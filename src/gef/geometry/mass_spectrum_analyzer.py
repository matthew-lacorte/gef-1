"""
GEF N_w Mass Spectrum Analysis Module

This module contains the core logic for running systematic sweeps over topological
winding numbers to generate mass spectra for different particle types within the
GEF framework.

Author: GEF Research Team
Date: 2025
"""

import datetime
import logging
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import MaxNLocator

# Ensure we can import from the GEF package
try:
    from gef.core.hopfion_relaxer import HopfionRelaxer
except ImportError:
    # Fallback for development environment: add project root and retry
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from gef.core.hopfion_relaxer import HopfionRelaxer

# Set up a fallback logger if the core logger isn't available
try:
    from gef.core.logging import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


class ResonanceModel:
    """Calculates effective coupling parameters based on resonance theory."""

    @staticmethod
    def calculate_g_eff_squared(nw: int, resonance_params: Dict) -> float:
        """
        Calculate effective g_squared based on resonance model.

        Args:
            nw: Topological winding number
            resonance_params: Dictionary containing resonance parameters

        Returns:
            Effective g_squared value for the given winding number
        """
        g_base_sq = resonance_params['g_base_squared']
        invert_amplitudes = resonance_params.get('invert_amplitudes', False)

        g_hook_sq = 0.0
        for peak in resonance_params.get('peaks', []):
            center = peak['center']
            amplitude = peak['amplitude']
            sigma = peak['sigma']

            if invert_amplitudes:
                amplitude *= -1.0

            g_hook_sq += amplitude * np.exp(-(nw - center)**2 / (2 * sigma**2))

        return g_base_sq + g_hook_sq


class SimulationWorker:
    """Handles individual simulation runs for parallel processing."""

    @staticmethod
    def run_single_simulation(nw: int, base_config: Dict) -> Dict:
        """
        Run a single Hopfion simulation for a given winding number.
        """
        quiet = bool(base_config.get('quiet', False))
        resonance_params = base_config['resonance_parameters']
        g_eff_sq = ResonanceModel.calculate_g_eff_squared(nw, resonance_params)

        if not quiet:
            logger.info(f"Starting simulation for N_w={nw}, g_eff²={g_eff_sq:.4f}")

        current_config = base_config.copy()
        current_config['g_squared'] = g_eff_sq

        try:
            solver = HopfionRelaxer(current_config)
            solver.initialize_field(nw=nw)

            t0 = datetime.datetime.now()
            record_series = bool(base_config.get('record_series', True))
            _, final_energy = solver.run_relaxation(record_series=record_series)
            dt = (datetime.datetime.now() - t0).total_seconds()

            if not quiet:
                logger.info(f"Finished N_w={nw} in {dt:.1f}s; E_final={final_energy}")
            
            converged = bool(np.isfinite(final_energy))
            if not converged:
                final_energy = np.nan

        except Exception as e:
            logger.error(f"Simulation for N_w={nw} failed with error: {e}")
            final_energy = np.nan
            converged = False

        return {
            'winding_number': nw,
            'g_eff_squared': g_eff_sq,
            'final_energy': final_energy,
            'converged': converged,
        }


class MassSpectrumPlotter:
    """Handles visualization of mass spectrum results."""

    def __init__(self, config: Dict):
        self.config = config

    def create_spectrum_plot(self, results_df: pd.DataFrame, output_dir: Path) -> Path:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))

        plotting_cfg = self.config.get('plotting', {}) or {}
        stable_df = results_df.dropna(subset=['mass'])
        unstable_df = results_df[results_df['mass'].isna()]

        if not stable_df.empty:
            ax.plot(stable_df['winding_number'], stable_df['mass'], 'o-', color='blue', linewidth=2, markersize=8, label='Stable States')

        if not unstable_df.empty:
            y_min = stable_df['mass'].min() if not stable_df.empty else 0
            ax.scatter(unstable_df['winding_number'], [y_min * 0.95] * len(unstable_df), marker='x', color='red', s=100, label='Unstable/Non-Converged', zorder=5)

        ax.set_xlabel('Topological Winding Number (N_w)', fontsize=12)
        ax.set_ylabel('Emergent Mass (E_final - E_vacuum)', fontsize=12)
        particle_type = self.config.get('particle_type', 'Unknown')
        ax.set_title(f'GEF Mass Spectrum: {particle_type} Particles', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        nw_values = list(self.config['nw_sweep_range'])
        if len(nw_values) <= 12:
            ax.set_xticks(nw_values)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))

        plot_path = output_dir / 'mass_vs_nw_spectrum.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return plot_path


class NwMassSpectrumAnalyzer:
    """Main class for running mass spectrum analysis."""

    def __init__(self, config_or_path):
        self.logger = logger
        if isinstance(config_or_path, dict):
            self.config_path = None
            self.config = config_or_path
        else:
            self.config_path = Path(config_or_path)
            self.config = self._load_config()

        self.output_dir = self._setup_output_directory()
        self.quiet = bool(self.config.get('quiet', False))

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_output_directory(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        particle_type = self.config.get('particle_type', 'unknown')
        output_dir_name = f"nw_spectrum_{particle_type}_{timestamp}"
        output_dir = Path(self.config['output_dir']) / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.quiet:
            self.logger.info(f"Output directory: {output_dir}")
        return output_dir

    def run_parallel_simulations(self) -> pd.DataFrame:
        nw_values = self.config['nw_sweep_range']
        num_cores = int(self.config.get('processes', multiprocessing.cpu_count() - 1))
        if num_cores <= 0:
            num_cores = 1
        
        if not self.quiet:
            self.logger.info(f"Running on {num_cores} processes for N_w values: {nw_values}")

        worker_func = partial(SimulationWorker.run_single_simulation, base_config=self.config)

        if num_cores == 1:
            results = list(map(worker_func, nw_values))
        else:
            with multiprocessing.Pool(processes=num_cores) as pool:
                results = pool.map(worker_func, nw_values)

        results_df = pd.DataFrame(results)
        vacuum_energy = self.config.get('vacuum_energy', 0.0)
        results_df['mass'] = results_df['final_energy'] - vacuum_energy
        return results_df

    def save_results(self, results_df: pd.DataFrame) -> Path:
        csv_path = self.output_dir / 'mass_spectrum_results.csv'
        results_df.to_csv(csv_path, index=False)
        if not self.quiet:
            self.logger.info(f"Results saved to: {csv_path}")
        return csv_path

    def generate_plots(self, results_df: pd.DataFrame) -> Path:
        plotter = MassSpectrumPlotter(self.config)
        plot_path = plotter.create_spectrum_plot(results_df, self.output_dir)
        if not self.quiet:
            self.logger.info(f"Plot saved to: {plot_path}")
        return plot_path

    def run_full_analysis(self) -> Tuple[pd.DataFrame, Path, Path]:
        self.logger.info("Starting N_w mass spectrum analysis")
        results_df = self.run_parallel_simulations()

        csv_path = self.save_results(results_df) if not self.config.get('skip_save') else self.output_dir / 'mass_spectrum_results.csv'
        plot_path = self.generate_plots(results_df) if not self.config.get('skip_plots') else self.output_dir / 'mass_vs_nw_spectrum.png'

        self.logger.info("Analysis complete")
        return results_df, csv_path, plot_path
