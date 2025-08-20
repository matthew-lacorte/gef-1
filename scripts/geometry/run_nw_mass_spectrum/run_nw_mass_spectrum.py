#!/usr/bin/env python3
"""
GEF N_w Mass Spectrum Analysis

This module runs systematic sweeps over topological winding numbers (N_w) to
generate mass spectra. It varies physical parameters based on N_w according
to a resonance model and uses a parallelized relaxation solver to find the
final energy (mass) for each configuration.
"""

import datetime
import logging
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# --- Robust Package Imports ---
try:
    from gef.physics_core.solvers.hopfion_relaxer import HopfionRelaxer
    from gef.core.logging import logger
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from gef.physics_core.solvers.hopfion_relaxer import HopfionRelaxer
        from gef.core.logging import logger
    except ImportError:
        # Fallback to local import and standard logger if package structure fails
        from hopfion_relaxer import HopfionRelaxer
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

# Optional import for visualization
try:
    from gef.visualization.spectrum_plots import MassSpectrumPlotter
except ImportError:
    MassSpectrumPlotter = None  # Use local implementation


class ResonanceModel:
    """Calculates effective coupling parameters based on a resonance model."""

    @staticmethod
    def calculate_g_eff_squared(nw: int, resonance_params: Dict) -> float:
        """Calculate effective g_squared based on a sum of Gaussian resonance peaks."""
        g_base_sq = resonance_params.get('g_base_squared', 0.0)
        invert_amplitudes = resonance_params.get('invert_amplitudes', False)
        g_offset = 0.0
        
        # Robustly get peaks from config
        peaks = resonance_params.get('peaks', []) or []
        
        for peak in peaks:
            center = peak.get('center', 0.0)
            amplitude = peak.get('amplitude', 0.0)
            sigma = peak.get('sigma', 1.0)

            if invert_amplitudes:
                amplitude *= -1.0
            
            g_offset += amplitude * np.exp(-(nw - center)**2 / (2 * sigma**2))
            
        return g_base_sq + g_offset


class SimulationWorker:
    """Handles individual simulation runs for parallel processing."""
    
    @staticmethod
    def run_single_simulation(nw: int, base_config: Dict) -> Dict:
        """Run a single Hopfion simulation for a given winding number."""
        quiet = base_config.get('quiet', False)
        
        # Robustly determine resonance parameters
        resonance_params = base_config.get('resonance_parameters', {})
        if 'peaks' not in resonance_params:
            resonance_params['peaks'] = base_config.get('peaks', [])

        g_eff_sq = ResonanceModel.calculate_g_eff_squared(nw, resonance_params)
        
        if not quiet:
            logger.info(f"Starting simulation for N_w={nw}, g_eff²={g_eff_sq:.4f}")
        
        current_config = base_config.copy()
        current_config['g_squared'] = g_eff_sq
        
        try:
            solver = HopfionRelaxer(current_config)
            solver.initialize_field(nw=nw)

            n_skip = base_config.get('relaxation_n_skip', 1000)
            n_iter = base_config.get('relaxation_n_iter', 1024)
            record_series = base_config.get('record_series', False)

            t0 = datetime.datetime.now()
            # The relaxer now returns a tuple (series, energy)
            _, final_energy = solver.run_relaxation(
                n_skip=n_skip, n_iter=n_iter, record_series=record_series
            )
            dt = (datetime.datetime.now() - t0).total_seconds()

            converged = np.isfinite(final_energy)
            if not quiet:
                status = "converged" if converged else "FAILED"
                logger.info(f"Finished N_w={nw} in {dt:.1f}s; E_final={final_energy:.6f} ({status})")

            if not converged:
                final_energy = np.nan
                
        except Exception:
            # IMPROVEMENT: Use logger.exception to capture the full traceback for debugging
            logger.exception(f"CRITICAL ERROR in simulation for N_w={nw}. See traceback below.")
            final_energy = np.nan
            converged = False
        
        return {
            'winding_number': nw,
            'g_eff_squared': g_eff_sq,
            'final_energy': final_energy,
            'converged': converged,
        }


class MassSpectrumAnalyzer:
    """Main class for running, analyzing, and plotting mass spectrum sweeps."""
    
    def __init__(self, config_or_path):
        self.logger = logger
        if isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            self.config_path = Path(config_or_path)
            self.config = self._load_config()
        
        self.output_dir = self._setup_output_directory()
        self.quiet = self.config.get('quiet', False)
        
    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def _setup_output_directory(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        particle_type = self.config.get('particle_type', 'unknown')
        output_dir_name = f"nw_spectrum_{particle_type}_{timestamp}"
        output_dir = Path(self.config.get('output_dir', '.')) / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.quiet:
            self.logger.info(f"Output directory created: {output_dir}")
        return output_dir
        
    def run_parallel_simulations(self) -> pd.DataFrame:
        """Run simulations in parallel across the configured winding numbers."""
        nw_values = list(self.config['nw_sweep_range'])
        
        # Configure multiprocessing and Numba threading
        requested_procs = self.config.get('processes', 0)
        num_cores = max(1, multiprocessing.cpu_count() - 1) if requested_procs <= 0 else requested_procs
        
        threads_per_proc = self.config.get('threads_per_process', 1)
        os.environ['GEF_NUMBA_NUM_THREADS'] = str(threads_per_proc) # Custom knob
        os.environ['NUMBA_NUM_THREADS'] = str(threads_per_proc)     # Numba's knob

        if not self.quiet:
            self.logger.info(f"Running simulations for {len(nw_values)} N_w values: {nw_values}")
            self.logger.info(f"Using {num_cores} parallel processes, with {threads_per_proc} Numba thread(s) each.")
        
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
            self.logger.info(f"Results summary:\n{results_df.to_string()}")
        return csv_path
        
    def generate_plots(self, results_df: pd.DataFrame) -> Path:
        # Use local plotter if the package one isn't available
        plotter_cls = MassSpectrumPlotter if MassSpectrumPlotter else LocalMassSpectrumPlotter
        plotter = plotter_cls(self.config)
        plot_path = plotter.create_spectrum_plot(results_df, self.output_dir)
        if not self.quiet:
            self.logger.info(f"Plot saved to: {plot_path}")
        return plot_path
        
    def run_full_analysis(self) -> Tuple[pd.DataFrame, Path, Path]:
        self.logger.info("Starting N_w mass spectrum analysis.")
        results_df = self.run_parallel_simulations()
        
        csv_path = self.save_results(results_df) if not self.config.get('skip_save') else None
        plot_path = self.generate_plots(results_df) if not self.config.get('skip_plots') else None
        
        self.logger.info("Analysis complete.")
        return results_df, csv_path, plot_path


class LocalMassSpectrumPlotter:
    """Handles visualization of mass spectrum results. (Local fallback)"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def create_spectrum_plot(self, results_df: pd.DataFrame, output_dir: Path) -> Path:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))

        plotting_cfg = self.config.get('plotting', {})
        normalize_y = plotting_cfg.get('normalize_y', False)
        
        # Prepare data for plotting
        plot_df = results_df.copy()
        stable_df = plot_df.dropna(subset=['mass'])
        unstable_df = plot_df[plot_df['mass'].isna()]
        
        y_label = 'Emergent Mass (E_final - E_vacuum)'
        if normalize_y and not stable_df.empty:
            m_min, m_max = stable_df['mass'].min(), stable_df['mass'].max()
            m_span = m_max - m_min
            if m_span > 1e-9:
                plot_df['mass'] = plot_df['mass'].apply(lambda v: (v - m_min) / m_span if np.isfinite(v) else np.nan)
                stable_df = plot_df.dropna(subset=['mass']) # Re-calculate stable_df after normalization
                y_label = 'Normalized Emergent Mass'

        # Plot stable states
        if not stable_df.empty:
            ax.plot(stable_df['winding_number'], stable_df['mass'], 'o-',
                    color='blue', linewidth=2, markersize=8, label='Stable States')

        # Determine plot limits from stable data before plotting unstable points
        y_min, y_max = ax.get_ylim()
        y_pad = (y_max - y_min) * 0.05 if (y_max > y_min) else 0.1
        y_bottom = y_min - y_pad

        # Plot unstable/non-converged states
        if not unstable_df.empty:
            ax.scatter(unstable_df['winding_number'], [y_bottom] * len(unstable_df),
                       marker='x', color='red', s=100, label='Unstable/Non-Converged', zorder=5)
            # BUG FIX: The second, redundant scatter plot call has been removed.

        # Formatting
        ax.set_xlabel('Topological Winding Number (N_w)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'GEF Mass Spectrum: {self.config.get("particle_type", "Unknown")}', fontsize=14)

        # Add overlays (observed masses and resonance peaks)
        self._add_overlays(ax, stable_df, normalize_y)
        
        ax.legend(fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Auto-adjust limits with padding
        ax.margins(x=0.05, y=0.05)
        
        plot_path = output_dir / 'mass_vs_nw_spectrum.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return plot_path

    def _add_overlays(self, ax, stable_df, is_normalized):
        """Helper to add horizontal/vertical lines for observed masses and peaks."""
        # Observed masses (horizontal lines)
        if is_normalized and not stable_df.empty:
            m_min, m_max = self.config.get('mass_min', 0), self.config.get('mass_max', 1) # Assume these would be calculated and stored
            m_span = m_max - m_min
        
        for item in self.config.get('observed_masses', []):
            y = item.get('mass')
            if y is None: continue
            if is_normalized and m_span > 1e-9:
                y = (y - m_min) / m_span
            ax.axhline(y=y, color='gray', linestyle=':', lw=1.2, label=item.get('label'))

        # Resonance peaks (vertical lines)
        peaks = (self.config.get('resonance_parameters', {}).get('peaks', []) or
                 self.config.get('peaks', []))
        for peak in peaks:
            x = peak.get('center')
            if x is not None:
                ax.axvline(x=x, color='purple', linestyle='--', lw=1.0, alpha=0.6, label=f'Resonance @ {x}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run GEF N_w mass spectrum analysis")
    parser.add_argument('config_path', help='Path to YAML configuration file')
    args = parser.parse_args()
    
    try:
        analyzer = MassSpectrumAnalyzer(args.config_path)
        _, csv_path, plot_path = analyzer.run_full_analysis()
        print("\nAnalysis completed successfully!")
        if csv_path: print(f"Results saved to: {csv_path}")
        if plot_path: print(f"Plot saved to: {plot_path}")
    except Exception as e:
        logger.exception("Analysis failed with a critical error.")
        sys.exit(1)


if __name__ == '__main__':
    # For multiprocessing on Windows/macOS, it's safer to use 'forkserver' or 'spawn'
    if sys.platform != "linux":
        multiprocessing.set_start_method("spawn", force=True)
    main()