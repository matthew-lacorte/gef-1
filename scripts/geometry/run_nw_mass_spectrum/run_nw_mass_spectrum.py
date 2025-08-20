"""
Calculate the mass spectrum based on the NW (Nambu-Wheeler) model.

This script implements the NW model for calculating the theoretical mass spectrum
of particles based on geometric principles, comparing the results with observed
particle masses.

* Reads parameters from a YAML config (default: ./configs/default_nw_mass_spectrum.yml)
* Calculates theoretical mass values using the NW model equations
* Compares calculated masses with observed standard model particles
* Identifies potential resonances and mass relationships
* Outputs data files and visualizations in a timestamped directory

This script should implement:
1. NW model equations and parameters
2. Mass spectrum calculation functions
3. Comparison with standard model masses
4. Resonance identification algorithms
5. Visualization of mass spectrum patterns
"""

# TODO: Implement full script with:
# - Imports (numpy, matplotlib, yaml, etc.)
# - NW model equations and calculation functions
# - Standard model mass comparison
# - Resonance pattern identification
# - Command line interface
# - Main execution logic

#!/usr/bin/env python3
"""
GEF N_w Mass Spectrum Analysis

This module runs systematic sweeps over topological winding numbers to generate
mass spectra for different particle types within the GEF framework.

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
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Ensure we can import from the GEF package
try:
    # Prefer the layout present in this repo
    from gef.core.hopfion_relaxer import HopfionRelaxer
except Exception:
    try:
        # Alternate layout (older experiments tree)
        from gef.physics_core.solvers.hopfion_relaxer import HopfionRelaxer
    except Exception:
        # Fallback for development environment: add project root and retry
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from gef.core.hopfion_relaxer import HopfionRelaxer

# Optional imports; provide local fallbacks if not available
try:
    from gef.analysis.resonance_models import ResonanceModel  # type: ignore
except Exception:
    pass
try:
    from gef.utils.logging_utils import setup_logger  # type: ignore
except Exception:
    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
try:
    from gef.visualization.spectrum_plots import MassSpectrumPlotter  # type: ignore
except Exception:
    MassSpectrumPlotter = None  # We'll use the local implementation below


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
        
        g_hook_sq = 0.0
        for peak in resonance_params['peaks']:
            center = peak['center']
            amplitude = peak['amplitude']
            sigma = peak['sigma']
            
            # Gaussian resonance peak
            g_hook_sq += amplitude * np.exp(-(nw - center)**2 / (2 * sigma**2))
            
        return g_base_sq + g_hook_sq


class SimulationWorker:
    """Handles individual simulation runs for parallel processing."""
    
    @staticmethod
    def run_single_simulation(nw: int, base_config: Dict) -> Dict:
        """
        Run a single Hopfion simulation for given winding number.
        
        Args:
            nw: Topological winding number
            base_config: Base configuration dictionary
            
        Returns:
            Dictionary containing simulation results
        """
        logger = logging.getLogger(__name__)
        
        # Calculate effective coupling for this winding number
        resonance_params = base_config['resonance_parameters']
        g_eff_sq = ResonanceModel.calculate_g_eff_squared(nw, resonance_params)
        
        logger.info(f"Starting simulation for N_w={nw}, g_eff²={g_eff_sq:.4f}")
        
        # Create configuration for this specific run
        current_config = base_config.copy()
        current_config['g_squared'] = g_eff_sq
        
        try:
            # Initialize and run solver
            solver = HopfionRelaxer(current_config)
            solver.initialize_field(nw=nw)

            # Optional relaxation controls from config
            n_skip = int(base_config.get('relaxation_n_skip', 1000))
            n_iter = int(base_config.get('relaxation_n_iter', 1024))

            t0 = datetime.datetime.now()
            phi_series, final_energy = solver.run_relaxation(n_skip=n_skip, n_iter=n_iter)
            dt = (datetime.datetime.now() - t0).total_seconds()
            logger.info(f"Finished N_w={nw} in {dt:.1f}s; E_final={final_energy}")
            converged = bool(np.isfinite(final_energy))

            # Validate results
            if not converged:
                logger.warning(f"Simulation for N_w={nw} produced invalid energy")
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


class MassSpectrumAnalyzer:
    """Main class for running mass spectrum analysis."""
    
    def __init__(self, config_or_path):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config_or_path: Either a path to a YAML config file, or a dict config
        """
        self.logger = setup_logger(__name__)

        # Accept either a dict or a path-like
        if isinstance(config_or_path, dict):
            self.config_path = None
            self.config = config_or_path
        else:
            self.config_path = Path(config_or_path)
            self.config = self._load_and_validate_config()
        
        # Setup output directory
        self.output_dir = self._setup_output_directory()
        
    def _load_and_validate_config(self) -> Dict:
        """Load and validate configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate configuration (implement this in utils module)
        # validate_spectrum_config(config)
        
        return config
        
    def _setup_output_directory(self) -> Path:
        """Create timestamped output directory."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        particle_type = self.config.get('particle_type', 'unknown')
        
        output_dir_name = f"nw_spectrum_{particle_type}_{timestamp}"
        output_dir = Path(self.config['output_dir']) / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory created: {output_dir}")
        return output_dir
        
    def run_parallel_simulations(self) -> pd.DataFrame:
        """
        Run simulations in parallel across different winding numbers.
        
        Returns:
            DataFrame containing all simulation results
        """
        nw_values = self.config['nw_sweep_range']
        # Allow user to cap process count to avoid oversubscription with Numba threads
        requested = int(self.config.get('processes', 0))
        if requested > 0:
            num_cores = requested
        else:
            num_cores = max(1, min(2, multiprocessing.cpu_count() - 1))
        
        # Configure per-process Numba threads via environment so children inherit it
        threads_per_proc = int(self.config.get('threads_per_process', 0))
        if threads_per_proc > 0:
            os.environ['NUMBA_NUM_THREADS'] = str(threads_per_proc)
            os.environ['GEF_NUMBA_NUM_THREADS'] = str(threads_per_proc)
        threading_layer = str(self.config.get('numba_threading_layer', '') or '').strip()
        if threading_layer:
            os.environ['NUMBA_THREADING_LAYER'] = threading_layer

        self.logger.info(f"Running simulations for N_w values: {nw_values}")
        self.logger.info(f"Using {num_cores} parallel processes; NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS','')} layer={os.environ.get('NUMBA_THREADING_LAYER','')} ")
        
        # Build worker function
        worker_func = partial(
            SimulationWorker.run_single_simulation,
            base_config=self.config,
        )

        # Run simulations: sequential if single process to avoid pickling issues
        if num_cores == 1:
            results = list(map(worker_func, nw_values))
        else:
            with multiprocessing.Pool(processes=num_cores) as pool:
                results = pool.map(worker_func, nw_values)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate masses relative to vacuum
        vacuum_energy = self.config.get('vacuum_energy', 0.0)
        results_df['mass'] = results_df['final_energy'] - vacuum_energy
        
        return results_df
        
    def save_results(self, results_df: pd.DataFrame) -> Path:
        """Save results to CSV file."""
        csv_path = self.output_dir / 'mass_spectrum_results.csv'
        results_df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Results saved to: {csv_path}")
        self.logger.info(f"Results summary:\n{results_df}")
        
        return csv_path
        
    def generate_plots(self, results_df: pd.DataFrame) -> Path:
        """Generate mass spectrum visualization."""
        plotter = MassSpectrumPlotter(self.config)
        plot_path = plotter.create_spectrum_plot(results_df, self.output_dir)
        
        self.logger.info(f"Plot saved to: {plot_path}")
        return plot_path
        
    def run_full_analysis(self) -> Tuple[pd.DataFrame, Path, Path]:
        """
        Run complete mass spectrum analysis.
        
        Returns:
            Tuple of (results_dataframe, csv_path, plot_path)
        """
        self.logger.info("Starting N_w mass spectrum analysis")
        
        # Run simulations
        results_df = self.run_parallel_simulations()
        
        # Save results
        csv_path = self.save_results(results_df)
        
        # Generate plots
        plot_path = self.generate_plots(results_df)
        
        self.logger.info("Analysis complete")
        
        return results_df, csv_path, plot_path


class MassSpectrumPlotter:
    """Handles visualization of mass spectrum results."""
    
    def __init__(self, config: Dict):
        """Initialize plotter with configuration."""
        self.config = config
        
    def create_spectrum_plot(self, results_df: pd.DataFrame, output_dir: Path) -> Path:
        """
        Create and save mass spectrum plot.
        
        Args:
            results_df: DataFrame containing simulation results
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot file
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))

        # Axis normalization / padding options (compute BEFORE plotting)
        plotting_cfg = self.config.get('plotting', {}) or {}
        normalize_y = bool(plotting_cfg.get('normalize_y', False))
        x_pad_frac = float(plotting_cfg.get('x_pad_frac', 0.1))
        y_pad_frac = float(plotting_cfg.get('y_pad_frac', 0.1))

        nw_values = list(self.config['nw_sweep_range'])
        x_min, x_max = float(min(nw_values)), float(max(nw_values))
        x_span = x_max - x_min
        if x_span <= 0:
            x_pad = 1.0
        else:
            x_pad = max(1e-3, x_pad_frac * x_span)

        # Prepare data for plotting with optional normalization
        base_stable_df = results_df.dropna(subset=['mass'])
        stable_masses = base_stable_df['mass'].values if not base_stable_df.empty else np.array([0.0])
        m_min = float(np.nanmin(stable_masses)) if stable_masses.size > 0 else 0.0
        m_max = float(np.nanmax(stable_masses)) if stable_masses.size > 0 else 1.0
        m_span = m_max - m_min
        if normalize_y and stable_masses.size > 0 and m_span > 0:
            def norm(m):
                return (m - m_min) / m_span
            plot_df = results_df.copy()
            plot_df['mass'] = plot_df['mass'].apply(lambda v: norm(v) if np.isfinite(v) else np.nan)
            y_label = 'Normalized Emergent Mass (unitless)'
        else:
            plot_df = results_df
            y_label = 'Emergent Mass (E_final - E_vacuum)'

        # Split after potential normalization
        stable_df = plot_df.dropna(subset=['mass'])
        unstable_df = plot_df[plot_df['mass'].isna()]

        if stable_masses.size > 0:
            y_vals = stable_df['mass'].values if not stable_df.empty else np.array([0.0])
            y_min = float(np.nanmin(y_vals))
            y_max = float(np.nanmax(y_vals))
        else:
            y_min, y_max = 0.0, 1.0
        y_span = y_max - y_min
        y_pad = max(1e-6, y_pad_frac * (y_span if y_span > 0 else 1.0))

        # In MassSpectrumPlotter, after you calculate y_min and y_pad
        y_bottom = y_min - y_pad

        if not unstable_df.empty:
            ax.scatter(
                unstable_df['winding_number'],
                # Plot them at the very bottom of the chart for clarity
                [y_bottom] * len(unstable_df), 
                marker='x',
                color='red',
                s=100,
                label='Unstable/Non-Converged',
                zorder=5
            )

        # Apply padded limits to avoid smushed plots
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        # Now plot points/lines using prepared data
        if not stable_df.empty:
            ax.plot(
                stable_df['winding_number'],
                stable_df['mass'],
                'o-',
                color='blue',
                linewidth=2,
                markersize=8,
                label='Stable States'
            )

        if not unstable_df.empty:
            ax.scatter(
                unstable_df['winding_number'],
                np.zeros_like(unstable_df['winding_number']),
                marker='x',
                color='red',
                s=100,
                label='Unstable/Non-Converged',
                zorder=5
            )

        # Formatting
        ax.set_xlabel('Topological Winding Number (N_w)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        particle_type = self.config.get('particle_type', 'Unknown')
        ax.set_title(f'GEF Mass Spectrum: {particle_type} Particles', fontsize=14)
        
        # Optional overlays: observed masses (horizontal lines)
        observed = self.config.get('observed_masses', []) or []
        for item in observed:
            try:
                y = float(item.get('mass'))
                # Normalize observed overlay if plot is normalized
                if normalize_y and m_span > 0:
                    y = (y - m_min) / m_span
                label = str(item.get('label', 'Observed'))
                ax.axhline(y=y, color='gray', linestyle=':', linewidth=1.2, alpha=0.8)
                ax.text(
                    x=ax.get_xlim()[0], y=y, s=f"  {label} ({y:g})",
                    va='center', ha='left', color='gray', fontsize=10
                )
            except Exception:
                continue

        # Optional overlays: resonance peak centers from config
        peaks = (self.config.get('resonance_parameters') or {}).get('peaks', [])
        if peaks:
            for pk in peaks:
                try:
                    x = float(pk.get('center'))
                    ax.axvline(x=x, color='purple', linestyle='--', linewidth=1.0, alpha=0.6)
                except Exception:
                    continue

        ax.legend(fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Set x-ticks to winding numbers (cap to avoid overcrowding)
        if len(nw_values) <= 12:
            ax.set_xticks(nw_values)
        else:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
        
        # Save plot
        plot_path = output_dir / 'mass_vs_nw_spectrum.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return plot_path


def setup_logger(name: str) -> logging.Logger:
    """Setup logger for the module."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run GEF N_w mass spectrum analysis"
    )
    parser.add_argument(
        'config_path',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    try:
        # Run analysis
        analyzer = MassSpectrumAnalyzer(args.config_path)
        results_df, csv_path, plot_path = analyzer.run_full_analysis()
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {csv_path}")
        print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()