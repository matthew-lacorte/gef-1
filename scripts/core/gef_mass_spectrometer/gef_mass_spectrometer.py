# experiments/scripts/gef_mass_spectrometer.py
"""
Computes the GEF geometric properties (frequency, tick, radius) from a
particle's rest mass. This tool acts as a "mass spectrometer" for the GEF framework.
"""
from __future__ import annotations
import argparse
import datetime as dt
import sys
import uuid
from pathlib import Path
from typing import Iterable, List

import numpy as np
import sympy as sp
import yaml
import matplotlib.pyplot as plt

# --- GEF Infrastructure (assumes this structure exists) ---
# from gef.core.logging import logger, setup_logfile
# from gef.core.constants import CONSTANTS_DICT

# For this standalone script, let's define constants directly
c_const = 299_792_458.0
planck_const = 6.62607015e-34
eV_const = 1.602176634e-19
# logger = ... # A real implementation would use the project's logger

# ==============================================================================
# 1. CORE COMPUTATION
# ==============================================================================

def compute_radius_from_mass(mass_MeV: float) -> tuple[float, float, float]:
    """Returns (frequency Hz, tick s, radius m) for a given rest mass in MeV/c²."""
    if mass_MeV <= 0:
        return np.nan, np.nan, np.nan
    energy_joules = mass_MeV * 1e6 * eV_const
    frequency = energy_joules / planck_const
    tick_time = 1 / frequency
    radius_meters = (c_const * tick_time) / (2 * np.pi)
    return frequency, tick_time, radius_meters

# ==============================================================================
# 2. DATA GENERATION
# ==============================================================================

def generate_mass_sweep(cfg: dict) -> np.ndarray:
    """Yields an array of masses (MeV) according to the sweep config."""
    sweep_cfg = cfg['sweep']
    if sweep_cfg['scale'] == 'log':
        return np.logspace(
            np.log10(sweep_cfg['start_MeV']),
            np.log10(sweep_cfg['stop_MeV']),
            sweep_cfg['num_points']
        )
    elif sweep_cfg['scale'] == 'linear':
        return np.linspace(
            sweep_cfg['start_MeV'],
            sweep_cfg['stop_MeV'],
            sweep_cfg['num_points']
        )
    else:
        raise ValueError(f"Unknown sweep scale: {sweep_cfg['scale']}")

# ==============================================================================
# 3. MAIN EXECUTION LOGIC
# ==============================================================================

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GEF Mass-to-Geometry Spectrometer.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    default_config = Path(__file__).parent / "configs" / "default_gef_mass_spectrometer.yml"
    parser.add_argument("-c", "--config", type=Path, default=default_config,
                        help="Path to the YAML configuration file.")
    args = parser.parse_args(argv)

    if not args.config.is_file():
        print(f"Error: Config file not found at '{args.config}'")
        return 1

    cfg = yaml.safe_load(args.config.read_text())
    
    # --- Setup Output Directory ---
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Perform Calculation ---
    if cfg['mode'] == 'single':
        masses_to_process = [cfg['single_mass_MeV']]
    elif cfg['mode'] == 'sweep':
        masses_to_process = generate_mass_sweep(cfg)
    else:
        print(f"Error: Invalid mode '{cfg['mode']}' in config file.")
        return 1

    print("Calculating geometric properties for specified masses...")
    results = [compute_radius_from_mass(m) for m in masses_to_process]
    
    # --- Save Data ---
    radii_fm = np.array([r * 1e15 for f, t, r in results])
    csv_filename = f"{cfg['base_filename']}_{timestamp}_{run_id}.csv"
    csv_path = output_dir / csv_filename
    np.savetxt(csv_path, np.c_[masses_to_process, radii_fm],
               header="mass_MeV,radius_fm", delimiter=",")
    print(f"Results saved to: {csv_path}")

    # --- Generate Plot ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=tuple(cfg['plotting']['figsize']))
    
    # Plot the main theoretical curve
    if cfg['mode'] == 'sweep':
        ax.plot(masses_to_process, radii_fm, label='GEF Prediction (r ∝ 1/m)',
                color='lightblue', zorder=1)
    
    # Overlay the known particles
    print("Overlaying known particles on the plot...")
    for name, mass, color in cfg['known_particles']:
        freq, tick, radius = compute_radius_from_mass(mass)
        radius_fm = radius * 1e15
        ax.plot(mass, radius_fm, 'o', ms=10, label=f'{name} ({mass:.1f} MeV)',
                color=color, zorder=2, markeredgecolor='black')
    
    # --- Plot Cosmetics ---
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(cfg['plotting']['title'], fontsize=18, weight='bold')
    ax.set_xlabel(cfg['plotting']['xlabel'], fontsize=14)
    ax.set_ylabel(cfg['plotting']['ylabel'], fontsize=14)
    ax.legend()
    ax.grid(True, which="both", linestyle='--')
    
    plot_filename = f"{cfg['base_filename']}_{timestamp}_{run_id}.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=cfg['plotting']['dpi'])
    print(f"Plot saved to: {plot_path}")
    
    plt.show()

    return 0

if __name__ == "__main__":
    # To run, ensure your config file is at:
    # experiments/scripts/configs/mass_spectrometer_v1.yml
    # Or provide the path with the -c flag.
    # Note: Added a simplified main hook for direct execution.
    # A real implementation might get the config path from a more robust location.
    sys.exit(main())