"""
Computes the GEF geometric properties (frequency, tick, radius) from a
particle's rest mass. This tool acts as a "mass spectrometer" for the GEF framework.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import sympy as sp
import yaml
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────── Logging setup
import logging

# Create a logger
logger = logging.getLogger("gef_mass_spectrometer")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
logger.addHandler(console_handler)

# Setup function for log file
def setup_logfile(log_path):
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(file_handler)

# ────────────────────────────────────────────────────── GEF infrastructure
from gef.core.constants import CONSTANTS_DICT

# ────────────────────────────────────────────────────── Physical constants
try:
    c_const = CONSTANTS_DICT["c"]                  # speed of light
    planck_const = CONSTANTS_DICT["planck"]        # Planck constant
    eV_const = CONSTANTS_DICT["electron_volt"]     # 1 eV in joule
except KeyError as err:
    raise RuntimeError(f"Constant {err} missing from CONSTANTS_DICT")

c = c_const.value or 299_792_458                   # fallback exact CODATA
if planck_const.value is None:
    raise ValueError("Numeric value for Planck constant not set in constants.py")
planck = planck_const.value
electron_volt = eV_const.value

# ==============================================================================
# 1. CORE COMPUTATION
# ==============================================================================

def compute_radius_from_mass(mass_MeV: float) -> Tuple[float, float, float]:
    """Return (frequency Hz, tick s, radius m) for a given rest mass in MeV/c²."""
    if mass_MeV <= 0:
        return np.nan, np.nan, np.nan
    E_P = mass_MeV * 1e6 * electron_volt           # J
    f_P = E_P / planck                             # Hz
    t_P = 1 / f_P                                  # s
    r_P = c * t_P / (2 * sp.pi)                    # m
    return f_P, t_P, float(r_P)

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

def _default_cfg(script_dir: Path) -> Path:
    return script_dir / "configs" / "default_gef_mass_spectrometer.yml"

def _new_outdir(script_dir: Path, base: Path | None) -> Path:
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    out_base = (base or script_dir / "outputs").resolve()
    out_dir = out_base / f"{timestamp}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "results").mkdir()
    return out_dir

def main(argv: List[str] | None = None) -> int:
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="GEF Mass-to-Geometry Spectrometer.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-c", "--config", type=Path, default=_default_cfg(script_dir),
                        help="Path to the YAML configuration file.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Base directory for output files")
    parser.add_argument("--log", type=Path, default=None,
                        help="Path to log file")
    args = parser.parse_args(argv)

    if args.log:
        setup_logfile(str(args.log))

    if not args.config.is_file():
        logger.error(f"Config file not found at '{args.config}'")
        return 1

    try:
        cfg = yaml.safe_load(args.config.read_text())
        if cfg is None:
            logger.error(f"Config file is empty or invalid: {args.config}")
            return 1
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return 1
    
    # --- Setup Output Directory ---
    out_dir = _new_outdir(script_dir, args.output_dir)
    # Save the config used for this run
    (out_dir / "used_config.yml").write_text(yaml.dump(cfg))
    logger.info(f"Output directory: {out_dir}")
    
    # --- Perform Calculation ---
    if cfg['mode'] == 'single':
        masses_to_process = [cfg['single_mass_MeV']]
    if cfg['mode'] == 'sweep':
        masses_to_process = generate_mass_sweep(cfg)
    else:
        logger.error(f"Error: Invalid mode '{cfg['mode']}' in config file.")
        return 1

    logger.info("Calculating geometric properties for specified masses...")
    results = [compute_radius_from_mass(m) for m in masses_to_process]
    
    # --- Save Data ---
    radii_fm = np.array([r * 1e15 for f, t, r in results])
    csv_filename = "mass_spectrum_results.csv"
    csv_path = out_dir / "results" / csv_filename
    
    # Create a more detailed CSV with all parameters
    with csv_path.open("w") as csvfile:
        csvfile.write("mass_MeV,f_P_Hz,t_P_s,r_P_fm\n")
        for i, mass in enumerate(masses_to_process):
            f_P, t_P, r_P = results[i]
            r_P_fm = r_P * 1e15  # Convert to femtometers
            csvfile.write(f"{mass:.6f},{f_P:.8e},{t_P:.8e},{r_P_fm:.8f}\n")
    
    logger.info(f"Results saved to: {csv_path}")

    # --- Generate Plot ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=tuple(cfg['plotting']['figsize']))
    
    # Plot the main theoretical curve
    if cfg['mode'] == 'sweep':
        ax.plot(masses_to_process, radii_fm, label='GEF Prediction (r ∝ 1/m)',
                color='lightblue', zorder=1)
    
    # Overlay the known particles
    logger.info("Overlaying known particles on the plot...")
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
    
    plot_filename = "mass_spectrum_plot.png"
    plot_path = out_dir / "results" / plot_filename
    plt.savefig(plot_path, dpi=cfg['plotting']['dpi'])
    logger.info(f"Plot saved to: {plot_path}")
    
    # Don't call plt.show() in production scripts as it blocks execution
    # plt.show()

    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())