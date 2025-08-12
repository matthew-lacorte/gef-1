#!/usr/bin/env python3
"""
GEF Mass→Geometry "Spectrometer"

Given particle rest mass m (MeV/c^2), compute:
  E = m * (1e6 eV) * (J/eV)
  f = E / h
  T = 1/f
  λ̄ = c*T/(2π) = ħ c / E = ħ/(m c)   (reduced Compton length)

We plot λ̄ (fm) vs. m (MeV/c^2). This is *not* a charge/classical radius;
it's the reduced Compton wavelength, which scales ~ 1/m.

Notes:
- Uses observed (benchmark) constants from gef.core.constants.CONSTANTS_DICT
- Robust config validation via Pydantic (accepts both list-of-dicts and legacy list-of-lists for known_particles)
- Extended prediction line spans [min_mass, max_mass] across particles + sweep
- Legend uses scientific notation for very large masses (e.g., Planck mass)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import math
import sys
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Literal

# ────────────────────────────────────────────────────────────────────────────
# Constants plumbing (from your registry)
# ────────────────────────────────────────────────────────────────────────────

from gef.core.constants import CONSTANTS_DICT  # expects keys: c_observed, hbar, eV

# Pull numeric values with clear error messages + safe fallbacks (where valid).
try:
    c_const = CONSTANTS_DICT["c_observed"]   # m/s
    hbar_const = CONSTANTS_DICT["hbar"]      # J*s
    eV_const = CONSTANTS_DICT["eV"]          # J
except KeyError as err:
    raise RuntimeError(f"Constant {err} missing from CONSTANTS_DICT")

c = c_const.value or 299_792_458.0
if hbar_const.value is None:
    raise ValueError("Numeric value for reduced Planck constant (ħ) not set in constants.py")
hbar = float(hbar_const.value)
electron_volt = float(eV_const.value)
planck = hbar * (getattr(math, "tau", 2.0 * math.pi))  # h = 2π ħ
tau = getattr(math, "tau", 2.0 * math.pi)

# ────────────────────────────────────────────────────────────────────────────
# Config schema (Pydantic v2)
# ────────────────────────────────────────────────────────────────────────────

class KnownParticle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    mass: float  # MeV
    color: str = Field("#000000", description="Any Matplotlib color")

class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_MeV: float
    stop_MeV: float
    num_points: int = Field(ge=2)
    scale: Literal["linear", "log"] = "log"

    @field_validator("start_MeV", "stop_MeV")
    @classmethod
    def _positive(cls, v):
        if v <= 0:
            raise ValueError("start_MeV/stop_MeV must be > 0")
        return v

    @field_validator("stop_MeV")
    @classmethod
    def _ordered(cls, v, info):
        start = info.data.get("start_MeV", None)
        if start is not None and v <= start:
            raise ValueError("stop_MeV must be > start_MeV")
        return v

class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dpi: int = 200
    figsize: Tuple[float, float] = (14, 10)
    title: str = "GEF Mass-to-Geometry Spectrometer"
    xlabel: str = "Particle Rest Mass (MeV/c²)"
    ylabel: str = "Predicted Reduced Compton Length λ̄ (fm)"

class AppConfig(BaseModel):
    """
    Main configuration model.

    known_particles can be:
      - preferred: list of dicts {name, mass, color}
      - legacy:   list of [name, mass, color]
    """
    model_config = ConfigDict(extra="forbid")

    mode: Literal["single", "sweep"]
    plotting: PlotConfig = PlotConfig()
    sweep: SweepConfig | None = None
    single_mass_MeV: float | None = None
    known_particles: List[KnownParticle] = Field(default_factory=list)

    @field_validator("known_particles", mode="before")
    @classmethod
    def _coerce_known_particles(cls, v):
        if v is None:
            return []
        # Backward-compat: allow [[name, mass, color], ...]
        if isinstance(v, list) and v and isinstance(v[0], list):
            coerced = []
            for triple in v:
                if len(triple) != 3:
                    raise ValueError("Each legacy known_particle triple must have 3 items: [name, mass, color]")
                name, mass, color = triple
                coerced.append({"name": name, "mass": float(mass), "color": color})
            return coerced
        return v

    @field_validator("single_mass_MeV")
    @classmethod
    def _single_ok(cls, v, info):
        mode = info.data.get("mode")
        if mode == "single":
            if v is None or not (isinstance(v, (int, float)) and v > 0):
                raise ValueError("single_mass_MeV must be a positive number for mode='single'")
        return v

    @field_validator("sweep")
    @classmethod
    def _sweep_ok(cls, v, info):
        mode = info.data.get("mode")
        if mode == "sweep" and v is None:
            raise ValueError("sweep config is required for mode='sweep'")
        return v

# ────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path | None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("gef.mass_spectrometer")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
        logger.addHandler(sh)

    if log_path and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
        logger.addHandler(fh)
    return logger

# ────────────────────────────────────────────────────────────────────────────
# Core numerics
# ────────────────────────────────────────────────────────────────────────────

def compton_from_mass_MeV(mass_MeV: float) -> tuple[float, float, float]:
    """
    Return (f_Hz, period_s, lambda_bar_m) for rest mass in MeV/c^2.
    λ̄ = ħ/(mc). Returns (nan, nan, nan) for invalid inputs.
    """
    try:
        m = float(mass_MeV)
    except Exception:
        return math.nan, math.nan, math.nan
    if not (m > 0 and math.isfinite(m)):
        return math.nan, math.nan, math.nan

    E_J = m * 1e6 * electron_volt               # J
    f_Hz = E_J / planck                          # 1/s
    T_s  = 1.0 / f_Hz
    lambda_bar_m = c * T_s / tau                 # = ħc/E = ħ/(mc)
    return f_Hz, T_s, lambda_bar_m

def compute_series_lambda_bar_fm(masses_MeV: np.ndarray) -> np.ndarray:
    """Vectorized λ̄ (fm) for array of masses in MeV."""
    E_J = masses_MeV * 1e6 * electron_volt
    with np.errstate(divide="ignore", invalid="ignore"):
        lamb_m = (hbar * c) / E_J
    return lamb_m * 1e15  # fm

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _default_cfg_path(script_dir: Path) -> Path:
    return script_dir / "configs" / "default_gef_mass_spectrometer.yml"

def _new_outdir(script_dir: Path, base: Path | None) -> Path:
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    out_base = (base or script_dir / "outputs").resolve()
    out_dir = out_base / f"{timestamp}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "results").mkdir()
    return out_dir

def mass_sweep_MeV(sweep: SweepConfig) -> np.ndarray:
    if sweep.scale == "log":
        return np.logspace(np.log10(sweep.start_MeV), np.log10(sweep.stop_MeV), sweep.num_points)
    return np.linspace(sweep.start_MeV, sweep.stop_MeV, sweep.num_points)

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="GEF Mass-to-Geometry Spectrometer (λ̄ vs mass)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-c", "--config", type=Path, default=_default_cfg_path(script_dir),
                        help="Path to the YAML configuration file.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Base directory for output files.")
    parser.add_argument("--log", type=Path, default=None,
                        help="Path to log file.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    args = parser.parse_args(argv)

    logger = setup_logging(args.log, args.log_level)

    # Load + validate config
    if not args.config.is_file():
        logger.error(f"Config file not found at '{args.config}'")
        return 1

    try:
        raw_cfg = yaml.safe_load(args.config.read_text())
        if raw_cfg is None:
            logger.error(f"Config file is empty or invalid: {args.config}")
            return 1
        cfg = AppConfig.model_validate(raw_cfg)
    except Exception as e:
        logger.error(f"Error loading/validating config: {e}")
        return 1

    # Output directory
    out_dir = _new_outdir(script_dir, args.output_dir)
    (out_dir / "used_config.yml").write_text(yaml.dump(cfg.model_dump(), sort_keys=False))
    logger.info(f"Output directory: {out_dir}")

    # Assemble masses to process
    if cfg.mode == "single":
        masses = np.array([float(cfg.single_mass_MeV)], dtype=float)
    else:
        masses = mass_sweep_MeV(cfg.sweep)

    logger.info("Computing λ̄ for specified masses...")
    # Vector form for CSV + plot line (we still compute per-point for CSV with f, T too)
    radii_fm_vec = compute_series_lambda_bar_fm(masses)

    # CSV with detailed columns
    csv_path = out_dir / "results" / "mass_spectrum_results.csv"
    with csv_path.open("w") as f:
        f.write("mass_MeV,f_Hz,T_s,lambda_bar_fm\n")
        for m in masses:
            f_Hz, T_s, lam_m = compton_from_mass_MeV(float(m))
            lam_fm = lam_m * 1e15
            f.write(f"{m:.6f},{f_Hz:.8e},{T_s:.8e},{lam_fm:.8e}\n")
    logger.info(f"Results saved to: {csv_path}")

    # Plot
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=cfg.plotting.figsize)

    # Determine the range for the *prediction line* using all relevant masses
    known_masses = np.array([kp.mass for kp in cfg.known_particles], dtype=float) if cfg.known_particles else np.array([])
    all_masses_for_span = np.concatenate([masses, known_masses]) if known_masses.size else masses
    m_min = float(np.nanmin(all_masses_for_span[all_masses_for_span > 0])) if all_masses_for_span.size else 0.1
    m_max = float(np.nanmax(all_masses_for_span)) if all_masses_for_span.size else 1e6
    if not (math.isfinite(m_min) and math.isfinite(m_max) and m_max > m_min):
        m_min, m_max = 0.1, 1e6

    # Smooth line across the entire span (removes visual confusion near Planck mass)
    x_line = np.logspace(np.log10(m_min), np.log10(m_max), 400)
    y_line = compute_series_lambda_bar_fm(x_line)
    ax.plot(x_line, y_line, label="GEF Prediction: λ̄ ∝ 1/m", linewidth=2.0)

    # Overlay the sweep points only if that’s helpful (optional; comment out to keep it ultra clean)
    if cfg.mode == "sweep":
        ax.scatter(masses, radii_fm_vec, s=10, alpha=0.6, label="Sweep samples")

    # Overlay known particles
    logger.info("Overlaying known particles...")
    for kp in cfg.known_particles:
        mass_val = float(kp.mass)
        _, _, lam_m = compton_from_mass_MeV(mass_val)
        lam_fm = lam_m * 1e15
        # Legend label with sci-notation for huge masses
        if mass_val >= 1e6:
            label_text = f"{kp.name} ({mass_val:.2e} MeV)"
        else:
            label_text = f"{kp.name} ({mass_val:.1f} MeV)"
        ax.plot(mass_val, lam_fm, 'o', ms=10, label=label_text,
                color=kp.color, zorder=3, markeredgecolor='black')

    # Cosmetics
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(cfg.plotting.title, fontsize=18, weight="bold")
    ax.set_xlabel(cfg.plotting.xlabel, fontsize=14)
    ax.set_ylabel(cfg.plotting.ylabel, fontsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Place a compact legend
    leg = ax.legend(loc="best", frameon=True)
    for lh in leg.legend_handles:
        try:
            lh.set_alpha(0.9)
        except Exception:
            pass

    plot_path = out_dir / "results" / "mass_spectrum_plot.png"
    plt.savefig(plot_path, dpi=cfg.plotting.dpi, bbox_inches="tight")
    logger.info(f"Plot saved to: {plot_path}")

    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
