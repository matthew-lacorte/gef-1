#!/usr/bin/env python3
"""
GEF Stability Map (2D parameter sweep, refactored)

- Proper package imports (uses gef.geometry.hopfion_relaxer)
- Loguru + Rich progress integration
- Single-directory config structure (default: ./configs/default_run_stability_map.yml)
- Applies corrected δU/δφ logic via HopfionRelaxer (your fixed solver)
- Saves CSV + NPZ + heatmap into timestamped output directory

Config-driven sweep:
- Two axes, each maps to a solver config key (e.g. "mu_squared", "g_squared", "P_env", etc.)
- Optional transform per axis ("linear" or "log10")
- Stability metric = 1{converged} * exp(- mass_scale * max(mass, 0)) where
  mass = E_final - E_vac (analytic vacuum) and mass_scale is a config knob.

Example CLI:
    python run_stability_map.py -c ./configs/default_run_stability_map.yml
"""
from __future__ import annotations

import argparse
import datetime as _dt
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- Logging (Loguru preferred, fallback to stdlib) -------------------------
try:
    from loguru import logger
    _USING_LOGURU = True
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("gef.stability_map")
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(_h)
        logger.setLevel(logging.INFO)
    _USING_LOGURU = False

# ---- Rich progress (optional) ----------------------------------------------
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    _console = Console()
except Exception:  # pragma: no cover
    _console = None

# ---- Solver import ----------------------------------------------------------
try:
    from gef.geometry.hopfion_relaxer import HopfionRelaxer  # type: ignore
except Exception:  # pragma: no cover
    # Last-resort local import for dev/testing
    from hopfion_relaxer import HopfionRelaxer  # type: ignore

# =============================================================================
# Helpers and caches
# =============================================================================

# tiny in-process cache for numeric vacuum energy E0
VACUUM_CACHE: Dict[tuple, float] = {}

def analytic_vacuum_energy(config: Dict) -> float:
    """Analytic vacuum energy for uniform field (grad terms vanish).

    U_iso = -1/2 mu^2 phi0^2 + 1/4 lambda phi0^4
    U_P   = -P (1 - phi0^2)

    With phi0^2 = 0        if mu^2 <= 2P,
                   (mu^2-2P)/lambda otherwise.
    """
    mu2 = float(config["mu_squared"])     # μ²
    lam = float(config["lambda_val"])     # λ (>0)
    P   = float(config.get("P_env", 0.0)) # pressure
    dx  = float(config["dx"])             # lattice spacing
    n0, n1, n2, n3 = map(int, config["lattice_size"])  # type: ignore
    ncells = n0 * n1 * n2 * n3

    if mu2 <= 2.0 * P:
        phi0_sq = 0.0
    else:
        phi0_sq = max(0.0, (mu2 - 2.0 * P) / lam)

    U_iso = -0.5 * mu2 * phi0_sq + 0.25 * lam * (phi0_sq ** 2)
    U_P   = -P * (1.0 - phi0_sq)
    return (U_iso + U_P) * (dx ** 4) * ncells


# =============================================================================
# Dataclasses for config
# =============================================================================

@dataclass(frozen=True)
class AxisConfig:
    name: str
    key: str  # solver config key to set (e.g. "mu_squared", "g_squared", "P_env")
    start: float
    stop: float
    num_points: int
    transform: str = "linear"  # "linear" or "log10"

    def values(self) -> np.ndarray:
        if self.transform == "log10":
            # interpret start/stop as decades
            return np.logspace(self.start, self.stop, self.num_points)
        return np.linspace(self.start, self.stop, self.num_points)


# =============================================================================
# Core worker: evaluate one grid point
# =============================================================================

def _evaluate_point(i_j: Tuple[int, int], ax1_vals: np.ndarray, ax2_vals: np.ndarray,
                    base_cfg: Dict, axis1: AxisConfig, axis2: AxisConfig,
                    scoring_cfg: Dict, vacuum_cache: Dict, energy_check_only: bool) -> Dict:
    i, j = i_j
    v1, v2 = float(ax1_vals[i]), float(ax2_vals[j])

    cfg = dict(base_cfg)
    cfg[axis1.key] = v1
    cfg[axis2.key] = v2

    # Keep threads per worker modest to avoid oversubscription
    threads_per_process = int(base_cfg.get("threads_per_process", 1))
    os.environ["GEF_NUMBA_NUM_THREADS"] = str(threads_per_process)
    os.environ["NUMBA_NUM_THREADS"] = str(threads_per_process)

    # Build solver
    solver = HopfionRelaxer(cfg)

    # Seed and initial energy (honor either seed_nw or seed_topology)
    nw_seed = int(base_cfg.get("seed_topology", base_cfg.get("seed_nw", 1)))
    solver.initialize_field(nw=nw_seed)
    E0 = solver.compute_total_energy()

    # Iteration budget: opt-in overrides; otherwise use solver's internal max_iterations
    n_skip = base_cfg.get("relaxation_n_skip")
    n_iter = base_cfg.get("relaxation_n_iter")
    record_series = bool(base_cfg.get("record_series", False))

    # Early-exit mode: optionally skip relaxation for a super-fast probe
    if energy_check_only:
        E_final = E0
        converged = False
    else:
        _, E_final = solver.run_relaxation(n_skip=n_skip, n_iter=n_iter, record_series=record_series)
        converged = bool(np.isfinite(E_final))
        if not converged:
            E_final = np.nan

    # Numeric vacuum subtraction (E0_num): relax nw=0 with same params
    use_num_vac = bool(scoring_cfg.get("use_numeric_vacuum", False))
    cache_num_vac = bool(scoring_cfg.get("cache_numeric_vacuum", True))

    mu2 = float(cfg["mu_squared"])     # for cache key
    P   = float(cfg.get("P_env", 0.0))
    g2  = float(cfg.get("g_squared", 0.0))
    h2  = float(cfg.get("g_H_squared", cfg.get("h_squared", 0.0)))
    lam = float(cfg["lambda_val"]) 
    dx  = float(cfg["dx"]) 
    shape_key = tuple(int(x) for x in cfg["lattice_size"])  # type: ignore
    

    E0_num = np.nan
    if use_num_vac and converged:
        cache_key = (mu2, P, g2, h2, lam, dx, shape_key)
        if cache_num_vac and cache_key in vacuum_cache:
            E0_num = float(vacuum_cache[cache_key])
        else:
            cfg0 = dict(cfg)
            solver0 = HopfionRelaxer(cfg0)
            solver0.initialize_field(nw=0)
            # seed exact uniform vacuum to avoid long slides / hook issues
            try:
                phi0 = float(np.sqrt(solver0._vacuum_phi0_sq()))
            except Exception:
                phi0 = 1.0
            solver0.phi.fill(phi0)
            _, E0_tmp = solver0.run_relaxation(n_skip=n_skip, n_iter=n_iter, record_series=False)
            E0_num = float(E0_tmp) if np.isfinite(E0_tmp) else np.nan
            if cache_num_vac and np.isfinite(E0_num):
                vacuum_cache[cache_key] = E0_num

    # If numeric vacuum disabled or failed, fall back to analytic
    if not np.isfinite(E0_num):
        E0_num = float(analytic_vacuum_energy(cfg)) if np.isfinite(E_final) else np.nan

    # Mass density and bounded score
    n0, n1, n2, n3 = map(int, cfg["lattice_size"])  # type: ignore
    # normalization: default 4D volume; legacy 3D available via config
    norm = scoring_cfg.get("density_normalization", "4D")
    vol4 = float(n0 * n1 * n2 * n3) * (dx ** 4)
    vol3 = float(n0 * n1 * n2) * (dx ** 3)
    if norm == "3D":
        volume = vol3
    else:
        volume = vol4

    mass_density = (E_final - E0_num) / volume if (np.isfinite(E_final) and np.isfinite(E0_num) and volume > 0) else np.nan
    # clamp small negative due to numerics
    if np.isfinite(mass_density) and mass_density < 0:
        mass_density = 0.0

    m0 = float(scoring_cfg.get("score_ref_mass_density", 50.0))
    score = (1.0 / (1.0 + (mass_density / m0))) if np.isfinite(mass_density) else 0.0
    score = score if converged else 0.0

    # Diagnostics logging
    try:
        phi_max = float(np.max(solver.phi))
        phi_rms = float(np.sqrt(np.mean(solver.phi * solver.phi)))
        max_dphi = float(getattr(solver, "last_max_update", np.nan))
        breakdown = solver.compute_energy_breakdown() if converged else {}
        Lw = float(n3 * dx)
        logger.debug(
            "[%s=%g, %s=%g] E1=%.6e E0=%.6e md=%.6e score=%.4f phi_max=%.3f phi_rms=%.3f maxΔφ=%.3e Lw=%.3f breakdown=%s",
            axis1.name, v1, axis2.name, v2,
            float(E_final) if np.isfinite(E_final) else np.nan,
            float(E0_num) if np.isfinite(E0_num) else np.nan,
            float(mass_density) if np.isfinite(mass_density) else np.nan,
            float(score), phi_max, phi_rms, max_dphi, Lw, breakdown,
        )
    except Exception:
        pass

    return {
        "i": i,
        "j": j,
        axis1.name: v1,
        axis2.name: v2,
        "E_initial": float(E0) if np.isfinite(E0) else np.nan,
        "E_final": float(E_final) if np.isfinite(E_final) else np.nan,
        "E0_numeric": float(E0_num) if np.isfinite(E0_num) else np.nan,
        "E0_analytic": float(analytic_vacuum_energy(cfg)) if np.isfinite(E_final) else np.nan,
        "mass_density": float(mass_density) if np.isfinite(mass_density) else np.nan,
        "converged": bool(converged),
        "stability_metric": float(score),
        "phi_max": float(phi_max) if 'phi_max' in locals() else np.nan,
        "phi_rms": float(phi_rms) if 'phi_rms' in locals() else np.nan,
        "max_delta_phi": float(max_dphi) if 'max_dphi' in locals() else np.nan,
        "last_max_update": float(max_dphi) if 'max_dphi' in locals() else np.nan,
        "Lw": float(Lw) if 'Lw' in locals() else np.nan,
        "core_frac": float(np.mean(np.abs(solver.phi) < 0.5)) if converged else 0.0,
        "E_kin": float(breakdown.get("kinetic", np.nan)),
        "E_iso": float(breakdown.get("iso", np.nan)),
        "E_aniso": float(breakdown.get("aniso", np.nan)),
        "E_hook": float(breakdown.get("hook", np.nan)),
        "E_press": float(breakdown.get("pressure", np.nan)),

    }


# =============================================================================
# Plotting
# =============================================================================

def plot_stability_map(ax1_vals: np.ndarray, ax2_vals: np.ndarray, grid_vals: np.ndarray,
                       axis1: AxisConfig, axis2: AxisConfig, out_dir: Path,
                       title: str = "GEF Configuration Stability Map") -> Path:
    # 1D fallback: if one axis has only 1 value, render a line plot instead of pcolormesh
    if len(ax1_vals) == 1 or len(ax2_vals) == 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        if len(ax2_vals) == 1:
            y = grid_vals[:, 0]
            x = ax1_vals
            ax.plot(x, y, marker="o", lw=1.5)
            ax.set_xlabel(axis1.name)
            ax.set_ylabel("Stability Metric")
            ax.set_title(title + f" (1D along {axis1.name}, {axis2.name}={ax2_vals[0]:.3g})")
        else:
            y = grid_vals[0, :]
            x = ax2_vals
            ax.plot(x, y, marker="o", lw=1.5)
            ax.set_xlabel(axis2.name)
            ax.set_ylabel("Stability Metric")
            ax.set_title(title + f" (1D along {axis2.name}, {axis1.name}={ax1_vals[0]:.3g})")
        ax.grid(True, linestyle="--", alpha=0.6)
        path = out_dir / "stability_map.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    # 2D heatmap case
    fig, ax = plt.subplots(figsize=(10, 8))

    # Blue (low) -> Red (high)
    colors = [(0.0, 0.0, 0.8), (0.0, 0.8, 0.8), (0.8, 0.8, 0.0), (0.8, 0.0, 0.0)]
    cmap = LinearSegmentedColormap.from_list("stability_cmap", colors, N=256)

    # Use pcolormesh for correct alignment on irregular axes (log sweeps)
    X, Y = np.meshgrid(ax1_vals, ax2_vals, indexing="ij")
    h = ax.pcolormesh(X, Y, grid_vals, cmap=cmap, shading="auto")
    cbar = fig.colorbar(h, ax=ax)
    cbar.set_label("Stability Metric", rotation=90)

    ax.set_xlabel(axis1.name)
    ax.set_ylabel(axis2.name)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)

    path = out_dir / "stability_map.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# =============================================================================
# Orchestration
# =============================================================================

def _load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _setup_output_dir(root: Path) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / f"stability_map_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_sweep(cfg: Dict) -> Tuple[pd.DataFrame, Path, Path]:
    # Build axis configs
    ax1 = AxisConfig(**cfg["param1"])  # expects {name,key,start,stop,num_points,transform?}
    ax2 = AxisConfig(**cfg["param2"])  # ditto

    # Build axis values
    ax1_vals = ax1.values()
    ax2_vals = ax2.values()

    # Output directory
    out_root = Path(cfg.get("output_dir", "./outputs"))
    out_dir = _setup_output_dir(out_root)

    # Persist used config
    (out_dir / "used_config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Controls
    energy_check_only = bool(cfg.get("energy_check_only", False))
    scoring_cfg = dict(cfg.get("scoring", {}))

    # Multiprocessing knobs
    procs = int(cfg.get("processes", 0))
    if procs <= 0:
        procs = max(1, mp.cpu_count() - 1)
    threads_per_proc = int(cfg.get("threads_per_process", 1))

    os.environ["GEF_NUMBA_NUM_THREADS"] = str(threads_per_proc)
    os.environ["NUMBA_NUM_THREADS"] = str(threads_per_proc)

    # Build base solver config (things not controlled by axes)
    base_cfg = dict(cfg.get("solver", {}))
    # Ensure required keys exist
    for key in ("lattice_size", "dx", "mu_squared", "lambda_val"):
        if key not in base_cfg:
            raise KeyError(f"Missing required solver key: {key}")

    # Prepare work items
    ij_list = [(i, j) for i in range(len(ax1_vals)) for j in range(len(ax2_vals))]

    logger.info(f"Stability sweep: {ax1.name} × {ax2.name} with shapes {ax1_vals.shape[0]} × {ax2_vals.shape[0]}")
    logger.info(f"Using {procs} process(es) × {threads_per_proc} Numba thread(s)")

    worker = partial(_evaluate_point, ax1_vals=ax1_vals, ax2_vals=ax2_vals,
                     base_cfg=base_cfg, axis1=ax1, axis2=ax2,
                     scoring_cfg=scoring_cfg, vacuum_cache=VACUUM_CACHE,
                     energy_check_only=energy_check_only)

    results: List[Dict] = []

    if procs == 1:
        if _console:
            with Progress("{task.description}", BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%",
                           TimeElapsedColumn(), TimeRemainingColumn(), console=_console) as progress:
                task = progress.add_task("Scanning grid", total=len(ij_list))
                for ij in ij_list:
                    results.append(worker(ij))
                    progress.advance(task)
        else:
            for ij in ij_list:
                results.append(worker(ij))
    else:
        # Multiprocessing pool with Rich progress
        if _console:
            with Progress("{task.description}", BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%",
                           TimeElapsedColumn(), TimeRemainingColumn(), console=_console) as progress:
                task = progress.add_task("Scanning grid (mp)", total=len(ij_list))
                with mp.Pool(processes=procs) as pool:
                    for rec in pool.imap_unordered(worker, ij_list):
                        results.append(rec)
                        progress.advance(task)
        else:
            with mp.Pool(processes=procs) as pool:
                results = pool.map(worker, ij_list)

    # Assemble DataFrame and grids
    df = pd.DataFrame(results)
    df.sort_values(["i", "j"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    grid_metric = df["stability_metric"].to_numpy().reshape(len(ax1_vals), len(ax2_vals))
    grid_mass   = df.get("mass_density", pd.Series(np.nan, index=df.index)).to_numpy().reshape(len(ax1_vals), len(ax2_vals))
    grid_conv   = df["converged"].to_numpy().reshape(len(ax1_vals), len(ax2_vals))

    # Save raw arrays
    np.savez(out_dir / "stability_data.npz",
             axis1_vals=ax1_vals, axis2_vals=ax2_vals,
             stability_metric=grid_metric, mass_density=grid_mass, converged=grid_conv)

    # Save CSV (one row per grid point)
    csv_path = out_dir / "stability_results.csv"
    df.to_csv(csv_path, index=False)

    # Plot heatmap
    plot_path = plot_stability_map(ax1_vals, ax2_vals, grid_metric, ax1, ax2, out_dir)

    logger.info(f"Stability map saved to {plot_path}")
    logger.info(f"Tabular results saved to {csv_path}")

    return df, csv_path, plot_path


# =============================================================================
# CLI
# =============================================================================

def _default_cfg_path(script_dir: Path) -> Path:
    return script_dir / "configs" / "default_run_stability_map.yml"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="run_stability_map",
                                     description="Run 2D parameter sweep to build a GEF stability map")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("-c", "--config", type=Path, default=_default_cfg_path(script_dir), help="YAML config path")
    args = parser.parse_args(argv)

    try:
        cfg = _load_config(args.config)
    except Exception:
        logger.exception(f"Failed to load config: {args.config}")
        return 1

    try:
        run_sweep(cfg)
        print("\nStability sweep completed successfully!")
        return 0
    except Exception:
        logger.exception("Stability sweep failed with a critical error.")
        return 1


if __name__ == "__main__":
    if sys.platform != "linux":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    sys.exit(main())
