#!/usr/bin/env python3
"""
GEF κ-Flow Simulator

Local time-flow (kappa) from the GEF ansatz:
    ||κ(r)|| = κ̄ * (1 - β * W(r)),  with  W(r) ≈ Σ_i M_i / r_i(ε)

We plot the normalized field κ/κ̄ = 1 - β W. Optionally overlay contours and
show a second panel of the wake potential W(r).

Config supports legacy keys: 'kappa_inf' -> 'kappa_bar', 'G_effective' -> 'beta'.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Config models (Pydantic v2)
# ─────────────────────────────────────────────────────────────────────────────

class MassSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mass: float = Field(..., gt=0, description="Mass parameter (dimensionless scale).")
    position: Tuple[float, float] = Field(..., description="(x, y) in window coordinates.")
    color: str | None = Field(None, description="Optional marker color for this mass.")

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid_size: int = Field(600, ge=16, description="Grid resolution along each axis.")
    window: Tuple[float, float, float, float] = Field(
        (-10.0, 10.0, -10.0, 10.0), description="[x_min, x_max, y_min, y_max]"
    )
    masses: List[MassSpec] = Field(default_factory=list)
    beta: float = Field(0.05, ge=0.0, description="Wake coupling β (legacy: G_effective).")
    kappa_bar: float = Field(1.0, gt=0.0, description="Normalization κ̄.")
    softening_epsilon: float = Field(1e-6, gt=0.0, description="Softening to avoid r=0.")
    clip_min: float = Field(0.0, description="Minimum of κ/κ̄ for visualization.")
    clip_max: float = Field(1.0, description="Maximum of κ/κ̄ for visualization.")

    @field_validator("window")
    @classmethod
    def _validate_window(cls, v):
        x0, x1, y0, y1 = v
        if not (x1 > x0 and y1 > y0):
            raise ValueError("window must satisfy x_max > x_min and y_max > y_min")
        return v

    @field_validator("clip_max")
    @classmethod
    def _clip_order(cls, v, info):
        mn = info.data.get("clip_min", 0.0)
        if v <= mn:
            raise ValueError("clip_max must be > clip_min")
        return v

class ContourConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    show: bool = True
    levels: int = 10
    color: str = "white"
    linewidth: float = 0.8
    alpha: float = 0.6

class PotentialPanelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    show: bool = True
    cmap: str = "viridis"
    scale: Literal["linear", "log"] = "linear"
    # If vmin/vmax are None, use robust percentiles
    vmin: float | None = None
    vmax: float | None = None
    clip_percentiles: Tuple[float, float] = (2.0, 98.0)

class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dpi: int = 200
    figsize: Tuple[float, float] = (14.0, 10.0)
    cmap: str = "magma"
    interp: Literal["nearest", "bilinear", "bicubic"] = "bilinear"
    title_kappa: str = "GEF Time Dilation (κ/κ̄)"
    title_potential: str = "Wake Potential W(r)"
    facecolor: str = "black"
    show_masses: bool = True
    base_marker_size: float = 120.0
    marker_scale: float = 600.0
    mass_size_exponent: float = 1.0 / 3.0
    contours: ContourConfig = ContourConfig()
    potential_panel: PotentialPanelConfig = PotentialPanelConfig()

class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    simulation: SimulationConfig = SimulationConfig()
    plotting: PlotConfig = PlotConfig()
    output_dir: str | None = None  # optional override


# ─────────────────────────────────────────────────────────────────────────────
# Legacy key migration (pre-validate)
# ─────────────────────────────────────────────────────────────────────────────

def _migrate_legacy_keys(raw: dict) -> dict:
    """Map legacy keys to new names in-place (returns the dict)."""
    if not isinstance(raw, dict):
        return raw
    sim = raw.get("simulation")
    if isinstance(sim, dict):
        if "kappa_inf" in sim and "kappa_bar" not in sim:
            sim["kappa_bar"] = sim.pop("kappa_inf")
        if "G_effective" in sim and "beta" not in sim:
            sim["beta"] = sim.pop("G_effective")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", logfile: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("gef.k_flow")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
        logger.addHandler(sh)

    if logfile and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(str(logfile))
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
        logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────

def compute_wake_potential(cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute wake potential W on a (g x g) grid and return (xx, yy, W).
    W(r) ≈ Σ_i M_i / sqrt((x - x_i)^2 + (y - y_i)^2 + ε^2)
    """
    g = cfg.grid_size
    x0, x1, y0, y1 = cfg.window
    eps2 = float(cfg.softening_epsilon) ** 2

    x = np.linspace(x0, x1, g)
    y = np.linspace(y0, y1, g)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    if len(cfg.masses) == 0:
        W = np.zeros_like(xx)
        return xx, yy, W

    M = np.array([m.mass for m in cfg.masses], dtype=float)              # (n,)
    PX = np.array([m.position[0] for m in cfg.masses], dtype=float)      # (n,)
    PY = np.array([m.position[1] for m in cfg.masses], dtype=float)      # (n,)

    dx = xx[..., None] - PX[None, None, :]  # (g, g, n)
    dy = yy[..., None] - PY[None, None, :]
    r = np.sqrt(dx * dx + dy * dy + eps2)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.sum(M[None, None, :] / r, axis=-1)  # (g, g)
    return xx, yy, W


def run_simulation(cfg: SimulationConfig, logger: logging.Logger | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute κ/κ̄ field and return (kappa_ratio, kappa_raw, W).
    κ/κ̄ = 1 - β W,  κ = κ̄ * (κ/κ̄)
    """
    xx, yy, W = compute_wake_potential(cfg)
    kappa_ratio = 1.0 - float(cfg.beta) * W
    kappa_ratio = np.clip(kappa_ratio, cfg.clip_min, cfg.clip_max)
    kappa_raw = float(cfg.kappa_bar) * kappa_ratio

    if logger:
        logger.debug(
            f"Stats κ/κ̄: min={np.min(kappa_ratio):.4f}, max={np.max(kappa_ratio):.4f}, mean={np.mean(kappa_ratio):.4f}"
        )
        logger.debug(
            f"Stats  W : min={np.min(W):.4e}, max={np.max(W):.4e}, mean={np.mean(W):.4e}"
        )
    return kappa_ratio, kappa_raw, W


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _mass_marker_sizes(masses: list[MassSpec], base: float, scale: float, expo: float) -> np.ndarray:
    vals = np.array([m.mass for m in masses], dtype=float)
    if vals.size == 0:
        return np.array([])
    m_exp = np.power(vals, float(expo))
    denom = np.max(m_exp) if np.max(m_exp) > 0 else 1.0
    return base + scale * (m_exp / denom)

def _robust_limits(arr: np.ndarray, pcts: Tuple[float, float]) -> tuple[float, float]:
    p_lo, p_hi = pcts
    vmin = float(np.nanpercentile(arr, p_lo))
    vmax = float(np.nanpercentile(arr, p_hi))
    if not (math.isfinite(vmin) and math.isfinite(vmax) and vmax > vmin):
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    return vmin, vmax

def plot_results(
    kappa_ratio: np.ndarray,
    W: np.ndarray,
    app_cfg: AppConfig,
    out_png: Path,
    logger: logging.Logger | None = None,
) -> None:
    sim = app_cfg.simulation
    pc = app_cfg.plotting
    show_potential = pc.potential_panel.show

    if logger:
        logger.info("Rendering visualization...")

    # Decide figure layout
    ncols = 2 if show_potential else 1
    fig, axes = plt.subplots(
        1, ncols,
        figsize=pc.figsize,
        facecolor=pc.facecolor,
        squeeze=False
    )
    ax_k = axes[0, 0]
    ax_p = axes[0, 1] if show_potential else None

    # Panel 1: κ/κ̄
    im1 = ax_k.imshow(
        kappa_ratio,
        cmap=pc.cmap,
        extent=sim.window,
        origin="lower",
        interpolation=pc.interp,
        vmin=sim.clip_min,
        vmax=sim.clip_max,
    )
    cbar1 = fig.colorbar(im1, ax=ax_k, fraction=0.046, pad=0.04)
    tick_col = "white" if pc.facecolor == "black" else "black"
    cbar1.set_label("Local Time Flow (κ/κ̄)", color=tick_col, fontsize=14)
    cbar1.ax.yaxis.set_tick_params(color=tick_col)
    for t in cbar1.ax.get_yticklabels():
        t.set_color(tick_col)

    # Contours on κ/κ̄ (optional)
    if pc.contours.show:
        levels = np.linspace(sim.clip_min, sim.clip_max, pc.contours.levels)
        ax_k.contour(
            np.linspace(sim.window[0], sim.window[1], kappa_ratio.shape[1]),
            np.linspace(sim.window[2], sim.window[3], kappa_ratio.shape[0]),
            kappa_ratio,
            levels=levels,
            colors=pc.contours.color,
            linewidths=pc.contours.linewidth,
            alpha=pc.contours.alpha,
        )

    # Mass markers on κ/κ̄
    if pc.show_masses and len(sim.masses) > 0:
        sizes = _mass_marker_sizes(sim.masses, pc.base_marker_size, pc.marker_scale, pc.mass_size_exponent)
        for i, m in enumerate(sim.masses):
            mx, my = m.position
            ax_k.scatter(
                mx, my,
                s=float(sizes[i]),
                c=m.color or "white",
                edgecolors="red",
                linewidths=1.5,
                alpha=0.9,
                zorder=3,
            )

    ax_k.set_title(pc.title_kappa, color=tick_col, fontsize=20, pad=20)
    ax_k.set_aspect("equal")
    ax_k.axis("off")

    # Panel 2: W(r) (optional)
    if show_potential and ax_p is not None:
        pp = pc.potential_panel
        if pp.vmin is None or pp.vmax is None:
            vmin, vmax = _robust_limits(W, pp.clip_percentiles)
        else:
            vmin, vmax = pp.vmin, pp.vmax

        norm = LogNorm(vmin=max(vmin, 1e-12), vmax=vmax) if pp.scale == "log" else None

        im2 = ax_p.imshow(
            W,
            cmap=pp.cmap,
            extent=sim.window,
            origin="lower",
            interpolation=pc.interp,
            vmin=None if pp.scale == "log" else vmin,
            vmax=None if pp.scale == "log" else vmax,
            norm=norm,
        )
        cbar2 = fig.colorbar(im2, ax=ax_p, fraction=0.046, pad=0.04)
        cbar2.set_label("Wake Potential W(r) (arb.)", color=tick_col, fontsize=14)
        cbar2.ax.yaxis.set_tick_params(color=tick_col)
        for t in cbar2.ax.get_yticklabels():
            t.set_color(tick_col)

        # Mass markers on W panel too
        if pc.show_masses and len(sim.masses) > 0:
            sizes = _mass_marker_sizes(sim.masses, pc.base_marker_size, pc.marker_scale, pc.mass_size_exponent)
            for i, m in enumerate(sim.masses):
                mx, my = m.position
                ax_p.scatter(
                    mx, my,
                    s=float(sizes[i]),
                    c=m.color or "white",
                    edgecolors="red",
                    linewidths=1.5,
                    alpha=0.9,
                    zorder=3,
                )

        ax_p.set_title(pc.title_potential, color=tick_col, fontsize=20, pad=20)
        ax_p.set_aspect("equal")
        ax_p.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, facecolor=pc.facecolor, dpi=pc.dpi, bbox_inches="tight")
    plt.close(fig)

    if logger:
        logger.info(f"Visualization saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI / Main
# ─────────────────────────────────────────────────────────────────────────────

def _default_cfg_path(script_dir: Path) -> Path:
    return script_dir / "configs" / "default_k_flow_simulator.yml"

def _default_out_base(script_dir: Path) -> Path:
    # keep outputs folder alphabetically after configs/
    return script_dir / "_outputs"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="GEF κ-Flow Simulator.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("-c", "--config", type=Path, default=_default_cfg_path(script_dir),
                   help="Path to YAML config.")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="Base output directory (default: SCRIPT_DIR/_outputs).")
    p.add_argument("--save-npy", action="store_true",
                   help="Also save raw arrays (kappa_ratio, kappa_raw, W) as .npy.")
    p.add_argument("--log-level", type=str, default="INFO",
                   help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Optional log file.")
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_level, args.log_file)

    script_dir = Path(__file__).resolve().parent

    # Load + validate config
    if not args.config.is_file():
        logger.error(f"Config file not found at '{args.config}'")
        return 1
    try:
        raw = yaml.safe_load(args.config.read_text())
        if raw is None:
            raise ValueError("Config is empty")
        raw = _migrate_legacy_keys(raw)
        cfg = AppConfig.model_validate(raw)
    except Exception as e:
        logger.error(f"Error loading/validating config: {e}")
        return 1

    # Output directory
    base_out = Path(cfg.output_dir) if cfg.output_dir else (args.output_dir or _default_out_base(script_dir))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(base_out) / f"{timestamp}").resolve()
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    # Save effective config
    used_cfg_path = out_dir / "used_config.yml"
    used_cfg_path.write_text(yaml.dump(cfg.model_dump(), sort_keys=False))
    logger.info(f"Output directory: {out_dir}")

    # Run simulation
    t0 = time.perf_counter()
    kappa_ratio, kappa_raw, W = run_simulation(cfg.simulation, logger=logger)
    dt = time.perf_counter() - t0
    logger.info(f"Simulation completed in {dt:.3f} s")

    # Save artifacts
    png_path = out_dir / "results" / "k_flow.png"
    plot_results(kappa_ratio, W, cfg, png_path, logger=logger)

    if args.save_npy:
        np.save(out_dir / "results" / "kappa_ratio.npy", kappa_ratio)
        np.save(out_dir / "results" / "kappa_raw.npy", kappa_raw)
        np.save(out_dir / "results" / "W.npy", W)
        logger.info("Saved arrays: kappa_ratio.npy, kappa_raw.npy, W.npy")

    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
