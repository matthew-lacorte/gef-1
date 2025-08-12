#!/usr/bin/env python3
"""
GEF κ-Flow Simulator

Visualizes local time-flow rate around massive bodies from the GEF ansatz:
    ||κ(r)|| = κ_inf * (1 - β * W(r)),   with   W(r) ~ Σ_i M_i / r_i

We plot the normalized field κ/κ_inf = (1 - β * W).
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
import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Config models (Pydantic v2)
# ─────────────────────────────────────────────────────────────────────────────

class MassSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mass: float = Field(..., gt=0, description="Mass parameter (dimensionless scale).")
    position: Tuple[float, float] = Field(..., description="(x, y) position in window coordinates.")
    color: str | None = Field(None, description="Optional marker color for this mass.")

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid_size: int = Field(600, ge=16, description="Grid resolution along each axis.")
    window: Tuple[float, float, float, float] = Field(
        (-10.0, 10.0, -10.0, 10.0), description="[x_min, x_max, y_min, y_max]"
    )
    masses: List[MassSpec] = Field(default_factory=list)
    beta: float = Field(0.05, ge=0.0, description="Wake coupling β (legacy: G_effective).")
    kappa_inf: float = Field(1.0, gt=0.0, description="Normalization κ_∞.")
    softening_epsilon: float = Field(1e-6, gt=0.0, description="Softening to avoid r=0.")
    clip_min: float = Field(0.0, description="Minimum of κ/κ∞ for visualization.")
    clip_max: float = Field(1.0, description="Maximum of κ/κ∞ for visualization.")

    # Back-compat: allow 'G_effective' to map into 'beta'
    @field_validator("beta", mode="before")
    @classmethod
    def _coerce_beta(cls, v, info):
        if v is not None:
            return v
        # check raw input
        raw = info.data.get("__root__") or {}
        legacy = raw.get("G_effective") if isinstance(raw, dict) else None
        return legacy if legacy is not None else 0.05

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

class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dpi: int = 200
    figsize: Tuple[float, float] = (12.0, 10.0)
    cmap: str = "magma"
    interp: Literal["nearest", "bilinear", "bicubic"] = "bilinear"
    title: str = "GEF Time Dilation Well"
    facecolor: str = "black"
    show_masses: bool = True
    base_marker_size: float = 120.0
    marker_scale: float = 600.0
    mass_size_exponent: float = 1.0 / 3.0  # visually: ~radius ∝ mass^(1/3)

class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    simulation: SimulationConfig = SimulationConfig()
    plotting: PlotConfig = PlotConfig()
    output_dir: str | None = None  # optional override


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

def run_simulation(cfg: SimulationConfig, logger: logging.Logger | None = None) -> np.ndarray:
    """
    Returns κ/κ_inf field (dimensionless), clipped to [clip_min, clip_max].
    W(r) ≈ Σ_i M_i / sqrt((x - x_i)^2 + (y - y_i)^2 + ε^2)
    κ/κ_inf = 1 - β * W
    """
    g = cfg.grid_size
    x0, x1, y0, y1 = cfg.window
    eps2 = float(cfg.softening_epsilon) ** 2

    # Build grid
    x = np.linspace(x0, x1, g)
    y = np.linspace(y0, y1, g)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    # Vectorized potential from all masses
    if len(cfg.masses) == 0:
        W = np.zeros_like(xx)
    else:
        M = np.array([m.mass for m in cfg.masses], dtype=float)              # (n,)
        PX = np.array([m.position[0] for m in cfg.masses], dtype=float)      # (n,)
        PY = np.array([m.position[1] for m in cfg.masses], dtype=float)      # (n,)

        dx = xx[..., None] - PX[None, None, :]  # (g, g, n)
        dy = yy[..., None] - PY[None, None, :]
        r = np.sqrt(dx * dx + dy * dy + eps2)   # soft-squared distance
        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.sum(M[None, None, :] / r, axis=-1)  # (g, g)

    kappa_ratio = 1.0 - float(cfg.beta) * W

    # Clip for visualization / physical floor (no negative time flow)
    kappa_ratio = np.clip(kappa_ratio, cfg.clip_min, cfg.clip_max)

    if logger:
        logger.debug(
            f"Simulation stats: min={np.min(kappa_ratio):.4f}, "
            f"max={np.max(kappa_ratio):.4f}, mean={np.mean(kappa_ratio):.4f}"
        )
    return kappa_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    kappa_ratio: np.ndarray,
    app_cfg: AppConfig,
    out_png: Path,
    logger: logging.Logger | None = None,
) -> None:
    sim = app_cfg.simulation
    pc = app_cfg.plotting

    if logger:
        logger.info("Rendering visualization...")

    fig, ax = plt.subplots(figsize=pc.figsize, facecolor=pc.facecolor)
    im = ax.imshow(
        kappa_ratio,
        cmap=pc.cmap,
        extent=sim.window,
        origin="lower",
        interpolation=pc.interp,
        vmin=sim.clip_min,
        vmax=sim.clip_max,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Local Time Flow Rate (κ/κ∞)", color="white" if pc.facecolor == "black" else "black", fontsize=14)
    # Colorbar tick color to match dark facecolor
    tick_col = "white" if pc.facecolor == "black" else "black"
    cbar.ax.yaxis.set_tick_params(color=tick_col)
    for t in cbar.ax.get_yticklabels():
        t.set_color(tick_col)

    # Mass markers
    if pc.show_masses and len(sim.masses) > 0:
        mass_vals = np.array([m.mass for m in sim.masses], dtype=float)
        m_exp = np.power(mass_vals, float(pc.mass_size_exponent))
        denom = np.max(m_exp) if np.max(m_exp) > 0 else 1.0
        sizes = pc.base_marker_size + pc.marker_scale * (m_exp / denom)

        for i, m in enumerate(sim.masses):
            mx, my = m.position
            ax.scatter(
                mx, my,
                s=float(sizes[i]),
                c=m.color or "white",
                edgecolors="red",
                linewidths=1.5,
                alpha=0.9,
                zorder=3,
            )

    ax.set_title(pc.title, color=tick_col, fontsize=20, pad=20)
    ax.set_aspect("equal")
    ax.axis("off")

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
    # your request: keep outputs folder alphabetically after configs/
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
                   help="Also save raw κ/κ∞ array as .npy.")
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
    field = run_simulation(cfg.simulation, logger=logger)
    dt = time.perf_counter() - t0
    logger.info(f"Simulation completed in {dt:.3f} s")

    # Save artifacts
    png_path = out_dir / "results" / "k_flow.png"
    plot_results(field, cfg, png_path, logger=logger)

    if args.save_npy:
        npy_path = out_dir / "results" / "k_flow.npy"
        np.save(npy_path, field)
        logger.info(f"Raw grid saved: {npy_path}")

    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
