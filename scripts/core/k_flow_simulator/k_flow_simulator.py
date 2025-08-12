#!/usr/bin/env python3
"""
GEF κ-Flow Simulator (with physical mapping, auto-β, Δ panel)

Local time-flow (kappa) from the GEF ansatz:
    ||κ(r)|| = κ̄ * (1 - β * W(r)),  with  W(r) ≈ Σ_i M_i / r_i(ε)

We render:
  - κ/κ̄ (time-flow), with optional contours and zero-level dashed boundary
  - W(r) (wake potential), optional
  - Δ = β W = 1 - κ/κ̄ (drop), optional

Physical mapping (optional):
  Choose unit scales (M0, L0). If beta_source='from_units',
    β = (G/c^2) * (M0/L0)
Masses and window can be provided in dimensionless units or in SI via
'mass_kg' and 'window_m' (converted by M0 and L0).
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
# SI constants (exact/standard)
# ─────────────────────────────────────────────────────────────────────────────

C_SI = 299_792_458.0               # m/s (exact)
G_SI = 6.67430e-11                  # m^3 kg^-1 s^-2 (2018 CODATA)


# ─────────────────────────────────────────────────────────────────────────────
# Config models (Pydantic v2)
# ─────────────────────────────────────────────────────────────────────────────

class UnitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["dimensionless", "physical"] = "dimensionless"
    # Only used if mode == "physical"
    M0_kg: float | None = Field(None, gt=0, description="Mass scale in kg.")
    L0_m: float | None = Field(None, gt=0, description="Length scale in m.")
    beta_source: Literal["config", "from_units"] = "config"
    beta_scale_factor: float = Field(1.0, gt=0.0, description="Optional ×scale after deriving β from units.")

    @field_validator("beta_source")
    @classmethod
    def _need_scales_if_from_units(cls, v, info):
        if v == "from_units":
            M0 = info.data.get("M0_kg")
            L0 = info.data.get("L0_m")
            if not (M0 and L0):
                raise ValueError("beta_source='from_units' requires M0_kg and L0_m.")
        return v


class MassSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Provide either 'mass' (dimensionless) OR 'mass_kg' (physical). If both,
    # 'mass_kg' wins in physical mode; 'mass' is used otherwise.
    mass: float | None = Field(None, gt=0)
    mass_kg: float | None = Field(None, gt=0)
    position: Tuple[float, float] | None = None       # dimensionless
    position_m: Tuple[float, float] | None = None     # physical meters
    color: str | None = Field(None)

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid_size: int = Field(600, ge=16)
    # Provide either 'window' (dimensionless [xmin,xmax,ymin,ymax]) or 'window_m' (meters).
    window: Tuple[float, float, float, float] | None = None
    window_m: Tuple[float, float, float, float] | None = None
    masses: List[MassSpec] = Field(default_factory=list)

    beta: float | None = Field(0.05, ge=0.0, description="Wake coupling β if beta_source='config'.")
    kappa_bar: float = Field(1.0, gt=0.0, description="Normalization κ̄.")
    softening_epsilon: float = Field(1e-6, gt=0.0, description="Softening to avoid r=0 (dimensionless).")
    clip_min: float = 0.0
    clip_max: float = 1.0

    # Auto-β to keep dynamic range
    auto_beta: bool = True
    auto_beta_target_min: float = Field(0.05, ge=0.0, le=0.5,
                                        description="Target minimum κ/κ̄ at the deepest point.")
    auto_beta_percentile: float = Field(99.9, gt=0, le=100,
                                        description="Use this W-percentile instead of absolute Wmax.")

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
    levels: int = 12
    color: str = "white"
    linewidth: float = 0.8
    alpha: float = 0.6
    show_zero_boundary: bool = True          # dashed contour for raw (preclip) κ/κ̄ = 0
    zero_color: str = "white"
    zero_ls: str = "--"
    zero_lw: float = 1.2
    zero_alpha: float = 0.9


class PotentialPanelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    show: bool = True
    cmap: str = "viridis"
    scale: Literal["linear", "log"] = "linear"
    vmin: float | None = None
    vmax: float | None = None
    clip_percentiles: Tuple[float, float] = (2.0, 98.0)


class DropPanelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    show: bool = True
    cmap: str = "plasma"
    vmax: float | None = None
    clip_percentiles: Tuple[float, float] = (2.0, 98.0)


class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dpi: int = 220
    figsize: Tuple[float, float] = (16.0, 9.0)
    cmap: str = "magma"
    interp: Literal["nearest", "bilinear", "bicubic"] = "bilinear"
    title_kappa: str = "GEF Time Dilation (κ/κ̄)"
    title_potential: str = "Wake Potential W(r)"
    title_drop: str = "Δ = β W"
    facecolor: str = "black"
    show_masses: bool = True
    base_marker_size: float = 120.0
    marker_scale: float = 600.0
    mass_size_exponent: float = 1.0 / 3.0
    contours: ContourConfig = ContourConfig()
    potential_panel: PotentialPanelConfig = PotentialPanelConfig()
    drop_panel: DropPanelConfig = DropPanelConfig()


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    units: UnitsConfig = UnitsConfig()
    simulation: SimulationConfig = SimulationConfig()
    plotting: PlotConfig = PlotConfig()
    output_dir: str | None = None  # optional override


# ─────────────────────────────────────────────────────────────────────────────
# Legacy key migration (pre-validate)
# ─────────────────────────────────────────────────────────────────────────────

def _migrate_legacy_keys(raw: dict) -> dict:
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
# Unit handling (dimensionless ↔ physical)
# ─────────────────────────────────────────────────────────────────────────────

def _to_dimless_window(sim: SimulationConfig, units: UnitsConfig) -> Tuple[float, float, float, float]:
    if sim.window is not None:
        x0, x1, y0, y1 = sim.window
        if not (x1 > x0 and y1 > y0):
            raise ValueError("window must satisfy x_max > x_min and y_max > y_min")
        return x0, x1, y0, y1
    if sim.window_m is not None:
        if units.mode != "physical":
            raise ValueError("window_m requires units.mode='physical'")
        L0 = units.L0_m or 1.0
        x0, x1, y0, y1 = sim.window_m
        if not (x1 > x0 and y1 > y0):
            raise ValueError("window_m must satisfy x_max > x_min and y_max > y_min")
        return x0 / L0, x1 / L0, y0 / L0, y1 / L0
    # default window
    return (-10.0, 10.0, -10.0, 10.0)


def _to_dimless_masses(masses: List[MassSpec], units: UnitsConfig) -> List[MassSpec]:
    out: List[MassSpec] = []
    if units.mode == "physical":
        M0 = units.M0_kg or 1.0
        L0 = units.L0_m or 1.0
        for m in masses:
            # mass
            if m.mass_kg is not None:
                mass = m.mass_kg / M0
            elif m.mass is not None:
                mass = m.mass
            else:
                raise ValueError("Each mass requires 'mass' or 'mass_kg'.")
            # position
            if m.position_m is not None:
                pos = (m.position_m[0] / L0, m.position_m[1] / L0)
            elif m.position is not None:
                pos = m.position
            else:
                raise ValueError("Each mass requires 'position' or 'position_m'.")
            out.append(MassSpec(mass=mass, position=pos, color=m.color))
        return out
    else:
        # dimensionless mode
        for m in masses:
            if m.mass is None or m.position is None:
                raise ValueError("In dimensionless mode, each mass needs 'mass' and 'position'.")
            out.append(MassSpec(mass=m.mass, position=m.position, color=m.color))
        return out


def _derive_beta_from_units(units: UnitsConfig) -> float:
    # β = (G/c^2)*(M0/L0)
    return units.beta_scale_factor * (G_SI / (C_SI ** 2)) * ((units.M0_kg or 1.0) / (units.L0_m or 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────

def compute_wake_potential(grid_size: int,
                           window: Tuple[float, float, float, float],
                           eps: float,
                           masses: List[MassSpec]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xx, yy, W) on a (g x g) grid, using softened 1/r."""
    g = grid_size
    x0, x1, y0, y1 = window
    eps2 = float(eps) ** 2

    x = np.linspace(x0, x1, g)
    y = np.linspace(y0, y1, g)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    if len(masses) == 0:
        return xx, yy, np.zeros_like(xx)

    M = np.array([m.mass for m in masses], dtype=float)              # (n,)
    PX = np.array([m.position[0] for m in masses], dtype=float)      # (n,)
    PY = np.array([m.position[1] for m in masses], dtype=float)      # (n,)

    dx = xx[..., None] - PX[None, None, :]
    dy = yy[..., None] - PY[None, None, :]
    r = np.sqrt(dx * dx + dy * dy + eps2)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.sum(M[None, None, :] / r, axis=-1)
    return xx, yy, W


def run_simulation(app_cfg: AppConfig, logger: logging.Logger | None = None):
    """
    Returns dict with fields:
      xx, yy, W, beta_base, beta_eff, kappa_ratio_raw, kappa_ratio, kappa_raw, drop
    """
    sim = app_cfg.simulation
    units = app_cfg.units

    # Unit conversion for window & masses
    window = _to_dimless_window(sim, units)
    masses_dimless = _to_dimless_masses(sim.masses, units)

    # Derive or use β
    if units.beta_source == "from_units":
        beta_base = _derive_beta_from_units(units)
    else:
        if sim.beta is None:
            raise ValueError("beta_source='config' requires simulation.beta")
        beta_base = float(sim.beta)

    # Wake potential
    xx, yy, W = compute_wake_potential(sim.grid_size, window, sim.softening_epsilon, masses_dimless)

    # Auto-β (reduce only, to preserve intended upper bound)
    W_ref = float(np.nanpercentile(W, sim.auto_beta_percentile)) if sim.auto_beta else float(np.nanmax(W))
    if W_ref <= 0.0:
        beta_eff = beta_base
    else:
        target_min = sim.auto_beta_target_min
        beta_cap = (1.0 - target_min) / W_ref
        beta_eff = min(beta_base, beta_cap) if sim.auto_beta else beta_base

    # κ/κ̄ before/after clipping
    kappa_ratio_raw = 1.0 - beta_eff * W
    kappa_ratio = np.clip(kappa_ratio_raw, sim.clip_min, sim.clip_max)
    kappa_raw = sim.kappa_bar * kappa_ratio
    drop = beta_eff * W  # Δ = 1 - κ/κ̄

    if logger:
        floor = np.mean(kappa_ratio <= sim.clip_min + 1e-12)
        ceil = np.mean(kappa_ratio >= sim.clip_max - 1e-12)
        logger.info(
            f"β_base={beta_base:.6g}, β_eff={beta_eff:.6g} "
            f"(auto={'on' if sim.auto_beta else 'off'}, W_ref@{sim.auto_beta_percentile:.3g}th={W_ref:.4g})"
        )
        logger.debug(
            f"κ/κ̄ raw stats: min={np.min(kappa_ratio_raw):.4f}, max={np.max(kappa_ratio_raw):.4f}, mean={np.mean(kappa_ratio_raw):.4f}"
        )
        logger.info(
            f"κ/κ̄ clipped stats: min={np.min(kappa_ratio):.4f}, max={np.max(kappa_ratio):.4f}, "
            f"mean={np.mean(kappa_ratio):.4f}, floor%={100*floor:.2f}, ceil%={100*ceil:.2f}"
        )

    return {
        "xx": xx, "yy": yy, "W": W,
        "beta_base": beta_base, "beta_eff": beta_eff,
        "kappa_ratio_raw": kappa_ratio_raw,
        "kappa_ratio": kappa_ratio,
        "kappa_raw": kappa_raw,
        "drop": drop,
        "masses_dimless": masses_dimless,
        "window": window,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _mass_marker_sizes(masses: list[MassSpec], base: float, scale: float, expo: float) -> np.ndarray:
    if not masses:
        return np.array([])
    vals = np.array([m.mass for m in masses], dtype=float)
    m_exp = np.power(vals, float(expo))
    denom = np.max(m_exp) if np.max(m_exp) > 0 else 1.0
    return base + scale * (m_exp / denom)

def _robust_limits(arr: np.ndarray, pcts: Tuple[float, float]) -> tuple[float, float]:
    lo, hi = pcts
    vmin = float(np.nanpercentile(arr, lo))
    vmax = float(np.nanpercentile(arr, hi))
    if not (math.isfinite(vmin) and math.isfinite(vmax) and vmax > vmin):
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    return vmin, vmax

def plot_results(res: dict, app_cfg: AppConfig, out_png: Path, logger: logging.Logger | None = None) -> None:
    sim, pc = app_cfg.simulation, app_cfg.plotting
    show_p = pc.potential_panel.show
    show_d = pc.drop_panel.show

    ncols = 1 + int(show_p) + int(show_d)
    fig, axes = plt.subplots(1, ncols, figsize=pc.figsize, facecolor=pc.facecolor, squeeze=False)

    tick_col = "white" if pc.facecolor == "black" else "black"

    # Panel A: κ/κ̄
    axA = axes[0, 0]
    imA = axA.imshow(
        res["kappa_ratio"], cmap=pc.cmap, extent=res["window"], origin="lower",
        interpolation=pc.interp, vmin=sim.clip_min, vmax=sim.clip_max
    )
    cbarA = fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)
    cbarA.set_label("Local Time Flow (κ/κ̄)", color=tick_col, fontsize=14)
    cbarA.ax.yaxis.set_tick_params(color=tick_col)
    for t in cbarA.ax.get_yticklabels(): t.set_color(tick_col)

    # Contours on κ/κ̄
    if pc.contours.show:
        levels = np.linspace(sim.clip_min, sim.clip_max, pc.contours.levels)
        xlin = np.linspace(res["window"][0], res["window"][1], res["kappa_ratio"].shape[1])
        ylin = np.linspace(res["window"][2], res["window"][3], res["kappa_ratio"].shape[0])
        axA.contour(xlin, ylin, res["kappa_ratio"], levels=levels,
                    colors=pc.contours.color, linewidths=pc.contours.linewidth, alpha=pc.contours.alpha)
    # Zero boundary of raw κ/κ̄
    if pc.contours.show_zero_boundary:
        raw = res["kappa_ratio_raw"]
        if np.nanmin(raw) < 0 < np.nanmax(raw):
            xlin = np.linspace(res["window"][0], res["window"][1], raw.shape[1])
            ylin = np.linspace(res["window"][2], res["window"][3], raw.shape[0])
            axA.contour(xlin, ylin, raw, levels=[0.0], colors=[pc.contours.zero_color],
                        linewidths=[pc.contours.zero_lw], linestyles=[pc.contours.zero_ls],
                        alpha=pc.contours.zero_alpha)

    # Mass markers
    if pc.show_masses and res["masses_dimless"]:
        sizes = _mass_marker_sizes(res["masses_dimless"], pc.base_marker_size, pc.marker_scale, pc.mass_size_exponent)
        for i, m in enumerate(res["masses_dimless"]):
            axA.scatter(m.position[0], m.position[1], s=float(sizes[i]),
                        c=m.color or "white", edgecolors="red", linewidths=1.5, alpha=0.9, zorder=3)

    # Annotated title with β
    beta_base, beta_eff = res["beta_base"], res["beta_eff"]
    beta_note = f"β={beta_eff:.3g}" + (f" (base {beta_base:.3g}, auto)" if abs(beta_eff - beta_base) > 1e-15 else "")
    axA.set_title(f"{pc.title_kappa}   {beta_note}", color=tick_col, fontsize=20, pad=20)
    axA.set_aspect("equal"); axA.axis("off")

    col = 1

    # Panel B: W
    if show_p:
        axB = axes[0, col]; col += 1
        W = res["W"]
        pp = pc.potential_panel
        if pp.vmin is None or pp.vmax is None:
            vmin, vmax = _robust_limits(W, pp.clip_percentiles)
        else:
            vmin, vmax = pp.vmin, pp.vmax
        norm = LogNorm(vmin=max(vmin, 1e-12), vmax=vmax) if pp.scale == "log" else None
        imB = axB.imshow(W, cmap=pp.cmap, extent=res["window"], origin="lower",
                         interpolation=pc.interp,
                         vmin=None if pp.scale == "log" else vmin,
                         vmax=None if pp.scale == "log" else vmax,
                         norm=norm)
        cbarB = fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)
        cbarB.set_label("Wake Potential W(r) (arb.)", color=tick_col, fontsize=14)
        cbarB.ax.yaxis.set_tick_params(color=tick_col)
        for t in cbarB.ax.get_yticklabels(): t.set_color(tick_col)
        if pc.show_masses and res["masses_dimless"]:
            sizes = _mass_marker_sizes(res["masses_dimless"], pc.base_marker_size, pc.marker_scale, pc.mass_size_exponent)
            for i, m in enumerate(res["masses_dimless"]):
                axB.scatter(m.position[0], m.position[1], s=float(sizes[i]),
                            c=m.color or "white", edgecolors="red", linewidths=1.5, alpha=0.9, zorder=3)
        axB.set_title(pc.title_potential, color=tick_col, fontsize=20, pad=20)
        axB.set_aspect("equal"); axB.axis("off")

    # Panel C: Δ = β W
    if show_d:
        axC = axes[0, col]; col += 1
        drop = res["drop"]
        dp = pc.drop_panel
        if dp.vmax is None:
            vmin, vmax = _robust_limits(drop, dp.clip_percentiles)
            vmax = vmax if vmax > 0 else 1.0
        else:
            vmin, vmax = 0.0, dp.vmax
        imC = axC.imshow(drop, cmap=dp.cmap, extent=res["window"], origin="lower",
                         interpolation=pc.interp, vmin=0.0, vmax=vmax)
        cbarC = fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04)
        cbarC.set_label("Δ = β W (arb.)", color=tick_col, fontsize=14)
        cbarC.ax.yaxis.set_tick_params(color=tick_col)
        for t in cbarC.ax.get_yticklabels(): t.set_color(tick_col)
        if pc.show_masses and res["masses_dimless"]:
            sizes = _mass_marker_sizes(res["masses_dimless"], pc.base_marker_size, pc.marker_scale, pc.mass_size_exponent)
            for i, m in enumerate(res["masses_dimless"]):
                axC.scatter(m.position[0], m.position[1], s=float(sizes[i]),
                            c=m.color or "white", edgecolors="red", linewidths=1.5, alpha=0.9, zorder=3)
        axC.set_title(pc.title_drop, color=tick_col, fontsize=20, pad=20)
        axC.set_aspect("equal"); axC.axis("off")

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
    return script_dir / "_outputs"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="GEF κ-Flow Simulator (with physical mapping, auto-β, Δ panel).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("-c", "--config", type=Path, default=_default_cfg_path(script_dir),
                   help="Path to YAML config.")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="Base output directory (default: SCRIPT_DIR/_outputs).")
    p.add_argument("--save-npy", action="store_true",
                   help="Save arrays: kappa_ratio, kappa_ratio_raw, kappa_raw, W, drop.")
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

    # Output dir
    base_out = Path(cfg.output_dir) if cfg.output_dir else (args.output_dir or _default_out_base(script_dir))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(base_out) / timestamp).resolve()
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    # Save effective config
    (out_dir / "used_config.yml").write_text(yaml.dump(cfg.model_dump(), sort_keys=False))
    logger.info(f"Output directory: {out_dir}")

    # Run
    t0 = time.perf_counter()
    res = run_simulation(cfg, logger=logger)
    logger.info(f"Simulation completed in {time.perf_counter() - t0:.3f} s")

    # Plot
    png_path = out_dir / "results" / "k_flow.png"
    plot_results(res, cfg, png_path, logger=logger)

    # Save arrays
    if args.save_npy:
        np.save(out_dir / "results" / "kappa_ratio.npy", res["kappa_ratio"])
        np.save(out_dir / "results" / "kappa_ratio_raw.npy", res["kappa_ratio_raw"])
        np.save(out_dir / "results" / "kappa_raw.npy", res["kappa_raw"])
        np.save(out_dir / "results" / "W.npy", res["W"])
        np.save(out_dir / "results" / "drop.npy", res["drop"])
        logger.info("Saved arrays: kappa_ratio.npy, kappa_ratio_raw.npy, kappa_raw.npy, W.npy, drop.npy")

    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
