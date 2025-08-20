#!/usr/bin/env python3
"""
GEF N_w Mass Spectrum Analysis (refactored)

What changed vs. your original:
- Physics-aware baseline: computes ANALYTIC vacuum energy from (μ², λ, P) and lattice volume.
- Safer resonance model: supports per-peak types (gaussian/lorentzian), clamps g² ≥ 0 with warnings.
- Robust parallelism: clean environment/thread knobs per process, Windows/macOS spawn-safe.
- Better logging & error paths: full tracebacks; records convergence; reproducible output folder.
- Plotter cleanup: pure Matplotlib (no seaborn), deduped legend, stable/unstable handling.
- Works with the refactored HopfionRelaxer (energy backtracking, sign fixes, parameter checks).

NOTE (topology): This analysis still treats a real-scalar "Hopfion-like" seed; if true Hopf charge
is required, migrate to O(3)/CP¹ and update HopfionRelaxer accordingly.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# --- Robust Package Imports ---
try:
    from gef.geometry.hopfion_relaxer import HopfionRelaxer, calculate_full_potential_derivative
    from gef.core.logging import logger
except Exception:  # pragma: no cover
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from gef.geometry.hopfion_relaxer import HopfionRelaxer, calculate_full_potential_derivative
        from gef.core.logging import logger
    except Exception:  # Last resort fallback
        from hopfion_relaxer import HopfionRelaxer, calculate_full_potential_derivative  # type: ignore
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            _h = logging.StreamHandler()
            _f = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            _h.setFormatter(_f)
            logger.addHandler(_h)
            logger.setLevel(logging.INFO)

# Optional import for visualization palettes (not required)
try:
    from gef.visualization.spectrum_plots import MassSpectrumPlotter  # pragma: no cover
except Exception:
    MassSpectrumPlotter = None  # type: ignore


# =============================================================================
# Physics helpers
# =============================================================================

def analytic_vacuum_energy(config: Dict) -> float:
    """Compute the analytic vacuum energy of a uniform field on the lattice.

    Vacuum φ₀ is set by U_iso + U_P: for μ² ≤ 2P -> φ₀=0; else φ₀²=(μ²-2P)/λ.
    Only isotropic + pressure contribute for uniform states; gradient terms vanish.
    Total energy = (U_iso(φ₀)+U_P(φ₀)) * Volume, where Volume = Ncells * dx⁴.
    """
    mu2 = float(config["mu_squared"])     # μ²
    lam = float(config["lambda_val"])     # λ (>0)
    P = float(cfg.get("P_env", 0.0))      # Pressure
    if cfg["mu_squared"] <= 2.0 * P:
        phi0 = 0.0
    else:
        phi0 = np.sqrt(max(0.0, (cfg["mu_squared"] - 2.0 * P) / cfg["lambda_val"]))
    test = np.full(solver.lattice_shape, phi0, dtype=np.float64)
    dU = calculate_full_potential_derivative(test, solver.mu2, solver.lam, solver.g_sq, solver.P_env, solver.h_sq, solver.dx)
    if not np.allclose(dU, 0.0, atol=1e-12):
        logger.warning("Uniform vacuum derivative not ~0 (max |δU/δφ|=%.3e). Check signs.", np.abs(dU).max())
    else:
        logger.info("Uniform vacuum derivative ~0 (max |δU/δφ|=%.3e).", np.abs(dU).max())
        
    dx = float(config["dx"])              # lattice spacing
    n0, n1, n2, n3 = map(int, config["lattice_size"])  # type: ignore
    ncells = n0 * n1 * n2 * n3

    if mu2 <= 2.0 * P:
        phi0_sq = 0.0
    else:
        phi0_sq = max(0.0, (mu2 - 2.0 * P) / lam)
    # φ₀ sign irrelevant for even powers
    U_iso = -0.5 * mu2 * phi0_sq + 0.25 * lam * (phi0_sq ** 2)
    U_P = -P * (1.0 - phi0_sq)
    return (U_iso + U_P) * (dx ** 4) * ncells


# =============================================================================
# Resonance model
# =============================================================================

class ResonanceModel:
    """Effective anisotropic coupling model g_eff²(N_w).

    Supports per-peak shapes: 'gaussian' (default) and 'lorentzian'. If a peak
    specifies a negative/zero sigma or gamma, it is ignored with a warning.
    """

    @staticmethod
    def _gaussian(nw: float, center: float, amplitude: float, sigma: float) -> float:
        return amplitude * np.exp(-((nw - center) ** 2) / (2.0 * sigma * sigma))

    @staticmethod
    def _lorentzian(nw: float, center: float, amplitude: float, gamma: float) -> float:
        return amplitude * (gamma * gamma) / (((nw - center) ** 2) + (gamma * gamma))

    @classmethod
    def calculate_g_eff_squared(cls, nw: int, rp: Dict, log: Optional[logging.Logger] = None) -> float:
        g_base_sq = float(rp.get("g_base_squared", 0.0))
        invert = bool(rp.get("invert_amplitudes", False))
        peaks = rp.get("peaks", []) or []
        offset = 0.0

        for peak in peaks:
            center = float(peak.get("center", 0.0))
            amp = float(peak.get("amplitude", 0.0))
            if invert:
                amp = -amp
            shape = str(peak.get("type", "gaussian")).lower()

            if shape == "lorentzian":
                gamma = float(peak.get("gamma", peak.get("sigma", 1.0)))
                if gamma <= 0:
                    if log: log.warning("Ignoring Lorentzian peak with nonpositive gamma: %s", peak)
                    continue
                offset += cls._lorentzian(nw, center, amp, gamma)
            else:  # gaussian default
                sigma = float(peak.get("sigma", 1.0))
                if sigma <= 0:
                    if log: log.warning("Ignoring Gaussian peak with nonpositive sigma: %s", peak)
                    continue
                offset += cls._gaussian(nw, center, amp, sigma)

        g_eff = g_base_sq + offset
        if g_eff < 0:
            if log: log.warning("Computed g_eff²=%.4g < 0 for N_w=%s; clamping to 0 (unphysical).", g_eff, nw)
            g_eff = 0.0
        return g_eff


# =============================================================================
# Worker (per-N_w simulation)
# =============================================================================

class SimulationWorker:
    """Handles individual simulation runs for parallel processing."""

    @staticmethod
    def run_single_simulation(nw: int, base_config: Dict) -> Dict:
        """Run one relaxation for the given winding number."""
        quiet = bool(base_config.get("quiet", False))
        rp = dict(base_config.get("resonance_parameters", {}))
        if "peaks" not in rp:
            rp["peaks"] = base_config.get("peaks", [])

        g_eff_sq = ResonanceModel.calculate_g_eff_squared(nw, rp, logger)
        if not quiet:
            logger.info(f"Starting N_w={nw} with g_eff²={g_eff_sq:.6g}")

        cfg = dict(base_config)
        cfg["g_squared"] = float(g_eff_sq)  # pass effective anisotropy to solver

        try:
            solver = HopfionRelaxer(cfg)
            
            # Quick uniform-vacuum check (gradient terms must vanish)
            phi0 = 0.0 if cfg["mu_squared"] <= 2.0*cfg.get("P_env",0.0) else np.sqrt(max(0.0,(cfg["mu_squared"]-2.0*cfg.get("P_env",0.0))/cfg["lambda_val"]))
            test = np.full(solver.lattice_shape, phi0, dtype=np.float64)
            dU = calculate_full_potential_derivative(test, solver.mu2, solver.lam, solver.g_sq, solver.P_env, solver.h_sq, solver.dx)
            if not np.allclose(dU, 0.0, atol=1e-12):
                logger.warning(f"Uniform vacuum derivative not ~0 (max |δU/δφ|={np.abs(dU).max():.3e}). Check signs.")
            
            solver.initialize_field(nw=nw)

            record_series = bool(base_config.get("record_series", False))

            t0 = _dt.datetime.now()
            _, final_energy = solver.run_relaxation(record_series=record_series)
            dt = (_dt.datetime.now() - t0).total_seconds()

            converged = bool(np.isfinite(final_energy))
            status = "converged" if converged else "FAILED"
            if not quiet:
                logger.info(f"Finished N_w={nw} in {dt:.2f}s; E_final={final_energy:.6g} ({'converged' if converged else 'FAILED'})")

            if not converged:
                final_energy = np.nan

        except Exception:  # pragma: no cover
            logger.exception(f"CRITICAL ERROR in simulation for N_w={nw}")
            final_energy = np.nan
            converged = False

        return {
            "winding_number": int(nw),
            "g_eff_squared": float(g_eff_sq),
            "final_energy": float(final_energy) if np.isfinite(final_energy) else np.nan,
            "converged": bool(converged),
        }


# =============================================================================
# Analyzer & plotting
# =============================================================================

class MassSpectrumAnalyzer:
    """Run, analyze, and plot the N_w mass spectrum."""

    def __init__(self, config_or_path: Dict | str | Path):
        self.logger = logger
        if isinstance(config_or_path, dict):
            self.config = dict(config_or_path)
            self.config_path = None
        else:
            self.config_path = Path(config_or_path)
            self.config = self._load_config()

        self.output_dir = self._setup_output_directory()
        self.quiet = bool(self.config.get("quiet", False))

        # Precompute analytic vacuum energy unless explicitly provided
        if "vacuum_energy" in self.config and self.config["vacuum_energy"] is not None:
            self.vacuum_energy = float(self.config["vacuum_energy"])  # user override
        else:
            try:
                self.vacuum_energy = analytic_vacuum_energy(self.config)
                if not self.quiet:
                    self.logger.info(f"Analytic vacuum energy: {self.vacuum_energy:.6g}")
            except Exception:
                self.logger.exception("Failed computing analytic vacuum energy; defaulting to 0.0")
                self.vacuum_energy = 0.0

    # ---------------------------- IO & setup ---------------------------- #
    def _load_config(self) -> Dict:
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_output_directory(self) -> Path:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        particle_type = self.config.get("particle_type", "unknown")
        root = Path(self.config.get("output_dir", "."))
        out = root / f"nw_spectrum_{particle_type}_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        if not self.config.get("quiet", False):
            self.logger.info(f"Output directory: {out}")
        return out

    # ------------------------ Simulation runner ------------------------ #
    def run_parallel_simulations(self) -> pd.DataFrame:
        nw_values = list(self.config["nw_sweep_range"])  # type: ignore

        # Processes & threading knobs
        requested_procs = int(self.config.get("processes", 0))
        if requested_procs <= 0:
            num_procs = max(1, mp.cpu_count() - 1)
        else:
            num_procs = max(1, requested_procs)

        threads_per_proc = int(self.config.get("threads_per_process", 1))
        # These env vars are inherited by spawned workers
        os.environ["GEF_NUMBA_NUM_THREADS"] = str(threads_per_proc)
        os.environ["NUMBA_NUM_THREADS"] = str(threads_per_proc)

        if not self.quiet:
            self.logger.info(f"N_w sweep ({len(nw_values)} values): {nw_values}")
            self.logger.info(f"Using {num_procs} process(es) × {threads_per_proc} Numba thread(s)")

        worker_func = partial(SimulationWorker.run_single_simulation, base_config=self.config)

        if num_procs == 1:
            results = list(map(worker_func, nw_values))
        else:
            # On Windows/macOS we set spawn in __main__ guard
            with mp.Pool(processes=num_procs) as pool:
                results = pool.map(worker_func, nw_values)

        df = pd.DataFrame(results).sort_values("winding_number").reset_index(drop=True)
        df["vacuum_energy"] = self.vacuum_energy
        # Mass is E_final - E_vac; keep NaN for non-converged
        df["mass"] = df["final_energy"] - self.vacuum_energy
        
        # Safety check: negative masses indicate sign errors in δU/δφ
        tol = 1e-10
        neg_mask = (df["mass"].notna()) & (df["mass"] < -tol)
        if neg_mask.any():
            self.logger.error(
                f"Negative masses detected (min={df.loc[neg_mask, 'mass'].min():.3e}). "
                "This usually means δU/δφ signs don't match the energy. Re-check the solver."
            )
        
        return df

    # ----------------------------- Saving ------------------------------ #
    def save_results(self, results_df: pd.DataFrame) -> Path:
        csv_path = self.output_dir / "mass_spectrum_results.csv"
        results_df.to_csv(csv_path, index=False)
        if not self.quiet:
            self.logger.info(f"Saved results: {csv_path}")
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                self.logger.info(f"Results summary:\n{results_df.to_string(index=False)}")
        return csv_path

    # ----------------------------- Plotting ---------------------------- #
    def generate_plots(self, results_df: pd.DataFrame) -> Optional[Path]:
        plotter_cls = MassSpectrumPlotter if MassSpectrumPlotter else LocalMassSpectrumPlotter
        plotter = plotter_cls(self.config)
        try:
            path = plotter.create_spectrum_plot(results_df, self.output_dir)
        except Exception:
            self.logger.exception("Plot generation failed")
            return None
        if not self.quiet:
            self.logger.info(f"Saved plot: {path}")
        return path

    # ---------------------------- Orchestration ------------------------ #
    def run_full_analysis(self) -> Tuple[pd.DataFrame, Optional[Path], Optional[Path]]:
        self.logger.info("Starting N_w mass spectrum analysis…")
        df = self.run_parallel_simulations()
        csv_path = None if self.config.get("skip_save") else self.save_results(df)
        plot_path = None if self.config.get("skip_plots") else self.generate_plots(df)
        self.logger.info("Analysis complete.")
        return df, csv_path, plot_path


# ------------------------- Local plotter fallback ------------------------- #
class LocalMassSpectrumPlotter:
    """Simple Matplotlib plotter for mass-vs-N_w spectra."""

    def __init__(self, config: Dict):
        self.config = dict(config)

    def create_spectrum_plot(self, results_df: pd.DataFrame, output_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

        plotting_cfg = self.config.get("plotting", {})
        normalize_y = bool(plotting_cfg.get("normalize_y", False))

        # Prepare data
        plot_df = results_df.copy()
        stable_df = plot_df.dropna(subset=["mass"])
        unstable_df = plot_df[plot_df["mass"].isna()]

        y_label = "Emergent Mass (E_final − E_vacuum)"
        
        # --- FIX: Capture original mass range BEFORE normalization ---
        orig_m_min, orig_m_max = (None, None)
        if not stable_df.empty:
            orig_m_min = stable_df["mass"].min()
            orig_m_max = stable_df["mass"].max()

        if normalize_y and orig_m_min is not None and orig_m_max is not None:
            m_span = orig_m_max - orig_m_min
            if m_span > 1e-12:
                # Apply normalization to the 'mass' column for plotting
                plot_df.loc[plot_df["mass"].notna(), "mass"] = (
                    (plot_df.loc[plot_df["mass"].notna(), "mass"] - orig_m_min) / m_span
                )
                stable_df = plot_df.dropna(subset=["mass"])  # Refresh with normalized values
                y_label = "Normalized Emergent Mass"
            else:
                normalize_y = False  # Avoid division by zero if all masses are the same

        # Plot stable points
        if not stable_df.empty:
            ax.plot(
                stable_df["winding_number"],
                stable_df["mass"],
                marker="o", linestyle="-", linewidth=2, markersize=7,
                label="Stable",
            )

        # Compute y-bottom for unstable marks using the (potentially normalized) data range
        if not stable_df.empty:
            ymin, ymax = stable_df["mass"].min(), stable_df["mass"].max()
            pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
            y_bottom = ymin - pad
        else:
            y_bottom = -0.05 # Place slightly below zero if no stable points exist

        # Plot non-converged as X along a baseline
        if not unstable_df.empty:
            ax.scatter(
                unstable_df["winding_number"],
                np.full(len(unstable_df), y_bottom),
                marker="x", s=90, color="red",
                label="Unstable / Non-converged",
                zorder=5,
            )

        ax.set_xlabel("Topological Winding Number (N_w)", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"GEF Mass Spectrum: {self.config.get('particle_type', 'Unknown')}", fontsize=14)

        # Overlays - Pass the original mass range for correct normalization
        self._add_overlays(ax, normalize_y, orig_m_min, orig_m_max)

        # Deduplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=11)

        ax.margins(x=0.05, y=0.08)
        path = output_dir / "mass_vs_nw_spectrum.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    def _add_overlays(self, ax, normalized: bool, orig_m_min: Optional[float], orig_m_max: Optional[float]) -> None:
        """Adds horizontal (observed mass) and vertical (resonance) lines to the plot."""
        # Observed masses (horizontal lines)
        m_span = None
        if normalized and orig_m_min is not None and orig_m_max is not None:
            m_span = orig_m_max - orig_m_min

        for item in self.config.get("observed_masses", []):
            y_obs = item.get("mass")
            label = item.get("label", "Observed mass")
            if y_obs is None: continue

            # --- FIX: Use the original mass range for normalization ---
            if normalized and m_span is not None and m_span > 1e-12:
                y_plot = (y_obs - orig_m_min) / m_span
            else:
                y_plot = y_obs
            
            ax.axhline(y=y_plot, linestyle=":", linewidth=1.5, color="gray", alpha=0.8)
            # Add text label directly on the plot instead of cluttering the legend
            ax.text(ax.get_xlim()[1], y_plot, f' {label}', color='gray',
                    ha='left', va='center', fontsize=10)

        # Resonance peaks (vertical lines)
        peaks = (self.config.get("resonance_parameters", {}).get("peaks", []) or self.config.get("peaks", []))
        for peak in peaks:
            x = peak.get("center")
            if x is not None:
                ax.axvline(x=float(x), linestyle="--", linewidth=1.2, color="purple", alpha=0.6)
            

# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run GEF N_w mass spectrum analysis")
    p.add_argument("config_path", help="Path to YAML configuration file")
    args = p.parse_args(argv)

    try:
        analyzer = MassSpectrumAnalyzer(args.config_path)
        _, csv_path, plot_path = analyzer.run_full_analysis()
        print("\nAnalysis completed successfully!")
        if csv_path:
            print(f"Results saved to: {csv_path}")
        if plot_path:
            print(f"Plot saved to: {plot_path}")
        return 0
    except Exception:
        logger.exception("Analysis failed with a critical error.")
        return 1


if __name__ == "__main__":
    # For multiprocessing on Windows/macOS, prefer 'spawn'. Linux can use default.
    if sys.platform != "linux":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    sys.exit(main())
