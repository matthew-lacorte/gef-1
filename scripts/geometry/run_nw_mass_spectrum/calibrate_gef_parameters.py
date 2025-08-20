#!/usr/bin/env python3
"""
GEF Parameter Calibration Engine (Unified)

Combines the optimized operational pipeline with a more physically meaningful
objective based on integrated valley mass. Switch objective via config:

objective:
  mode: integrated_ratio   # or: min_ratio
  use_log_error: true      # (optional, defaults true for ratios)

Supports parameter specs by name, and optionally dotted config paths.

Author: GEF Research Team
Date: 2025
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.integrate import simpson

# --- Add project root to path for robust imports (optional & safe) ---
project_root = Path(__file__).resolve().parents[2]  # up two (e.g., .../gef)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Logger resolution: try package logger, then a utility, then stdlib ---
def _get_logger() -> logging.Logger:
    try:
        # Preferred: your package logger
        from gef.core.logging import logger as pkg_logger  # type: ignore
        return pkg_logger  # type: ignore[return-value]
    except Exception:
        pass
    try:
        # Secondary: a utility that returns a configured logger
        from gef.utils.logging_utils import setup_logger  # type: ignore
        return setup_logger(__name__)
    except Exception:
        pass

    # Fallback: stdlib
    _logger = logging.getLogger("gef.calibration.fit_lepton_hierarchy")
    if not _logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(fmt)
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
    return _logger


logger = _get_logger()

import subprocess

# ---------- Helpers for config parameter mapping ----------

def _set_nested_config(cfg: dict, dotted_path: str, value: Any) -> None:
    """
    Set cfg["a"]["b"][2]["c"] via dotted_path like 'a.b[2].c'.
    """
    import re

    def _parse_tokens(path: str):
        # Split on dots, but keep bracket indices as part of tokens
        tokens = []
        for part in path.split("."):
            # Split "peaks[1]" into ("peaks", 1)
            m = re.findall(r"([^\[\]]+)(?:\[(\d+)\])?", part)
            if not m:
                raise ValueError(f"Bad path token: {part}")
            name, idx = m[0]
            tokens.append((name, None if idx == "" else int(idx)))
        return tokens

    ref = cfg
    tokens = _parse_tokens(dotted_path)
    for i, (name, maybe_idx) in enumerate(tokens):
        if i == len(tokens) - 1:
            # final set
            if maybe_idx is None:
                ref[name] = value
            else:
                ref[name][maybe_idx] = value
            return
        # descend
        ref = ref[name] if maybe_idx is None else ref[name][maybe_idx]


@dataclass
class ParamSpec:
    name: str
    bounds: Tuple[float, float]
    initial: float
    # Optional explicit config path override like "resonance_parameters.peaks[0].sigma"
    config_path: Optional[str] = None


def _build_param_specs(opt_cfg: dict) -> List[ParamSpec]:
    """
    Supports either:
      optimization_params.parameter_info: [{name, initial, min, max, config_path?}]
    or keeps compatibility with optimization_params.parameters (name list).
    """
    specs: List[ParamSpec] = []
    if "parameter_info" in opt_cfg:
        for p in opt_cfg["parameter_info"]:
            specs.append(
                ParamSpec(
                    name=p["name"],
                    initial=float(p["initial"]),
                    bounds=(float(p["min"]), float(p["max"])),
                    config_path=p.get("config_path"),
                )
            )
    elif "parameters" in opt_cfg and "initial_guess" in opt_cfg:
        # legacy fallback
        names = opt_cfg["parameters"]
        init = opt_cfg["initial_guess"]
        bnds = opt_cfg.get("bounds", [(None, None)] * len(names))
        for i, nm in enumerate(names):
            low, high = bnds[i]
            specs.append(
                ParamSpec(
                    name=nm,
                    initial=float(init[i]),
                    bounds=(float(low), float(high)),
                    config_path=None,
                )
            )
    else:
        raise KeyError(
            "optimization_params must include 'parameter_info' or ('parameters' and 'initial_guess')."
        )
    return specs


# ---------- Physics-centric utilities ----------

def _calculate_integrated_mass(
    df: pd.DataFrame,
    nw_center: float,
    sigma: float,
    width_factor: float,
    sea_level_energy: float,
) -> float:
    """
    Physically meaningful integrated mass ~ ∫ max(0, E_sea - E(N_w)) dN_w
    integrated over a window centered on the resonance valley center.

    Window width = width_factor * sigma  (NOT center-scaled).
    """
    if sigma <= 0 or width_factor <= 0:
        return 0.0

    half_width = 0.5 * width_factor * sigma
    nw_min = nw_center - half_width
    nw_max = nw_center + half_width

    peak_df = df[(df["winding_number"] >= nw_min) & (df["winding_number"] <= nw_max)].copy()
    if peak_df.empty or len(peak_df) < 3:
        return 0.0

    peak_df.sort_values("winding_number", inplace=True)
    energy_deficit = sea_level_energy - peak_df["final_energy"].to_numpy()
    energy_deficit = np.where(energy_deficit > 0.0, energy_deficit, 0.0)

    integrated = simpson(energy_deficit, peak_df["winding_number"].to_numpy())
    return float(integrated if integrated > 0 else 0.0)


def _infer_sea_level_energy(df: pd.DataFrame) -> float:
    """
    Estimate 'sea' (baseline) energy as mean over very low winding numbers,
    with sensible fallbacks if exact bins are missing.
    """
    if "final_energy" not in df.columns or "winding_number" not in df.columns:
        raise KeyError("DataFrame must contain 'final_energy' and 'winding_number'.")

    low_region = df[df["winding_number"] <= 10]
    if len(low_region) >= 3:
        return float(low_region["final_energy"].mean())

    # Fallbacks
    w1 = df[df["winding_number"] == 1.0]
    if not w1.empty:
        return float(w1["final_energy"].iloc[0])

    return float(df["final_energy"].median())


def _pick_valley_min(df: pd.DataFrame, nw_min: float, nw_max: Optional[float]) -> float:
    mask = df["winding_number"] > nw_min
    if nw_max is not None:
        mask &= (df["winding_number"] < nw_max)
    sub = df[mask]
    if sub.empty:
        return np.inf
    return float(sub["final_energy"].min())


# ---------- Objective functions (unified) ----------

@dataclass
class ObjectiveContext:
    base_config: dict
    param_specs: List[ParamSpec]
    mode: str  # "integrated_ratio" or "min_ratio"
    use_log_error: bool


def _apply_params_to_config(cfg: dict, specs: List[ParamSpec], values: np.ndarray) -> None:
    """
    Update cfg in-place using either well-known names (muon/tau amplitude/sigma)
    or explicit dotted config_path if provided in the spec.
    """
    name_to_val = {specs[i].name: float(values[i]) for i in range(len(specs))}
    for spec in specs:
        val = name_to_val[spec.name]
        if spec.config_path:
            _set_nested_config(cfg, spec.config_path, val)
            continue

        # Backward-compatible mapping by conventional names:
        if spec.name == "muon_amplitude":
            cfg["resonance_parameters"]["peaks"][0]["amplitude"] = val
        elif spec.name == "muon_sigma":
            cfg["resonance_parameters"]["peaks"][0]["sigma"] = val
        elif spec.name == "tau_amplitude":
            cfg["resonance_parameters"]["peaks"][1]["amplitude"] = val
        elif spec.name == "tau_sigma":
            cfg["resonance_parameters"]["peaks"][1]["sigma"] = val
        else:
            # As a last resort, set a top-level key if it exists
            if spec.name in cfg:
                cfg[spec.name] = val
            else:
                logger.debug(f"No mapping rule for '{spec.name}'. Consider providing 'config_path' in parameter_info.")


def _convert_paths_to_strings_in_config(cfg: Any) -> Any:
    """
    Recursively traverses a config dict/list and converts Path objects to strings.
    """
    if isinstance(cfg, dict):
        return {k: _convert_paths_to_strings_in_config(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_convert_paths_to_strings_in_config(i) for i in cfg]
    if isinstance(cfg, Path):
        return str(cfg)
    return cfg


def _run_analyzer(temp_cfg: dict) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Writes a temporary config and runs the modular analysis script in a
    separate process. This is crucial for avoiding multiprocessing issues
    with Numba in the main optimization loop.

    Returns:
        A tuple of (results_dataframe, csv_path, plot_path).
    """
    # Define a unique path for the temporary config for this run
    temp_dir = Path(temp_cfg["output_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_dir / "_temp_run_config.yml"

    # --- Convert all Path objects to strings before dumping to YAML ---
    serializable_cfg = _convert_paths_to_strings_in_config(temp_cfg)

    with open(temp_config_path, "w") as f:
        yaml.dump(serializable_cfg, f)

    # Get the path to the new, modular runner script
    script_path = Path(__file__).parent / "run_analysis.py"
    # Use a more robust method for invoking python
    cmd = ["/usr/bin/env", "python3", str(script_path), str(temp_config_path)]

    logger.info(f"Using python executable: /usr/bin/env python3")
    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        # Run the script as a separate process.
        # Stream output so progress is visible during warm-up and runs.
        subprocess.run(cmd, check=True, text=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Subprocess analysis failed for {temp_config_path}")
        # e.stdout / e.stderr may be None when not capturing output
        if getattr(e, "stdout", None):
            logger.error(f"STDOUT:\n{e.stdout}")
        if getattr(e, "stderr", None):
            logger.error(f"STDERR:\n{e.stderr}")
        raise e

    # The analysis script creates its own timestamped output directory inside
    # the one we specified. We need to find it to retrieve the results.
    # It will be the most recently created directory.
    sub_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
    if not sub_dirs:
        raise FileNotFoundError(f"No output subdirectory found in {temp_dir}")

    # Sort by creation time to find the newest one
    latest_output_dir = max(sub_dirs, key=os.path.getmtime)

    csv_path = latest_output_dir / 'mass_spectrum_results.csv'
    plot_path = latest_output_dir / 'mass_vs_nw_spectrum.png'

    if not csv_path.exists():
        raise FileNotFoundError(f"Result CSV not found at {csv_path}")

    results_df = pd.read_csv(csv_path)
    return results_df, csv_path, plot_path


def _objective(params: np.ndarray, ctx: ObjectiveContext) -> float:
    """
    Unified objective wrapper: builds temp config, runs analyzer, computes error.
    """
    temp_cfg = copy.deepcopy(ctx.base_config)

    # --- Apply trial parameters ---
    _apply_params_to_config(temp_cfg, ctx.param_specs, params)

    # --- Trial run settings (fast & robust) ---
    trial_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_out = Path(ctx.base_config["output_dir"])
    # Convert to string to avoid YAML serialization issues with Path objects
    temp_cfg["output_dir"] = str(base_out / "_temp_optimizer_run" / trial_stamp)

    # Avoid multiprocessing pickling issues during optimization; allow CPU threads
    temp_cfg["processes"] = 1
    temp_cfg["threads_per_process"] = int(ctx.base_config.get("threads_per_process", 1))
    temp_cfg["quiet"] = True
    temp_cfg["skip_plots"] = True

    # Let config control CSV saves during optimization; default False for speed
    opt_cfg = ctx.base_config.get("optimization_params", {})
    save_csv = bool(opt_cfg.get("save_csv_during_opt", False))
    temp_cfg["skip_save"] = not save_csv
    temp_cfg["record_series"] = False

    # --- Run the simulation/analysis ---
    try:
        results_df, csv_path, _ = _run_analyzer(temp_cfg)
    except Exception as e:
        logger.error(f"Simulation failed during optimization: {e}")
        return 1e12

    # --- Basic sanity logs ---
    if "final_energy" in results_df.columns:
        emin = float(results_df["final_energy"].min())
        emax = float(results_df["final_energy"].max())
        logger.info(f"Sim OK. {len(results_df)} rows. E-range [{emin:.3g}, {emax:.3g}] CSV: {csv_path or '<skip>'}")

    # --- Objective choice ---
    mode = ctx.mode.lower()
    target = ctx.base_config.get("calibration_target", {})

    if mode == "integrated_ratio":
        # Physically meaningful integrated valley masses
        peaks = ctx.base_config["resonance_parameters"]["peaks"]
        mu_center = float(peaks[0]["center"])
        tau_center = float(peaks[1]["center"])
        mu_sigma = float(peaks[0]["sigma"])
        tau_sigma = float(peaks[1]["sigma"])

        mu_wf = float(opt_cfg.get("muon_integration_width_factor", 6.0))
        tau_wf = float(opt_cfg.get("tau_integration_width_factor", 6.0))

        sea = _infer_sea_level_energy(results_df)

        mass_mu = _calculate_integrated_mass(results_df, mu_center, mu_sigma, mu_wf, sea)
        mass_tau = _calculate_integrated_mass(results_df, tau_center, tau_sigma, tau_wf, sea)

        if mass_mu <= 0 or mass_tau <= 0:
            logger.warning("Non-positive integrated masses (mu or tau). Penalizing.")
            return 1e9

        sim_ratio = mass_tau / mass_mu
        tgt_ratio = float(target["tau_to_muon_mass_ratio"])

        if ctx.use_log_error:
            err = (np.log(sim_ratio) - np.log(tgt_ratio)) ** 2
        else:
            err = (sim_ratio - tgt_ratio) ** 2

        logger.info(f"Integrated masses: mu={mass_mu:.6g}, tau={mass_tau:.6g} | tau/mu sim={sim_ratio:.6g} tgt={tgt_ratio:.6g} -> err={err:.6g}")
        return float(err)

    elif mode == "min_ratio":
        # Legacy: use minima in valleys & explicit electron baseline
        # Expect DataFrame with columns: 'winding_number', 'final_energy', 'mass'
        if not {"winding_number", "final_energy", "mass"}.issubset(results_df.columns):
            logger.warning("Results missing required columns for min_ratio. Penalizing.")
            return 1e9

        # electron (baseline)
        m_e_series = results_df.loc[results_df["winding_number"] == 1.0, "mass"]
        if m_e_series.empty:
            logger.warning("No N_w=1 mass found. Penalizing.")
            return 1e9
        m_e = float(m_e_series.iloc[0])

        # muon valley (heuristic range)
        mu_min_e = _pick_valley_min(results_df, nw_min=10.0, nw_max=1000.0)
        tau_min_e = _pick_valley_min(results_df, nw_min=1000.0, nw_max=None)

        # convert minima in energy valley to mass via provided 'mass' column minima:
        m_mu = float(results_df.loc[(results_df["winding_number"] > 10.0) & (results_df["winding_number"] < 1000.0), "mass"].min())
        m_tau = float(results_df.loc[(results_df["winding_number"] > 1000.0), "mass"].min())

        if any(x <= 0 for x in (m_e, m_mu, m_tau)):
            logger.warning(f"Non-positive masses encountered: me={m_e}, mmu={m_mu}, mtau={m_tau}. Penalizing.")
            return 1e3 + 1e3 * abs(min(m_e, m_mu, m_tau))

        sim_mu_e = m_mu / m_e
        sim_tau_e = m_tau / m_e

        tgt_mu_e = float(target["muon_to_electron_mass_ratio"])
        tgt_tau_e = float(target["tau_to_electron_mass_ratio"])

        if ctx.use_log_error:
            e_mu = (np.log(sim_mu_e) - np.log(tgt_mu_e)) ** 2
            e_tau = (np.log(sim_tau_e) - np.log(tgt_tau_e)) ** 2
        else:
            e_mu = (sim_mu_e - tgt_mu_e) ** 2
            e_tau = (sim_tau_e - tgt_tau_e) ** 2

        err = float(e_mu + e_tau)
        logger.info(f"Min-ratio masses: me={m_e:.6g}, mmu={m_mu:.6g}, mtau={m_tau:.6g} | mu/e={sim_mu_e:.6g} (tgt {tgt_mu_e:.6g}), tau/e={sim_tau_e:.6g} (tgt {tgt_tau_e:.6g}) -> err={err:.6g}")
        return err

    else:
        logger.error(f"Unknown objective mode '{ctx.mode}'. Use 'integrated_ratio' or 'min_ratio'.")
        return 1e12


# ---------- Main CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Run GEF Parameter Calibration Engine (Unified)")
    parser.add_argument("config_path", help="Path to the master calibration YAML file")
    args = parser.parse_args()

    logger.info("--- Starting GEF Grand Fit Calibration (Unified) ---")

    # 1) Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Ensure output dir exists
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    # 2) Warm-up run to JIT / allocate once (best effort)
    try:
        warm_cfg = copy.deepcopy(config)
        warm_cfg["nw_sweep_range"] = [1]
        warm_cfg["relaxation_n_skip"] = int(config.get("relaxation_n_skip", 200))
        # Make warm-up quicker and more talkative
        warm_cfg["relaxation_n_iter"] = 64
        warm_cfg["processes"] = 1
        warm_cfg["threads_per_process"] = int(config.get("threads_per_process", 1))
        warm_cfg["quiet"] = False  # show logs during warm-up so progress is visible
        warm_cfg["skip_plots"] = True
        warm_cfg["skip_save"] = True
        warm_cfg["record_series"] = False
        _run_analyzer(warm_cfg)
        logger.info("Warm-up run completed.")
    except Exception:
        # Warm-up is optional; ignore failures
        pass

    # 3) Prepare optimization inputs
    opt_cfg = config["optimization_params"]
    specs = _build_param_specs(opt_cfg)
    initial = [p.initial for p in specs]
    bounds = [p.bounds for p in specs]

    obj_cfg = config.get("objective", {})
    mode = obj_cfg.get("mode", "integrated_ratio")
    use_log = bool(obj_cfg.get("use_log_error", True))

    ctx = ObjectiveContext(
        base_config=config,
        param_specs=specs,
        mode=mode,
        use_log_error=use_log,
    )

    # 4) Optimize
    result = minimize(
        fun=_objective,
        x0=np.asarray(initial, dtype=float),
        args=(ctx,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"disp": True, "maxiter": int(opt_cfg.get("max_iterations", 50))},
    )

    # 5) Report & save best params
    logger.info("--- Calibration Finished ---")
    logger.info(f"Success: {result.success}")
    logger.info(f"Message: {result.message}")
    logger.info(f"Final Error (fun): {result.fun:.8g}")

    best = result.x
    for i, spec in enumerate(specs):
        logger.info(f"  {spec.name}: {best[i]:.9g}")

    # 6) Golden config
    golden = copy.deepcopy(config)
    _apply_params_to_config(golden, specs, best)

    out_dir = Path(config["output_dir"])
    golden_path = out_dir / ("golden_config_leptons.yml" if mode == "min_ratio" else "golden_config_leptons_integrated.yml")
    with open(golden_path, "w") as f:
        yaml.dump(golden, f, sort_keys=False)
    logger.info(f"Golden configuration saved to: {golden_path}")

    # 7) Final high-res analysis w/ best params
    logger.info("Running final high-resolution analysis with best-fit parameters...")
    golden["processes"] = 1
    golden["threads_per_process"] = int(config.get("threads_per_process", 1))
    try:
        df_final, _, money_plot = _run_analyzer(golden)
        if money_plot:
            logger.info(f"Final 'Money Plot' saved to {money_plot}")
        else:
            logger.info("Final analysis complete (no plot path returned).")
    except Exception as e:
        logger.error(f"Final analysis failed: {e}")

    logger.info("--- Calibration Complete ---")


if __name__ == "__main__":
    main()
