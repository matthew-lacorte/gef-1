
"""
Compute the GEF Planck‑particle frequency, tick, and compactification radius.

- Reads run parameters from a YAML config (default: ./configs/default_planck_particle.yml).
- Operates in “single” (one mass value) or “sweep” (range of masses) mode.
- Outputs results and full provenance to a unique timestamped directory.
"""

from __future__ import annotations

import argparse
import sys
import shutil
import yaml
import datetime
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple
import sympy as sp

# ──────────────────────────────────────────────────────────────────────────────
# GEF infrastructure imports
from gef.core.logging import logger, setup_logfile
from gef.core.constants import CONSTANTS_DICT
from gef.core.validators import asdict, positive_value

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants from the GEF constants system
c_const = CONSTANTS_DICT["c"]
planck_const = CONSTANTS_DICT["hbar"]
electron_volt_const = CONSTANTS_DICT["eV"]

# Extract the actual values for calculations
c = c_const.value if c_const.value is not None else 299_792_458  # Default to c in m/s if not set
planck = planck_const.value if planck_const.value is not None else 6.626_070_15e-34  # Default value
electron_volt = electron_volt_const.value if electron_volt_const.value is not None else 1.602_176_634e-19  # Default value

# ──────────────────────────────────────────────────────────────────────────────
# Computation helpers
def compute_radius(mass_MeV: float) -> Tuple[float, float, float]:
    """
    Return (frequency Hz, tick s, radius m) for a given rest mass in MeV/c².
    """
    E_P = mass_MeV * 1e6 * electron_volt          # J
    f_P = E_P / planck                            # Hz
    t_P = 1 / f_P                                 # s
    r_P = c * t_P / (2 * sp.pi)                   # m
    return f_P, t_P, float(r_P)

def mass_iterator(cfg: dict) -> Iterable[float]:
    """
    Yield one or many masses (MeV) according to config.
    """
    mode = cfg.get("mode", "single").lower()
    if mode == "single":
        try:
            # Try to use the GEF Planck Particle mass from constants if available
            m_info = CONSTANTS_DICT.get("m_GEF_PP") or CONSTANTS_DICT.get("m_PlanckParticle")
            if m_info and m_info.value is not None:
                yield m_info.value
            else:
                # Fall back to config value
                yield cfg["single"]["mass_MeV"]
        except KeyError:
            # Fall back to config value
            yield cfg["single"]["mass_MeV"]
    elif mode == "sweep":
        s = cfg["sweep"]
        current = s["start_MeV"]
        while current <= s["stop_MeV"]:
            yield current
            current += s["step_MeV"]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'single' or 'sweep'.")

def get_default_config_path(script_dir: Path) -> Path:
    """Return the path to the default config in ./configs/"""
    script_name = script_dir.name
    return script_dir / "configs" / f"default_{script_name}.yml"

def make_output_dir(script_dir: Path, base_output_dir: Path | None = None) -> Path:
    """Create a unique, timestamped output directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    out_base = (base_output_dir or (script_dir / "outputs")).resolve()
    out_dir = out_base / f"{timestamp}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir

def save_run_config(output_dir: Path, config: dict, config_file_path: Path):
    """Save the merged config and copy base config to output dir."""
    with open(output_dir / "used_config.yml", "w") as f:
        yaml.dump(config, f)
    shutil.copy(str(config_file_path), output_dir / "base_config.yml")

# ──────────────────────────────────────────────────────────────────────────────
# CLI and run logic
def main(argv: List[str] | None = None) -> int:
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        prog="planck_particle",
        description="GEF Planck‑particle geometric radius calculator",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=get_default_config_path(script_dir),
        help="Path to YAML configuration file (default: configs/default_planck_particle.yml)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Base directory for outputs (default: ./outputs/ under script dir)"
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional path to write a rotating log file",
    )
    args = parser.parse_args(argv)

    # Setup logging
    if args.log:
        setup_logfile(str(args.log))
    logger.info(f"Starting Planck‑particle calculation with config: {args.config}")

    # Load config
    cfg = yaml.safe_load(args.config.read_text())

    # Prepare output dir
    out_dir = make_output_dir(script_dir, args.output_dir)
    (out_dir / "results").mkdir(exist_ok=True)
    logger.info(f"Results will be saved in: {out_dir}")

    # Save provenance: config, base config
    save_run_config(out_dir, cfg, args.config)

    # Output header and CSV setup
    header = f"{'mass (MeV)':>10} | {'f_P (Hz)':>13} | {'t_P (s)':>11} | {'r_P (fm)':>11}"
    print(header)
    print("-" * len(header))
    csv_path = out_dir / "results" / "planck_particle_results.csv"
    with open(csv_path, "w") as csvfile:
        csvfile.write("mass_MeV,f_P_Hz,t_P_s,r_P_fm\n")
        for m in mass_iterator(cfg):
            f_P, t_P, r_P = compute_radius(m)
            logger.debug(f"m={m} MeV, f_P={f_P}, t_P={t_P}, r_P={r_P}")
            print(f"{m:10.2f} | {f_P:13.5e} | {t_P:11.5e} | {r_P*1e15:11.5f}")
            csvfile.write(f"{m:.6f},{f_P:.8e},{t_P:.8e},{r_P*1e15:.8f}\n")

    logger.info(f"Done. Results: {csv_path}")
    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
