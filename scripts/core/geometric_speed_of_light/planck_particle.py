"""
Compute the GEF Planck‑particle frequency, tick and compactification radius.

* Reads run parameters from a YAML config (default: ./configs/default_planck_particle.yml).
* Operates in “single” (one mass value) or “sweep” (range of masses) mode.
* Writes human‑readable stdout, CSV and Loguru logs into a timestamped output dir.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple

import sympy as sp
import yaml

# ────────────────────────────────────────────────────── GEF infrastructure
from gef.core.logging import logger, setup_logfile
from gef.core.constants import CONSTANTS_DICT

# ────────────────────────────────────────────────────── Physical constants
try:
    c_const = CONSTANTS_DICT["c"]                  # speed of light
    planck_const = CONSTANTS_DICT["planck"]        # Planck constant
    eV_const = CONSTANTS_DICT["electron_volt"]     # 1 eV in joule
except KeyError as err:
    raise RuntimeError(f"Constant {err} missing from CONSTANTS_DICT")

c = c_const.value or 299_792_458                   # fallback exact CODATA
if planck_const.value is None:
    raise ValueError("Numeric value for Planck constant not set in constants.py")
planck = planck_const.value
electron_volt = eV_const.value

# ────────────────────────────────────────────────────── Computation helpers
def compute_radius(mass_MeV: float) -> Tuple[float, float, float]:
    """Return (frequency Hz, tick s, radius m) for a given rest mass in MeV/c²."""
    E_P = mass_MeV * 1e6 * electron_volt           # J
    f_P = E_P / planck                             # Hz
    t_P = 1 / f_P                                  # s
    r_P = c * t_P / (2 * sp.pi)                    # m
    return f_P, t_P, float(r_P)

def mass_iterator(cfg: dict) -> Iterable[float]:
    """Yield one or many masses (MeV) according to config."""
    mode = cfg.get("mode", "single").lower()

    if mode == "single":
        try:
            m_info = CONSTANTS_DICT["m_PlanckParticle"]
            if m_info.value is None:
                raise ValueError("m_PlanckParticle has no numeric value")
            yield m_info.value
        except KeyError:
            yield cfg["single"]["mass_MeV"]

    elif mode == "sweep":
        sweep = cfg["sweep"]
        current = sweep["start_MeV"]
        while current <= sweep["stop_MeV"]:
            yield current
            current += sweep["step_MeV"]

    else:
        raise ValueError("mode must be 'single' or 'sweep'")

# ────────────────────────────────────────────────────── CLI / run logic
def _default_config(script_dir: Path) -> Path:
    return script_dir / "configs" / "default_planck_particle.yml"

def _make_output_dir(script_dir: Path, base: Path | None) -> Path:
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    out_base = (base or script_dir / "outputs").resolve()
    out_dir = out_base / f"{timestamp}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "results").mkdir()
    return out_dir

def main(argv: List[str] | None = None) -> int:
    script_dir = Path(__file__).parent

    p = argparse.ArgumentParser(prog="planck_particle")
    p.add_argument("-c", "--config", type=Path, default=_default_config(script_dir))
    p.add_argument("-o", "--output-dir", type=Path, default=None)
    p.add_argument("--log", type=Path, default=None, help="Path to rotating log file")
    args = p.parse_args(argv)

    if args.log:
        setup_logfile(str(args.log))
    logger.info("Starting Planck‑particle run")

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = _make_output_dir(script_dir, args.output_dir)
    (out_dir / "used_config.yml").write_text(yaml.dump(cfg))
    logger.info(f"Output directory: {out_dir}")

    header = f"{'mass (MeV)':>10} | {'f_P (Hz)':>13} | {'t_P (s)':>11} | {'r_P (fm)':>11}"
    print(header); print("-"*len(header))
    csv_path = out_dir / "results" / "planck_particle_results.csv"
    with csv_path.open("w") as csvfile:
        csvfile.write("mass_MeV,f_P_Hz,t_P_s,r_P_fm\n")
        for m in mass_iterator(cfg):
            f_P, t_P, r_P = compute_radius(m)
            logger.debug(f"m={m} MeV  f_P={f_P}  t_P={t_P}  r_P={r_P}")
            print(f"{m:10.2f} | {f_P:13.5e} | {t_P:11.5e} | {r_P*1e15:11.5f}")
            csvfile.write(f"{m:.6f},{f_P:.8e},{t_P:.8e},{r_P*1e15:.8f}\n")

    logger.info(f"Run complete. CSV: {csv_path}")
    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
