"""
Borwein‐type integral with a “Mexican‑Hat” (GEF) stabiliser.

* Reads optimisation settings from YAML (default: ./configs/default_borwein.yml)
* Supports Loguru file logging via --log
* Writes a CSV of optimisation history + a PNG of the final stabiliser curve
* Conforms to gef_core logging / constants / output‑dir conventions
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import uuid
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import yaml

from gef.core.logging import logger, setup_logfile
from gef.core.constants import CONSTANTS_DICT

# ─────────────────────────────────────────────────── CODATA constants in SI
PI = np.pi
# (No other physical constants needed for this script.)

# ─────────────────────────────────────────────────── GEF stabilised integrand
def borwein_integrand(x: float, n_terms: int, a: float, b: float) -> float:
    """Borwein product × GEF Mexican‑Hat stabiliser S(x) = (1‑a x²) e^{‑b x²}."""
    if x == 0.0:
        return 1.0
    product = np.prod([np.sinc(x / (PI * (2 * k + 1))) for k in range(n_terms)])
    stabiliser = (1.0 - a * x * x) * np.exp(-b * x * x)
    return product * stabiliser

def integral_value(n_terms: int, a: float, b: float) -> float:
    """Numerical ∫_{-∞}^{∞} I_n(x) S(x) dx (truncated to ±100)."""
    val, _err = quad(borwein_integrand, -100, 100, args=(n_terms, a, b), limit=200)
    return val

# ─────────────────────────────────────────────────── Optimisation objective
def objective(params: np.ndarray, n_terms: int) -> float:
    a, b = params
    if b <= 0.0:                              # enforce damping
        return 1e12
    val = integral_value(n_terms, a, b)
    err_sq = (val - PI) ** 2
    logger.debug(f"a={a:.3e}  b={b:.3e}  I={val:.6f}  err²={err_sq:.2e}")
    return err_sq

# ─────────────────────────────────────────────────── CLI helpers
def _default_cfg(script_dir: Path) -> Path:
    return script_dir / "configs" / "default_borwein.yml"

def _new_outdir(script_dir: Path, base: Path | None) -> Path:
    t = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    out_base = (base or script_dir / "outputs").resolve()
    out_dir = out_base / f"{t}-{run_id}"
    out_dir.mkdir(parents=True)
    return out_dir

# ─────────────────────────────────────────────────── Main
def main(argv: List[str] | None = None) -> int:
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(prog="borwein_stabilisation")
    p.add_argument("-c", "--config", type=Path, default=_default_cfg(script_dir))
    p.add_argument("-o", "--output-dir", type=Path, default=None)
    p.add_argument("--log", type=Path, default=None)
    args = p.parse_args(argv)

    if args.log:
        setup_logfile(str(args.log))

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = _new_outdir(script_dir, args.output_dir)
    (out_dir / "used_config.yml").write_text(yaml.dump(cfg))
    logger.info(f"Output dir: {out_dir}")

    n_terms = cfg["n_terms"]
    initial = cfg["initial_guess"]

    logger.info(f"Integrand I_{2*n_terms-1} (n_terms={n_terms}) — starting optimisation")
    unstable = integral_value(n_terms, 0.0, 0.0)
    logger.info(f"I_{2*n_terms-1}(0,0) = {unstable:.10f}   error²={(unstable-PI)**2:.2e}")

    hist_csv = out_dir / "optim_history.csv"
    hist_csv.write_text("step,a,b,error_sq\n")        # header

    def callback(xk):
        a, b = xk
        err = objective(xk, n_terms)
        step = len(list(hist_csv.read_text().splitlines())) - 1
        hist_csv.write_text(hist_csv.read_text() + f"{step},{a:.6e},{b:.6e},{err:.6e}\n")

    res = minimize(
        objective,
        initial,
        args=(n_terms,),
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12, "disp": True},
        callback=callback,
    )

    if not res.success:
        logger.error(f"Optimiser failed: {res.message}")
        return 1

    a_opt, b_opt = res.x
    logger.success(f"Converged: a={a_opt:.6e}, b={b_opt:.6e}")

    fixed = integral_value(n_terms, a_opt, b_opt)
    logger.info(f"Stabilised I = {fixed:.10f}   final error²={(fixed-PI)**2:.2e}")

    # --- Plot stabiliser ---------------------------------------------
    x = np.linspace(-30, 30, 1000)
    Sx = (1 - a_opt * x**2) * np.exp(-b_opt * x**2)
    plt.figure(figsize=(8, 5))
    plt.plot(x, Sx, label=f"S(x)   a={a_opt:.3e}, b={b_opt:.3e}")
    plt.axhline(1, ls="--", c="k")
    plt.ylim(0.9, 1.1)
    plt.title("GEF Mexican‑Hat Stabiliser")
    plt.xlabel("x"); plt.ylabel("S(x)")
    plt.grid(True); plt.legend()
    fig_path = out_dir / "stabiliser.png"
    plt.savefig(fig_path, dpi=150)
    logger.info(f"Plot saved → {fig_path}")
    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
