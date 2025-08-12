#!/usr/bin/env python3
"""
GEF Planck-Particle "Compactification Radius" (Reduced Compton Length)

This standalone script computes a characteristic length scale r_P from a given
rest mass m (default in MeV/c^2), using the relations:

    E = m c^2
    f = E / h
    T = 1 / f
    r_P = c T / (2π) = ħ c / E = ħ / (m c)

So r_P is *exactly* the reduced Compton wavelength λ̄ of the mass m.
We keep the GEF narrative, but name the quantity honestly to avoid confusion
with charge/classical radii.

Usage examples:
  python calculate_radius.py
  python calculate_radius.py --mass 938.272 --unit MeV
  python calculate_radius.py --mass 0.938272 --unit GeV --json
  python calculate_radius.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants (CODATA exact where applicable)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PhysicalConstants:
    """Container for fundamental physical constants."""
    # SI base
    c_light: float = 299_792_458.0            # m/s (exact)
    h_planck: float = 6.62607015e-34          # J·s (exact definition)
    hbar: float = 1.054571817e-34             # J·s (CODATA 2018)
    eV_to_joule: float = 1.602176634e-19      # J per eV (exact)

    # Reference scales for context (not used in computation)
    proton_charge_radius_fm: float = 0.84     # ~fm
    qcd_scale_fm: float = 1.0                 # ~fm


@dataclass
class GEFResults:
    """Container for calculation results."""
    input_mass: float
    input_unit: str
    mass_kg: float
    compton_frequency_hz: float
    fundamental_period_s: float
    compactification_radius_m: float
    compactification_radius_fm: float

    def __str__(self) -> str:
        return (
            "GEF Results (Reduced Compton Length):\n"
            f"  Input Mass: {self.input_mass:g} {self.input_unit}/c²\n"
            f"  Mass: {self.mass_kg:.6e} kg\n"
            f"  Compton Frequency: {self.compton_frequency_hz:.6e} Hz\n"
            f"  Fundamental Period: {self.fundamental_period_s:.6e} s\n"
            f"  r_P = λ̄: {self.compactification_radius_m:.6e} m"
            f"  ({self.compactification_radius_fm:.4f} fm)"
        )


class GEFCalculator:
    """Standalone calculator for the reduced Compton length (λ̄) of a mass."""

    def __init__(self, constants: PhysicalConstants | None = None):
        self.constants = constants or PhysicalConstants()
        self._tau = getattr(math, "tau", 2.0 * math.pi)

    # ───────────────────────────── Units ─────────────────────────────

    def mass_to_kg(self, mass: float, unit: str = "MeV") -> float:
        """
        Convert a rest mass from {eV, keV, MeV, GeV, TeV, kg} to kilograms.
        The value is assumed to be in 'unit' *per c^2* (i.e., MeV/c^2, etc.)
        except for 'kg' which is already mass.
        """
        unit = unit.strip().lower()
        if unit == "kg":
            if mass <= 0 or not math.isfinite(mass):
                raise ValueError("Mass (kg) must be positive and finite.")
            return float(mass)

        # energy in eV (per c^2), then convert to J, then to kg by E=mc^2
        scale = {
            "ev": 1.0,
            "kev": 1e3,
            "mev": 1e6,
            "gev": 1e9,
            "tev": 1e12,
        }.get(unit)
        if scale is None:
            raise ValueError("Unit must be one of: eV, keV, MeV, GeV, TeV, kg")

        if mass <= 0 or not math.isfinite(mass):
            raise ValueError("Mass must be positive and finite.")

        energy_eV = mass * scale
        energy_J = energy_eV * self.constants.eV_to_joule
        mass_kg = energy_J / (self.constants.c_light ** 2)
        return mass_kg

    # ────────────────────── Core physics relations ───────────────────

    def compton_frequency_and_period(self, mass_kg: float) -> Tuple[float, float]:
        """
        f = (m c^2)/h,  T = 1/f
        """
        E = mass_kg * (self.constants.c_light ** 2)   # J
        f = E / self.constants.h_planck               # Hz
        T = 1.0 / f                                   # s
        return f, T

    def compactification_radius_m(self, period_s: float) -> float:
        """
        r_P = c T / (2π)  = (ħ c)/E  = ħ/(m c)
        We compute via the c T / τ route for clarity.
        """
        return (self.constants.c_light * period_s) / self._tau

    def calculate(self, mass_value: float, unit: str, verbose: bool = True) -> GEFResults:
        """
        Full pipeline: (mass, unit) → kg → (f, T) → r_P (λ̄).
        """
        if verbose:
            print("=== GEF Compactification Radius (Reduced Compton) ===\n")

        mass_kg = self.mass_to_kg(mass_value, unit)
        if verbose:
            print("1) Mass conversion")
            print(f"   Input: {mass_value:g} {unit}/c²")
            print(f"   Mass:  {mass_kg:.6e} kg\n")

        f, T = self.compton_frequency_and_period(mass_kg)
        if verbose:
            print("2) Compton properties")
            print(f"   f = E/h: {f:.6e} Hz")
            print(f"   T = 1/f: {T:.6e} s\n")

        r_m = self.compactification_radius_m(T)
        r_fm = r_m * 1e15
        if verbose:
            print("3) Compactification radius (reduced Compton length)")
            print(f"   r_P = c T / (2π) = ħ/(m c)")
            print(f"   r_P = {r_m:.6e} m  = {r_fm:.4f} fm\n")

        results = GEFResults(
            input_mass=mass_value,
            input_unit=unit,
            mass_kg=mass_kg,
            compton_frequency_hz=f,
            fundamental_period_s=T,
            compactification_radius_m=r_m,
            compactification_radius_fm=r_fm,
        )

        if verbose:
            self._contextual_comparison(results)

        return results

    # ──────────────────────── Contextual info ────────────────────────

    def _contextual_comparison(self, res: GEFResults) -> None:
        """Print context against hadronic scales (informational only)."""
        print("=== Contextual Comparison ===")
        print(f"Derived r_P: {res.compactification_radius_fm:.4f} fm")
        print(f"Proton charge radius: ~{self.constants.proton_charge_radius_fm:.2f} fm")
        print(f"QCD scale (Λ_QCD^-1): ~{self.constants.qcd_scale_fm:.1f} fm")

        p = self.constants.proton_charge_radius_fm
        diff_pct = abs(res.compactification_radius_fm - p) / p * 100.0
        print(f"\nDifference vs proton charge radius: {diff_pct:.1f}%")

        # Rough qualitative tag:
        r = res.compactification_radius_fm
        if 0.3 <= r <= 3.0:
            tag = "hadronic-scale (O(1) fm)"
        elif r < 0.03:
            tag = "sub-hadronic (≪ 1 fm)"
        else:
            tag = "super-hadronic (≫ 1 fm)"
        print(f"Scale classification: {tag}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def validate_inputs(mass: float, unit: str) -> None:
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("Mass must be positive and finite.")
    unit_l = unit.lower()
    if unit_l not in {"ev", "kev", "mev", "gev", "tev", "kg"}:
        raise ValueError("Unit must be one of: eV, keV, MeV, GeV, TeV, kg")

def run_self_tests() -> int:
    """
    Minimal numeric sanity checks:
      - Electron mass: λ̄ ≈ 386.159267 fm
      - Proton mass:   λ̄ ≈ 0.210308 fm
    """
    const = PhysicalConstants()
    calc = GEFCalculator(const)

    def assert_close(a, b, rel=5e-6, label=""):
        if not math.isclose(a, b, rel_tol=rel, abs_tol=0.0):
            print(f"[SELF-TEST FAIL] {label}: got {a:.9g}, want {b:.9g}")
            return False
        return True

    ok = True
    # Electron: 0.51099895 MeV → λ̄ ≈ 386.159267 fm
    e_res = calc.calculate(0.51099895, "MeV", verbose=False)
    ok &= assert_close(e_res.compactification_radius_fm, 386.159267, rel=5e-6, label="electron λ̄")

    # Proton: 938.2720813 MeV → λ̄ ≈ 0.210308 fm
    p_res = calc.calculate(938.2720813, "MeV", verbose=False)
    ok &= assert_close(p_res.compactification_radius_fm, 0.210308, rel=5e-6, label="proton λ̄")

    if ok:
        print("[SELF-TEST PASS] All checks OK.")
        return 0
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone GEF compactification radius (reduced Compton length) calculator.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--mass", type=float, default=235.0,
                   help="Rest mass value (default: 235.0). Interpreted in --unit.")
    p.add_argument("--unit", type=str, default="MeV",
                   help="Unit for mass: eV | keV | MeV | GeV | TeV | kg (default: MeV).")
    p.add_argument("--quiet", action="store_true", help="Suppress verbose step-by-step output.")
    p.add_argument("--json", action="store_true", help="Emit results as JSON to stdout.")
    p.add_argument("--self-test", action="store_true", help="Run internal numeric checks and exit.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_tests()

    try:
        validate_inputs(args.mass, args.unit)
        calc = GEFCalculator()
        res = calc.calculate(args.mass, args.unit, verbose=not args.quiet)

        if args.json:
            print(json.dumps({
                "input_mass": args.mass,
                "input_unit": args.unit,
                "mass_kg": res.mass_kg,
                "compton_frequency_hz": res.compton_frequency_hz,
                "fundamental_period_s": res.fundamental_period_s,
                "compactification_radius_m": res.compactification_radius_m,
                "compactification_radius_fm": res.compactification_radius_fm,
            }, indent=2))
        else:
            print("=" * 60)
            print("FINAL RESULTS SUMMARY")
            print("=" * 60)
            print(res)
            print("=" * 60)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
