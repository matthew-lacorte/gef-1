"""
Calculates the fundamental geometric radius of the GEF Planck Particle.

This script implements a core calculation of the General Euclidean Flow (GEF)
framework. It derives a characteristic length scale (the compactification radius, r_P)
from three fundamental constants: the speed of light (c), the Planck constant (h),
and the hypothesized mass of the GEF Planck Particle (M_fund).

The central hypothesis is that the speed of light is a geometric property,
defined as the circumference of the particle's compactified fourth dimension
traveled in one fundamental unit of its internal "clock" time.

Usage:
    python gef_radius_calculator.py
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class PhysicalConstants:
    """Container for fundamental physical constants."""
    # CODATA 2018 values
    c_light: float = 299_792_458.0          # Speed of light in vacuum (m/s)
    h_planck: float = 6.62607015e-34        # Planck constant (J·s)
    hbar: float = 1.054571817e-34           # Reduced Planck constant (J·s)
    ev_to_joule: float = 1.602176634e-19    # Conversion factor eV to J
    
    # Reference values for comparison
    proton_charge_radius_fm: float = 0.84   # Proton charge radius (fm)
    qcd_scale_fm: float = 1.0               # Characteristic QCD scale (fm)


@dataclass
class GEFResults:
    """Container for GEF calculation results."""
    input_mass_mev: float
    mass_kg: float
    compton_frequency_hz: float
    fundamental_period_s: float
    compactification_radius_m: float
    compactification_radius_fm: float
    
    def __str__(self) -> str:
        return (
            f"GEF Results:\n"
            f"  Input Mass: {self.input_mass_mev} MeV/c²\n"
            f"  Mass: {self.mass_kg:.4e} kg\n"
            f"  Compton Frequency: {self.compton_frequency_hz:.4e} Hz\n"
            f"  Fundamental Period: {self.fundamental_period_s:.4e} s\n"
            f"  Compactification Radius: {self.compactification_radius_fm:.4f} fm"
        )


class GEFCalculator:
    """Calculator for GEF Planck Particle properties."""
    
    def __init__(self, constants: PhysicalConstants = None):
        """Initialize with physical constants."""
        self.constants = constants or PhysicalConstants()
    
    def mev_to_kg(self, mass_mev: float) -> float:
        """Convert mass from MeV/c² to kg.
        
        Args:
            mass_mev: Mass in MeV/c²
            
        Returns:
            Mass in kg
        """
        energy_ev = mass_mev * 1e6  # Convert MeV to eV
        energy_j = energy_ev * self.constants.ev_to_joule  # Convert eV to J
        mass_kg = energy_j / (self.constants.c_light ** 2)  # E = mc² → m = E/c²
        return mass_kg
    
    def calculate_compton_properties(self, mass_kg: float) -> Tuple[float, float]:
        """Calculate Compton frequency and corresponding period.
        
        The Compton frequency is derived from E = hf, where E = mc².
        This represents the fundamental "tick rate" of the particle.
        
        Args:
            mass_kg: Mass in kg
            
        Returns:
            Tuple of (frequency_hz, period_s)
        """
        rest_energy_j = mass_kg * (self.constants.c_light ** 2)  # E = mc²
        frequency_hz = rest_energy_j / self.constants.h_planck   # f = E/h
        period_s = 1.0 / frequency_hz                            # T = 1/f
        return frequency_hz, period_s
    
    def calculate_compactification_radius(self, period_s: float) -> float:
        """Calculate the compactification radius from the fundamental period.
        
        Based on the GEF hypothesis: c = (2π × r_P) / T_P
        Therefore: r_P = (c × T_P) / (2π)
        
        Args:
            period_s: Fundamental period in seconds
            
        Returns:
            Compactification radius in meters
        """
        radius_m = (self.constants.c_light * period_s) / (2 * np.pi)
        return radius_m
    
    def calculate_fundamental_radius(self, mass_mev: float, verbose: bool = True) -> GEFResults:
        """
        Perform complete GEF calculation for given particle mass.

        Args:
            mass_mev: Mass of the GEF Planck Particle in MeV/c²
            verbose: Whether to print intermediate steps

        Returns:
            GEFResults object containing all calculated values
        """
        if verbose:
            print("=== GEF Fundamental Radius Calculation ===\n")

        # Step 1: Convert mass to SI units
        mass_kg = self.mev_to_kg(mass_mev)
        if verbose:
            print(f"1. Mass Conversion:")
            print(f"   Input: {mass_mev} MeV/c²")
            print(f"   Output: {mass_kg:.6e} kg\n")

        # Step 2: Calculate Compton frequency and period
        frequency_hz, period_s = self.calculate_compton_properties(mass_kg)
        if verbose:
            print(f"2. Compton Properties:")
            print(f"   Frequency (f_P): {frequency_hz:.6e} Hz")
            print(f"   Period (T_P): {period_s:.6e} s\n")

        # Step 3: Calculate compactification radius
        radius_m = self.calculate_compactification_radius(period_s)
        radius_fm = radius_m * 1e15  # Convert to femtometers
        
        if verbose:
            print(f"3. Compactification Radius:")
            print(f"   r_P = {radius_m:.6e} m")
            print(f"   r_P = {radius_fm:.4f} fm\n")

        # Create results object
        results = GEFResults(
            input_mass_mev=mass_mev,
            mass_kg=mass_kg,
            compton_frequency_hz=frequency_hz,
            fundamental_period_s=period_s,
            compactification_radius_m=radius_m,
            compactification_radius_fm=radius_fm
        )

        if verbose:
            self._print_comparison(results)

        return results
    
    def _print_comparison(self, results: GEFResults) -> None:
        """Print comparison with known physical scales."""
        print("=== Contextual Comparison ===")
        print(f"Derived radius: {results.compactification_radius_fm:.4f} fm")
        print(f"Proton charge radius: {self.constants.proton_charge_radius_fm:.2f} fm")
        print(f"QCD scale: ~{self.constants.qcd_scale_fm:.1f} fm")
        
        # Calculate percentage difference from proton radius
        diff_percent = abs(results.compactification_radius_fm - 
                          self.constants.proton_charge_radius_fm) / \
                      self.constants.proton_charge_radius_fm * 100
        
        print(f"\nDifference from proton charge radius: {diff_percent:.1f}%")
        print("The derived radius aligns well with the hadronic scale.\n")


def validate_inputs(mass_mev: float) -> None:
    """Validate input parameters."""
    if mass_mev <= 0:
        raise ValueError("Mass must be positive")
    if mass_mev > 1e6:  # Sanity check: > 1 TeV seems unrealistic
        raise ValueError("Mass seems unrealistically large (> 1 TeV)")


def main():
    """Main execution function with error handling."""
    # Default GEF model parameters
    DEFAULT_MASS_MEV = 235.0  # MeV/c²
    
    try:
        # Validate inputs
        validate_inputs(DEFAULT_MASS_MEV)
        
        # Create calculator and perform calculation
        calculator = GEFCalculator()
        results = calculator.calculate_fundamental_radius(DEFAULT_MASS_MEV)
        
        # Print final summary
        print("=" * 50)
        print("FINAL RESULTS SUMMARY")
        print("=" * 50)
        print(results)
        print("=" * 50)
        
    except Exception as e:
        print(f"Error in calculation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())