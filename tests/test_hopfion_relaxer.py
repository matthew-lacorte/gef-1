"""Tests for gef.geometry.hopfion_relaxer.HopfionRelaxer

These tests validate small, fast code paths that do not require running the full
relaxation. They focus on configuration handling, vacuum analysis, and seeding.
"""

import pytest
import numpy as np
from gef.geometry.hopfion_relaxer import (
    HopfionRelaxer, 
    calculate_full_total_energy, 
    calculate_full_potential_derivative
)


def _minimal_config(**overrides):
    """Return a minimal valid config for constructing HopfionRelaxer."""
    cfg = {
        "lattice_size": (8, 8, 8, 8),
        "dx": 0.5,
        "mu_squared": 1.0,
        "lambda_val": 1.0,
        # Optional physics knobs default to zero for speed/stability in tests
        "g_squared": 0.0,
        "g_H_squared": 0.0,
        "P_env": 0.0,
        # Keep iterations tiny since we don't run relaxation in these tests
        "relaxation_dt": 1e-5,
        "max_iterations": 1,
    }
    cfg.update(overrides)
    return cfg


def test_vacuum_phi0_sq():
    # Case 1: mu^2 <= 2P → vacuum at phi=0
    config1 = _minimal_config(mu_squared=1.0, P_env=1.0, lambda_val=1.0)  # mu^2 < 2P
    solver1 = HopfionRelaxer(config1)
    assert solver1._vacuum_phi0_sq() == 0.0

    # Case 2: mu^2 > 2P → phi0^2 = (mu^2 - 2P)/lambda
    config2 = _minimal_config(mu_squared=4.0, P_env=1.0, lambda_val=2.0)  # mu^2 > 2P
    solver2 = HopfionRelaxer(config2)
    expected_phi0_sq = (4.0 - 2.0) / 2.0
    assert solver2._vacuum_phi0_sq() == pytest.approx(expected_phi0_sq)


def test_initialize_field():
    shape = (16, 16, 16, 16)
    config = _minimal_config(lattice_size=shape, dx=0.1)
    solver = HopfionRelaxer(config)

    # nw=0 → uniform field of ones
    solver.initialize_field(nw=0)
    assert solver.phi.shape == shape
    assert (solver.phi == 1.0).all()

    # nw=1 → seeded non-uniform field (should not be all ones)
    solver.initialize_field(nw=1)
    assert solver.phi.shape == shape
    assert not (solver.phi == 1.0).all()


def test_compute_energy_breakdown_keys_and_total_consistency():
    """Energy breakdown should contain expected keys and sum to total."""
    config = _minimal_config(lattice_size=(4, 4, 4, 4), dx=0.25, mu_squared=1.0, lambda_val=1.0, P_env=0.0)
    solver = HopfionRelaxer(config)
    solver.initialize_field(nw=0)  # uniform ones → zero gradients

    breakdown = solver.compute_energy_breakdown()
    for key in ["kinetic", "iso", "aniso", "pressure", "hook", "total"]:
        assert key in breakdown

    # With uniform ones and default params: no gradients, no pressure
    # iso per-site = -0.5*mu2*1 + 0.25*lambda*1 = -0.25
    # total should equal iso
    total_from_parts = breakdown["kinetic"] + breakdown["iso"] + breakdown["aniso"] + breakdown["pressure"] + breakdown["hook"]
    assert breakdown["total"] == pytest.approx(total_from_parts)


def test_gH_fallback_to_h_squared():
    """If g_H_squared is absent, use h_squared for the Hook term parameter."""
    config = _minimal_config()
    # Remove g_H_squared and set h_squared explicitly
    config.pop("g_H_squared", None)
    config["h_squared"] = 3.14
    solver = HopfionRelaxer(config)
    assert solver.h_sq == pytest.approx(3.14)


def test_run_relaxation_early_exit_threshold_triggers_nan():
    """Setting an extremely low early-exit threshold forces an immediate early return with NaN energy."""
    config = _minimal_config(lattice_size=(4, 4, 4, 4), dx=0.25)
    # Force early exit by using a very low threshold so E > threshold
    config["early_exit_energy_threshold"] = -1e300
    # Ensure we check energy every step for rapid early exit
    config["energy_check_interval"] = 1
    solver = HopfionRelaxer(config)
    solver.initialize_field(nw=1)

    phi_series, final_energy = solver.run_relaxation(record_series=True)
    assert phi_series is not None
    assert np.isnan(final_energy)


def test_timestep_adapts_down_when_updates_too_large():
    """If the proposed update is too large, dt should be reduced by dt_down_factor."""
    config = _minimal_config(lattice_size=(4, 4, 4, 4), dx=0.25)
    config.update({
        "relaxation_dt": 1e-2,            # large enough to trigger big updates
        "max_phi_change_per_step": 1e-9,  # extremely small threshold to force reduction
        "min_dt": 1e-12,
        "dt_down_factor": 0.1,
        "energy_check_interval": 1000,    # avoid backtracking affecting dt in this short run
        "max_iterations": 1,
        "early_exit_energy_threshold": 1e300,  # avoid early exit
    })
    solver = HopfionRelaxer(config)
    solver.initialize_field(nw=1)

    dt_initial = solver.dt
    _series, _E = solver.run_relaxation(record_series=False)
    assert solver.dt < dt_initial

    # A pytest fixture creates a reusable object for your tests.
@pytest.fixture
def base_config():
    """A default configuration for a small lattice."""
    return {
        "lattice_size": (8, 8, 8, 8),
        "dx": 0.5,
        "mu_squared": 2.0,
        "lambda_val": 1.5,
        "g_squared": 0.5,
        "h_squared": 0.2,
        "P_env": 0.1,
        # Set relaxation parameters to something simple
        "relaxation_dt": 1e-4, 
        "friction": 0.9,
    }

def test_potential_derivative_vs_numerical(base_config):
    """
    Compare the analytic potential derivative with a numerical one
    calculated from the total energy function. This is a robust check.
    """
    # ARRANGE
    # Use a non-trivial field configuration, like a sine wave
    shape = base_config["lattice_size"]
    dims = [np.arange(s, dtype=np.float64) for s in shape]
    x, y, z, w = np.meshgrid(dims[0], dims[1], dims[2], dims[3], indexing="ij")
    
    phi = np.sin(2 * np.pi * x / shape[0]) * np.cos(2 * np.pi * w / shape[3])
    
    # The point we'll test the derivative at
    test_point = (shape[0]//4, shape[1]//4, shape[2]//4, shape[3]//4)
    
    epsilon = 1e-6  # A small perturbation
    
    # Create perturbed versions of the field
    phi_plus = phi.copy()
    phi_plus[test_point] += epsilon
    
    phi_minus = phi.copy()
    phi_minus[test_point] -= epsilon

    # ACT
    # 1. Calculate the analytic derivative using your function
    # Note: We pass all config values directly to the numba function
    dUdphi_analytic = calculate_full_potential_derivative(
        phi,
        mu_squared=base_config["mu_squared"],
        lambda_val=base_config["lambda_val"],
        g_squared=base_config["g_squared"],
        P_env=base_config["P_env"],
        h_squared=base_config["h_squared"],
        dx=base_config["dx"]
    )
    analytic_result = dUdphi_analytic[test_point]

    # 2. Calculate the total energy for the perturbed fields
    # We need a helper function to pass all args to the energy kernel
    def get_energy(p):
        return calculate_full_total_energy(
            p,
            dx=base_config["dx"],
            mu_squared=base_config["mu_squared"],
            lambda_val=base_config["lambda_val"],
            g_squared=base_config["g_squared"],
            P_env=base_config["P_env"],
            h_squared=base_config["h_squared"]
        )

    energy_plus = get_energy(phi_plus)
    energy_minus = get_energy(phi_minus)
    
    # The energy is an integral (sum) over the volume. The derivative δU/δφ is a density.
    # So, we must divide by the volume element of the single point, which is dx^4.
    dv = base_config["dx"]**4
    numerical_result = (energy_plus - energy_minus) / (2 * epsilon * dv)
    
    # ASSERT
    # We expect them to be very close, using a relative tolerance
    assert analytic_result == pytest.approx(numerical_result, rel=1e-5)

def test_potential_derivative_uniform_field(base_config):
    """
    In a uniform field, only the isotropic and pressure terms should contribute.
    """
    # ARRANGE
    shape = base_config["lattice_size"]
    phi_val = 0.5  # An arbitrary value for the field
    phi = np.full(shape, phi_val, dtype=np.float64)
    
    mu2 = base_config["mu_squared"]
    lam = base_config["lambda_val"]
    P = base_config["P_env"]

    # ACT
    dUdphi = calculate_full_potential_derivative(
        phi,
        mu_squared=mu2,
        lambda_val=lam,
        g_squared=base_config["g_squared"], # Pass them in, but they should not matter
        h_squared=base_config["h_squared"],
        P_env=P,
        dx=base_config["dx"]
    )

    # ASSERT
    # The expected result from the docstring formulas:
    expected_dUdphi = -mu2 * phi_val + lam * phi_val**3 + 2 * P * phi_val
    
    # Every single point in the output array should have this value
    assert np.allclose(dUdphi, expected_dUdphi)

def test_hook_coupling_is_confined_to_core(base_config):
    """
    Tests the (1-φ²) factor. The Hook term's contribution should vanish
    in the vacuum (φ²=1), even if gradients are present.
    """
    # ARRANGE
    shape = base_config["lattice_size"]
    dx = base_config["dx"]
    h2 = base_config["h_squared"]
    
    # Create a field that is mostly vacuum (phi=1) but has a "dip" (the core)
    # in the middle where phi is not 1.
    phi = np.ones(shape, dtype=np.float64)
    
    # Define the core region
    center = shape[0] // 2
    radius = shape[0] // 4
    core_slice = slice(center - radius, center + radius)
    
    # Give the core a gradient (e.g., a sine wave profile)
    x = np.arange(2 * radius)
    core_profile = 0.5 * np.sin(np.pi * x / (2 * radius)) + 0.5
    
    # Place this profile into the larger grid in the x-dimension
    phi[core_slice, :, :, :] = core_profile[:, np.newaxis, np.newaxis, np.newaxis]
    
    # ACT
    # Calculate the derivative, turning OFF all other physics to isolate the Hook term
    dUdphi_hook_only = calculate_full_potential_derivative(
        phi,
        mu_squared=0.0,
        lambda_val=0.0,
        g_squared=0.0,
        P_env=0.0,
        h_squared=h2,
        dx=dx
    )

    # ASSERT
    # The derivative should be NON-ZERO inside the core
    core_region_values = dUdphi_hook_only[core_slice, :, :, :]
    assert np.any(core_region_values != 0)

    # The derivative should be EXACTLY ZERO in vacuum regions that are at least
    # one cell away from the core boundary (so local gradients vanish too).
    core_start, core_stop = core_slice.start, core_slice.stop
    left_vacuum = slice(0, max(core_start - 1, 0))
    right_vacuum = slice(min(core_stop + 1, shape[0]), shape[0])

    if left_vacuum.stop > left_vacuum.start:
        left_vals = dUdphi_hook_only[left_vacuum, :, :, :]
        assert np.all(left_vals == 0)
    if right_vacuum.stop > right_vacuum.start:
        right_vals = dUdphi_hook_only[right_vacuum, :, :, :]
        assert np.all(right_vals == 0)