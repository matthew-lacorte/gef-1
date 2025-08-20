# src/gef/physics_core/solvers/hopfion_relaxer.py

# --------------------------------------------------------------------------
# This file will contain the core, JIT-compiled numerical methods for
# relaxing the 4D scalar field.
# --------------------------------------------------------------------------

import os

import numpy as np
from numba import config, njit, prange

# Configure Numba threading conservatively and respect existing environment
# Do NOT override if the user already set NUMBA_NUM_THREADS.
if "NUMBA_NUM_THREADS" not in os.environ:
    # Allow alternate env knob 'GEF_NUMBA_NUM_THREADS'; default to 1 to avoid oversubscription under multiprocessing.
    os.environ["NUMBA_NUM_THREADS"] = os.environ.get("GEF_NUMBA_NUM_THREADS", "1")

# Prefer TBB threading layer on ARM if not specified by the environment
if "NUMBA_THREADING_LAYER" not in os.environ:
    config.THREADING_LAYER = "tbb"


# Enable fast math for additional speedup (if numerical precision allows)
@njit(parallel=True, fastmath=True, cache=True)
def calculate_laplacian_4d(phi: np.ndarray, dx: float) -> np.ndarray:
    """
    4D Laplacian calculation.
    """
    laplacian = np.zeros_like(phi)
    dx2_inv = 1.0 / (dx * dx)  # Pre-compute inverse

    n0, n1, n2, n3 = phi.shape

    # Use explicit loops with better cache locality
    for i in prange(n0):
        ip1 = (i + 1) % n0
        im1 = (i - 1) % n0

        for j in range(n1):
            jp1 = (j + 1) % n1
            jm1 = (j - 1) % n1

            for k in range(n2):
                kp1 = (k + 1) % n2
                km1 = (k - 1) % n2

                for l in range(n3):
                    lp1 = (l + 1) % n3
                    lm1 = (l - 1) % n3

                    center = phi[i, j, k, l]
                    neighbors_sum = (
                        phi[ip1, j, k, l]
                        + phi[im1, j, k, l]
                        + phi[i, jp1, k, l]
                        + phi[i, jm1, k, l]
                        + phi[i, j, kp1, l]
                        + phi[i, j, km1, l]
                        + phi[i, j, k, lp1]
                        + phi[i, j, k, lm1]
                    )

                    laplacian[i, j, k, l] = (neighbors_sum - 8.0 * center) * dx2_inv

    return laplacian


@njit(fastmath=True, cache=True)
def calculate_potential_derivative(
    phi: np.ndarray, mu_squared: float, lambda_val: float
) -> np.ndarray:
    """Potential derivative calculation."""
    return -mu_squared * phi + lambda_val * (phi * phi * phi)


@njit(fastmath=True, cache=True)
def calculate_total_energy(
    phi: np.ndarray, laplacian: np.ndarray, mu_squared: float, lambda_val: float, dx: float
) -> float:
    """Energy calculation."""
    dx4 = dx * dx * dx * dx

    # Calculate energy components in single pass
    total_energy = 0.0
    phi_flat = phi.flat
    lap_flat = laplacian.flat

    for i in range(phi_flat.size):
        phi_val = phi_flat[i]
        phi_sq = phi_val * phi_val

        # Kinetic energy density
        kinetic = -0.5 * phi_val * lap_flat[i]

        # Potential energy density
        potential = -0.5 * mu_squared * phi_sq + 0.25 * lambda_val * phi_sq * phi_sq

        total_energy += kinetic + potential

    return total_energy * dx4


@njit(parallel=False, fastmath=True, cache=True)
def calculate_anisotropic_potential_derivative(
    phi: np.ndarray, mu_squared: float, lambda_val: float, g_squared: float, P_env: float, dx: float
) -> np.ndarray:
    """Anisotropic potential derivative."""
    result = np.zeros_like(phi)
    inv_2dx = 0.5 / dx
    dx2_inv = 1.0 / (dx * dx)

    n0, n1, n2, n3 = phi.shape

    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                for l in range(n3):
                    phi_val = phi[i, j, k, l]
                    phi_sq = phi_val * phi_val

                    # Isotropic part
                    iso_part = -mu_squared * phi_val + lambda_val * phi_val * phi_sq

                    # Pressure term from L_pressure = P * (1 - Φ²)
                    # The potential is U_pressure = -P * (1 - Φ²), so dU/dΦ = 2 * P * Φ
                    pressure_part = 2.0 * P_env * phi_val

                    # Anisotropic part - w-direction derivatives
                    l_plus = (l + 1) % n3
                    l_minus = (l - 1) % n3

                    # First derivative in w
                    grad_w = (phi[i, j, k, l_plus] - phi[i, j, k, l_minus]) * inv_2dx
                    grad_w_sq = grad_w * grad_w

                    # Second derivative in w
                    d2_phi_dw2 = (
                        phi[i, j, k, l_plus] - 2.0 * phi_val + phi[i, j, k, l_minus]
                    ) * dx2_inv

                    one_minus_phi_sq = 1.0 - phi_sq

                    term1 = -4.0 * g_squared * one_minus_phi_sq * phi_val * grad_w_sq
                    term2 = -2.0 * g_squared * one_minus_phi_sq * one_minus_phi_sq * d2_phi_dw2

                    result[i, j, k, l] = iso_part + pressure_part - (term1 + term2)

    return result


# In your new optimized solver file...

# def seed_hopfion_optimized(lattice_shape: tuple, nw: int, dx: float) -> np.ndarray:
#     """
#     Optimized Hopfion seeding with proper multi-winding support via product ansatz.
#     """
#     if nw == 0:
#         print("INFO: Seeding lattice with N_w = 0 (vacuum state).")
#         return np.ones(lattice_shape, dtype=np.float64)

#     print(f"INFO: Seeding lattice with analytical Hopfion for N_w = {nw}...")

#     dims = [np.arange(-s / 2. + 0.5, s / 2. + 0.5, dtype=np.float32) * dx for s in lattice_shape]
#     x, y, z, w = np.meshgrid(dims[0], dims[1], dims[2], dims[3], indexing='ij')

#     epsilon = 1e-9

#     # 1. Define the base N_w=1 complex fields
#     z0 = x + 1j * y
#     z1 = z + 1j * w

#     # 2. Use the product ansatz for N_w > 1
#     # We model a multi-winding Hopfion by twisting one of the complex planes.
#     # The complex field for the topology is Psi = z0^nw / z1.
#     # We use z0 and z1 to define our real scalar field Phi.

#     # The field is defined by the ratio of the norms of the "twisted" complex numbers
#     # Let the twisted numbers be Z0 = z0^nw and Z1 = z1.
#     # |Z0|² = |z0^nw|² = (|z0|^nw)² = (x²+y²)^(nw)
#     # |Z1|² = |z1|² = z²+w²
#     # This leads to a complex formula.

#     # A more standard and stable approach uses the complex phase directly.
#     # Let Psi = z0 / z1.
#     # A multi-winding state is created by multiplying the phase of Psi by nw.
#     # This is equivalent to twisting the z0 plane.

#     # Let's use the simplest formulation that works:
#     # Φ(x,y,z,w) = ( |z₁|² - |z₀|² ) / ( |z₁|² + |z₀|² )
#     # A multi-winding state can be seeded by taking the argument of z0 (atan2(y,x))
#     # and multiplying it by nw before reconstructing x' and y'.

#     if nw > 1:
#         # This is the proper implementation of the product ansatz
#         angle_z0 = np.arctan2(y, x)
#         radius_z0 = np.sqrt(x*x + y*y)

#         # Multiply the angle by the winding number
#         new_angle_z0 = nw * angle_z0

#         # Reconstruct the "twisted" x and y coordinates
#         x_twisted = radius_z0 * np.cos(new_angle_z0)
#         y_twisted = radius_z0 * np.sin(new_angle_z0)

#         # Calculate the final Phi using the twisted coordinates
#         phi_final = (z*z + w*w - x_twisted**2 - y_twisted**2) / (x_twisted**2 + y_twisted**2 + z*z + w*w + epsilon)

#     else: # nw == 1
#         r_sq = x*x + y*y + z*z + w*w
#         phi_final = (z*z + w*w - x*x - y*y) / (r_sq + epsilon)

#     print("INFO: Seeding complete.")
#     return phi_final.astype(np.float64)

# def seed_hopfion(lattice_shape: tuple, nw: int, dx: float) -> np.ndarray:
#     """
#     Optimized Hopfion seeding with proper multi-winding support via product ansatz.
#     """
#     if nw == 0:
#         print("INFO: Seeding lattice with N_w = 0 (vacuum state).")
#         return np.ones(lattice_shape, dtype=np.float64)

#     print(f"INFO: Seeding lattice with analytical Hopfion for N_w = {nw}...")

#     dims = [np.arange(-s / 2. + 0.5, s / 2. + 0.5, dtype=np.float64) * dx for s in lattice_shape]
#     x, y, z, w = np.meshgrid(dims[0], dims[1], dims[2], dims[3], indexing='ij')

#     epsilon = 1e-9

#     # Use the product ansatz: A multi-winding state Ψ_k = (Ψ_1)^k
#     # can be modeled by multiplying the angle in one of the complex planes.
#     # Let z₀ = x + iy. We will "twist" this plane.

#     # Calculate the angle and radius in the z₀ (xy) plane
#     angle_z0 = np.arctan2(y, x)
#     radius_z0_sq = x**2 + y**2

#     # Multiply the angle by the winding number to create the twist
#     new_angle_z0 = nw * angle_z0

#     # Reconstruct the "twisted" x and y coordinates
#     # We need the radius, not radius squared for this
#     radius_z0 = np.sqrt(radius_z0_sq)
#     x_twisted = radius_z0 * np.cos(new_angle_z0)
#     y_twisted = radius_z0 * np.sin(new_angle_z0)

#     # Calculate the final Phi using the twisted coordinates
#     # Φ = ( |z₁|² - |z₀'|² ) / ( |z₁|² + |z₀'|² )
#     # where z₀' are the twisted coordinates
#     phi_final = (z**2 + w**2 - x_twisted**2 - y_twisted**2) / (x_twisted**2 + y_twisted**2 + z**2 + w**2 + epsilon)

#     print("INFO: Seeding complete.")
#     return phi_final


# This is the ONLY version of seed_hopfion we need now.
def seed_hopfion(lattice_shape: tuple, nw: int, dx: float) -> np.ndarray:
    """
    Creates the analytical seed for a fundamental N_w=1 Hopfion.
    The nw parameter is kept for API consistency but should always be 1 for this experiment.
    """
    if nw == 0:
        return np.ones(lattice_shape, dtype=np.float64)

    print("INFO: Seeding lattice with fundamental N_w=1 topology...")

    dims = [np.arange(-s / 2.0 + 0.5, s / 2.0 + 0.5, dtype=np.float64) * dx for s in lattice_shape]
    x, y, z, w = np.meshgrid(dims[0], dims[1], dims[2], dims[3], indexing="ij")

    epsilon = 1e-9
    r_sq = x**2 + y**2 + z**2 + w**2

    # The standard, fundamental N_w=1 formula
    phi_final = (z**2 + w**2 - x**2 - y**2) / (r_sq + epsilon)

    print("INFO: Seeding complete.")
    return phi_final


class HopfionRelaxer:
    """Optimized relaxation solver for Hopfion field configurations using comprehensive physics functions."""

    def __init__(self, config: dict):
        """Initialize the solver with the given configuration."""
        self.config = config

        # Grid parameters
        self.lattice_shape = tuple(config["lattice_size"])
        self.dx = config["dx"]

        # Physics parameters
        self.mu2 = config["mu_squared"]
        self.lam = config["lambda_val"]
        self.g_sq = config.get("g_squared", 0.0)
        self.P_env = config.get("P_env", 0.0)
        self.h_sq = config.get("h_squared", 0.0)

        # Relaxation parameters
        self.dt = config["relaxation_dt"]
        self.friction = config.get("friction", 0.95)

        # Adaptive timestepping
        self.min_dt = config.get("min_dt", 1e-9)
        self.max_dt = config.get("max_dt", 1e-3)
        self.dt_down_factor = config.get("dt_down_factor", 0.5)
        self.dt_up_factor = config.get("dt_up_factor", 1.01)
        self.max_phi_change_per_step = config.get("max_phi_change_per_step", 0.01)

        # Convergence & Exit conditions
        self.max_iterations = config.get("max_iterations", 20000)
        self.convergence_threshold = config.get("convergence_threshold", 1e-8)
        self.early_exit_energy_threshold = config.get("early_exit_energy_threshold", 1e6)

        # Buffers
        self.phi = np.zeros(self.lattice_shape, dtype=np.float64)
        self.velocity_buffer = np.zeros_like(self.phi)
        self.laplacian_buffer = np.zeros_like(self.phi)

    def initialize_field(self, nw: int):
        """Initialize the field with an analytical seed."""
        self.phi = seed_hopfion(self.lattice_shape, nw, self.dx)

    def run_relaxation(self, n_skip=None, n_iter=None, record_series: bool = False):
        """Main relaxation loop with adaptive timestepping and convergence checks."""
        last_energy = 0.0
        converged = False
        phi_series = [] if record_series else None

        # The calibration script uses n_skip and n_iter to control total iterations
        # for its warmup and main runs. We respect that if provided.
        total_iterations = self.max_iterations
        if n_skip is not None and n_iter is not None:
            total_iterations = n_skip + n_iter

        for i in range(total_iterations):
            # Calculate force
            force = calculate_laplacian_4d(
                self.phi, self.dx, out=self.laplacian_buffer
            ) - calculate_full_potential_derivative(
                self.phi, self.mu2, self.lam, self.g_sq, self.P_env, self.h_sq, self.dx
            )

            # Update velocity and position with momentum
            self.velocity_buffer = (self.friction * self.velocity_buffer) + (self.dt * force)
            update_step = self.velocity_buffer

            # --- Adaptive Timestep Control ---
            max_change = np.max(np.abs(update_step))
            if max_change > self.max_phi_change_per_step:
                self.dt = max(self.min_dt, self.dt * self.dt_down_factor)
                # Reject step by re-calculating with smaller dt
                self.velocity_buffer = (
                    self.friction * self.velocity_buffer
                ) + (self.dt * force)
                update_step = self.velocity_buffer

            # Slowly increase timestep if stable
            if i % 50 == 0 and i > 0:
                self.dt = min(self.max_dt, self.dt * self.dt_up_factor)

            self.phi += update_step

            # --- Convergence & Early Exit Checks (every 100 steps) ---
            if i % 100 == 0:
                current_energy = calculate_full_total_energy(
                    self.phi, self.dx, self.mu2, self.lam, self.g_sq, self.P_env, self.h_sq
                )

                if record_series:
                    phi_series.append(self.phi[self.lattice_shape[0] // 4, self.lattice_shape[1] // 4, self.lattice_shape[2] // 4, self.lattice_shape[3] // 4])

                # 1. Check for unstable explosion
                if not np.isfinite(current_energy) or current_energy > self.early_exit_energy_threshold:
                    return None if not record_series else np.array(phi_series), np.nan

                # 2. Check for convergence
                if i > 100:
                    energy_change = abs(current_energy - last_energy)
                    # Power dissipation is a good metric for convergence
                    power_dissipation = energy_change / self.dt
                    if power_dissipation < self.convergence_threshold:
                        converged = True
                        break
                last_energy = current_energy

        final_energy = calculate_full_total_energy(
            self.phi, self.dx, self.mu2, self.lam, self.g_sq, self.P_env, self.h_sq
        )

        if not converged:
            # If we maxed out iterations without converging, it's not a stable state
            final_energy = np.nan

        return None if not record_series else np.array(phi_series), final_energy

@njit(fastmath=True, cache=True)
def calculate_final_total_energy(
    phi: np.ndarray,
    dx: float,
    mu_squared: float,
    lambda_val: float,
    g_squared: float,
    P_env: float,
    h_squared: float,
) -> float:
    """Anisotropic energy calculation."""
    dx4 = dx * dx * dx * dx
    inv_2dx = 0.5 / dx

    total_energy = 0.0
    n0, n1, n2, n3 = phi.shape

    # Single pass through the array
    for i in range(n0):
        ip1 = (i + 1) % n0
        im1 = (i - 1) % n0

        for j in range(n1):
            jp1 = (j + 1) % n1
            jm1 = (j - 1) % n1

            for k in range(n2):
                kp1 = (k + 1) % n2
                km1 = (k - 1) % n2

                for l in range(n3):
                    lp1 = (l + 1) % n3
                    lm1 = (l - 1) % n3

                    phi_val = phi[i, j, k, l]
                    phi_sq = phi_val * phi_val

                    # Kinetic energy (laplacian calculation inline)
                    neighbors_sum = (
                        phi[ip1, j, k, l]
                        + phi[im1, j, k, l]
                        + phi[i, jp1, k, l]
                        + phi[i, jm1, k, l]
                        + phi[i, j, kp1, l]
                        + phi[i, j, km1, l]
                        + phi[i, j, k, lp1]
                        + phi[i, j, k, lm1]
                    )

                    laplacian_val = (neighbors_sum - 8.0 * phi_val) / (dx * dx)
                    kinetic_density = -0.5 * phi_val * laplacian_val

                    # Isotropic potential
                    potential_iso_density = (
                        -0.5 * mu_squared * phi_sq + 0.25 * lambda_val * phi_sq * phi_sq
                    )

                    # Anisotropic potential
                    grad_w = (phi[i, j, k, lp1] - phi[i, j, k, lm1]) * inv_2dx
                    grad_w_sq = grad_w * grad_w
                    one_minus_phi_sq = 1.0 - phi_sq
                    potential_aniso_density = (
                        0.5 * g_squared * one_minus_phi_sq * one_minus_phi_sq * grad_w_sq
                    )

                    # Environmental pressure potential energy U_pressure = -L_pressure
                    potential_pressure_density = -P_env * one_minus_phi_sq

                    # Hook coupling potential energy
                    grad_x = (phi[ip1, j, k, l] - phi[im1, j, k, l]) * inv_2dx
                    grad_y = (phi[i, jp1, k, l] - phi[i, jm1, k, l]) * inv_2dx
                    grad_x_sq = grad_x * grad_x
                    grad_y_sq = grad_y * grad_y
                    potential_hook_density = (
                        0.5 * h_squared * (grad_x_sq + grad_y_sq) * one_minus_phi_sq
                    )

                    total_energy += (
                        kinetic_density
                        + potential_iso_density
                        + potential_aniso_density
                        + potential_pressure_density
                        + potential_hook_density
                    )

    return total_energy * dx4


@njit(parallel=False, fastmath=True, cache=True)
def calculate_full_potential_derivative(
    phi: np.ndarray,
    mu_squared: float,
    lambda_val: float,
    g_squared: float,
    P_env: float,
    h_squared: float,
    dx: float,
) -> np.ndarray:
    """
    Calculates dU/dΦ for the FULL model: U_iso + U_aniso + U_pressure.
    This is the CORRECTED version with the proper sign for the anisotropic term.
    """
    result = np.zeros_like(phi)
    inv_2dx = 0.5 / dx
    dx2_inv = 1.0 / (dx * dx)

    n0, n1, n2, n3 = phi.shape

    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                for l in range(n3):
                    phi_val = phi[i, j, k, l]
                    phi_sq = phi_val * phi_val

                    # Term 1: Isotropic Mexican-hat Potential
                    dUdPhi_iso = -mu_squared * phi_val + lambda_val * phi_val * phi_sq

                    # Term 2: External Environmental Pressure
                    dUdPhi_pressure = 2.0 * P_env * phi_val

                    # Term 3: Anisotropic Internal Stabilizer
                    l_plus = (l + 1) % n3
                    l_minus = (l - 1) % n3

                    grad_w = (phi[i, j, k, l_plus] - phi[i, j, k, l_minus]) * inv_2dx
                    grad_w_sq = grad_w * grad_w

                    d2_phi_dw2 = (
                        phi[i, j, k, l_plus] - 2.0 * phi_val + phi[i, j, k, l_minus]
                    ) * dx2_inv

                    one_minus_phi_sq = 1.0 - phi_sq

                    # The functional derivative of U_aniso. These terms represent dU_aniso/dΦ.
                    aniso_term1 = -4.0 * g_squared * one_minus_phi_sq * phi_val * grad_w_sq
                    aniso_term2 = -2.0 * g_squared * (one_minus_phi_sq**2) * d2_phi_dw2

                    dUdPhi_aniso = aniso_term1 + aniso_term2

                    # Term 4: Hook Coupling Force
                    # From L_hook = -½ h² * ( (∂_x Φ)² + (∂_y Φ)² ) * (1 - Φ²)
                    # U_hook =  ½ h² * ( (∂_x Φ)² + (∂_y Φ)² ) * (1 - Φ²)
                    # The derivative dU_hook/dΦ has two parts.

                    # First derivatives in x and y
                    i_plus = (i + 1) % n0
                    i_minus = (i - 1) % n0
                    j_plus = (j + 1) % n1
                    j_minus = (j - 1) % n1

                    grad_x = (phi[i_plus, j, k, l] - phi[i_minus, j, k, l]) * inv_2dx
                    grad_y = (phi[i, j_plus, k, l] - phi[i, j_minus, k, l]) * inv_2dx

                    # Second derivatives in x and y
                    d2_phi_dx2 = (
                        phi[i_plus, j, k, l] - 2.0 * phi_val + phi[i_minus, j, k, l]
                    ) * dx2_inv
                    d2_phi_dy2 = (
                        phi[i, j_plus, k, l] - 2.0 * phi_val + phi[i, j_minus, k, l]
                    ) * dx2_inv

                    hook_term1 = h_squared * (grad_x**2 + grad_y**2) * (-2.0 * phi_val)
                    hook_term2 = -h_squared * (1.0 - phi_sq) * 2.0 * (d2_phi_dx2 + d2_phi_dy2)

                    dUdPhi_hook = hook_term1 + hook_term2

                    # The total derivative dU_total/dΦ is the sum of all parts.
                    result[i, j, k, l] = dUdPhi_iso + dUdPhi_pressure + dUdPhi_aniso + dUdPhi_hook

    return result


@njit(parallel=True, fastmath=True, cache=True)
def calculate_full_total_energy(
    phi: np.ndarray,
    dx: float,
    mu_squared: float,
    lambda_val: float,
    g_squared: float,
    P_env: float,
    h_squared: float,
) -> float:
    """Calculates total energy for the full anisotropic potential + pressure model."""
    dx4 = dx * dx * dx * dx
    inv_2dx = 0.5 / dx

    # Use a parallel reduction approach for energy calculation
    energy_array = np.zeros(phi.shape[0], dtype=np.float64)
    n0, n1, n2, n3 = phi.shape

    for i in prange(n0):
        local_energy = 0.0
        ip1 = (i + 1) % n0
        im1 = (i - 1) % n0
        for j in range(n1):
            jp1 = (j + 1) % n1
            jm1 = (j - 1) % n1
            for k in range(n2):
                kp1 = (k + 1) % n2
                km1 = (k - 1) % n2
                for l in range(n3):
                    lp1 = (l + 1) % n3
                    lm1 = (l - 1) % n3

                    phi_val = phi[i, j, k, l]
                    phi_sq = phi_val * phi_val

                    # Kinetic Energy Density (from integration by parts: -½Φ∇²Φ)
                    neighbors_sum = (
                        phi[ip1, j, k, l]
                        + phi[im1, j, k, l]
                        + phi[i, jp1, k, l]
                        + phi[i, jm1, k, l]
                        + phi[i, j, kp1, l]
                        + phi[i, j, km1, l]
                        + phi[i, j, k, lp1]
                        + phi[i, j, k, lm1]
                    )
                    laplacian_val = (neighbors_sum - 8.0 * phi_val) / (dx * dx)
                    kinetic_density = -0.5 * phi_val * laplacian_val

                    # Isotropic Potential Density
                    potential_iso_density = -0.5 * mu_squared * phi_sq + 0.25 * lambda_val * (
                        phi_sq**2
                    )

                    # Anisotropic Stabilizer Potential Density
                    grad_w_sq = ((phi[i, j, k, lp1] - phi[i, j, k, lm1]) * inv_2dx) ** 2
                    one_minus_phi_sq = 1.0 - phi_sq
                    potential_aniso_density = 0.5 * g_squared * (one_minus_phi_sq**2) * grad_w_sq

                    # Environmental Pressure Potential Density U_pressure = -L_pressure
                    potential_pressure_density = -P_env * one_minus_phi_sq

                    # Hook Coupling Potential Density
                    grad_x_sq = ((phi[ip1, j, k, l] - phi[im1, j, k, l]) * inv_2dx) ** 2
                    grad_y_sq = ((phi[i, jp1, k, l] - phi[i, jm1, k, l]) * inv_2dx) ** 2
                    potential_hook_density = (
                        0.5 * h_squared * (grad_x_sq + grad_y_sq) * (1.0 - phi_sq)
                    )

                    local_energy += (
                        kinetic_density
                        + potential_iso_density
                        + potential_aniso_density
                        + potential_pressure_density
                        + potential_hook_density
                    )

        energy_array[i] = local_energy

    return np.sum(energy_array) * dx4


# Add missing out parameter support
@njit(parallel=True, fastmath=True, cache=True)
def calculate_laplacian_4d(phi: np.ndarray, dx: float, out: np.ndarray = None) -> np.ndarray:
    """
    4D Laplacian with optional output buffer.
    """
    if out is None:
        out = np.zeros_like(phi)

    dx2_inv = 1.0 / (dx * dx)
    n0, n1, n2, n3 = phi.shape

    for i in prange(n0):
        ip1 = (i + 1) % n0
        im1 = (i - 1) % n0

        for j in range(n1):
            jp1 = (j + 1) % n1
            jm1 = (j - 1) % n1

            for k in range(n2):
                kp1 = (k + 1) % n2
                km1 = (k - 1) % n2

                for l in range(n3):
                    lp1 = (l + 1) % n3
                    lm1 = (l - 1) % n3

                    center = phi[i, j, k, l]
                    neighbors_sum = (
                        phi[ip1, j, k, l]
                        + phi[im1, j, k, l]
                        + phi[i, jp1, k, l]
                        + phi[i, jm1, k, l]
                        + phi[i, j, kp1, l]
                        + phi[i, j, km1, l]
                        + phi[i, j, k, lp1]
                        + phi[i, j, k, lm1]
                    )

                    out[i, j, k, l] = (neighbors_sum - 8.0 * center) * dx2_inv

    return out
