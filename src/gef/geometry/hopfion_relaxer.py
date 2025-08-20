# src/gef/physics_core/solvers/hopfion_relaxer.py

# --------------------------------------------------------------------------
# This file will contain the core, JIT-compiled numerical methods for
# relaxing the 4D scalar field.
# --------------------------------------------------------------------------

import os
import logging
import numpy as np
from numba import config, njit, prange

# --- Numba Configuration ---
logger = logging.getLogger(__name__)

if "NUMBA_NUM_THREADS" not in os.environ:
    os.environ["NUMBA_NUM_THREADS"] = os.environ.get("GEF_NUMBA_NUM_THREADS", "1")

if "NUMBA_THREADING_LAYER" not in os.environ:
    config.THREADING_LAYER = "tbb"

# ==========================================================================
# CORE NUMERICAL KERNELS (JIT-COMPILED)
# ==========================================================================

@njit(parallel=True, fastmath=True, cache=True)
def calculate_laplacian_4d(phi: np.ndarray, dx: float, out: np.ndarray) -> np.ndarray:
    """
    4D Laplacian with optional output buffer.
    """
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
                        phi[ip1, j, k, l] + phi[im1, j, k, l] +
                        phi[i, jp1, k, l] + phi[i, jm1, k, l] +
                        phi[i, j, kp1, l] + phi[i, j, km1, l] +
                        phi[i, j, k, lp1] + phi[i, j, k, lm1]
                    )
                    out[i, j, k, l] = (neighbors_sum - 8.0 * center) * dx2_inv
    return out


@njit(parallel=True, fastmath=True, cache=True)
def calculate_full_potential_derivative(
    phi: np.ndarray, mu_squared: float, lambda_val: float, g_squared: float,
    P_env: float, h_squared: float, dx: float
) -> np.ndarray:
    """
    Calculates dU/dΦ for the FULL model, consistent with the total energy functional.
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
                    one_minus_phi_sq = 1.0 - phi_sq

                    # Indices for finite differences
                    ip1, im1 = (i + 1) % n0, (i - 1) % n0
                    jp1, jm1 = (j + 1) % n1, (j - 1) % n1
                    lp1, lm1 = (l + 1) % n3, (l - 1) % n3

                    # Term 1: Isotropic Mexican-hat Potential
                    # U_iso = -½μ²Φ² + ¼λΦ⁴  =>  dU/dΦ = -μ²Φ + λΦ³
                    dUdPhi_iso = -mu_squared * phi_val + lambda_val * phi_val * phi_sq

                    # Term 2: External Environmental Pressure
                    # U_pressure = -P(1 - Φ²)  =>  dU/dΦ = 2PΦ
                    dUdPhi_pressure = 2.0 * P_env * phi_val

                    # Term 3: Anisotropic Internal Stabilizer
                    # U_aniso = ½g²(1-Φ²)²(∂_wΦ)²
                    # δU/δΦ = -2g²(1-Φ²)Φ(∂_wΦ)² - g²(1-Φ²)²(∂²_wΦ)
                    grad_w = (phi[i, j, k, lp1] - phi[i, j, k, lm1]) * inv_2dx
                    d2_phi_dw2 = (phi[i, j, k, lp1] - 2.0 * phi_val + phi[i, j, k, lm1]) * dx2_inv
                    
                    # CORRECTED: Factors of 4->2 and 2->1 removed to match energy functional
                    aniso_term1 = -2.0 * g_squared * one_minus_phi_sq * phi_val * (grad_w**2)
                    aniso_term2 = -1.0 * g_squared * (one_minus_phi_sq**2) * d2_phi_dw2
                    dUdPhi_aniso = aniso_term1 + aniso_term2

                    # Term 4: Hook Coupling Force
                    # U_hook = ½h²(1-Φ²)((∂_xΦ)²+(∂_yΦ)²)
                    # δU/δΦ = -h²Φ((∂_xΦ)²+(∂_yΦ)²) - h²(1-Φ²)(∂²_xΦ+∂²_yΦ)
                    grad_x = (phi[ip1, j, k, l] - phi[im1, j, k, l]) * inv_2dx
                    grad_y = (phi[i, jp1, k, l] - phi[i, jm1, k, l]) * inv_2dx
                    d2_phi_dx2 = (phi[ip1, j, k, l] - 2.0 * phi_val + phi[im1, j, k, l]) * dx2_inv
                    d2_phi_dy2 = (phi[i, jp1, k, l] - 2.0 * phi_val + phi[i, jm1, k, l]) * dx2_inv

                    # CORRECTED: Factors of 2 removed from both terms
                    hook_term1 = -1.0 * h_squared * phi_val * (grad_x**2 + grad_y**2)
                    hook_term2 = -1.0 * h_squared * one_minus_phi_sq * (d2_phi_dx2 + d2_phi_dy2)
                    dUdPhi_hook = hook_term1 + hook_term2
                    
                    result[i, j, k, l] = dUdPhi_iso + dUdPhi_pressure + dUdPhi_aniso + dUdPhi_hook

    return result


@njit(parallel=True, fastmath=True, cache=True)
def calculate_full_total_energy(
    phi: np.ndarray, dx: float, mu_squared: float, lambda_val: float,
    g_squared: float, P_env: float, h_squared: float
) -> float:
    """Calculates total energy for the full anisotropic potential + pressure model."""
    dx4 = dx**4
    inv_2dx = 0.5 / dx
    energy_array = np.zeros(phi.shape[0], dtype=np.float64)
    n0, n1, n2, n3 = phi.shape

    for i in prange(n0):
        local_energy = 0.0
        ip1, im1 = (i + 1) % n0, (i - 1) % n0
        for j in range(n1):
            jp1, jm1 = (j + 1) % n1, (j - 1) % n1
            for k in range(n2):
                kp1, km1 = (k + 1) % n2, (k - 1) % n2
                for l in range(n3):
                    lp1, lm1 = (l + 1) % n3, (l - 1) % n3

                    phi_val = phi[i, j, k, l]
                    phi_sq = phi_val**2
                    one_minus_phi_sq = 1.0 - phi_sq

                    # Kinetic Energy Density (from integration by parts: -½Φ∇²Φ)
                    neighbors_sum = (phi[ip1,j,k,l]+phi[im1,j,k,l]+phi[i,jp1,k,l]+phi[i,jm1,k,l]+
                                     phi[i,j,kp1,l]+phi[i,j,km1,l]+phi[i,j,k,lp1]+phi[i,j,k,lm1])
                    laplacian_val = (neighbors_sum - 8.0 * phi_val) / (dx * dx)
                    kinetic_density = -0.5 * phi_val * laplacian_val

                    # Isotropic Potential Density
                    potential_iso_density = -0.5 * mu_squared * phi_sq + 0.25 * lambda_val * (phi_sq**2)

                    # Anisotropic Stabilizer Potential Density
                    grad_w_sq = ((phi[i,j,k,lp1] - phi[i,j,k,lm1]) * inv_2dx)**2
                    potential_aniso_density = 0.5 * g_squared * (one_minus_phi_sq**2) * grad_w_sq

                    # Environmental Pressure Potential Density
                    potential_pressure_density = -P_env * one_minus_phi_sq

                    # Hook Coupling Potential Density
                    grad_x_sq = ((phi[ip1,j,k,l] - phi[im1,j,k,l]) * inv_2dx)**2
                    grad_y_sq = ((phi[i,jp1,k,l] - phi[i,jm1,k,l]) * inv_2dx)**2
                    potential_hook_density = 0.5 * h_squared * (grad_x_sq + grad_y_sq) * one_minus_phi_sq

                    local_energy += (kinetic_density + potential_iso_density + potential_aniso_density +
                                     potential_pressure_density + potential_hook_density)
        energy_array[i] = local_energy

    return np.sum(energy_array) * dx4


# ==========================================================================
# SIMULATION CONTROL
# ==========================================================================

def seed_hopfion(lattice_shape: tuple, nw: int, dx: float) -> np.ndarray:
    """
    Creates an analytical seed for a Hopfion with a given winding number N_w.
    """
    if nw == 0:
        return np.ones(lattice_shape, dtype=np.float64)

    logger.info(f"Seeding lattice with a proper N_w={nw} topological seed...")
    dims = [np.arange(-s/2.0 + 0.5, s/2.0 + 0.5, dtype=np.float64) * dx for s in lattice_shape]
    x, y, z, w = np.meshgrid(dims[0], dims[1], dims[2], dims[3], indexing="ij")

    epsilon = 1e-12
    a = x**2 + y**2
    b = z**2 + w**2
    denom_ab = a + b + epsilon
    a_norm = np.clip(a / denom_ab, 0.0, 1.0)
    b_norm = np.clip(b / denom_ab, 0.0, 1.0)

    nw_half = float(nw) / 2.0
    a_pow = np.power(a_norm, nw_half)
    b_pow = np.power(b_norm, nw_half)

    numerator = b_pow - a_pow
    denominator = b_pow + a_pow + epsilon
    phi_final = numerator / denominator
    phi_final = np.where(np.isfinite(phi_final), phi_final, 0.0)
    
    logger.info(f"Seeding for N_w={nw} complete.")
    return phi_final


class HopfionRelaxer:
    """Optimized relaxation solver for Hopfion field configurations using comprehensive physics functions."""

    def __init__(self, config: dict):
        self.config = config
        self.lattice_shape = tuple(config["lattice_size"])
        self.dx = config["dx"]
        self.mu2 = config["mu_squared"]
        self.lam = config["lambda_val"]
        self.g_sq = config.get("g_squared", 0.0)
        self.P_env = config.get("P_env", 0.0)
        self.h_sq = config.get("h_squared", 0.0)
        self.dt = config["relaxation_dt"]
        self.friction = config.get("friction", 0.95)
        self.min_dt = config.get("min_dt", 1e-9)
        self.max_dt = config.get("max_dt", 1e-3)
        self.dt_down_factor = config.get("dt_down_factor", 0.5)
        self.dt_up_factor = config.get("dt_up_factor", 1.01)
        self.max_phi_change_per_step = config.get("max_phi_change_per_step", 0.01)
        self.max_iterations = config.get("max_iterations", 20000)
        self.convergence_threshold = config.get("convergence_threshold", 1e-8)
        self.early_exit_energy_threshold = config.get("early_exit_energy_threshold", 1e6)
        self.phi = np.zeros(self.lattice_shape, dtype=np.float64)
        self.velocity_buffer = np.zeros_like(self.phi)
        self.laplacian_buffer = np.zeros_like(self.phi)

    def initialize_field(self, nw: int):
        self.phi = seed_hopfion(self.lattice_shape, nw, self.dx)

    def run_relaxation(self, n_skip=None, n_iter=None, record_series: bool = False):
        last_energy = 0.0
        converged = False
        phi_series = [] if record_series else None
        
        total_iterations = self.max_iterations
        if n_skip is not None and n_iter is not None:
            total_iterations = n_skip + n_iter

        for i in range(total_iterations):
            calculate_laplacian_4d(self.phi, self.dx, out=self.laplacian_buffer)
            
            dU_dphi = calculate_full_potential_derivative(
                self.phi, self.mu2, self.lam, self.g_sq, self.P_env, self.h_sq, self.dx
            )
            force = self.laplacian_buffer - dU_dphi

            # Store old velocity for potential step rejection
            # This makes the rejection logic cleaner
            prev_velocity = np.copy(self.velocity_buffer)

            self.velocity_buffer = (self.friction * self.velocity_buffer) + (self.dt * force)
            update_step = self.dt * self.velocity_buffer # The update is dt * velocity

            max_change = np.max(np.abs(update_step))
            if max_change > self.max_phi_change_per_step:
                self.dt = max(self.min_dt, self.dt * self.dt_down_factor)
                # Recalculate update with smaller dt, using the velocity from *before* this step
                self.velocity_buffer = (self.friction * prev_velocity) + (self.dt * force)
                update_step = self.dt * self.velocity_buffer
            
            # Slowly increase timestep if stable
            elif i % 50 == 0 and i > 0:
                self.dt = min(self.max_dt, self.dt * self.dt_up_factor)

            self.phi += update_step

            if i % 100 == 0:
                current_energy = calculate_full_total_energy(
                    self.phi, self.dx, self.mu2, self.lam, self.g_sq, self.P_env, self.h_sq
                )

                if record_series:
                    phi_series.append(self.phi[self.lattice_shape[0]//4, self.lattice_shape[1]//4, self.lattice_shape[2]//4, self.lattice_shape[3]//4])

                if not np.isfinite(current_energy) or current_energy > self.early_exit_energy_threshold:
                    return (np.array(phi_series), np.nan) if record_series else (None, np.nan)

                if i > 100:
                    energy_change = abs(current_energy - last_energy)
                    power_dissipation = energy_change / (100 * self.dt) # Avg power over 100 steps
                    if power_dissipation < self.convergence_threshold:
                        converged = True
                        break
                last_energy = current_energy

        final_energy = calculate_full_total_energy(
            self.phi, self.dx, self.mu2, self.lam, self.g_sq, self.P_env, self.h_sq
        )

        if not converged and i >= total_iterations - 1:
            final_energy = np.nan

        return (np.array(phi_series), final_energy) if record_series else (None, final_energy)