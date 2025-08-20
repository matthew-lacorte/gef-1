# src/gef/geometry/hopfion_relaxer.py
"""
HopfionRelaxer: physics-correct relaxation of a 4D real scalar field with
anisotropic stabilizer and Hook-type gradient coupling.

Key improvements in this refactor:
- Ensured physically correct signs for all terms in the potential derivative (δU/δφ).
- Added lightweight energy backtracking (line-search style) for stability.
- Parameter sanity checks to keep the vacuum within |φ| ≤ 1 when Hook term is active.
- Config compatibility for `g_H_squared` ↔ `h_squared`.
- Safer Numba threading-layer selection.
- Optional clipping guardrail `clip_phi_to_unit`.

NOTE (topology): a single real scalar field cannot carry a true Hopf charge.
The current solver can still relax shaped seeds; for topological protection you
will want at least an O(3) / CP¹ field.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple, List

import numpy as np
from numba import config as numba_config, njit, prange

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Numba configuration
# -----------------------------------------------------------------------------
if "NUMBA_NUM_THREADS" not in os.environ:
    os.environ["NUMBA_NUM_THREADS"] = os.environ.get("GEF_NUMBA_NUM_THREADS", "1")

if "NUMBA_THREADING_LAYER" not in os.environ:
    try:
        numba_config.THREADING_LAYER = "tbb"  # Prefer TBB when available
    except Exception:  # pragma: no cover
        # Numba will fall back to another available layer (e.g. workqueue/omp)
        pass

# =============================================================================
# CORE NUMERICAL KERNELS (JIT-COMPILED)
# =============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def calculate_laplacian_4d(phi: np.ndarray, dx: float, out: np.ndarray) -> np.ndarray:
    """Compute 4D periodic Laplacian using 2nd-order finite differences.

    Parameters
    ----------
    phi : (n0,n1,n2,n3) ndarray
        Field values.
    dx : float
        Lattice spacing (assumed uniform).
    out : ndarray
        Output buffer (same shape as `phi`).

    Returns
    -------
    out : ndarray
        Filled with ∇²φ.
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
                        phi[ip1, j, k, l] + phi[im1, j, k, l]
                        + phi[i, jp1, k, l] + phi[i, jm1, k, l]
                        + phi[i, j, kp1, l] + phi[i, j, km1, l]
                        + phi[i, j, k, lp1] + phi[i, j, k, lm1]
                    )
                    out[i, j, k, l] = (neighbors_sum - 8.0 * center) * dx2_inv
    return out


@njit(parallel=True, fastmath=True, cache=True)
def calculate_full_potential_derivative(
    phi: np.ndarray,
    mu_squared: float,
    lambda_val: float,
    g_squared: float,
    P_env: float,
    h_squared: float,
    dx: float,
) -> np.ndarray:
    """Compute δU/δφ for the full model, consistent with the energy density.

    U_iso     = -½ μ² φ² + ¼ λ φ⁴
    U_P       = -P (1-φ²)
    U_aniso   = ½ g² (1-φ²)² (∂_w φ)²
    U_hook    = ½ h² (1-φ²) [ (∂_x φ)² + (∂_y φ)² ]

    Resulting Euler–Lagrange derivative (δU/δφ):

    δU_iso/δφ   = -μ² φ + λ φ³
    δU_P/δφ     = +2 P φ
    δU_aniso/δφ = - g² (1-φ²)² ∂²_w φ + 2 g² φ (1-φ²) (∂_w φ)²
    δU_hook/δφ  = - h² (1-φ²) (∂²_x φ + ∂²_y φ) + h² φ [ (∂_x φ)² + (∂_y φ)² ]
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

                    # neighbor indices
                    ip1, im1 = (i + 1) % n0, (i - 1) % n0
                    jp1, jm1 = (j + 1) % n1, (j - 1) % n1
                    lp1, lm1 = (l + 1) % n3, (l - 1) % n3

                    # (1) Isotropic Mexican hat
                    dUdPhi_iso = -mu_squared * phi_val + lambda_val * phi_val * phi_sq

                    # (2) Pressure
                    dUdPhi_pressure = 2.0 * P_env * phi_val

                    # (3) Anisotropic stabilizer (w-direction)
                    grad_w = (phi[i, j, k, lp1] - phi[i, j, k, lm1]) * inv_2dx
                    d2_phi_dw2 = (phi[i, j, k, lp1] - 2.0 * phi_val + phi[i, j, k, lm1]) * dx2_inv
                    # --- aniso ---
                    aniso_term1 = +2.0 * g_squared * one_minus_phi_sq * phi_val * (grad_w * grad_w)
                    aniso_term2 = -1.0 * g_squared * (one_minus_phi_sq * one_minus_phi_sq) * d2_phi_dw2                    # second term: - g² (1-φ²)² ∂²_w φ
                    dUdPhi_aniso = aniso_term1 + aniso_term2

                    # (4) Hook coupling (x,y)
                    grad_x = (phi[ip1, j, k, l] - phi[im1, j, k, l]) * inv_2dx
                    grad_y = (phi[i, jp1, k, l] - phi[i, jm1, k, l]) * inv_2dx
                    d2_phi_dx2 = (phi[ip1, j, k, l] - 2.0 * phi_val + phi[im1, j, k, l]) * dx2_inv
                    d2_phi_dy2 = (phi[i, jp1, k, l] - 2.0 * phi_val + phi[i, jm1, k, l]) * dx2_inv
                    # --- hook ---
                    # was: hook_term1 = -1.0 * h_squared * phi_val * (grad_x * grad_x + grad_y * grad_y)
                    hook_term1 = +1.0 * h_squared * phi_val * (grad_x * grad_x + grad_y * grad_y)
                    hook_term2 = -1.0 * h_squared * one_minus_phi_sq * (d2_phi_dx2 + d2_phi_dy2)
                    dUdPhi_hook = hook_term1 + hook_term2

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
    """Total energy for the full anisotropic + pressure + Hook model.

    Uses the identity ½|∇φ|² = -½ φ ∇²φ for the kinetic term under periodic BCs.
    """
    dx4 = dx ** 4
    inv_2dx = 0.5 / dx
    n0, n1, n2, n3 = phi.shape

    energy_array = np.zeros(n0, dtype=np.float64)

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
                    phi_sq = phi_val * phi_val
                    one_minus_phi_sq = 1.0 - phi_sq

                    # Kinetic via -½ φ ∇²φ
                    neighbors_sum = (
                        phi[ip1, j, k, l] + phi[im1, j, k, l]
                        + phi[i, jp1, k, l] + phi[i, jm1, k, l]
                        + phi[i, j, kp1, l] + phi[i, j, km1, l]
                        + phi[i, j, k, lp1] + phi[i, j, k, lm1]
                    )
                    laplacian_val = (neighbors_sum - 8.0 * phi_val) / (dx * dx)
                    kinetic_density = -0.5 * phi_val * laplacian_val

                    # Potentials
                    potential_iso_density = -0.5 * mu_squared * phi_sq + 0.25 * lambda_val * (phi_sq * phi_sq)

                    grad_w_sq = ((phi[i, j, k, lp1] - phi[i, j, k, lm1]) * inv_2dx) ** 2
                    potential_aniso_density = 0.5 * g_squared * (one_minus_phi_sq * one_minus_phi_sq) * grad_w_sq

                    potential_pressure_density = -P_env * one_minus_phi_sq

                    grad_x_sq = ((phi[ip1, j, k, l] - phi[im1, j, k, l]) * inv_2dx) ** 2
                    grad_y_sq = ((phi[i, jp1, k, l] - phi[i, jm1, k, l]) * inv_2dx) ** 2
                    potential_hook_density = 0.5 * h_squared * (grad_x_sq + grad_y_sq) * one_minus_phi_sq

                    local_energy += (
                        kinetic_density
                        + potential_iso_density
                        + potential_aniso_density
                        + potential_pressure_density
                        + potential_hook_density
                    )

        energy_array[i] = local_energy

    return np.sum(energy_array) * dx4


# =============================================================================
# SIMULATION CONTROL
# =============================================================================

def seed_hopfion(lattice_shape: Tuple[int, int, int, int], nw: int, dx: float) -> np.ndarray:
    """Analytic seed for a Hopfion-like shape (not topologically protected for a real scalar).

    For nw == 0, returns a uniform field φ ≈ +1.
    """
    if nw == 0:
        return np.ones(lattice_shape, dtype=np.float64)

    logger.info("Seeding lattice with a proper N_w=%s topological seed (shape only for real scalar)...", nw)
    dims = [np.arange(-s / 2.0 + 0.5, s / 2.0 + 0.5, dtype=np.float64) * dx for s in lattice_shape]
    x, y, z, w = np.meshgrid(dims[0], dims[1], dims[2], dims[3], indexing="ij")

    epsilon = 1e-12
    a = x * x + y * y
    b = z * z + w * w
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

    logger.info("Seeding for N_w=%s complete.", nw)
    return phi_final


class HopfionRelaxer:
    """Relaxation solver for a 4D scalar with anisotropic/Hook gradient couplings.

    The update is a damped (heavy-ball) gradient flow with periodic backtracking
    on energy increase.
    """

    def __init__(self, config: dict):
        self.config = dict(config)  # shallow copy to avoid accidental external mutation

        # Lattice and discretization
        self.lattice_shape: Tuple[int, int, int, int] = tuple(self.config["lattice_size"])  # type: ignore
        self.dx: float = float(self.config["dx"])  # lattice spacing

        # Potential parameters
        self.mu2: float = float(self.config["mu_squared"])  # μ²
        self.lam: float = float(self.config["lambda_val"])  # λ
        self.g_sq: float = float(self.config.get("g_squared", 0.0))  # g² (anisotropic)
        # Support both keys: prefer g_H_squared, fall back to h_squared
        self.h_sq: float = float(self.config.get("g_H_squared", self.config.get("h_squared", 0.0)))  # h²
        self.P_env: float = float(self.config.get("P_env", 0.0))  # environmental pressure P

        # Integrator controls
        self.dt: float = float(self.config.get("relaxation_dt", 1e-4))
        self.friction: float = float(self.config.get("friction", 0.95))
        self.min_dt: float = float(self.config.get("min_dt", 1e-9))
        self.max_dt: float = float(self.config.get("max_dt", 1e-3))
        self.dt_down_factor: float = float(self.config.get("dt_down_factor", 0.5))
        self.dt_up_factor: float = float(self.config.get("dt_up_factor", 1.01))
        self.max_phi_change_per_step: float = float(self.config.get("max_phi_change_per_step", 0.01))
        self.max_iterations: int = int(self.config.get("max_iterations", 20000))
        self.convergence_threshold: float = float(self.config.get("convergence_threshold", 1e-8))
        self.early_exit_energy_threshold: float = float(self.config.get("early_exit_energy_threshold", 1e6))

        # Backtracking controls
        self.energy_check_interval: int = int(self.config.get("energy_check_interval", 10))
        self.energy_increase_tolerance: float = float(self.config.get("energy_increase_tolerance", 1e-14))

        # Optional safety clamp
        self.clip_phi_to_unit: bool = bool(self.config.get("clip_phi_to_unit", False))

        # Buffers
        self.phi: np.ndarray = np.zeros(self.lattice_shape, dtype=np.float64)
        self.velocity_buffer: np.ndarray = np.zeros_like(self.phi)
        self.laplacian_buffer: np.ndarray = np.zeros_like(self.phi)

        # Parameter sanity checks
        self._validate_parameters()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _vacuum_phi0_sq(self) -> float:
        """Return φ0² for the homogeneous vacuum minimizing U_iso + U_P."""
        # For μ² ≤ 2P, minimum at φ=0; otherwise φ0² = (μ² - 2P)/λ
        if self.mu2 <= 2.0 * self.P_env:
            return 0.0
        return max(0.0, (self.mu2 - 2.0 * self.P_env) / self.lam)

    def _validate_parameters(self) -> None:
        if self.lam <= 0.0:
            raise ValueError("lambda_val must be positive for a bounded-from-below Mexican hat.")

        if self.h_sq > 0.0:
            phi0_sq = self._vacuum_phi0_sq()
            if phi0_sq > 1.0:
                logger.warning(
                    "Hook term active (h^2>0) but vacuum |phi0|=%.3f > 1. "
                    "This makes U_hook negative (unbounded) for large gradients. "
                    "Consider increasing lambda_val or P_env, or reducing mu_squared.",
                    float(np.sqrt(phi0_sq)),
                )

    def compute_total_energy(self) -> float:
        return calculate_full_total_energy(
            self.phi, self.dx, self.mu2, self.lam, self.g_sq, self.P_env, self.h_sq
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize_field(self, nw: int) -> None:
        self.phi = seed_hopfion(self.lattice_shape, nw, self.dx)
        self.velocity_buffer.fill(0.0)

    def run_relaxation(
        self,
        n_skip: Optional[int] = None,
        n_iter: Optional[int] = None,
        record_series: bool = False,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Run damped gradient relaxation with periodic backtracking.

        Parameters
        ----------
        n_skip, n_iter : Optional[int]
            If provided, total iterations = n_skip + n_iter (kept for compatibility).
        record_series : bool
            If True, record a probe point through time and return it.

        Returns
        -------
        (phi_series, final_energy)
            phi_series is None unless record_series=True.
            final_energy is NaN if the run reaches max iterations without convergence
            (to signal non-convergence to callers).
        """
        # Determine iteration budget
        total_iterations = self.max_iterations
        if n_skip is not None and n_iter is not None:
            total_iterations = int(n_skip) + int(n_iter)

        # Initial energy
        last_energy = self.compute_total_energy()
        if not np.isfinite(last_energy):
            return (None, np.nan) if not record_series else (np.array([]), np.nan)

        converged = False
        phi_series: Optional[List[float]] = [] if record_series else None

        for i in range(total_iterations):
            # Core force = ∇²φ - δU/δφ
            calculate_laplacian_4d(self.phi, self.dx, out=self.laplacian_buffer)
            dU_dphi = calculate_full_potential_derivative(
                self.phi, self.mu2, self.lam, self.g_sq, self.P_env, self.h_sq, self.dx
            )
            force = self.laplacian_buffer - dU_dphi

            # Save previous state for potential backtracking
            prev_velocity = self.velocity_buffer.copy()
            prev_phi = None  # only allocated when checking energy
            prev_dt = self.dt

            # Proposed update
            self.velocity_buffer = (self.friction * self.velocity_buffer) + (self.dt * force)
            update_step = self.dt * self.velocity_buffer

            # Guard on max field increment
            max_change = float(np.max(np.abs(update_step)))
            if max_change > self.max_phi_change_per_step:
                self.dt = max(self.min_dt, self.dt * self.dt_down_factor)
                self.velocity_buffer = (self.friction * prev_velocity) + (self.dt * force)
                update_step = self.dt * self.velocity_buffer

            # Apply update; checkpoint for energy check
            if (i % self.energy_check_interval) == 0:
                prev_phi = self.phi.copy()

            self.phi += update_step

            # Optional clamp (debug guardrail while tuning parameters)
            if self.clip_phi_to_unit:
                np.clip(self.phi, -1.0, 1.0, out=self.phi)

            # Gentle dt growth when stable
            if (i % 50) == 0 and i > 0:
                self.dt = min(self.max_dt, self.dt * self.dt_up_factor)

            # Energy check + backtracking
            if (i % self.energy_check_interval) == 0:
                current_energy = self.compute_total_energy()

                if record_series and phi_series is not None:
                    phi_series.append(
                        float(
                            self.phi[
                                self.lattice_shape[0] // 4,
                                self.lattice_shape[1] // 4,
                                self.lattice_shape[2] // 4,
                                self.lattice_shape[3] // 4,
                            ]
                        )
                    )

                if (not np.isfinite(current_energy)) or (current_energy > self.early_exit_energy_threshold):
                    return (np.array(phi_series), np.nan) if record_series else (None, np.nan)

                # Backtrack if energy increased beyond tolerance
                if current_energy > last_energy + self.energy_increase_tolerance:
                    # Reject, shrink dt, try once with reduced dt
                    if prev_phi is not None:
                        self.phi[:] = prev_phi
                    self.dt = max(self.min_dt, prev_dt * self.dt_down_factor)
                    self.velocity_buffer[:] = prev_velocity

                    # Re-apply with smaller dt
                    self.velocity_buffer = (self.friction * self.velocity_buffer) + (self.dt * force)
                    self.phi += self.dt * self.velocity_buffer

                    if self.clip_phi_to_unit:
                        np.clip(self.phi, -1.0, 1.0, out=self.phi)

                    current_energy = self.compute_total_energy()

                rel_drop = abs(current_energy - last_energy) / max(abs(current_energy), 1.0)
                max_update = float(np.max(np.abs(update_step)))

                rel_tol = float(self.config.get("energy_rel_tol", 1e-5))
                step_tol = float(self.config.get("max_update_tol", 5e-5))
                warmup = int(self.config.get("convergence_warmup_iters", 500))

                # Debug log every 50th energy check
                checks_done = (i // self.energy_check_interval)
                if checks_done % 50 == 0:
                    logger.debug(
                        "it=%d E=%.6f dt=%.3e maxΔφ=%.3e",
                        i, float(current_energy), float(self.dt), float(max_update)
                    )

                if i >= warmup and (rel_drop < rel_tol or max_update < step_tol):
                    converged = True
                    last_energy = current_energy
                    break

                last_energy = current_energy

        final_energy = self.compute_total_energy()
        if not converged and i >= total_iterations - 1:
            # Signal non-convergence upstream
            final_energy = np.nan

        return (np.array(phi_series), final_energy) if record_series else (None, final_energy)

