# src/gef/physics_core/s3_geodesic_engine.py

import numpy as np
from numba import njit

# --- The Geometry of the 4-Sphere (S³) ---
@njit(cache=True)
def get_s3_surface_gradient(position: np.ndarray) -> np.ndarray:
    """
    For a point in 4D, returns the normalized outward-pointing vector,
    which is normal to the surface of a 4-sphere centered at the origin.
    """
    norm = np.sqrt(np.sum(position**2))
    if norm == 0:
        return np.zeros_like(position)
    return position / norm

@njit(cache=True)
def project_to_tangent_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Projects a vector onto the tangent plane defined by the normal vector,
    removing the component of the vector that is perpendicular to the surface.
    """
    return vector - np.dot(vector, normal) * normal

# --- REFACTORED: Class renamed for accuracy ---
class S3GeodesicEngine:
    def __init__(self, sphere_params: dict):
        """
        Initializes the engine with the geometric properties of the 4-sphere.
        """
        self.R = sphere_params['radius'] # The radius of the 4-sphere.

    def find_trajectory(self, initial_radius_on_surface: float, initial_impulse: float, max_steps: int, dt: float) -> dict:
        """
        Simulates the path of a particle on the S³ surface using a
        Leapfrog (Verlet) integrator for improved energy conservation.
        """
        # --- 1. Set Initial Conditions ---
        # Start at a point on the surface. We can scale the initial position
        # to ensure it lies exactly on the sphere of radius R.
        pos = np.array([initial_radius_on_surface, 0.0, 0.0, 0.0], dtype=np.float64)
        pos = pos / np.sqrt(np.sum(pos**2)) * self.R if np.sum(pos**2) > 0 else pos
        
        # Initial velocity is a "kick" in a tangent direction (e.g., along w-axis).
        vel = np.array([0.0, 0.0, 0.0, initial_impulse], dtype=np.float64)
        
        path = np.zeros((max_steps, 4))
        path[0] = pos
        initial_pos = pos.copy()
        
        # Store velocities for curvature calculation
        velocities = np.zeros((max_steps, 4))
        velocities[0] = vel.copy()
        
        # --- 2. The Main Evolution Loop (Leapfrog Integrator) ---
        
        # Calculate initial force at the starting position
        # For a geodesic on a sphere, the only "force" is the centripetal
        # constraint force that curves the path.
        current_force = -get_s3_surface_gradient(pos) * (np.sum(vel**2) / self.R)
        
        for i in range(1, max_steps):
            # --- Leapfrog (Verlet) Integration Scheme ---
            # 1. Update position by a full timestep using current velocity.
            pos += vel * dt + 0.5 * current_force * dt**2
            
            # 2. Calculate the force at the *new* position.
            # The velocity term in the centripetal force should be the one
            # used to generate the new position, so we estimate it.
            vel_midpoint_approx = vel + 0.5 * current_force * dt
            new_force = -get_s3_surface_gradient(pos) * (np.sum(vel_midpoint_approx**2) / self.R)
            
            # 3. Update velocity by a full timestep using the *average* of the old and new forces.
            vel += 0.5 * (current_force + new_force) * dt
            
            # --- Brute-force constraint enforcement (to correct long-term drift) ---
            # This should be needed less often with a better integrator.
            if i % 100 == 0: # Only correct every 100 steps
                # Project velocity back onto the tangent plane
                vel = project_to_tangent_plane(vel, get_s3_surface_gradient(pos))
                # Re-normalize position to strictly enforce the constraint |pos| = R
                pos = pos / np.sqrt(np.sum(pos**2)) * self.R

            # Store position and velocity for analysis
            path[i] = pos.copy()
            velocities[i] = vel.copy()
            
            current_force = new_force

        # --- 3. Analyze the Path for Stability ---
        final_pos = path[-1]
        final_distance_from_origin = np.sqrt(np.sum((final_pos - initial_pos)**2))
        
        # A stable orbit should be periodic, ending near its starting point.
        # A tighter threshold can be used now due to better energy conservation.
        is_stable = final_distance_from_origin < 1e-3

        # Calculate path metrics
        path_length = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
        
        # Calculate velocity changes for curvature
        vel_diffs = np.diff(velocities, axis=0)
        vel_diff_magnitudes = np.sqrt(np.sum(vel_diffs**2, axis=1))
        avg_curvature = np.mean(vel_diff_magnitudes) / dt if len(vel_diff_magnitudes) > 0 else 0.0
        
        # The energy proxy can still be path length or average curvature.
        energy_proxy = path_length
        
        return {
            'is_stable': is_stable,
            'path_length': path_length,
            'avg_curvature': avg_curvature,
            'final_distance_from_origin': final_distance_from_origin,
            'energy_proxy': energy_proxy
        }