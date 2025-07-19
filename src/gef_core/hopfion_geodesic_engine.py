# src/gef/physics_core/hopfion_geodesic_engine.py

import numpy as np
from numba import njit
from scipy.spatial import KDTree  # We need a fast way to find nearest neighbors
from ..geometry.manifolds import HopfionGeometry # Import your existing class

@njit(cache=True)
def project_to_tangent_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Projects a vector onto the tangent plane defined by the normal vector,
    removing the component of the vector that is perpendicular to the surface.
    
    Args:
        vector (np.ndarray): The vector to project.
        normal (np.ndarray): The normal vector defining the tangent plane.
        
    Returns:
        np.ndarray: The projected vector, tangent to the surface.
    """
    return vector - np.dot(vector, normal) * normal

# This helper function will live inside the class or can be a standalone numba func
@njit(cache=True)
def perform_pca_on_neighbors(neighbors: np.ndarray) -> np.ndarray:
    """
    Performs Principal Component Analysis (PCA) on a set of points.
    The last singular vector corresponds to the direction of least variance,
    which is our normal vector to the local surface patch.

    Args:
        neighbors (np.ndarray): An array of shape (k, 4) of neighboring points.

    Returns:
        np.ndarray: The 4D normal vector.
    """
    # 1. Center the data
    center = np.mean(neighbors, axis=0)
    centered_data = neighbors - center

    # 2. Compute the covariance matrix (or use SVD directly for stability)
    # Using SVD is numerically more stable than forming the covariance matrix.
    # U, S, Vt = np.linalg.svd(centered_data)
    # The last row of Vt (which is V transposed) is the principal component
    # with the smallest eigenvalue, i.e., the normal vector.
    # Numba's linalg support is limited, so we may need to implement SVD or
    # fallback to numpy mode for this part if it's not supported.
    # For now, let's assume a simplified covariance method works for numba.
    cov_matrix = np.cov(centered_data, rowvar=False)
    
    # 3. Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. The normal vector is the eigenvector associated with the smallest eigenvalue
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
    return normal_vector

class HopfionGeodesicEngine:
    """
    Simulates geodesic paths on the surface of a 4D Hopfion topology,
    which is defined by a discrete point cloud.
    """
    def __init__(self, hopfion_geometry: HopfionGeometry, k_neighbors: int):
        """
        Initializes the engine by generating the Hopfion manifold.

        Args:
            hopfion_geometry (HopfionGeometry): The pre-configured geometry generator.
            k_neighbors (int): The number of nearest neighbors to use for
                                 local geometry calculations (e.g., 10-20).
        """
        print("Engine Initializing: Generating Hopfion manifold point cloud...")
        # 1. Generate the static point cloud that defines our "surface"
        self.manifold_points, _ = hopfion_geometry.generate_manifold_point_cloud()
        self.R = hopfion_geometry.R # Get radius from the geometry object
        self.k = k_neighbors
        
        print("Building KD-Tree for fast nearest-neighbor lookups...")
        # 2. Build a KD-Tree for extremely fast nearest-neighbor searches.
        # This is a critical optimization.
        self.kdtree = KDTree(self.manifold_points)
        print("Engine ready.")

    def _get_local_geometry(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the local surface normal and closest manifold point
        for a given position in 4D space.
        
        This is the core of the new engine.

        Returns:
            (normal_vector, closest_point_on_manifold)
        """
        # 1. Query the KD-Tree to find the k nearest neighbors on the manifold
        distances, indices = self.kdtree.query(position, k=self.k)
        
        if indices is None: # Should not happen if k is valid
             return np.array([1.0, 0, 0, 0]), position

        neighbor_points = self.manifold_points[indices]

        # 2. Perform PCA to find the normal vector of the local patch
        # This requires a function that can run outside of numba-jitted class methods
        # Note: PCA via SVD is better, but this is a start.
        # We need a non-class, numba-jitted function for this.
        # Let's mock this for now and assume we have it.
        # A simple but effective normal is the vector from the patch center to the query point.
        patch_center = np.mean(neighbor_points, axis=0)
        normal_vector = position - patch_center
        norm = np.linalg.norm(normal_vector)
        if norm > 0:
            normal_vector /= norm
        else: # Particle is exactly at the center of the patch
            # This is tricky. We need a more robust normal calculation.
            # Let's use PCA as originally planned.
            # We'll assume a helper function `_pca_normal` exists.
            normal_vector = self._pca_normal(neighbor_points)

        # 3. The closest point on the manifold can be approximated as the center of the patch
        closest_point = patch_center

        return normal_vector, closest_point

    def _pca_normal(self, neighbors: np.ndarray) -> np.ndarray:
        """Helper to run PCA. SVD is better but this is simpler."""
        center = np.mean(neighbors, axis=0)
        centered = neighbors - center
        u, s, vt = np.linalg.svd(centered)
        # The last row of vt is the vector corresponding to the smallest singular value
        normal = vt[-1]
        return normal


    def find_trajectory(self, initial_pos_guess: np.ndarray, initial_impulse: float, max_steps: int, dt: float) -> dict:
        """
        Simulates the path of a particle on the Hopfion surface using Leapfrog.
        """
        # --- 1. Set Initial Conditions ---
        # Find the actual closest point on the manifold to our initial guess
        _, pos = self._get_local_geometry(initial_pos_guess)
        
        # Initial velocity is a kick in a tangent direction
        vel = np.array([0.0, 0.0, 0.0, initial_impulse], dtype=np.float64)
        
        # Ensure initial velocity is perfectly tangent to the surface
        initial_normal, _ = self._get_local_geometry(pos)
        vel = project_to_tangent_plane(vel, initial_normal)

        path = np.zeros((max_steps, 4))
        path[0] = pos
        initial_pos_on_manifold = pos.copy()
        
        # Store velocities for curvature calculation
        velocities = np.zeros((max_steps, 4))
        velocities[0] = vel.copy()
        
        # --- 2. Main Evolution Loop (Leapfrog Integrator) ---
        # Calculate initial force
        current_force = -initial_normal * (np.sum(vel**2) / self.R) # Simplified centripetal force

        for i in range(1, max_steps):
            # 1. Update position (naively)
            pos_intermediate = pos + vel * dt + 0.5 * current_force * dt**2

            # 2. Find the new local geometry based on the naive new position
            new_normal, closest_manifold_point = self._get_local_geometry(pos_intermediate)

            # 3. Snap the position to the manifold surface
            pos = closest_manifold_point
            
            # 4. Calculate the new force at the corrected position
            # Approximate the velocity that would have led to this point
            vel_midpoint_approx = vel + 0.5 * current_force * dt
            new_force = -new_normal * (np.sum(vel_midpoint_approx**2) / self.R)

            # 5. Update velocity with the average force
            vel += 0.5 * (current_force + new_force) * dt

            # 6. Project velocity onto the new tangent plane to remove drift
            vel = project_to_tangent_plane(vel, new_normal)

            # Store position and velocity for analysis
            path[i] = pos.copy()
            velocities[i] = vel.copy()
            
            current_force = new_force

        # --- 3. Analyze the Path for Stability ---
        final_pos = path[-1]
        final_distance_from_origin = np.linalg.norm(final_pos - initial_pos_on_manifold)
        
        is_stable = final_distance_from_origin < 1e-2 # Tune this threshold

        # Calculate path metrics
        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        
        # Calculate velocity changes for curvature
        vel_diffs = np.diff(velocities, axis=0)
        vel_diff_magnitudes = np.sqrt(np.sum(vel_diffs**2, axis=1))
        avg_curvature = np.mean(vel_diff_magnitudes) / dt if len(vel_diff_magnitudes) > 0 else 0.0
        
        return {
            'is_stable': is_stable,
            'path_length': path_length,
            'avg_curvature': avg_curvature,
            'final_distance_from_origin': final_distance_from_origin,
            'energy_proxy': path_length
        }