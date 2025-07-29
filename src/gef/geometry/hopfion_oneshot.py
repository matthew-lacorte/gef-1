# gef-framework/gef/geometry/hopfion.py
import numpy as np

class HopfionGeometry:
    """
    Generates a 4D point cloud representing an N_w=1 Hopfion soliton.
    
    The parameterization maps a 2-torus (angles u, v) to the 3-sphere (S^3),
    which is then stereographically projected to R^3 and a 'w' dimension.
    This gives us a (x, y, z, w) point cloud in 4D Euclidean space.
    """
    def __init__(self, num_points=50000):
        if int(np.sqrt(num_points))**2 != num_points:
            raise ValueError("num_points must be a perfect square for uniform sampling.")
        self.num_points = num_points
        self.points = self._generate_points()

    def _generate_points(self):
        """Generates the 4D point cloud."""
        n_sqrt = int(np.sqrt(self.num_points))
        u = np.linspace(0, 2 * np.pi, n_sqrt)
        v = np.linspace(0, 2 * np.pi, n_sqrt)
        u, v = np.meshgrid(u, v)

        # Hopf Fibration Coordinates (on the 3-sphere)
        x1 = np.cos(v) * np.sin(u)
        x2 = np.sin(v) * np.sin(u)
        x3 = np.cos(v) * np.cos(u)
        x4 = np.sin(v) * np.cos(u)

        # For our simulation, we can directly use these 4D coordinates.
        # Let's map them to (x, y, z, w)
        # This is a simplification; a true GEF model might have a more
        # complex mapping from the 3-sphere to R^4.
        points_4d = np.stack([x1, x2, x3, x4], axis=-1).reshape(-1, 4)
        
        return points_4d

    def get_points(self):
        """Returns a copy of the point cloud."""
        return self.points.copy()