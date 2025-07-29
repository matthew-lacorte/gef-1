# In a temporary test script, e.g., test_hopfion.py
import matplotlib.pyplot as plt
from gef_geometry.hopfion_oneshot import HopfionGeometry

# Generate the particle
# Note: For testing, use a smaller number of points.
particle = HopfionGeometry(num_points=10000)
points = particle.get_points()

# Visualize a 3D projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.3)
ax.set_title("3D Projection of 4D Hopfion")
plt.show()