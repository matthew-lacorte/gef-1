import numpy as np

G  = 6.67408e-11
M  = 1.98847e30          # Sun
R  = 6.9634e8            # solar radius
c  = 2.99792458e8

# PPN deflection formula using γ=1 from earlier snippet
alpha = 4*G*M/(c**2*R)   # radians
print("α =", alpha, "rad  ≈", alpha*206265, "arcsec")
