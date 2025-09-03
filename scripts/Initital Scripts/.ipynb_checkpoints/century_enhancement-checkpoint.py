from math import pi

G  = 6.67408e-11
M  = 1.98847e30           # Sun mass
a  = 5.791e10             # Merc. semi-major axis  (m)
e  = 0.2056               # Merc. eccentricity
c  = 2.99792458e8

# GR / PPN advance per orbit (rad)
dphi = 6*pi*G*M/(c**2 * a*(1 - e**2))
# convert to arcsec per century (415 orbits / century)
advance_arcsec = dphi * 206265 * 415
print("Δϖ =", advance_arcsec, "arcsec/century")