import sympy as sp

# symbols & constants
G, M, r1, r2, c = sp.symbols('G M r1 r2 c', positive=True)

# Newtonian potential
U1 = G*M/r1
U2 = G*M/r2

# kappa-induced clock factor
z = (1 - U2/c**2)/(1 - U1/c**2) - 1   # fractional red-shift

# Quick numeric example: Earth surface to 20 000 km GPS orbit
num = {
    G: 6.67408e-11,
    M: 5.972e24,
    r1: 6.371e6,
    r2: 2.6371e7,
    c: 2.99792458e8
}
print("Δf/f =", sp.N(z.subs(num), 12))