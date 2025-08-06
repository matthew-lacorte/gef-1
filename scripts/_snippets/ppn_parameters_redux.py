import sympy as sp

# Symbols
G, M, r, c = sp.symbols('G M r c', positive=True)
beta = sp.symbols('beta', positive=True)   # GEF coupling (we set beta=1 later)

U = G*M/r                     # Newtonian potential
kappa_ratio = 1 - beta*U/c**2 # κ(r)/κ_inf

g00 = -(kappa_ratio)**2
gij_prefactor = (1 + beta*U/c**2)**2  # spatial metric factor

# Series to O(U^2)
g00_series = sp.series(g00.expand(), U, 0, 3).removeO()
gij_series = sp.series(gij_prefactor.expand(), U, 0, 2).removeO()

print("g00 =", g00_series)
print("g_ij prefactor =", gij_series, "* δ_ij")

# Match to PPN form
γ_ppn = gij_series.expand().coeff(U/c**2, 1)
β_ppn = -g00_series.expand().coeff((U/c**2)**2, 1) / 2

print("\nPPN parameters:")
print("gamma =", sp.simplify(γ_ppn.subs(beta, 1)))
print("beta  =", sp.simplify(β_ppn.subs(beta, 1)))