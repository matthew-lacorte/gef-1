import sympy as sp

# Symbols
G, M, r, c = sp.symbols('G M r c', positive=True)
beta = sp.symbols('beta', positive=True)      # coupling; set beta=1 later

U = G*M/r
kappa = 1 - beta*U/c**2

delta = sp.symbols('delta')      # expect delta ≈ beta
g00 = -(1 - beta*U/c**2)**2 - delta*(U/c**2)**2
gij = (1 + beta*U/c**2)**2     # spatial prefactor

# Expand
g00s = sp.series(g00, U, 0, 3).removeO()
gijs = sp.series(gij, U, 0, 2).removeO()

# Extract PPN parameters properly
gamma_ppn = (gijs.expand().coeff(U/c**2, 1)) / 2          # divide by 2
beta_ppn  = -(g00s.expand().coeff((U/c**2)**2, 1)) / 2    # divide by –2

print("γ_PPN =", sp.simplify(gamma_ppn.subs(beta, 1)))
print("β_PPN =", sp.simplify(beta_ppn.subs(beta, 1)))
