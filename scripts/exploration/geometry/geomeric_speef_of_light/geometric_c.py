import sympy as sp

print("=== Verification of GEF Calculation (Corrected Version) ===\n")

# -----------------------------------------------------------------
# Physical constants (2019 SI "exact" values where available)
# -----------------------------------------------------------------
c  = 299_792_458            # speed of light, m s−1
h  = 6.626_070_15e-34       # Planck constant, J s
eV = 1.602_176_634e-19      # 1 eV in joules, J

print("Physical Constants:")
print(f"c  = {c:,} m/s")
print(f"h  = {h:.10e} J⋅s")
print(f"eV = {eV:.10e} J")

# -----------------------------------------------------------------
# 1) Planck-particle rest energy & mass  (200 MeV)
# -----------------------------------------------------------------
E_P = 200 * 1e6 * eV        # J
m_P = E_P / c**2            # kg

print(f"\n1) Planck Particle Properties:")
print(f"E_P = 200 MeV = {E_P:.10e} J")
print(f"m_P = E_P/c² = {m_P:.10e} kg")

# -----------------------------------------------------------------
# 2) Fundamental internal rotation frequency & tick
# -----------------------------------------------------------------
f_P = E_P / h               # Hz  (== m_P c2 / h)
t_P = 1 / f_P               # s

print(f"\n2) Fundamental Frequency and Time:")
print(f"f_P = E_P/h = {f_P:.10e} Hz")
print(f"t_P = 1/f_P = {t_P:.10e} s")

# -----------------------------------------------------------------
# 3) Compactification radius that makes c a "one-tick" length
# -----------------------------------------------------------------
r_P = c * t_P / (2*sp.pi)   # m

print(f"\n3) Geometric Radius:")
print(f"r_P = c⋅t_P/(2π) = {float(r_P):.10e} m")
print(f"r_P = {float(r_P*1e15):.5f} fm")

# Pretty-print summary (convert to floats for easy reading)
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
for name, val in {
    "Rest-energy  E_P":  E_P,
    "Mass         m_P":  m_P,
    "Frequency    f_P":  f_P,
    "Time unit    t_P":  t_P,
    "Radius       r_P":  r_P,
}.items():
    print(f"{name:18s}= {float(val):.5e}")

print(f"\nRadius in femtometres  = {float(r_P*1e15):.5e} fm")

# -----------------------------------------------------------------
# 4) Verification: Check that c = c_geom / t_P
# -----------------------------------------------------------------
print(f"\n{'='*50}")
print("VERIFICATION")
print(f"{'='*50}")

c_geom = 2 * sp.pi * r_P  # geometric circumference
c_calculated = c_geom / t_P

print(f"Geometric circumference: c_geom = 2π⋅r_P = {float(c_geom):.10e} m")
print(f"Calculated speed: c_calc = c_geom/t_P = {float(c_calculated):.10e} m/s")
print(f"Known speed of light: c = {c:.10e} m/s")
print(f"Difference: {abs(float(c_calculated) - c):.2e} m/s")
print(f"Relative error: {abs(float(c_calculated) - c)/c * 100:.2e}%")

# -----------------------------------------------------------------
# 5) Additional context and comparisons
# -----------------------------------------------------------------
print(f"\n{'='*50}")
print("PHYSICAL CONTEXT")
print(f"{'='*50}")

print(f"Planck Particle radius: {float(r_P*1e15):.3f} fm")
print(f"Proton charge radius:    ~0.88 fm")
print(f"Atomic nucleus scale:    ~1-10 fm")
print(f"Classical Planck length: ~1.6×10^-20 fm")

ratio_to_planck = float(r_P) / 1.616e-35  # Approximate Planck length
print(f"\nRadius ratio r_P/ℓ_Planck ≈ {ratio_to_planck:.2e}")

print(f"\nFundamental frequency: {f_P:.2e} Hz")
print(f"This corresponds to a period of: {t_P:.2e} s")
print(f"For comparison, light travels: {c * t_P:.2e} m in one tick")
print(f"Which equals the circumference: {float(2*sp.pi*r_P):.2e} m")