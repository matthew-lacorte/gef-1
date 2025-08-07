import numpy as np

def logistic_iter(r, x0=0.5, n_skip=1000, n_iter=4096):
    """Iterate the logistic map x_{k+1}=r x (1-x).
       Returns the final n_iter states after skipping transients."""
    x = x0
    for _ in range(n_skip):
        x = r * x * (1 - x)
    out = np.empty(n_iter)
    for i in range(n_iter):
        x = r * x * (1 - x)
        out[i] = x
    return out

def fundamental_period(xs, max_period=1024, tol=1e-10):
    """Detect the smallest p (<=max_period) such that x[k]==x[k-p] for all k in tail."""
    n = len(xs)
    for p in range(1, max_period + 1):
        if np.allclose(xs[n-p:], xs[n-2*p:n-p], atol=tol, rtol=0):
            return p
    return None  # appears aperiodic/chaotic within given max_period

def find_bifurcation(prev_r, target_period, r_hi=4.0, tol=1e-10, max_iter=60):
    """Bisection search for the parameter r where the orbit first attains
       the given target_period (which is 2×previous period)."""
    lo = prev_r
    hi = r_hi
    period_lo = fundamental_period(logistic_iter(lo))
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        period_mid = fundamental_period(logistic_iter(mid))
        if period_mid is None or period_mid > target_period:
            hi = mid  # chaos or higher period -> move left
        elif period_mid < target_period:
            lo = mid  # still lower period
        else:  # period_mid == target_period
            hi = mid
        if hi - lo < tol:
            return 0.5 * (lo + hi)
    return 0.5 * (lo + hi)

# Find bifurcation points r_1 (3.0), r_2, r_3...
r_values = [3.0]  # r_1 (period 2)
period = 2
for n in range(1, 8):  # up to period 2^8
    period *= 2
    r_prev = r_values[-1] + 1e-6
    r_n = find_bifurcation(r_prev, period, 4.0, tol=1e-12)
    r_values.append(r_n)

# compute deltas
deltas = []
for i in range(2, len(r_values)):
    deltas.append((r_values[i-1] - r_values[i-2]) / (r_values[i] - r_values[i-1]))

# Print results
print("n (period 2^n)    r_n")
for i, r in enumerate(r_values, start=1):
    print(f"{i:>2} ({2**i:>4})     {r:.12f}")
print("\nFeigenbaum delta estimates:")
for i, d in enumerate(deltas, start=3):
    print(f"δ_{i} = {d:.12f}")
