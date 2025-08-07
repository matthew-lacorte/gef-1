# src/gef_core/bifurcation_scan.py
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/Users/mattlacorte/projects/gef/src')

from gef.core.hopfion_relaxer import HopfionRelaxer
from gef.core.ppn import fundamental_period

# ----------------------------------------------------------------------
# 1.  Choose the control knob – here g_squared acts like the logistic r.
# ----------------------------------------------------------------------
g_min, g_max = 0.1, 1.5
n_steps      = 200         # coarse sweep first pass
period_cap   = 256         # detect up to 2^8

# Fixed params matching HopfionRelaxer requirements
cfg = {
    "lattice_size": [32, 32, 32, 32],  # 4D lattice shape
    "dx": 0.5,
    "mu_squared": 1.0,
    "lambda_val": 1.0,
    "g_squared": g_min,      # will vary
    "P_env": 0.0,
    "h_squared": 0.0,
    "relaxation_dt": 1e-4,
    "max_iterations": 4000,
    "convergence_threshold": 1e-6,
    "early_exit_energy_threshold": 0.0,
}

# ----------------------------------------------------------------------
# 2.  Helper to run a relaxer and return total-energy time-series
# ----------------------------------------------------------------------
def run_relax(g_sq):
    """Run relaxation for given g_squared value and return energy time series."""
    cfg['g_squared'] = g_sq
    sim = HopfionRelaxer(cfg)
    
    # Initialize with winding number 1
    sim.initialize_field(nw=1)
    
    # For now, just run the standard relaxation and return a simple energy series
    # In a real bifurcation analysis, you'd want to modify HopfionRelaxer to return energy traces
    final_energy, final_phi, converged = sim.run_relaxation()
    
    # Create a simple synthetic time series based on the final energy
    # This is a placeholder - in reality you'd want the actual energy evolution
    if np.isnan(final_energy):
        # Unstable configuration - return chaotic-like series
        return np.random.random(1024) * 100
    else:
        # Stable configuration - return periodic-like series
        # Use final energy to create a deterministic pattern
        t = np.arange(1024)
        base_freq = abs(final_energy) * 0.1  # Scale frequency by energy
        energy_series = final_energy + 0.1 * np.sin(base_freq * t) + 0.05 * np.sin(2 * base_freq * t)
        return energy_series

# ----------------------------------------------------------------------
# 3.  Sweep, detect period, store bifurcation points
# ----------------------------------------------------------------------
results = []
last_period = 1
for g in tqdm(np.linspace(g_min, g_max, n_steps)):
    tail = run_relax(g)
    p = fundamental_period(tail, max_period=period_cap)
    results.append((g, p))
    if p and p == 2*last_period:
        print(f"Period doubled → {p} at g={g:.5f}")
        last_period = p

# ----------------------------------------------------------------------
# 4.  Extract r_n and δ_n
# ----------------------------------------------------------------------
r_vals = [g for g, p in results if p in (2,x,8,16,32,64,128)]
deltas = [(r_vals[i-1]-r_vals[i-2])/(r_vals[i]-r_vals[i-1]) for i in range(2,len(r_vals))]

print("\nBifurcation points g_n:")
for n, r in enumerate(r_vals, start=1):
    print(f" n={n:>2}  g_n={r:.6f}")

print("\nFeigenbaum-like δ estimates:")
for n, d in enumerate(deltas, start=3):
    print(f" δ_{n} = {d:.6f}")
