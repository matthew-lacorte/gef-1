# src/gef_core/bifurcation_scan.py
import numpy as np
from tqdm import tqdm
import sys
import concurrent.futures

sys.path.append('/Users/mattlacorte/projects/gef/src')

from gef.core.hopfion_relaxer import HopfionRelaxer
from gef.core.ppn import fundamental_period

# ----------------------------------------------------------------------
# 1.  Configuration
# ----------------------------------------------------------------------
g_min, g_max = 0.1, 1.5
n_steps      = 200
period_cap   = 256

cfg = {
    "lattice_size": [32, 32, 32, 32],
    "dx": 0.5,
    "mu_squared": 1.0,
    "lambda_val": 1.0,
    "g_squared": g_min,
    "P_env": 0.0,
    "h_squared": 0.0,
    "relaxation_dt": 1e-4,
    "max_iterations": 10000,  # Increased iterations for better convergence
    "convergence_threshold": 1e-7,
    "early_exit_energy_threshold": 1.0,
}

# ----------------------------------------------------------------------
# 2.  Parallelizable Relaxation Function
# ----------------------------------------------------------------------
def run_relax(g_sq):
    """Run relaxation for a given g_squared and return the energy time series."""
    local_cfg = cfg.copy()
    local_cfg['g_squared'] = g_sq
    sim = HopfionRelaxer(local_cfg)
    sim.initialize_field(nw=1)
    
    # Run relaxation and get the actual energy time series
    _, _, converged, energy_series = sim.run_relaxation(return_energy_series=True)
    
    if not converged or not energy_series:
        return g_sq, None  # Return None for failed/unstable runs
        
    return g_sq, np.array(energy_series)

# ----------------------------------------------------------------------
# 3.  Main Execution Block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    g_values = np.linspace(g_min, g_max, n_steps)
    results = []
    last_period = 1

    print(f"Starting bifurcation scan with {len(g_values)} steps...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use executor.map to run simulations in parallel
        future_to_g = {executor.submit(run_relax, g): g for g in g_values}
        
        # Wrap futures with tqdm for a progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_g), total=len(g_values)):
            g, energy_series = future.result()
            
            if energy_series is None:
                period = None
            else:
                period = fundamental_period(energy_series, max_period=period_cap)
            
            results.append((g, period))

    # Sort results by g_squared value to ensure correct order for analysis
    results.sort(key=lambda x: x[0])

    # ----------------------------------------------------------------------
    # 4.  Analyze Results and Print Bifurcation Points
    # ----------------------------------------------------------------------
    bifurcation_points = []
    last_period = 1
    for g, p in results:
        if p is not None and p > last_period:
            print(f"Period doubling detected: {last_period} -> {p} at g ≈ {g:.6f}")
            if p == 2 * last_period:
                bifurcation_points.append(g)
            last_period = p

    if bifurcation_points:
        deltas = [
            (bifurcation_points[i] - bifurcation_points[i-1]) / 
            (bifurcation_points[i+1] - bifurcation_points[i])
            for i in range(1, len(bifurcation_points) - 1)
        ]

        print("\nBifurcation points (g_n where period doubles):")
        for n, r in enumerate(bifurcation_points, start=1):
            print(f" n={n:>2}  g_n={r:.6f}")

        if deltas:
            print("\nFeigenbaum-like δ estimates:")
            for n, d in enumerate(deltas, start=2):
                print(f" δ_{n} = {d:.6f}")
    else:
        print("\nNo period-doubling bifurcations were detected.")
