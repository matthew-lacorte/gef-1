# src/gef/scripts/core/bifurcation_scan.py
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import concurrent.futures
import yaml
from pathlib import Path
import time

# Make sure the gef package is in the Python path
# (This is a common way to handle local package imports in scripts)
script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent.parent))

from gef.core.hopfion_relaxer import HopfionRelaxer
from gef.core.ppn import fundamental_period

# --- Configuration ---
# It's better to load this from a YAML for reproducibility
CONFIG_PATH = script_dir / "configs" / "default_bifurcation_scan.yml"
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

# --- Parallelizable Relaxation Function ---
def run_relax(g_sq: float):
    """
    Runs relaxation for a single g_squared value and returns the result.
    """ 
    try:
        local_cfg = cfg['solver_config']
        local_cfg['g_squared'] = g_sq
        sim = HopfionRelaxer(local_cfg)
        sim.initialize_field(nw=1)

        phi_series, final_energy = sim.run_relaxation(
            n_skip=cfg['scan_config']['n_skip'],
            n_iter=cfg['scan_config']['n_iter']
        )
        
        period = fundamental_period(
            phi_series, 
            max_period=cfg['scan_config']['period_cap'],
            tol=cfg['scan_config']['period_tolerance']
        )
        
        return g_sq, period, final_energy
    except Exception as e:
        print(f"ERROR: Simulation for g_sq={g_sq:.4f} failed: {e}")
        return g_sq, -1, np.nan # Use -1 to indicate an error

# --- Main Execution Block ---
if __name__ == "__main__":
    scan_cfg = cfg['scan_config']
    g_values = np.linspace(scan_cfg['g_min'], scan_cfg['g_max'], scan_cfg['n_steps'])
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(cfg['output_dir']) / f"bifurcation_scan_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    results = []
    print(f"Starting bifurcation scan for g_squared from {scan_cfg['g_min']} to {scan_cfg['g_max']}...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_g = {executor.submit(run_relax, g): g for g in g_values}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_g), total=len(g_values)):
            results.append(future.result())

    # --- Analyze Results ---
    results_df = pd.DataFrame(results, columns=['g_squared', 'period', 'final_energy']).sort_values('g_squared')
    
    # Save raw results
    results_df.to_csv(output_dir / "raw_bifurcation_data.csv", index=False)
    print(f"\nRaw data saved to raw_bifurcation_data.csv")

    bifurcation_points = []
    last_period = 1
    for _, row in results_df.iterrows():
        g, p = row['g_squared'], row['period']
        if p is not None and p > last_period:
            print(f"Period doubling detected: {last_period} -> {int(p)} at g_squared ≈ {g:.6f}")
            if p == 2 * last_period:
                bifurcation_points.append(g)
            last_period = int(p)

    if len(bifurcation_points) > 2:
        deltas = [
            (bifurcation_points[i] - bifurcation_points[i-1]) / 
            (bifurcation_points[i+1] - bifurcation_points[i])
            for i in range(1, len(bifurcation_points) - 1)
        ]

        print("\nBifurcation points (g_n where period doubles):")
        for n, r in enumerate(bifurcation_points, start=1):
            print(f" n={n:>2}  g_n={r:.6f}")

        print("\nFeigenbaum-like δ estimates:")
        for n, d in enumerate(deltas, start=2):
            print(f" δ_{n} = {d:.6f}")
    else:
        print("\nFewer than 3 bifurcation points found. Cannot calculate δ.")