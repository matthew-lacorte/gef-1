# src/gef_core/bifurcation_scan.py

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import concurrent.futures
import yaml
from pathlib import Path
import time
import logging

# --- Setup Project Path ---
# This makes imports robust, assuming the script is in src/gef_core/
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

# --- Local Imports ---
# Use try-except for robustness if run outside the package structure
try:
    from gef.core.hopfion_relaxer import HopfionRelaxer
    from gef.core.ppn import fundamental_period
except ImportError:
    print("Could not import from src.gef..., trying relative paths.")
    from hopfion_relaxer import HopfionRelaxer
    # Assuming ppn is in the same directory for standalone execution
    from ppn import fundamental_period


# --- Worker Function ---
# This function is what each child process will execute.
# It's best to keep it self-contained at the top level of the script.
def run_relax_worker(g_sq: float, config: dict):
    """
    Worker function for a single relaxation run.
    It's designed to be pickled and sent to child processes.
    """
    try:
        solver_cfg = config['solver_config']
        solver_cfg['g_squared'] = g_sq
        sim = HopfionRelaxer(solver_cfg)
        sim.initialize_field(nw=1)

        phi_series, final_energy = sim.run_relaxation(
            n_skip=config['scan_config']['n_skip'],
            n_iter=config['scan_config']['n_iter']
        )
        
        period = fundamental_period(
            phi_series, 
            max_period=config['scan_config']['period_cap'],
            tol=config['scan_config']['period_tolerance']
        )
        
        return g_sq, period, final_energy
    except Exception:
        # It's crucial to catch exceptions in the worker, otherwise they can hang the pool
        # For a failed run, we return a clear error indicator.
        return g_sq, -1, np.nan

# --- Main Guard ---
if __name__ == "__main__":
    # --- Configuration ---
    CONFIG_PATH = script_dir / "configs" / "default_bifurcation_scan.yml"
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    scan_cfg = cfg['scan_config']
    g_values = np.linspace(scan_cfg['g_min'], scan_cfg['g_max'], scan_cfg['n_steps'])
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(cfg['output_dir']) / f"bifurcation_scan_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    results = []
    print(f"Starting bifurcation scan for g_squared from {scan_cfg['g_min']} to {scan_cfg['g_max']}...")

    # --- Multiprocessing Execution ---
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Pass the config to each worker
        future_to_g = {executor.submit(run_relax_worker, g, cfg): g for g in g_values}
        
        # This loop correctly updates tqdm as futures complete
        for future in tqdm(concurrent.futures.as_completed(future_to_g), total=len(g_values)):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                g_val = future_to_g[future]
                print(f"ERROR: g_squared={g_val:.4f} generated an exception: {exc}")
                results.append((g_val, -1, np.nan)) # Log failure

    # --- Analysis ---
    results_df = pd.DataFrame(results, columns=['g_squared', 'period', 'final_energy']).sort_values('g_squared')
    
    results_df.to_csv(output_dir / "raw_bifurcation_data.csv", index=False)
    print(f"\nRaw data saved to raw_bifurcation_data.csv")

    bifurcation_points = []
    last_period = 1
    # Use .itertuples() for efficient iteration
    for row in results_df.itertuples():
        g, p = row.g_squared, row.period
        # Check for a valid, positive period that's larger than the last one
        if p is not None and p > 0 and p > last_period:
            print(f"Period doubling detected: {last_period} -> {int(p)} at g_squared ≈ {g:.6f}")
            # We are only interested in true period-doubling events
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
        print("\nFewer than 3 period-doubling bifurcations were found. Cannot calculate δ.")