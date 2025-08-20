#!/usr/bin/env python3
"""
GEF Parameter Calibration Engine

This script uses numerical optimization to find the set of fundamental GEF
parameters that best reproduces observed physical constants, such as particle
mass ratios.

It reads a master calibration config, defines an objective function to minimize,
and uses scipy.optimize to search the parameter space.

Author: GEF Research Team
Date: 2025
"""

import logging
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec

import numpy as np
import yaml
from scipy.optimize import minimize

# --- Add project root to path for robust imports (optional) ---
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Package logger (Loguru) ---
try:
    from gef.core.logging import logger  # type: ignore
except Exception:
    # Fallback to stdlib logger if package logger unavailable
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


# --- Load MassSpectrumAnalyzer from sibling script robustly ---
def _load_mass_spectrum_analyzer():
    runner_path = (Path(__file__).resolve().parent / "run_nw_mass_spectrum.py").resolve()
    loader = SourceFileLoader("nw_mass_runner", str(runner_path))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)  # type: ignore[attr-defined]
    return module.MassSpectrumAnalyzer

MassSpectrumAnalyzer = _load_mass_spectrum_analyzer()

# --- The Core of the Calibrator: The Objective Function ---

def objective_function(params: np.ndarray, base_config: dict) -> float:
    """
    This is the function the optimizer will try to minimize.

    It takes a set of trial parameters, runs the full mass spectrum simulation,
    and returns a single "error" value indicating how far the simulation's
    results are from the real-world target values.
    """
    # Use package logger
    # 1. Create a temporary config for this specific simulation run
    temp_config = base_config.copy()
    
    # 2. Unpack the 'params' array and update the temp_config.
    #    This mapping must match the 'optimization_params' in the YAML.
    opt_params_map = base_config['optimization_params']['parameters']
    
    temp_config['resonance_parameters']['peaks'][0]['amplitude'] = params[opt_params_map.index('muon_amplitude')]
    temp_config['resonance_parameters']['peaks'][0]['sigma'] = params[opt_params_map.index('muon_sigma')]
    temp_config['resonance_parameters']['peaks'][1]['amplitude'] = params[opt_params_map.index('tau_amplitude')]
    temp_config['resonance_parameters']['peaks'][1]['sigma'] = params[opt_params_map.index('tau_sigma')]
    
    # You can add other parameters like P_env here if you want to optimize them too
    # temp_config['P_env'] = params[opt_params_map.index('P_env')]

    logger.info("-" * 60)
    logger.info(f"Optimizer trying new parameters:")
    logger.info(f"  Muon Amp: {params[0]:.4f}, Muon Sigma: {params[1]:.4f}")
    logger.info(f"  Tau Amp:  {params[2]:.4f}, Tau Sigma:  {params[3]:.4f}")

    # 3. Run simulations only (no plotting/saving) to speed up optimization
    #    Create a unique subdir for each trial so CSVs accumulate for inspection.
    import datetime as _dt
    trial_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_config['output_dir'] = Path(base_config['output_dir']) / "_temp_optimizer_run" / trial_stamp
    # Avoid multiprocessing pickling issues under dynamic import by forcing sequential
    temp_config['processes'] = 1
    # Allow thread-level parallelism
    temp_config['threads_per_process'] = int(base_config.get('threads_per_process', 1))
    # Quiet mode and skip heavy I/O during optimization (but DO save CSVs)
    temp_config['quiet'] = True
    temp_config['skip_plots'] = True
    temp_config['skip_save'] = False
    temp_config['record_series'] = False
    
    try:
        analyzer = MassSpectrumAnalyzer(temp_config)
        results_df = analyzer.run_parallel_simulations()
        # Write CSVs during optimization if not skipped
        csv_path = None
        if not temp_config.get('skip_save', False):
            csv_path = analyzer.save_results(results_df)
    except Exception as e:
        logger.error(f"A simulation run failed during optimization: {e}")
        return 1e12 # Return a huge error value if the simulation crashes

    # 4. Calculate the emergent masses and their ratios
    target_info = base_config['calibration_target']
    
    # Define the ground state mass (the "electron")
    # This is the stable energy of the "sea" state, far from resonances.
    sea_energy = results_df[results_df['winding_number'] == 1]['final_energy'].iloc[0]
    
    # Find the minimum energy (most stable state) in each generation's valley
    muon_valley_df = results_df[
        (results_df['winding_number'] > 10) & 
        (results_df['winding_number'] < 1000)
    ]
    tau_valley_df = results_df[
        (results_df['winding_number'] > 1000)
    ]
    
    if muon_valley_df.empty or tau_valley_df.empty or muon_valley_df['final_energy'].min() > sea_energy:
        logger.warning("Optimizer chose parameters that did not produce stable generation valleys.")
        return 1e9 # Return a large error if generations are not stable

    min_muon_energy = muon_valley_df['final_energy'].min()
    min_tau_energy = tau_valley_df['final_energy'].min()
    
    # Mass is energy above the deepest ground state (the Tau valley)
    deepest_ground_state = min_tau_energy
    
    mass_1 = sea_energy - deepest_ground_state
    mass_2 = min_muon_energy - deepest_ground_state
    mass_3 = 0 # By definition

    # Avoid division by zero if a mass is zero or negative
    if mass_1 <= 0 or mass_2 <= 0:
        logger.warning("Non-positive masses calculated, returning large error.")
        return 1e6

    simulated_mu_e_ratio = mass_2 / mass_1
    simulated_tau_e_ratio = mass_3 / mass_1 # This needs re-evaluation, let's focus on mu/e for now

    # Let's refine the mass definition. Mass should be energy above the *sea*.
    # The valleys represent binding energy.
    mass_electron = 1.0 # Define the sea as mass = 1 in arbitrary units
    mass_muon = (sea_energy - min_muon_energy)
    mass_tau = (sea_energy - min_tau_energy)

    if mass_muon <= 0 or mass_tau <= 0:
        logger.warning("Non-positive binding energies calculated, returning large error.")
        return 1e6

    simulated_mu_e_ratio = mass_muon
    simulated_tau_e_ratio = mass_tau

    # 5. Calculate the final error value (sum of squared log differences)
    target_mu_e_ratio = target_info['muon_to_electron_mass_ratio']
    target_tau_e_ratio = target_info['tau_to_electron_mass_ratio']
    
    # We need to scale our arbitrary units. Let's scale our simulation so that the
    # simulated electron mass matches the real one. The "sea" state is not the electron.
    # The N_w=1 state is the electron. Let's find its energy.
    m_e_energy = results_df[results_df['winding_number'] == 1.0]['final_energy'].iloc[0]
    m_mu_energy = min_muon_energy
    m_tau_energy = min_tau_energy

    # Mass is energy above the deepest state.
    deepest_state = m_tau_energy
    m_e = m_e_energy - deepest_state
    m_mu = m_mu_energy - deepest_state
    m_tau = 0 # By definition
    
    if m_e <= 0 or m_mu <= 0:
        return 1e6

    simulated_mu_e_ratio = m_mu / m_e
    simulated_tau_e_ratio = m_tau / m_e # This is zero, let's fix the definition.

    # --- REVISED AND CORRECTED MASS DEFINITION ---
    # Let's assume the N_w=1 state is the electron. Its mass is M_e.
    # The energy valleys for Muon and Tau represent states with mass M_mu and M_tau.
    # Mass = E_final - E_vacuum.
    m_e = results_df[results_df['winding_number'] == 1.0]['mass'].iloc[0]
    m_mu = results_df.loc[muon_valley_df['mass'].idxmin()]['mass']
    m_tau = results_df.loc[tau_valley_df['mass'].idxmin()]['mass']

    if m_e <= 0 or m_mu <= 0 or m_tau <=0:
        logger.warning("Negative mass calculated. Returning large error.")
        return 1e6

    simulated_mu_e_ratio = m_mu / m_e
    simulated_tau_e_ratio = m_tau / m_e
    
    error_mu = (np.log(simulated_mu_e_ratio) - np.log(target_mu_e_ratio))**2
    error_tau = (np.log(simulated_tau_e_ratio) - np.log(target_tau_e_ratio))**2
    
    total_error = error_mu + error_tau
    
    # Emit a concise summary for each trial regardless of quiet flag
    try:
        summary = {
            "csv": str(csv_path) if csv_path else "<skip_save>",
            "sea_energy": float(sea_energy),
            "min_muon_energy": float(min_muon_energy),
            "min_tau_energy": float(min_tau_energy),
            "m_e": float(m_e),
            "m_mu": float(m_mu),
            "m_tau": float(m_tau),
            "mu/e": float(simulated_mu_e_ratio),
            "tau/e": float(simulated_tau_e_ratio),
            "error": float(total_error),
        }
        logger.info(f"Trial results: {summary}")
    except Exception:
        pass
    
    return total_error

# --- Main Calibration Execution ---

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GEF Parameter Calibration Engine")
    parser.add_argument('config_path', help='Path to the master calibration YAML file')
    args = parser.parse_args()

    logger = get_logger(__name__)
    logger.info("--- Starting GEF Grand Fit Calibration ---")

    # 1. Load the master calibration config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Optional: perform a tiny warm-up run to JIT-compile kernels once
    try:
        warm_cfg = config.copy()
        warm_cfg['nw_sweep_range'] = [1]
        warm_cfg['relaxation_n_skip'] = int(config.get('relaxation_n_skip', 200))
        warm_cfg['relaxation_n_iter'] = int(config.get('relaxation_n_iter', 256))
        warm_cfg['processes'] = 1
        warm_cfg['threads_per_process'] = int(config.get('threads_per_process', 1))
        warm_cfg['quiet'] = True
        warm_cfg['skip_plots'] = True
        warm_cfg['skip_save'] = True
        warm_cfg['record_series'] = False
        MassSpectrumAnalyzer(warm_cfg).run_parallel_simulations()
    except Exception:
        pass

    # 3. Prepare for optimization
    opt_params = config['optimization_params']
    initial_guess = [p['initial'] for p in opt_params['parameter_info']]
    bounds = [(p['min'], p['max']) for p in opt_params['parameter_info']]

    # 4. Run the optimizer!
    result = minimize(
        fun=objective_function,
        x0=initial_guess,
        args=(config,),
        method='L-BFGS-B', # A good choice for bound-constrained problems
        bounds=bounds,
        options={'disp': True, 'maxiter': opt_params.get('max_iterations', 50)}
    )

    # 5. Process and save the final results
    logger.info("--- Calibration Finished ---")
    logger.info(f"Success: {result.success}")
    logger.info(f"Message: {result.message}")
    logger.info(f"Final Error (fun): {result.fun:.8f}")
    
    best_params = result.x
    logger.info("Best-fit parameters found:")
    for i, p_info in enumerate(opt_params['parameter_info']):
        logger.info(f"  {p_info['name']}: {best_params[i]:.6f}")

    # 6. Save the "Golden" configuration file
    golden_config = config.copy()
    golden_config['resonance_parameters']['peaks'][0]['amplitude'] = best_params[0]
    golden_config['resonance_parameters']['peaks'][0]['sigma'] = best_params[1]
    golden_config['resonance_parameters']['peaks'][1]['amplitude'] = best_params[2]
    golden_config['resonance_parameters']['peaks'][1]['sigma'] = best_params[3]
    
    output_dir = Path(config['output_dir'])
    golden_config_path = output_dir / "golden_config_leptons.yml"
    with open(golden_config_path, 'w') as f:
        yaml.dump(golden_config, f, sort_keys=False)
    logger.info(f"Golden configuration saved to: {golden_config_path}")

    # 7. Run the final, high-resolution analysis with the best parameters
    logger.info("Running final high-resolution analysis with best-fit parameters...")
    # Force sequential for the final high-res run as well (safe default)
    golden_config['processes'] = 1
    golden_config['threads_per_process'] = int(config.get('threads_per_process', 1))
    final_analyzer = MassSpectrumAnalyzer(golden_config)
    _, _, plot_path = final_analyzer.run_full_analysis()
    logger.info(f"Final 'Money Plot' saved to {plot_path}")
    logger.info("--- Calibration Complete ---")

if __name__ == '__main__':
    main()