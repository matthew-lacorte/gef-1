#!/usr/bin/env python3
"""
GEF Excitation Chain Runner

This script implements the "Electron -> Muon -> Tau" roadmap.
It runs a sequence of simulations where each step can use the relaxed
state of the previous step as its starting point.

Workflow:
1.  Step 1 (Ground State): Run a simulation from a topological seed (nw=1)
    and save the final field configuration.
    `python run_excitation_chain.py -c configs/config_step1_electron.yml --step 1`

2.  Step 2 (First Overtone): Load the field from Step 1, apply a kinetic
    perturbation ("ping"), relax the system again, and save the new state.
    `python run_excitation_chain.py -c configs/config_step2_muon.yml --step 2`

3.  Step 3 (Second Overtone): Repeat the process, starting from the Step 2 state.
    `python run_excitation_chain.py -c configs/config_step3_tau.yml --step 3`
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import yaml
from loguru import logger

try:
    from gef.geometry.hopfion_relaxer import HopfionRelaxer
except Exception:
    from hopfion_relaxer import HopfionRelaxer

# --- State Management ---

def save_state(solver: HopfionRelaxer, config: Dict, output_path: Path):
    """Saves the field, velocity, and config to a compressed NPZ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        phi=solver.phi,
        velocity=solver.velocity_buffer,
        config=config,
    )
    logger.info(f"Solver state saved to {output_path}")

def load_state(solver: HopfionRelaxer, input_path: Path) -> Dict:
    """Loads a field and velocity state into the solver from an NPZ file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input state file not found: {input_path}")
    
    data = np.load(input_path, allow_pickle=True)
    
    phi = data['phi']
    if phi.shape != solver.phi.shape:
        raise ValueError(f"Shape mismatch: loaded phi {phi.shape} vs solver {solver.phi.shape}")
        
    solver.phi[:] = phi
    if 'velocity' in data:
        solver.velocity_buffer[:] = data['velocity']
    else:
        solver.velocity_buffer.fill(0.0)
        
    logger.info(f"Loaded solver state from {input_path}")
    return data['config'].item()

def analytic_uniform_vacuum(config: Dict) -> Optional[float]:
    """Calculate the analytic value of phi in the uniform vacuum."""
    try:
        mu2 = float(config["mu_squared"])
        P = float(config["P_env"])
        lam = float(config["lambda_val"])
        
        if mu2 <= 2.0 * P:
            return 0.0
        
        phi0_sq = (mu2 - 2.0 * P) / lam
        
        # This is the physical constraint check from your previous work
        if phi0_sq > 1.0:
            logger.warning(f"Analytic vacuum |phi0|={phi0_sq**0.5:.3f} > 1. Potential may be unstable.")
            
        return phi0_sq**0.5
    except KeyError as e:
        logger.error(f"Missing key for analytic vacuum calculation: {e}")
        return None


# --- Physics Perturbation ---

def apply_gaussian_punch(solver: HopfionRelaxer, config: Dict):
    """
    Applies a localized, asymmetric kinetic "punch" to the soliton core.
    This is a more violent perturbation designed to break symmetries and
    potentially trigger a transition to a different topological state.
    """
    p_config = config.get("perturbation", {})
    amplitude = float(p_config.get("amplitude", 0.5))
    width = float(p_config.get("width", 4.0)) # Width of the Gaussian in grid units
    
    logger.info(f"Applying 'gaussian_punch' with amplitude {amplitude:.2e}, width {width:.1f}")
    
    shape = solver.lattice_shape
    dims = [np.arange(s, dtype=np.float64) for s in shape]
    x, y, z, w = np.meshgrid(dims[0], dims[1], dims[2], dims[3], indexing="ij")

    # Center coordinates on the soliton's core
    x_c = shape[0] / 2.0
    y_c = shape[1] / 2.0
    z_c = shape[2] / 2.0
    w_c = shape[3] / 2.0
    
    # Create a 4D Gaussian packet, offset from the center to make it asymmetric
    offset = width
    squared_dist = (
        ((x - x_c - offset)**2) + 
        ((y - y_c)**2) + 
        ((z - z_c)**2) + 
        ((w - w_c)**2)
    )
    
    punch_field = np.exp(-squared_dist / (2 * width**2))
    
    # Apply to the velocity buffer
    solver.velocity_buffer += amplitude * punch_field
    logger.success("Gaussian punch applied.")

# Then, modify the main `apply_perturbation` function to choose which ping to use
def apply_perturbation(solver: HopfionRelaxer, config: Dict):
    """
    Applies a kinetic "ping" to excite the system.
    This adds energy without directly moving the field, allowing it to find
    resonant overtones.
    """
    p_config = config.get("perturbation", {})
    mode_shape = p_config.get("mode_shape", "symmetric_breathing")

    if mode_shape == "gaussian_punch":
        apply_gaussian_punch(solver, config)
    else: # Default to the original symmetric mode
        # (The original code for symmetric_breathing goes here)
        p_config = config.get("perturbation", {})
        amplitude = float(p_config.get("amplitude", 1e-2))
        mode_shape = p_config.get("mode_shape", "symmetric_breathing")
        
        logger.info(f"Applying '{mode_shape}' perturbation with amplitude {amplitude:.2e}")
        
        shape = solver.lattice_shape
        dims = [np.arange(s, dtype=np.float64) for s in shape]
        x, y, z, w = np.meshgrid(dims[0], dims[1], dims[2], dims[3], indexing="ij")

        # Center the coordinates
        x -= shape[0] / 2.0
        y -= shape[1] / 2.0
        z -= shape[2] / 2.0
        w -= shape[3] / 2.0
        
        # Create a symmetric "breathing mode" perturbation
        # This mode smoothly goes to zero at the boundaries.
        perturbation_field = (
            np.cos(2 * np.pi * x / shape[0]) *
            np.cos(2 * np.pi * y / shape[1]) *
            np.cos(2 * np.pi * z / shape[2]) *
            np.cos(2 * np.pi * w / shape[3])
        )
        
        # Apply to the velocity buffer to add kinetic energy
        solver.velocity_buffer += amplitude * perturbation_field
        logger.success("Perturbation applied to velocity buffer.")


# --- Main Orchestration ---

def run_step(config: Dict, step: int):
    """Orchestrates a single step in the excitation chain."""
    logger.info(f"--- Running Excitation Chain: Step {step} ---")

    # Setup output directory and file paths
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping (dict), got {type(config)}. Check the YAML at the provided path.")

    output_dir = Path(config.get("output_dir", "./data/sim_outputs/default_chain"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_state_path = output_dir / f"step_{step}_final_state.npz"
    
    # Persist the config used for this step
    (output_dir / f"step_{step}_used_config.yml").write_text(yaml.safe_dump(config, sort_keys=False))

    # Build the solver from the config's 'solver' section
    if "solver" not in config or not isinstance(config["solver"], dict):
        raise KeyError("YAML is missing a 'solver' section or it is not a mapping.")
    solver_config = config["solver"]
    solver = HopfionRelaxer(solver_config)
    
    # --- State Initialization ---
    if step == 1:
        # Step 1: Initialize from a topological seed
        nw_seed = int(solver_config.get("seed_topology", 1))
        logger.info(f"Initializing field with topological seed nw={nw_seed}")
        solver.initialize_field(nw=nw_seed)
    else:
        # Subsequent steps: Load state from the previous step
        input_path = Path(config["input_state_path"])
        load_state(solver, input_path)
        
        # Apply the perturbation to the loaded state
        if "perturbation" in config:
            apply_perturbation(solver, config)
        else:
            logger.warning("Running subsequent step without a 'perturbation' config. System may relax to the same state.")

    # --- Run Relaxation ---
    E_initial = solver.compute_total_energy()
    logger.info(f"Initial Energy (E_i): {E_initial:.6f}")
    
    # Run relaxation and record energy series for diagnostics
    _, E_final, energy_history = solver.run_relaxation(record_energy_series=True)
    
    if not np.isfinite(E_final):
        logger.error("Relaxation failed to converge (E_final is NaN). State will not be saved.")
        return

    logger.success(f"Relaxation Converged. Final Energy (E_f): {E_final:.6f}")
    
    # --- Report and Save ---
    breakdown = solver.compute_energy_breakdown()
    logger.info(f"Final Energy Breakdown: {breakdown}")
    
    # To calculate mass density, we need the vacuum energy
    logger.info("Calculating numeric vacuum energy for mass density...")
    vac_solver = HopfionRelaxer(solver_config)
    
    # Initialize the vacuum simulation with a smarter guess
    phi0 = analytic_uniform_vacuum(solver_config)
    if phi0 is not None:
        logger.info(f"Seeding vacuum simulation with analytic uniform value phi0 = {phi0:.6f}")
        vac_solver.phi.fill(phi0)
        vac_solver.velocity_buffer.fill(0.0)
    else:
        # Fallback to the old method if analytic calc fails
        vac_solver.initialize_field(nw=0)

    # Give the vacuum extra time to converge on these flat potentials
    _, E_vac, _ = vac_solver.run_relaxation(n_iter=200000) # Give it a generous budget
    
    mass = E_final - E_vac
    vol4 = np.prod(solver.lattice_shape) * (solver.dx**4)
    mass_density = mass / vol4
    
    logger.info(f"Vacuum Energy (E_vac): {E_vac:.6f}")
    logger.info(f"Total Mass (E_f - E_vac): {mass:.6f}")
    logger.info(f"Mass Density (ρ): {mass_density:.6f}")

    # Machine-readable one-line summary for CLI parsing (CSV-like)
    try:
        amp = config.get("perturbation", {}).get("amplitude", None)
    except Exception:
        amp = None
    print(f"SUMMARY amplitude={amp if amp is not None else 'NA'} mass={mass:.6f} density={mass_density:.6f}")
    
    save_state(solver, config, output_state_path)

    # Optionally persist the energy history for this step
    if energy_history is not None:
        try:
            p_config = config.get("perturbation", {})
            amp = float(p_config.get("amplitude", 0.0))
        except Exception:
            amp = 0.0
        fric = float(solver.friction)
        history_filename = output_dir / f"step_{step}_energy_history_amp{amp:.2e}_fric{fric:.2f}.txt"
        np.savetxt(history_filename, energy_history)
        logger.info(f"Saved energy history to {history_filename}")

        # Plot and save a visualization of the energy history
        try:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(np.arange(len(energy_history)), energy_history, lw=1.5)
            ax.set_xlabel("Energy check index")
            ax.set_ylabel("Total Energy")
            ax.set_title(f"Energy vs. Checks (step={step}, amp={amp:.2e}, fric={fric:.2f})")
            ax.grid(True, linestyle='--', alpha=0.5)
            plot_path = output_dir / f"step_{step}_energy_history_amp{amp:.2e}_fric{fric:.2f}.png"
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved energy history plot to {plot_path}")
        except Exception:
            logger.exception("Failed to generate energy history plot.")

def _load_yaml_config(path: Path) -> Dict:
    """Load a YAML file and ensure it returns a mapping.

    Raises a clear error if the file is empty or does not contain a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config path not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"YAML file is empty or only contains comments: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML must be a mapping (dict), got {type(data)} in {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Run a step in the GEF excitation chain.")
    parser.add_argument("-c", "--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--step", type=int, required=True, choices=[1, 2, 3], help="Which step of the chain to run.")
    parser.add_argument("--amplitude", type=float, help="Override the perturbation amplitude in the config.")
    parser.add_argument("--friction", type=float, help="Override the solver friction parameter.")

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # If the CLI argument is provided, override the config value
        if args.amplitude is not None:
            if "perturbation" not in config:
                config["perturbation"] = {}
            logger.info(f"Overriding perturbation amplitude with CLI value: {args.amplitude}")
            config["perturbation"]["amplitude"] = args.amplitude

        # Optional: override solver friction via CLI
        if args.friction is not None:
            if "solver" not in config or not isinstance(config["solver"], dict):
                raise KeyError("YAML is missing a 'solver' section; cannot override friction.")
            logger.info(f"Overriding solver friction with CLI value: {args.friction}")
            config["solver"]["friction"] = float(args.friction)

        run_step(config, args.step)
    except Exception:
        logger.exception(f"A critical error occurred in Step {args.step}.")
        return 1
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())