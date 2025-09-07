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


# --- Physics Perturbation ---

def apply_perturbation(solver: HopfionRelaxer, config: Dict):
    """
    Applies a kinetic "ping" to excite the system.
    This adds energy without directly moving the field, allowing it to find
    resonant overtones.
    """
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
    
    _, E_final = solver.run_relaxation()
    
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
    vac_solver.initialize_field(nw=0)
    _, E_vac = vac_solver.run_relaxation()
    
    mass = E_final - E_vac
    vol4 = np.prod(solver.lattice_shape) * (solver.dx**4)
    mass_density = mass / vol4
    
    logger.info(f"Vacuum Energy (E_vac): {E_vac:.6f}")
    logger.info(f"Total Mass (E_f - E_vac): {mass:.6f}")
    logger.info(f"Mass Density (ρ): {mass_density:.6f}")
    
    save_state(solver, config, output_state_path)

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
    args = parser.parse_args()

    try:
        config = _load_yaml_config(args.config)
        run_step(config, args.step)
    except Exception:
        logger.exception(f"A critical error occurred in Step {args.step}.")
        return 1
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())