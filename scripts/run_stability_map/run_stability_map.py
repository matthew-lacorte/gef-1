"""
Run a 2D parameter sweep to generate a stability map for GEF configurations.

* Reads parameters from a YAML config (default: ./configs/default_run_stability_map.yml)
* Performs a 2D grid scan over specified parameter ranges
* Computes stability metrics for each point in the parameter space
* Outputs a heatmap visualization and data files in a timestamped directory
"""

# THIS HASN'T BEEN REVIEWED AT ALL

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap

from gef.core.logging import logger, setup_logfile
from gef.core.constants import CONSTANTS_DICT

# ─────────────────────────────────────────────────── Physical constants
c = CONSTANTS_DICT["c"].value
planck = CONSTANTS_DICT["planck"].value
electron_volt = CONSTANTS_DICT["electron_volt"].value

# ─────────────────────────────────────────────────── Stability calculation
def compute_stability_metric(param1: float, param2: float) -> float:
    """
    Compute a stability metric for a given set of parameters.
    
    This is a placeholder function that should be replaced with actual
    stability calculations based on GEF theory.
    """
    # Example: Simple stability metric based on distance from a reference point
    # In a real implementation, this would use GEF equations to calculate stability
    reference_point = (0.5, 0.5)
    distance = np.sqrt((param1 - reference_point[0])**2 + (param2 - reference_point[1])**2)
    stability = np.exp(-5 * distance)  # Higher values = more stable
    return stability

def generate_stability_map(param1_range: Tuple[float, float, int],
                          param2_range: Tuple[float, float, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a stability map by sweeping through parameter ranges.
    
    Args:
        param1_range: Tuple of (start, stop, num_points) for first parameter
        param2_range: Tuple of (start, stop, num_points) for second parameter
        
    Returns:
        Tuple of (param1_grid, param2_grid, stability_values)
    """
    param1_values = np.linspace(*param1_range)
    param2_values = np.linspace(*param2_range)
    
    param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)
    stability_values = np.zeros_like(param1_grid)
    
    # Compute stability for each point in the grid
    for i in range(param1_grid.shape[0]):
        for j in range(param1_grid.shape[1]):
            stability_values[i, j] = compute_stability_metric(param1_grid[i, j], param2_grid[i, j])
    
    return param1_grid, param2_grid, stability_values

# ─────────────────────────────────────────────────── Visualization
def plot_stability_map(param1_grid: np.ndarray, param2_grid: np.ndarray, 
                      stability_values: np.ndarray, 
                      param1_name: str, param2_name: str,
                      output_dir: Path) -> Path:
    """
    Create a heatmap visualization of the stability map.
    
    Args:
        param1_grid, param2_grid: Meshgrid of parameter values
        stability_values: Computed stability metrics
        param1_name, param2_name: Names of parameters for axis labels
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    plt.figure(figsize=(10, 8))
    
    # Create a custom colormap: blue (unstable) to red (stable)
    colors = [(0, 0, 0.8), (0, 0.8, 0.8), (0.8, 0.8, 0), (0.8, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('stability_cmap', colors, N=256)
    
    contour = plt.contourf(param1_grid, param2_grid, stability_values, 50, cmap=cmap)
    plt.colorbar(contour, label='Stability Metric')
    
    plt.xlabel(param1_name)
    plt.ylabel(param2_name)
    plt.title('GEF Configuration Stability Map')
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = output_dir / 'stability_map.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

# ─────────────────────────────────────────────────── CLI helpers
def _default_cfg(script_dir: Path) -> Path:
    return script_dir / "configs" / "default_run_stability_map.yml"

def _new_outdir(script_dir: Path, base: Path | None) -> Path:
    t = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    out_base = (base or script_dir / "outputs").resolve()
    out_dir = out_base / f"{t}-{run_id}"
    out_dir.mkdir(parents=True)
    return out_dir

# ─────────────────────────────────────────────────── Main
def main(argv: List[str] | None = None) -> int:
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(prog="run_stability_map")
    p.add_argument("-c", "--config", type=Path, default=_default_cfg(script_dir))
    p.add_argument("-o", "--output-dir", type=Path, default=None)
    p.add_argument("--log", type=Path, default=None)
    args = p.parse_args(argv)

    if args.log:
        setup_logfile(str(args.log))

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = _new_outdir(script_dir, args.output_dir)
    (out_dir / "used_config.yml").write_text(yaml.dump(cfg))
    logger.info(f"Output dir: {out_dir}")
    
    # Extract configuration parameters
    param1_cfg = cfg["param1"]
    param2_cfg = cfg["param2"]
    
    param1_range = (param1_cfg["start"], param1_cfg["stop"], param1_cfg["num_points"])
    param2_range = (param2_cfg["start"], param2_cfg["stop"], param2_cfg["num_points"])
    
    logger.info(f"Starting 2D parameter sweep: {param1_cfg['name']} × {param2_cfg['name']}")
    
    # Generate the stability map
    param1_grid, param2_grid, stability_values = generate_stability_map(param1_range, param2_range)
    
    # Save the raw data
    np.savez(out_dir / 'stability_data.npz', 
             param1=param1_grid, 
             param2=param2_grid, 
             stability=stability_values)
    
    # Create visualization
    plot_path = plot_stability_map(param1_grid, param2_grid, stability_values,
                                  param1_cfg['name'], param2_cfg['name'], out_dir)
    logger.info(f"Stability map saved to {plot_path}")
    
    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
