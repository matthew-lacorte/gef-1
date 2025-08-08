"""
GEF κ-Flow Simulator: A tool for visualizing how massive objects
perturb the local rate of time, as described by the GEF framework.

This script implements the core GEF law for gravitational time dilation:
||κ(r)|| = κ_∞ * (1 - β * W(r))

It generates a 2D heatmap of the local time flow rate around a configurable
set of massive bodies.
"""
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import time
import argparse
import sys
from typing import Dict, List

def run_simulation(config: Dict) -> np.ndarray:
    """
    Calculates the magnitude of the κ-flow across a 2D grid.
    
    Args:
        config: A dictionary containing all simulation parameters.
        
    Returns:
        A 2D numpy array representing the local time flow rate at each point.
    """
    p_sim = config['simulation']
    grid_size = p_sim['grid_size']
    win = p_sim['window']
    masses = p_sim['masses']
    
    # G_eff is the effective gravitational coupling, combining α and β
    G_eff = p_sim['G_effective']
    
    # Create the coordinate grid for the simulation space
    x = np.linspace(win[0], win[1], grid_size)
    y = np.linspace(win[2], win[3], grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Initialize the total wake potential across the grid
    total_potential = np.zeros_like(xx)
    
    print("Calculating total wake potential from all masses...")
    for mass_obj in masses:
        M = mass_obj['mass']
        mx, my = mass_obj['position']
        
        # Calculate distance from this mass to every point on the grid
        r = np.sqrt((xx - mx)**2 + (yy - my)**2)
        
        # Add a small epsilon to avoid division by zero at the mass's center
        r += 1e-9
        
        # The Wake Potential W(r) = M/r
        potential = M / r
        total_potential += potential
        
    # Apply the core GEF law. We assume κ_∞ = 1 for a normalized plot.
    # ||κ(r)|| = κ_∞ * (1 - G_eff * Σ(M_i / r_i))
    k_flow_magnitude = 1.0 - G_eff * total_potential
    
    # Clip values for visualization: time can't flow backwards
    # and we can cap extreme values for better color mapping.
    k_flow_magnitude = np.maximum(0.0, k_flow_magnitude)
    
    return k_flow_magnitude

def plot_results(k_flow: np.ndarray, config: Dict, output_path: Path):
    """
    Generates and saves a high-quality visualization of the κ-flow field.
    """
    print("Generating visualization...")
    p_sim = config['simulation']
    p_plot = config['plotting']
    win = p_sim['window']
    masses = p_sim['masses']
    
    fig, ax = plt.subplots(figsize=tuple(p_plot['figsize']), facecolor='black')
    
    # Plot the κ-flow magnitude as a heatmap
    im = ax.imshow(k_flow, cmap=p_plot['cmap'], extent=win, origin='lower',
                   interpolation='bilinear')
                   
    # Add a colorbar to show the scale of time dilation
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Local Time Flow Rate (κ/κ_∞)", color='white', fontsize=14)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Overlay the massive objects on the plot
    # The size of the marker is proportional to the log of the mass for better visuals
    mass_values = [m['mass'] for m in masses]
    sizes = 100 + 300 * (np.log(mass_values) / np.log(max(mass_values)))
    
    for i, mass_obj in enumerate(masses):
        mx, my = mass_obj['position']
        ax.scatter(mx, my, s=sizes[i], c='white', edgecolors='red', linewidths=1.5, alpha=0.8)

    ax.set_title("GEF Time Dilation Well", color='white', fontsize=20, pad=20)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor='black', dpi=p_plot['dpi'])
    plt.close(fig)
    print(f"Visualization saved to {output_path}")

def main():
    """Main entry point: parses args, loads config, and runs the simulation."""
    script_dir = Path(__file__).parent.absolute()
    default_config = script_dir / "configs" / "default_k_flow_simulator.yml"
    
    parser = argparse.ArgumentParser(description="GEF κ-Flow Simulator.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', '-c', type=Path, default=default_config,
                        help=f"Path to the YAML configuration file.\nDefault: {default_config}")
    args = parser.parse_args()

    try:
        if not args.config.is_file():
            raise FileNotFoundError(f"Config file not found at '{args.config}'")
        with args.config.open() as f:
            config = yaml.safe_load(f)
        
        # --- Run Simulation ---
        k_flow_field = run_simulation(config)
        
        # --- Plot Results ---
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"time_dilation_well_{timestamp}.png"
        
        plot_results(k_flow_field, config, output_path)

    except (ValueError, FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"\nConfiguration or Input Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()