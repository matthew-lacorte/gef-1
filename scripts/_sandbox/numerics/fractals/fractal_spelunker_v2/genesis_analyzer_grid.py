"""
GEF Genesis Grid Analyzer: A tool for analyzing the output of 2D parameter
grid sweeps from the Fractal Spelunker's "quantize-gravity" mode.

This script creates a 2D heatmap of "Complexity vs. Law," revealing the
full phase diagram of the emergent universe and identifying the "razor's edge"
of stability where maximal complexity occurs.
"""
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import argparse
import sys
from scipy import ndimage
from tqdm import tqdm
import re

# --- (The Numba-jitted box_counting function is identical to the previous analyzer) ---
try:
    from numba import njit
except ImportError:
    print("Numba not found → running in pure-Python (will be slow).")
    def njit(f=None, **kw): return (lambda g: g)(f) if f else (lambda g: g)

@njit
def box_counting(binary_image: np.ndarray, min_box_size: int = 2) -> float:
    pixels = np.where(binary_image > 0)
    if len(pixels[0]) == 0: return 0.0
    max_box_size = min(binary_image.shape) // 2
    if max_box_size <= min_box_size: return 0.0
    
    sizes = np.geomspace(min_box_size, max_box_size, num=10, dtype=np.int32)
    sizes = np.unique(sizes)
    
    counts = []
    for size in sizes:
        H, _, _ = np.histogram2d(pixels[0], pixels[1], bins=(np.arange(0, binary_image.shape[0] + size, size),
                                                              np.arange(0, binary_image.shape[1] + size, size)))
        counts.append(np.sum(H > 0))

    if len(counts) < 2: return 0.0

    log_sizes = np.log(1.0 / sizes)
    log_counts = np.log(np.array(counts, dtype=np.float64))
    
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return coeffs[0]

def analyze_grid_run(run_directory: Path):
    """Loads a 2D grid of data, calculates dimensions, and plots the heatmap."""
    config_path = run_directory / 'run_config.yaml'
    if not config_path.is_file():
        raise FileNotFoundError(f"run_config.yaml not found in {run_directory}")
    with config_path.open() as f:
        config = yaml.safe_load(f)
    
    print(f"Analyzing grid run in: {run_directory}")

    # --- 1. Reconstruct the Grid from Filenames ---
    p_sweep = config['genesis_sweep_params']
    num_theta = p_sweep['theta_sweep']['num_frames']
    num_target = p_sweep['target_sweep']['num_frames']
    
    complexity_grid = np.zeros((num_theta, num_target))
    
    npy_files = list(run_directory.glob("frame_*_*_data.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy data files found in {run_directory}.")

    print("Calculating fractal dimension for each point in the grid...")
    for file_path in tqdm(npy_files):
        match = re.search(r"frame_(\d+)_(\d+)_data.npy", file_path.name)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            
            law_grid = np.load(file_path)
            sx = ndimage.sobel(law_grid, axis=0, mode='constant')
            sy = ndimage.sobel(law_grid, axis=1, mode='constant')
            gradient_map = np.hypot(sx, sy)
            threshold = np.percentile(gradient_map, 95)
            binary_boundary = (gradient_map > threshold).astype(np.uint8)
            
            dimension = box_counting(binary_boundary)
            if i < num_theta and j < num_target:
                complexity_grid[i, j] = dimension

    # --- 2. Generate the Final Heatmap ---
    print("Generating final complexity heatmap...")
    theta_params = p_sweep['theta_sweep']
    target_params = p_sweep['target_sweep']
    
    extent = [
        target_params['start_value'], target_params['end_value'],
        theta_params['start_value'], theta_params['end_value']
    ]
    
    fig, ax = plt.subplots(figsize=(16, 12))
    # Transpose the grid because imshow's first index is rows (y-axis)
    im = ax.imshow(complexity_grid.T, extent=extent, origin='lower',
                   aspect='auto', cmap='magma', interpolation='bicubic')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Complexity (Fractal Dimension)", fontsize=14)
    
    # --- 3. Find and Mark the Peak Complexity ---
    peak_loc_indices = np.unravel_index(np.argmax(complexity_grid), complexity_grid.shape)
    peak_theta_idx, peak_target_idx = peak_loc_indices
    
    theta_axis = np.linspace(theta_params['start_value'], theta_params['end_value'], num_theta)
    target_axis = np.linspace(target_params['start_value'], target_params['end_value'], num_target)
    
    peak_theta = theta_axis[peak_theta_idx]
    peak_target = target_axis[peak_target_idx]
    peak_complexity = complexity_grid[peak_theta_idx, peak_target_idx]
    
    ax.scatter(peak_target, peak_theta, s=200, facecolors='none', edgecolors='cyan', linewidth=2)
    ax.text(peak_target, peak_theta + 0.05 * (extent[3]-extent[2]), 
            f"  Peak Complexity: {peak_complexity:.3f}\n  θ = {peak_theta:.3f}π\n  target = {peak_target:.3f}",
            color='cyan', fontsize=12, ha='center')

    ax.set_xlabel("Genesis Parameter: genesis_target", fontsize=14)
    ax.set_ylabel("Stability Parameter: fixed_angle_pi (x π)", fontsize=14)
    ax.set_title("Phase Diagram of Universal Complexity", fontsize=18, pad=20)
    
    final_plot_path = run_directory / "analysis_complexity_heatmap.png"
    plt.savefig(final_plot_path, dpi=200)
    plt.close(fig)

    print("\nAnalysis complete!")
    print(f"Peak complexity found at θ={peak_theta:.4f}π, target={peak_target:.4f}")
    print(f"Final heatmap saved to: {final_plot_path}")

def main():
    """Main entry point for the grid analyzer script."""
    parser = argparse.ArgumentParser(description="GEF Genesis Grid Analyzer.")
    parser.add_argument('run_directory', type=Path,
                        help="Path to the simulation output directory containing indexed .npy files.")
    args = parser.parse_args()

    try:
        analyze_grid_run(args.run_directory)
    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()