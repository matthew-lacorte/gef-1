"""
GEF Genesis Analyzer: A tool for analyzing the output of "quantize-gravity"
simulations from the Fractal Spelunker.

This script performs the following scientific workflow:
1. Loads a series of raw numerical (.npy) data files from a simulation run.
2. Applies a robust edge-detection filter to isolate the fractal boundary
   (the "crust") of the emergent Genesis State in each file.
3. Calculates the fractal dimension of this boundary using a fast, Numba-jitted
   box-counting algorithm.
4. Generates a final plot of "Complexity (Fractal Dimension) vs. Law (Genesis Target)"
   to reveal the "quantized" values where complexity peaks.
"""
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import argparse
import sys
from scipy import ndimage
from tqdm import tqdm

# --- Numba Set-up (graceful fallback) ---
try:
    from numba import jit
except ImportError:
    print("Numba not found → running in pure-Python (will be slow).")
    jit = None

def _maybe_jit(func):
    """Apply a safe JIT if Numba is available; otherwise return the function.
    Use object mode to support numpy.polyfit and histogram2d.
    """
    if 'jit' in globals() and jit is not None:
        return jit(forceobj=True, cache=True)(func)
    return func

@_maybe_jit
def box_counting(binary_image: np.ndarray, min_box_size: int = 2, max_box_size: int = 512) -> float:
    """
    Calculates the fractal dimension of a binary image using the box-counting method.
    
    Args:
        binary_image: A 2D numpy array with values of 0 or 1.
        min_box_size: The smallest box size to test.
        max_box_size: The largest box size to test (should not exceed image dimensions).
        
    Returns:
        The calculated fractal dimension (slope of the log-log plot).
    """
    pixels = np.where(binary_image > 0)
    
    # Ensure there are points to count
    if len(pixels[0]) == 0:
        return 0.0

    # Generate a series of box sizes to test, on a log scale
    sizes = np.geomspace(min_box_size, max_box_size, num=15, dtype=np.int32)
    sizes = np.unique(sizes) # Remove duplicates
    
    counts = []
    for size in sizes:
        # Create a grid of bins of the current box size
        H, _, _ = np.histogram2d(pixels[0], pixels[1], bins=(np.arange(0, binary_image.shape[0] + size, size),
                                                              np.arange(0, binary_image.shape[1] + size, size)))
        # Count how many boxes contained part of the boundary
        counts.append(np.sum(H > 0))

    # The arrays for the log-log plot
    log_sizes = np.log(1.0 / sizes)
    log_counts = np.log(counts)

    # Fit a line to the log-log plot. The slope is the fractal dimension.
    # Using np.polyfit is a standard way to do linear regression.
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    
    return coeffs[0]

def analyze_simulation_run(run_directory: Path, config_path: Path | None = None):
    """
    Main analysis function. Loads data, calculates dimensions, and plots results.
    """
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Simulation directory not found at: {run_directory}")

    # Determine config path: prefer explicit override; otherwise use used_config.yml in run dir
    if config_path is None:
        config_path = run_directory / 'used_config.yml'
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Could not find 'used_config.yml' in {run_directory}. "
                f"Provide --config to point to the original config file.")

    with config_path.open() as f:
        config = yaml.safe_load(f)
    
    print(f"Analyzing run in: {run_directory}")

    # --- 1. Load Data and Parameters ---
    npy_files = sorted(run_directory.glob("frame_*_data.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy data files found in {run_directory}. Did the simulation run correctly?")

    p_sweep = config['genesis_sweep_params']
    param_name = p_sweep['parameter_to_sweep']
    sweep_values = np.linspace(p_sweep['start_value'], p_sweep['end_value'], p_sweep['num_frames'])

    if len(sweep_values) != len(npy_files):
        print(f"Warning: Mismatch between number of frames in config ({len(sweep_values)}) "
              f"and .npy files found ({len(npy_files)}). Using file count.")
        sweep_values = np.linspace(p_sweep['start_value'], p_sweep['end_value'], len(npy_files))

    fractal_dimensions = []
    
    print("Calculating fractal dimension for each frame...")
    for i, file_path in enumerate(tqdm(npy_files)):
        # --- 2. Isolate the Boundary ---
        law_grid = np.load(file_path)
        
        # Apply Sobel filter to find gradients (edges)
        sx = ndimage.sobel(law_grid, axis=0, mode='constant')
        sy = ndimage.sobel(law_grid, axis=1, mode='constant')
        gradient_map = np.hypot(sx, sy)
        
        # Threshold the gradient map to get a clean binary image of the "crust"
        # We use a percentile-based threshold to be robust to different data scales
        threshold = np.percentile(gradient_map, 95)
        binary_boundary = (gradient_map > threshold).astype(np.uint8)
        
        # Save one example boundary image for verification
        if i == len(npy_files) // 2:
            plt.imsave(run_directory / "analysis_boundary_map_example.png",
                       binary_boundary, cmap='gray')

        # --- 3. Calculate Fractal Dimension ---
        dimension = box_counting(binary_boundary, max_box_size=law_grid.shape[0] // 2)
        fractal_dimensions.append(dimension)

    # --- 4. Generate the Final Plot ---
    print("Generating final complexity plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(sweep_values, fractal_dimensions, 'o-', label='Fractal Dimension')
    ax.set_xlabel(f"Genesis Parameter: {param_name}", fontsize=14)
    ax.set_ylabel("Complexity (Fractal Dimension)", fontsize=14)
    ax.set_title("Complexity of Emergent Physical Laws vs. Genesis Parameter", fontsize=16, pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight the peaks - the "quantized" solutions
    max_dim = np.max(fractal_dimensions)
    peak_threshold = max_dim * 0.95 # Highlight points within 95% of the max
    peaks = np.where(fractal_dimensions >= peak_threshold)[0]
    
    if len(peaks) > 0:
        ax.plot(sweep_values[peaks], np.array(fractal_dimensions)[peaks], 'ro', markersize=10, label='Quantized States')
    
    ax.legend()
    plt.tight_layout()
    final_plot_path = run_directory / "analysis_complexity_vs_law.png"
    plt.savefig(final_plot_path, dpi=200)
    plt.close(fig)
    
    print(f"\nAnalysis complete!")
    print(f"Example boundary map saved to: {run_directory / 'analysis_boundary_map_example.png'}")
    print(f"Final complexity plot saved to: {final_plot_path}")

def main():
    """Main entry point for the analyzer script."""
    parser = argparse.ArgumentParser(description="GEF Genesis Analyzer.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('run_directory', type=Path,
                        help="Path to a single simulation run directory (e.g., quantize-gravity_YYYY-MM-DD_HH-MM-SS).\n"
                             "The analyzer will load frames from this folder and read configuration\n"
                             "from 'used_config.yml' saved by the simulation.")
    parser.add_argument('--config', '-c', type=Path, default=None,
                        help="Optional: explicit path to a YAML configuration file.\n"
                             "If omitted, the analyzer looks for 'used_config.yml' inside the run directory.")
    args = parser.parse_args()

    try:
        analyze_simulation_run(args.run_directory, args.config)
    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()