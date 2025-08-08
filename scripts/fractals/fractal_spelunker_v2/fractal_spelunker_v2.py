"""
Fractal Spelunker: A high-performance engine for exploring and analyzing
fractals derived from the General Euclidean Flow (GEF) model.

This version includes an animated 'quantize-gravity' mode to simulate the 
co-evolution of local geometry and physical laws, searching for stable, 
self-consistent "Genesis Configurations" across a parameter sweep.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import yaml
from pathlib import Path
import math
import time
from tqdm import tqdm
import argparse
import sys
import traceback
from typing import Dict, Any

# --- Numba Set-up (graceful fallback) ---
try:
    from numba import njit, prange, float64, boolean, complex128
    from numba.experimental import jitclass
except ImportError:
    print("Numba not found → running in pure-Python (will be slow).")
    def njit(f=None, **kw): return (lambda g: g)(f) if f else (lambda g: g)
    def jitclass(spec): return lambda cls: cls
    prange = range

# ==============================================================================
# 1. PARAMETER DATA STRUCTURES & CORE ITERATORS
# ==============================================================================
# ... (This section is unchanged as its logic is sound) ...
param_spec = [
    ('scale_x', float64), ('scale_y', float64),
    ('use_spatial_lambda', boolean),
    ('base_lambda', float64), ('amp', float64), ('sigma_sq', float64)
]
@jitclass(param_spec)
class IteratorParams:
    """Numba-compatible data structure to hold fractal iteration parameters."""
    def __init__(self, scale_x, scale_y, use_spatial_lambda, base_lambda, amp, sigma_sq):
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.use_spatial_lambda = use_spatial_lambda
        self.base_lambda = base_lambda
        self.amp = amp
        self.sigma_sq = sigma_sq
@njit(inline='always')
def gef_iter(z: complex, c: complex, cos_th: float, sin_th: float, params: IteratorParams) -> complex:
    """Performs one iteration of the GEF model, handling all anisotropy cases."""
    if params.use_spatial_lambda:
        r_squared = min(z.real**2 + z.imag**2, 100.0)
        lam = params.base_lambda + params.amp * math.exp(-r_squared / params.sigma_sq)
        scale_y = max(0.1, min(lam, 2.0))
    else:
        scale_y = params.scale_y

    rot = cos_th + 1j * sin_th
    irot = cos_th - 1j * sin_th
    
    # 1. Rotate the point
    z_rot = z * rot
    # 2. Fold the point (Burning Ship behavior)
    z_abs = abs(z_rot.real) + 1j * abs(z_rot.imag)
    # 3. Apply anisotropic scaling
    z_scaled = z_abs.real * params.scale_x + 1j * z_abs.imag * scale_y
    # 4. Rotate back
    z_prime = z_scaled * irot
    
    return z_prime * z_prime + c
@njit(inline='always')
def gef_iter_quantized(z: complex, c: complex, cos_th: float, sin_th: float,
                         scale_x: float, k_flow_val: float,
                         genesis_target: float, genesis_strength: float) -> complex:
    """
    Performs one GEF iteration where anisotropy is dynamically coupled to the
    local gravity (κ-flow).
    """
    scale_y = genesis_target + genesis_strength * (k_flow_val - 1.0)
    scale_y = max(0.1, min(scale_y, 2.0)) # Clamp to sane values
    rot = cos_th + 1j * sin_th
    irot = cos_th - 1j * sin_th
    z_rot = z * rot
    z_abs = abs(z_rot.real) + 1j * abs(z_rot.imag)
    z_scaled = z_abs.real * scale_x + 1j * z_abs.imag * scale_y
    z_prime = z_scaled * irot
    return z_prime * z_prime + c

# ==============================================================================
# 2. RENDER ENGINES
# ==============================================================================
# ... (_render_fractal, _render_orbit, and _render_fractal_quantized are unchanged) ...
@njit(parallel=True)
def _render_fractal(width: int, height: int, x_min: float, x_max: float,
                    y_min: float, y_max: float, max_iter: int, cos_th: float,
                    sin_th: float, is_julia: bool, c_julia: complex,
                    iter_params: IteratorParams) -> np.ndarray:
    """Renders a GEF fractal (Mandelbrot or Julia) using integer escape-time."""
    img = np.zeros((height, width), dtype=np.uint16)
    dx = (x_max - x_min) / width
    dy = (y_max - y_min) / height

    for py in prange(height):
        y0 = y_min + py * dy
        for px in range(width):
            x0 = x_min + px * dx
            c = c_julia if is_julia else complex(x0, y0)
            z = complex(x0, y0) if is_julia else 0j
            
            for n in range(max_iter):
                if z.real**2 + z.imag**2 > 4.0:
                    img[py, px] = n
                    break
                z = gef_iter(z, c, cos_th, sin_th, iter_params)
    return img

@njit
def _render_orbit(c: complex, cos_th: float, sin_th: float, num_points: int,
                  transient_skip: int, iter_params: IteratorParams) -> np.ndarray:
    """Calculates the critical point orbit with robust escape/convergence checks."""
    orbit_points = np.zeros(num_points, dtype=np.complex128)
    z = 0j
    escape_radius_sq = 100.0

    for _ in range(transient_skip):
        z_prev = z
        z = gef_iter(z, c, cos_th, sin_th, iter_params)
        mag_sq = z.real**2 + z.imag**2
        if mag_sq > escape_radius_sq or not math.isfinite(mag_sq):
            return np.array([0j])
        if abs(z - z_prev) < 1e-12:
            break
    
    for i in range(num_points):
        z = gef_iter(z, c, cos_th, sin_th, iter_params)
        mag_sq = z.real**2 + z.imag**2
        if mag_sq > escape_radius_sq or not math.isfinite(mag_sq):
            return orbit_points[:i]
        orbit_points[i] = z
    return orbit_points
@njit(parallel=True)
def _render_fractal_quantized(width: int, height: int, x_min: float, x_max: float,
                              y_min: float, y_max: float, max_iter: int, cos_th: float,
                              sin_th: float, is_julia: bool, c_julia: complex,
                              scale_x: float, genesis_target: float,
                              genesis_strength: float, k_damping: float) -> np.ndarray:
    """
    Renders a fractal with co-evolving gravity. The final image shows the
    equilibrium state of the physical laws (the anisotropy field).
    """
    # Initialize grids
    img = np.zeros((height, width), dtype=np.uint16)      # Stores escape time
    z_grid = np.zeros((height, width), dtype=np.complex128)# Stores current z value
    k_flow_grid = np.ones((height, width), dtype=np.float64)# Stores local time rate (gravity)
    
    dx = (x_max - x_min) / width
    dy = (y_max - y_min) / height
    
    # Initialize z_grid based on mode
    for py in prange(height):
        y0 = y_min + py * dy
        for px in range(width):
            x0 = x_min + px * dx
            if not is_julia:
                z_grid[py, px] = 0j
            else:
                z_grid[py, px] = complex(x0, y0)

    # Main co-evolution loop
    for n in range(max_iter):
        active_points = 0
        for py in prange(height):
            y0 = y_min + py * dy
            for px in range(width):
                # Only update points that haven't escaped
                if img[py, px] == 0:
                    active_points += 1
                    z = z_grid[py, px]
                    
                    # Check for escape
                    if z.real**2 + z.imag**2 > 4.0:
                        img[py, px] = n
                        continue
                        
                    c = c_julia if is_julia else complex(x_min + px * dx, y_min + py * dy)
                    k_val = k_flow_grid[py, px]
                    
                    # Iterate z using the gravity-coupled law
                    z_new = gef_iter_quantized(z, c, cos_th, sin_th, scale_x, k_val,
                                               genesis_target, genesis_strength)
                    z_grid[py, px] = z_new

                    # Update local gravity (κ-flow) based on the "mass density" of z
                    mass_proxy = 1.0 / (1.0 + z_new.real**2 + z_new.imag**2)
                    target_k = 1.0 - mass_proxy 
                    
                    # Smoothly update the k_flow grid towards the target value
                    k_flow_grid[py, px] = k_damping * k_flow_grid[py, px] + (1.0 - k_damping) * target_k
        
        if active_points == 0:
            break
            
    final_law_grid = np.zeros_like(k_flow_grid)
    for py in prange(height):
        for px in range(width):
            final_law_grid[py, px] = genesis_target + genesis_strength * (k_flow_grid[py, px] - 1.0)

    return final_law_grid
# ==============================================================================
# 3. PLOTTING AND ANALYSIS
# ==============================================================================
# ... (These functions are unchanged and fully reusable) ...
def plot_and_save_frame(img_data: np.ndarray,
                        frame_path: Path,
                        title: str,
                        cmap: str,
                        dpi: int,
                        figsize: tuple,
                        extent: list = None,
                        log_norm: bool = False,
                        norm_type: str = "linear"):
    """A centralized utility for plotting and saving a single frame."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    
    if log_norm and img_data.max() > 0:
        norm = colors.LogNorm(vmin=1, vmax=img_data.max())
    elif norm_type == "relative":
        vmin, vmax = img_data.min(), img_data.max()
        norm = None if vmin == vmax else colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm_type == "percentile":
        if img_data.size > 0:
            vmin, vmax = np.percentile(img_data, [0.05, 99.95])
            norm = None if vmin == vmax else colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = None
    else:
        norm = None
    
    ax.imshow(img_data, cmap=cmap, extent=extent, origin='lower', norm=norm)
    ax.set_title(title, color="white", fontsize=16)
    ax.axis("off")
    fig.tight_layout(pad=0)
    try:
        plt.savefig(frame_path, facecolor="black", dpi=dpi)
    finally:
        plt.close(fig)

def calculate_fractal_metrics(frame_data: np.ndarray) -> Dict[str, float]:
    """Calculates metrics from a rendered fractal frame."""
    total_points = frame_data.size
    escaped_mask = frame_data > 0
    trapped_points = total_points - np.count_nonzero(escaped_mask)
    
    metrics = {'trapped_fraction': trapped_points / total_points}
    if escaped_mask.any():
        escaped_values = frame_data[escaped_mask]
        metrics['mean_escape_time'] = np.mean(escaped_values)
        metrics['escape_time_std'] = np.std(escaped_values)
    return metrics

def generate_area_curve_plot(thetas_pi: np.ndarray, area_fractions: list, output_dir: Path, config: Dict):
    """Generates a plot of trapped area vs. angle, highlighting transitions."""
    p_analysis = config['analysis']
    fig, ax = plt.subplots(figsize=tuple(p_analysis.get('area_curve_figsize', [10, 6])))
    ax.plot(thetas_pi, area_fractions, 'b-', linewidth=2, label='Trapped Area')
    ax.set_xlabel('Angle (θ/π)', fontsize=14)
    ax.set_ylabel('Trapped Area Fraction', fontsize=14)
    ax.set_title('Trapped Area vs. Rotation Angle', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    if len(area_fractions) > 1:
        area_diff = np.abs(np.diff(area_fractions))
        threshold = np.mean(area_diff) + 2 * np.std(area_diff)
        for idx in np.where(area_diff > threshold)[0]:
            ax.axvline(x=thetas_pi[idx], color='r', linestyle='--', alpha=0.7, label=f'Transition at {thetas_pi[idx]:.3f}π')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis_area_vs_angle.png", dpi=150)
    plt.close(fig)
# ==============================================================================
# 4. CONFIGURATION AND ORCHESTRATION
# ==============================================================================
def load_and_validate_config(config_path: Path) -> Dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at '{config_path}'")
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    required = ['mode', 'output_dir', 'image_size', 'fractal_window', 'max_iterations']
    for key in required:
        if key not in cfg: raise ValueError(f"Missing required config key: '{key}'")

    valid_modes = ['mandelbrot', 'julia', 'orbit', 'quantize-gravity']
    if cfg['mode'] not in valid_modes:
        raise ValueError(f"Invalid mode: '{cfg['mode']}'. Must be one of {valid_modes}")

    if 'sweep_parameters' not in cfg and cfg['mode'] in ['mandelbrot', 'julia', 'orbit']:
         raise ValueError(f"Mode '{cfg['mode']}' requires 'sweep_parameters' section.")
    
    if 'genesis_sweep_params' not in cfg and cfg['mode'] == 'quantize-gravity':
        raise ValueError("Mode 'quantize-gravity' requires 'genesis_sweep_params' section.")

    # ... (rest of validation is the same) ...
    if not (16 <= cfg['image_size'] <= 16384):
        raise ValueError(f"image_size is out of reasonable range (16-16384): {cfg['image_size']}")
    if len(cfg['fractal_window']) != 4:
        raise ValueError("fractal_window must have 4 values: [x_min, x_max, y_min, y_max]")
    
    if cfg['mode'] in ['julia', 'orbit'] and 'julia_set_c' not in cfg:
        raise ValueError(f"Mode '{cfg['mode']}' requires 'julia_set_c' parameter in config.")
        
    return cfg

def run_simulation(config: Dict):
    mode = config['mode']
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(config["output_dir"]) / f"{mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    win, p_plot = config['fractal_window'], config['plotting']
    
    # --- Dispatch to the correct simulation loop ---
    if mode == 'quantize-gravity':
        run_genesis_sweep(config, output_dir)
    else:
        run_standard_sweep(config, output_dir)

    print(f"\nSimulation complete.")

def run_genesis_sweep(config: Dict, output_dir: Path):
    """Runs an animated sweep for the quantize-gravity mode."""
    p_qg = config['quantize_gravity_params']
    p_sweep = config['genesis_sweep_params']
    
    param_to_sweep = p_sweep['parameter_to_sweep']
    sweep_values = np.linspace(p_sweep['start_value'], p_sweep['end_value'], p_sweep['num_frames'])
    
    # Base arguments for the renderer, to be updated in the loop
    render_kwargs = {
        "width": config["image_size"], "height": config["image_size"],
        "x_min": config["fractal_window"][0], "x_max": config["fractal_window"][1],
        "y_min": config["fractal_window"][2], "y_max": config["fractal_window"][3],
        "max_iter": config['max_iterations'],
        "is_julia": p_qg.get('is_julia', False),
        "c_julia": complex(*config.get('julia_set_c', [0,0])),
        "scale_x": config['anisotropy_scaling']['x'],
        "k_damping": p_qg['k_damping']
    }
    # Set fixed parameters
    render_kwargs['genesis_target'] = p_qg.get('genesis_target', 0.6)
    render_kwargs['genesis_strength'] = p_qg.get('genesis_strength', 0.5)

    th_pi = p_qg.get('fixed_angle_pi', 0.0)
    th_rad = th_pi * np.pi
    render_kwargs['cos_th'], render_kwargs['sin_th'] = math.cos(th_rad), math.sin(th_rad)

    desc = f"Rendering Genesis Sweep of '{param_to_sweep}'"
    for i, sweep_val in enumerate(tqdm(sweep_values, desc=desc)):
        # Update the parameter that is being swept for this frame
        render_kwargs[param_to_sweep] = sweep_val
        
        img_data = _render_fractal_quantized(**render_kwargs)
        
        title = f"Genesis State | {param_to_sweep} = {sweep_val:.4f}"
        
        plot_and_save_frame(
            img_data, output_dir / f"frame_{i:04d}.png", title,
            config['plotting']['cmap'], config['plotting']['dpi'],
            tuple(config['plotting']['animation_figsize']), config['fractal_window'],
            norm_type=config['plotting'].get('normalization', 'relative')
        )
        
def run_standard_sweep(config: Dict, output_dir: Path):
    """Runs the standard animated sweep for mandelbrot, julia, or orbit modes."""
    # ... (This is the original simulation loop, now in its own function) ...
    mode = config['mode']
    thetas_pi = np.linspace(
        config['sweep_parameters']['start_angle_pi'], 
        config['sweep_parameters']['end_angle_pi'],
        config['sweep_parameters']['num_frames'], endpoint=True)
    
    iter_params = setup_iterator_params(config)
    win, p_plot = config['fractal_window'], config['plotting']
    p_analysis = config.get('analysis', {})
    do_area_curve = p_analysis.get('generate_area_curve', False)
    area_fractions = [] if do_area_curve else None
    norm_type = p_plot.get('normalization', 'linear')

    desc = f"Rendering {mode.capitalize()} Frames"
    for i, th_pi in enumerate(tqdm(thetas_pi, desc=desc)):
        th_rad, cos_th, sin_th = th_pi * np.pi, math.cos(th_pi * np.pi), math.sin(th_pi * np.pi)
        title = f"θ = {th_pi:.4f}π"
        
        if mode in ['mandelbrot', 'julia']:
            c_julia = complex(*config.get('julia_set_c', [0, 0]))
            img_data = _render_fractal(config["image_size"], config["image_size"], *win,
                                       config['max_iterations'], cos_th, sin_th,
                                       (mode == 'julia'), c_julia, iter_params)
            if mode == 'julia': title += f" | c = {c_julia.real:.2f}{c_julia.imag:+.2f}i"
            if do_area_curve: area_fractions.append(calculate_fractal_metrics(img_data)['trapped_fraction'])
        elif mode == 'orbit':
            p_orbit = config['orbit_render']
            c_julia = complex(*config['julia_set_c'])
            title += f" | c = {c_julia.real:.2f}{c_julia.imag:+.2f}i"
            orbit_data = _render_orbit(c_julia, cos_th, sin_th, p_orbit['num_points'],
                                       p_orbit['transient_skip'], iter_params)
            if orbit_data.size > 1:
                bins = p_orbit['resolution'] if isinstance(p_orbit['resolution'], int) else tuple(p_orbit['resolution'])
                hist, _, _ = np.histogram2d(orbit_data.real, orbit_data.imag, bins=bins, range=[win[:2], win[2:]])
                img_data = hist.T
            else:
                img_data = np.zeros((config['image_size'], config['image_size']))
        
        plot_and_save_frame(img_data, output_dir / f"frame_{i:04d}.png", title,
                            p_plot['cmap'], p_plot['dpi'], tuple(p_plot['animation_figsize']),
                            win, log_norm=(mode == 'orbit'), norm_type=norm_type)

    if do_area_curve and area_fractions:
        print("\nGenerating Trapped Area vs. Angle plot...")
        generate_area_curve_plot(thetas_pi, area_fractions, output_dir, config)

# ... (main function is identical to the previous version) ...
def main():
    """Main entry point: parses args, loads config, and runs the simulation."""
    script_dir = Path(__file__).parent.absolute()
    default_config = script_dir / "configs" / "default_fractal_spelunker_v2.yml"
    
    parser = argparse.ArgumentParser(description="GEF Fractal Spelunker.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', '-c', type=Path, default=default_config,
                        help=f"Path to the YAML configuration file.\nDefault: {default_config}")
    args = parser.parse_args()
    
    try:
        config = load_and_validate_config(args.config)
        run_simulation(config)
    except (ValueError, FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"\nConfiguration or Input Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()