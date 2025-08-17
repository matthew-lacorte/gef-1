# experiments/scripts/prime_isomorphism/benfords_law/benfords_law.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import yaml
from pathlib import Path
import datetime

def get_first_digit(n):
    """Returns the first significant digit of a number."""
    if n <= 0: return None
    while n < 1: n *= 10
    return int(str(n)[0])

def analyze_and_plot_benford(ax, data, title, config):
    """Analyzes and plots the first-digit distribution of a dataset."""
    positive_data = data[data > 0]
    if len(positive_data) == 0:
        ax.text(0.5, 0.5, "No positive data to analyze.", ha='center', va='center')
        ax.set_title(title)
        return

    first_digits = [d for d in [get_first_digit(n) for n in positive_data] if d is not None]
    
    digit_counts = np.bincount(first_digits, minlength=10)[1:10]
    observed_proportions = digit_counts / np.sum(digit_counts)
    
    digits = np.arange(1, 10)
    benford_proportions = np.log10(1 + 1 / digits)
    
    expected_counts = benford_proportions * np.sum(digit_counts)
    
    # Handle cases where an expected count is zero to avoid division errors in chi-squared
    # This is good practice although unlikely with Benford's law.
    non_zero_mask = expected_counts > 0
    if not np.all(non_zero_mask):
        print(f"Warning: Zero expected counts found in '{title}'. Chi-squared test may be unreliable.")

    chi2_stat, p_value = chisquare(
        f_obs=digit_counts[non_zero_mask], 
        f_exp=expected_counts[non_zero_mask]
    )
    
    # Plotting
    ax.bar(digits, observed_proportions, label='Observed Data', zorder=2)
    ax.plot(digits, benford_proportions, 'ro-', lw=2, label="Benford's Law Prediction", zorder=3)
    ax.set_title(title, fontsize=config['plotting']['title_fontsize'])
    ax.set_xlabel("Leading Digit", fontsize=config['plotting']['label_fontsize'])
    ax.set_ylabel("Proportion", fontsize=config['plotting']['label_fontsize'])
    ax.legend(fontsize=config['plotting']['legend_fontsize'])
    ax.grid(True, linestyle='--', alpha=0.6)
    
    text_str = f"Chi-squared test:\np-value = {p_value:.4f}"
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

def run_benford_analysis(config_path):
    """Main function to run the full analysis based on a config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 1. Load or Generate Data ---
    if config['input_data']['generate_fake_data']:
        print("Generating placeholder log-uniform data for GEF energies...")
        num_points = config['input_data']['num_fake_points']
        # This simulates a process with multiplicative noise spanning many orders of magnitude
        gef_energies = np.exp(np.random.uniform(np.log(1e-8), np.log(1e8), num_points))
    else:
        print(f"Loading real GEF energy data from: {config['input_data']['real_data_path']}")
        data_path = Path(config['input_data']['real_data_path'])
        if not data_path.is_file():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        gef_energies = np.load(data_path).flatten() # Flatten in case it's a 2D map

    # Generate control group data
    print("Generating uniform random data for control group...")
    cg_params = config['control_group']
    control_data = np.random.uniform(
        cg_params['low_bound'], 
        cg_params['high_bound'], 
        cg_params['num_points']
    )

    # --- 2. Create Plots ---
    print("Creating plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=config['plotting']['figsize'])
    
    analyze_and_plot_benford(
        axes[0],
        data=np.abs(gef_energies),
        title="First-Digit Distribution of GEF Stable Energies",
        config=config
    )
    
    analyze_and_plot_benford(
        axes[1],
        data=control_data,
        title="First-Digit Distribution of Uniform Random Data (Control)",
        config=config
    )
    
    fig.tight_layout(pad=3.0)
    fig.suptitle("GEF Isomorphism Test: Benford's Law Analysis", fontsize=18, weight='bold')
    
    # --- 3. Save Output ---
    output_path = Path(config['output_dir']) / config['plot_filename']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=config['plotting']['dpi'], bbox_inches='tight')
    print(f"\nAnalysis complete. Plot saved to: {output_path}")
    plt.show()

if __name__ == '__main__':
    config_file = 'scripts/fractals/benfords_law/config/benford_analysis_config.yml'
    run_benford_analysis(config_file)