# experiments/scripts/prime_isomorphism/anisotropic_sieve.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def generate_ulam_grid(size):
    """Generates a grid of numbers in a spiral."""
    if size % 2 == 0:
        size += 1  # Ensure odd size for a clear center
    
    grid = np.zeros((size, size), dtype=int)
    x, y = size // 2, size // 2  # Start at the center
    grid[y, x] = 1  # Place 1 at the center
    
    # Direction vectors: right, up, left, down
    dx, dy = 1, 0
    num = 2  # Start with 2 since 1 is already placed
    
    # Variables to control the spiral pattern
    steps = 1  # How many steps in the current direction
    step_count = 0  # Counter for steps taken in current direction
    direction_changes = 0  # Counter for direction changes
    
    while num <= size * size:
        # Move in the current direction
        x += dx
        y += dy
        
        # Check if we're still within bounds
        if 0 <= x < size and 0 <= y < size:
            grid[y, x] = num
            num += 1
        else:
            # If we went out of bounds, step back and break
            x -= dx
            y -= dy
            break
        
        # Increment step counter
        step_count += 1
        
        # Check if we need to change direction
        if step_count == steps:
            # Reset step counter
            step_count = 0
            
            # Change direction (right->up->left->down->right...)
            dx, dy = -dy, dx
            
            # Increment direction change counter
            direction_changes += 1
            
            # Every two direction changes, increase the step length
            if direction_changes % 2 == 0:
                steps += 1
    
    return grid

def sieve_on_grid(grid):
    """Applies the Sieve of Eratosthenes to a grid of numbers."""
    max_val = grid.max()
    is_prime = np.ones(max_val + 1, dtype=bool)
    is_prime[:2] = False
    
    for p in range(2, int(max_val**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p : max_val+1 : p] = False
            
    # Create a boolean mask of primes for the grid shape
    prime_mask = is_prime[grid]
    return prime_mask

def apply_gef_anisotropy(grid_shape):
    """Creates a diagonal weighting mask mimicking the GEF κ-flow."""
    size = grid_shape[0]
    center = size // 2
    x, y = np.ogrid[-center:center+1, -center:center+1]
    
    # This creates a "cross" of higher probability along the diagonals
    # The strength of the anisotropy can be tuned with the 'bias_strength' param
    bias_strength = 0.7 
    diagonal_bias = np.exp(-((x - y)**2) / (2 * (size/6)**2))
    anti_diagonal_bias = np.exp(-((x + y)**2) / (2 * (size/6)**2))
    
    # Combine the two diagonal masks
    anisotropy_mask = np.maximum(diagonal_bias, anti_diagonal_bias)
    
    # Normalize and apply strength
    anisotropy_mask = anisotropy_mask / anisotropy_mask.max()
    
    # We want a mask that enhances, not replaces. So it should be close to 1.
    return 1.0 + bias_strength * anisotropy_mask

def generate_gef_field(size, prime_mask, anisotropy_field, seed=None):
    """Generate a single GEF field with the given seed."""
    if seed is not None:
        np.random.seed(seed)
    
    # Create a random noise field to represent the "quantum foam"
    random_field = np.random.rand(size, size)
    
    # The final "GEF field" is the product of all three influences:
    # 1. Is it a prime number location? (Binary: 0 or 1)
    # 2. What is the GEF anisotropic bias here? (Continuous: 1.0 to 1.7)
    # 3. What is the random local fluctuation? (Continuous: 0.0 to 1.0)
    gef_prime_values = prime_mask * anisotropy_field * random_field
    
    return gef_prime_values

def plot_spirals(size=201, save_output=True, num_iterations=200):
    """Generates and plots the standard Ulam spiral and the GEF-enhanced version.
    
    Args:
        size: Size of the grid (will be forced to odd number)
        save_output: Whether to save the plots to files
        num_iterations: Number of random seeds to aggregate for the enhanced visualization
    """
    
    print("Generating number grid...")
    number_grid = generate_ulam_grid(size)
    print(f"  Grid shape: {number_grid.shape}, Max value: {number_grid.max()}")
    
    print("Sieving for primes...")
    prime_mask = sieve_on_grid(number_grid)
    prime_count = np.sum(prime_mask)
    print(f"  Found {prime_count} primes in the grid ({prime_count/size**2*100:.2f}% of cells)")
    
    print("Creating GEF anisotropic bias field...")
    anisotropy_field = apply_gef_anisotropy(number_grid.shape)
    print(f"  Anisotropy field range: {anisotropy_field.min():.2f} to {anisotropy_field.max():.2f}")
    
    # Generate multiple GEF fields with different random seeds
    print(f"Generating {num_iterations} GEF fields with different random seeds...")
    gef_fields = []
    
    for i in range(num_iterations):
        print(f"  Generating field {i+1}/{num_iterations} with seed {i}")
        gef_field = generate_gef_field(size, prime_mask, anisotropy_field, seed=i)
        gef_fields.append(gef_field)
    
    # Aggregate the fields
    print("Aggregating GEF fields...")
    master_grid = np.sum(gef_fields, axis=0)
    
    # Normalize the master grid for better visualization
    master_grid = master_grid / num_iterations
    
    print(f"Master grid range: {master_grid.min():.4f} to {master_grid.max():.4f}")
    
    # --- Plotting ---
    print("Creating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
    
    # Plot 1: Standard Ulam Spiral
    ax1 = axes[0]
    ax1.imshow(prime_mask, cmap='gray_r', interpolation='nearest')
    ax1.set_title("Standard Ulam Spiral\n(Primes Only)", color='white')
    ax1.axis('off')

    # Plot 2: Single GEF-Enhanced Spiral (first seed)
    ax2 = axes[1]
    single_gef = gef_fields[0]
    ax2.imshow(single_gef, cmap='inferno', interpolation='nearest')
    ax2.set_title("Single GEF-Enhanced Spiral\n(One Random Seed)", color='white')
    ax2.axis('off')
    
    # Plot 3: Aggregated GEF-Enhanced Spiral
    ax3 = axes[2]
    master_plot = ax3.imshow(master_grid, cmap='viridis', interpolation='nearest')
    ax3.set_title(f"Aggregated GEF-Enhanced Spiral\n({num_iterations} Random Seeds)", color='white')
    ax3.axis('off')
    
    fig.tight_layout()
    
    # Save the plots to files
    if save_output:
        import os
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the plots
        standard_path = os.path.join(output_dir, f'ulam_spiral_{size}_{timestamp}.png')
        gef_path = os.path.join(output_dir, f'gef_spiral_single_{size}_{timestamp}.png')
        master_path = os.path.join(output_dir, f'gef_spiral_aggregated_{num_iterations}_{size}_{timestamp}.png')
        
        print(f"Saving standard Ulam spiral to: {standard_path}")
        plt.figure(figsize=(10, 10), facecolor='black')
        plt.imshow(prime_mask, cmap='gray_r', interpolation='nearest')
        plt.title("Standard Ulam Spiral", color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(standard_path, facecolor='black', bbox_inches='tight')
        
        print(f"Saving single GEF-enhanced spiral to: {gef_path}")
        plt.figure(figsize=(10, 10), facecolor='black')
        plt.imshow(single_gef, cmap='inferno', interpolation='nearest')
        plt.title("Single GEF-Enhanced Spiral", color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(gef_path, facecolor='black', bbox_inches='tight')
        
        print(f"Saving aggregated GEF-enhanced spiral to: {master_path}")
        plt.figure(figsize=(10, 10), facecolor='black')
        plt.imshow(master_grid, cmap='viridis', interpolation='nearest')
        plt.title(f"Aggregated GEF-Enhanced Spiral ({num_iterations} Seeds)", color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(master_path, facecolor='black', bbox_inches='tight')
    
    # Try to show the plot interactively
    print("Attempting to display plots interactively...")
    try:
        plt.show()
        print("Interactive display successful")
    except Exception as e:
        print(f"Note: Could not display plots interactively: {e}")
        print("But don't worry, the plots were saved to files.")
    
    return prime_mask, single_gef, master_grid

if __name__ == "__main__":
    # You can change the size to see the effect at different resolutions
    # Larger sizes will show clearer patterns but take longer to compute.
    # 201 is a good starting point. 401 is even better.
    plot_spirals(size=401)