# experiments/scripts/prime_isomorphism/borwein_stabilization.py

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize # Use a more powerful minimizer for 2D problems
import matplotlib.pyplot as plt

# --- 1. Define the Core Functions ---

def borwein_integrand_v2(x, n_terms, a, b):
    """
    Calculates the value of the Borwein integrand with a "Mexican Hat" stabilizer.
    Stabilizer S(x) = (1 - a*x^2) * exp(-b*x^2)
    """
    if x == 0.0:
        # sinc(0)=1, exp(0)=1, (1-0)=1. Result is 1.
        return 1.0
    
    product = 1.0
    for k in range(n_terms):
        denominator = 2 * k + 1
        product *= np.sinc(x / (np.pi * denominator)) # Use np.sinc for stability
        
    # Apply the new, more sophisticated GEF stabilization factor
    stabilization_factor = (1 - a * x**2) * np.exp(-b * x**2)
    
    return product * stabilization_factor

def calculate_integral_value_v2(n_terms, a, b):
    """Numerically calculates the integral with the new stabilizer."""
    # Note: quad is very good but can be slow for oscillating functions.
    # We might need to increase the limit for a robust result.
    result, error = quad(borwein_integrand_v2, -100, 100, args=(n_terms, a, b), limit=200)
    return result

# --- 2. The Objective Function for 2D Optimization ---

def objective_function_v2(params, n_terms_to_fix):
    """
    This is the function we want to MINIMIZE.
    We want the squared difference between our integral and pi to be zero.
    
    Args:
        params (list): A list [a, b] of the parameters we are optimizing.
        n_terms_to_fix (int): The number of terms for the "broken" integral.
    """
    a, b = params
    
    # We must constrain 'b' to be positive for the Gaussian to be a damper.
    if b <= 0:
        return 1e12 # Return a large penalty if 'b' is invalid

    integral_value = calculate_integral_value_v2(n_terms_to_fix, a, b)
    
    # We want to minimize the squared error.
    error_squared = (integral_value - np.pi)**2
    
    # Print progress during optimization
    print(f"  Trying [a={a:.6f}, b={b:.6f}] -> Integral={integral_value:.9f}, Err^2={error_squared:.2e}")
    
    return error_squared

# --- 3. The Main Execution Block ---

if __name__ == "__main__":
    # --- Configuration ---
    BROKEN_N_TERMS = 8  # Corresponds to I_15

    # --- Step 1: Verify the "Broken" Integral ---
    unstable_value = calculate_integral_value_v2(BROKEN_N_TERMS, a=0.0, b=0.0)
    print("Verifying the 'broken' integral (I_15)...")
    print(f"  Value of I_15 without stabilization: {unstable_value:.15f}")
    print(f"  Target value (pi):               {np.pi:.15f}")
    print(f"  Initial Error^2:                 {(unstable_value - np.pi)**2:.2e}")
    print("-" * 50)

    # --- Step 2: Find the GEF Stabilization Parameters [a, b] ---
    print(f"Searching for stabilization parameters [a, b] for N={BROKEN_N_TERMS} terms...")
    
    # Initial guess for the parameters [a, b]
    initial_guess = [0.001, 0.001]
    
    # Use a powerful optimization algorithm like Nelder-Mead or BFGS
    result = minimize(
        objective_function_v2,
        initial_guess,
        args=(BROKEN_N_TERMS,),
        method='Nelder-Mead', # Good for noisy or complex functions
        options={'xatol': 1e-12, 'fatol': 1e-12, 'disp': True}
    )

    print("-" * 50)
    if result.success:
        solution_a, solution_b = result.x
        print(f"SUCCESS: Optimizer converged on a solution!")
        print(f"  Found parameters: a = {solution_a:.15f}")
        print(f"                    b = {solution_b:.15f}")
        
        # --- Step 3: Verify the "Fixed" Integral with the solution ---
        print("\nVerifying the integral with the found stabilizer...")
        fixed_value = calculate_integral_value_v2(BROKEN_N_TERMS, solution_a, solution_b)
        print(f"  Value of stabilized I_15: {fixed_value:.15f}")
        print(f"  Value of pi:              {np.pi:.15f}")
        print(f"  Final Error^2:            {(fixed_value - np.pi)**2:.2e}")
        
        # --- Step 4: Visualize the Stabilizer Function ---
        x_vals = np.linspace(-30, 30, 1000)
        stabilizer_vals = (1 - solution_a * x_vals**2) * np.exp(-solution_b * x_vals**2)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, stabilizer_vals, label=f'S(x) with a={solution_a:.4f}, b={solution_b:.4f}')
        plt.axhline(1, color='r', linestyle='--', label='y = 1 (No effect)')
        plt.title("GEF 'Mexican Hat' Stabilization Field k(x)")
        plt.xlabel("x")
        plt.ylabel("Stabilization Factor S(x)")
        plt.grid(True)
        plt.legend()
        plt.ylim(0.9, 1.1) # Zoom in to see the effect
        plt.show()

    else:
        print("FAILURE: The optimizer did not converge.")
        print(f"  Message: {result.message}")