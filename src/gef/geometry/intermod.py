import numpy as np
import matplotlib.pyplot as plt

def generate_intermodulation_chart():
    """
    Generates a theoretical chart predicting the power of GEF intermodulation
    products against standard equipment artifacts.
    """
    # Define the x-axis: Input Power in dBm (decibels relative to 1 milliwatt)
    input_power_dbm = np.linspace(-30, 10, 100) # From very low to high power

    # --- 1. Fundamental Signal ---
    # The output power of the fundamental signal (at f1 or f2) has a slope of 1.
    # P_out = P_in + gain. We'll assume gain = 0 for simplicity.
    fundamental_out_dbm = input_power_dbm

    # --- 2. Standard Equipment Intermodulation (IMD3) ---
    # Third-order intermodulation is the most common artifact in real RF gear.
    # Its power increases with a slope of 3. We need to define the "Third-Order
    # Intercept Point" (IP3), a standard measure of a device's linearity.
    # A typical high-end transmitter might have an IP3 of +40 dBm.
    ip3_dbm = 40
    imd3_out_dbm = 3 * input_power_dbm - 2 * ip3_dbm

    # --- 3. Predicted GEF Intermodulation (IMD2) ---
    # Our hypothesis is that GEF creates a second-order non-linearity.
    # Its power increases with a slope of 2. We need to hypothesize a value
    # for its intercept point, IP2. Since this is new physics, this value is
    # unknown, but we expect it to be very high (meaning the effect is very weak).
    ip2_dbm_hypothesis = 80 # Let's hypothesize a very high IP2 of +80 dBm
    imd2_gef_out_dbm = 2 * input_power_dbm - ip2_dbm_hypothesis
    
    # --- 4. The Noise Floor ---
    # The sensitivity limit of your spectrum analyzer.
    # A good lab analyzer might have a noise floor around -110 dBm with a narrow RBW.
    noise_floor_dbm = -110

    # --- 5. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(input_power_dbm, fundamental_out_dbm, 'b', lw=2.5, label='Fundamental Signal (Slope=1)')
    ax.plot(input_power_dbm, imd3_out_dbm, 'r--', lw=2, label='Typical Equipment IMD3 (Slope=3)')
    ax.plot(input_power_dbm, imd2_gef_out_dbm, 'g-', lw=3, alpha=0.8, label='Predicted GEF Signal (IMD2, Slope=2)')
    
    ax.axhline(noise_floor_dbm, color='k', ls=':', lw=2, label=f'Analyzer Noise Floor ({noise_floor_dbm} dBm)')

    # --- Finding the "Discovery Window" ---
    # Find the intersection where the GEF signal rises above the noise floor
    # but is still below the equipment's own artifacts.
    discovery_mask = (imd2_gef_out_dbm > noise_floor_dbm) & (imd2_gef_out_dbm < imd3_out_dbm)
    if np.any(discovery_mask):
        ax.fill_between(input_power_dbm, imd2_gef_out_dbm, noise_floor_dbm, 
                        where=discovery_mask, color='limegreen', alpha=0.3, label='Discovery Window')

    ax.set_xlabel('Input Power per Tone (dBm)', fontsize=14)
    ax.set_ylabel('Output Power (dBm)', fontsize=14)
    ax.set_title('Predictive GEF Intermodulation Chart', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_ylim(-130, 20)
    ax.set_xlim(-30, 10)
    ax.grid(True, which='both', linestyle='--')

    plt.show()

if __name__ == '__main__':
    generate_intermodulation_chart()