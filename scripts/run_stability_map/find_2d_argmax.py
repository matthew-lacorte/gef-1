# save as scripts/run_stability_map/analysis/find_2d_argmax.py
import sys, pandas as pd
p = sys.argv[1]  # path to stability_results.csv
df = pd.read_csv(p, header=None)
# Column meanings (as emitted): i, j, P_env, mu_squared, E_initial, E_final, E0_numeric, E0_analytic,
# mass_density, converged, stability_metric, phi_max, phi_rms, max_delta_phi, last_max_update, Lw,
# core_frac, E_kin, E_iso, E_aniso, E_hook, E_press
df.columns = ["i","j","P_env","mu2","E_initial","E_final","E0_numeric","E0_analytic",
              "mass_density","converged","stability_metric","phi_max","phi_rms",
              "max_delta_phi","last_max_update","Lw","core_frac","E_kin","E_iso",
              "E_aniso","E_hook","E_press"]
dfc = df[df["converged"]==True]
row = dfc.loc[dfc["stability_metric"].idxmax()]
print(f"P_STAR={row.P_env:.6f}")
print(f"MU2_STAR={row.mu2:.6f}")
print("Argmax row:\n", row.to_string())