import argparse
import yaml

def check_parameters(mu2, P, lam):
    """Calculates the vacuum phi0 and checks for stability."""
    if lam <= 0:
        print("❌ ERROR: lambda_val must be positive.")
        return

    phi0_sq = (mu2 - 2 * P) / lam if (mu2 > 2 * P) else 0.0

    print(f"--- Vacuum Stability Check ---")
    print(f"  mu_squared: {mu2}")
    print(f"  P_env:      {P}")
    print(f"  lambda_val: {lam}")
    print("-" * 30)
    
    if phi0_sq < 0:
        phi0_sq = 0.0

    phi0 = phi0_sq**0.5
    print(f"  Calculated phi0_sq = {phi0_sq:.4f}")
    print(f"  Calculated |phi0| = {phi0:.4f}")

    if phi0 > 1.0:
        print(f"⚠️  WARNING: |phi0| > 1. The Hook term is unstable.")
    else:
        print(f"✅ OK: |phi0| <= 1. The Hook term is stable.")

def main():
    parser = argparse.ArgumentParser(description="Check GEF vacuum stability for a set of parameters.")
    parser.add_argument("-c", "--config", type=argparse.FileType('r'), help="Path to a solver YAML config file.")
    parser.add_argument("--mu2", type=float, help="Value for mu_squared.")
    parser.add_argument("--p", type=float, help="Value for P_env.")
    parser.add_argument("--lam", type=float, help="Value for lambda_val.")
    args = parser.parse_args()

    if args.config:
        config = yaml.safe_load(args.config)
        solver_cfg = config['solver']
        mu2 = solver_cfg['mu_squared']
        P = solver_cfg['P_env']
        lam = solver_cfg['lambda_val']
    elif all([args.mu2, args.p, args.lam]):
        mu2 = args.mu2
        P = args.p
        lam = args.lam
    else:
        parser.error("Please provide either a config file (-c) or all three parameters (--mu2, --p, --lam).")
        return

    check_parameters(mu2, P, lam)

if __name__ == "__main__":
    main()