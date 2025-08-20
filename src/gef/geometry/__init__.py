        # # --- Parameter sanity checks ---
        # if self.lam <= 0.0:
        #     raise ValueError("lambda_val must be positive for a bounded-from-below Mexican hat.")

        # # If Hook term is active, prefer a vacuum within |phi| <= 1
        # if self.h_sq > 0.0:
        #     phi0_sq = max(0.0, (self.mu2 - 2.0 * self.P_env) / self.lam)
        #     if phi0_sq > 1.0:
        #         logger.warning(
        #             "Hook term active (h^2>0) but vacuum |phi0|=%.3f > 1. "
        #             "This makes U_hook negative (unbounded) for large gradients. "
        #             "Consider increasing lambda_val or P_env, or reducing mu_squared.",
        #             np.sqrt(phi0_sq),
        #         )

self.config.setdefault("energy_rel_tol", 1e-5)
self.config.setdefault("max_update_tol", 5e-5)
self.config.setdefault("convergence_warmup_iters", 500)