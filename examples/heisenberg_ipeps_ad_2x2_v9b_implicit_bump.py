#!/usr/bin/env python3
"""v9b: 2-site bipartite implicit AD + in-CTM chi-bump (post-#516 chi-lock).

Configuration: ``gs_c4v=False`` + ``gs_implicit_ad=True`` (implicit AD
via custom_vjp + fixed-point adjoint), ``chi_initial=9``, ``chi_max=24``,
``chi_auto_bump=True``, no schedule.

Acceptance gate (per docs/plans/2026-05-20-chi-lock-design.md Section
"Bench-level acceptance gate"):
- wall-clock <= v7b (8h52m at fixed chi=24)
- energy within numerical tolerance of v7b (~-0.6225)
- NOT a QMC-parity gate -- fixed chi remains the production protocol;
  this run only verifies the bump saves cycles without changing the
  fixed-point.

QMC reference: E/site ~ -0.6694 (informational only -- implicit AD at
chi=24 plateaus at -0.6225 per v7b).

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python examples/heisenberg_ipeps_ad_2x2_v9b_implicit_bump.py
"""

from __future__ import annotations

import logging
import time

import jax

jax.config.update("jax_enable_x64", True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
# Uncomment for chi-lock diagnostics:
# logging.getLogger("tenax.ctm.gmres").setLevel(logging.DEBUG)

from tenax import (  # noqa: E402
    CTMConfig,
    aligned_ctm_schedules,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
)


def main() -> None:
    print("iPEPS-AD v9b: bipartite implicit AD + in-CTM chi-bump (#516)")
    print("H = J sum_<ij> S_i . S_j,   J = 1")
    print("QMC reference: E/site ~ -0.6694 (informational only)")
    print("v7b baseline:  E/site ~ -0.6225 at fixed chi=24 in 8h52m\n")

    H = heisenberg_gate()

    conv_tol_schedule, patience_schedule = aligned_ctm_schedules(
        [(0.0, 1e-5, 20), (0.7, 1e-7, None)]
    )

    D = 3
    config = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=300,
        dt=0.05,
        unit_cell="2site",
        gs_num_steps=80,
        ctm=CTMConfig(
            chi=9,
            max_iter=75,
            min_iter=10,
            conv_tol=1e-5,
            projector_method="svd",
            forward_gauge="phase",
            adjoint_method="fixed_point",
            gmres_tol=1e-7,
            gmres_maxiter=75,
            chi_auto_bump=True,
            chi_auto_bump_eps=1e-6,
            chi_auto_bump_step=2,
            chi_max=24,
        ),
        gs_c4v=False,
        gs_implicit_ad=True,  # implicit AD via custom_vjp
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_line_search_max_steps=40,
        gs_metric_precond=True,
        metric_gmres_maxiter=3,
        gs_ctm_conv_tol_schedule=conv_tol_schedule,
        gs_plateau_patience_schedule=patience_schedule,
        gs_conv_tol=1e-5,
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e-5,
        gs_stall_recovery="reset",
        gs_noise_amplitude=1e-4,
        gs_noise_recovery_retries=5,
        su_init=True,
        gs_verbose=True,
        gs_log_interval=5,
    )

    t0 = time.perf_counter()
    (A_opt, B_opt), envs, E_gs = optimize_gs_ad(H, None, config)
    dt = time.perf_counter() - t0

    print(f"\n  D       = {D}")
    print("  chi_max = 24 (initial=9, bump active)")
    print(f"  E/site  = {E_gs:.8f}")
    print(f"  time    = {dt:.1f}s ({dt / 3600:.2f}h)")
    v7b_wallclock_s = 8 * 3600 + 52 * 60
    print(
        f"  wall-clock gate: {'PASS' if dt <= v7b_wallclock_s else 'FAIL'} (v7b 8h52m)"
    )


if __name__ == "__main__":
    main()
