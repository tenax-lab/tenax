#!/usr/bin/env python3
"""v7c: 2-site bipartite explicit AD at fixed chi=24 (#328 probe).

No schedule, no auto-bump. Tests whether the bipartite-explicit-AD drift
(issue #328) is purely an intermediate-chi effect (would be rescued
here because chi is always 24, never below) or whether explicit AD
drifts even at chi=24.

If E stays above QMC (-0.6694): drift was intermediate-chi, fix is to
start at variational chi (or in-CTM auto-bump per #492).
If E still drifts below QMC: bug is in the explicit backward pass.

QMC reference: E/site ≈ −0.6694.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python examples/heisenberg_ipeps_ad_2x2_v7c_explicit_fixed.py
"""

from __future__ import annotations

import logging
import time

import jax

jax.config.update("jax_enable_x64", True)

logging.basicConfig(level=logging.INFO, format="%(message)s")

from tenax import (  # noqa: E402
    CTMConfig,
    aligned_ctm_schedules,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
)


def main():
    print("iPEPS-AD v7c: bipartite explicit AD fixed chi=24 (#328 probe)")
    print("H = J sum_<ij> S_i . S_j,   J = 1")
    print("QMC reference: E/site ~ -0.6694 (drift signature: E < -0.6694)\n")

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
            chi=24,
            max_iter=75,
            min_iter=10,
            conv_tol=1e-5,
            projector_method="svd",
            forward_gauge="phase",
            adjoint_method="fixed_point",
            gmres_tol=1e-7,
            gmres_maxiter=75,
            chi_auto_bump=False,
        ),
        gs_c4v=False,
        gs_implicit_ad=False,  # explicit AD through unrolled CTM
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
    print("  chi     = 24 (fixed)")
    print(f"  E/site  = {E_gs:.8f}")
    print(f"  time    = {dt:.1f}s")
    print(f"  drift   = {'YES (E < QMC)' if E_gs < -0.6694 else 'NO'}")


if __name__ == "__main__":
    main()
