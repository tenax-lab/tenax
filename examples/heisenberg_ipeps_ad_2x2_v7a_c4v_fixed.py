#!/usr/bin/env python3
"""v7a: 2-site C4v at fixed chi=24 (peps-torch / YASTN pattern).

No chi_schedule, no chi_auto_bump — fixed chi=24 from step 1. Compares
to v6 (C4v + chi_schedule + auto_bump, E=-0.66498121) to isolate the
cost of the chi=8 warm-up stage on the C4v ansatz.

QMC reference: E/site ≈ −0.6694.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python examples/heisenberg_ipeps_ad_2x2_v7a_c4v_fixed.py
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
    print("iPEPS-AD v7a: C4v fixed chi=24 (no schedule, no auto-bump)")
    print("H = J sum_<ij> S_i . S_j,   J = 1")
    print("QMC reference: E/site ~ -0.6694\n")

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
            chi=24,  # fixed chi
            max_iter=75,
            min_iter=10,
            conv_tol=1e-5,
            projector_method="svd",
            forward_gauge="phase",
            adjoint_method="fixed_point",
            gmres_tol=1e-7,
            gmres_maxiter=75,
            chi_auto_bump=False,  # off — fixed-chi pattern
        ),
        gs_c4v=True,
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


if __name__ == "__main__":
    main()
