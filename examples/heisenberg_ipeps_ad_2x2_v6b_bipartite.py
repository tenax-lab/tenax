#!/usr/bin/env python3
"""v6b: v6 compose benchmark with gs_c4v=False (bipartite 2-site).

Identical to ``heisenberg_ipeps_ad_2x2_v6_compose.py`` except for
``gs_c4v=False`` — A and B are two independent site tensors, not B
derived from A via sublattice rotation. Same implicit-AD path
(``adjoint_method='fixed_point'`` + default ``gs_implicit_ad=True``).

Goal: apples-to-apples vs variPEPS bipartite reference (D=3, χ=24,
E=-0.668168 from 2026-05-13). v6 C4v reached E=-0.66498121 — variPEPS
is 3.2×10⁻³ lower. This run isolates whether the gap is from the C4v
constraint or from optimizer/stall handling (v6 hit 5/5 stalls; variPEPS
ran 68+ steps stall-free).

QMC reference: E/site ≈ −0.6694.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python examples/heisenberg_ipeps_ad_2x2_v6b_bipartite.py
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
    optimize_gs_ad_chi_schedule,
)


def main():
    print("iPEPS-AD v6b: bipartite (gs_c4v=False) compose")
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
        ctm=CTMConfig(
            chi=16,  # start at chi=16: bipartite drifts below QMC at chi<16
            max_iter=75,
            min_iter=10,
            conv_tol=1e-5,
            projector_method="svd",
            forward_gauge="phase",
            adjoint_method="fixed_point",
            gmres_tol=1e-7,
            gmres_maxiter=75,
            chi_auto_bump=True,
            chi_auto_bump_eps=1e-5,
            chi_auto_bump_step=2,
        ),
        gs_c4v=False,  # v6b: bipartite — A and B independent
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

    # Skip the chi=8 stage: bipartite implicit AD is non-variational below
    # chi=16 (warning at ipeps_optimize.py:2089), and the v6b/v6c chi=8
    # probe at 2026-05-16 21:50 showed the optimizer pushes E to -0.85 by
    # step 5 — already in the drift basin before reactive bumps can catch up.
    chi_schedule = [(16, 30), (24, 30)]

    t0 = time.perf_counter()
    (A_opt, B_opt), envs, E_gs = optimize_gs_ad_chi_schedule(
        H, None, config, chi_schedule
    )
    dt = time.perf_counter() - t0

    print(f"\n  D       = {D}")
    print(f"  E/site  = {E_gs:.8f}")
    print(f"  time    = {dt:.1f}s")


if __name__ == "__main__":
    main()
