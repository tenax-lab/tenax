#!/usr/bin/env python3
"""v8b: bipartite implicit AD with 9→12→16→18 chi ramp on the #493-fixed branch.

Configuration: ``gs_c4v=False`` + ``gs_implicit_ad=True`` (implicit AD via
fixed-point adjoint), ``chi_schedule=[(9,20),(12,20),(16,20),(18,20)]``,
``chi_auto_bump=False`` (explicit ramp only, no auto-bump noise).

The schedule deliberately caps at χ=18 (well below variPEPS's canonical
χ=24) to keep wall-clock manageable while still probing all chi-bump
transitions on the #494-fixed energy.

Verdict logic:
- v7b (fixed χ=24, post-fix) stayed above QMC through step 35.
- If v8b also stays above QMC across all four chi stages, then the
  implicit-AD path remains variational under chi-ramping, which is
  the production code path users will hit.

QMC reference: E/site ≈ −0.6694.  variPEPS bipartite ref: -0.668168.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python examples/heisenberg_ipeps_ad_2x2_v8b_bipartite_ramp.py
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
    print("iPEPS-AD v8b: bipartite implicit AD + chi-ramp (post-#493 fix)")
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
            chi_auto_bump=False,
        ),
        gs_c4v=False,
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

    chi_schedule = [(9, 20), (12, 20), (16, 20), (18, 20)]

    t0 = time.perf_counter()
    (A_opt, B_opt), envs, E_gs = optimize_gs_ad_chi_schedule(
        H, None, config, chi_schedule
    )
    dt = time.perf_counter() - t0

    print(f"\n  D       = {D}")
    print(f"  E/site  = {E_gs:.8f}")
    print(f"  time    = {dt:.1f}s")
    print(f"  drift   = {'YES (E < QMC)' if E_gs < -0.6694 else 'NO (E >= QMC)'}")


if __name__ == "__main__":
    main()
