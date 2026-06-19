#!/usr/bin/env python3
"""v6: heisenberg_ipeps_ad_2x2 + reactive ε_T auto-bump composed with the
v5 chi-schedule (#455 follow-up benchmark, 2026-05-15).

Identical to ``heisenberg_ipeps_ad_2x2.py`` except for:
    - ``chi_auto_bump=True``
    - ``chi_auto_bump_eps=1e-5`` (variPEPS §2.8.2 default)
    - ``chi_auto_bump_step=2`` (default)
    - ``chi=8`` start in CTMConfig (matches v5 first stage)

Goal: test whether reactive ε_T-driven bumps fire *earlier* than the
scheduled stage budget, advancing chi=8→16→24 before the
``[(8,30),(16,30),(24,20)]`` schedule's nominal step boundaries. The
compose order in ``_optimize_gs_ad_tensor_2site`` is reactive first
(``_maybe_bump_chi``), scheduled second (``_advance_chi_stage_if_due``),
so reactive can pre-empt the schedule but not vice-versa. Schedule
serves as an upper bound.

Comparison target:
    - v5 (#455-baseline): chi-schedule only — E=-0.66422809 in 8h52m,
      chi=24 stalled without improvement over chi=16.
    - variPEPS bipartite (pure reactive auto-bump): documented in
      ``project_chi_schedule_v5_validated.md``.

PR #481 (#474) makes ε_T real on the 2x2 plaquette path; before that
patch, eps_T was identically 0.0 and reactive bumps could never fire.

PR #484 adds INFO-level log lines to ``_maybe_bump_chi`` so reactive
bump events are visible in stdout; this script sets the logger to INFO.

QMC reference: E/site ≈ −0.6694.

Usage::

    JAX_PLATFORMS=cpu uv run python examples/heisenberg_ipeps_ad_2x2_v6_compose.py
"""

from __future__ import annotations

import logging
import time

import jax

jax.config.update("jax_enable_x64", True)

# PR #484 observability: surface _maybe_bump_chi INFO log lines so we can see
# when reactive bumps fire vs schedule-only advances. Set DEBUG to also see
# per-step (chi, eps_T, threshold) trace useful for retuning chi_auto_bump_eps.
logging.basicConfig(level=logging.INFO, format="%(message)s")

from tenax import (  # noqa: E402
    CTMConfig,
    aligned_ctm_schedules,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad_chi_schedule,
)


def main():
    print("iPEPS-AD v6: chi_schedule + chi_auto_bump compose (#455 follow-up)")
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
            chi=8,  # start small; schedule + auto-bump ramps to chi_max
            max_iter=75,
            min_iter=10,
            conv_tol=1e-5,
            projector_method="svd",
            forward_gauge="phase",
            adjoint_method="fixed_point",
            gmres_tol=1e-7,
            gmres_maxiter=75,
            # v6: enable reactive ε_T auto-bump (variPEPS §2.8.2)
            chi_auto_bump=True,
            chi_auto_bump_eps=1e-5,
            chi_auto_bump_step=2,
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

    # Same v5 schedule — auto-bump may pre-empt some stage advances.
    chi_schedule = [(8, 30), (16, 30), (24, 20)]

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
