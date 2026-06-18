#!/usr/bin/env python3
"""v6c: 2-site bipartite explicit AD with the new chi_auto_bump stack.

Same as v6b (gs_c4v=False, bipartite) but with gs_implicit_ad=False —
explicit AD through the unrolled CTM. The 2-site explicit AD path is
known non-variational at finite χ (issue #328, ipeps_optimize.py:2079
warning): the optimizer drifts below QMC E=-0.6694 to ~-0.566 because
truncation bias at the projector level lets it "cheat".

Goal: test whether the post-#481 / #484 toolkit rescues this:
    - #481: real ε_T on the 2x2 plaquette projector (was 0.0 prior).
    - #484: INFO logging of ε_T per step and reactive bump events.
    - #472/#473: chi_auto_bump on the 2-site AD path — dynamically
      raises χ when ε_T exceeds threshold, forcing the optimizer back
      into the variational regime.
    - stop_gradient on projector outputs (#481): blocks "cheating"
      through projector gradients on the 2x2 path.

If chi_auto_bump=True + chi_auto_bump_eps=1e-5 keeps E above the QMC
floor, the drift was finite-χ truncation bias and the explicit AD path
is reusable. If E still drifts, the bug is in the backward pass itself.

QMC reference: E/site ≈ −0.6694 (drift if E goes below).

Usage::

    CUDA_VISIBLE_DEVICES=1 uv run python examples/heisenberg_ipeps_ad_2x2_v6c_explicit.py
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
    print("iPEPS-AD v6c: bipartite explicit AD with chi_auto_bump (issue #328 probe)")
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
        gs_c4v=False,  # bipartite
        gs_implicit_ad=False,  # v6c: explicit AD through unrolled CTM
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

    # Skip chi=8 stage — see v6b for rationale.
    chi_schedule = [(16, 30), (24, 30)]

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
