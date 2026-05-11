#!/usr/bin/env python3
"""Diagnose CTM iteration count and final residual for the production
Heisenberg iPEPS setup, isolating the CTM convergence path from L-BFGS.

Background: Tenax CTM is suspected to plateau before reaching
``conv_tol=1e-5`` on iPEPS site tensors, forcing the full ``max_iter``
budget on every AD-energy evaluation (see project memo
``project_tenax_ctm_doesnt_converge_random_init.md``, 2026-05-10).
This script confirms whether that still happens on the 2-site C4v +
2x2 plaquette projector path at production-relevant χ.

Outputs (per stage):
    chi=8/16 D=3 SU init: iters, final sv_diff, max ε_T, wall-time
"""

from __future__ import annotations

import time

import jax

jax.config.update("jax_enable_x64", True)

from tenax import (  # noqa: E402
    CTMConfig,
    heisenberg_gate,
    ipeps,
    iPEPSConfig,
)
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    CHECKERBOARD_NEIGHBORS,
)


def _run(site_tensors, chi, conv_tol, max_iter, label):
    t0 = time.perf_counter()
    envs, info = python_loop_ctm_converge(
        site_tensors,
        CHECKERBOARD_NEIGHBORS,
        chi=chi,
        max_iter=max_iter,
        min_iter=10,
        conv_tol=conv_tol,
        conv_method="elementwise",
        renormalize=True,
        projector_method="svd",
    )
    # second call: prove JIT-cached run-time
    t1 = time.perf_counter()
    envs2, info2 = python_loop_ctm_converge(
        site_tensors,
        CHECKERBOARD_NEIGHBORS,
        chi=chi,
        max_iter=max_iter,
        min_iter=10,
        conv_tol=conv_tol,
        conv_method="elementwise",
        renormalize=True,
        projector_method="svd",
        env_init=envs,
    )
    t2 = time.perf_counter()

    print(
        f"  {label}: iters={info.iterations:>3d} "
        f"converged={info.converged!s:>5s} "
        f"sv_diff={info.sv_diff:.3e} "
        f"eps_T={info.max_truncation_error:.3e} "
        f"t_cold={t1 - t0:6.1f}s "
        f"t_warm={t2 - t1:6.1f}s",
        flush=True,
    )


def main():
    print("CTM convergence diagnostic: 2-site AFM Heisenberg, D=3, 2x2 projector")
    print("(reports iters to reach conv_tol; if iters == max_iter, plateau)\n")

    H = heisenberg_gate()
    D = 3

    # Get an SU-init 2-site iPEPS (matches what optimize_gs_ad uses internally
    # with su_init=True).
    su_cfg = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=200,
        dt=0.05,
        unit_cell="2site",
        ctm=CTMConfig(chi=10, max_iter=40),
    )
    _, peps, _ = ipeps(H, None, su_cfg)
    A, B = peps  # 2-site SU output

    site_tensors = {(0, 0): A, (1, 0): B}

    print(f"SU done: A.norm={A.norm():.4f}, B.norm={B.norm():.4f}\n")

    print("Tolerance sweep at chi=8, max_iter=100:")
    for tol in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        _run(site_tensors, chi=8, conv_tol=tol, max_iter=100, label=f"tol={tol:.0e}")

    print("\nTolerance sweep at chi=16, max_iter=100:")
    for tol in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        _run(site_tensors, chi=16, conv_tol=tol, max_iter=100, label=f"tol={tol:.0e}")


if __name__ == "__main__":
    main()
