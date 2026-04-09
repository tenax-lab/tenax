#!/usr/bin/env python3
"""2D Heisenberg ground state via iPEPS AD optimization.

Finds the ground-state energy of the spin-1/2 antiferromagnetic Heisenberg
model on the infinite square lattice using automatic differentiation (AD)
through the Corner Transfer Matrix (CTM) fixed-point equation.

Two configurations are compared:

1. **L-BFGS with sigma gauge** (recommended) — sigma gauge fixing stabilizes
   the CTM iteration, enabling GMRES implicit differentiation with L-BFGS
   optimizer, Hager-Zhang line search, and metric preconditioning.
2. **Adam baseline** — simple Adam optimizer for comparison.

Both use simple update (SU) initialization and C4v symmetry enforcement.

The exact ground-state energy per site is E/N ~ -0.6694 (QMC reference).
At D=2, chi=8, the best variational energy is E ~ -0.6625.

Usage::

    uv run python examples/heisenberg_ipeps_ad.py
"""

from __future__ import annotations

import time

import jax

jax.config.update("jax_enable_x64", True)

from tenax import (  # noqa: E402
    CTMConfig,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
    sublattice_rotate_gate,
)

# ---------------------------------------------------------------------------
# Run AD optimization
# ---------------------------------------------------------------------------


def run_ad(gate, config, label: str = ""):
    """Run optimize_gs_ad and print results."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  D={config.max_bond_dim}, chi={config.ctm.chi}")
    print(f"  optimizer={config.gs_optimizer}, gauge={config.ctm.forward_gauge}")
    print(f"{'─' * 60}")

    t0 = time.perf_counter()
    A_opt, env, E_gs = optimize_gs_ad(gate, None, config)
    dt = time.perf_counter() - t0

    print(f"  E/site = {E_gs:.6f}")
    print(f"  Time   = {dt:.1f}s")

    return A_opt, env, E_gs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("iPEPS AD ground-state optimization: 2D Heisenberg model")
    print("H = sum_{<i,j>} S_i . S_j   (J=1, antiferromagnetic)")
    print("Exact E/site ~ -0.6694 (QMC)")

    gate = heisenberg_gate()
    H_rot = sublattice_rotate_gate(gate)

    # --- Recommended: L-BFGS + sigma gauge + GMRES ---
    config_lbfgs = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=200,
        dt=0.05,
        ctm=CTMConfig(
            chi=8,
            max_iter=100,
            conv_tol=1e-8,
            projector_method="eigh",
            forward_gauge="sigma",
            ad_backward_method="gmres",
        ),
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_metric_precond=True,
        gs_c4v=True,
        gs_verbose=True,
        gs_log_interval=5,
        gs_num_steps=30,
        su_init=True,
    )
    run_ad(H_rot, config_lbfgs, label="L-BFGS + sigma gauge (recommended)")

    # --- Baseline: Adam ---
    config_adam = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=200,
        dt=0.05,
        ctm=CTMConfig(
            chi=8,
            max_iter=100,
            forward_gauge="sigma",
            ad_backward_method="gmres",
        ),
        gs_optimizer="adam",
        gs_learning_rate=1e-3,
        gs_num_steps=30,
        gs_verbose=True,
        gs_log_interval=5,
        su_init=True,
    )
    run_ad(H_rot, config_adam, label="Adam baseline")


if __name__ == "__main__":
    main()
