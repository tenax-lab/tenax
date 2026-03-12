#!/usr/bin/env python3
"""2D Heisenberg ground state via iPEPS AD optimization.

Finds the ground-state energy of the spin-1/2 antiferromagnetic Heisenberg
model on the infinite square lattice using automatic differentiation (AD)
through the Corner Transfer Matrix (CTM) fixed-point equation.

Two initialization strategies are compared:

1. **Random init** — site tensor drawn from a normal distribution.
2. **Simple update init** (``su_init=True``) — imaginary time evolution
   provides a physically reasonable starting tensor before AD refinement.

The exact ground-state energy per site is E/N ~ -0.6694 (QMC reference).
With the small D and chi used here, the variational energy will be above
this exact value.

Usage::

    uv run python examples/heisenberg_ipeps_ad.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, heisenberg_gate, iPEPSConfig, optimize_gs_ad

# ---------------------------------------------------------------------------
# Run AD optimization
# ---------------------------------------------------------------------------


def run_ad(
    gate: jnp.ndarray,
    D: int,
    chi: int,
    gs_steps: int,
    lr: float,
    su_init: bool,
    su_steps: int = 200,
    su_dt: float = 0.01,
    A_init: jnp.ndarray | None = None,
    label: str = "",
):
    """Run optimize_gs_ad and print results."""
    config = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=su_steps,
        dt=su_dt,
        ctm=CTMConfig(chi=chi, max_iter=100),
        gs_optimizer="adam",
        gs_learning_rate=lr,
        gs_num_steps=gs_steps,
        su_init=su_init,
    )

    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  D={D}, chi={chi}, lr={lr}, AD steps={gs_steps}")
    if su_init:
        print(f"  SU init: {su_steps} steps, dt={su_dt}")
    else:
        print("  Random initialization")
    print(f"{'─' * 60}")

    t0 = time.perf_counter()
    A_opt, env, E_gs = optimize_gs_ad(gate, A_init, config)
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

    # --- D=2, random init ---
    run_ad(
        gate,
        D=2,
        chi=4,
        gs_steps=20,
        lr=1e-2,
        su_init=False,
        label="D=2, random init",
    )

    # --- D=2, SU init ---
    run_ad(
        gate,
        D=2,
        chi=4,
        gs_steps=20,
        lr=1e-2,
        su_init=True,
        su_steps=200,
        su_dt=0.05,
        label="D=2, simple update init",
    )


if __name__ == "__main__":
    main()
