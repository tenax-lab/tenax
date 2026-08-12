#!/usr/bin/env python3
"""2D Heisenberg ground state via iPEPS simple update (2-site unit cell).

Finds the ground-state energy of the spin-1/2 antiferromagnetic Heisenberg
model on the infinite square lattice using imaginary time evolution (simple
update) followed by Corner Transfer Matrix (CTM) environment contraction.

The 2-site (checkerboard) unit cell captures Neel-type antiferromagnetic
order.  The exact ground-state energy per site is E/N ~ -0.6694 (QMC
reference).

Usage::

    uv run python examples/heisenberg_ipeps_su.py
"""

from __future__ import annotations

import time

import jax

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, heisenberg_gate, ipeps, iPEPSConfig

# ---------------------------------------------------------------------------
# Run simple update
# ---------------------------------------------------------------------------


def run_simple_update(
    gate,
    D: int,
    chi: int,
    num_steps: int,
    dt: float,
    label: str = "",
):
    """Run iPEPS 2-site simple update + CTM and print results."""
    config = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=num_steps,
        dt=dt,
        ctm=CTMConfig(chi=chi, max_iter=100),
    )

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  D={D}, chi={chi}")
    print(f"  SU steps={num_steps}, dt={dt}")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    energy, (A, B), (env_A, env_B) = ipeps(gate, initial_peps=None, config=config)
    elapsed = time.perf_counter() - t0

    print(f"  E/site = {energy:.6f}")
    print(f"  Time   = {elapsed:.1f}s")

    return energy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("iPEPS simple update: 2D Heisenberg model")
    print("H = sum_{<i,j>} S_i . S_j   (J=1, antiferromagnetic)")
    print("QMC reference: E/site ~ -0.6694")

    gate = heisenberg_gate()

    # --- 2-site (checkerboard) unit cell, D=2 ---
    run_simple_update(
        gate,
        D=2,
        chi=16,
        num_steps=200,
        # dt=0.3 predates the #667 fix, when lam_2 was proportional to dt and a
        # large step was the only thing keeping the state off the product state.
        dt=0.05,
        label="2-site checkerboard, D=2",
    )


if __name__ == "__main__":
    main()
