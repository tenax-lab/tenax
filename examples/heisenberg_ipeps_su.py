#!/usr/bin/env python3
"""2D Heisenberg ground state via iPEPS simple update.

Finds the ground-state energy of the spin-1/2 antiferromagnetic Heisenberg
model on the infinite square lattice using imaginary time evolution (simple
update) followed by Corner Transfer Matrix (CTM) environment contraction.

Two unit cells are demonstrated:

1. **1x1 unit cell, D=2** -- single-site translational invariance.
2. **2x2 unit cell, D=2** -- checkerboard (2-site) cell that can capture
   Neel-type antiferromagnetic order.

The exact ground-state energy per site is E/N ~ -0.6694 (QMC reference).
The 1x1 unit cell gives E ~ 0.5 (product state) because a single tensor
cannot represent the two-sublattice Neel order.  The 2x2 unit cell breaks
this symmetry and yields a much lower energy.

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
    unit_cell: str = "1x1",
    label: str = "",
):
    """Run iPEPS simple update + CTM and print results."""
    config = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=num_steps,
        dt=dt,
        ctm=CTMConfig(chi=chi, max_iter=100),
        unit_cell=unit_cell,
    )

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  D={D}, chi={chi}, unit_cell={unit_cell}")
    print(f"  SU steps={num_steps}, dt={dt}")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    energy, peps, env = ipeps(gate, initial_peps=None, config=config)
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

    # --- 1x1 unit cell, D=2 ---
    run_simple_update(
        gate,
        D=2,
        chi=16,
        num_steps=500,
        dt=0.05,
        unit_cell="1x1",
        label="1x1 unit cell, D=2",
    )

    # --- 2x2 (checkerboard) unit cell, D=2 ---
    run_simple_update(
        gate,
        D=2,
        chi=16,
        num_steps=200,
        dt=0.3,
        unit_cell="2site",
        label="2x2 checkerboard unit cell, D=2",
    )


if __name__ == "__main__":
    main()
