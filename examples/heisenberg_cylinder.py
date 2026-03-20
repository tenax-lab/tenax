#!/usr/bin/env python3
"""2D Heisenberg model on a cylinder via DMRG.

Maps a 2D square lattice with periodic boundary conditions in y (circumference
``Ly``) and open boundary conditions in x (length ``Lx``) onto a 1D chain
using column-major ordering::

    site index  i = x * Ly + y,    x in [0, Lx),  y in [0, Ly)

The spin-1/2 antiferromagnetic Heisenberg Hamiltonian is

    H = J * sum_{<i,j>} (Sz_i Sz_j + 0.5 * (S+_i S-_j + S-_i S+_j))

with nearest-neighbour couplings on the cylinder:

- **Within-ring** (y-direction, periodic): ``(x, y) -- (x, (y+1) % Ly)``
- **Between-ring** (x-direction, open):    ``(x, y) -- (x+1, y)``

AutoMPO handles long-range 1D terms arising from the 2D geometry automatically.
MPO compression is used to reduce the bond dimension of long-range terms.

Usage::

    uv run python examples/heisenberg_cylinder.py
"""

from __future__ import annotations

import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax import AutoMPO, DMRGConfig, FiniteMPS, dmrg

# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------


def site(x: int, y: int, Ly: int) -> int:
    """Map 2D coordinate (x, y) to a 1D site index (column-major)."""
    return x * Ly + y


def cylinder_bonds(Lx: int, Ly: int) -> list[tuple[int, int]]:
    """Return all nearest-neighbour bonds on an Lx x Ly cylinder.

    y-direction is periodic (circumference Ly), x-direction is open.
    Each bond is returned as ``(i, j)`` with ``i < j`` in 1D ordering.
    """
    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            # Within-ring bond: (x, y) -- (x, (y+1) % Ly)
            y_next = (y + 1) % Ly
            i, j = site(x, y, Ly), site(x, y_next, Ly)
            bonds.append((min(i, j), max(i, j)))

            # Between-ring bond: (x, y) -- (x+1, y)
            if x < Lx - 1:
                i, j = site(x, y, Ly), site(x + 1, y, Ly)
                bonds.append((min(i, j), max(i, j)))
    return bonds


# ---------------------------------------------------------------------------
# MPO construction via AutoMPO
# ---------------------------------------------------------------------------


def build_heisenberg_cylinder_mpo(
    Lx: int,
    Ly: int,
    J: float = 1.0,
    compress: bool = True,
):
    """Build the Heisenberg MPO on an Lx x Ly cylinder.

    Args:
        Lx: Number of rings (open direction).
        Ly: Circumference (periodic direction).
        J: Coupling constant (positive = antiferromagnetic).
        compress: Apply SVD compression to reduce MPO bond dimension.

    Returns:
        Tuple of ``(mpo, bond_dims)`` where ``mpo`` is a TensorNetwork and
        ``bond_dims`` is the list of uncompressed MPO bond dimensions.
    """
    N = Lx * Ly
    auto = AutoMPO(L=N, d=2)

    for i, j in cylinder_bonds(Lx, Ly):
        auto += (J, "Sz", i, "Sz", j)
        auto += (J / 2, "Sp", i, "Sm", j)
        auto += (J / 2, "Sm", i, "Sp", j)

    bond_dims = auto.bond_dims()
    mpo = auto.to_mpo(compress=compress)
    return mpo, bond_dims


# ---------------------------------------------------------------------------
# Exact diagonalisation reference (small systems only)
# ---------------------------------------------------------------------------


def build_heisenberg_cylinder_exact(
    Lx: int,
    Ly: int,
    J: float = 1.0,
) -> float:
    """Compute the exact ground-state energy via full diagonalisation.

    Only feasible for N = Lx * Ly <= ~16 sites.

    Returns:
        Ground-state energy (float).
    """
    N = Lx * Ly
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]])
    I2 = np.eye(2)

    def kron_chain(ops: list[np.ndarray]) -> np.ndarray:
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    dim = 2**N
    H = np.zeros((dim, dim))

    for i, j in cylinder_bonds(Lx, Ly):
        for op_i, op_j in [(Sz, Sz), (Sp, Sm), (Sm, Sp)]:
            ops = [I2] * N
            ops[i] = op_i
            ops[j] = op_j
            coeff = J if op_i is Sz else J / 2
            H += coeff * kron_chain(ops)

    return float(np.linalg.eigvalsh(H)[0])


# ---------------------------------------------------------------------------
# Run DMRG on a cylinder
# ---------------------------------------------------------------------------


def run_cylinder_dmrg(
    Lx: int,
    Ly: int,
    J: float = 1.0,
    max_bond_dim: int = 64,
    num_sweeps: int = 10,
    initial_bond_dim: int = 16,
    ed_check: bool = False,
):
    """Run DMRG on an Lx x Ly Heisenberg cylinder and print results.

    Args:
        Lx: Number of rings (open direction).
        Ly: Circumference (periodic direction).
        max_bond_dim: Maximum MPS bond dimension (chi).
        num_sweeps: Number of DMRG sweeps.
        initial_bond_dim: Bond dimension for the random initial MPS.
        ed_check: If True, compare against exact diagonalisation.
    """
    N = Lx * Ly
    print(f"\n{'=' * 60}")
    print(f"  Heisenberg cylinder  Lx={Lx}, Ly={Ly}  (N={N} sites)")
    print(f"  max_bond_dim={max_bond_dim}, sweeps={num_sweeps}")
    print(f"{'=' * 60}")

    # Build MPO
    t0 = time.perf_counter()
    mpo, bond_dims_uncomp = build_heisenberg_cylinder_mpo(Lx, Ly, J=J)
    t_mpo = time.perf_counter() - t0
    print(f"  MPO built in {t_mpo:.2f}s")
    print(
        f"  Uncompressed MPO bond dims: max={max(bond_dims_uncomp)}, "
        f"all={bond_dims_uncomp}"
    )

    # Build initial MPS
    mps = FiniteMPS.random(L=N, d=2, chi=initial_bond_dim, key=jax.random.PRNGKey(42))

    # Run DMRG
    config = DMRGConfig(
        max_bond_dim=max_bond_dim,
        num_sweeps=num_sweeps,
        convergence_tol=1e-8,
        two_site=True,
        lanczos_max_iter=50,
        noise=0.0,
        verbose=True,
    )

    t0 = time.perf_counter()
    result = dmrg(mpo, mps, config)
    t_dmrg = time.perf_counter() - t0

    print(f"\n  DMRG finished in {t_dmrg:.1f}s")
    print(f"  Ground-state energy:  E = {result.energy:.8f}")
    print(f"  Energy per site:      E/N = {result.energy / N:.8f}")
    print(f"  Converged: {result.converged}")

    if result.energies_per_sweep:
        print(
            f"  Energies per sweep: {[f'{e:.6f}' for e in result.energies_per_sweep]}"
        )

    # ED cross-check
    if ed_check:
        print(f"\n  Running exact diagonalisation (N={N})...")
        t0 = time.perf_counter()
        e_exact = build_heisenberg_cylinder_exact(Lx, Ly, J=J)
        t_ed = time.perf_counter() - t0
        print(f"  ED energy:   E_exact = {e_exact:.8f}  ({t_ed:.2f}s)")
        print(f"  ED E/N:      {e_exact / N:.8f}")
        diff = abs(result.energy - e_exact)
        print(f"  |E_dmrg - E_exact| = {diff:.2e}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Heisenberg model on a cylinder: 2D DMRG via AutoMPO")
    print("H = J * sum_{<i,j>} S_i . S_j   (J=1, antiferromagnetic)")

    # --- Configuration 1: 4x2 cylinder (8 sites) with ED check ---
    run_cylinder_dmrg(
        Lx=4,
        Ly=2,
        max_bond_dim=32,
        num_sweeps=10,
        initial_bond_dim=8,
        ed_check=True,
    )

    # --- Configuration 2: 6x3 cylinder (18 sites) ---
    run_cylinder_dmrg(
        Lx=6,
        Ly=3,
        max_bond_dim=100,
        num_sweeps=10,
        initial_bond_dim=16,
        ed_check=False,
    )

    # --- Configuration 3: 8x4 cylinder (32 sites) ---
    run_cylinder_dmrg(
        Lx=8,
        Ly=4,
        max_bond_dim=200,
        num_sweeps=15,
        initial_bond_dim=16,
        ed_check=False,
    )


if __name__ == "__main__":
    main()
