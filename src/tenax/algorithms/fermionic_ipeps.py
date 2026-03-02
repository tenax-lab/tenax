"""Fermionic infinite Projected Entangled Pair States (fPEPS) algorithm.

This module implements iPEPS for fermionic systems using FermionParity symmetry.
The state is represented as a PEPS with fermionic tensor structure, where
Koszul signs are automatically handled by SymmetricTensor operations.

Currently supports:
- Spinless fermion Hamiltonian: H = -t(c†c + h.c.) + V(n_i n_j)
- Trotter decomposition for imaginary time evolution
- fPEPS site tensor initialization with FermionParity

Reference:
- Corboz et al., PRB 81, 165104 (2010)
- Barthel, Pineda, Eisert, PRA 80, 042333 (2009)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor


@dataclass
class FPEPSConfig:
    """Configuration for fermionic iPEPS.

    Attributes:
        D:   Virtual bond dimension.
        t:   Hopping amplitude.
        V:   Nearest-neighbour interaction strength.
        dt:  Imaginary time step size for Trotter decomposition.
    """

    D: int = 2
    t: float = 1.0
    V: float = 0.0
    dt: float = 0.01


def spinless_fermion_gate(config: FPEPSConfig) -> SymmetricTensor:
    """Build the 2-site Hamiltonian H = -t(c†c + h.c.) + V(n_i n_j).

    The Hamiltonian acts on two spinless fermion sites with local
    Hilbert space {|0>, |1>} (empty, occupied). The fermionic
    anti-commutation relations are encoded via FermionParity symmetry.

    Args:
        config: FPEPSConfig with hopping t and interaction V.

    Returns:
        SymmetricTensor with 4 legs (si, sj, si_out, sj_out),
        shape (2, 2, 2, 2), using FermionParity symmetry.
    """
    t = config.t
    V = config.V

    # Build the dense 4x4 Hamiltonian matrix in the basis
    # |00>, |01>, |10>, |11> (site i tensor site j)
    #
    # c†_i c_j: |10><01| (with fermionic sign from Jordan-Wigner = +1 here)
    # c†_j c_i: |01><10|
    # n_i n_j:  |11><11|
    H = np.zeros((4, 4), dtype=np.float64)

    # Hopping: -t (c†_i c_j + c†_j c_i)
    # |01> -> |10>: c†_i c_j |01> = c†_i |00> = |10>, sign = +1
    # |10> -> |01>: c†_j c_i |10> = c†_j |00> = |01>, sign = +1
    H[2, 1] = -t  # <10|H|01>
    H[1, 2] = -t  # <01|H|10>

    # Interaction: V * n_i * n_j
    H[3, 3] = V  # <11|H|11>

    # Reshape to (2, 2, 2, 2): (si, sj, si_out, sj_out)
    H_4leg = H.reshape(2, 2, 2, 2)

    # Create TensorIndex objects with FermionParity
    sym = FermionParity()
    charges = np.array([0, 1], dtype=np.int32)

    indices = (
        TensorIndex(sym, charges, FlowDirection.IN, label="si"),
        TensorIndex(sym, charges, FlowDirection.IN, label="sj"),
        TensorIndex(sym, charges, FlowDirection.OUT, label="si_out"),
        TensorIndex(sym, charges, FlowDirection.OUT, label="sj_out"),
    )

    return SymmetricTensor.from_dense(jnp.array(H_4leg), indices)


def _trotter_gate(H: SymmetricTensor, dt: float) -> SymmetricTensor:
    """Compute the Trotter gate exp(-dt * H).

    Uses dense eigendecomposition: H = U diag(E) U†, then
    exp(-dt * H) = U diag(exp(-dt * E)) U†.

    Args:
        H:  2-site Hamiltonian as SymmetricTensor with 4 legs.
        dt: Imaginary time step (real-valued).

    Returns:
        SymmetricTensor with same indices as H, representing exp(-dt * H).
    """
    dense = H.todense().reshape(4, 4)
    dense_np = np.array(dense)

    # Eigendecomposition of the Hermitian matrix
    eigvals, eigvecs = np.linalg.eigh(dense_np)

    # Compute exp(-dt * H)
    exp_eigvals = np.exp(-dt * eigvals)
    gate = eigvecs @ np.diag(exp_eigvals) @ eigvecs.T

    # Reshape back to (2, 2, 2, 2)
    gate_4leg = gate.reshape(2, 2, 2, 2)

    return SymmetricTensor.from_dense(jnp.array(gate_4leg), H.indices)


def _initialize_fpeps(config: FPEPSConfig, key: jax.Array) -> SymmetricTensor:
    """Create a random fPEPS site tensor A[u, d, l, r, phys].

    The tensor has FermionParity symmetry on all legs. Virtual bond
    charges alternate 0, 1, 0, 1, ... for bond dimension D.
    Physical charges are [0, 1] (empty, occupied).

    Flows:
        u = OUT, d = IN, l = OUT, r = IN, phys = IN

    Args:
        config: FPEPSConfig with bond dimension D.
        key:    JAX random key.

    Returns:
        SymmetricTensor with 5 legs (u, d, l, r, phys).
    """
    D = config.D
    sym = FermionParity()

    # Virtual charges: [i % 2 for i in range(D)]
    virt_charges = np.array([i % 2 for i in range(D)], dtype=np.int32)

    # Physical charges: [0, 1]
    phys_charges = np.array([0, 1], dtype=np.int32)

    indices = (
        TensorIndex(sym, virt_charges, FlowDirection.OUT, label="u"),
        TensorIndex(sym, virt_charges, FlowDirection.IN, label="d"),
        TensorIndex(sym, virt_charges, FlowDirection.OUT, label="l"),
        TensorIndex(sym, virt_charges, FlowDirection.IN, label="r"),
        TensorIndex(sym, phys_charges, FlowDirection.IN, label="phys"),
    )

    return SymmetricTensor.random_normal(indices, key)
