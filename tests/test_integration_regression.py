"""Integration regression tests for block preservation across algorithms.

These tests verify the cross-cutting invariant: symmetric block structure
survives end-to-end through every algorithm pipeline. Individual algorithm
tests exist but test each in isolation; these catch regressions where a
change in one area (e.g., contraction engine, SVD) breaks block preservation
across multiple algorithms.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor import (
    _build_double_layer_tensor,
    _ctm_tensor_sweep,
    compute_energy_ctm_tensor,
    ctm_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._split_ctm_tensor import (
    _split_ctm_tensor_sweep,
    ctm_split_tensor,
    initialize_split_ctm_tensor_env,
)
from tenax.algorithms.auto_mpo import build_auto_mpo
from tenax.algorithms.dmrg import (
    DMRGConfig,
    build_random_mps,
    build_random_symmetric_mps,
    dmrg,
)
from tenax.algorithms.hotrg import HOTRGConfig, _hotrg_step_horizontal, hotrg
from tenax.algorithms.trg import TRGConfig, _trg_step, compute_ising_tensor, trg
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionicU1, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor
from tenax.network.network import TensorNetwork

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _build_symmetric_heisenberg_mpo(L: int, Jz: float = 1.0, Jxy: float = 1.0):
    """Build U(1)-symmetric Heisenberg MPO."""
    terms = []
    for i in range(L - 1):
        terms.append((Jz, "Sz", i, "Sz", i + 1))
        terms.append((Jxy / 2, "Sp", i, "Sm", i + 1))
        terms.append((Jxy / 2, "Sm", i, "Sp", i + 1))
    return build_auto_mpo(terms, L=L, symmetric=True)


def _densify_tensor_network(tn: TensorNetwork) -> TensorNetwork:
    """Convert all tensors in a TensorNetwork to DenseTensor."""
    result = TensorNetwork()
    for nid in tn.node_ids():
        t = tn.get_tensor(nid)
        if isinstance(t, DenseTensor):
            result.add_node(nid, t)
        else:
            result.add_node(nid, DenseTensor(t.todense(), t.indices))
    return result


def _make_symmetric_peps(key, sym, virt_charges, phys_charges):
    """Build a random SymmetricTensor iPEPS site tensor."""
    indices = (
        TensorIndex(sym, virt_charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex(sym, virt_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex(sym, virt_charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex(sym, virt_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
    )
    return SymmetricTensor.random_normal(indices, key)


# ------------------------------------------------------------------ #
# Block Preservation Tests                                             #
# ------------------------------------------------------------------ #


class TestBlockPreservation:
    """Verify block structure survives full algorithm pipelines."""

    def test_dmrg_symmetric_blocks_preserved(self):
        """U(1) symmetric DMRG: every MPS site retains blocks after sweeps."""
        L = 6
        mpo = _build_symmetric_heisenberg_mpo(L)
        mps = build_random_symmetric_mps(L, bond_dim=4, seed=42)
        config = DMRGConfig(max_bond_dim=8, num_sweeps=4, lanczos_max_iter=20)
        result = dmrg(mpo, mps, config)

        for i in range(L):
            t = result.mps.get_tensor(i)
            assert isinstance(t, SymmetricTensor), (
                f"Site {i}: expected SymmetricTensor, got {type(t).__name__}"
            )
            assert t.n_blocks > 0, f"Site {i}: zero blocks after DMRG"

    def test_trg_symmetric_blocks_preserved(self):
        """Z2 symmetric TRG: blocks don't collapse across coarse-graining."""
        tensor = compute_ising_tensor(beta=0.3, symmetric=True)
        initial_blocks = tensor.n_blocks
        assert initial_blocks == 8  # sanity

        T = tensor
        for step in range(5):
            T, _ = _trg_step(T, max_bond_dim=8, svd_trunc_err=None)
            assert isinstance(T, SymmetricTensor), (
                f"Step {step}: lost SymmetricTensor type"
            )
            assert T.n_blocks >= initial_blocks, (
                f"Step {step}: block count collapsed from {initial_blocks} "
                f"to {T.n_blocks}"
            )

    def test_hotrg_symmetric_blocks_preserved(self):
        """Z2 symmetric HOTRG: blocks preserved across RG steps."""
        tensor = compute_ising_tensor(beta=0.3, symmetric=True)
        initial_blocks = tensor.n_blocks
        assert initial_blocks == 8

        T = tensor
        for step in range(3):
            T, _ = _hotrg_step_horizontal(T, max_bond_dim=8)
            assert isinstance(T, SymmetricTensor), (
                f"Step {step}: lost SymmetricTensor type"
            )
            assert T.n_blocks >= initial_blocks, (
                f"Step {step}: block count collapsed from {initial_blocks} "
                f"to {T.n_blocks}"
            )

    def test_ctm_tensor_symmetric_blocks_preserved(self):
        """U(1) standard CTM: blocks preserved across CTM sweeps."""
        sym = U1Symmetry()
        charges = np.zeros(2, dtype=np.int32)
        phys_charges = np.zeros(2, dtype=np.int32)
        A = _make_symmetric_peps(jax.random.PRNGKey(99), sym, charges, phys_charges)

        chi = 4
        env = initialize_ctm_tensor_env(A, chi)
        a = _build_double_layer_tensor(A)

        init_blocks = min(t.n_blocks for t in env)
        assert init_blocks > 0

        for sweep in range(10):
            env = _ctm_tensor_sweep(env, a, chi, renormalize=True)

        for t in env:
            assert isinstance(t, SymmetricTensor), (
                "Env tensor lost SymmetricTensor type after sweeps"
            )
            assert t.n_blocks > 0, "Env tensor has zero blocks after CTM"

    def test_split_ctm_tensor_symmetric_blocks_preserved(self):
        """FermionicU1 split CTM: charge sectors survive CTM sweeps."""
        sym = FermionicU1()
        virt_charges = np.array([0, 1], dtype=np.int32)
        phys_charges = np.array([0, 1], dtype=np.int32)
        A = _make_symmetric_peps(
            jax.random.PRNGKey(77), sym, virt_charges, phys_charges
        )

        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(A, chi, chi_I)
        init_blocks = min(len(t._blocks) for t in env)

        for _ in range(5):
            env = _split_ctm_tensor_sweep(env, A, chi, chi_I, True)

        for t in env:
            assert isinstance(t, SymmetricTensor), (
                "Env tensor lost SymmetricTensor type after split CTM sweeps"
            )
            assert len(t._blocks) >= init_blocks, (
                f"Block count dropped from {init_blocks} to {len(t._blocks)}: "
                f"charge sectors collapsed"
            )


# ------------------------------------------------------------------ #
# Symmetric-vs-Dense Equivalence Tests                                 #
# ------------------------------------------------------------------ #


class TestSymmetricMatchesDense:
    """Verify symmetric path produces same physics as dense path."""

    def test_dmrg_energy_matches_dense(self):
        """Symmetric DMRG energy matches dense DMRG for L=6 Heisenberg."""
        L = 6

        # Symmetric path
        mpo_sym = _build_symmetric_heisenberg_mpo(L)
        mps_sym = build_random_symmetric_mps(L, bond_dim=4, seed=7)
        config = DMRGConfig(
            max_bond_dim=8,
            num_sweeps=6,
            lanczos_max_iter=30,
            convergence_tol=1e-10,
        )
        result_sym = dmrg(mpo_sym, mps_sym, config)

        # Dense path
        mpo_dense = _densify_tensor_network(mpo_sym)
        mps_dense = _densify_tensor_network(mps_sym)
        result_dense = dmrg(mpo_dense, mps_dense, config)

        np.testing.assert_allclose(
            result_sym.energy,
            result_dense.energy,
            atol=1e-6,
            err_msg="Symmetric DMRG energy diverges from dense DMRG",
        )

    def test_trg_free_energy_matches_dense(self):
        """Symmetric TRG free energy matches dense TRG for 2D Ising."""
        beta = 0.3
        config = TRGConfig(max_bond_dim=16, num_steps=20)

        tensor_sym = compute_ising_tensor(beta=beta, symmetric=True)
        tensor_dense = compute_ising_tensor(beta=beta, symmetric=False)

        log_z_sym = float(trg(tensor_sym, config))
        log_z_dense = float(trg(tensor_dense, config))

        F_sym = -log_z_sym / beta
        F_dense = -log_z_dense / beta

        # Block-sparse SVD truncation differs from dense at finite chi,
        # so tolerance matches existing test_trg.py symmetric-vs-dense check.
        np.testing.assert_allclose(
            F_sym,
            F_dense,
            atol=0.01,
            err_msg="Symmetric TRG free energy diverges from dense TRG",
        )

    def test_ctm_tensor_energy_matches_dense(self):
        """Symmetric CTM energy matches DenseTensor CTM."""
        sym = U1Symmetry()
        charges = np.zeros(2, dtype=np.int32)
        phys_charges = np.zeros(2, dtype=np.int32)
        A_sym = _make_symmetric_peps(jax.random.PRNGKey(99), sym, charges, phys_charges)
        A_dense = DenseTensor(A_sym.todense(), A_sym.indices)

        # Heisenberg gate
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        gate = H.reshape(d, d, d, d)

        chi = 6

        env_dense = ctm_tensor(A_dense, chi=chi, max_iter=40, conv_tol=1e-8)
        E_dense = float(compute_energy_ctm_tensor(A_dense, env_dense, gate, d=d))

        env_sym = ctm_tensor(A_sym, chi=chi, max_iter=40, conv_tol=1e-8)
        E_sym = float(compute_energy_ctm_tensor(A_sym, env_sym, gate, d=d))

        np.testing.assert_allclose(
            E_sym,
            E_dense,
            atol=1e-4,
            err_msg="Symmetric CTM energy diverges from dense CTM",
        )
