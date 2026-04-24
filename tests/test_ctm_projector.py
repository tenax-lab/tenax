"""Tests for block-sparse SVD (Fishman) projector in _ctm_projector.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_projector import _compute_projector_tensor
from tenax.algorithms._ctm_tensor import (
    _fuse_pair_by_label,
    initialize_ctm_tensor_env,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_peps_symmetric():
    """Random SymmetricTensor iPEPS with trivial U(1) charges."""
    D, d = 2, 2
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    rng = np.random.RandomState(99)
    data = jnp.array(rng.standard_normal((D, D, D, D, d)))
    return SymmetricTensor.from_dense(data, indices)


@pytest.fixture
def fpeps_tensor():
    """Random SymmetricTensor iPEPS with FermionParity (nontrivial charges)."""
    key = jax.random.PRNGKey(7)
    sym = FermionParity()
    virt_charges = np.array([0, 1], dtype=np.int32)
    phys_charges = np.array([0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(
            sym, virt_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, virt_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    return SymmetricTensor.random_normal(indices, key)


# ------------------------------------------------------------------ #
# Helpers                                                               #
# ------------------------------------------------------------------ #


def _build_grown_corners(A, chi):
    """Build grown corners from a symmetric iPEPS tensor (left move)."""
    from tenax.contraction.contractor import contract

    env = initialize_ctm_tensor_env(A, chi)

    C1_r = env.C1.relabel("c1_r", "t1_l")
    C1g = contract(C1_r, env.T1)
    C1g = _fuse_pair_by_label(C1g, "c1_d", "u2", "fused", FlowDirection.IN)

    C4_u = env.C4.relabel("c4_u", "t3_r")
    C4g = contract(C4_u, env.T3)
    C4g = _fuse_pair_by_label(C4g, "c4_r", "d2", "fused", FlowDirection.IN)

    return C1g, C4g


# ------------------------------------------------------------------ #
# Tests                                                                 #
# ------------------------------------------------------------------ #


class TestSVDProjectorSymmetric:
    """Tests for block-sparse Fishman SVD projector."""

    def test_svd_projector_returns_symmetric_tensors(self, small_peps_symmetric):
        """SVD projector should return SymmetricTensor pair."""
        A = small_peps_symmetric
        chi = 4
        C1g, C4g = _build_grown_corners(A, chi)

        P1, P2 = _compute_projector_tensor(C1g, C4g, chi, projector_method="svd")

        assert isinstance(P1, SymmetricTensor)
        assert isinstance(P2, SymmetricTensor)
        assert P1.labels() == ("fused", "chi_new")
        assert P2.labels() == ("fused", "chi_new")

    def test_svd_projector_biorthogonality(self, small_peps_symmetric):
        """P1^H @ P2 should be close to identity."""
        A = small_peps_symmetric
        chi = 4
        C1g, C4g = _build_grown_corners(A, chi)

        P1, P2 = _compute_projector_tensor(C1g, C4g, chi, projector_method="svd")

        P1d = P1.todense()
        P2d = P2.todense()
        gram = P1d.conj().T @ P2d
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-10)

    def test_svd_projector_matches_dense_subspace(self, small_peps_symmetric):
        """Block-sparse SVD projector truncated env matches dense SVD projector."""
        A = small_peps_symmetric
        chi = 4
        C1g, C4g = _build_grown_corners(A, chi)

        # Block-sparse path
        P1_sym, P2_sym = _compute_projector_tensor(
            C1g, C4g, chi, projector_method="svd"
        )

        # Dense reference: convert to DenseTensor to force dense path
        fused_pos = C1g.labels().index("fused")
        fused_idx = C1g.indices[fused_pos]
        col1_idx = C1g.indices[1 - fused_pos]
        col2_idx = C4g.indices[1 - fused_pos]

        C1g_dense = DenseTensor(C1g.todense(), (fused_idx, col1_idx))
        C4g_dense = DenseTensor(C4g.todense(), (fused_idx, col2_idx))
        P1_ref, P2_ref = _compute_projector_tensor(
            C1g_dense, C4g_dense, chi, projector_method="svd"
        )

        # The truncated environment P1^H @ C1g should be similar (up to gauge)
        # Compare truncated cross-product: P1^H @ C4g and P2^H @ C1g
        C4gd = C4g.todense()

        trunc_sym = P1_sym.todense().conj().T @ C4gd
        trunc_ref = P1_ref.todense().conj().T @ C4gd

        # The truncated matrices should span the same row space.
        # Compare via SVD singular values of the truncated matrices.
        sv_sym = np.sort(np.linalg.svd(np.array(trunc_sym), compute_uv=False))[::-1]
        sv_ref = np.sort(np.linalg.svd(np.array(trunc_ref), compute_uv=False))[::-1]
        np.testing.assert_allclose(sv_sym, sv_ref, atol=1e-10)

    def test_svd_projector_fpeps_returns_symmetric(self, fpeps_tensor):
        """SVD projector works with FermionParity (nontrivial charges)."""
        A = fpeps_tensor
        chi = 4
        C1g, C4g = _build_grown_corners(A, chi)

        P1, P2 = _compute_projector_tensor(C1g, C4g, chi, projector_method="svd")

        assert isinstance(P1, SymmetricTensor)
        assert isinstance(P2, SymmetricTensor)

        # Biorthogonality
        P1d = P1.todense()
        P2d = P2.todense()
        gram = P1d.conj().T @ P2d
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-10)

    def test_svd_projector_fpeps_matches_dense(self, fpeps_tensor):
        """Block-sparse SVD projector singular values match dense for FermionParity."""
        A = fpeps_tensor
        chi = 4
        C1g, C4g = _build_grown_corners(A, chi)

        # Block-sparse
        P1_sym, P2_sym = _compute_projector_tensor(
            C1g, C4g, chi, projector_method="svd"
        )

        # Dense reference
        fused_pos = C1g.labels().index("fused")
        fused_idx = C1g.indices[fused_pos]
        col1_idx = C1g.indices[1 - fused_pos]
        col2_idx = C4g.indices[1 - fused_pos]
        C1g_dense = DenseTensor(C1g.todense(), (fused_idx, col1_idx))
        C4g_dense = DenseTensor(C4g.todense(), (fused_idx, col2_idx))
        P1_ref, P2_ref = _compute_projector_tensor(
            C1g_dense, C4g_dense, chi, projector_method="svd"
        )

        # Compare truncated cross-product singular values
        C4gd = C4g.todense()
        trunc_sym = P1_sym.todense().conj().T @ C4gd
        trunc_ref = P1_ref.todense().conj().T @ C4gd
        sv_sym = np.sort(np.linalg.svd(np.array(trunc_sym), compute_uv=False))[::-1]
        sv_ref = np.sort(np.linalg.svd(np.array(trunc_ref), compute_uv=False))[::-1]
        np.testing.assert_allclose(sv_sym, sv_ref, atol=1e-10)

    def test_svd_no_todense_on_symmetric_path(self, small_peps_symmetric):
        """Block-sparse SVD path must not call todense()."""
        from unittest.mock import patch

        A = small_peps_symmetric
        chi = 4
        C1g, C4g = _build_grown_corners(A, chi)

        todense_calls = []
        orig_todense = SymmetricTensor.todense

        def tracking_todense(self):
            todense_calls.append(True)
            return orig_todense(self)

        with patch.object(SymmetricTensor, "todense", tracking_todense):
            P1, P2 = _compute_projector_tensor(C1g, C4g, chi, projector_method="svd")

        assert isinstance(P1, SymmetricTensor)
        assert len(todense_calls) == 0, (
            "todense() was called on the block-sparse SVD path"
        )
