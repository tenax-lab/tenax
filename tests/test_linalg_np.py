"""Tests for _truncated_svd_symmetric_np (numpy symmetric SVD)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._block_array import BlockArray, ba_to_symmetric
from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor
from tenax.linalg import (
    _qr_symmetric,
    _qr_symmetric_np,
    _truncated_svd_symmetric,
    _truncated_svd_symmetric_np,
)

IN = FlowDirection.IN
OUT = FlowDirection.OUT


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def sym_2leg():
    """Random 2-leg U(1)-symmetric tensor for SVD tests."""
    sym = U1Symmetry()
    # Left: charge 0 x3, charge 1 x3, charge -1 x2
    charges_l = np.array([0, 0, 0, 1, 1, 1, -1, -1], dtype=np.int32)
    idx_l = TensorIndex(sym, charges_l, IN, label="l")
    # Right: must be dual (opposite flow) with same charge multiplicities
    # for conservation sum(flow*q)==0: block (q,q) needs right to have
    # charge q with the same multiplicity.
    charges_r = np.array([0, 0, 0, 1, 1, 1, -1, -1], dtype=np.int32)
    idx_r = TensorIndex(sym, charges_r, OUT, label="r")
    rng = np.random.default_rng(42)
    blocks = {
        (0, 0): jnp.array(rng.standard_normal((3, 3))),
        (1, 1): jnp.array(rng.standard_normal((3, 3))),
        (-1, -1): jnp.array(rng.standard_normal((2, 2))),
    }
    return SymmetricTensor(blocks, (idx_l, idx_r))


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #


class TestTruncatedSvdSymmetricNp:
    def test_matches_jax_version(self, sym_2leg):
        """Numpy SVD produces the same singular values as the JAX version."""
        U_jax, s_jax, Vh_jax, s_full_jax = _truncated_svd_symmetric(
            sym_2leg, ["l"], ["r"], None, None, "bond", False
        )
        U_np, s_np, Vh_np, s_full_np = _truncated_svd_symmetric_np(
            sym_2leg, ["l"], ["r"], None, None, "bond", False
        )

        # Return types
        assert isinstance(U_np, BlockArray)
        assert isinstance(Vh_np, BlockArray)
        assert isinstance(s_np, np.ndarray)
        assert isinstance(s_full_np, np.ndarray)

        # Singular values match (sorted descending)
        s_jax_sorted = np.sort(np.asarray(s_jax))[::-1]
        s_np_sorted = np.sort(s_np)[::-1]
        np.testing.assert_allclose(s_np_sorted, s_jax_sorted, rtol=1e-12)

        # Full spectrum matches too
        s_full_jax_sorted = np.sort(np.asarray(s_full_jax))[::-1]
        s_full_np_sorted = np.sort(s_full_np)[::-1]
        np.testing.assert_allclose(s_full_np_sorted, s_full_jax_sorted, rtol=1e-12)

    def test_truncation(self, sym_2leg):
        """max_singular_values=4 produces exactly 4 singular values."""
        _, s_np, _, s_full_np = _truncated_svd_symmetric_np(
            sym_2leg, ["l"], ["r"], 4, None, "bond", False
        )
        assert len(s_np) == 4
        # Full spectrum should contain all singular values (3+3+2 = 8)
        assert len(s_full_np) == 8

    def test_reconstruction(self, sym_2leg):
        """U @ diag(s) @ Vh reconstructs the original tensor."""
        U_ba, s_np, Vh_ba, _ = _truncated_svd_symmetric_np(
            sym_2leg, ["l"], ["r"], None, None, "bond", False
        )

        # Convert BlockArrays back to SymmetricTensors for contraction
        U_sym = ba_to_symmetric(U_ba)
        Vh_sym = ba_to_symmetric(Vh_ba)

        # Scale U by singular values: U @ diag(s)
        s_jax = jnp.array(s_np)
        Us = scale_bond_axis(U_sym, "bond", s_jax)

        # Contract Us @ Vh
        reconstructed = contract(Us, Vh_sym)

        # Compare dense representations
        orig_dense = sym_2leg.todense()
        recon_dense = reconstructed.todense()
        np.testing.assert_allclose(
            np.asarray(recon_dense), np.asarray(orig_dense), atol=1e-12
        )

    def test_truncation_error(self, sym_2leg):
        """max_truncation_err limits the truncation error."""
        _, s_np, _, s_full_np = _truncated_svd_symmetric_np(
            sym_2leg, ["l"], ["r"], None, 0.1, "bond", False
        )
        # The kept singular values should cover most of the weight
        total_sq = np.sum(s_full_np**2)
        kept_sq = np.sum(s_np**2)
        trunc_err = np.sqrt(1.0 - kept_sq / total_sq)
        assert trunc_err <= 0.1 + 1e-14

    def test_normalize(self, sym_2leg):
        """normalize=True makes singular values sum to 1."""
        _, s_np, _, _ = _truncated_svd_symmetric_np(
            sym_2leg, ["l"], ["r"], None, None, "bond", True
        )
        np.testing.assert_allclose(np.sum(s_np), 1.0, atol=1e-14)

    def test_indices_correct(self, sym_2leg):
        """Output BlockArrays have correct index structure."""
        U_ba, _, Vh_ba, _ = _truncated_svd_symmetric_np(
            sym_2leg, ["l"], ["r"], None, None, "bond", False
        )

        # U should have indices: (l, bond)
        assert len(U_ba.indices) == 2
        assert U_ba.indices[0].label == "l"
        assert U_ba.indices[1].label == "bond"
        assert U_ba.indices[1].flow == OUT

        # Vh should have indices: (bond, r)
        assert len(Vh_ba.indices) == 2
        assert Vh_ba.indices[0].label == "bond"
        assert Vh_ba.indices[0].flow == IN
        assert Vh_ba.indices[1].label == "r"


# ------------------------------------------------------------------ #
# QR Tests                                                              #
# ------------------------------------------------------------------ #


class TestQrSymmetricNp:
    def test_reconstruction(self, sym_2leg):
        """Q @ R reconstructs the original tensor."""
        Q_ba, R_ba = _qr_symmetric_np(sym_2leg, ["l"], ["r"], "bond")

        # Return types
        assert isinstance(Q_ba, BlockArray)
        assert isinstance(R_ba, BlockArray)

        # Convert BlockArrays back to SymmetricTensors for contraction
        Q_sym = ba_to_symmetric(Q_ba)
        R_sym = ba_to_symmetric(R_ba)

        # Contract Q @ R
        reconstructed = contract(Q_sym, R_sym)

        # Compare dense representations
        orig_dense = sym_2leg.todense()
        recon_dense = reconstructed.todense()
        np.testing.assert_allclose(
            np.asarray(recon_dense), np.asarray(orig_dense), atol=1e-12
        )

    def test_matches_jax_version(self, sym_2leg):
        """NumPy QR reconstruction matches JAX QR reconstruction."""
        # JAX version
        Q_jax, R_jax = _qr_symmetric(sym_2leg, ["l"], ["r"], "bond")
        recon_jax = contract(Q_jax, R_jax)

        # NumPy version
        Q_ba, R_ba = _qr_symmetric_np(sym_2leg, ["l"], ["r"], "bond")
        Q_sym = ba_to_symmetric(Q_ba)
        R_sym = ba_to_symmetric(R_ba)
        recon_np = contract(Q_sym, R_sym)

        # Both reconstructions should match
        np.testing.assert_allclose(
            np.asarray(recon_np.todense()),
            np.asarray(recon_jax.todense()),
            atol=1e-12,
        )
