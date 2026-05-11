"""Tests for SymmetricTensor support in the 2x2 plaquette projector (#416)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_projector_2x2 import (
    _compute_2x2_projector,
    _gauge_fix_symmetric_svd,
    _scale_bond_by_diag,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor


def _make_test_matrix_tensor(seed: int = 0) -> SymmetricTensor:
    """Build a small 2-leg SymmetricTensor matrix with two U(1) charge sectors."""
    sym = U1Symmetry()
    left_charges = np.array([0, 0, 1, 1], dtype=np.int32)
    right_charges = np.array([0, 0, 1, 1], dtype=np.int32)
    left_idx = TensorIndex.from_charges(
        sym, left_charges, FlowDirection.IN, label="left"
    )
    right_idx = TensorIndex.from_charges(
        sym, right_charges, FlowDirection.OUT, label="right"
    )
    return SymmetricTensor.random_normal(
        (left_idx, right_idx), jax.random.PRNGKey(seed)
    )


def test_gauge_fix_symmetric_svd_preserves_reconstruction():
    """After gauge fix, U_T @ diag(S) @ Vh_T == original matrix (per sector)."""
    from tenax.linalg import svd as tensor_svd

    M_T = _make_test_matrix_tensor(seed=0)
    U_T, s, Vh_T, _ = tensor_svd(
        M_T, left_labels=("left",), right_labels=("right",), new_bond_label="bond"
    )
    U_fixed, Vh_fixed = _gauge_fix_symmetric_svd(U_T, Vh_T)

    from tenax.contraction.contractor import contract

    U_scaled = _scale_bond_by_diag(U_fixed, s, bond_label="bond")
    M_reconstructed = contract(U_scaled, Vh_fixed)
    np.testing.assert_allclose(
        np.asarray(M_reconstructed.todense()),
        np.asarray(M_T.todense()),
        atol=1e-10,
        err_msg="gauge-fixed SVD must preserve reconstruction U·diag(s)·Vh == M",
    )


def test_gauge_fix_symmetric_svd_real_positive_max_row():
    """After gauge fix, the entry of largest |U[:, j]| is real-positive for every j."""
    from tenax.linalg import svd as tensor_svd

    M_T = _make_test_matrix_tensor(seed=1)
    U_T, s, Vh_T, _ = tensor_svd(
        M_T, left_labels=("left",), right_labels=("right",), new_bond_label="bond"
    )
    U_fixed, _ = _gauge_fix_symmetric_svd(U_T, Vh_T)

    U_dense = np.asarray(U_fixed.todense())
    for j in range(U_dense.shape[1]):
        col = U_dense[:, j]
        if np.max(np.abs(col)) == 0.0:
            continue
        max_row = int(np.argmax(np.abs(col)))
        entry = col[max_row]
        assert entry.imag == pytest.approx(0.0, abs=1e-10), (
            f"column {j}: max-abs entry should be real, got {entry}"
        )
        assert entry.real >= 0.0, (
            f"column {j}: max-abs entry should be non-negative, got {entry}"
        )
