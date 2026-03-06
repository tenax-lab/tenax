"""Tests for tenax.linalg module (svd, qr, eigh)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor
from tenax.linalg import eigh

IN = FlowDirection.IN
OUT = FlowDirection.OUT


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def hermitian_dense():
    """Small Hermitian DenseTensor (4x4 reshaped as 2x2 x 2x2)."""
    sym = U1Symmetry()
    key = jax.random.PRNGKey(42)
    M = jax.random.normal(key, (4, 4))
    M = M @ M.T  # make positive semidefinite
    indices = (
        TensorIndex(sym, np.zeros(2, dtype=np.int32), IN, label="i"),
        TensorIndex(sym, np.zeros(2, dtype=np.int32), OUT, label="j"),
        TensorIndex(sym, np.zeros(2, dtype=np.int32), OUT, label="k"),
        TensorIndex(sym, np.zeros(2, dtype=np.int32), IN, label="l"),
    )
    return DenseTensor(M.reshape(2, 2, 2, 2), indices)


@pytest.fixture
def hermitian_symmetric():
    """Small Hermitian SymmetricTensor with nontrivial U(1) charges."""
    sym = U1Symmetry()
    charges = np.array([-1, 0, 1], dtype=np.int32)
    idx_in = TensorIndex(sym, charges, IN, label="row")
    idx_out = TensorIndex(sym, charges, OUT, label="col")

    # Build a symmetric-compatible random tensor, then form M M^T
    key = jax.random.PRNGKey(7)
    A = SymmetricTensor.random_normal((idx_in, idx_out), key)
    # Form Hermitian: A @ A^dagger via dense
    A_d = A.todense()
    M = A_d @ A_d.conj().T
    return SymmetricTensor.from_dense(M, (idx_in, idx_out))


# ------------------------------------------------------------------ #
# eigh tests                                                           #
# ------------------------------------------------------------------ #


class TestEigh:
    def test_eigh_dense_basic(self, hermitian_dense):
        """eigh of a known Hermitian DenseTensor gives correct eigenvalues."""
        T = hermitian_dense
        V, eigvals = eigh(T, ["i", "j"], ["k", "l"], new_bond_label="ev")

        # Verify V has correct labels
        assert V.labels() == ("i", "j", "ev")
        # Eigenvalues should be sorted descending
        assert jnp.all(eigvals[:-1] >= eigvals[1:])
        # Eigenvalues should be non-negative (positive semidefinite input)
        assert jnp.all(eigvals >= -1e-10)

    def test_eigh_dense_reconstruction(self, hermitian_dense):
        """V @ diag(eigenvalues) @ V^T reconstructs the original matrix."""
        T = hermitian_dense
        V, eigvals = eigh(T, ["i", "j"], ["k", "l"], new_bond_label="ev")

        # Reconstruct: V @ diag(eigvals) @ V^T
        V_mat = V.todense().reshape(4, 4)
        reconstructed = V_mat @ jnp.diag(eigvals) @ V_mat.T
        original = T.todense().reshape(4, 4)
        np.testing.assert_allclose(reconstructed, original, atol=1e-10)

    def test_eigh_symmetric_matches_dense(self, hermitian_symmetric):
        """eigh of SymmetricTensor matches dense path."""
        T_sym = hermitian_symmetric
        V_sym, eigvals_sym = eigh(T_sym, ["row"], ["col"], new_bond_label="ev")

        # Dense reference
        M = T_sym.todense()
        M = 0.5 * (M + M.conj().T)
        eigvals_ref, eigvecs_ref = jnp.linalg.eigh(M)
        eigvals_ref = eigvals_ref[::-1]

        # Compare eigenvalues (sorted descending)
        np.testing.assert_allclose(
            np.sort(np.array(eigvals_sym))[::-1],
            np.sort(np.array(eigvals_ref))[::-1],
            atol=1e-10,
        )

    def test_eigh_truncation(self, hermitian_dense):
        """max_eigenvalues truncation keeps only top-k."""
        T = hermitian_dense
        V, eigvals = eigh(
            T, ["i", "j"], ["k", "l"], new_bond_label="ev", max_eigenvalues=2
        )
        assert len(eigvals) == 2
        assert V.todense().shape[-1] == 2


# ------------------------------------------------------------------ #
# Import compatibility tests                                           #
# ------------------------------------------------------------------ #


class TestImportCompat:
    def test_svd_import_from_contractor(self):
        """truncated_svd is still importable from contractor."""
        from tenax.contraction.contractor import truncated_svd

        assert callable(truncated_svd)

    def test_qr_import_from_contractor(self):
        """qr_decompose is still importable from contractor."""
        from tenax.contraction.contractor import qr_decompose

        assert callable(qr_decompose)

    def test_svd_from_init(self):
        """svd is importable from tenax top-level."""
        from tenax import svd as svd_init  # noqa: F811

        assert callable(svd_init)

    def test_qr_from_init(self):
        """qr is importable from tenax top-level."""
        from tenax import qr as qr_init  # noqa: F811

        assert callable(qr_init)

    def test_eigh_from_init(self):
        """eigh is importable from tenax top-level."""
        from tenax import eigh as eigh_init  # noqa: F811

        assert callable(eigh_init)

    def test_svd_is_truncated_svd(self):
        """svd and truncated_svd are the same function."""
        from tenax import svd as svd_top  # noqa: F811
        from tenax import truncated_svd
        from tenax.linalg import svd as linalg_svd

        assert truncated_svd is linalg_svd
        assert svd_top is linalg_svd
