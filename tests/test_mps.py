"""Tests for FiniteMPS class."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

IN = FlowDirection.IN
OUT = FlowDirection.OUT


def _make_dense_mps(L=4, d=2, chi=3, key=None):
    """Helper: build a random dense MPS as list[DenseTensor]."""
    if key is None:
        key = jax.random.PRNGKey(0)
    u1 = U1Symmetry()
    tensors = []
    for i in range(L):
        if i == 0:
            shape = (d, chi)
            indices = (
                TensorIndex(u1, np.zeros(d, dtype=np.int32), IN, label=f"p{i}"),
                TensorIndex(
                    u1, np.zeros(chi, dtype=np.int32), OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == L - 1:
            shape = (chi, d)
            indices = (
                TensorIndex(
                    u1, np.zeros(chi, dtype=np.int32), IN, label=f"v{i - 1}_{i}"
                ),
                TensorIndex(u1, np.zeros(d, dtype=np.int32), IN, label=f"p{i}"),
            )
        else:
            shape = (chi, d, chi)
            indices = (
                TensorIndex(
                    u1, np.zeros(chi, dtype=np.int32), IN, label=f"v{i - 1}_{i}"
                ),
                TensorIndex(u1, np.zeros(d, dtype=np.int32), IN, label=f"p{i}"),
                TensorIndex(
                    u1, np.zeros(chi, dtype=np.int32), OUT, label=f"v{i}_{i + 1}"
                ),
            )
        key, subkey = jax.random.split(key)
        data = jax.random.normal(subkey, shape)
        tensors.append(DenseTensor(data, indices))
    return tensors


class TestFiniteMPSConstruction:
    def test_from_tensors_basic(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4, d=2, chi=3)
        mps = FiniteMPS.from_tensors(tensors)
        assert len(mps) == 4
        assert mps.orth_center is None
        assert mps.singular_values == [None, None, None]

    def test_from_tensors_with_orth_center(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4)
        mps = FiniteMPS.from_tensors(tensors, orth_center=2)
        assert mps.orth_center == 2

    def test_len(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=6))
        assert mps.L == 6
        assert len(mps) == 6

    def test_getitem(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4)
        mps = FiniteMPS.from_tensors(tensors)
        assert mps[0] is tensors[0]
        assert mps[3] is tensors[3]

    def test_setitem_invalidates_orth_center(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4)
        mps = FiniteMPS.from_tensors(tensors, orth_center=2)
        assert mps.orth_center == 2
        mps[1] = tensors[1]  # replace with same tensor
        assert mps.orth_center is None

    def test_iter(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4)
        mps = FiniteMPS.from_tensors(tensors)
        assert list(mps) == tensors


class TestFiniteMPSProperties:
    def test_bond_dims(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        assert mps.bond_dims == [3, 3, 3]

    def test_phys_dims(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        assert mps.phys_dims == [2, 2, 2, 2]

    def test_max_bond_dim(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        assert mps.max_bond_dim == 3

    def test_is_symmetric_dense(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4))
        assert mps.is_symmetric is False


class TestFiniteMPSCanonicalize:
    def _check_left_canonical(self, tensor):
        """Check that tensor is left-isometric: A^dag A = I on the bond index."""
        d = tensor.todense()
        if d.ndim == 2:
            mat = d.reshape(-1, d.shape[-1])
        else:
            mat = d.reshape(-1, d.shape[-1])
        eye = mat.conj().T @ mat
        np.testing.assert_allclose(eye, np.eye(eye.shape[0]), atol=1e-12)

    def _check_right_canonical(self, tensor):
        """Check that tensor is right-isometric: A A^dag = I on the bond index."""
        d = tensor.todense()
        if d.ndim == 2:
            mat = d.reshape(d.shape[0], -1)
        else:
            mat = d.reshape(d.shape[0], -1)
        eye = mat @ mat.conj().T
        np.testing.assert_allclose(eye, np.eye(eye.shape[0]), atol=1e-12)

    def test_right_canonicalize_dense(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=6, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)
        mps_r = mps.right_canonicalize()
        assert mps_r.orth_center == 0
        for i in range(1, 6):
            self._check_right_canonical(mps_r[i])

    def test_left_canonicalize_dense(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=6, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)
        mps_l = mps.left_canonicalize()
        assert mps_l.orth_center == 5
        for i in range(5):
            self._check_left_canonical(mps_l[i])

    def test_canonicalize_center_dense(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=6, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)
        mps_c = mps.canonicalize(center=3)
        assert mps_c.orth_center == 3
        for i in range(3):
            self._check_left_canonical(mps_c[i])
        for i in range(4, 6):
            self._check_right_canonical(mps_c[i])

    def test_canonicalize_preserves_state(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4, d=2, chi=3)
        mps = FiniteMPS.from_tensors(tensors)
        mps_c = mps.canonicalize(center=2)

        def _to_statevector(mps_tensors):
            v = mps_tensors[0].todense()
            for t in mps_tensors[1:]:
                v = jnp.tensordot(v, t.todense(), axes=([-1], [0]))
            return v.ravel()

        psi_orig = _to_statevector(mps.tensors)
        psi_canon = _to_statevector(mps_c.tensors)
        psi_orig = psi_orig / jnp.linalg.norm(psi_orig)
        psi_canon = psi_canon / jnp.linalg.norm(psi_canon)
        overlap = jnp.abs(jnp.dot(psi_orig.conj(), psi_canon))
        np.testing.assert_allclose(float(overlap), 1.0, atol=1e-12)

    def test_canonicalize_singular_values(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=6, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors).canonicalize(center=3)
        assert mps.singular_values[3] is not None
        assert len(mps.singular_values[3]) > 0
        sv = np.array(mps.singular_values[3])
        assert np.all(sv >= -1e-15)
        np.testing.assert_allclose(sv, np.sort(sv)[::-1], atol=1e-15)
