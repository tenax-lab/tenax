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
