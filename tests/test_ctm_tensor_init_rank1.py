"""Init invariants for bug 3a (chi_init=1, rank-1 padded init)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_init import (
    _init_symmetric_standard_corner,
    _init_symmetric_standard_edge,
    _make_dense_standard_edge,
    initialize_ctm_tensor_env,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


def _peps_dense(D: int = 2, d: int = 2):
    sym = U1Symmetry()
    rng = np.random.RandomState(0)
    data = jnp.array(rng.standard_normal((D, D, D, D, d)))
    indices = (
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="d"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="r"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label="phys"
        ),
    )
    return DenseTensor(data, indices)


@pytest.mark.parametrize("D, chi", [(2, 4), (2, 16), (3, 9)])
def test_initialize_dense_corner_rank1(D, chi):
    """Standard-CTM dense corner is rank-1: only entry (0, 0) is non-zero."""
    A = _peps_dense(D=D)
    env = initialize_ctm_tensor_env(A, chi)
    for C in (env.C1, env.C2, env.C3, env.C4):
        arr = np.asarray(C._data)
        assert arr.shape == (chi, chi)
        assert arr[0, 0] == pytest.approx(1.0)
        # Everything else exactly zero.
        mask = np.ones_like(arr, dtype=bool)
        mask[0, 0] = False
        assert np.all(arr[mask] == 0), "corner has non-zero entries outside (0, 0)"


@pytest.mark.parametrize("D, chi", [(2, 4), (2, 16), (3, 9)])
def test_initialize_dense_edge_rank1(D, chi):
    """Standard-CTM dense edge has only D non-zero entries at (0, j*(D+1), 0)."""
    A = _peps_dense(D=D)
    env = initialize_ctm_tensor_env(A, chi)
    for T in (env.T1, env.T2, env.T3, env.T4):
        arr = np.asarray(T._data)
        assert arr.shape == (chi, D * D, chi)
        # Expected: T[0, diag_idx, 0] = 1, everything else 0.
        for j in range(D):
            assert arr[0, j * (D + 1), 0] == pytest.approx(1.0)
        mask = np.ones_like(arr, dtype=bool)
        for j in range(D):
            mask[0, j * (D + 1), 0] = False
        assert np.all(arr[mask] == 0), (
            "edge has non-zero entries outside the rank-1 slot"
        )


def _peps_symmetric(D: int = 2, d: int = 2):
    sym = U1Symmetry()
    rng = np.random.RandomState(99)
    data = jnp.array(rng.standard_normal((D, D, D, D, d)))
    indices = (
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="d"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="r"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label="phys"
        ),
    )
    return SymmetricTensor.from_dense(data, indices)


@pytest.mark.parametrize("D, chi", [(2, 4), (2, 16), (3, 9)])
def test_initialize_symmetric_corner_rank1(D, chi):
    """Symmetric standard corner is rank-1: dense view has only (0, 0) = 1."""
    A = _peps_symmetric(D=D)
    env = initialize_ctm_tensor_env(A, chi)
    for C in (env.C1, env.C2, env.C3, env.C4):
        assert isinstance(C, SymmetricTensor)
        arr = np.asarray(C.todense())
        assert arr.shape == (chi, chi)
        assert arr[0, 0] == pytest.approx(1.0)
        mask = np.ones_like(arr, dtype=bool)
        mask[0, 0] = False
        assert np.all(arr[mask] == 0)


@pytest.mark.parametrize("D, chi", [(2, 4), (2, 16), (3, 9)])
def test_initialize_symmetric_edge_rank1(D, chi):
    """Symmetric standard edge: dense view has only D non-zero entries at (0, j*(D+1), 0)."""
    A = _peps_symmetric(D=D)
    env = initialize_ctm_tensor_env(A, chi)
    for T in (env.T1, env.T2, env.T3, env.T4):
        assert isinstance(T, SymmetricTensor)
        arr = np.asarray(T.todense())
        assert arr.shape == (chi, D * D, chi)
        for j in range(D):
            assert arr[0, j * (D + 1), 0] == pytest.approx(1.0)
        mask = np.ones_like(arr, dtype=bool)
        for j in range(D):
            mask[0, j * (D + 1), 0] = False
        assert np.all(arr[mask] == 0)
