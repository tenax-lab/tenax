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
