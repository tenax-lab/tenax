"""Tests for ``_flow_flip_no_conj`` in CTM absorption.

The helper flips every leg's flow direction without conjugating the data
(since ``_compute_2x2_projector`` already returns ``U^H``).  The dense
branch was correct; the SymmetricTensor branch crashed because the
constructor expects a blocks dict but received the flat data buffer
(see PR #422 review thread).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_tensor_moves import _flow_flip_no_conj
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


def test_flow_flip_no_conj_dense_preserves_data_flips_flow():
    sym = U1Symmetry()
    idx_a = TensorIndex.from_charges(
        sym, np.zeros(3, dtype=np.int32), FlowDirection.IN, label="a"
    )
    idx_b = TensorIndex.from_charges(
        sym, np.zeros(3, dtype=np.int32), FlowDirection.OUT, label="b"
    )
    rng = np.random.RandomState(0)
    data = jnp.array(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    t = DenseTensor(data, (idx_a, idx_b))

    flipped = _flow_flip_no_conj(t)

    assert isinstance(flipped, DenseTensor)
    np.testing.assert_array_equal(np.asarray(flipped._data), np.asarray(t._data))
    assert flipped.indices[0].flow == FlowDirection.OUT
    assert flipped.indices[1].flow == FlowDirection.IN


def test_flow_flip_no_conj_symmetric_trivial_charges():
    """SymmetricTensor with trivial U(1) charges: data preserved, flows flipped."""
    sym = U1Symmetry()
    idx_a = TensorIndex.from_charges(
        sym, np.zeros(3, dtype=np.int32), FlowDirection.IN, label="a"
    )
    idx_b = TensorIndex.from_charges(
        sym, np.zeros(3, dtype=np.int32), FlowDirection.OUT, label="b"
    )
    rng = np.random.RandomState(1)
    data = jnp.array(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    t = SymmetricTensor.from_dense(data, (idx_a, idx_b))

    flipped = _flow_flip_no_conj(t)

    assert isinstance(flipped, SymmetricTensor)
    assert flipped.indices[0].flow == FlowDirection.OUT
    assert flipped.indices[1].flow == FlowDirection.IN
    np.testing.assert_allclose(
        np.asarray(flipped.todense()),
        np.asarray(t.todense()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_flow_flip_no_conj_symmetric_nontrivial_charges():
    """SymmetricTensor with non-trivial U(1) charges: keys preserved (flipping every
    flow leaves ``sum(flow_i * c_i) = 0`` invariant), block data uncon­jugated."""
    sym = U1Symmetry()
    idx_a = TensorIndex.from_charges(
        sym, np.array([0, 1], dtype=np.int32), FlowDirection.OUT, label="a"
    )
    idx_b = TensorIndex.from_charges(
        sym, np.array([0, 1], dtype=np.int32), FlowDirection.IN, label="b"
    )
    rng = np.random.RandomState(7)
    raw = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    # Conservation (OUT, IN) with charges (0/1, 0/1) requires c_a == c_b,
    # so only diagonal entries belong to allowed sectors.
    raw[0, 1] = 0.0
    raw[1, 0] = 0.0
    data = jnp.array(raw)
    t = SymmetricTensor.from_dense(data, (idx_a, idx_b))

    flipped = _flow_flip_no_conj(t)

    assert isinstance(flipped, SymmetricTensor)
    assert flipped.indices[0].flow == FlowDirection.IN
    assert flipped.indices[1].flow == FlowDirection.OUT
    # Block keys unchanged.
    assert set(flipped._block_keys) == set(t._block_keys)
    for k in t._block_keys:
        np.testing.assert_allclose(
            np.asarray(flipped.blocks[k]),
            np.asarray(t.blocks[k]),
            rtol=1e-12,
            atol=1e-12,
        )
