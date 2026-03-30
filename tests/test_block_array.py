"""Tests for BlockArray: lightweight numpy-backed block-sparse array."""

import numpy as np
import pytest

from tenax import U1Symmetry
from tenax.algorithms._block_array import (
    BlockArray,
    ba_add,
    ba_conj,
    ba_inner,
    ba_norm,
    ba_scale,
    ba_sub,
    ba_to_symmetric,
    symmetric_to_ba,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import SymmetricTensor


@pytest.fixture
def simple_ba():
    """A BlockArray with two blocks for basic arithmetic tests."""
    sym = U1Symmetry()
    # charges: [0, 0, 1, 1, 1] -> 2 states charge 0, 3 states charge 1
    idx_a = TensorIndex(
        sym, np.array([0, 0, 1, 1, 1], dtype=np.int32), FlowDirection.IN, label="a"
    )
    # charges: [0, 0, 0, 0, 1, 1] -> 4 states charge 0, 2 states charge 1
    idx_b = TensorIndex(
        sym, np.array([0, 0, 0, 0, 1, 1], dtype=np.int32), FlowDirection.OUT, label="b"
    )
    blocks = {
        (0, 0): np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        (1, 1): np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
    }
    return BlockArray(blocks=blocks, indices=(idx_a, idx_b))


@pytest.fixture
def another_ba():
    """A BlockArray with same structure but different values."""
    sym = U1Symmetry()
    idx_a = TensorIndex(
        sym, np.array([0, 0, 1, 1, 1], dtype=np.int32), FlowDirection.IN, label="a"
    )
    idx_b = TensorIndex(
        sym, np.array([0, 0, 0, 0, 1, 1], dtype=np.int32), FlowDirection.OUT, label="b"
    )
    blocks = {
        (0, 0): np.ones((2, 4)),
        (1, 1): np.ones((3, 2)) * 2.0,
    }
    return BlockArray(blocks=blocks, indices=(idx_a, idx_b))


class TestBaScale:
    def test_scale_multiplies_all_blocks(self, simple_ba):
        result = ba_scale(simple_ba, 3.0)
        for key in simple_ba.blocks:
            np.testing.assert_allclose(result.blocks[key], simple_ba.blocks[key] * 3.0)

    def test_scale_zero(self, simple_ba):
        result = ba_scale(simple_ba, 0.0)
        for key in result.blocks:
            np.testing.assert_allclose(result.blocks[key], 0.0)

    def test_scale_preserves_indices(self, simple_ba):
        result = ba_scale(simple_ba, 2.0)
        assert result.indices == simple_ba.indices


class TestBaAdd:
    def test_add_same_keys(self, simple_ba, another_ba):
        result = ba_add(simple_ba, another_ba)
        for key in simple_ba.blocks:
            expected = simple_ba.blocks[key] + another_ba.blocks[key]
            np.testing.assert_allclose(result.blocks[key], expected)

    def test_add_different_keys_union(self):
        """Adding with disjoint keys should produce union."""
        sym = U1Symmetry()
        idx = TensorIndex(
            sym, np.array([0, 1], dtype=np.int32), FlowDirection.IN, label="x"
        )
        idx_out = TensorIndex(
            sym, np.array([0, 1], dtype=np.int32), FlowDirection.OUT, label="y"
        )
        ba1 = BlockArray(blocks={(0, 0): np.array([[1.0]])}, indices=(idx, idx_out))
        ba2 = BlockArray(blocks={(1, 1): np.array([[2.0]])}, indices=(idx, idx_out))
        result = ba_add(ba1, ba2)
        assert (0, 0) in result.blocks
        assert (1, 1) in result.blocks
        np.testing.assert_allclose(result.blocks[(0, 0)], [[1.0]])
        np.testing.assert_allclose(result.blocks[(1, 1)], [[2.0]])

    def test_add_preserves_indices(self, simple_ba, another_ba):
        result = ba_add(simple_ba, another_ba)
        assert result.indices == simple_ba.indices


class TestBaSub:
    def test_sub_same_keys(self, simple_ba, another_ba):
        result = ba_sub(simple_ba, another_ba)
        for key in simple_ba.blocks:
            expected = simple_ba.blocks[key] - another_ba.blocks[key]
            np.testing.assert_allclose(result.blocks[key], expected)


class TestBaInner:
    def test_inner_product(self, simple_ba):
        # Frobenius inner product = sum of element-wise products
        expected = sum(np.sum(b * b) for b in simple_ba.blocks.values())
        result = ba_inner(simple_ba, simple_ba)
        np.testing.assert_allclose(result, expected)

    def test_inner_different_arrays(self, simple_ba, another_ba):
        expected = sum(
            np.sum(simple_ba.blocks[k] * another_ba.blocks[k]) for k in simple_ba.blocks
        )
        result = ba_inner(simple_ba, another_ba)
        np.testing.assert_allclose(result, expected)


class TestBaNorm:
    def test_norm_matches_sqrt_inner(self, simple_ba):
        expected = np.sqrt(ba_inner(simple_ba, simple_ba))
        result = ba_norm(simple_ba)
        np.testing.assert_allclose(result, expected)


class TestBaConj:
    def test_conj_real_is_identity(self, simple_ba):
        result = ba_conj(simple_ba)
        for key in simple_ba.blocks:
            np.testing.assert_allclose(result.blocks[key], simple_ba.blocks[key])

    def test_conj_preserves_indices(self, simple_ba):
        result = ba_conj(simple_ba)
        assert result.indices == simple_ba.indices


class TestRoundtrip:
    def test_symmetric_to_ba_and_back(self):
        """SymmetricTensor -> BlockArray -> SymmetricTensor preserves data."""
        sym = U1Symmetry()
        # charges for idx_a: [0, 0, 1, 1, 1]  (2 with q=0, 3 with q=1)
        idx_a = TensorIndex(
            sym, np.array([0, 0, 1, 1, 1], dtype=np.int32), FlowDirection.IN, label="a"
        )
        # charges for idx_b: [0, 0, 0, 0, 1, 1]  (4 with q=0, 2 with q=1)
        idx_b = TensorIndex(
            sym,
            np.array([0, 0, 0, 0, 1, 1], dtype=np.int32),
            FlowDirection.OUT,
            label="b",
        )
        rng = np.random.default_rng(42)
        blocks = {
            (0, 0): rng.standard_normal((2, 4)),
            (1, 1): rng.standard_normal((3, 2)),
        }
        import jax.numpy as jnp

        jax_blocks = {k: jnp.array(v) for k, v in blocks.items()}
        t = SymmetricTensor(jax_blocks, (idx_a, idx_b))

        ba = symmetric_to_ba(t)
        assert isinstance(ba, BlockArray)
        assert ba.indices == t.indices

        t2 = ba_to_symmetric(ba)
        assert isinstance(t2, SymmetricTensor)
        for key in t.blocks:
            np.testing.assert_allclose(
                np.asarray(t2.blocks[key]),
                np.asarray(t.blocks[key]),
                rtol=1e-12,
            )
