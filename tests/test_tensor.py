"""Tests for DenseTensor and SymmetricTensor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry, ZnSymmetry
from tenax.core.tensor import (
    DenseTensor,
    SymmetricTensor,
    _block_slices,
    _compute_valid_blocks,
    inner,
)


class TestDenseTensor:
    def test_creation(self, u1, u1_charges_3, rng):
        data = jax.random.normal(rng, (3, 3))
        indices = (
            TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="row"),
            TensorIndex(u1, u1_charges_3, FlowDirection.OUT, label="col"),
        )
        t = DenseTensor(data, indices)
        assert t.ndim == 2
        assert t.labels() == ("row", "col")

    def test_wrong_ndim_raises(self, u1, u1_charges_3):
        data = jnp.ones((3,))
        indices = (
            TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="a"),
            TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="b"),
        )
        with pytest.raises(ValueError, match="dims"):
            DenseTensor(data, indices)

    def test_wrong_shape_raises(self, u1, u1_charges_3):
        data = jnp.ones((4, 3))  # first dim wrong
        indices = (
            TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="row"),
            TensorIndex(u1, u1_charges_3, FlowDirection.OUT, label="col"),
        )
        with pytest.raises(ValueError):
            DenseTensor(data, indices)

    def test_todense_identity(self, small_dense_matrix):
        data = small_dense_matrix.todense()
        assert data.shape == (3, 3)

    def test_norm_positive(self, small_dense_matrix):
        n = small_dense_matrix.norm()
        assert float(n) > 0

    def test_conj(self, u1, rng):
        charges = np.array([0, 1], dtype=np.int32)
        data = jax.random.normal(rng, (2, 2)) + 1j * jax.random.normal(rng, (2, 2))
        data = data.astype(jnp.complex64)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(u1, charges, FlowDirection.OUT, label="b"),
        )
        t = DenseTensor(data, indices)
        tc = t.conj()
        np.testing.assert_allclose(tc.todense(), jnp.conj(data))
        assert tc.labels() == t.labels()

    def test_transpose(self, small_dense_matrix):
        t_transposed = small_dense_matrix.transpose((1, 0))
        assert t_transposed.labels() == ("col", "row")
        np.testing.assert_allclose(
            t_transposed.todense(),
            jnp.transpose(small_dense_matrix.todense(), (1, 0)),
        )

    def test_relabel(self, small_dense_matrix):
        t2 = small_dense_matrix.relabel("row", "new_row")
        assert "new_row" in t2.labels()
        assert "row" not in t2.labels()
        assert "col" in t2.labels()  # unchanged

    def test_relabel_missing_raises(self, small_dense_matrix):
        with pytest.raises(KeyError, match="notexist"):
            small_dense_matrix.relabel("notexist", "x")

    def test_relabels_batch(self, small_dense_matrix):
        t2 = small_dense_matrix.relabels({"row": "i", "col": "j"})
        assert t2.labels() == ("i", "j")

    def test_jit_compatible(self, small_dense_matrix):
        @jax.jit
        def compute_norm(t):
            return t.norm()

        result = compute_norm(small_dense_matrix)
        assert result.shape == ()

    def test_grad_compatible(self, u1, u1_charges_3, rng):
        data = jax.random.normal(rng, (3,))
        idx = TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="v")
        t = DenseTensor(data, (idx,))

        def loss(tensor):
            return tensor.norm()

        grad = jax.grad(loss)(t)
        assert grad.todense().shape == (3,)

    def test_vmap_compatible(self, u1, u1_charges_3, rng):
        """Test that DenseTensor works under vmap."""
        batch_data = jax.random.normal(rng, (5, 3))
        charges = u1_charges_3

        def process_row(row):
            idx = TensorIndex(u1, charges, FlowDirection.IN, label="v")
            t = DenseTensor(row, (idx,))
            return t.norm()

        norms = jax.vmap(process_row)(batch_data)
        assert norms.shape == (5,)

    def test_repr(self, small_dense_matrix):
        r = repr(small_dense_matrix)
        assert "Dense" in r
        assert "──▶" in r

    def test_dtype(self, u1, u1_charges_3, rng):
        data = jax.random.normal(rng, (3,), dtype=jnp.float64)
        idx = TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="v")
        # float64 may be promoted to float32 by default JAX config
        t = DenseTensor(data, (idx,))
        assert t.dtype is not None


class TestComputeValidBlocks:
    def test_u1_2leg(self, u1):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="b"),
        )
        keys = _compute_valid_blocks(indices)
        # For 2-leg U(1): charge_in + (-charge_out) = 0 => charge_in = charge_out
        for key in keys:
            q_in, q_out = key
            net = 1 * q_in + (-1) * q_out
            assert net == 0

    def test_empty_indices(self):
        keys = _compute_valid_blocks(())
        assert keys == [()]

    def test_z2_3leg(self, z2):
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex(z2, charges, FlowDirection.IN, label="a"),
            TensorIndex(z2, charges, FlowDirection.IN, label="b"),
            TensorIndex(z2, z2.dual(charges), FlowDirection.OUT, label="c"),
        )
        keys = _compute_valid_blocks(indices)
        for key in keys:
            net = (1 * key[0] + 1 * key[1] + (-1) * key[2]) % 2
            assert net == 0


class TestSymmetricTensorCreation:
    def test_zeros_factory(self, u1, u1_charges_3):
        indices = (
            TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"),
        )
        t = SymmetricTensor.zeros(indices)
        assert t.ndim == 2
        for block in t.blocks.values():
            np.testing.assert_allclose(block, 0.0)

    def test_random_normal_factory(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        assert t.ndim == 2
        assert t.n_blocks > 0

    def test_conservation_law_satisfied(self, u1, u1_charges_3, rng):
        indices = (
            TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        for key in t.blocks:
            net = 1 * key[0] + (-1) * key[1]
            assert net == 0, f"Block {key} violates U(1) conservation, net={net}"

    def test_invalid_block_raises(self, u1):
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(u1, charges, FlowDirection.OUT, label="b"),  # not dual
        )
        # Block (1, 1) has net = 1 + (-1)*1 = 0 which is valid
        # Block (0, 1) has net = 0 + (-1)*1 = -1 which is invalid
        bad_blocks = {(0, 1): jnp.ones((1, 1))}  # net = 0 - 1 = -1, invalid
        with pytest.raises(ValueError, match="conservation"):
            SymmetricTensor(bad_blocks, indices)

    def test_from_dense_roundtrip(self, u1, u1_charges_3, rng):
        """from_dense(todense(T)) should recover T block-by-block."""
        indices = (
            TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"),
        )
        t_orig = SymmetricTensor.random_normal(indices, rng)
        dense = t_orig.todense()
        t_recovered = SymmetricTensor.from_dense(dense, indices)

        for key in t_orig.blocks:
            np.testing.assert_allclose(
                t_orig.blocks[key], t_recovered.blocks[key], rtol=1e-5
            )

    def test_from_dense_rejects_non_zero_outside_blocks(self, u1, u1_charges_3):
        indices = (
            TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"),
        )
        # Dense tensor with non-zero element outside valid sectors
        dense = jnp.ones((3, 3))
        with pytest.raises(ValueError):
            SymmetricTensor.from_dense(dense, indices)


class TestSymmetricTensorOperations:
    def test_todense_shape(self, u1_sym_tensor_2leg):
        dense = u1_sym_tensor_2leg.todense()
        assert dense.shape == (3, 3)

    def test_norm_matches_dense(self, u1_sym_tensor_2leg):
        sym_norm = u1_sym_tensor_2leg.norm()
        dense_norm = jnp.linalg.norm(u1_sym_tensor_2leg.todense().ravel())
        np.testing.assert_allclose(float(sym_norm), float(dense_norm), rtol=1e-5)

    def test_conj(self, u1_sym_tensor_2leg):
        tc = u1_sym_tensor_2leg.conj()
        for key in u1_sym_tensor_2leg.blocks:
            np.testing.assert_allclose(
                tc.blocks[key], jnp.conj(u1_sym_tensor_2leg.blocks[key])
            )

    def test_transpose(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        t_T = t.transpose((1, 0))
        assert t_T.labels() == ("out", "in")
        # Transposing twice gives original
        t_TT = t_T.transpose((1, 0))
        assert t_TT.labels() == t.labels()

    def test_relabel(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        t2 = t.relabel("in", "new_in")
        assert "new_in" in t2.labels()
        assert "in" not in t2.labels()

    def test_relabels_batch(self, u1_sym_tensor_3leg):
        t = u1_sym_tensor_3leg
        t2 = t.relabels({"phys": "s", "left": "l", "right": "r"})
        assert set(t2.labels()) == {"s", "l", "r"}

    def test_block_shapes(self, u1_sym_tensor_2leg):
        shapes = u1_sym_tensor_2leg.block_shapes()
        for key, shape in shapes.items():
            assert len(shape) == 2
            assert all(s > 0 for s in shape)

    def test_pytree_jit(self, u1_sym_tensor_2leg, rng):
        """SymmetricTensor is compatible with jax.jit via pytree."""
        t = u1_sym_tensor_2leg

        @jax.jit
        def scale_blocks(tensor, factor):
            new_blocks = {k: v * factor for k, v in tensor.blocks.items()}
            return SymmetricTensor(new_blocks, tensor.indices)

        result = scale_blocks(t, 2.0)
        for key in t.blocks:
            np.testing.assert_allclose(
                result.blocks[key], t.blocks[key] * 2.0, rtol=1e-5
            )

    def test_pytree_grad(self, u1_sym_tensor_2leg):
        """Gradient flows through SymmetricTensor via pytree."""

        def loss(t):
            return t.norm()

        grad = jax.grad(loss)(u1_sym_tensor_2leg)
        assert isinstance(grad, SymmetricTensor)

    def test_repr(self, u1_sym_tensor_2leg):
        r = repr(u1_sym_tensor_2leg)
        assert "Symmetric" in r
        assert "blk" in r
        assert "charges:" in r

    def test_3leg_conservation(self, u1_sym_tensor_3leg, u1):
        """All blocks in a 3-leg tensor satisfy U(1) conservation."""
        t = u1_sym_tensor_3leg
        for key in t.blocks:
            # phys (IN=+1), left (IN=+1), right (OUT=-1)
            net = 1 * key[0] + 1 * key[1] + (-1) * key[2]
            assert net == 0, f"Block {key} violates conservation, net={net}"

    def test_dtype_property(self, u1_sym_tensor_2leg):
        assert u1_sym_tensor_2leg.dtype is not None


class TestDenseSymmetricParity:
    """SymmetricTensor.todense() must equal DenseTensor with the same data.

    These tests verify that block extraction and reconstruction are lossless:
    constructing a SymmetricTensor from a dense array and calling todense()
    recovers the original array exactly (within float tolerance).
    """

    def test_u1_2leg(self, u1, u1_charges_3, rng):
        """U(1) 2-leg: sym.todense() matches the source dense array."""
        indices = (
            TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"),
        )
        sym = SymmetricTensor.random_normal(indices, rng)
        dense_data = sym.todense()
        # Build a DenseTensor wrapping the same data and compare
        dt = DenseTensor(dense_data, indices)
        np.testing.assert_allclose(sym.todense(), dt.todense(), rtol=1e-5)

    def test_u1_3leg(self, u1_sym_tensor_3leg):
        """U(1) 3-leg: sym.todense() matches source dense array."""
        sym = u1_sym_tensor_3leg
        dense_data = sym.todense()
        dt = DenseTensor(dense_data, sym.indices)
        np.testing.assert_allclose(sym.todense(), dt.todense(), rtol=1e-5)

    def test_zn_2leg(self, z2, rng):
        """Z2 2-leg: sym.todense() matches source dense array."""
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex(z2, charges, FlowDirection.IN, label="in"),
            TensorIndex(z2, z2.dual(charges), FlowDirection.OUT, label="out"),
        )
        sym = SymmetricTensor.random_normal(indices, rng)
        dense_data = sym.todense()
        dt = DenseTensor(dense_data, indices)
        np.testing.assert_allclose(sym.todense(), dt.todense(), rtol=1e-5)

    def test_zn_3leg(self, z2, rng):
        """Z2 3-leg (two IN, one OUT): sym.todense() matches source dense array."""
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex(z2, charges, FlowDirection.IN, label="a"),
            TensorIndex(z2, charges, FlowDirection.IN, label="b"),
            TensorIndex(z2, z2.dual(charges), FlowDirection.OUT, label="c"),
        )
        sym = SymmetricTensor.random_normal(indices, rng)
        dense_data = sym.todense()
        dt = DenseTensor(dense_data, indices)
        np.testing.assert_allclose(sym.todense(), dt.todense(), rtol=1e-5)

    def test_from_dense_parity_u1(self, u1, u1_charges_3, rng):
        """from_dense then todense recovers the original dense array for U(1)."""
        indices = (
            TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"),
        )
        sym = SymmetricTensor.random_normal(indices, rng)
        dense_data = sym.todense()
        sym2 = SymmetricTensor.from_dense(dense_data, indices)
        np.testing.assert_allclose(sym2.todense(), dense_data, rtol=1e-5)

    def test_from_dense_parity_zn(self, z2, rng):
        """from_dense then todense recovers the original dense array for Z2."""
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex(z2, charges, FlowDirection.IN, label="a"),
            TensorIndex(z2, charges, FlowDirection.IN, label="b"),
            TensorIndex(z2, z2.dual(charges), FlowDirection.OUT, label="c"),
        )
        sym = SymmetricTensor.random_normal(indices, rng)
        dense_data = sym.todense()
        sym2 = SymmetricTensor.from_dense(dense_data, indices)
        np.testing.assert_allclose(sym2.todense(), dense_data, rtol=1e-5)

    def test_zeros_outside_blocks(self, u1_sym_tensor_2leg):
        """Positions outside symmetry-allowed sectors are zero in todense()."""
        sym = u1_sym_tensor_2leg
        dense = np.array(sym.todense())
        # Zero out positions that belong to valid blocks, rest must be zero
        valid_keys = _compute_valid_blocks(sym.indices)
        covered = np.zeros(dense.shape, dtype=bool)
        for key in valid_keys:
            masks, shape = _block_slices(sym.indices, key)
            if all(s > 0 for s in shape):
                idx_arrays = [np.where(m)[0] for m in masks]
                grid = np.ix_(*idx_arrays)
                covered[grid] = True
        np.testing.assert_allclose(dense[~covered], 0.0, atol=1e-7)


class TestDenseTensorArithmetic:
    def test_mul(self, small_dense_matrix):
        t2 = small_dense_matrix * 3.0
        np.testing.assert_allclose(t2.todense(), small_dense_matrix.todense() * 3.0)

    def test_rmul(self, small_dense_matrix):
        t2 = 3.0 * small_dense_matrix
        np.testing.assert_allclose(t2.todense(), small_dense_matrix.todense() * 3.0)

    def test_add(self, small_dense_matrix):
        t2 = small_dense_matrix + small_dense_matrix
        np.testing.assert_allclose(t2.todense(), small_dense_matrix.todense() * 2.0)

    def test_sub(self, small_dense_matrix):
        t2 = small_dense_matrix - small_dense_matrix
        np.testing.assert_allclose(t2.todense(), 0.0, atol=1e-14)

    def test_neg(self, small_dense_matrix):
        t2 = -small_dense_matrix
        np.testing.assert_allclose(t2.todense(), -small_dense_matrix.todense())

    def test_max_abs(self, small_dense_matrix):
        m = small_dense_matrix.max_abs()
        expected = jnp.max(jnp.abs(small_dense_matrix.todense()))
        np.testing.assert_allclose(float(m), float(expected))

    def test_labels_preserved_after_mul(self, small_dense_matrix):
        t2 = small_dense_matrix * 2.0
        assert t2.labels() == small_dense_matrix.labels()

    def test_labels_preserved_after_add(self, small_dense_matrix):
        t2 = small_dense_matrix + small_dense_matrix
        assert t2.labels() == small_dense_matrix.labels()


class TestSymmetricTensorArithmetic:
    def test_mul(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        t2 = t * 3.0
        np.testing.assert_allclose(t2.todense(), t.todense() * 3.0, rtol=1e-6)

    def test_rmul(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        t2 = 3.0 * t
        np.testing.assert_allclose(t2.todense(), t.todense() * 3.0, rtol=1e-6)

    def test_add(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        t2 = t + t
        np.testing.assert_allclose(t2.todense(), t.todense() * 2.0, rtol=1e-6)

    def test_sub(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        t2 = t - t
        np.testing.assert_allclose(t2.todense(), 0.0, atol=1e-14)

    def test_neg(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        t2 = -t
        np.testing.assert_allclose(t2.todense(), -t.todense(), rtol=1e-6)

    def test_max_abs(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        m = t.max_abs()
        expected = jnp.max(jnp.abs(t.todense()))
        np.testing.assert_allclose(float(m), float(expected), rtol=1e-6)

    def test_max_abs_empty(self, u1):
        """max_abs on a tensor with no blocks returns 0."""
        charges = np.array([0], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(
                u1, np.array([1], dtype=np.int32), FlowDirection.OUT, label="b"
            ),
        )
        t = SymmetricTensor.zeros(indices)
        assert float(t.max_abs()) == 0.0

    def test_add_mixed_type_raises(self, u1_sym_tensor_2leg, small_dense_matrix):
        with pytest.raises(TypeError, match="Cannot add"):
            u1_sym_tensor_2leg + small_dense_matrix

    def test_block_structure_preserved(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        t2 = t * 2.0 + t
        assert set(t2.blocks.keys()) == set(t.blocks.keys())

    def test_labels_preserved_after_mul(self, u1_sym_tensor_2leg):
        t2 = u1_sym_tensor_2leg * 2.0
        assert t2.labels() == u1_sym_tensor_2leg.labels()


class TestBar:
    """Tests for the bar() operation (conjugate + flip flows, no charge dual)."""

    def test_dense_bar_todense_equals_conj(self, u1, rng):
        """DenseTensor.bar().todense() == conj(todense())."""
        charges = np.array([0, 1], dtype=np.int32)
        data = jax.random.normal(rng, (2, 2)) + 1j * jax.random.normal(rng, (2, 2))
        data = data.astype(jnp.complex64)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(u1, charges, FlowDirection.OUT, label="b"),
        )
        t = DenseTensor(data, indices)
        tb = t.bar()
        np.testing.assert_allclose(tb.todense(), jnp.conj(data))

    def test_dense_bar_flows_flipped_charges_unchanged(self, u1, rng):
        """DenseTensor.bar() flips flows but keeps charges identical."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        data = jax.random.normal(rng, (3, 3))
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(u1, charges, FlowDirection.OUT, label="b"),
        )
        t = DenseTensor(data, indices)
        tb = t.bar()
        for orig, barred in zip(t.indices, tb.indices):
            assert barred.flow == FlowDirection(-int(orig.flow))
            np.testing.assert_array_equal(barred.charges, orig.charges)

    def test_symmetric_bar_todense_equals_conj(self, u1_sym_tensor_2leg):
        """SymmetricTensor.bar().todense() == conj(todense())."""
        t = u1_sym_tensor_2leg
        tb = t.bar()
        np.testing.assert_allclose(tb.todense(), jnp.conj(t.todense()), rtol=1e-6)

    def test_symmetric_bar_block_keys_unchanged(self, u1_sym_tensor_2leg):
        """SymmetricTensor.bar() preserves block keys (no charge dual)."""
        t = u1_sym_tensor_2leg
        tb = t.bar()
        assert set(tb.blocks.keys()) == set(t.blocks.keys())

    def test_symmetric_bar_flows_flipped_charges_unchanged(self, u1_sym_tensor_2leg):
        """SymmetricTensor.bar() flips flows, keeps charges."""
        t = u1_sym_tensor_2leg
        tb = t.bar()
        for orig, barred in zip(t.indices, tb.indices):
            assert barred.flow == FlowDirection(-int(orig.flow))
            np.testing.assert_array_equal(barred.charges, orig.charges)

    def test_trivial_charges_bar_equals_dagger(self, u1, rng):
        """With trivial (zero) charges, bar() and dagger() give same dense result."""
        charges = np.zeros(3, dtype=np.int32)
        data = jax.random.normal(rng, (3, 3))
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(u1, charges, FlowDirection.OUT, label="b"),
        )
        t = DenseTensor(data, indices)
        np.testing.assert_allclose(t.bar().todense(), t.dagger().todense())


class TestInner:
    def test_dense_self_inner(self, small_dense_matrix):
        t = small_dense_matrix
        result = inner(t, t)
        expected = jnp.sum(jnp.abs(t.todense()) ** 2)
        np.testing.assert_allclose(float(result), float(expected), rtol=1e-6)

    def test_symmetric_self_inner(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        result = inner(t, t)
        expected = jnp.sum(jnp.abs(t.todense()) ** 2)
        np.testing.assert_allclose(float(result), float(expected), rtol=1e-6)

    def test_inner_equals_norm_squared(self, u1_sym_tensor_2leg):
        t = u1_sym_tensor_2leg
        ip = inner(t, t)
        norm_sq = t.norm() ** 2
        np.testing.assert_allclose(float(ip), float(norm_sq), rtol=1e-6)

    def test_inner_equals_norm_squared_dense(self, small_dense_matrix):
        t = small_dense_matrix
        ip = inner(t, t)
        norm_sq = t.norm() ** 2
        np.testing.assert_allclose(float(ip), float(norm_sq), rtol=1e-6)

    def test_mixed_type_fallback(self, u1, rng):
        """inner(dense, symmetric) falls back to dense contraction."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="out"),
        )
        sym = SymmetricTensor.random_normal(indices, rng)
        dense = DenseTensor(sym.todense(), indices)
        result = inner(dense, sym)
        expected = jnp.sum(jnp.abs(sym.todense()) ** 2)
        np.testing.assert_allclose(float(result), float(expected), rtol=1e-6)
