"""Tests for DenseTensor and SymmetricTensor."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex, _net_charge
from tenax.core.symmetry import (
    FermionParity,
    ProductSymmetry,
    U1Symmetry,
    ZnSymmetry,
)
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
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="row"),
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.OUT, label="col"),
        )
        t = DenseTensor(data, indices)
        assert t.ndim == 2
        assert t.labels() == ("row", "col")

    def test_wrong_ndim_raises(self, u1, u1_charges_3):
        data = jnp.ones((3,))
        indices = (
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="b"),
        )
        with pytest.raises(ValueError, match="dims"):
            DenseTensor(data, indices)

    def test_wrong_shape_raises(self, u1, u1_charges_3):
        data = jnp.ones((4, 3))  # first dim wrong
        indices = (
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="row"),
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.OUT, label="col"),
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
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
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
        idx = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="v")
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
            idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="v")
            t = DenseTensor(row, (idx,))
            return t.norm()

        norms = jax.vmap(process_row)(batch_data)
        assert norms.shape == (5,)

    def test_repr(self, small_dense_matrix):
        r = repr(small_dense_matrix)
        assert "Dense" in r
        assert "──>" in r

    def test_dtype(self, u1, u1_charges_3, rng):
        data = jax.random.normal(rng, (3,), dtype=jnp.float64)
        idx = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="v")
        # float64 may be promoted to float32 by default JAX config
        t = DenseTensor(data, (idx,))
        assert t.dtype is not None


class TestComputeValidBlocks:
    def test_u1_2leg(self, u1):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="b"
            ),
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
            TensorIndex.from_charges(z2, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(z2, charges, FlowDirection.IN, label="b"),
            TensorIndex.from_charges(
                z2, z2.dual(charges), FlowDirection.OUT, label="c"
            ),
        )
        keys = _compute_valid_blocks(indices)
        for key in keys:
            net = (1 * key[0] + 1 * key[1] + (-1) * key[2]) % 2
            assert net == 0

    # The three tests above re-derive conservation from the keys they were
    # handed, so they can only catch keys that should not be there.  They say
    # nothing about *completeness*, which is the half that fails silently: a
    # missing key is a dropped block, not an error.  The ProductSymmetry cases
    # below assert the exact key set against exhaustive filtering, and cover
    # both branches the old implementation split on -- ``n_values() is None``,
    # where it did U(1) integer algebra on bit-packed charges, and the finite
    # branch, where ``flow_last * q`` was an integer negation of a bitfield.

    @pytest.mark.parametrize(
        "sym,pairs",
        [
            # n_values() is None -> the old code took the U(1) branch and
            # solved q_last = (target - prev) * flow_last in plain integers.
            (
                ProductSymmetry(FermionParity(), U1Symmetry()),
                [(n % 2, n) for n in range(4)],
            ),
            # Both factors finite -> the old code took the enumeration branch,
            # whose flow weighting was ``flow_last * q`` on a packed bitfield.
            (
                ProductSymmetry(ZnSymmetry(3), ZnSymmetry(2)),
                [(q1, q2) for q1 in range(3) for q2 in range(2)],
            ),
        ],
        ids=["fp_x_u1", "z3_x_z2"],
    )
    @pytest.mark.parametrize(
        "flows",
        [
            (FlowDirection.IN, FlowDirection.IN, FlowDirection.OUT),
            (FlowDirection.OUT, FlowDirection.IN, FlowDirection.IN),
            (FlowDirection.OUT, FlowDirection.OUT, FlowDirection.OUT),
        ],
        ids=["iio", "oii", "ooo"],
    )
    def test_product_symmetry_keys_are_exactly_the_conserving_ones(
        self, sym, pairs, flows
    ):
        sectors = np.unique(
            sym.canonicalize_charges(
                np.array([ProductSymmetry.encode(*p) for p in pairs], dtype=np.int32)
            )
        )
        indices = tuple(
            TensorIndex(
                sym,
                sectors,
                np.full(len(sectors), 2, dtype=np.int32),
                flow,
                label=lbl,
            )
            for flow, lbl in zip(flows, "abc")
        )

        expected = {
            key
            for key in itertools.product(*(idx.sectors.tolist() for idx in indices))
            if _net_charge(indices, key) == sym.identity()
        }
        got = _compute_valid_blocks(indices)

        assert len(got) == len(set(got)), got  # no duplicates
        assert set(got) == expected
        # The trap this is really for: a hollowed-out enumerator returns a
        # *subset* and nothing complains.  There must be something to drop.
        assert 0 < len(expected) < len(sectors) ** 3, (len(expected), len(sectors))


class TestSymmetricTensorCreation:
    def test_zeros_factory(self, u1, u1_charges_3):
        indices = (
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"
            ),
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
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"
            ),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        for key in t.blocks:
            net = 1 * key[0] + (-1) * key[1]
            assert net == 0, f"Block {key} violates U(1) conservation, net={net}"

    def test_invalid_block_raises(self, u1):
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(
                u1, charges, FlowDirection.OUT, label="b"
            ),  # not dual
        )
        # Block (1, 1) has net = 1 + (-1)*1 = 0 which is valid
        # Block (0, 1) has net = 0 + (-1)*1 = -1 which is invalid
        bad_blocks = {(0, 1): jnp.ones((1, 1))}  # net = 0 - 1 = -1, invalid
        with pytest.raises(ValueError, match="conservation"):
            SymmetricTensor(bad_blocks, indices)

    def test_from_dense_roundtrip(self, u1, u1_charges_3, rng):
        """from_dense(todense(T)) should recover T block-by-block."""
        indices = (
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"
            ),
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
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"
            ),
        )
        # Dense tensor with non-zero element outside valid sectors
        dense = jnp.ones((3, 3))
        with pytest.raises(ValueError):
            SymmetricTensor.from_dense(dense, indices)

    def test_flat_buffer_storage(self, u1_sym_tensor_2leg):
        """SymmetricTensor stores data in a single flat buffer."""
        t = u1_sym_tensor_2leg
        assert hasattr(t, "_data"), "Expected flat buffer _data attribute"
        assert isinstance(t._data, jax.Array), "_data should be a jax.Array"
        assert t._data.ndim == 1, "_data should be 1D"
        total = sum(v.size for v in t.blocks.values())
        assert t._data.size == total


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
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"
            ),
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
            TensorIndex.from_charges(z2, charges, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                z2, z2.dual(charges), FlowDirection.OUT, label="out"
            ),
        )
        sym = SymmetricTensor.random_normal(indices, rng)
        dense_data = sym.todense()
        dt = DenseTensor(dense_data, indices)
        np.testing.assert_allclose(sym.todense(), dt.todense(), rtol=1e-5)

    def test_zn_3leg(self, z2, rng):
        """Z2 3-leg (two IN, one OUT): sym.todense() matches source dense array."""
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(z2, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(z2, charges, FlowDirection.IN, label="b"),
            TensorIndex.from_charges(
                z2, z2.dual(charges), FlowDirection.OUT, label="c"
            ),
        )
        sym = SymmetricTensor.random_normal(indices, rng)
        dense_data = sym.todense()
        dt = DenseTensor(dense_data, indices)
        np.testing.assert_allclose(sym.todense(), dt.todense(), rtol=1e-5)

    def test_from_dense_parity_u1(self, u1, u1_charges_3, rng):
        """from_dense then todense recovers the original dense array for U(1)."""
        indices = (
            TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="out"
            ),
        )
        sym = SymmetricTensor.random_normal(indices, rng)
        dense_data = sym.todense()
        sym2 = SymmetricTensor.from_dense(dense_data, indices)
        np.testing.assert_allclose(sym2.todense(), dense_data, rtol=1e-5)

    def test_from_dense_parity_zn(self, z2, rng):
        """from_dense then todense recovers the original dense array for Z2."""
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(z2, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(z2, charges, FlowDirection.IN, label="b"),
            TensorIndex.from_charges(
                z2, z2.dual(charges), FlowDirection.OUT, label="c"
            ),
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

    def test_add_mismatched_labels_raises(self, u1):
        charges = np.array([0, 1], dtype=np.int32)
        idx_a = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a")
        idx_b = TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b")
        idx_c = TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="c")
        t1 = DenseTensor(jnp.ones((2, 2)), (idx_a, idx_b))
        t2 = DenseTensor(jnp.ones((2, 2)), (idx_a, idx_c))
        with pytest.raises(ValueError, match="label mismatch"):
            t1 + t2

    def test_add_mismatched_dim_raises(self, u1):
        idx_a = TensorIndex.from_charges(
            u1, np.array([0, 1], dtype=np.int32), FlowDirection.IN, label="a"
        )
        idx_b2 = TensorIndex.from_charges(
            u1, np.array([0, 1], dtype=np.int32), FlowDirection.OUT, label="b"
        )
        idx_b3 = TensorIndex.from_charges(
            u1, np.array([0, 1, 2], dtype=np.int32), FlowDirection.OUT, label="b"
        )
        t1 = DenseTensor(jnp.ones((2, 2)), (idx_a, idx_b2))
        t2 = DenseTensor(jnp.ones((2, 3)), (idx_a, idx_b3))
        with pytest.raises(ValueError, match="dimension mismatch"):
            t1 + t2

    def test_add_mismatched_flow_raises(self, u1):
        charges = np.array([0, 1], dtype=np.int32)
        idx_a = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a")
        idx_b_out = TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b")
        idx_b_in = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="b")
        t1 = DenseTensor(jnp.ones((2, 2)), (idx_a, idx_b_out))
        t2 = DenseTensor(jnp.ones((2, 2)), (idx_a, idx_b_in))
        with pytest.raises(ValueError, match="flow mismatch"):
            t1 + t2

    def test_sub_mismatched_labels_raises(self, u1):
        charges = np.array([0, 1], dtype=np.int32)
        idx_a = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a")
        idx_b = TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b")
        idx_c = TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="c")
        t1 = DenseTensor(jnp.ones((2, 2)), (idx_a, idx_b))
        t2 = DenseTensor(jnp.ones((2, 2)), (idx_a, idx_c))
        with pytest.raises(ValueError, match="label mismatch"):
            t1 - t2


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
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(
                u1, np.array([1], dtype=np.int32), FlowDirection.OUT, label="b"
            ),
        )
        t = SymmetricTensor.zeros(indices)
        assert float(t.max_abs()) == 0.0

    def test_add_mixed_type_raises(self, u1_sym_tensor_2leg, small_dense_matrix):
        with pytest.raises(TypeError, match="Cannot add"):
            u1_sym_tensor_2leg + small_dense_matrix

    def test_add_mismatched_labels_raises(self, u1):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t1 = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
            ),
            key=jax.random.PRNGKey(0),
        )
        t2 = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="c"),
            ),
            key=jax.random.PRNGKey(1),
        )
        with pytest.raises(ValueError, match="label mismatch"):
            t1 + t2

    def test_sub_mismatched_labels_raises(self, u1):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t1 = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
            ),
            key=jax.random.PRNGKey(0),
        )
        t2 = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="c"),
            ),
            key=jax.random.PRNGKey(1),
        )
        with pytest.raises(ValueError, match="label mismatch"):
            t1 - t2

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
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
        )
        t = DenseTensor(data, indices)
        tb = t.bar()
        np.testing.assert_allclose(tb.todense(), jnp.conj(data))

    def test_dense_bar_flows_flipped_charges_unchanged(self, u1, rng):
        """DenseTensor.bar() flips flows but keeps charges identical."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        data = jax.random.normal(rng, (3, 3))
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
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
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
        )
        t = DenseTensor(data, indices)
        np.testing.assert_allclose(t.bar().todense(), t.dagger().todense())

    def test_fermionic_bar_todense_equals_conj(self, rng):
        """FermionParity bar().todense() == conj(todense())."""
        from tenax.core.symmetry import FermionParity

        sym = FermionParity()
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="b"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        tb = t.bar()
        np.testing.assert_allclose(tb.todense(), jnp.conj(t.todense()), rtol=1e-6)

    def test_fermionic_bar_no_twist(self, rng):
        """FermionParity bar() differs from dagger() for odd-parity blocks."""
        from tenax.core.symmetry import FermionParity

        sym = FermionParity()
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="b"),
            TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="c"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        tb = t.bar()
        # bar() has no twist, dagger() has twist — generally different
        # but both should give valid todense that equals conj of original
        np.testing.assert_allclose(tb.todense(), jnp.conj(t.todense()), rtol=1e-6)
        # block keys should be same for bar (no charge dual)
        assert set(tb.blocks.keys()) == set(t.blocks.keys())
        # block keys differ for dagger (charges are dualled)
        # flows differ too: bar flips flows, dagger flips flows + duals charges

    def test_fermionic_u1_bar_todense_equals_conj(self, rng):
        """FermionicU1 bar().todense() == conj(todense())."""
        from tenax.core.symmetry import FermionicU1

        sym = FermionicU1()
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="b"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        tb = t.bar()
        np.testing.assert_allclose(tb.todense(), jnp.conj(t.todense()), rtol=1e-6)

    # bar_super was removed in #555: it was a per-block Koszul-phase
    # compensator that worked for the single-tensor self-contraction case
    # but broke multi-tensor inner products. Post-#555 the contractor no
    # longer auto-tracks Koszul, so the compensator is unnecessary; bra
    # construction is just bar().


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
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="out"
            ),
        )
        sym = SymmetricTensor.random_normal(indices, rng)
        dense = DenseTensor(sym.todense(), indices)
        result = inner(dense, sym)
        expected = jnp.sum(jnp.abs(sym.todense()) ** 2)
        np.testing.assert_allclose(float(result), float(expected), rtol=1e-6)


# --- ProductSymmetry end to end (#734 Task 4) ------------------------------
#
# Mixed-flow ``ProductSymmetry`` tensors did not work at all before #734: every
# charge-arithmetic site in the tree assumed the group inverse is integer
# negation and the group operation integer addition.  For bit-packed charges
# both are false -- ``-encode(1, 1)`` decodes as ``(-1, -2)``, and adding two
# packed charges carries across the 16-bit factor boundary -- so a conserving
# block fused to ``-65536`` instead of the identity.  These tests walk the whole
# path (construct -> validate -> decompose -> contract) on the two shapes of
# product group that fail differently: one factor infinite (no ``% n``
# reduction anywhere) and both factors finite (where ``n_values()`` returns the
# *cardinality* 2*3 == 6, which is not a modulus for the packed integer).

_PRODUCT_CASES = [
    (
        "Prod(Z2,U1)",
        ProductSymmetry(ZnSymmetry(2), U1Symmetry()),
        [ProductSymmetry.encode(0, 0), ProductSymmetry.encode(1, 1)],
    ),
    (
        "Prod(Z2,Z3)",
        ProductSymmetry(ZnSymmetry(2), ZnSymmetry(3)),
        [ProductSymmetry.encode(0, 0), ProductSymmetry.encode(1, 1)],
    ),
]


def _product_leg(sym, sectors, flow, label):
    return TensorIndex(
        symmetry=sym,
        sectors=np.array(sectors, dtype=np.int32),
        multiplicities=np.array([2, 2], dtype=np.int32),
        flow=flow,
        label=label,
    )


@pytest.mark.parametrize(
    "name,sym,sectors", _PRODUCT_CASES, ids=[c[0] for c in _PRODUCT_CASES]
)
def test_product_symmetry_mixed_flow_tensor_constructs(name, sym, sectors):
    """Before #734 this raised: fused=-65536 for the conserving block (q, q)."""
    t = SymmetricTensor.random_normal_np(
        (
            _product_leg(sym, sectors, FlowDirection.IN, "a"),
            _product_leg(sym, sectors, FlowDirection.OUT, "b"),
        ),
        np.random.RandomState(0),
    )
    assert t.blocks, f"{name}: no valid blocks found"
    t._validate()
    # Both sectors are actually populated -- an enumerator that kept only the
    # identity block would also "construct and validate".
    assert set(t._block_keys) == {(int(q), int(q)) for q in sectors}


@pytest.mark.parametrize(
    "name,sym,sectors", _PRODUCT_CASES, ids=[c[0] for c in _PRODUCT_CASES]
)
def test_product_symmetry_svd_reconstructs(name, sym, sectors):
    """SVD -> contract round trip on a mixed-flow product-symmetry tensor.

    ``tenax.linalg.svd`` holds the singular values out of U and Vh, so they are
    folded back in before contracting.  The comparison is against the dense
    original, which catches a dropped block (the #734 failure mode) as a numeric
    difference rather than only as a missing key.
    """
    import tenax.linalg as tl
    from tenax.contraction.contractor import contract
    from tenax.core._tensor_utils import scale_bond_axis

    t = SymmetricTensor.random_normal_np(
        (
            _product_leg(sym, sectors, FlowDirection.IN, "a"),
            _product_leg(sym, sectors, FlowDirection.OUT, "b"),
        ),
        np.random.RandomState(0),
    )
    U, s, Vh, _ = tl.svd(t, left_labels=["a"], right_labels=["b"])
    rebuilt = contract(scale_bond_axis(U, "bond", s), Vh)

    ref = t.todense()
    assert float(jnp.linalg.norm(ref)) > 1.0, "reference is degenerate"
    got = rebuilt.todense()
    # A silently-dropped block reads as an all-zero slab, not as noise.
    assert float(jnp.linalg.norm(got)) > 1.0, f"{name}: reconstruction collapsed"
    assert float(jnp.max(jnp.abs(got - ref))) < 1e-8


@pytest.mark.parametrize(
    "name,sym,sectors", _PRODUCT_CASES, ids=[c[0] for c in _PRODUCT_CASES]
)
def test_product_symmetry_svd_over_a_multi_leg_row_reconstructs(name, sym, sectors):
    """The same round trip, but with two labels grouped into the SVD's row leg.

    ``tenax.linalg`` matricises through ``_group_blocks_by_bond_charge``, which
    fuses the flow-weighted charges of *several* legs into one bond charge —
    a rank-3 case exercises that reduction, where the rank-2 case above only
    ever fuses one leg per side.  (It does not reach
    ``_tensor_utils.fuse_indices``; ``test_tensor_utils.py`` covers that path
    directly.)
    """
    import tenax.linalg as tl
    from tenax.contraction.contractor import contract
    from tenax.core._tensor_utils import scale_bond_axis

    t = SymmetricTensor.random_normal_np(
        (
            _product_leg(sym, sectors, FlowDirection.IN, "a"),
            _product_leg(sym, sectors, FlowDirection.IN, "b"),
            _product_leg(sym, sectors, FlowDirection.OUT, "c"),
        ),
        np.random.RandomState(5),
    )
    assert t.blocks, f"{name}: no valid blocks found"

    U, s, Vh, _ = tl.svd(t, left_labels=["a", "b"], right_labels=["c"])
    rebuilt = contract(scale_bond_axis(U, "bond", s), Vh)

    ref = t.todense()
    assert float(jnp.linalg.norm(ref)) > 1.0, "reference is degenerate"
    perm = tuple(rebuilt.labels().index(lbl) for lbl in t.labels())
    got = rebuilt.transpose(perm).todense()
    assert float(jnp.linalg.norm(got)) > 1.0, f"{name}: reconstruction collapsed"
    assert float(jnp.max(jnp.abs(got - ref))) < 1e-8


@pytest.mark.parametrize(
    "name,sym,sectors", _PRODUCT_CASES, ids=[c[0] for c in _PRODUCT_CASES]
)
def test_product_symmetry_contraction_is_not_silently_zero(name, sym, sectors):
    """``contract`` on two mixed-flow product-symmetry tensors matches dense.

    The target-charge inference in ``_parse_contraction_prelude`` and the block
    enumerator both run here; either getting the packed arithmetic wrong yields
    an all-zero result rather than an error.
    """
    from tenax.contraction.contractor import contract

    left = SymmetricTensor.random_normal_np(
        (
            _product_leg(sym, sectors, FlowDirection.IN, "a"),
            _product_leg(sym, sectors, FlowDirection.OUT, "m"),
        ),
        np.random.RandomState(1),
    )
    right = SymmetricTensor.random_normal_np(
        (
            _product_leg(sym, sectors, FlowDirection.IN, "m"),
            _product_leg(sym, sectors, FlowDirection.OUT, "c"),
        ),
        np.random.RandomState(2),
    )
    got = contract(left, right).todense()
    ref = left.todense() @ right.todense()
    assert float(jnp.linalg.norm(ref)) > 1.0, "reference is degenerate"
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-10)
