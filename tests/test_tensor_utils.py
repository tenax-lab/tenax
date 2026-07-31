"""Tests for _tensor_utils shared helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._tensor_utils import (
    absorb_sqrt_singular_values,
    double_layer_tensor,
    fuse_indices,
    max_abs_normalize,
    scale_bond_axis,
    split_index,
)
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import (
    FermionParity,
    ProductSymmetry,
    U1Symmetry,
    ZnSymmetry,
)
from tenax.core.tensor import DenseTensor, SymmetricTensor, _compute_valid_blocks


class TestScaleBondAxis:
    def test_dense_basic(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="row"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="col"),
        )
        data = jax.random.normal(rng, (3, 3))
        T = DenseTensor(data, indices)
        scale = jnp.array([2.0, 3.0, 4.0])

        result = scale_bond_axis(T, "col", scale)

        expected = data * scale.reshape(1, 3)
        np.testing.assert_allclose(result.todense(), expected, rtol=1e-6)

    def test_dense_preserves_labels(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="row"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="col"),
        )
        data = jax.random.normal(rng, (3, 3))
        T = DenseTensor(data, indices)
        scale = jnp.array([1.0, 2.0, 3.0])
        result = scale_bond_axis(T, "row", scale)
        assert result.labels() == T.labels()

    def test_symmetric_parity(self, u1, rng):
        """Scaling a SymmetricTensor and materializing to dense matches
        scaling the dense version directly."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="out"
            ),
        )
        T = SymmetricTensor.random_normal(indices, rng)
        scale = jnp.array([2.0, 3.0, 4.0])

        result_sym = scale_bond_axis(T, "out", scale)

        # Compare with dense path
        T_dense = DenseTensor(T.todense(), T.indices)
        result_dense = scale_bond_axis(T_dense, "out", scale)

        np.testing.assert_allclose(
            result_sym.todense(), result_dense.todense(), rtol=1e-6
        )

    def test_symmetric_preserves_type(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="out"
            ),
        )
        T = SymmetricTensor.random_normal(indices, rng)
        scale = jnp.array([1.0, 2.0, 3.0])
        result = scale_bond_axis(T, "in", scale)
        assert isinstance(result, SymmetricTensor)


class TestMaxAbsNormalize:
    def test_dense_normalized(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="row"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="col"),
        )
        data = jax.random.normal(rng, (3, 3)) * 10.0
        T = DenseTensor(data, indices)

        T_norm, log_norm = max_abs_normalize(T)
        np.testing.assert_allclose(float(T_norm.max_abs()), 1.0, rtol=1e-6)

    def test_symmetric_normalized(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="out"
            ),
        )
        T = SymmetricTensor.random_normal(indices, rng) * 5.0

        T_norm, log_norm = max_abs_normalize(T)
        np.testing.assert_allclose(float(T_norm.max_abs()), 1.0, rtol=1e-6)

    def test_log_norm_value(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="row"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="col"),
        )
        data = jax.random.normal(rng, (3, 3)) * 10.0
        T = DenseTensor(data, indices)

        _, log_norm = max_abs_normalize(T)
        np.testing.assert_allclose(
            float(log_norm), float(jnp.log(T.max_abs())), rtol=1e-6
        )


class TestAbsorbSqrtSingularValues:
    def test_reconstruct_dense(self, u1, rng):
        """F_left @ F_right should approximately reconstruct U @ diag(s) @ Vh."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="row"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="col"),
        )
        data = jax.random.normal(rng, (3, 3))
        T = DenseTensor(data, indices)

        U, s, Vh, _ = truncated_svd(
            T,
            left_labels=["row"],
            right_labels=["col"],
            new_bond_label="bond",
        )

        F_left, F_right = absorb_sqrt_singular_values(U, s, Vh, "bond")

        # Contract F_left and F_right
        reconstructed = contract(F_left, F_right)

        np.testing.assert_allclose(reconstructed.todense(), T.todense(), rtol=1e-5)

    def test_reconstruct_symmetric(self, u1, rng):
        """Symmetric version: F_left @ F_right reconstructs U @ diag(s) @ Vh."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="out"
            ),
        )
        T = SymmetricTensor.random_normal(indices, rng)

        U, s, Vh, _ = truncated_svd(
            T,
            left_labels=["in"],
            right_labels=["out"],
            new_bond_label="bond",
        )

        F_left, F_right = absorb_sqrt_singular_values(U, s, Vh, "bond")

        reconstructed = contract(F_left, F_right)

        np.testing.assert_allclose(reconstructed.todense(), T.todense(), rtol=1e-5)


class TestFuseIndices:
    """Tests for fuse_indices on DenseTensor and SymmetricTensor."""

    def _make_dense_4leg(self, u1, rng):
        """Helper: 4-leg DenseTensor (2,3,3,2) with trivial charges."""
        shapes = [2, 3, 3, 2]
        labels = ["up", "down", "left", "right"]
        flows = [
            FlowDirection.IN,
            FlowDirection.OUT,
            FlowDirection.IN,
            FlowDirection.OUT,
        ]
        charges = [np.zeros(s, dtype=np.int32) for s in shapes]
        indices = tuple(
            TensorIndex.from_charges(u1, charges[i], flows[i], label=labels[i])
            for i in range(4)
        )
        data = jax.random.normal(rng, tuple(shapes))
        return DenseTensor(data, indices)

    def test_dense_fuse_adjacent(self, u1, rng):
        """Fusing two adjacent axes of a DenseTensor gives correct shape."""
        T = self._make_dense_4leg(u1, rng)
        # Fuse axes 0 (up, dim=2) and 1 (down, dim=3) → fused dim=6
        result = fuse_indices(T, 0, 1, "ud", FlowDirection.IN)
        assert result.todense().shape == (6, 3, 2)
        assert list(result.labels()) == ["ud", "left", "right"]

    def test_dense_fuse_non_adjacent(self, u1, rng):
        """Fusing non-adjacent axes transposes then reshapes correctly."""
        T = self._make_dense_4leg(u1, rng)
        # Fuse axes 0 (up, dim=2) and 2 (left, dim=3) → fused dim=6
        result = fuse_indices(T, 0, 2, "ul", FlowDirection.IN)
        assert result.todense().shape == (6, 3, 2)
        assert "ul" in result.labels()

    def test_dense_fuse_data_consistency(self, u1, rng):
        """Fused DenseTensor has the same elements as manual reshape."""
        charges = np.zeros(2, dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="c"),
        )
        data = jax.random.normal(rng, (2, 2, 2))
        T = DenseTensor(data, indices)

        result = fuse_indices(T, 0, 1, "ab", FlowDirection.IN)
        # axes a,b are adjacent (0,1), so fuse = reshape(4, 2)
        expected = data.reshape(4, 2)
        np.testing.assert_allclose(result.todense(), expected, rtol=1e-6)

    def test_symmetric_fuse_matches_dense(self, u1, rng):
        """Fusing a SymmetricTensor matches fusing its dense representation."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="b"
            ),
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="c"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="d"
            ),
        )
        T_sym = SymmetricTensor.random_normal(indices, rng)
        T_dense = DenseTensor(T_sym.todense(), T_sym.indices)

        fused_sym = fuse_indices(T_sym, 0, 1, "ab", FlowDirection.IN)
        fused_dense = fuse_indices(T_dense, 0, 1, "ab", FlowDirection.IN)

        np.testing.assert_allclose(
            fused_sym.todense(),
            fused_dense.todense(),
            rtol=1e-5,
            err_msg="Symmetric fuse_indices doesn't match dense",
        )

    def test_symmetric_preserves_type(self, u1, rng):
        """fuse_indices on SymmetricTensor returns SymmetricTensor."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="b"
            ),
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="c"),
        )
        T = SymmetricTensor.random_normal(indices, rng)
        result = fuse_indices(T, 0, 1, "ab", FlowDirection.IN)
        assert isinstance(result, SymmetricTensor)

    def test_fuse_reduces_ndim(self, u1, rng):
        """Fusing two legs reduces ndim by 1."""
        T = self._make_dense_4leg(u1, rng)
        result = fuse_indices(T, 1, 3, "dr", FlowDirection.OUT)
        assert result.ndim == 3

    def test_dense_fuse_populates_fuse_info(self, u1, rng):
        """Fused DenseTensor leg should have FuseInfo with parent indices."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="c"),
        )
        data = jax.random.normal(rng, (3, 3, 3))
        T = DenseTensor(data, indices)
        result = fuse_indices(T, 0, 1, "ab", FlowDirection.IN)

        fused_idx = result.indices[0]
        assert fused_idx.fuse_info is not None
        assert len(fused_idx.fuse_info.parent_indices) == 2
        assert fused_idx.fuse_info.parent_indices[0].label == "a"
        assert fused_idx.fuse_info.parent_indices[1].label == "b"

    def test_symmetric_fuse_populates_fuse_info(self, u1, rng):
        """Fused SymmetricTensor leg should have FuseInfo with parent indices."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="b"
            ),
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="c"),
        )
        T = SymmetricTensor.random_normal(indices, rng)
        result = fuse_indices(T, 0, 1, "ab", FlowDirection.IN)

        fused_idx = result.indices[0]
        assert fused_idx.fuse_info is not None
        assert len(fused_idx.fuse_info.parent_indices) == 2
        assert fused_idx.fuse_info.parent_indices[0].label == "a"
        assert fused_idx.fuse_info.parent_indices[1].label == "b"

    def test_fuse_sectors_computation(self, u1):
        """_compute_fused_sectors gives correct sectors and total dim."""
        from tenax.algorithms._tensor_utils import _compute_fused_sectors

        charges_a = np.array([-1, 0, 1], dtype=np.int32)
        idx_a = TensorIndex.from_charges(u1, charges_a, FlowDirection.IN, label="a")
        idx_b = TensorIndex.from_charges(
            u1, u1.dual(charges_a), FlowDirection.OUT, label="b"
        )

        sectors, mults = _compute_fused_sectors(idx_a, idx_b, FlowDirection.IN, u1)

        assert len(sectors) == len(mults)
        assert int(np.sum(mults)) == 9  # 3 * 3
        # sectors should be sorted
        np.testing.assert_array_equal(sectors, np.sort(sectors))

    @pytest.mark.parametrize(
        "sym",
        [
            U1Symmetry(),
            ZnSymmetry(3),
            ProductSymmetry(ZnSymmetry(2), U1Symmetry()),
            ProductSymmetry(ZnSymmetry(2), ZnSymmetry(3)),
        ],
        ids=["u1", "z3", "z2_x_u1", "z2_x_z3"],
    )
    @pytest.mark.parametrize(
        "flows",
        [
            (FlowDirection.IN, FlowDirection.IN),
            (FlowDirection.IN, FlowDirection.OUT),
            (FlowDirection.OUT, FlowDirection.OUT),
        ],
        ids=["ii", "io", "oo"],
    )
    @pytest.mark.parametrize("fused_flow", [FlowDirection.IN, FlowDirection.OUT])
    def test_fuse_sectors_agrees_with_fused_charges(self, sym, flows, fused_flow):
        """The sector-level and charge-level fusions must be one function.

        ``_compute_fused_sectors`` has no production caller today — only
        ``_compute_fused_charges`` is on the live path — so a bug in it is
        invisible to every end-to-end test, and reverting its arithmetic to the
        raw ``flow * q`` sum killed nothing.  What pins it is the relation it
        exists to compute cheaply: the sector table it returns must equal
        ``np.unique(_compute_fused_charges(...), return_counts=True)``.  Under
        ``ProductSymmetry`` the two part company hard, because ``n_values()`` is
        the group cardinality (``2 * 3 == 6``) and not a modulus for the packed
        charge (#734).
        """
        from tenax.algorithms._tensor_utils import (
            _compute_fused_charges,
            _compute_fused_sectors,
        )

        if isinstance(sym, ProductSymmetry):
            raw = np.array(
                [ProductSymmetry.encode(q1, q2) for q1 in range(2) for q2 in range(3)],
                dtype=np.int32,
            )
        else:
            raw = np.array([0, 1, 2], dtype=np.int32)
        charges = np.unique(sym.canonicalize_charges(raw))

        idx_a = TensorIndex.from_charges(sym, charges, flows[0], label="a")
        idx_b = TensorIndex.from_charges(sym, charges, flows[1], label="b")

        sectors, mults = _compute_fused_sectors(idx_a, idx_b, fused_flow, sym)
        want_sectors, want_mults = np.unique(
            _compute_fused_charges(idx_a, idx_b, fused_flow, sym), return_counts=True
        )

        np.testing.assert_array_equal(sectors, want_sectors)
        np.testing.assert_array_equal(mults, want_mults)
        assert int(np.sum(mults)) == idx_a.dim * idx_b.dim
        assert len(sectors) > 1, "degenerate fixture: nothing to get wrong"


class TestSplitIndex:
    """Tests for split_index — inverse of fuse_indices."""

    def test_dense_fuse_split_roundtrip(self, u1, rng):
        """Fuse then split should recover the original DenseTensor."""
        charges = np.zeros(3, dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="c"),
        )
        data = jax.random.normal(rng, (3, 3, 3))
        T = DenseTensor(data, indices)

        fused = fuse_indices(T, 0, 1, "ab", FlowDirection.IN)
        recovered = split_index(fused, 0)

        np.testing.assert_allclose(recovered.todense(), T.todense(), rtol=1e-6)
        assert recovered.labels() == T.labels()

    def test_dense_nontrivial_charges_roundtrip(self, u1, rng):
        """Fuse-split roundtrip with nontrivial U(1) charges."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="c"),
        )
        data = jax.random.normal(rng, (3, 3, 3))
        T = DenseTensor(data, indices)

        fused = fuse_indices(T, 0, 1, "ab", FlowDirection.IN)
        recovered = split_index(fused, 0)

        np.testing.assert_allclose(recovered.todense(), T.todense(), rtol=1e-6)

    def test_split_no_fuse_info_raises(self, u1, rng):
        """split_index on unfused leg should raise ValueError."""
        charges = np.zeros(2, dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
        )
        T = DenseTensor(jax.random.normal(rng, (2, 2)), indices)
        with pytest.raises(ValueError, match="fuse_info"):
            split_index(T, 0)

    def test_symmetric_fuse_split_roundtrip(self, u1, rng):
        """Fuse then split on SymmetricTensor recovers original."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="b"
            ),
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="c"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="d"
            ),
        )
        T = SymmetricTensor.random_normal(indices, rng)

        fused = fuse_indices(T, 0, 1, "ab", FlowDirection.IN)
        recovered = split_index(fused, 0)

        np.testing.assert_allclose(recovered.todense(), T.todense(), rtol=1e-5)

    def test_split_matches_dense(self, u1, rng):
        """Symmetric split_index matches dense split_index."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="b"
            ),
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="c"),
        )
        T_sym = SymmetricTensor.random_normal(indices, rng)
        T_dense = DenseTensor(T_sym.todense(), T_sym.indices)

        fused_sym = fuse_indices(T_sym, 0, 1, "ab", FlowDirection.IN)
        fused_dense = fuse_indices(T_dense, 0, 1, "ab", FlowDirection.IN)

        split_sym = split_index(fused_sym, 0)
        split_dense = split_index(fused_dense, 0)

        np.testing.assert_allclose(
            split_sym.todense(), split_dense.todense(), rtol=1e-5
        )

    def test_non_adjacent_fuse_split_roundtrip(self, u1, rng):
        """Fuse-split roundtrip for non-adjacent axes."""
        charges = np.zeros(2, dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="c"),
        )
        data = jax.random.normal(rng, (2, 2, 2))
        T = DenseTensor(data, indices)

        fused = fuse_indices(T, 0, 2, "ac", FlowDirection.IN)
        recovered = split_index(fused, 0)

        # After fuse(0,2) and split, we get axes (a, c, b) — not (a, b, c)
        # because fuse transposes non-adjacent axes to be adjacent first
        assert recovered.ndim == 3


class TestDoubleLayerTensor:
    """Tests for double_layer_tensor on DenseTensor and SymmetricTensor."""

    def _make_peps_dense(self, u1, rng, D=2, d=2):
        """Helper: 5-leg DenseTensor (D,D,D,D,d) with trivial charges."""
        labels = ["up", "down", "left", "right", "phys"]
        flows = [
            FlowDirection.IN,
            FlowDirection.OUT,
            FlowDirection.IN,
            FlowDirection.OUT,
            FlowDirection.IN,
        ]
        shapes = [D, D, D, D, d]
        charges = [np.zeros(s, dtype=np.int32) for s in shapes]
        indices = tuple(
            TensorIndex.from_charges(u1, charges[i], flows[i], label=labels[i])
            for i in range(5)
        )
        data = jax.random.normal(rng, tuple(shapes))
        return DenseTensor(data, indices)

    def test_dense_shape(self, u1, rng):
        """Dense double-layer tensor has shape (D², D², D², D²)."""
        D = 2
        A = self._make_peps_dense(u1, rng, D=D)
        dl = double_layer_tensor(A)
        assert dl.todense().shape == (D * D, D * D, D * D, D * D)

    def test_dense_labels(self, u1, rng):
        """Double-layer tensor preserves spatial labels."""
        A = self._make_peps_dense(u1, rng)
        dl = double_layer_tensor(A)
        assert set(dl.labels()) == {"up", "down", "left", "right"}

    def test_dense_matches_einsum(self, u1, rng):
        """Dense double-layer matches manual einsum."""
        D, d = 2, 2
        A = self._make_peps_dense(u1, rng, D=D, d=d)
        data = A.todense()

        # Manual: contract over phys, fuse ket/bra pairs
        dl_manual = jnp.einsum("udlrs,UDLRs->uUdDlLrR", data, jnp.conj(data))
        dl_manual = dl_manual.reshape(D * D, D * D, D * D, D * D)

        dl = double_layer_tensor(A)
        np.testing.assert_allclose(dl.todense(), dl_manual, rtol=1e-5)

    def test_symmetric_matches_dense(self, u1, rng):
        """Symmetric double-layer tensor matches dense version."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        phys_charges = np.array([-1, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="up"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="down"
            ),
            TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="left"),
            TensorIndex.from_charges(
                u1, u1.dual(charges), FlowDirection.OUT, label="right"
            ),
            TensorIndex.from_charges(u1, phys_charges, FlowDirection.IN, label="phys"),
        )
        A_sym = SymmetricTensor.random_normal(indices, rng)
        A_dense = DenseTensor(A_sym.todense(), A_sym.indices)

        dl_sym = double_layer_tensor(A_sym)
        dl_dense = double_layer_tensor(A_dense)

        np.testing.assert_allclose(
            dl_sym.todense(),
            dl_dense.todense(),
            rtol=1e-5,
            err_msg="Symmetric double_layer_tensor doesn't match dense",
        )


# ------------------------------------------------------------------ #
# Vectorized fuse/split scatter-gather (#569, Milestone A)             #
# ------------------------------------------------------------------ #


class TestFuseSplitVectorized:
    """The vectorized scatter (fuse) / gather (split) must be exactly
    equivalent to the per-element loops they replaced — including under AD
    and jit, and for fermionic tensors (the scatter/gather move data only,
    so Koszul signs in the block values are carried through untouched)."""

    def _u1_3leg(self, seed=0):
        sym = U1Symmetry()
        ch = np.array([-1, 0, 1], dtype=np.int32)
        ch2 = np.array([0, 1], dtype=np.int32)
        idx_a = TensorIndex.from_charges(sym, ch, FlowDirection.IN, label="a")
        idx_b = TensorIndex.from_charges(sym, ch2, FlowDirection.IN, label="b")
        idx_c = TensorIndex.from_charges(sym, ch, FlowDirection.OUT, label="c")
        return SymmetricTensor.random_normal(
            (idx_a, idx_b, idx_c), jax.random.PRNGKey(seed), dtype=jnp.float64
        )

    def test_fermionic_fuse_split_roundtrip(self):
        from tenax.core.symmetry import FermionParity

        sym = FermionParity()
        ch = np.array([0, 0, 1, 1], dtype=np.int32)
        idx_a = TensorIndex.from_charges(sym, ch, FlowDirection.IN, label="a")
        idx_b = TensorIndex.from_charges(sym, ch, FlowDirection.IN, label="b")
        idx_c = TensorIndex.from_charges(sym, ch, FlowDirection.OUT, label="c")
        T = SymmetricTensor.random_normal(
            (idx_a, idx_b, idx_c), jax.random.PRNGKey(1), dtype=jnp.float64
        )
        fused = fuse_indices(T, 0, 1, "ab", FlowDirection.IN)
        recovered = split_index(fused, 0)
        np.testing.assert_allclose(
            recovered.todense(), T.todense(), rtol=1e-12, atol=1e-14
        )

    def test_fuse_split_jit_bit_identical(self):
        # The vectorized scatter/gather must give the SAME result under jit as
        # eager (data movement is a pure scatter/gather primitive).
        T = self._u1_3leg(seed=2)
        fused_eager = fuse_indices(T, 0, 1, "ab", FlowDirection.IN)
        split_eager = split_index(fused_eager, 0)

        fused_jit = jax.jit(lambda t: fuse_indices(t, 0, 1, "ab", FlowDirection.IN))(T)
        split_jit = jax.jit(lambda t: split_index(t, 0))(fused_jit)
        np.testing.assert_array_equal(
            np.asarray(fused_jit.todense()), np.asarray(fused_eager.todense())
        )
        np.testing.assert_array_equal(
            np.asarray(split_jit.todense()), np.asarray(split_eager.todense())
        )

    def test_fuse_split_ad_roundtrip(self):
        # Differentiating through fuse->split (an identity on data) must give a
        # gradient equal to differentiating the same scalar of the input —
        # i.e. the scatter then gather compose to the identity VJP. Compares
        # analytic grad to central finite difference on the flat buffer.
        T = self._u1_3leg(seed=3)

        leaves, treedef = jax.tree_util.tree_flatten(T)
        x0 = leaves[0]

        def loss(x):
            Tx = jax.tree_util.tree_unflatten(treedef, [x])
            fused = fuse_indices(Tx, 0, 1, "ab", FlowDirection.IN)
            split = split_index(fused, 0)
            # weighted sum so the gradient is a fixed nonconstant tensor
            return jnp.sum(split.todense() ** 2)

        g = jax.grad(loss)(x0)
        x0n = np.asarray(x0)
        eps = 1e-6
        fd = np.zeros_like(x0n)
        for i in range(x0n.shape[0]):
            xp = x0n.copy()
            xp[i] += eps
            xm = x0n.copy()
            xm[i] -= eps
            fd[i] = (float(loss(jnp.asarray(xp))) - float(loss(jnp.asarray(xm)))) / (
                2 * eps
            )
        np.testing.assert_allclose(np.asarray(g), fd, rtol=1e-5, atol=1e-7)


# ------------------------------------------------------------------ #
# Fused indices keep charges and sectors on the same representatives   #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "sym,pairs",
    [
        (
            ProductSymmetry(ZnSymmetry(2), ZnSymmetry(3)),
            [(q1, q2) for q1 in range(2) for q2 in range(3)],
        ),
        (
            ProductSymmetry(FermionParity(), U1Symmetry()),
            [(n % 2, n) for n in range(3)],
        ),
    ],
    ids=["z2_x_z3", "fp_x_u1"],
)
@pytest.mark.parametrize("fused_flow", [FlowDirection.IN, FlowDirection.OUT])
def test_fused_charges_never_name_a_sector_the_index_lacks(sym, pairs, fused_flow):
    """``idx.charges`` and ``idx.sectors`` must use the same representatives.

    ``fuse_indices`` derives the fused sectors from the fused *charges*, then
    overwrites ``_charges_cache`` with those same charges — but the
    ``TensorIndex`` constructor canonicalises ``sectors`` on the way through
    (#734), so the two halves stop agreeing unless the charges are canonicalised
    too.  U(1)/``Z_n`` are unaffected because ``_compute_fused_charges`` already
    reduces mod ``n``; ``ProductSymmetry``'s bit-packed charges are not reduced
    by that, and were measured landing outside their own sector table:
    ``Z2 x Z3`` produced charge ``2`` against sectors ``[0, 1]``, and
    ``FP x U(1)`` produced a charge set *disjoint* from its sectors.

    A charge with no matching sector is not an error anywhere — it just means
    the blocks sitting on it are dropped.
    """
    charges = np.unique(
        sym.canonicalize_charges(
            np.array([ProductSymmetry.encode(*p) for p in pairs], dtype=np.int32)
        )
    )
    idx_a = TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="a")
    idx_b = TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="b")
    idx_c = TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="c")

    dense = DenseTensor(
        jnp.ones((idx_a.dim, idx_b.dim, idx_c.dim)), (idx_a, idx_b, idx_c)
    )
    fused = fuse_indices(dense, 0, 1, "ab", fused_flow)
    leg = fused.indices[0]
    assert set(leg.charges.tolist()) <= set(leg.sectors.tolist()), (
        sorted(set(leg.charges.tolist())),
        leg.sectors.tolist(),
    )
    assert len(leg.charges) == leg.dim
    for q in leg.sectors:
        assert int(np.sum(leg.charges == q)) == leg.multiplicity(int(q))

    # And the same through the block-sparse path, which caches its own copy.
    keys = _compute_valid_blocks((idx_a, idx_b, idx_c))
    assert keys, "fixture must have at least one conserving block"
    sym_tensor = SymmetricTensor(
        {
            k: jnp.ones(
                tuple(i.multiplicity(int(q)) for i, q in zip((idx_a, idx_b, idx_c), k))
            )
            for k in keys
        },
        (idx_a, idx_b, idx_c),
    )
    fused_sym = fuse_indices(sym_tensor, 0, 1, "ab", fused_flow)
    leg = fused_sym.indices[0]
    assert set(leg.charges.tolist()) <= set(leg.sectors.tolist()), (
        sorted(set(leg.charges.tolist())),
        leg.sectors.tolist(),
    )


@pytest.mark.parametrize(
    "sym",
    [
        ProductSymmetry(ZnSymmetry(2), U1Symmetry()),
        ProductSymmetry(ZnSymmetry(2), ZnSymmetry(3)),
    ],
    ids=["z2_x_u1", "z2_x_z3"],
)
@pytest.mark.parametrize(
    "flows",
    [
        (FlowDirection.IN, FlowDirection.IN, FlowDirection.OUT),
        (FlowDirection.IN, FlowDirection.OUT, FlowDirection.OUT),
        (FlowDirection.OUT, FlowDirection.OUT, FlowDirection.OUT),
    ],
    ids=["iio", "ioo", "ooo"],
)
@pytest.mark.parametrize("fused_flow", [FlowDirection.IN, FlowDirection.OUT])
def test_fused_block_keys_name_sectors_their_leg_carries(sym, flows, fused_flow):
    """Every key ``fuse_indices`` writes must exist in its own leg's sector table.

    The companion test above pins the *charges* side of this, which #734 Task 2
    fixed.  The keys were still derived from an independent copy of
    ``flow_a * qa + flow_b * qb`` reduced by ``% n_values()`` -- twice over, in
    ``_fuse_indices_symmetric``'s ``fused_groups`` grouping and again when
    rewriting each block key.  Measured on ``ProductSymmetry(Z2, Z3)``, that
    produced fused keys ``{0, 1, 3, 4, 5}`` against a fused sector table of
    ``{0, 1}``: keys naming charges the leg does not carry.  Nothing raises on
    such a key -- ``todense`` and every later contraction simply drop the block.

    ``ProductSymmetry(Z2, U(1))`` is the control: ``n_values()`` is ``None``
    there, so no bogus modulus was applied and only the flow weighting was
    wrong.  ``ProductSymmetry(Z2, Z3)`` gets both errors at once.
    """
    sectors = np.array(
        [ProductSymmetry.encode(0, 0), ProductSymmetry.encode(1, 1)], dtype=np.int32
    )

    def leg(flow, label):
        return TensorIndex(
            symmetry=sym,
            sectors=sectors.copy(),
            multiplicities=np.array([2, 2], dtype=np.int32),
            flow=flow,
            label=label,
        )

    indices = tuple(leg(f, lbl) for f, lbl in zip(flows, "abc"))
    keys = _compute_valid_blocks(indices)
    assert keys, "fixture must have at least one conserving block"
    t = SymmetricTensor(
        {
            k: jnp.asarray(
                np.random.RandomState(abs(hash(k)) % 2**31).standard_normal(
                    tuple(i.multiplicity(int(q)) for i, q in zip(indices, k))
                )
            )
            for k in keys
        },
        indices,
    )

    fused = fuse_indices(t, 0, 1, "ab", fused_flow)

    assert fused.blocks, "fusion dropped every block"
    for key in fused._block_keys:
        for idx, q in zip(fused.indices, key):
            assert idx.has_sector(int(q)), (
                key,
                idx.label,
                idx.sectors.tolist(),
            )
    # A key that names a real sector can still violate conservation.
    fused._validate()
    # No data lost: fusing is a relabelling of the same numbers.
    np.testing.assert_allclose(
        float(jnp.linalg.norm(fused.todense())),
        float(jnp.linalg.norm(t.todense())),
        rtol=1e-12,
    )
    assert float(jnp.linalg.norm(t.todense())) > 1.0, "degenerate fixture"

    # ... and the split inverts it, which pins the two ``(qa, qb) -> q_f``
    # groupings (fuse's and split's) to the same map.
    unfused = split_index(fused, 0)
    assert set(unfused._block_keys) == set(t._block_keys)
    for key in t._block_keys:
        np.testing.assert_allclose(
            np.asarray(unfused.blocks[key]), np.asarray(t.blocks[key]), atol=1e-12
        )
