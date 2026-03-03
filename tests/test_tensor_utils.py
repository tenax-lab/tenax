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
)
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


class TestScaleBondAxis:
    def test_dense_basic(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="row"),
            TensorIndex(u1, charges, FlowDirection.OUT, label="col"),
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
            TensorIndex(u1, charges, FlowDirection.IN, label="row"),
            TensorIndex(u1, charges, FlowDirection.OUT, label="col"),
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
            TensorIndex(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="out"),
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
            TensorIndex(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="out"),
        )
        T = SymmetricTensor.random_normal(indices, rng)
        scale = jnp.array([1.0, 2.0, 3.0])
        result = scale_bond_axis(T, "in", scale)
        assert isinstance(result, SymmetricTensor)


class TestMaxAbsNormalize:
    def test_dense_normalized(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="row"),
            TensorIndex(u1, charges, FlowDirection.OUT, label="col"),
        )
        data = jax.random.normal(rng, (3, 3)) * 10.0
        T = DenseTensor(data, indices)

        T_norm, log_norm = max_abs_normalize(T)
        np.testing.assert_allclose(float(T_norm.max_abs()), 1.0, rtol=1e-6)

    def test_symmetric_normalized(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="out"),
        )
        T = SymmetricTensor.random_normal(indices, rng) * 5.0

        T_norm, log_norm = max_abs_normalize(T)
        np.testing.assert_allclose(float(T_norm.max_abs()), 1.0, rtol=1e-6)

    def test_log_norm_value(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="row"),
            TensorIndex(u1, charges, FlowDirection.OUT, label="col"),
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
            TensorIndex(u1, charges, FlowDirection.IN, label="row"),
            TensorIndex(u1, charges, FlowDirection.OUT, label="col"),
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
            TensorIndex(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="out"),
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
            TensorIndex(u1, charges[i], flows[i], label=labels[i]) for i in range(4)
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
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(u1, charges, FlowDirection.OUT, label="b"),
            TensorIndex(u1, charges, FlowDirection.IN, label="c"),
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
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="b"),
            TensorIndex(u1, charges, FlowDirection.IN, label="c"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="d"),
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
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="b"),
            TensorIndex(u1, charges, FlowDirection.IN, label="c"),
        )
        T = SymmetricTensor.random_normal(indices, rng)
        result = fuse_indices(T, 0, 1, "ab", FlowDirection.IN)
        assert isinstance(result, SymmetricTensor)

    def test_fuse_reduces_ndim(self, u1, rng):
        """Fusing two legs reduces ndim by 1."""
        T = self._make_dense_4leg(u1, rng)
        result = fuse_indices(T, 1, 3, "dr", FlowDirection.OUT)
        assert result.ndim == 3


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
            TensorIndex(u1, charges[i], flows[i], label=labels[i]) for i in range(5)
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
            TensorIndex(u1, charges, FlowDirection.IN, label="up"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="down"),
            TensorIndex(u1, charges, FlowDirection.IN, label="left"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="right"),
            TensorIndex(u1, phys_charges, FlowDirection.IN, label="phys"),
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
