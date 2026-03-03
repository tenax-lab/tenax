"""Tests for _tensor_utils shared helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._tensor_utils import (
    absorb_sqrt_singular_values,
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
