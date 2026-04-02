"""Tests for the randomized SVD (rsvd) function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor
from tenax.linalg import rsvd, svd


@pytest.fixture
def u1():
    return U1Symmetry()


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


def _make_low_rank_dense(u1, key, m=20, n=15, true_rank=5):
    """Create a low-rank DenseTensor (matrix) with known rank."""
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (m, true_rank))
    B = jax.random.normal(k2, (true_rank, n))
    data = A @ B
    charges_m = np.zeros(m, dtype=np.int32)
    charges_n = np.zeros(n, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(u1, charges_m, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(u1, charges_n, FlowDirection.OUT, label="right"),
    )
    return DenseTensor(data, indices)


def test_rsvd_returns_correct_shapes(u1, key):
    """Shape check for dense rsvd."""
    m, n, rank = 20, 15, 5
    tensor = _make_low_rank_dense(u1, key, m, n, true_rank=rank)
    U, s, Vh = rsvd(tensor, ["left"], ["right"], rank=rank, key=key)
    assert U.todense().shape == (m, rank)
    assert s.shape == (rank,)
    assert Vh.todense().shape == (rank, n)


def test_rsvd_approximates_full_svd(u1, key):
    """Singular values from rsvd should match full SVD for a low-rank matrix."""
    true_rank = 5
    tensor = _make_low_rank_dense(u1, key, m=20, n=15, true_rank=true_rank)
    _, s_full_all, _, _ = svd(tensor, ["left"], ["right"])
    _, s_rsvd, _ = rsvd(
        tensor, ["left"], ["right"], rank=true_rank, key=key, n_power_iter=2
    )
    # The top singular values should be close
    np.testing.assert_allclose(
        np.sort(np.array(s_rsvd))[::-1],
        np.sort(np.array(s_full_all[:true_rank]))[::-1],
        rtol=1e-6,
    )


def test_rsvd_reconstruction_error(u1, key):
    """A approximately equals U @ diag(s) @ Vh for low-rank input."""
    true_rank = 5
    tensor = _make_low_rank_dense(u1, key, m=20, n=15, true_rank=true_rank)
    U, s, Vh = rsvd(
        tensor, ["left"], ["right"], rank=true_rank, key=key, n_power_iter=2
    )
    reconstructed = U.todense() @ jnp.diag(s) @ Vh.todense()
    original = tensor.todense()
    rel_err = jnp.linalg.norm(original - reconstructed) / jnp.linalg.norm(original)
    assert float(rel_err) < 1e-6


def test_rsvd_labels(u1, key):
    """Output labels should be correct."""
    tensor = _make_low_rank_dense(u1, key, m=10, n=8, true_rank=3)
    U, s, Vh = rsvd(
        tensor, ["left"], ["right"], new_bond_label="svd_bond", rank=3, key=key
    )
    assert U.labels() == ("left", "svd_bond")
    assert Vh.labels() == ("svd_bond", "right")


def test_rsvd_multidimensional(u1, key):
    """rsvd works on a 4-leg tensor."""
    d1, d2, d3, d4 = 4, 5, 3, 6
    data = jax.random.normal(key, (d1, d2, d3, d4))
    indices = (
        TensorIndex.from_charges(
            u1, np.zeros(d1, dtype=np.int32), FlowDirection.IN, label="a"
        ),
        TensorIndex.from_charges(
            u1, np.zeros(d2, dtype=np.int32), FlowDirection.IN, label="b"
        ),
        TensorIndex.from_charges(
            u1, np.zeros(d3, dtype=np.int32), FlowDirection.OUT, label="c"
        ),
        TensorIndex.from_charges(
            u1, np.zeros(d4, dtype=np.int32), FlowDirection.OUT, label="d"
        ),
    )
    tensor = DenseTensor(data, indices)
    rank = 7
    U, s, Vh = rsvd(tensor, ["a", "b"], ["c", "d"], rank=rank, key=key)
    assert U.todense().shape == (d1, d2, rank)
    assert Vh.todense().shape == (rank, d3, d4)
    assert s.shape == (rank,)


def test_rsvd_symmetric_runs(u1, key):
    """U(1) SymmetricTensor rsvd should not crash."""
    charges = np.array([-1, 0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(
            u1, u1.dual(charges), FlowDirection.OUT, label="right"
        ),
    )
    tensor = SymmetricTensor.random_normal(indices, key)
    U, s, Vh = rsvd(tensor, ["left"], ["right"], rank=3, key=key)
    assert s.shape[0] <= 3
    assert s.shape[0] >= 1
    assert isinstance(U, SymmetricTensor)
    assert isinstance(Vh, SymmetricTensor)


def test_rsvd_symmetric_matches_dense(u1, key):
    """Symmetric rsvd singular values should roughly match dense rsvd."""
    charges = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(
            u1, u1.dual(charges), FlowDirection.OUT, label="right"
        ),
    )
    tensor = SymmetricTensor.random_normal(indices, key)
    rank = 3

    # Symmetric path
    _, s_sym, _ = rsvd(tensor, ["left"], ["right"], rank=rank, key=key)

    # Dense path: convert to DenseTensor
    dense_data = tensor.todense()
    dense_indices = (
        TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(
            u1, u1.dual(charges), FlowDirection.OUT, label="right"
        ),
    )
    dense_tensor = DenseTensor(dense_data, dense_indices)
    _, s_dense, _ = rsvd(dense_tensor, ["left"], ["right"], rank=rank, key=key)

    # Sort both descending and compare
    s_sym_sorted = np.sort(np.array(s_sym))[::-1]
    s_dense_sorted = np.sort(np.array(s_dense))[::-1]

    # Only compare up to the minimum length
    n_compare = min(len(s_sym_sorted), len(s_dense_sorted))
    np.testing.assert_allclose(
        s_sym_sorted[:n_compare], s_dense_sorted[:n_compare], rtol=0.3
    )
