"""Tests for Controlled Bond Expansion (CBE) algorithm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax import (
    DMRGConfig,
    build_mpo_heisenberg,
    build_random_symmetric_mps,
    dmrg,
)
from tenax.algorithms.cbe import expand_bond
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_left_canonical_site(chi_l, d, chi_r, key, dtype=jnp.float64):
    """Create a left-canonical DenseTensor site with 3 legs."""
    sym = U1Symmetry()
    charges_l = np.zeros(chi_l, dtype=np.int32)
    charges_d = np.zeros(d, dtype=np.int32)
    charges_r = np.zeros(chi_r, dtype=np.int32)

    data = jax.random.normal(key, (chi_l, d, chi_r), dtype=dtype)
    mat = data.reshape(chi_l * d, chi_r)
    q, _ = jnp.linalg.qr(mat)
    q = q[:, :chi_r]
    data_lc = q.reshape(chi_l, d, chi_r)

    indices = (
        TensorIndex(sym, charges_l, FlowDirection.IN, label="left"),
        TensorIndex(sym, charges_d, FlowDirection.IN, label="phys"),
        TensorIndex(sym, charges_r, FlowDirection.OUT, label="right"),
    )
    return DenseTensor(data_lc, indices)


def _make_right_tensor(chi_r, d2, chi_rr, key, dtype=jnp.float64):
    """Create a DenseTensor for the right neighbor site."""
    sym = U1Symmetry()
    charges_r = np.zeros(chi_r, dtype=np.int32)
    charges_d2 = np.zeros(d2, dtype=np.int32)
    charges_rr = np.zeros(chi_rr, dtype=np.int32)

    data = jax.random.normal(key, (chi_r, d2, chi_rr), dtype=dtype)

    indices = (
        TensorIndex(sym, charges_r, FlowDirection.IN, label="right"),
        TensorIndex(sym, charges_d2, FlowDirection.IN, label="phys2"),
        TensorIndex(sym, charges_rr, FlowDirection.OUT, label="right2"),
    )
    return DenseTensor(data, indices)


def _make_random_hermitian_matvec(theta_shape, key):
    """Create a random Hermitian matvec that mixes all directions.

    Returns a matvec function f(v_flat, shape) -> result_flat where the
    linear operator is a random symmetric matrix, ensuring that the CBE
    sketch produces vectors outside the original column space.
    """
    n = 1
    for s in theta_shape:
        n *= s
    # Random symmetric matrix
    M = jax.random.normal(key, (n, n))
    M = (M + M.T) / 2.0

    def matvec(v_flat, shape):
        return M @ v_flat

    return matvec


class TestExpandedShape:
    """Test that expand_bond produces the correct output shape."""

    def test_expanded_shape(self):
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        chi_l, d, chi_r = 4, 2, 3
        d2, chi_rr = 2, 5
        k = 4

        site = _make_left_canonical_site(chi_l, d, chi_r, k1)
        right = _make_right_tensor(chi_r, d2, chi_rr, k2)
        theta_shape = (chi_l, d, d2, chi_rr)
        matvec = _make_random_hermitian_matvec(theta_shape, k3)

        result = expand_bond(
            site, matvec, right, k=k, oversampling=3, key=jax.random.PRNGKey(42)
        )
        dense = result.todense()
        assert dense.shape == (chi_l, d, chi_r + k), (
            f"Expected shape {(chi_l, d, chi_r + k)}, got {dense.shape}"
        )

    def test_expanded_shape_small_k(self):
        key = jax.random.PRNGKey(1)
        k1, k2, k3 = jax.random.split(key, 3)
        chi_l, d, chi_r = 3, 2, 2
        d2, chi_rr = 2, 3
        k = 2

        site = _make_left_canonical_site(chi_l, d, chi_r, k1)
        right = _make_right_tensor(chi_r, d2, chi_rr, k2)
        theta_shape = (chi_l, d, d2, chi_rr)
        matvec = _make_random_hermitian_matvec(theta_shape, k3)

        result = expand_bond(
            site, matvec, right, k=k, oversampling=2, key=jax.random.PRNGKey(42)
        )
        dense = result.todense()
        assert dense.shape == (chi_l, d, chi_r + k)


class TestExpansionOrthogonal:
    """Test that expansion columns are orthogonal to original site columns."""

    def test_expansion_orthogonal_to_original(self):
        key = jax.random.PRNGKey(10)
        k1, k2, k3 = jax.random.split(key, 3)
        chi_l, d, chi_r = 6, 2, 4
        d2, chi_rr = 2, 5
        k = 3

        site = _make_left_canonical_site(chi_l, d, chi_r, k1)
        right = _make_right_tensor(chi_r, d2, chi_rr, k2)
        theta_shape = (chi_l, d, d2, chi_rr)
        matvec = _make_random_hermitian_matvec(theta_shape, k3)

        result = expand_bond(
            site, matvec, right, k=k, oversampling=3, key=jax.random.PRNGKey(42)
        )
        dense = result.todense()
        A_mat = site.todense().reshape(chi_l * d, chi_r)
        new_cols = dense.reshape(chi_l * d, chi_r + k)[:, chi_r:]

        # A_mat^dagger @ new_cols should be approximately zero
        overlap = A_mat.conj().T @ new_cols
        assert jnp.allclose(overlap, 0.0, atol=1e-10), (
            f"Overlap between original and new columns too large: "
            f"{float(jnp.max(jnp.abs(overlap)))}"
        )


class TestComplexTensors:
    """Test that CBE works with complex-valued tensors."""

    def test_complex_tensors(self):
        key = jax.random.PRNGKey(20)
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        chi_l, d, chi_r = 4, 2, 3
        d2, chi_rr = 2, 4
        k = 2

        # Build complex site tensor (left-canonical)
        sym = U1Symmetry()
        charges_l = np.zeros(chi_l, dtype=np.int32)
        charges_d = np.zeros(d, dtype=np.int32)
        charges_r = np.zeros(chi_r, dtype=np.int32)

        re = jax.random.normal(k1, (chi_l, d, chi_r))
        im = jax.random.normal(k2, (chi_l, d, chi_r))
        data = re + 1j * im
        mat = data.reshape(chi_l * d, chi_r)
        q, _ = jnp.linalg.qr(mat)
        q = q[:, :chi_r]
        data_lc = q.reshape(chi_l, d, chi_r)

        indices = (
            TensorIndex(sym, charges_l, FlowDirection.IN, label="left"),
            TensorIndex(sym, charges_d, FlowDirection.IN, label="phys"),
            TensorIndex(sym, charges_r, FlowDirection.OUT, label="right"),
        )
        site = DenseTensor(data_lc, indices)
        right = _make_right_tensor(chi_r, d2, chi_rr, k3, dtype=jnp.complex128)

        # Complex Hermitian matvec
        n = chi_l * d * d2 * chi_rr
        M = jax.random.normal(k4, (n, n)) + 1j * jax.random.normal(k5, (n, n))
        M = (M + M.conj().T) / 2.0

        def matvec(v_flat, shape):
            return M @ v_flat

        result = expand_bond(
            site, matvec, right, k=k, oversampling=3, key=jax.random.PRNGKey(99)
        )
        dense = result.todense()

        # Check shape
        assert dense.shape == (chi_l, d, chi_r + k)

        # Check orthogonality
        A_mat = data_lc.reshape(chi_l * d, chi_r)
        new_cols = dense.reshape(chi_l * d, chi_r + k)[:, chi_r:]
        overlap = A_mat.conj().T @ new_cols
        assert jnp.allclose(overlap, 0.0, atol=1e-10), (
            f"Complex overlap too large: {float(jnp.max(jnp.abs(overlap)))}"
        )

        # Check dtype preserved
        assert jnp.iscomplexobj(dense)


class TestEnergyImprovement:
    """Integration test: 2-site DMRG gives lower energy than 1-site."""

    def test_energy_improves_with_expansion(self):
        L = 6
        mpo = build_mpo_heisenberg(L)

        # 1-site DMRG with small bond dim
        mps = build_random_symmetric_mps(L, bond_dim=4, seed=0)
        config_1site = DMRGConfig(
            max_bond_dim=4,
            num_sweeps=5,
            two_site=False,
            verbose=False,
        )
        result_1site = dmrg(mpo, mps, config_1site)
        E_1site = result_1site.energy

        # 2-site DMRG with larger bond dim (more variational freedom)
        mps2 = build_random_symmetric_mps(L, bond_dim=4, seed=0)
        config_2site = DMRGConfig(
            max_bond_dim=6,
            num_sweeps=10,
            two_site=True,
            verbose=False,
        )
        result_2site = dmrg(mpo, mps2, config_2site)
        E_2site = result_2site.energy

        # 2-site should achieve lower or equal energy
        assert E_2site <= E_1site + 1e-8, (
            f"2-site energy {E_2site} should be <= 1-site energy {E_1site}"
        )
