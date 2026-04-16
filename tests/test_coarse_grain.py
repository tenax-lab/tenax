"""Tests for coarse-grained iPEPS gate construction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.coarse_grain import CGGates, honeycomb_cg_gates


@pytest.fixture(autouse=True)
def _enable_x64():
    """Enable float64 for this test module and restore afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_returns_cg_gates():
    """honeycomb_cg_gates returns a CGGates with n_sites == 2."""
    gates = honeycomb_cg_gates()
    assert isinstance(gates, CGGates)
    assert gates.n_sites == 2


def test_h_intra_shape_and_hermiticity():
    """h_intra is a (4,4) Hermitian matrix."""
    gates = honeycomb_cg_gates()
    assert gates.h_intra.shape == (4, 4)
    np.testing.assert_allclose(gates.h_intra, gates.h_intra.conj().T, atol=1e-14)


def test_h_intra_eigenvalues():
    """S*S eigenvalues are [-3/4, +1/4, +1/4, +1/4] (singlet-triplet)."""
    gates = honeycomb_cg_gates()
    eigvals = np.sort(np.linalg.eigvalsh(np.asarray(gates.h_intra)))
    expected = np.array([-3 / 4, 1 / 4, 1 / 4, 1 / 4])
    np.testing.assert_allclose(eigvals, expected, atol=1e-14)


def test_h_inter_keys_and_shapes():
    """h_inter has keys {'h', 'v'}, each with shape (4,4,4,4)."""
    gates = honeycomb_cg_gates()
    assert set(gates.h_inter.keys()) == {"h", "v"}
    for key in ("h", "v"):
        assert gates.h_inter[key].shape == (4, 4, 4, 4)


def test_h_inter_hermiticity():
    """Each h_inter reshaped to (16,16) is Hermitian."""
    gates = honeycomb_cg_gates()
    for key in ("h", "v"):
        mat = gates.h_inter[key].reshape(16, 16)
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-14)
