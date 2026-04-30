"""Tests for the Arnoldi spectral-radius estimator."""

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._arnoldi import (
    arnoldi_spectral_radius,
    arnoldi_spectral_radius_pytree,
)


def test_contractive_matrix():
    """Diagonal matrix with max eigenvalue 0.9 should give rho ~ 0.9."""
    eigs = jnp.array([0.9, 0.5, 0.3, 0.1])
    A = jnp.diag(eigs)
    v0 = jnp.ones(4)
    rho = arnoldi_spectral_radius(lambda v: A @ v, v0, n_iter=20)
    assert 0.85 < rho < 0.95, f"Expected rho ~ 0.9, got {rho}"


def test_non_contractive_matrix():
    """Diagonal matrix with eigenvalue 1.5 should give rho > 1."""
    eigs = jnp.array([1.5, 0.5, 0.3])
    A = jnp.diag(eigs)
    v0 = jnp.ones(3)
    rho = arnoldi_spectral_radius(lambda v: A @ v, v0, n_iter=20)
    assert rho > 1.0, f"Expected rho > 1, got {rho}"


def test_complex_matrix():
    """Complex diagonal matrix with |eigenvalues| < 1 should give rho < 1."""
    eigs = jnp.array([0.5 + 0.5j, 0.3 - 0.4j, 0.1 + 0.2j])
    A = jnp.diag(eigs)
    v0 = jnp.ones(3, dtype=jnp.complex128)
    rho = arnoldi_spectral_radius(lambda v: A @ v, v0, n_iter=20)
    expected_max = float(jnp.max(jnp.abs(eigs)))  # ~0.707
    assert rho < 1.0, f"Expected rho < 1, got {rho}"
    assert abs(rho - expected_max) < 0.1, f"Expected rho ~ {expected_max}, got {rho}"


def test_pytree_matvec():
    """Pytree matvec with tuple-of-arrays, scaled by (0.5, 0.9)."""

    def matvec(x):
        a, b = x
        return (0.5 * a, 0.9 * b)

    v0 = (jnp.ones(3), jnp.ones(4))
    rho = arnoldi_spectral_radius_pytree(matvec, v0, n_iter=20)
    assert 0.85 < rho < 0.95, f"Expected rho ~ 0.9, got {rho}"
