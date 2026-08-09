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


# --- #828: there must be exactly one implementation ------------------------


def test_ad_utils_reexports_the_reviewed_implementation():
    """``ad_utils`` must not carry its own copy of this function.

    It did, and the copy was wrong on complex input: real Krylov buffers and an
    unconjugated ``jnp.dot`` for the Gram-Schmidt projection.  The explicit-AD
    CTM backward calls it as a *divergence precheck* -- ``if rho >= threshold:
    raise CTMRGGradientError`` -- so under-reporting rho passes a divergent
    adjoint through the guard that exists to catch it.

    Pinned as an identity rather than a value check: a future edit that
    reintroduces a local definition fails here even if it happens to be
    correct, because two copies is the defect (#828, and #829 the same week).
    """
    from tenax.algorithms import _arnoldi, ad_utils

    assert ad_utils.arnoldi_spectral_radius is _arnoldi.arnoldi_spectral_radius


def test_the_explicit_ad_entry_point_is_correct_on_complex_input():
    """The symptom #828 was filed for, at the name the backward actually calls.

    ``diag(2i, 1, 0.5)`` has rho = 2 unambiguously.  The stale copy returned
    0.972 -- below even the rho >= 1 divergence line.
    """
    from tenax.algorithms.ad_utils import arnoldi_spectral_radius as rho_fn

    A = jnp.diag(jnp.array([2j, 1.0 + 0j, 0.5 + 0j]))
    v0 = jnp.array([1.0 + 1j, 0.3 - 0.2j, 0.1 + 0.4j])
    rho = rho_fn(lambda v: A @ v, v0, n_iter=3)
    assert abs(rho - 2.0) < 1e-9, f"expected rho = 2.0, got {rho}"
