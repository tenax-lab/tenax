"""Tests for lax.while_loop-based GMRES(m) solver."""

import jax
import jax.numpy as jnp

from tenax.algorithms._gmres_lax import gmres_lax


def _matvec_from_matrix(A):
    """Return a matvec closure for matrix A."""

    def matvec(x):
        return A @ x

    return matvec


def test_gmres_solves_simple_linear_system():
    """Solve A x = b where A = 2*I (trivial)."""
    n = 10
    A = 2.0 * jnp.eye(n)
    b = jnp.ones(n)
    x, info = gmres_lax(_matvec_from_matrix(A), b, tol=1e-10, maxiter=50)
    assert jnp.allclose(x, 0.5 * jnp.ones(n), atol=1e-8)
    assert info == 0  # converged


def test_gmres_solves_spd_system():
    """Solve (I - 0.5*J) x = b where J is a contraction."""
    key = jax.random.PRNGKey(42)
    n = 20
    M = jax.random.normal(key, (n, n))
    J = 0.5 * M / jnp.linalg.norm(M, ord=2)
    A = jnp.eye(n) - J
    b = jax.random.normal(jax.random.PRNGKey(1), (n,))
    x, info = gmres_lax(_matvec_from_matrix(A), b, tol=1e-8, maxiter=100)
    x_ref = jnp.linalg.solve(A, b)
    assert jnp.allclose(x, x_ref, atol=1e-6)


def test_gmres_works_with_spectral_radius_above_one():
    """GMRES should still solve even when rho(J) > 1 (unlike Neumann)."""
    key = jax.random.PRNGKey(0)
    n = 15
    M = jax.random.normal(key, (n, n))
    J = 2.0 * M / jnp.linalg.norm(M, ord=2)
    A = jnp.eye(n) - J
    b = jax.random.normal(jax.random.PRNGKey(1), (n,))
    x, info = gmres_lax(_matvec_from_matrix(A), b, tol=1e-6, maxiter=200)
    x_ref = jnp.linalg.solve(A, b)
    assert jnp.allclose(x, x_ref, atol=1e-4)


def test_gmres_is_jit_compatible():
    """Must work inside jax.jit."""
    n = 10
    A = 2.0 * jnp.eye(n)
    b = jnp.ones(n)
    matvec = _matvec_from_matrix(A)

    @jax.jit
    def solve(b):
        x, info = gmres_lax(matvec, b, tol=1e-10, maxiter=50)
        return x

    x = solve(b)
    assert jnp.allclose(x, 0.5 * jnp.ones(n), atol=1e-8)


def test_gmres_restart():
    """GMRES(m) with restart should still converge."""
    key = jax.random.PRNGKey(7)
    n = 50
    M = jax.random.normal(key, (n, n))
    A = jnp.eye(n) + 0.3 * (M + M.T) / n
    b = jax.random.normal(jax.random.PRNGKey(2), (n,))
    x, info = gmres_lax(_matvec_from_matrix(A), b, tol=1e-8, maxiter=200, restart=10)
    x_ref = jnp.linalg.solve(A, b)
    assert jnp.allclose(x, x_ref, atol=1e-5)
