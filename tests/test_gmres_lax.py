"""Tests for lax.while_loop-based GMRES(m) solver."""

import jax
import jax.numpy as jnp

from tenax.algorithms._gmres_lax import gmres_lax, gmres_pytree


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


def test_gmres_pytree_dict():
    """GMRES on a pytree of arrays (simulating CTM env)."""
    key = jax.random.PRNGKey(99)
    tree_template = {
        "C1": jnp.zeros((4, 4)),
        "T1": jnp.zeros((4, 3, 4)),
    }
    leaves, treedef = jax.tree.flatten(tree_template)
    flat_sizes = [leaf.size for leaf in leaves]
    total = sum(flat_sizes)
    M = jax.random.normal(key, (total, total))
    A = jnp.eye(total) + 0.3 * (M + M.T) / total

    def matvec_pytree(v_tree):
        v_flat = jnp.concatenate([leaf.ravel() for leaf in jax.tree.leaves(v_tree)])
        y_flat = A @ v_flat
        out_leaves = []
        offset = 0
        for sz, leaf in zip(flat_sizes, jax.tree.leaves(v_tree)):
            out_leaves.append(y_flat[offset : offset + sz].reshape(leaf.shape))
            offset += sz
        return jax.tree.unflatten(treedef, out_leaves)

    b_tree = jax.tree.map(
        lambda x: jax.random.normal(jax.random.PRNGKey(3), x.shape), tree_template
    )
    x_tree, info = gmres_pytree(matvec_pytree, b_tree, tol=1e-8, maxiter=100)

    residual = jax.tree.map(lambda ax, b: ax - b, matvec_pytree(x_tree), b_tree)
    res_norm = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(residual)))
    b_norm = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(b_tree)))
    assert res_norm / b_norm < 1e-6


def test_gmres_pytree_jit_compatible():
    """gmres_pytree must work inside jax.jit."""
    tree = {"a": jnp.ones(5), "b": jnp.ones(3)}

    def matvec(t):
        return jax.tree.map(lambda x: 2.0 * x, t)

    @jax.jit
    def solve(b):
        x, info = gmres_pytree(matvec, b, tol=1e-10, maxiter=50)
        return x

    x = solve(tree)
    assert jnp.allclose(x["a"], 0.5 * jnp.ones(5), atol=1e-8)
    assert jnp.allclose(x["b"], 0.5 * jnp.ones(3), atol=1e-8)
