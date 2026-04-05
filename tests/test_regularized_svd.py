"""Tests for regularized (non-truncated) SVD with stable backward."""

import jax
import jax.numpy as jnp
import pytest


class TestRegularizedSVD:
    def test_forward_matches_jnp_svd(self):
        """Forward pass should match jnp.linalg.svd."""
        from tenax.algorithms.ad_utils import regularized_svd

        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))
        U, s, Vh = regularized_svd(M)
        U_ref, s_ref, Vh_ref = jnp.linalg.svd(M, full_matrices=False)
        assert jnp.allclose(jnp.abs(U), jnp.abs(U_ref), atol=1e-5)
        assert jnp.allclose(s, s_ref, atol=1e-5)

    def test_gradient_finite(self):
        """Gradient through regularized_svd should be finite."""
        from tenax.algorithms.ad_utils import regularized_svd

        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))

        def loss(M):
            U, s, Vh = regularized_svd(M)
            return jnp.sum(s[:3] ** 2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_with_degeneracy(self):
        """Gradient should stay finite even with degenerate singular values."""
        from tenax.algorithms.ad_utils import regularized_svd

        # Build a matrix with degenerate singular values but non-trivial
        # structure so that gradient is nonzero.
        key = jax.random.PRNGKey(42)
        Q1, _ = jnp.linalg.qr(jax.random.normal(key, (6, 6)))
        key2 = jax.random.PRNGKey(7)
        Q2, _ = jnp.linalg.qr(jax.random.normal(key2, (4, 4)))
        s = jnp.array([2.0, 2.0, 1.0, 1.0])
        M = Q1[:, :4] @ jnp.diag(s) @ Q2

        def loss(M):
            U, s, Vh = regularized_svd(M)
            return jnp.sum(s**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 0
