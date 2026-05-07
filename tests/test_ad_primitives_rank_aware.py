"""Rank-aware truncated SVD: zero modes inside [0, chi) must be pruned."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ad_primitives import truncated_svd_ad

jax.config.update("jax_enable_x64", True)


@pytest.mark.core
def test_rank_aware_truncation_zeros_subrank_singular_values():
    """A rank-4 8x8 matrix truncated to chi=8 must return s[4:] == 0 exactly.

    Currently fails because top-k truncation keeps numerical-noise zeros
    (~1e-15) instead of zeroing them. Multiplet-aware guard only fires
    when chi cuts through a degenerate group at the boundary; it does not
    catch zero modes that survive because chi >= rank.
    """
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (8, 4), dtype=jnp.float64)
    M = A @ A.T  # 8x8 symmetric, rank 4

    U, s, Vh = truncated_svd_ad(M, chi=8)
    assert s.shape == (8,)
    # Top 4 singular values must be > 0
    assert jnp.all(s[:4] > 1e-6), f"top-4 SVs collapsed: {s[:4]}"
    # Bottom 4 must be EXACTLY 0 after rank-aware truncation
    assert jnp.all(s[4:] == 0.0), (
        f"zero modes leaked into kept set: s[4:] = {s[4:]} "
        "(rank-aware truncation should set them to 0)"
    )


@pytest.mark.core
def test_rank_aware_backward_matches_chi_equals_rank():
    """Gradient through chi=8 (with rank-4 matrix) must equal gradient
    through chi=4 (proper rank truncation). If zero modes contaminate
    the chi=8 backward via gauge-artifact F-matrix entries or perp-
    projector dimension errors, the two gradients differ.
    """
    key = jax.random.PRNGKey(1)
    A = jax.random.normal(key, (8, 4), dtype=jnp.float64)
    M = A @ A.T  # rank 4

    target = jax.random.normal(jax.random.PRNGKey(2), (8, 8), dtype=jnp.float64)

    def loss(M_in, chi):
        U, s, Vh = truncated_svd_ad(M_in, chi=chi)
        M_rec = U @ jnp.diag(s) @ Vh
        return jnp.sum((M_rec - target) ** 2)

    g_chi8 = jax.grad(loss, argnums=0)(M, 8)
    g_chi4 = jax.grad(loss, argnums=0)(M, 4)
    np.testing.assert_allclose(
        np.asarray(g_chi8),
        np.asarray(g_chi4),
        atol=1e-10,
        err_msg="rank-aware: chi=rank+slack must give same grad as chi=rank",
    )
