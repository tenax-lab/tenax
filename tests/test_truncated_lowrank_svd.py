"""Tests for the AD-stable truncated low-rank SVD (fast projector decomposition).

The 2x2 CTM projector half-systems are exactly rank-χ but were full-SVD'd at
χD²×χD² (slow GPU cuSOLVER). truncated_lowrank_svd computes the top-k SVD via a
randomized range finder + a stable small-matrix truncated SVD VJP, exact for
rank ≤ k.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ad_primitives import truncated_lowrank_svd, truncated_svd_ad


def _rank_k_matrix(m, n, k, seed=0):
    ka, kb = jax.random.split(jax.random.PRNGKey(seed))
    return jax.random.normal(ka, (m, k)) @ jax.random.normal(kb, (k, n))


def test_reconstructs_rank_k_matrix():
    """For a rank-k matrix, the top-k SVD reconstructs it to machine precision."""
    M = _rank_k_matrix(200, 200, 16, seed=1)
    U, s, Vh = truncated_lowrank_svd(M, 16)
    assert U.shape == (200, 16)
    assert s.shape == (16,)
    assert Vh.shape == (16, 200)
    recon = (U * s) @ Vh
    assert float(jnp.max(jnp.abs(recon - M))) < 1e-9


def test_singular_values_match_full_svd():
    """Top-k singular values equal the full SVD's (rank-k matrix)."""
    M = _rank_k_matrix(256, 256, 16, seed=2)
    _, s, _ = truncated_lowrank_svd(M, 16)
    s_full = np.asarray(jnp.linalg.svd(M, compute_uv=False))[:16]
    np.testing.assert_allclose(np.sort(np.asarray(s))[::-1], s_full, rtol=1e-8, atol=1e-9)


def test_gradient_matches_reference_truncated_svd():
    """The singular-value gradient (gauge-invariant) matches the reference
    stable truncated SVD — confirms the randomized path's VJP is correct."""
    M = _rank_k_matrix(128, 128, 16, seed=3)
    g_lr = jax.grad(lambda M: jnp.sum(truncated_lowrank_svd(M, 16)[1]))(M)
    g_ref = jax.grad(lambda M: jnp.sum(truncated_svd_ad(M, 16)[1]))(M)
    assert float(jnp.max(jnp.abs(g_lr - g_ref))) < 1e-7


def test_power_iterations_sharpen_decaying_spectrum():
    """For a slowly-decaying (not sharply-truncated) spectrum, power iterations
    make the randomized top-k singular values accurate."""
    m = n = 128
    k = 8
    qu, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(10), (m, m)))
    qv, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(11), (n, n)))
    s_true = 0.85 ** jnp.arange(min(m, n))  # slow geometric decay, full rank
    M = (qu * s_true) @ qv.conj().T
    s_true_k = np.asarray(s_true[:k])

    def topk_err(n_power):
        s = truncated_lowrank_svd(M, k, n_power_iterations=n_power)[1]
        return float(jnp.max(jnp.abs(jnp.sort(s)[::-1] - s_true_k)))

    err0, err2 = topk_err(0), topk_err(2)
    assert err2 < err0  # power iterations help
    assert err2 < 1e-6  # and reach high accuracy


def test_jit_compatible():
    """Works under jit (the projector runs inside a jitted CTM step)."""
    M = _rank_k_matrix(200, 200, 16, seed=4)
    U, s, Vh = jax.jit(lambda M: truncated_lowrank_svd(M, 16))(M)
    assert float(jnp.max(jnp.abs((U * s) @ Vh - M))) < 1e-9
