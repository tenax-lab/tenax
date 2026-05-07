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
def test_rank_aware_backward_finite_at_chi_geq_rank():
    """chi >= rank with rank-aware truncation must give a finite gradient.

    The F-matrix entries (sigma_i^2 - 0)/((sigma_i^2)^2 + eps^2) ~ 1/sigma_i^2
    are gauge artifacts when one of the pair is a zero mode. Without rank-
    aware F-masking they pump arbitrary cotangent components into dM through
    1/sigma_i^2; this test pins a finite gradient as the contract.

    (We do NOT assert chi=rank+slack gradient equals chi=rank gradient —
    finite differences confirm those two truncation schedules give genuinely
    different gradients, since perturbing M shifts modes across the chi cut
    differently in each scheme.)
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
    assert jnp.all(jnp.isfinite(g_chi8)), (
        f"chi=rank+slack backward not finite: g = {g_chi8}"
    )
    g_chi4 = jax.grad(loss, argnums=0)(M, 4)
    assert jnp.all(jnp.isfinite(g_chi4)), f"chi=rank backward not finite: g = {g_chi4}"


@pytest.mark.core
def test_rank_aware_vh_only_zeros_subrank_singular_values():
    """Same rank-aware contract for the half-SVD vh_only variant."""
    from tenax.algorithms._ad_primitives import truncated_svd_ad_vh_only

    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (8, 4), dtype=jnp.float64)
    M = A @ A.T

    s, Vh = truncated_svd_ad_vh_only(M, chi=8)
    assert s.shape == (8,)
    assert jnp.all(s[:4] > 1e-6)
    assert jnp.all(s[4:] == 0.0), f"vh_only zero modes leaked: s[4:] = {s[4:]}"


@pytest.mark.core
def test_rank_aware_regularized_svd_backward():
    """regularized_svd on a rank-deficient input must give a finite gradient.

    The Wan-Narayanan / Francuz adjoint as implemented in
    ``_svd_sector_backward`` is not the same VJP as JAX's standard
    ``jnp.linalg.svd`` adjoint — even on full-rank inputs the two
    differ by O(1) (gauge convention in U/Vh phases), so we cannot assert
    ``g == 2(M - target)``. The rank-aware F-mask added in Task 3 is the
    point of this test: pre-existing F gauge artifacts at (sigma>0, sigma=0)
    pairs no longer pollute the gradient.
    """
    from tenax.algorithms._ad_primitives import regularized_svd

    key = jax.random.PRNGKey(3)
    A = jax.random.normal(key, (6, 3), dtype=jnp.float64)
    M = A @ A.T  # 6x6, rank 3

    target = jax.random.normal(jax.random.PRNGKey(4), (6, 6), dtype=jnp.float64)

    def loss(M_in):
        U, s, Vh = regularized_svd(M_in)
        M_rec = U @ jnp.diag(s) @ Vh
        return jnp.sum((M_rec - target) ** 2)

    g = jax.grad(loss)(M)
    assert jnp.all(jnp.isfinite(g)), "non-finite grad on rank-deficient input"


@pytest.mark.algorithm
def test_ctm_projector_rank_deficient_corner():
    """_compute_projector_tensor on a rank-deficient corner-cross-product
    must return finite projectors and a finite gradient through them."""
    from tenax.algorithms._ctm_projector import _compute_projector_tensor
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor

    # Synthetic corners (D2, col) where the cross-product M = C1g^T @ C4g is
    # rank-deficient: build C1g, C4g so the dummy last column of each is 0.
    sym = U1Symmetry()
    D2 = 8
    col = 6
    rank = 4

    fused_idx = TensorIndex.from_charges(
        sym, np.zeros(D2, dtype=np.int32), FlowDirection.IN, label="fused"
    )
    col1_idx = TensorIndex.from_charges(
        sym, np.zeros(col, dtype=np.int32), FlowDirection.OUT, label="col1"
    )
    col2_idx = TensorIndex.from_charges(
        sym, np.zeros(col, dtype=np.int32), FlowDirection.OUT, label="col2"
    )

    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    C1_data_full = jax.random.normal(k1, (D2, rank), dtype=jnp.float64)
    C4_data_full = jax.random.normal(k2, (D2, rank), dtype=jnp.float64)
    # Pad to (D2, col) with zero columns — makes M = C1g^T @ C4g rank-deficient.
    C1_data = jnp.concatenate(
        [C1_data_full, jnp.zeros((D2, col - rank), dtype=jnp.float64)], axis=1
    )
    C4_data = jnp.concatenate(
        [C4_data_full, jnp.zeros((D2, col - rank), dtype=jnp.float64)], axis=1
    )

    C1g = DenseTensor(C1_data, (fused_idx, col1_idx))
    C4g = DenseTensor(C4_data, (fused_idx, col2_idx))

    chi = 4
    P1, P2 = _compute_projector_tensor(C1g, C4g, chi, projector_method="svd")

    P1d = np.asarray(P1.todense())
    P2d = np.asarray(P2.todense())
    assert np.all(np.isfinite(P1d)), "P1 contains non-finite entries"
    assert np.all(np.isfinite(P2d)), "P2 contains non-finite entries"

    # Differentiability: gradient w.r.t. C1g data must be finite.
    def loss(C1_data_in):
        C1g_in = DenseTensor(C1_data_in, (fused_idx, col1_idx))
        P1_in, _ = _compute_projector_tensor(C1g_in, C4g, chi, projector_method="svd")
        return jnp.sum(P1_in.todense() ** 2)

    g = jax.grad(loss)(C1_data)
    assert np.all(np.isfinite(g)), (
        f"non-finite grad through rank-deficient projector: max|g|={float(jnp.max(jnp.abs(g)))}"
    )
