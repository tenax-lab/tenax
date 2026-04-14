"""Tests for the projector backward dispatch (Task 7).

Covers two levels of the Task 7 plumbing:

(a) At the *leaf* — confirm that the standard backward
    ``ad_utils.regularized_eigh`` and the Lorentzian backward
    ``_lorentzian_eigh.truncated_eigh_regularized`` produce the same
    projector columns (to sign) and agree on their gradients on a
    non-degenerate 6x6 PSD density matrix ``rho = C @ C.T``. This is
    the "close to" check that motivates the dispatch: both backwards
    are valid on a non-degenerate spectrum, so they should give
    equivalent gradients.

(b) At the *entry point* — confirm that ``ctm_tensor`` accepts the
    new ``projector_backward`` keyword and runs one sweep without
    raising, producing finite output. This is the end-to-end wiring
    smoke test: it fails before plumbing (unknown kwarg) and passes
    after.

See docs/plans/2026-04-13-multisite-c4v-reference-ad-plan.md Task 7.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._lorentzian_eigh import truncated_eigh_regularized
from tenax.algorithms.ad_utils import regularized_eigh
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _projector_cols_standard(rho: jax.Array, k: int) -> jax.Array:
    """Top-k projector columns via ``regularized_eigh`` (ascending → reverse)."""
    eigvals, eigvecs = regularized_eigh(rho)
    return eigvecs[:, -k:][:, ::-1]


def _projector_cols_lorentzian(rho: jax.Array, k: int) -> jax.Array:
    """Top-k projector columns via ``truncated_eigh_regularized(-rho, k)``."""
    _, P = truncated_eigh_regularized(-rho, k)
    return P


def test_projector_forward_columns_match_on_non_degenerate_rho():
    """Forward columns from standard vs Lorentzian branch must match.

    This confirms the dispatch doesn't change the fixed-point forward:
    ``truncated_eigh_regularized(-rho, k)`` returns the same top-k
    eigenvectors (up to floating-point noise) as
    ``regularized_eigh(rho)[:, -k:][:, ::-1]``.
    """
    key = jax.random.PRNGKey(0)
    n, k = 6, 3
    C = jax.random.normal(key, (n, n))
    rho = C @ C.T
    rho = 0.5 * (rho + rho.T)

    P_std = _projector_cols_standard(rho, k)
    P_lor = _projector_cols_lorentzian(rho, k)

    # Eigenvectors can differ by a ±1 phase per column. Compare |Pᵀ P'|
    # which is diag(±1) for matching columns.
    overlap = jnp.abs(P_std.T @ P_lor)
    np.testing.assert_allclose(np.asarray(overlap), np.eye(k), atol=1e-10)


def test_lorentzian_projector_backward_matches_fd():
    """Lorentzian kernel backward vs finite differences on a projector loss.

    The Lorentzian kernel is the primary object this plumbing wires in;
    here we pin it: its VJP for the ``truncated_eigh_regularized(-rho, k)``
    projector path must match FD to ~1e-5 on a non-degenerate 6x6 PSD ρ.
    The loss ``tr(PᵀMP)`` is a standard gauge-invariant subspace loss.
    """
    key = jax.random.PRNGKey(0)
    key_c, key_m, key_h = jax.random.split(key, 3)
    n, k = 6, 3
    C = jax.random.normal(key_c, (n, n))
    M = jax.random.normal(key_m, (n, n))
    M = 0.5 * (M + M.T)

    def loss(c):
        rho = c @ c.T
        rho = 0.5 * (rho + rho.T)
        P = _projector_cols_lorentzian(rho, k)
        return jnp.trace(P.T @ M @ P)

    g = jax.grad(loss)(C)
    h = jax.random.normal(key_h, C.shape)
    eps = 1e-5
    fd = (loss(C + eps * h) - loss(C - eps * h)) / (2.0 * eps)
    ad = jnp.sum(g * h)
    np.testing.assert_allclose(float(ad), float(fd), atol=1e-5, rtol=1e-5)


def test_ctm_tensor_accepts_projector_backward_lorentzian():
    """``ctm_tensor`` must accept ``projector_backward='lorentzian'`` and run."""
    A_raw = jax.random.normal(jax.random.PRNGKey(0), (2, 2, 2, 2, 2))
    A = _wrap_as_dense_tensor(A_raw)
    env = ctm_tensor(
        A,
        chi=4,
        max_iter=2,
        projector_method="eigh",
        projector_backward="lorentzian",
    )
    # Output must be finite
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        t = getattr(env, name)
        assert jnp.all(jnp.isfinite(t.todense())), f"{name} has non-finite entries"
