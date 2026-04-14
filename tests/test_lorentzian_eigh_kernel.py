"""Tests for the shared Lorentzian-regularized truncated-eigh kernel.

Task 2 of the multisite c4v_reference AD plan hoists the kernel
(``_truncated_eigh_lorentzian_backward`` + ``truncated_eigh_regularized``)
out of ``_ctm_tensor_c4v_reference_ad`` into its own module so it can be
reused by the projector backward path. See
``docs/plans/2026-04-13-multisite-c4v-reference-ad-plan.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms._lorentzian_eigh import (
    _truncated_eigh_lorentzian_backward,  # noqa: F401 — surface check
    truncated_eigh_regularized,
)


def test_lorentzian_dense_matches_fd_on_symmetric_matrix():
    """FD vs AD agreement for the custom_vjp Lorentzian-regularized eigh."""
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (6, 6))
    A = 0.5 * (A + A.T)
    chi = 4

    def loss(mat):
        w, U = truncated_eigh_regularized(mat, chi)
        return jnp.sum(jnp.sin(U)) + jnp.sum(w**2)

    grad_ad = jax.grad(loss)(A)

    eps = 1e-4
    grad_fd = jnp.zeros_like(A)
    for i in range(6):
        for j in range(6):
            Ap = A.at[i, j].add(eps)
            Am = A.at[i, j].add(-eps)
            grad_fd = grad_fd.at[i, j].set((loss(Ap) - loss(Am)) / (2 * eps))

    assert jnp.allclose(grad_ad, grad_fd, atol=1e-7)
