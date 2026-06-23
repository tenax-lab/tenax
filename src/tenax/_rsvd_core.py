"""Shared randomized-SVD core (Halko, Martinsson & Tropp 2011).

A single qr/svd-parameterized HMT randomized SVD, used by BOTH
:func:`tenax.linalg.rsvd` (plain ``jnp.linalg`` decompositions) and
:func:`tenax.algorithms._ad_primitives.truncated_lowrank_svd` (AD-stable
``regularized_qr`` / ``truncated_svd_ad`` decompositions). Injecting the qr/svd
functions lets the two callers share the algorithm despite living on opposite
sides of the ``algorithms ↔ linalg`` import boundary — this module depends only
on ``jax``/``numpy`` so it is a safe leaf for both.
"""

from __future__ import annotations

from collections.abc import Callable

import jax


def hmt_rsvd(
    M: jax.Array,
    k: int,
    *,
    oversampling: int,
    n_power_iter: int,
    key: jax.Array,
    qr_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    svd_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Randomized top-*k* SVD of a 2-D array (HMT 2011), returning ``(U, s, Vh)``
    with shapes ``(m, r)``, ``(r,)``, ``(r, n)`` for ``r = min(k, ·)``.

    Range finder (Gaussian sketch + ``qr_fn``, with re-orthogonalized subspace
    iteration) → project to ``B = Qᴴ M`` → ``svd_fn(B)`` → lift ``U = Q U_B``.

    ``qr_fn(X) -> (Q, R)`` and ``svd_fn(B) -> (U, s, Vh)`` are injected so the
    caller chooses plain vs AD-stable (Lorentzian-VJP) decompositions. Uses
    ``conjᵀ`` so it is correct for complex ``M`` as well as real.
    """
    m, n = M.shape
    ell = min(k + oversampling, m, n)
    omega = jax.random.normal(key, (n, ell), dtype=M.dtype)
    q = qr_fn(M @ omega)[0]
    for _ in range(n_power_iter):
        q = qr_fn(M.conj().T @ q)[0]
        q = qr_fn(M @ q)[0]
    b = q.conj().T @ M
    u_hat, s, vh = svd_fn(b)
    u = q @ u_hat
    r = min(k, s.shape[0])
    return u[:, :r], s[:r], vh[:r, :]
