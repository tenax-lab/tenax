"""Lorentzian-regularized truncated symmetric eigendecomposition kernel.

This module hosts the dense Lorentzian-regularized truncated-eigh kernel
and its custom-VJP wrapper. It was hoisted out of
``_ctm_tensor_c4v_reference_ad`` so that the same backward can be reused
by the projector path for multi-sublattice CTM (checkerboard, honeycomb,
kagome) without recreating a second copy of the formula. See
``docs/plans/2026-04-13-multisite-c4v-reference-ad-design.md`` for the
design rationale.

The kernel uses a Lorentzian-regularized eigen-gap inverse for both
kept-kept and kept-discarded couplings, which is the
truncation-correction adjoint from:

- Francuz, Schmoll, Rizzi, Eisert, Naumann, "Stable and efficient
  differentiation of tensor network algorithms",
  Phys. Rev. Research 7, 013237 (2025). arXiv:2311.11894.

The matrix construction is scatter-free (uses ``jnp.eye`` masks and
``jnp.concatenate`` instead of ``.at[...].set(...)``) so that the
backward can run inside ``jax.lax.custom_linear_solve``, whose
transpose rule does not support scatter.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

__all__ = [
    "_truncated_eigh_lorentzian_backward",
    "truncated_eigh_regularized",
]


def _truncated_eigh_lorentzian_backward(
    w_full: jax.Array,
    v_full: jax.Array,
    dw_k: jax.Array,
    dV_k: jax.Array,
    *,
    k: int,
    eps: float = 1e-12,
) -> jax.Array:
    """Regularized truncated eigendecomposition backward.

    Uses a Lorentzian-regularized eigen-gap inverse for both kept-kept and
    kept-discarded couplings. The kept-discarded term is the critical
    truncation correction missing in naive truncated-eigh adjoints.
    """
    n = int(w_full.shape[0])
    k = min(int(k), n)
    v_h = jnp.conj(v_full).T
    dtype = jnp.result_type(v_full.dtype, dV_k.dtype, dw_k.dtype)

    dw_full = jnp.concatenate([dw_k.astype(dtype), jnp.zeros((n - k,), dtype=dtype)])

    # Differential coefficients K_{j,i} = <v_j, dV_i> / (w_i - w_j), i in kept.
    # Symmetrizing this n x n matrix recovers both the kept-kept anti-gauge term
    # and the kept-discarded truncation correction from the Francuz et al.
    # regularized differential.
    inner = v_h @ dV_k.astype(dtype)  # (n, k)
    denom = w_full[:k][None, :] - w_full[:, None]  # (n, k): w_i - w_j
    inv_gap = denom / (denom**2 + eps**2)
    K = inner * inv_gap
    if k > 0:
        # Zero the kept-kept diagonal without a scatter (scatter transpose
        # is unsupported inside custom_linear_solve).
        diag_mask = (1.0 - jnp.eye(n, k)).astype(dtype)
        K = K * diag_mask

    # Pad K (n, k) to K_full (n, n) by concatenation instead of scatter.
    K_full = jnp.concatenate([K, jnp.zeros((n, n - k), dtype=dtype)], axis=1)
    C = jnp.diag(dw_full) + 0.5 * (K_full + jnp.conj(K_full).T)
    dM = v_full @ C @ v_h
    return 0.5 * (dM + jnp.conj(dM).T)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def truncated_eigh_regularized(
    M: jax.Array,
    chi: int,
) -> tuple[jax.Array, jax.Array]:
    """Truncated symmetric eigendecomposition with a regularized VJP."""
    w, v = jnp.linalg.eigh(M)
    k = min(int(chi), int(w.shape[0]))
    return w[:k], v[:, :k]


def _truncated_eigh_regularized_fwd(M: jax.Array, chi: int):
    M_h = 0.5 * (M + jnp.conj(M).T)
    w, v = jnp.linalg.eigh(M_h)
    k = min(int(chi), int(w.shape[0]))
    return (w[:k], v[:, :k]), (w, v, k)


def _truncated_eigh_regularized_bwd(chi: int, residuals, g):
    w, v, k = residuals
    dw_k, dV_k = g
    dM = _truncated_eigh_lorentzian_backward(w, v, dw_k, dV_k, k=k)
    return (dM,)


truncated_eigh_regularized.defvjp(
    _truncated_eigh_regularized_fwd, _truncated_eigh_regularized_bwd
)
