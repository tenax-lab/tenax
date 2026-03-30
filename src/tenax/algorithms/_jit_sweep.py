"""JIT-compatible dense environment update functions for DMRG sweeps.

These functions operate on raw JAX arrays (not Tensor objects) and produce
padded outputs with fixed shapes, making them compatible with ``jax.jit``
and ``jax.lax.scan``.

The einsum contraction patterns match exactly those used in
:func:`tenax.algorithms.dmrg._update_left_env` and
:func:`tenax.algorithms.dmrg._update_right_env`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def update_left_env_dense_jit(
    L_env: jax.Array,
    A: jax.Array,
    W: jax.Array,
    chi_max: int,
) -> jax.Array:
    """Compute a left environment update and pad output to fixed shape.

    Contracts: new_L[d,e,f] = L[a,b,c] * A[a,p,d] * W[b,p,x,e] * conj(A)[c,x,f]

    This matches the einsum ``"abc,apd,bpxe,cxf->def"`` from ``_update_left_env``.

    Args:
        L_env: Left environment, shape ``(chi_l, D_w_in, chi_l)``.
            May already be padded to ``(chi_max, D_w, chi_max)`` -- zero
            padding is handled correctly by einsum.
        A: MPS site tensor, shape ``(chi_l, d, chi_r)``.
            May be padded to ``(chi_max, d, chi_max)``.
        W: MPO site tensor, shape ``(D_w_in, d, d, D_w_out)``.
            Not padded (MPO bond dimension is fixed).
        chi_max: Maximum bond dimension for padding.

    Returns:
        Padded left environment of shape ``(chi_max, D_w_out, chi_max)``.
    """
    D_w_out = W.shape[3]

    new_L = jnp.einsum("abc,apd,bpxe,cxf->def", L_env, A, W, jnp.conj(A))

    # Pad to (chi_max, D_w_out, chi_max)
    chi_r = new_L.shape[0]
    padded = jnp.zeros((chi_max, D_w_out, chi_max), dtype=new_L.dtype)
    padded = padded.at[:chi_r, :D_w_out, :chi_r].set(new_L)
    return padded


def update_right_env_dense_jit(
    R_env: jax.Array,
    B: jax.Array,
    W: jax.Array,
    chi_max: int,
) -> jax.Array:
    """Compute a right environment update and pad output to fixed shape.

    Contracts: new_R[d,e,f] = R[a,b,c] * B[d,p,a] * W[e,p,x,b] * conj(B)[f,x,c]

    This matches the einsum ``"abc,dpa,epxb,fxc->def"`` from ``_update_right_env``.

    Args:
        R_env: Right environment, shape ``(chi_r, D_w_in, chi_r)``.
            May already be padded to ``(chi_max, D_w, chi_max)``.
        B: MPS site tensor, shape ``(chi_l, d, chi_r)``.
            May be padded to ``(chi_max, d, chi_max)``.
        W: MPO site tensor, shape ``(D_w_out, d, d, D_w_in)``.
            Not padded (MPO bond dimension is fixed).
        chi_max: Maximum bond dimension for padding.

    Returns:
        Padded right environment of shape ``(chi_max, D_w_out, chi_max)``.
    """
    D_w_out = W.shape[0]

    new_R = jnp.einsum("abc,dpa,epxb,fxc->def", R_env, B, W, jnp.conj(B))

    # Pad to (chi_max, D_w_out, chi_max)
    chi_l = new_R.shape[0]
    padded = jnp.zeros((chi_max, D_w_out, chi_max), dtype=new_R.dtype)
    padded = padded.at[:chi_l, :D_w_out, :chi_l].set(new_R)
    return padded
