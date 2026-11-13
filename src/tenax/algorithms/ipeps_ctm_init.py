"""CTM algorithm for iPEPS — double-layer construction and environment initialization."""

from __future__ import annotations

__all__ = [
    "_build_double_layer",
    "_initialize_split_ctm_env",
]

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import SplitCTMEnvironment


def _build_double_layer(A: jax.Array) -> jax.Array:
    """Build the double-layer tensor from a PEPS site tensor.

    For a tensor A with shape (D,...,d) where d is the physical dimension
    and D's are virtual bond dimensions, the double-layer tensor is:
    a[virtual...] = sum_s A[virtual..., s] * conj(A[virtual..., s])

    This traces out the physical index.
    """
    if A.ndim == 5:
        # A[u, d, l, r, s] — fuse ket/bra pairs per spatial direction
        return jnp.einsum("udlrs,UDLRs->uUdDlLrR", A, jnp.conj(A))
    elif A.ndim == 3:
        # A[l, r, s] — simplified 2D
        return jnp.einsum("lrs,LRs->lrLR", A, jnp.conj(A))
    else:
        # Generic: assume last index is physical
        # Squeeze to remove degenerate dims
        s_idx = "".join(chr(97 + i) for i in range(A.ndim))
        phys = s_idx[-1]
        virt1 = s_idx[:-1]
        virt2 = virt1.upper()
        return jnp.einsum(f"{s_idx},{virt2}{phys}->{virt1}{virt2}", A, jnp.conj(A))


def _initialize_split_ctm_env(
    A: jax.Array,
    chi: int,
    chi_I: int,
) -> SplitCTMEnvironment:
    """Initialize a SplitCTMEnvironment from the PEPS site tensor.

    Args:
        A:     Site tensor of shape ``(D, D, D, D, d)``.
        chi:   Environment bond dimension.
        chi_I: Interlayer bond dimension.
    """
    D = A.shape[0]
    dtype = A.dtype

    # Corners: identity-like (chi x chi)
    C = jnp.eye(min(chi, D), dtype=dtype)
    C_pad = jnp.zeros((chi, chi), dtype=dtype)
    C_pad = C_pad.at[: C.shape[0], : C.shape[1]].set(C)

    # Split edges: identity-like structure
    chi_D = min(chi, D)
    chi_I_D = min(chi_I, D)

    T_ket = jnp.zeros((chi, D, chi_I), dtype=dtype)
    for i in range(min(chi_D, chi_I_D)):
        T_ket = T_ket.at[i, :, i].set(jnp.ones(D))

    T_bra = jnp.zeros((chi_I, D, chi), dtype=dtype)
    for i in range(min(chi_I_D, chi_D)):
        T_bra = T_bra.at[i, :, i].set(jnp.ones(D))

    return SplitCTMEnvironment(
        C1=C_pad,
        C2=C_pad,
        C3=C_pad,
        C4=C_pad,
        T1_ket=T_ket,
        T1_bra=T_bra,
        T2_ket=T_ket,
        T2_bra=T_bra,
        T3_ket=T_ket,
        T3_bra=T_bra,
        T4_ket=T_ket,
        T4_bra=T_bra,
    )
