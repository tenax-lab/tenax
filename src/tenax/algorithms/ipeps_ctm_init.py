"""CTM algorithm for iPEPS — double-layer construction and environment initialization."""

from __future__ import annotations

__all__ = [
    "_build_double_layer",
    "_initialize_ctm_env",
    "_initialize_split_ctm_env",
]

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import (
    CTMEnvironment,
    SplitCTMEnvironment,
)


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


def _initialize_ctm_env(a: jax.Array, chi: int) -> CTMEnvironment:
    """Initialize CTM environment tensors from the PEPS double-layer tensor.

    Uses a simple initialization: corners and edges built from partial traces
    of the double-layer tensor.

    Args:
        a:   Double-layer tensor of shape (D2, D2, D2, D2).
        chi: Environment bond dimension.
    """
    D2 = a.shape[0]
    dtype = a.dtype

    # Initialize corners as identity matrices (chi x chi)
    C = jnp.eye(min(chi, D2), dtype=dtype)
    C_small = jnp.zeros((chi, chi), dtype=dtype)
    C_small = C_small.at[: C.shape[0], : C.shape[1]].set(
        C[: min(chi, C.shape[0]), : min(chi, C.shape[1])]
    )

    # Initialize edges as a slice of the double-layer tensor
    # T[chi, D2, chi] — use first chi values
    T_chi = min(chi, D2)
    T_init = jnp.zeros((chi, D2, chi), dtype=dtype)
    # Fill with identity-like structure
    for i in range(min(T_chi, chi)):
        T_init = T_init.at[i, :, i].add(jnp.ones(D2))

    return CTMEnvironment(
        C1=C_small,
        C2=C_small,
        C3=C_small,
        C4=C_small,
        T1=T_init,
        T2=T_init,
        T3=T_init,
        T4=T_init,
    )


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
