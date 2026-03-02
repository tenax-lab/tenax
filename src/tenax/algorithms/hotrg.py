"""Higher-Order Tensor Renormalization Group (HOTRG) algorithm.

HOTRG improves upon TRG by using Higher-Order Singular Value Decomposition
(HOSVD) to compute truncation isometries. Instead of pairwise SVD splits,
HOTRG constructs an optimal projector by computing the truncated SVD of the
"environment tensor" M obtained by contracting two tensors over shared bonds.

Reference: Xie et al., PRB 86, 045139 (2012).

Algorithm (horizontal coarse-graining step):
  1. Form M[u,U,d,D] = sum_{l,r} T[u,d,l,r] * T[U,D,r,l]
     (contract two adjacent tensors over their shared left-right bonds)
  2. Reshape M to (d_u*d_U, d_d*d_D) and SVD to get paired isometries
     U_u of shape (d_u^2, chi) and U_d of shape (d_d^2, chi).
  3. Contract the two T tensors over the shared bond, apply paired
     isometries to compress the doubled up/down indices:
     T_new[a,b,l,r] = U_u[(u,U),a] * T_merged[(u,U),(d,D),l,r] * U_d[(d,D),b]

The vertical step is analogous with l/r bonds.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core import LOG_EPS
from tenax.core.tensor import DenseTensor, Tensor


@dataclass
class HOTRGConfig:
    """Configuration for HOTRG coarse-graining.

    Attributes:
        max_bond_dim:    Maximum bond dimension chi after each coarse-graining step.
        num_steps:       Number of coarse-graining iterations.
        direction_order: Order of coarse-graining directions.
                         "alternating": alternate horizontal/vertical (default).
                         "horizontal": horizontal only.
                         "vertical": vertical only.
        svd_trunc_err:   Optional maximum truncation error per HOSVD.
    """

    max_bond_dim: int = 16
    num_steps: int = 10
    direction_order: str = "alternating"
    svd_trunc_err: float | None = None


def hotrg(
    tensor: Tensor,
    config: HOTRGConfig,
) -> jax.Array:
    """HOTRG coarse-graining for a 2D square lattice partition function.

    Uses Higher-Order SVD (HOSVD) for computing truncation isometries,
    providing better accuracy than TRG at the same bond dimension.

    Args:
        tensor: Initial site tensor with 4 legs (up, down, left, right).
                Can be a DenseTensor or raw tensor-like with shape (d,d,d,d).
        config: HOTRGConfig parameters.

    Returns:
        Scalar JAX array: estimated log(Z)/N (free energy per site).
    """
    valid_directions = ("alternating", "horizontal", "vertical")
    if config.direction_order not in valid_directions:
        raise ValueError(
            f"Invalid direction_order {config.direction_order!r}. "
            f"Must be one of {valid_directions}."
        )

    if isinstance(tensor, DenseTensor):
        T = tensor.todense()
    else:
        T = tensor.todense()

    log_norm_total = jnp.zeros((), dtype=T.dtype)

    for step in range(config.num_steps):
        if config.direction_order == "alternating":
            if step % 2 == 0:
                T, log_norm = _hotrg_step_horizontal(
                    T, config.max_bond_dim, config.svd_trunc_err
                )
            else:
                T, log_norm = _hotrg_step_vertical(
                    T, config.max_bond_dim, config.svd_trunc_err
                )
        elif config.direction_order == "horizontal":
            T, log_norm = _hotrg_step_horizontal(
                T, config.max_bond_dim, config.svd_trunc_err
            )
        else:
            T, log_norm = _hotrg_step_vertical(
                T, config.max_bond_dim, config.svd_trunc_err
            )

        # Each HOTRG step halves the number of tensors.
        log_norm_total = log_norm_total + log_norm / (2.0 ** (step + 1))

    return log_norm_total


def _apply_svd_trunc_err(s: jax.Array, chi: int, svd_trunc_err: float) -> int:
    """Reduce chi so that the relative truncation error stays within bound.

    Args:
        s:             Singular values (sorted descending).
        chi:           Current maximum number of values to keep.
        svd_trunc_err: Maximum allowed relative truncation error.

    Returns:
        Possibly reduced chi.
    """
    s_np = np.array(s[:chi])
    total_sq = float(np.sum(np.array(s) ** 2))
    if total_sq == 0:
        return chi
    trunc_sq = 0.0
    for i in range(len(s_np) - 1, 0, -1):
        trunc_sq += float(s_np[i] ** 2)
        if trunc_sq / total_sq > svd_trunc_err**2:
            return i + 1
    return chi


def _hotrg_step_horizontal(
    T: jax.Array,
    max_bond_dim: int,
    svd_trunc_err: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single horizontal HOTRG coarse-graining step.

    Contracts two adjacent tensors horizontally and uses HOSVD to find
    the optimal truncation isometries for the paired up and down bonds.

    Args:
        T:             Site tensor of shape (d_u, d_d, d_l, d_r).
        max_bond_dim:  Maximum chi after truncation.
        svd_trunc_err: Optional maximum truncation error per HOSVD.

    Returns:
        (T_new, log_norm) where T_new has compressed up/down bonds.
    """
    d_u, d_d, d_l, d_r = T.shape

    # Step 1: Form environment tensor M by contracting over horizontal bonds
    # M[u,U,d,D] = sum_{l,r} T[u,d,l,r] * T[U,D,r,l]
    M = jnp.einsum("udlr,UDrl->uUdD", T, T)
    # Shape: (d_u, d_u, d_d, d_d)

    # Step 2: Compute paired isometries via SVD of M reshaped to (d_u^2, d_d^2)
    chi = min(max_bond_dim, d_u * d_u, d_d * d_d)
    M_mat = M.reshape(d_u * d_u, d_d * d_d)
    U_full, s_full, Vh_full = jnp.linalg.svd(M_mat, full_matrices=False)

    if svd_trunc_err is not None:
        chi = _apply_svd_trunc_err(s_full, chi, svd_trunc_err)

    U_u = U_full[:, :chi]  # (d_u^2, chi)
    U_d = Vh_full[:chi, :].T  # (d_d^2, chi)

    # Step 3: Contract two T tensors over shared horizontal bond,
    # then apply paired isometries
    # T_merged[u,U,d,D,l,r] = T[u,d,l,k] * T[U,D,k,r]
    T_merged = jnp.einsum("udlk,UDkr->uUdDlr", T, T)
    # Reshape paired indices: (u,U) -> d_u^2, (d,D) -> d_d^2
    T_merged = T_merged.reshape(d_u * d_u, d_d * d_d, d_l, d_r)
    # Apply isometries: T_new[a,b,l,r] = U_u[(uU),a] * T_merged[(uU),(dD),l,r] * U_d[(dD),b]
    T_new = jnp.einsum("ua,udlr,db->ablr", U_u, T_merged, U_d)

    # Normalize
    norm = jnp.max(jnp.abs(T_new))
    log_norm = jnp.log(norm + LOG_EPS)
    T_new = T_new / (norm + LOG_EPS)

    return T_new, log_norm


def _hotrg_step_vertical(
    T: jax.Array,
    max_bond_dim: int,
    svd_trunc_err: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single vertical HOTRG coarse-graining step.

    Analogous to horizontal step but contracts along the up-down direction.

    Args:
        T:             Site tensor of shape (d_u, d_d, d_l, d_r).
        max_bond_dim:  Maximum chi after truncation.
        svd_trunc_err: Optional maximum truncation error per HOSVD.

    Returns:
        (T_new, log_norm) where T_new has compressed left/right bonds.
    """
    d_u, d_d, d_l, d_r = T.shape

    # Form environment: M[l,L,r,R] = sum_{u,d} T[u,d,l,r] * T[d,u,L,R]
    M = jnp.einsum("udlr,duLR->lLrR", T, T)
    # Shape: (d_l, d_l, d_r, d_r)

    # Paired isometries via SVD of M reshaped to (d_l^2, d_r^2)
    chi = min(max_bond_dim, d_l * d_l, d_r * d_r)
    M_mat = M.reshape(d_l * d_l, d_r * d_r)
    U_full, s_full, Vh_full = jnp.linalg.svd(M_mat, full_matrices=False)

    if svd_trunc_err is not None:
        chi = _apply_svd_trunc_err(s_full, chi, svd_trunc_err)

    U_l = U_full[:, :chi]  # (d_l^2, chi)
    U_r = Vh_full[:chi, :].T  # (d_r^2, chi)

    # Contract two T tensors over shared vertical bond,
    # then apply paired isometries
    # T_merged[u,d,l,L,r,R] = T[u,k,l,r] * T[k,d,L,R]
    T_merged = jnp.einsum("uklr,kdLR->udlLrR", T, T)
    # Reshape paired indices: (l,L) -> d_l^2, (r,R) -> d_r^2
    T_merged = T_merged.reshape(d_u, d_d, d_l * d_l, d_r * d_r)
    # Apply isometries
    T_new = jnp.einsum("la,udlr,rb->udab", U_l, T_merged, U_r)

    # Normalize
    norm = jnp.max(jnp.abs(T_new))
    log_norm = jnp.log(norm + LOG_EPS)
    T_new = T_new / (norm + LOG_EPS)

    return T_new, log_norm
