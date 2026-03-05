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

from tenax.algorithms._tensor_utils import max_abs_normalize
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core.tensor import Tensor


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
        tensor: Initial site tensor (DenseTensor or SymmetricTensor) with
                4 legs labeled ("up", "down", "left", "right").
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

    if not isinstance(tensor, Tensor):
        raise TypeError(f"hotrg() requires a Tensor, got {type(tensor).__name__}")

    T = tensor
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


def _hotrg_step_horizontal(
    T: Tensor,
    max_bond_dim: int,
    svd_trunc_err: float | None = None,
) -> tuple[Tensor, jax.Array]:
    """Single horizontal HOTRG coarse-graining step (polymorphic).

    Contracts two adjacent tensors horizontally and uses SVD to find
    the optimal truncation isometries for the paired up and down bonds.

    Args:
        T:             Site tensor with labels ("up", "down", "left", "right").
        max_bond_dim:  Maximum chi after truncation.
        svd_trunc_err: Optional maximum truncation error per HOSVD.

    Returns:
        (T_new, log_norm) where T_new has compressed up/down bonds.
    """
    # Step 1: Form environment M by contracting T with itself over (left, right).
    # T has labels (up, down, left, right). Second copy relabeled to avoid collision.
    T_copy = T.relabels({"up": "U", "down": "D", "left": "right", "right": "left"})
    # T_copy: (U, D, right, left) — shares "left" and "right" with T
    M = contract(T, T_copy)  # contracts left↔left, right↔right → (up, down, U, D)

    # Step 2: Isometries via SVD of environment
    # Group (up, U) vs (down, D) → get paired isometries
    U_iso, _, Vh_iso, _ = truncated_svd(
        M,
        left_labels=["up", "U"],
        right_labels=["down", "D"],
        new_bond_label="a",
        max_singular_values=max_bond_dim,
        max_truncation_err=svd_trunc_err,
    )
    # U_iso: (up, U, a),  Vh_iso: (a, down, D)

    # Step 3: Merge two T copies over the shared horizontal bond
    T_left = T.relabel("right", "k")  # (up, down, left, k)
    T_right = T.relabels({"up": "U", "down": "D", "left": "k"})  # (U, D, k, right)
    T_merged = contract(T_left, T_right)  # contracts k → (up, down, left, U, D, right)

    # Step 4: Apply isometries to compress (up, U) → a and (down, D) → b
    # Dagger flips flow directions so contracted legs have opposite flows,
    # which is required for SymmetricTensor charge conservation.
    # Use two-step contraction (multi-tensor symmetric contraction has
    # limitations when different tensor pairs share different bonds).
    U_iso_dag = U_iso.dagger()  # (up_out, U_out, a_in)
    Vh_iso_b = Vh_iso.relabel("a", "b").dagger()  # (b_out, down_in, D_in)
    T_tmp = contract(U_iso_dag, T_merged)  # contracts up, U → (a, down, left, D, right)
    T_new = contract(T_tmp, Vh_iso_b, output_labels=("a", "b", "left", "right"))
    T_new = T_new.relabels({"a": "up", "b": "down"})

    # Step 5: Normalize
    T_new, log_norm = max_abs_normalize(T_new)
    return T_new, log_norm


def _hotrg_step_vertical(
    T: Tensor,
    max_bond_dim: int,
    svd_trunc_err: float | None = None,
) -> tuple[Tensor, jax.Array]:
    """Single vertical HOTRG coarse-graining step (polymorphic).

    Analogous to horizontal step but contracts along the up-down direction.

    Args:
        T:             Site tensor with labels ("up", "down", "left", "right").
        max_bond_dim:  Maximum chi after truncation.
        svd_trunc_err: Optional maximum truncation error per HOSVD.

    Returns:
        (T_new, log_norm) where T_new has compressed left/right bonds.
    """
    # Step 1: Form environment M by contracting T with itself over (up, down).
    T_copy = T.relabels({"left": "L", "right": "R", "up": "down", "down": "up"})
    # T_copy: (down, up, L, R) — shares "up" and "down" with T
    M = contract(T, T_copy)  # contracts up↔up, down↔down → (left, right, L, R)

    # Step 2: Isometries via SVD of environment
    U_iso, _, Vh_iso, _ = truncated_svd(
        M,
        left_labels=["left", "L"],
        right_labels=["right", "R"],
        new_bond_label="a",
        max_singular_values=max_bond_dim,
        max_truncation_err=svd_trunc_err,
    )
    # U_iso: (left, L, a),  Vh_iso: (a, right, R)

    # Step 3: Merge two T copies over the shared vertical bond
    T_top = T.relabel("down", "k")  # (up, k, left, right)
    T_bottom = T.relabels({"left": "L", "right": "R", "up": "k"})  # (k, down, L, R)
    T_merged = contract(T_top, T_bottom)  # contracts k → (up, left, right, down, L, R)

    # Step 4: Apply isometries to compress (left, L) → a and (right, R) → b
    # Dagger flips flow directions for SymmetricTensor charge conservation.
    # Use two-step contraction (see horizontal step comment).
    U_iso_dag = U_iso.dagger()
    Vh_iso_b = Vh_iso.relabel("a", "b").dagger()
    T_tmp = contract(U_iso_dag, T_merged)  # contracts left, L → (up, right, down, R, a)
    T_new = contract(T_tmp, Vh_iso_b, output_labels=("up", "down", "a", "b"))
    T_new = T_new.relabels({"a": "left", "b": "right"})

    # Step 5: Normalize
    T_new, log_norm = max_abs_normalize(T_new)
    return T_new, log_norm
