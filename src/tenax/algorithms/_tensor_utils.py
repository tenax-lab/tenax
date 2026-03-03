"""Shared tensor utilities for algorithms.

Functions here work polymorphically on both DenseTensor and SymmetricTensor
via the Tensor protocol.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.index import Label
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor


def scale_bond_axis(T: Tensor, label: Label, scale: jax.Array) -> Tensor:
    """Scale a tensor along a labeled axis by a diagonal vector.

    For DenseTensor: broadcasts scale along the named axis.
    For SymmetricTensor: delegates to block-wise scaling.

    Args:
        T:     Input tensor.
        label: Label of the axis to scale along.
        scale: 1D JAX array of length matching the axis dimension.

    Returns:
        New tensor with the specified axis scaled.
    """
    labels = T.labels()
    axis = labels.index(label)

    if isinstance(T, SymmetricTensor):
        return _scale_bond_axis_symmetric(T, axis, scale)

    # DenseTensor path
    data = T.todense()
    shape = [1] * T.ndim
    shape[axis] = data.shape[axis]
    return DenseTensor(data * scale.reshape(shape), T.indices)


def _scale_bond_axis_symmetric(
    T: SymmetricTensor, axis: int, scale: jax.Array
) -> SymmetricTensor:
    """Block-wise scaling for SymmetricTensor (same logic as fermionic_ipeps)."""
    new_blocks = {}
    idx = T.indices[axis]
    for key, block in T._blocks.items():
        charge_val = key[axis]
        positions = np.where(idx.charges == charge_val)[0]
        block_size = block.shape[axis]
        scale_slice = scale[positions[:block_size]]
        shape = [1] * T.ndim
        shape[axis] = block_size
        new_blocks[key] = block * scale_slice.reshape(shape)

    obj = object.__new__(SymmetricTensor)
    obj._indices = T._indices
    obj._blocks = new_blocks
    return obj


def max_abs_normalize(T: Tensor) -> tuple[Tensor, jax.Array]:
    """Normalize tensor by its max absolute value.

    Args:
        T: Input tensor.

    Returns:
        (T_normalized, log_norm) where T_normalized = T / max_abs(T)
        and log_norm = log(max_abs(T)).
    """
    from tenax.core import LOG_EPS

    norm = T.max_abs()
    log_norm = jnp.log(norm + LOG_EPS)
    T_norm = T * (1.0 / (norm + LOG_EPS))
    return T_norm, log_norm


def absorb_sqrt_singular_values(
    U: Tensor,
    s: jax.Array,
    Vh: Tensor,
    bond_label: Label,
) -> tuple[Tensor, Tensor]:
    """Absorb sqrt(s) into both U and Vh along their shared bond.

    Args:
        U:          Left factor from SVD with bond_label as a leg.
        s:          1D singular values.
        Vh:         Right factor from SVD with bond_label as a leg.
        bond_label: Label of the SVD bond on both U and Vh.

    Returns:
        (F_left, F_right) with sqrt(s) absorbed into each.
    """
    sqrt_s = jnp.sqrt(s)
    F_left = scale_bond_axis(U, bond_label, sqrt_s)
    F_right = scale_bond_axis(Vh, bond_label, sqrt_s)
    return F_left, F_right
