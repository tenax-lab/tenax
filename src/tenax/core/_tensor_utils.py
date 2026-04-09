"""Core tensor utilities.

Functions here work polymorphically on both DenseTensor and SymmetricTensor
via the Tensor protocol and depend only on tenax.core.
"""

from __future__ import annotations

import jax
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
    for key, block in T.blocks.items():
        charge_val = key[axis]
        positions = np.where(idx.charges == charge_val)[0]
        block_size = block.shape[axis]
        scale_slice = scale[positions[:block_size]]
        shape = [1] * T.ndim
        shape[axis] = block_size
        new_blocks[key] = block * scale_slice.reshape(shape)

    obj = object.__new__(SymmetricTensor)
    obj._indices = T._indices
    obj._init_flat_buffer(new_blocks)
    return obj
