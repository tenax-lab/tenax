"""PaddedBlockArray: accelerator-native padded representation of block-sparse tensors.

Stores all charge-sector blocks of a SymmetricTensor as a single
(num_blocks, M_max, N_max) JAX array with zero-padding and a boolean mask.
This enables jax.vmap and jax.lax.scan over blocks without dynamic shapes,
which is essential for TPU/GPU accelerated DMRG.

Each block is treated as a 2D matrix. For tensors with more than 2 legs,
the block shape is the product of all dimensions grouped into "row" and
"column" sets following the SymmetricTensor's internal flat-buffer layout:
for a 3-leg MPS tensor (phys, left, right) with block shape (d, chi_l, chi_r),
the 2D shape is (d * chi_l, chi_r).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.tensor import BlockKey, SymmetricTensor, _block_slices


class PaddedBlockArray:
    """Padded block array for accelerator-native block-sparse operations.

    All blocks are zero-padded to (M_max, N_max) and stacked into a single
    3D JAX array of shape (num_blocks, M_max, N_max). A boolean mask array
    of the same shape marks which entries are real data (True) vs padding (False).

    Attributes:
        data:           JAX array of shape (num_blocks, M_max, N_max).
        mask:           Bool array of same shape; True = real data (computed on demand).
        block_charges:  Tuple of BlockKey tuples identifying each block's charges.
        block_shapes:   Tuple of (rows, cols) for each block's unpadded 2D shape.
        indices:        Tuple of TensorIndex objects from the source tensor.
        symmetry:       The symmetry object from the source tensor's indices.
    """

    def __init__(
        self,
        *,
        data: jax.Array,
        block_charges: tuple[BlockKey, ...],
        block_shapes: tuple[tuple[int, int], ...],
        indices: tuple,
        symmetry: object,
    ) -> None:
        self.data = data
        self.block_charges = block_charges
        self.block_shapes = block_shapes
        self.indices = indices
        self.symmetry = symmetry

    @property
    def mask(self):
        """Boolean mask: True for real data, False for padding."""
        num_blocks = self.data.shape[0]
        M_max, N_max = self.data.shape[1], self.data.shape[2]
        mask = np.zeros((num_blocks, M_max, N_max), dtype=bool)
        for i, (m, n) in enumerate(self.block_shapes):
            mask[i, :m, :n] = True
        return jnp.array(mask)

    # --- JAX pytree registration ---

    def tree_flatten(self):
        children = (self.data,)
        aux = (
            self.block_charges,
            self.block_shapes,
            self.indices,
            self.symmetry,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        block_charges, block_shapes, indices, symmetry = aux
        (data,) = children
        return cls(
            data=data,
            block_charges=block_charges,
            block_shapes=block_shapes,
            indices=indices,
            symmetry=symmetry,
        )

    # --- Conversion from/to SymmetricTensor ---

    @classmethod
    def from_symmetric(cls, tensor: SymmetricTensor) -> PaddedBlockArray:
        """Convert a SymmetricTensor to padded block form.

        Each block is reshaped to 2D by treating all legs except the last
        as rows and the last leg as columns. Blocks are then zero-padded
        to (M_max, N_max) and stacked.

        Args:
            tensor: The source SymmetricTensor.

        Returns:
            PaddedBlockArray with all blocks padded and stacked.
        """
        n_blocks = tensor.n_blocks
        block_keys = tensor._block_keys
        block_shapes_nd = tensor._block_shapes

        if n_blocks == 0:
            sym = tensor.indices[0].symmetry if tensor.indices else None
            return cls(
                data=jnp.zeros((0, 0, 0), dtype=tensor.dtype),
                block_charges=(),
                block_shapes=(),
                indices=tensor.indices,
                symmetry=sym,
            )

        # Convert each block's ND shape to 2D: (product of all but last, last)
        shapes_2d: list[tuple[int, int]] = []
        for shape in block_shapes_nd:
            if len(shape) == 1:
                shapes_2d.append((shape[0], 1))
            elif len(shape) == 2:
                shapes_2d.append((shape[0], shape[1]))
            else:
                # Fuse all legs except the last into rows
                row_dim = 1
                for d in shape[:-1]:
                    row_dim *= d
                shapes_2d.append((row_dim, shape[-1]))

        M_max = max(s[0] for s in shapes_2d)
        N_max = max(s[1] for s in shapes_2d)

        # Build padded data array using direct block access (avoids dict construction)
        padded_blocks = []

        for i in range(n_blocks):
            block = tensor._get_block(i)
            rows, cols = shapes_2d[i]
            block_2d = block.reshape(rows, cols)

            # Zero-pad to (M_max, N_max)
            padded = jnp.zeros((M_max, N_max), dtype=block.dtype)
            padded = padded.at[:rows, :cols].set(block_2d)
            padded_blocks.append(padded)

        data = jnp.stack(padded_blocks, axis=0)

        sym = tensor.indices[0].symmetry if tensor.indices else None

        return cls(
            data=data,
            block_charges=block_keys,
            block_shapes=tuple(shapes_2d),
            indices=tensor.indices,
            symmetry=sym,
        )

    def to_symmetric(self) -> SymmetricTensor:
        """Convert back to a SymmetricTensor by stripping padding.

        Returns:
            SymmetricTensor with the original block structure and data.
        """
        blocks: dict[BlockKey, jax.Array] = {}

        # Recover the original ND shapes from the indices
        # We need the original block shapes to reshape 2D back to ND
        original_nd_shapes = _compute_nd_shapes(self.indices, self.block_charges)

        for i, key in enumerate(self.block_charges):
            rows, cols = self.block_shapes[i]
            block_2d = self.data[i, :rows, :cols]
            nd_shape = original_nd_shapes[i]
            blocks[key] = block_2d.reshape(nd_shape)

        return SymmetricTensor._from_blocks_unchecked(blocks, self.indices)


def _compute_nd_shapes(
    indices: tuple, block_charges: tuple[BlockKey, ...]
) -> list[tuple[int, ...]]:
    """Compute the original ND block shapes from indices and block charges.

    For each block key, finds how many basis states have the matching charge
    on each leg, giving the block dimensions.
    """
    shapes = []
    for key in block_charges:
        _, shape = _block_slices(indices, key)
        shapes.append(shape)
    return shapes


def pad_dense(data: jax.Array, chi_max: int) -> jax.Array:
    """Pad a dense MPS tensor (chi_l, d, chi_r) to (chi_max, d, chi_max).

    Zero-pads the first and last dimensions to chi_max. The physical
    dimension (middle axis) is left unchanged.

    Args:
        data:    Dense JAX array of shape (chi_l, d, chi_r).
        chi_max: Target bond dimension for padding.

    Returns:
        Padded JAX array of shape (chi_max, d, chi_max).
    """
    chi_l, d, chi_r = data.shape
    if chi_l == chi_max and chi_r == chi_max:
        return data
    result = jnp.zeros((chi_max, d, chi_max), dtype=data.dtype)
    result = result.at[:chi_l, :, :chi_r].set(data)
    return result


# Register as JAX pytree
jax.tree_util.register_pytree_node(
    PaddedBlockArray,
    PaddedBlockArray.tree_flatten,
    PaddedBlockArray.tree_unflatten,
)
