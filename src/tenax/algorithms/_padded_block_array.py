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

import string as _string
from collections import defaultdict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.tensor import (
    BlockKey,
    SymmetricTensor,
    _block_slices,
    _compute_valid_blocks,
)


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


# ===== PaddedContractionPlan and contract_padded =====


@dataclass(frozen=True)
class PaddedContractionPlan:
    """Static contraction plan for block-sparse matmul on PaddedBlockArrays.

    Precomputes which block pairs from two tensors contribute to each output
    block based on charge conservation on the contracted index. All fields
    are static (not traced by JAX), so the plan can be captured as a closure
    inside jax.jit without recompilation as long as the charge structure
    doesn't change.

    Attributes:
        left_indices:     Which blocks of tensor A participate (one per pair).
        right_indices:    Which blocks of tensor B participate (one per pair).
        output_indices:   Which output block each pair contributes to (scatter-add).
        num_output_blocks: Total number of output blocks.
        output_M_max:     Max row dimension of output blocks.
        output_N_max:     Max column dimension of output blocks.
        output_charges:   BlockKey for each output block (sorted).
        output_shapes:    (rows, cols) for each output block's unpadded 2D shape.
        output_tensor_indices: TensorIndex metadata for the output tensor legs.
        subscripts:       Einsum subscript for one block pair (e.g. "ij,jk->ik").
    """

    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    num_output_blocks: int
    output_M_max: int
    output_N_max: int
    output_charges: tuple[BlockKey, ...]
    output_shapes: tuple[tuple[int, int], ...]
    output_tensor_indices: tuple  # tuple of TensorIndex for the output legs
    subscripts: str

    @classmethod
    def build(cls, A: SymmetricTensor, B: SymmetricTensor) -> PaddedContractionPlan:
        """Build a contraction plan for A @ B (2-tensor matrix multiplication).

        Identifies which block pairs (one from A, one from B) share a
        contracted-leg charge value, and maps each pair to its output block.

        For A with legs (i, j) and B with legs (j, k), the contracted leg is j.
        A block A[q_i, q_j] and B[q_j, q_k] with matching q_j produce output
        at C[q_i, q_k].

        Args:
            A: Left SymmetricTensor (2-leg).
            B: Right SymmetricTensor (2-leg).

        Returns:
            PaddedContractionPlan with all static metadata for contract_padded.
        """
        # Find the contracted leg: the shared label between A and B.
        a_labels = {idx.label: i for i, idx in enumerate(A.indices)}
        b_labels = {idx.label: i for i, idx in enumerate(B.indices)}
        shared_labels = set(a_labels.keys()) & set(b_labels.keys())
        if not shared_labels:
            raise ValueError("No shared labels between A and B for contraction.")

        contracted_label = shared_labels.pop()
        a_contracted_pos = a_labels[contracted_label]
        b_contracted_pos = b_labels[contracted_label]

        # Free legs: all legs not contracted
        a_free_positions = [i for i in range(len(A.indices)) if i != a_contracted_pos]
        b_free_positions = [i for i in range(len(B.indices)) if i != b_contracted_pos]

        # Build subscript string
        # Assign letters: a's legs get consecutive letters, b's contracted leg
        # reuses the same letter, b's free legs get new letters.
        chars = _string.ascii_lowercase
        char_idx = 0
        a_chars = []
        contracted_char = None
        for i in range(len(A.indices)):
            a_chars.append(chars[char_idx])
            if i == a_contracted_pos:
                contracted_char = chars[char_idx]
            char_idx += 1

        b_chars = []
        for i in range(len(B.indices)):
            if i == b_contracted_pos:
                b_chars.append(contracted_char)
            else:
                b_chars.append(chars[char_idx])
                char_idx += 1

        out_chars = [a_chars[p] for p in a_free_positions] + [
            b_chars[p] for p in b_free_positions
        ]
        subscripts = (
            "".join(a_chars) + "," + "".join(b_chars) + "->" + "".join(out_chars)
        )

        # Output tensor indices: free legs of A followed by free legs of B
        out_tensor_indices = tuple(A.indices[p] for p in a_free_positions) + tuple(
            B.indices[p] for p in b_free_positions
        )

        # Index A's blocks by the charge on the contracted leg
        a_by_contracted: dict[int, list[int]] = defaultdict(list)
        for idx_a, key_a in enumerate(A._block_keys):
            q_contracted = key_a[a_contracted_pos]
            a_by_contracted[q_contracted].append(idx_a)

        # Index B's blocks by the charge on the contracted leg
        b_by_contracted: dict[int, list[int]] = defaultdict(list)
        for idx_b, key_b in enumerate(B._block_keys):
            q_contracted = key_b[b_contracted_pos]
            b_by_contracted[q_contracted].append(idx_b)

        # Compute valid output blocks and their 2D shapes
        valid_out_keys = sorted(set(_compute_valid_blocks(out_tensor_indices)))
        out_key_to_idx = {key: i for i, key in enumerate(valid_out_keys)}

        out_shapes_2d: list[tuple[int, int]] = []
        for key in valid_out_keys:
            _, nd_shape = _block_slices(out_tensor_indices, key)
            if len(nd_shape) == 1:
                out_shapes_2d.append((nd_shape[0], 1))
            elif len(nd_shape) == 2:
                out_shapes_2d.append((nd_shape[0], nd_shape[1]))
            else:
                row_dim = 1
                for d in nd_shape[:-1]:
                    row_dim *= d
                out_shapes_2d.append((row_dim, nd_shape[-1]))

        # Find matching block pairs (same charge on contracted leg)
        left_list: list[int] = []
        right_list: list[int] = []
        output_list: list[int] = []

        common_charges = set(a_by_contracted.keys()) & set(b_by_contracted.keys())
        for q in sorted(common_charges):
            for idx_a in a_by_contracted[q]:
                key_a = A._block_keys[idx_a]
                for idx_b in b_by_contracted[q]:
                    key_b = B._block_keys[idx_b]
                    # Build output key from free legs
                    out_key = tuple(key_a[p] for p in a_free_positions) + tuple(
                        key_b[p] for p in b_free_positions
                    )
                    if out_key in out_key_to_idx:
                        left_list.append(idx_a)
                        right_list.append(idx_b)
                        output_list.append(out_key_to_idx[out_key])

        # Output dimensions
        if out_shapes_2d:
            out_M_max = max(s[0] for s in out_shapes_2d)
            out_N_max = max(s[1] for s in out_shapes_2d)
        else:
            out_M_max = 0
            out_N_max = 0

        return cls(
            left_indices=tuple(left_list),
            right_indices=tuple(right_list),
            output_indices=tuple(output_list),
            num_output_blocks=len(valid_out_keys),
            output_M_max=out_M_max,
            output_N_max=out_N_max,
            output_charges=tuple(valid_out_keys),
            output_shapes=tuple(out_shapes_2d),
            output_tensor_indices=out_tensor_indices,
            subscripts=subscripts,
        )


def contract_padded(
    plan: PaddedContractionPlan,
    A: PaddedBlockArray,
    B: PaddedBlockArray,
) -> PaddedBlockArray:
    """Execute a padded contraction plan using vmap.

    Gathers participating blocks by index, vmaps a per-block einsum over
    all block pairs, then scatter-adds results into output blocks.

    The plan is entirely static (captured as a closure in jax.jit).
    Only A.data and B.data are traced JAX arrays.

    Args:
        plan:  PaddedContractionPlan from PaddedContractionPlan.build().
        A:     Left PaddedBlockArray.
        B:     Right PaddedBlockArray.

    Returns:
        PaddedBlockArray with the contraction result.
    """
    n_pairs = len(plan.left_indices)
    if n_pairs == 0 or plan.num_output_blocks == 0:
        # No contributing pairs: return zeros
        data = jnp.zeros(
            (plan.num_output_blocks, plan.output_M_max, plan.output_N_max),
            dtype=A.data.dtype,
        )
        return PaddedBlockArray(
            data=data,
            block_charges=plan.output_charges,
            block_shapes=plan.output_shapes,
            indices=plan.output_tensor_indices,
            symmetry=A.symmetry,
        )

    # Parse subscripts to get per-block subscripts (no batch dim needed,
    # since we're vmapping over a gathered batch of block pairs).
    # plan.subscripts is e.g. "ij,jk->ik" -- this is the per-block subscript.
    lhs, rhs = plan.subscripts.split("->")
    sub_a, sub_b = lhs.split(",")

    # Gather blocks for all pairs: shape (n_pairs, M_max_A, N_max_A) etc.
    left_idx = jnp.array(plan.left_indices, dtype=jnp.int32)
    right_idx = jnp.array(plan.right_indices, dtype=jnp.int32)

    a_gathered = A.data[left_idx]  # (n_pairs, M_a, N_a)
    b_gathered = B.data[right_idx]  # (n_pairs, M_b, N_b)

    # vmap the per-block einsum over the pair dimension
    def single_pair_einsum(a_block, b_block):
        return jnp.einsum(sub_a + "," + sub_b + "->" + rhs, a_block, b_block)

    pair_results = jax.vmap(single_pair_einsum)(a_gathered, b_gathered)
    # pair_results: (n_pairs, output_M_max_padded, output_N_max_padded)
    # The padded dimensions come from A's N_max and B's N_max via einsum.
    # But we need the output padded to (output_M_max, output_N_max).

    # The einsum "ij,jk->ik" on padded blocks of shape (M_a, N_a) and (M_b, N_b)
    # produces (M_a, N_b). We need to ensure this fits in (output_M_max, output_N_max).
    # Since A's M_max >= all block row dims and B's N_max >= all block col dims,
    # and output blocks have row=A_free, col=B_free, the einsum result shape is
    # (A.M_max, B.N_max) which may differ from (output_M_max, output_N_max).
    # We need to pad/slice to the correct output size.
    result_M = pair_results.shape[1]
    result_N = pair_results.shape[2]

    if result_M < plan.output_M_max or result_N < plan.output_N_max:
        # Pad to output size
        pad_M = max(0, plan.output_M_max - result_M)
        pad_N = max(0, plan.output_N_max - result_N)
        pair_results = jnp.pad(pair_results, ((0, 0), (0, pad_M), (0, pad_N)))
    elif result_M > plan.output_M_max or result_N > plan.output_N_max:
        # Slice to output size (padding in inputs produced larger intermediates)
        pair_results = pair_results[:, : plan.output_M_max, : plan.output_N_max]

    # Scatter-add: accumulate pair results into output blocks
    output_idx = jnp.array(plan.output_indices, dtype=jnp.int32)
    output_data = jnp.zeros(
        (plan.num_output_blocks, plan.output_M_max, plan.output_N_max),
        dtype=pair_results.dtype,
    )
    output_data = output_data.at[output_idx].add(pair_results)

    # Zero out padding regions in the output to keep it clean.
    # Build a mask for valid data in each output block.
    output_mask = np.zeros(
        (plan.num_output_blocks, plan.output_M_max, plan.output_N_max),
        dtype=bool,
    )
    for i, (m, n) in enumerate(plan.output_shapes):
        output_mask[i, :m, :n] = True
    output_data = jnp.where(jnp.array(output_mask), output_data, 0.0)

    return PaddedBlockArray(
        data=output_data,
        block_charges=plan.output_charges,
        block_shapes=plan.output_shapes,
        indices=plan.output_tensor_indices,
        symmetry=A.symmetry,
    )
