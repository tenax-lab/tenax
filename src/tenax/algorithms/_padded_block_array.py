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

from tenax.core.index import TensorIndex
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


# ===== PBA vector operations for Lanczos =====


def pba_inner(a: PaddedBlockArray, b: PaddedBlockArray) -> jax.Array:
    """Hermitian inner product: sum(conj(a) * b) over non-padding elements.

    Uses the mask to zero out padding before summation so the result is
    independent of pad size.  Returns a scalar JAX array (stays on device
    for JIT compatibility).
    """
    return jnp.sum(jnp.conj(a.data) * b.data * a.mask)


def pba_scale(a: PaddedBlockArray, scalar) -> PaddedBlockArray:
    """Element-wise scaling: new PBA with data = a.data * scalar.

    Padding stays zero because scalar * 0 = 0.
    """
    return PaddedBlockArray(
        data=a.data * scalar,
        block_charges=a.block_charges,
        block_shapes=a.block_shapes,
        indices=a.indices,
        symmetry=a.symmetry,
    )


def pba_add(a: PaddedBlockArray, b: PaddedBlockArray) -> PaddedBlockArray:
    """Element-wise addition of two PBAs with the same structure.

    Both must share the same block_charges and block_shapes.
    Padding: 0 + 0 = 0, so the result is correctly padded.
    """
    return PaddedBlockArray(
        data=a.data + b.data,
        block_charges=a.block_charges,
        block_shapes=a.block_shapes,
        indices=a.indices,
        symmetry=a.symmetry,
    )


def pba_axpy(alpha, x: PaddedBlockArray, y: PaddedBlockArray) -> PaddedBlockArray:
    """Compute y + alpha * x (AXPY), commonly used in Lanczos orthogonalization.

    Both PBAs must have the same structure.
    """
    return PaddedBlockArray(
        data=y.data + alpha * x.data,
        block_charges=y.block_charges,
        block_shapes=y.block_shapes,
        indices=y.indices,
        symmetry=y.symmetry,
    )


def pba_norm(a: PaddedBlockArray) -> jax.Array:
    """Frobenius norm: sqrt(real(inner(a, a))).

    Returns a scalar JAX array.
    """
    return jnp.sqrt(pba_inner(a, a).real)


def pba_zeros_like(a: PaddedBlockArray) -> PaddedBlockArray:
    """Zero PBA with the same structure (block_charges, block_shapes, etc.)."""
    return PaddedBlockArray(
        data=jnp.zeros_like(a.data),
        block_charges=a.block_charges,
        block_shapes=a.block_shapes,
        indices=a.indices,
        symmetry=a.symmetry,
    )


def pba_expand_blocks(
    a: PaddedBlockArray,
    target_charges: tuple,
    target_shapes: tuple[tuple[int, int], ...],
    target_indices: tuple,
) -> PaddedBlockArray:
    """Expand a PBA to a (possibly larger) target block structure.

    Copies existing blocks to their matching positions in the target layout
    and fills new blocks with zeros.  Both the source and target must share
    the same symmetry.  The target block set must be a superset of the
    source blocks (every source charge key must appear in the target).

    This is useful when a ``MultiContractionPlan`` discovers more valid
    output blocks than the input tensor has.  Expanding the input to
    match the plan's output structure ensures the matvec return shape is
    consistent with the Lanczos basis vectors.

    Args:
        a:               Source PaddedBlockArray.
        target_charges:  Block charge keys for the target layout.
        target_shapes:   Unpadded 2D shapes for the target layout.
        target_indices:  TensorIndex metadata for the target layout.

    Returns:
        PaddedBlockArray with ``target_charges`` / ``target_shapes``.
    """
    n_target = len(target_charges)
    if n_target == 0:
        return PaddedBlockArray(
            data=jnp.zeros((0, 0, 0), dtype=a.data.dtype),
            block_charges=target_charges,
            block_shapes=target_shapes,
            indices=target_indices,
            symmetry=a.symmetry,
        )

    M_max = max(s[0] for s in target_shapes)
    N_max = max(s[1] for s in target_shapes)

    # Build lookup from source charge key -> source block index
    src_map = {key: i for i, key in enumerate(a.block_charges)}

    # Assemble target data: copy existing blocks, zero-fill new ones
    blocks = []
    for key in target_charges:
        if key in src_map:
            src_idx = src_map[key]
            src_block = a.data[src_idx]
            # Pad/slice to target (M_max, N_max) if source has different pad dims
            src_M, src_N = src_block.shape
            if src_M == M_max and src_N == N_max:
                blocks.append(src_block)
            else:
                padded = jnp.zeros((M_max, N_max), dtype=a.data.dtype)
                min_M = min(src_M, M_max)
                min_N = min(src_N, N_max)
                padded = padded.at[:min_M, :min_N].set(src_block[:min_M, :min_N])
                blocks.append(padded)
        else:
            blocks.append(jnp.zeros((M_max, N_max), dtype=a.data.dtype))

    data = jnp.stack(blocks, axis=0)
    return PaddedBlockArray(
        data=data,
        block_charges=target_charges,
        block_shapes=target_shapes,
        indices=target_indices,
        symmetry=a.symmetry,
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
    output_tensor_indices: tuple[TensorIndex, ...]
    subscripts: str
    output_mask_np: np.ndarray  # precomputed bool mask (num_out, M_max, N_max)

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
        if len(shared_labels) != 1:
            raise ValueError(
                f"PaddedContractionPlan.build() currently supports exactly 1 contracted "
                f"label, got {len(shared_labels)}: {shared_labels}"
            )

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

        # Precompute output mask (static numpy array, not traced by JAX)
        n_out = len(valid_out_keys)
        if n_out > 0:
            output_mask = np.zeros((n_out, out_M_max, out_N_max), dtype=bool)
            for i, (m, n) in enumerate(out_shapes_2d):
                output_mask[i, :m, :n] = True
        else:
            output_mask = np.zeros((0, 0, 0), dtype=bool)

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
            output_mask_np=output_mask,
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

    # Zero out padding regions using the precomputed mask from the plan.
    output_data = jnp.where(jnp.array(plan.output_mask_np), output_data, 0.0)

    return PaddedBlockArray(
        data=output_data,
        block_charges=plan.output_charges,
        block_shapes=plan.output_shapes,
        indices=plan.output_tensor_indices,
        symmetry=A.symmetry,
    )


# ===== Multi-tensor contraction (N-tensor einsum on PBAs) =====


def _per_leg_max_dims(indices: tuple[TensorIndex, ...]) -> tuple[int, ...]:
    """Compute the maximum charge-sector dimension for each leg.

    For each leg, groups basis states by charge and returns the size of the
    largest charge sector. This gives the per-leg padded dimension for ND
    block reshaping.

    Args:
        indices: Tuple of TensorIndex objects, one per tensor leg.

    Returns:
        Tuple of ints, one per leg, giving the max sector size.
    """
    dims = []
    for idx in indices:
        charges = np.asarray(idx.charges)
        unique_q = np.unique(charges)
        max_d = max(int(np.sum(charges == q)) for q in unique_q)
        dims.append(max_d)
    return tuple(dims)


@dataclass(frozen=True)
class MultiContractionPlan:
    """Plan for N-tensor contraction on PaddedBlockArrays.

    Precomputes which block combinations from N input tensors contribute to
    each output block, based on charge conservation on shared subscript chars.
    All fields are static (not traced by JAX).

    Attributes:
        n_inputs:             Number of input tensors.
        subscripts:           Full einsum subscript (e.g. "abc,apd,bpxe,cxf->def").
        combo_indices:        Per-input tuple of block indices for each combo.
                              combo_indices[i] is a tuple of length n_combos giving
                              which block of input i participates in each combo.
        output_indices:       Which output block each combo maps to.
        n_combos:             Total number of valid block combinations.
        num_output_blocks:    Number of output blocks.
        input_nd_shapes:      Per-input padded ND shape (max dims per leg).
        output_nd_shape:      Padded ND shape of output blocks.
        output_charges:       BlockKey for each output block (sorted).
        output_shapes:        Unpadded 2D (rows, cols) for each output block.
        output_tensor_indices: TensorIndex metadata for the output legs.
        output_mask_np:       Precomputed bool mask for output (num_out, M_max, N_max).
    """

    n_inputs: int
    subscripts: str
    combo_indices: tuple  # tuple of n_inputs tuples of ints
    output_indices: tuple[int, ...]
    n_combos: int
    num_output_blocks: int
    input_nd_shapes: tuple  # tuple of n_inputs tuples of ints
    output_nd_shape: tuple[int, ...]
    output_charges: tuple[BlockKey, ...]
    output_shapes: tuple[tuple[int, int], ...]
    output_tensor_indices: tuple[TensorIndex, ...]
    output_mask_np: np.ndarray

    @classmethod
    def build(
        cls,
        tensors: list[SymmetricTensor],
        subscripts: str,
        output_indices: tuple[TensorIndex, ...],
    ) -> MultiContractionPlan:
        """Build a multi-tensor contraction plan.

        Args:
            tensors:        List of N SymmetricTensor inputs.
            subscripts:     Full einsum subscript (e.g. "abc,apd,bpxe,cxf->def").
            output_indices: TensorIndex metadata for the output legs.

        Returns:
            MultiContractionPlan with all static metadata.
        """
        input_part, output_part = subscripts.split("->")
        input_subs = input_part.split(",")
        n_inputs = len(tensors)

        # 1. Compute per-input ND shapes (max dim per leg)
        input_nd_shapes = []
        for t in tensors:
            input_nd_shapes.append(_per_leg_max_dims(t.indices))

        # Output ND shape
        output_nd_shape = _per_leg_max_dims(output_indices)

        # 2. Build block-key-to-index lookup for each input
        input_key_to_idx: list[dict[BlockKey, int]] = []
        for t in tensors:
            key_map = {key: i for i, key in enumerate(t._block_keys)}
            input_key_to_idx.append(key_map)

        # 3. Enumerate all charge-compatible combos using backtracking
        #    (same algorithm as _precompute_block_plan in dmrg.py)
        #
        # NOTE: We do NOT pre-filter output blocks via _compute_valid_blocks
        # because environment tensors (from DMRG left/right env updates) may
        # have output charge combinations that violate the standard conservation
        # law.  Instead, we discover valid output keys from the backtracking
        # itself, matching the behaviour of _blockwise_contract in dmrg.py.
        raw_combos: list[tuple[list[int], tuple[int, ...]]] = []
        combo_keys: list[int] = []  # block indices per input
        char_charges: dict[str, int] = {}

        def _recurse(tensor_idx: int) -> None:
            if tensor_idx == n_inputs:
                # All inputs matched: compute output key
                output_key = tuple(char_charges.get(c, 0) for c in output_part)
                raw_combos.append(
                    ([combo_keys[i] for i in range(n_inputs)], output_key)
                )
                return

            subs = input_subs[tensor_idx]
            t = tensors[tensor_idx]
            for blk_idx, key in enumerate(t._block_keys):
                added_chars: list[str] = []
                compatible = True
                for char, q in zip(subs, key):
                    qi = int(q)
                    if char in char_charges:
                        if char_charges[char] != qi:
                            compatible = False
                            break
                    else:
                        char_charges[char] = qi
                        added_chars.append(char)

                if compatible:
                    combo_keys.append(blk_idx)
                    _recurse(tensor_idx + 1)
                    combo_keys.pop()

                for char in added_chars:
                    del char_charges[char]

        _recurse(0)

        # Collect unique output keys discovered by backtracking (sorted for
        # determinism), then assign indices and compute 2D shapes.
        seen_keys: set[tuple[int, ...]] = set()
        for _, out_key in raw_combos:
            seen_keys.add(out_key)
        valid_out_keys = sorted(seen_keys)
        out_key_to_idx = {key: i for i, key in enumerate(valid_out_keys)}

        # Compute output 2D shapes
        out_shapes_2d: list[tuple[int, int]] = []
        for key in valid_out_keys:
            _, nd_shape = _block_slices(output_indices, key)
            if len(nd_shape) == 1:
                out_shapes_2d.append((nd_shape[0], 1))
            elif len(nd_shape) == 2:
                out_shapes_2d.append((nd_shape[0], nd_shape[1]))
            else:
                row_dim = 1
                for d in nd_shape[:-1]:
                    row_dim *= d
                out_shapes_2d.append((row_dim, nd_shape[-1]))

        # Now build combo_lists and output_list using the discovered mapping
        combo_lists: list[list[int]] = [[] for _ in range(n_inputs)]
        output_list: list[int] = []
        for combo_k, out_key in raw_combos:
            out_idx = out_key_to_idx[out_key]
            for inp in range(n_inputs):
                combo_lists[inp].append(combo_k[inp])
            output_list.append(out_idx)

        n_combos = len(output_list)

        # Output dimensions (2D max)
        if out_shapes_2d:
            out_M_max = max(s[0] for s in out_shapes_2d)
            out_N_max = max(s[1] for s in out_shapes_2d)
        else:
            out_M_max = 0
            out_N_max = 0

        # Precompute output mask
        n_out = len(valid_out_keys)
        if n_out > 0:
            output_mask = np.zeros((n_out, out_M_max, out_N_max), dtype=bool)
            for i, (m, n) in enumerate(out_shapes_2d):
                output_mask[i, :m, :n] = True
        else:
            output_mask = np.zeros((0, 0, 0), dtype=bool)

        return cls(
            n_inputs=n_inputs,
            subscripts=subscripts,
            combo_indices=tuple(tuple(cl) for cl in combo_lists),
            output_indices=tuple(output_list),
            n_combos=n_combos,
            num_output_blocks=n_out,
            input_nd_shapes=tuple(tuple(s) for s in input_nd_shapes),
            output_nd_shape=output_nd_shape,
            output_charges=tuple(valid_out_keys),
            output_shapes=tuple(out_shapes_2d),
            output_tensor_indices=output_indices,
            output_mask_np=output_mask,
        )


def contract_multi_padded(
    plan: MultiContractionPlan,
    pbas: list[PaddedBlockArray],
) -> PaddedBlockArray:
    """Execute a multi-tensor contraction plan using vmap.

    Reshapes PBA blocks from 2D to ND, gathers blocks per combo, vmaps
    the full ND einsum over all combos, then scatter-adds results into
    output PBA blocks.

    Args:
        plan: MultiContractionPlan from MultiContractionPlan.build().
        pbas: List of PaddedBlockArrays, one per input tensor.

    Returns:
        PaddedBlockArray with the contraction result.
    """
    if plan.n_combos == 0 or plan.num_output_blocks == 0:
        # Compute output 2D max dims
        if plan.output_shapes:
            out_M = max(s[0] for s in plan.output_shapes)
            out_N = max(s[1] for s in plan.output_shapes)
        else:
            out_M = 0
            out_N = 0
        data = jnp.zeros(
            (plan.num_output_blocks, out_M, out_N),
            dtype=pbas[0].data.dtype if pbas else jnp.float64,
        )
        return PaddedBlockArray(
            data=data,
            block_charges=plan.output_charges,
            block_shapes=plan.output_shapes,
            indices=plan.output_tensor_indices,
            symmetry=pbas[0].symmetry if pbas else None,
        )

    # Parse subscripts for per-block einsum
    input_part, output_part = plan.subscripts.split("->")
    input_subs = input_part.split(",")

    # 1. Reshape each input PBA from 2D to ND
    nd_arrays = []
    for i, pba in enumerate(pbas):
        nd_shape = plan.input_nd_shapes[i]
        num_blocks = pba.data.shape[0]
        # Reshape from (num_blocks, M_max, N_max) to (num_blocks, *nd_shape)
        # M_max = product(nd_shape[:-1]), N_max = nd_shape[-1]
        # But the 2D M_max/N_max may be larger than product(nd_shape)/nd_shape[-1]
        # due to padding. We need to slice first, then reshape.
        nd_M = 1
        for d in nd_shape[:-1]:
            nd_M *= d
        nd_N = nd_shape[-1] if len(nd_shape) > 0 else 1

        if len(nd_shape) <= 2:
            # Already effectively 2D (or 1D), just slice to nd_M x nd_N
            nd_data = pba.data[:, :nd_M, :nd_N]
            if len(nd_shape) == 1:
                nd_data = nd_data.reshape(num_blocks, nd_shape[0])
            # For 2D, already correct shape
        else:
            # Slice to exact product dimensions, then reshape
            nd_data = pba.data[:, :nd_M, :nd_N]
            nd_data = nd_data.reshape(num_blocks, *nd_shape)
        nd_arrays.append(nd_data)

    # 2. Gather blocks per combo for each input
    gathered = []
    for i in range(plan.n_inputs):
        idx = jnp.array(plan.combo_indices[i], dtype=jnp.int32)
        gathered.append(nd_arrays[i][idx])  # (n_combos, *nd_shape_i)

    # 3. Build the per-block einsum subscript and vmap it
    # Build subscript without batch dimensions
    per_block_sub = ",".join(input_subs) + "->" + output_part

    def single_combo_einsum(*blocks):
        return jnp.einsum(per_block_sub, *blocks)

    # vmap over the combo dimension (axis 0 of each gathered array)
    vmapped_einsum = jax.vmap(single_combo_einsum)
    combo_results = vmapped_einsum(*gathered)
    # combo_results: (n_combos, *output_nd_shape)

    # 4. Scatter-add combo results in ND, then convert per-block to 2D.
    #
    # We cannot simply reshape the padded ND output to 2D globally because
    # different output blocks have different per-leg dims.  The PBA 2D layout
    # uses product-of-actual-dims, not product-of-per-leg-maxes.  Reshaping
    # with per-leg maxes would place valid data at wrong 2D row positions for
    # blocks whose per-leg dims differ from the maxes.
    #
    # Instead: scatter-add in ND space, then convert each block individually
    # from padded ND to the correct 2D layout.
    out_nd = plan.output_nd_shape

    output_idx = jnp.array(plan.output_indices, dtype=jnp.int32)

    if len(out_nd) <= 2:
        # For 1D/2D outputs, ND == 2D so the old reshape is correct.
        if len(out_nd) == 1:
            combo_results_2d = combo_results.reshape(plan.n_combos, out_nd[0], 1)
        else:
            combo_results_2d = combo_results  # already (n_combos, M, N)

        if plan.output_shapes:
            target_M = max(s[0] for s in plan.output_shapes)
            target_N = max(s[1] for s in plan.output_shapes)
        else:
            target_M = combo_results_2d.shape[1]
            target_N = combo_results_2d.shape[2]

        result_M = combo_results_2d.shape[1]
        result_N = combo_results_2d.shape[2]
        if result_M < target_M or result_N < target_N:
            pad_M = max(0, target_M - result_M)
            pad_N = max(0, target_N - result_N)
            combo_results_2d = jnp.pad(
                combo_results_2d, ((0, 0), (0, pad_M), (0, pad_N))
            )
        elif result_M > target_M or result_N > target_N:
            combo_results_2d = combo_results_2d[:, :target_M, :target_N]

        output_data = jnp.zeros(
            (plan.num_output_blocks, target_M, target_N),
            dtype=combo_results_2d.dtype,
        )
        output_data = output_data.at[output_idx].add(combo_results_2d)
    else:
        # ND output (>2 legs): scatter-add in ND, then remap per-block to 2D.
        output_nd_data = jnp.zeros(
            (plan.num_output_blocks, *out_nd),
            dtype=combo_results.dtype,
        )
        output_nd_data = output_nd_data.at[output_idx].add(combo_results)

        # Compute per-block actual ND shapes and target 2D max dims
        block_nd_shapes = _compute_nd_shapes(
            plan.output_tensor_indices, plan.output_charges
        )
        if plan.output_shapes:
            target_M = max(s[0] for s in plan.output_shapes)
            target_N = max(s[1] for s in plan.output_shapes)
        else:
            out_M_nd = 1
            for d in out_nd[:-1]:
                out_M_nd *= d
            target_M = out_M_nd
            target_N = out_nd[-1] if out_nd else 1

        # Convert each block from padded ND to correct 2D layout
        blocks_2d = []
        for blk_i in range(plan.num_output_blocks):
            nd_shape = block_nd_shapes[blk_i]
            # Slice padded ND block to actual per-leg dims
            slices = tuple(slice(0, d) for d in nd_shape)
            block_nd = output_nd_data[blk_i][slices]
            # Flatten all legs except the last to get the correct 2D layout
            rows_2d = 1
            for d in nd_shape[:-1]:
                rows_2d *= d
            cols_2d = nd_shape[-1]
            block_2d = block_nd.reshape(rows_2d, cols_2d)
            # Pad to (target_M, target_N)
            padded = jnp.zeros((target_M, target_N), dtype=block_2d.dtype)
            padded = padded.at[:rows_2d, :cols_2d].set(block_2d)
            blocks_2d.append(padded)

        output_data = jnp.stack(blocks_2d, axis=0)

    # Zero out padding regions
    output_data = jnp.where(jnp.array(plan.output_mask_np), output_data, 0.0)

    return PaddedBlockArray(
        data=output_data,
        block_charges=plan.output_charges,
        block_shapes=plan.output_shapes,
        indices=plan.output_tensor_indices,
        symmetry=pbas[0].symmetry,
    )


def update_left_env_blocks_jit(
    plan: MultiContractionPlan,
    L_pba: PaddedBlockArray,
    A_pba: PaddedBlockArray,
    W_pba: PaddedBlockArray,
    A_conj_pba: PaddedBlockArray,
) -> PaddedBlockArray:
    """Left environment update via multi-tensor contraction on PBAs.

    Computes: new_L[d,e,f] = L[a,b,c] * A[a,p,d] * W[b,p,x,e] * conj(A)[c,x,f]

    Args:
        plan:       MultiContractionPlan built for the left env subscript.
        L_pba:      Left environment as PaddedBlockArray.
        A_pba:      MPS site tensor as PaddedBlockArray.
        W_pba:      MPO site tensor as PaddedBlockArray.
        A_conj_pba: Conjugate MPS tensor as PaddedBlockArray (data = conj(A.data)).

    Returns:
        Updated left environment as PaddedBlockArray.
    """
    return contract_multi_padded(plan, [L_pba, A_pba, W_pba, A_conj_pba])


def update_right_env_blocks_jit(
    plan: MultiContractionPlan,
    R_pba: PaddedBlockArray,
    B_pba: PaddedBlockArray,
    W_pba: PaddedBlockArray,
    B_conj_pba: PaddedBlockArray,
) -> PaddedBlockArray:
    """Right environment update via multi-tensor contraction on PBAs.

    Computes: new_R[d,e,f] = R[a,b,c] * B[d,p,a] * W[e,p,x,b] * conj(B)[f,x,c]

    Args:
        plan:       MultiContractionPlan built for the right env subscript.
        R_pba:      Right environment as PaddedBlockArray.
        B_pba:      MPS site tensor as PaddedBlockArray.
        W_pba:      MPO site tensor as PaddedBlockArray.
        B_conj_pba: Conjugate MPS tensor as PaddedBlockArray (data = conj(B.data)).

    Returns:
        Updated right environment as PaddedBlockArray.
    """
    return contract_multi_padded(plan, [R_pba, B_pba, W_pba, B_conj_pba])
