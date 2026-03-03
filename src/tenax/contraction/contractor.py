r"""Tensor contraction engine with label-based API.

Primary API::

    contract(\*tensors, output_labels=None, optimize="auto") -> Tensor

Labels drive contraction: legs with the same label across different tensors
are contracted (summed over). Free labels (unique to one tensor) become
output legs. This is the Cytnx-style label-based contraction model.

Under the hood, labels are translated to einsum subscript strings which
are fed to opt_einsum for optimal contraction path finding, then executed
with the JAX backend.

Lower-level API::

    contract_with_subscripts(tensors, subscripts, output_indices, optimize) -> Tensor
    truncated_svd(tensor, left_labels, right_labels, ...) -> (U, s, Vh, s_full)
    qr_decompose(tensor, left_labels, right_labels, ...) -> (Q, R)
"""

from __future__ import annotations

import functools
import itertools
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum

from tenax.core.index import FlowDirection, Label, TensorIndex
from tenax.core.tensor import (
    BlockKey,
    DenseTensor,
    SymmetricTensor,
    Tensor,
    _compute_valid_blocks,
    _koszul_sign,
)

# ---------- Label → Subscript Translation ----------


def _labels_to_subscripts(
    tensors: Sequence[Tensor],
    output_labels: Sequence[Label] | None = None,
) -> tuple[str, tuple[TensorIndex, ...]]:
    """Build an einsum subscript string from tensor labels.

    Algorithm:
    1. Count how many times each label appears across all tensors.
    2. Labels appearing >= 2 times are contracted (summed over).
    3. Labels appearing exactly once are free (output) legs.
    4. Assign a unique letter from the alphabet to each unique label.
    5. Build the subscript string "legs_t0,legs_t1,...->output_legs".

    Args:
        tensors:       Sequence of Tensor objects.
        output_labels: Explicit ordering of free labels in the output.
                       If None, uses the order: free labels of t0, t1, ...

    Returns:
        (subscripts, output_indices) where output_indices are TensorIndex
        objects for the output legs in output_labels order.

    Raises:
        ValueError: If a label appears more than 2 times (ambiguous).
        ValueError: If output_labels contains a label not present as a free label.
    """
    # Count label occurrences across all tensors
    label_counts: Counter[Label] = Counter()
    label_to_index: dict[Label, TensorIndex] = {}

    for tensor in tensors:
        for idx in tensor.indices:
            label_counts[idx.label] += 1
            # Keep the first-seen index metadata for each label
            if idx.label not in label_to_index:
                label_to_index[idx.label] = idx

    # Validate: no label appears more than 2 times
    for label, count in label_counts.items():
        if count > 2:
            raise ValueError(
                f"Label {label!r} appears {count} times across tensors. "
                f"Labels must appear at most 2 times (one per tensor to contract)."
            )

    # Identify free labels (appear exactly once) and contracted labels (appear twice)
    free_labels = [lbl for lbl, cnt in label_counts.items() if cnt == 1]
    # contracted_labels = [lbl for lbl, cnt in label_counts.items() if cnt == 2]

    # Assign letters to labels (need at most 52 unique labels for a-zA-Z)
    # For larger networks use a different encoding (multi-char not supported by einsum)
    all_labels = sorted(label_counts.keys(), key=str)
    if len(all_labels) > 52:
        raise ValueError(
            f"Too many unique labels ({len(all_labels)}) for einsum encoding. "
            f"Maximum supported is 52 (a-z + A-Z)."
        )

    available_chars = string.ascii_lowercase + string.ascii_uppercase
    label_to_char: dict[Label, str] = {
        lbl: available_chars[i] for i, lbl in enumerate(all_labels)
    }

    # Build subscript strings per tensor
    tensor_subscripts = []
    for tensor in tensors:
        subs = "".join(label_to_char[idx.label] for idx in tensor.indices)
        tensor_subscripts.append(subs)

    # Determine output label ordering
    if output_labels is None:
        # Default: free labels in the order they appear across tensors
        seen: set[Label] = set()
        ordered_free: list[Label] = []
        for tensor in tensors:
            for idx in tensor.indices:
                if idx.label in free_labels and idx.label not in seen:
                    ordered_free.append(idx.label)
                    seen.add(idx.label)
        output_labels = ordered_free
    else:
        # Validate user-specified output labels
        free_set = set(free_labels)
        for lbl in output_labels:
            if lbl not in free_set:
                raise ValueError(
                    f"output_labels contains {lbl!r} which is not a free label. "
                    f"Free labels are: {free_labels}"
                )

    output_subs = "".join(label_to_char[lbl] for lbl in output_labels)
    subscripts = ",".join(tensor_subscripts) + "->" + output_subs

    # Build output TensorIndex objects (use first-seen index for each free label)
    output_indices = tuple(label_to_index[lbl] for lbl in output_labels)

    return subscripts, output_indices


# ---------- Dense contraction path cache ----------


@functools.lru_cache(maxsize=256)
def _cached_contraction_path(
    subscripts: str,
    shapes: tuple[tuple[int, ...], ...],
    optimize: str,
) -> list[tuple[int, ...]]:
    """Cache opt_einsum contraction paths by (subscripts, shapes, optimize).

    The path depends only on the subscript string and tensor shapes, not on
    the actual data.  Caching avoids repeating the O(n!) path search on
    every contraction call with the same shape signature — a key contributor
    to DMRG warmup time.
    """
    # Build dummy arrays (zeros) just for path planning — never executed on device
    dummy = [np.empty(s) for s in shapes]
    _, path_info = opt_einsum.contract_path(subscripts, *dummy, optimize=optimize)
    return path_info.path


# ---------- Dense contraction ----------


def _contract_dense(
    tensors: Sequence[DenseTensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    optimize: str = "auto",
) -> DenseTensor:
    """Contract dense tensors using opt_einsum with JAX backend.

    Uses a cached contraction path to avoid repeated path planning overhead.

    Args:
        tensors:        Sequence of DenseTensor.
        subscripts:     Einsum subscript string (e.g., "ij,jk->ik").
        output_indices: TensorIndex metadata for the output legs.
        optimize:       opt_einsum optimizer ('auto', 'greedy', 'dp', etc.).

    Returns:
        Contracted DenseTensor.
    """
    arrays = [t.todense() for t in tensors]
    shapes = tuple(a.shape for a in arrays)

    # Look up cached contraction path (or compute & cache it)
    path = _cached_contraction_path(subscripts, shapes, optimize)

    # Execute contraction with cached path and JAX backend (GPU-compatible)
    result = opt_einsum.contract(subscripts, *arrays, optimize=path, backend="jax")

    return DenseTensor(result, output_indices)


# ---------- Fermionic sign helpers ----------


def _contraction_inversion_pairs(
    input_subs: list[str],
    output_part: str,
) -> list[tuple[str, str]]:
    """Compute inversion pairs for fermionic contraction sign.

    The contraction conceptually reorders legs:
    1. For each input tensor, contracted legs move to the right.
    2. Free legs are then reordered to match the output order.

    We compute the composite permutation and return pairs of subscript
    characters whose exchange could contribute a fermionic sign.

    Args:
        input_subs: List of subscript strings, one per input tensor.
        output_part: Output subscript string.

    Returns:
        List of (char_i, char_j) pairs. For each pair, if both charges
        have odd parity, the overall sign flips.
    """
    # Build the "natural" order: all input legs concatenated in order
    all_chars: list[str] = []
    for subs in input_subs:
        all_chars.extend(subs)

    # Count occurrences to identify contracted vs free
    counts = Counter(all_chars)
    contracted = {c for c, n in counts.items() if n >= 2}

    # Build target order: free legs in output_part order, then contracted
    # legs in the order they first appear (they cancel out but the reordering
    # to bring them together matters).
    seen_contracted: set[str] = set()

    # For each input tensor, the contracted legs come at the end
    # We want pairs of (i, j) from `all_chars` where i appears after j
    # in the target ordering but before j in the natural ordering.
    # This is equivalent to computing the permutation and finding inversions.

    # Target ordering: for each input tensor, keep free legs in original
    # order, move contracted legs to the right (standard convention).
    # Then merge: free legs match output_part order; contracted legs pair up.

    # Step 1: Build canonical target list
    target: list[str] = list(output_part)
    for c in all_chars:
        if c in contracted and c not in seen_contracted:
            # Each contracted char appears twice; we just need it once
            # in the "contracted zone" to pair with itself
            target.append(c)
            seen_contracted.add(c)

    # Step 2: Build position map for each occurrence in all_chars
    # Each char in all_chars needs a target position
    char_positions_in_target: dict[str, list[int]] = {}
    for i, c in enumerate(target):
        char_positions_in_target.setdefault(c, []).append(i)

    # Assign target positions to each element in all_chars
    char_use_count: dict[str, int] = {}
    perm_targets: list[int] = []
    for c in all_chars:
        use_idx = char_use_count.get(c, 0)
        if c in contracted:
            # Contracted chars: both occurrences map to the same target position
            # (they'll be summed over), so we use the contracted-zone position
            perm_targets.append(char_positions_in_target[c][0] * 2 + use_idx)
        else:
            perm_targets.append(char_positions_in_target[c][0] * 2)
        char_use_count[c] = use_idx + 1

    # Step 3: Find inversion pairs (i < j but perm[i] > perm[j])
    pairs: list[tuple[str, str]] = []
    for i in range(len(all_chars)):
        for j in range(i + 1, len(all_chars)):
            if perm_targets[i] > perm_targets[j]:
                pairs.append((all_chars[i], all_chars[j]))

    return pairs


# ---------- Symmetric (block-sparse) contraction ----------


def _contract_symmetric(
    tensors: Sequence[SymmetricTensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    optimize: str = "auto",
) -> SymmetricTensor:
    """Contract block-sparse symmetric tensors using charge-indexed matching.

    Instead of iterating over the full Cartesian product of all input blocks
    (which is O(product of block counts) and mostly incompatible), this
    implementation pre-indexes blocks by their contracted-leg charge
    signatures and iterates only over compatible combinations.

    Algorithm:
    1. Parse subscripts to identify contracted and free legs per tensor.
    2. For each tensor, index blocks by (contracted-leg-charges) signature.
    3. Find contracted-charge tuples shared across all tensors.
    4. For each shared tuple, iterate over the (much smaller) product of
       matching blocks and accumulate into output blocks.

    Args:
        tensors:        Sequence of SymmetricTensor with the same symmetry group.
        subscripts:     Einsum subscript string.
        output_indices: TensorIndex metadata for output legs.
        optimize:       opt_einsum optimizer for within-block contractions.

    Returns:
        Contracted SymmetricTensor.
    """
    # Parse subscripts: e.g., "ij,jk->ik" → inputs=["ij","jk"], output="ik"
    input_part, output_part = subscripts.split("->")
    input_subs = input_part.split(",")

    # Map each character to the corresponding TensorIndex
    char_to_index: dict[str, TensorIndex] = {}
    for tensor, subs in zip(tensors, input_subs):
        for char, idx in zip(subs, tensor.indices):
            char_to_index[char] = idx

    # Build output_indices list in output_part order
    out_indices_ordered = tuple(char_to_index[c] for c in output_part)

    # Identify contracted characters (appear in multiple input tensors)
    char_counts: dict[str, int] = Counter(input_part.replace(",", ""))
    contracted_chars = {c for c, n in char_counts.items() if n >= 2}

    # Infer the output target charge from input tensors.
    # For U(1): output target = sum of input targets, since contracted legs
    # have opposite flows and cancel.  This allows contracting tensors with
    # non-identity targets (e.g. boundary MPS tensors targeting Sz != 0).
    # We only count a tensor's target if ALL its blocks agree on the same
    # value of sum(flow*q).  Mixed-charge tensors (e.g. operators that
    # create/annihilate particles) contribute 0.
    output_target: int | None = None
    total_target = 0
    for tensor in tensors:
        if tensor.blocks:
            targets = set()
            for key in tensor.blocks:
                t = sum(int(idx.flow) * int(q) for idx, q in zip(tensor.indices, key))
                targets.add(t)
            if len(targets) == 1:
                total_target += targets.pop()
    if total_target != 0:
        output_target = total_target

    # Precompute valid output keys as a set for O(1) lookup
    valid_output_set = set(
        _compute_valid_blocks(out_indices_ordered, target=output_target)
    )

    # Precompute fermionic sign structure (once, outside block loop)
    sym = tensors[0].indices[0].symmetry if tensors and tensors[0].indices else None
    is_fermionic = sym is not None and sym.is_fermionic
    inversion_pairs: list[tuple[str, str]] = []
    if is_fermionic:
        inversion_pairs = _contraction_inversion_pairs(input_subs, output_part)

    # For each tensor, build an index:
    #   contracted_charge_sig -> list of (block_key, block_array)
    # where contracted_charge_sig = tuple of charges on contracted legs
    # in a canonical order (sorted contracted chars).
    contracted_chars_sorted = sorted(contracted_chars)

    tensor_indices_by_sig: list[dict[tuple[int, ...], list[tuple[BlockKey, Any]]]] = []
    for tensor_i, (tensor, subs) in enumerate(zip(tensors, input_subs)):
        # Find which positions in this tensor's subscript are contracted
        contracted_positions = [
            pos for pos, c in enumerate(subs) if c in contracted_chars
        ]
        # Map contracted char -> position in contracted_chars_sorted
        char_to_contracted_pos = {c: i for i, c in enumerate(contracted_chars_sorted)}
        # For this tensor, map each contracted char to its position in subs
        contracted_char_positions = [
            (char_to_contracted_pos[subs[pos]], pos) for pos in contracted_positions
        ]
        # Sort by canonical contracted char order
        contracted_char_positions.sort(key=lambda x: x[0])

        sig_index: dict[tuple[int, ...], list[tuple[BlockKey, Any]]] = {}
        for key, array in tensor.blocks.items():
            # Extract charges at contracted leg positions, ordered canonically
            sig = tuple(int(key[pos]) for _, pos in contracted_char_positions)
            sig_index.setdefault(sig, []).append((key, array))
        tensor_indices_by_sig.append(sig_index)

    # Find contracted-charge signatures shared across all tensors
    if tensor_indices_by_sig:
        common_sigs = set(tensor_indices_by_sig[0].keys())
        for idx_map in tensor_indices_by_sig[1:]:
            common_sigs &= set(idx_map.keys())
    else:
        common_sigs = set()

    # Cache for within-block contraction expressions
    block_expr_cache: dict[tuple[tuple[int, ...], ...], Any] = {}

    output_blocks: dict[BlockKey, Any] = {}

    for sig in common_sigs:
        # Get matching blocks for each tensor
        matching_lists = [idx_map[sig] for idx_map in tensor_indices_by_sig]

        # Iterate over the product of matching blocks only
        for combo in itertools.product(*matching_lists):
            # combo: tuple of (key, array) pairs, one per tensor
            keys = [c[0] for c in combo]
            arrays = [c[1] for c in combo]

            # Build char -> charge mapping
            char_to_charge: dict[str, int] = {}
            compatible = True
            for tensor_i, (key, subs) in enumerate(zip(keys, input_subs)):
                for char, charge in zip(subs, key):
                    charge_int = int(charge)
                    if char in char_to_charge:
                        if char_to_charge[char] != charge_int:
                            compatible = False
                            break
                    else:
                        char_to_charge[char] = charge_int
                if not compatible:
                    break

            if not compatible:
                continue

            # Determine output block key
            output_key = tuple(char_to_charge.get(c, 0) for c in output_part)
            if output_key not in valid_output_set:
                continue

            # Contract using cached expression or opt_einsum
            block_shapes = tuple(a.shape for a in arrays)
            cache_key = (block_shapes,)
            if cache_key in block_expr_cache:
                expr = block_expr_cache[cache_key]
                result_array = expr(*arrays, backend="jax")
            else:
                try:
                    expr = opt_einsum.contract_expression(
                        subscripts,
                        *block_shapes,
                        optimize=optimize,
                    )
                    block_expr_cache[cache_key] = expr
                    result_array = expr(*arrays, backend="jax")
                except Exception:
                    continue

            # Apply fermionic sign from leg reordering
            if is_fermionic and inversion_pairs:
                sign = 1
                for ci, cj in inversion_pairs:
                    pi = int(sym.parity(np.array([char_to_charge[ci]]))[0])
                    pj = int(sym.parity(np.array([char_to_charge[cj]]))[0])
                    if pi and pj:
                        sign = -sign
                if sign < 0:
                    result_array = -result_array

            # Accumulate into output block
            if output_key in output_blocks:
                output_blocks[output_key] = output_blocks[output_key] + result_array
            else:
                output_blocks[output_key] = result_array

    if output_target is not None:
        # Non-identity target: bypass conservation validation
        obj = object.__new__(SymmetricTensor)
        obj._indices = out_indices_ordered
        obj._blocks = output_blocks
        return obj
    return SymmetricTensor(output_blocks, out_indices_ordered)


# ---------- Public API ----------


def contract(
    *tensors: Tensor,
    output_labels: Sequence[Label] | None = None,
    optimize: str = "auto",
) -> Tensor:
    """Contract tensors by matching shared labels (Cytnx-style).

    Legs with the same label across different tensors are automatically
    contracted (summed over). Legs with unique labels become output legs.

    Args:
        *tensors:       Two or more Tensor objects to contract.
        output_labels:  Explicit ordering of output legs by label.
                        If None, uses the natural order (labels of first tensor
                        that is free, then second, etc.).
        optimize:       opt_einsum path optimizer strategy.

    Returns:
        Contracted Tensor with indices corresponding to free labels.

    Raises:
        ValueError: If a label appears more than 2 times (ambiguous contraction).
        TypeError:  If tensors have mixed DenseTensor/SymmetricTensor types.

    Example:
        >>> # A has labels ('i', 'j', 'k'), B has labels ('k', 'l', 'm')
        >>> result = contract(A, B)
        >>> result.labels()
        ('i', 'j', 'l', 'm')
    """
    if not tensors:
        raise ValueError("contract() requires at least one tensor")

    subscripts, output_indices = _labels_to_subscripts(tensors, output_labels)

    # If a single tensor with no contractions needed, return it as-is
    if len(tensors) == 1 and "->" in subscripts:
        lhs, rhs = subscripts.split("->")
        if lhs == rhs:
            return tensors[0]

    return contract_with_subscripts(tensors, subscripts, output_indices, optimize)


def contract_with_subscripts(
    tensors: Sequence[Tensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    optimize: str = "auto",
) -> Tensor:
    """Contract tensors using an explicit einsum subscript string.

    Lower-level API for power users who prefer subscript notation.
    The output_indices must provide TensorIndex metadata for each output leg.

    Args:
        tensors:        Sequence of Tensor objects.
        subscripts:     Einsum subscript string (e.g., "ij,jk->ik").
        output_indices: TensorIndex metadata for output legs in subscript order.
        optimize:       opt_einsum optimizer.

    Returns:
        Contracted Tensor.

    Raises:
        TypeError: If tensors have mixed DenseTensor/SymmetricTensor types.
    """
    all_dense = all(isinstance(t, DenseTensor) for t in tensors)
    all_sym = all(isinstance(t, SymmetricTensor) for t in tensors)

    if all_dense:
        return _contract_dense(list(tensors), subscripts, output_indices, optimize)  # type: ignore[arg-type]
    elif all_sym:
        return _contract_symmetric(list(tensors), subscripts, output_indices, optimize)  # type: ignore[arg-type]
    else:
        types = [type(t).__name__ for t in tensors]
        raise TypeError(
            f"Cannot mix DenseTensor and SymmetricTensor in a single contraction. "
            f"Got types: {types}. Convert all tensors to the same type first."
        )


# ---------- Block-sparse decomposition helpers ----------


def _group_blocks_by_bond_charge(
    tensor: SymmetricTensor,
    left_leg_positions: list[int],
    right_leg_positions: list[int],
) -> dict[int, list[tuple[BlockKey, BlockKey, jax.Array]]]:
    """Group tensor blocks by their bond charge sector.

    For each block, the "bond charge" is determined by fusing the flow-weighted
    charges of the left legs.  Blocks sharing the same bond charge belong to
    the same diagonal block in the matrix representation.

    Args:
        tensor:              SymmetricTensor to decompose.
        left_leg_positions:  Axis positions belonging to the left (U / Q) factor.
        right_leg_positions: Axis positions belonging to the right (Vh / R) factor.

    Returns:
        Dict mapping bond charge ``q`` to a list of
        ``(left_subkey, right_subkey, block_array)`` tuples.
    """
    sym = tensor.indices[0].symmetry
    grouped: dict[int, list[tuple[BlockKey, BlockKey, jax.Array]]] = {}

    for key, block in tensor.blocks.items():
        # Compute bond charge from left legs
        effective = [
            np.array([int(tensor.indices[i].flow) * int(key[i])], dtype=np.int32)
            for i in left_leg_positions
        ]
        q = int(sym.fuse_many(effective)[0])

        left_subkey = tuple(key[i] for i in left_leg_positions)
        right_subkey = tuple(key[i] for i in right_leg_positions)
        grouped.setdefault(q, []).append((left_subkey, right_subkey, block))

    return grouped


def _truncated_svd_symmetric(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    max_singular_values: int | None,
    max_truncation_err: float | None,
    new_bond_label: Label,
    normalize: bool,
) -> tuple[SymmetricTensor, jax.Array, SymmetricTensor, jax.Array]:
    """Block-diagonal SVD for SymmetricTensor.

    Each charge sector is decomposed independently, then singular values
    are merged and truncated globally.

    Returns ``(U, s_truncated, Vh, s_full)`` where *s_full* contains all
    singular values (sorted descending) before truncation.
    """
    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    # Check if fermionic signs are needed for leg reordering
    sym = tensor.indices[0].symmetry
    is_fermionic = sym.is_fermionic
    # The permutation from original leg order to (left_axes, right_axes)
    decomp_perm = tuple(left_axes + right_axes)

    # For each charge sector, we need to know the row/col dimensions of the
    # block-diagonal matrix.  Rows are indexed by unique left_subkeys within
    # the sector; columns by unique right_subkeys.

    # Per-sector SVD results
    sector_results: dict[
        int,
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            list[BlockKey],
            list[BlockKey],
            list[int],
            list[int],
        ],
    ] = {}

    for q, entries in grouped.items():
        # Collect unique left / right subkeys (preserving order for determinism)
        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
        for lk, rk, _ in entries:
            if lk not in left_subkeys_seen:
                left_subkeys_seen[lk] = len(left_subkeys_seen)
            if rk not in right_subkeys_seen:
                right_subkeys_seen[rk] = len(right_subkeys_seen)

        left_subkeys = list(left_subkeys_seen.keys())
        right_subkeys = list(right_subkeys_seen.keys())

        # Determine row size per left_subkey and col size per right_subkey
        # by computing the product of charge-multiplicities along each leg.
        left_row_sizes: list[int] = []
        for lk in left_subkeys:
            size = 1
            for leg_pos, charge_val in zip(left_axes, lk):
                idx = tensor.indices[leg_pos]
                size *= int(np.sum(idx.charges == charge_val))
            left_row_sizes.append(size)

        right_col_sizes: list[int] = []
        for rk in right_subkeys:
            size = 1
            for leg_pos, charge_val in zip(right_axes, rk):
                idx = tensor.indices[leg_pos]
                size *= int(np.sum(idx.charges == charge_val))
            right_col_sizes.append(size)

        total_rows = sum(left_row_sizes)
        total_cols = sum(right_col_sizes)

        if total_rows == 0 or total_cols == 0:
            continue

        # Assemble the block matrix for this charge sector
        matrix = jnp.zeros((total_rows, total_cols), dtype=tensor.dtype)
        for lk, rk, block in entries:
            li = left_subkeys_seen[lk]
            ri = right_subkeys_seen[rk]
            row_start = sum(left_row_sizes[:li])
            col_start = sum(right_col_sizes[:ri])
            flat_block = block.reshape(left_row_sizes[li], right_col_sizes[ri])
            # Apply Koszul sign for leg reordering (original -> left+right)
            if is_fermionic:
                full_key = [0] * len(tensor.indices)
                for ax, ch in zip(left_axes, lk):
                    full_key[ax] = ch
                for ax, ch in zip(right_axes, rk):
                    full_key[ax] = ch
                parities = tuple(
                    int(sym.parity(np.array([full_key[i]]))[0])
                    for i in range(len(full_key))
                )
                ksign = _koszul_sign(parities, decomp_perm)
                if ksign < 0:
                    flat_block = -flat_block
            matrix = matrix.at[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ].set(flat_block)

        # SVD this sector
        U_q, s_q, Vh_q = jnp.linalg.svd(matrix, full_matrices=False)
        sector_results[q] = (
            U_q,
            s_q,
            Vh_q,
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
        )

    # Global truncation: merge all singular values across sectors
    all_sv_pairs: list[
        tuple[float, int, int]
    ] = []  # (value, sector_q, index_in_sector)
    for q, (_, s_q, _, _, _, _, _) in sector_results.items():
        s_np = np.array(s_q)
        for i, val in enumerate(s_np):
            all_sv_pairs.append((float(val), q, i))

    # Sort descending by singular value
    all_sv_pairs.sort(key=lambda x: -x[0])

    # Preserve the full singular-value spectrum before truncation
    s_full = jnp.array([v for v, _, _ in all_sv_pairs])

    # Determine global keep count
    n_total = len(all_sv_pairs)
    n_keep = n_total

    if max_truncation_err is not None and n_total > 0:
        total_sq = sum(x[0] ** 2 for x in all_sv_pairs)
        if total_sq > 0:
            trunc_sq = 0.0
            for i in range(n_total - 1, 0, -1):
                trunc_sq += all_sv_pairs[i][0] ** 2
                if trunc_sq / total_sq > max_truncation_err**2:
                    n_keep = i + 1
                    break
            else:
                n_keep = n_total

    if max_singular_values is not None:
        n_keep = min(n_keep, max_singular_values)

    n_keep = max(1, min(n_keep, n_total))

    # Count per-sector keep
    kept = all_sv_pairs[:n_keep]
    sector_keep_count: dict[int, int] = {}
    for _, q, _ in kept:
        sector_keep_count[q] = sector_keep_count.get(q, 0) + 1

    # Build the bond index charges: one entry per kept singular value,
    # charge = q for the sector it belongs to.
    # We need to order them: iterate sectors in sorted order.
    bond_charges_list: list[int] = []
    # Collect the final singular values in the same order
    final_sv_list: list[float] = []

    # Also build per-sector offset in the bond dimension
    sector_bond_offset: dict[int, int] = {}

    for q in sorted(sector_keep_count.keys()):
        sector_bond_offset[q] = len(bond_charges_list)
        n_q = sector_keep_count[q]
        bond_charges_list.extend([q] * n_q)
        s_q_np = np.array(sector_results[q][1])
        final_sv_list.extend(s_q_np[:n_q].tolist())

    bond_charges = np.array(bond_charges_list, dtype=np.int32)
    s_final = jnp.array(final_sv_list)

    if normalize and jnp.sum(s_final) > 0:
        s_final = s_final / jnp.sum(s_final)

    sym = tensor.indices[0].symmetry

    bond_index_out = TensorIndex(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    # Reconstruct U blocks: keys are (left_subkey..., bond_charge_q)
    # U has indices: (left_indices..., bond_index_out)
    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices

    U_blocks: dict[BlockKey, jax.Array] = {}
    Vh_blocks: dict[BlockKey, jax.Array] = {}

    for q in sorted(sector_keep_count.keys()):
        U_q, _, Vh_q, left_subkeys, right_subkeys, left_row_sizes, right_col_sizes = (
            sector_results[q]
        )
        n_q = sector_keep_count[q]

        # Slice U_q and Vh_q to keep only n_q singular vectors
        U_q_trunc = U_q[:, :n_q]
        Vh_q_trunc = Vh_q[:n_q, :]

        # Split U_q rows back into individual left_subkey blocks
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            u_slice = U_q_trunc[row_offset : row_offset + n_rows, :]
            # Reshape: (prod(left_shape_for_lk), n_q) -> (left_shape_for_lk..., n_q)
            left_shape = tuple(
                int(np.sum(tensor.indices[ax].charges == ch))
                for ax, ch in zip(left_axes, lk)
            )
            u_block = u_slice.reshape(left_shape + (n_q,))
            block_key = lk + (q,)
            U_blocks[block_key] = u_block
            row_offset += n_rows

        # Split Vh_q cols back into individual right_subkey blocks
        col_offset = 0
        for ri, rk in enumerate(right_subkeys):
            n_cols = right_col_sizes[ri]
            vh_slice = Vh_q_trunc[:, col_offset : col_offset + n_cols]
            right_shape = tuple(
                int(np.sum(tensor.indices[ax].charges == ch))
                for ax, ch in zip(right_axes, rk)
            )
            vh_block = vh_slice.reshape((n_q,) + right_shape)
            block_key = (q,) + rk
            Vh_blocks[block_key] = vh_block
            col_offset += n_cols

    # Check if input tensor has a non-identity target (e.g. boundary MPS
    # tensor targeting Sz != 0).  If so, the output tensors may also have
    # non-identity targets and need to bypass standard validation.
    input_target = 0
    if tensor.blocks:
        key0 = next(iter(tensor.blocks))
        input_target = sum(
            int(idx.flow) * int(q) for idx, q in zip(tensor.indices, key0)
        )

    if input_target != 0:
        # Bypass validation for non-identity targets
        U_tensor = object.__new__(SymmetricTensor)
        U_tensor._indices = U_indices
        U_tensor._blocks = U_blocks
        Vh_tensor = object.__new__(SymmetricTensor)
        Vh_tensor._indices = Vh_indices
        Vh_tensor._blocks = Vh_blocks
    else:
        U_tensor = SymmetricTensor(U_blocks, U_indices)
        Vh_tensor = SymmetricTensor(Vh_blocks, Vh_indices)

    return U_tensor, s_final, Vh_tensor, s_full


def _qr_symmetric(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Block-diagonal QR decomposition for SymmetricTensor.

    Each charge sector is decomposed independently; the bond index carries
    the sector charge with multiplicity = min(left_dim, right_dim) per sector.
    """
    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    # Check if fermionic signs are needed for leg reordering
    sym_qr = tensor.indices[0].symmetry
    is_fermionic_qr = sym_qr.is_fermionic
    decomp_perm_qr = tuple(left_axes + right_axes)

    # Per-sector QR results
    sector_results: dict[
        int,
        tuple[
            jax.Array,
            jax.Array,
            list[BlockKey],
            list[BlockKey],
            list[int],
            list[int],
            int,
        ],
    ] = {}

    bond_charges_list: list[int] = []
    sector_bond_offset: dict[int, int] = {}

    for q in sorted(grouped.keys()):
        entries = grouped[q]

        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
        for lk, rk, _ in entries:
            if lk not in left_subkeys_seen:
                left_subkeys_seen[lk] = len(left_subkeys_seen)
            if rk not in right_subkeys_seen:
                right_subkeys_seen[rk] = len(right_subkeys_seen)

        left_subkeys = list(left_subkeys_seen.keys())
        right_subkeys = list(right_subkeys_seen.keys())

        left_row_sizes: list[int] = []
        for lk in left_subkeys:
            size = 1
            for leg_pos, charge_val in zip(left_axes, lk):
                idx = tensor.indices[leg_pos]
                size *= int(np.sum(idx.charges == charge_val))
            left_row_sizes.append(size)

        right_col_sizes: list[int] = []
        for rk in right_subkeys:
            size = 1
            for leg_pos, charge_val in zip(right_axes, rk):
                idx = tensor.indices[leg_pos]
                size *= int(np.sum(idx.charges == charge_val))
            right_col_sizes.append(size)

        total_rows = sum(left_row_sizes)
        total_cols = sum(right_col_sizes)

        if total_rows == 0 or total_cols == 0:
            continue

        # Assemble block matrix
        matrix = jnp.zeros((total_rows, total_cols), dtype=tensor.dtype)
        for lk, rk, block in entries:
            li = left_subkeys_seen[lk]
            ri = right_subkeys_seen[rk]
            row_start = sum(left_row_sizes[:li])
            col_start = sum(right_col_sizes[:ri])
            flat_block = block.reshape(left_row_sizes[li], right_col_sizes[ri])
            # Apply Koszul sign for leg reordering (original -> left+right)
            if is_fermionic_qr:
                full_key = [0] * len(tensor.indices)
                for ax, ch in zip(left_axes, lk):
                    full_key[ax] = ch
                for ax, ch in zip(right_axes, rk):
                    full_key[ax] = ch
                parities = tuple(
                    int(sym_qr.parity(np.array([full_key[i]]))[0])
                    for i in range(len(full_key))
                )
                ksign = _koszul_sign(parities, decomp_perm_qr)
                if ksign < 0:
                    flat_block = -flat_block
            matrix = matrix.at[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ].set(flat_block)

        Q_q, R_q = jnp.linalg.qr(matrix)
        bond_dim_q = Q_q.shape[1]

        sector_bond_offset[q] = len(bond_charges_list)
        bond_charges_list.extend([q] * bond_dim_q)
        sector_results[q] = (
            Q_q,
            R_q,
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
            bond_dim_q,
        )

    bond_charges = np.array(bond_charges_list, dtype=np.int32)
    sym = tensor.indices[0].symmetry

    bond_index_out = TensorIndex(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    Q_indices = left_indices + (bond_index_out,)
    R_indices = (bond_index_in,) + right_indices

    Q_blocks: dict[BlockKey, jax.Array] = {}
    R_blocks: dict[BlockKey, jax.Array] = {}

    for q, (
        Q_q,
        R_q,
        left_subkeys,
        right_subkeys,
        left_row_sizes,
        right_col_sizes,
        bond_dim_q,
    ) in sector_results.items():
        # Split Q rows back into left_subkey blocks
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            q_slice = Q_q[row_offset : row_offset + n_rows, :]
            left_shape = tuple(
                int(np.sum(tensor.indices[ax].charges == ch))
                for ax, ch in zip(left_axes, lk)
            )
            q_block = q_slice.reshape(left_shape + (bond_dim_q,))
            Q_blocks[lk + (q,)] = q_block
            row_offset += n_rows

        # Split R cols back into right_subkey blocks
        col_offset = 0
        for ri, rk in enumerate(right_subkeys):
            n_cols = right_col_sizes[ri]
            r_slice = R_q[:, col_offset : col_offset + n_cols]
            right_shape = tuple(
                int(np.sum(tensor.indices[ax].charges == ch))
                for ax, ch in zip(right_axes, rk)
            )
            r_block = r_slice.reshape((bond_dim_q,) + right_shape)
            R_blocks[(q,) + rk] = r_block
            col_offset += n_cols

    Q_tensor = SymmetricTensor(Q_blocks, Q_indices)
    R_tensor = SymmetricTensor(R_blocks, R_indices)

    return Q_tensor, R_tensor


# ---------- Truncated SVD ----------


def truncated_svd(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
    max_singular_values: int | None = None,
    max_truncation_err: float | None = None,
    normalize: bool = False,
) -> tuple[Tensor, jax.Array, Tensor, jax.Array]:
    """Reshape tensor into matrix, compute SVD, truncate, reshape back.

    The tensor is first reshaped into a matrix by grouping left_labels as
    rows and right_labels as columns. After SVD and truncation, the result
    is reshaped back.

    The new bond leg (connecting U and Vh factors) is given label
    new_bond_label, making it immediately usable in label-based contractions.

    Output labels::

        U:  (left_labels..., new_bond_label)
        Vh: (new_bond_label, right_labels...)

    Note:
        This function is not JIT-able as a whole because the truncation
        cutoff is determined dynamically from singular values (dynamic shape).
        Apply ``@jax.jit`` to the inner SVD step only; call this at Python level.

    Args:
        tensor:               Tensor to decompose.
        left_labels:          Labels forming the "left" (U) factor.
        right_labels:         Labels forming the "right" (Vh) factor.
        new_bond_label:       Label for the new virtual bond.
        max_singular_values:  Hard cap on bond dimension after truncation.
        max_truncation_err:   Truncate until relative truncation error <= this.
        normalize:            Normalize singular values to sum to 1.

    Returns:
        ``(U_tensor, singular_values, Vh_tensor, singular_values_full)``
        -- U has labels ``(left_labels..., new_bond_label)``.
        Vh has labels ``(new_bond_label, right_labels...)``.
        singular_values is a 1-D JAX float array (truncated).
        singular_values_full is a 1-D JAX float array containing **all**
        singular values before truncation (length = min(left_dim, right_dim)),
        useful for computing truncation error without a second SVD.

    Raises:
        ValueError: If left_labels + right_labels don't cover all tensor labels.
    """
    all_labels = tensor.labels()
    all_labels_set = set(all_labels)
    left_set = set(left_labels)
    right_set = set(right_labels)

    if left_set | right_set != all_labels_set:
        raise ValueError(
            f"left_labels {list(left_labels)} + right_labels {list(right_labels)} "
            f"must cover all tensor labels {list(all_labels)}"
        )
    if left_set & right_set:
        raise ValueError(
            f"left_labels and right_labels must be disjoint, "
            f"got overlap: {left_set & right_set}"
        )

    # Dispatch to block-sparse path for SymmetricTensor
    if isinstance(tensor, SymmetricTensor):
        return _truncated_svd_symmetric(
            tensor,
            left_labels,
            right_labels,
            max_singular_values,
            max_truncation_err,
            new_bond_label,
            normalize,
        )

    # Build axis ordering: left labels first, then right labels
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]

    # Get dense representation and reshape
    dense = tensor.todense()
    perm = left_axes + right_axes
    dense_perm = jnp.transpose(dense, perm)

    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)
    left_dim = int(np.prod([idx.dim for idx in left_indices]))
    right_dim = int(np.prod([idx.dim for idx in right_indices]))

    matrix = dense_perm.reshape(left_dim, right_dim)

    # SVD (not JIT-able at this level due to dynamic truncation)
    U, s, Vh = jnp.linalg.svd(matrix, full_matrices=False)

    # Preserve the full singular-value spectrum before truncation
    s_full = s

    # Determine truncation cutoff
    s_np = np.array(s)
    n_keep = len(s_np)

    if max_truncation_err is not None:
        # Keep singular values until truncation error <= max_truncation_err
        total_sq = float(np.sum(s_np**2))
        trunc_sq = 0.0
        for i in range(len(s_np) - 1, -1, -1):
            trunc_sq += float(s_np[i] ** 2)
            if trunc_sq / total_sq > max_truncation_err**2:
                n_keep = i + 1
                break
        else:
            n_keep = len(s_np)

    if max_singular_values is not None:
        n_keep = min(n_keep, max_singular_values)

    n_keep = max(1, n_keep)  # always keep at least one

    # Truncate
    U = U[:, :n_keep]
    s = s[:n_keep]
    Vh = Vh[:n_keep, :]

    if normalize:
        s = s / jnp.sum(s)

    # Reshape back and build output tensors
    left_shape = tuple(idx.dim for idx in left_indices)
    right_shape = tuple(idx.dim for idx in right_indices)

    U_dense = U.reshape(left_shape + (n_keep,))
    Vh_dense = Vh.reshape((n_keep,) + right_shape)

    # Build new bond index
    # Convention: bond on U is OUT (outgoing from left side)
    #             bond on Vh is IN (incoming to right side)
    # The charges on the bond index are 0..n_keep-1 (no symmetry on singular values)
    # For a dense SVD we use a trivial bond with all charges = 0
    # (symmetric SVD with charge-preserving structure is handled separately)
    bond_charges_out = np.zeros(n_keep, dtype=np.int32)
    if left_indices:
        sym = left_indices[0].symmetry
    elif right_indices:
        sym = right_indices[0].symmetry
    else:
        from tenax.core.symmetry import U1Symmetry

        sym = U1Symmetry()

    bond_index_out = TensorIndex(
        sym, bond_charges_out, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex(
        sym, bond_charges_out, FlowDirection.IN, label=new_bond_label
    )

    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices

    U_tensor = DenseTensor(U_dense, U_indices)
    Vh_tensor = DenseTensor(Vh_dense, Vh_indices)

    return U_tensor, s, Vh_tensor, s_full


# ---------- QR Decomposition ----------


def qr_decompose(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
) -> tuple[Tensor, Tensor]:
    """QR decomposition of a tensor for canonical form in DMRG.

    Reshapes tensor into a matrix, performs QR, then reshapes back.

    Output labels::

        Q: (left_labels..., new_bond_label)
        R: (new_bond_label, right_labels...)

    Args:
        tensor:          Tensor to decompose.
        left_labels:     Labels forming the Q (isometric) factor.
        right_labels:    Labels forming the R (upper triangular) factor.
        new_bond_label:  Label for the new virtual bond.

    Returns:
        (Q_tensor, R_tensor) where Q is isometric (Q^dag Q = I).
    """
    # Dispatch to block-sparse path for SymmetricTensor
    if isinstance(tensor, SymmetricTensor):
        return _qr_symmetric(tensor, left_labels, right_labels, new_bond_label)

    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]

    dense = tensor.todense()
    perm = left_axes + right_axes
    dense_perm = jnp.transpose(dense, perm)

    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)
    left_dim = int(np.prod([idx.dim for idx in left_indices]))
    right_dim = int(np.prod([idx.dim for idx in right_indices]))

    matrix = dense_perm.reshape(left_dim, right_dim)
    Q, R = jnp.linalg.qr(matrix)

    bond_dim = Q.shape[1]
    left_shape = tuple(idx.dim for idx in left_indices)
    right_shape = tuple(idx.dim for idx in right_indices)

    Q_dense = Q.reshape(left_shape + (bond_dim,))
    R_dense = R.reshape((bond_dim,) + right_shape)

    bond_charges = np.zeros(bond_dim, dtype=np.int32)
    if left_indices:
        sym = left_indices[0].symmetry
    else:
        from tenax.core.symmetry import U1Symmetry

        sym = U1Symmetry()

    bond_index_out = TensorIndex(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    Q_indices = left_indices + (bond_index_out,)
    R_indices = (bond_index_in,) + right_indices

    Q_tensor = DenseTensor(Q_dense, Q_indices)
    R_tensor = DenseTensor(R_dense, R_indices)

    return Q_tensor, R_tensor
