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

Linear algebra decompositions (``svd``, ``qr``, ``eigh``) live in
``tenax.linalg``; legacy names ``truncated_svd`` and ``qr_decompose``
are re-exported here for backwards compatibility.
"""

from __future__ import annotations

import functools
import itertools
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import opt_einsum

from tenax.core.index import Label, TensorIndex
from tenax.core.tensor import (
    BlockKey,
    DenseTensor,
    SymmetricTensor,
    Tensor,
    _compute_valid_blocks,
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


# ---------- Linear algebra re-exports (moved to tenax.linalg) ----------

from tenax.linalg import eigh  # noqa: F401, E402
from tenax.linalg import qr as qr_decompose  # noqa: F401, E402
from tenax.linalg import svd as truncated_svd  # noqa: F401, E402
