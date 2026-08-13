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
import os
import string
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum

from tenax.core.index import Label, TensorIndex, _net_charges
from tenax.core.tensor import (
    BlockKey,
    DenseTensor,
    SymmetricTensor,
    Tensor,
    _compute_valid_blocks,
)

# Test-observability counter: incremented each time the gated even-D stacked
# block-sparse contraction path (TENAX_STACK_BLOCKSPARSE) actually executes.
_STACK_FIRED = {"n": 0}

# Persist-hit-rate instrumentation (#566 P1d slice 1). Each stacked contraction
# records, per operand, whether the input arrived as a StackedSymmetricTensor
# (its stacked arrays PERSISTED from an upstream contraction, no _data gather) or
# as a plain SymmetricTensor (the chain was INTERRUPTED upstream by a
# fuse/bar/svd/_data-read, forcing a gather from _data).
_STACK_PERSIST = {
    "calls": 0,  # stacked contractions executed
    "persisted_inputs": 0,  # operands that arrived already-stacked
    "gathered_inputs": 0,  # operands that had to gather from _data
    "fully_persisted": 0,  # calls where BOTH operands persisted
}

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

    # Map labels to single characters for einsum subscripts (a-z + A-Z = 52 max).
    # This limit comes from NumPy/JAX einsum's single-character subscript format.
    # In practice, 52 labels is sufficient for all standard tensor network
    # algorithms (DMRG, TRG, iPEPS).  For very large custom networks exceeding
    # this limit, split the contraction into smaller pairwise steps.
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

    # Optional cuTensorNet GPU path for dense contractions (~5x over opt_einsum).
    # Enabled when TENAX_USE_CUTENSORNET=1 and cuTensorNet is available on GPU.
    import os

    if os.environ.get("TENAX_USE_CUTENSORNET", "0") == "1":
        from tenax.contraction.cutensornet_backend import is_available as _cutn_ok

        if _cutn_ok():
            from tenax.contraction.cutensornet_backend import contract_ad

            result = contract_ad(subscripts, *arrays)
            return DenseTensor(result, output_indices)

    shapes = tuple(a.shape for a in arrays)

    # Look up cached contraction path (or compute & cache it)
    path = _cached_contraction_path(subscripts, shapes, optimize)

    # Execute contraction with cached path and JAX backend (GPU-compatible)
    result = opt_einsum.contract(subscripts, *arrays, optimize=path, backend="jax")

    return DenseTensor(result, output_indices)


# ---------- Symmetric (block-sparse) contraction ----------


def _contract_symmetric_batched(
    sig_iter: Any,
    tensor_partial_indices: list[
        tuple[list[tuple[int, int]], dict[tuple[int, ...], list[tuple[BlockKey, Any]]]]
    ],
    input_subs: list[str],
    output_part: str,
    input_part: str,
    valid_output_set: set[tuple[int, ...]],
) -> dict[BlockKey, Any]:
    """Batched execution of the surviving block-combos (issue #568, M-B).

    Performs the SAME filtering as the per-combo path (compatibility check +
    valid_output_set membership) to collect surviving units, each carrying its
    ``output_key`` and tuple of input arrays.  Surviving units are then grouped
    by their input block-shape signature; per group a single batched
    ``jnp.einsum`` runs over a fresh leading batch axis, and
    ``jax.ops.segment_sum`` collapses combos that share an ``output_key``.
    Groups are merged additively into the final ``output_blocks`` dict.

    The subscripts are constant for the whole call, so the input block-shape
    signature fully determines the einsum kernel; grouping on it guarantees
    ``jnp.stack`` sees identical shapes per tensor-position.
    """
    # Collect surviving units: (output_key, tuple-of-arrays).
    survivors: list[tuple[tuple[int, ...], tuple[Any, ...]]] = []

    for full_sig in sig_iter:
        matching_lists: list[list[tuple[BlockKey, Any]]] = []
        skip = False
        for covered, partial_sig_index in tensor_partial_indices:
            partial_sig = tuple(full_sig[canonical_pos] for canonical_pos, _ in covered)
            matching = partial_sig_index.get(partial_sig)
            if not matching:
                skip = True
                break
            matching_lists.append(matching)
        if skip:
            continue

        for combo in itertools.product(*matching_lists):
            keys = [c[0] for c in combo]
            arrays = tuple(c[1] for c in combo)

            char_to_charge: dict[str, int] = {}
            compatible = True
            for key, subs in zip(keys, input_subs):
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

            output_key = tuple(char_to_charge.get(c, 0) for c in output_part)
            if output_key not in valid_output_set:
                continue

            survivors.append((output_key, arrays))

    output_blocks: dict[BlockKey, Any] = {}
    if not survivors:
        return output_blocks

    # Pick a batch label not already used in the subscripts.
    used_chars = set(input_part.replace(",", "")) | set(output_part)
    batch_char = None
    for c in string.ascii_letters:
        if c not in used_chars:
            batch_char = c
            break
    if batch_char is None:  # pragma: no cover - 52-label ceiling guards this
        raise ValueError("No free einsum label available for batch axis.")

    # Build the batched subscripts: prepend the batch char to every input
    # operand and to the output operand.
    batched_inputs = ",".join(batch_char + s for s in input_subs)
    batched_subscripts = batched_inputs + "->" + batch_char + output_part

    # Group surviving units by full input block-shape signature.
    groups: dict[tuple[tuple[int, ...], ...], list[int]] = {}
    for unit_i, (_okey, arrays) in enumerate(survivors):
        shape_sig = tuple(a.shape for a in arrays)
        groups.setdefault(shape_sig, []).append(unit_i)

    n_pos = len(input_subs)

    for unit_indices in groups.values():
        group_units = [survivors[i] for i in unit_indices]

        # Stable per-group ordering of distinct output_keys -> segment ids.
        distinct_keys: list[tuple[int, ...]] = []
        key_to_seg: dict[tuple[int, ...], int] = {}
        seg_ids: list[int] = []
        for okey, _arrays in group_units:
            seg = key_to_seg.get(okey)
            if seg is None:
                seg = len(distinct_keys)
                key_to_seg[okey] = seg
                distinct_keys.append(okey)
            seg_ids.append(seg)

        # Stack each tensor-position's arrays along a new leading batch axis.
        stacked = [
            jnp.stack([u[1][pos] for u in group_units], axis=0) for pos in range(n_pos)
        ]

        # One batched einsum over the batch axis.
        batched_result = jnp.einsum(batched_subscripts, *stacked)

        # Sum combos sharing an output_key via segment_sum over the batch axis.
        segments = jnp.asarray(seg_ids, dtype=jnp.int32)
        summed = jax.ops.segment_sum(
            batched_result, segments, num_segments=len(distinct_keys)
        )

        # Merge across groups: ADD to any existing entry.
        for seg, okey in enumerate(distinct_keys):
            block = summed[seg]
            if okey in output_blocks:
                output_blocks[okey] = output_blocks[okey] + block
            else:
                output_blocks[okey] = block

    return output_blocks


def _parse_contraction_prelude(
    tensors: Sequence[SymmetricTensor],
    subscripts: str,
) -> tuple[
    list[str],
    str,
    tuple[TensorIndex, ...],
    set[str],
    list[str],
    dict[str, int],
    int | None,
    set[BlockKey],
]:
    """Shared prelude for the per-block and stacked symmetric contraction paths.

    Returns the subscript/charge bookkeeping common to both; per-tensor block
    indexing (arrays vs stacked rows) is built separately by each caller.

    Returns:
        input_subs, output_part, out_indices_ordered, contracted_chars (set),
        contracted_chars_sorted, char_to_canonical_pos, output_target,
        valid_output_set.
    """
    # Parse subscripts: e.g., "ij,jk->ik" -> inputs=["ij","jk"], output="ik".
    input_part, output_part = subscripts.split("->")
    input_subs = input_part.split(",")

    # Map each character to the corresponding TensorIndex, then build the
    # output index tuple in output_part order.
    char_to_index: dict[str, TensorIndex] = {}
    for tensor, subs in zip(tensors, input_subs):
        for char, idx in zip(subs, tensor.indices):
            char_to_index[char] = idx
    out_indices_ordered = tuple(char_to_index[c] for c in output_part)

    # Identify contracted characters (appear in multiple input tensors).
    char_counts: dict[str, int] = Counter(input_part.replace(",", ""))
    contracted_chars = {c for c, n in char_counts.items() if n >= 2}

    # Infer the output target charge from input tensors.
    # Output target = fusion of the input targets, since contracted legs have
    # opposite flows and cancel.  This allows contracting tensors with
    # non-identity targets (e.g. boundary MPS tensors targeting Sz != 0).
    # We only count a tensor's target if ALL its blocks agree on the same net
    # charge.  Mixed-charge tensors (e.g. operators that create/annihilate
    # particles) contribute the identity.  Iterating ``_block_keys`` is
    # equivalent to iterating ``.blocks`` keys but cheaper.
    # ``indices`` is part of the SymmetricTensor contract, so read it directly.
    # A ``getattr(..., None)`` here read as "tensors might not have indices",
    # which is not true -- the loop above already dereferences ``tensor.indices``
    # unconditionally -- and would have turned a future rename into "no target
    # inferred", silently, which is the #734 failure mode rather than an
    # AttributeError.  Unobservable today; the point is that it stays that way.
    output_target: int | None = None
    sym = None
    for tensor in tensors:
        if tensor.indices:
            sym = tensor.indices[0].symmetry
            break

    if sym is not None:
        # Accumulate with fuse, not +=: adding targets as plain integers is the
        # same category error as weighting a charge by int(flow) (#734).  For
        # the bit-packed charges of ProductSymmetry the sum carries across the
        # 16-bit factor boundary and lands on a different charge entirely.
        total = np.array([sym.identity()], dtype=np.int32)
        for tensor in tensors:
            if not tensor.indices or not getattr(tensor, "_block_keys", None):
                continue
            targets = np.unique(_net_charges(tensor.indices, tensor._block_keys))
            if targets.size == 1:
                total = sym.fuse(total, targets)
        total_target = int(total[0])
        if total_target != sym.identity():
            output_target = total_target

    # Precompute valid output keys as a set for O(1) lookup.
    valid_output_set = set(
        _compute_valid_blocks(out_indices_ordered, target=output_target)
    )

    contracted_chars_sorted = sorted(contracted_chars)
    char_to_canonical_pos = {c: i for i, c in enumerate(contracted_chars_sorted)}

    return (
        input_subs,
        output_part,
        out_indices_ordered,
        contracted_chars,
        contracted_chars_sorted,
        char_to_canonical_pos,
        output_target,
        valid_output_set,
    )


def _contract_symmetric_stacked(
    tensors: Sequence[SymmetricTensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    optimize: str = "auto",
    backend: Any = None,
    plan: Any = None,
) -> SymmetricTensor | None:
    """Even-D stacked block-sparse contraction (issue #566, P1b; seam A2).

    Sources operands DIRECTLY from each tensor's contiguous ``_data`` buffer via
    ``stacked_blocks()`` (one row-gather per shape-group) + ``jnp.take`` (one
    op per operand), runs the contraction through the supplied ``backend`` (the
    backend returns the canonical-ordered stacked output array — same shape
    contract as ``stacked_execute``), then assembles the output
    ``SymmetricTensor``.  It builds partial charge signatures from the STATIC
    ``_block_keys`` — NOT ``.blocks`` / ``_get_block`` — so no per-block slicing
    is emitted at trace time.

    Scoped tightly (returns ``None`` to fall back to the per-block path):
      * only 2-tensor contractions;
      * only when EVERY input tensor is single-shape-group
        (``len(set(t._block_shapes)) <= 1``) — the even-D case.

    No Koszul signs are applied (matching the per-block path; planar networks
    need none — see ``_contract_symmetric``).  Value and gradient match the
    per-block path within the fp tier.

    Args:
        backend: a :class:`~tenax.contraction.blocksparse_backend.BlockSparseContractBackend`
            whose ``execute`` produces the canonical stacked output array. When
            ``None``, the default :class:`StackedJaxBackend` is used (legacy
            direct-call behavior).
        plan: a prebuilt :class:`BlockContractPlan` (the dispatcher builds it once
            and reuses it for backend selection); built here if ``None``.

    Returns the contracted ``SymmetricTensor``, or ``None`` to fall back.
    """
    # ``optimize`` is accepted for signature parity with the dispatcher and is
    # unused here (the backend kernel takes no optimize argument).
    del optimize

    from tenax.contraction.blocksparse_backend import StackedJaxBackend
    from tenax.contraction.blocksparse_plan import build_block_contract_plan
    from tenax.core.stacked_tensor import StackedSymmetricTensor
    from tenax.core.stacked_view import StackedView, StackGroup

    # --- Backend-agnostic plan from STATIC metadata only (charge matching,
    # valid-output filter, combo grouping, segment/accumulation structure,
    # batched subscripts, output keys/shapes). None => out of scope -> fall back.
    if plan is None:
        plan = build_block_contract_plan(tensors, subscripts, output_indices)
    if plan is None:
        return None

    if backend is None:
        backend = StackedJaxBackend()

    out_indices_ordered = plan.out_indices
    output_target = plan.output_target

    # --- Source operands from the contiguous _data buffer (one gather/group).
    # If an operand already carries a cached StackedView (produced by an upstream
    # stacked contraction), stacked_blocks() returns it with NO gather from
    # _data — the chain persisted. Otherwise stacked_blocks() gathers from the
    # flat _data buffer (chain interrupted upstream). The single shape-group is
    # guaranteed by the plan's scope check.
    operand_stacks: list[Any] = []
    n_pos = len(tensors)
    n_persisted = 0
    for tensor in tensors:
        if isinstance(tensor, StackedSymmetricTensor):
            n_persisted += 1
            _STACK_PERSIST["persisted_inputs"] += 1
        else:
            _STACK_PERSIST["gathered_inputs"] += 1
        view = tensor.stacked_blocks()
        (group,) = view.groups.values()
        operand_stacks.append(group.array)

    # Empty plan (no surviving blocks): assemble an empty tensor exactly as before.
    if not plan.groups:
        output_blocks: dict[BlockKey, Any] = {}
        if output_target is not None:
            return SymmetricTensor._from_blocks_unchecked(
                output_blocks, out_indices_ordered
            )
        return SymmetricTensor(output_blocks, out_indices_ordered)

    # --- Data-level execution through the selected backend. The backend returns
    # the canonical-ordered stacked output array of shape
    # (n_out_blocks, *out_block_shape); rows map to ``plan.out_block_keys``
    # positionally. Backend-neutral: the assembly below is identical regardless
    # of which backend produced ``out_stack``.
    out_stack = backend.execute(operand_stacks, plan)

    _STACK_FIRED["n"] += 1
    _STACK_PERSIST["calls"] += 1
    if n_persisted == n_pos:
        _STACK_PERSIST["fully_persisted"] += 1

    # Diagnostic sub-flag: TENAX_STACK_PERSIST_RETURN=0 falls back to the
    # original _from_blocks_unchecked return (no persisting subclass). Isolates
    # the persisting-return change from the batched-einsum core for debugging.
    import os

    _persist_return = os.environ.get(
        "TENAX_STACK_PERSIST_RETURN", "1"
    ).strip().lower() in ("1", "true", "yes", "on")
    if not _persist_return:
        out_blocks = {
            plan.out_block_keys[i]: out_stack[i]
            for i in range(len(plan.out_block_keys))
        }
        if output_target is not None:
            return SymmetricTensor._from_blocks_unchecked(
                out_blocks, out_indices_ordered
            )
        return SymmetricTensor(out_blocks, out_indices_ordered)

    # --- Build the output as a PERSISTING StackedSymmetricTensor (#566 P1d).
    # Keep the batched output array as the cached StackedView (lazy _data). The
    # plan supplies canonical sorted-key block metadata; ``out_stack`` rows are
    # already in that same canonical order, so it IS the canonical stacked array.
    out_keys = plan.out_block_keys
    out_shapes = plan.out_block_shapes
    out_offsets = plan.out_block_offsets
    total_size = plan.total_size
    out_shape = out_shapes[0]

    out_group = StackGroup(keys=out_keys, array=out_stack)
    out_view = StackedView(
        groups={out_shape: out_group}, indices=tuple(out_indices_ordered)
    )
    return StackedSymmetricTensor.from_stacked(
        view=out_view,
        indices=tuple(out_indices_ordered),
        block_keys=out_keys,
        block_shapes=out_shapes,
        block_offsets=out_offsets,
        total_size=total_size,
        dtype=out_stack.dtype,
    )


def _strict_contract() -> bool:
    """Whether ``TENAX_STRICT_CONTRACT`` arms the #834 cross-representation checks.

    Off by default, and the reason is measured rather than cautious.  Both
    checks below are *structural* -- they read leg charges, flows and block
    keys -- while whether the two representations actually differ depends on the
    blocks' **values**, which are traced.  Scoring every ``_contract_symmetric``
    call of a D=2, chi=8 charged U(1)-Sz sweep against the densified contraction
    of the same operands:

    ==========================================  =====  ==============
    call site                                   calls  max rel. gap
    ==========================================  =====  ==============
    *refused* by the leg-pairing check
      ``_apply_proj_unfused``                     40   1.7e-16
      ``_build_enlarged_corner`` (4 frames)       82   0.0
      ``_ctm_tensor_absorb_*_2plaq`` (4)          16   0.0
    *allowed* by it
      ``_apply_proj_unfused``                     56   **8.3e-01**
    ==========================================  =====  ==============

    Every refusal is a false alarm, and the one genuinely wrong site is allowed.
    A default-on structural guard would therefore break the default CTM path
    while still missing the defect on it.  Read per call rather than cached at
    import, so an audit can toggle it around a single call.

    While armed, this **also pins execution to the reference per-block path**:
    the cuTENSOR, stacked and batched backends each return from
    ``_contract_symmetric`` before the discard check and drop out-of-set output
    keys with their own bare ``continue``, so leaving them enabled would report
    clean on precisely the products never inspected.  A partial audit is worse
    than none -- the contract of a diagnostic is that silence means agreement --
    and since this is a diagnostic, the lost throughput costs nothing that
    matters.  Do not "restore" the accelerated paths under the flag without
    implementing the discard check in each of them; the test
    ``test_strict_mode_is_not_bypassed_by_an_accelerated_backend`` pins this.
    """
    return os.environ.get("TENAX_STRICT_CONTRACT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _leg_charges_on(tensor: SymmetricTensor, axis: int) -> set[int]:
    """Charge values that actually carry a block on ``axis`` of ``tensor``.

    Reads ``_block_keys`` rather than ``.blocks`` so this does not materialise
    the block-array dict, and works unchanged under tracing: block keys are
    static Python metadata, never traced values.
    """
    keys = getattr(tensor, "_block_keys", None)
    if keys is None:
        keys = tuple(tensor.blocks)
    return {int(key[axis]) for key in keys}


def _leg_pairing_fault(
    idx_a: TensorIndex,
    idx_b: TensorIndex,
    live_a: Callable[[], set[int]],
    live_b: Callable[[], set[int]],
) -> str | None:
    """Say why block-sparse and dense would *mis-pair* on this leg pair, or None.

    Block-sparse contraction pairs **charge value q with charge value q**; dense
    einsum pairs **position p with position p**.  Wherever either path pairs
    anything at position ``p``, the two legs must carry the same charge there,
    or the two are contracting different slots and the results differ.  That is
    the mechanism #834 documents and the one this refuses.

    **This is not the only way the two paths can differ, and the other way is
    deliberately not checked here.**  Products whose output key falls outside
    the output legs' valid set are discarded by the per-block loop; dense keeps
    them.  Whether that changes the answer depends on whether those products are
    *numerically* zero, which is a traced value and not visible to any
    structural check.  Both outcomes occur in one CTM sweep at D=2, chi=8 on a
    charged U(1)-Sz state:

    ======================================  ==========  ===============
    site                                    discarded   any non-zero?
    ======================================  ==========  ===============
    ``_build_enlarged_corner`` (6 frames)    ~2000       no -- exact
    ``_apply_proj_unfused``                  560         **192, max 8.15**
    ======================================  ==========  ===============

    So refusing on the structural condition would hard-error the default CTM
    path on contractions measured exact (#852), while permitting it hides real
    weight loss.  ``TENAX_STRICT_CONTRACT=1`` turns the discard itself into an
    error for anyone who wants to audit a path; it is off by default because it
    fires on both rows above.

    ``live_a`` / ``live_b`` are read lazily: on the sanctioned convention this
    returns before either is needed, so the common case costs one array
    comparison and never scans the block keys.  That matters because this runs
    on every contraction, and the eager symmetric path is host-bound (#618).

    Reported rather than raised so the caller can name the legs.
    """
    charges_a = np.asarray(idx_a.charges)
    charges_b = np.asarray(idx_b.charges)

    # Fast path: opposite flows with element-wise equal charges is the
    # convention flip_flow()/bar() produce and the one the block matching
    # implements.  Value-pairing and position-pairing then coincide by
    # construction, whatever blocks the operands carry.
    if (
        idx_a.flow != idx_b.flow
        and charges_a.shape == charges_b.shape
        and np.array_equal(charges_a, charges_b)
    ):
        return None

    if charges_a.shape != charges_b.shape:
        return (
            f"their dimensions differ ({charges_a.size} vs {charges_b.size}), so "
            f"there is no pairing at all"
        )

    populated_a = live_a()
    populated_b = live_b()
    shared = populated_a & populated_b

    # The value-pairing and the position-pairing must select the same slots.
    # A position matters when either path pairs something there: the
    # block-sparse path pairs a shared charge, or the dense path finds both
    # operands populated.
    shared_arr = np.array(sorted(shared), dtype=charges_a.dtype)
    live_a_arr = np.array(sorted(populated_a), dtype=charges_a.dtype)
    live_b_arr = np.array(sorted(populated_b), dtype=charges_b.dtype)
    paired = (
        np.isin(charges_a, shared_arr)
        | np.isin(charges_b, shared_arr)
        | (np.isin(charges_a, live_a_arr) & np.isin(charges_b, live_b_arr))
    )
    offending = np.flatnonzero(paired & (charges_a != charges_b))
    if offending.size:
        p = int(offending[0])
        return (
            f"their charges differ where it matters: slot {p} carries charge "
            f"{int(charges_a[p])} on one leg and {int(charges_b[p])} on the "
            f"other ({offending.size} such slot(s)). Block pairing is by charge "
            f"value and dense pairing is positional, so the two pair different "
            f"slots"
        )

    return None


def _validate_contracted_legs(
    tensors: Sequence[SymmetricTensor], subscripts: str
) -> None:
    """Refuse a contraction whose legs the two representations pair differently.

    **Armed only by ``TENAX_STRICT_CONTRACT``** -- see :func:`_strict_contract`
    for the measurement that made default-on untenable.

    Tenax treats :class:`DenseTensor` and :class:`SymmetricTensor` as
    interchangeable representations of the same object.  On legs whose charge
    arrays do not line up that breaks: measured 22 of 64 flow/charge
    configurations returned a *different* tensor with no error and no warning,
    by 4.2e-01 to 1.5e+00 relative.  ``|sym|/|den|`` ran from 0.458 to 1.514 --
    a result larger than the true one, so blocks were being mis-paired, not
    merely dropped.

    **Refusing is the only available answer, not a conservative choice.**  The
    dense result of such a contraction is generally not a symmetric tensor at
    all.  Worked example (U(1), ``A.k`` OUT ``[-1,0,1]`` against ``B.k`` IN
    ``[1,0,-1]``): dense pairs position 0 of each, charge -1 against charge +1,
    and puts weight on output block ``(i=1, j=-1)``, whose charge under ``i``
    OUT / ``j`` IN is ``-2``.  No relabelling can make a ``SymmetricTensor``
    over those indices carry it.

    The check consults the operands' **blocks**, not just their leg metadata.
    The purely structural condition -- opposite flows and element-wise equal
    charges, what ``bar()``/``flip_flow()`` produce -- refuses 88 of 256 measured
    configurations that are exact; restricting it to populated sectors cuts that
    to 0 while still refusing every one of the 88 that are wrong.

    It is deliberately not :meth:`TensorIndex.is_dual_of`, this tree's other
    duality convention (opposite flow + *negated* charges): negation preserves
    the charge set but permutes the position->charge map, and it admits 28 of
    those 256 configurations that produce wrong answers.

    Raises:
        ValueError: If any contracted leg pair would make the block-sparse and
            dense results differ, and strict mode is on.
    """
    if not _strict_contract():
        return

    groups = subscripts.split("->")[0].split(",")
    if len(groups) != len(tensors):
        return  # malformed subscripts are the caller's problem, not ours

    occurrences: dict[str, list[tuple[int, int]]] = {}
    for t_i, group in enumerate(groups):
        for axis, char in enumerate(group):
            occurrences.setdefault(char, []).append((t_i, axis))

    for char, places in occurrences.items():
        if len(places) != 2:
            continue  # a free leg, or an n-way index this does not model
        (t1, a1), (t2, a2) = places
        if t1 == t2:
            continue  # a trace within one tensor: no pairing between operands

        idx_a = tensors[t1].indices[a1]
        idx_b = tensors[t2].indices[a2]
        fault = _leg_pairing_fault(
            idx_a,
            idx_b,
            lambda t=t1, ax=a1: _leg_charges_on(tensors[t], ax),
            lambda t=t2, ax=a2: _leg_charges_on(tensors[t], ax),
        )
        if fault is None:
            continue

        raise ValueError(
            f"Cannot contract leg {idx_a.label!r} ({idx_a.flow.name}) with leg "
            f"{idx_b.label!r} ({idx_b.flow.name}) as subscript {char!r}: {fault}. "
            f"The block-sparse and dense results would differ, silently (#834). "
            f"Build the second leg with flip_flow() / bar() -- opposite flow, "
            f"identical charges -- which is the convention the block matching "
            f"implements. Note this is stricter than TensorIndex.is_dual_of(), "
            f"which admits negated charges and is unsound here."
        )


def _contract_symmetric(
    tensors: Sequence[SymmetricTensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    optimize: str = "auto",
) -> SymmetricTensor:
    """Contract block-sparse symmetric tensors using charge-indexed matching.

    When ``TENAX_USE_CUTENSOR_BLOCKSPARSE=1`` and a CUDA GPU is available,
    pairwise contractions use cuTENSOR's native block-sparse API (~32μs per
    contraction vs ~1ms Python per-block).

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
    _validate_contracted_legs(tensors, subscripts)

    # An armed audit has to inspect every block product, so it pins execution to
    # the reference per-block path below.  Each accelerated backend returns
    # before the discard check and drops out-of-set output keys with its own
    # bare ``continue``, so leaving them enabled would report clean on exactly
    # the products never inspected -- worse than not offering the flag, since
    # the contract of a diagnostic is that silence means agreement.  The flag is
    # a diagnostic, so giving up the accelerated path while it is on costs
    # nothing that matters.
    strict = _strict_contract()

    # --- Optional cuTENSOR block-sparse GPU path ---
    sym = tensors[0].indices[0].symmetry if tensors and tensors[0].indices else None
    if (
        not strict
        and len(tensors) == 2
        and os.environ.get("TENAX_USE_CUTENSOR_BLOCKSPARSE", "0") == "1"
        and not any(isinstance(t._data, jax.core.Tracer) for t in tensors)
        and (sym is None or not sym.is_fermionic)
    ):
        from tenax.contraction.cutensor_blocksparse import is_available as _ct_ok

        if _ct_ok():
            from tenax.contraction.cutensor_blocksparse import contract_blocksparse

            return contract_blocksparse(
                tensors[0], tensors[1], subscripts, output_indices
            )

    # --- Backend seam: opt-in block-sparse contraction backend (issue #200) ---
    # Build the STATIC plan once (None => out of even-D/2-tensor scope) and let
    # select_backend pick a backend (or None => per-block). Precedence:
    # TENAX_BLOCKSPARSE_BACKEND (stacked/cutensornet/perblock/auto) then the
    # legacy TENAX_STACK_BLOCKSPARSE flag. With NO backend env set, no backend is
    # selected and the per-block path below runs byte-identically to today.
    from tenax.contraction.blocksparse_backend import _backend_opt_in, select_backend

    if not strict and _backend_opt_in():
        from tenax.contraction.blocksparse_plan import build_block_contract_plan

        plan = build_block_contract_plan(tensors, subscripts, output_indices)
        if plan is not None:
            backend = select_backend(tensors, plan)
            if backend is not None:
                stacked_result = _contract_symmetric_stacked(
                    tensors,
                    subscripts,
                    output_indices,
                    optimize,
                    backend=backend,
                    plan=plan,
                )
                if stacked_result is not None:
                    return stacked_result

    # Prelude: parse subscripts + output index order + contracted chars +
    # inferred output target charge + valid output set + canonical contracted
    # positions.  Shared verbatim with the stacked path (issue #566).
    (
        input_subs,
        output_part,
        out_indices_ordered,
        contracted_chars,
        contracted_chars_sorted,
        char_to_canonical_pos,
        output_target,
        valid_output_set,
    ) = _parse_contraction_prelude(tensors, subscripts)

    # Fermionic sign convention (#555): the contractor does NOT auto-apply
    # Koszul signs from leg permutations.  Tenax's auto-tracking (the original
    # PR #13 design) was a *different* convention from TensorKit's fusion-tree
    # braiding and produced incorrect multi-tensor inner products on planar
    # PEPS networks (Tier 7-bound violation in PR #556).  For planar networks
    # — the only kind Tenax's CTM/RDM/energy code uses — no signs are needed:
    # FermionParity's R-symbol contributes only at physical line crossings,
    # which planar diagrams have none of.  For future non-planar applications
    # an explicit ``twist`` primitive can be added.

    # For each tensor, build an index:
    #   partial_sig -> list of (block_key, block_array)
    # where partial_sig is the tuple of this tensor's contracted-leg charges
    # in *canonical order*, restricted to the contracted chars the tensor
    # carries.  Per-tensor partial sigs allow correct multi-input contraction
    # even when some pair of inputs shares no contracted labels (issue #553);
    # the previous full-sig intersection silently dropped to zero whenever
    # two tensors' sig tuples had different lengths.  ``contracted_chars_sorted``
    # / ``char_to_canonical_pos`` come from the shared prelude above.
    tensor_partial_indices: list[
        tuple[list[tuple[int, int]], dict[tuple[int, ...], list[tuple[BlockKey, Any]]]]
    ] = []
    for tensor, subs in zip(tensors, input_subs):
        # For each canonical contracted char that this tensor carries, record
        # ``(canonical_pos, position_in_subs)`` so we can extract the partial
        # sig from a block key.  Sorted by canonical order so partial sigs
        # are comparable across tensors that share the same labels.
        covered: list[tuple[int, int]] = sorted(
            (char_to_canonical_pos[c], pos)
            for pos, c in enumerate(subs)
            if c in contracted_chars
        )

        partial_sig_index: dict[tuple[int, ...], list[tuple[BlockKey, Any]]] = {}
        for key, array in tensor.blocks.items():
            partial_sig = tuple(int(key[pos]) for _, pos in covered)
            partial_sig_index.setdefault(partial_sig, []).append((key, array))
        tensor_partial_indices.append((covered, partial_sig_index))

    # Fast path: if every tensor carries every contracted char, partial sigs
    # equal full sigs and we can iterate via direct set intersection (cheaper
    # than the cartesian product for the common case where the contraction
    # graph is "complete").
    all_complete = all(
        len(covered) == len(contracted_chars_sorted)
        for covered, _ in tensor_partial_indices
    )
    if all_complete and tensor_partial_indices:
        common_sigs: set[tuple[int, ...]] = set(tensor_partial_indices[0][1].keys())
        for _, idx_map in tensor_partial_indices[1:]:
            common_sigs &= set(idx_map.keys())
        sig_iter = iter(common_sigs)
    else:
        # General path: iterate over the cartesian product of allowed charges
        # at each canonical position, then look up each tensor's partial sig.
        # For each canonical contracted position, intersect (over tensors
        # carrying that label) the set of charges they have on that leg.
        # Tensors that don't carry the label contribute no constraint (their
        # blocks tensor-product over it).
        allowed_per_pos: list[set[int]] = []
        for pos_idx in range(len(contracted_chars_sorted)):
            constraint: set[int] | None = None
            for covered, partial_sig_index in tensor_partial_indices:
                local_pos = None
                for ci, (canonical_pos, _tensor_pos) in enumerate(covered):
                    if canonical_pos == pos_idx:
                        local_pos = ci
                        break
                if local_pos is None:
                    continue
                charges_here = {sig[local_pos] for sig in partial_sig_index}
                constraint = (
                    charges_here if constraint is None else constraint & charges_here
                )
            allowed_per_pos.append(constraint if constraint is not None else {0})
        sig_iter = itertools.product(*[sorted(s) for s in allowed_per_pos])

    # --- Gate: opt-in batched block-sparse execution path (issue #568, M-B) ---
    # When TENAX_BATCH_BLOCKSPARSE is truthy, surviving (output_key, arrays)
    # units are grouped by input block-shape signature and contracted with a
    # single batched jnp.einsum + segment_sum per group, instead of one
    # opt_einsum expression per combo.  Default (falsey) keeps the per-combo
    # path byte-identical to before.
    # Allowlist parse (truthy only for explicit on-values) so non-canonical
    # falsey strings like "FALSE"/"no"/"off" are not misread as enabled.
    _batch_blocksparse = os.environ.get(
        "TENAX_BATCH_BLOCKSPARSE", "0"
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if _batch_blocksparse and not strict:
        output_blocks = _contract_symmetric_batched(
            sig_iter,
            tensor_partial_indices,
            input_subs,
            output_part,
            ",".join(input_subs),
            valid_output_set,
        )
        if output_target is not None:
            return SymmetricTensor._from_blocks_unchecked(
                output_blocks, out_indices_ordered
            )
        return SymmetricTensor(output_blocks, out_indices_ordered)

    # Cache for within-block contraction expressions
    block_expr_cache: dict[tuple[tuple[int, ...], ...], Any] = {}

    output_blocks: dict[BlockKey, Any] = {}

    for full_sig in sig_iter:
        # For each tensor, extract its partial sig from full_sig and look up
        # matching blocks. Missing partial sigs (no blocks for that
        # assignment) terminate this iteration early.
        matching_lists: list[list[tuple[BlockKey, Any]]] = []
        skip = False
        for covered, partial_sig_index in tensor_partial_indices:
            partial_sig = tuple(full_sig[canonical_pos] for canonical_pos, _ in covered)
            matching = partial_sig_index.get(partial_sig)
            if not matching:
                skip = True
                break
            matching_lists.append(matching)
        if skip:
            continue

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
                # This product exists in the densified contraction and not here
                # (#834).  Whether that changes the answer depends on the
                # block's *values*, which are traced, so the discard cannot be
                # judged structurally -- see ``_leg_pairing_fault``.  Opt in to
                # make it an error while auditing a path.
                if strict:
                    raise ValueError(
                        f"Discarding the block product {tuple(keys)} -> "
                        f"{output_key} for subscripts {subscripts!r}: it is "
                        f"outside the output legs' valid set, so the "
                        f"block-sparse result omits weight the densified "
                        f"contraction keeps (#834). Set "
                        f"TENAX_STRICT_CONTRACT=0 to allow it; note that a "
                        f"discarded product is harmless when its value happens "
                        f"to be zero, which is why this is opt-in."
                    )
                continue

            # Contract using cached expression or opt_einsum
            block_shapes = tuple(a.shape for a in arrays)
            cache_key = (block_shapes,)
            if cache_key in block_expr_cache:
                expr = block_expr_cache[cache_key]
                result_array = expr(*arrays, backend="jax")
            else:
                expr = opt_einsum.contract_expression(
                    subscripts,
                    *block_shapes,
                    optimize=optimize,
                )
                block_expr_cache[cache_key] = expr
                result_array = expr(*arrays, backend="jax")

            # Accumulate into output block
            if output_key in output_blocks:
                output_blocks[output_key] = output_blocks[output_key] + result_array
            else:
                output_blocks[output_key] = result_array

    if output_target is not None:
        # Non-identity target: bypass conservation validation
        return SymmetricTensor._from_blocks_unchecked(
            output_blocks, out_indices_ordered
        )
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

    For :class:`SymmetricTensor` operands, two legs may be contracted when they
    have **opposite flows and identical charges** -- what :meth:`TensorIndex.
    flip_flow` and :meth:`Tensor.bar` produce.  This is *not*
    :meth:`TensorIndex.is_dual_of` / :meth:`Tensor.dagger`, which negate the
    charges: block pairing is by charge value while dense pairing is
    positional, so mixing the conventions makes this return a
    representation-dependent answer, silently (#834).  Set
    ``TENAX_STRICT_CONTRACT=1`` to raise instead of returning one.

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
