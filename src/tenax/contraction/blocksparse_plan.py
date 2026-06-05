"""Backend-agnostic block-sparse contraction plan + pure-JAX execution (#200, A1).

This module is the first slice of the kernel-agnostic block-sparse backend seam.
It splits the even-D stacked contraction (formerly inline in
``_contract_symmetric_stacked``) into:

* :func:`build_block_contract_plan` — STATIC: everything computable from block
  metadata (``_block_keys`` / ``_block_shapes``) and the subscript string alone,
  with NO tensor data. Charge matching, valid-output filtering, block-combo
  grouping, per-operand row selection, segment/accumulation structure, batched
  subscripts, and output block keys/shapes/offsets all live here. Returns
  :class:`BlockContractPlan`, or ``None`` when the contraction is out of scope
  (so the caller falls back to the per-block path).

* :func:`stacked_execute` — DATA: given each operand's stacked array(s) (leading
  block axis), runs the batched ``jnp.einsum`` + ``jax.ops.segment_sum`` per
  group, then reorders to canonical sorted-key layout. Pure ``jnp`` so it is
  naturally differentiable.

This is a PURE REFACTOR of the prior inline logic; behavior is byte-identical.
The backend protocol / dispatch (Task A2) and the VJP seam (Task A3) are NOT
introduced here.
"""

from __future__ import annotations

import itertools
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from tenax.core.index import TensorIndex
from tenax.core.tensor import BlockKey, SymmetricTensor


@dataclass(frozen=True)
class PlanGroup:
    """Per input-shape-group execution structure (static).

    For the even-D single-shape-group scope there is exactly one ``PlanGroup``.

    Attributes:
        operand_rows: per operand, the stacked-row indices (into that operand's
            stacked group array) feeding the batched einsum, in survivor order.
        segment_ids: output-key accumulation segment id per survivor (for
            ``segment_sum`` over the batch axis).
        num_segments: number of distinct output keys in this group.
        out_shape: output block shape produced by this group's einsum.
    """

    operand_rows: tuple[tuple[int, ...], ...]
    segment_ids: tuple[int, ...]
    num_segments: int
    out_shape: tuple[int, ...]
    # Canonical sorted-key row -> segment (first-seen) row. Reorders the
    # segment_sum output into ``BlockContractPlan.out_block_keys`` order.
    canonical_perm: tuple[int, ...]


@dataclass(frozen=True)
class BlockContractPlan:
    """Backend-agnostic plan for a block-sparse contraction (static metadata only).

    Attributes:
        out_indices: output ``TensorIndex`` tuple (subscript output order).
        out_block_keys: output block keys in canonical sorted order.
        out_block_shapes: output block shapes (parallel to ``out_block_keys``).
        out_block_offsets: flat-buffer offsets for the canonical layout.
        total_size: total flat-buffer element count for the output.
        batched_subscripts: einsum subscripts with a fresh batch label prepended.
        groups: per input-shape-group :class:`PlanGroup` tuple.
        output_target: inferred output target charge (``int``) or ``None``;
            drives ``_from_blocks_unchecked`` vs ``SymmetricTensor`` assembly.
        dtype: output dtype (from operands), or ``None`` when there are no
            surviving blocks.
    """

    out_indices: tuple
    out_block_keys: tuple
    out_block_shapes: tuple
    out_block_offsets: tuple
    total_size: int
    batched_subscripts: str
    groups: tuple
    output_target: object
    dtype: object


def build_block_contract_plan(
    tensors: Sequence[SymmetricTensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
) -> BlockContractPlan | None:
    """Build the static block-contraction plan, or ``None`` if unsupported.

    Backend-agnostic: touches only block metadata (``_block_keys`` /
    ``_block_shapes``) and the subscript string — NO tensor data. Returns
    ``None`` (caller falls back to the per-block path) when the contraction is
    out of the even-D scope:

      * not exactly 2 tensors;
      * any tensor is multi-shape-group (``len(set(t._block_shapes)) > 1``);
      * any tensor has no blocks.
    """
    # Lazy import to avoid a module-level cycle (contractor imports this module).
    from tenax.contraction.contractor import _parse_contraction_prelude

    # YAGNI scope: even-D, 2-tensor, single-shape-group only.
    if len(tensors) != 2:
        return None
    for t in tensors:
        if len(set(t._block_shapes)) > 1:
            return None
        if not t._block_keys:
            return None

    # --- Prelude: shared with the per-block path (parse + targets + valid set).
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

    # --- Static per-tensor row indexing from _block_keys (no .blocks / data).
    tensor_covered: list[list[tuple[int, int]]] = []
    tensor_partial_rows: list[dict[tuple[int, ...], list[int]]] = []
    tensor_keys: list[tuple[BlockKey, ...]] = []
    operand_dtype = None

    for tensor, subs in zip(tensors, input_subs):
        # Single shape-group guaranteed by the scope check; read the static keys
        # straight off the tensor metadata (sorted-key order matches the stacked
        # group array the caller will gather).
        keys = tuple(tensor._block_keys)
        tensor_keys.append(keys)
        operand_dtype = tensor.dtype

        covered: list[tuple[int, int]] = sorted(
            (char_to_canonical_pos[c], pos)
            for pos, c in enumerate(subs)
            if c in contracted_chars
        )
        tensor_covered.append(covered)

        partial_rows: dict[tuple[int, ...], list[int]] = {}
        for row, key in enumerate(keys):
            partial_sig = tuple(int(key[pos]) for _, pos in covered)
            partial_rows.setdefault(partial_sig, []).append(row)
        tensor_partial_rows.append(partial_rows)

    # --- Enumerate surviving combos via the all_complete intersection.
    all_complete = all(
        len(cov) == len(contracted_chars_sorted) for cov in tensor_covered
    )
    assert all_complete, "2-tensor scope guarantees every char is contracted in both"
    common_sigs: set[tuple[int, ...]] = set(tensor_partial_rows[0].keys())
    for partial_rows in tensor_partial_rows[1:]:
        common_sigs &= set(partial_rows.keys())

    n_pos = len(input_subs)

    survivors: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for full_sig in common_sigs:
        matching_rows: list[list[int]] = []
        skip = False
        for covered, partial_rows in zip(tensor_covered, tensor_partial_rows):
            partial_sig = tuple(full_sig[canonical_pos] for canonical_pos, _ in covered)
            rows = partial_rows.get(partial_sig)
            if not rows:
                skip = True
                break
            matching_rows.append(rows)
        if skip:
            continue

        for combo_rows in itertools.product(*matching_rows):
            keys = [tensor_keys[i][combo_rows[i]] for i in range(n_pos)]

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

            survivors.append((output_key, tuple(combo_rows)))

    # Batched subscripts: prepend a fresh batch label to every operand + output.
    used_chars = set("".join(input_subs)) | set(output_part)
    batch_char = None
    for c in string.ascii_letters:
        if c not in used_chars:
            batch_char = c
            break
    if batch_char is None:  # pragma: no cover - 52-label ceiling guards this
        raise ValueError("No free einsum label available for batch axis.")
    batched_inputs = ",".join(batch_char + s for s in input_subs)
    batched_subscripts = batched_inputs + "->" + batch_char + output_part

    # No surviving blocks: empty plan (caller assembles an empty tensor).
    if not survivors:
        return BlockContractPlan(
            out_indices=tuple(out_indices_ordered),
            out_block_keys=(),
            out_block_shapes=(),
            out_block_offsets=(),
            total_size=0,
            batched_subscripts=batched_subscripts,
            groups=(),
            output_target=output_target,
            dtype=None,
        )

    # Single shape-group on both sides -> exactly one input shape-sig group.
    # Stable per-group ordering of distinct output_keys -> segment ids.
    distinct_keys: list[tuple[int, ...]] = []
    key_to_seg: dict[tuple[int, ...], int] = {}
    seg_ids: list[int] = []
    per_op_rows: list[list[int]] = [[] for _ in range(n_pos)]
    for okey, combo_rows in survivors:
        seg = key_to_seg.get(okey)
        if seg is None:
            seg = len(distinct_keys)
            key_to_seg[okey] = seg
            distinct_keys.append(okey)
        seg_ids.append(seg)
        for pos in range(n_pos):
            per_op_rows[pos].append(combo_rows[pos])

    # Output block shape (single shape-group: all output blocks share it).
    # Statically determined by the einsum subscripts + the single input block
    # shapes — no tensor data needed.
    in_block_shapes = [t._block_shapes[0] for t in tensors]
    out_shape = _einsum_output_shape(batched_subscripts, in_block_shapes)

    # Canonical sorted-key layout for output block metadata.
    sorted_keys = sorted(distinct_keys)
    out_keys = tuple(sorted_keys)
    out_shapes = tuple(out_shape for _ in out_keys)
    block_size = 1
    for d in out_shape:
        block_size *= d
    out_offsets = tuple(i * block_size for i in range(len(out_keys)))
    total_size = block_size * len(out_keys)

    # Canonical sorted-key row -> segment (first-seen) row. ``seg`` is the
    # first-seen rank of each distinct key, so ``key_to_seg`` is exactly the
    # inverse layout map the original code built via ``seg_of_key``.
    canonical_perm = tuple(key_to_seg[k] for k in sorted_keys)

    group = PlanGroup(
        operand_rows=tuple(tuple(rows) for rows in per_op_rows),
        segment_ids=tuple(seg_ids),
        num_segments=len(distinct_keys),
        out_shape=tuple(out_shape),
        canonical_perm=canonical_perm,
    )

    return BlockContractPlan(
        out_indices=tuple(out_indices_ordered),
        out_block_keys=out_keys,
        out_block_shapes=out_shapes,
        out_block_offsets=out_offsets,
        total_size=total_size,
        batched_subscripts=batched_subscripts,
        groups=(group,),
        output_target=output_target,
        dtype=operand_dtype,
    )


def _einsum_output_shape(
    batched_subscripts: str, in_block_shapes: Sequence[tuple[int, ...]]
) -> tuple[int, ...]:
    """STATIC output block shape for the batched einsum (drops the batch axis).

    ``batched_subscripts`` has a fresh batch label prepended to every operand and
    the output. The per-block (non-batch) operand shapes are
    ``in_block_shapes``; we size each output character from the operand char that
    carries it.
    """
    input_part, output_part = batched_subscripts.split("->")
    operand_subs = input_part.split(",")
    # operand_subs[i][0] is the batch char; the rest map to in_block_shapes[i].
    char_to_dim: dict[str, int] = {}
    for subs, shape in zip(operand_subs, in_block_shapes):
        body = subs[1:]  # strip batch char
        for char, dim in zip(body, shape):
            char_to_dim[char] = dim
    # output_part[0] is the batch char; the remainder is the per-block out shape.
    return tuple(char_to_dim[c] for c in output_part[1:])


def stacked_execute(
    operand_stacks: Sequence[Any], plan: BlockContractPlan
) -> dict[BlockKey, Any]:
    """Data-level execution shared by the pure-JAX path.

    Given each operand's stacked group array (leading block axis, in the same
    sorted-key row order the plan's ``operand_rows`` index into), run the batched
    einsum + ``segment_sum`` per group, then reorder rows to canonical
    sorted-key layout. Returns ``{out_block_key: array}`` keyed in canonical
    sorted order. Pure ``jnp`` -> differentiable.

    For an empty plan (no surviving blocks) returns ``{}``.
    """
    if not plan.groups:
        return {}

    # Even-D scope: exactly one group.
    (group,) = plan.groups
    n_pos = len(operand_stacks)

    # Gather each operand's needed rows from its stacked group array in ONE op
    # (jnp.take, axis=0) — no per-block slicing.
    gathered = [
        jnp.take(operand_stacks[pos], jnp.asarray(group.operand_rows[pos]), axis=0)
        for pos in range(n_pos)
    ]

    # ONE batched einsum over the combo batch axis.
    batched_result = jnp.einsum(plan.batched_subscripts, *gathered)

    # Collapse combos sharing an output_key via segment_sum over the batch axis.
    segments = jnp.asarray(group.segment_ids, dtype=jnp.int32)
    summed = jax.ops.segment_sum(
        batched_result, segments, num_segments=group.num_segments
    )

    # ``summed`` rows are in survivor/first-seen (segment) order; reorder ONCE to
    # canonical sorted-key layout so it matches ``plan.out_block_keys``.
    canon_perm = jnp.asarray(group.canonical_perm, dtype=jnp.int32)
    out_stack = jnp.take(summed, canon_perm, axis=0)

    return {key: out_stack[i] for i, key in enumerate(plan.out_block_keys)}
