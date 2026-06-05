"""Hand-written transposed-plan backward for the block-sparse seam (#200, A3).

A block-sparse contraction is *multilinear* in its operands, so its VJP w.r.t.
operand ``i`` is itself a block-sparse contraction — of the output cotangent with
the *other* operand(s). For ``einsum("ij,jk->ik", A, B)``::

    dA = einsum("ik,jk->ij", gC, B)
    dB = einsum("ij,ik->jk", A, gC)

This module expresses that backward through the SAME plan machinery the forward
uses (:func:`build_block_contract_plan` + :func:`stacked_execute`), so EVERY
backend reuses it. This is the de-risking crux for opaque FFI backends
(cuTensorNet, Pallas): a ``custom_call`` has no autodiff, so its backward MUST be
hand-written — and here we prove the transposed-plan backward reproduces the
per-block / pure-JAX autodiff gradient at fp tier on CPU, before any GPU exists.

Charge-flow subtlety (the exact thing a cuTensorNet VJP author hits): the output
cotangent ``gC`` shares the forward output's block structure, but its legs carry
the forward *output* flows. In the backward contraction those same legs are
contracted against the surviving forward operand, whose matching legs have the
*same* flow (they were passed through in the forward). For the charge machinery
to pair them as a contraction (opposite flows), the cotangent's indices are
flow-flipped (:meth:`TensorIndex.flip_flow`: opposite flow, identical charge
values) before building the backward plan. ``flip_flow`` is purely about
charge-value matching in the block machinery — it is NOT a conjugation and has
nothing to do with complex dtypes.

No conjugation rationale (read before touching the backward for Phase B):
the VJP of a bilinear contraction transposes the UNCONJUGATED surviving operand
for BOTH real and complex dtypes (this matches ``jax.vjp`` of the same
``einsum`` / ``dot_general``). Conjugation in complex autodiff enters only
through non-holomorphic LEAF operations (``abs``, ``conj``, real-valued loss),
which are outside this contraction seam. Therefore ``backward_contraction``
applies NO conjugation and is correct for complex128 as well as real — do NOT
add a conjugation here. (Proven empirically against ``jax.vjp(stacked_execute)``
ground truth at fp tier for complex128 in ``tests/stacked/test_vjp_seam.py``.)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp

from tenax.contraction.blocksparse_plan import (
    BlockContractPlan,
    build_block_contract_plan,
    stacked_execute,
)
from tenax.core.tensor import SymmetricTensor


def _cotangent_tensor(
    plan: BlockContractPlan, out_cotangent_stack: Any
) -> SymmetricTensor:
    """Wrap the output-cotangent stacked array as a SymmetricTensor.

    Block structure = the forward output's (``plan.out_block_keys`` /
    ``plan.out_indices``), but with every leg flow-flipped so the backward
    contraction pairs the cotangent's legs with the surviving forward operand as
    a genuine contraction (opposite flows, identical charge values). Charge
    VALUES are untouched (``flip_flow``), so block-key matching is by-value as
    the per-block path expects.
    """
    flipped_indices = tuple(idx.flip_flow() for idx in plan.out_indices)
    blocks = {
        plan.out_block_keys[i]: out_cotangent_stack[i]
        for i in range(len(plan.out_block_keys))
    }
    return SymmetricTensor._from_blocks_unchecked(blocks, flipped_indices)


def _backward_subscripts(subscripts: str, wrt: int) -> str:
    """Backward einsum subscripts for the VJP w.r.t. operand ``wrt``.

    Forward ``subA,subB->subO`` -> backward (wrt=0) ``subO,subB->subA``;
    (wrt=1) ``subA,subO->subB``. The operand order in the returned string is
    ``(other_operand, cotangent)`` for ``wrt==0`` and ``(operand, cotangent)``
    matching the caller's operand assembly in :func:`backward_contraction`.
    """
    inputs, output = subscripts.split("->")
    in_subs = inputs.split(",")
    if len(in_subs) != 2:
        raise ValueError(
            f"transposed-plan backward supports 2-operand forward only; got "
            f"{len(in_subs)} operands in {subscripts!r}"
        )
    sub_a, sub_b = in_subs
    # The surviving operand is contracted UNCONJUGATED (real and complex alike;
    # matches jax.vjp of this einsum). Conjugation in complex autodiff comes only
    # from non-holomorphic leaf ops, which live outside this seam.
    if wrt == 0:
        # dA = einsum(subO, subB -> subA): cotangent first, surviving operand B.
        return f"{output},{sub_b}->{sub_a}"
    if wrt == 1:
        # dB = einsum(subA, subO -> subB): surviving operand A, cotangent.
        return f"{sub_a},{output}->{sub_b}"
    raise ValueError(f"wrt must be 0 or 1, got {wrt}")


def backward_contraction(
    subscripts: str,
    fwd_operands: Sequence[SymmetricTensor],
    out_cotangent_stack: Any,
    fwd_plan: BlockContractPlan,
    wrt: int,
) -> Any:
    """Cotangent stack for forward operand ``wrt`` via the transposed plan.

    Reuses :func:`build_block_contract_plan` + :func:`stacked_execute` on the
    backward contraction (cotangent paired with the other forward operand) —
    NOT a bespoke charge matcher — so the same machinery serves every backend.

    Args:
        subscripts: the FORWARD einsum subscripts ``subA,subB->subO``.
        fwd_operands: the two forward operand SymmetricTensors ``(A, B)``.
        out_cotangent_stack: cotangent for the forward output, in the SAME
            canonical stacked layout the forward backend returned
            (``(n_out_blocks, *out_block_shape)``, rows aligned with
            ``fwd_plan.out_block_keys``).
        fwd_plan: the forward :class:`BlockContractPlan`.
        wrt: which forward operand to differentiate w.r.t. (0 or 1).

    Returns:
        The cotangent stacked array for operand ``wrt``, row-aligned with that
        operand's ``stacked_blocks()`` group order (so it can be scattered back
        into the operand's flat ``_data`` buffer), or ``None`` if the backward
        contraction has no surviving blocks (zero cotangent).
    """
    if len(fwd_operands) != 2:
        raise ValueError("transposed-plan backward supports 2-operand forward only")

    cot = _cotangent_tensor(fwd_plan, out_cotangent_stack)
    other = fwd_operands[1 - wrt]
    bwd_subs = _backward_subscripts(subscripts, wrt)

    # Operand order MUST match bwd_subs: wrt==0 -> (cot, other); wrt==1 -> (other, cot).
    if wrt == 0:
        bwd_operands: tuple[SymmetricTensor, SymmetricTensor] = (cot, other)
    else:
        bwd_operands = (other, cot)

    target = fwd_operands[wrt]
    bwd_plan = build_block_contract_plan(bwd_operands, bwd_subs, target.indices)
    if bwd_plan is None:
        raise RuntimeError(
            "transposed-plan backward fell out of the even-D/2-tensor scope; "
            "the forward was in scope so this should not happen"
        )

    operand_stacks = []
    for t in bwd_operands:
        view = t.stacked_blocks()
        (group,) = view.groups.values()
        operand_stacks.append(group.array)

    grad_stack = stacked_execute(operand_stacks, bwd_plan)

    # Re-key the backward output (in bwd_plan.out_block_keys order) onto the
    # target operand's stacked-blocks row order, zero-filling any block the
    # backward contraction did not produce (those have zero cotangent).
    return _align_to_operand(grad_stack, bwd_plan, target)


def _align_to_operand(
    grad_stack: Any, bwd_plan: BlockContractPlan, target: SymmetricTensor
) -> Any:
    """Reorder/zero-fill ``grad_stack`` to the target operand's stacked layout.

    ``stacked_execute`` returns rows in ``bwd_plan.out_block_keys`` order; the
    target operand's ``stacked_blocks()`` group lists its blocks in its own
    sorted-key order. We build a row per target block, taking the matching
    backward-output row where present and zeros otherwise.
    """
    view = target.stacked_blocks()
    (tgt_group,) = view.groups.values()
    tgt_keys = tgt_group.keys
    tgt_array = tgt_group.array  # (n_tgt_blocks, *block_shape)

    if grad_stack is None:
        return jnp.zeros_like(tgt_array)

    bwd_key_to_row = {k: i for i, k in enumerate(bwd_plan.out_block_keys)}
    rows = []
    zero_block = jnp.zeros(tgt_array.shape[1:], dtype=tgt_array.dtype)
    for k in tgt_keys:
        r = bwd_key_to_row.get(k)
        if r is None:
            rows.append(zero_block)
        else:
            rows.append(grad_stack[r].astype(tgt_array.dtype))
    return jnp.stack(rows, axis=0)
