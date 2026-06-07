"""Block-sparse contraction backend protocol + selection (#200, A2).

This is the second slice of the kernel-agnostic block-sparse backend seam.
Task A1 split the even-D stacked contraction into a STATIC
:class:`~tenax.contraction.blocksparse_plan.BlockContractPlan` plus a pure-JAX
``stacked_execute``. Here we introduce the *backend protocol* and a *selection*
function so the contractor only ever sees the protocol + the plan — never any
particular kernel's internals.

A backend takes the gathered per-operand stacked arrays plus the (static) plan
and returns the *canonical-ordered stacked output array* — the exact same shape
contract as ``stacked_execute`` (``(n_out_blocks, *out_block_shape)``). The
contractor's assembly logic (persist / non-persist / output_target / counters)
runs identically off that returned array regardless of which backend produced
it. A future cuTensorNet backend (Task B) returns the same canonical stacked
array (or a flat buffer the seam reshapes/slices) — it does NOT leak FFI-isms
through the protocol.

The default (no backend env set) is ``None`` (per-block fallback) — byte
identical to today. The legacy ``TENAX_STACK_BLOCKSPARSE`` flag still selects
the pure-JAX :class:`StackedJaxBackend`. cuTensorNet/Pallas backends are NOT
registered here (that is Task B); the ``cutensornet`` env value is *accepted*
(no error) but resolves to ``None`` until a backend is registered.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from tenax.contraction.blocksparse_plan import BlockContractPlan, stacked_execute


@runtime_checkable
class BlockSparseContractBackend(Protocol):
    """Protocol every block-sparse contraction backend implements.

    A backend is selected by :func:`select_backend` from the (static) plan and
    the input tensors, then driven by the contractor:
    ``backend.execute(operand_stacks, plan)``.

    Attributes:
        name: short identifier (matches the ``TENAX_BLOCKSPARSE_BACKEND`` value).
    """

    name: str

    def available(self) -> bool:
        """Whether this backend can run on the current platform/libraries.

        Pure-platform check (CUDA present, FFI library importable, ...) with NO
        reference to a specific contraction. ``False`` => never selected.
        """
        ...

    def supports(self, tensors: Sequence[Any], plan: BlockContractPlan) -> bool:
        """Whether this backend supports THIS contraction (dtype/symmetry/shape).

        The ``plan`` already encodes the even-D / 2-tensor scope (it is ``None``
        otherwise and selection never reaches here), so a pure-JAX backend can
        return ``True`` unconditionally. A hardware backend may reject e.g.
        complex dtypes or block shapes it cannot handle.
        """
        ...

    def execute(self, operand_stacks: Sequence[Any], plan: BlockContractPlan) -> Any:
        """Run the contraction, returning the canonical stacked output array.

        Same shape contract as
        :func:`~tenax.contraction.blocksparse_plan.stacked_execute`:
        ``(n_out_blocks, *out_block_shape)`` in ``plan.out_block_keys`` order, or
        ``None`` for an empty plan. This is backend-neutral: a flat-buffer
        backend reshapes/slices to this layout before returning.
        """
        ...


class StackedJaxBackend:
    """Pure-JAX stacked block-sparse backend (the A1 ``stacked_execute`` path).

    Available on every platform (it is pure ``jnp``) and supports any plan the
    seam hands it (the plan already encodes the even-D / 2-tensor scope). This is
    the backend the legacy ``TENAX_STACK_BLOCKSPARSE`` flag selects.
    """

    name = "stacked"

    def available(self) -> bool:
        return True

    def supports(self, tensors: Sequence[Any], plan: BlockContractPlan) -> bool:
        # The plan already encodes the supported scope (else it is None and we
        # are never called); the pure-JAX kernel handles every such plan.
        return True

    def execute(self, operand_stacks: Sequence[Any], plan: BlockContractPlan) -> Any:
        return stacked_execute(operand_stacks, plan)


class MockFFIBackend:
    """Opaque-FFI stand-in proving the hand-written VJP seam (#200, A3).

    Test/dev-only. Genuinely simulates an opaque ``custom_call`` (cuTensorNet,
    Pallas): its ``execute`` is wrapped in :func:`jax.custom_vjp`, so autodiff
    CANNOT leak through the forward — the ONLY gradient path is the hand-written
    transposed-plan backward
    (:func:`~tenax.contraction.blocksparse_vjp.backward_contraction`). The
    forward kernel happens to reuse ``stacked_execute`` (a real FFI backend would
    call into CUDA here), but its custom-VJP backward does NOT differentiate
    through that — it rebuilds the cotangent via the SAME plan machinery the
    forward used, exactly as a cuTensorNet VJP author must.

    The backward needs the FORWARD ``subscripts`` and per-operand block metadata
    (keys/indices) to rebuild the forward operand(s) — and as of the Phase-B
    seam refinement (#200) those facts ride on the PLAN itself
    (``plan.subscripts`` / ``plan.operand_*``). So this backend captures NOTHING
    at construction: ``execute(operand_stacks, plan)`` + the residual operand
    stacks are all the hand-written backward consumes. The protocol gap flagged
    in A3 review is closed; a real ``CuTensorNetBackend`` slots in the same way.

    Args:
        bwd_calls: optional shared counter ``dict`` whose ``["n"]`` is bumped
            every time the hand-written backward fires (opacity probe).
    """

    name = "mockffi"

    def __init__(self, bwd_calls=None):
        self._bwd_calls = bwd_calls

    def available(self) -> bool:
        return True

    def supports(self, tensors: Sequence[Any], plan: BlockContractPlan) -> bool:
        return True

    def execute(self, operand_stacks: Sequence[Any], plan: BlockContractPlan) -> Any:
        return self._execute_opaque(tuple(operand_stacks), plan)

    def _execute_opaque(self, operand_stacks, plan):
        """``jax.custom_vjp``-wrapped opaque kernel (forward + hand-written bwd).

        ``plan`` is static (closed over / nondiff) and self-contained for the
        backward; only ``operand_stacks`` is differentiated.
        """
        import jax

        @jax.custom_vjp
        def opaque(operand_stacks):
            # FORWARD: an opaque kernel. (Here it reuses stacked_execute, but its
            # gradient is NOT taken through this — see opaque_bwd.)
            return stacked_execute(operand_stacks, plan)

        def opaque_fwd(operand_stacks):
            out = stacked_execute(operand_stacks, plan)
            # Residual carries ONLY the operand stacks the hand-written bwd
            # consumes — no forward intermediates -> autodiff cannot leak through.
            return out, operand_stacks

        def opaque_bwd(operand_stacks_res, out_cotangent_stack):
            from tenax.contraction.blocksparse_vjp import backward_contraction

            grads = tuple(
                backward_contraction(
                    operand_stacks_res,
                    out_cotangent_stack,
                    plan,
                    wrt,
                )
                for wrt in range(len(operand_stacks_res))
            )
            if self._bwd_calls is not None:
                self._bwd_calls["n"] += 1
            return (tuple(grads),)

        opaque.defvjp(opaque_fwd, opaque_bwd)
        return opaque(operand_stacks)


_TRUTHY = ("1", "true", "yes", "on")


def _backend_opt_in() -> bool:
    """True iff a block-sparse backend env opt-in is active (cheap: env reads only, no plan)."""
    choice = os.environ.get("TENAX_BLOCKSPARSE_BACKEND", "").strip().lower()
    if choice in ("stacked", "cutensornet"):
        return True
    if choice == "perblock":
        return False  # explicit force-off: skip plan-building entirely
    # auto / unset / unknown -> defer to legacy flag
    return os.environ.get("TENAX_STACK_BLOCKSPARSE", "0").strip().lower() in _TRUTHY


def _stacked_if_ok(
    tensors: Sequence[Any], plan: BlockContractPlan
) -> BlockSparseContractBackend | None:
    """Return a :class:`StackedJaxBackend` if it is available + supports the plan."""
    backend = StackedJaxBackend()
    if backend.available() and backend.supports(tensors, plan):
        return backend
    return None


def select_backend(
    tensors: Sequence[Any], plan: BlockContractPlan
) -> BlockSparseContractBackend | None:
    """Choose a block-sparse contraction backend, or ``None`` (per-block fallback).

    Precedence:

    1. Explicit ``TENAX_BLOCKSPARSE_BACKEND`` in
       ``{stacked, cutensornet, perblock, auto}``:

       * ``perblock`` -> ``None`` (force the per-block path);
       * ``stacked``  -> :class:`StackedJaxBackend` if available + supports, else
         ``None``;
       * ``cutensornet`` -> the cuTensorNet backend if registered + available +
         supports, else ``None`` (NOT registered yet -> ``None``, no error);
       * ``auto`` / unset -> fall through to step 2.

    2. Back-compat: if ``TENAX_STACK_BLOCKSPARSE`` is truthy -> :class:`StackedJaxBackend`
       (if it supports the plan).

    3. Otherwise ``None`` (per-block default; byte-identical to today).

    A ``None`` plan never reaches here (the contractor only calls this with a
    non-``None`` plan), but selection stays total regardless.
    """
    choice = os.environ.get("TENAX_BLOCKSPARSE_BACKEND", "").strip().lower()

    if choice == "perblock":
        return None
    if choice == "stacked":
        return _stacked_if_ok(tensors, plan)
    if choice == "cutensornet":
        # Accepted but not registered yet (Task B). Resolve gracefully to None.
        return _select_cutensornet(tensors, plan)
    # "auto", "", or any other value -> fall through to the legacy flag.

    legacy = os.environ.get("TENAX_STACK_BLOCKSPARSE", "0").strip().lower()
    if legacy in _TRUTHY:
        return _stacked_if_ok(tensors, plan)

    return None


def _select_cutensornet(
    tensors: Sequence[Any], plan: BlockContractPlan
) -> BlockSparseContractBackend | None:
    """Resolve the cuTensorNet backend, or ``None`` if not registered/usable.

    The GPU :class:`~tenax.contraction.blocksparse_cutensor.CuTensorNetBackend`
    (Phase B) is selected iff it is ``available()`` (CUDA present + cuTENSOR /
    cuQuantum importable) and ``supports`` the plan. On a non-GPU host (or without
    cuQuantum installed) ``available()`` is ``False`` and this returns ``None``, so
    the ``cutensornet`` env value is accepted without error and falls back to the
    per-block path. Imported lazily so non-GPU environments never import cupy.
    """
    from tenax.contraction.blocksparse_cutensor import CuTensorNetBackend

    backend = CuTensorNetBackend()
    if backend.available() and backend.supports(tensors, plan):
        return backend
    return None
