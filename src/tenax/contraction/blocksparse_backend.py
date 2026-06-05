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


_TRUTHY = ("1", "true", "yes", "on")


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

    Task B registers a real backend here; until then this returns ``None`` so the
    ``cutensornet`` env value is accepted without error and falls back to the
    per-block path.
    """
    return None
