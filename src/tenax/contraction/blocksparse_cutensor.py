"""cuTENSOR forward block-sparse backend (#200, Phase B).

The GPU sibling of :class:`~tenax.contraction.blocksparse_backend.StackedJaxBackend`.
It honours the SAME seam contract — ``execute(operand_stacks, plan)`` returning the
canonical-ordered stacked output array — but performs the one expensive step (the
batched block contraction) on the GPU through cuTENSOR instead of an XLA-compiled
``jnp.einsum``.

Structure mirrors ``stacked_execute`` exactly (gather rows -> batched contraction
-> ``segment_sum`` accumulation -> canonical reorder); the ONLY substitution is the
batched contraction, routed through ``nvmath.tensor.binary_contraction`` (explicit
cuTENSOR pairwise contract) inside a single ``jax.pure_callback`` so it is ONE
opaque XLA op. This is handoff §4 build-shape (b): drive the per-``PlanGroup``
batched contraction via the library and ``segment_sum``-accumulate.

NB: this is NOT the legacy ``cutensornet_backend.py`` / ``cutensor_blocksparse.py``
(eager, full-tensor ``custom_vjp``, non-seam-conformant — kept only as API
reference). This module is seam-conformant: ``(operand_stacks, plan) -> canonical
stacked array``.

P-B1 (this slice) is the FORWARD kernel only — value-validated against
:class:`StackedJaxBackend` at fp 1e-12, real + complex128. The ``jax.custom_vjp``
wrap + hand-written ``backward_contraction`` reuse, plus registration in
``_select_cutensornet``, land in P-B2.

Default OFF: selected only via ``TENAX_BLOCKSPARSE_BACKEND=cutensornet`` AND
``available()`` (CUDA present + cuTENSOR importable); never auto-selected, so the
default path stays byte-identical.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

from tenax.contraction.blocksparse_plan import BlockContractPlan

# Lazy availability cache (the platform probe is mildly expensive; do it once).
_available: bool | None = None


def cutensor_available() -> bool:
    """True iff a GPU is present AND the cuTENSOR (nvmath) binding is importable.

    Pure-platform check (no reference to a specific contraction). ``False`` => the
    backend is never selected and the seam falls back to the per-block path.
    """
    global _available
    if _available is not None:
        return _available
    try:
        import jax

        if not any(d.platform == "gpu" for d in jax.devices()):
            _available = False
            return False
        import cupy  # noqa: F401
        import nvmath.tensor  # noqa: F401

        _available = True
    except Exception:
        # ImportError (no cuquantum/cupy), RuntimeError (no CUDA), AttributeError.
        _available = False
    return _available


def _batched_contract_host(subscripts: str, a_h, b_h):
    """Eager cuTENSOR batched binary contraction (host -> device -> host).

    ``pure_callback`` delivers host numpy; upload to the GPU, run the real
    cuTENSOR pairwise contraction (``nvmath.tensor.binary_contraction``), return
    host numpy. The host round-trip is the callback bridge's cost (a true FFI
    handler would stay on-device) — for the forward VALUE it is irrelevant, and
    for the #200 compile-wall measurement what matters is that the whole
    contraction is ONE opaque op, which the ``pure_callback`` guarantees.
    """
    import cupy as cp
    import nvmath

    a_d = cp.asarray(a_h)
    b_d = cp.asarray(b_h)
    out_d = nvmath.tensor.binary_contraction(subscripts, a_d, b_d)
    return cp.asnumpy(out_d)


def _cutensor_batched_einsum(subscripts: str, a, b, out_shape_dtype):
    """JAX-visible batched contraction: one opaque ``pure_callback`` to cuTENSOR."""
    import jax

    host = partial(_batched_contract_host, subscripts)
    return jax.pure_callback(host, out_shape_dtype, a, b)


def cutensor_forward(operand_stacks: Sequence[Any], plan: BlockContractPlan) -> Any:
    """Forward block-sparse contraction returning the canonical stacked output.

    Same shape contract as
    :func:`~tenax.contraction.blocksparse_plan.stacked_execute`:
    ``(n_out_blocks, *out_block_shape)`` in ``plan.out_block_keys`` order, or
    ``None`` for an empty plan. The batched contraction runs on the GPU via
    cuTENSOR; the gather / ``segment_sum`` / reorder stay in JAX (each already a
    single block-count-independent op).
    """
    import jax
    import jax.numpy as jnp

    if not plan.groups:
        return None

    # Even-D scope: exactly one group, exactly two operands (binary contraction).
    (group,) = plan.groups
    n_pos = len(operand_stacks)
    if n_pos != 2:
        # binary_contraction is pairwise; the even-D plan is always 2-operand, but
        # guard rather than mis-call cuTENSOR with the wrong arity.
        raise ValueError(f"cuTensorNetBackend forward expects 2 operands, got {n_pos}")

    # Gather each operand's needed rows from its stacked group array in ONE op.
    gathered = [
        jnp.take(operand_stacks[pos], jnp.asarray(group.operand_rows[pos]), axis=0)
        for pos in range(n_pos)
    ]

    # ONE batched cuTENSOR contraction over the combo batch axis. Output shape is
    # static: (n_combos, *out_block_shape), n_combos = number of survivor combos.
    n_combos = len(group.segment_ids)
    out_block_shape = plan.out_block_shapes[0]
    out_dtype = jnp.result_type(gathered[0].dtype, gathered[1].dtype)
    out_sd = jax.ShapeDtypeStruct((n_combos, *out_block_shape), out_dtype)
    batched_result = _cutensor_batched_einsum(
        plan.batched_subscripts, gathered[0], gathered[1], out_sd
    )

    # Collapse combos sharing an output_key, then reorder to canonical layout.
    segments = jnp.asarray(group.segment_ids, dtype=jnp.int32)
    summed = jax.ops.segment_sum(
        batched_result, segments, num_segments=group.num_segments
    )
    canon_perm = jnp.asarray(group.canonical_perm, dtype=jnp.int32)
    return jnp.take(summed, canon_perm, axis=0)


class CuTensorNetBackend:
    """GPU cuTENSOR forward block-sparse backend (#200, Phase B).

    Implements the
    :class:`~tenax.contraction.blocksparse_backend.BlockSparseContractBackend`
    protocol. ``execute`` returns the canonical stacked output array — the exact
    shape contract :class:`StackedJaxBackend` honours — so the contractor's
    assembly logic runs identically regardless of which backend produced it.

    The cuTENSOR forward is opaque to XLA (a ``pure_callback``), so autodiff
    CANNOT trace through it; the ONLY gradient path is the hand-written
    transposed-plan backward
    (:func:`~tenax.contraction.blocksparse_vjp.backward_contraction`), wired via
    ``jax.custom_vjp`` exactly as a real FFI backend must. The backward is itself
    a block-sparse contraction expressed through the SAME plan machinery (it reads
    ``plan.subscripts`` / ``plan.operand_*`` and rebuilds operands), so the
    backend hands it nothing but ``(operand_stacks, plan)``.
    """

    name = "cutensornet"

    def available(self) -> bool:
        return cutensor_available()

    def supports(self, tensors: Sequence[Any], plan: BlockContractPlan) -> bool:
        # The plan already encodes the even-D / 2-tensor single-shape-group scope
        # (else it is None and selection never reaches here); the binary cuTENSOR
        # contraction handles every such plan in real or complex128.
        return True

    def execute(self, operand_stacks: Sequence[Any], plan: BlockContractPlan) -> Any:
        operand_stacks = tuple(operand_stacks)
        # Empty plan: no surviving blocks -> nothing to differentiate. Skip the
        # custom_vjp wrap (cutensor_forward returns None, which the contractor
        # assembles into an empty tensor).
        if not plan.groups:
            return None
        return self._execute_opaque(operand_stacks, plan)

    def _execute_opaque(self, operand_stacks, plan):
        """``jax.custom_vjp``-wrapped opaque cuTENSOR kernel (forward + hand bwd).

        Structurally identical to
        :meth:`~tenax.contraction.blocksparse_backend.MockFFIBackend._execute_opaque`,
        but the forward is the real GPU cuTENSOR contraction
        (:func:`cutensor_forward`). ``plan`` is static (closed over / nondiff) and
        self-contained for the backward; only ``operand_stacks`` is differentiated.
        """
        import jax

        @jax.custom_vjp
        def opaque(operand_stacks):
            # FORWARD: the opaque cuTENSOR kernel. Its gradient is NOT taken
            # through this (see opaque_bwd) — autodiff stops at the pure_callback.
            return cutensor_forward(operand_stacks, plan)

        def opaque_fwd(operand_stacks):
            out = cutensor_forward(operand_stacks, plan)
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
            return (tuple(grads),)

        opaque.defvjp(opaque_fwd, opaque_bwd)
        return opaque(operand_stacks)
