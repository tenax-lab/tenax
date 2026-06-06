"""Triple-equivalence VJP seam tests for an opaque block-sparse backend (#200, A3).

The de-risking crux for cuTensorNet: a real FFI backend is an opaque
``custom_call`` JAX cannot autodiff, so its backward MUST be hand-written. Here a
:class:`MockFFIBackend` (genuine ``jax.custom_vjp``, opacity enforced) is driven
through the SAME plan machinery the forward uses, and we assert:

1. **Value**: MockFFIBackend == per-block == StackedJaxBackend.
2. **Gradient (crux)**: ``jax.grad`` through MockFFIBackend's hand-written
   transposed-plan VJP == ``jax.grad`` per-block == ``jax.grad`` StackedJaxBackend.
3. **Opacity**: the hand-written bwd is the ONLY gradient path (counter probe +
   ``stop_gradient`` through the forward kernel still yields the correct grad).
4. **Two distinct operands**: dA and dB independently checked on an A·B contraction
   with B NOT sharing A's data buffer.

All comparisons are at the fp tier (rtol/atol 1e-12) — never gauge/loosened
(this is contraction, no SVD).
"""

import jax
import jax.numpy as jnp
import pytest

from tenax.contraction.blocksparse_backend import MockFFIBackend, StackedJaxBackend
from tenax.contraction.blocksparse_plan import (
    build_block_contract_plan,
    stacked_execute,
)
from tenax.contraction.blocksparse_vjp import backward_contraction
from tenax.contraction.contractor import _labels_to_subscripts, contract
from tests.stacked._harness import assert_tiered, canonical_tensors


def _bra_from(A):
    """Bra = A.bar() with virtual legs relabelled to survive as free output legs."""
    return A.bar().relabels({"u": "U", "d": "D", "l": "L", "r": "R"})


def _stack_of(t):
    view = t.stacked_blocks()
    (group,) = view.groups.values()
    return group.array


@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_vjp_seam_value_triple_equivalence(name):
    A = dict(canonical_tensors())[name]
    Abar = _bra_from(A)
    subs, out_indices = _labels_to_subscripts([A, Abar])
    plan = build_block_contract_plan([A, Abar], subs, out_indices)
    assert plan is not None

    stacks = [_stack_of(A), _stack_of(Abar)]

    # per-block reference
    ref = contract(A, Abar)
    assert ref._data.size > 1, "double-layer collapsed to a scalar"

    stacked = StackedJaxBackend().execute(stacks, plan)
    mock = MockFFIBackend().execute(stacks, plan)

    # Stacked / mock are canonical-row arrays aligned with plan.out_block_keys.
    assert plan.out_block_keys == ref._block_keys
    assert_tiered(stacked, mock, tier="fp")
    # Compare against the per-block flat buffer reshaped to canonical rows.
    ref_rows = jnp.stack(
        [ref._get_block(i) for i in range(len(ref._block_keys))], axis=0
    )
    assert_tiered(ref_rows, mock, tier="fp")


@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_vjp_seam_grad_triple_equivalence(name):
    """The crux: hand-written transposed-plan VJP == per-block == StackedJax grad."""
    A0 = dict(canonical_tensors())[name]
    Abar0 = _bra_from(A0)
    subs, out_indices = _labels_to_subscripts([A0, Abar0])
    plan = build_block_contract_plan([A0, Abar0], subs, out_indices)
    assert plan is not None

    # --- per-block reference: differentiate through contract() on the data buffer.
    indices, keys, shapes, offsets = (
        A0.indices,
        A0._block_keys,
        A0._block_shapes,
        A0._block_offsets,
    )
    bar_indices, bar_keys, bar_shapes, bar_offsets = (
        Abar0.indices,
        Abar0._block_keys,
        Abar0._block_shapes,
        Abar0._block_offsets,
    )

    def loss_perblock(data):
        A = type(A0)._raw(
            indices=indices,
            data=data,
            block_keys=keys,
            block_shapes=shapes,
            block_offsets=offsets,
        )
        Abar = type(A0)._raw(
            indices=bar_indices,
            data=jnp.conj(data),
            block_keys=bar_keys,
            block_shapes=bar_shapes,
            block_offsets=bar_offsets,
        )
        out = contract(A, Abar)
        return jnp.sum(out._data**2)

    g_perblock = jax.grad(loss_perblock)(A0._data)

    # --- stacked-array losses (StackedJax vs MockFFI) over the operand stacks.
    sA = _stack_of(A0)
    sB = _stack_of(Abar0)

    def loss_stacked(sA, sB):
        out = stacked_execute([sA, sB], plan)
        return jnp.sum(out**2)

    gA_stacked, gB_stacked = jax.grad(loss_stacked, argnums=(0, 1))(sA, sB)

    backend = MockFFIBackend()

    def loss_mock(sA, sB):
        out = backend.execute([sA, sB], plan)
        return jnp.sum(out**2)

    gA_mock, gB_mock = jax.grad(loss_mock, argnums=(0, 1))(sA, sB)

    # Mock (hand-written VJP) == StackedJax (autodiff) on the stacked operands.
    assert_tiered(gA_stacked, gA_mock, tier="fp")
    assert_tiered(gB_stacked, gB_mock, tier="fp")

    # Mock grad on stacks == per-block grad, mapped through the shared data buffer.
    # A and Abar share A0's data (Abar = conj(A); real x64 -> same buffer), so the
    # per-block data-grad is the sum of the ket and bra contributions. Reconstruct
    # the same combination from the stacked grads to compare on the flat buffer.
    def scatter_like(t, gstack):
        view = t.stacked_blocks()
        (group,) = view.groups.values()
        new_view = type(view)(
            groups={
                next(iter(view.groups)): type(group)(keys=group.keys, array=gstack)
            },
            indices=view.indices,
        )
        return t.from_stacked_blocks(new_view)._data

    g_from_mock = scatter_like(A0, gA_mock) + jnp.conj(scatter_like(Abar0, gB_mock))
    assert_tiered(g_perblock, g_from_mock, tier="fp")


def test_vjp_seam_opacity_counter():
    """Opacity (a): the hand-written custom-VJP backward IS the gradient path.

    The counter bumps under ``jax.grad`` (the bwd fired), and the grad equals the
    hand-written ``backward_contraction`` computed independently — i.e. the
    gradient is exactly the transposed-plan backward, not autodiff-through-execute.
    """
    A0 = dict(canonical_tensors())["ferm_D2"]
    Abar0 = _bra_from(A0)
    subs, out_indices = _labels_to_subscripts([A0, Abar0])
    plan = build_block_contract_plan([A0, Abar0], subs, out_indices)

    sA = _stack_of(A0)
    sB = _stack_of(Abar0)

    counter = {"n": 0}
    backend = MockFFIBackend(bwd_calls=counter)

    def loss(sA, sB):
        return jnp.sum(backend.execute([sA, sB], plan) ** 2)

    g = jax.grad(loss, argnums=(0, 1))(sA, sB)
    assert counter["n"] > 0, "hand-written custom-VJP backward never fired"

    out = backend.execute([sA, sB], plan)
    gO = 2.0 * out  # cotangent of sum(out**2)
    gA_direct = backward_contraction([sA, sB], gO, plan, 0)
    gB_direct = backward_contraction([sA, sB], gO, plan, 1)
    assert_tiered(g[0], gA_direct, tier="fp")
    assert_tiered(g[1], gB_direct, tier="fp")


def test_vjp_seam_opacity_stop_gradient_forward():
    """Opacity (b): autodiff CANNOT leak through the opaque forward kernel.

    If the forward's internal ``stacked_execute`` is wrapped in
    ``jax.lax.stop_gradient`` (simulating a truly opaque ``custom_call`` whose
    forward is a black box), the grad is UNCHANGED — proof that the entire
    gradient comes from the hand-written ``custom_vjp`` backward, never from
    differentiating the forward kernel.
    """
    A0 = dict(canonical_tensors())["ferm_D2"]
    Abar0 = _bra_from(A0)
    subs, out_indices = _labels_to_subscripts([A0, Abar0])
    plan = build_block_contract_plan([A0, Abar0], subs, out_indices)

    sA = _stack_of(A0)
    sB = _stack_of(Abar0)

    # Normal opaque backend.
    backend = MockFFIBackend()

    # Opaque backend whose forward kernel is stop_gradient-ed (a stricter black
    # box). custom_vjp's bwd is unaffected; the value path is identical.
    class _StopGradFwdMock(MockFFIBackend):
        def _execute_opaque(self, operand_stacks, plan):
            orig = stacked_execute

            def _sg(stacks, p):
                return jax.lax.stop_gradient(orig(stacks, p))

            import tenax.contraction.blocksparse_backend as _m

            saved = _m.stacked_execute
            _m.stacked_execute = _sg
            try:
                return MockFFIBackend._execute_opaque(self, operand_stacks, plan)
            finally:
                _m.stacked_execute = saved

    backend_sg = _StopGradFwdMock()

    def make_loss(b):
        def loss(sA, sB):
            return jnp.sum(b.execute([sA, sB], plan) ** 2)

        return loss

    g = jax.grad(make_loss(backend), argnums=(0, 1))(sA, sB)
    g_sg = jax.grad(make_loss(backend_sg), argnums=(0, 1))(sA, sB)

    assert_tiered(g[0], g_sg[0], tier="fp")
    assert_tiered(g[1], g_sg[1], tier="fp")


def _complexify(A, scale=1 + 0.7j, shift=0.3j):
    """Promote a real tensor's data buffer to complex128 (distinct real/imag)."""
    return type(A)._raw(
        indices=A._indices,
        data=(A._data * scale + shift).astype(jnp.complex128),
        block_keys=A._block_keys,
        block_shapes=A._block_shapes,
        block_offsets=A._block_offsets,
    )


def test_vjp_seam_grad_complex128_triple_equivalence():
    """complex128 crux: hand-written VJP == StackedJax autodiff == jax.vjp truth.

    The production dtype is complex128. The premise that the backward needs a
    conjugation for complex is WRONG: the VJP of a bilinear contraction
    transposes the UNCONJUGATED surviving operand for both real and complex
    (conjugation enters only via non-holomorphic LEAF ops, outside this seam).

    Ground truth is ``jax.vjp(stacked_execute)`` (NOT the per-block shared-buffer
    reconstruction, whose hand-rolled conj bookkeeping is awkward for complex).
    The loss is the real-valued ``sum|out|**2`` so the complex grad of a real
    loss is well-defined and matches physical energy/AD. dA and dB are checked on
    two genuinely distinct complex operands.
    """
    A_real = dict(canonical_tensors())["ferm_D2"]
    # Distinct complex bra: bar() of a *different*, perturbed tensor so dA and dB
    # are independent, then promote both operands to complex128.
    B_src_real = dict(canonical_tensors())["ferm_D2"]
    B_src_real = type(B_src_real)._raw(
        indices=B_src_real.indices,
        data=B_src_real._data * 1.7 + 0.3,
        block_keys=B_src_real._block_keys,
        block_shapes=B_src_real._block_shapes,
        block_offsets=B_src_real._block_offsets,
    )
    A = _complexify(A_real, scale=1 + 0.7j, shift=0.3j)
    B = _complexify(_bra_from(B_src_real), scale=1 - 0.4j, shift=-0.2j)

    subs, out_indices = _labels_to_subscripts([A, B])
    plan = build_block_contract_plan([A, B], subs, out_indices)
    assert plan is not None

    sA = _stack_of(A)
    sB = _stack_of(B)
    assert sA.dtype == jnp.complex128 and sB.dtype == jnp.complex128

    def real_loss(out):
        # Real-valued: matches physical energy / non-holomorphic AD.
        return jnp.sum(jnp.abs(out) ** 2)

    # --- Ground truth: jax.grad of the pure-JAX forward (stacked_execute),
    # which is jax.vjp(stacked_execute) composed with the real loss. This is the
    # robust truth — no hand-rolled conj bookkeeping.
    gA_truth, gB_truth = jax.grad(
        lambda a, b: real_loss(stacked_execute([a, b], plan)),
        argnums=(0, 1),
    )(sA, sB)

    # --- StackedJaxBackend autodiff.
    def loss_stacked(a, b):
        return real_loss(stacked_execute([a, b], plan))

    gA_stacked, gB_stacked = jax.grad(loss_stacked, argnums=(0, 1))(sA, sB)

    # --- MockFFIBackend hand-written transposed-plan VJP.
    backend = MockFFIBackend()

    def loss_mock(a, b):
        return real_loss(backend.execute([a, b], plan))

    gA_mock, gB_mock = jax.grad(loss_mock, argnums=(0, 1))(sA, sB)

    # Triple match at fp tier for BOTH dA and dB on the production dtype.
    assert_tiered(gA_truth, gA_stacked, tier="fp")
    assert_tiered(gB_truth, gB_stacked, tier="fp")
    assert_tiered(gA_truth, gA_mock, tier="fp")
    assert_tiered(gB_truth, gB_mock, tier="fp")


def test_vjp_seam_two_distinct_operands():
    """dA and dB independently checked on A·B with B NOT sharing A's buffer."""
    A = dict(canonical_tensors())["ferm_D2"]
    # Distinct bra: bar() of a *different* tensor so dA and dB are independent.
    B_src = dict(canonical_tensors())["ferm_D2"]
    # Perturb B's data so it is genuinely distinct from A.
    B_src = type(B_src)._raw(
        indices=B_src.indices,
        data=B_src._data * 1.7 + 0.3,
        block_keys=B_src._block_keys,
        block_shapes=B_src._block_shapes,
        block_offsets=B_src._block_offsets,
    )
    B = _bra_from(B_src)

    subs, out_indices = _labels_to_subscripts([A, B])
    plan = build_block_contract_plan([A, B], subs, out_indices)
    assert plan is not None

    sA = _stack_of(A)
    sB = _stack_of(B)

    def loss_stacked(sA, sB):
        return jnp.sum(stacked_execute([sA, sB], plan) ** 2)

    gA_ref, gB_ref = jax.grad(loss_stacked, argnums=(0, 1))(sA, sB)

    backend = MockFFIBackend()

    def loss_mock(sA, sB):
        return jnp.sum(backend.execute([sA, sB], plan) ** 2)

    gA_mock, gB_mock = jax.grad(loss_mock, argnums=(0, 1))(sA, sB)

    assert_tiered(gA_ref, gA_mock, tier="fp")
    assert_tiered(gB_ref, gB_mock, tier="fp")
