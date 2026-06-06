"""P-B3: validate the GPU CuTensorNetBackend against the VJP seam spine (#200).

This is :mod:`tests.stacked.test_vjp_seam` with ``MockFFIBackend`` swapped for the
real :class:`~tenax.contraction.blocksparse_cutensor.CuTensorNetBackend`. The Mock
proved the SEAM (a hand-written ``custom_vjp`` backward plugs into the contractor);
here we prove the actual GPU kernel honours that seam end to end:

1. **Value**: CuTensorNet == per-block == StackedJaxBackend.
2. **Gradient (crux)**: ``jax.grad`` through CuTensorNet's opaque ``pure_callback``
   forward + hand-written transposed-plan VJP == ``jax.grad`` per-block ==
   ``jax.grad`` StackedJaxBackend.
3. **Opacity**: the hand-written bwd is the ONLY gradient path — ``stop_gradient``
   through the opaque cuTENSOR forward leaves the grad unchanged.
4. **complex128**: the production dtype, dA and dB on two distinct operands.

All comparisons at the fp tier (rtol/atol 1e-12) — never gauge/loosened (this is
contraction, no SVD). GPU-only: skipped unless CUDA + cuTENSOR (nvmath) are present.
The ``pure_callback`` bridge needs a CPU device to stage host inputs, so run with
``JAX_PLATFORMS=cuda,cpu`` (handoff §P-B0).
"""

import jax
import jax.numpy as jnp
import pytest

from tenax.contraction.blocksparse_backend import StackedJaxBackend
from tenax.contraction.blocksparse_cutensor import (
    CuTensorNetBackend,
    cutensor_available,
)
from tenax.contraction.blocksparse_plan import (
    build_block_contract_plan,
    stacked_execute,
)
from tenax.contraction.blocksparse_vjp import backward_contraction
from tenax.contraction.contractor import _labels_to_subscripts, contract
from tests.stacked._harness import assert_tiered, canonical_tensors

pytestmark = pytest.mark.skipif(
    not cutensor_available(),
    reason="cuTENSOR backend unavailable (needs CUDA + nvmath/cuquantum)",
)


def _bra_from(A):
    """Bra = A.bar() with virtual legs relabelled to survive as free output legs."""
    return A.bar().relabels({"u": "U", "d": "D", "l": "L", "r": "R"})


def _stack_of(t):
    view = t.stacked_blocks()
    (group,) = view.groups.values()
    return group.array


def _complexify(A, scale=1 + 0.7j, shift=0.3j):
    """Promote a real tensor's data buffer to complex128 (distinct real/imag)."""
    return type(A)._raw(
        indices=A._indices,
        data=(A._data * scale + shift).astype(jnp.complex128),
        block_keys=A._block_keys,
        block_shapes=A._block_shapes,
        block_offsets=A._block_offsets,
    )


@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_cutensor_value_triple_equivalence(name):
    """VALUE: CuTensorNet == per-block == StackedJaxBackend (fp 1e-12)."""
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
    cut = CuTensorNetBackend().execute(stacks, plan)

    assert plan.out_block_keys == ref._block_keys
    assert_tiered(stacked, cut, tier="fp")
    ref_rows = jnp.stack(
        [ref._get_block(i) for i in range(len(ref._block_keys))], axis=0
    )
    assert_tiered(ref_rows, cut, tier="fp")


@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_cutensor_grad_triple_equivalence(name):
    """CRUX: CuTensorNet custom_vjp grad == per-block == StackedJax grad."""
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

    sA = _stack_of(A0)
    sB = _stack_of(Abar0)

    def loss_stacked(sA, sB):
        out = stacked_execute([sA, sB], plan)
        return jnp.sum(out**2)

    gA_stacked, gB_stacked = jax.grad(loss_stacked, argnums=(0, 1))(sA, sB)

    backend = CuTensorNetBackend()

    def loss_cut(sA, sB):
        out = backend.execute([sA, sB], plan)
        return jnp.sum(out**2)

    gA_cut, gB_cut = jax.grad(loss_cut, argnums=(0, 1))(sA, sB)

    # CuTensorNet (hand-written VJP) == StackedJax (autodiff) on the stacked operands.
    assert_tiered(gA_stacked, gA_cut, tier="fp")
    assert_tiered(gB_stacked, gB_cut, tier="fp")

    # CuTensorNet grad on stacks == per-block grad, mapped through the shared buffer.
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

    g_from_cut = scatter_like(A0, gA_cut) + jnp.conj(scatter_like(Abar0, gB_cut))
    assert_tiered(g_perblock, g_from_cut, tier="fp")


def test_cutensor_opacity_counter():
    """Opacity (a): the grad equals the hand-written ``backward_contraction``.

    Computing the transposed-plan backward independently and matching it against
    ``jax.grad`` proves the gradient is exactly the hand-written backward, not
    autodiff leaking through the opaque cuTENSOR forward.
    """
    A0 = dict(canonical_tensors())["ferm_D2"]
    Abar0 = _bra_from(A0)
    subs, out_indices = _labels_to_subscripts([A0, Abar0])
    plan = build_block_contract_plan([A0, Abar0], subs, out_indices)

    sA = _stack_of(A0)
    sB = _stack_of(Abar0)
    backend = CuTensorNetBackend()

    def loss(sA, sB):
        return jnp.sum(backend.execute([sA, sB], plan) ** 2)

    g = jax.grad(loss, argnums=(0, 1))(sA, sB)

    out = backend.execute([sA, sB], plan)
    gO = 2.0 * out  # cotangent of sum(out**2)
    gA_direct = backward_contraction([sA, sB], gO, plan, 0)
    gB_direct = backward_contraction([sA, sB], gO, plan, 1)
    assert_tiered(g[0], gA_direct, tier="fp")
    assert_tiered(g[1], gB_direct, tier="fp")


def test_cutensor_opacity_stop_gradient_forward():
    """Opacity (b): autodiff CANNOT leak through the opaque cuTENSOR forward.

    Wrapping the cuTENSOR forward in ``stop_gradient`` (a stricter black box) leaves
    the grad UNCHANGED — proof the entire gradient comes from the hand-written
    ``custom_vjp`` backward, never from differentiating the forward kernel.
    """
    A0 = dict(canonical_tensors())["ferm_D2"]
    Abar0 = _bra_from(A0)
    subs, out_indices = _labels_to_subscripts([A0, Abar0])
    plan = build_block_contract_plan([A0, Abar0], subs, out_indices)

    sA = _stack_of(A0)
    sB = _stack_of(Abar0)
    backend = CuTensorNetBackend()

    def loss(sA, sB):
        return jnp.sum(backend.execute([sA, sB], plan) ** 2)

    g = jax.grad(loss, argnums=(0, 1))(sA, sB)

    # Monkeypatch the module-level forward with a stop_gradient-ed variant; the
    # custom_vjp backward is unaffected, the value path is identical.
    import tenax.contraction.blocksparse_cutensor as _m

    saved = _m.cutensor_forward

    def _sg_forward(stacks, p):
        return jax.lax.stop_gradient(saved(stacks, p))

    _m.cutensor_forward = _sg_forward
    try:
        g_sg = jax.grad(loss, argnums=(0, 1))(sA, sB)
    finally:
        _m.cutensor_forward = saved

    assert_tiered(g[0], g_sg[0], tier="fp")
    assert_tiered(g[1], g_sg[1], tier="fp")


def test_cutensor_grad_complex128_triple_equivalence():
    """complex128 crux (production dtype): CuTensorNet == StackedJax == jax.vjp truth.

    Ground truth is ``jax.grad`` of the pure-JAX forward (``stacked_execute``); the
    real-valued ``sum|out|**2`` loss makes the complex grad of a real loss
    well-defined and matches physical energy / non-holomorphic AD. dA and dB are
    checked on two genuinely distinct complex operands.
    """
    A_real = dict(canonical_tensors())["ferm_D2"]
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
        return jnp.sum(jnp.abs(out) ** 2)

    gA_truth, gB_truth = jax.grad(
        lambda a, b: real_loss(stacked_execute([a, b], plan)),
        argnums=(0, 1),
    )(sA, sB)

    backend = CuTensorNetBackend()
    gA_cut, gB_cut = jax.grad(
        lambda a, b: real_loss(backend.execute([a, b], plan)),
        argnums=(0, 1),
    )(sA, sB)

    assert_tiered(gA_truth, gA_cut, tier="fp")
    assert_tiered(gB_truth, gB_cut, tier="fp")


def test_cutensor_two_distinct_operands():
    """dA and dB independently checked on A·B with B NOT sharing A's buffer."""
    A = dict(canonical_tensors())["ferm_D2"]
    B_src = dict(canonical_tensors())["ferm_D2"]
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

    backend = CuTensorNetBackend()

    def loss_cut(sA, sB):
        return jnp.sum(backend.execute([sA, sB], plan) ** 2)

    gA_cut, gB_cut = jax.grad(loss_cut, argnums=(0, 1))(sA, sB)

    assert_tiered(gA_ref, gA_cut, tier="fp")
    assert_tiered(gB_ref, gB_cut, tier="fp")
