"""P-B1 forward-kernel validation: cuTENSOR block-sparse == StackedJax (#200).

P-B0 proved the JAX<->GPU contraction bridge in isolation. P-B1 wires a real
even-D double-layer block-sparse plan through the cuTENSOR forward kernel
(``CuTensorNetBackend.execute``, handoff build-shape (b): gather -> batched
cuTENSOR contraction via ``nvmath.tensor.binary_contraction`` in one
``pure_callback`` -> ``segment_sum`` -> canonical reorder) and asserts the
canonical stacked output VALUE matches the pure-JAX ``StackedJaxBackend`` oracle
at fp tier (1e-12), for **real float64 AND complex128**, on ferm_D2 and ferm_D4.

This is the value gate before P-B2 (custom_vjp + hand-written backward) and the
P-B4 compile-wall measurement. It is forward-only and undifferentiated on purpose.

Run:
  JAX_PLATFORMS=cuda,cpu uv run python examples/probe_200_cutensor_forward.py
(Tenax forces jax_enable_x64=True on import; cuda,cpu is needed so pure_callback
has a CPU device to stage host inputs.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Tenax import forces x64; do it before any array work.
from tenax.algorithms.fermionic_ipeps import FPEPSConfig, _build_initial_fpeps_tensor
from tenax.contraction.blocksparse_backend import StackedJaxBackend
from tenax.contraction.blocksparse_cutensor import CuTensorNetBackend, cutensor_available
from tenax.contraction.blocksparse_plan import build_block_contract_plan
from tenax.contraction.contractor import _labels_to_subscripts

FP_TOL = 1e-12


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


def _max_abs_diff(ref, got):
    ref = jnp.asarray(ref)
    got = jnp.asarray(got)
    assert ref.shape == got.shape, f"shape {ref.shape} != {got.shape}"
    return float(jnp.max(jnp.abs(ref - got)))


def _check_case(name: str, D: int, complex_: bool) -> bool:
    A = _build_initial_fpeps_tensor(FPEPSConfig(D=D), jax.random.PRNGKey(0))
    Abar = _bra_from(A)
    if complex_:
        A = _complexify(A, scale=1 + 0.7j, shift=0.3j)
        Abar = _complexify(Abar, scale=1 - 0.4j, shift=-0.2j)

    subs, out_indices = _labels_to_subscripts([A, Abar])
    plan = build_block_contract_plan([A, Abar], subs, out_indices)
    assert plan is not None, f"{name}: plan is None (out of even-D scope?)"

    stacks = [_stack_of(A), _stack_of(Abar)]
    dtype = stacks[0].dtype

    # Oracle (pure JAX) vs cuTENSOR forward, both jitted (the production path).
    # The plan is static metadata — close over it; jit only over the stacks.
    stacked_backend = StackedJaxBackend()
    cutensor_backend = CuTensorNetBackend()
    oracle = jax.jit(lambda s: stacked_backend.execute(s, plan))(stacks)
    got = jax.jit(lambda s: cutensor_backend.execute(s, plan))(stacks)

    err = _max_abs_diff(oracle, got)
    value_ok = err < FP_TOL and bool(jnp.all(jnp.isfinite(got)))

    # OP-COUNT (compile-collapse premise): the jaxpr must be O(#groups) callback
    # ops, NOT O(#blocks) structural ops. On a REAL block-sparse contraction with
    # n_blocks output blocks, the equation count must stay a small handful,
    # independent of block count, with #callbacks == #groups.
    n_blocks = len(plan.out_block_keys)
    n_groups = len(plan.groups)
    jaxpr = jax.make_jaxpr(lambda s: cutensor_backend.execute(s, plan))(stacks).jaxpr
    n_eqns = len(jaxpr.eqns)
    n_callbacks = sum(
        "callback" in str(e.primitive) or "custom_call" in str(e.primitive)
        for e in jaxpr.eqns
    )
    # A handful of structural ops (gather/segment_sum/reorder per group) + one
    # callback per group; emphatically << n_blocks. Bound generously but well
    # below block count (128) to prove the collapse.
    opcount_ok = n_callbacks == n_groups and n_eqns < 20 and n_eqns < n_blocks

    passed = value_ok and opcount_ok
    print(
        f"  {name:9s} D={D} {str(dtype):11s} "
        f"out_blocks={n_blocks:3d} shape={tuple(got.shape)}  "
        f"max|Δ|={err:.2e}  eqns={n_eqns:2d}(cb={n_callbacks}/grp={n_groups})  "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def main() -> int:
    print(f"devices: {jax.devices()}")
    print(f"x64: {jax.config.read('jax_enable_x64')}  "
          f"cutensor_available: {cutensor_available()}")
    if not cutensor_available():
        print("\nP-B1: SKIP — cuTENSOR backend not available on this platform.")
        return 2

    ok = True
    print("\ncuTENSOR forward vs StackedJax oracle (fp 1e-12) + op-count (O(#groups)):")
    for D in (2, 4):
        ok &= _check_case("ferm_real", D, complex_=False)
        ok &= _check_case("ferm_c128", D, complex_=True)

    print(f"\nP-B1 forward kernel: "
          f"{'PASS — cuTENSOR == StackedJax (real + c128)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
