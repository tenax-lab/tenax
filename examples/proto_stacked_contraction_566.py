#!/usr/bin/env python3
"""#566 narrow prototype: per-block-loop vs stacked-batched block-sparse einsum.

Phase-0 + op-histogram (issue #566) localized the symmetric-iPEPS-AD compile
wall to per-block STRUCTURAL op emission in the contraction (reshape/broadcast/
slice + the per-combo einsum), NOT the contraction math itself, and showed the
flat `_data` buffer (#87) is intentional accelerator-substrate groundwork
(#200/#568/#569) that must NOT be reverted. The lever is therefore to express a
block-sparse contraction as O(1) JAX primitives over the flat buffer instead of
O(n_combos) Python-traced per-block ops -- via a stacked/vmapped einsum
(even-D, uniform shapes) or a single custom kernel (general).

This prototype validates that op-count collapse on the REAL contraction inside
the double-layer build: ``contract(A, A.bar())`` over the physical leg, for a
FermionParity iPEPS site tensor. It compares, by jaxpr primitive count:

  * LOOP   -- one einsum per surviving (ket-block, bra-block) combo (the current
              `_contract_symmetric` pattern), operands supplied as separate
              arrays;
  * STACK  -- operands stacked along a combo-batch axis, ONE batched einsum per
              operand-shape group (vmap-equivalent), then the per-combo outputs.

Both are validated numerically against the production ``contract(A, A.bar())``
block-for-block. We report op counts at D=2 (even -> all blocks one shape ->
single stacked einsum) and D=3 (odd -> distinct shapes -> stack fragments into
groups), so the even/odd payoff is explicit.

Usage::

    JAX_PLATFORMS=cpu uv run python examples/proto_stacked_contraction_566.py --D 2 3
"""

from __future__ import annotations

import argparse
import collections

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.fermionic_ipeps import (  # noqa: E402
    FPEPSConfig,
    _build_initial_fpeps_tensor,
)
from tenax.contraction.contractor import contract  # noqa: E402

# physical leg is the last (axis 4) of both A (u,d,l,r,phys) and A.bar().
# contract over phys -> 8-leg block keyed (u,d,l,r,U,D,L,R); einsum:
_KET = "udlrp"
_BRA = "UDLRp"
_OUT = "udlrUDLR"
_EINSUM = f"{_KET},{_BRA}->{_OUT}"


def _as_jaxpr(v):
    if hasattr(v, "eqns"):
        return v
    inner = getattr(v, "jaxpr", None)
    return inner if (inner is not None and hasattr(inner, "eqns")) else None


def count_primitives(jaxpr) -> collections.Counter:
    counts: collections.Counter = collections.Counter()

    def walk(jx):
        for eqn in jx.eqns:
            counts[eqn.primitive.name] += 1
            for val in eqn.params.values():
                sub = _as_jaxpr(val)
                if sub is not None:
                    walk(sub)
                elif isinstance(val, (tuple, list)):
                    for it in val:
                        s = _as_jaxpr(it)
                        if s is not None:
                            walk(s)

    walk(_as_jaxpr(jaxpr) or jaxpr.jaxpr)
    return counts


def _layout(T):
    """(flat _data, offsets, shapes, keys) for a SymmetricTensor."""
    return T._data, T._block_offsets, T._block_shapes, T._block_keys


def _slice_block(data, offset, shape):
    """Mirror SymmetricTensor._get_block: slice from flat buffer + reshape."""
    size = int(np.prod(shape)) if shape else 1
    return jax.lax.dynamic_slice(data, (offset,), (size,)).reshape(shape)


def enumerate_combos(A):
    """Surviving combos of contract(A, A.bar()) over phys + validation refs.

    Returns (info, ref_blocks) where info has the flat-buffer layouts and the
    combo index lists (combo -> ket/bra block index), so both paths can source
    operands from the same flat ``_data`` buffers (the real representation).
    """
    A_bar = A.bar().relabels({"u": "U", "d": "D", "l": "L", "r": "R"})
    kd, koff, kshp, kkeys = _layout(A)
    bd, boff, bshp, bkeys = _layout(A_bar)

    ket_idx, bra_idx, out_keys = [], [], []
    for ki, kk in enumerate(kkeys):
        for bi, bk in enumerate(bkeys):
            if kk[-1] != bk[-1]:  # physical charge must match (contracted leg)
                continue
            ket_idx.append(ki)
            bra_idx.append(bi)
            out_keys.append(tuple(kk[:4]) + tuple(bk[:4]))

    ref = contract(A, A_bar)
    ref_labels = ref.labels()
    perm = [ref_labels.index(lbl) for lbl in _OUT]
    ref_blocks = {}
    for k, arr in ref.blocks.items():
        key_out = tuple(k[ref_labels.index(lbl)] for lbl in _OUT)
        ref_blocks[key_out] = jnp.transpose(arr, perm)

    info = dict(
        kd=kd,
        koff=koff,
        kshp=kshp,
        bd=bd,
        boff=boff,
        bshp=bshp,
        ket_idx=ket_idx,
        bra_idx=bra_idx,
        out_keys=out_keys,
    )
    return info, ref_blocks


def loop_contract(kd, bd, info):
    """Current pattern: per combo, slice both blocks from the flat buffers
    (the real `_get_block` slice+reshape) and einsum.  O(n_combos) ops."""
    out = []
    for ki, bi in zip(info["ket_idx"], info["bra_idx"]):
        a = _slice_block(kd, info["koff"][ki], info["kshp"][ki])
        b = _slice_block(bd, info["boff"][bi], info["bshp"][bi])
        out.append(jnp.einsum(_EINSUM, a, b))
    return out


def stack_contract_uniform(kd, bd, info):
    """Even-D stacked pattern: ONE reshape of the flat buffer to a block-stack,
    ONE gather of the combo operands, ONE batched einsum.  O(1) ops.

    Requires all ket blocks one shape and all bra blocks one shape (even D).
    This is exactly what a flat-buffer stacked/vmapped contraction emits -- the
    reshape-to-stack is free *because* the blocks are contiguous in `_data`.
    """
    kshp0 = info["kshp"][0]
    bshp0 = info["bshp"][0]
    nk = len(info["kshp"])
    nb = len(info["bshp"])
    ket_stack = kd.reshape((nk, *kshp0))  # 1 reshape, no per-block slicing
    bra_stack = bd.reshape((nb, *bshp0))
    ks = jnp.take(ket_stack, jnp.asarray(info["ket_idx"]), axis=0)  # 1 gather
    bs = jnp.take(bra_stack, jnp.asarray(info["bra_idx"]), axis=0)
    res = jnp.einsum(f"n{_KET},n{_BRA}->n{_OUT}", ks, bs)  # 1 batched einsum
    return res  # (n_combos, u,d,l,r,U,D,L,R) -- output STAYS stacked


def run(D: int) -> None:
    A = _build_initial_fpeps_tensor(
        FPEPSConfig(D=D, t=1.0, V=0.0), jax.random.PRNGKey(42)
    )
    info, ref_blocks = enumerate_combos(A)
    kd, bd = info["kd"], info["bd"]
    n = len(info["ket_idx"])
    uniform = len(set(info["kshp"])) == 1 and len(set(info["bshp"])) == 1

    # ---- numerical validation: LOOP (+ STACK if uniform) == production ----
    loop_out = loop_contract(kd, bd, info)
    max_err = 0.0
    for ok, lo in zip(info["out_keys"], loop_out):
        ref = ref_blocks.get(ok)
        if ref is not None:
            max_err = max(max_err, float(jnp.max(jnp.abs(lo - ref))))
    if uniform:
        stk = stack_contract_uniform(kd, bd, info)
        for i, ok in enumerate(info["out_keys"]):
            ref = ref_blocks.get(ok)
            if ref is not None:
                max_err = max(max_err, float(jnp.max(jnp.abs(stk[i] - ref))))

    # ---- op counts (jaxpr primitives), both sourced from the flat _data and
    #      both PRODUCING THE OUTPUT FLAT BUFFER (apples-to-apples): LOOP must
    #      concatenate its N block ravels; STACK ravels its one batched result.
    def loop_to_buffer(k, b):
        return jnp.concatenate([o.reshape(-1) for o in loop_contract(k, b, info)])

    cl = count_primitives(jax.make_jaxpr(loop_to_buffer)(kd, bd))
    tl = sum(cl.values())

    print(f"== D={D}  (contract(A, A.bar()) over phys) ==")
    print(f"  surviving combos = {n}   blocks uniform-shape: {uniform}")
    print(f"  numerical match vs production contract: max_err={max_err:.2e}")
    print("  LOOP  (per-combo slice+reshape+einsum from flat buffer):")
    print(
        f"    jaxpr ops={tl:>6}  dot_general={cl.get('dot_general', 0):>4} "
        f"dynamic_slice={cl.get('dynamic_slice', 0):>4} reshape={cl.get('reshape', 0):>4}"
    )
    if uniform:
        cs = count_primitives(
            jax.make_jaxpr(lambda k, b: stack_contract_uniform(k, b, info).reshape(-1))(
                kd, bd
            )
        )
        ts = sum(cs.values())
        print("  STACK (1 reshape + 1 gather + 1 batched einsum from flat buffer):")
        print(
            f"    jaxpr ops={ts:>6}  dot_general={cs.get('dot_general', 0):>4} "
            f"reshape={cs.get('reshape', 0):>4} gather={cs.get('gather', 0):>4} "
            f"broadcast={cs.get('broadcast_in_dim', 0):>4}"
        )
        print(f"  >>> op reduction LOOP/STACK = {tl / max(ts, 1):.1f}x")
    else:
        ngroups = len({(s, t) for s, t in zip(info["kshp"], info["bshp"])})
        print(
            f"  STACK: blocks have DISTINCT shapes (odd D) -> cannot reshape the\n"
            f"        flat buffer to one stack; would fragment into per-shape\n"
            f"        groups (>= {ngroups} einsums). Same-shape stacking buys little here;\n"
            f"        general fix is a single custom kernel (lever c)."
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, nargs="+", default=[2, 3])
    args = ap.parse_args()
    print(
        "# #566 stacked-vs-loop block-sparse contraction prototype "
        f"| x64={jax.config.read('jax_enable_x64')}"
    )
    print("# unit = contract(A, A.bar()) over phys (the double-layer's heavy step)\n")
    for D in args.D:
        run(D)
    print(
        "Reading: STACK collapses LOOP's per-combo einsums+scaffolding to one "
        "batched einsum PER OPERAND-SHAPE GROUP.\n"
        "  even D -> 1 group -> full collapse; odd D -> many groups -> partial.\n"
        "  This is what a flat-buffer stacked/vmapped contraction (lever b) or a\n"
        "  single custom kernel (lever c) would emit -- O(groups) vs O(combos)."
    )


if __name__ == "__main__":
    main()
