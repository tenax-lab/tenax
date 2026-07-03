r"""Gate: does the REAL reduced-corner 1×1 CTM move achieve ~1/N per-device peak
memory when GSPMD-sharded?  (#570 payoff-2, follow-up to probe_qr_shard_reopen_570.)

!!! WARNING — this script's RELIEF number is NOT trustworthy. !!!
It uses ``memory_analysis().temp_size_in_bytes`` and returns a single leaf, so XLA's
dead-code elimination prunes everything except the one sharded absorption and reports a
spurious ~4× relief. The FAITHFUL measurement (real ``peak_bytes_in_use`` high-water,
full env live) is ``examples/bench_1x1_shard_highwater.py`` and shows the full 1×1
SWEEP does NOT shard (~1×, NO-GO). See
``docs/superpowers/handoffs/2026-07-02-570-reduced-corner-shard-gate.md``. Kept only as
a record of the methodology error.

Background
----------
The #632 multi-GPU NO-GO was measured on the **dense 2×2** path (full χD²×χD² SVD),
which XLA all-gathers -> replicates the dominant intermediate -> ~1.2× relief.
`probe_qr_shard_reopen_570.py` showed (HLO) that the reduced-corner decomposition
only touches a D²-smaller operand.  This gate tests the **real** move end-to-end:

    _ctm_tensor_move_left(env, env, a, chi, projector_method="svd")

with the double-layer's surviving D² leg sharded (as the production 2×2 path does via
`constrain_double_layer_for_move`, which the 1×1 sweep branch does NOT yet call).

The dense 1×1 SVD projector (`_ctm_projector.py:978-1018`) is:
    M = C1g^H @ C4g        # (χ,χ) -- REDUCTION over the sharded χD² axis (all-reduce)
    U,S,Vh = svd(M)        # tiny χ×χ, replicated
    P = C4g @ (V S^-1/2)   # (χD²,χ) -- stays sharded on χD²
so in principle the whole move stays sharded with only a ~χ² all-reduce.  The RISK:
the fuse/reembed ops (merging a replicated χ axis with a sharded D² axis into χD²)
may force an all-gather that re-replicates the χ²D⁶ absorption.  This gate measures
which happens.

GATE: reduced-corner 1×1 SVD move per-device peak should drop ~N× (recover the #632
rung-1 no-SVD profile).  GO if relief >> the dense-2×2 ~1.2× baseline.

Run
---
    # compile-only per-device peak estimate (cheap; D=8/10), fake devices:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run python \
        examples/gate_reduced_corner_shard_570.py --D 8 --chi 16

    # real 2-GPU high-water validation (NCCL 4-way deadlocks on this box):
    CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 uv run python \
        examples/gate_reduced_corner_shard_570.py --D 8 --chi 16 --exec
"""

from __future__ import annotations

import argparse
import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_moves import _ctm_tensor_move_left
from tenax.algorithms.ctm_sharding import (
    _AXIS,
    build_ctm_mesh,
    commit_env,
    constrain_double_layer_for_move,
    double_layer_move_partition_spec,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

_COLL = ("all-gather", "all-reduce", "all-to-all", "reduce-scatter", "collective-permute")
_SHAPE_RE = re.compile(r"(?:f64|f32|c128|c64)\[[\d,]+\]")


def _build_A(D, d=2, seed=42):
    rng = np.random.RandomState(seed)
    data = jnp.asarray(rng.standard_normal((D, D, D, D, d)))
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    ch = np.zeros(D, dtype=np.int32)
    pch = np.zeros(d, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pch.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, idx)


def _randomize(tree, seed=0):
    """Replace every array leaf with a normalized random normal of the same shape,
    so corners/edges are full-rank (realistic memory), keeping index structure."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    out = []
    for i, x in enumerate(leaves):
        r = np.random.RandomState(seed + i).standard_normal(x.shape)
        r = jnp.asarray(r) / (float(np.linalg.norm(r)) + 1e-10)
        out.append(r.astype(x.dtype))
    return jax.tree_util.tree_unflatten(treedef, out)


def _bytes(tok):
    dt, dims = tok.split("[")
    n = 1
    for x in dims.rstrip("]").split(","):
        n *= int(x)
    return n * {"f64": 8, "f32": 4, "c128": 16, "c64": 8}[dt]


def scan_collectives(hlo):
    found = []
    for line in hlo.splitlines():
        for op in _COLL:
            if f"{op}(" in line or f"{op}-start(" in line or f"{op}-done" in line:
                shapes = _SHAPE_RE.findall(line)
                if shapes:
                    biggest = max(shapes, key=_bytes)
                    found.append((op, biggest, _bytes(biggest)))
                # matched this line's collective; stop so it isn't double-counted
                # under a later op. The break MUST stay inside the `if` — at the
                # `for op` level it would only ever test the first op (all-gather)
                # and miss all-reduce/all-to-all/reduce-scatter/collective-permute.
                break
    return found


def _mem(exe):
    try:
        ma = exe.memory_analysis()
        temp = getattr(ma, "temp_size_in_bytes", 0)
        out = getattr(ma, "output_size_in_bytes", 0)
        arg = getattr(ma, "argument_size_in_bytes", 0)
        return temp, out, arg
    except Exception as e:  # noqa: BLE001
        return -1, -1, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--chi", type=int, default=16)
    ap.add_argument("--method", default="svd", choices=["svd", "qr"])
    ap.add_argument("--exec", action="store_true", help="real multi-GPU high-water run")
    ap.add_argument("--grad", action="store_true", help="also gate value_and_grad (backward)")
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    D, chi = args.D, args.chi
    n = jax.device_count()
    mesh = build_ctm_mesh()

    A = _build_A(D)
    env0 = _randomize(initialize_ctm_tensor_env(A, chi), seed=1)
    a0 = _randomize(_build_double_layer_tensor(A), seed=500)

    print(f"# gate_reduced_corner_shard  devices={n}  D={D} D2={D*D}  chi={chi}  "
          f"method={args.method}")
    print(f"#   dominant absorption χ²D⁶ = {chi*chi*D**6} elems "
          f"= {chi*chi*D**6*8/1e9:.3f} GB (f64, single-device)")
    print(f"#   reduced corner χD²×χ     = {chi*D*D*chi} elems "
          f"= {chi*D*D*chi*8/1e6:.3f} MB")

    def move(env, a, shard):
        if shard:
            a = constrain_double_layer_for_move(a, "left", mesh)
        new_env, _eps = _ctm_tensor_move_left(env, env, a, chi, args.method)
        # return the renormalized edge (biggest sharded output) as a raw array
        leaves = jax.tree_util.tree_leaves(new_env.T4)
        return leaves[0]

    # ---- replicated (1-device-equivalent) baseline on the same mesh ----
    repl = NamedSharding(mesh, PartitionSpec())  # fully replicated
    env_r = jax.device_put(env0, repl)
    a_r = jax.device_put(a0, repl)
    exe_1 = jax.jit(lambda e, a: move(e, a, False)).lower(env_r, a_r).compile()
    t1, o1, g1 = _mem(exe_1)

    # ---- sharded: edges D²-sharded, corners replicated; a surviving-leg sharded ----
    env_s = commit_env(env0, mesh)
    a_s = jax.device_put(a0, NamedSharding(mesh, double_layer_move_partition_spec("left")))
    exe_n = jax.jit(lambda e, a: move(e, a, True)).lower(env_s, a_s).compile()
    tn, on, gn = _mem(exe_n)

    colls = scan_collectives(exe_n.as_text())
    maxb = max((b for _, _, b in colls), default=0)

    def gb(x):
        return x / 1e9 if x >= 0 else x

    relief = (t1 / tn) if (tn and tn > 0) else float("nan")
    print(f"\n  per-device TEMP peak (memory_analysis):")
    print(f"     replicated (1-dev-equiv) = {gb(t1):.4f} GB")
    print(f"     sharded  ({n}-dev)        = {gb(tn):.4f} GB")
    print(f"     RELIEF = {relief:.2f}x   (ideal {n}x; dense-2×2 baseline ~1.2x)")
    print(f"  output+arg (repl / shard): {gb(o1):.3f}/{gb(g1):.3f}  |  "
          f"{gb(on):.3f}/{gb(gn):.3f} GB")

    print(f"\n  collectives in the SHARDED move ({len(colls)}); "
          f"biggest operand {maxb/1e6:.3f} MB "
          f"[χ²D⁶={chi*chi*D**6*8/1e6:.1f}MB would = re-replication]:")
    seen = set()
    for op, shp, b in colls:
        if (op, shp) in seen:
            continue
        seen.add((op, shp))
        print(f"       {op:20s} {shp:26s} {b/1e6:.3f} MB")
        if len(seen) >= 10:
            break

    if args.grad:
        # Backward gate: value_and_grad through the move (traces the projector
        # backward -> tracer SVD path).  Loss = sum(T4_new**2).
        def loss(a, env, shard):
            return jnp.sum(move(env, a, shard) ** 2)

        vg = jax.value_and_grad(loss, argnums=0)
        exe_g1 = jax.jit(lambda a, e: vg(a, e, False)).lower(a_r, env_r).compile()
        exe_gn = jax.jit(lambda a, e: vg(a, e, True)).lower(a_s, env_s).compile()
        tg1, _, _ = _mem(exe_g1)
        tgn, _, _ = _mem(exe_gn)
        gcolls = scan_collectives(exe_gn.as_text())
        gmax = max((b for _, _, b in gcolls), default=0)
        grelief = (tg1 / tgn) if (tgn and tgn > 0) else float("nan")
        print(f"\n  BACKWARD (value_and_grad) per-device TEMP peak:")
        print(f"     replicated = {gb(tg1):.4f} GB   sharded({n}) = {gb(tgn):.4f} GB   "
              f"RELIEF = {grelief:.2f}x")
        print(f"     biggest collective in backward = {gmax/1e6:.3f} MB "
              f"[χ²D⁶={chi*chi*D**6*8/1e6:.1f}MB = re-replication]")

    if args.exec and n > 1:
        # real high-water validation + parity
        jax.devices()  # ensure init
        out_r = jax.jit(lambda e, a: move(e, a, False))(env_r, a_r)
        out_s = jax.jit(lambda e, a: move(e, a, True))(env_s, a_s)
        err = float(jnp.max(jnp.abs(jax.device_get(out_r) - jax.device_get(out_s))))
        try:
            hw = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
        except Exception:
            hw = float("nan")
        print(f"\n  [--exec] parity |repl - shard|max = {err:.2e}  "
              f"(cumulative high-water dev0 = {hw:.3f} GB)")


if __name__ == "__main__":
    main()
