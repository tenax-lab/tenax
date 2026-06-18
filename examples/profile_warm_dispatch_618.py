#!/usr/bin/env python3
"""#618 step 2: attribute the U(1)-Sz CTM-AD WARM-step wall (host dispatch vs device).

Step 1 (issue #618) showed ``TENAX_BATCH_BLOCKSPARSE=1`` helps the #566 *compile*
wall ~21% but HURTS the *warm* step ~14% -> the warm wall is NOT in the (batched)
decompositions.  This script attributes ONE warm ``value_and_grad`` step to decide
WHERE the ~24x u1sz/dense warm gap lives:

  (A) HOST / eager per-block dispatch  -- Python orchestration of the forward
      python-loop CTM steps: pytree flatten/unflatten over O(n_blocks) leaves,
      per-block SymmetricTensor assembly/disassembly, argument handling.  A
      Python-level batching of same-shape blocks would shrink this.
  (B) DEVICE  -- the COMPILED graph is a huge pile of tiny ops (block-pack
      slice/scatter/reshape + per-block dot_general); warm time is the GPU
      executing ~30k tiny kernels.  Only a graph-level change (vmap/batched
      einsum over the block axis -> fewer, bigger kernels) helps; Python
      batching does nothing.

Decisive split
--------------
* cold-compile vg(A) once (warmup, block_until_ready).
* **dispatch-vs-sync:** ``t_dispatch`` = wall to RETURN from ``vg(A)`` (all jit
  units issued async, not synced); ``t_sync`` = extra wall for
  ``block_until_ready``.  Large ``t_dispatch`` => HOST/dispatch bound (A);
  small ``t_dispatch`` with large ``t_sync`` => DEVICE bound (B).
* **pipelining probe:** K calls blocking only once at the end vs K x single
  blocked.  If batched << K x single, the host runs ahead of the device
  (device-bound); if ~=, each call fully serializes (host- or saturation-bound).
* **cProfile** one warm call -> top funcs by tottime (host CPU) and cumtime.
  Heavy *tottime* in tenax Python (contraction / block iteration / pytree) =>
  HOST; cumtime concentrated in a ``block_until_ready`` / xla-execute C-frame
  with thin tenax tottime => DEVICE.

Usage
-----
    uv run python examples/profile_warm_dispatch_618.py \
        --sym dense u1sz --D 3 --chi 12 --depth 8 --reps 5 \
        --json profile_warm_dispatch_618.json

Set ``--gate`` to run under ``TENAX_BATCH_BLOCKSPARSE=1`` (step-1 arm).
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import io
import json
import os
import pathlib
import platform
import pstats
import statistics
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Reuse make_site_and_gate / build_loss from the shared #566 profiler so site
# construction + the production loss dispatcher are a single source of truth.
_prof_spec = importlib.util.spec_from_file_location(
    "profile_ctm_ad_wall_566",
    pathlib.Path(__file__).parent / "profile_ctm_ad_wall_566.py",
)
_PROF = importlib.util.module_from_spec(_prof_spec)
_prof_spec.loader.exec_module(_PROF)

FLAG = "TENAX_BATCH_BLOCKSPARSE"


def _median(xs):
    return statistics.median(xs) if xs else float("nan")


def profile_sym(sym, D, chi, depth, reps, *, pipeline_k=4):
    """Cold-compile then attribute one warm value_and_grad step for `sym`."""
    A, gate = _PROF.make_site_and_gate(sym, D, seed=42)
    n_blocks = getattr(A, "n_blocks", 1)
    loss_fn = _PROF.build_loss(gate, chi, depth, explicit=False, warmup=3)
    vg = jax.value_and_grad(loss_fn)

    # --- cold compile (warmup) -------------------------------------------- #
    t0 = time.perf_counter()
    out = vg(A)
    jax.block_until_ready(out)
    compile_wall = time.perf_counter() - t0
    E0 = float(out[0])

    # --- uninstrumented warm wall (median of reps blocked calls) ---------- #
    warm = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = vg(A)
        jax.block_until_ready(out)
        warm.append(time.perf_counter() - t0)

    # --- dispatch-vs-sync split ------------------------------------------- #
    disp, sync = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = vg(A)  # returns once all jit units are dispatched (async)
        t1 = time.perf_counter()
        jax.block_until_ready(out)
        t2 = time.perf_counter()
        disp.append(t1 - t0)
        sync.append(t2 - t1)

    # --- pipelining probe ------------------------------------------------- #
    # K calls, block only at the very end: if the host can run ahead of the
    # device, wall/K < single blocked warm (device-bound).
    t0 = time.perf_counter()
    last = None
    for _ in range(pipeline_k):
        last = vg(A)
    jax.block_until_ready(last)
    pipe_wall = time.perf_counter() - t0

    # --- cProfile one warm call ------------------------------------------- #
    pr = cProfile.Profile()
    pr.enable()
    out = vg(A)
    jax.block_until_ready(out)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(30)
    tottime_dump = s.getvalue()

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("cumulative")
    ps2.print_stats(30)
    cumtime_dump = s2.getvalue()

    # Programmatic extraction of the largest tottime frames for the JSON.
    top_tottime = []
    stats_dict = ps.stats  # {(file,line,name): (cc, nc, tt, ct, callers)}
    rows = sorted(stats_dict.items(), key=lambda kv: kv[1][2], reverse=True)
    for (fn_file, fn_line, fn_name), (cc, nc, tt, ct, _callers) in rows[:25]:
        short = f"{pathlib.Path(fn_file).name}:{fn_line}({fn_name})"
        top_tottime.append({"func": short, "ncalls": nc, "tottime": tt, "cumtime": ct})

    warm_med = _median(warm)
    disp_med = _median(disp)
    sync_med = _median(sync)

    result = {
        "sym": sym,
        "D": D,
        "chi": chi,
        "depth": depth,
        "n_blocks": int(n_blocks),
        "gate": os.environ.get(FLAG, "0"),
        "energy": E0,
        "compile_wall_s": compile_wall,
        "warm_med_s": warm_med,
        "warm_all_s": warm,
        "dispatch_med_s": disp_med,
        "sync_med_s": sync_med,
        "dispatch_frac": (disp_med / warm_med) if warm_med else float("nan"),
        "pipeline_k": pipeline_k,
        "pipeline_wall_s": pipe_wall,
        "pipeline_per_call_s": pipe_wall / pipeline_k,
        "pipeline_speedup_vs_serial": (warm_med * pipeline_k / pipe_wall)
        if pipe_wall
        else float("nan"),
        "top_tottime": top_tottime,
    }
    return result, tottime_dump, cumtime_dump


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sym", nargs="+", default=["dense", "u1sz"])
    ap.add_argument("--D", type=int, default=3)
    ap.add_argument("--chi", type=int, default=12)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pipeline-k", type=int, default=4)
    ap.add_argument("--gate", action="store_true", help="Run under TENAX_BATCH_BLOCKSPARSE=1.")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    os.environ[FLAG] = "1" if args.gate else "0"

    dev = jax.devices()[0]
    print("=" * 92)
    print("# #618 step-2: warm-step host-vs-device attribution")
    print(f"# platform : {dev.platform}  device0={dev.device_kind}")
    print(f"# host     : {platform.platform()}")
    print(f"# x64={jax.config.read('jax_enable_x64')}  {FLAG}={os.environ[FLAG]}")
    print(f"# D={args.D} chi={args.chi} depth={args.depth} reps={args.reps}")
    print("=" * 92)

    meta = {
        "platform": dev.platform,
        "device_kind": dev.device_kind,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "gate": os.environ[FLAG],
        "D": args.D,
        "chi": args.chi,
        "depth": args.depth,
        "reps": args.reps,
    }
    rows = []
    for sym in args.sym:
        print(f"\n### {sym} (D={args.D} chi={args.chi}) ###", flush=True)
        try:
            r, tot, cum = profile_sym(
                sym, args.D, args.chi, args.depth, args.reps, pipeline_k=args.pipeline_k
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {type(exc).__name__}: {exc}")
            rows.append({"sym": sym, "error": f"{type(exc).__name__}: {exc}"})
            if args.json:
                with open(args.json, "w") as fh:
                    json.dump({**meta, "rows": rows}, fh, indent=2)
            continue
        print(
            f"  n_blocks={r['n_blocks']}  E={r['energy']:.6f}  "
            f"compile={r['compile_wall_s']:.1f}s"
        )
        print(
            f"  warm_med={r['warm_med_s'] * 1e3:.1f} ms  "
            f"dispatch={r['dispatch_med_s'] * 1e3:.1f} ms "
            f"({100 * r['dispatch_frac']:.1f}% of warm)  "
            f"sync={r['sync_med_s'] * 1e3:.1f} ms"
        )
        print(
            f"  pipeline x{r['pipeline_k']}: per_call={r['pipeline_per_call_s'] * 1e3:.1f} ms  "
            f"speedup_vs_serial={r['pipeline_speedup_vs_serial']:.2f}x  "
            f"(>1 => device-bound, host runs ahead)"
        )
        print("  -- top tottime (host CPU) --")
        for t in r["top_tottime"][:12]:
            print(
                f"    {t['tottime']:>8.3f}s  ncalls={t['ncalls']:>9}  {t['func']}"
            )
        print("\n  ===== cProfile tottime (top 30) =====")
        print(tot)
        rows.append(r)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump({**meta, "rows": rows}, fh, indent=2)

    # Summary verdict heuristic
    print("\n" + "=" * 92)
    print("# READING: dispatch_frac high (>~50%) => HOST/eager-dispatch bound "
          "(Python batching can help).")
    print("#          dispatch_frac low + pipeline speedup >1 => DEVICE bound "
          "(need graph-level batching: fewer/bigger kernels).")
    if args.json:
        print(f"\nJSON -> {args.json}")


if __name__ == "__main__":
    main()
