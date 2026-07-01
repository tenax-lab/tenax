"""#632 shard-only reach benchmark — dense CTM-AD ``value_and_grad`` (forward CTM +
implicit-AD backward), single-GPU vs N-GPU GSPMD mesh, at large D.

This is the *shard-only* lever (rung-2 GSPMD ``device_mesh``), NOT chunking: the
#632 Increment-2 gate showed chunking the backward is a NO-GO, so the large-D
backward memory relief is GSPMD sharding. The backward shards via GSPMD
propagating from the sharded env residuals (rung-2). Reports per-device peak of
the full ``value_and_grad`` and whether it fit or OOM'd, extending the rung-2
table (single ceiling D=8, 4-GPU D=10 at chi=24) toward D=10-12.

NB: the mesh dim N must divide D^2 (the sharded double-layer leg). D=10 -> 100
(N in {2,4,5}); D=12 -> 144 (N in {2,3,4,6}); D=11 -> 121 is NOT shardable.

Usage (one D per process for a clean peak; PREALLOCATE=false for a faithful peak):
    # single-GPU ceiling:
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_ctm_shard_reach_grad.py --D 10 --chi 24
    # 2-GPU sharded:
    CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_ctm_shard_reach_grad.py --D 10 --chi 24 --shard
"""

import argparse
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ctm_sharding import build_ctm_mesh  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from _rung2_grad_probe import implicit_energy_and_grad  # noqa: E402


def peak_gb():
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, nargs="+", default=[10, 12])
    ap.add_argument("--chi", type=int, default=24)
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--max-iter", type=int, default=30)
    args = ap.parse_args()

    n = jax.device_count()
    mesh = build_ctm_mesh() if args.shard else None
    mesh_n = n if args.shard else 1
    print(
        f"# devices={n} shard={args.shard} mesh_n={mesh_n} chi={args.chi} "
        f"max_iter={args.max_iter} recipe=2x2 well_conditioned=True x64=True"
    )
    for D in args.D:
        if args.shard and (D * D) % mesh_n != 0:
            print(f"D={D}: SKIP (D^2={D*D} not divisible by mesh_n={mesh_n})")
            continue
        t0 = time.perf_counter()
        try:
            e, g = implicit_energy_and_grad(
                D=D, chi=args.chi, device_mesh=mesh, seed=0,
                well_conditioned=True, max_iter=args.max_iter,
            )
            gnorm = float((g ** 2).sum() ** 0.5)
            dt = time.perf_counter() - t0
            print(
                f"D={D}: OK  E={e:.6f}  |g|={gnorm:.3e}  "
                f"per_device_peak={peak_gb():.2f} GB  wall={dt:.1f}s"
            )
        except Exception as ex:  # noqa: BLE001
            dt = time.perf_counter() - t0
            print(f"D={D}: FAILED ({type(ex).__name__}: {str(ex)[:110]})  wall={dt:.1f}s")


if __name__ == "__main__":
    main()
