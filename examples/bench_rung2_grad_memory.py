"""Rung-2 gate 2B/2C: value_and_grad (fwd+implicit-AD bwd) memory, single vs N-GPU.

Usage (real GPUs):
    # single-GPU ceiling:
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_rung2_grad_memory.py --D 8 10 --chi 24
    # 4-GPU sharded backward:
    CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_rung2_grad_memory.py --D 8 10 12 --chi 24 --shard

Reports per-device peak (peak_bytes_in_use) of the FULL value_and_grad — forward
CTM + implicit-AD backward — so 2B shows whether GSPMD shards the backward and 2C
the largest D that runs. One D per process for a clean peak. Throwaway gate bench.
"""

import argparse
import os
import sys
import time

import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from _rung2_grad_probe import implicit_energy_and_grad  # noqa: E402

from tenax.algorithms.ctm_sharding import build_ctm_mesh  # noqa: E402


def peak_gb():
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, nargs="+", default=[8])
    ap.add_argument("--chi", type=int, default=24)
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--max-iter", type=int, default=30)
    args = ap.parse_args()
    mesh = build_ctm_mesh() if args.shard else None
    n = jax.device_count() if args.shard else 1
    print(
        f"# value_and_grad devices={jax.device_count()} shard={args.shard} "
        f"mesh_n={n} chi={args.chi} max_iter={args.max_iter} "
        f"x64={jax.config.jax_enable_x64}"
    )
    for D in args.D:
        t0 = time.perf_counter()
        try:
            e, g = implicit_energy_and_grad(
                D=D, chi=args.chi, device_mesh=mesh, seed=0,
                well_conditioned=True, max_iter=args.max_iter,
            )
            dt = time.perf_counter() - t0
            gnorm = float((g**2).sum() ** 0.5)
            print(
                f"D={D}: OK  E={e:.6f}  |grad|={gnorm:.4e}  "
                f"per_device_peak={peak_gb():.2f} GB  wall={dt:.1f}s"
            )
        except Exception as ex:  # noqa: BLE001 - bench reports OOM and continues
            dt = time.perf_counter() - t0
            print(f"D={D}: FAILED ({type(ex).__name__}: {str(ex)[:90]})  wall={dt:.1f}s")


if __name__ == "__main__":
    main()
