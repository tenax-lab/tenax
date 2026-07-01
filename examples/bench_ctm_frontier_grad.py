"""#632 large-D x large-chi multi-GPU frontier benchmark — value_and_grad reach.

Per-device peak of ONE value_and_grad step across (all at recipe=1x1):
  --path split               1-GPU  ctm_energy_split_implicit (chi^2 * D^4)
  --path dense               1-GPU  ctm_energy_implicit recipe=1x1 (chi^2 * D^6)
  --path dense --shard       N-GPU  + GSPMD device_mesh
  --path dense --shard --chunk K   N-GPU  + device_mesh + ctm_chunk_size

One (path, D, chi) per process for a clean cumulative peak (shard-reach method).

Two orthogonal walls (see the design spec):
  divisibility : --shard needs D^2 % mesh_n == 0  -> SKIP (decided before running)
  memory (OOM) : a shardable config can still exceed RAM -> FAILED(RESOURCE_EXHAUSTED)

Usage (PREALLOCATE=false for a faithful peak; one config per process):
    CUDA_VISIBLE_DEVICES=1   XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_ctm_frontier_grad.py --path split --D 10 --chi 48
    CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_ctm_frontier_grad.py --path dense --D 10 --chi 24 --shard --chunk 8
"""

import argparse
import os
import sys
import time


def skip_reason(D, mesh_n, shard):
    """SKIP reason iff a sharded config is un-shardable (D^2 not divisible by N)."""
    if shard and (D * D) % mesh_n != 0:
        return f"D^2={D * D} % mesh_n={mesh_n} != 0"
    return None


def peak_gb():
    import jax

    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:  # noqa: BLE001
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["dense", "split"], required=True)
    ap.add_argument("--D", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--chi", type=int, default=24)
    ap.add_argument("--chi-I", type=int, default=None, dest="chi_I")
    ap.add_argument("--shard", action="store_true")
    ap.add_argument(
        "--chunk", type=int, default=0, help="ctm_chunk_size (0=off; dense only)"
    )
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    from tenax.algorithms.ctm_sharding import build_ctm_mesh

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from _frontier_grad_probe import frontier_energy_and_grad

    n = jax.device_count()
    shard = args.shard and args.path == "dense"
    if args.shard and args.path == "split":
        print("# WARN: --shard ignored on split path (single-GPU only)")
    chunk = args.chunk if (args.chunk > 0 and args.path == "dense") else None
    if args.chunk > 0 and args.path == "split":
        print("# WARN: --chunk ignored on split path")
    mesh = build_ctm_mesh() if shard else None
    mesh_n = n if shard else 1
    print(
        f"# path={args.path} devices={n} shard={shard} mesh_n={mesh_n} "
        f"chi={args.chi} chi_I={args.chi_I} chunk={chunk} max_iter={args.max_iter} "
        f"recipe=1x1 x64=True"
    )
    for D in args.D:
        reason = skip_reason(D, mesh_n, shard)
        if reason is not None:
            print(f"path={args.path} D={D} chi={args.chi}: SKIP ({reason})")
            continue
        t0 = time.perf_counter()
        try:
            e, g = frontier_energy_and_grad(
                path=args.path,
                D=D,
                chi=args.chi,
                chi_I=args.chi_I,
                device_mesh=mesh,
                ctm_chunk_size=chunk,
                seed=args.seed,
                well_conditioned=True,
                max_iter=args.max_iter,
            )
            gnorm = float((g**2).sum() ** 0.5)
            dt = time.perf_counter() - t0
            print(
                f"path={args.path} D={D} chi={args.chi} OK  E={e:.6f}  |g|={gnorm:.3e}  "
                f"per_device_peak={peak_gb():.2f} GB  wall={dt:.1f}s"
            )
        except Exception as ex:  # noqa: BLE001
            dt = time.perf_counter() - t0
            print(
                f"path={args.path} D={D} chi={args.chi} "
                f"FAILED({type(ex).__name__}: {str(ex)[:110]})  wall={dt:.1f}s"
            )


if __name__ == "__main__":
    main()
