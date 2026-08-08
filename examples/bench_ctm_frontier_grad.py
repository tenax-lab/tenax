"""#632 large-D x large-chi multi-GPU frontier benchmark — value_and_grad reach.

Per-device peak of ONE value_and_grad step across:
  --path split               1-GPU  ctm_energy_split_implicit
  --path dense               1-GPU  ctm_energy_implicit
  --path dense --shard       N-GPU  + GSPMD device_mesh
  --path dense --shard --chunk K   N-GPU  + device_mesh + ctm_chunk_size

One (path, D, chi) per process for a clean cumulative peak (shard-reach method).

``--recipe`` selects the CTM projector on both arms, default ``2x2`` (the
library default since #746). **The original 2026-07-01 run pinned the dense arm
to ``1x1``**, whose corner-pair projector collapses the environment to a rank-1
corner (#723/#726), so every (D, chi) reach number it produced describes a
chi_eff=1 mean-field boundary rather than a converged CTM (#747). At D=8 the
forward re-run showed split's realized peak equals dense's to within 3% once the
recipe is correct -- the chi^2*D^4 vs chi^2*D^6 separation was a ``1x1``
property. Pass ``--recipe 1x1`` to reproduce the historical run.

``--gate`` runs the forward CTM instead of value_and_grad and reports
``rank(C1)``: the sound collapse detector (#747 comment 3). Expect rank == chi
on ``2x2`` and rank == 1 on ``1x1``. Run it on a few representative cells before
trusting a grid.

Usage (PREALLOCATE=false for a faithful peak; one config per process):
    CUDA_VISIBLE_DEVICES=1   XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_ctm_frontier_grad.py --path split --D 10 --chi 48
    CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_ctm_frontier_grad.py --path dense --D 10 --chi 24 --shard --chunk 8
    CUDA_VISIBLE_DEVICES=1   uv run python examples/bench_ctm_frontier_grad.py \
        --path split --D 10 --chi 32 --gate
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
    """Max ``peak_bytes_in_use`` across the mesh, in GB.

    Reads **every** device, not just ``devices()[0]``: on a sharded run the
    per-device high-water marks differ, so sampling device 0 alone can
    under-report the true per-device peak and flatter the sharded arm
    (unresolved review comment on PR #673).
    """
    import jax

    peaks = []
    for dev in jax.devices():
        try:
            peaks.append(dev.memory_stats()["peak_bytes_in_use"])
        except Exception:  # noqa: BLE001, PERF203
            continue
    return max(peaks) / 1e9 if peaks else float("nan")


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
    ap.add_argument(
        "--recipe",
        choices=["1x1", "2x2"],
        default="2x2",
        help="CTM projector on both arms (default 2x2; 1x1 reproduces the "
        "collapsed 2026-07-01 run, see #747)",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="run the forward CTM and report rank(C1) instead of value_and_grad",
    )
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    from tenax.algorithms.ctm_sharding import build_ctm_mesh

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from _frontier_grad_probe import frontier_corner_rank, frontier_energy_and_grad

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
        f"recipe={args.recipe} gate={args.gate} x64=True"
    )
    for D in args.D:
        reason = skip_reason(D, mesh_n, shard)
        if reason is not None:
            print(f"path={args.path} D={D} chi={args.chi}: SKIP ({reason})")
            continue
        t0 = time.perf_counter()
        if args.gate:
            try:
                rank, _ = frontier_corner_rank(
                    path=args.path,
                    D=D,
                    chi=args.chi,
                    chi_I=args.chi_I,
                    seed=args.seed,
                    well_conditioned=True,
                    max_iter=args.max_iter,
                    recipe=args.recipe,
                )
                dt = time.perf_counter() - t0
                verdict = "COLLAPSED" if rank <= 1 else "ok"
                print(
                    f"GATE path={args.path} D={D} chi={args.chi} "
                    f"recipe={args.recipe} corner_rank={rank}/{args.chi} "
                    f"{verdict}  wall={dt:.1f}s"
                )
            except Exception as ex:  # noqa: BLE001
                dt = time.perf_counter() - t0
                print(
                    f"GATE path={args.path} D={D} chi={args.chi} "
                    f"FAILED({type(ex).__name__}: {str(ex)[:110]})  wall={dt:.1f}s"
                )
            continue
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
                recipe=args.recipe,
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
