"""Memory feasibility: dense CTM at large D, single-GPU vs N-GPU GSPMD mesh.

Usage (real GPUs):
    # 1) find the single-GPU OOM ceiling:
    CUDA_VISIBLE_DEVICES=0 uv run python examples/bench_ctm_sharding_memory.py --D 6 8 --chi 24
    # 2) show it fits on N GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python examples/bench_ctm_sharding_memory.py --D 6 8 --chi 24 --shard

Reports per-device peak memory (peak_bytes_in_use) and whether the run completed
or OOM'd, for each D.

Note on the reported peak: the probe runs FULL CTM convergence (all four
directional moves per sweep), so the per-device peak naturally reflects the
WORST move. This matters because per-move resharding shards the double-layer
tensor ``a`` on a different surviving D²-leg for each directional move; a
single-move benchmark would understate the true peak. Running the whole sweep
captures the dominant move's footprint.

Throwaway manual benchmark: needs real GPUs to show the headline (1-GPU OOM ->
N-GPU OK) result. Not a CI test; only smoke-run on CPU to confirm the harness
wires up.
"""

import argparse
import os
import sys

import jax

from tenax.algorithms.ctm_sharding import build_ctm_mesh

# This throwaway benchmark deliberately reuses the test-layer probe helper to
# avoid duplicating the iPEPS construction.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from _ctm_sharding_probe import heisenberg_dense_probe_energy  # noqa: E402


def peak_gb():
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--chi", type=int, default=24)
    ap.add_argument("--shard", action="store_true")
    args = ap.parse_args()
    mesh = build_ctm_mesh() if args.shard else None
    n = jax.device_count() if args.shard else 1
    print(
        f"# devices={jax.device_count()} shard={args.shard} mesh_n={n} chi={args.chi}"
    )
    for D in args.D:
        try:
            e = heisenberg_dense_probe_energy(
                D=D, chi=args.chi, device_mesh=mesh, seed=0
            )
            print(f"D={D}: OK  E={float(e):.6f}  per_device_peak={peak_gb():.2f} GB")
        except Exception as ex:  # noqa: BLE001 - benchmark reports OOM and continues
            print(f"D={D}: FAILED ({type(ex).__name__}: {str(ex)[:80]})")


if __name__ == "__main__":
    main()
