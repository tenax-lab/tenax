"""CLI entry point for Tenax benchmarks.

Usage::

    python -m benchmarks.run --backend cpu --algorithm trg --size small --trials 1
    python -m benchmarks.run --backend cuda --algorithm all --size all -o results.json
    python -m benchmarks.run --list-backends
"""

from __future__ import annotations

import argparse
import datetime
import sys

_ALGORITHM_MODULES = {
    "dmrg": "benchmarks.bench_dmrg",
    "idmrg": "benchmarks.bench_idmrg",
    "trg": "benchmarks.bench_trg",
    "hotrg": "benchmarks.bench_hotrg",
    "ipeps": "benchmarks.bench_ipeps",
    "ipeps_ad": "benchmarks.bench_ipeps_ad",
    "dmrg_sharded": "benchmarks.bench_dmrg_sharded",
}

_ALL_SIZES = ["small", "medium", "large"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tenax benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        "-b",
        default="auto",
        help="Backend: cpu | cuda | gpu | tpu | metal | auto (default: auto)",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        nargs="+",
        default=["all"],
        help="Algorithms to benchmark (default: all)",
    )
    parser.add_argument(
        "--size",
        "-s",
        nargs="+",
        default=["all"],
        help="Sizes to benchmark: small | medium | large | all (default: all)",
    )
    parser.add_argument(
        "--trials",
        "-n",
        type=int,
        default=3,
        help="Number of timed trials per benchmark (default: 3)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="JSON output path (default: benchmarks/results/<backend>_<timestamp>.json)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Print available backends and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # --- configure backend BEFORE importing JAX ---
    from benchmarks.backend import configure_backend

    configure_backend(args.backend)

    # --- now safe to import JAX-dependent code ---
    import importlib

    from benchmarks.backend import default_dtype, get_backend_info
    from benchmarks.results import (
        print_summary_table,
        save_results_csv,
        save_results_json,
    )
    from benchmarks.runner import run_benchmark

    if args.list_backends:
        info = get_backend_info()
        print("Current JAX backend info:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    backend_info = get_backend_info()
    dtype = default_dtype(backend_info["backend"])
    dtype_str = getattr(dtype, "__name__", str(dtype))
    backend_name = backend_info["backend"]

    print(
        f"Backend: {backend_name} | dtype: {dtype_str} | x64: {backend_info['x64_enabled']}"
    )
    print(f"Device: {backend_info['device_kind']} (x{backend_info['device_count']})")

    # Resolve algorithms
    if "all" in args.algorithm:
        algo_names = list(_ALGORITHM_MODULES.keys())
    else:
        algo_names = []
        for a in args.algorithm:
            if a not in _ALGORITHM_MODULES:
                print(
                    f"Unknown algorithm: {a}. Available: {', '.join(_ALGORITHM_MODULES)}"
                )
                sys.exit(1)
            algo_names.append(a)

    # Resolve sizes
    sizes = _ALL_SIZES if "all" in args.size else args.size

    # Collect benchmarks
    all_benchmarks = []
    for algo_name in algo_names:
        mod = importlib.import_module(_ALGORITHM_MODULES[algo_name])
        for bench in mod.get_benchmarks(dtype=dtype):
            if bench["size_label"] in sizes:
                all_benchmarks.append(bench)

    if not all_benchmarks:
        print("No benchmarks matched the selection.")
        sys.exit(0)

    print(
        f"\nRunning {len(all_benchmarks)} benchmark(s), {args.trials} trial(s) each...\n"
    )

    # Resolve output paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.output or f"benchmarks/results/{backend_name}_{timestamp}.json"

    # Run benchmarks, saving incrementally after each one
    results = []
    for i, bench in enumerate(all_benchmarks, 1):
        label = f"[{i}/{len(all_benchmarks)}] {bench['name']} / {bench['size_label']}"
        print(f"{label} ...", end=" ", flush=True)
        result = run_benchmark(
            name=bench["name"],
            size_label=bench["size_label"],
            params=bench["params"],
            setup_fn=bench["setup_fn"],
            run_fn=bench["run_fn"],
            backend_info=backend_info,
            dtype_str=dtype_str,
            num_trials=args.trials,
        )
        if result.error:
            print("ERROR")
        else:
            print(f"{result.mean_time_s:.3f}s (min={result.min_time_s:.3f}s)")
        results.append(result)
        save_results_json(results, json_path, quiet=True)

    # Print summary
    print_summary_table(results)

    # Final save (with message)
    save_results_json(results, json_path)

    if args.csv:
        save_results_csv(results, args.csv)


if __name__ == "__main__":
    main()
