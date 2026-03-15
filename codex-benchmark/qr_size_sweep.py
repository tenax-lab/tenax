from __future__ import annotations

import argparse
import statistics

import jax

from qr_dispatch_benchmark import (
    Timing,
    bench_loop_dispatch_then_sync,
    bench_loop_sync_each,
    bench_two_gpu_split_dispatch_then_sync,
    bench_vmap_jit_compile_and_first_call,
    bench_vmap_jit_second_call,
    bench_vmap_jit,
    device_summary,
    has_two_gpu_devices,
    make_inputs,
    prepare_two_gpu_split_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QR dispatch benchmarks over a range of matrix sizes."
    )
    parser.add_argument("--num-matrices", type=int, default=64)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[128, 256, 512, 1024]
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--two-gpu-split", action="store_true")
    parser.add_argument("--include-vmap-compile", action="store_true")
    return parser.parse_args()


def summarize_mode(timings: list[Timing]) -> tuple[float, float, float]:
    return (
        1e3 * statistics.mean(t.issue_phase_s for t in timings),
        1e3 * statistics.mean(t.tail_wait_s for t in timings),
        1e3 * statistics.mean(t.total_s for t in timings),
    )


def main() -> None:
    args = parse_args()
    summary = device_summary()
    use_two_gpu_split = args.two_gpu_split and has_two_gpu_devices()
    benchmarks = [
        bench_loop_sync_each,
        bench_loop_dispatch_then_sync,
        bench_vmap_jit,
    ]
    compile_benchmarks = []
    if use_two_gpu_split:
        benchmarks.append(bench_two_gpu_split_dispatch_then_sync)
    if args.include_vmap_compile:
        compile_benchmarks = [
            bench_vmap_jit_compile_and_first_call,
            bench_vmap_jit_second_call,
        ]

    print("QR size sweep")
    print("-------------")
    print(f"jax version  : {jax.__version__}")
    print(f"backend      : {summary['default_backend']}")
    print(f"platforms    : {summary['platforms']}")
    print(f"device count : {summary['device_count']}")
    print(f"primary dev  : {summary['primary_device']}")
    print(f"all devices  : {summary['devices']}")
    print(f"num_matrices : {args.num_matrices}")
    print(f"sizes        : {args.sizes}")
    print(f"trials       : {args.trials}")
    print(f"warmup       : {args.warmup}")
    print(f"dtype        : {args.dtype}")

    for size in args.sizes:
        mats = make_inputs(args.num_matrices, size, args.dtype, args.seed)
        two_gpu_inputs = None
        if use_two_gpu_split:
            two_gpu_inputs = prepare_two_gpu_split_inputs(mats)
        for _ in range(args.warmup):
            for benchmark in benchmarks:
                if benchmark is bench_two_gpu_split_dispatch_then_sync and two_gpu_inputs:
                    benchmark(two_gpu_inputs)
                else:
                    benchmark(mats)

        per_mode: dict[str, list[Timing]] = {}
        for _ in range(args.trials):
            for benchmark in benchmarks + compile_benchmarks:
                if benchmark is bench_two_gpu_split_dispatch_then_sync and two_gpu_inputs:
                    result = benchmark(two_gpu_inputs)
                else:
                    result = benchmark(mats)
                per_mode.setdefault(result.name, []).append(result)

        print()
        print(f"size={size}")
        header = (
            f"{'mode':<32} {'issue_ms':>12} {'tail_wait_ms':>14} {'total_ms':>12}"
        )
        print(header)
        print("-" * len(header))
        for name, timings in per_mode.items():
            issue_ms, tail_wait_ms, total_ms = summarize_mode(timings)
            print(
                f"{name:<32} {issue_ms:12.3f} {tail_wait_ms:14.3f} {total_ms:12.3f}"
            )


if __name__ == "__main__":
    main()
