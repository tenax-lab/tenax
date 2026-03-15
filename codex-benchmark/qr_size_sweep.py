from __future__ import annotations

import argparse
import statistics

import jax

from qr_dispatch_benchmark import (
    Timing,
    bench_loop_dispatch_then_sync,
    bench_multi_device_split_dispatch_then_sync,
    bench_loop_sync_each,
    bench_two_gpu_split_dispatch_then_sync,
    bench_vmap_jit_compile_and_first_call,
    bench_vmap_jit_second_call,
    bench_vmap_jit,
    configure_precision,
    device_summary,
    has_at_least_n_devices,
    has_two_gpu_devices,
    make_inputs,
    prepare_device_split_inputs,
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
    parser.add_argument("--split-devices", type=int, default=0)
    parser.add_argument("--two-gpu-split", action="store_true")
    parser.add_argument("--include-vmap-compile", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    return parser.parse_args()


def summarize_mode(timings: list[Timing]) -> tuple[float, float, float]:
    return (
        1e3 * statistics.mean(t.issue_phase_s for t in timings),
        1e3 * statistics.mean(t.tail_wait_s for t in timings),
        1e3 * statistics.mean(t.total_s for t in timings),
    )


def main() -> None:
    args = parse_args()
    configure_precision(args.dtype)
    summary = device_summary()
    requested_split_devices = args.split_devices
    if args.two_gpu_split:
        requested_split_devices = max(requested_split_devices, 2)

    use_split_devices = requested_split_devices > 1 and has_at_least_n_devices(
        requested_split_devices
    )
    use_two_gpu_split = args.two_gpu_split and has_two_gpu_devices()
    split_only = args.split_only and use_split_devices
    benchmarks = [
        bench_loop_sync_each,
        bench_loop_dispatch_then_sync,
        bench_vmap_jit,
    ]
    split_benchmark = None
    compile_benchmarks = []
    if use_split_devices:
        if args.two_gpu_split and requested_split_devices == 2:
            split_benchmark = bench_two_gpu_split_dispatch_then_sync
        else:
            def run_split(prepared_inputs):
                return bench_multi_device_split_dispatch_then_sync(
                    prepared_inputs,
                    name=f"split_{requested_split_devices}_devices_dispatch_then_sync",
                )

            split_benchmark = run_split
        benchmarks.append(split_benchmark)
    if split_only:
        benchmarks = [split_benchmark]
    if args.include_vmap_compile:
        compile_benchmarks = [
            bench_vmap_jit_compile_and_first_call,
            bench_vmap_jit_second_call,
        ]
    if split_only:
        compile_benchmarks = []

    print("QR size sweep")
    print("-------------")
    print(f"jax version  : {jax.__version__}")
    print(f"backend      : {summary['default_backend']}")
    print(f"platforms    : {summary['platforms']}")
    print(f"device count : {summary['device_count']}")
    print(f"primary dev  : {summary['primary_device']}")
    print(f"x64 enabled  : {summary['x64_enabled']}")
    print(f"all devices  : {summary['devices']}")
    print(f"num_matrices : {args.num_matrices}")
    print(f"sizes        : {args.sizes}")
    print(f"trials       : {args.trials}")
    print(f"warmup       : {args.warmup}")
    print(f"dtype        : {args.dtype}")
    if args.split_devices > 1 and not use_split_devices:
        print(
            "note         : "
            f"--split-devices {args.split_devices} requested, but fewer visible "
            "devices were found"
        )
    if args.two_gpu_split and not use_two_gpu_split:
        print("note         : --two-gpu-split requested, but fewer than two GPUs found")
    if args.split_only and not use_split_devices:
        print("note         : --split-only requested without an active split benchmark")

    for size in args.sizes:
        host_mats = make_inputs(args.num_matrices, size, args.dtype, args.seed)
        mats = None
        split_inputs = None
        if use_split_devices:
            if args.two_gpu_split and requested_split_devices == 2:
                split_inputs = prepare_two_gpu_split_inputs(host_mats)
            else:
                split_inputs = prepare_device_split_inputs(
                    host_mats, requested_split_devices
                )
        for _ in range(args.warmup):
            for benchmark in benchmarks:
                if split_benchmark is not None and benchmark is split_benchmark and split_inputs:
                    benchmark(split_inputs)
                else:
                    if mats is None:
                        mats = jax.device_put(host_mats)
                    benchmark(mats)

        per_mode: dict[str, list[Timing]] = {}
        for _ in range(args.trials):
            for benchmark in benchmarks + compile_benchmarks:
                if split_benchmark is not None and benchmark is split_benchmark and split_inputs:
                    result = benchmark(split_inputs)
                else:
                    if mats is None:
                        mats = jax.device_put(host_mats)
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
