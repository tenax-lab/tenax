from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Timing:
    name: str
    issue_phase_s: float
    tail_wait_s: float
    total_s: float
    checksum: float


def device_summary() -> dict[str, str]:
    devices = jax.devices()
    backend = jax.default_backend()
    platforms = sorted({device.platform for device in devices})
    return {
        "default_backend": backend,
        "device_count": str(len(devices)),
        "platforms": ",".join(platforms),
        "devices": str(devices),
        "primary_device": str(devices[0]) if devices else "none",
        "x64_enabled": str(jax.config.x64_enabled),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark independent JAX QR calls from a Python loop."
    )
    parser.add_argument("--num-matrices", type=int, default=32)
    parser.add_argument("--matrix-size", type=int, default=96)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-devices",
        type=int,
        default=0,
        help=(
            "If at least this many visible JAX devices are present on the active "
            "backend, split the workload across the first N devices and interleave "
            "QR submissions."
        ),
    )
    parser.add_argument(
        "--two-gpu-split",
        action="store_true",
        help=(
            "Compatibility alias for --split-devices 2 on GPU."
        ),
    )
    parser.add_argument(
        "--include-vmap-compile",
        action="store_true",
        help=(
            "Also measure a fresh first-call JIT compile+run and the immediate "
            "second call for the vmap QR path."
        ),
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help=(
            "When a split-device benchmark is requested, run only the split mode. "
            "This avoids materializing the full batch on a single default device."
        ),
    )
    return parser.parse_args()


def make_inputs(
    num_matrices: int, matrix_size: int, dtype_name: str, seed: int
) -> np.ndarray:
    dtype = getattr(np, dtype_name)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(
        (num_matrices, matrix_size, matrix_size)
    ).astype(dtype, copy=False)


def configure_precision(dtype_name: str) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        jax.config.update("jax_enable_x64", True)


def _checksum_terms(q: jax.Array, r: jax.Array) -> jax.Array:
    # Keep a tiny dependency on both outputs without building a long reduction chain.
    return q[0, 0] + r[0, 0]


def bench_loop_sync_each(mats: jax.Array) -> Timing:
    start = time.perf_counter()
    checksum = 0.0
    issue_phase_s = 0.0
    tail_wait_s = 0.0

    for i in range(mats.shape[0]):
        t0 = time.perf_counter()
        q, r = jnp.linalg.qr(mats[i])
        t1 = time.perf_counter()
        term = _checksum_terms(q, r)
        term = jax.block_until_ready(term)
        t2 = time.perf_counter()
        checksum += float(term)
        issue_phase_s += t1 - t0
        tail_wait_s += t2 - t1

    total_s = time.perf_counter() - start
    return Timing(
        name="loop_sync_each",
        issue_phase_s=issue_phase_s,
        tail_wait_s=tail_wait_s,
        total_s=total_s,
        checksum=checksum,
    )


def bench_loop_dispatch_then_sync(mats: jax.Array) -> Timing:
    start = time.perf_counter()
    dispatch_start = time.perf_counter()
    outputs = []
    for i in range(mats.shape[0]):
        q, r = jnp.linalg.qr(mats[i])
        outputs.append(_checksum_terms(q, r))
    dispatch_end = time.perf_counter()

    sync_start = time.perf_counter()
    stacked = jnp.stack(outputs)
    stacked = jax.block_until_ready(stacked)
    sync_end = time.perf_counter()

    checksum = float(jnp.sum(stacked))
    total_s = time.perf_counter() - start
    return Timing(
        name="loop_dispatch_then_sync",
        issue_phase_s=dispatch_end - dispatch_start,
        tail_wait_s=sync_end - sync_start,
        total_s=total_s,
        checksum=checksum,
    )


def _vmap_qr_kernel(mats: jax.Array) -> jax.Array:
    q, r = jax.vmap(jnp.linalg.qr)(mats)
    return q[:, 0, 0] + r[:, 0, 0]


VMAP_QR = jax.jit(_vmap_qr_kernel)


def make_fresh_vmap_qr():
    def fresh_vmap_qr(mats: jax.Array) -> jax.Array:
        return _vmap_qr_kernel(mats)

    return jax.jit(fresh_vmap_qr)


def bench_vmap_jit(mats: jax.Array) -> Timing:
    return _run_jitted_vmap_qr(VMAP_QR, mats, "batched_vmap_jit")


def _run_jitted_vmap_qr(
    qr_fn: Callable[[jax.Array], jax.Array],
    mats: jax.Array,
    name: str,
) -> Timing:
    start = time.perf_counter()
    dispatch_start = time.perf_counter()
    outputs = qr_fn(mats)
    dispatch_end = time.perf_counter()

    sync_start = time.perf_counter()
    outputs = jax.block_until_ready(outputs)
    sync_end = time.perf_counter()

    checksum = float(jnp.sum(outputs))
    total_s = time.perf_counter() - start
    return Timing(
        name=name,
        issue_phase_s=dispatch_end - dispatch_start,
        tail_wait_s=sync_end - sync_start,
        total_s=total_s,
        checksum=checksum,
    )


def bench_vmap_jit_compile_and_first_call(mats: jax.Array) -> Timing:
    fresh_vmap_qr = make_fresh_vmap_qr()
    return _run_jitted_vmap_qr(
        fresh_vmap_qr,
        mats,
        "batched_vmap_jit_first_call",
    )


def bench_vmap_jit_second_call(mats: jax.Array) -> Timing:
    fresh_vmap_qr = make_fresh_vmap_qr()
    warm_outputs = fresh_vmap_qr(mats)
    jax.block_until_ready(warm_outputs)
    return _run_jitted_vmap_qr(
        fresh_vmap_qr,
        mats,
        "batched_vmap_jit_second_call",
    )


def has_two_gpu_devices() -> bool:
    gpus = [device for device in jax.devices() if device.platform == "gpu"]
    return len(gpus) >= 2


def visible_devices(platform: str | None = None) -> list[jax.Device]:
    devices = jax.devices()
    if platform is None:
        return devices
    return [device for device in devices if device.platform == platform]


def has_at_least_n_devices(n: int, platform: str | None = None) -> bool:
    return len(visible_devices(platform)) >= n


def prepare_device_split_inputs(
    mats: Any, num_devices: int, platform: str | None = None
) -> tuple[jax.Array, ...]:
    target_devices = visible_devices(platform)
    if len(target_devices) < num_devices:
        platform_msg = f" on platform '{platform}'" if platform else ""
        raise RuntimeError(
            f"split benchmark requires at least {num_devices} visible devices"
            f"{platform_msg}"
        )

    if mats.shape[0] < num_devices:
        raise ValueError(
            "split benchmark requires at least as many matrices as devices so the "
            "workload can be partitioned across devices"
        )

    selected_devices = target_devices[:num_devices]
    device_batches = []
    start = 0
    total = mats.shape[0]
    for offset, device in enumerate(selected_devices):
        remaining = total - start
        devices_left = len(selected_devices) - offset
        batch_size = remaining // devices_left
        host_batch = mats[start : start + batch_size]
        start += batch_size
        with jax.default_device(device):
            device_batches.append(jax.device_put(host_batch, device))
    return tuple(device_batches)


def prepare_two_gpu_split_inputs(mats: jax.Array) -> tuple[jax.Array, jax.Array]:
    device_batches = prepare_device_split_inputs(mats, num_devices=2, platform="gpu")
    return device_batches[0], device_batches[1]


def _is_device_batch_tuple(mats: Any) -> bool:
    return isinstance(mats, tuple) and all(hasattr(batch, "shape") for batch in mats)


def bench_multi_device_split_dispatch_then_sync(
    mats: Any,
    num_devices: int | None = None,
    platform: str | None = None,
    name: str | None = None,
) -> Timing:
    """Dispatch independent QR calls across pre-placed device batches.

    Inputs are expected to already live on distinct devices if this function is
    used from the timed benchmark path. Calls are interleaved across the
    device-resident batches so the host alternates submissions rather than
    draining one device fully before touching the next.
    """
    if _is_device_batch_tuple(mats):
        device_batches = mats
    else:
        if num_devices is None:
            raise ValueError("num_devices is required when mats are not pre-split")
        device_batches = prepare_device_split_inputs(mats, num_devices, platform)

    mode_name = name or f"split_{len(device_batches)}_devices_dispatch_then_sync"

    start = time.perf_counter()

    dispatch_start = time.perf_counter()
    outputs_by_device = [[] for _ in device_batches]
    max_len = max(batch.shape[0] for batch in device_batches)
    for i in range(max_len):
        for device_index, batch in enumerate(device_batches):
            if i < batch.shape[0]:
                q, r = jnp.linalg.qr(batch[i])
                outputs_by_device[device_index].append(_checksum_terms(q, r))
    dispatch_end = time.perf_counter()

    sync_start = time.perf_counter()
    stacked_outputs = []
    for device_outputs in outputs_by_device:
        stacked = jnp.stack(device_outputs)
        stacked_outputs.append(jax.block_until_ready(stacked))
    sync_end = time.perf_counter()

    checksum = sum(float(jnp.sum(stacked)) for stacked in stacked_outputs)
    total_s = time.perf_counter() - start
    return Timing(
        name=mode_name,
        issue_phase_s=dispatch_end - dispatch_start,
        tail_wait_s=sync_end - sync_start,
        total_s=total_s,
        checksum=checksum,
    )


def bench_two_gpu_split_dispatch_then_sync(
    mats: Any,
) -> Timing:
    return bench_multi_device_split_dispatch_then_sync(
        mats,
        num_devices=2,
        platform="gpu",
        name="two_gpu_split_dispatch_then_sync",
    )


def summarize(results: list[Timing]) -> None:
    grouped: dict[str, list[Timing]] = {}
    for result in results:
        grouped.setdefault(result.name, []).append(result)

    print()
    print("Summary")
    print("-------")
    header = (
        f"{'mode':<32} {'issue_ms':>12} {'tail_wait_ms':>14} "
        f"{'total_ms':>12} {'checksum':>16}"
    )
    print(header)
    print("-" * len(header))
    for name, timings in grouped.items():
        issue_ms = 1e3 * statistics.mean(t.issue_phase_s for t in timings)
        tail_wait_ms = 1e3 * statistics.mean(t.tail_wait_s for t in timings)
        total_ms = 1e3 * statistics.mean(t.total_s for t in timings)
        checksum = statistics.mean(t.checksum for t in timings)
        print(
            f"{name:<32} {issue_ms:12.3f} {tail_wait_ms:14.3f} "
            f"{total_ms:12.3f} {checksum:16.6f}"
        )


def main() -> None:
    args = parse_args()
    configure_precision(args.dtype)
    host_mats = make_inputs(args.num_matrices, args.matrix_size, args.dtype, args.seed)
    summary = device_summary()
    mats: jax.Array | None = None
    split_inputs = None
    split_benchmark: Callable[[Any], Timing] | None = None
    split_only = False

    print("JAX QR dispatch benchmark")
    print("-------------------------")
    print(f"jax version  : {jax.__version__}")
    print(f"backend      : {summary['default_backend']}")
    print(f"platforms    : {summary['platforms']}")
    print(f"device count : {summary['device_count']}")
    print(f"primary dev  : {summary['primary_device']}")
    print(f"x64 enabled  : {summary['x64_enabled']}")
    print(f"shape        : {host_mats.shape}")
    print(f"dtype        : {host_mats.dtype}")
    print(f"trials       : {args.trials}")
    print(f"warmup       : {args.warmup}")
    print(f"all devices  : {summary['devices']}")

    benchmarks: list = [
        bench_loop_sync_each,
        bench_loop_dispatch_then_sync,
        bench_vmap_jit,
    ]

    requested_split_devices = args.split_devices
    if args.two_gpu_split:
        requested_split_devices = max(requested_split_devices, 2)

    if requested_split_devices > 1:
        if has_at_least_n_devices(requested_split_devices):
            split_inputs = prepare_device_split_inputs(
                host_mats, requested_split_devices
            )

            def run_split(prepared_inputs: Any) -> Timing:
                return bench_multi_device_split_dispatch_then_sync(
                    prepared_inputs,
                    name=(
                        "two_gpu_split_dispatch_then_sync"
                        if args.two_gpu_split and requested_split_devices == 2
                        else f"split_{requested_split_devices}_devices_dispatch_then_sync"
                    ),
                )

            split_benchmark = run_split
            benchmarks.append(split_benchmark)
            split_only = args.split_only
        else:
            print(
                "note         : "
                f"--split-devices {requested_split_devices} requested, but fewer "
                "visible devices were found"
            )

    if args.split_only and split_benchmark is None:
        print("note         : --split-only requested without an active split benchmark")

    if split_only:
        benchmarks = [split_benchmark]

    if args.two_gpu_split:
        if not has_two_gpu_devices():
            print("note         : --two-gpu-split requested, but fewer than two GPUs found")

    compile_benchmarks: list = []
    if args.include_vmap_compile:
        compile_benchmarks = [
            bench_vmap_jit_compile_and_first_call,
            bench_vmap_jit_second_call,
        ]
    if split_only:
        compile_benchmarks = []

    for _ in range(args.warmup):
        for benchmark in benchmarks:
            if split_benchmark is not None and benchmark is split_benchmark and split_inputs:
                benchmark(split_inputs)
            else:
                if mats is None:
                    mats = jax.device_put(host_mats)
                benchmark(mats)

    results: list[Timing] = []
    for trial in range(args.trials):
        print()
        print(f"Trial {trial + 1}")
        print(
            f"{'mode':<32} {'issue_ms':>12} {'tail_wait_ms':>14} {'total_ms':>12}"
        )
        print("-" * 76)
        for benchmark in benchmarks + compile_benchmarks:
            if split_benchmark is not None and benchmark is split_benchmark and split_inputs:
                result = benchmark(split_inputs)
            else:
                if mats is None:
                    mats = jax.device_put(host_mats)
                result = benchmark(mats)
            results.append(result)
            print(
                f"{result.name:<32} {1e3 * result.issue_phase_s:12.3f} "
                f"{1e3 * result.tail_wait_s:14.3f} {1e3 * result.total_s:12.3f}"
            )

    summarize(results)


if __name__ == "__main__":
    main()
