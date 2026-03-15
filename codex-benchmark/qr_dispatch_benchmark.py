from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp


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
        "--two-gpu-split",
        action="store_true",
        help=(
            "If two or more GPU devices are available, split the workload across "
            "the first two GPUs and benchmark concurrent dispatch."
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
    return parser.parse_args()


def make_inputs(
    num_matrices: int, matrix_size: int, dtype_name: str, seed: int
) -> jax.Array:
    dtype = getattr(jnp, dtype_name)
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(
        key, (num_matrices, matrix_size, matrix_size), dtype=dtype
    )


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


def prepare_two_gpu_split_inputs(mats: jax.Array) -> tuple[jax.Array, jax.Array]:
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(gpu_devices) < 2:
        raise RuntimeError("two_gpu_split benchmark requires at least two GPU devices")

    split = mats.shape[0] // 2
    if split == 0 or split == mats.shape[0]:
        raise ValueError(
            "two_gpu_split benchmark requires at least two matrices so the workload "
            "can be split across devices"
        )

    d0, d1 = gpu_devices[:2]
    host_mats_0 = mats[:split]
    host_mats_1 = mats[split:]

    with jax.default_device(d0):
        mats_0 = jax.device_put(host_mats_0, d0)
    with jax.default_device(d1):
        mats_1 = jax.device_put(host_mats_1, d1)
    return mats_0, mats_1


def bench_two_gpu_split_dispatch_then_sync(
    mats: Any,
) -> Timing:
    """Dispatch independent QR calls across two pre-placed GPU batches.

    Inputs are expected to already live on distinct devices if this function is
    used from the timed benchmark path. Calls are interleaved across the two
    device-resident batches so the host alternates submissions to ``cuda:0`` and
    ``cuda:1`` rather than draining one device fully before touching the other.
    """
    if (
        isinstance(mats, tuple)
        and len(mats) == 2
        and hasattr(mats[0], "shape")
        and hasattr(mats[1], "shape")
    ):
        mats_0, mats_1 = mats
    else:
        mats_0, mats_1 = prepare_two_gpu_split_inputs(mats)

    start = time.perf_counter()

    dispatch_start = time.perf_counter()
    outputs_0 = []
    outputs_1 = []
    max_len = max(mats_0.shape[0], mats_1.shape[0])
    for i in range(max_len):
        if i < mats_0.shape[0]:
            q0, r0 = jnp.linalg.qr(mats_0[i])
            outputs_0.append(_checksum_terms(q0, r0))
        if i < mats_1.shape[0]:
            q1, r1 = jnp.linalg.qr(mats_1[i])
            outputs_1.append(_checksum_terms(q1, r1))
    dispatch_end = time.perf_counter()

    sync_start = time.perf_counter()
    stacked_0 = jnp.stack(outputs_0)
    stacked_1 = jnp.stack(outputs_1)
    stacked_0 = jax.block_until_ready(stacked_0)
    stacked_1 = jax.block_until_ready(stacked_1)
    sync_end = time.perf_counter()

    checksum = float(jnp.sum(stacked_0)) + float(jnp.sum(stacked_1))
    total_s = time.perf_counter() - start
    return Timing(
        name="two_gpu_split_dispatch_then_sync",
        issue_phase_s=dispatch_end - dispatch_start,
        tail_wait_s=sync_end - sync_start,
        total_s=total_s,
        checksum=checksum,
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
    mats = make_inputs(args.num_matrices, args.matrix_size, args.dtype, args.seed)
    summary = device_summary()
    two_gpu_inputs = None

    print("JAX QR dispatch benchmark")
    print("-------------------------")
    print(f"jax version  : {jax.__version__}")
    print(f"backend      : {summary['default_backend']}")
    print(f"platforms    : {summary['platforms']}")
    print(f"device count : {summary['device_count']}")
    print(f"primary dev  : {summary['primary_device']}")
    print(f"shape        : {mats.shape}")
    print(f"dtype        : {mats.dtype}")
    print(f"trials       : {args.trials}")
    print(f"warmup       : {args.warmup}")
    print(f"all devices  : {summary['devices']}")

    benchmarks: list = [
        bench_loop_sync_each,
        bench_loop_dispatch_then_sync,
        bench_vmap_jit,
    ]

    if args.two_gpu_split:
        if has_two_gpu_devices():
            two_gpu_inputs = prepare_two_gpu_split_inputs(mats)
            benchmarks.append(bench_two_gpu_split_dispatch_then_sync)
        else:
            print("note         : --two-gpu-split requested, but fewer than two GPUs found")

    compile_benchmarks: list = []
    if args.include_vmap_compile:
        compile_benchmarks = [
            bench_vmap_jit_compile_and_first_call,
            bench_vmap_jit_second_call,
        ]

    for _ in range(args.warmup):
        for benchmark in benchmarks:
            if benchmark is bench_two_gpu_split_dispatch_then_sync and two_gpu_inputs:
                benchmark(two_gpu_inputs)
            else:
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
            if benchmark is bench_two_gpu_split_dispatch_then_sync and two_gpu_inputs:
                result = benchmark(two_gpu_inputs)
            else:
                result = benchmark(mats)
            results.append(result)
            print(
                f"{result.name:<32} {1e3 * result.issue_phase_s:12.3f} "
                f"{1e3 * result.tail_wait_s:14.3f} {1e3 * result.total_s:12.3f}"
            )

    summarize(results)


if __name__ == "__main__":
    main()
