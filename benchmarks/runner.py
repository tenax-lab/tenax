"""Timing infrastructure for Tenax benchmarks."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp


@dataclass
class BenchmarkResult:
    algorithm: str = ""
    size_label: str = ""
    parameters: dict = field(default_factory=dict)
    backend: str = ""
    device_info: dict = field(default_factory=dict)
    dtype: str = ""
    warmup_time_s: float = 0.0
    times_s: list[float] = field(default_factory=list)
    mean_time_s: float = 0.0
    std_time_s: float = 0.0
    min_time_s: float = 0.0
    result_value: dict = field(default_factory=dict)
    error: str | None = None


def _sync() -> None:
    """Block until device computation completes."""
    jnp.zeros(1).block_until_ready()


def _extract_result(algorithm: str, raw: Any) -> dict:
    """Pull JSON-serializable values from algorithm output."""
    if raw is None:
        return {}
    algo = algorithm.lower()
    if algo in ("dmrg", "dmrg_sharded"):
        return {
            "energy": float(raw.energy),
            "converged": bool(raw.converged),
            "num_sweeps": len(raw.energies_per_sweep),
        }
    if algo == "idmrg":
        return {
            "energy_per_site": float(raw.energy_per_site),
            "converged": bool(raw.converged),
            "num_steps": len(raw.energies_per_step),
        }
    if algo in ("trg", "hotrg"):
        return {"free_energy": float(raw)}
    if algo == "ipeps":
        energy, _peps, _env = raw
        return {"energy_per_site": float(energy)}
    if algo in ("ipeps_ad", "ipeps_ad_c4v_reference"):
        _A, _env, energy = raw
        return {"energy_per_site": float(energy)}
    return {}


def run_benchmark(
    name: str,
    size_label: str,
    params: dict,
    setup_fn: Callable[[], tuple],
    run_fn: Callable[..., Any],
    backend_info: dict,
    dtype_str: str,
    num_trials: int = 3,
) -> BenchmarkResult:
    """Run a single benchmark with warmup and timed trials."""
    result = BenchmarkResult(
        algorithm=name,
        size_label=size_label,
        parameters=params,
        backend=backend_info.get("backend", "unknown"),
        device_info=backend_info,
        dtype=dtype_str,
    )

    try:
        # --- warmup ---
        args = setup_fn()
        _sync()
        t0 = time.perf_counter()
        raw = run_fn(*args)
        _sync()
        result.warmup_time_s = time.perf_counter() - t0
        result.result_value = _extract_result(name, raw)

        # --- timed trials ---
        times: list[float] = []
        for _ in range(num_trials):
            args = setup_fn()
            _sync()
            t0 = time.perf_counter()
            run_fn(*args)
            _sync()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        result.times_s = times
        result.mean_time_s = sum(times) / len(times)
        result.min_time_s = min(times)
        if len(times) > 1:
            mean = result.mean_time_s
            result.std_time_s = (
                sum((t - mean) ** 2 for t in times) / (len(times) - 1)
            ) ** 0.5

    except Exception:
        result.error = traceback.format_exc()

    return result
