"""JSON/CSV serialization and summary table printer for benchmark results."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict

from benchmarks.runner import BenchmarkResult


def save_results_json(
    results: list[BenchmarkResult], path: str, *, quiet: bool = False
) -> None:
    """Save benchmark results to a timestamped JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    if not quiet:
        print(f"Results saved to {path}")


def save_results_csv(results: list[BenchmarkResult], path: str) -> None:
    """Save benchmark results to a flat CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "algorithm",
        "size_label",
        "backend",
        "dtype",
        "warmup_time_s",
        "mean_time_s",
        "std_time_s",
        "min_time_s",
        "num_trials",
        "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "algorithm": r.algorithm,
                    "size_label": r.size_label,
                    "backend": r.backend,
                    "dtype": r.dtype,
                    "warmup_time_s": f"{r.warmup_time_s:.4f}",
                    "mean_time_s": f"{r.mean_time_s:.4f}",
                    "std_time_s": f"{r.std_time_s:.4f}",
                    "min_time_s": f"{r.min_time_s:.4f}",
                    "num_trials": len(r.times_s),
                    "error": r.error or "",
                }
            )
    print(f"CSV saved to {path}")


def print_summary_table(results: list[BenchmarkResult]) -> None:
    """Print an aligned summary table to stdout."""
    header = f"{'Algorithm':<12} {'Size':<8} {'Backend':<8} {'dtype':<10} {'Warmup(s)':>10} {'Mean(s)':>10} {'Std(s)':>10} {'Min(s)':>10} {'Status':<10}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in results:
        status = "ERROR" if r.error else "OK"
        print(
            f"{r.algorithm:<12} {r.size_label:<8} {r.backend:<8} {r.dtype:<10} "
            f"{r.warmup_time_s:>10.3f} {r.mean_time_s:>10.3f} {r.std_time_s:>10.3f} {r.min_time_s:>10.3f} {status:<10}"
        )
    print(sep)
    print()
