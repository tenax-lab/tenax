"""Plot benchmark JSON results into PNG charts."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_SIZE_ORDER = ["small", "medium", "large"]
_METRICS = {"mean_time_s", "min_time_s", "warmup_time_s"}
_AGGREGATORS = {"mean", "median", "min"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Tenax benchmark JSON results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-glob",
        default="benchmarks/results/*.json",
        help="Input JSON glob (default: benchmarks/results/*.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results/plots",
        help="Output directory for PNG files (default: benchmarks/results/plots)",
    )
    parser.add_argument(
        "--metric",
        default="mean_time_s",
        choices=sorted(_METRICS),
        help="Metric to plot (default: mean_time_s)",
    )
    parser.add_argument(
        "--aggregate",
        default="mean",
        choices=sorted(_AGGREGATORS),
        help="How to combine duplicate entries across files (default: mean)",
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        default=None,
        help="Optional algorithm filter (e.g. dmrg hotrg)",
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        default=None,
        help="Optional backend filter (e.g. cpu cuda)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG DPI (default: 180)",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic y-axis",
    )
    return parser.parse_args()


def _aggregate(values: list[float], method: str) -> float:
    if method == "mean":
        return statistics.mean(values)
    if method == "median":
        return statistics.median(values)
    if method == "min":
        return min(values)
    raise ValueError(f"Unsupported aggregation: {method}")


def _normalize_size_order(sizes: set[str]) -> list[str]:
    known = [s for s in _SIZE_ORDER if s in sizes]
    unknown = sorted(s for s in sizes if s not in _SIZE_ORDER)
    return known + unknown


def _load_records(
    args: argparse.Namespace,
) -> tuple[list[Path], dict[tuple[str, str, str], list[float]]]:
    paths = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise SystemExit(f"No files matched: {args.input_glob}")

    values_by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in paths:
        with path.open() as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            print(f"Skipping non-list JSON file: {path}", file=sys.stderr)
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            if row.get("error"):
                continue

            algorithm = str(row.get("algorithm", ""))
            backend = str(row.get("backend", ""))
            size_label = str(row.get("size_label", ""))

            if not algorithm or not backend or not size_label:
                continue
            if args.algorithm and algorithm not in args.algorithm:
                continue
            if args.backend and backend not in args.backend:
                continue

            try:
                metric_value = float(row.get(args.metric, math.nan))
            except (TypeError, ValueError):
                continue
            if math.isnan(metric_value):
                continue
            values_by_key[(algorithm, backend, size_label)].append(metric_value)

    if not values_by_key:
        raise SystemExit("No usable benchmark rows found after filtering.")
    return paths, values_by_key


def _plot_algorithm(
    algorithm: str,
    backends: list[str],
    sizes: list[str],
    aggregated: dict[tuple[str, str, str], float],
    metric: str,
    use_logy: bool,
    output_dir: Path,
    dpi: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_backends = len(backends)
    width = 0.8 / max(n_backends, 1)
    centers = list(range(len(sizes)))
    offset0 = -0.4 + 0.5 * width

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for i, backend in enumerate(backends):
        x = [c + offset0 + i * width for c in centers]
        y = []
        for size in sizes:
            y.append(aggregated.get((algorithm, backend, size), math.nan))
        ax.bar(x, y, width=width, label=backend)

    ax.set_xticks(centers, sizes)
    ax.set_xlabel("size")
    ax.set_ylabel(f"{metric} (s)")
    if use_logy:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"{algorithm}: {metric}")
    if n_backends > 1:
        ax.legend(title="backend")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{algorithm}_{metric}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    args = _parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Run with "
            "`uv run --with matplotlib python -m benchmarks.plot_results ...`."
        ) from exc

    paths, values_by_key = _load_records(args)
    aggregated = {
        key: _aggregate(vals, args.aggregate) for key, vals in values_by_key.items()
    }

    algorithms = sorted({key[0] for key in aggregated})
    backends = sorted({key[1] for key in aggregated})
    sizes = _normalize_size_order({key[2] for key in aggregated})

    output_dir = Path(args.output_dir)
    print(f"Loaded {len(paths)} file(s), {len(values_by_key)} benchmark group(s).")
    print(f"Metric: {args.metric} | Aggregate: {args.aggregate}")

    for algorithm in algorithms:
        algorithm_backends = sorted(
            {backend for (algo, backend, _size) in aggregated if algo == algorithm}
        )
        algorithm_sizes = _normalize_size_order(
            {size for (algo, _backend, size) in aggregated if algo == algorithm}
        )
        out_path = _plot_algorithm(
            algorithm=algorithm,
            backends=algorithm_backends,
            sizes=algorithm_sizes,
            aggregated=aggregated,
            metric=args.metric,
            use_logy=args.logy,
            output_dir=output_dir,
            dpi=args.dpi,
        )
        print(f"Saved {out_path}")

    if not algorithms or not backends or not sizes:
        raise SystemExit("No plots generated.")


if __name__ == "__main__":
    main()
