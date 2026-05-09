"""Orchestrator: enumerate grid, build payload per point, spawn both runners,
merge JSONs, write report.json + summary.md + trajectory plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .payload import save_payload  # noqa: E402
from .protocol import (  # noqa: E402
    GRID,
    MAX_STEPS,
    SUBPROCESS_TIMEOUT_SEC,
    TOL,
    grid_key,
)
from .su_init import (  # noqa: E402
    build_heisenberg_gate,
    build_sublattice_rotated_gate,
    make_init,
)

_logger = logging.getLogger(__name__)


@dataclass
class PointResult:
    key: str
    path: str
    D: int
    chi: int
    tenax: dict | None
    varipeps: dict | None
    delta_final_energy: float | None
    delta_num_steps: int | None
    tenax_speedup: float | None  # (varipeps total) / (tenax total)


def _run_subprocess(cmd: list[str], log_file: Path, timeout: int) -> tuple[bool, str]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_file.open("w") as lf:
            subprocess.run(
                cmd,
                check=True,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except subprocess.CalledProcessError as exc:
        return False, f"exit code {exc.returncode}; see {log_file}"


def _build_payload(point: dict, results_dir: Path) -> Path:
    key = grid_key(point["path"], point["D"], point["chi"])
    payload = results_dir / f"{key}_payload.npz"
    if payload.exists():
        return payload
    if point["path"] == "single_site":
        gate = build_sublattice_rotated_gate()
    elif point["path"] == "bipartite_2site":
        gate = build_heisenberg_gate()
    else:
        raise ValueError(f"unknown path: {point['path']}")
    init = make_init(path=point["path"], D=point["D"], seed=0)
    save_payload(payload, init=init, gate=gate, meta={**point, "seed": 0})
    return payload


def _run_one_point(point: dict, results_dir: Path, *, force: bool) -> PointResult:
    key = grid_key(point["path"], point["D"], point["chi"])
    payload = _build_payload(point, results_dir)

    tenax_json = results_dir / f"{key}_tenax.json"
    varipeps_json = results_dir / f"{key}_varipeps.json"
    common_args = [
        "--payload",
        str(payload),
        "--path",
        point["path"],
        "--D",
        str(point["D"]),
        "--chi",
        str(point["chi"]),
        "--tol",
        str(TOL),
        "--max-steps",
        str(MAX_STEPS),
    ]

    if force or not tenax_json.exists():
        ok, msg = _run_subprocess(
            [
                sys.executable,
                "-m",
                "benchmarks.varipeps_compare.run_tenax",
                *common_args,
                "--out",
                str(tenax_json),
            ],
            results_dir / f"{key}_tenax.log",
            SUBPROCESS_TIMEOUT_SEC,
        )
        if not ok:
            tenax_json.write_text(json.dumps({"status": "error", "msg": msg}))

    if force or not varipeps_json.exists():
        ok, msg = _run_subprocess(
            [
                sys.executable,
                "-m",
                "benchmarks.varipeps_compare.run_varipeps",
                *common_args,
                "--out",
                str(varipeps_json),
            ],
            results_dir / f"{key}_varipeps.log",
            SUBPROCESS_TIMEOUT_SEC,
        )
        if not ok:
            varipeps_json.write_text(json.dumps({"status": "error", "msg": msg}))

    tenax = json.loads(tenax_json.read_text())
    varipeps = json.loads(varipeps_json.read_text())

    if "final_energy" in tenax and "final_energy" in varipeps:
        delta_e = tenax["final_energy"] - varipeps["final_energy"]
        delta_n = tenax["num_steps"] - varipeps["num_steps"]
        speedup = varipeps["total_wall_clock"] / tenax["total_wall_clock"]
    else:
        delta_e = delta_n = speedup = None

    return PointResult(
        key=key,
        path=point["path"],
        D=point["D"],
        chi=point["chi"],
        tenax=tenax,
        varipeps=varipeps,
        delta_final_energy=delta_e,
        delta_num_steps=delta_n,
        tenax_speedup=speedup,
    )


def _plot_trajectory(point: PointResult, out_png: Path):
    if "energy_history" not in (point.tenax or {}) or "energy_history" not in (
        point.varipeps or {}
    ):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        point.tenax["energy_history"],
        "o-",
        label=f"tenax ({point.tenax['lib_version']})",
    )
    ax.plot(
        point.varipeps["energy_history"],
        "x-",
        label=f"varipeps ({point.varipeps['lib_version']})",
    )
    ax.set_xlabel("AD step")
    ax.set_ylabel("E / site")
    ax.set_title(f"{point.path}, D={point.D}, χ={point.chi}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _write_summary(results: list[PointResult], out_md: Path):
    lines = [
        "# Tenax ↔ variPEPS — Square Heisenberg AFM benchmark",
        "",
        "| key | path | D | χ | E_tenax | E_varipeps | ΔE | n_tenax | n_vp | t_tenax (s) | t_vp (s) | speedup |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if (
            r.tenax
            and "final_energy" in r.tenax
            and r.varipeps
            and "final_energy" in r.varipeps
        ):
            lines.append(
                f"| {r.key} | {r.path} | {r.D} | {r.chi} "
                f"| {r.tenax['final_energy']:.8f} | {r.varipeps['final_energy']:.8f} "
                f"| {r.delta_final_energy:+.2e} "
                f"| {r.tenax['num_steps']} | {r.varipeps['num_steps']} "
                f"| {r.tenax['total_wall_clock']:.1f} | {r.varipeps['total_wall_clock']:.1f} "
                f"| {r.tenax_speedup:.2f}x |"
            )
        else:
            lines.append(
                f"| {r.key} | {r.path} | {r.D} | {r.chi} | error | error | — | — | — | — | — | — |"
            )
    out_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--results-dir", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--force", action="store_true", help="Re-run even if JSON exists")
    args = ap.parse_args()

    import os

    os.environ["JAX_PLATFORMS"] = args.device

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(results_dir / "run.log"),
            logging.StreamHandler(),
        ],
    )

    all_points: list[PointResult] = []
    for point in GRID:
        _logger.info("running %s", grid_key(point["path"], point["D"], point["chi"]))
        try:
            r = _run_one_point(point, results_dir, force=args.force)
        except Exception:
            _logger.exception("orchestrator failure on %s", point)
            continue
        all_points.append(r)
        _plot_trajectory(r, results_dir / f"{r.key}_trajectory.png")

    report = {r.key: asdict(r) for r in all_points}
    (results_dir / "report.json").write_text(json.dumps(report, indent=2))
    _write_summary(all_points, results_dir / "summary.md")
    _logger.info("done — report at %s", results_dir / "summary.md")


if __name__ == "__main__":
    main()
