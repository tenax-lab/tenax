#!/usr/bin/env python3
"""Dense 2D-Heisenberg iPEPS-AD characterization study (issue #570).

Sweeps Tenax's dense single-site (C4v, sigma gauge) iPEPS-AD over a grid of
bond dimension D and environment bond dimension chi.  For each (D, chi) cell
the script records:

  - Final variational energy E_final and deviation dE = E_final - REF_ENERGY
    from the QMC reference E/site ~ -0.6694430.
  - Cold XLA-compile time (first L-BFGS step).
  - Total wall time and warm step time (median of steps 1+).
  - Number of steps taken and convergence flag.

Purpose: establish a runtime / accuracy baseline for the dense path before
evaluating block-sparse QR-CTMRG speedups in Phase 3 of issue #570.

Usage::

    # CPU smoke test (D=2, chi=8, 30 steps ~ 1-2 min)
    JAX_PLATFORMS=cpu uv run python examples/bench_heisenberg_largeD.py \\
        --D-list 2 --chi-list 8 --gs-steps 30

    # Full grid with JSON checkpointing (A100 recommended for D>=3)
    uv run python examples/bench_heisenberg_largeD.py \\
        --D-list 2 3 4 --chi-list 8 16 32 --gs-steps 200 \\
        --json results/heisenberg_largeD_a100.json

    # Resume an interrupted run (skips already-completed cells)
    uv run python examples/bench_heisenberg_largeD.py \\
        --D-list 2 3 4 --chi-list 8 16 32 --gs-steps 200 \\
        --json results/heisenberg_largeD_a100.json

    # Limit cost to avoid long-running cells
    uv run python examples/bench_heisenberg_largeD.py \\
        --D-list 2 3 --chi-list 8 16 32 --gs-steps 100 --max-cost 200
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from tenax import (  # noqa: E402
    CTMConfig,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
    sublattice_rotate_gate,
)

# ---------------------------------------------------------------------------
# Reference energy (QMC)
# ---------------------------------------------------------------------------

REF_ENERGY = -0.6694430


# ---------------------------------------------------------------------------
# Problem builder
# ---------------------------------------------------------------------------


def build_problem(D: int, chi: int, gs_steps: int):
    """Build gate, config, and A_init for a single (D, chi) cell.

    Returns (gate, A_init, config).  A_init=None lets optimize_gs_ad
    run simple-update initialization (su_init=True).
    """
    gate = sublattice_rotate_gate(heisenberg_gate())
    ctm = CTMConfig(
        chi=chi,
        max_iter=100,
        conv_tol=1e-8,
        # forward_gauge defaults to "phase" — required for implicit AD
        # (validate_ctm_for_implicit_ad enforces phase+svd+elementwise)
    )
    config = iPEPSConfig(
        max_bond_dim=D,
        ctm=ctm,
        gs_c4v=True,
        gs_num_steps=gs_steps,
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e-5,
        su_init=True,
        return_history=True,
    )
    A_init = None
    return gate, A_init, config


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------


def run_cell(D: int, chi: int, gs_steps: int) -> dict:
    """Run one (D, chi) optimization cell and return a metrics dict."""
    gate, A_init, config = build_problem(D, chi, gs_steps)

    t_wall_start = time.perf_counter()
    result = optimize_gs_ad(gate, A_init, config)
    total_wall_s = time.perf_counter() - t_wall_start

    # Unpack: with return_history=True the result is (A, env, E_gs, history)
    _A, _env, E_gs, history = result

    energies = history["energies"]
    step_times = history["step_times"]
    jit_compile_s = float(history["jit_compile_time"])
    num_steps = int(history["num_steps"])
    converged = bool(history["converged"])

    E_final = float(min(energies)) if energies else float("nan")
    dE = E_final - REF_ENERGY

    # Warm step = median of all step_times: the history accumulator stores
    # jit_compile_time separately (step 0) and step_times contains only steps
    # 1+ (all "warm" — already compiled), so no further slicing needed.
    warm_step_s = statistics.median(step_times) if step_times else float("nan")

    return {
        "D": D,
        "chi": chi,
        "E_final": E_final,
        "dE": dE,
        "jit_compile_s": jit_compile_s,
        "total_wall_s": total_wall_s,
        "warm_step_s": warm_step_s,
        "num_steps": num_steps,
        "converged": converged,
        "below_ref": E_final < REF_ENERGY,
    }


# ---------------------------------------------------------------------------
# JSON checkpoint helpers
# ---------------------------------------------------------------------------


def _write_json(path: str, meta: dict, rows: list[dict]) -> None:
    """Write meta + rows to a JSON file (atomic-ish via temp rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    tmp.rename(p)


def _load_rows(path: str) -> list[dict]:
    """Load rows from a JSON checkpoint; return [] if missing or invalid."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open() as f:
            data = json.load(f)
        return data.get("rows", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_HEADER = (
    f"{'D':>3}  {'chi':>4}  {'E_final':>12}  {'dE':>10}  "
    f"{'jit_s':>7}  {'wall_s':>7}  {'step_s':>7}  {'steps':>5}  "
    f"{'conv':>5}  {'<ref':>5}"
)
_SEP = "-" * len(_HEADER)


def _print_row(r: dict) -> None:
    """Print a single result row (handles error rows)."""
    if "error" in r:
        print(f"  D={r['D']:>2}  chi={r['chi']:>4}  !! {r['error']}")
        return
    conv_s = "yes" if r.get("converged") else "no"
    ref_s = "yes" if r.get("below_ref") else "no"
    jit = r.get("jit_compile_s", float("nan"))
    wall = r.get("total_wall_s", float("nan"))
    step = r.get("warm_step_s", float("nan"))
    print(
        f"{r['D']:>3}  {r['chi']:>4}  {r['E_final']:>12.7f}  {r['dE']:>+10.6f}  "
        f"{jit:>7.1f}  {wall:>7.1f}  {step:>7.3f}  {r['num_steps']:>5}  "
        f"{conv_s:>5}  {ref_s:>5}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dense 2D-Heisenberg iPEPS-AD characterization study (#570)."
    )
    parser.add_argument(
        "--D-list",
        nargs="+",
        type=int,
        default=[2],
        metavar="D",
        help="Bond dimensions to sweep (default: 2)",
    )
    parser.add_argument(
        "--chi-list",
        nargs="+",
        type=int,
        default=[8],
        metavar="CHI",
        help="Environment bond dimensions to sweep (default: 8)",
    )
    parser.add_argument(
        "--gs-steps",
        type=int,
        default=100,
        metavar="N",
        help="Max GS-AD optimizer steps per cell (default: 100)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="JSON file path for checkpointing/resuming results",
    )
    parser.add_argument(
        "--max-cost",
        type=int,
        default=None,
        metavar="C",
        help="Skip cells where D*D*chi > C (default: no limit)",
    )
    args = parser.parse_args()

    # Build metadata
    try:
        device_kind = jax.devices()[0].device_kind
    except Exception:
        device_kind = "unknown"

    meta = {
        "platform": platform.node(),
        "device_kind": device_kind,
        "x64": True,
        "ref_energy": REF_ENERGY,
        "D_list": args.D_list,
        "chi_list": args.chi_list,
        "gs_steps": args.gs_steps,
    }

    # Load existing rows (resume support)
    rows: list[dict] = _load_rows(args.json) if args.json else []
    done: set[tuple[int, int]] = {
        (r["D"], r["chi"]) for r in rows if "error" not in r
    }

    # Print header
    print(f"\nDense Heisenberg iPEPS-AD sweep  |  ref E = {REF_ENERGY}")
    print(f"device: {device_kind}  |  x64: True  |  gs_steps: {args.gs_steps}")
    if args.max_cost is not None:
        print(f"max-cost: {args.max_cost}  (D*D*chi)")
    print(_SEP)
    print(_HEADER)
    print(_SEP)

    # Reprint resumed rows
    for r in rows:
        _print_row(r)

    # Build work list: cross-product sorted cheap-first (D^2 * chi)
    all_cells = [
        (D, chi)
        for D in args.D_list
        for chi in args.chi_list
        if (D, chi) not in done
    ]
    all_cells.sort(key=lambda dc: dc[0] * dc[0] * dc[1])

    # Run each cell
    for D, chi in all_cells:
        cost = D * D * chi
        if args.max_cost is not None and cost > args.max_cost:
            row: dict = {
                "D": D,
                "chi": chi,
                "error": f"skipped: cost {cost} > {args.max_cost}",
            }
        else:
            try:
                row = run_cell(D, chi, args.gs_steps)
            except Exception as exc:
                row = {"D": D, "chi": chi, "error": f"{type(exc).__name__}: {exc}"}

        rows.append(row)
        _print_row(row)

        if args.json:
            _write_json(args.json, meta, rows)

    print(_SEP)
    print(f"Total cells: {len(rows)}  (done this run: {len(all_cells)})")
    if args.json:
        print(f"Results saved to: {args.json}")


if __name__ == "__main__":
    main()
