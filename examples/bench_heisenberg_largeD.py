#!/usr/bin/env python3
"""Dense 2D-Heisenberg iPEPS-AD characterization study (issue #570).

Sweeps Tenax's dense single-site (C4v, phase gauge, implicit-AD) iPEPS over a
grid of bond dimension D and environment bond dimension chi.  For each (D, chi)
cell the script records:

  - Final variational energy E_final and deviation dE = E_final - REF_ENERGY
    from the QMC reference E/site ~ -0.6694430.
  - Cold XLA-compile time (first L-BFGS step).
  - Total wall time and warm step time (median of steps 1+).
  - Number of steps taken and convergence flag.

Purpose: establish a runtime / accuracy baseline for the dense path before
evaluating block-sparse QR-CTMRG speedups in Phase 3 of issue #570.

Anchor: D=2, chi=8, 100 steps → E ~ -0.6602 (implicit-AD / phase-gauge path).
QMC reference: E/site ~ -0.6694430.  Note: the ``heisenberg_ipeps_ad.py``
example docstring quotes E ~ -0.6625 for a sigma+eigh+gmres config that is
incompatible with the current API (``validate_ctm_for_implicit_ad`` enforces
phase+svd+elementwise); the reachable energy on the implicit-AD path is -0.6602.

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
        num_imaginary_steps=200,
        dt=0.05,
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

    # E_final: the energy of the actual final state returned by the
    # optimizer (E_gs).  best_seen tracks the lowest energy recorded over
    # the trajectory — only used for the variational-floor watch below, so a
    # CTM-noisy transient can't be mistaken for the reported final energy.
    E_final = float(E_gs)
    # Lowest energy anywhere — the final state (E_gs) is often lower than any
    # value in history["energies"], so include it in the minimum.
    best_seen = min([E_final, *map(float, energies)]) if energies else E_final
    dE = E_final - REF_ENERGY

    # grad_eval_s: history["step_times"] is stamped immediately after the
    # jax.value_and_grad(loss_fn)(params) call, BEFORE the L-BFGS search
    # direction + Hager-Zhang line search run, so its median is a gradient-
    # evaluation time, NOT a full optimizer step.  The honest per-step wall is
    # total_wall_s / num_steps (which includes line-search CTM re-evaluations).
    grad_eval_s = statistics.median(step_times) if step_times else float("nan")
    wall_per_step_s = total_wall_s / num_steps if num_steps else float("nan")

    return {
        "D": D,
        "chi": chi,
        "E_final": E_final,
        "best_seen": best_seen,
        "dE": dE,
        "jit_compile_s": jit_compile_s,
        "total_wall_s": total_wall_s,
        "wall_per_step_s": wall_per_step_s,
        "grad_eval_s": grad_eval_s,
        "num_steps": num_steps,
        "converged": converged,
        # Variational-floor watch: flag if the lowest energy *anywhere* on the
        # trajectory dips below the QMC reference (signals a normalization/gauge
        # bug), using best_seen rather than just the final state.
        "below_ref": best_seen < REF_ENERGY,
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
        return [_backfill_row(r) for r in data.get("rows", [])]
    except Exception:
        return []


def _backfill_row(r: dict) -> dict:
    """Upgrade a legacy checkpoint row to the current schema in place.

    Rows written by an earlier version carry ``warm_step_s`` but not
    ``wall_per_step_s``. The per-step wall is derivable as
    ``total_wall_s / num_steps``, so backfill it (and drop the stale
    ``warm_step_s`` label) before the row is reprinted or rewritten —
    otherwise resumed legacy rows print ``nan`` for ``w/step`` and a
    rewrite would persist a mixed schema. The energy fields are left as
    stored: ``E_gs`` is not recoverable for legacy rows, and the
    difference from the old ``min(energies)`` is negligible (5th–6th
    decimal).
    """
    if "error" in r or "wall_per_step_s" in r:
        return r
    total = r.get("total_wall_s")
    steps = r.get("num_steps")
    if total is not None and steps:
        r["wall_per_step_s"] = total / steps
    r.pop("warm_step_s", None)
    return r


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_HEADER = (
    f"{'D':>3}  {'chi':>4}  {'E_final':>12}  {'dE':>10}  "
    f"{'jit_s':>7}  {'wall_s':>7}  {'w/step':>7}  {'steps':>5}  "
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
    # Per-step wall (full optimizer step incl. line search), not grad-eval.
    w_step = r.get("wall_per_step_s", float("nan"))
    print(
        f"{r['D']:>3}  {r['chi']:>4}  {r['E_final']:>12.7f}  {r['dE']:>+10.6f}  "
        f"{jit:>7.1f}  {wall:>7.1f}  {w_step:>7.3f}  {r['num_steps']:>5}  "
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

    # Load existing rows (resume support)
    rows: list[dict] = _load_rows(args.json) if args.json else []
    done: set[tuple[int, int]] = {
        (r["D"], r["chi"]) for r in rows if "error" not in r
    }

    # Build metadata.  On a resume with a narrower CLI selection, the file may
    # already hold rows outside the current --D-list/--chi-list; record the
    # UNION of the D/chi actually present (existing rows ∪ this run's request)
    # so the top-level metadata describes the whole sweep, not just this call.
    d_union = sorted(set(args.D_list) | {r["D"] for r in rows})
    chi_union = sorted(set(args.chi_list) | {r["chi"] for r in rows})
    meta = {
        "platform": platform.node(),
        "device_kind": device_kind,
        "x64": True,
        "ref_energy": REF_ENERGY,
        "D_list": d_union,
        "chi_list": chi_union,
        "gs_steps": args.gs_steps,
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
