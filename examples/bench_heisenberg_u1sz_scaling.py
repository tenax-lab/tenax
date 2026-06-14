#!/usr/bin/env python3
"""U(1)-Sz symmetric vs dense 2-site Heisenberg iPEPS-AD D-chi scaling (#570).

Now that #602 (charged-CTM collapse) is fixed, this script measures whether
U(1)-Sz block-sparsity is a *runtime* lever for 2-site iPEPS-AD, sweeping a
grid of bond dimension ``D`` and environment bond dimension ``chi``.  For each
(D, chi) cell and each arm:

  - **symmetric**: U(1)-Sz ``SymmetricTensor`` init (``heisenberg_u1sz_init_pair``)
  - **dense**:     the *densified same* init (apples-to-apples, like the GO-gate)

it records the final energy plus a compile/runtime timing split:

  - ``compile_s``   wall of a ``gs_num_steps=1`` run (cold XLA compile + 1 step)
  - ``total_s``     wall of the ``gs_num_steps=N`` run
  - ``per_step_s``  warm per-step = ``(total_s - compile_s) / (N - 1)``

The headline metric is the **symmetric / dense per-step ratio** (``speedup``;
>1 means symmetric is faster) as (D, chi) grows, plus the symmetric arm's
block-sparsity (``n_blocks``, ``density``).

Timing note: the 2-site path does not expose ``return_history``, so timing is
external wall-clock.  ``gs_grad_norm_tol`` is set very tight so both arms run
the full ``N`` steps (the per-step split assumes N steps ran).

Usage::

    # CPU smoke test (D=2, chi=8, N=4 ~ a few min)
    JAX_PLATFORMS=cpu uv run python examples/bench_heisenberg_u1sz_scaling.py \\
        --D-list 2 --chi-list 8 --gs-steps 4

    # Scaling grid with JSON checkpointing (A100 recommended for D>=3)
    uv run python examples/bench_heisenberg_u1sz_scaling.py \\
        --D-list 2 3 4 --chi-list 8 16 32 --gs-steps 20 \\
        --json benchmarks/results/u1sz_scaling.json

    # Only the symmetric arm; cap cost to avoid long cells
    uv run python examples/bench_heisenberg_u1sz_scaling.py \\
        --D-list 2 3 --chi-list 8 16 --arms symmetric --max-cost 200
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad  # noqa: E402
from tenax.algorithms.ipeps import (  # noqa: E402
    heisenberg_gate,
    heisenberg_u1sz_init_pair,
)


# ---------------------------------------------------------------------------
# Single arm: time a 1-step (compile) and N-step (total) optimize run
# ---------------------------------------------------------------------------


def _time_arm(gate, init, D: int, chi: int, ctm_iter: int, gs_steps: int) -> dict:
    """Time one arm (symmetric or dense) for a single (D, chi) cell.

    ``init`` is the iPEPS pair passed to ``optimize_gs_ad`` — SymmetricTensors
    for the symmetric arm, raw dense arrays for the dense arm.
    """

    def _cfg(n: int) -> iPEPSConfig:
        return iPEPSConfig(
            max_bond_dim=D,
            ctm=CTMConfig(chi=chi, max_iter=ctm_iter, conv_tol=1e-8),
            gs_num_steps=n,
            unit_cell="2site",
            gs_grad_norm_tol=1e-30,  # effectively never converge -> full n steps
        )

    # 1-step run: dominated by cold XLA compile.
    t0 = time.perf_counter()
    res1 = optimize_gs_ad(gate, init, _cfg(1))
    compile_s = time.perf_counter() - t0

    # N-step run.
    t0 = time.perf_counter()
    resN = optimize_gs_ad(gate, init, _cfg(gs_steps))
    total_s = time.perf_counter() - t0

    E_final = float(resN[2])  # ((A,B), env, E_gs) for the 2-site path
    per_step_s = (
        (total_s - compile_s) / (gs_steps - 1) if gs_steps > 1 else float("nan")
    )
    return {
        "E_final": E_final,
        "compile_s": compile_s,
        "total_s": total_s,
        "per_step_s": per_step_s,
    }


def _block_sparsity(A) -> dict:
    """n_blocks and stored-element density of a SymmetricTensor site tensor."""
    import numpy as np

    blocks = A.blocks
    stored = int(sum(int(np.asarray(b).size) for b in blocks.values()))
    dense_numel = 1
    for ix in A.indices:
        dense_numel *= len(np.asarray(ix.charges))
    return {
        "n_blocks": len(blocks),
        "stored_elems": stored,
        "dense_numel": int(dense_numel),
        "density": stored / dense_numel if dense_numel else float("nan"),
    }


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------


def run_cell(D: int, chi: int, ctm_iter: int, gs_steps: int, arms: list[str]) -> dict:
    """Run the requested arms for one (D, chi) cell; return a metrics row."""
    A_sym, B_sym = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    gate = heisenberg_gate().todense()  # dense gate, identical numerics both arms

    row: dict = {"D": D, "chi": chi, **_block_sparsity(A_sym)}

    if "symmetric" in arms:
        row["sym"] = _time_arm(gate, (A_sym, B_sym), D, chi, ctm_iter, gs_steps)
    if "dense" in arms:
        dense_init = (A_sym.todense(), B_sym.todense())
        row["dense"] = _time_arm(gate, dense_init, D, chi, ctm_iter, gs_steps)

    if "sym" in row and "dense" in row:
        sps = row["sym"]["per_step_s"]
        dps = row["dense"]["per_step_s"]
        row["speedup"] = dps / sps if sps and sps == sps and sps > 0 else float("nan")
        row["dE_sym_dense"] = abs(row["sym"]["E_final"] - row["dense"]["E_final"])
    return row


# ---------------------------------------------------------------------------
# JSON checkpoint helpers
# ---------------------------------------------------------------------------


def _write_json(path: str, meta: dict, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    tmp.rename(p)


def _load_rows(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open() as f:
            return json.load(f).get("rows", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

_HEADER = (
    f"{'D':>3} {'chi':>4} {'blk':>4} {'dens':>5}  "
    f"{'symE':>11} {'sym/step':>9} {'symcc':>7}  "
    f"{'denE':>11} {'den/step':>9} {'dencc':>7}  {'speedup':>7}"
)
_SEP = "-" * len(_HEADER)


def _fmt(x, w, p=3):
    return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) and x == x else f"{'-':>{w}}"


def _print_row(r: dict) -> None:
    if "error" in r:
        print(f"  D={r['D']:>2} chi={r['chi']:>4}  !! {r['error']}")
        return
    sym = r.get("sym", {})
    den = r.get("dense", {})
    print(
        f"{r['D']:>3} {r['chi']:>4} {r.get('n_blocks', 0):>4} "
        f"{_fmt(r.get('density'), 5, 2)}  "
        f"{_fmt(sym.get('E_final'), 11, 6)} {_fmt(sym.get('per_step_s'), 9)} "
        f"{_fmt(sym.get('compile_s'), 7, 1)}  "
        f"{_fmt(den.get('E_final'), 11, 6)} {_fmt(den.get('per_step_s'), 9)} "
        f"{_fmt(den.get('compile_s'), 7, 1)}  {_fmt(r.get('speedup'), 7, 2)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="U(1)-Sz symmetric vs dense 2-site Heisenberg iPEPS-AD "
        "D-chi scaling (#570)."
    )
    parser.add_argument("--D-list", nargs="+", type=int, default=[2], metavar="D")
    parser.add_argument("--chi-list", nargs="+", type=int, default=[8], metavar="CHI")
    parser.add_argument("--gs-steps", type=int, default=8, metavar="N")
    parser.add_argument("--ctm-iter", type=int, default=30, metavar="M")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["symmetric", "dense"],
        default=["symmetric", "dense"],
    )
    parser.add_argument("--json", type=str, default=None, metavar="PATH")
    parser.add_argument(
        "--max-cost",
        type=int,
        default=None,
        metavar="C",
        help="Skip cells where D*D*chi > C",
    )
    args = parser.parse_args()

    try:
        device_kind = jax.devices()[0].device_kind
    except Exception:
        device_kind = "unknown"

    rows = _load_rows(args.json) if args.json else []
    done = {(r["D"], r["chi"]) for r in rows if "error" not in r}

    meta = {
        "platform": platform.node(),
        "device_kind": device_kind,
        "x64": True,
        "arms": args.arms,
        "gs_steps": args.gs_steps,
        "ctm_iter": args.ctm_iter,
        "D_list": sorted(set(args.D_list) | {r["D"] for r in rows}),
        "chi_list": sorted(set(args.chi_list) | {r["chi"] for r in rows}),
    }

    print(f"\nU(1)-Sz vs dense 2-site Heisenberg iPEPS-AD scaling  (#570 / #602)")
    print(
        f"device: {device_kind}  |  arms: {args.arms}  |  "
        f"gs_steps: {args.gs_steps}  |  ctm_iter: {args.ctm_iter}"
    )
    print(_SEP)
    print(_HEADER)
    print(_SEP)
    for r in rows:
        _print_row(r)

    cells = [
        (D, chi)
        for D in args.D_list
        for chi in args.chi_list
        if (D, chi) not in done
    ]
    cells.sort(key=lambda dc: dc[0] * dc[0] * dc[1])

    for D, chi in cells:
        cost = D * D * chi
        if args.max_cost is not None and cost > args.max_cost:
            row = {"D": D, "chi": chi, "error": f"skipped: cost {cost} > {args.max_cost}"}
        else:
            try:
                row = run_cell(D, chi, args.ctm_iter, args.gs_steps, args.arms)
            except Exception as exc:
                row = {"D": D, "chi": chi, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        _print_row(row)
        if args.json:
            _write_json(args.json, meta, rows)

    print(_SEP)
    print(f"Total cells: {len(rows)}  (done this run: {len(cells)})")
    if args.json:
        print(f"Results saved to: {args.json}")


if __name__ == "__main__":
    main()
