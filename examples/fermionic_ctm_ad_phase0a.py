#!/usr/bin/env python3
"""Phase 0a campaign driver for the fermionic CTM-AD reform (PR #986).

Runs the GO/NO-GO measurement the design mandates before any reform code:
the fermionic-minus-bosonic *delta* at matched block structure is the only
cost swap gates + retyping can remove, so it is measured first, under the
design's pinned protocol:

* arms       : ``fermionic`` vs ``z2boson`` (the same tensor retyped onto
               bosonic ZnSymmetry(2) — matched block structure by
               construction; see ``profile_ctm_ad_wall_566.py``)
* unit cells : 1site AND the 2site checkerboard (Phase 5's primary workload)
* D          : 2 and 3;  seeds: 42 and 43;  chi = 16
* engine     : the fused Neumann VJP in BOTH arms (production implicit
               default — same-engine parity)
* fixed count: conv_tol=1e-30 so every arm runs exactly --depth sweeps
               (iteration-matched timing; the raw wall delta then measures
               graded-branch overhead, not convergence-rate differences)
* hygiene    : ONE FRESH PROCESS PER CELL (no _JIT_STEP_CACHE / _VJP_CACHE
               reuse across cells), per-cold-call empty persistent-compile
               cache dir (inside the harness), block_until_ready,
               JAX_PLATFORMS=cpu pinned, compile vs post-warm separated.

Verdict: NO-GO if the median fermionic-minus-z2boson value_and_grad compile
delta is below 30% of the fermionic compile wall (the reform cannot pay);
GO otherwise. Both numbers print per cell pair and in the summary.

Usage::

    uv run python examples/fermionic_ctm_ad_phase0a.py --outdir phase0a_out
    # smoke (one cheap cell pair):
    uv run python examples/fermionic_ctm_ad_phase0a.py --outdir /tmp/p0a \
        --D 2 --seeds 42 --unit-cells 1site --depth 4 --chi 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).parent / "profile_ctm_ad_wall_566.py"


def run_cell(outdir: Path, sym, D, seed, unit_cell, chi, depth) -> dict:
    """One harness cell in a fresh subprocess; returns its single row."""
    tag = f"{sym}_D{D}_s{seed}_{unit_cell}"
    cell_json = outdir / f"{tag}.json"
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    cmd = [
        sys.executable,
        str(HARNESS),
        "--D",
        str(D),
        "--chi",
        str(chi),
        "--depth",
        str(depth),
        "--sym",
        sym,
        "--seed",
        str(seed),
        "--unit-cell",
        unit_cell,
        "--conv-tol",
        "1e-30",
        "--json",
        str(cell_json),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0 or not cell_json.exists():
        return {
            "sym": sym,
            "D": D,
            "seed": seed,
            "unit_cell": unit_cell,
            "error": f"exit={proc.returncode}",
            "stderr_tail": proc.stderr[-800:],
            "cell_process_wall_s": wall,
        }
    data = json.loads(cell_json.read_text())
    row = data["rows"][0]
    row["cell_process_wall_s"] = wall
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--D", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument(
        "--unit-cells",
        nargs="+",
        default=["1site", "2site"],
        choices=["1site", "2site"],
    )
    ap.add_argument("--chi", type=int, default=16)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--nogo-fraction", type=float, default=0.30)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cells = [
        (sym, D, seed, uc)
        for D in args.D
        for seed in args.seeds
        for uc in args.unit_cells
        for sym in ("fermionic", "z2boson")
    ]
    print(f"# Phase 0a: {len(cells)} cells, chi={args.chi}, depth={args.depth}")
    rows = []
    for sym, D, seed, uc in cells:
        print(f"-- {sym:>9} D={D} seed={seed} {uc} ...", flush=True)
        row = run_cell(args.outdir, sym, D, seed, uc, args.chi, args.depth)
        rows.append(row)
        status = (
            f"ERROR {row['error']}"
            if "error" in row
            else f"vg_compile={row['vg_compile_s']:.2f}s "
            f"vg_wall={row['vg_wall_s']:.2f}s warm={row['warm_step_s'] * 1e3:.0f}ms "
            f"blocks={row['n_blocks']} grad_finite={row['grad_finite']}"
        )
        print(f"   {status}", flush=True)
        (args.outdir / "campaign.json").write_text(
            json.dumps({"chi": args.chi, "depth": args.depth, "rows": rows}, indent=2)
        )

    # ── delta analysis: pair fermionic vs z2boson per (D, seed, unit_cell) ──
    def key(r):
        return (r["D"], r["seed"], r["unit_cell"])

    ferm = {key(r): r for r in rows if r.get("sym") == "fermionic" and "error" not in r}
    bos = {key(r): r for r in rows if r.get("sym") == "z2boson" and "error" not in r}
    pairs, fracs = [], []
    print("\n#  D seed cell    ferm_cmp  z2b_cmp   delta  delta/ferm  warm f/z2b")
    for k in sorted(ferm):
        if k not in bos:
            continue
        f, b = ferm[k], bos[k]
        delta = f["vg_compile_s"] - b["vg_compile_s"]
        frac = delta / f["vg_compile_s"] if f["vg_compile_s"] > 0 else float("nan")
        warm_ratio = (
            f["warm_step_s"] / b["warm_step_s"] if b["warm_step_s"] else float("nan")
        )
        fracs.append(frac)
        pairs.append(
            {
                "key": list(k),
                "delta_s": delta,
                "delta_fraction": frac,
                "warm_ratio": warm_ratio,
            }
        )
        print(
            f"  {k[0]:>2} {k[1]:>4} {k[2]:>6} {f['vg_compile_s']:>8.2f} "
            f"{b['vg_compile_s']:>8.2f} {delta:>7.2f} {frac:>10.2%} "
            f"{warm_ratio:>10.2f}"
        )

    verdict = "NO-DATA"
    med = float("nan")
    if fracs:
        med = statistics.median(fracs)
        verdict = "GO" if med >= args.nogo_fraction else "NO-GO"
    print(
        f"\n  median delta fraction = {med:.2%}  "
        f"(threshold {args.nogo_fraction:.0%})  ==> {verdict}"
    )
    (args.outdir / "campaign.json").write_text(
        json.dumps(
            {
                "chi": args.chi,
                "depth": args.depth,
                "rows": rows,
                "pairs": pairs,
                "median_delta_fraction": med,
                "nogo_fraction": args.nogo_fraction,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"  campaign JSON -> {args.outdir / 'campaign.json'}")


if __name__ == "__main__":
    main()
