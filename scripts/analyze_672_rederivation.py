#!/usr/bin/env python3
"""Turn ``runs/672_rederivation/results.jsonl`` into the #672 re-derivation tables.

Emits, per D: the reach ceiling of each arm at ``recipe=2x2``, the split-vs-dense
peak ratio at matched chi, and a side-by-side against the recorded 2026-07-01
``1x1`` figures that this run re-derives.

Peaks are decimal GB (bytes / 1e9) from ``peak_bytes_in_use``, matching the
original harness; XLA's own OOM messages are in GiB, so the two units appear
side by side in the wall annotations exactly as they did in the original table.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "runs" / "672_rederivation" / "results.jsonl"
GATES = REPO / "runs" / "672_rederivation" / "gate.jsonl"

# The recorded 2026-07-01 numbers this run re-derives (PR #672 body + #673
# ceiling addendum). All at recipe=1x1, BFC allocator, 2x A100.
ORIGINAL = {
    (8, "split", 1): "chi>=128 @ 16.9 GB (ceiling chi=224 @ 51.2 GB, #673)",
    (10, "split", 1): "chi>=128 @ 39.9 GB (ceiling chi=128 @ 37.3 GB, #673)",
    (12, "split", 1): "chi=96 @ 46 GB",
    (8, "dense", 2): "chi=32 @ 16.5 GB",
    (10, "dense", 2): "chi=16 @ 19.64 GB (chi24 OOM)",
    (12, "dense", 2): "cannot compile chi=16 (autotuner)",
}

# Kept identical to ``bench_672_rederivation.WALL``.  When the two disagree,
# the ladder climbs past a cell this reports as a ceiling, and the published
# reach is whichever of the two happened to be looser.
WALL = ("OOM", "COMPILE_FAIL", "FAILED", "TIMEOUT", "NO_OUTPUT")


def load(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main():
    rows = load(RESULTS)
    if not rows:
        print(f"no results at {RESULTS}")
        return 1

    # Only the recipe=2x2, non-autotune0 cells form the re-derived frontier. The
    # 1x1 control rows and the autotune0 probes live in the same file and would
    # otherwise be folded into the per-arm REACH lines -- e.g. reporting the 1x1
    # control's chi=224 as the D=8 split 2x2 reach.
    arms = {}
    for r in rows:
        if r.get("recipe") != "2x2" or r.get("autotune0"):
            continue
        arms.setdefault((r["D"], r["path"], r["n_devices"]), []).append(r)
    for v in arms.values():
        v.sort(key=lambda r: r["chi"])

    print("=" * 78)
    print("#672 RE-DERIVATION @ recipe=2x2 (cuda_async, one cell per process)")
    print("=" * 78)

    for key in sorted(arms):
        D, path, n = key
        cells = arms[key]
        ok = [c for c in cells if c["status"] == "OK"]
        wall = [c for c in cells if c["status"] in WALL]
        reach = max((c["chi"] for c in ok), default=None)
        peak = next((c["peak_gb"] for c in ok if c["chi"] == reach), None)

        print(f"\n--- D={D} {path} {n}-GPU ---")
        for c in cells:
            pk = f"{c['peak_gb']:8.2f} GB" if c.get("peak_gb") is not None else " " * 11
            print(f"    chi={c['chi']:4d}  {c['status']:12s} {pk}  {c['wall_s']:7.1f}s")
        if reach is not None:
            print(f"  REACH @2x2 : chi={reach} @ {peak:.2f} GB")
        if wall:
            w = wall[0]
            print(f"  WALL       : chi={w['chi']} {w['status']}")
            if w.get("detail"):
                print(f"               {w['detail'][:150]}")
        if key in ORIGINAL:
            print(
                f"  RECORDED   : {ORIGINAL[key]}   <- 1x1, refuted-or-confirmed above"
            )

    # split vs dense at matched chi, 1 GPU
    print("\n" + "=" * 78)
    print("SPLIT vs DENSE peak at matched (D, chi), 1 GPU, recipe=2x2")
    print("=" * 78)
    print(f"{'D':>3s} {'chi':>5s} {'split GB':>10s} {'dense GB':>10s} {'ratio':>8s}")
    for D in sorted({r["D"] for r in rows}):
        s = {c["chi"]: c for c in arms.get((D, "split", 1), []) if c["status"] == "OK"}
        d = {c["chi"]: c for c in arms.get((D, "dense", 1), []) if c["status"] == "OK"}
        for chi in sorted(set(s) & set(d)):
            sp, dp = s[chi]["peak_gb"], d[chi]["peak_gb"]
            print(f"{D:3d} {chi:5d} {sp:10.2f} {dp:10.2f} {dp / sp:7.2f}x")

    gates = load(GATES)
    if gates:
        print("\n" + "=" * 78)
        print("COLLAPSE GATE — rank(C1); must be > 1, else the cell is meaningless")
        print("=" * 78)
        for g in sorted(gates, key=lambda g: (g["D"], g["path"], g["chi"])):
            print(
                f"  D={g['D']:2d} {g['path']:6s} chi={g['chi']:4d} "
                f"corner_rank={g.get('corner_rank')}  {g['status']}"
            )
        # An execution failure is not evidence about rank.  Called out
        # separately so a crashed gate cell cannot be read as a collapse --
        # the gate exists to test whether 2x2 collapses, so that conflation
        # would be self-confirming.
        unrun = [g for g in gates if g["status"] in ("GATE_NO_RANK", "GATE_TIMEOUT")]
        if unrun:
            print(
                f"\n  {len(unrun)} gate cell(s) produced no rank "
                "(crash/timeout) — these are UNMEASURED, not collapsed:"
            )
            for g in unrun:
                print(
                    f"    D={g['D']:2d} {g['path']:6s} chi={g['chi']:4d} {g['status']}"
                )
    else:
        print("\n(no collapse-gate results yet)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
