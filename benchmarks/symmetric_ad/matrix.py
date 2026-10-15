"""Run the spec §6 matrix: one subprocess per point, sequentially, on GPU 0
(or ``$MATRIX_GPU``), then a summary table."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.symmetric_ad.codehash import code_hash

POINTS = list(
    itertools.product(("xxz03", "heis"), ("u1", "z2"), (2, 3), (8, 16), (0, 1))
)


def _read(d, name):
    f = d / name
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def _complete(d, chash, accepted=()):
    """Done when complete and its hash is the current one or an accepted one."""
    s = _read(d, "summary.json")
    return (
        s is not None
        and bool(s.get("complete"))
        and s.get("code_hash") in (chash, *accepted)
    )


def _row(name, rc, d):
    s = _read(d, "summary.json") or {}
    stages = []
    for k in ("S1", "S2", "S3", "S4", "S5"):
        r = _read(d, f"{k}.json")
        stages.append("?" if r is None else ("P" if r["passed"] else "F"))
    if rc != 0 or not s.get("complete"):
        status = f"CRASHED(rc={rc})"
    elif s.get("stopped_at"):
        err = " error" if s.get("error_stage") else ""
        status = f"FAILED at {s['stopped_at']}{err}"
    else:
        status = "vacuous (chi not binding)" if s.get("vacuous") else "PASS"
    return (
        name,
        status,
        "".join(stages),
        s.get("near_degenerate"),
        s.get("vacuous"),
        s.get("code_hash"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--only", default=None, help="substring filter on the point name")
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="rerun points that already have a complete summary of the current code",
    )
    ap.add_argument(
        "--accept-hash",
        action="append",
        default=[],
        metavar="HASH",
        help="also count a complete summary of this code hash as done (repeatable)",
    )
    a = ap.parse_args()
    chash = code_hash()
    env = dict(
        os.environ,
        JAX_PLATFORMS="cuda,cpu",
        CUDA_VISIBLE_DEVICES=os.environ.get("MATRIX_GPU", "0"),
    )
    rows = []
    for model, sym, D, chi, seed in POINTS:
        name = f"{model}_{sym}_D{D}_chi{chi}_s{seed}"
        if a.only and a.only not in name:
            continue
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        if not a.no_resume and _complete(d, chash, a.accept_hash):
            rows.append(_row(name, 0, d))
            continue
        for f in (
            "summary.json",
            "S1.json",
            "S2.json",
            "S3.json",
            "S4.json",
            "S5.json",
        ):
            (d / f).unlink(missing_ok=True)  # never report a stale result
        with open(d / "run.log", "w") as log:
            rc = subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.symmetric_ad.pipeline",
                    "--model",
                    model,
                    "--sym",
                    sym,
                    "--D",
                    str(D),
                    "--chi",
                    str(chi),
                    "--seed",
                    str(seed),
                    "--cache-dir",
                    str(a.cache_dir),
                    "--out",
                    str(d),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        rows.append(_row(name, rc, d))
    lines = [
        "| point | status | S1-S5 (P/F/?) | near-degenerate | vacuous | code_hash |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {n} | {st} | {sg} | {nd} | {v} | {h} |" for n, st, sg, nd, v, h in rows
    ]
    (a.out / "matrix_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
