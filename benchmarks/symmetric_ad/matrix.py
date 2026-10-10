"""Run the spec §6 matrix: one subprocess per point, sequentially, on GPU 0,
then a summary table."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

POINTS = list(
    itertools.product(("xxz03", "heis"), ("u1", "z2"), (2, 3), (8, 16), (0, 1))
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--only", default=None, help="substring filter on the point name")
    a = ap.parse_args()
    env = dict(os.environ, JAX_PLATFORMS="cuda,cpu", CUDA_VISIBLE_DEVICES="0")
    rows = []
    for model, sym, D, chi, seed in POINTS:
        name = f"{model}_{sym}_D{D}_chi{chi}_s{seed}"
        if a.only and a.only not in name:
            continue
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
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
        s = (
            json.loads((d / "summary.json").read_text())
            if (d / "summary.json").exists()
            else {}
        )
        rows.append((name, rc, s.get("stopped_at"), s.get("near_degenerate")))
    lines = ["| point | rc | stopped at | near-degenerate |", "|---|---|---|---|"]
    lines += [f"| {n} | {rc} | {st or '-'} | {nd} |" for n, rc, st, nd in rows]
    (a.out / "matrix_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
