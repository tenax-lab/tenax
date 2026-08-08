#!/usr/bin/env python3
"""#672 re-derivation driver: the (D, chi) value_and_grad frontier at recipe=2x2.

The 2026-07-01 #672 benchmark concluded "split-CTM on 1 GPU dominates dense
multi-GPU everywhere on the frontier" (D=10: split chi>=128 vs dense-2GPU chi16;
D=12: split chi96 vs dense compile-fail).  Both arms ran ``recipe="1x1"``, whose
corner-pair projector collapses the CTM environment to a rank-1 corner
(#723/#726) -- so every reach number describes how cheaply a chi_eff=1
mean-field boundary scales, not the reach of a converged CTM (#747).

This walks a chi ladder per (path, D, n_devices) at ``recipe=2x2``, one cell per
process for a clean cumulative peak (the shard-reach protocol), and stops the
ladder at the first wall.  It distinguishes the two wall types the original run
found:

    OOM             RESOURCE_EXHAUSTED -- a memory wall
    COMPILE_FAIL    XLA autotuner failure on the giant (chi*D) gemm

Results append to ``<outdir>/results.jsonl``; a completed cell is cached and
skipped on re-run, so the ladder is resumable.  A cell that hits the wall is
cached too -- delete its line to retry.

Usage:
    uv run python scripts/bench_672_rederivation.py \
        --path split --D 10 --gpus 0 --chi 16 24 32 48 64 96 128 160
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
DRIVER = REPO / "examples" / "bench_ctm_frontier_grad.py"


def _cell_key(path, D, chi, n_dev, recipe, chunk, autotune0=False):
    key = f"{path}|D{D}|chi{chi}|n{n_dev}|{recipe}|chunk{chunk}"
    return key + "|at0" if autotune0 else key


def _load_done(results_path):
    done = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[rec["key"]] = rec
    return done


def _classify(stdout, returncode, timed_out):
    """Map driver output to (status, peak_gb, energy, gnorm, detail)."""
    if timed_out:
        return "TIMEOUT", None, None, None, "wall-clock timeout"
    for line in stdout.splitlines():
        if line.startswith("path=") and " OK " in line:
            peak = energy = gnorm = None
            for tok in line.split():
                if tok.startswith("E="):
                    energy = float(tok[2:])
                elif tok.startswith("|g|="):
                    gnorm = float(tok[4:])
                elif tok.startswith("per_device_peak="):
                    peak = float(tok.split("=")[1])
            return "OK", peak, energy, gnorm, line.strip()
        if line.startswith("path=") and "FAILED(" in line:
            detail = line.strip()
            low = detail.lower()
            if "resource_exhausted" in low or "out of memory" in low:
                return "OOM", None, None, None, detail
            if "autotun" in low or "no valid" in low or "internal" in low:
                return "COMPILE_FAIL", None, None, None, detail
            return "FAILED", None, None, None, detail
        if line.startswith("path=") and "SKIP" in line:
            return "SKIP", None, None, None, line.strip()
    return "NO_OUTPUT", None, None, None, (stdout[-400:] if stdout else "")


def run_cell(
    *, path, D, chi, gpus, recipe, chunk, shard, max_iter, timeout_s, gate, autotune0
):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
    if autotune0:
        # The (chi*D) giant-transpose/gemm autotuner failure is a SOFT wall:
        # disabling autotuning recovers configs that fit in memory but that XLA
        # refused to compile (#673).  Probe it before reporting any COMPILE_FAIL
        # as a ceiling -- otherwise the reported reach is an artifact of a knob.
        prev = env.get("XLA_FLAGS", "")
        env["XLA_FLAGS"] = (prev + " --xla_gpu_autotune_level=0").strip()

    cmd = [
        "uv",
        "run",
        "python",
        str(DRIVER),
        "--path",
        path,
        "--D",
        str(D),
        "--chi",
        str(chi),
        "--recipe",
        recipe,
        "--max-iter",
        str(max_iter),
    ]
    if shard:
        cmd.append("--shard")
    if chunk:
        cmd += ["--chunk", str(chunk)]
    if gate:
        cmd.append("--gate")

    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout_s
        )
        out = proc.stdout + "\n" + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as ex:
        out = (ex.stdout or "") if isinstance(ex.stdout, str) else ""
        rc = -1
        timed_out = True
    wall = time.perf_counter() - t0

    if gate:
        rank = None
        for line in out.splitlines():
            if line.startswith("GATE") and "corner_rank=" in line:
                for tok in line.split():
                    if tok.startswith("corner_rank="):
                        rank = int(tok.split("=")[1].split("/")[0])
        status = "GATE_OK" if rank and rank > 1 else "GATE_COLLAPSED"
        return {"status": status, "corner_rank": rank, "wall_s": round(wall, 1)}

    status, peak, energy, gnorm, detail = _classify(out, rc, timed_out)
    return {
        "status": status,
        "peak_gb": peak,
        "energy": energy,
        "gnorm": gnorm,
        "wall_s": round(wall, 1),
        "detail": detail[:300],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["dense", "split"], required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--chi", type=int, nargs="+", required=True)
    ap.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES value")
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--recipe", choices=["1x1", "2x2"], default="2x2")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--timeout-s", type=int, default=7200)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument(
        "--autotune0",
        action="store_true",
        help="set --xla_gpu_autotune_level=0 to probe whether a COMPILE_FAIL "
        "is a soft autotuner wall rather than a real ceiling (#673)",
    )
    ap.add_argument("--outdir", default="runs/672_rederivation")
    ap.add_argument(
        "--no-stop",
        action="store_true",
        help="keep climbing past a wall instead of stopping the ladder",
    )
    args = ap.parse_args()

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / ("gate.jsonl" if args.gate else "results.jsonl")
    done = _load_done(results_path)

    n_dev = len(args.gpus.split(",")) if args.shard else 1
    label = f"{args.path} D={args.D} n={n_dev} recipe={args.recipe}"
    print(f"### ladder: {label} chi={args.chi} gpus={args.gpus}", flush=True)

    for chi in args.chi:
        key = _cell_key(
            args.path, args.D, chi, n_dev, args.recipe, args.chunk, args.autotune0
        )
        if key in done:
            rec = done[key]
            print(f"  [cached] chi={chi:4d} {rec['status']}", flush=True)
            if not args.no_stop and rec["status"] in ("OOM", "COMPILE_FAIL", "FAILED"):
                print(f"  ladder stops at chi={chi} ({rec['status']}, cached)")
                break
            continue

        print(f"  running chi={chi} ...", flush=True)
        res = run_cell(
            path=args.path,
            D=args.D,
            chi=chi,
            gpus=args.gpus,
            recipe=args.recipe,
            chunk=args.chunk,
            shard=args.shard,
            max_iter=args.max_iter,
            timeout_s=args.timeout_s,
            gate=args.gate,
            autotune0=args.autotune0,
        )
        rec = {
            "key": key,
            "path": args.path,
            "D": args.D,
            "chi": chi,
            "n_devices": n_dev,
            "recipe": args.recipe,
            "chunk": args.chunk,
            "autotune0": args.autotune0,
            "gpus": args.gpus,
            **res,
        }
        with results_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

        peak = rec.get("peak_gb")
        peak_s = f"{peak:.2f} GB" if peak is not None else "-"
        print(
            f"  chi={chi:4d} {rec['status']:12s} peak={peak_s:>10s} "
            f"wall={rec['wall_s']}s",
            flush=True,
        )
        if not args.no_stop and rec["status"] in ("OOM", "COMPILE_FAIL", "FAILED"):
            print(f"  ladder stops at chi={chi} ({rec['status']})", flush=True)
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
