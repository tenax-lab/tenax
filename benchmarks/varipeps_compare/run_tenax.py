"""Tenax CLI runner for the variPEPS compare benchmark.

Usage:
    python -m benchmarks.varipeps_compare.run_tenax \\
        --payload payload.npz --path single_site --D 2 --chi 16 \\
        --tol 1e-6 --max-steps 100 --out tenax_<key>.json
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad

from .payload import load_payload
from .protocol import CTM_MAX_ITER, CTM_TOL


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _tenax_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _build_config(
    *, path: str, D: int, chi: int, tol: float, max_steps: int
) -> iPEPSConfig:
    """Build the iPEPSConfig for a benchmark point.

    If ``VARIPEPS_COMPARE_TEST_FAST=1`` is set in the environment, swap the
    production implicit-AD + L-BFGS path for explicit-AD + Adam with a tiny
    CTM unroll.  This keeps JIT compile under ~1 min so the smoke test can
    exercise the runner module end-to-end.  The accumulator code in the
    history hook is path-agnostic, so the schema contract is still
    validated.  Production benchmark runs always use the implicit-AD path.
    """
    import os

    test_fast = os.environ.get("VARIPEPS_COMPARE_TEST_FAST") == "1"

    ctm = CTMConfig(
        chi=chi,
        max_iter=5 if test_fast else CTM_MAX_ITER,
        conv_tol=CTM_TOL,
        projector_method="svd",  # Fishman, matches variPEPS default
    )
    common: dict = dict(
        max_bond_dim=D,
        ctm=ctm,
        gs_num_steps=max_steps,
        gs_conv_tol=tol,
        su_init=False,
        return_history=True,
    )
    if test_fast:
        common.update(
            gs_optimizer="adam",
            gs_implicit_ad=False,
            gs_explicit_ad_steps=2,
            gs_explicit_ad_warmup=1,
            gs_learning_rate=1e-2,
        )
    else:
        common.update(gs_optimizer="lbfgs", gs_implicit_ad=True)

    if path == "single_site":
        # Unconstrained 1×1 + sublattice-rotated gate.  No C4v — same ansatz
        # as variPEPS structure=[[0]] so parameter counts match.
        return iPEPSConfig(unit_cell="1x1", gs_c4v=False, **common)
    elif path == "bipartite_2site":
        return iPEPSConfig(unit_cell="2site", **common)
    else:
        raise ValueError(f"unknown path: {path}")


def _load_init(init: np.ndarray, path: str):
    if path == "single_site":
        return jnp.asarray(init)
    elif path == "bipartite_2site":
        # init shape: (2, D, D, D, D, d) → (A, B)
        return (jnp.asarray(init[0]), jnp.asarray(init[1]))
    else:
        raise ValueError(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, type=Path)
    ap.add_argument("--path", required=True, choices=("single_site", "bipartite_2site"))
    ap.add_argument("--D", required=True, type=int)
    ap.add_argument("--chi", required=True, type=int)
    ap.add_argument("--tol", required=True, type=float)
    ap.add_argument("--max-steps", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    init_np, gate_np, _ = load_payload(args.payload)
    gate = jnp.asarray(gate_np)
    A_init = _load_init(init_np, args.path)
    config = _build_config(
        path=args.path,
        D=args.D,
        chi=args.chi,
        tol=args.tol,
        max_steps=args.max_steps,
    )

    t0 = time.perf_counter()
    result = optimize_gs_ad(gate, A_init, config)
    dt = time.perf_counter() - t0

    history = result[-1]
    final_energy = float(result[-2])

    out = {
        "lib": "tenax",
        "path": args.path,
        "D": args.D,
        "chi": args.chi,
        "dtype": "complex128",
        "seed": 0,
        "energy_history": [float(e) for e in history["energies"]],
        "step_times": [float(t) for t in history["step_times"]],
        "jit_compile_time": float(history["jit_compile_time"]),
        "final_energy": final_energy,
        "num_steps": int(history["num_steps"]),
        "converged": bool(history["converged"]),
        "total_wall_clock": dt,
        "peak_memory_mb": _peak_rss_mb(),
        "device": str(jax.devices()[0]).lower(),
        "lib_version": _tenax_git_sha(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
