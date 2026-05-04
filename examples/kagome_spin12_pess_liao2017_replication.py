#!/usr/bin/env python3
"""Liao 2017 (PRL 118 137202) 3-PESS SU replication audit.

Runs a D-sweep at D in {4, 6, 8, 10} on the spin-1/2 kagome AFM Heisenberg
model. For each D, measures E/site two ways:

  P1 — Husimi-tree mean-field probe (`pess_local_energy`): contracts one
       triangle's 3-site RDM with bond-λ gauges as the environment. No CTM.
  P2 — Convention-C + CTM (`build_pess_loss`): maps the PESS to a
       square-iPEPS supersite, runs CTM at χ = 2*D**2, evaluates energy
       through the standard 2-site RDM helpers.

Both numbers are written to JSON alongside Liao 2017's hand-digitized
Fig 1(a) target. The diff (P1 - target, P2 - target) tells us whether
the SU kernel matches Liao (P1 ≈ target) and whether Convention-C is
the bias source (P2 above target).

Usage::

    python examples/kagome_spin12_pess_liao2017_replication.py \\
        --output examples/kagome_spin12_pess_liao2017_replication.json
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import jax

from tenax.algorithms._liao2017_targets import (
    LIAO2017_3PESS_SU_FIG1A,
    LIAO2017_3PESS_SU_INF,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import (
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    kagome_xxz_pess_cg_gates,
    pess_local_energy,
    pess_simple_update,
)
from tenax.algorithms.pess_optimize import build_pess_loss

DELTA = 1.0
D_PHYS = 2
SU_SCHEDULE = [(0.1, 200), (0.01, 200), (0.001, 100)]
D_LIST_DEFAULT = (4, 6, 8, 10)
# JAX trace cache snowballs across SU stages with different dt — at D≥6 it
# OOMs a 256 GB box. Mirror tests/conftest.py: clear caches between stages
# once peak RSS crosses this threshold.
_RSS_CLEAR_THRESHOLD_MB = 6000


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _maybe_clear_jax_caches() -> None:
    if _peak_rss_mb() < _RSS_CLEAR_THRESHOLD_MB:
        return
    jax.clear_caches()
    gc.collect()


def _make_ctm_config(chi: int) -> CTMConfig:
    return CTMConfig(
        chi=chi,
        max_iter=30,
        min_iter=4,
        conv_tol=1e-7,
        projector_method="svd",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
        gmres_tol=1e-5,
        gmres_maxiter=80,
        gmres_restart=30,
        chi_ramp=None,
    )


def run_one(D: int, seed: int = 0, verbose: bool = False) -> dict:
    H = kagome_triangle_xxz_hamiltonian(delta=DELTA, d=D_PHYS)
    cg_gates = kagome_xxz_pess_cg_gates(delta=DELTA, d=D_PHYS)

    state = IPESSState.random(D=D, d=D_PHYS, key=jax.random.PRNGKey(seed))

    t_su = time.perf_counter()
    for stage in SU_SCHEDULE:
        state = pess_simple_update(state, H, dt_schedule=[stage], D_max=D)
        _maybe_clear_jax_caches()
    t_su = time.perf_counter() - t_su

    t_p1 = time.perf_counter()
    e_p1 = float(pess_local_energy(state, H))
    t_p1 = time.perf_counter() - t_p1

    chi = 2 * D * D
    config = _make_ctm_config(chi=chi)
    loss_fn = build_pess_loss(cg_gates, config)
    t_p2 = time.perf_counter()
    e_p2 = float(loss_fn(state).real)
    t_p2 = time.perf_counter() - t_p2

    target = LIAO2017_3PESS_SU_FIG1A.get(D)

    record = {
        "D": D,
        "chi": chi,
        "seed": seed,
        "su_schedule": [list(s) for s in SU_SCHEDULE],
        "e_p1_husimi": e_p1,
        "e_p2_ctm": e_p2,
        "liao2017_target": target,
        "delta_p1_target": (e_p1 - target) if target is not None else None,
        "delta_p2_target": (e_p2 - target) if target is not None else None,
        "t_su_seconds": t_su,
        "t_p1_seconds": t_p1,
        "t_p2_seconds": t_p2,
    }
    if verbose:
        print(
            f"  D={D:2d} χ={chi:3d}  P1={e_p1:.6f}  P2={e_p2:.6f}  "
            f"target={target:.6f}  Δ_P1={record['delta_p1_target']:+.4f}  "
            f"Δ_P2={record['delta_p2_target']:+.4f}",
            flush=True,
        )
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--D",
        type=int,
        nargs="*",
        default=list(D_LIST_DEFAULT),
        help="Bond dimensions (default: 4 6 8 10)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
        help="Path to JSON results file",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("Liao 2017 PRL 118 137202 3-PESS SU replication", flush=True)
    print(f"  asymptotic target E0 = {LIAO2017_3PESS_SU_INF:.6f}", flush=True)
    print(f"  D list: {args.D}", flush=True)
    print(f"  SU schedule: {SU_SCHEDULE}", flush=True)

    results = []
    for D in args.D:
        print(f"\n=== D = {D} ===", flush=True)
        record = run_one(D=D, seed=args.seed, verbose=True)
        results.append(record)

    payload = {
        "reference": "Liao et al., PRL 118, 137202 (2017), arXiv:1610.04727",
        "asymptotic_target_E0": LIAO2017_3PESS_SU_INF,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(results)} record(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
