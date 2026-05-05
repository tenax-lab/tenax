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


def _peak_rss_gb() -> float:
    # ru_maxrss: bytes on macOS, KB on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**3) if sys.platform == "darwin" else rss / (1024**2)


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


def _compute_e_p2_split(state, cg_gates, chi: int, max_iter: int = 30) -> float:
    """Forward-only P2 probe via split-CTM + ``compute_energy_cg_split``.

    Builds the kagome supersite, drives ``ctm_split_tensor`` to convergence,
    and evaluates the CG energy through the split-aware RDM dispatcher.
    Stays at the ``chi^2 * D^4 * d`` peak instead of the std path's
    ``chi^2 * D^6`` (matters at ``D >= 6`` where ``chi = 2 D^2`` makes the
    std-path projector SVD allocate >100 GB on a 256 GB box).
    """
    import jax.numpy as jnp_local

    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
    from tenax.algorithms.coarse_grain import compute_energy_cg_split
    from tenax.algorithms.pess import pess_to_kagome_supersite
    from tenax.algorithms.pess_optimize import _make_supersite_indices
    from tenax.core.tensor import DenseTensor

    A_super = pess_to_kagome_supersite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.lambdas
    )
    A_super = A_super / (jnp_local.linalg.norm(A_super) + 1e-12)
    D_super = A_super.shape[0]
    d_eff = int(cg_gates.h_intra.shape[0])
    indices = _make_supersite_indices(D_super, d_eff)
    A_tensor = DenseTensor(A_super, indices)

    env = ctm_split_tensor(A_tensor, chi=chi, max_iter=max_iter, chi_I=chi)
    return float(compute_energy_cg_split(A_tensor, env, cg_gates, d_eff).real)


def run_one(
    D: int,
    seed: int = 0,
    verbose: bool = False,
    skip_p2: bool = False,
    use_split_ctm: bool = False,
) -> dict:
    H = kagome_triangle_xxz_hamiltonian(delta=DELTA, d=D_PHYS)
    cg_gates = kagome_xxz_pess_cg_gates(delta=DELTA, d=D_PHYS)

    print(f"  [start] RSS={_peak_rss_gb():.2f} GB", flush=True)
    state = IPESSState.random(D=D, d=D_PHYS, key=jax.random.PRNGKey(seed))

    t_su = time.perf_counter()
    state = pess_simple_update(state, H, dt_schedule=SU_SCHEDULE, D_max=D)
    t_su = time.perf_counter() - t_su
    print(f"  [SU done] t={t_su:.1f}s RSS={_peak_rss_gb():.2f} GB", flush=True)

    t_p1 = time.perf_counter()
    e_p1 = float(pess_local_energy(state, H))
    t_p1 = time.perf_counter() - t_p1
    print(
        f"  [P1 done] t={t_p1:.1f}s e_p1={e_p1:.6f} RSS={_peak_rss_gb():.2f} GB",
        flush=True,
    )

    chi = 2 * D * D
    p2_path: str | None = None
    if skip_p2:
        e_p2: float | None = None
        t_p2 = 0.0
        print("  [P2 skipped]", flush=True)
    elif use_split_ctm:
        p2_path = "split-ctm"
        print(
            f"  [P2 split-CTM start] χ={chi} RSS={_peak_rss_gb():.2f} GB",
            flush=True,
        )
        t_p2 = time.perf_counter()
        e_p2 = _compute_e_p2_split(state, cg_gates, chi=chi)
        t_p2 = time.perf_counter() - t_p2
        print(
            f"  [P2 split-CTM done] t={t_p2:.1f}s e_p2={e_p2:.6f} "
            f"RSS={_peak_rss_gb():.2f} GB",
            flush=True,
        )
    else:
        p2_path = "std-ctm"
        config = _make_ctm_config(chi=chi)
        loss_fn = build_pess_loss(cg_gates, config)
        print(f"  [P2 start] χ={chi} RSS={_peak_rss_gb():.2f} GB", flush=True)
        t_p2 = time.perf_counter()
        e_p2 = float(loss_fn(state).real)
        t_p2 = time.perf_counter() - t_p2
        print(
            f"  [P2 done] t={t_p2:.1f}s e_p2={e_p2:.6f} RSS={_peak_rss_gb():.2f} GB",
            flush=True,
        )

    target = LIAO2017_3PESS_SU_FIG1A.get(D)

    record = {
        "D": D,
        "chi": chi,
        "seed": seed,
        "su_schedule": [list(s) for s in SU_SCHEDULE],
        "e_p1_husimi": e_p1,
        "e_p2_ctm": e_p2,
        "p2_path": p2_path,
        "liao2017_target": target,
        "delta_p1_target": (e_p1 - target) if target is not None else None,
        "delta_p2_target": (
            (e_p2 - target) if (target is not None and e_p2 is not None) else None
        ),
        "t_su_seconds": t_su,
        "t_p1_seconds": t_p1,
        "t_p2_seconds": t_p2,
    }
    if verbose:
        e_p2_str = f"{e_p2:.6f}" if e_p2 is not None else "(skipped)"
        d_p2_str = (
            f"{record['delta_p2_target']:+.4f}"
            if record["delta_p2_target"] is not None
            else "(skipped)"
        )
        target_str = f"{target:.6f}" if target is not None else "n/a"
        print(
            f"  D={D:2d} χ={chi:3d}  P1={e_p1:.6f}  P2={e_p2_str}  "
            f"target={target_str}  Δ_P1={record['delta_p1_target']:+.4f}  "
            f"Δ_P2={d_p2_str}",
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
    p.add_argument(
        "--skip-p2",
        action="store_true",
        help="Skip the Convention-C + CTM probe (P2). Useful at D≥6 where "
        "the std-CTM projector SVD allocates >100 GB at χ=2D².",
    )
    p.add_argument(
        "--use-split-ctm",
        action="store_true",
        help="Run P2 via the split-aware CTM forward + compute_energy_cg_split "
        "(peak ≈ χ²·D⁴·d) instead of the std AD path (peak ≈ χ²·D⁶). "
        "Forward-only — no AD, no implicit-CTM. Lets D≥6 fit on a 32 GB box.",
    )
    args = p.parse_args()

    print("Liao 2017 PRL 118 137202 3-PESS SU replication", flush=True)
    print(f"  asymptotic target E0 = {LIAO2017_3PESS_SU_INF:.6f}", flush=True)
    print(f"  D list: {args.D}", flush=True)
    print(f"  SU schedule: {SU_SCHEDULE}", flush=True)

    results = []
    for D in args.D:
        print(f"\n=== D = {D} ===", flush=True)
        record = run_one(
            D=D,
            seed=args.seed,
            verbose=True,
            skip_p2=args.skip_p2,
            use_split_ctm=args.use_split_ctm,
        )
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
