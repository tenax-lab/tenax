"""Phase C.3 diagnostic: tight-CTM re-evaluation of the χ=16 AD-converged state.

Purpose:
    Distinguish "the C.3 χ=16 result -0.4448 is real (CTM fixed-point
    issue)" from "it's optimisation noise from a loosely-converged inner
    CTM during L-BFGS." The wavefunction-fidelity test (option C) showed
    the encoding and the energy formula are correct at D∈{1..6} on the
    1-cell PBC torus; the C.3 anomaly therefore lives on the CTM side.

Procedure:
    1. Reproduce the SU warmstart at D=4, seed=0 (matches
       ``kagome_pess_multisite_phase_c3_probe.py``).
    2. Run multisite L-BFGS at χ=16 for ``--ad-iter`` steps (same loose
       CTM the C.3 probe used: ``max_iter=30, conv_tol=1e-7``). This
       reproduces the -0.4448 stuck point.
    3. Re-evaluate the resulting state's energy at the same χ but with a
       MUCH tighter CTM (``max_iter=300, conv_tol=1e-12``). No AD, no
       optimisation — just one forward CTM pass + energy.

Decision tree:
    * If E_tight ≥ -0.43752: the optimiser was misled by a
      loosely-converged CTM. Fix is to tighten the CTM in
      ``optimize_pess_3site_multisite_ad``.
    * If E_tight ≈ -0.4448 (still below floor): the multisite-CTM has a
      genuine fixed-point pathology at D=4 — likely related to the
      dim-1 v-w bonds creating ill-conditioned transfer matrices. Need
      a deeper structural fix (e.g. χ-ramp, alternative gauge,
      regularisation).

Usage::

    python examples/kagome_pess_multisite_phase_c3_tight_ctm_probe.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

from tenax.algorithms._pess_multisite_energy import kagome_3site_bond_gates
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import (
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    pess_simple_update,
)
from tenax.algorithms.pess_optimize import (
    build_pess_loss_3site_multisite,
    optimize_pess_3site_multisite_ad,
)

DELTA = 1.0
D_PHYS = 2
LIAO_ASYMPTOTIC = -0.43752


def _ctm_config(chi: int, max_iter: int, conv_tol: float) -> CTMConfig:
    return CTMConfig(
        chi=chi,
        max_iter=max_iter,
        min_iter=4,
        conv_tol=conv_tol,
        projector_method="svd",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
        gmres_tol=1e-5,
        gmres_maxiter=80,
        gmres_restart=30,
        chi_ramp=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument(
        "--ad-iter", type=int, default=30, help="L-BFGS iterations (loose CTM)"
    )
    parser.add_argument(
        "--tight-max-iter",
        type=int,
        default=300,
        help="Inner CTM max_iter for re-evaluation",
    )
    parser.add_argument(
        "--tight-conv-tol",
        type=float,
        default=1e-12,
        help="Inner CTM conv_tol for re-evaluation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    args = parser.parse_args()

    print(
        f"=== Phase C.3 tight-CTM diagnostic: D={args.D}, χ={args.chi} ===", flush=True
    )
    print(
        "  Loose CTM:  max_iter=30, conv_tol=1e-7 (reproduces -0.4448 stuck point)",
        flush=True,
    )
    print(
        f"  Tight CTM:  max_iter={args.tight_max_iter}, conv_tol={args.tight_conv_tol}",
        flush=True,
    )

    H = kagome_triangle_xxz_hamiltonian(delta=DELTA, d=D_PHYS)
    bond_gates = kagome_3site_bond_gates(delta=DELTA, d=D_PHYS)
    su_steps = ((0.1, 200), (0.01, 200), (0.001, 100))

    print("\n  [1/3] SU warmstart (D=4, seed=0)...", flush=True)
    t0 = time.perf_counter()
    state0 = IPESSState.random(D=args.D, d=D_PHYS, key=jax.random.PRNGKey(args.seed))
    state0 = pess_simple_update(state0, H, dt_schedule=list(su_steps), D_max=args.D)
    t_su = time.perf_counter() - t0

    loose_cfg = _ctm_config(args.chi, max_iter=30, conv_tol=1e-7)

    e_su_loose = float(
        build_pess_loss_3site_multisite(bond_gates, loose_cfg)(state0).real
    )
    print(f"    [SU readout, loose CTM] E = {e_su_loose:.6f}", flush=True)

    print(
        f"\n  [2/3] Multisite L-BFGS, {args.ad_iter} steps (loose CTM)...",
        flush=True,
    )
    t0 = time.perf_counter()
    state_ad, e_loose = optimize_pess_3site_multisite_ad(
        state0, bond_gates, loose_cfg, max_iter=args.ad_iter, verbose=True
    )
    t_ad = time.perf_counter() - t0
    print(f"    [AD-converged, loose CTM] E = {e_loose:.6f}", flush=True)

    # The decision: re-evaluate the AD-converged state's energy with a
    # much tighter inner CTM.  No optimisation, no AD — just forward CTM.
    print(
        f"\n  [3/3] Re-evaluating AD state with tight CTM "
        f"(max_iter={args.tight_max_iter}, conv_tol={args.tight_conv_tol})...",
        flush=True,
    )
    tight_cfg = _ctm_config(
        args.chi, max_iter=args.tight_max_iter, conv_tol=args.tight_conv_tol
    )
    t0 = time.perf_counter()
    e_tight = float(
        build_pess_loss_3site_multisite(bond_gates, tight_cfg)(state_ad).real
    )
    t_tight = time.perf_counter() - t0

    delta_loose_to_tight = e_tight - e_loose
    delta_tight_to_liao = e_tight - LIAO_ASYMPTOTIC

    result = {
        "D": args.D,
        "chi": args.chi,
        "ad_iter": args.ad_iter,
        "seed": args.seed,
        "loose_max_iter": 30,
        "loose_conv_tol": 1e-7,
        "tight_max_iter": args.tight_max_iter,
        "tight_conv_tol": args.tight_conv_tol,
        "e_su_loose": e_su_loose,
        "e_ad_loose": float(e_loose),
        "e_ad_tight": e_tight,
        "delta_loose_to_tight": delta_loose_to_tight,
        "liao_asymptotic": LIAO_ASYMPTOTIC,
        "delta_tight_to_liao": delta_tight_to_liao,
        "wall_seconds_su": t_su,
        "wall_seconds_ad": t_ad,
        "wall_seconds_tight": t_tight,
    }

    args.output.write_text(json.dumps(result, indent=2))

    print(f"\n  E_loose      = {e_loose:.6f}", flush=True)
    print(f"  E_tight      = {e_tight:.6f}", flush=True)
    print(f"  Δ tight - loose = {delta_loose_to_tight:+.6e}", flush=True)
    print(
        f"  Δ tight - Liao  = {delta_tight_to_liao:+.6e}  (variational principle: ≥ 0)",
        flush=True,
    )

    print(f"\nWrote result to {args.output}", flush=True)

    if e_tight >= LIAO_ASYMPTOTIC:
        print(
            "\n✓ Tight CTM moves E above the Liao floor — the C.3 anomaly "
            "is CTM convergence noise during optimisation.  Fix: tighten "
            "the inner CTM in optimize_pess_3site_multisite_ad.",
            flush=True,
        )
    else:
        print(
            "\n✗ Tight CTM stays below the Liao floor — the multisite-CTM "
            "fixed point itself is unphysical at D=4.  Likely culprit: "
            "dim-1 v-w bonds making the transfer matrix ill-conditioned. "
            "Needs a deeper structural fix.",
            flush=True,
        )


if __name__ == "__main__":
    main()
