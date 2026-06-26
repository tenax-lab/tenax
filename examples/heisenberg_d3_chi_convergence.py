#!/usr/bin/env python3
r"""D=3 χ-convergence study for the 2D square Heisenberg AFM via iPEPS-AD.

Computes the ground-state energy of the spin-1/2 antiferromagnetic Heisenberg
model on the infinite square lattice at fixed bond dimension ``D=3``, sweeping
the CTM environment dimension χ along a ladder (16 → 24 → 32 → 48) to study the
convergence of the variational energy in χ.

Method (variational + crash-resilient)
--------------------------------------
* **Implicit AD** (``gs_implicit_ad=True``) through the CTM fixed point — the
  energy is a true variational expectation value (explicit/unrolled AD is
  non-variational, see issue #328, so it is *not* used here).
* **C4v symmetrization** (``gs_c4v=True``).  The square-lattice AFM ground state
  is C4v-symmetric; enforcing it removes the iPEPS bond-gauge freedom that
  otherwise makes the implicit-AD backward linear-solve ill-conditioned (NaN
  gradients).  With default ``ctm_ad_mode=None`` this routes to the
  checkpoint-wired ``_optimize_gs_ad_tensor`` (NOT the un-checkpointed
  ``c4v_reference`` path).
* **Fixed-χ-per-stage, warm-started** across the ladder: each stage optimizes at
  one fixed χ starting from the previous stage's optimized tensor.  This avoids
  the mid-run χ-bump that violates the implicit-AD variational precondition
  (zero-padded env, issue #511).
* **Checkpointing** every 2 steps within each stage, plus a per-stage optimized
  tensor saved on completion.  Re-running the script resumes the ladder where it
  left off (completed stages are skipped; an interrupted stage resumes from its
  in-stage checkpoint) — so a long run survives crashes / preemption.

Reference energy: E/site ≈ -0.6694 (QMC).  D=2 χ=8 gives ≈ -0.6625; D=3 should
sit below that and approach the QMC value as χ grows.

Usage::

    # quick validation (D=3, χ=16 only, few steps) — also the CI smoke path
    uv run python examples/heisenberg_d3_chi_convergence.py --smoke

    # full production ladder (single GPU; ~1-2 GPU-days)
    uv run python examples/heisenberg_d3_chi_convergence.py --outdir runs/d3chi

    # resume after a crash: just re-run the same command
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import jax

jax.config.update("jax_enable_x64", True)

from tenax import (  # noqa: E402
    CTMConfig,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
    sublattice_rotate_gate,
)

QMC_REF = -0.6694  # E/site, square-lattice Heisenberg AFM (QMC)


def make_config(
    *, chi: int, ckpt_path: str, num_steps: int, resume: bool, probe_max_iter: int | None
) -> iPEPSConfig:
    """Validated variational config: implicit AD + C4v on the checkpoint path."""
    return iPEPSConfig(
        max_bond_dim=3,
        num_imaginary_steps=200,
        dt=0.05,
        ctm=CTMConfig(
            chi=chi,
            max_iter=100,
            conv_tol=1e-8,
            projector_method="svd",   # implicit-AD requires svd/qr (not eigh)
            forward_gauge="phase",    # implicit-AD requires phase gauge
            probe_max_iter=probe_max_iter,  # #503: cap HZ line-search CTM probes
        ),
        unit_cell="1x1",
        gs_c4v=True,                  # removes bond-gauge freedom -> stable backward
        gs_implicit_ad=True,          # variational (ctm_ad_mode=None -> ckpt-wired)
        gs_recipe="1x1",
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_metric_precond=True,
        gs_num_steps=num_steps,
        gs_verbose=True,
        gs_log_interval=1,
        su_init=True,                 # only runs when A_init is None (stage 1)
        gs_checkpoint_path=ckpt_path,
        gs_checkpoint_every=2,
        gs_resume=resume,
    )


def run_ladder(
    outdir: str, ladder: list[tuple[int, int]], probe_max_iter: int | None
) -> list[dict]:
    """Run the warm-started fixed-χ ladder with ladder-level resume."""
    os.makedirs(outdir, exist_ok=True)
    H = sublattice_rotate_gate(heisenberg_gate())

    results_path = os.path.join(outdir, "chi_convergence.json")
    results: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path) as fh:
            results = json.load(fh)
    done = {r["chi"]: r for r in results}

    A_warm = None  # warm-start tensor threaded across stages
    for chi, nsteps in ladder:
        stage_tensor = os.path.join(outdir, f"A_chi{chi}.pkl")
        if chi in done and os.path.exists(stage_tensor):
            with open(stage_tensor, "rb") as fh:
                A_warm = pickle.load(fh)
            print(f"[ladder] χ={chi}: already complete (E={done[chi]['E']:.6f}), "
                  f"loaded warm-start tensor", flush=True)
            continue

        ckpt = os.path.join(outdir, f"ckpt_chi{chi}", "ckpt")
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        resume = os.path.exists(ckpt + ".last.pkl")
        print(f"\n{'=' * 60}\n[ladder] χ={chi}  (resume={resume}, warm_start="
              f"{A_warm is not None})\n{'=' * 60}", flush=True)

        cfg = make_config(chi=chi, ckpt_path=ckpt, num_steps=nsteps,
                          resume=resume, probe_max_iter=probe_max_iter)
        t0 = time.perf_counter()
        A_opt, _env, E = optimize_gs_ad(H, A_warm, cfg)
        dt = time.perf_counter() - t0
        E = float(E)

        with open(stage_tensor, "wb") as fh:
            pickle.dump(A_opt, fh)
        results = [r for r in results if r["chi"] != chi]
        results.append({"chi": chi, "E": E, "time_s": dt,
                        "err_vs_qmc": E - QMC_REF})
        results.sort(key=lambda r: r["chi"])
        with open(results_path, "w") as fh:
            json.dump(results, fh, indent=2)
        A_warm = A_opt
        print(f"[ladder] χ={chi} DONE: E/site={E:.6f}  err_vs_qmc={E - QMC_REF:+.4f}"
              f"  ({dt:.0f}s)", flush=True)

    print(f"\n{'=' * 60}\nχ-convergence curve (D=3 Heisenberg AFM, QMC ref "
          f"{QMC_REF})\n{'=' * 60}", flush=True)
    for r in sorted(results, key=lambda r: r["chi"]):
        print(f"  χ={r['chi']:3d}   E/site={r['E']:.6f}   "
              f"err_vs_qmc={r['err_vs_qmc']:+.5f}   ({r['time_s']:.0f}s)", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="runs/d3_chi_convergence",
                    help="directory for checkpoints, tensors, and results JSON")
    ap.add_argument("--smoke", action="store_true",
                    help="quick validation: χ=16 only, few steps")
    ap.add_argument("--steps-per-stage", type=int, default=80,
                    help="max optimizer steps per χ stage (full run)")
    ap.add_argument("--probe-max-iter", type=int, default=15,
                    help="#503 cap on HZ line-search CTM probe sweeps "
                         "(None disables; lower = faster, looser line search)")
    args = ap.parse_args()

    probe = None if args.probe_max_iter <= 0 else args.probe_max_iter
    if args.smoke:
        ladder = [(16, 12)]
        outdir = args.outdir + "_smoke"
    else:
        n = args.steps_per_stage
        ladder = [(16, n), (24, n), (32, n), (48, n)]
        outdir = args.outdir

    print(f"D=3 χ-convergence study | ladder={ladder} | probe_max_iter={probe} | "
          f"outdir={outdir}", flush=True)
    run_ladder(outdir, ladder, probe)


if __name__ == "__main__":
    main()
