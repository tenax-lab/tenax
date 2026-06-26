#!/usr/bin/env python3
r"""D=3 χ-convergence study for the 2D square Heisenberg AFM via iPEPS-AD.

Computes the ground-state energy of the spin-1/2 antiferromagnetic Heisenberg
model on the infinite square lattice at fixed bond dimension ``D=3`` and studies
the convergence of the energy with the CTM environment dimension χ.

Design (clean χ-convergence)
----------------------------
The variational state is optimized **once** at a high χ, then its energy is
measured on a χ-ladder with a fully-converged CTM (no AD, no optimizer noise).
This separates the two sources of error cleanly:

* **Optimization** (one run): implicit AD through the CTM fixed point (the energy
  is a true variational expectation value — explicit/unrolled AD is
  non-variational, #328) with **C4v symmetrization** (``gs_c4v=True``).  The
  square-lattice AFM ground state is C4v-symmetric; enforcing it removes the
  iPEPS bond-gauge freedom that makes the implicit-AD backward ill-conditioned.
  Two safety nets handle the rare non-variational CTM spike: ``gs_energy_floor``
  (rejects sub-GS artifacts from best-state tracking) and ``gs_grad_spike_ratio``
  (rolls a >Nx gradient blowup back to best before the line search thrashes).
  Checkpointed for crash/preempt resilience; the optimized tensor is saved.

* **χ-scan** (cheap): for each χ, run CTM to convergence on the *fixed* optimized
  tensor and evaluate ⟨H⟩.  E(χ) is clean and converges monotonically — this is
  the "is χ large enough" plot.

Reference energy: E/site ≈ -0.6694 (QMC).  D=2 χ=8 gives ≈ -0.6625; D=3 should
sit below that and the χ-scan should plateau as χ grows.

Usage::

    # quick validation (small χ, few steps) — also the CI smoke path
    uv run python examples/heisenberg_d3_chi_convergence.py --smoke

    # full study (single GPU)
    uv run python examples/heisenberg_d3_chi_convergence.py --outdir runs/d3chi

    # resume after a crash: just re-run the same command (optimization resumes;
    # if the optimized tensor already exists, it jumps straight to the χ-scan)
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
    compute_energy_ctm_tensor,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
    sublattice_rotate_gate,
)

# The χ-scan must contract the optimized state with the SAME CTM the optimizer
# uses (phase-gauge fixing, element-wise convergence) — the public ``ctm_tensor``
# is the standard 4-move CTM without phase-gauge fixing and converges to a
# different fixed point for this C4v-optimized state, giving inconsistent
# energies.  We therefore reuse the optimizer's own fresh-CTM eval path
# (``python_loop_ctm_converge`` + ``ctm_converge_kwargs``), exactly mirroring the
# ``_eval_fresh`` helper that produces ``optimize_gs_ad``'s returned energy.
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs  # noqa: E402

QMC_REF = -0.6694  # E/site, square-lattice Heisenberg AFM (QMC)


def make_opt_config(
    *, chi: int, ckpt_path: str, num_steps: int, resume: bool, probe_max_iter: int | None
) -> iPEPSConfig:
    """Validated variational optimizer config: implicit AD + C4v + safety nets."""
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
        gs_conv_criterion="grad_norm",  # dE is fragile: a spike rollback makes
                                        # dE=0 and falsely "converges". grad_norm
                                        # (||g||<tol) is a true stationarity test;
                                        # with the finite-χ noise floor it simply
                                        # runs the full step budget.
        gs_energy_floor=QMC_REF,      # reject sub-GS CTM-artifact spikes (#298)
        gs_grad_spike_ratio=5.0,      # roll back >5x gradient blowups before the
                                      # line search thrashes (#524)
        gs_verbose=True,
        gs_log_interval=1,
        su_init=True,
        gs_checkpoint_path=ckpt_path,
        gs_checkpoint_every=2,
        gs_resume=resume,
    )


def optimize_state(outdir: str, chi_opt: int, num_steps: int, probe_max_iter: int | None):
    """Optimize the D=3 state once at ``chi_opt``; cache the optimized tensor."""
    H = sublattice_rotate_gate(heisenberg_gate())
    tensor_path = os.path.join(outdir, "A_opt.pkl")
    if os.path.exists(tensor_path):
        with open(tensor_path, "rb") as fh:
            A_opt = pickle.load(fh)
        print(f"[opt] optimized tensor already cached ({tensor_path}); "
              f"skipping optimization", flush=True)
        return A_opt, H

    ckpt = os.path.join(outdir, "ckpt_opt", "ckpt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    resume = os.path.exists(ckpt + ".last.pkl")
    print(f"\n{'=' * 60}\n[opt] optimizing D=3 state at χ={chi_opt} "
          f"(resume={resume}, {num_steps} steps)\n{'=' * 60}", flush=True)
    cfg = make_opt_config(chi=chi_opt, ckpt_path=ckpt, num_steps=num_steps,
                          resume=resume, probe_max_iter=probe_max_iter)
    t0 = time.perf_counter()
    A_opt, _env, E = optimize_gs_ad(H, None, cfg)
    print(f"[opt] done in {time.perf_counter() - t0:.0f}s; in-loop E_best={float(E):.6f}",
          flush=True)
    with open(tensor_path, "wb") as fh:
        pickle.dump(A_opt, fh)
    return A_opt, H


def scan_chi(A_opt, H, chi_ladder: list[int], outdir: str) -> list[dict]:
    """Clean E(χ) on the fixed optimized state: converge CTM, evaluate ⟨H⟩."""
    results_path = os.path.join(outdir, "chi_convergence.json")
    results: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path) as fh:
            results = json.load(fh)
    done = {r["chi"] for r in results}

    print(f"\n{'=' * 60}\n[scan] clean E(χ) on the fixed optimized state\n{'=' * 60}",
          flush=True)
    for chi in chi_ladder:
        if chi in done:
            print(f"[scan] χ={chi}: cached", flush=True)
            continue
        t0 = time.perf_counter()
        # Same CTM the optimizer's _eval_fresh uses: phase gauge, elementwise
        # convergence (via ctm_converge_kwargs), converged tightly at this χ.
        ctm_cfg = CTMConfig(chi=chi, max_iter=200, conv_tol=1e-10,
                            projector_method="svd", forward_gauge="phase")
        envs, info = python_loop_ctm_converge(
            {(0, 0): A_opt}, SINGLE_SITE_NEIGHBORS,
            **ctm_converge_kwargs(ctm_cfg),
        )
        E = float(compute_energy_ctm_tensor(A_opt, envs[(0, 0)], H, 2))
        dt = time.perf_counter() - t0
        results = [r for r in results if r["chi"] != chi]
        results.append({"chi": chi, "E": E,
                        "err_vs_qmc": E - QMC_REF, "time_s": dt})
        results.sort(key=lambda r: r["chi"])
        with open(results_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[scan] χ={chi:3d}: E/site={E:.6f}  err_vs_qmc={E - QMC_REF:+.5f}  "
              f"({dt:.0f}s)", flush=True)

    print(f"\n{'=' * 60}\nχ-convergence curve (D=3 Heisenberg AFM, QMC ref {QMC_REF})"
          f"\n{'=' * 60}", flush=True)
    for r in sorted(results, key=lambda r: r["chi"]):
        print(f"  χ={r['chi']:3d}   E/site={r['E']:.6f}   "
              f"err_vs_qmc={r['err_vs_qmc']:+.5f}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="runs/d3_chi_convergence",
                    help="directory for checkpoint, optimized tensor, results JSON")
    ap.add_argument("--smoke", action="store_true",
                    help="quick validation: small χ, few steps, short scan")
    ap.add_argument("--chi-opt", type=int, default=24,
                    help="χ for the one-time variational optimization. Kept "
                         "moderate: the implicit-AD backward is spike-prone at "
                         "large χ (frequent rollbacks), while χ≈24 optimizes "
                         "stably; the fixed-state χ-scan (forward-only) still "
                         "runs up to 96.")
    ap.add_argument("--opt-steps", type=int, default=100,
                    help="optimizer steps for the one-time optimization")
    ap.add_argument("--probe-max-iter", type=int, default=15,
                    help="#503 cap on HZ line-search CTM probe sweeps "
                         "(<=0 disables)")
    args = ap.parse_args()

    probe = None if args.probe_max_iter <= 0 else args.probe_max_iter
    if args.smoke:
        outdir = args.outdir + "_smoke"
        chi_opt, opt_steps = 16, 12
        chi_ladder = [8, 12, 16, 24]
    else:
        outdir = args.outdir
        chi_opt, opt_steps = args.chi_opt, args.opt_steps
        chi_ladder = [8, 16, 24, 32, 48, 64, 96]

    print(f"D=3 χ-convergence study | optimize once at χ={chi_opt} ({opt_steps} steps)"
          f" | scan χ={chi_ladder} | outdir={outdir}", flush=True)
    os.makedirs(outdir, exist_ok=True)
    A_opt, H = optimize_state(outdir, chi_opt, opt_steps, probe)
    scan_chi(A_opt, H, chi_ladder, outdir)


if __name__ == "__main__":
    main()
