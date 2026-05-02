"""Spin-1 kagome Heisenberg AD-iPESS benchmark.

Spin-1 analog of :mod:`examples.kagome_spin12_pess_ad_benchmark`. d=3
(``d_eff = 27``), Δ=1 isotropic. Reference: Picot et al., Phys. Rev. B
93, 060407(R) (2016) — large-D iPESS gives E/site ≈ −1.41.

At small ``D`` (D=2) the iPESS pipeline lands around E/site ≈ −1.0,
significantly below the classical limit (Néel-like product state energy
−3/4 for spin-1) but well above the converged large-D value. Going to
``D=3`` or ``D=4`` is the cheapest meaningful step toward the published
target.

Usage:
    python examples/kagome_spin1_pess_ad_benchmark.py --D 2 --chi 8
    python examples/kagome_spin1_pess_ad_benchmark.py --D 4 --chi 32 --max-iter 80
    python examples/kagome_spin1_pess_ad_benchmark.py --sweep    # D ∈ {2,3,4}, χ=2D²
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import (
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    kagome_xxz_pess_cg_gates,
    pess_simple_update,
)
from tenax.algorithms.pess_optimize import build_pess_loss, optimize_pess_ad

DELTA = 1.0  # isotropic Heisenberg
D_PHYS = 3  # spin-1


def _make_ctm_config(chi: int, max_iter: int = 30) -> CTMConfig:
    return CTMConfig(
        chi=chi,
        max_iter=max_iter,
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


def run_kagome_spin1_benchmark(
    D: int,
    chi: int,
    max_iter: int = 30,
    su_steps: tuple[tuple[float, int], ...] = (
        (0.1, 200),
        (0.01, 200),
        (0.001, 100),
    ),
    seed: int = 0,
    verbose: bool = False,
) -> tuple[IPESSState, float, float]:
    """Run SU warm-start + AD optimization. Returns ``(state, e_ad, e_su)``."""
    H = kagome_triangle_xxz_hamiltonian(delta=DELTA, d=D_PHYS)
    cg_gates = kagome_xxz_pess_cg_gates(delta=DELTA, d=D_PHYS)
    state = IPESSState.random(D=D, d=D_PHYS, key=jax.random.PRNGKey(seed))
    state = pess_simple_update(state, H, dt_schedule=list(su_steps), D_max=D)

    config = _make_ctm_config(chi=chi)
    loss_fn = build_pess_loss(cg_gates, config)
    e_su = float(loss_fn(state).real)
    if verbose:
        print(f"  [SU only] E/site = {e_su:.6f}", flush=True)

    state, e_ad = optimize_pess_ad(
        state, cg_gates, config, max_iter=max_iter, verbose=verbose
    )
    return state, e_ad, e_su


def _sweep(args: argparse.Namespace) -> list[dict]:
    results: list[dict] = []
    for D in args.D_list:
        chi = 2 * D * D if args.chi is None else args.chi
        print(f"\n=== D = {D}, χ = {chi} ===", flush=True)
        t0 = time.perf_counter()
        _, e_ad, e_su = run_kagome_spin1_benchmark(
            D=D,
            chi=chi,
            max_iter=args.max_iter,
            verbose=True,
        )
        dt = time.perf_counter() - t0
        print(
            f"  [AD]      E/site = {e_ad:.6f}  (Δ vs SU = {e_ad - e_su:+.6f})",
            flush=True,
        )
        results.append(
            {
                "D": D,
                "chi": chi,
                "max_iter": args.max_iter,
                "e_per_site_su": e_su,
                "e_per_site_ad": e_ad,
                "wall_seconds": dt,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=None, help="Bond dimension D")
    parser.add_argument("--chi", type=int, default=None, help="CTM χ (default 2D²)")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a default sweep over D ∈ {2, 3, 4}.",
    )
    parser.add_argument("--max-iter", type=int, default=80, help="L-BFGS iterations")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
        help="Path to JSON results file",
    )
    args = parser.parse_args()

    if args.sweep or args.D is None:
        args.D_list = [2, 3, 4]
    else:
        args.D_list = [args.D]

    results = _sweep(args)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} result(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
