"""Spin-½ kagome AFM Heisenberg AD-iPESS benchmark.

Reproduces the differentiable-iPESS pipeline of Liao et al., PRX 9, 031041
(2019) on the spin-½ kagome antiferromagnet at the isotropic point Δ=1.
For each bond dimension ``D``, run a triangle simple-update warm start
(:func:`tenax.pess_simple_update`) then optimize ``(R_a, R_b, R_c, T_u,
lambdas)`` via L-BFGS through the square CG-iPEPS CTM
(:func:`tenax.optimize_pess_ad`).

Usage:
    python examples/kagome_spin12_pess_ad_benchmark.py --D 2 --chi 8
    python examples/kagome_spin12_pess_ad_benchmark.py --D 4 --chi 32 --max-iter 80
    python examples/kagome_spin12_pess_ad_benchmark.py --sweep   # D ∈ {2,4,6}, χ=2D²

Output:
    JSON file at ``--output`` (default
    ``examples/kagome_spin12_pess_ad_benchmark.json``) with one entry per
    ``(D, chi)`` pair containing the per-kagome-site SU and AD energies.

Reference (Liao 2019, Table I): E/site → −0.4378 in the large-``D`` limit.
At very small ``D`` (D=2) the iPESS variational manifold is dominated by
the classical 120° state at E/site ≈ −0.25; quantum corrections start
showing up at ``D=4`` and converge from above as ``D`` grows.
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
D_PHYS = 2  # spin-½


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


def run_kagome_spin12_benchmark(
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
        _, e_ad, e_su = run_kagome_spin12_benchmark(
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
        help="Run a default sweep over D ∈ {2, 4, 6}.",
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
        args.D_list = [2, 4, 6]
    else:
        args.D_list = [args.D]

    results = _sweep(args)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} result(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
