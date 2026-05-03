"""Spin-½ kagome AFM Heisenberg AD-iPESS benchmark.

Pipeline:
  1. Triangle simple update (3-PESS, HOSVD truncation per simplex) —
     this matches the SU kernel of Liao et al., PRL 118, 137202 (2017),
     arXiv:1610.04727 ("Gapless spin-liquid ground state in the S=1/2
     kagome antiferromagnet"). Liao 2017 reports E/site → -0.43752(6)
     in the large-D limit (Fig 1(b) inset).
  2. AD optimization through the Tenax square-CG-iPEPS CTM
     (Convention C: PESS -> 1-site square supersite, chi = 2*D**2).
     This step is a Tenax extension; Liao 2017 has no AD optimization
     for kagome PESS. The AD machinery is reused from Liao et al.,
     PRX 9, 031041 (2019), "Differentiable Programming Tensor
     Networks", which applies AD to *square-lattice* iPEPS, not
     kagome PESS.

For an SU-only Liao 2017 replication audit (no AD, two energy probes),
see ``kagome_spin12_pess_liao2017_replication.py``.

Usage:
    python examples/kagome_spin12_pess_ad_benchmark.py --D 2 --chi 8
    python examples/kagome_spin12_pess_ad_benchmark.py --D 4 --chi 32 --max-iter 80
    python examples/kagome_spin12_pess_ad_benchmark.py --sweep   # D in {2,4,6}, chi=2D^2

Output:
    JSON file at ``--output`` (default
    ``examples/kagome_spin12_pess_ad_benchmark.json``) with one entry per
    ``(D, chi)`` pair containing the per-kagome-site SU and AD energies.
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
