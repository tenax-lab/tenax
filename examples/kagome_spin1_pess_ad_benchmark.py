"""Spin-1 kagome Heisenberg AD-iPESS benchmark.

Spin-1 analog of :mod:`examples.kagome_spin12_pess_ad_benchmark`. d=3
(``d_eff = 27``), Δ=1 isotropic. Reference: Picot et al., Phys. Rev. B
93, 060407(R) (2016) — large-D iPESS gives E/site ≈ −1.41.

Both the ``[SU only]`` readout and the AD stage go through the EXACT
supersite blocking (#991: ``build_pess_loss_exact`` /
``optimize_pess_ad(..., loss_builder="exact")``; ``T_d`` contracted
explicitly and optimized). Until #1002 this script measured and
optimized the Convention-C loss (``build_pess_loss``), whose CTM
collapses to rank-1 corners on SU-converged states and reads
backend-dependent values that are not the kagome energy — the
"E/site ≈ −1.0 at D=2" claim of the earlier docstring came through that
broken probe and should be discarded. On the exact path the D=2 SU
state (seed 0, default schedule) reads E/site = -1.270151,
consistent with the independent Husimi-tree probe
(:func:`tenax.algorithms.pess.pess_local_energy`: -1.269909 on the
same state, agreement 2.4e-4).

The default ``--sweep`` covers ``D ∈ {2, 4}`` (even only). The odd-D AD
instability documented for the Convention-C supersite came from its
dummy 4th leg's zero singular values in the CTM projector spectrum (see
:func:`tenax.algorithms.pess.pess_to_kagome_supersite`); the exact
blocking has no dummy leg, so that mechanism does not apply, but odd
``D`` on the exact path has not been characterised — the sweep default
stays even-D. Pass ``--include-odd-D`` to extend the sweep to
``{2, 3, 4}``.

Usage:
    python examples/kagome_spin1_pess_ad_benchmark.py --D 2 --chi 8
    python examples/kagome_spin1_pess_ad_benchmark.py --D 4 --chi 32 --max-iter 80
    python examples/kagome_spin1_pess_ad_benchmark.py --sweep    # D ∈ {2,4}, χ=2D²
    python examples/kagome_spin1_pess_ad_benchmark.py --sweep --include-odd-D
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
    kagome_xxz_pess_cg_gates_exact,
    pess_simple_update,
)
from tenax.algorithms.pess_optimize import build_pess_loss_exact, optimize_pess_ad

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
    cg_gates = kagome_xxz_pess_cg_gates_exact(delta=DELTA, d=D_PHYS)
    state = IPESSState.random(D=D, d=D_PHYS, key=jax.random.PRNGKey(seed))
    state = pess_simple_update(state, H, dt_schedule=list(su_steps), D_max=D)

    config = _make_ctm_config(chi=chi)
    loss_fn = build_pess_loss_exact(cg_gates, config)
    e_su = float(loss_fn(state).real)
    if verbose:
        print(f"  [SU only] E/site = {e_su:.6f}", flush=True)

    state, e_ad = optimize_pess_ad(
        state,
        cg_gates,
        config,
        max_iter=max_iter,
        verbose=verbose,
        loss_builder="exact",
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
        help="Run a default sweep over D ∈ {2, 4} (use --include-odd-D for {2,3,4}).",
    )
    parser.add_argument(
        "--include-odd-D",
        action="store_true",
        help=(
            "Include D=3 in the sweep. The historical odd-D AD instability "
            "was a Convention-C dummy-leg artifact that does not exist on "
            "the exact blocking this script now runs; odd D is simply "
            "uncharacterised there, so it stays opt-in."
        ),
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
        args.D_list = [2, 3, 4] if args.include_odd_D else [2, 4]
    else:
        args.D_list = [args.D]

    results = _sweep(args)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} result(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
