"""Phase C.2a invariant probe: D=2 multisite ≤ supersite.

Stop-and-ask checkpoint #4 from
``docs/plans/2026-05-05-multisite-kagome-pess.md``.

The 3-site multisite encoding has strictly more variational parameters
than the 1-site CG-iPEPS supersite encoding (``T_d`` is variational in
multisite, frozen-and-absorbed in supersite). On the same SU warm-start
and the same CTM χ, AD-optimised multisite energy must therefore be
lower than (or equal to within optimisation noise of) AD-optimised
supersite energy. A regression points at an AD-graph bug, not at
physics.

Usage::

    python examples/kagome_pess_multisite_invariant_probe.py --D 2 --chi 8

Asserts ``E_multisite ≤ E_supersite + 1e-6`` and writes both energies
plus the gap to ``examples/kagome_pess_multisite_invariant_probe.json``.

Reference values (manual run of the *reverted* C.1 attempt at ``ef6ade5``
on the *broken* 6-NN energy summation)::

    D=2, χ=8, 30 L-BFGS steps, SU(0.1×100, 0.01×100) warmstart:
      E_supersite = -0.3072
      E_multisite = -0.3707  (6.3% lower)

The corrected energy at ``d53909a`` may give a different multisite
number; the only assertion is the invariant ``E_ms ≤ E_ss + 1e-6``.
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
    kagome_xxz_pess_cg_gates,
    pess_simple_update,
)
from tenax.algorithms.pess_optimize import (
    build_pess_loss,
    build_pess_loss_3site_multisite,
    optimize_pess_3site_multisite_ad,
    optimize_pess_ad,
)

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


def run_invariant_probe(
    D: int,
    chi: int,
    max_iter: int,
    su_steps: tuple[tuple[float, int], ...] = (
        (0.1, 200),
        (0.01, 200),
        (0.001, 100),
    ),
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Run the C.2a probe and return a dict with both AD energies."""
    H = kagome_triangle_xxz_hamiltonian(delta=DELTA, d=D_PHYS)
    cg_gates = kagome_xxz_pess_cg_gates(delta=DELTA, d=D_PHYS)
    bond_gates = kagome_3site_bond_gates(delta=DELTA, d=D_PHYS)
    config = _make_ctm_config(chi=chi)

    state0 = IPESSState.random(D=D, d=D_PHYS, key=jax.random.PRNGKey(seed))
    state0 = pess_simple_update(state0, H, dt_schedule=list(su_steps), D_max=D)

    e_su_ss = float(build_pess_loss(cg_gates, config)(state0).real)
    e_su_ms = float(build_pess_loss_3site_multisite(bond_gates, config)(state0).real)
    if verbose:
        print(f"  [SU only] E_supersite = {e_su_ss:.6f}", flush=True)
        print(f"  [SU only] E_multisite = {e_su_ms:.6f}", flush=True)

    print("  Running supersite L-BFGS...", flush=True)
    t0 = time.perf_counter()
    _, e_ad_ss = optimize_pess_ad(
        state0, cg_gates, config, max_iter=max_iter, verbose=verbose
    )
    t_ss = time.perf_counter() - t0

    print("  Running multisite L-BFGS...", flush=True)
    t0 = time.perf_counter()
    _, e_ad_ms = optimize_pess_3site_multisite_ad(
        state0, bond_gates, config, max_iter=max_iter, verbose=verbose
    )
    t_ms = time.perf_counter() - t0

    return {
        "D": D,
        "chi": chi,
        "max_iter": max_iter,
        "seed": seed,
        "e_supersite_su": e_su_ss,
        "e_multisite_su": e_su_ms,
        "e_supersite_ad": e_ad_ss,
        "e_multisite_ad": e_ad_ms,
        "delta_ad": e_ad_ms - e_ad_ss,
        "wall_seconds_supersite": t_ss,
        "wall_seconds_multisite": t_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=2, help="Bond dimension D")
    parser.add_argument("--chi", type=int, default=8, help="CTM χ")
    parser.add_argument(
        "--max-iter", type=int, default=30, help="L-BFGS iterations per path"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-6, help="Invariant tolerance")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
        help="Path to JSON results file",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"=== Phase C.2a probe: D={args.D}, χ={args.chi} ===", flush=True)
    result = run_invariant_probe(
        D=args.D,
        chi=args.chi,
        max_iter=args.max_iter,
        seed=args.seed,
        verbose=args.verbose,
    )

    print(f"\n  E_supersite_AD = {result['e_supersite_ad']:.6f}", flush=True)
    print(f"  E_multisite_AD = {result['e_multisite_ad']:.6f}", flush=True)
    print(
        f"  Δ = E_ms - E_ss = {result['delta_ad']:+.6e}  "
        f"(invariant: ≤ +{args.tol:.0e})",
        flush=True,
    )

    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nWrote result to {args.output}", flush=True)

    # Stop-and-ask checkpoint #4: assert the invariant.
    assert result["delta_ad"] <= args.tol, (
        f"Phase C.2a invariant FAIL: E_multisite ({result['e_multisite_ad']:.6f}) "
        f"exceeds E_supersite ({result['e_supersite_ad']:.6f}) by "
        f"{result['delta_ad']:.6e} > {args.tol:.0e}. Multisite has strictly "
        "more variational parameters (T_d is real, not frozen) — a regression "
        "here points at an AD-graph bug, not at physics."
    )
    print("\n✓ Phase C.2a invariant PASS", flush=True)


if __name__ == "__main__":
    main()
