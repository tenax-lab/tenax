"""Phase C.3 probe: D=4 multisite vs Liao 2017 asymptotic floor.

Stop-and-ask checkpoint #5 from
``docs/plans/2026-05-05-multisite-kagome-pess.md``.

The variational principle requires ``E_multisite ≥ E_GS_kagome_AFH``,
where the best literature estimate of the kagome S=1/2 antiferromagnetic
Heisenberg ground-state energy per site is the Liao 2017 asymptotic
``E_∞ = -0.43752`` (Liao et al., PRL 118, 137202 (2017),
arXiv:1610.04727, Fig. 1(b) inset).  Any AD-converged multisite energy
*below* this is variationally impossible and signals a bug in the
energy formula.

The reverted Phase C.1 attempt at ``ef6ade5`` (built on the broken
6-NN energy summation) gave ``E_ms = -0.4766`` at D=4, χ=16, which is
the failure signature for that bug.  This probe asserts that the
rebuilt energy formula at ``d53909a`` does NOT reproduce the anomaly.

Reference values at D=4 from
``examples/kagome_spin12_pess_liao2017_replication.json``::

    Liao 2017 reported (D=4):                -0.429
    P1 Husimi-tree probe (best at D=4 SU):   -0.420
    P2 Convention-C CTM (SU only):           -0.347
    Liao 2017 asymptotic (D→∞):              -0.43752  ← variational floor

Expected outcome at D=4, χ=16:
* multisite encoding has strictly more parameters than supersite
  (Convention C), so should beat the P2 SU floor of -0.347;
* no diagonal-RDM term, so reaches further into the variational
  manifold than the supersite AD path (which stalls near -0.343 at D=4
  per ``project_pess_ad_stalls_d4.md``);
* must remain above the Liao asymptotic -0.43752 (variational principle).

Plausible target: ``E_ms ∈ [-0.42, -0.36]`` at D=4, χ=16, 30 L-BFGS
steps.  Tighter than the supersite AD ceiling, looser than the Liao
floor.

Usage::

    python examples/kagome_pess_multisite_phase_c3_probe.py --D 4 --chi 16
    python examples/kagome_pess_multisite_phase_c3_probe.py --D 4 --chi 24 --max-iter 50
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

DELTA = 1.0  # isotropic Heisenberg
D_PHYS = 2  # spin-½

# Variational floor: Liao 2017 D→∞ asymptotic.  Any E_ms below this is
# variationally impossible and signals a broken energy formula.
LIAO_ASYMPTOTIC = -0.43752


def _make_ctm_config(chi: int, max_iter: int = 100) -> CTMConfig:
    """Tight inner CTM for D=4 multisite optimisation.

    The looser ``max_iter=30, conv_tol=1e-7`` defaults used at D=2 leave
    enough CTM convergence noise at D=4 χ=16 to create spurious
    sub-Liao-floor wells in the energy landscape (verified by the HZ
    diagnostic: ``phi(α=0.25) = -0.487`` even though the true energy of
    that state is ~-0.20).  Tightening to ``max_iter=100, conv_tol=1e-9``
    is ~3× more inner CTM iterations per energy eval but the gradients
    become physical.
    """
    return CTMConfig(
        chi=chi,
        max_iter=max_iter,
        min_iter=4,
        conv_tol=1e-9,
        projector_method="svd",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
        gmres_tol=1e-5,
        gmres_maxiter=80,
        gmres_restart=30,
        chi_ramp=None,
    )


def run_c3_probe(
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
    """Run the C.3 probe and return a dict with the multisite AD energy."""
    H = kagome_triangle_xxz_hamiltonian(delta=DELTA, d=D_PHYS)
    bond_gates = kagome_3site_bond_gates(delta=DELTA, d=D_PHYS)
    config = _make_ctm_config(chi=chi)

    print(f"  Running SU warmstart (D={D})...", flush=True)
    t0 = time.perf_counter()
    state0 = IPESSState.random(D=D, d=D_PHYS, key=jax.random.PRNGKey(seed))
    state0 = pess_simple_update(state0, H, dt_schedule=list(su_steps), D_max=D)
    t_su = time.perf_counter() - t0

    e_su_ms = float(build_pess_loss_3site_multisite(bond_gates, config)(state0).real)
    print(f"  [SU only] E_multisite = {e_su_ms:.6f}", flush=True)

    print(f"  Running multisite L-BFGS ({max_iter} steps)...", flush=True)
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
        "e_multisite_su": e_su_ms,
        "e_multisite_ad": e_ad_ms,
        "liao_asymptotic": LIAO_ASYMPTOTIC,
        "delta_to_liao": e_ad_ms - LIAO_ASYMPTOTIC,
        "wall_seconds_su": t_su,
        "wall_seconds_ad": t_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=4, help="Bond dimension D")
    parser.add_argument("--chi", type=int, default=16, help="CTM χ")
    parser.add_argument("--max-iter", type=int, default=30, help="L-BFGS iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
        help="Path to JSON results file",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"=== Phase C.3 probe: D={args.D}, χ={args.chi} ===", flush=True)
    print(f"  Liao 2017 asymptotic floor: {LIAO_ASYMPTOTIC}", flush=True)
    result = run_c3_probe(
        D=args.D,
        chi=args.chi,
        max_iter=args.max_iter,
        seed=args.seed,
        verbose=args.verbose,
    )

    print(f"\n  E_multisite_AD = {result['e_multisite_ad']:.6f}", flush=True)
    print(
        f"  Δ to Liao floor = {result['delta_to_liao']:+.6e}  "
        "(variational principle: ≥ 0)",
        flush=True,
    )

    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nWrote result to {args.output}", flush=True)

    # Stop-and-ask checkpoint #5: assert the Liao variational floor.
    # Any E_ms below the asymptotic D→∞ ground-state energy is impossible
    # at finite D — the reverted ef6ade5 attempt gave -0.4766 here, which
    # is the failure signature of the broken 6-NN energy summation.
    assert result["e_multisite_ad"] >= LIAO_ASYMPTOTIC, (
        f"Phase C.3 invariant FAIL: E_multisite ({result['e_multisite_ad']:.6f}) "
        f"is BELOW the Liao 2017 asymptotic floor ({LIAO_ASYMPTOTIC}) by "
        f"{LIAO_ASYMPTOTIC - result['e_multisite_ad']:.6e}. This is "
        "variationally impossible at finite D — the reverted attempt at "
        "ef6ade5 gave -0.4766 here, that's the failure signature of the "
        "broken 6-NN energy formula.  Diagnose the multisite energy path "
        "(``compute_energy_pess_3site_multisite``) before proceeding."
    )
    print("\n✓ Phase C.3 invariant PASS (E_ms ≥ Liao asymptotic)", flush=True)


if __name__ == "__main__":
    main()
