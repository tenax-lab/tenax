"""In-process optimizer runs must be order-independent (#973).

The module-level implicit-AD warm-start cache used to leak λ seeds across
``optimize_gs_ad`` invocations: two back-to-back runs with bit-identical
configs differed by ~8e-5 after three steps, because the second run's
adjoint solves warm-started from the first run's state.  That broke the
determinism assumption every in-process A/B comparison relies on
(parameter scans, benchmark ladders, the test suite itself).

These are the end-to-end regressions on the original repro config
(D=2, χ=4, 1x1 C4v, implicit AD, Hager-Zhang line search, metric
preconditioning, deterministic ``su_init``):

* repeated-run: two identical runs return bit-identical energies;
* A/B/A: an interposed different run must not shift a repeat.

Bit-equality (not a tolerance) is deliberate — the entry invalidation
restores exact reproducibility on the CPU backend, and any looser gate
would re-admit sub-tolerance leaks of exactly the kind being pinned.

The wiring tests (each entry point invalidates before tensor work) live
in ``test_warm_start_entry_invalidation_973.py`` (core bucket).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from tenax import (
    CTMConfig,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
    sublattice_rotate_gate,
)
from tenax.algorithms._ctm_energy_ad import invalidate_implicit_ad_warm_start


def _repro_config(num_steps: int = 3) -> iPEPSConfig:
    """The #973 repro: smallest config exercising the cached implicit backward."""
    return iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=20,
        dt=0.05,
        ctm=CTMConfig(
            chi=4,
            max_iter=50,
            conv_tol=1e-8,
            projector_method="svd",
            forward_gauge="phase",
        ),
        unit_cell="1x1",
        gs_c4v=True,
        gs_implicit_ad=True,
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_metric_precond=True,
        gs_num_steps=num_steps,
        gs_verbose=False,
        su_init=True,
    )


def _assert_regime() -> None:
    """The runs above must actually have gone through the cached backward.

    ``invalidate_implicit_ad_warm_start`` returns the number of cache
    entries carrying warm-start plumbing; zero would mean a refactor
    rerouted this config off ``_VJP_CACHE`` and these tests stopped
    testing anything.  Called only *after* the energy comparisons, since
    it clears the very seeds whose leakage is under test.
    """
    n = invalidate_implicit_ad_warm_start()
    assert n >= 1, (
        "repro config no longer routes through the cached implicit-AD "
        "backward (_VJP_CACHE untouched) — these determinism tests have "
        "gone vacuous; re-point them at the current implicit path"
    )


def test_back_to_back_identical_runs_are_bit_identical():
    gate = sublattice_rotate_gate(heisenberg_gate())
    _a1, _e1, E1 = optimize_gs_ad(gate, None, _repro_config())
    _a2, _e2, E2 = optimize_gs_ad(gate, None, _repro_config())
    assert float(E1) == float(E2), (
        f"#973 regression: identical in-process runs differ — "
        f"run1={float(E1)!r} run2={float(E2)!r} "
        f"(diff={abs(float(E1) - float(E2)):.3e}); a cross-run solver "
        "seed is leaking again"
    )
    _assert_regime()


def test_interposed_different_run_does_not_shift_a_repeat():
    gate = sublattice_rotate_gate(heisenberg_gate())
    _a1, _e1, E_a1 = optimize_gs_ad(gate, None, _repro_config())
    # B: same statics (no recompile), different trajectory — warms the
    # cache with a λ from a different parameter point.
    optimize_gs_ad(gate, None, _repro_config(num_steps=2))
    _a2, _e2, E_a2 = optimize_gs_ad(gate, None, _repro_config())
    assert float(E_a1) == float(E_a2), (
        f"#973 regression: A/B/A ordering shifted a repeated run — "
        f"A1={float(E_a1)!r} A2={float(E_a2)!r} "
        f"(diff={abs(float(E_a1) - float(E_a2)):.3e})"
    )
    _assert_regime()
