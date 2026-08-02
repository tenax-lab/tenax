"""Integration test for the 1-site grad-spike guard (ported from 2-site, #524).

The 1-site implicit-AD optimizer occasionally produces a transient gradient
blowup (a non-variational CTM artifact); ``gs_grad_spike_ratio`` rolls such a
step back to best before the line search thrashes.  The guard is gated on the
config field (default ``None`` → off) and must not perturb a clean short run.

These tests verify the 1-site path accepts the field and runs to completion with
a finite, physical energy when the guard is enabled — catching port-level
regressions (broken loop control flow, missing state resets) without needing to
reproduce a real spike (which requires a slow, large-χ optimization).
"""

from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

from tenax import (
    CTMConfig,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
    sublattice_rotate_gate,
)

_QMC_GS = -0.6694


def _tiny_config(grad_spike_ratio):
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
        gs_recipe="1x1",
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_metric_precond=True,
        gs_num_steps=3,
        gs_energy_floor=_QMC_GS,
        gs_grad_spike_ratio=grad_spike_ratio,
        gs_verbose=False,
        su_init=True,
    )


def test_grad_spike_guard_1site_runs_and_is_physical():
    """1-site implicit optimize with the guard ENABLED completes; E is physical."""
    gate = sublattice_rotate_gate(heisenberg_gate())
    _A, _env, E = optimize_gs_ad(gate, None, _tiny_config(grad_spike_ratio=5.0))
    E = float(E)
    assert math.isfinite(E), f"energy not finite: {E}"
    # variational: a real D=2 iPEPS energy sits above the true GS and is bound.
    assert _QMC_GS <= E < 0.0, f"energy not physical: {E}"


def test_grad_spike_guard_off_matches_baseline():
    """With no spike in a clean short run, the guard is inert (same energy)."""
    gate = sublattice_rotate_gate(heisenberg_gate())
    _A0, _e0, E_off = optimize_gs_ad(gate, None, _tiny_config(grad_spike_ratio=None))
    _A1, _e1, E_on = optimize_gs_ad(gate, None, _tiny_config(grad_spike_ratio=5.0))
    # deterministic su_init + optimization → identical when the guard never fires
    assert math.isclose(float(E_off), float(E_on), rel_tol=0, abs_tol=1e-9), (
        f"guard altered a clean run: off={float(E_off)} on={float(E_on)}"
    )
