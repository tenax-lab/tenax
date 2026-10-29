"""A line-search accept whose converged energy rose above φ0 is rolled back.

The line search accepts α on its φ probe: a forward-only CTM warm-started
from the step's env.  A probe that stops on a CTM plateau can report a
decrease that the converged fixed point does not have (S5, 2026-10-29: the
probe saw E below φ0 = -0.5729735284; the next step's converged gradient
forward gave -0.5728248944).  With ``gs_reject_energy_rise`` (default on)
the next step compares its converged energy with the φ0 of the line search
that produced the point, and rolls back to ``best_params`` on a rise.

The tests mock the line search to "accept" a step taken UPHILL (negative α
along a descent direction) with a fake decrease, so every accepted point is
a false accept.  They test the mechanism, not convergence: D=2, χ=4.
"""

from __future__ import annotations

import math
import warnings

import jax.numpy as jnp
import pytest

import tenax.algorithms._line_search as _ls_mod
import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

RETRIES = 3
NUM_STEPS = 12


def _heisenberg_gate():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def _unit_cell(name):
    if name == "multisite":
        from tenax.core.lattice import checkerboard

        return checkerboard()
    return name


def _config(unit_cell, method, **overrides):
    kw = dict(
        unit_cell=_unit_cell(unit_cell),
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        su_init=False,
        gs_num_steps=NUM_STEPS,
        gs_line_search_method=method,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=RETRIES,
        # Never converge on dE/|g|: the run length is the observable.
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e-30,
        return_history=True,
    )
    kw.update(overrides)
    return iPEPSConfig(**kw)


def _run(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        out = _opt.optimize_gs_ad(_heisenberg_gate(), None, cfg)
    history = out[-1]
    assert isinstance(history, dict)
    return history


def _patch_uphill_accept(monkeypatch, method, calls):
    """Every line search 'accepts' an uphill step with a fake decrease."""
    if method == "hager_zhang":

        def _hz(_phi, _dphi, phi0, _slope, **kwargs):
            calls["n"] += 1
            return -0.5 * kwargs["alpha_init"], phi0 - 1.0, True

        monkeypatch.setattr(_ls_mod, "hager_zhang_line_search", _hz)
    else:

        def _bt(params, direction, grad, energy, loss_fn_fwd, **_kw):
            calls["n"] += 1
            ratio = _opt._tree_dot(params, params) / _opt._tree_dot(
                direction, direction
            )
            step = -0.1 * math.sqrt(ratio)
            new = _opt._normalize_params(
                _opt._tree_add(params, _opt._tree_scale(direction, step))
            )
            return new, energy - 1.0, step

        monkeypatch.setattr(_opt, "_backtracking_line_search", _bt)


UNIT_CELLS = ["1x1", "2site", "multisite"]
METHODS = ["hager_zhang", "armijo"]


@pytest.mark.core
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("unit_cell", UNIT_CELLS)
def test_false_accept_is_rolled_back(monkeypatch, unit_cell, method):
    """Each false accept is rejected; RETRIES + 1 in a row end the run."""
    calls = {"n": 0}
    _patch_uphill_accept(monkeypatch, method, calls)
    history = _run(_config(unit_cell, method))
    energies = history["energies"]

    # Regime: the mocked accept really moved uphill (else the test is vacuous).
    assert energies[1] > energies[0] + 1e-9, energies[:2]
    # Accept, reject, (rolled back) accept, reject, ... until RETRIES + 1
    # consecutive rejects: RETRIES + 1 line searches, 2 * (RETRIES + 1) steps.
    assert calls["n"] == RETRIES + 1, (unit_cell, method, calls["n"], energies)
    assert len(energies) == 2 * (RETRIES + 1), (unit_cell, method, energies)
    # Every step after a reject re-measures the best (first) point.
    for e in energies[2::2]:
        assert e == pytest.approx(energies[0], abs=1e-7)
    assert history["converged"] is False


@pytest.mark.core
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("unit_cell", UNIT_CELLS)
def test_flag_off_keeps_the_false_accept(monkeypatch, unit_cell, method):
    """``gs_reject_energy_rise=False`` keeps the old behaviour."""
    calls = {"n": 0}
    _patch_uphill_accept(monkeypatch, method, calls)
    history = _run(_config(unit_cell, method, gs_reject_energy_rise=False))
    energies = history["energies"]

    assert calls["n"] == NUM_STEPS, (unit_cell, method, calls["n"])
    assert len(energies) == NUM_STEPS
    assert energies[-1] > energies[0] + 1e-9


def test_helper_flags_a_rise():
    assert _opt._accepted_step_rose(-0.5, (-0.6, 4), chi=4, rolled_back=False)


def test_helper_ignores_noise_and_decrease():
    noise = _opt._STALL_NOISE_FLOOR
    assert not _opt._accepted_step_rose(
        -0.6 + 0.5 * noise, (-0.6, 4), chi=4, rolled_back=False
    )
    assert not _opt._accepted_step_rose(-0.7, (-0.6, 4), chi=4, rolled_back=False)


def test_helper_skips_without_a_comparable_phi0():
    # No accepted line search pending.
    assert not _opt._accepted_step_rose(-0.5, None, chi=4, rolled_back=False)
    # A rollback since the accept: the point is best_params, not the trial.
    assert not _opt._accepted_step_rose(-0.5, (-0.6, 4), chi=4, rolled_back=True)
    # chi changed since the accept: the energies are not comparable.
    assert not _opt._accepted_step_rose(-0.5, (-0.6, 4), chi=6, rolled_back=False)
    # Non-finite energies are never a rise.
    assert not _opt._accepted_step_rose(
        float("inf"), (-0.6, 4), chi=4, rolled_back=False
    )
