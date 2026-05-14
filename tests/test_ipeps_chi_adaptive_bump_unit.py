"""Truth-table unit tests for ``_advance_chi_stage_if_due`` (#455 PR 2).

These tests inject the signal inputs directly and assert state
transitions on the helper — no optimizer run, no JAX, milliseconds
each. Per ``feedback_test_mechanism_not_convergence``, convergence on
a real physics problem is the production benchmark's job, not a
unit test.

Marker: ``core``.
"""

import pytest

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import _advance_chi_stage_if_due


def _make_config(
    *,
    conv_criterion="grad_norm",
    grad_norm_tol=1e-5,
    conv_tol=1e-8,
    stall_recovery="reset",
    stall_retries=3,
):
    """Build an iPEPSConfig pinned to test-relevant fields."""
    return iPEPSConfig(
        ctm=CTMConfig(chi=2, chi_max=8),
        gs_conv_criterion=conv_criterion,
        gs_grad_norm_tol=grad_norm_tol,
        gs_conv_tol=conv_tol,
        gs_stall_recovery=stall_recovery,
        gs_stall_recovery_retries=stall_retries,
    )


@pytest.mark.core
def test_budget_exhausted_non_final_advances():
    """Test #1: budget hit at non-final stage → bump, advance, no break."""
    ctm = CTMConfig(chi=2, chi_max=8)
    env = {}
    schedule = [(2, 3), (4, 3)]

    new_ctm, _new_env, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm,
        env,
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=3,
        config=_make_config(),
        grad_norm=1e3,
        delta_energy=1e3,
        stall_count=0,
    )
    assert bump_fired is True
    assert should_break is False
    assert new_idx == 1
    assert new_ctm.chi == 4


@pytest.mark.core
def test_grad_norm_signal_non_final_advances():
    """Test #2: grad_norm < tol with criterion=grad_norm → bump."""
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    _, _, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm,
        {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,  # well within budget
        config=_make_config(conv_criterion="grad_norm", grad_norm_tol=1e-3),
        grad_norm=1e-6,
        delta_energy=1.0,
        stall_count=0,
    )
    assert bump_fired is True
    assert should_break is False
    assert new_idx == 1


@pytest.mark.core
def test_dE_signal_non_final_advances():
    """Test #3: |dE| < tol with criterion=dE → bump."""
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    _, _, new_idx, bump_fired, _ = _advance_chi_stage_if_due(
        ctm,
        {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,
        config=_make_config(conv_criterion="dE", conv_tol=1e-3),
        grad_norm=1.0,
        delta_energy=1e-6,
        stall_count=0,
    )
    assert bump_fired is True
    assert new_idx == 1


@pytest.mark.core
def test_stall_cap_reset_advances():
    """Test #4: stall_count ≥ retries with recovery=reset → bump."""
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    _, _, new_idx, bump_fired, _ = _advance_chi_stage_if_due(
        ctm,
        {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,
        config=_make_config(stall_recovery="reset", stall_retries=3),
        grad_norm=1.0,
        delta_energy=1.0,
        stall_count=3,
    )
    assert bump_fired is True
    assert new_idx == 1


@pytest.mark.core
def test_stall_cap_noise_does_not_advance():
    """Test #5: stall_count ≥ retries but recovery=noise → no bump.

    Noise path has its own retry budget; PR 2 explicitly gates the
    stall-cap bump signal to recovery=reset.
    """
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    _, _, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm,
        {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,
        config=_make_config(stall_recovery="noise", stall_retries=3),
        grad_norm=1.0,
        delta_energy=1.0,
        stall_count=99,
    )
    assert bump_fired is False
    assert should_break is False
    assert new_idx == 0


@pytest.mark.core
def test_final_stage_any_signal_breaks():
    """Test #6: at final stage, any signal returns should_break=True, no bump."""
    ctm = CTMConfig(chi=4, chi_max=8)
    schedule = [(2, 3), (4, 3)]

    # Budget signal at final stage.
    _, _, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm,
        {},
        chi_schedule=schedule,
        current_stage_idx=1,
        steps_in_stage=3,
        config=_make_config(),
        grad_norm=1.0,
        delta_energy=1.0,
        stall_count=0,
    )
    assert bump_fired is False
    assert should_break is True
    assert new_idx == 1


@pytest.mark.core
def test_no_signal_no_action():
    """Test #7: no signal tripped → no-op."""
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    new_ctm, _, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm,
        {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,
        config=_make_config(),
        grad_norm=1.0,
        delta_energy=1.0,
        stall_count=0,
    )
    assert bump_fired is False
    assert should_break is False
    assert new_idx == 0
    assert new_ctm.chi == 2


@pytest.mark.core
def test_simultaneous_signals_advance_once():
    """Test #8: grad_norm AND stall-cap together → single bump (idempotent)."""
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    _, _, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm,
        {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,
        config=_make_config(grad_norm_tol=1e-3, stall_retries=3),
        grad_norm=1e-6,
        delta_energy=1.0,
        stall_count=99,
    )
    assert bump_fired is True
    assert new_idx == 1  # advanced exactly once, not twice
    assert should_break is False
