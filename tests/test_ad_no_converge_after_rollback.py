"""A dE measured across a rollback-to-best is not evidence of convergence.

Every recovery path in ``optimize_gs_ad`` that restores ``params =
best_params`` and continues (line-search stall reset, CTM-adjoint-error
reset, gradient-spike guard, stall-cap χ-stage advance) makes the NEXT step
re-evaluate a state that was already evaluated.  ``delta_energy = |E -
prev_energy|`` is then ~0 by construction, and under the default
``gs_conv_criterion="dE"`` the optimizer used to report ``converged=True``
with a large gradient.  Observed on a 2-site Heisenberg run::

    step 5/200 E=0.1048858544 dE=1.208e-01 |g|=2.189e+00
    [iPEPS-AD] stall #1, reset L-BFGS history (rollback to best, retry 1/5)
    step 6/200 E=0.1048858544 dE=3.045e-11 |g|=2.189e+00
    converged at step 6 (dE=3.045e-11 < tol=1.000e-06)

These tests force each rollback with a mocked signal (failed line search,
raised ``CTMRGGradientError``, spiked gradient norm, forced stage advance)
and assert the run is NOT reported converged.  They test the mechanism, not
convergence: every run is a handful of D=2, χ=4 steps.

The multisite loop additionally gates dE convergence on ``step > 5 and
stall_count == 0``, which already blocks the stall-reset and CTM-error
cases there (stall_count > 0 after those rollbacks); only the χ-stage
rollback (which zeroes stall_count) is reachable through that gate.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

import tenax.algorithms._line_search as _ls_mod
import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms._ad_primitives import CTMRGGradientError
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

RETRIES = 5  # > 5 so a post-rollback step clears the multisite warmup gate


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


def _config(unit_cell, **overrides):
    kw = dict(
        unit_cell=_unit_cell(unit_cell),
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        su_init=False,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=RETRIES,
        # Default criterion and tolerance -- the ones the defect fired under.
        gs_conv_criterion="dE",
        gs_conv_tol=1e-6,
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


def _always_fail_factory(calls):
    def _always_fail(_phi, _dphi, phi0, _slope, **_kwargs):
        # alpha=0, f(alpha)=phi0: no descent -> stall_count += 1 -> rollback.
        calls["n"] += 1
        return 0.0, phi0, False

    return _always_fail


UNIT_CELLS = ["1x1", "2site", "multisite"]


@pytest.mark.core
@pytest.mark.parametrize("unit_cell", UNIT_CELLS)
def test_stall_reset_rollback_does_not_converge_on_dE(monkeypatch, unit_cell):
    """Line search always fails: every step is a stall reset to best_params.

    The step after each reset re-evaluates best_params, dE ~ 0.  Correct
    behaviour is to exhaust the retry budget (RETRIES + 1 line searches) and
    report converged=False -- not to stop one step after the first reset.
    """
    calls = {"n": 0}
    monkeypatch.setattr(_ls_mod, "hager_zhang_line_search", _always_fail_factory(calls))
    history = _run(_config(unit_cell, gs_num_steps=20))

    assert history["converged"] is False, (
        f"{unit_cell}: reported converged after a stall rollback "
        f"({calls['n']} line searches) -- dE across a rollback is ~0 by "
        f"construction, not a convergence signal"
    )
    assert calls["n"] == RETRIES + 1, (
        f"{unit_cell}: expected the stall budget to be exhausted "
        f"({RETRIES + 1} line searches), got {calls['n']}"
    )


@pytest.mark.core
@pytest.mark.parametrize("unit_cell", UNIT_CELLS)
def test_stall_cap_stage_advance_rollback_does_not_converge_on_dE(
    monkeypatch, unit_cell
):
    """Stall-cap χ-stage advance rolls back to best_params with stall_count=0.

    ``_advance_chi_stage_if_due`` is mocked to report one bump at the stall
    cap (χ left unchanged, so the post-rollback energy is bit-identical).
    """
    calls = {"n": 0}
    fired = {"n": 0}
    monkeypatch.setattr(_ls_mod, "hager_zhang_line_search", _always_fail_factory(calls))

    def _fake_advance(ctm_cfg, env_cache, **kw):
        idx = kw["current_stage_idx"]
        cfg = kw["config"]
        if fired["n"] == 0 and kw["stall_count"] > cfg.gs_stall_recovery_retries:
            fired["n"] += 1
            return ctm_cfg, env_cache, idx + 1, True, False
        return ctm_cfg, env_cache, idx, False, False

    monkeypatch.setattr(_opt, "_advance_chi_stage_if_due", _fake_advance)
    history = _run(
        _config(
            unit_cell,
            gs_num_steps=30,
            gs_chi_schedule_steps=[(4, 1000), (4, 1000)],
        )
    )

    assert fired["n"] == 1, f"{unit_cell}: stage-advance rollback never fired"
    assert history["converged"] is False, (
        f"{unit_cell}: reported converged right after the stall-cap "
        f"stage-advance rollback ({calls['n']} line searches)"
    )
    # Two full stall budgets: one before the advance, one after.
    assert calls["n"] == 2 * (RETRIES + 1), (
        f"{unit_cell}: expected {2 * (RETRIES + 1)} line searches, got {calls['n']}"
    )


@pytest.mark.core
@pytest.mark.parametrize("unit_cell", UNIT_CELLS)
def test_ctm_error_reset_rollback_does_not_converge_on_dE(monkeypatch, unit_cell):
    """A CTMRGGradientError at step 1 rolls back to the step-0 params.

    Adam (no line search) so step 0 genuinely moves the params; step 2 then
    re-evaluates the step-0 params and sees dE ~ 0 against step 0's energy.
    """
    raised = {"n": 0}
    calls = {"n": 0}
    orig_take = _opt._AcceptedProbeEval.take

    def _take(self, params, cfg):
        calls["n"] += 1
        if calls["n"] == 2:
            raised["n"] += 1
            raise CTMRGGradientError(spectral_radius=1.5)
        return orig_take(self, params, cfg)

    monkeypatch.setattr(_opt._AcceptedProbeEval, "take", _take)
    history = _run(
        _config(unit_cell, gs_num_steps=4, gs_optimizer="adam", gs_learning_rate=1e-2)
    )

    assert raised["n"] == 1, f"{unit_cell}: CTM-error path never exercised"
    assert history["converged"] is False, (
        f"{unit_cell}: reported converged right after the CTM-error rollback"
    )


@pytest.mark.core
@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_grad_spike_rollback_does_not_converge_on_dE(monkeypatch, unit_cell):
    """The |g|-spike guard rolls back to best_params before the line search.

    ``_grad_l2_norm`` is mocked to spike once (step 1); step 2 re-evaluates
    the step-0 params.  (The multisite loop has no spike guard.)
    """
    calls = {"n": 0}
    spiked = {"n": 0}
    orig = _opt._grad_l2_norm

    def _gnorm(grads):
        calls["n"] += 1
        val = orig(grads)
        if calls["n"] == 2:
            spiked["n"] += 1
            return 1e6
        return val

    monkeypatch.setattr(_opt, "_grad_l2_norm", _gnorm)
    history = _run(
        _config(
            unit_cell,
            gs_num_steps=4,
            gs_optimizer="adam",
            gs_learning_rate=1e-2,
            gs_grad_spike_ratio=10.0,
        )
    )

    assert spiked["n"] == 1, f"{unit_cell}: spike never injected"
    assert history["converged"] is False, (
        f"{unit_cell}: reported converged right after the |g|-spike rollback"
    )


@pytest.mark.core
@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_rollback_flag_survives_checkpoint_resume(monkeypatch, tmp_path, unit_cell):
    """A checkpoint written right after a rollback must carry the flag.

    Phase A runs one step whose line search fails, so it ends on a stall
    rollback (params == best_params) and checkpoints that state.  Phase B
    resumes: its first step re-evaluates best_params, dE ~ 0 against the saved
    prev_energy, and must not be read as convergence.  (Multisite has no
    checkpointing: gs_checkpoint_path is rejected for Lattice unit cells.)
    """
    calls = {"n": 0}
    monkeypatch.setattr(_ls_mod, "hager_zhang_line_search", _always_fail_factory(calls))
    ckpt = str(tmp_path / f"ckpt_{unit_cell}")
    common = dict(gs_checkpoint_path=ckpt, gs_checkpoint_every=1)

    _run(_config(unit_cell, gs_num_steps=1, gs_resume=False, **common))
    assert calls["n"] == 1, "phase A must end on exactly one stall rollback"

    history = _run(_config(unit_cell, gs_num_steps=2, gs_resume=True, **common))
    assert history["converged"] is False, (
        f"{unit_cell}: resumed run converged on its first step -- the "
        f"rollback flag was not checkpointed/restored"
    )


@pytest.mark.core
def test_chi_ceiling_bailout_flag_survives_checkpoint_resume(tmp_path):
    """The 2-site chi-ceiling bail-out rolls back and checkpoints -- flag it.

    Phase A pins chi at chi_max with an auto-bump threshold every step
    exceeds, so step 1 bails out to best_params and saves.  Phase B resumes
    with the bail-out off: its first step re-evaluates the saved best state,
    and dE ~ 0 must not be read as convergence.
    """
    ckpt = str(tmp_path / "ckpt_bailout")
    ctm = CTMConfig(chi=4, chi_max=4, chi_auto_bump_eps=1e-300)
    common = dict(ctm=ctm, gs_checkpoint_path=ckpt, gs_checkpoint_every=1)

    phase_a = _run(_config("2site", gs_num_steps=5, gs_chi_ceiling_bailout=1, **common))
    assert len(phase_a["energies"]) < 5, "phase A must end on the bail-out"

    history = _run(
        _config(
            "2site", gs_num_steps=6, gs_chi_ceiling_bailout=0, gs_resume=True, **common
        )
    )
    assert history["converged"] is False, (
        "resumed run converged on its first step -- the chi-ceiling bail-out "
        "did not flag its rollback in the checkpoint"
    )


@pytest.mark.core
@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_legacy_checkpoint_without_flag_does_not_converge_on_resume(
    monkeypatch, tmp_path, unit_cell
):
    """A pre-#1073 checkpoint has no ``rolled_back`` key -- assume it rolled back.

    Phase A ends on a stall rollback and checkpoints it; the key is then
    stripped to mimic an old file.  Whether that old run had just rolled back
    is unknowable, so the resume must invalidate its first dE test rather
    than converge on dE ~ 0.
    """
    from tenax.algorithms._checkpoint import load_checkpoint, save_checkpoint

    calls = {"n": 0}
    monkeypatch.setattr(_ls_mod, "hager_zhang_line_search", _always_fail_factory(calls))
    ckpt = str(tmp_path / f"ckpt_legacy_{unit_cell}")
    common = dict(gs_checkpoint_path=ckpt, gs_checkpoint_every=1)

    _run(_config(unit_cell, gs_num_steps=1, gs_resume=False, **common))
    bundle = load_checkpoint(ckpt)
    assert bundle.pop("rolled_back") is True
    save_checkpoint(bundle, ckpt)

    history = _run(_config(unit_cell, gs_num_steps=2, gs_resume=True, **common))
    assert history["converged"] is False, (
        f"{unit_cell}: a resume from a checkpoint with no rolled_back key "
        f"converged on its first dE"
    )
