"""CTM unconverged-forward policy (#1059 hotspot 8, #1060).

Spec: docs/superpowers/specs/2026-10-01-ctm-unconverged-policy-design.md
"""

from __future__ import annotations

import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms._ctm_energy_ad as _cea
import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms._ctm_convergence_policy import (
    CTMNotConvergedError,
    CTMNotConvergedWarning,
    check_ctm_converged,
    format_step_multiplier,
)
from tenax.algorithms._ctm_python_loop import CTMConvergeInfo
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


def _info(converged, iterations=500, sv_diff=3.9e-7, **extra):
    info = CTMConvergeInfo(converged=converged, iterations=iterations, sv_diff=sv_diff)
    if not extra:
        return info

    class _WithExtra:
        pass

    obj = _WithExtra()
    for f in info._fields:
        setattr(obj, f, getattr(info, f))
    for k, v in extra.items():
        setattr(obj, k, v)
    return obj


def test_check_passes_when_converged():
    assert check_ctm_converged(_info(True), site="gradient", policy="raise") is True


def test_check_raises_with_diagnostics():
    with pytest.raises(CTMNotConvergedError) as ei:
        check_ctm_converged(
            _info(False),
            site="gradient",
            policy="raise",
            step=7,
            conv_tol=1e-10,
            chi=12,
        )
    msg = str(ei.value)
    for needle in ("gradient", "step 7", "500", "3.9e-07", "1e-10", "chi=12"):
        assert needle in msg, (needle, msg)
    assert ei.value.site == "gradient" and ei.value.step == 7


def test_check_warns_under_warn():
    with pytest.warns(CTMNotConvergedWarning, match="gradient"):
        out = check_ctm_converged(_info(False), site="gradient", policy="warn", step=3)
    assert out is False


def test_message_shows_numeric_step_multiplier():
    with pytest.raises(CTMNotConvergedError, match=r"step multiplier -0\.99"):
        check_ctm_converged(
            _info(False, step_multiplier=-0.99), site="g", policy="raise"
        )


@pytest.mark.parametrize("value", [None, float("nan")])
def test_message_shows_na_when_multiplier_missing_or_nan(value):
    info = _info(False) if value is None else _info(False, step_multiplier=value)
    assert format_step_multiplier(info) == "n/a"
    with pytest.raises(CTMNotConvergedError, match=r"step multiplier n/a"):
        check_ctm_converged(info, site="g", policy="raise")


def test_ctmconfig_default_and_validation():
    assert CTMConfig().on_unconverged == "raise"
    assert CTMConfig(on_unconverged="warn").on_unconverged == "warn"
    with pytest.raises(ValueError, match="on_unconverged"):
        CTMConfig(on_unconverged="bogus")


def test_public_exports():
    import tenax

    assert tenax.CTMNotConvergedError is CTMNotConvergedError
    assert tenax.CTMNotConvergedWarning is CTMNotConvergedWarning
    assert "CTMNotConvergedError" in tenax.__all__
    assert "CTMNotConvergedWarning" in tenax.__all__


# ---------------------------------------------------------------------------
# Task 2: site 1 (gradient forward) and optimizer recovery, 2-site and 1-site.
# unit_cell="1x1" routes optimize_gs_ad to _optimize_gs_ad_tensor (1-site).
# ---------------------------------------------------------------------------


def _heisenberg_gate():
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    return (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real


def _rand(D, d, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    return jnp.asarray(a / np.linalg.norm(a))


def _cfg(unit_cell, policy, *, max_iter=3, steps=3, ckpt=None, retries=2):
    return iPEPSConfig(
        unit_cell=unit_cell,
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=4, max_iter=max_iter, min_iter=1, conv_tol=1e-14, on_unconverged=policy
        ),
        gs_num_steps=steps,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=retries,
        su_init=False,
        gs_conv_criterion="grad_norm",
        return_history=True,
        gs_checkpoint_path=ckpt,
    )


def _init(unit_cell):
    return (_rand(2, 2, 0), _rand(2, 2, 1)) if unit_cell == "2site" else _rand(2, 2, 0)


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site1_raises_and_checkpoints(unit_cell, tmp_path):
    cfg = _cfg(unit_cell, "raise", ckpt=str(tmp_path / "ck"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(CTMNotConvergedError) as ei:
            _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    assert ei.value.site == "gradient"
    assert (tmp_path / "ck" / "ckpt.last.pkl").exists()


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site1_warn_completes_and_records(unit_cell):
    cfg = _cfg(unit_cell, "warn")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    history = out[-1]
    assert history["ctm_converged"], "no per-step convergence recorded"
    assert all(c is False for c in history["ctm_converged"])
    assert len(history["ctm_sv_diff"]) == len(history["ctm_converged"])
    assert len(history["ctm_step_multiplier"]) == len(history["ctm_converged"])


def test_site1_missing_diagnostic_skips_check(monkeypatch):
    """Review Focus 2: no forward_converged key -> no check, no stale reuse."""
    monkeypatch.setattr(_cea, "get_last_implicit_ad_diagnostics", lambda: {})
    cfg = _cfg("2site", "raise", max_iter=200, steps=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), cfg)
    assert out[-1]["ctm_converged"] == []


def test_reset_forward_diagnostics_pops_keys():
    _cea._F3_LAST_DIAGNOSTICS["forward_converged"] = False
    _cea._F3_LAST_DIAGNOSTICS["forward_stationarity_residual"] = 1.0
    _cea.reset_forward_diagnostics()
    d = _cea.get_last_implicit_ad_diagnostics()
    assert "forward_converged" not in d and "forward_stationarity_residual" not in d


def test_unconverged_at_best_raises_immediately(monkeypatch, tmp_path):
    """Review Focus 3: failure at best_params must not burn the stall budget."""
    calls = {"n": 0}
    real = _cea.get_last_implicit_ad_diagnostics

    def fake():
        d = real()
        if "forward_converged" in d:
            calls["n"] += 1
            d["forward_converged"] = False  # every gradient forward "fails"
        return d

    monkeypatch.setattr(_cea, "get_last_implicit_ad_diagnostics", fake)
    cfg = _cfg(
        "2site", "raise", max_iter=200, steps=10, retries=5, ckpt=str(tmp_path / "ck")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(CTMNotConvergedError):
            _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), cfg)
    # step 1 fails at the initial params, which are best_params -> raise at once
    assert calls["n"] == 1


def test_should_restore_best_env():
    """Review Focus 4: a best env at a stale chi must not be restored."""
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
    from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
    from tenax.algorithms.ipeps_optimize import _should_restore_best_env

    # ctm_tensor_2site rejects a bare jax.Array; wrap it as a DenseTensor.
    A = _wrap_as_dense_tensor(_rand(2, 2, 0))
    B = _wrap_as_dense_tensor(_rand(2, 2, 1))
    eA, eB = ctm_tensor_2site(A, B, chi=4, max_iter=5)
    envs = {(0, 0): eA, (1, 0): eB}
    assert _should_restore_best_env(envs, 4) is True
    assert _should_restore_best_env(envs, 6) is False
    assert _should_restore_best_env(None, 4) is False


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_unconverged_after_best_resets_and_recovers(unit_cell, monkeypatch, caplog):
    """Section 3: a failure away from best_params resets to best (logged with
    the spec's line) and the run continues instead of raising."""
    import logging

    calls = {"n": 0}
    real = _cea.get_last_implicit_ad_diagnostics

    def fake():
        d = real()
        if "forward_converged" in d:
            calls["n"] += 1
            if calls["n"] == 2:  # only the step-2 gradient forward "fails"
                d["forward_converged"] = False
        return d

    monkeypatch.setattr(_cea, "get_last_implicit_ad_diagnostics", fake)
    cfg = _cfg(unit_cell, "raise", max_iter=200, steps=3, retries=2)
    with warnings.catch_warnings(), caplog.at_level(logging.WARNING):
        warnings.simplefilter("ignore", UserWarning)
        out = _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    assert calls["n"] == 3
    assert any(
        "CTM forward not converged at step 2 (sweeps n/a" in r.getMessage()
        and "reset to best (#1/2)" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    # the failing step is not recorded; the two good steps are
    assert out[-1]["ctm_converged"] == [True, True]


# ---------------------------------------------------------------------------
# Task 2 fix round 1: the raise-and-checkpoint branches and the restore
# closure's chi-mismatch fallback, pinned on the 2-site loop.
# ---------------------------------------------------------------------------


def _fail_forwards(monkeypatch, failing):
    """Make the gradient forwards whose 1-based call index is in ``failing``
    report forward_converged=False."""
    calls = {"n": 0}
    real = _cea.get_last_implicit_ad_diagnostics

    def fake():
        d = real()
        if "forward_converged" in d:
            calls["n"] += 1
            if calls["n"] in failing:
                d["forward_converged"] = False
        return d

    monkeypatch.setattr(_cea, "get_last_implicit_ad_diagnostics", fake)
    return calls


def test_2site_stall_budget_exhausted_raises_and_checkpoints(
    monkeypatch, tmp_path, caplog
):
    """With no reset budget (retries=0) a failure away from best exceeds it
    and must raise (no reset) with the checkpoint written."""
    import logging

    _fail_forwards(monkeypatch, {2})
    cfg = _cfg(
        "2site", "raise", max_iter=200, steps=4, retries=0, ckpt=str(tmp_path / "ck")
    )
    with warnings.catch_warnings(), caplog.at_level(logging.WARNING):
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(CTMNotConvergedError):
            _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), cfg)
    assert not any("reset to best" in r.getMessage() for r in caplog.records)
    assert (tmp_path / "ck" / "ckpt.last.pkl").exists()


def test_2site_non_reset_recovery_raises_without_reset(monkeypatch, tmp_path, caplog):
    """gs_stall_recovery != "reset": the first failure away from best raises
    at once (no reset) and the checkpoint is written."""
    import logging
    from dataclasses import replace

    _fail_forwards(monkeypatch, {2})
    cfg = replace(
        _cfg("2site", "raise", max_iter=200, steps=4, ckpt=str(tmp_path / "ck")),
        gs_stall_recovery="noise",
    )
    with warnings.catch_warnings(), caplog.at_level(logging.WARNING):
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(CTMNotConvergedError):
            _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), cfg)
    assert not any("reset to best" in r.getMessage() for r in caplog.records)
    assert (tmp_path / "ck" / "ckpt.last.pkl").exists()


def test_2site_restore_closure_chi_mismatch_clears_env_cache(monkeypatch):
    """Through the 2-site closure: a best env at a stale chi is not restored;
    the reset clears the cache so the next forward cold-starts (env_init None),
    whereas a matching chi restores it (env_init not None)."""
    import tenax.algorithms.ad_utils as _adu
    import tenax.algorithms.ipeps_ad_policy as _pol

    def run(mismatch):
        seen = []
        real_kwargs = _pol.ctm_converge_kwargs

        def spy(cfg, env_init=None, **kw):
            seen.append(env_init is None)
            return real_kwargs(cfg, env_init=env_init, **kw)

        with monkeypatch.context() as m:
            m.setattr(_pol, "ctm_converge_kwargs", spy)
            _fail_forwards(m, {2})
            if mismatch:
                real_chi = _adu._env_chi
                m.setattr(_adu, "_env_chi", lambda envs: real_chi(envs) + 1)
            cfg = _cfg("2site", "raise", max_iter=200, steps=3, retries=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), cfg)
        return seen

    restored = run(mismatch=False)
    cleared = run(mismatch=True)
    assert len(restored) == len(cleared)
    assert cleared.count(True) > restored.count(True), (restored, cleared)
