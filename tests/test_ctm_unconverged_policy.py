"""CTM unconverged-forward policy (#1059 hotspot 8, #1060).

Spec: docs/superpowers/specs/2026-10-01-ctm-unconverged-policy-design.md
"""

from __future__ import annotations

import math
import re
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


def test_message_reports_stationarity_separately():
    """Final-review I3: the #841 stationarity residual is printed under its own
    label and its own threshold, never as the loop's sv_diff."""
    info = _info(False, stationarity_residual=7e-8, stationarity_threshold=1e-7)
    with pytest.raises(CTMNotConvergedError) as ei:
        check_ctm_converged(info, site="gradient", policy="raise", conv_tol=1e-9)
    msg = str(ei.value)
    assert "sweeps 500, sv_diff 3.9e-07 vs conv_tol 1e-09" in msg, msg
    assert "stationarity residual 7e-08 (#841 threshold 1e-07)" in msg, msg
    # absent -> not printed at all
    with pytest.raises(CTMNotConvergedError) as ei:
        check_ctm_converged(_info(False), site="gradient", policy="raise")
    assert "stationarity" not in str(ei.value)


def test_message_names_the_plateau_bail():
    """Final-review I3: a plateau bail (the returned env trails the last sweep)
    says so, and that raising max_iter alone will not help."""
    bailed = _info(False, iterations=45, best_iteration=25)
    with pytest.raises(CTMNotConvergedError) as ei:
        check_ctm_converged(bailed, site="gradient", policy="raise")
    msg = str(ei.value)
    assert "plateau" in msg and "max_iter alone will not help" in msg, msg
    for hint in ("plateau_patience", "on_unconverged='warn'", "chi"):
        assert hint in msg, (hint, msg)
    # budget exhausted (best_iteration == iterations) -> no plateau claim
    spent = _info(False, iterations=500, best_iteration=500)
    with pytest.raises(CTMNotConvergedError) as ei:
        check_ctm_converged(spent, site="gradient", policy="raise")
    assert "plateau" not in str(ei.value)


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
    # Final-review I3: the loop's own sweep count and sv_diff, not "n/a" and
    # not the #841 residual; the residual comes separately with its threshold.
    msg = str(ei.value)
    assert "sweeps n/a" not in msg and "sweeps 3," in msg, msg
    assert ei.value.info.iterations == 3
    assert "stationarity residual" in msg and "#841 threshold 1e-08" in msg, msg


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
    assert len(history["ctm_stationarity"]) == len(history["ctm_converged"])


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site1_history_sv_diff_is_the_loops(unit_cell, monkeypatch):
    """Final-review I3: history ctm_sv_diff holds the CTM loop's own metric
    (what the loop compared with conv_tol), and ctm_stationarity the #841
    residual, not the other way round."""
    real = _cea._run_ctm_loop_with_bump
    loop_sv = []

    def spy(*a, **k):
        res = real(*a, **k)
        loop_sv.append(float(res.sv_diff))
        return res

    monkeypatch.setattr(_cea, "_run_ctm_loop_with_bump", spy)
    real_diag = _cea.get_last_implicit_ad_diagnostics
    residuals = []

    def diag_spy():
        d = real_diag()
        if "forward_converged" in d:
            residuals.append(d["forward_stationarity_residual"])
        return d

    monkeypatch.setattr(_cea, "get_last_implicit_ad_diagnostics", diag_spy)
    cfg = _cfg(unit_cell, "warn")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    history = out[-1]
    assert history["ctm_sv_diff"] and loop_sv
    for v in history["ctm_sv_diff"]:
        assert v in loop_sv, (v, loop_sv)
    assert history["ctm_stationarity"] == residuals


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
        re.search(r"CTM forward not converged at step 2 \(sweeps \d+,", r.getMessage())
        and "reset to best (#1/2)" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    # the failing step is not recorded; the two good steps are
    assert out[-1]["ctm_converged"] == [True, True]


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_reset_does_not_false_converge_under_de(unit_cell, monkeypatch):
    """Final-review I1: a reset puts params back on best_params, which the
    step before evaluated from a converged env.  The re-evaluation therefore
    reproduces the previous energy to ~1e-16, and with the default "dE"
    criterion the run used to stop right there reporting converged=True.
    The reset must forget prev_energy so the run carries on."""
    import dataclasses

    calls = _fail_forwards(monkeypatch, {2})
    cfg = dataclasses.replace(
        _cfg(unit_cell, "raise", max_iter=200, steps=6, retries=2),
        gs_conv_criterion="dE",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    history = out[-1]
    # forward 1 = step 1, forward 2 = the failing step, forward 3 = the
    # re-evaluation of best_params straight after the reset.  Stopping there
    # is the false convergence.
    assert calls["n"] > 3, (calls["n"], history["converged"], history["energies"])


def test_1site_reset_reinits_optax_lbfgs_state(monkeypatch):
    """Final-review I2: the 1-site CTMNotConvergedError reset re-initialises
    the optax L-BFGS state, as the 2-site reset does."""
    import dataclasses

    import optax

    init_calls = {"n": 0}
    real_build = _opt._build_optimizer

    def counting_build(cfg):
        optimizer = real_build(cfg)
        if optimizer is None:
            return None
        real_init = optimizer.init

        def counting_init(params):
            init_calls["n"] += 1
            return real_init(params)

        return optax.GradientTransformation(init=counting_init, update=optimizer.update)

    monkeypatch.setattr(_opt, "_build_optimizer", counting_build)
    calls = _fail_forwards(monkeypatch, {2})
    cfg = dataclasses.replace(
        _cfg("1x1", "raise", max_iter=200, steps=3, retries=2),
        gs_optimizer="lbfgs",
        gs_metric_precond=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _opt.optimize_gs_ad(_heisenberg_gate(), _init("1x1"), cfg)
    assert calls["n"] >= 2, "the failing forward never ran"
    # once at set-up, once from the reset
    assert init_calls["n"] >= 2, init_calls


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


# ---------------------------------------------------------------------------
# Task 3: site 2 -- the warm-start env cache never keeps an unconverged env.
# ---------------------------------------------------------------------------

import tenax.algorithms._ctm_python_loop as _cpl  # noqa: E402


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
@pytest.mark.parametrize("policy", ["raise", "warn"])
def test_site2_refuses_to_cache_unconverged(monkeypatch, policy, unit_cell):
    """Spy on the env each python_loop call is SEEDED with. An unconverged
    warm-start refresh must not become the next call's env_init."""
    import inspect

    real = _cpl.python_loop_ctm_converge
    refresh = "_update_env_cache_2s" if unit_cell == "2site" else "_update_env_cache"
    seen = []
    poisoned = {"env": None, "armed": True}

    def spy(*a, **k):
        seen.append(k.get("env_init"))
        envs, info = real(*a, **k)
        in_refresh = any(f.function == refresh for f in inspect.stack(0))
        if poisoned["armed"] and in_refresh and k.get("env_init") is not None:
            poisoned["armed"] = False
            poisoned["env"] = envs
            return envs, info._replace(converged=False)
        return envs, info

    monkeypatch.setattr(_cpl, "python_loop_ctm_converge", spy)
    cfg = _cfg(unit_cell, policy, max_iter=300, steps=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    assert poisoned["env"] is not None, "spy never saw a warm refresh"
    assert all(e is not poisoned["env"] for e in seen), (
        "unconverged env was used as a seed"
    )


def _fake_info(converged):
    return CTMConvergeInfo(converged=converged, iterations=3, sv_diff=1.4)


@pytest.mark.parametrize("policy", ["raise", "warn"])
def test_refresh_env_cache_cases(policy, caplog):
    """Cold start caches; a later unconverged refresh keeps the previous env;
    the log names which case; warn mode emits CTMNotConvergedWarning."""
    cfg = CTMConfig(chi=4, conv_tol=1e-9, on_unconverged=policy)
    cache: dict = {}
    first, second, third, good = object(), object(), object(), object()

    def refresh(envs, converged):
        caplog.clear()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with caplog.at_level("WARNING"):
                _opt._refresh_env_cache(cache, envs, _fake_info(converged), cfg)
        n_warn = sum(issubclass(x.category, CTMNotConvergedWarning) for x in w)
        assert n_warn == (1 if policy == "warn" and not converged else 0)
        return caplog.text

    assert "caching it anyway" in refresh(first, False)  # cold start
    assert cache["envs"] is first
    assert "also unconverged" in refresh(second, False)
    assert cache["envs"] is first
    assert refresh(good, True) == ""
    assert cache["envs"] is good
    assert "previous (converged) env" in refresh(third, False)
    assert cache["envs"] is good
    # a probe-style write leaves the verdict stale -> reported as unknown
    cache["envs"] = second
    assert "unknown" in refresh(third, False)
    assert cache["envs"] is second


def test_refresh_warn_call_is_pinned_in_optimizer(monkeypatch):
    """Under warn, a real optimizer run surfaces CTMNotConvergedWarning from
    the env-cache site (guards deleting the check_ctm_converged call)."""
    cfg = _cfg("2site", "warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), cfg)
    assert any(
        issubclass(x.category, CTMNotConvergedWarning) and "env_cache" in str(x.message)
        for x in w
    )


# ---------------------------------------------------------------------------
# Task 4: site 3 (line-search phi) rejects unconverged trial points.
# ---------------------------------------------------------------------------


def _probe_loss(monkeypatch, unit_cell, policy, *, probe_max_iter=None):
    """Run a tiny optimization whose line-search forwards report unconverged
    and return what the line search's phi returned (plus the caught warnings)."""
    import tenax.algorithms._line_search as _ls

    real_cpl = _cpl.python_loop_ctm_converge
    in_phi = {"on": False}

    phi_calls = []  # (env_init seen, envs returned) per forward inside phi

    def unconverged_in_phi(*a, **k):
        envs, info = real_cpl(*a, **k)
        if in_phi["on"]:
            phi_calls.append((k.get("env_init"), envs))
            return envs, info._replace(converged=False)
        return envs, info

    phis = []

    def spy_hz(phi, dphi, phi0, slope, **kw):
        in_phi["on"] = True
        try:
            # Two probes: the second one's env_init reveals whether the first
            # wrote its (unconverged) env to the cache.
            phis.append(phi(1e-3))
            phis.append(phi(1e-3))
        finally:
            in_phi["on"] = False
        return 0.0, phi0, False

    monkeypatch.setattr(_cpl, "python_loop_ctm_converge", unconverged_in_phi)
    monkeypatch.setattr(_ls, "hager_zhang_line_search", spy_hz)
    ctm = CTMConfig(
        chi=4,
        max_iter=300,
        min_iter=1,
        conv_tol=1e-8,
        on_unconverged=policy,
        probe_max_iter=probe_max_iter,
    )
    cfg = iPEPSConfig(
        unit_cell=unit_cell,
        max_bond_dim=2,
        ctm=ctm,
        gs_num_steps=1,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
        except CTMNotConvergedError:
            pass  # the real gradient forward may not reach conv_tol
    return phis, w, phi_calls


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site3_unconverged_probe_is_inf(monkeypatch, unit_cell):
    phis, _, calls = _probe_loss(monkeypatch, unit_cell, "raise")
    assert phis and all(math.isinf(p) for p in phis)
    # An unconverged probe must not poison the env cache: no later forward is
    # seeded with an env a rejected probe returned.
    assert len(calls) >= 2
    rejected = [envs for _, envs in calls]
    assert all(seed is not r for seed, _ in calls[1:] for r in rejected)


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site3_warn_returns_energy(monkeypatch, unit_cell):
    phis, w, _ = _probe_loss(monkeypatch, unit_cell, "warn")
    assert phis and all(math.isfinite(p) for p in phis)
    assert any(
        issubclass(x.category, CTMNotConvergedWarning)
        and "line_search" in str(x.message)
        for x in w
    )


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site3_probe_override_keeps_legacy(monkeypatch, unit_cell):
    """Review Focus 1: probe_max_iter set -> truncated probes are expected."""
    phis, w, _ = _probe_loss(monkeypatch, unit_cell, "raise", probe_max_iter=7)
    assert phis and all(math.isfinite(p) for p in phis)
    assert not any(
        issubclass(x.category, CTMNotConvergedWarning)
        and "line_search" in str(x.message)
        for x in w
    )


# ---------------------------------------------------------------------------
# Task 5: site 4 (fresh final evaluation) falls back to the converged warm env.
# ---------------------------------------------------------------------------


def _final_eval_unconverged(monkeypatch):
    """Make every COLD (env_init=None) forward that comes AFTER at least one
    warm forward report unconverged: the loop's warm calls precede the cold
    final evaluations (#899 removed env_init from the final evaluation only)."""
    real = _cpl.python_loop_ctm_converge
    state = {"warm_seen": False, "armed_seen": False}

    def spy(*a, **k):
        envs, info = real(*a, **k)
        if k.get("env_init") is not None:
            state["warm_seen"] = True
        elif state["warm_seen"]:
            state["armed_seen"] = True
            return envs, info._replace(converged=False)
        return envs, info

    monkeypatch.setattr(_cpl, "python_loop_ctm_converge", spy)
    return state


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site4_falls_back_to_warm(monkeypatch, unit_cell):
    state = _final_eval_unconverged(monkeypatch)
    warm = {}
    real_use = _opt._final_eval_use_warm

    def spy_use(info, cache, cfg, **kw):
        out = real_use(info, cache, cfg, **kw)
        if out == "warm":
            warm["envs"] = cache["envs"]
        return out

    monkeypatch.setattr(_opt, "_final_eval_use_warm", spy_use)
    cfg = _cfg(unit_cell, "raise", max_iter=300, steps=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    assert state["armed_seen"], "final cold evaluation was never reached"
    assert out[-1]["final_env_source"] == "warm_fallback"
    assert math.isfinite(out[2])
    # the returned env IS the warm env (a tag-only mutant must fail this)
    assert "envs" in warm
    ret = out[1] if unit_cell == "2site" else (out[1],)
    assert all(ret[i] is warm["envs"][(i, 0)] for i in range(len(ret)))


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site4_raises_without_converged_warm(monkeypatch, unit_cell):
    state = _final_eval_unconverged(monkeypatch)
    monkeypatch.setattr(_opt, "_should_restore_best_env", lambda *a, **k: False)
    cfg = _cfg(unit_cell, "raise", max_iter=300, steps=1)
    with pytest.raises(CTMNotConvergedError) as ei:
        _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    assert state["armed_seen"]
    assert ei.value.site == "final_energy"


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site4_warn_legacy(monkeypatch, unit_cell):
    state = _final_eval_unconverged(monkeypatch)
    monkeypatch.setattr(_opt, "_should_restore_best_env", lambda *a, **k: False)
    cfg = _cfg(unit_cell, "warn", max_iter=300, steps=1)
    with pytest.warns(CTMNotConvergedWarning, match="final_energy"):
        out = _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    assert state["armed_seen"]
    assert out[-1]["final_env_source"] == "fresh"
    assert math.isfinite(out[2])


def test_site4_converged_fresh_is_tagged_fresh():
    cfg = _cfg("2site", "raise", max_iter=300, steps=1)
    out = _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), cfg)
    assert out[-1]["final_env_source"] == "fresh"


def test_site4_warm_requires_certified_converged_verdict(monkeypatch):
    """R9: only a cache whose verdict is identity-matched AND converged may
    supply the fallback; unconverged / missing / stale verdicts must raise."""
    monkeypatch.setattr(_opt, "_should_restore_best_env", lambda *a, **k: True)
    cfg = CTMConfig(chi=4, conv_tol=1e-9, on_unconverged="raise")
    bad = _fake_info(False)
    env, other = {"e": 1}, {"e": 2}

    def use(cache):
        return _opt._final_eval_use_warm(bad, cache, cfg)

    assert use({"envs": env, _opt._VERDICT_KEY: (env, True, None)}) == "warm"
    assert _opt._final_eval_use_warm(_fake_info(True), None, cfg) == "fresh"
    assert _opt._final_eval_use_warm(bad, None, cfg, skippable=True) == "skip"
    for cache in (
        {"envs": env, _opt._VERDICT_KEY: (env, False, None)},  # unconverged
        {"envs": env},  # missing verdict
        {"envs": env, _opt._VERDICT_KEY: (other, True, None)},  # stale verdict
        {"envs": env, _opt._VERDICT_KEY: (env, True)},  # legacy 2-tuple
        {},
        None,
    ):
        assert not _opt._cache_env_known_converged(cache)
        with pytest.raises(CTMNotConvergedError) as ei:
            use(cache)
        assert ei.value.site == "final_energy"


def test_site4_verdict_binds_the_params(monkeypatch):
    """R11 / I1: a refused refresh keeps the previous env AND its verdict, so a
    best snapshot taken right after must not qualify for the NEW params."""
    monkeypatch.setattr(_opt, "_should_restore_best_env", lambda *a, **k: True)
    cfg = CTMConfig(chi=4, conv_tol=1e-9, on_unconverged="raise")
    p0, p1 = object(), object()
    e0, e1 = {"e": 0}, {"e": 1}
    cache: dict = {}
    _opt._refresh_env_cache(cache, e0, _fake_info(True), cfg, p0)
    _opt._refresh_env_cache(cache, e1, _fake_info(False), cfg, p1)  # refused
    best_snapshot = dict(cache)
    assert best_snapshot["envs"] is e0
    assert _opt._cache_env_known_converged(best_snapshot, p0)
    assert not _opt._cache_env_known_converged(best_snapshot, p1)
    assert (
        _opt._final_eval_use_warm(_fake_info(False), best_snapshot, cfg, params=p0)
        == "warm"
    )
    with pytest.raises(CTMNotConvergedError):
        _opt._final_eval_use_warm(_fake_info(False), best_snapshot, cfg, params=p1)


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site4_warn_keeps_legacy_even_with_certified_warm(monkeypatch, unit_cell):
    """R12: under warn the warm fallback is NOT used; legacy cold result,
    tagged fresh, plus CTMNotConvergedWarning(site=final_energy)."""
    state = _final_eval_unconverged(monkeypatch)
    seen, verdicts = [], []
    real_use = _opt._final_eval_use_warm

    def spy_use(info, cache, cfg, **kw):
        if not info.converged:
            seen.append(_opt._cache_env_known_converged(cache, kw.get("params")))
        out = real_use(info, cache, cfg, **kw)
        verdicts.append(out)
        return out

    monkeypatch.setattr(_opt, "_final_eval_use_warm", spy_use)
    cfg = _cfg(unit_cell, "warn", max_iter=300, steps=1)
    with pytest.warns(CTMNotConvergedWarning, match="final_energy"):
        out = _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    assert state["armed_seen"]
    assert out[-1]["final_env_source"] == "fresh"
    assert any(seen), "no certified warm env was present; test is vacuous"
    # whichever evaluation was picked, no evaluation may have used the warm env
    assert "warm" not in verdicts, verdicts


# ---------------------------------------------------------------------------
# Task 6: site 5 -- ctm_tensor_2site(strict=) is opt-in; the default keeps the
# legacy warn-and-return behaviour for direct callers.
# ---------------------------------------------------------------------------


def test_site5_ctm_tensor_2site_strict():
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
    from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor

    # ctm_tensor_2site takes Tensors, not raw arrays (the plan's snippet passed
    # raw arrays, which fail in the double-layer build before CTM runs).
    A = _wrap_as_dense_tensor(_rand(2, 2, 0))
    B = _wrap_as_dense_tensor(_rand(2, 2, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eA, eB = ctm_tensor_2site(
            A, B, chi=4, max_iter=2, conv_tol=1e-14
        )  # default: no raise
        with pytest.raises(CTMNotConvergedError) as ei:
            ctm_tensor_2site(A, B, chi=4, max_iter=2, conv_tol=1e-14, strict=True)
    assert ei.value.site == "ctm_tensor_2site"
    assert ei.value.info.converged is False
    assert math.isfinite(ei.value.info.sv_diff)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctm_tensor_2site(
            A, B, chi=4, max_iter=400, conv_tol=1e-6, strict=True
        )  # converges: no raise
