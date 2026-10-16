"""Layout re-check between optimizer steps: cheap (core) tests."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tenax.algorithms._ctm_python_loop as cpl


def _near_tie_record(gap):
    """Two sectors, s_max = 1.  Kept: 1.0, 0.5 (sector 0) and 0.9 (sector 1).
    Dropped: 0.1 (sector 0) and 0.5 + gap (sector 1).  So the largest dropped
    value exceeds the smallest kept one by exactly ``gap``."""
    return {
        "alloc": [2, 1],
        "floor": [0, 0],
        "s": [np.array([1.0, 0.5, 0.1]), np.array([0.9, 0.5 + gap])],
    }


def test_the_probe_tie_tolerance_is_the_hysteresis_margin():
    rec = [_near_tie_record(1e-9)]
    # Default (1e-10): a dropped value 1e-9 above a kept one is a wrong cut.
    assert not cpl._probe_says_layout_is_eager(rec)
    # Margin 1e-8: the same cut is a tie, so the layout is kept.
    assert cpl._probe_says_layout_is_eager(rec, rtol=1e-8)
    # A real gap above the margin is still rejected.
    assert not cpl._probe_says_layout_is_eager([_near_tie_record(1e-6)], rtol=1e-8)


@pytest.mark.parametrize("bad", [-1e-8, float("nan"), float("inf")])
def test_python_loop_refuses_a_bad_layout_tie_rtol(bad):
    with pytest.raises(ValueError, match="layout_tie_rtol"):
        cpl.python_loop_ctm_converge({}, {}, chi=4, layout_tie_rtol=bad)


from tenax.algorithms.ipeps_config import iPEPSConfig  # noqa: E402


def test_the_layout_recheck_defaults():
    cfg = iPEPSConfig()
    assert cfg.gs_layout_recheck is True
    assert cfg.gs_layout_tie_rtol == 1e-8


@pytest.mark.parametrize("bad", [-1e-8, float("nan"), float("inf")])
def test_the_config_refuses_a_bad_layout_tie_rtol(bad):
    with pytest.raises(ValueError, match="gs_layout_tie_rtol"):
        iPEPSConfig(gs_layout_tie_rtol=bad)


def test_a_dense_seed_runs_no_layout_recheck(monkeypatch):
    """A dense seed has no chi layout (``_chi_layout`` is empty), so the
    step-boundary re-check could never see a change: it must not run.

    Every signal is mocked, so no CTM and no AD runs: the CTM driver returns
    its seed as converged, the first gradient is a fixed scaled copy of the
    sites, and the second gradient call (step 2, after the point where the
    re-check would run) stops the run.  Plain optax Adam, no line search."""
    import jax
    import jax.numpy as jnp

    import tenax.algorithms.ipeps_optimize as ipo
    from tenax.algorithms._ctm_convergence_policy import CTMConvergeInfo
    from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
    from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
    from tenax.algorithms.ipeps_config import CTMConfig

    chi = 4
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    A, B = (_wrap_as_dense_tensor(jax.random.normal(k, (2, 2, 2, 2, 2))) for k in keys)
    envs = {
        (0, 0): initialize_ctm_tensor_env(A, chi),
        (1, 0): initialize_ctm_tensor_env(B, chi),
    }
    assert cpl._chi_layout(envs) == ()  # regime: the seed carries no layout
    sz = np.diag([0.5, -0.5])
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    h = np.kron(sz, sz) + 0.5 * (np.kron(sp, sp.T) + np.kron(sp.T, sp))
    gate = jnp.asarray(h.reshape(2, 2, 2, 2))

    callers = []

    def _ctm(*a, **kw):
        import sys

        callers.append(sys._getframe(1).f_code.co_name)
        return kw.get("env_init") or envs, CTMConvergeInfo(True, 1, 0.0)

    class _Stop(Exception):
        pass

    n_vg = []

    def _vg(loss_fn, params, **kw):
        n_vg.append(kw["step_index"])
        if len(n_vg) > 1:
            raise _Stop
        return jnp.asarray(-0.25), tuple(p * 0.1 for p in params)

    monkeypatch.setattr(cpl, "python_loop_ctm_converge", _ctm)
    monkeypatch.setattr(ipo, "_site1_value_and_grad", _vg)
    cfg = iPEPSConfig(
        max_bond_dim=2,
        unit_cell="2site",
        su_init=False,
        gs_implicit_ad=False,
        gs_optimizer="adam",
        gs_line_search=False,
        gs_num_steps=3,
        gs_verbose=False,
        ctm=CTMConfig(chi=chi, max_iter=5, conv_tol=1e-8),
    )
    assert cfg.gs_layout_recheck  # the default is on: only the dense seed gates it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(_Stop):
            ipo.optimize_gs_ad(gate, (A, B), cfg, envs_init=envs)
    assert n_vg == [1, 2], n_vg  # regime: step 2 started
    assert "_recheck_layout_2s" not in callers, callers
    assert "_update_env_cache_2s" in callers, callers  # the spy saw the loop


# --- I1: ``layout_certified`` is the layout verdict, not the sv verdict ------


def _scripted_loop(monkeypatch, *, outcome, check=True, needs_phase=True, probe=None):
    """Drive ``python_loop_ctm_converge`` with every signal mocked.

    The layout phase returns ``outcome``; the jitted loop always reports
    ``converged=True`` after one sweep; the end-of-call check returns
    ``check``.  ``probe`` (if set) makes the pre-loop probe run and say
    ``probe``.  Returns the call's info and the outcomes the phase handed out.
    """
    from tenax.algorithms._ctm_loop_core import CTMLoopResult

    envs = {(0, 0): "ENV"}
    phases = []

    def _adapt(site_tensors, envs_, **kw):
        phases.append(outcome)
        return envs_, 2, outcome

    def _loop(jit_step, site_tensors, envs_start, **k):
        return CTMLoopResult(
            envs=envs_start,
            converged=True,
            iterations=1,
            sv_diff=0.0,
            max_truncation_error=0.0,
            max_smallest_S=0.0,
            final_chi=k["chi_current"],
            bump_extra_sweeps=0,
            best_iteration=1,
        )

    def _jit_factory(*a, **k):
        def _step(site_tensors, envs_, **kw):
            return envs_, 0.0, 0.0, ["record"]

        return _step

    monkeypatch.setattr(cpl, "_make_jit_ctm_step", _jit_factory)
    monkeypatch.setattr(cpl, "_validate_chi_bump_args", lambda **k: k["chi"])
    monkeypatch.setattr(cpl, "_needs_eager_layout_phase", lambda *a: needs_phase)
    monkeypatch.setattr(cpl, "_chi_layout", lambda e: ("L",))
    monkeypatch.setattr(cpl, "_cold_init_layout", lambda st, chi: ("COLD",))
    monkeypatch.setattr(cpl, "_probe_can_certify", lambda st, r: probe is not None)
    monkeypatch.setattr(cpl, "_probe_says_layout_is_eager", lambda r, rtol: probe)
    monkeypatch.setattr(cpl, "_adapt_symmetric_layout", _adapt)
    monkeypatch.setattr(cpl, "_run_ctm_loop_with_bump", _loop)
    monkeypatch.setattr(cpl, "_end_of_call_layout_check", lambda *a, **k: (check, 1))
    max_iter = 1 if probe is False else 100
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, info = cpl.python_loop_ctm_converge(
            {(0, 0): None},
            {},
            chi=4,
            max_iter=max_iter,
            min_iter=0,
            env_init=envs,
            plateau_patience=None,
        )
    msgs = [str(w.message) for w in caught if "#1098" in str(w.message)]
    return info, phases, msgs


@pytest.mark.parametrize(
    "outcome, check, probe, warned",
    [
        ("cap", True, None, "still changing"),
        ("cycle", True, None, "alternated"),
        ("stable", False, None, "end-of-call"),
        # The rejected probe used up the budget: the phase reports "cap".
        ("stable", True, False, "rejected the layout"),
    ],
)
def test_an_uncertified_layout_is_converged_but_not_certified(
    monkeypatch, outcome, check, probe, warned
):
    info, phases, msgs = _scripted_loop(
        monkeypatch, outcome=outcome, check=check, probe=probe
    )
    assert len(msgs) == 1 and warned in msgs[0], msgs  # regime: that branch
    assert info.converged is True, info
    assert info.layout_certified is False, info


def test_a_certified_layout_and_a_dense_call_are_certified(monkeypatch):
    info, phases, msgs = _scripted_loop(monkeypatch, outcome="stable", check=True)
    assert phases == ["stable"] and msgs == [], (phases, msgs)  # regime
    assert info.converged is True and info.layout_certified is True, info

    info, phases, msgs = _scripted_loop(
        monkeypatch, outcome="cap", check=False, needs_phase=False
    )
    assert phases == [] and msgs == [], (phases, msgs)  # regime: no layout phase
    assert info.layout_certified is True, info


def test_the_default_converge_info_is_certified():
    from tenax.algorithms._ctm_convergence_policy import CTMConvergeInfo

    assert CTMConvergeInfo(True, 1, 0.0).layout_certified is True


def _scripted_seeded_run(monkeypatch, *, recheck_info, num_steps=3, layouts=None):
    """``optimize_gs_ad`` on a dense D=2 pair with every CTM and gradient
    mocked, made to look seeded: ``_chi_layout`` is scripted, so the
    step-boundary re-check runs.

    The re-check CTM returns a fresh env dict (``layouts`` maps the n-th
    re-check to the layout its output reports; default: a new one each
    time) with info ``recheck_info``; every other CTM call returns its
    ``env_init``.  Returns ``(history, calls)`` with ``calls`` the
    ``(caller, env_init)`` of every CTM call."""
    import sys

    import jax
    import jax.numpy as jnp

    import tenax.algorithms.ipeps_optimize as ipo
    from tenax.algorithms._ctm_convergence_policy import CTMConvergeInfo
    from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
    from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
    from tenax.algorithms.ipeps_config import CTMConfig

    chi = 4
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    A, B = (_wrap_as_dense_tensor(jax.random.normal(k, (2, 2, 2, 2, 2))) for k in keys)
    seed = {
        (0, 0): initialize_ctm_tensor_env(A, chi),
        (1, 0): initialize_ctm_tensor_env(B, chi),
    }
    sz = np.diag([0.5, -0.5])
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    h = np.kron(sz, sz) + 0.5 * (np.kron(sp, sp.T) + np.kron(sp.T, sp))
    gate = jnp.asarray(h.reshape(2, 2, 2, 2))

    layout_of = {id(seed): ("SEED",)}
    calls = []
    n_rechecks = []
    keep = []

    def _layout(envs):
        return layout_of.get(id(envs), ("SEED",))

    def _ctm(*a, **kw):
        who = sys._getframe(1).f_code.co_name
        env_init = kw.get("env_init")
        calls.append((who, env_init))
        if who == "_recheck_layout_2s":
            n = len(n_rechecks)
            n_rechecks.append(1)
            out = dict(env_init or seed)
            keep.append(out)  # alive, so its id is never reused
            layout_of[id(out)] = layouts[n] if layouts else ("NEW", n)
            return out, recheck_info
        return env_init or seed, CTMConvergeInfo(True, 1, 0.0)

    def _vg(loss_fn, params, **kw):
        return jnp.asarray(-0.25), tuple(p * 0.1 for p in params)

    monkeypatch.setattr(cpl, "python_loop_ctm_converge", _ctm)
    monkeypatch.setattr(cpl, "_chi_layout", _layout)
    monkeypatch.setattr(ipo, "_site1_value_and_grad", _vg)
    cfg = iPEPSConfig(
        max_bond_dim=2,
        unit_cell="2site",
        su_init=False,
        gs_implicit_ad=False,
        gs_optimizer="adam",
        gs_line_search=False,
        gs_num_steps=num_steps,
        gs_verbose=False,
        return_history=True,
        ctm=CTMConfig(chi=chi, max_iter=5, conv_tol=1e-8),
    )
    *_, hist = ipo.optimize_gs_ad(gate, (A, B), cfg, envs_init=seed)
    assert len(n_rechecks) == num_steps - 1, calls  # regime: steps 2..N
    return hist, calls, layout_of


def test_a_converged_but_uncertified_recheck_is_not_a_change(monkeypatch):
    """I1: ``converged=True`` with ``layout_certified=False`` and a changed
    layout records no change, and the uncertified env never becomes the
    cache a frozen forward is seeded from."""
    from tenax.algorithms._ctm_convergence_policy import CTMConvergeInfo

    info = CTMConvergeInfo(True, 1, 0.0, layout_certified=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist, calls, layout_of = _scripted_seeded_run(monkeypatch, recheck_info=info)
    assert hist["layout_changes"] == [], hist["layout_changes"]
    later = [e for who, e in calls if who != "_recheck_layout_2s" and e is not None]
    assert later, calls  # regime: forwards ran after the re-checks
    assert all(layout_of.get(id(e), ("SEED",)) == ("SEED",) for e in later), calls


def test_a_certified_recheck_with_a_new_layout_is_a_change(monkeypatch):
    """Control for the test above: the same script, certified, is a change."""
    from tenax.algorithms._ctm_convergence_policy import CTMConvergeInfo

    info = CTMConvergeInfo(True, 1, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist, calls, _ = _scripted_seeded_run(monkeypatch, recheck_info=info)
    assert hist["layout_changes"] == [2, 3], hist["layout_changes"]


# --- I2: a layout flip-flop is visible -------------------------------------


def test_an_alternating_layout_is_keyed_and_its_recurrence_warns(monkeypatch):
    """Re-checks at steps 2, 3, 4 return B, A, B with A the seed's layout.
    ``layout_changes`` alone cannot tell this from A -> B -> C -> D; the
    keys can, and each return to an earlier layout warns."""
    from tenax.algorithms._ctm_convergence_policy import CTMConvergeInfo

    info = CTMConvergeInfo(True, 1, 0.0)
    with pytest.warns(RuntimeWarning, match="recurred") as rec:
        hist, _, _ = _scripted_seeded_run(
            monkeypatch,
            recheck_info=info,
            num_steps=4,
            layouts=[("B",), ("SEED",), ("B",)],
        )
    assert hist["layout_changes"] == [2, 3, 4], hist["layout_changes"]
    keys = hist["layout_change_keys"]
    seed_key = hist["layout_seed_key"]
    assert isinstance(seed_key, str) and len(seed_key) == 12, seed_key
    assert len(keys) == 3 and keys[0] != seed_key, keys
    assert keys == [keys[0], seed_key, keys[0]], (keys, seed_key)
    recur = [str(w.message) for w in rec if "recurred" in str(w.message)]
    assert len(recur) == 2, recur
    assert "step 3" in recur[0] and "step 4" in recur[1], recur


def test_distinct_layouts_do_not_warn(monkeypatch):
    """Control: B, C (all new) gives distinct keys and no recurrence warning."""
    from tenax.algorithms._ctm_convergence_policy import CTMConvergeInfo

    info = CTMConvergeInfo(True, 1, 0.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hist, _, _ = _scripted_seeded_run(
            monkeypatch, recheck_info=info, layouts=[("B",), ("C",)]
        )
    keys = hist["layout_change_keys"]
    assert len(set(keys + [hist["layout_seed_key"]])) == 3, (keys, hist)
    assert not [w for w in rec if "recurred" in str(w.message)], rec
