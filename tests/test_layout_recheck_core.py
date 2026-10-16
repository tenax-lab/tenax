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
