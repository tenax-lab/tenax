"""#1083: ``on_unconverged="warn"`` emits one warning per (site, step).

Every unconverged forward warned independently from five call sites with
no cross-call state, so a 3-step run produced ~19 near-identical warnings
(Python's once-per-location filter cannot dedupe: the messages differ).
"""

from __future__ import annotations

import re
import warnings
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms._ctm_convergence_policy import (
    CTMConvergeInfo,
    CTMNotConvergedWarning,
    check_ctm_converged,
    ctm_warning_scope,
    set_ctm_warning_step,
)
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

jax.config.update("jax_enable_x64", True)

_INFO = CTMConvergeInfo(converged=False, iterations=3, sv_diff=0.1)


def _check(site, **kw):
    return check_ctm_converged(_INFO, site=site, policy="warn", **kw)


def test_scope_emits_first_per_site_step_and_counts_the_rest():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with ctm_warning_scope() as scope:
            set_ctm_warning_step(0)
            _check("line_search")
            _check("line_search")
            _check("line_search")
            _check("env_cache")
            set_ctm_warning_step(1)
            _check("line_search")
    msgs = [str(x.message) for x in w]
    assert len(msgs) == 3  # (ls,0) (env,0) (ls,1)
    assert "similar suppressed" not in msgs[0]
    assert "(2 similar suppressed)" in msgs[1]  # next emitted carries the count
    assert scope.suppressed == 2


def test_distinct_keys_at_one_step_are_not_merged():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with ctm_warning_scope():
            _check("final_energy", key="best")
            _check("final_energy", key="last")
    assert len(w) == 2


def test_outside_scope_every_call_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check("env_cache")
        _check("env_cache")
    assert len(w) == 2


def test_scope_is_exception_safe_and_not_shared():
    with pytest.raises(RuntimeError):
        with ctm_warning_scope():
            raise RuntimeError
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check("env_cache")
        _check("env_cache")
    assert len(w) == 2  # state did not leak out of the failed scope
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with ctm_warning_scope():
            _check("x")
            with ctm_warning_scope() as inner:  # nested run: fresh state
                _check("x")
            assert inner.suppressed == 0
    assert len(w) == 2


def _gate():
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    return (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real


def _rand(seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2, 2, 2, 2, 2)) + 1j * rng.standard_normal((2, 2, 2, 2, 2))
    return jnp.asarray(a / np.linalg.norm(a))


def _run_recording(monkeypatch, steps=3, noop_step=False):
    import contextlib

    import tenax.algorithms._ctm_convergence_policy as P

    scopes = []
    real_scope = _opt.ctm_warning_scope

    @contextlib.contextmanager
    def rec_scope():
        with real_scope() as sc:
            scopes.append(sc)
            yield sc

    monkeypatch.setattr(_opt, "ctm_warning_scope", rec_scope)
    emitted = []
    orig = P.check_ctm_converged

    def rec(info, **kw):
        sc = P._WARN_SCOPE.get()
        before = len(sc.seen)
        r = orig(info, **kw)
        if len(sc.seen) > before:  # a new (site, step, key) => one emitted
            emitted.append((kw.get("site"), sc.step, kw.get("key")))
        return r

    monkeypatch.setattr(P, "check_ctm_converged", rec)
    if noop_step:
        monkeypatch.setattr(_opt, "set_ctm_warning_step", lambda s: None)
    cfg = iPEPSConfig(
        unit_cell="1x1",
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=4, max_iter=3, min_iter=1, conv_tol=1e-14, on_unconverged="warn"
        ),
        gs_num_steps=steps,
        su_init=False,
        gs_conv_criterion="grad_norm",
        return_history=True,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = _opt.optimize_gs_ad(_gate(), _rand(0), cfg)
    ws = [x for x in w if issubclass(x.category, CTMNotConvergedWarning)]
    return out, ws, emitted


def _assert_per_step(out, ws, emitted, steps):
    sites = Counter(re.search(r"site=([a-z_]+)", str(x.message)).group(1) for x in ws)
    # conv_tol=1e-14 with max_iter=3: every forward is unconverged, so the
    # counts are exact (a per-run dedupe would give 1 per site, not `steps`).
    for site in ("line_search", "env_cache", "gradient"):
        assert sites[site] == steps, sites
    assert sites["final_energy"] == 2, sites
    assert len(ws) == len(emitted)
    assert len(set(emitted)) == len(emitted)  # no (site, step, key) twice
    assert {st for (site, st, _) in emitted if site == "gradient"} == set(range(steps))
    assert {st for (site, st, _) in emitted if site == "line_search"} == set(
        range(steps)
    )
    assert out[-1]["ctm_warnings_suppressed"] > 0
    assert any("similar suppressed" in str(x.message) for x in ws)


def test_optimize_gs_ad_warn_mode_dedupes_per_step(monkeypatch):
    out, ws, emitted = _run_recording(monkeypatch)
    _assert_per_step(out, ws, emitted, 3)


def test_per_step_check_fails_if_step_is_never_set(monkeypatch):
    """Guard the guard: a no-op ``set_ctm_warning_step`` (per-run dedupe) fails."""
    out, ws, emitted = _run_recording(monkeypatch, noop_step=True)
    with pytest.raises(AssertionError):
        _assert_per_step(out, ws, emitted, 3)
