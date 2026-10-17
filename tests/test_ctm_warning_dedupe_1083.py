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


def test_optimize_gs_ad_warn_mode_dedupes_per_step():
    steps = 3
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
    sites = Counter(re.search(r"site=([a-z_]+)", str(x.message)).group(1) for x in ws)
    # At most one per step for each per-step site (was ~4 line_search/step).
    for site in ("line_search", "env_cache", "gradient"):
        assert sites[site] <= steps, sites
    assert sites["final_energy"] <= 2, sites
    h = out[-1]
    assert h["ctm_warnings_suppressed"] > 0
    assert any("similar suppressed" in str(x.message) for x in ws)
