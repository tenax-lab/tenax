"""Metric preconditioning is turned off for fermionic input (#1059 finding 27).

``_metric_precond`` builds its norm matrix with the sign-free contractor and
``.todense()``, so on fermionic tensors it is the hard-core-boson metric.
``optimize_gs_ad`` now drops it for fermions, with a warning.  The branch
targets are spies, so no tensor work runs.
"""

from __future__ import annotations

import warnings

import jax
import pytest

from tenax.algorithms import ipeps_optimize as iom
from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _build_initial_fpeps_tensor,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

_FCFG = FPEPSConfig(D=2, t=1.0, V=1.0)
_H_F = spinless_fermion_gate(_FCFG)


def _f(seed=0):
    return _build_initial_fpeps_tensor(_FCFG, jax.random.PRNGKey(seed))


def _cfg(**overrides):
    kw = dict(
        max_bond_dim=2,
        su_init=False,
        gs_num_steps=1,
        ctm=CTMConfig(chi=4, max_iter=10, conv_tol=1e-4),
    )
    kw.update(overrides)
    return iPEPSConfig(**kw)


@pytest.fixture
def seen(monkeypatch):
    configs = []

    def fake(gate, A_init, config, **kw):
        configs.append(config)

    for name in ("_optimize_gs_ad_tensor", "_optimize_gs_ad_2site"):
        monkeypatch.setattr(iom, name, fake)
    return configs


@pytest.mark.parametrize("optimizer", ["lbfgs", "cg"])
@pytest.mark.parametrize(
    "unit_cell, A_init",
    [("1x1", lambda: _f()), ("2site", lambda: (_f(1), _f(2)))],
    ids=["1x1", "2site"],
)
def test_fermions_drop_the_metric_with_a_warning(seen, unit_cell, A_init, optimizer):
    cfg = _cfg(unit_cell=unit_cell, gs_optimizer=optimizer, gs_metric_precond=True)
    with pytest.warns(UserWarning, match=r"gs_metric_precond.*#1059"):
        iom.optimize_gs_ad(_H_F, A_init(), cfg)
    assert seen[0].gs_metric_precond is False


def test_fpeps_entry_drops_it_too(seen):
    with pytest.warns(UserWarning, match=r"gs_metric_precond.*#1059"):
        iom.optimize_fpeps_ad(_H_F, _f(), _cfg(gs_metric_precond=True))
    assert seen[0].gs_metric_precond is False


def test_bosons_keep_the_metric(seen):
    A = jax.random.normal(jax.random.PRNGKey(0), (2, 2, 2, 2, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        iom.optimize_gs_ad(heisenberg_gate(), A, _cfg(gs_metric_precond=True))
    assert seen[0].gs_metric_precond is True


@pytest.mark.parametrize(
    "overrides",
    [dict(gs_metric_precond=False), dict(gs_optimizer="adam")],
    ids=["already-off", "adam"],
)
def test_no_warning_when_the_metric_would_not_run(seen, overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        iom.optimize_gs_ad(_H_F, _f(), _cfg(**overrides))
    assert len(seen) == 1
