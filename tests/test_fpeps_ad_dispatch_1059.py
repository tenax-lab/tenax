"""optimize_fpeps_ad goes through optimize_gs_ad (#1059 finding 1).

It used to call the private 1-site loop ``_optimize_gs_ad_tensor`` directly,
so ``unit_cell="2site"`` silently ran a uniform 1-site ansatz (which cannot
hold a CDW), ``ctm_ad_mode`` was ignored, and none of optimize_gs_ad's config
checks ran.  The dispatch tests replace each branch target with a spy, so
they do no tensor work.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest

from tenax.algorithms import ipeps_optimize as iom
from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _build_initial_fpeps_tensor,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.core.lattice import checkerboard
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor

_FCFG = FPEPSConfig(D=2, t=1.0, V=0.0)


def _cfg(**overrides):
    kw = dict(
        max_bond_dim=2,
        su_init=False,
        gs_num_steps=1,
        ctm=CTMConfig(chi=4, max_iter=10, conv_tol=1e-4),
    )
    kw.update(overrides)
    return iPEPSConfig(**kw)


def _A(seed=0):
    return _build_initial_fpeps_tensor(_FCFG, jax.random.PRNGKey(seed))


@pytest.fixture
def spy(monkeypatch):
    """Replace every branch target of optimize_gs_ad; record which one ran."""
    calls = []

    def make(name):
        def fake(gate, A_init, config, **kw):
            calls.append((name, A_init, config, kw))
            return name

        return fake

    for name in (
        "_optimize_gs_ad_tensor",
        "_optimize_gs_ad_2site",
        "_optimize_gs_ad_multisite",
    ):
        monkeypatch.setattr(iom, name, make(name))
    from tenax.algorithms import ipeps_optimize_root_implicit as rim

    monkeypatch.setattr(
        rim, "optimize_gs_ad_root_implicit", make("optimize_gs_ad_root_implicit")
    )
    return calls


def test_1x1_still_runs_the_1site_loop(spy):
    A = _A()
    assert iom.optimize_fpeps_ad(spinless_fermion_gate(_FCFG), A, _cfg()) == (
        "_optimize_gs_ad_tensor"
    )
    assert spy[0][1] is A


def test_2site_runs_the_2site_optimizer(spy):
    AB = (_A(1), _A(2))
    out = iom.optimize_fpeps_ad(
        spinless_fermion_gate(_FCFG), AB, _cfg(unit_cell="2site")
    )
    assert out == "_optimize_gs_ad_2site"
    assert spy[0][1] is AB


def test_envs_init_is_passed_through(spy):
    seed = {(0, 0): object(), (1, 0): object()}
    iom.optimize_fpeps_ad(
        spinless_fermion_gate(_FCFG),
        (_A(1), _A(2)),
        _cfg(unit_cell="2site"),
        envs_init=seed,
    )
    assert spy[0][3]["envs_init"] is seed


def test_lattice_runs_the_multisite_optimizer(spy):
    A = {"A": _A(1), "B": _A(2)}
    out = iom.optimize_fpeps_ad(
        spinless_fermion_gate(_FCFG), A, _cfg(unit_cell=checkerboard())
    )
    assert out == "_optimize_gs_ad_multisite"


def test_ctm_ad_mode_reaches_the_root_implicit_engine(spy):
    cfg = _cfg()
    cfg = replace(cfg, ctm=replace(cfg.ctm, ctm_ad_mode="root_implicit"))
    out = iom.optimize_fpeps_ad(spinless_fermion_gate(_FCFG), _A(), cfg)
    assert out == "optimize_gs_ad_root_implicit"


def test_2site_init_builds_two_distinct_fermionic_tensors(spy):
    iom.optimize_fpeps_ad(
        spinless_fermion_gate(_FCFG), None, _cfg(unit_cell="2site"), _FCFG
    )
    A, B = spy[0][1]
    for t in (A, B):
        assert isinstance(t, SymmetricTensor)
        assert isinstance(t.indices[0].symmetry, FermionParity)
    assert not np.allclose(np.asarray(A.todense()), np.asarray(B.todense()))


def test_1x1_init_builds_one_tensor(spy):
    iom.optimize_fpeps_ad(spinless_fermion_gate(_FCFG), None, _cfg(), _FCFG)
    assert isinstance(spy[0][1], SymmetricTensor)


def test_lattice_init_is_refused(spy):
    with pytest.raises(ValueError, match="Lattice"):
        iom.optimize_fpeps_ad(
            spinless_fermion_gate(_FCFG), None, _cfg(unit_cell=checkerboard()), _FCFG
        )
    assert spy == []


def test_optimize_gs_ad_checks_now_apply(spy):
    gate = spinless_fermion_gate(_FCFG)
    with pytest.raises(ValueError, match="gs_num_steps"):
        iom.optimize_fpeps_ad(gate, _A(), _cfg(gs_num_steps=-1))
    with pytest.raises(ValueError, match="envs_init"):
        iom.optimize_fpeps_ad(gate, _A(), _cfg(), envs_init={(0, 0): object()})
    assert spy == []


@pytest.mark.slow
def test_2site_end_to_end_returns_the_pair():
    """One real 2-site step: the result is the pair, not one uniform tensor."""
    cfg = _cfg(
        unit_cell="2site",
        ctm=CTMConfig(chi=4, max_iter=100, conv_tol=1e-8),
    )
    (A, B), (env_a, env_b), E = iom.optimize_fpeps_ad(
        spinless_fermion_gate(_FCFG), None, cfg, _FCFG
    )
    assert isinstance(A, SymmetricTensor) and isinstance(B, SymmetricTensor)
    assert env_a is not None and env_b is not None
    assert np.isfinite(E)
