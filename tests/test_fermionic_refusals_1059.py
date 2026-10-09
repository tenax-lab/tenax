"""Fermionic input is refused on paths that would drop its signs (#1059).

Grading lives in the tensor type, so a fermionic tensor made dense, or
contracted by a sign-free engine, silently becomes a hard-core boson:
- finding 2: ``gs_c4v=True`` rebuilds the site from a dense C4v basis;
- finding 3: ``ctm_ad_mode="root_implicit_symmetric"`` contracts its graded
  double layer with the sign-free contractor;
- a fermionic gate with non-fermionic sites (incl. ``A_init=None``, which
  builds a dense site).

The optimizer branch targets are replaced by spies, so no tensor work runs.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import pytest
from test_ctm_root_implicit_symmetric import _site_tensor as _u1_site

from tenax.algorithms import ipeps_optimize as iom
from tenax.algorithms import ipeps_optimize_root_implicit as rim
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


def _root_symmetric(cfg):
    # The path has no line search or metric; turning them off keeps its
    # config warnings out of the test output.
    return replace(
        cfg,
        gs_line_search=False,
        gs_metric_precond=False,
        ctm=replace(cfg.ctm, ctm_ad_mode="root_implicit_symmetric"),
    )


@pytest.fixture
def spy(monkeypatch):
    calls = []

    def make(name):
        def fake(gate, A_init, config, **kw):
            calls.append(name)
            return name

        return fake

    for name in (
        "_optimize_gs_ad_tensor",
        "_optimize_gs_ad_2site",
        "_optimize_gs_ad_multisite",
        "_optimize_gs_ad_tensor_reference_c4v",
    ):
        monkeypatch.setattr(iom, name, make(name))
    return calls


# --- finding 2: gs_c4v ------------------------------------------------------


@pytest.mark.parametrize(
    "unit_cell, A_init",
    [("1x1", lambda: _f()), ("2site", lambda: (_f(1), _f(2)))],
    ids=["1x1", "2site"],
)
def test_c4v_refuses_fermionic_sites(spy, unit_cell, A_init):
    with pytest.raises(NotImplementedError, match=r"gs_c4v.*#1059"):
        iom.optimize_gs_ad(_H_F, A_init(), _cfg(unit_cell=unit_cell, gs_c4v=True))
    assert spy == []


def test_c4v_still_accepts_a_bosonic_state(spy):
    A = jax.random.normal(jax.random.PRNGKey(0), (2, 2, 2, 2, 2))
    iom.optimize_gs_ad(heisenberg_gate(), A, _cfg(gs_c4v=True))
    assert len(spy) == 1


# --- a fermionic gate needs fermionic sites ---------------------------------


def test_fermionic_gate_with_no_init_is_refused(spy):
    with pytest.raises(
        NotImplementedError, match=r"A_init=None.*#1059|#1059.*A_init=None"
    ):
        iom.optimize_gs_ad(_H_F, None, _cfg())
    assert spy == []


def test_fermionic_gate_with_a_dense_site_is_refused(spy):
    A = jax.random.normal(jax.random.PRNGKey(0), (2, 2, 2, 2, 2))
    with pytest.raises(NotImplementedError, match="fermionic site tensors"):
        iom.optimize_gs_ad(_H_F, A, _cfg())
    assert spy == []


def test_fermionic_gate_with_one_non_fermionic_site_is_refused(spy):
    with pytest.raises(NotImplementedError, match="fermionic site tensors"):
        iom.optimize_gs_ad(_H_F, (_f(1), _u1_site()), _cfg(unit_cell="2site"))
    assert spy == []


@pytest.mark.parametrize(
    "unit_cell, A_init",
    [("1x1", lambda: _f()), ("2site", lambda: (_f(1), _f(2)))],
    ids=["1x1", "2site"],
)
def test_fermionic_gate_and_sites_are_accepted(spy, unit_cell, A_init):
    iom.optimize_gs_ad(_H_F, A_init(), _cfg(unit_cell=unit_cell))
    assert len(spy) == 1


# --- finding 3: root_implicit_symmetric -------------------------------------


def test_root_implicit_symmetric_refuses_a_fermionic_site():
    with pytest.raises(NotImplementedError, match=r"root_implicit_symmetric.*#1059"):
        iom.optimize_gs_ad(_H_F, _f(), _root_symmetric(_cfg()))


def test_the_engine_entry_refuses_a_fermionic_site_on_its_own():
    """Called directly, not through optimize_gs_ad's guard."""
    with pytest.raises(NotImplementedError, match=r"root_implicit_symmetric.*#1059"):
        rim.optimize_gs_ad_root_implicit(_H_F, _f(), _root_symmetric(_cfg()))


def test_the_engine_entry_refuses_a_fermionic_gate_on_a_bosonic_site():
    with pytest.raises(NotImplementedError, match=r"root_implicit_symmetric.*#1059"):
        rim.optimize_gs_ad_root_implicit(_H_F, _u1_site(), _root_symmetric(_cfg()))


def test_root_implicit_symmetric_gets_past_the_guard_on_a_bosonic_site():
    """A U(1) site passes the fermionic check; the sentinel gate then fails
    the next step (``jnp.asarray``), proving the guard did not fire."""
    with pytest.raises(TypeError) as exc:
        rim.optimize_gs_ad_root_implicit(object(), _u1_site(), _root_symmetric(_cfg()))
    assert "#1059" not in str(exc.value)


# --- Codex review of #1097 --------------------------------------------------


def test_the_dense_root_entry_refuses_a_fermionic_gate():
    """``root_implicit`` (dense) called directly with a fermionic gate and a
    dense site would make the gate dense and run the boson model."""
    cfg = _cfg(gs_line_search=False, gs_metric_precond=False)
    cfg = replace(cfg, ctm=replace(cfg.ctm, ctm_ad_mode="root_implicit"))
    A = jax.random.normal(jax.random.PRNGKey(0), (2, 2, 2, 2, 2))
    with pytest.raises(NotImplementedError, match=r"'root_implicit'.*#1059"):
        rim.optimize_gs_ad_root_implicit(_H_F, A, cfg)


def test_fpeps_entry_refuses_c4v(spy):
    """``optimize_fpeps_ad`` does not go through ``optimize_gs_ad`` on this
    branch, so it must run the guard itself."""
    with pytest.raises(NotImplementedError, match=r"gs_c4v.*#1059"):
        iom.optimize_fpeps_ad(_H_F, _f(), _cfg(gs_c4v=True))
    assert spy == []


def test_fpeps_entry_refuses_a_dense_site(spy):
    A = jax.random.normal(jax.random.PRNGKey(0), (2, 2, 2, 2, 2))
    with pytest.raises(NotImplementedError, match="fermionic site tensors"):
        iom.optimize_fpeps_ad(_H_F, A, _cfg())
    assert spy == []


def test_fpeps_entry_accepts_a_fermionic_site(spy):
    iom.optimize_fpeps_ad(_H_F, _f(), _cfg())
    assert spy == ["_optimize_gs_ad_tensor"]
