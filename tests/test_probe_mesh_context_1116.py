"""#1116 item 2: the #1111 slope-check probe recompiled the forward inside the
backward.  A custom_vjp backward runs under its recorded equation context, which
sets JAX's abstract-mesh key to ``AbstractMesh(())`` where top-level calls see
``None``; every jit and eager primitive the forward had already compiled missed
the cache.  Measured on the limit-cycle fixture below (CPU, forward pre-warmed):
115 compiles / 3.7 s inside the probe, ``_step``, ``_jit_forward_energy`` and
``_bond_phase_fix_envs`` among them.  The probe now runs under the top-level
key."""

from __future__ import annotations

import contextlib
import types
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import numpy as np
import pytest
from test_flow_slope_check_1111 import _single_energy, _single_params

import tenax.algorithms._ctm_energy_ad as cea
from tenax.algorithms._ctm_energy_ad import get_last_implicit_ad_diagnostics

_COMPILED: list[str] = []


def _bare(name: str) -> str:
    """``jit(f)`` -> ``f``: XLA reports jitted functions wrapped in ``jit(...)``."""
    return name[4:-1] if name.startswith("jit(") and name.endswith(")") else name


jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _COMPILED.append(_bare(kw.get("fun_name", "?")))
        if event == "/jax/core/compile/backend_compile_duration"
        else None
    )
)

_FORWARD_JITS = ("_step", "_jit_forward_energy", "_bond_phase_fix_envs")
_EMPTY = jax.sharding.AbstractMesh((), ())


def _mesh_key():
    from jax._src import config

    return config.abstract_mesh_context_manager.value


def _grad_and_diag(params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = jax.grad(lambda p: _single_energy(p))(params)
    return g, dict(get_last_implicit_ad_diagnostics())


def test_probe_does_not_recompile_the_warm_forward(monkeypatch):
    # Clear the caches so an earlier test that already ran this probe (and so
    # compiled it under the backward's key) cannot hide the regression.
    jax.clear_caches()
    probe_compiles: list[list[str]] = []
    orig = cea._slope_probe_energy

    def spy(*a, **k):
        start = len(_COMPILED)
        out = orig(*a, **k)
        probe_compiles.append(_COMPILED[start:])
        return out

    monkeypatch.setattr(cea, "_slope_probe_energy", spy)
    p = _single_params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _single_energy(p)  # warm the forward at top level
    _, d = _grad_and_diag(p)
    assert d["projector_backward_used"] == "flow", (
        d.get("flow_fallback_reason"),
        d.get("flow_slope_ratio"),
    )
    assert len(probe_compiles) == 1, "premise: the probe ran exactly once"
    recompiled = [n for n in probe_compiles[0] if n in _FORWARD_JITS]
    assert recompiled == [], (
        f"the probe recompiled the warm forward: {recompiled} "
        f"({len(probe_compiles[0])} compiles in the probe in total)"
    )


def test_helper_swaps_only_the_empty_mesh_and_restores_it():
    assert _mesh_key() is None
    with cea._top_level_mesh_context():  # top level: nothing to change
        assert _mesh_key() is None
    with jax.sharding.use_abstract_mesh(_EMPTY):
        assert _mesh_key() == _EMPTY
        with cea._top_level_mesh_context():
            assert _mesh_key() is None
        assert _mesh_key() == _EMPTY
        with contextlib.suppress(RuntimeError):
            with cea._top_level_mesh_context():
                raise RuntimeError
        assert _mesh_key() == _EMPTY, "restored after an exception"


def _enter_use_abstract_mesh_inside_helper():
    """What ``jax.debug.callback`` does on the calling thread (debugging.py)."""
    with jax.sharding.use_abstract_mesh(_EMPTY):  # the backward's key
        with cea._top_level_mesh_context():
            with jax.sharding.use_abstract_mesh(_EMPTY):
                pass


def test_use_abstract_mesh_inside_the_helper_does_not_crash(monkeypatch):
    # With the true top-level value (``config_ext.unset``) this is clean.
    _enter_use_abstract_mesh_inside_helper()
    # The spike's literal ``None`` crashed here (mesh.py: ``prev.empty``),
    # which is how 2 of 6 probes became inconclusive; pin that it would.
    real = cea._abstract_mesh_manager()
    assert real is not None, "premise: this JAX has the private API"
    monkeypatch.setattr(cea, "_abstract_mesh_manager", lambda: (real[0], None))
    with pytest.raises(AttributeError, match="empty"):
        _enter_use_abstract_mesh_inside_helper()


def test_a_real_mesh_is_left_alone():
    mesh = jax.sharding.AbstractMesh((1,), ("x",))
    with jax.sharding.use_abstract_mesh(mesh):
        with cea._top_level_mesh_context():
            assert _mesh_key() == mesh
        assert _mesh_key() == mesh


def test_missing_private_api_is_a_no_op(monkeypatch):
    p = _single_params()
    g_ref, d_ref = _grad_and_diag(p)
    for absent in (types.SimpleNamespace(), None):
        monkeypatch.setattr(cea, "_jax_src_config", absent)
        with jax.sharding.use_abstract_mesh(_EMPTY):
            with cea._top_level_mesh_context():
                assert _mesh_key() == _EMPTY, "no-op without the private API"
    # A JAX without the ``config_ext.unset`` sentinel: no-op, not an error.
    from jax._src import config as real_config

    monkeypatch.setattr(
        cea,
        "_jax_src_config",
        types.SimpleNamespace(
            abstract_mesh_context_manager=real_config.abstract_mesh_context_manager
        ),
    )
    with jax.sharding.use_abstract_mesh(_EMPTY):
        with cea._top_level_mesh_context():
            assert _mesh_key() == _EMPTY, "no-op without config_ext.unset"
    # A manager lacking the swap methods is a no-op too.
    monkeypatch.setattr(
        cea,
        "_jax_src_config",
        types.SimpleNamespace(abstract_mesh_context_manager=types.SimpleNamespace()),
    )
    with cea._top_level_mesh_context():
        pass
    g, d = _grad_and_diag(p)
    assert d["projector_backward_used"] == d_ref["projector_backward_used"] == "flow"
    assert d["flow_slope_ratio"] == d_ref["flow_slope_ratio"]
    for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(g_ref)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_verdict_and_ratio_unchanged_by_the_fix(monkeypatch):
    p = _single_params()
    g_fix, d_fix = _grad_and_diag(p)
    monkeypatch.setattr(cea, "_top_level_mesh_context", contextlib.nullcontext)
    g_old, d_old = _grad_and_diag(p)
    assert d_fix["projector_backward_used"] == d_old["projector_backward_used"]
    assert d_fix.get("flow_fallback_reason") == d_old.get("flow_fallback_reason")
    assert d_fix["flow_slope_ratio"] == d_old["flow_slope_ratio"]
    assert 0.5 <= d_fix["flow_slope_ratio"] <= 2.0
    for a, b in zip(jax.tree.leaves(g_fix), jax.tree.leaves(g_old)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
