"""#1049: the implicit-AD compile cache is keyed by value, and bounded.

``_VJP_CACHE`` keyed the compiled backward on ``id(gate)``, ``id(energy_fn)``
and ``id(neighbors)``.  ``optimize_gs_ad`` builds its energy callback as a
fresh closure per call, so every call missed, re-traced and re-compiled the
whole backward (~82-92 s at D=2 chi=8; a D=3 compile is hours on CPU), and
left the stale entry behind for good.

The key now uses value fingerprints (``_cache_fingerprint``).  These tests pin
both directions: equal configurations share one compiled backward, and a
gate with different values does NOT reuse a backward that baked in the old
one -- a wrong hit would return a gradient for the wrong Hamiltonian.
"""

from __future__ import annotations

import collections
import functools
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _ctm_energy_ad as cea
from tenax.algorithms._cache_fingerprint import (
    _MAX_HASHED_ELEMENTS,
    cache_key_part,
    declare_cache_key,
    lru_get,
    lru_put,
)
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate

# --------------------------------------------------------------------------
# fingerprint: unit tests (milliseconds)
# --------------------------------------------------------------------------


def _key(obj):
    return cache_key_part(obj)[0]


def _exact(obj):
    return _key(obj)[0] == "val"


def test_equal_arrays_on_distinct_objects_share_a_key():
    a = jnp.arange(6.0).reshape(2, 3)
    b = jnp.arange(6.0).reshape(2, 3)
    assert a is not b
    assert _key(a) == _key(b)
    assert _key(np.asarray(a)) == _key(a)


def test_different_values_shapes_or_dtypes_get_different_keys():
    a = jnp.arange(6.0).reshape(2, 3)
    assert _key(a) != _key(a + 1e-12)
    assert _key(a) != _key(a.reshape(3, 2))
    assert _key(a) != _key(a.astype(jnp.float32))


def test_equal_tensors_share_a_key_and_scaled_ones_do_not():
    g1, g2 = heisenberg_gate(), heisenberg_gate()
    assert g1 is not g2
    assert _key(g1) == _key(g2)
    assert _key(g1) != _key(g1 * 2.0)


def _make_closure(d):
    def energy(site_tensors, envs, gate_):
        return d

    return energy


def _make_declared(d):
    return declare_cache_key(_make_closure(d), d)


def test_declared_closures_compare_by_their_declared_data():
    f1, f2, f3 = _make_declared(2), _make_declared(2), _make_declared(3)
    assert f1 is not f2
    assert _key(f1) == _key(f2), "a per-call tenax closure must not miss the cache"
    assert _key(f1) != _key(f3), "different declared data is a different program"


def test_undeclared_callbacks_keep_the_identity_key():
    """A user callback can read state no fingerprint sees (module globals,
    class attributes, registries -- Codex review of #1090), so it keeps the
    pre-#1049 identity key: two equal-looking closures still miss."""
    f1, f2 = _make_closure(2), _make_closure(2)
    assert not _exact(f1)
    assert _key(f1) != _key(f2)


def test_declared_closures_with_different_code_differ():
    def a(x):
        return x

    def b(x):
        return x

    assert _key(declare_cache_key(a)) != _key(declare_cache_key(b))


def test_opaque_objects_fall_back_to_the_whole_components_id():
    """Anything not keyable by value keys the WHOLE component by id -- the
    pre-#1049 key -- and keeps it alive, so it can only miss."""
    o = object()
    holder = {"x": o}
    key, keep = cache_key_part(holder)
    assert key == ("id", id(holder))
    assert any(k is holder for k in keep)
    assert key != _key({"x": object()})


def test_dict_insertion_order_is_part_of_the_key():
    """Codex review of #1090: ``.items()`` order is observable in a trace."""
    assert _key({"a": 1, "b": 2}) != _key({"b": 2, "a": 1})
    assert _key({"a": 1, "b": 2}) == _key({"a": 1, "b": 2})


class _Model:
    coupling = 1.0

    def energy(self, site_tensors, envs, gate_):
        return type(self).coupling


def test_user_objects_and_bound_methods_keep_the_identity_key():
    """Codex review of #1090: a method can read class-level state that two
    receivers with equal ``__dict__`` share, so a user object or its bound
    method is never keyed by value -- a fresh one misses."""
    m = _Model()
    assert not _exact(m)
    assert not _exact(m.energy)
    assert _key(m.energy) != _key(_Model().energy)


def test_large_arrays_keep_the_identity_key():
    """A big array is not hashed per call, so its component keeps its id."""
    big = np.zeros(_MAX_HASHED_ELEMENTS + 1)
    assert not _exact(big)
    assert not _exact(_make_declared(big))


def test_weak_type_is_part_of_the_array_key():
    """Codex review of #1090: weak and strong scalars promote differently."""
    assert _key(jnp.array(1.0)) != _key(jnp.array(1.0, dtype=jnp.float64))


def test_fuse_info_distinguishes_tensor_indices():
    """Codex review of #1090: ``TensorIndex.__eq__`` ignores ``fuse_info``,
    which ``split_index`` branches on; the fingerprint must not."""
    import dataclasses

    from tenax.core.index import FuseInfo, TensorIndex

    g = heisenberg_gate()
    idx = g.indices[0]
    assert isinstance(idx, TensorIndex)
    fused = dataclasses.replace(idx, fuse_info=FuseInfo(parent_indices=(idx, idx)))
    assert fused == idx, "premise: __eq__ ignores fuse_info"
    assert _key(fused) != _key(idx)


def test_a_self_referencing_closure_terminates():
    def outer():
        def rec(n):
            return rec(n - 1) if n else 0

        return rec

    key, _ = cache_key_part(outer())
    hash(key)


# --------------------------------------------------------------------------
# LRU bound
# --------------------------------------------------------------------------


def test_lru_evicts_the_oldest_config_and_never_a_sentinel():
    cache = collections.OrderedDict()
    cache["sentinel"] = "planted"
    for i in range(5):
        lru_put(cache, ("k", i), i, maxsize=3)
    assert "sentinel" in cache
    assert [k for k in cache if isinstance(k, tuple)] == [("k", 2), ("k", 3), ("k", 4)]
    assert lru_get(cache, ("k", 2)) == 2  # a hit becomes most recent
    lru_put(cache, ("k", 5), 5, maxsize=3)
    assert ("k", 3) not in cache and ("k", 2) in cache


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

_BACKWARD_FNS = {
    "_jit_fused_fixed_point_bwd",
    "_jit_dE_denv",
    "_jit_chain_rule",
    "_jit_apply_Jt",
    "_jit_eager_gmres_solve",
}
_TRACED: list[str] = []
jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _TRACED.append(kw.get("fun_name", "?"))
        if event == "/jax/core/compile/jaxpr_trace_duration"
        else None
    )
)


@pytest.fixture
def _fresh_cache():
    saved = collections.OrderedDict(cea._VJP_CACHE)
    cea._VJP_CACHE.clear()
    yield
    cea._VJP_CACHE.clear()
    cea._VJP_CACHE.update(saved)


def test_second_optimize_call_reuses_the_compiled_backward(_fresh_cache):
    from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)))
    B = jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)))
    conf = iPEPSConfig(
        max_bond_dim=2,
        unit_cell="2site",
        su_init=False,
        gs_implicit_ad=True,
        gs_num_steps=1,
        ctm=CTMConfig(chi=4, max_iter=40, conv_tol=1e-8, on_unconverged="warn"),
    )
    H = heisenberg_gate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimize_gs_ad(H, (A, B), conf)
        n_entries = len(cea._VJP_CACHE)
        mark = len(_TRACED)
        # A NEW but equal gate object, as a restart or a scan would pass.
        optimize_gs_ad(heisenberg_gate(), (A, B), conf)
    retraced = sorted(set(_TRACED[mark:]) & _BACKWARD_FNS)
    assert n_entries >= 1
    assert len(cea._VJP_CACHE) == n_entries, (
        "the second optimize_gs_ad call added a _VJP_CACHE entry, so it missed "
        "the cache for an identical configuration (#1049)"
    )
    assert not retraced, f"the second call re-traced the backward: {retraced}"


def _energy(A_arr, gate):
    return jnp.real(
        cea.ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(A_arr)},
            SINGLE_SITE_NEIGHBORS,
            gate,
            recipe="2x2",
            chi=4,
            max_iter=60,
            min_iter=8,
            conv_tol=1e-10,
        )
    )


def test_a_different_gate_does_not_reuse_the_old_backward(_fresh_cache):
    """The energy and its gradient are linear in the gate: doubling the gate
    must double both.  A wrong cache hit would run a backward that baked in
    the OLD gate and return the old gradient."""
    rng = np.random.default_rng(0)
    A0 = jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)))
    A0 = A0 / jnp.linalg.norm(A0)
    g1 = heisenberg_gate()
    g2 = g1 * 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e1, d1 = jax.value_and_grad(_energy)(A0, g1)
        e2, d2 = jax.value_and_grad(_energy)(A0, g2)
    assert len(cea._VJP_CACHE) == 2, "a different gate must get its own entry"
    np.testing.assert_allclose(float(e2), 2.0 * float(e1), rtol=1e-8)
    # rtol 1e-4, not tighter: each gradient carries the adjoint solve's own
    # tolerance (gmres_tol=1e-6; measured 1.8e-6 relative here).  A stale
    # backward would be off by a factor of 2, i.e. 50%.
    np.testing.assert_allclose(
        np.asarray(d2), 2.0 * np.asarray(d1), rtol=1e-4, atol=1e-10
    )


def test_run_start_rearms_warning_latches_and_mid_run_does_not():
    """Codex review of #1090: a shared entry's once-per-run warning latches
    are re-armed when a run starts, not on a mid-run reset."""
    calls = {"seed": 0, "latch": 0}
    key = "_test_1049_latch_sentinel"

    def _seed():
        calls["seed"] += 1

    def _latch():
        calls["latch"] += 1

    cea._VJP_CACHE[key] = (
        None,
        {"_invalidate_warm_start": _seed, "_reset_run_latches": _latch},
    )
    try:
        cea.invalidate_implicit_ad_warm_start()
        assert calls == {"seed": 1, "latch": 0}
        cea.invalidate_implicit_ad_warm_start(run_start=True)
        assert calls == {"seed": 2, "latch": 1}
    finally:
        cea._VJP_CACHE.pop(key, None)


def test_a_declaration_must_list_every_captured_value():
    """The declaration is a contract; ``declare_cache_key`` enforces it."""
    a, b = 1, 2

    def f(x):
        return x + a + b

    with pytest.raises(ValueError, match="not among the declared values"):
        declare_cache_key(f, a)
    declare_cache_key(f, a, b)  # complete: accepted

    # A tenax module-level function imported locally, as the optimizers do,
    # is a captured cell too; it is fixed code, so it needs no declaration.
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor

    def g(x):
        return compute_energy_ctm_tensor(x, x, x)

    declare_cache_key(g)
