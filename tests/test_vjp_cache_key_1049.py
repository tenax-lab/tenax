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
import dataclasses
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
    _key_numpy,
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
    big = jnp.zeros(_MAX_HASHED_ELEMENTS + 1)
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


# --------------------------------------------------------------------------
# Codex re-review of #1090 (73770f76)
# --------------------------------------------------------------------------


def test_cg_energy_gates_are_keyed_by_their_gate_data():
    """The CG energy callbacks capture ``cg_gates._energy_gates()``, a copy
    without ``map_fn``/``init_fn``, so a fresh callback still hits.  The full
    gates keep the identity key: a user callback may call ``gate.map_fn``
    (Codex review of #1090), so its functions are not dropped globally."""
    from tenax.algorithms.coarse_grain import CGGates

    def mk(map_fn):
        return CGGates(
            h_intra=jnp.eye(4),
            h_inter={"h": jnp.ones((16, 16))},
            n_sites=2,
            map_fn=map_fn,
            init_fn=lambda: None,
        )

    g1, g2 = mk(lambda x: x), mk(lambda x: 2 * x)
    assert not _exact(g1)
    assert _key(g1) != _key(g2)
    e1, e2 = g1._energy_gates(), g2._energy_gates()
    assert e1.map_fn is None and e1.init_fn is None
    assert _exact(e1)
    assert _key(e1) == _key(e2)
    assert _key(e1) != _key(dataclasses.replace(e1, n_sites=3))

    def factory(cg, d):
        def energy(site_tensors, envs, _gate):
            return cg, d

        return declare_cache_key(energy, cg, d)

    assert _exact(factory(e1, 4))
    assert _key(factory(e1, 4)) == _key(factory(e2, 4))


def test_numpy_values_keep_the_identity_key():
    """NumPy state goes beyond the values (flags, strides, ownership,
    writeability), so NumPy arrays and scalars are never keyed by value
    (Codex review of #1090).  Equal NumPy objects therefore miss."""
    c = np.arange(6.0).reshape(2, 3)
    for x in (
        c,
        np.asfortranarray(c),
        c.view(),
        np.float64(1.0),
        np.int64(1),
        np.asarray(2.0),
    ):
        assert not _exact(x)
        assert not _exact(_make_declared(x))
    assert _key(c) != _key(c.copy())
    assert _key(c) != _key(jnp.asarray(c))


def test_numpy_inside_tenax_objects_is_keyed_by_value():
    """A tensor's ``TensorIndex`` charges are NumPy arrays; they are tenax's
    own metadata, so a tenax gate is still keyed by value."""
    g = heisenberg_gate()
    assert any(isinstance(ix.charges, np.ndarray) for ix in g.indices)
    assert _exact(g)


def test_python_scalars_are_keyed_by_exact_type():
    """``np.float64`` subclasses ``float`` but is strongly typed under JAX."""
    assert _exact(1.0) and _exact(1) and _exact(True) and _exact(1j)
    assert _key(1.0) == _key(1.0)
    assert _key(1.0) != _key(np.float64(1.0))
    assert _key(1j) != _key(np.complex128(1j))
    assert _key(1) != _key(True)


def test_zero_dimensional_shape_is_part_of_the_array_key():
    """``np.ascontiguousarray`` promotes shape ``()`` to ``(1,)``; a callback
    may branch on ``gate.ndim``."""
    x = jnp.asarray(2.0, dtype=jnp.float64)
    assert _key(x) != _key(x.reshape(1))
    assert _key(x) == _key(jnp.asarray(2.0, dtype=jnp.float64))


def test_float8_dtypes_get_different_keys():
    """``dtype.str`` is ``"<V1"`` for every float8 variant (Codex review of
    #1090); equal bytes in different float8 types must not share a key."""
    a = jnp.array([0, 1], dtype=jnp.float8_e4m3fnuz)
    b = jax.lax.bitcast_convert_type(a, jnp.float8_e5m2fnuz)
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert np.asarray(a).dtype.str == np.asarray(b).dtype.str
    assert _key(a) != _key(b)
    assert _key(a) == _key(jnp.array([0, 1], dtype=jnp.float8_e4m3fnuz))
    # The same on the NumPy-inside-a-tenax-object path.
    assert _key_numpy(np.asarray(a)) != _key_numpy(np.asarray(b))


def test_placement_is_part_of_the_array_key():
    """A callback may branch on ``.committed`` or ``.sharding`` (Codex review
    of #1090): equal values placed differently must not share a key."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    x = jnp.arange(3.0)
    dev = jax.devices()[0]
    committed = jax.device_put(x, dev)
    assert not x.committed and committed.committed
    assert _key(x) != _key(committed)
    named = jax.device_put(x, NamedSharding(Mesh([dev], ("i",)), PartitionSpec()))
    assert _key(committed) != _key(named)
    # Same value, same placement: still one key.
    assert _key(committed) == _key(jax.device_put(jnp.arange(3.0), dev))
    assert _key(x) == _key(jnp.arange(3.0))


def test_a_different_captured_tenax_function_changes_the_key():
    """Captured module-level tenax functions are part of the declared key."""
    from tenax.algorithms import _ctm_tensor_energy as te

    def factory(fn):
        def energy(site_tensors, envs, gate):
            return fn(site_tensors, envs, gate)

        return declare_cache_key(energy)

    k_a = _key(factory(te.compute_energy_ctm_tensor))
    assert k_a == _key(factory(te.compute_energy_ctm_tensor))
    assert k_a != _key(factory(te.compute_energy_ctm_tensor_2site))


# --------------------------------------------------------------------------
# Codex round 6 on #1090
# --------------------------------------------------------------------------


def test_frozensets_are_keyed_by_value_and_sets_are_not():
    a = frozenset({("A", "right"), ("B", "left")})
    b = frozenset({("B", "left"), ("A", "right")})
    assert _exact(a) and _key(a) == _key(b)
    assert _key(a) != _key(frozenset({("A", "right"), ("C", "left")}))
    assert not _exact({("A", "right")})  # a mutable set
    assert not _exact(frozenset({object()}))


def test_np_dtype_objects_are_keyed_exactly():
    e4 = jnp.zeros(1, jnp.float8_e4m3fnuz).dtype
    e5 = jnp.zeros(1, jnp.float8_e5m2fnuz).dtype
    assert np.dtype(e4).str == np.dtype(e5).str
    assert _key(np.dtype(e4)) != _key(np.dtype(e5))


def test_multisite_pess_energy_callback_is_keyed_by_value(monkeypatch):
    """Its captured bond gates are keyed by frozenset bond IDs; before, the
    whole callback fell back to identity and every build missed."""
    from tenax.algorithms import pess_optimize as po
    from tenax.algorithms._pess_multisite_energy import kagome_3site_bond_gates
    from tenax.algorithms.ipeps_config import CTMConfig

    seen = []
    real = po.declare_cache_key
    monkeypatch.setattr(
        po, "declare_cache_key", lambda fn, *data: seen.append(real(fn, *data))
    )
    cfg = CTMConfig(chi=4)
    po.build_pess_loss_3site_multisite(kagome_3site_bond_gates(), cfg)
    po.build_pess_loss_3site_multisite(kagome_3site_bond_gates(), cfg)
    k1, k2 = (_key(fn) for fn in seen)
    assert k1[0] == "val" and k1 == k2


def test_a_user_callback_keys_its_inputs_by_identity():
    """A user callback may branch on ``gate is X``, so two equal gate objects
    must not share a backward; default and declared callbacks still do."""
    from tenax.algorithms._cache_fingerprint import callback_key_parts

    g1, g2 = heisenberg_gate(), heisenberg_gate()

    def user(site_tensors, envs, gate):
        return 0.0

    _, (k1,), _ = callback_key_parts(user, g1)
    _, (k2,), _ = callback_key_parts(user, g2)
    assert k1 != k2 and k1[0] == "id"

    _, (k1,), _ = callback_key_parts(None, g1)
    _, (k2,), _ = callback_key_parts(None, g2)
    assert k1 == k2 and k1[0] == "val"

    def declared(site_tensors, envs, gate):
        return 0.0

    declare_cache_key(declared)
    _, (k1,), _ = callback_key_parts(declared, g1)
    _, (k2,), _ = callback_key_parts(declared, g2)
    assert k1 == k2 and k1[0] == "val"


@pytest.fixture
def run_starts(monkeypatch):
    """Count run starts, via a sentinel cache entry."""
    from tenax.algorithms import pess_optimize as po

    monkeypatch.setattr(po, "_run_state_owner", None)
    calls = {"seed": 0, "latch": 0}
    key = "_test_1049_pess_owner_sentinel"

    def _seed():
        calls["seed"] += 1

    def _latch():
        calls["latch"] += 1

    cea._VJP_CACHE[key] = (
        None,
        {"_invalidate_warm_start": _seed, "_reset_run_latches": _latch},
    )
    yield calls
    cea._VJP_CACHE.pop(key, None)


def test_a_different_pess_loss_starts_a_fresh_run(run_starts):
    """Codex review of #1090: equal losses share a cache entry, so the run
    state resets whenever the loss using it changes, in any call order."""
    from tenax.algorithms import pess_optimize as po

    a, b = object(), object()
    for owner, starts in [(a, 1), (a, 1), (b, 2), (a, 3), (a, 3)]:
        po._claim_run_state(owner)
        assert run_starts == {"seed": starts, "latch": starts}


class _Claimed(Exception):
    pass


@pytest.mark.parametrize(
    "builder", ["build_pess_loss", "build_pess_loss_exact", "multisite"]
)
def test_each_pess_loss_claims_its_run_before_any_work(monkeypatch, builder):
    from tenax.algorithms import pess_optimize as po
    from tenax.algorithms._pess_multisite_energy import kagome_3site_bond_gates
    from tenax.algorithms.ipeps_config import CTMConfig
    from tenax.algorithms.pess import (
        kagome_xxz_pess_cg_gates,
        kagome_xxz_pess_cg_gates_exact,
    )

    build = {
        "build_pess_loss": lambda c: po.build_pess_loss(kagome_xxz_pess_cg_gates(), c),
        "build_pess_loss_exact": lambda c: po.build_pess_loss_exact(
            kagome_xxz_pess_cg_gates_exact(), c
        ),
        "multisite": lambda c: po.build_pess_loss_3site_multisite(
            kagome_3site_bond_gates(), c
        ),
    }[builder]
    owners = []

    def spy(owner):
        owners.append(owner)
        raise _Claimed

    monkeypatch.setattr(po, "_claim_run_state", spy)
    loss_a, loss_b = build(CTMConfig(chi=4)), build(CTMConfig(chi=4))
    assert owners == []  # building claims nothing
    for loss in (loss_a, loss_b, loss_a):
        with pytest.raises(_Claimed):
            loss(None)
    assert owners[0] is owners[2] and owners[0] is not owners[1]
