"""``_ctm_eager_jit.call``: jit concrete symmetric inputs, pass everything else through."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import tenax.algorithms._ctm_eager_jit as ej
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


def _sym(seed=0):
    U1 = U1Symmetry()
    q = np.array([-1, 0, 0, 1], np.int32)
    idx = (
        TensorIndex.from_charges(U1, q, FlowDirection.IN, "a"),
        TensorIndex.from_charges(U1, q, FlowDirection.OUT, "b"),
    )
    return SymmetricTensor.random_normal(idx, jax.random.PRNGKey(seed))


def _close(out, ref, factor):
    return jnp.allclose(out.todense(), factor * ref.todense())


def _double(x):
    return x * 2.0


def _scale(x, *, k):
    return x * k


def test_concrete_call_runs_through_one_cached_jit():
    ej._jitted.cache_clear()
    t = _sym()
    assert _close(ej.call(_double, t), t, 2.0)
    jitted = ej._jitted(_double, (), ej._env_flags())
    assert jitted._cache_size() == 1
    ej.call(_double, _sym(1))  # same structure: no new trace
    assert jitted._cache_size() == 1


def test_traced_call_bypasses_the_jit():
    ej._jitted.cache_clear()
    jax.jit(lambda x: ej.call(_double, x))(_sym())
    assert ej._jitted.cache_info().currsize == 0


def test_disabled_calls_the_raw_function(monkeypatch):
    ej._jitted.cache_clear()
    monkeypatch.setattr(ej, "_ENABLED", False)
    t = _sym()
    assert _close(ej.call(_double, t), t, 2.0)
    assert ej._jitted.cache_info().currsize == 0


def test_static_argnames_are_static():
    ej._jitted.cache_clear()
    t = _sym()
    a = ej.call(_scale, t, k=3.0, static_argnames=("k",))
    b = ej.call(_scale, t, k=5.0, static_argnames=("k",))
    assert _close(a, t, 3.0) and _close(b, t, 5.0)
    assert ej._jitted(_scale, ("k",), ej._env_flags())._cache_size() == 2


def test_a_patched_function_gets_its_own_jit():
    """Review Focus 3: the jit is keyed by the function object at call time."""
    ej._jitted.cache_clear()

    def f1(x):
        return x * 2.0

    def f2(x):
        return x * 10.0

    t = _sym()
    assert _close(ej.call(f1, t), t, 2.0)
    assert _close(ej.call(f2, t), t, 10.0)
    assert ej._jitted.cache_info().currsize == 2


# --------------------------------------- per-call contraction env flags (R14 F1)
_STRICT = "TENAX_STRICT_CONTRACT"


def test_armed_strict_mode_runs_every_call_eagerly(monkeypatch):
    """A cached executable must not skip the per-call #834 strict checks.

    ``_strict_contract`` is read per call so an audit can toggle it.  Inside a
    jit it is read only at trace time, so once the jit is warm with the flag
    off, a cached executable would skip the check.  Armed, ``call`` must run
    the Python body on every call.
    """
    from tenax.contraction import contractor

    ej._jitted.cache_clear()
    runs = []

    def body(x):
        runs.append(None)
        return x * (2.0 if contractor._strict_contract() else 1.0)

    t = _sym()
    monkeypatch.setenv(_STRICT, "0")
    assert _close(ej.call(body, t), t, 1.0)  # warm the jit
    ej.call(body, t)
    assert len(runs) == 1, "the jit did not warm"

    monkeypatch.setenv(_STRICT, "1")
    assert _close(ej.call(body, t), t, 2.0)
    assert _close(ej.call(body, t), t, 2.0)
    assert len(runs) == 3, "the strict check ran at trace time only"


def test_a_new_contraction_flag_value_gets_a_new_executable(monkeypatch):
    """A flag read at trace time is baked in; a new value must retrace."""
    import os

    ej._jitted.cache_clear()
    flag = "TENAX_BATCH_BLOCKSPARSE"
    assert flag in ej._CONTRACTION_ENV_FLAGS

    def body(x):
        return x * (2.0 if os.environ.get(flag) == "1" else 1.0)

    t = _sym()
    monkeypatch.setenv(flag, "0")
    assert _close(ej.call(body, t), t, 1.0)
    monkeypatch.setenv(flag, "1")
    assert _close(ej.call(body, t), t, 2.0)
    assert ej._jitted.cache_info().currsize == 2


# ------------------------------------------- dense input is not jitted (R14 F2)
def test_dense_input_calls_the_function_directly(monkeypatch):
    """The dense eager path must not change: no jit without a SymmetricTensor."""
    hits = []
    real = ej._jitted

    def spy(*a):
        hits.append(a)
        return real(*a)

    monkeypatch.setattr(ej, "_jitted", spy)
    runs = []

    def body(*xs):
        runs.append(None)
        return xs[0]

    d = DenseTensor(jnp.ones((4, 4)), _sym().indices)
    for args in ((d,), (jnp.ones(3),), ({"env": d}, jnp.ones(2))):
        ej.call(body, *args)
        ej.call(body, *args)
    assert hits == []
    assert len(runs) == 6

    ej.call(body, {"env": _sym()})  # a symmetric leaf container does jit
    assert len(hits) == 1
