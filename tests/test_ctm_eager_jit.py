"""``_ctm_eager_jit.call``: jit when the inputs are concrete, pass through when traced."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import tenax.algorithms._ctm_eager_jit as ej


def _double(x):
    return x * 2.0


def _scale(x, *, k):
    return x * k


def test_concrete_call_runs_through_one_cached_jit():
    ej._jitted.cache_clear()
    out = ej.call(_double, jnp.ones(3))
    assert jnp.allclose(out, 2.0)
    jitted = ej._jitted(_double, (), ej._env_flags())
    assert jitted._cache_size() == 1
    ej.call(_double, jnp.zeros(3))  # same shape: no new trace
    assert jitted._cache_size() == 1


def test_traced_call_bypasses_the_jit():
    ej._jitted.cache_clear()
    jax.jit(lambda x: ej.call(_double, x))(jnp.ones(3))
    assert ej._jitted.cache_info().currsize == 0


def test_disabled_calls_the_raw_function(monkeypatch):
    ej._jitted.cache_clear()
    monkeypatch.setattr(ej, "_ENABLED", False)
    out = ej.call(_double, jnp.ones(3))
    assert jnp.allclose(out, 2.0)
    assert ej._jitted.cache_info().currsize == 0


def test_static_argnames_are_static():
    ej._jitted.cache_clear()
    a = ej.call(_scale, jnp.ones(2), k=3.0, static_argnames=("k",))
    b = ej.call(_scale, jnp.ones(2), k=5.0, static_argnames=("k",))
    assert jnp.allclose(a, 3.0) and jnp.allclose(b, 5.0)
    assert ej._jitted(_scale, ("k",), ej._env_flags())._cache_size() == 2


def test_a_patched_function_gets_its_own_jit():
    """Review Focus 3: the jit is keyed by the function object at call time."""
    ej._jitted.cache_clear()

    def f1(x):
        return x * 2.0

    def f2(x):
        return x + 10.0

    assert jnp.allclose(ej.call(f1, jnp.ones(2)), 2.0)
    assert jnp.allclose(ej.call(f2, jnp.ones(2)), 11.0)
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

    monkeypatch.setenv(_STRICT, "0")
    assert jnp.allclose(ej.call(body, jnp.ones(3)), 1.0)  # warm the jit
    assert len(runs) == 1

    monkeypatch.setenv(_STRICT, "1")
    assert jnp.allclose(ej.call(body, jnp.ones(3)), 2.0)
    assert jnp.allclose(ej.call(body, jnp.ones(3)), 2.0)
    assert len(runs) == 3, "the strict check ran at trace time only"


def test_a_new_contraction_flag_value_gets_a_new_executable(monkeypatch):
    """A flag read at trace time is baked in; a new value must retrace."""
    import os

    ej._jitted.cache_clear()
    flag = "TENAX_BATCH_BLOCKSPARSE"
    assert flag in ej._CONTRACTION_ENV_FLAGS

    def body(x):
        return x * (2.0 if os.environ.get(flag) == "1" else 1.0)

    monkeypatch.setenv(flag, "0")
    assert jnp.allclose(ej.call(body, jnp.ones(3)), 1.0)
    monkeypatch.setenv(flag, "1")
    assert jnp.allclose(ej.call(body, jnp.ones(3)), 2.0)
    assert ej._jitted.cache_info().currsize == 2
