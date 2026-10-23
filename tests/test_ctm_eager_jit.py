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
    jitted = ej._jitted(_double, ())
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
    assert ej._jitted(_scale, ("k",))._cache_size() == 2


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
