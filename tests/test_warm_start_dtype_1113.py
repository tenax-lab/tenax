"""#1113: the implicit-AD warm start must not reuse a cached lambda of another dtype.

The run state is keyed on (energy_fn, gate, neighbors) identities and the
seed-compatibility check compared only shapes, so a complex lambda cached by
one backward seeded a real-dtype backward that shared the gate and neighbors
objects, and the backward raised a dtype ValueError instead of starting cold.
"""

from __future__ import annotations

import collections
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _ctm_energy_ad as cea
from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _heisenberg():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = Sp.T
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


#: One gate object for every call: the run state (and its cached lambda) is
#: keyed on its identity, which is what lets a seed cross between dtypes.
_GATE = _heisenberg()


def _site(dtype, seed=3):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    A = jax.random.normal(k1, (2, 2, 2, 2, 2))
    if dtype == "complex":
        A = A + 1j * jax.random.normal(k2, (2, 2, 2, 2, 2))
    return A / jnp.linalg.norm(A)


def _energy(A, adjoint_method):
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(A)},
            SINGLE_SITE_NEIGHBORS,
            _GATE,
            chi=8,
            max_iter=200,
            conv_tol=1e-10,
            adjoint_method=adjoint_method,
        )
    )


@pytest.fixture
def _fresh_vjp_cache():
    saved = collections.OrderedDict(cea._VJP_CACHE)
    cea._VJP_CACHE.clear()
    yield
    cea._VJP_CACHE.clear()
    cea._VJP_CACHE.update(saved)


@pytest.mark.usefixtures("_fresh_vjp_cache")
@pytest.mark.parametrize("adjoint_method", ["fixed_point", "gmres"])
@pytest.mark.parametrize("first, second", [("complex", "real"), ("real", "complex")])
def test_a_dtype_switch_starts_the_adjoint_cold(adjoint_method, first, second):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_first = jax.grad(lambda a: _energy(a, adjoint_method))(_site(first))
        assert np.all(np.isfinite(np.asarray(g_first)))
        # Same gate and neighbors objects: the run (and its cached lambda) is
        # reused, so only the dtype check stands between the two.
        g_second = jax.grad(lambda a: _energy(a, adjoint_method))(_site(second))
    assert jnp.asarray(g_second).dtype == _site(second).dtype
    assert np.all(np.isfinite(np.asarray(g_second)))
    # A cold start gives the same gradient as a fresh cache would.
    cea._VJP_CACHE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_fresh = jax.grad(lambda a: _energy(a, adjoint_method))(_site(second))
    np.testing.assert_allclose(
        np.asarray(g_second), np.asarray(g_fresh), rtol=1e-6, atol=1e-9
    )
