"""A non-finite environment must never be certified as converged (#974).

``max(0.0, float("nan"))`` is ``0.0``: every comparison against NaN is False,
so Python's ``max`` returns its *first* argument and ``if diff > worst`` never
fires.  Both idioms appear in the CTM convergence reducers, so a NaN leaf
difference vanished from the aggregate and the loop reported
``converged=True, sv_diff=0.0`` over an environment whose every tensor was NaN.

The trap is order-dependent, which is why it survived review: ``max(nan, 0.0)``
*is* ``nan``, so a reducer seeded from the data rather than from ``0.0`` would
have propagated it.

Distinct from #898: the spectral (``"sv"``) guard already refuses this
reproduction by returning ``inf``.  It is the elementwise reducer that accepts
it, and the honeycomb module repeats the defect in both of its methods.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_honeycomb_convergence import check_honeycomb_convergence
from tenax.algorithms._ctm_honeycomb_init import initialize_honeycomb_env
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import (
    SINGLE_SITE_NEIGHBORS,
    _max_env_leaf_diff,
)
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms.ipeps import _wrap_as_dense_tensor


def test_python_max_swallows_nan_in_the_accumulator_order():
    """Pin the language behaviour the defect rests on, so the fix reads as one."""
    assert max(0.0, float("nan")) == 0.0
    assert math.isnan(max(float("nan"), 0.0))
    assert (float("nan") > 0.0) is False


def _product_state_env():
    A = _wrap_as_dense_tensor(jnp.ones((1, 1, 1, 1, 2)))
    return A, initialize_ctm_tensor_env(A, 1)


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_square_leaf_reducer_reports_non_finite(bad_value):
    """Every leaf corrupt: the aggregate must not be a finite small number."""
    _A, env = _product_state_env()
    bad = jax.tree.map(lambda x: jnp.full_like(x, bad_value), env)
    got = _max_env_leaf_diff(env, bad)
    assert not math.isfinite(got), (
        f"corrupt-vs-healthy diff reported {got!r}, a finite residual that "
        "certifies a corrupt environment"
    )


def test_square_leaf_reducer_catches_a_single_corrupt_leaf():
    """One bad leaf among healthy ones is the realistic case."""
    _A, env = _product_state_env()
    leaves, treedef = jax.tree.flatten(env)
    assert len(leaves) > 1, "fixture needs more than one leaf to be meaningful"
    poisoned = [
        jnp.full_like(x, jnp.nan) if i == 0 else x for i, x in enumerate(leaves)
    ]
    bad = jax.tree.unflatten(treedef, poisoned)
    got = _max_env_leaf_diff(env, bad)
    assert not math.isfinite(got), f"single NaN leaf vanished; reported {got!r}"


def test_square_leaf_reducer_is_unchanged_on_finite_input():
    """The fix must not disturb healthy comparisons."""
    _A, env = _product_state_env()
    same = _max_env_leaf_diff(env, env)
    assert same == 0.0 and math.isfinite(same)
    shifted = jax.tree.map(lambda x: x + 0.25, env)
    got = _max_env_leaf_diff(env, shifted)
    assert math.isfinite(got)
    assert got == pytest.approx(0.25, abs=1e-9)


@pytest.mark.parametrize("method", ["elementwise", "sv"])
def test_the_loop_cannot_certify_a_corrupt_environment(method):
    """The loop-level contract, both methods.

    ``"sv"`` already passes; it is here so a future refactor cannot fix
    elementwise by breaking the method that was already correct.
    """
    A, env = _product_state_env()
    bad = jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), env)
    envs, info = python_loop_ctm_converge(
        {(0, 0): A},
        SINGLE_SITE_NEIGHBORS,
        chi=1,
        max_iter=4,
        min_iter=2,
        conv_method=method,
        env_init={(0, 0): bad},
        renormalize=True,
    )
    finite_env = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(envs))
    if not finite_env:
        assert not info.converged, (
            f"conv_method={method!r} certified a non-finite environment with "
            f"sv_diff={info.sv_diff!r} after {info.iterations} iterations"
        )


def _honeycomb_envs():
    from tests.test_ctm_honeycomb_energy import _make_random_honeycomb_site

    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(100))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(200))
    return initialize_honeycomb_env({(0, 0): A, (1, 0): B}, chi_init=4, seed=42)


@pytest.mark.parametrize("method", ["elementwise", "svd"])
def test_honeycomb_cannot_certify_a_corrupt_environment(method):
    """Both honeycomb reducers use ``if diff > worst``, so both swallow NaN."""
    envs = _honeycomb_envs()
    bad = jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), envs)
    assert not check_honeycomb_convergence(envs, bad, method=method, tol=1e-10), (
        f"honeycomb method={method!r} certified a fully NaN environment"
    )


@pytest.mark.parametrize("method", ["elementwise", "svd"])
def test_honeycomb_still_certifies_an_identical_environment(method):
    """The healthy direction, so the fix is not just 'always return False'."""
    envs = _honeycomb_envs()
    assert check_honeycomb_convergence(envs, envs, method=method, tol=1e-8)
