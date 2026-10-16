"""``gmres_single_matvec``: restarted GMRES with one matvec call site.

The implicit-AD adjoint used JAX's GMRES, whose program calls the matvec at
three places (initial residual, Arnoldi step, restart residual).  XLA inlines
each call, so the expensive CTM-sweep VJP was compiled three times there and
once more standalone for the seed choice and residual check: D=2 chi=8 on
CPU, 2,458 s + 435 s against 611 s for the fused Neumann module.  This solver
applies the matvec at exactly one site; these tests pin its numerics, its
#858 seed contract, and the one-site structure itself.
"""

from __future__ import annotations

import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_energy_ad import _adjoint_converged
from tenax.algorithms._gmres_lax import gmres_pytree_jax, gmres_single_matvec

N = 24


def _system(kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "well":
        A = np.eye(N) + 0.3 * rng.standard_normal((N, N)) / np.sqrt(N)
    elif kind == "ill":
        Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
        A = Q @ np.diag(np.logspace(-6, 0, N)) @ Q.T
    elif kind == "nonnormal":
        A = np.eye(N) + np.triu(2.0 * rng.standard_normal((N, N)), k=1)
    elif kind == "complex":
        A = np.eye(N) + 0.3 * (
            rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        ) / np.sqrt(N)
    else:
        raise ValueError(kind)
    b = rng.standard_normal(N) + (
        1j * rng.standard_normal(N) if kind == "complex" else 0
    )
    return jnp.asarray(A), jnp.asarray(b)


def _mv(A):
    def matvec(v):
        return (A @ v[0],)

    return matvec


def _solve(A, b, candidates=(), *, tol=1e-12, maxiter=10, restart=30):
    x, res, bn = jax.jit(
        lambda bb, cc: gmres_single_matvec(
            _mv(A), bb, list(cc), tol=tol, maxiter=maxiter, restart=restart
        )
    )((b,), tuple(candidates))
    return np.asarray(x[0]), float(res), float(bn)


@pytest.mark.parametrize("kind", ["well", "ill", "nonnormal"])
def test_converges_to_the_tolerance(kind):
    A, b = _system(kind)
    # cond(A) ~ 1e6-4e6 for "ill"/"nonnormal": even LAPACK's direct solve
    # leaves ||b - A x|| ~ 3e-11 there (f64 floor), so 1e-12 is unattainable.
    tol = 1e-12 if kind == "well" else 1e-9
    x, res, bn = _solve(A, b, tol=tol, maxiter=20, restart=N)
    true = np.linalg.norm(np.asarray(b) - np.asarray(A) @ x)
    assert res == pytest.approx(true, rel=1e-6, abs=1e-14)
    assert true <= max(tol * bn, tol) * (1 + 1e-6)
    assert _adjoint_converged(res, bn, tol)
    if kind != "ill":
        np.testing.assert_allclose(
            x, np.linalg.solve(np.asarray(A), np.asarray(b)), rtol=1e-8, atol=1e-10
        )


def test_complex_system():
    A, b = _system("complex")
    x, res, bn = _solve(A, b, tol=1e-12, maxiter=20, restart=N)
    np.testing.assert_allclose(
        x, np.linalg.solve(np.asarray(A), np.asarray(b)), rtol=1e-8, atol=1e-10
    )
    assert _adjoint_converged(res, bn, 1e-12)


def test_small_budget_reports_an_honest_residual():
    A, b = _system("nonnormal")
    tol = 1e-14
    x, res, bn = _solve(A, b, tol=tol, maxiter=1, restart=2)
    true = np.linalg.norm(np.asarray(b) - np.asarray(A) @ x)
    assert res == pytest.approx(true, rel=1e-10)
    assert res > max(tol * bn, tol)
    assert not _adjoint_converged(res, bn, tol)


def test_restart_one_still_converges():
    A, b = _system("well")
    x, res, bn = _solve(A, b, tol=1e-10, maxiter=500, restart=1)
    assert _adjoint_converged(res, bn, 1e-10)


def test_a_seed_worse_than_zero_is_never_chosen():
    A, b = _system("well")
    bad = -10.0 * b  # residual ||b + 10 A b|| >> ||b||
    assert np.linalg.norm(np.asarray(b - A @ bad)) > np.linalg.norm(np.asarray(b))
    x, res, bn = _solve(A, b, [bad], maxiter=0)
    np.testing.assert_array_equal(x, np.zeros(N))
    assert res == pytest.approx(bn, rel=1e-12)


def test_a_good_warm_start_is_chosen_and_used():
    A, b = _system("nonnormal")
    exact = jnp.asarray(np.linalg.solve(np.asarray(A), np.asarray(b)))
    good = exact + 1e-6 * jnp.ones(N)
    x0, _, _ = _solve(A, b, [good], maxiter=0)
    np.testing.assert_allclose(x0, np.asarray(good), rtol=0, atol=0)
    _, res_from_good, _ = _solve(A, b, [good], tol=1e-15, maxiter=1, restart=2)
    _, res_from_zero, _ = _solve(A, b, [], tol=1e-15, maxiter=1, restart=2)
    assert res_from_good < res_from_zero


def test_ties_go_to_zero():
    b = jnp.zeros(N).at[0].set(1.0)
    A = jnp.eye(N)
    tie = 2.0 * b  # ||b - 2b|| == ||b||
    x, res, bn = _solve(A, b, [tie], maxiter=0)
    np.testing.assert_array_equal(x, np.zeros(N))


def test_the_better_of_two_candidates_is_chosen():
    """Selection is by measured residual, not by position.  (That an earlier
    candidate beats a later *equal* one cannot be observed from the output:
    equal residuals need equal-quality seeds, and the strict ``<`` in the
    seed phase is what pins it, mirrored from ``_best_adjoint_seed``.)"""
    A, b = _system("well")
    exact = jnp.asarray(np.linalg.solve(np.asarray(A), np.asarray(b)))
    worse = exact + 1e-2 * jnp.ones(N)
    better = exact + 1e-5 * jnp.ones(N)
    for cands in ([worse, better], [better, worse]):
        x, _, _ = _solve(A, b, cands, maxiter=0)
        np.testing.assert_array_equal(x, np.asarray(better))


def test_breakdown_returns_the_exact_solution():
    A = jnp.diag(jnp.arange(1.0, N + 1))
    b = jnp.zeros(N).at[0].set(1.0).at[1].set(1.0)  # invariant 2-d subspace
    x, res, bn = _solve(A, b, tol=1e-14, maxiter=3, restart=N)
    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(x, np.asarray(b) / np.arange(1.0, N + 1), atol=1e-13)
    assert res <= 1e-13


def test_zero_rhs():
    A, _ = _system("well")
    x, res, bn = _solve(A, jnp.zeros(N), maxiter=5)
    np.testing.assert_array_equal(x, np.zeros(N))
    assert res == 0.0 and bn == 0.0


def test_nan_matvec_terminates_and_fails_closed():
    _, b = _system("well")

    def nan_matvec(v):
        return (v[0] * jnp.nan,)

    x, res, bn = jax.jit(
        lambda bb: gmres_single_matvec(
            nan_matvec, bb, [bb], tol=1e-10, maxiter=50, restart=5
        )
    )((b,))
    assert np.all(np.isfinite(np.asarray(x[0])))
    assert np.isnan(float(res))
    assert not _adjoint_converged(float(res), float(bn), 1e-10)


def test_nan_seed_candidate_does_not_poison_a_healthy_solve():
    A, b = _system("well")
    bad = jnp.full((N,), jnp.nan)
    x, res, bn = _solve(A, b, [bad], tol=1e-10, maxiter=20, restart=N)
    assert np.all(np.isfinite(x))
    assert np.isfinite(res)
    assert _adjoint_converged(res, bn, 1e-10)


def test_multi_leaf_pytree():
    rng = np.random.default_rng(5)
    shapes = [(3, 4), (5,), (2, 2, 2)]
    n = sum(int(np.prod(s)) for s in shapes)
    A = jnp.asarray(np.eye(n) + 0.2 * rng.standard_normal((n, n)) / np.sqrt(n))
    b = tuple(jnp.asarray(rng.standard_normal(s)) for s in shapes)

    def flat(t):
        return jnp.concatenate([x.ravel() for x in t])

    def matvec(v):
        y = A @ flat(v)
        out, off = [], 0
        for s in shapes:
            k = int(np.prod(s))
            out.append(y[off : off + k].reshape(s))
            off += k
        return tuple(out)

    x, res, bn = jax.jit(
        lambda bb: gmres_single_matvec(matvec, bb, [], tol=1e-12, maxiter=10, restart=n)
    )(b)
    assert [leaf.shape for leaf in x] == shapes
    np.testing.assert_allclose(
        flat(x),
        np.linalg.solve(np.asarray(A), np.asarray(flat(b))),
        rtol=1e-8,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# The structural property this solver exists for.
# ---------------------------------------------------------------------------

_A_STRUCT, _B_STRUCT = _system("well", seed=9)


@jax.jit
def _named_matvec(v):
    return (_A_STRUCT @ v[0],)


def _stablehlo_call_sites(fn):
    text = jax.jit(fn).lower((_B_STRUCT,)).as_text()
    return len(re.findall(r"call @_named_matvec\w*\(", text))


def _jaxpr_call_sites(fn):
    closed = jax.make_jaxpr(fn)((_B_STRUCT,))
    count = 0

    def walk(jaxpr):
        nonlocal count
        for eqn in jaxpr.eqns:
            if eqn.params.get("name") == "_named_matvec":
                count += 1
            for p in eqn.params.values():
                for sub in p if isinstance(p, (list, tuple)) else (p,):
                    inner = getattr(sub, "jaxpr", None)
                    if inner is not None and hasattr(inner, "eqns"):
                        walk(inner)
                    elif hasattr(sub, "eqns"):
                        walk(sub)

    walk(closed.jaxpr)
    return count


def _new(b):
    return gmres_single_matvec(_named_matvec, b, [b], tol=1e-10, maxiter=5, restart=4)


def _old(b):
    return gmres_pytree_jax(_named_matvec, b, b, tol=1e-10, maxiter=5, restart=4)


def test_the_counter_sees_jax_gmres_call_sites():
    """Premise: the counters can see more than one site, else a pass below
    would mean nothing."""
    assert _stablehlo_call_sites(_old) >= 2
    assert _jaxpr_call_sites(_old) >= 2


def test_the_solver_applies_the_matvec_at_exactly_one_site():
    assert _jaxpr_call_sites(_new) == 1
    assert _stablehlo_call_sites(_new) == 1


def test_a_singular_system_never_returns_worse_than_the_seed():
    """No exact solution (b has a null-space component): a Krylov cycle can
    blow up on a near-zero pivot.  The returned x must still be the best
    measured iterate, never worse than the seed or zero."""
    A = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))
    b = jnp.asarray([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    seed = jnp.asarray([1.0, 0.5, 1.0 / 3.0, 0.0, 0.0, 0.0])  # exact on range
    seed_res = float(jnp.linalg.norm(b - A @ seed))
    for restart in (3, 6):
        x, res, bn = jax.jit(
            lambda bb, cc, r=restart: gmres_single_matvec(
                _mv(A), bb, list(cc), tol=1e-12, maxiter=10, restart=r
            )
        )((b,), ((seed,),))
        xv = np.asarray(x[0])
        assert np.all(np.isfinite(xv))
        true = np.linalg.norm(np.asarray(b) - np.asarray(A) @ xv)
        assert float(res) == pytest.approx(true, rel=1e-10)
        assert float(res) <= seed_res * (1 + 1e-12)
        assert float(res) <= float(bn)


def test_real_rhs_with_complex_matvec_or_candidate_raises():
    A, b = _system("well")
    with pytest.raises(TypeError):
        gmres_single_matvec(
            _mv((1 + 1j) * A), (b,), [], tol=1e-10, maxiter=3, restart=4
        )
    with pytest.raises(TypeError):
        gmres_single_matvec(
            _mv(A), (b,), [(b.astype(jnp.complex128),)], tol=1e-10, maxiter=3, restart=4
        )
