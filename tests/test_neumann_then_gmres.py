"""``neumann_then_gmres``: the Neumann loop and GMRES at one matvec site.

The joint adjoint solve ran the Neumann loop and the joint GMRES solve as two
compiled programs, each holding a copy of the CTM-sweep transpose.  Here the
Neumann loop is the first phase of the GMRES phase machine, so one program
applies ``J^T`` at one call site.  The oracle is the composition the backward
ran before: ``_neumann_loop``, then ``gmres_single_matvec`` from the seeds
``[lam if solved else 0, b]``.  Spec:
docs/superpowers/specs/2026-10-25-single-transpose-adjoint-design.md
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _ctm_energy_ad as cea
from tenax.algorithms._gmres_lax import gmres_single_matvec, neumann_then_gmres

jax.config.update("jax_enable_x64", True)

TOL = 1e-12
RESTART = 6


def _tree(vec):
    """A flat (6,) vector as two leaves of different shapes (Review Focus 5)."""
    return (vec[:4].reshape(2, 2), vec[4:])


def _flat(t):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(t)])


def _contractive(rho, n=6, seed=0, complex_=False):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    if complex_:
        M = M + 1j * rng.standard_normal((n, n))
    M *= rho / np.max(np.abs(np.linalg.eigvals(M)))
    b = rng.standard_normal(n)
    if complex_:
        b = b + 1j * rng.standard_normal(n)
    return M, b


def _jt_aux(M, P):
    M = jnp.asarray(M)
    P = jnp.asarray(P)

    def jt_aux(v):
        f = _flat(v)
        return _tree(M.T @ f), P @ f

    return jt_aux


def _oracle(jt_aux, b, init, use_init, *, tol, maxiter, restart):
    """The two solvers as the backward composed them before this change."""
    diverged, converged, n_iter, lam_n, resid_n, _bn = cea._neumann_loop(
        lambda v: jt_aux(v)[0], b, init, use_init, tol=tol, maxiter=maxiter
    )
    solved = converged & ~diverged
    seed = jax.tree.map(lambda x: jnp.where(solved, x, jnp.zeros_like(x)), lam_n)

    def matvec(v):
        jt, a = jt_aux(v)
        return jax.tree.map(lambda vi, ji: vi - ji, v, jt), a

    lam, abs_resid, b_norm, a = gmres_single_matvec(
        matvec, b, [seed, b], tol=tol, maxiter=maxiter, restart=restart, aux=True
    )
    return diverged, converged, n_iter, lam_n, resid_n, lam, abs_resid, b_norm, a


def _assert_same(got, want, rtol=1e-12, atol=1e-14):
    gl, wl = jax.tree.leaves(got), jax.tree.leaves(want)
    assert len(gl) == len(wl) and gl
    for g, w in zip(gl, wl):
        g, w = np.asarray(g), np.asarray(w)
        assert g.shape == w.shape and g.dtype == w.dtype, (g.dtype, w.dtype)
        if np.issubdtype(g.dtype, np.inexact):
            np.testing.assert_allclose(g, w, rtol=rtol, atol=atol, equal_nan=True)
        else:
            np.testing.assert_array_equal(g, w)


_CASES = {
    "converges": dict(rho=0.5),
    "warm": dict(rho=0.5, warm=True),
    "diverges": dict(rho=1.5),
    "slow": dict(rho=0.97, maxiter=4),
    "zero_iter": dict(rho=0.5, maxiter=0),
    "complex": dict(rho=0.5, complex_=True),
    "zero_rhs": dict(rho=0.5, zero_b=True),
    "nonfinite": dict(rho=0.5, nan=True),
}


def _run(case):
    c = _CASES[case]
    M, b = _contractive(c["rho"], complex_=c.get("complex_", False))
    if c.get("nan"):
        M[1, 2] = np.nan
    if c.get("zero_b"):
        b = np.zeros_like(b)
    P = np.arange(18.0).reshape(3, 6) / 10
    jt_aux = _jt_aux(M, P)
    if c.get("warm"):
        exact = np.linalg.solve(np.eye(6) - M.T, b)
        init = _tree(jnp.asarray(exact + 1e-6))
        use_init = jnp.array(True)
    else:
        init = _tree(jnp.zeros(6, dtype=jnp.asarray(b).dtype))
        use_init = jnp.array(False)
    b_tree = _tree(jnp.asarray(b))
    kw = dict(tol=TOL, maxiter=c.get("maxiter", 200), restart=RESTART)
    got = jax.jit(lambda bb, i, u: neumann_then_gmres(jt_aux, bb, i, u, **kw))(
        b_tree, init, use_init
    )
    want = jax.jit(lambda bb, i, u: _oracle(jt_aux, bb, i, u, **kw))(
        b_tree, init, use_init
    )
    return got, want


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_one_program_matches_the_two_solvers_composed(case):
    got, want = _run(case)
    _assert_same(got, want)
    diverged, converged, n_iter, _ln, resid_n, lam, abs_resid, _bn, _a = got
    # The regime each case is for (fixtures go vacuous otherwise).
    if case in ("converges", "warm", "complex"):
        assert bool(converged) and not bool(diverged) and int(n_iter) > 1
    if case == "warm":
        cold, _ = _run("converges")
        assert int(n_iter) < int(cold[2])
    if case == "diverges":
        assert bool(diverged)
        assert float(abs_resid) <= TOL * 10 * max(1.0, float(got[7]))
    if case == "slow":
        assert not bool(converged) and not bool(diverged) and int(n_iter) == 4
    if case == "zero_iter":
        assert int(n_iter) == 0 and not bool(converged)
        assert np.isinf(float(resid_n))
    if case == "complex":
        assert jnp.iscomplexobj(lam[0])
    if case == "zero_rhs":
        assert bool(converged)
        assert all(np.all(np.asarray(x) == 0) for x in lam)
    if case == "nonfinite":
        assert not bool(converged) and np.isnan(float(abs_resid))


def test_the_neumann_phase_off_is_gmres_from_the_warm_start():
    M, b = _contractive(0.5)
    P = np.arange(18.0).reshape(3, 6) / 10
    jt_aux = _jt_aux(M, P)
    b_tree = _tree(jnp.asarray(b))
    warm = _tree(jnp.asarray(np.linalg.solve(np.eye(6) - M.T, b) + 1e-3))
    kw = dict(tol=TOL, maxiter=50, restart=RESTART)
    got = jax.jit(
        lambda bb, w: neumann_then_gmres(
            jt_aux, bb, w, jnp.array(True), neumann=False, **kw
        )
    )(b_tree, warm)

    def matvec(v):
        jt, a = jt_aux(v)
        return jax.tree.map(lambda vi, ji: vi - ji, v, jt), a

    want = jax.jit(
        lambda bb, w: gmres_single_matvec(matvec, bb, [w, bb], aux=True, **kw)
    )(b_tree, warm)
    _assert_same(got[5:], want, rtol=0, atol=0)
    assert not bool(got[0]) and not bool(got[1]) and int(got[2]) == 0
    assert np.isinf(float(got[4]))


_M_SITE, _B_SITE = _contractive(0.5, seed=9)


@jax.jit
def _named_jt(v):
    f = _flat(v)
    return _tree(jnp.asarray(_M_SITE).T @ f), f[:3]


def _call_sites(fn):
    b = _tree(jnp.asarray(_B_SITE))
    text = jax.jit(fn).lower(b, b, jnp.array(False)).as_text()
    return len(re.findall(r"call @_named_jt\w*\(", text))


def test_the_counter_sees_two_sites_in_the_two_solvers_composed():
    """Premise: the counter can see more than one site."""
    kw = dict(tol=1e-10, maxiter=5, restart=4)
    assert _call_sites(lambda b, i, u: _oracle(_named_jt, b, i, u, **kw)) == 2


@pytest.mark.parametrize("neumann", [True, False])
def test_the_operator_is_applied_at_exactly_one_site(neumann):
    kw = dict(tol=1e-10, maxiter=5, restart=4, neumann=neumann)
    assert (
        _call_sites(lambda b, i, u: neumann_then_gmres(_named_jt, b, i, u, **kw)) == 1
    )
