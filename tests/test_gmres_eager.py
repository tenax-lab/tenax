"""Eager-loop GMRES (#731).

The solver exists for one structural reason: the *loop* must not be traced, so
that XLA compiles the operator once instead of compiling a program that
contains the whole operator inside a ``while_loop``.  The root-implicit adjoint
operator is the VJP of the characteristic equations, and having it inside the
loop peaked at 8.6 GB at ``D=2, chi=4`` against ~7 GB CI runners.

Two families of test therefore matter equally:

* **numerical** -- it solves what it claims to solve, with
  ``jax.scipy.sparse.linalg.gmres(solve_method="batched")`` as the reference,
  since that is what it replaces;
* **structural** -- the operator is compiled once and *invoked many times*.
  A regression that put the loop back under ``jit`` would keep every numerical
  test green and silently restore the 8.6 GB.  That is the whole defect.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._gmres_eager import gmres_eager

jax.config.update("jax_enable_x64", True)


def _well_conditioned(n: int = 24, seed: int = 0):
    """A nonsymmetric, well-conditioned real matrix and a right-hand side."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    M = M + n * np.eye(n)  # push the spectrum away from the origin
    b = rng.standard_normal(n)
    return jnp.asarray(M), jnp.asarray(b)


# ---------------------------------------------------------------------------
# Numerical behaviour
# ---------------------------------------------------------------------------


def test_it_solves_a_dense_system_to_the_requested_tolerance():
    M, b = _well_conditioned()
    x, resid = gmres_eager(lambda v: M @ v, b, tol=1e-12, maxiter=20, restart=10)

    exact = jnp.linalg.solve(M, b)
    assert resid <= 1e-12
    # The residual is the contract; the error follows it through the condition
    # number, which is O(1) here by construction.
    assert float(jnp.linalg.norm(x - exact) / jnp.linalg.norm(exact)) < 1e-10


def test_it_agrees_with_the_jitted_gmres_it_replaces():
    """Parity with ``jax.scipy.sparse.linalg.gmres``, same controls."""
    M, b = _well_conditioned(n=32, seed=3)
    op = lambda v: M @ v  # noqa: E731

    x_eager, _ = gmres_eager(op, b, tol=1e-11, atol=0.0, maxiter=40, restart=15)
    x_ref, _info = jax.scipy.sparse.linalg.gmres(
        op,
        b,
        x0=b,
        tol=1e-11,
        atol=0.0,
        restart=15,
        maxiter=40,
        solve_method="batched",
    )

    rel = float(jnp.linalg.norm(x_eager - x_ref) / jnp.linalg.norm(x_ref))
    assert rel < 1e-8, rel


def test_it_converges_on_a_singular_but_consistent_system():
    """The case the root-implicit adjoint actually presents.

    ``∂_y F`` has an exact null space -- an independent phase on each
    environment tensor -- and the right-hand side is orthogonal to it.  A solver
    that only works on nonsingular systems is no use here.
    """
    n = 20
    rng = np.random.default_rng(11)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # Three exactly-zero singular values.
    s = np.concatenate([rng.uniform(0.5, 2.0, n - 3), np.zeros(3)])
    M = Q @ np.diag(s) @ Q.T
    M = jnp.asarray(M)

    # b in range(M) by construction, so the system is consistent.
    b = M @ jnp.asarray(rng.standard_normal(n))

    x, resid = gmres_eager(lambda v: M @ v, b, tol=1e-10, maxiter=40, restart=15)
    assert np.isfinite(resid)
    assert resid <= 1e-10, resid
    assert bool(jnp.all(jnp.isfinite(x)))


def test_the_returned_residual_is_measured_not_the_givens_estimate():
    """A starved solve must report the residual it really achieved.

    GMRES's internal ``|g[k]|`` tracks the least-squares residual of the
    *current* Krylov space; after a restart that stagnates it can sit below the
    true residual.  Callers gate a gradient on this number, so it is computed
    from a real matvec.
    """
    rng = np.random.default_rng(5)
    n = 40
    M = jnp.asarray(rng.standard_normal((n, n)) + 0.4 * np.eye(n))
    b = jnp.asarray(rng.standard_normal(n))

    # One restart of a 2-dimensional Krylov space: nowhere near converged.
    x, resid = gmres_eager(lambda v: M @ v, b, tol=1e-14, maxiter=1, restart=2)

    true_resid = float(jnp.linalg.norm(b - M @ x) / jnp.linalg.norm(b))
    assert resid == pytest.approx(true_resid, rel=1e-12), (resid, true_resid)
    assert resid > 1e-14  # it really did not converge, so this is not vacuous


def test_maxiter_counts_restarts_not_iterations():
    """Semantics parity with the solver this replaces.

    ``jax.scipy.sparse.linalg.gmres`` counts *restarts* in ``maxiter`` while
    ``tenax.algorithms._gmres_lax.gmres_lax`` counts total Arnoldi steps.  Get
    this backwards and every caller's budget changes by a factor of ``restart``.
    """
    M, b = _well_conditioned(n=30, seed=7)
    op = lambda v: M @ v  # noqa: E731

    # 1 restart x 30 Krylov directions spans the whole space, so it converges.
    _x_full, resid_full = gmres_eager(op, b, tol=1e-12, maxiter=1, restart=30)
    # 30 restarts x 1 Krylov direction is 30 steepest-descent-like steps.
    _x_thin, resid_thin = gmres_eager(op, b, tol=1e-12, maxiter=30, restart=1)

    assert resid_full <= 1e-12, resid_full
    # If ``maxiter`` were being read as a total-iteration budget the two would
    # do the same work.  They do not.
    assert resid_thin > resid_full


def test_each_restart_builds_the_whole_krylov_space():
    """Parity with ``_gmres_batched``, and it is load-bearing for accuracy.

    JAX's ``"batched"`` method opens with ``del ptol  # unused`` and loops on
    ``k < restart`` alone, so it never exits a restart early.  On the
    root-implicit adjoint those extra directions take the achieved residual six
    orders past the request (3.5e-15 against ``tol=1e-8``), which is the number
    every caller's gradient was measured against.  An "optimisation" that
    stopped at the target would keep every tolerance test green while quietly
    loosening the adjoint by six orders.
    """
    M, b = _well_conditioned(n=18, seed=17)
    steps = {"n": 0}

    def op(v):
        steps["n"] += 1
        return M @ v

    # A tolerance this loose is met almost immediately; a solver with an early
    # inner exit would stop after a couple of Arnoldi steps.
    _x, resid = gmres_eager(op, b, tol=1e-1, maxiter=1, restart=12)

    # 1 initial residual + 12 Arnoldi steps + 1 closing true-residual measure.
    assert steps["n"] == 14, steps
    # And the point of doing the work: seven orders past what was asked for.
    assert resid < 1e-6, resid


def test_x0_defaults_to_b_not_to_zero():
    """``_solve_root_adjoint`` passes ``x0=b``; that must survive the swap."""
    M, b = _well_conditioned(n=12, seed=2)
    # Zero restarts: the loop never runs, so what comes back IS x0.
    x, resid = gmres_eager(lambda v: M @ v, b, maxiter=0, restart=5)
    assert bool(jnp.allclose(x, b))
    expected = float(jnp.linalg.norm(b - M @ b) / jnp.linalg.norm(b))
    assert resid == pytest.approx(expected, rel=1e-12)


def test_an_already_solved_system_takes_no_step():
    """``x0 = b`` is already the answer when ``A`` is the identity."""
    b = jnp.asarray(np.random.default_rng(1).standard_normal(8))
    x, resid = gmres_eager(lambda v: v, b, tol=1e-12, maxiter=10, restart=5)
    assert resid == pytest.approx(0.0, abs=1e-15)
    assert bool(jnp.allclose(x, b))


def test_a_krylov_space_that_closes_early_does_not_divide_by_noise():
    """Happy breakdown: ``restart`` larger than the space the operator spans."""
    # Rank-2 action on a 6-vector: the Krylov space closes after 2 steps.
    M = jnp.asarray(np.diag([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]))
    b = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x, resid = gmres_eager(lambda v: M @ v, b, tol=1e-13, maxiter=3, restart=6)
    assert np.isfinite(resid)
    assert resid <= 1e-13
    assert bool(jnp.allclose(x, b / 3.0))


@pytest.mark.parametrize("scale", [1.0, 1e-10, 1e-20, 1e40])
def test_the_solve_is_invariant_under_rescaling_the_operator(scale):
    """The breakdown test must measure orthogonality, not magnitude.

    ``A x = b`` and ``(sA) x = (s b)`` have the same solution, so GMRES should
    do the same work on both. It did not: the breakdown threshold was
    ``1e-14 * max(|H[j,j]|, 1.0)``, and that floor of 1.0 makes it an
    *absolute* test. Any operator smaller than the floor then reports a happy
    breakdown on its first Krylov vector, collapsing every restart to a
    one-dimensional update.

    Measured at ``s = 1e-20`` on a moderately conditioned nonsymmetric ``M``
    (``cond = 32``): **801 matvecs and a relative residual of 0.904**, against
    32 matvecs and 2.1e-15 at ``s = 1``. Not a slow path -- a wrong answer,
    returned after exhausting the whole 400-restart budget.
    """
    rng = np.random.default_rng(3)
    n = 30
    M = rng.standard_normal((n, n)) + 3.0 * np.eye(n)
    assert 10.0 < float(np.linalg.cond(M)) < 1e3, "fixture must stay moderate"
    b0 = rng.standard_normal(n)

    calls = {"n": 0}

    def op(v):
        calls["n"] += 1
        return jnp.asarray(scale * M) @ v

    x, resid = gmres_eager(
        op, jnp.asarray(scale * b0), tol=1e-12, maxiter=400, restart=30
    )

    assert resid <= 1e-12, (scale, resid)
    # The solution itself is scale-invariant, so compare against the unscaled one.
    exact = np.linalg.solve(M, b0)
    rel = float(jnp.linalg.norm(x - jnp.asarray(exact)) / np.linalg.norm(exact))
    assert rel < 1e-10, (scale, rel)
    # And it did not spend the whole budget getting there. The bound is loose
    # on purpose: the defect exhausted all 400 restarts at 801 matvecs, and a
    # converged solve here costs 32 (or ~125 at ``s = 1e40``, where the
    # convergence target itself starts feeling float64 rounding). 200
    # separates those two regimes without pinning a restart count that a
    # different BLAS could shift by one.
    assert calls["n"] < 200, (scale, calls["n"])


def test_a_zero_right_hand_side_returns_zero():
    b = jnp.zeros(9)
    x, resid = gmres_eager(lambda v: 2.0 * v, b, tol=1e-12, maxiter=5, restart=3)
    assert resid == 0.0
    assert bool(jnp.all(x == 0.0))


def test_a_non_finite_right_hand_side_fails_closed():
    """A NaN must not come back as a converged solve.

    Every root-implicit gate in this library has had to be rewritten once to
    fail closed on NaN (#796 / #787 / #784); the solver underneath them is not
    going to be the exception.
    """
    b = jnp.asarray([1.0, jnp.nan, 3.0])
    _x, resid = gmres_eager(lambda v: v, b, tol=1e-12, maxiter=5, restart=2)
    assert resid == float("inf")


def test_a_nan_arising_on_the_last_restart_also_fails_closed():
    """The guard is on the exit, not on the loop head.

    A NaN that appears *during* the final restart never reaches the
    ``isfinite`` check at the top of the next one, so a per-iteration guard
    alone would let it out as ``nan``, and ``nan > tol`` is False.
    """
    M, b = _well_conditioned(n=6, seed=13)
    calls = {"n": 0}

    # One restart of a 2-dimensional space applies the operator four times:
    # the initial residual, two Arnoldi steps, and the closing true-residual
    # measurement.  Poison only that last one.
    def op(v):
        calls["n"] += 1
        return (M @ v) * (jnp.nan if calls["n"] == 4 else 1.0)

    _x, resid = gmres_eager(op, b, tol=1e-30, maxiter=1, restart=2)
    assert calls["n"] == 4, calls  # the fixture must reach the closing matvec
    assert resid == float("inf")


def test_a_nan_arising_mid_arnoldi_returns_before_poisoning_the_step():
    """Fail closed *and* early, from inside the Krylov loop.

    Carrying a NaN through the Givens recurrence finishes the restart, takes a
    NaN step, and then measures a residual against the poisoned ``x`` -- so the
    number the caller's gate reads is about a solution that no longer exists.
    """
    M, b = _well_conditioned(n=8, seed=23)
    calls = {"n": 0}

    def op(v):
        calls["n"] += 1
        # Call 1 is the initial residual; call 2 is the first Arnoldi step.
        return (M @ v) * (jnp.nan if calls["n"] == 2 else 1.0)

    _x, resid = gmres_eager(op, b, tol=1e-12, maxiter=4, restart=6)

    assert resid == float("inf")
    # It stopped there rather than grinding out the rest of the budget.
    assert calls["n"] == 2, calls


def test_an_operator_that_returns_nan_fails_closed():
    b = jnp.asarray([1.0, 2.0, 3.0])
    _x, resid = gmres_eager(lambda v: v * jnp.nan, b, tol=1e-12, maxiter=5, restart=2)
    assert resid == float("inf")


def test_complex_input_is_refused_rather_than_silently_mishandled():
    """The callers' operators are real-linear only, so this is not a limitation
    to paper over -- a complex Krylov space would span the wrong thing."""
    b = jnp.asarray([1.0 + 1j, 2.0])
    with pytest.raises(TypeError, match="real embedding"):
        gmres_eager(lambda v: v, b)


def test_a_two_dimensional_right_hand_side_is_refused():
    with pytest.raises(ValueError, match="1-D"):
        gmres_eager(lambda v: v, jnp.zeros((3, 3)))


# ---------------------------------------------------------------------------
# The structural property this module exists for (#731)
# ---------------------------------------------------------------------------


def test_the_operator_is_compiled_once_and_driven_many_times():
    """The 8.6 GB defect, in the one form a test can see.

    Under ``jit`` a Python matvec body runs exactly once -- at trace time -- and
    the loop that calls it becomes part of the same XLA program.  Eagerly it
    runs once per Krylov step against a program compiled once.  So:

      * ``traces == 1``  -- the operator is compiled a single time; and
      * ``invocations >> 1`` -- the loop is *outside* that program.

    Both halves are load-bearing.  Dropping the ``jit`` would keep the second
    and break the first (recompiling every step); putting the loop back under
    ``jit`` would keep the first and break the second, which is exactly the
    regression that reinstates #731.
    """
    M, b = _well_conditioned(n=24, seed=9)

    traces = 0
    invocations = 0

    def raw(v):
        nonlocal traces
        traces += 1  # Python body: runs on trace only.
        return M @ v

    compiled = jax.jit(raw)

    def op(v):
        nonlocal invocations
        invocations += 1
        return compiled(v)

    _x, resid = gmres_eager(op, b, tol=1e-12, maxiter=20, restart=10)

    assert resid <= 1e-12
    assert traces == 1, f"operator compiled {traces} times, expected once"
    assert invocations > 5, f"only {invocations} matvecs -- is the loop traced?"


def test_the_krylov_basis_is_the_only_vector_memory_it_holds():
    """``(m+1) x n``, and nothing that scales with the number of restarts.

    The point of #731 is that the *data* was never the problem: at ``D=2,
    chi=4`` the real embedding is ``n = 384`` and a 30-dimensional basis is
    91 KB.  A solver that accumulated per-restart history would reintroduce a
    memory term this one does not have, so pin the shape by running many
    restarts on a system that cannot converge and checking nothing grows.
    """
    n = 16
    rng = np.random.default_rng(4)
    # Singular AND inconsistent: b has a component outside range(M), so no
    # number of restarts converges and the loop runs its full budget.
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.concatenate([rng.uniform(0.5, 2.0, n - 1), [0.0]])
    M = jnp.asarray(Q @ np.diag(s) @ Q.T)
    b = jnp.asarray(Q[:, -1])  # exactly the null direction

    live = []

    def op(v):
        live.append(v.shape)
        return M @ v

    _x, resid = gmres_eager(op, b, tol=1e-14, maxiter=25, restart=4)

    assert np.isfinite(resid)
    assert resid > 1e-8, "the fixture must NOT converge, or this proves nothing"
    # Every matvec is on a length-n vector: no widening, no batching over
    # restarts.
    assert set(live) == {(n,)}, set(live)


def test_it_is_a_drop_in_for_the_adjoint_solver_contract():
    """``_solve_root_adjoint`` calls it through the real embedding.

    Guards the seam rather than the solver: a pytree in, the same pytree out,
    and a relative residual the caller's gate can read.
    """
    from tenax.algorithms._ctm_c4v_root_implicit import _solve_root_adjoint

    rng = np.random.default_rng(21)
    # A pytree with mixed shapes and one complex leaf -- the real embedding is
    # what makes that solvable at all.
    rhs = {
        "c": jnp.asarray(rng.standard_normal((3, 3))),
        "e": jnp.asarray(
            rng.standard_normal((2, 4)) + 1j * rng.standard_normal((2, 4))
        ),
    }
    scale = {"c": 2.0, "e": 3.0}

    def matvec(tree):
        # Real-linear but NOT complex-linear: it conjugates, exactly like the
        # VJP of ``F``.
        return {
            "c": scale["c"] * tree["c"],
            "e": scale["e"] * jnp.conj(tree["e"]),
        }

    sol, resid = _solve_root_adjoint(matvec, rhs, tol=1e-12, maxiter=20, restart=20)

    assert float(resid) <= 1e-10, resid
    assert set(sol) == set(rhs)
    for k in rhs:
        assert sol[k].shape == rhs[k].shape
    applied = matvec(sol)
    for k in rhs:
        assert bool(jnp.allclose(applied[k], rhs[k], atol=1e-10)), k
