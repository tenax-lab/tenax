"""The implicit-AD backward must not use an unconverged adjoint silently.

The backward solves ``(I - J^T) λ = dE/denv`` and then contracts ``λ`` into the
gradient.  If that solve has not converged, the gradient is wrong — finite,
plausibly scaled, and indistinguishable downstream from a correct one.  The
C4v root-implicit path already refuses to let that pass silently (#716); this
pins the same guarantee for the *default* iPEPS AD gradient path (#801, first
raised on #341).

**Why the convergence flag cannot be the guard.**  Both call sites bind the
solver's second return value and discard it, and consuming it would not help:
``jax.scipy.sparse.linalg.gmres`` returns ``info = 0`` whether or not it
converged --

    >>> gmres(lambda v: A @ v, b, tol=1e-14, atol=1e-14, maxiter=1, restart=1)
    (Array([...]), Array(0, dtype=int64))   # residual 0.54

-- so a fix that reads ``info`` would look right and change nothing.  The
residual has to be measured.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_energy_ad import (
    ctm_energy_implicit,
    get_last_implicit_ad_diagnostics,
)
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _heisenberg_gate():
    d = 2
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(d, d, d, d)


def _random_peps(seed=2026, D=2, d=2):
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (D, D, D, D, d))
    return _wrap_as_dense_tensor(A / (jnp.linalg.norm(A) + 1e-10))


def _grad(A, H, *, maxiter, restart, tol, method="gmres", ctm_max_iter=20):
    """Gradient through the implicit-AD energy.

    ``ctm_max_iter`` is separate from ``maxiter`` on purpose, and the tests
    that assert *silence* must raise it.  The backward linearises about the
    forward CTM iterate: if the forward has not reached its fixed point,
    ``(I - Jᵀ)`` is not the operator the adjoint system assumes and GMRES
    stalls no matter how large its Krylov budget is.  Measured on this D=2
    chi=4 fixture:

        ctm_max_iter =  20  ->  adjoint residual 2.5e-02   (guard fires)
        ctm_max_iter = 100  ->  adjoint residual 1.3e-10
        ctm_max_iter = 300  ->  adjoint residual 3.4e-10

    At 20 the guard is *correct* to warn, so a silence assertion there is
    testing luck: it held on CPU and failed on CUDA, where the unconverged
    forward lands somewhere else.  Starve the Krylov budget to provoke the
    warning, never the forward sweep count.
    """

    def loss(A_):
        return ctm_energy_implicit(
            {(0, 0): A_},
            SINGLE_SITE_NEIGHBORS,
            H,
            chi=4,
            max_iter=ctm_max_iter,
            conv_tol=1e-8,
            gmres_tol=tol,
            gmres_maxiter=maxiter,
            gmres_restart=restart,
            adjoint_method=method,
        )

    return jax.grad(loss)(A)


def test_an_unconverged_gmres_adjoint_warns():
    """A starved Krylov budget must not return a gradient in silence."""
    A, H = _random_peps(), _heisenberg_gate()
    with pytest.warns(RuntimeWarning, match="adjoint solve did not converge"):
        g = _grad(A, H, maxiter=1, restart=1, tol=1e-14)
    # The gradient is still returned -- the guard reports, it does not abort.
    assert bool(jnp.all(jnp.isfinite(g if not hasattr(g, "todense") else g.todense())))


def test_a_converged_gmres_adjoint_is_silent():
    """The guard must not fire on a healthy solve, or it is noise.

    Without this the previous test passes against a helper that warns
    unconditionally.

    ``ctm_max_iter`` is raised so the forward actually reaches its fixed
    point — see ``_grad``. With the default 20 this asserts silence about a
    solve that genuinely does not converge, which is a backend-dependent
    coin flip rather than a property of the guard.
    """
    A, H = _random_peps(), _heisenberg_gate()

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _grad(A, H, maxiter=200, restart=20, tol=1e-6, ctm_max_iter=200)
    bad = [
        str(w.message)
        for w in rec
        if "adjoint solve did not converge" in str(w.message)
    ]
    assert not bad, f"guard fired on a converged solve: {bad}"


def test_the_warning_reports_the_residual_and_the_tolerance():
    """A bare 'did not converge' is not actionable.

    The C4v sibling names the residual, the tolerance and the remedy; this
    path should too, so a caller can tell a near-miss from a total failure.
    """
    A, H = _random_peps(), _heisenberg_gate()
    with pytest.warns(RuntimeWarning) as rec:
        _grad(A, H, maxiter=1, restart=1, tol=1e-14)
    msg = next(
        str(w.message)
        for w in rec
        if "adjoint solve did not converge" in str(w.message)
    )
    assert "relative residual" in msg, msg
    assert "gmres_maxiter" in msg, msg


def test_the_default_fixed_point_path_is_covered_too():
    """``fixed_point`` is the default, so it is the one that matters.

    It does not solve the system itself when the fused Neumann loop fails —
    it falls back to the same eager GMRES. Starving the budget therefore has
    to surface through the same guard, or the default gradient path stays
    silent while only the opt-out is protected.
    """
    A, H = _random_peps(), _heisenberg_gate()
    with pytest.warns(RuntimeWarning, match="adjoint solve did not converge"):
        _grad(A, H, maxiter=1, restart=1, tol=1e-14, method="fixed_point")


# --- review follow-ups on #804 -------------------------------------------


def test_the_gate_uses_the_same_criterion_as_the_solver():
    """A guard that cries wolf gets ignored, which would undo #801.

    ``gmres_pytree_jax`` configures JAX GMRES with ``tol=tol, atol=tol``, and
    that solver stops at ``‖r‖ <= max(tol·‖b‖, atol)``.  Gating on the
    *relative* residual alone is a stricter test, so whenever ``‖b‖ < 1`` the
    solver can legitimately stop above the gate and be reported as a failure.

    ``‖rhs‖`` is O(1) here (measured 1.285 on the D=2 chi=8 fixture), so this
    is one state away rather than hypothetical.
    """
    from tenax.algorithms._ctm_energy_ad import _adjoint_converged

    tol = 1e-6
    # ‖b‖ = 1: the two criteria coincide.
    assert _adjoint_converged(1e-7, 1.0, tol) is True
    assert _adjoint_converged(1e-5, 1.0, tol) is False

    # ‖b‖ = 1e-3: the solver's floor is atol = tol, so a residual of 1e-7 is
    # converged even though ‖r‖/‖b‖ = 1e-4 is far above tol.  This is the
    # false positive.
    assert _adjoint_converged(1e-7, 1e-3, tol) is True
    # Still rejects a genuinely bad solve at the same ‖b‖.
    assert _adjoint_converged(1e-2, 1e-3, tol) is False

    # ‖b‖ = 100: the relative criterion binds and is the looser of the two.
    assert _adjoint_converged(1e-5, 100.0, tol) is True


def test_the_gate_still_fails_closed_on_a_non_finite_residual():
    """``nan > tol`` is False; the guard must not take the silent branch.

    This is the #796 failure shape, pinned here so neither the original fix
    nor the tolerance change above can be written with the comparison that
    reintroduces it.
    """
    from tenax.algorithms._ctm_energy_ad import _adjoint_converged

    assert _adjoint_converged(float("nan"), 1.0, 1e-6) is False
    assert _adjoint_converged(float("inf"), 1.0, 1e-6) is False
    # ...including when ‖b‖ itself is degenerate.
    assert _adjoint_converged(float("nan"), float("nan"), 1e-6) is False


def test_a_non_finite_rhs_norm_cannot_make_the_threshold_swallow_the_residual():
    """The ‖b‖-relative threshold must not become the thing that fails open.

    Scaling the tolerance by ‖b‖ introduces a failure mode the residual-only
    check does not have: if the RHS norm overflows, ``max(tol·inf, tol)`` is
    ``inf``, and *every* residual satisfies ``r <= inf`` — including ``inf``
    itself.  So the exact situation the guard exists to catch, a blown-up
    adjoint solve, is the one it reports as converged.

    Note ``(nan, inf)`` already fails closed for an unrelated reason
    (``nan <= inf`` is False), so testing only nan residuals misses this.
    An infinite ‖b‖ means the RHS itself is garbage, so no residual against
    it is meaningful, however small.
    """
    from tenax.algorithms._ctm_energy_ad import _adjoint_converged

    inf = float("inf")
    assert _adjoint_converged(inf, inf, 1e-6) is False
    # A *small* residual against an infinite RHS is not evidence of anything.
    assert _adjoint_converged(1e-9, inf, 1e-6) is False
    assert _adjoint_converged(1e-9, float("nan"), 1e-6) is False


def test_a_successful_fixed_point_backward_replaces_a_stale_residual():
    """Diagnostics must describe the latest solve, not a previous one.

    Originally pinned as "the fused path clears the key", because that path
    measured nothing and absent was the best it could do.  It now measures its
    own residual (#824), so the same requirement is met more strongly: the
    value describes *this* solve rather than merely not describing the last
    one.  The assertion tracks that — what must never happen is the previous
    solve's number sitting next to this solve's ``converged=True``.
    """
    from tenax.algorithms import _ctm_energy_ad as M

    A, H = _random_peps(), _heisenberg_gate()
    # Starve the budget so the eager fallback runs and records a bad residual.
    with pytest.warns(RuntimeWarning):
        _grad(A, H, maxiter=1, restart=1, tol=1e-14)
    starved = M._F3_LAST_DIAGNOSTICS.get("adjoint_residual")
    assert starved is not None and starved > 1e-8, starved

    # Now a healthy fixed-point backward: the stale value must not survive it.
    # Converged forward (see _grad) so "healthy" is a fact, not a coin flip.
    _grad(
        A,
        H,
        maxiter=200,
        restart=20,
        tol=1e-6,
        method="fixed_point",
        ctm_max_iter=200,
    )
    fresh = M._F3_LAST_DIAGNOSTICS.get("adjoint_residual")
    assert fresh is not None, (
        "the fused fixed-point backward published no residual at all; a "
        "consumer cannot distinguish that from a solve it never ran"
    )
    assert fresh != starved, (
        f"stale adjoint_residual={starved} survived a successful fixed-point "
        "backward; get_last_implicit_ad_diagnostics() would report it "
        "alongside converged=True for a different solve"
    )
    assert fresh < 1e-5, f"healthy solve reported residual {fresh}"


# --- #824: the fused fixed-point happy path -------------------------------


def test_the_fixed_point_happy_path_publishes_a_residual():
    """The default solver must report a residual when it *succeeds*, too.

    ``test_the_default_fixed_point_path_is_covered_too`` above only reaches
    the eager-GMRES fallback — it starves the budget so the fused loop fails.
    When that loop stops on its own criterion it returns straight to the
    caller, and before #824 that path measured nothing: no residual, no
    warning, ``converged=True``.  Since it is the default ``adjoint_method``
    that was the hole covering the majority of real backward calls, and #824
    measured it returning a gradient wrong by 3e-3 on CUDA under exactly that
    clean bill of health.
    """
    A, H = _random_peps(), _heisenberg_gate()
    _grad(
        A,
        H,
        maxiter=200,
        restart=20,
        tol=1e-6,
        method="fixed_point",
        ctm_max_iter=200,
    )

    diag = get_last_implicit_ad_diagnostics()
    assert diag.get("converged") is True, (
        "precondition: this test must exercise the fused happy path, not the "
        f"eager fallback (diagnostics={diag})"
    )
    assert "adjoint_residual" in diag, (
        "the fused fixed-point loop returned without publishing a residual; "
        f"diagnostics={diag}"
    )
    assert diag["adjoint_residual"] < 1e-5, diag


def test_the_fixed_point_residual_tracks_the_tolerance():
    """Pins that the number is measured, not a constant.

    A hardcoded ``0.0`` — or an alias of the loop's own ``diff``, which is a
    different norm — passes the presence check above.  Stopping the Neumann
    iteration earlier via a looser ``gmres_tol`` has to show up as a *larger*
    reported residual, which only holds if ``‖(I - Jᵀ)λ - b‖/‖b‖`` is really
    being evaluated.
    """
    A, H = _random_peps(), _heisenberg_gate()
    seen = {}
    for tol in (1e-2, 1e-8):
        _grad(
            A,
            H,
            maxiter=200,
            restart=20,
            tol=tol,
            method="fixed_point",
            ctm_max_iter=200,
        )
        diag = get_last_implicit_ad_diagnostics()
        assert diag.get("converged") is True, (tol, diag)
        seen[tol] = diag["adjoint_residual"]

    assert seen[1e-2] > seen[1e-8], (
        f"residual did not respond to gmres_tol: {seen} — it is not being "
        "measured at the returned λ"
    )


def test_the_fused_happy_path_is_residual_gated_by_construction():
    """Why the fused branch carries no convergence warning: it cannot need one.

    Returning to the caller means the loop set ``converged``, i.e.
    ``diff <= gmres_tol``.  ``diff`` is the ell1-of-ell2 sum over leaves of the
    same step vector whose plain ell2 is the published residual, and
    ``||v||_2 <= ||v||_1``, so the residual is always at most ``gmres_tol``
    while the gate's threshold is ``max(tol*||b||, tol) >= tol``.  A warning
    there would be unreachable code that reads like a guard.

    Pinned as a property rather than argued in a comment: if someone changes
    the loop's stopping criterion so it no longer implies the residual bound,
    this fails and the branch genuinely does need a gate again.
    """
    A, H = _random_peps(), _heisenberg_gate()
    for tol in (1e-4, 1e-6, 1e-8):
        _grad(
            A,
            H,
            maxiter=200,
            restart=20,
            tol=tol,
            method="fixed_point",
            ctm_max_iter=200,
        )
        diag = get_last_implicit_ad_diagnostics()
        assert diag.get("converged") is True, (tol, diag)
        assert diag["adjoint_residual"] <= tol, (
            f"published residual {diag['adjoint_residual']:.3e} exceeds "
            f"gmres_tol {tol:.1e} on the happy path — the branch now needs "
            "the convergence gate that was removed as unreachable"
        )
