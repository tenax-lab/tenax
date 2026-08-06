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

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
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


def _grad(A, H, *, maxiter, restart, tol, method="gmres"):
    def loss(A_):
        return ctm_energy_implicit(
            {(0, 0): A_},
            SINGLE_SITE_NEIGHBORS,
            H,
            chi=4,
            max_iter=20,
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
    """
    A, H = _random_peps(), _heisenberg_gate()
    import warnings

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _grad(A, H, maxiter=200, restart=20, tol=1e-6)
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


def test_the_gate_fails_closed_on_a_nan_residual():
    """``nan > tol`` is False; the guard must not take the silent branch.

    This is the #796 failure shape, pinned here so the fix cannot be written
    with the comparison that reintroduces it.
    """
    from tenax.algorithms._ctm_energy_ad import _adjoint_converged

    assert _adjoint_converged(1e-9, 1e-6) is True
    assert _adjoint_converged(1e-3, 1e-6) is False
    assert _adjoint_converged(float("nan"), 1e-6) is False
    assert _adjoint_converged(float("inf"), 1e-6) is False
