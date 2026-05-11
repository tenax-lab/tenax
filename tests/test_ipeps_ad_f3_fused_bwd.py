"""Tests for the F3 fused-backward refactor of implicit-AD CTM.

The ``fixed_point`` and ``gmres`` adjoint methods solve the same
``(I - J^T) λ = b`` linear system to the same target tolerance, so
their gradients must agree within the solver tolerance shared by
both paths.  This test passes today against the F2 Python-loop
fixed_point implementation; once Task 4 wires the F3 fused JIT into
the same code path, the same assertion guards against any gradient
drift introduced by the JIT fusion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    return A / (jnp.linalg.norm(A) + 1e-10)


def _grad_for_method(method: str, gmres_maxiter: int = 200):
    H = _heisenberg_gate()
    A = _wrap_as_dense_tensor(_random_peps())

    def loss(A_):
        return ctm_energy_implicit(
            {(0, 0): A_},
            SINGLE_SITE_NEIGHBORS,
            H,
            chi=8,
            max_iter=40,
            conv_tol=1e-6,
            forward_gauge="phase",
            gmres_tol=1e-6,
            gmres_maxiter=gmres_maxiter,
            # Parity test: skip the divergence guard so both methods
            # actually run the solve. The chi=8 phase-gauge config is
            # well-posed (variPEPS converges here in <30 Neumann iters).
            arnoldi_precheck=False,
            adjoint_method=method,
        )

    return jax.grad(loss)(A)


@pytest.mark.algorithm
def test_fused_bwd_matches_gmres_at_chi8():
    """F3 fused JIT must match eager GMRES gradient within solver tol."""
    g_fp = _grad_for_method("fixed_point")
    g_gm = _grad_for_method("gmres")

    g_fp_arr = np.asarray(g_fp.todense() if hasattr(g_fp, "todense") else g_fp)
    g_gm_arr = np.asarray(g_gm.todense() if hasattr(g_gm, "todense") else g_gm)
    np.testing.assert_allclose(
        g_fp_arr,
        g_gm_arr,
        rtol=1e-5,
        atol=1e-7,
        err_msg="F3 fused fixed_point and GMRES gradients diverged",
    )


@pytest.mark.algorithm
def test_fused_bwd_converges_without_fallback():
    """The fused JIT must actually solve the system itself.

    A passing parity test is necessary but not sufficient: if the fused
    fixed-point iteration silently diverges every call, the in-loop
    guard fires and the eager-GMRES fallback rescues the gradient,
    making the parity test green while F3 does strictly more work than
    F2. This test asserts the fused JIT converges on a well-posed
    config without invoking the fallback.
    """
    from tenax.algorithms._ctm_energy_ad import _F3_LAST_DIAGNOSTICS

    _F3_LAST_DIAGNOSTICS.clear()
    _ = _grad_for_method("fixed_point")

    assert "diverged" in _F3_LAST_DIAGNOSTICS, (
        "f_bwd did not record diagnostics — fused-JIT path may not have run"
    )
    assert not _F3_LAST_DIAGNOSTICS["diverged"], (
        f"Fused fixed-point diverged: {_F3_LAST_DIAGNOSTICS}. "
        "F3 fell back to eager GMRES — the fused JIT is not doing the work."
    )
    assert _F3_LAST_DIAGNOSTICS["converged"], (
        f"Fused fixed-point did not converge: {_F3_LAST_DIAGNOSTICS}"
    )
    assert _F3_LAST_DIAGNOSTICS["n_iter"] > 1, (
        f"Suspicious n_iter (=initial value?): {_F3_LAST_DIAGNOSTICS}"
    )


@pytest.mark.algorithm
def test_slow_neumann_falls_back_to_gmres():
    """Regression for issue #420.

    When the fused Neumann iteration exits at ``gmres_maxiter`` without
    reaching ``gmres_tol`` and without tripping the divergence guard,
    the truncated iterate must NOT be silently consumed by the chain
    rule. Before the fix, ``converged=False & diverged=False`` skipped
    the eager-GMRES fallback and returned a biased gradient.

    Test construction:

    * Set ``gmres_maxiter=2``. The in-loop divergence guard is
      ``(diff > prev_diff) & (k > 5)``, so with ``k`` at most 2 the
      guard is structurally inactive and ``diverged`` is guaranteed
      ``False``.
    * ``gmres_tol=1e-6`` cannot be reached in two Neumann steps for
      this Heisenberg problem, so ``converged`` is ``False``.
    * Diagnostics therefore show ``converged=False & diverged=False`` —
      precisely the issue-#420 scenario, independent of ``ρ(J^T)``.

    Reference is the eager-GMRES adjoint path, which solves the same
    ``(I - J^T) λ = b`` system to the same tolerance and is unaffected
    by the fixed-point gate.
    """
    from tenax.algorithms._ctm_energy_ad import _F3_LAST_DIAGNOSTICS

    # Reference: eager-GMRES adjoint — gives the true (I - J^T)^{-1} b.
    g_ref = _grad_for_method("gmres", gmres_maxiter=200)

    # Starve the fused Neumann iteration. maxiter=2 < 6 ⇒ the
    # divergence guard cannot fire, regardless of ρ(J^T).
    _F3_LAST_DIAGNOSTICS.clear()
    g_starved = _grad_for_method("fixed_point", gmres_maxiter=2)
    assert not _F3_LAST_DIAGNOSTICS.get("diverged"), (
        "Starved Neumann recorded diverged=True — guard should be "
        f"structurally inactive at k<=2. Diagnostics: {_F3_LAST_DIAGNOSTICS}"
    )
    assert not _F3_LAST_DIAGNOSTICS.get("converged"), (
        "Starved Neumann unexpectedly converged in two iterations — pick "
        f"a tighter starvation budget. Diagnostics: {_F3_LAST_DIAGNOSTICS}"
    )

    g_ref_arr = np.asarray(g_ref.todense() if hasattr(g_ref, "todense") else g_ref)
    g_starved_arr = np.asarray(
        g_starved.todense() if hasattr(g_starved, "todense") else g_starved
    )
    np.testing.assert_allclose(
        g_starved_arr,
        g_ref_arr,
        rtol=1e-5,
        atol=1e-7,
        err_msg=(
            "Issue #420: starved Neumann returned a biased gradient. The "
            "eager-GMRES fallback must fire on converged=False & "
            "diverged=False, not only on the diverging case."
        ),
    )
