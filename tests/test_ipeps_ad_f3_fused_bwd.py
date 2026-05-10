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


def _grad_for_method(method: str):
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
            gmres_maxiter=200,
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
