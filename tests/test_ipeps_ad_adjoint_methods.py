"""Tests for the ``adjoint_method`` config knob in implicit-AD CTM.

Both ``"fixed_point"`` (Neumann iteration, default) and ``"gmres"``
(eager Krylov solve) solve the same linear system
``(I - J^T) λ = dE/denv``, so they must produce equivalent gradients
within the solver tolerance.  The Arnoldi precheck guarantees the
fixed-point loop converges by rejecting configurations with ρ(J^T) ≥ 1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ad_utils import CTMRGGradientError
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor, optimize_gs_ad


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


@pytest.mark.algorithm
def test_fixed_point_matches_gmres_gradient():
    """``adjoint_method`` must not change the GRADIENT.

    ``adjoint_method`` only selects the *backward* linear solver; both methods
    see the identical forward fixed point and therefore solve the identical
    system ``(I - J^T) λ = dE/denv``.  Compared at a single gradient evaluation
    the two agree to well inside ``gmres_tol`` (measured: cos = 1.000000000000,
    relative L2 difference 1.9e-7), so this is gated tightly.

    Why not compare ``optimize_gs_ad`` end states (issue #740)?  Because that
    measures chaos amplification through the CTM fixed point, not the solver.
    Since #710 restored ket-bra Z2 at the fixed point, the projector singular
    values come in exact degenerate pairs whose LAPACK basis choice is
    platform- and version-dependent, so a sub-solver-tolerance difference at
    step 1 explodes across subsequent steps.  Measured on the old fixture
    (D=2, chi=8, lr=1e-2, seed 2026):

        steps   |dE|       max|dA|
        1       2.2e-09    1.3e-08
        2       5.3e-06    3.8e-03    <- what this test used to assert on
        3       5.2e-03    5.0e-02

    ~3 orders of magnitude per step.  The old 2-step assertion passed only
    before #710 (verified: green at 3f25688^, red at 3f25688) because the
    pre-#710 transposed convention broke the degeneracy.  A single step is
    still stable, so the production path is covered by the second half below.
    """
    H = _heisenberg_gate()
    A = _wrap_as_dense_tensor(_random_peps())

    def grad_with(method):
        def loss(A_):
            return ctm_energy_implicit(
                {(0, 0): A_},
                SINGLE_SITE_NEIGHBORS,
                H,
                chi=8,
                max_iter=40,
                conv_tol=1e-8,
                gmres_tol=1e-6,
                gmres_maxiter=200,
                adjoint_method=method,
            )

        g = jax.grad(loss)(A)
        return np.asarray(g.todense() if hasattr(g, "todense") else g).ravel()

    g_fp = grad_with("fixed_point")
    g_gmres = grad_with("gmres")

    n_fp, n_gmres = np.linalg.norm(g_fp), np.linalg.norm(g_gmres)
    assert n_fp > 1e-8 and n_gmres > 1e-8, "gradient collapsed to zero"

    rel = float(np.linalg.norm(g_fp - g_gmres) / n_fp)
    assert rel < 1e-5, (
        f"adjoint_method changed the gradient beyond the solver tolerance: "
        f"rel={rel:.3e} (|g_fp|={n_fp:.6e}, |g_gmres|={n_gmres:.6e})"
    )

    # Production path: one optimizer step is still below the amplification
    # threshold, so the two methods must agree there element-wise.
    def make_config(method: str) -> iPEPSConfig:
        return iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(
                chi=8,
                max_iter=40,
                conv_tol=1e-8,
                adjoint_method=method,
            ),
            gs_num_steps=1,
            gs_learning_rate=1e-2,
            su_init=False,
            gs_metric_precond=False,
        )

    A_init = _random_peps()
    A_fp, _, E_fp = optimize_gs_ad(H, A_init, make_config("fixed_point"))
    A_gmres, _, E_gmres = optimize_gs_ad(H, A_init, make_config("gmres"))

    assert abs(float(E_fp) - float(E_gmres)) < 1e-6, (
        f"one-step energies should match: fixed_point={float(E_fp)}, "
        f"gmres={float(E_gmres)}, diff={abs(float(E_fp) - float(E_gmres))}"
    )
    A_fp_arr = np.asarray(A_fp.todense() if hasattr(A_fp, "todense") else A_fp)
    A_gmres_arr = np.asarray(
        A_gmres.todense() if hasattr(A_gmres, "todense") else A_gmres
    )
    np.testing.assert_allclose(
        A_fp_arr,
        A_gmres_arr,
        rtol=1e-5,
        atol=1e-6,
        err_msg="one-step tensors should match within solver tolerance",
    )


@pytest.mark.algorithm
def test_fixed_point_arnoldi_rejects_high_rho():
    """Arnoldi precheck must catch ρ(J^T) ≥ 1 before the loop runs.

    When the precheck is on and the spectrum is unfavorable (here:
    ``forward_gauge="none"`` at chi=4), the backward must raise
    ``CTMRGGradientError`` instead of looping to ``gmres_maxiter``.

    ``seed=42`` is chosen because it lands on ρ(J^T) ≈ 3.20 on the
    current CTM iteration — comfortably above the rejection threshold
    (1.0) and not borderline. The previous ``seed=1`` fixture drifted to
    ρ ≈ 0.54 < 1 after later CTM-iteration fixes shipped, so the precheck
    stopped firing (issue #469). Other rejecting seeds at this config: 25.
    """
    H = _heisenberg_gate()
    A = _wrap_as_dense_tensor(_random_peps(seed=42))

    def loss(A_):
        return ctm_energy_implicit(
            {(0, 0): A_},
            SINGLE_SITE_NEIGHBORS,
            H,
            chi=4,
            max_iter=20,
            conv_tol=1e-6,
            forward_gauge="none",
            gmres_tol=1e-6,
            gmres_maxiter=200,
            arnoldi_precheck=True,
            adjoint_method="fixed_point",
        )

    with pytest.raises(CTMRGGradientError):
        jax.grad(loss)(A)
