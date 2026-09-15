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
    system ``(I - J^T) λ = dE/denv``.

    The first assertion below is a solver SELF-CONSISTENCY check, not a
    gradient-accuracy check (#827): because the two evaluations share the
    forward environment, it certifies ~1e-7 agreement for a quantity the
    forward stopping criterion may determine far more loosely.  On a platform
    whose LAPACK/cuSOLVER build leaves this fixture's environment without an
    element-wise fixed point, #827 measured the gmres gradient itself moving
    2.5e-2..4.5e-2 relative as ``(max_iter, conv_tol)`` went over
    (40,1e-8)/(100,1e-10)/(300,1e-12) vs a (300,1e-14) reference, while this
    check still read 1.28e-7.

    Fixture and forward settings (#827):

    * The seed-2026 random tensor is KEPT, with the forward tightened to
      ``conv_tol=1e-10, max_iter=100``.  On the reference CPU env (jaxlib
      0.10.1) its phase-gauged CTM reaches a true element-wise fixed point
      (residual 1.3e-14 in 69 sweeps at chi=8) — the precondition for the
      implicit adjoint to linearize around an actual fixed point — and the
      gradient is then *determined*: drift vs a (300,1e-14) reference is
      8.0e-6 at the old (40,1e-8), 4.5e-11 at (100,1e-10) used here, and
      2.1e-13 at (300,1e-12).
    * The SU-relaxed D=2 Heisenberg fixture #827 proposed was measured and
      REJECTED: physical SU states carry near-degenerate corner multiplets
      (cut ratio 1.10 at chi=8; SVD-VJP factors 1/(s_i^2 - s_j^2) ~ 3e8),
      their environment never becomes an element-wise fixed point under the
      phase or sigma gauge (residual plateaus at 3.9e-2 after 300 sweeps),
      and their gradient drift plateaus at 1e-1..3.5e-1 — flat in conv_tol,
      chi (8 or the gap-respecting 10), and under symmetry-breaking
      perturbation.  Swapping it in would make this test's gradient *less*
      determined, not more.

    Why not compare ``optimize_gs_ad`` end states (issue #740)?  Because that
    measures chaos amplification through the CTM fixed point, not the solver.
    Since #710 restored ket-bra Z2 at the fixed point, the projector singular
    values come in exact degenerate pairs whose LAPACK basis choice is
    platform- and version-dependent, so a sub-solver-tolerance difference at
    step 1 explodes across subsequent steps.  Measured on this fixture
    (D=2, chi=8, lr=1e-2, seed 2026):

        steps   |dE|       max|dA|
        1       2.2e-09    1.3e-08
        2       5.3e-06    3.8e-03    <- what this test used to assert on
        3       5.2e-03    5.0e-02

    ~3 orders of magnitude per step.  The old 2-step assertion passed only
    before #710 (verified: green at 3f25688^, red at 3f25688) because the
    pre-#710 transposed convention broke the degeneracy.  A single step is
    the largest horizon on which the two methods are comparable at all, so
    the production path is covered by the second half below, with an energy
    tolerance derived from the measured single-step amplification.
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
                max_iter=100,
                conv_tol=1e-10,
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

    # Solver self-consistency (see docstring): identical forward, so the two
    # backward solvers must agree far inside gmres_tol.  Measured healthy:
    # 9.9e-8; a crippled adjoint (gmres_maxiter=1, gmres_restart=1) lands at
    # 7.6e-2, 7600x over this gate.
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
                max_iter=100,
                conv_tol=1e-10,
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

    # Tolerance derivation (#827).  The step tensors agree to
    # max|dA| ~ lr * |g| * rel ~ 1e-8 (measured 1.5e-8, and |dE| = 9.0e-9,
    # CPU jaxlib 0.10.1) — but E_fp and E_gmres each come from a FRESH CTM
    # run at those two nearby tensors, and the exactly degenerate ket-bra
    # projector pairs (#710) give that map platform-dependent fault lines: on
    # the CUDA box's CPU path this same comparison measured |dE| = 1.37e-4
    # with the gradients still agreeing to 1.28e-7 (#827).  5e-4 sits 3.6x
    # above that worst measured platform jump and 4.5x below the 2.28e-3 a
    # genuinely broken adjoint produces (gmres_maxiter=1, gmres_restart=1,
    # measured); the rel gate above and the element-wise comparison below
    # catch that same breakage at 7600x and 7900x respectively.
    assert abs(float(E_fp) - float(E_gmres)) < 5e-4, (
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
