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
def test_fixed_point_matches_gmres_at_chi8():
    """``adjoint_method`` choice must not change the optimization outcome.

    Both Neumann iteration and eager GMRES solve the same linear system
    to the same tolerance.  Running ``optimize_gs_ad`` from the same
    seed under both methods must yield matching energies and tensors.
    """
    H = _heisenberg_gate()
    A_init = _random_peps()

    def make_config(method: str) -> iPEPSConfig:
        return iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(
                chi=8,
                max_iter=40,
                conv_tol=1e-8,
                adjoint_method=method,
            ),
            gs_num_steps=2,
            gs_learning_rate=1e-2,
            su_init=False,
            gs_metric_precond=False,
        )

    A_fp, _, E_fp = optimize_gs_ad(H, A_init, make_config("fixed_point"))
    A_gmres, _, E_gmres = optimize_gs_ad(H, A_init, make_config("gmres"))

    # The two backward solvers do NOT converge to the same tolerance: GMRES
    # solves its linear system to ``gmres_tol`` (1e-6) while the Neumann
    # fixed-point loop targets ``adjoint_tol`` (1e-8). Their gradients can
    # therefore differ at the ~``gmres_tol`` level, which after two optimizer
    # steps (lr=1e-2) shows up as an energy/tensor disagreement of order 1e-5.
    # A 1e-6 match is thus tighter than the solvers themselves and fails on
    # some platforms' BLAS (observed CI diff ≈ 1.8e-5). Assert agreement at the
    # solver-tolerance scale instead — still tight enough to catch a genuine
    # method discrepancy (which would be orders of magnitude larger).
    assert abs(float(E_fp) - float(E_gmres)) < 1e-4, (
        f"Energies should match within solver tolerance: "
        f"fixed_point={float(E_fp)}, gmres={float(E_gmres)}, "
        f"diff={abs(float(E_fp) - float(E_gmres))}"
    )

    A_fp_arr = np.asarray(A_fp.todense() if hasattr(A_fp, "todense") else A_fp)
    A_gmres_arr = np.asarray(
        A_gmres.todense() if hasattr(A_gmres, "todense") else A_gmres
    )
    np.testing.assert_allclose(
        A_fp_arr,
        A_gmres_arr,
        rtol=1e-3,
        atol=1e-5,
        err_msg=("Final tensors should match element-wise within solver tolerance"),
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
