"""Tests for the F3 fused-backward refactor of implicit-AD CTM.

The fused JIT must produce element-wise identical gradients to the
F2 Python-loop fixed-point path within float roundoff.  It must also
match the GMRES path within the existing solver tolerance, since the
underlying linear system is the same.
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
