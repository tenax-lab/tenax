"""Tests for metric preconditioning (norm environment) for iPEPS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._metric_precond import (
    _contract_single_site_environment,
    lbfgs_two_loop,
    norm_environment_matvec,
    precondition_gradient,
)
from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _make_random_ipeps_tensor(D=2, d=2, seed=0):
    """Create a random DenseTensor iPEPS site tensor."""
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, d))
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    return DenseTensor(data, indices)


def _converge_ctm(A, chi=4, max_iter=50):
    """Run CTM to convergence and return the single-site environment."""
    return ctm_tensor(A, chi=chi, max_iter=max_iter, conv_tol=1e-10)


# ------------------------------------------------------------------ #
# Tests: _contract_single_site_environment                             #
# ------------------------------------------------------------------ #


class TestContractSingleSiteEnv:
    """Tests for the single-site environment contraction."""

    def test_output_shape(self):
        """E has shape (D^2, D^2, D^2, D^2)."""
        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        env = _converge_ctm(A)
        E = _contract_single_site_environment(env)
        assert E.shape == (D**2, D**2, D**2, D**2)

    def test_reproduces_norm(self):
        """E contracted with double-layer tensor gives a positive norm."""
        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        env = _converge_ctm(A)
        E = _contract_single_site_environment(env)

        # Build the double-layer tensor (u2, d2, l2, r2) and contract with E
        from tenax.contraction.contractor import contract as tensor_contract

        a2 = _build_double_layer_tensor(A)
        a2_dense = a2.todense()  # (D^2, D^2, D^2, D^2)

        # Norm = E_{u2,d2,l2,r2} * a2_{u2,d2,l2,r2}
        norm = jnp.einsum("ijkl,ijkl->", E, a2_dense)
        assert norm.real > 0, f"Expected positive norm, got {norm}"


# ------------------------------------------------------------------ #
# Tests: norm_environment_matvec                                       #
# ------------------------------------------------------------------ #


class TestNormEnvironmentMV:
    """Tests for the norm-environment matvec."""

    def test_output_shape(self):
        """N . v has shape (D, D, D, D, d)."""
        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        env = _converge_ctm(A)

        key = jax.random.PRNGKey(42)
        v = jax.random.normal(key, (D, D, D, D, d))
        Nv = norm_environment_matvec(A, env, v)
        assert Nv.shape == (D, D, D, D, d)

    def test_positive_definite(self):
        """<v|N|v> > 0 for random v."""
        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        env = _converge_ctm(A)

        key = jax.random.PRNGKey(42)
        v = jax.random.normal(key, (D, D, D, D, d))
        Nv = norm_environment_matvec(A, env, v)

        vNv = jnp.sum(v * Nv)
        assert vNv.real > 0, f"Expected <v|N|v> > 0, got {vNv}"

    def test_hermitian(self):
        """<u|N|v> = conj(<v|N|u>)."""
        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        env = _converge_ctm(A)

        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(99)
        u = jax.random.normal(key1, (D, D, D, D, d))
        v = jax.random.normal(key2, (D, D, D, D, d))

        Nu = norm_environment_matvec(A, env, u)
        Nv = norm_environment_matvec(A, env, v)

        uNv = jnp.sum(u * Nv)
        vNu = jnp.sum(v * Nu)
        np.testing.assert_allclose(
            uNv,
            jnp.conj(vNu),
            rtol=1e-5,
            atol=1e-10,
            err_msg="Norm environment is not Hermitian",
        )


# ------------------------------------------------------------------ #
# Tests: precondition_gradient                                         #
# ------------------------------------------------------------------ #


class TestPreconditionGradient:
    def test_finite_output(self):
        """Preconditioned gradient should be finite and nonzero."""
        A = _make_random_ipeps_tensor(D=2, d=2)
        env = _converge_ctm(A)
        grad = jax.random.normal(jax.random.PRNGKey(99), (2, 2, 2, 2, 2))
        config = iPEPSConfig(gs_metric_precond=True)
        g_precond = precondition_gradient(A, env, grad, delta=0.01, config=config)
        assert g_precond.shape == (2, 2, 2, 2, 2)
        assert jnp.all(jnp.isfinite(g_precond))
        assert float(jnp.sum(jnp.abs(g_precond))) > 0

    def test_identity_at_large_delta(self):
        """With very large delta, (N + delta*I)^{-1} approx (1/delta)*I."""
        A = _make_random_ipeps_tensor(D=2, d=2)
        env = _converge_ctm(A)
        grad = jax.random.normal(jax.random.PRNGKey(99), (2, 2, 2, 2, 2))
        config = iPEPSConfig(gs_metric_precond=True, metric_gmres_tol=1e-6)
        g_precond = precondition_gradient(A, env, grad, delta=1e6, config=config)
        expected = grad / 1e6
        assert jnp.allclose(g_precond, expected, rtol=0.1)


# ------------------------------------------------------------------ #
# Tests: lbfgs_two_loop                                                #
# ------------------------------------------------------------------ #


class TestLBFGSTwoLoop:
    def test_empty_history_returns_h0_grad(self):
        """With no history, result = h0_matvec(grad)."""
        grad = jnp.array([1.0, 2.0, 3.0])

        def h0_matvec(v):
            return 0.5 * v

        result = lbfgs_two_loop(grad, [], h0_matvec)
        assert jnp.allclose(result, 0.5 * grad)

    def test_one_step_history(self):
        """With one (s, y) pair, verify descent direction."""
        s = jnp.array([1.0, 0.0])
        y = jnp.array([2.0, 1.0])
        rho = 1.0 / jnp.dot(y, s)
        history = [(s, y, float(rho))]
        grad = jnp.array([1.0, 1.0])
        gamma = float(jnp.dot(s, y) / jnp.dot(y, y))

        def h0_matvec(v):
            return gamma * v

        result = lbfgs_two_loop(grad, history, h0_matvec)
        assert jnp.all(jnp.isfinite(result))
        assert float(jnp.dot(result, grad)) > 0  # descent direction

    def test_multi_step_history(self):
        """L-BFGS with multiple history entries should be finite and descent."""
        n = 5
        history = []
        for i in range(3):
            key = jax.random.PRNGKey(i)
            s = jax.random.normal(key, (n,))
            y = jax.random.normal(jax.random.PRNGKey(i + 100), (n,)) + 0.1 * s
            sy = float(jnp.dot(s, y))
            if sy > 0:
                history.append((s, y, 1.0 / sy))
        grad = jax.random.normal(jax.random.PRNGKey(42), (n,))

        def h0_matvec(v):
            return v

        result = lbfgs_two_loop(grad, history, h0_matvec)
        assert jnp.all(jnp.isfinite(result))
