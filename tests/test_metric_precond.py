"""Tests for metric preconditioning."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_random_ipeps_tensor(D=2, d=2, seed=0):
    """Create a random DenseTensor iPEPS site tensor (D,D,D,D,d)."""
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, d))
    sym = U1Symmetry()
    charges = [0] * D
    phys_charges = [0] * d
    indices = (
        TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, phys_charges, FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, indices)


def _converge_ctm(A, chi=4, max_iter=50):
    """Run CTM to convergence and return the environment."""
    env, _ = ctm_tensor(A, chi=chi, max_iter=max_iter, conv_tol=1e-10)
    return env


class TestContractSingleSiteEnv:
    """Tests for _contract_single_site_environment."""

    def test_output_shape(self):
        """E should have shape (D², D², D², D²)."""
        from tenax.algorithms._metric_precond import _contract_single_site_environment

        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        A = A * (1.0 / (A.norm() + 1e-10))
        env = _converge_ctm(A)
        E = _contract_single_site_environment(env)
        assert E.shape == (D * D, D * D, D * D, D * D)

    def test_reproduces_norm(self):
        """Contracting E with double-layer should reproduce <psi|psi>."""
        from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
        from tenax.algorithms._metric_precond import _contract_single_site_environment

        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        A = A * (1.0 / (A.norm() + 1e-10))
        env = _converge_ctm(A)
        E = _contract_single_site_environment(env)
        a = _build_double_layer_tensor(A).todense()
        # <psi|psi> = einsum(E, a) over all legs
        norm_val = jnp.einsum("ijkl,ijkl->", E, a)
        assert jnp.isfinite(norm_val)
        assert float(norm_val.real) > 0


class TestNormEnvironmentMV:
    """Tests for norm_environment_matvec."""

    def test_output_shape(self):
        """N·v should return a dense array with the same shape as A."""
        from tenax.algorithms._metric_precond import norm_environment_matvec

        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        A = A * (1.0 / (A.norm() + 1e-10))
        env = _converge_ctm(A)
        v = _make_random_ipeps_tensor(D=D, d=d, seed=42)
        Nv = norm_environment_matvec(A, env, v)
        assert Nv.shape == (D, D, D, D, d)

    def test_positive_definite(self):
        """<v|N|v> should be positive for any nonzero v."""
        from tenax.algorithms._metric_precond import norm_environment_matvec

        A = _make_random_ipeps_tensor(D=2, d=2)
        A = A * (1.0 / (A.norm() + 1e-10))
        env = _converge_ctm(A)
        v = _make_random_ipeps_tensor(D=2, d=2, seed=42)
        Nv = norm_environment_matvec(A, env, v)
        vNv = jnp.sum(jnp.conj(v.todense()) * Nv).real
        assert float(vNv) > 0.0

    def test_hermitian(self):
        """<u|N|v> should equal conj(<v|N|u>)."""
        from tenax.algorithms._metric_precond import norm_environment_matvec

        A = _make_random_ipeps_tensor(D=2, d=2)
        A = A * (1.0 / (A.norm() + 1e-10))
        env = _converge_ctm(A)
        u = _make_random_ipeps_tensor(D=2, d=2, seed=1)
        v = _make_random_ipeps_tensor(D=2, d=2, seed=2)
        Nu = norm_environment_matvec(A, env, u)
        Nv = norm_environment_matvec(A, env, v)
        uNv = jnp.sum(jnp.conj(u.todense()) * Nv)
        vNu = jnp.sum(jnp.conj(v.todense()) * Nu)
        assert jnp.allclose(uNv, jnp.conj(vNu), atol=1e-6)


class TestPreconditionGradient:
    """Tests for precondition_gradient."""

    def test_finite_output(self):
        """Preconditioned gradient should be finite and nonzero."""
        from tenax.algorithms._metric_precond import precondition_gradient
        from tenax.algorithms.ipeps_config import iPEPSConfig

        A = _make_random_ipeps_tensor(D=2, d=2)
        A = A * (1.0 / (A.norm() + 1e-10))
        env = _converge_ctm(A)
        grad = _make_random_ipeps_tensor(D=2, d=2, seed=99)
        config = iPEPSConfig(gs_metric_precond=True)
        delta = 0.01
        g_precond = precondition_gradient(A, env, grad, delta, config)
        assert g_precond.shape == (2, 2, 2, 2, 2)
        assert jnp.all(jnp.isfinite(g_precond))
        assert float(jnp.sum(jnp.abs(g_precond))) > 0

    def test_identity_at_large_delta(self):
        """With very large delta, (N + delta*I)^{-1} ~ (1/delta)*I."""
        from tenax.algorithms._metric_precond import precondition_gradient
        from tenax.algorithms.ipeps_config import iPEPSConfig

        A = _make_random_ipeps_tensor(D=2, d=2)
        A = A * (1.0 / (A.norm() + 1e-10))
        env = _converge_ctm(A)
        grad = _make_random_ipeps_tensor(D=2, d=2, seed=99)
        config = iPEPSConfig(gs_metric_precond=True, metric_gmres_tol=1e-6)
        delta = 1e6
        g_precond = precondition_gradient(A, env, grad, delta, config)
        # Should approximately equal grad / delta
        expected = grad.todense() / delta
        assert jnp.allclose(g_precond, expected, rtol=0.1)


class TestLBFGSTwoLoop:
    """Tests for lbfgs_two_loop."""

    def test_empty_history_returns_h0_grad(self):
        """With no history, result = h0_matvec(grad)."""
        from tenax.algorithms._metric_precond import lbfgs_two_loop

        grad = jnp.array([1.0, 2.0, 3.0])

        def h0_matvec(v):
            return 0.5 * v

        result = lbfgs_two_loop(grad, [], h0_matvec)
        assert jnp.allclose(result, 0.5 * grad)

    def test_one_step_history(self):
        """With one (s, y) pair, verify descent direction."""
        from tenax.algorithms._metric_precond import lbfgs_two_loop

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
        # Verify descent direction: result . grad > 0
        assert float(jnp.dot(result, grad)) > 0
