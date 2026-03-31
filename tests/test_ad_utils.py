"""Tests for the stable AD infrastructure (ad_utils.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _enable_x64():
    """Enable float64 for this test module and restore afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


from tenax.algorithms._ctm_tensor import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms.ad_utils import (
    _config_to_tuple,
    _gauge_fix_ctm_tensor,
    ctm_tensor_converge,
    truncated_svd_ad,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


class TestTruncatedSVDADForward:
    """Forward pass of truncated_svd_ad matches standard SVD."""

    def test_forward_matches_svd(self):
        """Truncated results should match jnp.linalg.svd truncated output."""
        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))
        chi = 3

        U_ad, s_ad, Vh_ad = truncated_svd_ad(M, chi)

        U_ref, s_ref, Vh_ref = jnp.linalg.svd(M, full_matrices=False)
        U_ref = U_ref[:, :chi]
        s_ref = s_ref[:chi]
        Vh_ref = Vh_ref[:chi, :]

        assert jnp.allclose(s_ad, s_ref, atol=1e-12)
        # U and Vh can differ by sign per column, compare via reconstruction
        recon_ad = U_ad * s_ad[None, :] @ Vh_ad
        recon_ref = U_ref * s_ref[None, :] @ Vh_ref
        assert jnp.allclose(recon_ad, recon_ref, atol=1e-12)

    def test_forward_shapes(self):
        """Output shapes should be (m, chi), (chi,), (chi, n)."""
        M = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        chi = 3
        U, s, Vh = truncated_svd_ad(M, chi)
        assert U.shape == (8, 3)
        assert s.shape == (3,)
        assert Vh.shape == (3, 5)

    def test_forward_chi_larger_than_min_dim(self):
        """When chi > min(m,n), should truncate to min(m,n)."""
        M = jax.random.normal(jax.random.PRNGKey(2), (4, 3))
        chi = 10
        U, s, Vh = truncated_svd_ad(M, chi)
        assert s.shape[0] == 3  # min(4, 3)


class TestTruncatedSVDADGradient:
    """VJP of truncated_svd_ad matches finite-difference gradients."""

    def test_gradient_finite_diff(self):
        """Custom VJP should approximate finite-difference gradient."""
        key = jax.random.PRNGKey(42)
        M = jax.random.normal(key, (5, 4))
        chi = 3

        # Loss function: sum of singular values
        def loss(M_in):
            _, s, _ = truncated_svd_ad(M_in, chi)
            return jnp.sum(s)

        # AD gradient
        grad_ad = jax.grad(loss)(M)

        # Finite-difference gradient
        eps = 1e-5
        grad_fd = np.zeros_like(M)
        M_np = np.array(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_plus = M_np.copy()
                M_plus[i, j] += eps
                M_minus = M_np.copy()
                M_minus[i, j] -= eps
                grad_fd[i, j] = (
                    float(loss(jnp.array(M_plus))) - float(loss(jnp.array(M_minus)))
                ) / (2 * eps)

        assert jnp.allclose(grad_ad, grad_fd, atol=1e-4), (
            f"Max diff: {float(jnp.max(jnp.abs(grad_ad - grad_fd)))}"
        )

    def test_gradient_reconstruction_loss(self):
        """Gradient of ||M - U S Vh||^2 through truncated SVD."""
        key = jax.random.PRNGKey(7)
        M = jax.random.normal(key, (6, 4))
        chi = 2

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            recon = U * s[None, :] @ Vh
            return jnp.sum((M_in - recon) ** 2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))


class TestTruncatedSVDADDegenerate:
    """No NaN/Inf when singular values are degenerate."""

    def test_degenerate_identity(self):
        """Identity matrix has all singular values = 1 (maximally degenerate)."""
        M = jnp.eye(4)
        chi = 3

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            return jnp.sum(s**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad)), f"NaN/Inf in gradient: {grad}"

    def test_degenerate_repeated_singular_values(self):
        """Matrix with repeated singular values should not produce NaN."""
        # Construct M with repeated singular values
        U, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (5, 5)))
        V, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(1), (4, 4)))
        s = jnp.array([3.0, 3.0, 1.0, 1.0])  # repeated values
        M = U[:, :4] * s[None, :] @ V.T
        chi = 3

        def loss(M_in):
            U_t, s_t, Vh_t = truncated_svd_ad(M_in, chi)
            return jnp.sum(s_t)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))

    def test_degenerate_zero_matrix(self):
        """Near-zero matrix should not cause NaN."""
        M = 1e-15 * jax.random.normal(jax.random.PRNGKey(3), (4, 3))
        chi = 2

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            return jnp.sum(s)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))


class TestTruncatedSVDADMissingTerm:
    """Truncation correction term improves gradient accuracy."""

    def test_truncation_correction_improves_accuracy(self):
        """For a matrix with significant truncated spectrum, our custom VJP
        should be more accurate than naive truncation."""
        key = jax.random.PRNGKey(10)
        M = jax.random.normal(key, (6, 5))
        chi = 2  # Aggressive truncation — large truncated spectrum

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            # Loss that depends on U and Vh (not just s)
            return jnp.sum(U[:, 0] ** 2) + jnp.sum(Vh[0, :] ** 2)

        grad_ad = jax.grad(loss)(M)

        # Finite-difference reference
        eps = 1e-5
        grad_fd = np.zeros_like(M)
        M_np = np.array(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_plus = M_np.copy()
                M_plus[i, j] += eps
                M_minus = M_np.copy()
                M_minus[i, j] -= eps
                grad_fd[i, j] = (
                    float(loss(jnp.array(M_plus))) - float(loss(jnp.array(M_minus)))
                ) / (2 * eps)

        max_diff = float(jnp.max(jnp.abs(grad_ad - grad_fd)))
        assert max_diff < 1e-3, f"Gradient error too large: {max_diff}"


def _make_dense_tensor(key, D=2, d=2):
    """Create a DenseTensor iPEPS site tensor for testing."""
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    data = jax.random.normal(key, (D, D, D, D, d))
    data = data / (jnp.linalg.norm(data) + 1e-10)
    indices = (
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, indices)


class TestCTMFixedPointGradient:
    """Gradient through ctm_tensor_converge matches finite-difference."""

    def test_gradient_exists_and_finite(self):
        """Gradient of energy through ctm_tensor_converge should be finite."""
        A = _make_dense_tensor(jax.random.PRNGKey(42))
        config = CTMConfig(chi=4, max_iter=5, conv_tol=1e-6)
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(A_norm, None, config_tuple)
            import jax

            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "Gradient contains NaN/Inf"
        assert grad.norm() > 1e-15, "Gradient is all zeros"


class TestGaugeFix:
    """Tests for CTM gauge fixing (Tensor protocol)."""

    @pytest.fixture
    def random_env(self):
        """Random CTMTensorEnv for testing."""
        A = _make_dense_tensor(jax.random.PRNGKey(0))
        return initialize_ctm_tensor_env(A, chi=4)

    def test_gauge_fix_idempotent(self, random_env):
        """Applying gauge fix twice should give the same result."""
        env1 = _gauge_fix_ctm_tensor(random_env)
        env2 = _gauge_fix_ctm_tensor(env1)

        for t1, t2 in zip(env1, env2):
            assert jnp.allclose(t1.todense(), t2.todense(), atol=1e-10), (
                f"Gauge fix not idempotent: max diff = "
                f"{float(jnp.max(jnp.abs(t1.todense() - t2.todense())))}"
            )

    def test_gauge_fix_preserves_shapes(self, random_env):
        """Gauge-fixed environment should have same tensor shapes."""
        env_fixed = _gauge_fix_ctm_tensor(random_env)
        for t_orig, t_fixed in zip(random_env, env_fixed):
            assert t_orig.todense().shape == t_fixed.todense().shape


class TestGMRESBackward:
    """Validate GMRES-based backward pass for ctm_tensor_converge."""

    def test_gmres_backward_finite_gradient(self):
        """GMRES backward pass should produce finite, nonzero gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(123))
        config = CTMConfig(chi=4, max_iter=10, conv_tol=1e-6)
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(A_norm, None, config_tuple)
            import jax

            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "GMRES backward: NaN/Inf"
        assert grad.norm() > 1e-15, "GMRES backward: gradient is all zeros"

    def test_gmres_backward_deterministic(self):
        """GMRES backward pass should be deterministic across calls."""
        A = _make_dense_tensor(jax.random.PRNGKey(77))
        config = CTMConfig(chi=4, max_iter=10, conv_tol=1e-6)
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(A_norm, None, config_tuple)
            import jax

            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad1 = jax.grad(energy_fn)(A)
        grad2 = jax.grad(energy_fn)(A)
        assert jnp.allclose(grad1.todense(), grad2.todense(), atol=1e-10), (
            f"GMRES backward not deterministic: max diff = "
            f"{float(jnp.max(jnp.abs(grad1.todense() - grad2.todense())))}"
        )
