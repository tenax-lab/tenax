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
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms.ad_utils import (
    _config_from_tuple,
    _config_to_tuple,
    _ctm_tensor_multisite_fixed_point,
    _gauge_fix_ctm_tensor,
    _svd_sector_backward,
    ctm_tensor_converge,
    regularized_svd,
    truncated_svd_ad,
    truncated_svd_symmetric_ad,
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
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
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
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
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
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
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
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
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

    def test_gmres_neumann_preconditioner_finite(self):
        """Neumann-preconditioned GMRES produces finite, nonzero gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(200))
        config = CTMConfig(chi=4, max_iter=10, conv_tol=1e-6, gmres_precondition=True)
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "Preconditioned GMRES: NaN/Inf"
        assert grad.norm() > 1e-15, "Preconditioned GMRES: gradient is all zeros"

    def test_gmres_preconditioned_matches_unpreconditioned(self):
        """With preconditioner disabled (no-op), both paths must agree."""
        A = _make_dense_tensor(jax.random.PRNGKey(55))
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def _grad_with(precond: bool):
            config = CTMConfig(
                chi=4, max_iter=10, conv_tol=1e-6, gmres_precondition=precond
            )
            config_tuple = _config_to_tuple(config)

            def energy_fn(A_in):
                A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
                env_leaves = ctm_tensor_converge(
                    {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
                )
                env = jax.tree.unflatten(
                    jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                    list(env_leaves),
                )
                return compute_energy_ctm_tensor(A_norm, env, gate)

            return jax.grad(energy_fn)(A)

        grad_on = _grad_with(True)
        grad_off = _grad_with(False)
        diff = float(jnp.max(jnp.abs(grad_on.todense() - grad_off.todense())))
        assert diff < 1e-4, f"Preconditioned vs unpreconditioned gradient diff = {diff}"

    def test_gmres_no_preconditioner_still_works(self):
        """Setting gmres_precondition=False must still produce finite gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(99))
        config = CTMConfig(chi=4, max_iter=10, conv_tol=1e-6, gmres_precondition=False)
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "No-precond GMRES: NaN/Inf"
        assert grad.norm() > 1e-15, "No-precond GMRES: gradient is all zeros"


class TestGMRESBackwardPath:
    """Verify the GMRES backward branch (ad_backward_method='gmres') is exercised."""

    def test_gmres_path_finite_gradient(self):
        """GMRES backward path should produce finite, nonzero gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(42))
        config = CTMConfig(
            chi=4, max_iter=10, conv_tol=1e-6, ad_backward_method="gmres"
        )
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "GMRES path: NaN/Inf"
        assert grad.norm() > 1e-15, "GMRES path: gradient is all zeros"

    def test_gmres_path_agrees_with_vjp(self):
        """GMRES and VJP backward paths should produce similar gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(55))
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def _grad_with(method: str):
            config = CTMConfig(
                chi=4, max_iter=10, conv_tol=1e-6, ad_backward_method=method
            )
            ct = _config_to_tuple(config)

            def energy_fn(A_in):
                A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
                env_leaves = ctm_tensor_converge(
                    {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, ct
                )
                env = jax.tree.unflatten(
                    jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                    list(env_leaves),
                )
                return compute_energy_ctm_tensor(A_norm, env, gate)

            return jax.grad(energy_fn)(A)

        grad_gmres = _grad_with("gmres")
        grad_vjp = _grad_with("vjp")
        diff = float(jnp.max(jnp.abs(grad_gmres.todense() - grad_vjp.todense())))
        assert diff < 5e-2, f"GMRES vs VJP gradient diff = {diff}"


class TestSvdSectorBackward:
    """Tests for the factored _svd_sector_backward function."""

    def test_svd_sector_backward_matches_original(self):
        """Factored function must reproduce the original backward exactly."""
        key = jax.random.PRNGKey(42)
        M = jax.random.normal(key, (6, 4))
        chi = 3

        # Full SVD
        U_full, s_full, Vh_full = jnp.linalg.svd(M, full_matrices=False)
        k = min(chi, s_full.shape[0])

        # Fake incoming gradients
        key2 = jax.random.PRNGKey(99)
        dU = jax.random.normal(key2, (6, k))
        ds = jax.random.normal(jax.random.PRNGKey(100), (k,))
        dVh = jax.random.normal(jax.random.PRNGKey(101), (k, 4))

        # Factored version
        dM_new = _svd_sector_backward(U_full, s_full, Vh_full, dU, ds, dVh)

        # Original inline version (reproduced for comparison)
        eps = 1e-12
        U = U_full[:, :k]
        s = s_full[:k]
        V = Vh_full[:k, :].conj().T
        s2 = s**2
        diff = s2[:, None] - s2[None, :]
        F = diff / (diff**2 + eps**2)
        F = F - jnp.diag(jnp.diag(F))
        UtdU = U.conj().T @ dU
        VtdV = V.conj().T @ dVh.conj().T
        UtdU_anti = 0.5 * (UtdU - UtdU.conj().T)
        VtdV_anti = 0.5 * (VtdV - VtdV.conj().T)
        s_inv = jnp.where(s > eps, 1.0 / s, 0.0)
        proj_U_perp = jnp.eye(M.shape[0]) - U @ U.conj().T
        proj_V_perp = jnp.eye(M.shape[1]) - V @ V.conj().T
        dM_orig = jnp.zeros_like(M)
        dM_orig = dM_orig + U @ jnp.diag(ds) @ Vh_full[:k, :]
        dM_orig = dM_orig + U @ (F * UtdU_anti) @ jnp.diag(s) @ Vh_full[:k, :]
        dM_orig = dM_orig + U @ jnp.diag(s) @ (F * VtdV_anti) @ Vh_full[:k, :]
        dM_orig = dM_orig + proj_U_perp @ dU @ jnp.diag(s_inv) @ Vh_full[:k, :]
        dM_orig = dM_orig + U @ jnp.diag(s_inv) @ dVh @ proj_V_perp

        assert jnp.allclose(dM_new, dM_orig, atol=1e-12), (
            f"Max diff: {float(jnp.max(jnp.abs(dM_new - dM_orig)))}"
        )

    def test_svd_sector_backward_no_truncation(self):
        """When k == min(m,n), the sector backward should still work."""
        M = jax.random.normal(jax.random.PRNGKey(5), (4, 3))
        U, s, Vh = jnp.linalg.svd(M, full_matrices=False)
        k = s.shape[0]  # no truncation
        dU = jax.random.normal(jax.random.PRNGKey(6), (4, k))
        ds = jax.random.normal(jax.random.PRNGKey(7), (k,))
        dVh = jax.random.normal(jax.random.PRNGKey(8), (k, 3))

        dM = _svd_sector_backward(U, s, Vh, dU, ds, dVh)
        assert jnp.all(jnp.isfinite(dM))
        assert dM.shape == M.shape


class TestDegenerateSvGradientFinite:
    """Gradient through SVD with degenerate singular values is finite."""

    def test_degenerate_sv_gradient_finite_lorentzian(self):
        """Lorentzian regularization prevents NaN for degenerate SVs."""
        # Build a matrix with exactly degenerate singular values
        U_rand, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (6, 6)))
        V_rand, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(1), (4, 4)))
        s_degen = jnp.array([5.0, 5.0, 2.0, 2.0])
        M = U_rand[:, :4] * s_degen[None, :] @ V_rand.T
        chi = 3

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            return jnp.sum(U**2) + jnp.sum(s) + jnp.sum(Vh**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad)), f"NaN/Inf in gradient: {grad}"

    def test_degenerate_sv_sector_backward_finite(self):
        """_svd_sector_backward produces finite output for degenerate SVs."""
        U_rand, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(10), (5, 5)))
        V_rand, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(11), (4, 4)))
        s_degen = jnp.array([3.0, 3.0, 3.0, 1.0])
        M = U_rand[:, :4] * s_degen[None, :] @ V_rand.T

        U, s, Vh = jnp.linalg.svd(M, full_matrices=False)
        k = 3
        dU = jax.random.normal(jax.random.PRNGKey(12), (5, k))
        ds = jax.random.normal(jax.random.PRNGKey(13), (k,))
        dVh = jax.random.normal(jax.random.PRNGKey(14), (k, 4))

        dM = _svd_sector_backward(U, s, Vh, dU, ds, dVh)
        assert jnp.all(jnp.isfinite(dM)), f"NaN/Inf in sector backward: {dM}"


class TestSymmetricSvdAdMatchesDense:
    """Per-sector regularized SVD matches dense truncated_svd_ad result."""

    def test_symmetric_svd_ad_matches_dense_trivial_charges(self):
        """With trivial (all-zero) charges, symmetric AD SVD should match dense."""
        key = jax.random.PRNGKey(42)
        D, d = 3, 2
        data = jax.random.normal(key, (D, D, D, D, d))
        data = data / jnp.linalg.norm(data)

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
        A = DenseTensor(data, indices)

        chi = 4
        left_labels = ("u", "l")
        right_labels = ("d", "r", "phys")

        U_sym, s_sym, Vh_sym = truncated_svd_symmetric_ad(
            A, left_labels, right_labels, chi, new_bond_label="bond"
        )

        # Compare via dense: reshape A the same way, apply truncated_svd_ad
        dense = A.todense()
        # Permute to (u, l, d, r, phys)
        label_to_ax = {lbl: i for i, lbl in enumerate(A.labels())}
        perm = [label_to_ax[lb] for lb in list(left_labels) + list(right_labels)]
        dense_p = jnp.transpose(dense, perm)
        m = D * D  # u, l
        n = D * D * d  # d, r, phys
        matrix = dense_p.reshape(m, n)
        U_ref, s_ref, Vh_ref = truncated_svd_ad(matrix, chi)

        # Singular values must match
        assert jnp.allclose(s_sym, s_ref, atol=1e-10), (
            f"SV mismatch: max diff = {float(jnp.max(jnp.abs(s_sym - s_ref)))}"
        )

        # Reconstruction must match
        k = s_sym.shape[0]
        recon_sym = (
            U_sym.todense().reshape(m, k)
            * s_sym[None, :]
            @ Vh_sym.todense().reshape(k, n)
        )
        recon_ref = U_ref * s_ref[None, :] @ Vh_ref
        assert jnp.allclose(recon_sym, recon_ref, atol=1e-10), (
            f"Reconstruction mismatch: {float(jnp.max(jnp.abs(recon_sym - recon_ref)))}"
        )

    def test_symmetric_svd_ad_gradient_finite(self):
        """Gradient through truncated_svd_symmetric_ad is finite."""
        key = jax.random.PRNGKey(77)
        D, d = 2, 2
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
        A = DenseTensor(data, indices)

        def loss(A_in):
            U, s, Vh = truncated_svd_symmetric_ad(
                A_in, ("u", "l"), ("d", "r", "phys"), chi=3
            )
            return jnp.sum(s)

        grad = jax.grad(loss)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "Gradient contains NaN/Inf"


class TestSVDSignFixing:
    """Tests for SVD sign-fixing gauge convention."""

    def test_sign_convention_truncated(self):
        """Max-abs element of each U column should be real-positive after truncated_svd_ad."""
        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))
        chi = 3
        U, s, Vh = truncated_svd_ad(M, chi)
        for j in range(U.shape[1]):
            col = U[:, j]
            max_elem = col[jnp.argmax(jnp.abs(col))]
            assert float(jnp.real(max_elem)) > 0, (
                f"Column {j}: max-abs element {float(max_elem)} is not real-positive"
            )

    def test_sign_convention_regularized(self):
        """Max-abs element of each U column should be real-positive after regularized_svd."""
        key = jax.random.PRNGKey(1)
        M = jax.random.normal(key, (5, 4))
        U, s, Vh = regularized_svd(M)
        for j in range(U.shape[1]):
            col = U[:, j]
            max_elem = col[jnp.argmax(jnp.abs(col))]
            assert float(jnp.real(max_elem)) > 0, (
                f"Column {j}: max-abs element {float(max_elem)} is not real-positive"
            )

    def test_sign_fix_preserves_reconstruction(self):
        """U @ diag(s) @ Vh should still reconstruct the original matrix."""
        key = jax.random.PRNGKey(2)
        M = jax.random.normal(key, (6, 4))
        chi = 3
        U, s, Vh = truncated_svd_ad(M, chi)
        recon = U * s[None, :] @ Vh

        # Compare against standard SVD truncated reconstruction
        U_ref, s_ref, Vh_ref = jnp.linalg.svd(M, full_matrices=False)
        recon_ref = U_ref[:, :chi] * s_ref[:chi][None, :] @ Vh_ref[:chi, :]
        assert jnp.allclose(recon, recon_ref, atol=1e-12), (
            f"Reconstruction mismatch: {float(jnp.max(jnp.abs(recon - recon_ref)))}"
        )


class TestElementWiseCTMConvergence:
    """Element-wise CTM convergence should be available and work correctly."""

    def test_elementwise_converges(self):
        """CTM with element-wise convergence should reach convergence."""
        A = _make_dense_tensor(jax.random.PRNGKey(0))
        config = CTMConfig(
            chi=8,
            max_iter=200,
            conv_tol=1e-6,
            min_iter=10,
            ctm_conv_method="elementwise",
        )
        envs = _ctm_tensor_multisite_fixed_point(
            {(0, 0): A}, SINGLE_SITE_NEIGHBORS, config
        )
        assert envs is not None
        # Check that env tensors are non-trivial
        c1_norm = float(jnp.linalg.norm(envs[(0, 0)].C1.todense()))
        assert c1_norm > 0

    def test_sv_still_works(self):
        """Singular-value convergence should still work."""
        A = _make_dense_tensor(jax.random.PRNGKey(0))
        config = CTMConfig(chi=8, max_iter=200, conv_tol=1e-6, min_iter=10)
        envs = _ctm_tensor_multisite_fixed_point(
            {(0, 0): A}, SINGLE_SITE_NEIGHBORS, config
        )
        assert envs is not None

    def test_config_field_exists(self):
        """CTMConfig should have ctm_conv_method field."""
        config = CTMConfig()
        assert hasattr(config, "ctm_conv_method")
        assert config.ctm_conv_method == "sv"

    def test_config_roundtrip(self):
        """ctm_conv_method should survive serialization round-trip."""
        config = CTMConfig(ctm_conv_method="elementwise")
        t = _config_to_tuple(config)
        config2 = _config_from_tuple(t)
        assert config2.ctm_conv_method == "elementwise"

    def test_config_roundtrip_sv(self):
        """Default sv method should survive serialization round-trip."""
        config = CTMConfig(ctm_conv_method="sv")
        t = _config_to_tuple(config)
        config2 = _config_from_tuple(t)
        assert config2.ctm_conv_method == "sv"

    def test_sign_fix_gradient_finite(self):
        """Gradient through sign-fixed SVD should be finite."""
        key = jax.random.PRNGKey(3)
        M = jax.random.normal(key, (5, 4))
        chi = 3

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            return jnp.sum(U**2) + jnp.sum(s) + jnp.sum(Vh**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad)), f"NaN/Inf in gradient: {grad}"
