"""Tests for regularized SVD, regularized eigh, and differentiable projectors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


class TestRegularizedSVD:
    def test_forward_matches_jnp_svd(self):
        from tenax.algorithms.ad_utils import regularized_svd

        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))
        U, s, Vh = regularized_svd(M)
        U_ref, s_ref, Vh_ref = jnp.linalg.svd(M, full_matrices=False)
        # Singular values should match exactly
        assert jnp.allclose(s, s_ref, atol=1e-5)
        # U columns may differ by sign — check U @ diag(s) @ Vh reconstructs M
        M_recon = U @ jnp.diag(s) @ Vh
        assert jnp.allclose(M_recon, M, atol=1e-5)

    def test_gradient_finite(self):
        from tenax.algorithms.ad_utils import regularized_svd

        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))

        def loss(M):
            U, s, Vh = regularized_svd(M)
            return jnp.sum(s[:3] ** 2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_with_degeneracy(self):
        from tenax.algorithms.ad_utils import regularized_svd

        # Near-degenerate singular values: standard SVD backward would
        # produce NaN from 1/(s_i^2 - s_j^2), but Lorentzian broadening
        # keeps it finite.
        key = jax.random.PRNGKey(42)
        M = jax.random.normal(key, (6, 4))
        # Make two pairs of nearly degenerate singular values
        U0, s0, Vh0 = jnp.linalg.svd(M, full_matrices=False)
        s_degen = jnp.array([2.0, 2.0 + 1e-14, 1.0, 1.0 + 1e-14])
        M_degen = U0 @ jnp.diag(s_degen) @ Vh0

        def loss(M):
            U, s, Vh = regularized_svd(M)
            return jnp.sum(s**2)

        grad = jax.grad(loss)(M_degen)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 0


class TestRegularizedEigh:
    def test_forward_matches_jnp_eigh(self):
        from tenax.algorithms.ad_utils import regularized_eigh

        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (6, 6))
        M = A @ A.T  # PSD matrix
        eigvals, eigvecs = regularized_eigh(M)
        eigvals_ref, eigvecs_ref = jnp.linalg.eigh(M)
        assert jnp.allclose(eigvals, eigvals_ref, atol=1e-5)
        # Eigenvectors may differ by sign; check reconstruction
        M_recon = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
        assert jnp.allclose(M_recon, M, atol=1e-5)

    def test_gradient_finite(self):
        from tenax.algorithms.ad_utils import regularized_eigh

        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (6, 6))
        M = A @ A.T

        def loss(M):
            eigvals, eigvecs = regularized_eigh(M)
            return jnp.sum(eigvals[-3:] ** 2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_with_degeneracy(self):
        from tenax.algorithms.ad_utils import regularized_eigh

        # Degenerate eigenvalues: standard eigh backward produces NaN
        key = jax.random.PRNGKey(42)
        Q, _ = jnp.linalg.qr(jax.random.normal(key, (6, 6)))
        eigvals_exact = jnp.array([3.0, 3.0, 2.0, 2.0, 1.0, 1.0])
        M = Q @ jnp.diag(eigvals_exact) @ Q.T
        M = 0.5 * (M + M.T)

        def loss(M):
            eigvals, eigvecs = regularized_eigh(M)
            return jnp.sum(eigvals**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 0


def _make_grown_corner(key, fused_dim, col_dim, label_col="col"):
    """Helper: create a DenseTensor grown corner (fused, col)."""
    data = jax.random.normal(key, (fused_dim, col_dim))
    sym = U1Symmetry()
    fused_idx = TensorIndex.from_charges(
        sym, np.zeros(fused_dim, dtype=np.int32), FlowDirection.IN, label="fused"
    )
    col_idx = TensorIndex.from_charges(
        sym, np.zeros(col_dim, dtype=np.int32), FlowDirection.OUT, label=label_col
    )
    return DenseTensor(data, (fused_idx, col_idx))


class TestDifferentiableProjectorQR:
    def test_qr_projector_gradient_flows(self):
        from tenax.algorithms._ctm_projector import _compute_projector_tensor

        chi, fused_dim = 4, 16
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        C1g = _make_grown_corner(k1, fused_dim, chi, "col1")
        C4g = _make_grown_corner(k2, fused_dim, chi, "col2")

        def loss(data1, data2):
            C1 = DenseTensor(data1, C1g.indices)
            C4 = DenseTensor(data2, C4g.indices)
            P = _compute_projector_tensor(C1, C4, chi, projector_method="qr")
            return jnp.sum(P.todense() ** 2)

        g1, g2 = jax.grad(loss, argnums=(0, 1))(C1g.todense(), C4g.todense())
        assert jnp.all(jnp.isfinite(g1))
        assert jnp.all(jnp.isfinite(g2))
        assert float(jnp.sum(jnp.abs(g1))) > 0


class TestDifferentiableProjectorEigh:
    def test_eigh_projector_gradient_flows(self):
        from tenax.algorithms._ctm_projector import _compute_projector_tensor

        chi, fused_dim = 4, 16
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        C1g = _make_grown_corner(k1, fused_dim, chi, "col1")
        C4g = _make_grown_corner(k2, fused_dim, chi, "col2")

        def loss(data1, data2):
            C1 = DenseTensor(data1, C1g.indices)
            C4 = DenseTensor(data2, C4g.indices)
            P = _compute_projector_tensor(C1, C4, chi, projector_method="eigh")
            return jnp.sum(P.todense() ** 2)

        g1, g2 = jax.grad(loss, argnums=(0, 1))(C1g.todense(), C4g.todense())
        assert jnp.all(jnp.isfinite(g1))
        assert jnp.all(jnp.isfinite(g2))
        assert float(jnp.sum(jnp.abs(g1))) > 0


def _make_random_dense_site_tensor(key, D=2, d=2):
    """Build a random DenseTensor site tensor with 5 legs (u, d, l, r, phys)."""
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    data = jax.random.normal(key, shape=(D, D, D, D, d))
    return DenseTensor(data, indices)


class TestExplicitADWarmup:
    def test_warmup_runs_without_error(self):
        """Explicit AD with warmup steps should run without error."""
        from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
        from tenax.algorithms.ad_utils import (
            _config_to_tuple,
            ctm_tensor_converge_explicit,
        )
        from tenax.algorithms.ipeps_config import CTMConfig

        key = jax.random.PRNGKey(0)
        A = _make_random_dense_site_tensor(key, D=2, d=2)
        site_tensors = {(0, 0): A}
        config = CTMConfig(chi=4, max_iter=5)
        config_tuple = _config_to_tuple(config)

        env_leaves = ctm_tensor_converge_explicit(
            site_tensors,
            None,
            SINGLE_SITE_NEIGHBORS,
            config_tuple,
            num_steps=3,
            warmup_steps=2,
        )
        assert all(jnp.all(jnp.isfinite(x)) for x in env_leaves)

    def test_warmup_detaches_gradient(self):
        """Warmup steps should not contribute to gradients (stop_gradient)."""
        from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
        from tenax.algorithms.ad_utils import (
            _config_to_tuple,
            _flatten_envs,
            ctm_tensor_converge_explicit,
        )
        from tenax.algorithms.ipeps_config import CTMConfig

        key = jax.random.PRNGKey(42)
        A = _make_random_dense_site_tensor(key, D=2, d=2)
        config = CTMConfig(chi=4, max_iter=5)
        config_tuple = _config_to_tuple(config)

        def loss_fn(data):
            A_inner = DenseTensor(data, A.indices)
            site_tensors = {(0, 0): A_inner}
            env_leaves = ctm_tensor_converge_explicit(
                site_tensors,
                None,
                SINGLE_SITE_NEIGHBORS,
                config_tuple,
                num_steps=3,
                warmup_steps=2,
            )
            return sum(jnp.sum(x**2) for x in env_leaves)

        grad = jax.grad(loss_fn)(A.todense())
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 0


class TestSplitCTMExplicitAD:
    def test_split_ctm_explicit_runs(self):
        """Split CTM explicit AD should run without error."""
        from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv
        from tenax.algorithms.ad_utils import ctm_split_tensor_converge_explicit

        key = jax.random.PRNGKey(0)
        A = _make_random_dense_site_tensor(key, D=2, d=2)
        env = ctm_split_tensor_converge_explicit(
            A,
            chi=4,
            num_steps=3,
            warmup_steps=2,
        )
        assert isinstance(env, SplitCTMTensorEnv)
        leaves = jax.tree.leaves(env)
        assert all(jnp.all(jnp.isfinite(x)) for x in leaves)

    @pytest.mark.skip(
        reason="Split CTM uses dynamic SVD truncation (np.array on traced values) "
        "which is incompatible with JAX AD tracing. Gradient support requires "
        "refactoring tenax.linalg.svd to use static truncation bounds."
    )
    def test_split_ctm_explicit_gradient_finite(self):
        """Split CTM explicit AD should produce finite gradients."""
        from tenax.algorithms.ad_utils import ctm_split_tensor_converge_explicit

        key = jax.random.PRNGKey(42)
        A = _make_random_dense_site_tensor(key, D=2, d=2)

        def loss_fn(data):
            A_inner = DenseTensor(data, A.indices)
            env = ctm_split_tensor_converge_explicit(
                A_inner,
                chi=4,
                num_steps=3,
                warmup_steps=2,
            )
            # Loss = sum of squared corner norms
            return sum(jnp.sum(x**2) for x in jax.tree.leaves(env))

        grad = jax.grad(loss_fn)(A.todense())
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 0
