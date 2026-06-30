"""Tests for regularized SVD, explicit AD warmup, and split CTM explicit AD."""

import jax
import jax.numpy as jnp
import pytest


class TestRegularizedSVD:
    def test_forward_matches_jnp_svd(self):
        """Forward pass should match jnp.linalg.svd."""
        from tenax.algorithms.ad_utils import regularized_svd

        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))
        U, s, Vh = regularized_svd(M)
        U_ref, s_ref, Vh_ref = jnp.linalg.svd(M, full_matrices=False)
        assert jnp.allclose(jnp.abs(U), jnp.abs(U_ref), atol=1e-5)
        assert jnp.allclose(s, s_ref, atol=1e-5)

    def test_gradient_finite(self):
        """Gradient through regularized_svd should be finite."""
        from tenax.algorithms.ad_utils import regularized_svd

        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))

        def loss(M):
            U, s, Vh = regularized_svd(M)
            return jnp.sum(s[:3] ** 2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_with_degeneracy(self):
        """Gradient should stay finite even with degenerate singular values."""
        from tenax.algorithms.ad_utils import regularized_svd

        # Build a matrix with degenerate singular values but non-trivial
        # structure so that gradient is nonzero.
        key = jax.random.PRNGKey(42)
        Q1, _ = jnp.linalg.qr(jax.random.normal(key, (6, 6)))
        key2 = jax.random.PRNGKey(7)
        Q2, _ = jnp.linalg.qr(jax.random.normal(key2, (4, 4)))
        s = jnp.array([2.0, 2.0, 1.0, 1.0])
        M = Q1[:, :4] @ jnp.diag(s) @ Q2

        def loss(M):
            U, s, Vh = regularized_svd(M)
            return jnp.sum(s**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 0


class TestExplicitADWarmup:
    def test_explicit_ad_warmup_forward_runs(self):
        """Explicit AD with warmup should run without error in forward mode."""
        from tenax.algorithms._ctm_tensor import (
            _build_double_layer_tensor,
            initialize_ctm_tensor_env,
        )
        from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
        from tenax.algorithms.ad_utils import (
            _config_to_tuple,
            _flatten_envs,
            ctm_tensor_converge_explicit,
        )
        from tenax.algorithms.ipeps_config import CTMConfig
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import DenseTensor

        key = jax.random.PRNGKey(0)
        sym = U1Symmetry()
        D, d = 2, 2
        A_data = jax.random.normal(key, (D, D, D, D, d))
        charges = jnp.zeros(D, dtype=jnp.int32)
        phys_charges = jnp.zeros(d, dtype=jnp.int32)
        indices = tuple(
            TensorIndex.from_charges(sym, charges, flow, label=lbl)
            for lbl, flow in [
                ("d", FlowDirection.IN),
                ("u", FlowDirection.OUT),
                ("r", FlowDirection.IN),
                ("l", FlowDirection.OUT),
            ]
        ) + (
            TensorIndex.from_charges(sym, phys_charges, FlowDirection.IN, label="phys"),
        )
        A = DenseTensor(A_data, indices)

        config = CTMConfig(chi=4, max_iter=10)
        config_tuple = _config_to_tuple(config)
        site_tensors = {(0, 0): A}

        env_leaves = ctm_tensor_converge_explicit(
            site_tensors,
            None,
            SINGLE_SITE_NEIGHBORS,
            config_tuple,
            num_steps=3,
            warmup_steps=2,
        )
        assert all(jnp.all(jnp.isfinite(x)) for x in env_leaves)

    def test_explicit_ad_warmup_gradient_finite(self):
        """Explicit AD with warmup should produce finite energy gradients."""
        from tenax.algorithms._ctm_tensor import (
            _build_double_layer_tensor,
            compute_energy_ctm_tensor,
            initialize_ctm_tensor_env,
        )
        from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
        from tenax.algorithms.ad_utils import (
            _config_to_tuple,
            ctm_tensor_converge_explicit,
        )
        from tenax.algorithms.ipeps_config import CTMConfig
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import DenseTensor

        key = jax.random.PRNGKey(0)
        sym = U1Symmetry()
        D, d = 2, 2
        A_data = jax.random.normal(key, (D, D, D, D, d))
        charges = jnp.zeros(D, dtype=jnp.int32)
        phys_charges = jnp.zeros(d, dtype=jnp.int32)
        indices = tuple(
            TensorIndex.from_charges(sym, charges, flow, label=lbl)
            for lbl, flow in [
                ("d", FlowDirection.IN),
                ("u", FlowDirection.OUT),
                ("r", FlowDirection.IN),
                ("l", FlowDirection.OUT),
            ]
        ) + (
            TensorIndex.from_charges(sym, phys_charges, FlowDirection.IN, label="phys"),
        )
        A = DenseTensor(A_data, indices)

        config = CTMConfig(chi=4, max_iter=10)
        config_tuple = _config_to_tuple(config)

        # Heisenberg gate
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        gate = H.reshape(d, d, d, d)

        env_treedef = jax.tree.structure(initialize_ctm_tensor_env(A, config.chi))

        def loss(A_data):
            A_t = DenseTensor(A_data, indices)
            A_norm = A_t * (1.0 / (A_t.norm() + 1e-10))
            site_tensors = {(0, 0): A_norm}
            env_leaves = ctm_tensor_converge_explicit(
                site_tensors,
                None,
                SINGLE_SITE_NEIGHBORS,
                config_tuple,
                num_steps=3,
                warmup_steps=2,
            )
            env = jax.tree.unflatten(env_treedef, env_leaves)
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(loss)(A_data)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 0


class TestSplitCTMExplicitAD:
    def test_split_ctm_explicit_forward_runs(self):
        """Split CTM explicit AD should run forward without error."""
        from tenax.algorithms.ad_utils import ctm_split_tensor_converge_explicit
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import DenseTensor

        key = jax.random.PRNGKey(0)
        sym = U1Symmetry()
        D, d = 2, 2
        A_data = jax.random.normal(key, (D, D, D, D, d))
        charges = jnp.zeros(D, dtype=jnp.int32)
        phys_charges = jnp.zeros(d, dtype=jnp.int32)
        indices = tuple(
            TensorIndex.from_charges(sym, charges, flow, label=lbl)
            for lbl, flow in [
                ("d", FlowDirection.IN),
                ("u", FlowDirection.OUT),
                ("r", FlowDirection.IN),
                ("l", FlowDirection.OUT),
            ]
        ) + (
            TensorIndex.from_charges(sym, phys_charges, FlowDirection.IN, label="phys"),
        )
        A = DenseTensor(A_data, indices)

        env = ctm_split_tensor_converge_explicit(
            A,
            chi=4,
            num_steps=3,
            warmup_steps=2,
        )
        leaves = jax.tree.leaves(env)
        assert all(jnp.all(jnp.isfinite(x)) for x in leaves)

    def test_split_ctm_explicit_gradient_finite(self):
        """Split CTM explicit AD should produce finite gradients."""
        from tenax.algorithms.ad_utils import ctm_split_tensor_converge_explicit
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import DenseTensor

        key = jax.random.PRNGKey(0)
        sym = U1Symmetry()
        D, d = 2, 2
        A_data = jax.random.normal(key, (D, D, D, D, d))
        charges = jnp.zeros(D, dtype=jnp.int32)
        phys_charges = jnp.zeros(d, dtype=jnp.int32)
        indices = tuple(
            TensorIndex.from_charges(sym, charges, flow, label=lbl)
            for lbl, flow in [
                ("d", FlowDirection.IN),
                ("u", FlowDirection.OUT),
                ("r", FlowDirection.IN),
                ("l", FlowDirection.OUT),
            ]
        ) + (
            TensorIndex.from_charges(sym, phys_charges, FlowDirection.IN, label="phys"),
        )

        def loss(A_data):
            A_t = DenseTensor(A_data, indices)
            env = ctm_split_tensor_converge_explicit(
                A_t,
                chi=4,
                num_steps=3,
                warmup_steps=2,
            )
            # NOTE: do *not* use ``sum(C1**2)`` here. The #463 double-layer
            # split path renormalizes each corner per move (max_abs_normalize),
            # and the boundary corner of a uniform 1-site iPEPS is genuinely
            # rank-1, so the normalized C1 collapses to a constant one-hot
            # matrix that is identically independent of A — its gradient is
            # exactly zero. The A dependence lives in the edge tensors, so sum
            # over the full environment to exercise the explicit-AD gradient.
            env_tensors = (
                env.C1,
                env.C2,
                env.C3,
                env.C4,
                env.T1_ket,
                env.T2_ket,
                env.T3_ket,
                env.T4_ket,
                env.T1_bra,
                env.T2_bra,
                env.T3_bra,
                env.T4_bra,
            )
            return sum(jnp.sum(t.todense() ** 2) for t in env_tensors)

        grad = jax.grad(loss)(A_data)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 0
