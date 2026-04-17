"""Tests for CTM chi-ramp (intra-convergence chi ramping)."""

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


from tenax.algorithms._ctm_tensor import initialize_ctm_tensor_env
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms.ad_utils import (
    _config_from_tuple,
    _config_to_tuple,
    _ctm_tensor_multisite_fixed_point,
    ctm_tensor_converge,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry


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


# ---- Unit tests for the CTMConfig field ----


class TestChiRampConfig:
    def test_chi_ramp_field_defaults_to_none(self):
        config = CTMConfig()
        assert config.chi_ramp is None

    def test_chi_ramp_field_accepts_schedule(self):
        config = CTMConfig(chi_ramp=[(8, 10), (16, 20)])
        assert config.chi_ramp == [(8, 10), (16, 20)]

    def test_chi_ramp_survives_config_roundtrip(self):
        config = CTMConfig(chi=16, chi_ramp=[(4, 5), (8, 10), (16, None)])
        t = _config_to_tuple(config)
        restored = _config_from_tuple(t)
        assert restored.chi == 16
        assert len(restored.chi_ramp) == 3
        assert restored.chi_ramp[0] == (4, 5)
        assert restored.chi_ramp[1] == (8, 10)
        assert restored.chi_ramp[2] == (16, None)

    def test_chi_ramp_none_survives_roundtrip(self):
        config = CTMConfig(chi=8)
        t = _config_to_tuple(config)
        restored = _config_from_tuple(t)
        assert restored.chi_ramp is None


# ---- Functional tests ----


class TestChiRampFunctional:
    def test_chi_ramp_converges(self):
        """chi_ramp produces environment at final chi size."""
        A = _make_dense_tensor(jax.random.PRNGKey(42))
        config = CTMConfig(
            chi=8,
            chi_ramp=[(4, 5), (8, None)],
            max_iter=20,
            conv_tol=1e-6,
            min_iter=3,
        )
        envs = _ctm_tensor_multisite_fixed_point(
            {(0, 0): A}, SINGLE_SITE_NEIGHBORS, config
        )
        env = envs[(0, 0)]
        # Final chi should be 8
        assert env.C1.todense().shape == (8, 8)

    def test_chi_ramp_matches_direct(self):
        """chi_ramp result has similar corner SVs to direct chi run."""
        A = _make_dense_tensor(jax.random.PRNGKey(42))

        # Direct run at chi=8
        config_direct = CTMConfig(chi=8, max_iter=30, conv_tol=1e-8, min_iter=5)
        envs_direct = _ctm_tensor_multisite_fixed_point(
            {(0, 0): A}, SINGLE_SITE_NEIGHBORS, config_direct
        )

        # Ramp: chi=4 for 10 steps, then chi=8 to convergence
        config_ramp = CTMConfig(
            chi=8,
            chi_ramp=[(4, 10), (8, None)],
            max_iter=30,
            conv_tol=1e-8,
            min_iter=5,
        )
        envs_ramp = _ctm_tensor_multisite_fixed_point(
            {(0, 0): A}, SINGLE_SITE_NEIGHBORS, config_ramp
        )

        # Compare corner singular values
        c1_direct = envs_direct[(0, 0)].C1.todense()
        c1_ramp = envs_ramp[(0, 0)].C1.todense()
        svs_direct = jnp.linalg.svdvals(c1_direct)
        svs_ramp = jnp.linalg.svdvals(c1_ramp)
        # Normalize for comparison
        svs_direct = svs_direct / svs_direct[0]
        svs_ramp = svs_ramp / svs_ramp[0]
        np.testing.assert_allclose(svs_ramp, svs_direct, atol=1e-3)

    def test_chi_ramp_ad_gradient(self):
        """chi_ramp works through the custom_vjp AD path."""
        A = _make_dense_tensor(jax.random.PRNGKey(42))
        config = CTMConfig(
            chi=4,
            chi_ramp=[(2, 3), (4, None)],
            max_iter=5,
            conv_tol=1e-6,
            adjoint_arnoldi_precheck=False,
        )
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, config.chi)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "Gradient contains NaN/Inf"
        assert grad.norm() > 1e-15, "Gradient is all zeros"
