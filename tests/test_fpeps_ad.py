"""Tests for AD-based fermionic iPEPS optimization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _build_initial_fpeps_tensor,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_fpeps_ad
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def fpeps_config():
    return FPEPSConfig(D=2, t=1.0, V=0.0)


@pytest.fixture
def ipeps_config_short():
    """iPEPSConfig for short smoke-test runs."""
    return iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=10, conv_tol=1e-4),
        gs_num_steps=3,
        gs_learning_rate=1e-2,
        gs_verbose=False,
    )


@pytest.fixture
def ipeps_config_medium():
    """iPEPSConfig for medium runs (energy decrease test)."""
    return iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=15, conv_tol=1e-4),
        gs_num_steps=10,
        gs_learning_rate=1e-2,
        gs_optimizer="adam",
        gs_verbose=False,
    )


# ------------------------------------------------------------------ #
# _build_initial_fpeps_tensor                                          #
# ------------------------------------------------------------------ #


class TestBuildInitialFPEPSTensor:
    """Tests for the extracted initialization helper."""

    def test_returns_symmetric_tensor(self, fpeps_config):
        A = _build_initial_fpeps_tensor(fpeps_config)
        assert isinstance(A, SymmetricTensor)

    def test_labels(self, fpeps_config):
        A = _build_initial_fpeps_tensor(fpeps_config)
        assert A.labels() == ("u", "d", "l", "r", "phys")

    def test_shape(self, fpeps_config):
        A = _build_initial_fpeps_tensor(fpeps_config)
        assert A.todense().shape == (2, 2, 2, 2, 2)

    def test_default_key(self, fpeps_config):
        """Calling with key=None should not raise."""
        A = _build_initial_fpeps_tensor(fpeps_config, key=None)
        assert jnp.all(jnp.isfinite(A.todense()))

    def test_explicit_key(self, fpeps_config):
        key = jax.random.PRNGKey(42)
        A = _build_initial_fpeps_tensor(fpeps_config, key=key)
        assert jnp.all(jnp.isfinite(A.todense()))


# ------------------------------------------------------------------ #
# optimize_fpeps_ad                                                    #
# ------------------------------------------------------------------ #


class TestOptimizeFpepsAd:
    """Tests for the AD-based fermionic iPEPS optimization entry point."""

    def test_optimize_fpeps_ad_runs(self, fpeps_config, ipeps_config_short):
        """Smoke test: 3 AD steps should complete and return finite energy."""
        H = spinless_fermion_gate(fpeps_config)
        A_opt, env, E_gs = optimize_fpeps_ad(
            H, A_init=None, config=ipeps_config_short, fpeps_config=fpeps_config
        )
        assert isinstance(A_opt, Tensor)
        assert np.isfinite(E_gs)

    def test_optimize_fpeps_ad_with_explicit_init(
        self, fpeps_config, ipeps_config_short
    ):
        """Passing an explicit A_init (SymmetricTensor) should also work."""
        H = spinless_fermion_gate(fpeps_config)
        A_init = _build_initial_fpeps_tensor(fpeps_config, jax.random.PRNGKey(7))
        assert isinstance(A_init, SymmetricTensor)
        A_opt, env, E_gs = optimize_fpeps_ad(
            H, A_init=A_init, config=ipeps_config_short
        )
        # SymmetricTensor is wrapped as DenseTensor for AD stability
        assert isinstance(A_opt, DenseTensor)
        assert np.isfinite(E_gs)

    def test_optimize_fpeps_ad_energy_decreases(
        self, fpeps_config, ipeps_config_medium
    ):
        """10 AD steps: final energy should be lower than the initial energy."""
        H = spinless_fermion_gate(fpeps_config)
        A_init = _build_initial_fpeps_tensor(fpeps_config, jax.random.PRNGKey(0))

        # Compute initial energy via a short 0-step run
        config_0 = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=15, conv_tol=1e-4),
            gs_num_steps=0,
            gs_learning_rate=1e-2,
            gs_verbose=False,
        )
        _, _, E_init = optimize_fpeps_ad(H, A_init=A_init, config=config_0)

        # Run 10 AD optimization steps
        A_opt, env, E_final = optimize_fpeps_ad(
            H, A_init=A_init, config=ipeps_config_medium
        )
        assert np.isfinite(E_final)
        assert E_final < E_init, (
            f"Energy should decrease: E_init={E_init:.8f}, E_final={E_final:.8f}"
        )

    def test_optimize_fpeps_ad_requires_fpeps_config(self, ipeps_config_short):
        """A_init=None without fpeps_config should raise ValueError."""
        H = spinless_fermion_gate(FPEPSConfig())
        with pytest.raises(ValueError, match="fpeps_config is required"):
            optimize_fpeps_ad(H, A_init=None, config=ipeps_config_short)


# ------------------------------------------------------------------ #
# todense() gradient flow                                              #
# ------------------------------------------------------------------ #


class TestTodenseGradientFlow:
    """Verify gradient flow through todense() in the AD path.

    The gauge-fixing step ``_gauge_fix_ctm_tensor`` calls ``todense()``
    on environment tensors and wraps them back via
    ``SymmetricTensor.from_dense(..., tol=inf)``.  For SymmetricTensor
    with non-trivial charges (e.g. FermionParity with both 0 and 1
    sectors), this round-trip produces NaN gradients in the backward
    pass.  The workaround used by ``optimize_fpeps_ad`` is to convert
    to DenseTensor before entering the AD loop.
    """

    @staticmethod
    def _build_loss_fn(A, config_tuple, env_treedef, prev_env_leaves, gate, d_phys):
        """Build a loss function that runs CTM + energy computation."""
        from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
        from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
        from tenax.algorithms.ad_utils import ctm_tensor_converge

        def loss_fn(A_param):
            A_norm = A_param * (1.0 / (A_param.norm() + 1e-10))
            site_tensors = {(0, 0): A_norm}
            env_leaves = ctm_tensor_converge(
                site_tensors, prev_env_leaves, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            env = jax.tree.unflatten(env_treedef, env_leaves)
            energy = compute_energy_ctm_tensor(A_norm, env, gate, d_phys)
            return energy

        return loss_fn

    @pytest.mark.algorithm
    def test_dense_tensor_gradient_finite(self):
        """DenseTensor (the workaround path) produces finite gradients."""
        from tenax.algorithms._ctm_tensor import initialize_ctm_tensor_env
        from tenax.algorithms.ad_utils import _config_to_tuple

        fpeps_cfg = FPEPSConfig(D=2, t=1.0, V=0.0)
        ctm_cfg = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10, conv_tol=1e-4),
        )

        # Build a SymmetricTensor then convert to DenseTensor (the workaround)
        A_sym = _build_initial_fpeps_tensor(fpeps_cfg, jax.random.PRNGKey(42))
        A_dense = DenseTensor(A_sym.todense(), A_sym.indices)

        gate = spinless_fermion_gate(fpeps_cfg).todense().reshape(2, 2, 2, 2)
        config_tuple = _config_to_tuple(ctm_cfg.ctm)

        env_template = initialize_ctm_tensor_env(A_dense, ctm_cfg.ctm.chi)
        env_treedef = jax.tree.structure(env_template)
        prev_env_leaves = tuple(jax.tree.leaves(env_template))

        loss_fn = self._build_loss_fn(
            A_dense, config_tuple, env_treedef, prev_env_leaves, gate, 2
        )

        energy, grads = jax.value_and_grad(loss_fn)(A_dense)

        # Energy should be finite
        assert np.isfinite(float(energy)), f"Energy is not finite: {energy}"

        # Gradient should be finite and non-zero
        grad_data = grads.todense()
        assert jnp.all(jnp.isfinite(grad_data)), "DenseTensor gradient has NaN/inf"
        assert float(jnp.linalg.norm(grad_data)) > 0, "DenseTensor gradient is zero"

    @pytest.mark.algorithm
    def test_symmetric_nontrivial_gradient_finite(self):
        """SymmetricTensor with non-trivial charges now produces finite gradients.

        Previously the gauge-fixing step called
        ``SymmetricTensor.from_dense(..., tol=inf)`` which broke gradient
        flow for non-trivial charge sectors, forcing ``optimize_fpeps_ad``
        to convert to DenseTensor before the AD loop.  The NaN has since
        been fixed, but ``optimize_fpeps_ad`` still wraps as DenseTensor
        for other stability reasons; this test now asserts the fixed
        behavior instead of documenting the old limitation.
        """
        from tenax.algorithms._ctm_tensor import initialize_ctm_tensor_env
        from tenax.algorithms.ad_utils import _config_to_tuple

        fpeps_cfg = FPEPSConfig(D=2, t=1.0, V=0.0)
        ctm_cfg = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10, conv_tol=1e-4),
        )

        # Use SymmetricTensor directly (non-trivial FermionParity charges)
        A_sym = _build_initial_fpeps_tensor(fpeps_cfg, jax.random.PRNGKey(42))
        assert isinstance(A_sym, SymmetricTensor)

        gate = spinless_fermion_gate(fpeps_cfg).todense().reshape(2, 2, 2, 2)
        config_tuple = _config_to_tuple(ctm_cfg.ctm)

        env_template = initialize_ctm_tensor_env(A_sym, ctm_cfg.ctm.chi)
        env_treedef = jax.tree.structure(env_template)
        prev_env_leaves = tuple(jax.tree.leaves(env_template))

        loss_fn = self._build_loss_fn(
            A_sym, config_tuple, env_treedef, prev_env_leaves, gate, 2
        )

        energy, grads = jax.value_and_grad(loss_fn)(A_sym)

        assert np.isfinite(float(energy)), f"Energy is not finite: {energy}"
        grad_leaves = jax.tree.leaves(grads)
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in grad_leaves), (
            "SymmetricTensor gradient contains NaN/inf — the gauge-fix "
            "round-trip may have regressed."
        )
