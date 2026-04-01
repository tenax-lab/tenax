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
