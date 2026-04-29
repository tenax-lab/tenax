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
        # optimize_fpeps_ad is polymorphic over the Tensor protocol (#297) —
        # returns a tensor of the same type as the input.
        assert isinstance(A_opt, type(A_init))
        assert np.isfinite(E_gs)

    @pytest.mark.xfail(
        reason=(
            "Pre-#357 this test passed because the broken bar() (no "
            "Koszul twist on fermionic SymmetricTensor) made the "
            "fermionic CTM energy collapse to ~0; near-zero energy "
            "plus near-zero gradient meant L-BFGS took small steps "
            "that satisfied E_final < E_init by numerical noise. After "
            "#357 the forward energy is correct (E_init ~ -4.44), but "
            "the implicit-AD backward through the gauge-fix path still "
            "produces wrong gradients for SymmetricTensor with "
            "non-trivial charges (same root cause as "
            "test_symmetric_nontrivial_gradient_finite below) — the "
            "optimizer now moves uphill with a correct landscape and a "
            "broken gradient. Will pass once the gauge-fix-AD issue is "
            "resolved; that's tracked in #354 bucket E follow-up."
        ),
        strict=False,
    )
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
        # adjoint_arnoldi_precheck=False: this fixture's backward has
        # rho ~ 6.97 which crosses the default raise threshold (5.0,
        # default since #334); the precheck tripping is incidental to
        # the gradient-finiteness contract this test exercises.
        ctm_cfg = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(
                chi=4, max_iter=10, conv_tol=1e-4, adjoint_arnoldi_precheck=False
            ),
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
    def test_symmetric_nontrivial_energy_finite(self):
        """SymmetricTensor with non-trivial fermionic charges produces a
        finite forward energy through the implicit CTM path.

        Regression for #357: the open-RDM construction used ``A.bar()``
        to build the bra layer, which on fermionic ``SymmetricTensor``
        (FermionParity / FermionicU1) missed the super-algebra Koszul
        twist ``(-1)^{sum_{i<j} p_i p_j}``. The Hermitization step in
        ``_rdm2x1_tensor`` then cancelled the sign-broken terms to
        produce ``trace ≈ -0.017`` and ``E_h ≈ 0`` (effectively zero,
        which would propagate 0/0 NaNs into any downstream gradient).
        Fixed by routing the bra construction through ``bar_super()``,
        which applies Koszul for fermionic symmetries and is identical
        to ``bar()`` for bosonic.

        This test pins the **forward** energy at a sensible non-zero
        magnitude. Gradient through the implicit-AD backward is still
        NaN due to a separate gauge-fix differentiation issue tracked
        in ``test_symmetric_nontrivial_gradient_finite`` below.
        """
        from tenax.algorithms._ctm_tensor import initialize_ctm_tensor_env
        from tenax.algorithms.ad_utils import _config_to_tuple

        fpeps_cfg = FPEPSConfig(D=2, t=1.0, V=0.0)
        ctm_cfg = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(
                chi=4,
                max_iter=10,
                conv_tol=1e-4,
                adjoint_arnoldi_precheck=False,
            ),
        )

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
        energy = float(loss_fn(A_sym))
        assert np.isfinite(energy), f"Energy is not finite: {energy}"
        # Pre-fix the energy was 6.3e-18 (effectively zero from sign
        # cancellation in the open RDM); post-fix it is ~0.1. Anchor a
        # loose lower bound on |E| so future regressions in the bar()
        # path (or its callers) trip immediately.
        assert abs(energy) > 1e-6, (
            f"Energy magnitude {abs(energy):.3e} is suspiciously small "
            "— bar_super() Koszul twist may have regressed."
        )

    @pytest.mark.algorithm
    @pytest.mark.xfail(
        reason=(
            "Implicit-AD backward through the gauge-fix path "
            "(_phase_fix_ctm_tensor / _wrap_tensor with "
            "from_dense(..., tol=inf), or the argmax-based phase pick) "
            "is not differentiation-safe for SymmetricTensor with "
            "non-trivial charges. Forward energy is now finite (#357 "
            "fixed the bar() Koszul twist), but the implicit VJP still "
            "produces all-NaN gradients on this fixture. Separate from "
            "#357; tracked in #354 bucket E."
        ),
        strict=True,
    )
    def test_symmetric_nontrivial_gradient_finite(self):
        """SymmetricTensor with non-trivial charges should produce finite gradients.

        The Koszul-twist part of this contract is now covered by
        ``test_symmetric_nontrivial_energy_finite`` (no xfail). What
        remains is the implicit-AD backward through the gauge-fix path,
        which is the bug the xfail marker documents.
        """
        from tenax.algorithms._ctm_tensor import initialize_ctm_tensor_env
        from tenax.algorithms.ad_utils import _config_to_tuple

        fpeps_cfg = FPEPSConfig(D=2, t=1.0, V=0.0)
        ctm_cfg = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(
                chi=4,
                max_iter=10,
                conv_tol=1e-4,
                adjoint_arnoldi_precheck=False,
            ),
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
