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
from tenax.algorithms.ipeps_optimize import (
    _log_ad_compile_notice,
    optimize_fpeps_ad,
    optimize_gs_ad,
)
from tenax.core.tensor import DenseTensor, SymmetricTensor

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
    """iPEPSConfig for medium runs (energy decrease test).

    ``chi=8`` is required: at ``chi=4`` the CTM converges to a
    non-physical metastable fixed point (E ≈ -4.44 vs E ≈ +0.18 at
    chi=8) and the perturbed-A energy landscape is wildly non-smooth
    — descent breaks out of that artifact basin and looks like
    "uphill" optimization. ``chi=8`` lands in the physical basin
    where the energy varies smoothly and Adam can descend.
    """
    return iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=8, max_iter=30, conv_tol=1e-4),
        gs_num_steps=10,
        gs_learning_rate=1e-2,
        gs_optimizer="adam",
        gs_verbose=False,
    )


# ------------------------------------------------------------------ #
# optimize_fpeps_ad                                                    #
# ------------------------------------------------------------------ #


class TestOptimizeFpepsAd:
    """Tests for the AD-based fermionic iPEPS optimization entry point."""

    @pytest.mark.slow
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

    def test_optimize_fpeps_ad_requires_fpeps_config(self, ipeps_config_short):
        """A_init=None without fpeps_config should raise ValueError."""
        H = spinless_fermion_gate(FPEPSConfig())
        with pytest.raises(ValueError, match="fpeps_config is required"):
            optimize_fpeps_ad(H, A_init=None, config=ipeps_config_short)


# ------------------------------------------------------------------ #
# #565: fermionic 2-site AD de-silencing + completion smoke test      #
# ------------------------------------------------------------------ #


class TestFermionic2SiteADRegression:
    """Regression coverage for #565.

    The 2-site fermionic AD path (``optimize_gs_ad(unit_cell="2site")``
    on FermionParity input) was reported as a *silent hang*: 30+ min at
    99% CPU with no output after the implicit-AD warning.  It is not a
    deadlock — the cost is a one-time trace+compile of the block-sparse
    CTM backward, which scales with charge-block count (tracked in #566).
    Two regressions are guarded here:

    1. The first-step compile notice fires (so the wait is never silent).
    2. The path actually completes with a finite energy and preserves
       the SymmetricTensor type — the multi-block symmetric AD that CI
       did not previously exercise (``test_block_sparse_ctm_ad`` uses
       trivial single-block charges).
    """

    def test_compile_notice_emitted_when_verbose(self, capsys):
        """The de-silencing notice prints (with the #566 ref) iff verbose."""
        verbose_cfg = iPEPSConfig(gs_verbose=True)
        _log_ad_compile_notice(verbose_cfg)
        out = capsys.readouterr().out
        assert "step 1" in out and "#566" in out, out

        quiet_cfg = iPEPSConfig(gs_verbose=False)
        _log_ad_compile_notice(quiet_cfg)
        assert capsys.readouterr().out == ""

    @pytest.mark.slow
    def test_fermionic_2site_ad_completes(self):
        """One step of 2-site FermionParity AD completes with finite energy.

        Wrapped in a generous wall-clock guard (the compile is the cost,
        not a deadlock — see #566): a regression to a true hang fails the
        test loudly instead of stalling the suite indefinitely.
        """
        import threading

        fcfg = FPEPSConfig(D=2, t=1.0, V=2.0)
        gate = spinless_fermion_gate(fcfg)
        A = _build_initial_fpeps_tensor(fcfg, jax.random.PRNGKey(1))
        B = _build_initial_fpeps_tensor(fcfg, jax.random.PRNGKey(2))
        assert isinstance(A, SymmetricTensor)

        config = iPEPSConfig(
            max_bond_dim=2,
            unit_cell="2site",
            su_init=False,
            gs_implicit_ad=True,
            gs_num_steps=1,
            gs_verbose=False,
            ctm=CTMConfig(chi=4, max_iter=20, conv_tol=1e-4),
        )

        result: dict = {}

        def _run():
            try:
                (A_opt, B_opt), _envs, E = optimize_gs_ad(gate, (A, B), config)
                result.update(A_opt=A_opt, B_opt=B_opt, E=E)
            except BaseException as exc:  # noqa: BLE001 - re-raise in main thread
                result["exc"] = exc

        # XLA compile releases the GIL, so join(timeout) on the main thread
        # observes a real wall-clock bound.  600s is generous for D=2 chi=4
        # (~2-4 min) yet well under the >30 min #565 hang.
        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout=600.0)

        assert not worker.is_alive(), (
            "Fermionic 2-site AD did not complete within 600s — possible "
            "regression to the #565 hang."
        )
        assert "exc" not in result, f"2-site fermionic AD raised: {result['exc']!r}"
        assert np.isfinite(float(result["E"])), f"Non-finite energy: {result['E']}"
        assert isinstance(result["A_opt"], SymmetricTensor)
        assert isinstance(result["B_opt"], SymmetricTensor)


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
    @pytest.mark.slow
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
    def test_symmetric_nontrivial_gradient_finite(self):
        """SymmetricTensor with non-trivial charges should produce finite gradients.

        Regression for the implicit-AD backward through the gauge-fix
        path (#362): pre-fix, ``jax.value_and_grad`` returned all-NaN
        gradient leaves on a fermionic ``SymmetricTensor`` with empty
        charge sectors. Root cause was the differentiable
        ``arr / (norm + 1e-30)`` step in ``_phase_fix_normalize_tensor``
        and ``_phase_fix_ctm_tensor`` — JAX evaluated both branches of
        ``jnp.where`` in the SVD/divide backward, and the dense layout's
        out-of-block zeros produced a 0/0 NaN at the fixed point. Fixed
        by wrapping the norm in ``jax.lax.stop_gradient`` (the radial
        component of the cotangent is discarded by the outer unit-sphere
        projection anyway).
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
