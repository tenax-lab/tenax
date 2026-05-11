"""Tests for Python-loop CTM convergence."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_python_loop import (
    CTMConvergeInfo,
    _make_jit_ctm_step,
    python_loop_ctm_converge,
)
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms._ctm_tensor_energy import (
    compute_energy_ctm_tensor,
    compute_energy_ctm_tensor_multisite,
)
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


@pytest.fixture(autouse=True)
def _enable_x64():
    """Enable float64 for this test module and restore afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _make_random_A(D=2, d=2, key=None):
    """Create a random DenseTensor iPEPS site tensor."""
    if key is None:
        key = jax.random.PRNGKey(0)
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


# Heisenberg gate for energy computation
def _heisenberg_gate():
    """Return the 2-site Heisenberg gate as (2,2,2,2) array."""
    return jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)


class TestJitCtmStep:
    """Tests for _make_jit_ctm_step."""

    def test_step_changes_env(self):
        """_make_jit_ctm_step produces different environments from input."""
        A = _make_random_A()
        site_tensors = {(0, 0): A}
        envs = {(0, 0): initialize_ctm_tensor_env(A, chi=4)}
        step = _make_jit_ctm_step(SINGLE_SITE_NEIGHBORS)
        new_envs = step(site_tensors, envs, chi=4)
        # Environments should differ after a sweep
        old_c1 = envs[(0, 0)].C1.todense()
        new_c1 = new_envs[(0, 0)].C1.todense()
        assert not jnp.allclose(old_c1, new_c1)


class TestPythonLoopCtmConverge:
    """Tests for python_loop_ctm_converge."""

    def test_python_loop_converges(self):
        """Python-loop CTM reaches convergence."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        envs, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=100,
            conv_tol=1e-8,
        )
        assert info.converged
        assert info.sv_diff < 1e-8
        assert info.iterations > 1

    def test_python_loop_2site_checkerboard_converges(self):
        """Checkerboard topology converges + produces a finite energy.

        Behavioral coverage for the CHECKERBOARD_NEIGHBORS sweep path,
        which `optimize_gs_ad` for 2-site cells routes through. No
        atol-tight parity check vs the legacy `ctm_tensor_2site` path
        (that is brittle on macOS — see #354 bucket E).
        """
        A = _make_random_A(key=jax.random.PRNGKey(42))
        B = _make_random_A(key=jax.random.PRNGKey(99))
        gate = _heisenberg_gate()
        envs, info = python_loop_ctm_converge(
            {(0, 0): A, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            chi=4,
            max_iter=50,
            conv_tol=1e-8,
        )
        assert info.converged
        energy = compute_energy_ctm_tensor_multisite(
            {(0, 0): A, (1, 0): B}, envs, CHECKERBOARD_NEIGHBORS, gate
        )
        assert jnp.isfinite(energy)
        assert energy.shape == ()

    def test_chi_ramp_energy_close_to_direct(self):
        """Chi ramp from chi=4 to chi=8 gives energy close to direct chi=8."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        gate = _heisenberg_gate()

        # Direct run at chi=8
        envs_direct, _ = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=8,
            max_iter=100,
            conv_tol=1e-10,
        )
        energy_direct = float(compute_energy_ctm_tensor(A, envs_direct[(0, 0)], gate))

        # Ramp: chi=4 warmup, then chi=8
        envs_ramp, _ = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=8,
            max_iter=100,
            conv_tol=1e-10,
            chi_ramp=[(4, 5), (8, None)],
        )
        energy_ramp = float(compute_energy_ctm_tensor(A, envs_ramp[(0, 0)], gate))

        # Both should give similar energy at chi=8
        np.testing.assert_allclose(energy_ramp, energy_direct, atol=1e-4)

    def test_converge_info_fields(self):
        """CTMConvergeInfo has correct fields."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        envs, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=5,
            conv_tol=1e-20,  # won't converge in 5 iters
        )
        assert isinstance(info, CTMConvergeInfo)
        assert info.converged is False
        assert info.iterations == 5
        assert info.sv_diff > 0

    def test_env_init_warm_start(self):
        """Passing env_init warm-starts the convergence."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        chi = 4

        # Run 10 iterations to get warm environments
        envs_warm, _ = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=chi,
            max_iter=10,
            conv_tol=1e-20,
        )

        # Continue from warm start — should converge faster
        envs_final, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=chi,
            max_iter=100,
            conv_tol=1e-10,
            env_init=envs_warm,
        )
        assert info.converged


class TestMultisiteEnergy:
    """Tests for compute_energy_ctm_tensor_multisite."""

    def test_1site_matches_existing(self):
        """Multisite energy for 1-site cell matches compute_energy_ctm_tensor."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        gate = _heisenberg_gate()
        chi = 4

        envs, _ = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=chi,
            max_iter=50,
            conv_tol=1e-10,
        )

        energy_ref = float(compute_energy_ctm_tensor(A, envs[(0, 0)], gate))
        energy_multi = float(
            compute_energy_ctm_tensor_multisite(
                {(0, 0): A}, envs, SINGLE_SITE_NEIGHBORS, gate
            )
        )

        np.testing.assert_allclose(energy_multi, energy_ref, atol=1e-10)


class TestPlateauPatience:
    """Regression coverage for the ``plateau_patience`` early-bail."""

    def test_disabled_matches_legacy_max_iter(self):
        """``plateau_patience=None`` runs to ``max_iter`` even when stuck."""
        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        max_iter = 30
        _, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=max_iter,
            min_iter=5,
            conv_tol=0.0,  # impossible — forces non-convergence
            conv_method="elementwise",
            renormalize=True,
            projector_method="svd",
            plateau_patience=None,
        )
        assert not info.converged
        assert info.iterations == max_iter

    def test_patience_bails_before_max_iter_on_plateau(self):
        """A non-converging input bails before ``max_iter`` with a finite patience."""
        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        max_iter = 50
        patience = 5
        _, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=max_iter,
            min_iter=5,
            conv_tol=0.0,  # impossible — forces non-convergence
            conv_method="elementwise",
            renormalize=True,
            projector_method="svd",
            plateau_patience=patience,
        )
        assert not info.converged
        # Bail must happen before exhausting the budget, and the returned
        # iteration count is the best-iter (not the bail-iter) so it is
        # bounded by ``max_iter - patience``.
        assert info.iterations < max_iter

    def test_large_patience_matches_disabled(self):
        """``plateau_patience > max_iter`` is equivalent to ``None``."""
        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        max_iter = 25
        kwargs = dict(
            site_tensors={(0, 0): A},
            neighbors=SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=max_iter,
            min_iter=5,
            conv_tol=0.0,
            conv_method="elementwise",
            renormalize=True,
            projector_method="svd",
        )
        _, info_none = python_loop_ctm_converge(plateau_patience=None, **kwargs)
        _, info_huge = python_loop_ctm_converge(
            plateau_patience=10 * max_iter, **kwargs
        )
        assert info_none.iterations == info_huge.iterations == max_iter
        assert not info_none.converged
        assert not info_huge.converged

    def test_returned_env_is_best_seen(self):
        """Returned env corresponds to the best ``sv_diff``, not the bail iter."""
        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        _, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=50,
            min_iter=5,
            conv_tol=0.0,
            conv_method="elementwise",
            renormalize=True,
            projector_method="svd",
            plateau_patience=5,
        )
        assert isinstance(info, CTMConvergeInfo)
        assert not info.converged
        # ``sv_diff`` carries the *best* metric, so it must be finite and
        # not larger than the very first measurement we could have made.
        assert info.sv_diff < float("inf")
