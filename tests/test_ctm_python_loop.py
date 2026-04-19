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
    Coord,
    ctm_tensor,
    ctm_tensor_2site,
)
from tenax.algorithms._ctm_tensor_energy import (
    compute_energy_ctm_tensor,
    compute_energy_ctm_tensor_2site,
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

    def test_single_step_runs(self):
        """_make_jit_ctm_step completes without error."""
        A = _make_random_A()
        site_tensors = {(0, 0): A}
        envs = {(0, 0): initialize_ctm_tensor_env(A, chi=4)}
        step = _make_jit_ctm_step(SINGLE_SITE_NEIGHBORS)
        new_envs = step(site_tensors, envs, chi=4)
        assert (0, 0) in new_envs
        # Corner shape should be (chi, chi)
        assert new_envs[(0, 0)].C1.todense().shape == (4, 4)

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

    def test_multisite_step(self):
        """_make_jit_ctm_step works with 2-site checkerboard."""
        A = _make_random_A(key=jax.random.PRNGKey(0))
        B = _make_random_A(key=jax.random.PRNGKey(1))
        site_tensors = {(0, 0): A, (1, 0): B}
        envs = {
            (0, 0): initialize_ctm_tensor_env(A, chi=4),
            (1, 0): initialize_ctm_tensor_env(B, chi=4),
        }
        step = _make_jit_ctm_step(CHECKERBOARD_NEIGHBORS)
        new_envs = step(site_tensors, envs, chi=4)
        assert (0, 0) in new_envs
        assert (1, 0) in new_envs


class TestPythonLoopCtmConverge:
    """Tests for python_loop_ctm_converge."""

    def test_python_loop_matches_existing_ctm(self):
        """Python-loop CTM must converge to same energy as existing code."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        chi = 4
        gate = _heisenberg_gate()

        # Reference: existing ctm_tensor
        env_ref = ctm_tensor(A, chi, max_iter=50, conv_tol=1e-10)
        energy_ref = float(compute_energy_ctm_tensor(A, env_ref, gate))

        # Python loop (min_iter=1 to match reference which has no min_iter gate)
        envs, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=chi,
            max_iter=50,
            min_iter=1,
            conv_tol=1e-10,
        )
        energy_loop = float(compute_energy_ctm_tensor(A, envs[(0, 0)], gate))

        np.testing.assert_allclose(energy_loop, energy_ref, atol=1e-6)

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

    def test_python_loop_2site_checkerboard(self):
        """Python-loop CTM converges for 2-site checkerboard."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        B = _make_random_A(key=jax.random.PRNGKey(99))
        chi = 4
        gate = _heisenberg_gate()

        # Reference: existing ctm_tensor_2site
        env_A_ref, env_B_ref = ctm_tensor_2site(A, B, chi, max_iter=50, conv_tol=1e-10)
        energy_ref = float(
            compute_energy_ctm_tensor_2site(A, B, env_A_ref, env_B_ref, gate)
        )

        # Python loop (min_iter=1 to match reference which has no min_iter gate)
        envs, info = python_loop_ctm_converge(
            {(0, 0): A, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            chi=chi,
            max_iter=50,
            min_iter=1,
            conv_tol=1e-10,
        )
        energy_loop = float(
            compute_energy_ctm_tensor_2site(A, B, envs[(0, 0)], envs[(1, 0)], gate)
        )

        np.testing.assert_allclose(energy_loop, energy_ref, atol=1e-6)

    def test_python_loop_chi_ramp(self):
        """Chi ramp works: starts at chi=4, ramps to chi=8."""
        A = _make_random_A(key=jax.random.PRNGKey(42))

        envs, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=8,
            max_iter=30,
            conv_tol=1e-8,
            chi_ramp=[(4, 10), (8, None)],
        )

        # Final chi should be 8
        assert envs[(0, 0)].C1.todense().shape == (8, 8)

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

    def test_qr_warmup(self):
        """QR projector with warmup runs without error."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        envs, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=20,
            conv_tol=1e-6,
            projector_method="qr",
            qr_warmup_steps=3,
        )
        # Should run without error; convergence not guaranteed but should progress
        assert info.iterations > 0

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

    def test_2site_matches_existing(self):
        """Multisite energy for checkerboard matches compute_energy_ctm_tensor_2site."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        B = _make_random_A(key=jax.random.PRNGKey(99))
        gate = _heisenberg_gate()
        chi = 4

        envs, _ = python_loop_ctm_converge(
            {(0, 0): A, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            chi=chi,
            max_iter=50,
            conv_tol=1e-10,
        )

        energy_ref = float(
            compute_energy_ctm_tensor_2site(A, B, envs[(0, 0)], envs[(1, 0)], gate)
        )
        energy_multi = float(
            compute_energy_ctm_tensor_multisite(
                {(0, 0): A, (1, 0): B}, envs, CHECKERBOARD_NEIGHBORS, gate
            )
        )

        # The multisite function counts all 4 bonds (AB and BA) and averages,
        # while the 2-site function exploits symmetry (only AB bonds).
        # They agree up to small numerical asymmetry in the environments.
        np.testing.assert_allclose(energy_multi, energy_ref, atol=1e-3)

    def test_multisite_returns_finite_scalar(self):
        """Multisite energy returns a finite scalar for a 2x2 unit cell."""
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        coords = [(0, 0), (1, 0), (0, 1), (1, 1)]
        site_tensors = {c: _make_random_A(key=k) for c, k in zip(coords, keys)}

        # 2x2 periodic neighbor map
        neighbors = {
            (0, 0): {
                "left": (1, 0),
                "right": (1, 0),
                "top": (0, 1),
                "bottom": (0, 1),
            },
            (1, 0): {
                "left": (0, 0),
                "right": (0, 0),
                "top": (1, 1),
                "bottom": (1, 1),
            },
            (0, 1): {
                "left": (1, 1),
                "right": (1, 1),
                "top": (0, 0),
                "bottom": (0, 0),
            },
            (1, 1): {
                "left": (0, 1),
                "right": (0, 1),
                "top": (1, 0),
                "bottom": (1, 0),
            },
        }

        chi = 4
        envs, _ = python_loop_ctm_converge(
            site_tensors,
            neighbors,
            chi=chi,
            max_iter=30,
            conv_tol=1e-6,
        )

        gate = _heisenberg_gate()
        energy = compute_energy_ctm_tensor_multisite(
            site_tensors, envs, neighbors, gate
        )

        assert jnp.isfinite(energy)
        assert energy.shape == ()  # scalar
