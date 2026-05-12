"""Tests for compiled raw-array CTM moves."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_compiled_moves import (
    _build_double_layer_raw,
    _compiled_move_left,
    _phase_fix_normalize_raw,
    compiled_ctm_sweep_raw,
)
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_moves import (
    _ctm_tensor_move_left,
    _phase_fix_normalize_tensor,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _random_site_tensor(D: int, d: int, seed: int = 0) -> DenseTensor:
    """Create a random iPEPS site tensor with 5 legs (u, d, l, r, phys)."""
    rng = np.random.RandomState(seed)
    data = jnp.array(rng.randn(D, D, D, D, d), dtype=jnp.float64)
    sym = U1Symmetry()
    indices = (
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="u"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="d"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="l"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="r"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label="phys"
        ),
    )
    return DenseTensor(data, indices)


def _env_to_raw(env: CTMTensorEnv) -> tuple[jnp.ndarray, ...]:
    """Extract raw arrays from a CTMTensorEnv."""
    return tuple(t.todense() for t in env)


class TestDoubleLayer:
    def test_matches_tensor_version(self):
        A = _random_site_tensor(D=2, d=2, seed=42)
        dl_tensor = _build_double_layer_tensor(A)
        dl_raw = _build_double_layer_raw(A.todense())
        np.testing.assert_allclose(dl_tensor.todense(), dl_raw, atol=1e-12)


class TestPhaseFixNormalize:
    def test_matches_tensor_version(self):
        rng = np.random.RandomState(0)
        arr = jnp.array(rng.randn(4, 4), dtype=jnp.float64)
        sym = U1Symmetry()
        T = DenseTensor(
            arr,
            (
                TensorIndex.from_charges(
                    sym, np.zeros(4, dtype=np.int32), FlowDirection.IN, label="a"
                ),
                TensorIndex.from_charges(
                    sym, np.zeros(4, dtype=np.int32), FlowDirection.OUT, label="b"
                ),
            ),
        )
        result_tensor = _phase_fix_normalize_tensor(T).todense()
        result_raw = _phase_fix_normalize_raw(arr)
        np.testing.assert_allclose(result_tensor, result_raw, atol=1e-12)


class TestCompiledLeftMove:
    def test_matches_tensor_left_move(self):
        """Compiled left move must match label-based Tensor left move."""
        D, d, chi = 2, 2, 4
        A = _random_site_tensor(D, d, seed=10)
        env = initialize_ctm_tensor_env(A, chi)

        # Use a different seed for the "neighbor" tensor
        A_nb = _random_site_tensor(D, d, seed=20)
        env_nb = initialize_ctm_tensor_env(A_nb, chi)

        # Build double-layer for neighbor
        dl_nb = _build_double_layer_tensor(A_nb)

        # Tensor path
        env_tensor, _ = _ctm_tensor_move_left(
            env, env_nb, dl_nb, chi, projector_method="svd"
        )

        # Raw path
        env_raw_result = _compiled_move_left(
            _env_to_raw(env),
            _env_to_raw(env_nb),
            dl_nb.todense(),
            chi,
        )

        # Compare updated tensors: C1, C4, T4
        np.testing.assert_allclose(
            env_tensor.C1.todense(),
            env_raw_result[0],
            atol=1e-10,
            err_msg="C1 mismatch",
        )
        np.testing.assert_allclose(
            env_tensor.C4.todense(),
            env_raw_result[3],
            atol=1e-10,
            err_msg="C4 mismatch",
        )
        np.testing.assert_allclose(
            env_tensor.T4.todense(),
            env_raw_result[7],
            atol=1e-10,
            err_msg="T4 mismatch",
        )


class TestCompiledGaugeFix:
    def test_phase_fix_matches_ad_utils(self):
        """Outer gauge fix must match _phase_fix_ctm_tensor."""
        from tenax.algorithms._ctm_compiled_moves import phase_fix_env_raw
        from tenax.algorithms.ad_utils import _phase_fix_ctm_tensor

        D, d, chi = 2, 2, 4
        A = _random_site_tensor(D, d, seed=50)
        env = initialize_ctm_tensor_env(A, chi)

        # Apply Tensor-based phase fix
        env_fixed = _phase_fix_ctm_tensor(env)

        # Apply raw-array phase fix
        raw_tuple = _env_to_raw(env)
        raw_fixed = phase_fix_env_raw(raw_tuple)

        for i, name in enumerate(["C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"]):
            np.testing.assert_allclose(
                env_fixed[i].todense(),
                raw_fixed[i],
                atol=1e-12,
                err_msg=f"{name} gauge-fix mismatch",
            )


class TestCompiledSweep:
    def test_single_site_sweep(self):
        """Full compiled sweep on a 1-site unit cell matches Tensor sweep."""
        from tenax.algorithms._ctm_tensor_convergence import (
            _ctm_tensor_sweep_multisite,
        )

        D, d, chi = 2, 2, 4
        A = _random_site_tensor(D, d, seed=30)
        coord = (0, 0)
        neighbors = {
            coord: {"left": coord, "right": coord, "top": coord, "bottom": coord}
        }

        env = initialize_ctm_tensor_env(A, chi)
        dl = _build_double_layer_tensor(A)

        # Tensor sweep
        envs_tensor, _ = _ctm_tensor_sweep_multisite(
            {coord: env},
            {coord: dl},
            neighbors,
            chi,
            renormalize=False,
            projector_method="svd",
        )

        # Raw sweep
        envs_raw = compiled_ctm_sweep_raw(
            {coord: A.todense()},
            {coord: _env_to_raw(env)},
            neighbors,
            chi,
        )

        # Compare all 8 env tensors
        tensor_env = envs_tensor[coord]
        raw_env = envs_raw[coord]
        names = ["C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"]
        for i, name in enumerate(names):
            np.testing.assert_allclose(
                tensor_env[i].todense(),
                raw_env[i],
                atol=1e-10,
                err_msg=f"{name} mismatch in single-site sweep",
            )

    def test_two_site_sweep(self):
        """Full compiled sweep on a 2-site checkerboard."""
        from tenax.algorithms._ctm_tensor_convergence import (
            _ctm_tensor_sweep_multisite,
        )

        D, d, chi = 2, 2, 4
        A = _random_site_tensor(D, d, seed=40)
        B = _random_site_tensor(D, d, seed=41)
        c0, c1 = (0, 0), (1, 0)
        neighbors = {
            c0: {"left": c1, "right": c1, "top": c1, "bottom": c1},
            c1: {"left": c0, "right": c0, "top": c0, "bottom": c0},
        }

        env_A = initialize_ctm_tensor_env(A, chi)
        env_B = initialize_ctm_tensor_env(B, chi)
        dl_A = _build_double_layer_tensor(A)
        dl_B = _build_double_layer_tensor(B)

        # Tensor sweep
        envs_tensor, _ = _ctm_tensor_sweep_multisite(
            {c0: env_A, c1: env_B},
            {c0: dl_A, c1: dl_B},
            neighbors,
            chi,
            renormalize=False,
            projector_method="svd",
        )

        # Raw sweep
        envs_raw = compiled_ctm_sweep_raw(
            {c0: A.todense(), c1: B.todense()},
            {c0: _env_to_raw(env_A), c1: _env_to_raw(env_B)},
            neighbors,
            chi,
        )

        names = ["C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"]
        for coord in [c0, c1]:
            for i, name in enumerate(names):
                np.testing.assert_allclose(
                    envs_tensor[coord][i].todense(),
                    envs_raw[coord][i],
                    atol=1e-10,
                    err_msg=f"{name} mismatch at {coord} in 2-site sweep",
                )


class TestCompiledBackwardGradient:
    def test_compiled_sweep_is_differentiable(self):
        """VJP through compiled_ctm_sweep_raw produces finite gradients."""
        from tenax.algorithms._ctm_compiled_moves import (
            env_arrays_to_leaves,
            leaves_to_env_arrays,
            phase_fix_env_raw,
        )

        D, d, chi = 2, 2, 4
        coord = (0, 0)
        neighbors = {
            coord: {"left": coord, "right": coord, "top": coord, "bottom": coord}
        }
        coords = [coord]

        rng = np.random.RandomState(100)
        A_arr = jnp.array(rng.randn(D, D, D, D, d), dtype=jnp.float64)

        A_tensor = _random_site_tensor(D, d, seed=100)
        env = initialize_ctm_tensor_env(A_tensor, chi)
        env_leaves = _env_to_raw(env)

        def sweep_and_sum(site_arr):
            """Sweep + phase fix + sum all env elements (scalar loss)."""
            sa = {coord: site_arr}
            ea = {coord: env_leaves}
            env_out = compiled_ctm_sweep_raw(sa, ea, neighbors, chi, has_tracers=True)
            env_out = {c: phase_fix_env_raw(env_out[c]) for c in coords}
            flat = env_arrays_to_leaves(env_out, coords)
            return sum(jnp.sum(x) for x in flat)

        grad_fn = jax.grad(sweep_and_sum)
        grad = grad_fn(A_arr)

        assert grad.shape == A_arr.shape, f"Gradient shape mismatch: {grad.shape}"
        assert jnp.all(jnp.isfinite(grad)), "Gradient contains NaN/Inf"
        assert jnp.any(grad != 0), "Gradient is all zeros"
