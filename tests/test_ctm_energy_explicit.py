"""Tests for ctm_energy_explicit: FD-AD gradient comparison."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit
from tenax.algorithms._ctm_python_loop import _make_jit_ctm_step
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _make_random_A(D=2, d=2, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (D, D, D, D, d), dtype=jnp.float64)
    # Normalize to avoid ill-conditioned CTM environments
    data = data / jnp.linalg.norm(data)
    return _wrap_as_dense_tensor(data)


@pytest.mark.slow
def test_ctm_energy_explicit_gradient_matches_fd():
    """Gradient from explicit backprop matches finite differences.

    We pre-compute warmed-up environments at the base point and pass them
    via ``env_init`` with ``warmup_steps=0``.  This way both the AD and FD
    paths differentiate through the same set of CTM sweeps, eliminating
    the mismatch caused by ``stop_gradient`` in the warmup phase.
    """
    A = _make_random_A(D=2, d=2)
    chi = 4
    gate = heisenberg_gate()
    neighbors = SINGLE_SITE_NEIGHBORS

    # Pre-warm the environments at the base point (no gradient needed)
    A_data = A.todense()
    base_A = _wrap_as_dense_tensor(A_data)
    base_tensors = {(0, 0): base_A}
    envs = {c: initialize_ctm_tensor_env(t, chi) for c, t in base_tensors.items()}
    jit_step = _make_jit_ctm_step(neighbors)
    for _ in range(20):
        envs = jit_step(
            base_tensors,
            envs,
            chi=chi,
            projector_method="eigh",
            renormalize=True,
            projector_backward="auto",
        )
    # Detach environments from computation graph
    envs = jax.tree.map(jax.lax.stop_gradient, envs)

    def energy_fn(params_data):
        A_local = _wrap_as_dense_tensor(params_data)
        site_tensors = {(0, 0): A_local}
        return ctm_energy_explicit(
            site_tensors,
            neighbors,
            gate,
            chi=chi,
            warmup_steps=0,
            backprop_steps=5,
            env_init=envs,
        )

    grad_ad = jax.grad(energy_fn)(A_data)

    eps = 1e-5
    n_check = 20
    flat = A_data.ravel()
    grad_ad_flat = grad_ad.ravel()

    # Compute FD gradient for first n_check elements, skipping any where
    # the energy evaluation returns degenerate values.
    checked = 0
    max_rel_err = 0.0
    for i in range(min(flat.size, n_check)):
        e_plus = energy_fn(flat.at[i].add(eps).reshape(A_data.shape))
        e_minus = energy_fn(flat.at[i].add(-eps).reshape(A_data.shape))
        fd_i = (e_plus - e_minus) / (2 * eps)
        # Skip if FD is degenerate (e.g. energy returns 0)
        if jnp.abs(fd_i) < 1e-10:
            continue
        if jnp.abs(fd_i) > 1e3:
            continue
        rel_err = float(jnp.abs(grad_ad_flat[i] - fd_i) / jnp.abs(fd_i))
        max_rel_err = max(max_rel_err, rel_err)
        checked += 1

    assert checked > 0, "No valid FD gradients computed"
    # Tolerance is 0.10 rather than 0.05 because Apple's Accelerate eigh
    # backward (used by the explicit-AD projector path on macOS) drifts
    # ~0.085 vs central-difference FD; OpenBLAS on Linux comes in well
    # below 0.05. See issue #369.
    assert max_rel_err < 0.10, (
        f"Gradient relative error {max_rel_err:.4e} > 0.10 ({checked} elements checked)"
    )
