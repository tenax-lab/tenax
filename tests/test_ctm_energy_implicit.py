"""Tests for ctm_energy_implicit: FD-AD gradient comparison."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate, ipeps
from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _make_su_tensor(D=2, d=2):
    """Create a simple-update initialized tensor (well-conditioned for CTM)."""
    gate = heisenberg_gate()
    config = iPEPSConfig(max_bond_dim=D, num_imaginary_steps=100, dt=0.05)
    _, (A_su, _), _ = ipeps(gate, None, config)
    data = A_su.todense()
    data = data / jnp.linalg.norm(data)
    return _wrap_as_dense_tensor(data)


@pytest.mark.slow
def test_ctm_energy_implicit_gradient_matches_fd():
    """Gradient from GMRES backward matches finite differences.

    Uses a simple-update initialized tensor (not random) for well-conditioned
    CTM convergence.  The implicit AD gradient is compared to central finite
    differences, filtering out elements where the FD gradient is very small
    (where relative error is amplified by noise/discontinuities).
    """
    A = _make_su_tensor(D=2, d=2)
    chi = 4
    gate = heisenberg_gate()
    neighbors = SINGLE_SITE_NEIGHBORS
    A_data = A.todense()

    def energy_fn(params_data):
        A_local = _wrap_as_dense_tensor(params_data)
        site_tensors = {(0, 0): A_local}
        return ctm_energy_implicit(
            site_tensors,
            neighbors,
            gate,
            chi=chi,
            max_iter=100,
            conv_tol=1e-12,
            gmres_tol=1e-10,
            gmres_maxiter=300,
            gmres_restart=50,
        )

    grad_ad = jax.grad(energy_fn)(A_data)

    n_check = 15
    flat = A_data.ravel()
    grad_ad_flat = grad_ad.ravel()

    # Compute FD gradients, using smaller eps if the default hits a CTM
    # convergence discontinuity (FD value > 100 suggests branch jump).
    rel_errs = []
    for i in range(min(flat.size, n_check)):
        fd_i = None
        for eps in [1e-5, 1e-6]:
            e_plus = energy_fn(flat.at[i].add(eps).reshape(A_data.shape))
            e_minus = energy_fn(flat.at[i].add(-eps).reshape(A_data.shape))
            fd_candidate = float((e_plus - e_minus) / (2 * eps))
            if abs(fd_candidate) < 100:
                fd_i = fd_candidate
                break
        if fd_i is None:
            continue
        # Skip elements with very small FD (amplified relative error)
        if abs(fd_i) < 5e-3:
            continue
        rel_err = abs(float(grad_ad_flat[i]) - fd_i) / abs(fd_i)
        rel_errs.append(rel_err)

    assert len(rel_errs) >= 5, f"Too few valid FD gradients: {len(rel_errs)}"

    # Use median relative error -- robust to outliers from near-zero gradients
    median_err = float(jnp.median(jnp.array(rel_errs)))
    max_err = max(rel_errs)
    assert median_err < 0.05, (
        f"Median gradient relative error {median_err:.4e} > 0.05 "
        f"(max {max_err:.4e}, {len(rel_errs)} elements checked)"
    )


@pytest.mark.slow
def test_ctm_energy_implicit_forward_runs():
    """Smoke test: forward pass produces a reasonable energy."""
    A = _make_su_tensor(D=2, d=2)
    chi = 4
    gate = heisenberg_gate()
    neighbors = SINGLE_SITE_NEIGHBORS

    site_tensors = {(0, 0): A}
    energy = ctm_energy_implicit(
        site_tensors,
        neighbors,
        gate,
        chi=chi,
        max_iter=40,
        conv_tol=1e-8,
    )
    assert jnp.isfinite(energy), f"Energy is not finite: {energy}"
    assert energy.shape == (), f"Energy should be scalar, got shape {energy.shape}"
