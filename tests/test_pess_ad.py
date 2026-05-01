"""Smoke tests for the kagome iPESS AD loss closure (Task 8)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import IPESSState, kagome_triangle_xxz_hamiltonian
from tenax.algorithms.pess_optimize import build_pess_loss


def _make_test_config(chi: int) -> CTMConfig:
    """Small CTM config for smoke tests: tight iteration budget, biorthogonal
    projectors (kagome supersites are non-isometric, A_u ≠ A_d)."""
    return CTMConfig(
        chi=chi,
        max_iter=20,
        min_iter=4,
        conv_tol=1e-6,
        projector_method="biorthogonal",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
        gmres_tol=1e-4,
        gmres_maxiter=50,
        gmres_restart=20,
        chi_ramp=None,
    )


def test_pess_loss_returns_finite_real_scalar():
    """The loss closure returns a finite real scalar for a random PESS state."""
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss(H, config)
    e0 = loss_fn(state)

    assert jnp.shape(e0) == ()
    assert jnp.isrealobj(e0) or jnp.allclose(e0.imag, 0.0, atol=1e-10)
    assert jnp.isfinite(e0)


def test_pess_loss_is_differentiable():
    """``jax.grad`` of the loss returns finite gradients on all 5 PESS primitives.

    Spec for Task 8: must produce finite gradients on R_a, R_b, R_c, T_u, T_d
    via implicit-AD CTM through the converged honeycomb environment.
    """
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss(H, config)
    e0 = loss_fn(state)
    g = jax.grad(loss_fn)(state)

    assert jnp.isfinite(e0)
    for arr in (g.R_a, g.R_b, g.R_c, g.T_u, g.T_d):
        assert jnp.all(jnp.isfinite(arr)), (
            "non-finite values in PESS gradient — implicit-AD CTM backward "
            "produced NaN/Inf on a smoke-test state."
        )
        # IPESSState.random returns complex128 tensors — gradients should
        # match dtype to keep the variational regime intact.
        assert arr.dtype == jnp.complex128


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_pess_loss_runs_at_multiple_anisotropies(delta: float):
    """Smoke check: closure runs for a few values of XXZ anisotropy."""
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(1))
    H = kagome_triangle_xxz_hamiltonian(delta=delta, d=3)
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss(H, config)
    e = loss_fn(state)
    assert jnp.isfinite(e)
