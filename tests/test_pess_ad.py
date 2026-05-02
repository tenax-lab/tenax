"""Smoke tests for the kagome iPESS AD loss closure (CG-iPEPS path)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import IPESSState, kagome_xxz_pess_cg_gates
from tenax.algorithms.pess_optimize import build_pess_loss, optimize_pess_ad


def _make_test_config(chi: int) -> CTMConfig:
    """Small CTM config for smoke tests: tight iteration budget, SVD
    projectors (the standard square CTM AD default)."""
    return CTMConfig(
        chi=chi,
        max_iter=20,
        min_iter=4,
        conv_tol=1e-6,
        projector_method="svd",
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
    cg_gates = kagome_xxz_pess_cg_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss(cg_gates, config)
    e0 = loss_fn(state)

    assert jnp.shape(e0) == ()
    assert jnp.isrealobj(e0) or jnp.allclose(e0.imag, 0.0, atol=1e-10)
    assert jnp.isfinite(e0)


def test_pess_loss_is_differentiable():
    """``jax.grad`` of the loss returns finite gradients on the iPESS primitives.

    The CG-iPEPS path optimizes ``(R_a, R_b, R_c, T_u, lambdas)``; ``T_d``
    is held frozen (its effect is absorbed via the down-bond ``sqrt(λ)``
    gauges).
    """
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    cg_gates = kagome_xxz_pess_cg_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss(cg_gates, config)
    e0 = loss_fn(state)
    g = jax.grad(loss_fn)(state)

    assert jnp.isfinite(e0)
    for arr in (g.R_a, g.R_b, g.R_c, g.T_u):
        assert jnp.all(jnp.isfinite(arr)), (
            "non-finite values in PESS gradient — implicit-AD CTM backward "
            "produced NaN/Inf on a smoke-test state."
        )
        # IPESSState.random returns complex128 tensors — gradients should
        # match dtype to keep the variational regime intact.
        assert arr.dtype == jnp.complex128
    for lam_g in g.lambdas:
        assert jnp.all(jnp.isfinite(lam_g))


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_pess_loss_runs_at_multiple_anisotropies(delta: float):
    """Smoke check: closure runs for a few values of XXZ anisotropy."""
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(1))
    cg_gates = kagome_xxz_pess_cg_gates(delta=delta, d=3)
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss(cg_gates, config)
    e = loss_fn(state)
    assert jnp.isfinite(e)


def test_optimize_pess_ad_decreases_energy():
    """L-BFGS lowers the triangle energy below the random-init baseline."""
    state0 = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(2))
    cg_gates = kagome_xxz_pess_cg_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    e0 = float(build_pess_loss(cg_gates, config)(state0))
    state_opt, e_opt = optimize_pess_ad(
        state0, cg_gates, config, max_iter=5, verbose=False
    )

    assert jnp.isfinite(e_opt)
    assert e_opt < e0, f"L-BFGS did not decrease energy: e0={e0}, e_opt={e_opt}"


def test_optimize_pess_ad_preserves_shapes_and_T_d():
    """Shapes/dtype preserved; T_d frozen unchanged (it's not a CG-path AD variable)."""
    state0 = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(3))
    cg_gates = kagome_xxz_pess_cg_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    state_opt, _ = optimize_pess_ad(state0, cg_gates, config, max_iter=2)

    assert state_opt.R_a.shape == state0.R_a.shape
    assert state_opt.R_b.shape == state0.R_b.shape
    assert state_opt.R_c.shape == state0.R_c.shape
    assert state_opt.T_u.shape == state0.T_u.shape
    assert state_opt.T_d.shape == state0.T_d.shape
    assert state_opt.R_a.dtype == jnp.complex128
    assert state_opt.T_u.dtype == jnp.complex128
    # T_d is not optimized in the CG path — it's preserved bit-exact.
    assert jnp.array_equal(state_opt.T_d, state0.T_d)
