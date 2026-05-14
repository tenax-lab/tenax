"""End-to-end: gs_chi_schedule_steps actually bumps env shape mid-run (#453)."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


def _heisenberg_gate():
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    return (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real


@pytest.mark.slow
def test_1site_chi_schedule_bumps_env_at_boundary():
    """Per-stage chi schedule bumps env between stages (#455).

    Two-stage schedule ``[(4, 1), (8, 1)]`` per #455 semantics means:
    stage 0 runs at logical chi=4 for 1 step, then the budget is
    exhausted, ``_advance_chi_stage_if_due`` fires and bumps to chi=8
    via ``_apply_chi_bump`` (which also pads the cached env in place);
    stage 1 runs at chi=8 for 1 step.  After 2 total optimizer steps
    the env corner ``C1`` must be padded to chi=8.  Adam is used (no
    line-search / L-BFGS rollback to interact with the bump).
    """
    d = 2
    D = 2
    rng = np.random.default_rng(0)
    A = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    A = A / jnp.linalg.norm(A)
    gate = _heisenberg_gate()

    cfg = iPEPSConfig(
        unit_cell="1x1",
        gs_c4v=True,
        ctm=CTMConfig(chi=4, chi_max=8),
        gs_num_steps=2,
        gs_chi_schedule_steps=[(4, 1), (8, 1)],
        gs_optimizer="adam",
        gs_verbose=False,
        su_init=False,
        gs_conv_criterion="grad_norm",
        gs_conv_tol=1e-30,
        gs_grad_norm_tol=1e-30,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        A_opt, env, E = _opt.optimize_gs_ad(gate, A, cfg)

    # After stage 0's 1-step budget is exhausted, env should be padded to chi=8.
    assert env.C1._data.shape == (8, 8), (
        f"expected env padded to chi=8 after stage-0 budget exhausted, "
        f"got {env.C1._data.shape}"
    )


@pytest.mark.slow
def test_chi_schedule_none_is_no_op():
    """With gs_chi_schedule_steps=None (default), env stays at logical chi."""
    d = 2
    D = 2
    rng = np.random.default_rng(0)
    A = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    A = A / jnp.linalg.norm(A)
    gate = _heisenberg_gate()

    cfg = iPEPSConfig(
        unit_cell="1x1",
        gs_c4v=True,
        ctm=CTMConfig(chi=4),
        gs_num_steps=3,
        gs_optimizer="lbfgs",
        gs_verbose=False,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        A_opt, env, E = _opt.optimize_gs_ad(gate, A, cfg)

    assert env.C1._data.shape == (4, 4), (
        f"expected env to stay at chi=4 with no schedule, got {env.C1._data.shape}"
    )
