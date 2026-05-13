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
    """1-site C4v run with gs_chi_schedule_steps=[(2, 8)] bumps env from chi=4 to chi=8."""
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
        gs_num_steps=3,
        gs_chi_schedule_steps=[(2, 8)],
        gs_optimizer="lbfgs",
        gs_verbose=False,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        A_opt, env, E = _opt.optimize_gs_ad(gate, A, cfg)

    # After 3 steps with bump at boundary=2, env should be padded to chi=8.
    assert env.C1._data.shape == (8, 8), (
        f"expected env padded to chi=8 after schedule boundary, "
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
