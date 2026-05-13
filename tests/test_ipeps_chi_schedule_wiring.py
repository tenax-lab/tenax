"""Wiring smoke test for ``optimize_gs_ad_chi_schedule`` (#455 PR 1).

Asserts that a 2-stage schedule actually bumps chi between stages,
exercising the helper that PR 1 introduces (and the legacy
``_maybe_scheduled_bump`` function pre-refactor).  This test pins
the *mechanism* — that a chi bump fires between stages — not
optimizer convergence.

Marker: ``core``.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad_chi_schedule


@pytest.mark.core
def test_chi_schedule_bumps_between_stages():
    """A 2-stage schedule ``[(chi=2, n=2), (chi=3, n=2)]`` must advance to chi=3.

    The legacy ``_maybe_scheduled_bump`` (pre-refactor) and the
    new ``_advance_chi_stage_if_due`` helper (post-refactor) must
    both produce the same observable: a final env at chi=3 after
    a 4-step run on a tiny 2-site D=2 Heisenberg problem.

    Convergence is intentionally disabled (``gs_conv_tol=1e-30``,
    ``gs_grad_norm_tol=1e-30``) so the optimizer doesn't early-
    exit; stall recovery is similarly disabled
    (``gs_stall_recovery_retries=99``) so a stall doesn't trip a
    break.  This isolates the chi-schedule bump from every other
    exit mechanism.
    """
    jax.config.update("jax_enable_x64", True)

    d = 2
    D = 2
    rng = np.random.default_rng(0)
    A_data = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    A_data = A_data / jnp.linalg.norm(A_data)

    # The 2-site C4v path takes a tuple (A, B); B is derived internally
    # from A via sublattice rotation when ``gs_c4v=True``, so any tuple
    # whose first element has shape (D,D,D,D,d) is acceptable.
    A_init = (A_data, A_data)

    gate = heisenberg_gate()

    cfg = iPEPSConfig(
        unit_cell="2site",
        max_bond_dim=D,
        ctm=CTMConfig(chi=2, chi_max=3, max_iter=10, conv_tol=1e-4),
        gs_c4v=True,
        gs_optimizer="lbfgs",
        gs_num_steps=4,  # overridden by the shim (sum of schedule budgets)
        gs_verbose=False,
        gs_conv_tol=1e-30,  # don't converge on dE
        gs_grad_norm_tol=1e-30,  # don't converge on grad_norm
        gs_stall_recovery_retries=99,  # don't trip stall cap
        su_init=False,
    )

    # Chi-ramp shim; final stage targets chi=3.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = optimize_gs_ad_chi_schedule(
            gate, A_init, cfg, chi_schedule=[(2, 2), (3, 2)]
        )

    final_chi = _extract_final_chi(result)
    assert final_chi == 3, (
        f"Expected final chi=3 after 2-stage schedule [(2,2),(3,2)], "
        f"got chi={final_chi}. The chi-schedule bump did not fire."
    )


def _extract_final_chi(result):
    """Pull final chi out of ``optimize_gs_ad_chi_schedule``'s return value.

    The shim forwards to ``optimize_gs_ad``, whose return signature is:

    * 1-site:     ``(A_opt, env, E_gs)``
    * 2-site:     ``((A_opt, B_opt), (env_A, env_B), E_gs)``
    * multi-site: ``(dict[str, Tensor], dict[str, CTMTensorEnv], E_gs)``

    For all three, ``env`` (or any single CTMTensorEnv) has corner
    tensor ``C1`` whose first leg dim is the logical chi (it is
    grown in-place by ``_apply_chi_bump``).

    The smoke test uses ``unit_cell="2site"`` so the middle entry is
    ``(env_A, env_B)``; both share the same chi.
    """
    _A, envs, _E = result

    if isinstance(envs, dict):
        # Multisite: dict[str, CTMTensorEnv].
        any_env = next(iter(envs.values()))
        return int(any_env.C1.indices[0].dim)

    if isinstance(envs, tuple):
        # 2-site: (env_A, env_B).  Both envs share the same chi.
        env_a = envs[0]
        return int(env_a.C1.indices[0].dim)

    # 1-site: single CTMTensorEnv.
    return int(envs.C1.indices[0].dim)
