"""``optimize_gs_ad``'s returned energy must be an evaluation of the tensor it
returns (#899).

The final re-evaluation advertises a *"fully converged fresh CTM"* and its
block comment gives the reason:

    Re-evaluate both final A and best_A with fully converged fresh CTM.
    In-loop energies use warm-started CTM that can produce unphysical values
    (non-variational at finite chi), so we compare fresh evaluations only.

It then seeds that evaluation from ``_env_cache["envs"]`` -- which
``_restore_env_cache_after_line_search`` has just reverted to the environment
converged at the *previous* parameters.  So the returned number is
``_eval_fresh(params_final, env(previous params))``: neither fresh nor an
evaluation of the tensor handed back.

The error does not wash out: #899 measured it bit-identical at ``max_iter``
40/100/300 and ``conv_tol`` down to 1e-12, 7.5e-06 to 1.4e-04 away from the
cold value, and 19x larger for one of two tensors that were themselves
1.27e-08 apart.  It is a seed-selected branch, not under-convergence.

The invariant asserted here is the weakest honest one: **re-converging the
returned tensor from scratch must reproduce the returned energy.**  It says
nothing about which fixed point is "right" -- only that the number and the
tensor describe the same state.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad


def _cfg() -> iPEPSConfig:
    """Cheap explicit-AD config; enough outer steps that the line-search
    reversion has something to revert to."""
    return iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=6, max_iter=12, conv_tol=1e-9),
        gs_num_steps=4,
        gs_learning_rate=1e-2,
        gs_implicit_ad=False,
        gs_explicit_ad_steps=2,
        gs_explicit_ad_warmup=1,
        # lbfgs, NOT adam: ``_use_line_search`` is true only for lbfgs/cg (or
        # an explicit gs_line_search), and the defect lives in the reversion
        # ``_restore_env_cache_after_line_search`` performs. Under adam the
        # cache is never reverted, the seed IS the current state's env, and
        # this fixture cannot see the bug at all.
        gs_optimizer="lbfgs",
        gs_line_search=True,
        su_init=False,
        return_history=False,
    )


@pytest.mark.core
def test_returned_energy_is_an_evaluation_of_the_returned_tensor():
    cfg = _cfg()
    gate = heisenberg_gate()
    A_final, _env, E_returned = optimize_gs_ad(gate, None, cfg)

    # Cold-converge the RETURNED tensor from scratch: no env_init, same CTM
    # settings the optimizer used for its own "fresh" evaluation.
    envs, _info = python_loop_ctm_converge(
        {(0, 0): A_final},
        SINGLE_SITE_NEIGHBORS,
        **ctm_converge_kwargs(cfg.ctm, env_init=None),
    )
    d_phys = A_final.indices[-1].dim
    E_cold = float(compute_energy_ctm_tensor(A_final, envs[(0, 0)], gate, d_phys))

    assert jnp.isfinite(E_returned) and jnp.isfinite(E_cold)
    assert E_returned == pytest.approx(E_cold, abs=1e-8), (
        f"returned E={E_returned!r} is not an evaluation of the returned "
        f"tensor (cold-converged E={E_cold!r}, spread "
        f"{abs(E_returned - E_cold):.3e}). The final evaluation is seeded "
        "from the line-search-reverted env cache, so it reports the energy "
        "of a different state -- see #899."
    )


@pytest.mark.core
def test_a_seeded_final_evaluation_would_be_detected():
    """Non-vacuity: the assertion above can fail.

    Seeding a CTM from a *different* state's environment at this chi moves the
    energy by far more than the 1e-8 gate, so the check is not passing merely
    because everything at D=2/chi=6 agrees to 1e-8 regardless.
    """
    cfg = _cfg()
    gate = heisenberg_gate()
    A_final, _env, _E = optimize_gs_ad(gate, None, cfg)
    d_phys = A_final.indices[-1].dim

    cold_envs, _ = python_loop_ctm_converge(
        {(0, 0): A_final},
        SINGLE_SITE_NEIGHBORS,
        **ctm_converge_kwargs(cfg.ctm, env_init=None),
    )
    E_cold = float(compute_energy_ctm_tensor(A_final, cold_envs[(0, 0)], gate, d_phys))

    # A deliberately foreign environment: converged for a different tensor.
    other, _, _ = optimize_gs_ad(gate, None, replace(_cfg(), gs_num_steps=1))
    foreign_envs, _ = python_loop_ctm_converge(
        {(0, 0): other},
        SINGLE_SITE_NEIGHBORS,
        **ctm_converge_kwargs(cfg.ctm, env_init=None),
    )
    seeded_envs, _ = python_loop_ctm_converge(
        {(0, 0): A_final},
        SINGLE_SITE_NEIGHBORS,
        **ctm_converge_kwargs(cfg.ctm, env_init=foreign_envs),
    )
    E_seeded = float(
        compute_energy_ctm_tensor(A_final, seeded_envs[(0, 0)], gate, d_phys)
    )

    # If a foreign seed changed nothing, this fixture cannot detect the defect
    # and the sibling test above would be passing for free.
    assert abs(E_seeded - E_cold) > 1e-9, (
        f"a foreign seed moved the energy by only {abs(E_seeded - E_cold):.3e}, "
        "so this D/chi cannot distinguish seeded from fresh and the sibling "
        "test proves nothing -- raise chi or D"
    )
