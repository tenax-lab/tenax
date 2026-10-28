"""``iPEPSConfig.gs_conv_patience``: the outer criterion must hold on that many
consecutive steps before ``optimize_gs_ad`` exits.

A huge ``gs_grad_norm_tol`` makes every step meet the criterion, so the exit
step is set by the patience alone: patience ``p`` exits on step ``p``.
"""

from dataclasses import replace

import jax.numpy as jnp
import pytest

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad
from tenax.algorithms._ipeps_optimize_shared import _next_conv_streak


def test_next_conv_streak_counts_and_resets():
    s = 0
    for met in (True, True, False, True):
        s = _next_conv_streak(s, met)
    assert s == 1
    assert _next_conv_streak(2, True) == 3
    assert _next_conv_streak(7, False) == 0


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
def test_config_rejects_bad_patience(bad):
    with pytest.raises(ValueError, match="gs_conv_patience"):
        iPEPSConfig(gs_conv_patience=bad)


def test_config_default_patience_is_one():
    assert iPEPSConfig().gs_conv_patience == 1
    assert iPEPSConfig(gs_conv_patience=3).gs_conv_patience == 3


def _heisenberg_gate():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def _cfg(unit_cell: str, patience: int) -> iPEPSConfig:
    # Same cheap explicit-AD setup as test_ipeps_ad_conv_criterion.py.
    return iPEPSConfig(
        max_bond_dim=2,
        unit_cell=unit_cell,
        ctm=CTMConfig(chi=4, max_iter=30),
        gs_num_steps=8,
        gs_learning_rate=1e-2,
        gs_implicit_ad=False,
        gs_explicit_ad_steps=2,
        gs_explicit_ad_warmup=1,
        gs_optimizer="adam",
        su_init=False,
        return_history=True,
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e30,  # every finite step meets the criterion
        gs_conv_patience=patience,
    )


@pytest.mark.algorithm
@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_patience_sets_the_exit_step(unit_cell):
    steps = {}
    for p in (1, 3):
        hist = optimize_gs_ad(_heisenberg_gate(), None, _cfg(unit_cell, p))[-1]
        assert hist["converged"] is True
        steps[p] = hist["num_steps"]
    # Patience 3 exits exactly two steps after patience 1.
    assert steps[3] - steps[1] == 2, steps
    assert steps[3] < 8
