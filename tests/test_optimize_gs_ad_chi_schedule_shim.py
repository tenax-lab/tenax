"""optimize_gs_ad_chi_schedule is now a thin shim (#453).

Old: calls optimize_gs_ad once per (chi, num_steps) stage, dropping the
env between stages and forcing per-stage JIT retraces.

New: calls optimize_gs_ad exactly ONCE with envs padded to max(chi)
from step 1 and gs_chi_schedule_steps stashed on the config so the
inner loop ramps logical chi via _maybe_scheduled_bump.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


def _trivial_gate():
    return jnp.zeros((2, 2, 2, 2), dtype=jnp.complex128)


@pytest.mark.core
def test_chi_schedule_shim_invokes_optimize_gs_ad_once(monkeypatch):
    """The shim should make a SINGLE optimize_gs_ad call with the
    unified config (chi_max set, gs_chi_schedule_steps set, gs_num_steps
    = sum of stage budgets)."""
    captured = []

    def _spy(gate, A_init, cfg):
        captured.append(cfg)
        # Return a no-op result of the right shape for the unit-cell.
        # Don't actually run; this is a contract test on the shim.
        return (A_init, None, 0.0)

    monkeypatch.setattr(_opt, "optimize_gs_ad", _spy)

    D = 2
    d = 2
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((D, D, D, D, d)))
    A = A / jnp.linalg.norm(A)

    base_cfg = iPEPSConfig(
        unit_cell="1x1",
        ctm=CTMConfig(chi=4),
        gs_num_steps=200,  # will be overridden by shim
        su_init=False,
    )

    _opt.optimize_gs_ad_chi_schedule(
        _trivial_gate(), A, base_cfg, chi_schedule=[(4, 3), (8, 3), (16, 4)]
    )

    assert len(captured) == 1, (
        f"shim should call optimize_gs_ad once, got {len(captured)}"
    )
    inner_cfg = captured[0]
    assert inner_cfg.gs_num_steps == 10, (
        f"expected gs_num_steps=10 (sum of stage budgets), got {inner_cfg.gs_num_steps}"
    )
    # schedule_targets entries are (cumulative_step_boundary, chi_to_bump_TO).
    # For chi_schedule=[(4, 3), (8, 3), (16, 4)] the intended stages are:
    #   - steps 1-3 at chi=4 (initial)
    #   - after step 3 (cum=3), bump to chi=8
    #   - steps 4-6 at chi=8
    #   - after step 6 (cum=6), bump to chi=16
    #   - steps 7-10 at chi=16
    # So schedule_targets is the (cum, next-stage-chi) pairs for bumps only —
    # the initial stage has no bump entry.
    assert inner_cfg.gs_chi_schedule_steps == [(3, 8), (6, 16)], (
        f"expected schedule_targets=[(3, 8), (6, 16)] (bump-to-next-stage), "
        f"got {inner_cfg.gs_chi_schedule_steps}"
    )
    assert inner_cfg.ctm.chi == 4, f"expected initial chi=4, got {inner_cfg.ctm.chi}"
    assert inner_cfg.ctm.chi_max == 16, (
        f"expected chi_max=16, got {inner_cfg.ctm.chi_max}"
    )


@pytest.mark.core
def test_chi_schedule_single_stage_has_no_bumps():
    """A single-stage chi_schedule should produce an empty schedule_targets list:
    no boundaries, no bumps. The optimizer simply runs gs_num_steps at the
    initial chi."""
    captured = []

    def _spy(gate, A_init, cfg):
        captured.append(cfg)
        return (A_init, None, 0.0)

    import pytest as _pytest

    base_cfg = iPEPSConfig(
        unit_cell="1x1",
        ctm=CTMConfig(chi=8),
        gs_num_steps=200,
        su_init=False,
    )
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(_opt, "optimize_gs_ad", _spy)
        _opt.optimize_gs_ad_chi_schedule(
            _trivial_gate(),
            jnp.zeros((2, 2, 2, 2, 2), dtype=jnp.complex128),
            base_cfg,
            chi_schedule=[(8, 10)],
        )
    assert captured[0].gs_chi_schedule_steps == [], (
        f"single-stage schedule should produce no bumps, "
        f"got {captured[0].gs_chi_schedule_steps}"
    )
    assert captured[0].gs_num_steps == 10
    assert captured[0].ctm.chi == 8
    assert captured[0].ctm.chi_max == 8
