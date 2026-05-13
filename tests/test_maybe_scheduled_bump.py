"""Unit tests for _maybe_scheduled_bump (#453).

Scheduled-bump fires when the optimizer step counter crosses a
``(boundary, target_chi)`` boundary in the schedule.  Reuses the
``_apply_chi_bump`` mechanism, so padding and in-place mutation are
inherited from the reactive-bump path.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _idx(dim: int, label: str, flow: FlowDirection) -> TensorIndex:
    sym = U1Symmetry()
    return TensorIndex.from_charges(
        sym, np.zeros(dim, dtype=np.int32), flow, label=label
    )


def _make_dense_env_cache(chi: int, D: int = 2) -> dict:
    """Build a minimal env_cache dict with a single CTMTensorEnv at logical χ.

    Mirrors ``tests/test_apply_chi_bump.py``.
    """
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 8)
    Csh = (chi, chi)
    Tsh = (chi, D**2, chi)
    chi_in = _idx(chi, "chi", FlowDirection.IN)
    chi_out = _idx(chi, "chi", FlowDirection.OUT)
    d_idx = _idx(D**2, "u2", FlowDirection.IN)

    def make_corner(k):
        return DenseTensor(jax.random.normal(k, Csh), (chi_in, chi_out))

    def make_edge(k):
        return DenseTensor(jax.random.normal(k, Tsh), (chi_in, d_idx, chi_out))

    env = CTMTensorEnv(
        C1=make_corner(keys[0]),
        C2=make_corner(keys[1]),
        C3=make_corner(keys[2]),
        C4=make_corner(keys[3]),
        T1=make_edge(keys[4]),
        T2=make_edge(keys[5]),
        T3=make_edge(keys[6]),
        T4=make_edge(keys[7]),
    )
    return {"envs": {(0, 0): env}, "max_truncation_error": 0.0}


@pytest.mark.core
def test_scheduled_bump_no_op_when_schedule_none():
    cfg = CTMConfig(chi=4)
    cache = _make_dense_env_cache(4)
    new_cfg, new_cache = _opt._maybe_scheduled_bump(
        cfg, cache, step=10, schedule_targets=None, base_charges=None
    )
    assert new_cfg is cfg
    assert new_cache is cache


@pytest.mark.core
def test_scheduled_bump_fires_at_boundary():
    # Schedule: at cumulative step 5, target chi=8.
    schedule = [(5, 8)]
    cfg = CTMConfig(chi=4)
    cache = _make_dense_env_cache(4)

    # Step 4: below boundary, no bump.
    new_cfg, _ = _opt._maybe_scheduled_bump(
        cfg, cache, step=4, schedule_targets=schedule, base_charges=None
    )
    assert new_cfg.chi == 4

    # Step 5: cross boundary, bump to 8.
    new_cfg, _ = _opt._maybe_scheduled_bump(
        cfg, cache, step=5, schedule_targets=schedule, base_charges=None
    )
    assert new_cfg.chi == 8
    assert cache["envs"][(0, 0)].C1._data.shape == (8, 8)


@pytest.mark.core
def test_scheduled_bump_idempotent_past_boundary():
    # Already at target chi=8; calls past boundary are no-ops.
    schedule = [(5, 8)]
    cfg = CTMConfig(chi=8)
    cache = _make_dense_env_cache(8)
    new_cfg, _ = _opt._maybe_scheduled_bump(
        cfg, cache, step=7, schedule_targets=schedule, base_charges=None
    )
    assert new_cfg.chi == 8  # No-op: would not bump down.
    assert new_cfg is cfg or new_cfg.chi == cfg.chi


@pytest.mark.core
def test_scheduled_bump_walks_through_multi_stage_schedule():
    # Multi-stage schedule: (5, 8) then (10, 16).
    schedule = [(5, 8), (10, 16)]
    cfg = CTMConfig(chi=4)
    cache = _make_dense_env_cache(4)

    # Step 5: bumps to 8 (first stage).
    cfg, _ = _opt._maybe_scheduled_bump(
        cfg, cache, step=5, schedule_targets=schedule, base_charges=None
    )
    assert cfg.chi == 8

    # Step 9: between stages, no further bump.
    new_cfg, _ = _opt._maybe_scheduled_bump(
        cfg, cache, step=9, schedule_targets=schedule, base_charges=None
    )
    assert new_cfg.chi == 8

    # Step 10: bumps to 16 (second stage).
    cfg, _ = _opt._maybe_scheduled_bump(
        cfg, cache, step=10, schedule_targets=schedule, base_charges=None
    )
    assert cfg.chi == 16


@pytest.mark.core
def test_scheduled_bump_respects_chi_max():
    # If chi_max=10 caps below the schedule target=16, only bump to 10.
    schedule = [(5, 16)]
    cfg = CTMConfig(chi=4, chi_max=10)
    cache = _make_dense_env_cache(4)
    new_cfg, _ = _opt._maybe_scheduled_bump(
        cfg, cache, step=5, schedule_targets=schedule, base_charges=None
    )
    assert new_cfg.chi == 10
