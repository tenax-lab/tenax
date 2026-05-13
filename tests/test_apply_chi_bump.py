"""Unit test for the extracted _apply_chi_bump mechanism (#453).

_apply_chi_bump is the pure mechanism shared by the reactive auto-χ_E
bump (variPEPS §2.8.2) and the upcoming scheduled-bump for outer-loop
χ-ramping.  No policy — just cfg-replace + in-place env-padding.
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
    """Build a minimal env_cache dict with a single CTMTensorEnv at logical χ."""
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
def test_apply_chi_bump_updates_cfg_chi():
    cfg = CTMConfig(chi=4)
    cache = _make_dense_env_cache(4)

    new_cfg, _ = _opt._apply_chi_bump(cfg, cache, 8, base_charges=None)

    assert new_cfg.chi == 8
    # Original cfg is not mutated (dataclass replace returns a new instance).
    assert cfg.chi == 4


@pytest.mark.core
def test_apply_chi_bump_pads_envs_in_place():
    cfg = CTMConfig(chi=4)
    cache = _make_dense_env_cache(4)

    _, new_cache = _opt._apply_chi_bump(cfg, cache, 8, base_charges=None)

    # Same dict object (in-place mutation contract).
    assert new_cache is cache
    # Envs padded to new chi.
    assert new_cache["envs"][(0, 0)].C1._data.shape == (8, 8)
    assert new_cache["envs"][(0, 0)].T1._data.shape[0] == 8
    assert new_cache["envs"][(0, 0)].T1._data.shape[-1] == 8


@pytest.mark.core
def test_apply_chi_bump_no_op_when_envs_missing():
    """When env_cache has no 'envs' key, the helper just updates cfg."""
    cfg = CTMConfig(chi=4)
    cache: dict = {}  # no 'envs' key

    new_cfg, new_cache = _opt._apply_chi_bump(cfg, cache, 8, base_charges=None)

    assert new_cfg.chi == 8
    assert new_cache is cache
    assert "envs" not in new_cache
