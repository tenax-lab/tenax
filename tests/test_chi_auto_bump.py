"""Integration tests for variPEPS §2.8.2 auto-χ_E bump."""

from __future__ import annotations

import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_random_site_tensor(D: int, d: int, seed: int = 0) -> DenseTensor:
    """Build a random DenseTensor iPEPS site tensor with trivial U(1) charges."""
    rng = np.random.RandomState(seed)
    data = rng.standard_normal((D, D, D, D, d)).astype(np.float64)
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    import jax.numpy as jnp

    return DenseTensor(jnp.array(data), indices)


class TestCtmTensorReturnsTuple:
    def test_ctm_tensor_returns_max_truncation_error(self):
        """ctm_tensor returns (env, max_truncation_error) tuple after Task 3."""
        A = _make_random_site_tensor(D=2, d=2, seed=0)
        result = ctm_tensor(A, chi=4, max_iter=10, conv_tol=1e-6)
        assert isinstance(result, tuple), (
            f"ctm_tensor must return a 2-tuple (env, max_eps), got {type(result)}"
        )
        assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"
        env, max_eps = result
        assert isinstance(max_eps, float), (
            f"max_truncation_error must be a Python float, got {type(max_eps)}"
        )
        assert 0.0 <= max_eps <= 1.0, (
            f"max_truncation_error must be in [0, 1], got {max_eps}"
        )

    def test_ctm_tensor_eps_finite(self):
        """max_truncation_error must be a finite number."""
        A = _make_random_site_tensor(D=2, d=2, seed=1)
        env, max_eps = ctm_tensor(A, chi=4, max_iter=5, conv_tol=1e-6)
        import math

        assert math.isfinite(max_eps), f"max_eps must be finite, got {max_eps}"

    def test_ctm_tensor_eps_decreases_with_chi(self):
        """Larger chi should yield smaller or equal truncation error."""
        A = _make_random_site_tensor(D=2, d=2, seed=2)
        # D=2 so chi=4 should already be saturating; chi=2 will be tighter
        _, eps_small = ctm_tensor(A, chi=2, max_iter=20, conv_tol=1e-8)
        _, eps_large = ctm_tensor(A, chi=4, max_iter=20, conv_tol=1e-8)
        # chi=4 >= D^2=4 is already exact: eps_large <= eps_small
        assert eps_large <= eps_small + 1e-10, (
            f"Larger chi should not yield larger truncation error: "
            f"chi=2 -> {eps_small}, chi=4 -> {eps_large}"
        )


def test_ctm_config_auto_bump_defaults_off():
    """Auto-bump must be opt-in to preserve existing behavior."""
    config = CTMConfig(chi=4)
    assert config.chi_auto_bump is False
    assert config.chi_auto_bump_eps == 1e-5
    assert config.chi_auto_bump_step == 2
    assert config.chi_max is None


def test_ctm_config_auto_bump_rejects_chi_ramp_combo():
    """`chi_ramp` is a deterministic schedule; reactive auto-bump conflicts."""
    with pytest.raises(ValueError, match="chi_ramp"):
        CTMConfig(
            chi=4,
            chi_auto_bump=True,
            chi_ramp=[(4, 10), (8, None)],
        )


def test_ctm_config_auto_bump_validates_step_positive():
    """chi_auto_bump_step <= 0 is rejected when chi_auto_bump=True."""
    with pytest.raises(ValueError, match="chi_auto_bump_step"):
        CTMConfig(chi=4, chi_auto_bump=True, chi_auto_bump_step=0)


def test_ctm_config_auto_bump_validates_chi_max_above_chi():
    """chi_max is always validated when set, regardless of chi_auto_bump."""
    with pytest.raises(ValueError, match="chi_max"):
        CTMConfig(chi=4, chi_max=2)


def test_maybe_bump_chi_disabled_returns_input_unchanged():
    """Auto-bump disabled: helper passes through."""
    from tenax.algorithms.ipeps_optimize import _maybe_bump_chi

    cfg = CTMConfig(chi=4)  # chi_auto_bump=False by default
    cache = {"envs": "sentinel"}
    new_cfg, new_cache = _maybe_bump_chi(cfg, cache, last_eps_t=1.0)
    assert new_cfg is cfg
    assert new_cache is cache


def test_maybe_bump_chi_below_threshold_no_bump():
    """ε_T below threshold: no change."""
    from tenax.algorithms.ipeps_optimize import _maybe_bump_chi

    cfg = CTMConfig(chi=4, chi_auto_bump=True, chi_auto_bump_eps=1e-3)
    cache = {}
    new_cfg, new_cache = _maybe_bump_chi(cfg, cache, last_eps_t=1e-5)
    assert new_cfg.chi == 4
    assert new_cache == cache


def test_maybe_bump_chi_above_threshold_bumps_and_pads():
    """ε_T above threshold: chi rises, env is padded."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
    from tenax.algorithms.ipeps_optimize import _maybe_bump_chi
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor

    sym = U1Symmetry()
    chi_old, D = 4, 2
    chi_in = TensorIndex.from_charges(
        sym, np.zeros(chi_old, dtype=np.int32), FlowDirection.IN, label="chi"
    )
    chi_out = TensorIndex.from_charges(
        sym, np.zeros(chi_old, dtype=np.int32), FlowDirection.OUT, label="chi"
    )
    d_idx = TensorIndex.from_charges(
        sym, np.zeros(D**2, dtype=np.int32), FlowDirection.IN, label="u2"
    )

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 8)
    Csh = (chi_old, chi_old)
    Tsh = (chi_old, D**2, chi_old)

    def C(k):
        return DenseTensor(jax.random.normal(k, Csh), (chi_in, chi_out))

    def T(k):
        return DenseTensor(jax.random.normal(k, Tsh), (chi_in, d_idx, chi_out))

    env = CTMTensorEnv(
        C1=C(keys[0]),
        C2=C(keys[1]),
        C3=C(keys[2]),
        C4=C(keys[3]),
        T1=T(keys[4]),
        T2=T(keys[5]),
        T3=T(keys[6]),
        T4=T(keys[7]),
    )
    cache = {"envs": {(0, 0): env}}
    cfg = CTMConfig(
        chi=4, chi_auto_bump=True, chi_auto_bump_eps=1e-5, chi_auto_bump_step=2
    )
    new_cfg, new_cache = _maybe_bump_chi(cfg, cache, last_eps_t=1e-3)
    assert new_cfg.chi == 6  # 4 + 2
    assert new_cache["envs"][(0, 0)].C1._data.shape == (6, 6)


def test_maybe_bump_chi_respects_chi_max_ceiling():
    """chi_max ceiling caps the bump."""
    from tenax.algorithms.ipeps_optimize import _maybe_bump_chi

    cfg = CTMConfig(
        chi=4,
        chi_auto_bump=True,
        chi_auto_bump_eps=1e-5,
        chi_auto_bump_step=2,
        chi_max=5,
    )
    cache = {}
    new_cfg, _ = _maybe_bump_chi(cfg, cache, last_eps_t=1e-3)
    assert new_cfg.chi == 5  # 4+2=6, capped at 5
