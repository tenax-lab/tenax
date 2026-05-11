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
        """compute_truncation_error decreases as chi increases on a known spectrum.

        In the CTM SVD projector, M = C1g^H @ C4g is a (chi × chi) matrix so
        its SVD has at most chi singular values — keeping all chi gives eps=0
        regardless of D.  The meaningful truncation happens at the level of the
        singular-value spectrum itself: given a fixed spectrum, keeping more
        singular values should give a strictly smaller eps_T.

        We test this directly on ``compute_truncation_error`` with a synthetic
        spectrum that has 9 non-trivial modes (matching D=3, D^2=9).
        """
        import jax.numpy as jnp

        from tenax.algorithms._ctm_truncation_error import compute_truncation_error

        # Synthetic spectrum with 9 non-trivial singular values (decaying)
        # Mimics what the SVD of a chi*D² × chi*D² matrix would look like.
        s = jnp.array([10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1])
        eps_chi4 = float(compute_truncation_error(s, chi=4))
        eps_chi8 = float(compute_truncation_error(s, chi=8))
        # chi=8 keeps 8 out of 9 modes; chi=4 keeps only 4 — chi=4 discards more
        assert eps_chi4 > 0.0, f"chi=4 should discard modes: eps={eps_chi4}"
        assert eps_chi8 > 0.0, f"chi=8 should still discard 1 mode: eps={eps_chi8}"
        assert eps_chi8 < eps_chi4, (
            f"Larger chi should yield strictly smaller truncation error: "
            f"chi=4 -> {eps_chi4:.6f}, chi=8 -> {eps_chi8:.6f}"
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
    # env_cache dict must be mutated in-place — closures that captured the
    # original dict (e.g. make_ctm_energy_fn) must see the padded envs.
    assert new_cache is cache, (
        "_maybe_bump_chi must mutate env_cache in-place, not create a new dict"
    )
    assert new_cache["envs"][(0, 0)].C1._data.shape == (6, 6)


def test_maybe_bump_chi_above_threshold_bumps_and_pads_symmetric():
    """variPEPS §2.8.2 auto-bump on SymmetricTensor envs (Issue #410).

    Previously, ``_maybe_bump_chi`` -> ``pad_dense_env_chi`` raised
    ``NotImplementedError`` for block-sparse envs.  After #410, the
    symmetric path tiles existing χ-leg charges via ``_derive_charges``
    and zero-pads each block along its χ axes.
    """
    import jax
    import numpy as np

    from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
    from tenax.algorithms.ipeps_optimize import _maybe_bump_chi
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import SymmetricTensor

    sym = U1Symmetry()
    bond_charges = np.array([0, 1], dtype=np.int32)
    phys_charges = np.array([0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    A_sym = SymmetricTensor.random_normal(indices, jax.random.PRNGKey(0))
    env_old = initialize_ctm_tensor_env(A_sym, chi=4)

    cache = {"envs": {(0, 0): env_old}}
    cfg = CTMConfig(
        chi=4, chi_auto_bump=True, chi_auto_bump_eps=1e-5, chi_auto_bump_step=2
    )
    new_cfg, new_cache = _maybe_bump_chi(cfg, cache, last_eps_t=1e-3)
    assert new_cfg.chi == 6
    assert new_cache is cache, (
        "_maybe_bump_chi must mutate env_cache in-place even on the symmetric path"
    )
    env_new = new_cache["envs"][(0, 0)]
    # χ axes grew to 6; D² leg untouched.
    assert env_new.C1.indices[0].dim == 6
    assert env_new.C1.indices[1].dim == 6
    assert env_new.T1.indices[0].dim == 6
    assert env_new.T1.indices[2].dim == 6
    assert env_new.T1.indices[1].dim == env_old.T1.indices[1].dim
    # Leading χ_old × χ_old block matches the pre-pad corner.
    chi_old = 4
    C1_old = env_old.C1.todense()
    C1_new = env_new.C1.todense()
    assert np.allclose(np.asarray(C1_new)[:chi_old, :chi_old], np.asarray(C1_old))


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


# Integration test moved to tests/test_ipeps_chi_bump_integration.py
# (algorithm-marked, ~3 min runtime; kept out of the core suite).
