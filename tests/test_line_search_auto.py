"""Per-χ-stage ``gs_line_search_method='auto'`` resolver (#509).

iPEPS-AD's gradient at small χ carries CTM truncation noise of order ε_T
that frequently exceeds Hager-Zhang's strong-Wolfe curvature tolerance
(σ ≈ 0.9), so HZ wastes ``value_and_grad`` calls hunting an unachievable
Wolfe point and falls back to a poor α anyway.  Armijo only checks
sufficient decrease and dodges the noise entirely at ~5-10× lower per-
probe cost — but at large χ (≥ 16 for the 2-site bipartite path) HZ's
strong-Wolfe pairing helps L-BFGS converge superlinearly.

``gs_line_search_method='auto'`` picks Armijo when ``ctm.chi`` is below
``gs_line_search_hz_chi_threshold`` and HZ otherwise; explicit
``'armijo'`` / ``'hager_zhang'`` choices pass through unchanged.
"""

from __future__ import annotations

import pytest

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import _resolve_line_search_method

# Bucket comes from ``_FILE_MARKERS`` in conftest, not from a module-level
# ``pytestmark`` (#933): conftest *adds* its marker, so a module-level core
# mark would override the per-test ``@pytest.mark.slow`` withholding below.


def test_explicit_hager_zhang_passes_through():
    config = iPEPSConfig(max_bond_dim=2, gs_line_search_method="hager_zhang")
    assert _resolve_line_search_method(config, CTMConfig(chi=8)) == "hager_zhang"
    assert _resolve_line_search_method(config, CTMConfig(chi=64)) == "hager_zhang"


def test_explicit_armijo_passes_through():
    config = iPEPSConfig(max_bond_dim=2, gs_line_search_method="armijo")
    assert _resolve_line_search_method(config, CTMConfig(chi=8)) == "armijo"
    assert _resolve_line_search_method(config, CTMConfig(chi=64)) == "armijo"


def test_auto_picks_armijo_below_default_threshold():
    """Default threshold is 16; chi < 16 → Armijo."""
    config = iPEPSConfig(max_bond_dim=2, gs_line_search_method="auto")
    assert _resolve_line_search_method(config, CTMConfig(chi=8)) == "armijo"
    assert _resolve_line_search_method(config, CTMConfig(chi=15)) == "armijo"


def test_auto_picks_hager_zhang_at_or_above_default_threshold():
    """Default threshold is 16; chi >= 16 → HZ."""
    config = iPEPSConfig(max_bond_dim=2, gs_line_search_method="auto")
    assert _resolve_line_search_method(config, CTMConfig(chi=16)) == "hager_zhang"
    assert _resolve_line_search_method(config, CTMConfig(chi=24)) == "hager_zhang"


def test_auto_honors_custom_threshold():
    config = iPEPSConfig(
        max_bond_dim=2,
        gs_line_search_method="auto",
        gs_line_search_hz_chi_threshold=24,
    )
    assert _resolve_line_search_method(config, CTMConfig(chi=16)) == "armijo"
    assert _resolve_line_search_method(config, CTMConfig(chi=23)) == "armijo"
    assert _resolve_line_search_method(config, CTMConfig(chi=24)) == "hager_zhang"
    assert _resolve_line_search_method(config, CTMConfig(chi=64)) == "hager_zhang"


def test_invalid_method_rejected_in_config():
    with pytest.raises(ValueError, match="gs_line_search_method must be one of"):
        iPEPSConfig(max_bond_dim=2, gs_line_search_method="bogus")


def test_invalid_threshold_rejected_in_config():
    with pytest.raises(
        ValueError, match="gs_line_search_hz_chi_threshold must be positive"
    ):
        iPEPSConfig(max_bond_dim=2, gs_line_search_hz_chi_threshold=0)
    with pytest.raises(
        ValueError, match="gs_line_search_hz_chi_threshold must be positive"
    ):
        iPEPSConfig(max_bond_dim=2, gs_line_search_hz_chi_threshold=-4)


def test_default_method_is_hager_zhang_for_back_compat():
    """The pre-#509 default was ``hager_zhang`` and stays unchanged so
    existing callers see no behavior change.  Auto is opt-in."""
    config = iPEPSConfig(max_bond_dim=2)
    assert config.gs_line_search_method == "hager_zhang"
    # ctm.chi value should not matter for the explicit-default branch.
    assert _resolve_line_search_method(config, CTMConfig(chi=4)) == "hager_zhang"
    assert _resolve_line_search_method(config, CTMConfig(chi=32)) == "hager_zhang"


def test_resolver_called_once_per_optimizer_dispatch():
    """Codex P2 on #549: ``_resolve_line_search_method`` must be called
    exactly once per ``optimize_gs_ad`` dispatch, not per line-search
    step.  Without caching, ``ctm_cfg.chi`` shifting mid-run under
    legacy ``chi_ramp`` / ``chi_auto_bump`` paths would silently flip
    the resolved method between Armijo and HZ — violating the
    documented "resolved once per dispatch" contract.

    This test patches the resolver, runs a short ``optimize_gs_ad``
    (1-site, default path), and asserts a single call.  The same
    invariant applies to the 2-site and multisite dispatchers by
    construction (all three optimizers now bind the resolver result
    to ``line_search_method`` before the optimizer loop).
    """
    import jax.numpy as jnp

    import tenax.algorithms.ipeps_optimize as opt_mod
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    # Heisenberg gate (small, fast).
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    gate = (jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))).reshape(
        2, 2, 2, 2
    )

    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=2,  # enough to traverse multiple line-search calls
        gs_line_search_method="auto",
        gs_implicit_ad=True,
        gs_optimizer="lbfgs",
    )

    real_resolver = opt_mod._resolve_line_search_method
    calls: list[tuple] = []

    def counting_resolver(cfg, ctm_cfg):
        calls.append((id(cfg), id(ctm_cfg)))
        return real_resolver(cfg, ctm_cfg)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(opt_mod, "_resolve_line_search_method", counting_resolver)
    try:
        optimize_gs_ad(gate, None, config)
    finally:
        monkey.undo()

    # Exactly one call per dispatch, regardless of gs_num_steps / line-search retries.
    assert len(calls) == 1, (
        f"expected resolver called once per dispatch, got {len(calls)} calls"
    )
