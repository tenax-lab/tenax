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

pytestmark = pytest.mark.core


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
