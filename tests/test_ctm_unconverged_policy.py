"""CTM unconverged-forward policy (#1059 hotspot 8, #1060).

Spec: docs/superpowers/specs/2026-10-01-ctm-unconverged-policy-design.md
"""

from __future__ import annotations

import math
import warnings

import pytest

from tenax.algorithms._ctm_convergence_policy import (
    CTMNotConvergedError,
    CTMNotConvergedWarning,
    check_ctm_converged,
    format_step_multiplier,
)
from tenax.algorithms._ctm_python_loop import CTMConvergeInfo
from tenax.algorithms.ipeps_config import CTMConfig


def _info(converged, iterations=500, sv_diff=3.9e-7, **extra):
    info = CTMConvergeInfo(converged=converged, iterations=iterations, sv_diff=sv_diff)
    if not extra:
        return info

    class _WithExtra:
        pass

    obj = _WithExtra()
    for f in info._fields:
        setattr(obj, f, getattr(info, f))
    for k, v in extra.items():
        setattr(obj, k, v)
    return obj


def test_check_passes_when_converged():
    assert check_ctm_converged(_info(True), site="gradient", policy="raise") is True


def test_check_raises_with_diagnostics():
    with pytest.raises(CTMNotConvergedError) as ei:
        check_ctm_converged(
            _info(False),
            site="gradient",
            policy="raise",
            step=7,
            conv_tol=1e-10,
            chi=12,
        )
    msg = str(ei.value)
    for needle in ("gradient", "step 7", "500", "3.9e-07", "1e-10", "chi=12"):
        assert needle in msg, (needle, msg)
    assert ei.value.site == "gradient" and ei.value.step == 7


def test_check_warns_under_warn():
    with pytest.warns(CTMNotConvergedWarning, match="gradient"):
        out = check_ctm_converged(_info(False), site="gradient", policy="warn", step=3)
    assert out is False


def test_message_shows_numeric_step_multiplier():
    with pytest.raises(CTMNotConvergedError, match=r"step multiplier -0\.99"):
        check_ctm_converged(
            _info(False, step_multiplier=-0.99), site="g", policy="raise"
        )


@pytest.mark.parametrize("value", [None, float("nan")])
def test_message_shows_na_when_multiplier_missing_or_nan(value):
    info = _info(False) if value is None else _info(False, step_multiplier=value)
    assert format_step_multiplier(info) == "n/a"
    with pytest.raises(CTMNotConvergedError, match=r"step multiplier n/a"):
        check_ctm_converged(info, site="g", policy="raise")


def test_ctmconfig_default_and_validation():
    assert CTMConfig().on_unconverged == "raise"
    assert CTMConfig(on_unconverged="warn").on_unconverged == "warn"
    with pytest.raises(ValueError, match="on_unconverged"):
        CTMConfig(on_unconverged="bogus")


def test_public_exports():
    import tenax

    assert tenax.CTMNotConvergedError is CTMNotConvergedError
    assert tenax.CTMNotConvergedWarning is CTMNotConvergedWarning
    assert "CTMNotConvergedError" in tenax.__all__
    assert "CTMNotConvergedWarning" in tenax.__all__
