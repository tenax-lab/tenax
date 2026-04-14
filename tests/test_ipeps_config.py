"""Tests for iPEPSConfig new fields (issue #298)."""

import pytest

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


def test_stall_recovery_default_is_none():
    cfg = iPEPSConfig()
    assert cfg.gs_stall_recovery is None
    assert cfg.gs_energy_floor is None


def test_stall_recovery_accepts_noise_and_reset():
    cfg_n = iPEPSConfig(gs_stall_recovery="noise")
    cfg_r = iPEPSConfig(gs_stall_recovery="reset")
    assert cfg_n.gs_stall_recovery == "noise"
    assert cfg_r.gs_stall_recovery == "reset"


def test_energy_floor_stores_float():
    cfg = iPEPSConfig(gs_energy_floor=-1.5)
    assert cfg.gs_energy_floor == -1.5


def test_stall_recovery_rejects_invalid_value():
    with pytest.raises(ValueError, match="gs_stall_recovery"):
        iPEPSConfig(gs_stall_recovery="noize")  # typo


# --- CTMConfig.projector_backward (Task 6, plan PR #315) ---


def test_projector_backward_default_is_auto():
    assert CTMConfig(chi=8).projector_backward == "auto"


def test_projector_backward_explicit_standard_is_honored():
    assert (
        CTMConfig(chi=8, projector_backward="standard").projector_backward == "standard"
    )


def test_projector_backward_explicit_lorentzian_is_honored():
    assert (
        CTMConfig(chi=8, projector_backward="lorentzian").projector_backward
        == "lorentzian"
    )


def test_projector_backward_bogus_value_rejected():
    with pytest.raises((ValueError, TypeError), match="projector_backward"):
        CTMConfig(chi=8, projector_backward="bogus")
