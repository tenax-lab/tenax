"""Tests for iPEPSConfig new fields (issue #298)."""

from tenax.algorithms.ipeps_config import iPEPSConfig


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
