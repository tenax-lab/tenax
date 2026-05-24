"""gs_chi_ceiling_bailout field tests."""

import pytest

from tenax.algorithms.ipeps_config import iPEPSConfig


def test_gs_chi_ceiling_bailout_defaults_to_zero():
    cfg = iPEPSConfig()
    assert cfg.gs_chi_ceiling_bailout == 0  # disabled


def test_gs_chi_ceiling_bailout_custom():
    cfg = iPEPSConfig(gs_chi_ceiling_bailout=3)
    assert cfg.gs_chi_ceiling_bailout == 3


def test_gs_chi_ceiling_bailout_negative_rejected():
    with pytest.raises(ValueError, match="gs_chi_ceiling_bailout"):
        iPEPSConfig(gs_chi_ceiling_bailout=-1)


def test_gs_chi_ceiling_bailout_zero_allowed():
    # 0 explicitly means disabled, not invalid.
    cfg = iPEPSConfig(gs_chi_ceiling_bailout=0)
    assert cfg.gs_chi_ceiling_bailout == 0
