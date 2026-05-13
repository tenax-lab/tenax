"""gs_stall_recovery_retries field tests (issue #454)."""

import pytest

from tenax.algorithms.ipeps_config import iPEPSConfig


def test_gs_stall_recovery_retries_defaults_to_5():
    cfg = iPEPSConfig()
    assert cfg.gs_stall_recovery_retries == 5


def test_gs_stall_recovery_retries_must_be_non_negative():
    with pytest.raises(ValueError, match="gs_stall_recovery_retries"):
        iPEPSConfig(gs_stall_recovery_retries=-1)


def test_gs_stall_recovery_retries_zero_is_allowed():
    # 0 means: no resets allowed; first stall exits immediately.
    cfg = iPEPSConfig(gs_stall_recovery_retries=0)
    assert cfg.gs_stall_recovery_retries == 0
