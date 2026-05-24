"""gs_hz_max_iter field tests."""

import pytest

from tenax.algorithms.ipeps_config import iPEPSConfig


def test_gs_hz_max_iter_defaults_to_40():
    cfg = iPEPSConfig()
    assert cfg.gs_hz_max_iter == 40


def test_gs_hz_max_iter_custom():
    cfg = iPEPSConfig(gs_hz_max_iter=15)
    assert cfg.gs_hz_max_iter == 15


@pytest.mark.parametrize("bad", [0, -1, -40])
def test_gs_hz_max_iter_must_be_positive(bad):
    with pytest.raises(ValueError, match="gs_hz_max_iter"):
        iPEPSConfig(gs_hz_max_iter=bad)
