"""gs_grad_spike_ratio / gs_grad_spike_window field tests."""

import pytest

from tenax.algorithms.ipeps_config import iPEPSConfig


def test_gs_grad_spike_defaults():
    cfg = iPEPSConfig()
    assert cfg.gs_grad_spike_ratio is None  # off by default
    assert cfg.gs_grad_spike_window == 5


def test_gs_grad_spike_ratio_custom():
    cfg = iPEPSConfig(gs_grad_spike_ratio=5.0, gs_grad_spike_window=3)
    assert cfg.gs_grad_spike_ratio == 5.0
    assert cfg.gs_grad_spike_window == 3


@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -1.0])
def test_gs_grad_spike_ratio_must_exceed_one(bad_ratio):
    with pytest.raises(ValueError, match="gs_grad_spike_ratio"):
        iPEPSConfig(gs_grad_spike_ratio=bad_ratio)


@pytest.mark.parametrize("bad_window", [0, -1])
def test_gs_grad_spike_window_must_be_positive(bad_window):
    with pytest.raises(ValueError, match="gs_grad_spike_window"):
        iPEPSConfig(gs_grad_spike_window=bad_window)
