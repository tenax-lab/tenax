"""Full D=2 spectrum: X and M, chi=10 against chi=16."""

import numpy as np
import pytest
from _ipeps_spectrum_fixtures import SX, SY_REAL, SZ, hasik_d2, rotated_gate

from tenax import SpectrumConfig, compute_spectrum

OPERATORS = {"Sy": SY_REAL, "Sx": SX, "Sz": SZ}


@pytest.mark.slow
@pytest.mark.parametrize(
    ("momentum", "omega_low", "weight_top"),
    [((np.pi, 0.0), 2.293, 0.1904), ((np.pi, np.pi), 0.6487, 0.7744)],
    ids=["X", "M"],
)
def test_full_spectrum(momentum, omega_low, weight_top):
    results = [
        compute_spectrum(
            hasik_d2(), rotated_gate(), momentum, OPERATORS, SpectrumConfig(chi=chi)
        )
        for chi in (10, 16)
    ]
    for result in results:
        assert result.omega[0] == pytest.approx(omega_low, abs=2e-3)
        assert result.weights["Sy"].max() == pytest.approx(weight_top, abs=2e-3)
        assert result.n_negative_modes == 0
        assert result.n_min_eig > -1e-4
        assert result.gauge_residual < 1e-3
        for deficit in result.sum_rule_deficit.values():
            assert abs(deficit) < 1e-3
    np.testing.assert_allclose(results[0].omega[:3], results[1].omega[:3], atol=1e-3)
