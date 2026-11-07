"""Full D=2 spectrum: X and M, chi=10 against chi=16."""

import numpy as np
import pytest
from _ipeps_spectrum_fixtures import (
    SX,
    SY_REAL,
    SZ,
    exact_u1_d2,
    hasik_d2,
    rotated_gate,
)

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


@pytest.mark.slow
@pytest.mark.filterwarnings("error:compute_spectrum")
def test_full_spectrum_on_an_exact_u1_state():
    """Default settings, all 32 columns. Reference: FD-FD of every entry in a charge
    block at chi=10, h=1e-3. The Lorentzian gave 8e3 here, and 6e29 at width 1e-10.

    Roundoff in the rotation of a pair leaves 2e-4 in entries with no weight.
    """
    result = compute_spectrum(
        exact_u1_d2(),
        rotated_gate(),
        (np.pi, 0.0),
        OPERATORS,
        SpectrumConfig(chi=10),
    )
    assert result.rank == 23
    assert result.n_negative_modes == 0
    assert result.h_asymmetry < 1e-3
    assert result.n_asymmetry < 1e-5
    np.testing.assert_allclose(result.omega[:3], [2.29373, 2.9606, 3.4618], atol=1e-4)
    assert result.structure_factor["Sy"] == pytest.approx(0.20435146, abs=1e-6)
    assert result.structure_factor["Sz"] == pytest.approx(0.09376483, abs=1e-6)
    first_moment = result.omega @ result.weights["Sy"]
    assert first_moment == pytest.approx(0.49878491, abs=2e-6)
    assert result.weights["Sy"].max() == pytest.approx(0.190746, abs=1e-5)
