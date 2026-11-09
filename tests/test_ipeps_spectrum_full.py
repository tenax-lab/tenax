"""Full D=2 spectrum: X and M, chi=10 against chi=16."""

import warnings

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
def test_full_spectrum_on_an_exact_u1_state():
    """Default settings, all 32 columns. Reference: FD-FD of every entry in a charge
    block at chi=10, h=1e-3. The Lorentzian gave 8e3 here, and 6e29 at width 1e-10.

    Roundoff in the rotation of a pair leaves asymmetries in entries with no weight, and
    their size depends on the roundoff. Over 9 changes of ``A`` by 3e-16 the H asymmetry is
    1.8e-4 to 9.6e-4 and the N asymmetry 1.6e-6 to 9.0e-6, with the numbers below the same
    to all digits. So the H asymmetry may warn here. No other warning may.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_spectrum(
            exact_u1_d2(),
            rotated_gate(),
            (np.pi, 0.0),
            OPERATORS,
            SpectrumConfig(chi=10),
        )
    messages = [str(w.message) for w in caught]
    assert not [
        m for m in messages if "compute_spectrum:" in m and "H asymmetry" not in m
    ]
    assert result.rank == 23
    assert result.n_negative_modes == 0
    assert result.h_asymmetry < 1e-2
    assert result.n_asymmetry < 1e-4
    np.testing.assert_allclose(result.omega[:3], [2.29373, 2.9606, 3.4618], atol=1e-4)
    assert result.structure_factor["Sy"] == pytest.approx(0.20435146, abs=1e-6)
    assert result.structure_factor["Sz"] == pytest.approx(0.09376483, abs=1e-6)
    first_moment = result.omega @ result.weights["Sy"]
    assert first_moment == pytest.approx(0.49878491, abs=2e-6)
    assert result.weights["Sy"].max() == pytest.approx(0.190746, abs=1e-5)
