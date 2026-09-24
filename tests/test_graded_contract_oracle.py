"""The reference graded contractor against #1038's exact Fock oracle (#1035).

Design §3.4 showed, in plain numpy, that three rules reproduce Fock with
nothing fitted.  These tests make the same claim about tenax's own
``SymmetricTensor`` + ``tenax.core._graded`` -- including the per-site order
a CTM builds, odd tensors, and an SVD regauge (the design's go/no-go).
"""

from __future__ import annotations

import _graded_cluster
import numpy as np
import pytest
from _fermionic_fock_oracle import (
    fock_psi,
    fock_psi_ordered,
    hop_energy,
    plain_amplitudes,
    random_even_tensors,
    z_gauge,
)
from _graded_cluster import cluster_energy

CLUSTERS = [(2, 2), pytest.param(2, 3, marks=pytest.mark.slow)]


def test_the_ordered_fock_state_is_the_oracle_state_for_even_tensors():
    As = random_even_tensors(2, 2, np.random.default_rng(0))
    np.testing.assert_allclose(
        fock_psi_ordered(2, 2, As), fock_psi(2, 2, As), atol=1e-14
    )


@pytest.mark.parametrize("order", ["global", "per_site"])
@pytest.mark.parametrize("R,C", CLUSTERS)
def test_graded_energy_matches_fock(R, C, order):
    rng = np.random.default_rng(7)
    for _ in range(2):
        As = random_even_tensors(R, C, rng)
        E, norm = cluster_energy(R, C, As, order=order)
        E_fock = hop_energy(R, C, fock_psi(R, C, z_gauge(R, C, As)), fermion=True)
        E_hcb = hop_energy(R, C, plain_amplitudes(R, C, As), fermion=False)
        assert norm > 0
        assert E == pytest.approx(E_fock, abs=1e-12)
        assert abs(E - E_hcb) > 1e-3  # regime: the bosonic answer is different


def test_graded_energy_matches_fock_for_complex_tensors():
    rng = np.random.default_rng(4)
    re, im = random_even_tensors(2, 2, rng), random_even_tensors(2, 2, rng)
    As = {s: re[s] + 1j * im[s] for s in re}
    E, norm = cluster_energy(2, 2, As)
    E_fock = hop_energy(2, 2, fock_psi(2, 2, z_gauge(2, 2, As)), fermion=True)
    assert norm > 0
    assert E == pytest.approx(E_fock, abs=1e-12)


def test_regime_an_ungraded_bar_misses_the_oracle(monkeypatch):
    """Rule 3 is load-bearing: today's ``bar()`` in place of ``graded_bar``."""
    monkeypatch.setattr(_graded_cluster, "graded_bar", lambda t: t.bar())
    As = random_even_tensors(2, 2, np.random.default_rng(7))
    E, _ = cluster_energy(2, 2, As)
    E_fock = hop_energy(2, 2, fock_psi(2, 2, z_gauge(2, 2, As)), fermion=True)
    assert abs(E - E_fock) > 1e-3
