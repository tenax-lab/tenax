"""Tests for the numerically stable infinite-TEBD (iTEBD).

Validates against the same Bethe-ansatz reference as iDMRG: the spin-1/2
Heisenberg chain has e_0 = 1/4 - ln 2 ~= -0.4431 per site.
"""

from __future__ import annotations

import math

from tenax.algorithms.itebd import (
    heisenberg_2site_h,
    itebd_groundstate,
)


class TestITEBDHeisenberg:
    def test_energy_matches_bethe_ansatz(self):
        """iTEBD reaches e_0 = 1/4 - ln 2 for the Heisenberg chain."""
        H = heisenberg_2site_h(Jz=1.0, Jxy=1.0)
        e, _ = itebd_groundstate(H, chi_max=16, steps_per_dt=600)
        e_exact = 0.25 - math.log(2)
        assert abs(e - e_exact) < 5e-3, f"e={e}, exact={e_exact}"

    def test_xx_chain_energy(self):
        """The XX chain (Jz=0) is the 1D free-fermion point; e_0 = -1/pi."""
        H = heisenberg_2site_h(Jz=0.0, Jxy=1.0)
        e, _ = itebd_groundstate(H, chi_max=24, steps_per_dt=600)
        # XX chain ground-state energy per site is -1/pi ~= -0.3183.
        assert abs(e - (-1.0 / math.pi)) < 1e-2, f"e={e}"

    def test_stability_no_nan(self):
        """The stable canonicalization must not produce NaN even with many
        steps and aggressive truncation (the regime where the naive lambda^-1
        scheme blows up)."""
        import numpy as np

        H = heisenberg_2site_h(Jz=1.0, Jxy=1.0)
        e, st = itebd_groundstate(H, chi_max=8, steps_per_dt=1500)
        assert np.isfinite(e)
        assert np.all(np.isfinite(st.lA)) and np.all(np.isfinite(st.lB))
