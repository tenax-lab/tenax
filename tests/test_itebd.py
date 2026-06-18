"""Tests for the numerically stable infinite-TEBD (iTEBD).

Validates against the same Bethe-ansatz reference as iDMRG: the spin-1/2
Heisenberg chain has e_0 = 1/4 - ln 2 ~= -0.4431 per site.
"""

from __future__ import annotations

import math

import numpy as np

from tenax.algorithms.itebd import (
    heisenberg_2site_h,
    itebd_groundstate,
)


def _random_left_canonical(chiL, d, chiR, seed):
    """Random left-canonical tensor (chiL, d, chiR): reshape(chiL*d, chiR) isometry."""

    rng = np.random.RandomState(seed)
    m = rng.standard_normal((chiL * d, chiR))
    q, _ = np.linalg.qr(m)  # (chiL*d, chiR) with orthonormal columns
    return q.reshape(chiL, d, chiR)


class TestHastingsUpdate:
    def test_identity_gate_is_inversion_free_and_exact(self):
        """With the identity gate and no truncation, the Hastings update must
        return a left-canonical A_new and reproduce the original weighted block
        A·B·diag(l_right) exactly (A_new·B_new·diag(l_right))."""

        from tenax.algorithms.itebd import _update_bond_hastings

        chi, d = 3, 2
        A = _random_left_canonical(chi, d, chi, seed=1)
        B = _random_left_canonical(chi, d, chi, seed=2)
        l_right = np.abs(np.random.RandomState(3).standard_normal(chi))
        l_right = l_right / np.linalg.norm(l_right)

        # identity gate: einsum("aijc,ijkl->aklc") becomes identity
        ident = np.einsum("ik,jl->ijkl", np.eye(d), np.eye(d))

        A_new, S, B_new = _update_bond_hastings(A, B, l_right, ident, chi_max=chi * d)

        # A_new is left-canonical (isometry)
        k = A_new.shape[2]
        flat = A_new.reshape(-1, k)
        assert np.allclose(flat.conj().T @ flat, np.eye(k), atol=1e-10)

        # new bond weights normalized
        assert abs(np.linalg.norm(S) - 1.0) < 1e-10

        # reconstruction: A_new·B_new·diag(l_right) == A·B·diag(l_right)
        recon = np.einsum("aik,kjc,c->aijc", A_new, B_new, l_right)
        orig = np.einsum("aik,kjc,c->aijc", A, B, l_right)
        assert np.allclose(recon, orig, atol=1e-10)


class TestHastingsEnergy:
    def test_product_state_energies(self):
        """In left-canonical form, <H> on a chi=1 product state matches the
        analytic Heisenberg diagonal: |up up> -> +0.25, |up down> -> -0.25."""
        import numpy as np

        from tenax.algorithms.itebd import _bond_energy_left, heisenberg_2site_h

        H = heisenberg_2site_h(Jz=1.0, Jxy=1.0)
        up = np.array([1.0, 0.0])
        down = np.array([0.0, 1.0])
        l_right = np.array([1.0])

        def site(vec):
            return vec.reshape(1, 2, 1)  # (chiL=1, d=2, chiR=1), already normalized

        e_uu = _bond_energy_left(site(up), site(up), l_right, H)
        e_ud = _bond_energy_left(site(up), site(down), l_right, H)
        assert abs(e_uu - 0.25) < 1e-12, f"e_uu={e_uu}"
        assert abs(e_ud - (-0.25)) < 1e-12, f"e_ud={e_ud}"


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

        H = heisenberg_2site_h(Jz=1.0, Jxy=1.0)
        e, st = itebd_groundstate(H, chi_max=8, steps_per_dt=1500)
        assert np.isfinite(e)
        assert np.all(np.isfinite(st.lA)) and np.all(np.isfinite(st.lB))
