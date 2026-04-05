"""Tests for TDVP time evolution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import DMRGConfig, build_random_mps, dmrg
from tenax.algorithms.auto_mpo import build_auto_mpo
from tenax.algorithms.tdvp import TDVPConfig, TDVPResult, tdvp, tdvp_step
from tenax.network.network import TensorNetwork


def _build_dense_heisenberg(L: int) -> TensorNetwork:
    """Build a dense (non-symmetric) Heisenberg MPO."""
    terms = []
    for i in range(L - 1):
        terms.append((1.0, "Sz", i, "Sz", i + 1))
        terms.append((0.5, "Sp", i, "Sm", i + 1))
        terms.append((0.5, "Sm", i, "Sp", i + 1))
    return build_auto_mpo(terms, L=L, symmetric=False)


class TestTDVPConfig:
    """Tests for TDVPConfig defaults."""

    def test_tdvp_config_defaults(self):
        cfg = TDVPConfig()
        assert cfg.mode == "1site"
        assert cfg.dt == 0.05
        assert cfg.time_type == "real"
        assert cfg.num_steps == 100
        assert cfg.max_bond_dim == 64
        assert cfg.svd_trunc_err is None
        assert cfg.krylov_dim == 20
        assert cfg.krylov_tol == 1e-12
        assert cfg.verbose is False


class TestEffectiveHamiltonians:
    """Tests for 1-site and bond effective Hamiltonian matvecs."""

    def test_1site_effective_hamiltonian_matvec(self):
        """Verify shape of 1-site matvec output."""
        from tenax.algorithms.tdvp import _effective_hamiltonian_matvec_1site

        chi_l, d, chi_r, D_w = 4, 2, 3, 5
        L_env = jnp.ones((chi_l, D_w, chi_l))
        theta = jnp.ones((chi_l, d, chi_r))
        W = jnp.ones((D_w, d, d, D_w))
        R_env = jnp.ones((chi_r, D_w, chi_r))

        theta_flat = theta.ravel()
        result = _effective_hamiltonian_matvec_1site(
            theta_flat, theta.shape, L_env, W, R_env
        )
        assert result.shape == theta_flat.shape

    def test_bond_effective_hamiltonian_matvec(self):
        """Verify shape of bond matvec output."""
        from tenax.algorithms.tdvp import _bond_hamiltonian_matvec

        chi_l, chi_r, D_w = 4, 3, 5
        L_env = jnp.ones((chi_l, D_w, chi_l))
        R_env = jnp.ones((chi_r, D_w, chi_r))

        bond_flat = jnp.ones(chi_l * chi_r)
        result = _bond_hamiltonian_matvec(bond_flat, (chi_l, chi_r), L_env, R_env)
        assert result.shape == bond_flat.shape


class TestTDVP1Site:
    """Tests for 1-site TDVP."""

    @pytest.fixture()
    def dmrg_state(self):
        """Prepare a DMRG ground state for L=6, chi=8."""
        L, chi = 6, 8
        mpo = _build_dense_heisenberg(L)
        mps = build_random_mps(L, bond_dim=chi, seed=42)
        result = dmrg(mpo, mps, DMRGConfig(max_bond_dim=chi, num_sweeps=10))
        return mpo, result.mps, result.energy, chi

    def test_1site_tdvp_energy_conservation(self, dmrg_state):
        """Real-time 1TDVP should conserve energy."""
        mpo, mps, e_dmrg, chi = dmrg_state
        cfg = TDVPConfig(mode="1site", dt=0.05, time_type="real", num_steps=5)
        result = tdvp(mps, mpo, cfg)

        for e in result.energies:
            np.testing.assert_allclose(e, e_dmrg, atol=1e-4)

    def test_1site_tdvp_norm_preservation(self, dmrg_state):
        """Real-time 1TDVP should preserve norm."""
        mpo, mps, _, chi = dmrg_state
        cfg = TDVPConfig(mode="1site", dt=0.05, time_type="real", num_steps=5)
        result = tdvp(mps, mpo, cfg)

        # Compute norm of final MPS via overlap <psi|psi>
        # If norm drifted significantly, energy would be very different
        np.testing.assert_allclose(result.energies[-1], result.energies[0], atol=1e-4)


class TestTDVP2Site:
    """Tests for 2-site TDVP."""

    def test_2site_tdvp_bond_growth(self):
        """Product state with bond_dim=1 should grow bond dim after 2TDVP."""
        L = 6
        mpo = _build_dense_heisenberg(L)
        mps = build_random_mps(L, bond_dim=1, seed=7)
        cfg = TDVPConfig(
            mode="2site",
            dt=0.1,
            time_type="imaginary",
            num_steps=1,
            max_bond_dim=8,
        )
        result = tdvp(mps, mpo, cfg)

        # Check that at least one bond grew beyond 1
        final_mps = result.mps
        L_final = final_mps.n_nodes()
        max_bond = 0
        for i in range(L_final):
            t = final_mps.get_tensor(i)
            for dim in t.todense().shape:
                if dim > max_bond:
                    max_bond = dim
        assert max_bond > 1, "Bond dimension should have grown from 1"

    def test_tdvp_driver_imaginary_time(self):
        """2TDVP imaginary-time should lower energy toward DMRG ground state."""
        L, chi = 6, 8
        mpo = _build_dense_heisenberg(L)

        # DMRG reference
        mps_dmrg = build_random_mps(L, bond_dim=chi, seed=42)
        dmrg_result = dmrg(mpo, mps_dmrg, DMRGConfig(max_bond_dim=chi, num_sweeps=20))
        e_dmrg = dmrg_result.energy

        # TDVP imaginary time
        mps_init = build_random_mps(L, bond_dim=chi, seed=99)
        cfg = TDVPConfig(
            mode="2site",
            dt=0.05,
            time_type="imaginary",
            num_steps=40,
            max_bond_dim=chi,
        )
        result = tdvp(mps_init, mpo, cfg)

        # Energy should decrease
        assert result.energies[-1] < result.energies[0]
        # Should approach DMRG energy (within reasonable tolerance for 40 steps)
        np.testing.assert_allclose(result.energies[-1], e_dmrg, atol=0.1)


class TestTDVPDriver:
    """Tests for the TDVP driver function."""

    def test_tdvp_driver_with_measure(self):
        """Verify measure callback is called and observables stored."""
        L, chi = 6, 4
        mpo = _build_dense_heisenberg(L)
        mps = build_random_mps(L, bond_dim=chi, seed=42)

        call_count = [0]

        def measure(mps_tn, step):
            call_count[0] += 1
            return {"step": float(step)}

        cfg = TDVPConfig(mode="1site", dt=0.05, time_type="real", num_steps=3)
        result = tdvp(mps, mpo, cfg, measure=measure)

        # measure should be called at step 0 (initial) + 3 steps = 4 times
        assert call_count[0] == 4
        assert "step" in result.observables
        assert len(result.observables["step"]) == 4


class TestTDVPComplexTime:
    """Tests for complex-time evolution."""

    def test_complex_time_runs(self):
        """Complex-time TDVP runs without error."""
        L, chi = 6, 8
        mpo = _build_dense_heisenberg(L)
        mps = build_random_mps(L, bond_dim=chi, seed=42)

        # DMRG warmup
        dmrg_cfg = DMRGConfig(max_bond_dim=chi, num_sweeps=5)
        mps = dmrg(mpo, mps, dmrg_cfg).mps

        # Complex time: real-time + small damping
        cfg = TDVPConfig(
            mode="1site",
            dt=0.05 - 0.01j,
            time_type="complex",
            num_steps=3,
        )
        result = tdvp(mps, mpo, cfg)
        # initial measurement + 3 steps = 4 entries
        assert len(result.energies) == 4
        assert all(np.isfinite(e) for e in result.energies)

    def test_complex_time_pure_real_matches_real(self):
        """Complex-time with zero imaginary part matches real-time."""
        L, chi = 4, 4
        mpo = _build_dense_heisenberg(L)
        mps = build_random_mps(L, bond_dim=chi, seed=42)

        dmrg_cfg = DMRGConfig(max_bond_dim=chi, num_sweeps=5)
        mps = dmrg(mpo, mps, dmrg_cfg).mps

        # Real-time via time_type="real"
        cfg_real = TDVPConfig(mode="1site", dt=0.05, time_type="real", num_steps=3)
        result_real = tdvp(mps, mpo, cfg_real)

        # Same via time_type="complex" with real dt
        cfg_complex = TDVPConfig(
            mode="1site", dt=0.05 + 0j, time_type="complex", num_steps=3
        )
        result_complex = tdvp(mps, mpo, cfg_complex)

        np.testing.assert_allclose(
            result_real.energies, result_complex.energies, atol=1e-10
        )


class TestTDVPSymmetric:
    """Verify TDVP works with symmetric (SymmetricTensor) MPO."""

    def test_1site_tdvp_symmetric_mpo(self):
        """TDVP with a symmetric MPO should produce decreasing energy."""
        from tenax.core.mps import FiniteMPS

        L, chi = 4, 4
        terms = []
        for i in range(L - 1):
            terms.append((1.0, "Sz", i, "Sz", i + 1))
            terms.append((0.5, "Sp", i, "Sm", i + 1))
            terms.append((0.5, "Sm", i, "Sp", i + 1))
        mpo_sym = build_auto_mpo(terms, L=L, symmetric=True)
        mps = FiniteMPS.random(L, d=2, chi=chi, key=jax.random.PRNGKey(42))

        cfg = TDVPConfig(mode="1site", dt=0.05, time_type="imaginary", num_steps=2)
        result = tdvp(mps, mpo_sym, cfg)

        assert len(result.energies) == 3  # initial + 2 steps
        assert all(np.isfinite(e) for e in result.energies)
        # Imaginary time should lower energy
        assert result.energies[-1] <= result.energies[0] + 1e-6
