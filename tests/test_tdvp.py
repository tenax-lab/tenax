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
from tenax.core.mps import FiniteMPS
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


# ------------------------------------------------------------------ #
# #942 / #943: the 2-site integrator against exact evolution           #
# ------------------------------------------------------------------ #
#
# Nothing above compares a 2-site evolved STATE against exact evolution, and
# the invariants that are checked are provably blind to both defects: any
# exp(-i c dt H) conserves <H> whatever c is (#942 multiplied the time by the
# bond count), and the imaginary-time path renormalizes away the exp(-dt H)
# scale that #943 leaked into "complex".  These oracles are marked ``core``:
# a state evolved by the wrong time or the wrong exponent is a shipped wrong
# number, and every case here is a <=16-dimensional exact computation.


def _product_plus_state(L: int) -> FiniteMPS:
    from tenax.algorithms.tdvp import _make_site_tensor

    sites = [
        _make_site_tensor(jnp.ones((1, 2, 1)) / np.sqrt(2.0), i, L) for i in range(L)
    ]
    return FiniteMPS.from_tensors(sites)


def _sz_at_site_1_mpo(L: int, mps: FiniteMPS) -> TensorNetwork:
    """MPO for the single local term H = Sz_1."""
    from tenax.algorithms.tdvp import _identity_mpo_site
    from tenax.core.tensor import DenseTensor

    sz = np.diag([0.5, -0.5])
    mpo = TensorNetwork()
    for i in range(L):
        w = _identity_mpo_site(mps.get_tensor(i))
        if i == 1:
            w = DenseTensor(jnp.array(sz.reshape(1, 2, 2, 1)), w.indices)
        mpo.add_node(i, w)
    return mpo


def _statevector(mps: FiniteMPS) -> np.ndarray:
    psi = np.asarray(mps.get_tensor(0).todense())
    for i in range(1, mps.n_nodes()):
        psi = np.tensordot(psi, np.asarray(mps.get_tensor(i).todense()), axes=(-1, 0))
    return psi.reshape(-1)


def _sz1_diagonal(L: int) -> np.ndarray:
    diag = np.ones(1)
    for i in range(L):
        # np.array, NOT np.diag: np.diag(vector) *constructs* a matrix, and
        # the resulting exact vector silently broadcasts to a matrix whose
        # distance from psi is coincidentally the size of the #942 defect.
        diag = np.kron(diag, np.array([0.5, -0.5]) if i == 1 else np.ones(2))
    assert diag.shape == (2**L,)
    return diag


class TestTwoSiteIntegrator942:
    @pytest.mark.core
    @pytest.mark.parametrize("L", [3, 4])
    def test_a_local_field_advances_by_dt_not_by_bond_count_times_dt(self, L):
        """H = Sz_1, product state: exact evolution stays in the product
        manifold, so truncation explains nothing.  Without the backward
        one-site steps every bond re-applied the term, advancing site 1 by
        (L-1)*dt (#942)."""
        dt = 0.1
        mps = _product_plus_state(L)
        out = tdvp_step(
            mps,
            _sz_at_site_1_mpo(L, mps),
            TDVPConfig(mode="2site", time_type="real", dt=dt, max_bond_dim=8),
        )
        psi = _statevector(out)
        initial = np.ones(2**L) / np.sqrt(2**L)
        diag = _sz1_diagonal(L)
        exact = np.exp(-1j * dt * diag) * initial
        wrong = np.exp(-1j * dt * (L - 1) * diag) * initial
        assert np.linalg.norm(psi - exact) < 1e-12
        # Anti-vacuous: the defect this guards is far away, not inside tol.
        assert np.linalg.norm(psi - wrong) > 0.04

    @pytest.mark.core
    def test_full_bond_heisenberg_matches_dense_expm(self):
        """At full bond dimension every effective Hamiltonian is H itself, so
        the projector splitting is exact and 2TDVP must reproduce dense
        exp(-i dt H) to Krylov/roundoff -- entangled dynamics, all bonds, both
        sweeps, no truncation excuse.  Codex's independent full-bond check saw
        the same (L-1)-fold excess evolution here before the fix."""
        L, dt = 4, 0.05
        mps = _product_plus_state(L)
        mpo = _build_dense_heisenberg(L)
        out = tdvp_step(
            mps,
            mpo,
            TDVPConfig(mode="2site", time_type="real", dt=dt, max_bond_dim=4),
        )
        psi = _statevector(out)

        Sz = np.diag([0.5, -0.5])
        Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = Sp.T
        I2 = np.eye(2)

        def kron_chain(ops):
            m = ops[0]
            for o in ops[1:]:
                m = np.kron(m, o)
            return m

        H = np.zeros((2**L, 2**L))
        for i in range(L - 1):
            for coeff, oi, oj in [(1.0, Sz, Sz), (0.5, Sp, Sm), (0.5, Sm, Sp)]:
                ops = [I2] * L
                ops[i], ops[i + 1] = oi, oj
                H += coeff * kron_chain(ops)
        evals, evecs = np.linalg.eigh(H)
        initial = np.ones(2**L) / np.sqrt(2**L)
        exact = evecs @ (np.exp(-1j * dt * evals) * (evecs.conj().T @ initial))
        assert np.linalg.norm(psi - exact) < 1e-8


class TestTwoSiteComplexTime943:
    @pytest.mark.core
    def test_complex_time_type_is_a_timestep_not_imaginary_evolution(self):
        """L=2 excludes #942 (a single bond has no backward step): a purely
        real complex timestep must give exp(-i dt H), and the pre-fix
        exp(-dt H) must be far away."""
        dt = 0.05 + 0j
        mps = _product_plus_state(2)
        out = tdvp_step(
            mps,
            _sz_at_site_1_mpo(2, mps),
            TDVPConfig(mode="2site", time_type="complex", dt=dt, max_bond_dim=8),
        )
        psi = _statevector(out)
        initial = np.ones(4) / 2.0
        diag = _sz1_diagonal(2)
        exact = np.exp(-1j * dt * diag) * initial
        wrong = np.exp(-dt * diag) * initial
        assert np.linalg.norm(psi - exact) < 1e-12
        assert np.linalg.norm(psi - wrong) > 0.03

    @pytest.mark.core
    def test_a_genuinely_complex_timestep_rotates_and_damps(self):
        """dt = a - ib must produce exp(-i dt H): rotation from a, damping
        from b -- distinguishing the exponent from both pure conventions."""
        dt = 0.05 - 0.02j
        mps = _product_plus_state(2)
        out = tdvp_step(
            mps,
            _sz_at_site_1_mpo(2, mps),
            TDVPConfig(mode="2site", time_type="complex", dt=dt, max_bond_dim=8),
        )
        psi = _statevector(out)
        initial = np.ones(4) / 2.0
        exact = np.exp(-1j * dt * _sz1_diagonal(2)) * initial
        assert np.linalg.norm(psi - exact) < 1e-12
