"""Tests for the DMRG3S subspace expansion module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.dmrg3s import (
    adapt_alpha,
    build_expansion_tensor_dense,
    expand_and_truncate_dense,
)


def _heisenberg_terms(L):
    """Return Heisenberg term specs for build_auto_mpo."""
    terms = []
    for i in range(L - 1):
        terms.append((1.0, "Sz", i, "Sz", i + 1))
        terms.append((0.5, "Sp", i, "Sm", i + 1))
        terms.append((0.5, "Sm", i, "Sp", i + 1))
    return terms


class TestBuildExpansionTensorDense:
    @pytest.fixture()
    def heisenberg_setup(self):
        from tenax.algorithms.auto_mpo import build_auto_mpo
        from tenax.algorithms.dmrg import (
            _build_left_environments_list,
            _build_right_environments_list,
            _dense_ops,
        )
        from tenax.core.mps import FiniteMPS

        L, d, chi = 4, 2, 6
        key = jax.random.PRNGKey(42)
        mps = FiniteMPS.random(L, d, chi, key=key)
        terms = _heisenberg_terms(L)
        mpo = build_auto_mpo(terms, L)
        mps_t = list(mps.tensors)
        mpo_t = [mpo.get_tensor(i) for i in range(L)]
        ops = _dense_ops()
        left_envs = _build_left_environments_list(mps_t, mpo_t, L, ops)
        right_envs = [None] * (L + 1)
        right_envs[L] = ops.build_trivial_right_env()
        for j in range(L - 1, 0, -1):
            right_envs[j] = ops.update_right_env(right_envs[j + 1], mps_t[j], mpo_t[j])
        return mps_t, mpo_t, left_envs, right_envs

    def test_ltr_shape(self, heisenberg_setup):
        """L->R expansion tensor shape is (chi_L_bra, d_phys, w * chi_R)."""
        mps_t, mpo_t, left_envs, right_envs = heisenberg_setup
        # Use site 1 for L->R: left_env at 1, site tensor 1, MPO tensor 1
        site_idx = 1
        env = left_envs[site_idx]
        site = mps_t[site_idx]
        mpo_site = mpo_t[site_idx]

        P = build_expansion_tensor_dense(
            env, site, mpo_site, alpha=1e-3, direction="left_to_right"
        )

        # env shape: (chi_L_ket, w_L, chi_L_bra)
        # site shape: (chi_L_ket, d, chi_R_ket)
        # mpo shape: (w_L, d_ket, d_bra, w_R)
        chi_L_bra = env.todense().shape[2]
        d_phys = site.todense().shape[1]
        w_R = mpo_site.todense().shape[3]
        chi_R_ket = site.todense().shape[2]

        assert P.shape == (chi_L_bra, d_phys, w_R * chi_R_ket)

    def test_rtl_shape(self, heisenberg_setup):
        """R->L expansion tensor shape is (w * chi_L, d_phys, chi_R_bra)."""
        mps_t, mpo_t, left_envs, right_envs = heisenberg_setup
        # Use site 2 for R->L: right_env at 3, site tensor 2, MPO tensor 2
        site_idx = 2
        env = right_envs[site_idx + 1]
        site = mps_t[site_idx]
        mpo_site = mpo_t[site_idx]

        P = build_expansion_tensor_dense(
            env, site, mpo_site, alpha=1e-3, direction="right_to_left"
        )

        # env shape: (chi_R_ket, w_R, chi_R_bra)
        # site shape: (chi_L_ket, d, chi_R_ket)
        # mpo shape: (w_L, d_ket, d_bra, w_R)
        chi_R_bra = env.todense().shape[2]
        d_phys = site.todense().shape[1]
        w_L = mpo_site.todense().shape[0]
        chi_L_ket = site.todense().shape[0]

        assert P.shape == (w_L * chi_L_ket, d_phys, chi_R_bra)

    def test_scales_with_alpha(self, heisenberg_setup):
        """P(2*alpha) should have twice the norm of P(alpha)."""
        mps_t, mpo_t, left_envs, right_envs = heisenberg_setup
        site_idx = 1
        env = left_envs[site_idx]
        site = mps_t[site_idx]
        mpo_site = mpo_t[site_idx]

        P1 = build_expansion_tensor_dense(
            env, site, mpo_site, alpha=1e-3, direction="left_to_right"
        )
        P2 = build_expansion_tensor_dense(
            env, site, mpo_site, alpha=2e-3, direction="left_to_right"
        )

        ratio = jnp.linalg.norm(P2) / jnp.linalg.norm(P1)
        np.testing.assert_allclose(float(ratio), 2.0, atol=1e-10)

    def test_zero_alpha_gives_zero(self, heisenberg_setup):
        """P should be all zeros when alpha=0."""
        mps_t, mpo_t, left_envs, right_envs = heisenberg_setup
        site_idx = 1
        env = left_envs[site_idx]
        site = mps_t[site_idx]
        mpo_site = mpo_t[site_idx]

        P = build_expansion_tensor_dense(
            env, site, mpo_site, alpha=0.0, direction="left_to_right"
        )

        np.testing.assert_allclose(P, 0.0, atol=1e-15)


class TestExpandAndTruncateDense:
    @pytest.fixture()
    def heisenberg_setup(self):
        from tenax.algorithms.auto_mpo import build_auto_mpo
        from tenax.algorithms.dmrg import (
            _build_left_environments_list,
            _build_right_environments_list,
            _dense_ops,
        )
        from tenax.core.mps import FiniteMPS

        L, d, chi = 4, 2, 6
        key = jax.random.PRNGKey(42)
        mps = FiniteMPS.random(L, d, chi, key=key)
        terms = _heisenberg_terms(L)
        mpo = build_auto_mpo(terms, L)
        mps_t = list(mps.tensors)
        mpo_t = [mpo.get_tensor(i) for i in range(L)]
        ops = _dense_ops()
        left_envs = _build_left_environments_list(mps_t, mpo_t, L, ops)
        right_envs = [None] * (L + 1)
        right_envs[L] = ops.build_trivial_right_env()
        for j in range(L - 1, 0, -1):
            right_envs[j] = ops.update_right_env(right_envs[j + 1], mps_t[j], mpo_t[j])
        return mps_t, mpo_t, left_envs, right_envs

    def test_ltr_bond_dim_preserved(self, heisenberg_setup):
        """After expand + truncate, bond dim <= max_bond_dim."""
        mps_t, mpo_t, left_envs, _ = heisenberg_setup
        i = 1
        A, B = expand_and_truncate_dense(
            mps_t[i].todense(),
            mps_t[i + 1].todense(),
            left_envs[i],
            mpo_t[i],
            alpha=1e-3,
            max_bond_dim=8,
            direction="left_to_right",
        )
        assert A.shape[-1] <= 8
        assert B.shape[0] == A.shape[-1]

    def test_rtl_bond_dim_preserved(self, heisenberg_setup):
        """R->L: bond dim <= max_bond_dim, dims match."""
        mps_t, mpo_t, _, right_envs = heisenberg_setup
        i = 2
        B, A = expand_and_truncate_dense(
            mps_t[i].todense(),
            mps_t[i - 1].todense(),
            right_envs[i + 1],
            mpo_t[i],
            alpha=1e-3,
            max_bond_dim=8,
            direction="right_to_left",
        )
        assert B.shape[0] <= 8
        assert A.shape[-1] == B.shape[0]

    def test_alpha_zero_preserves_state(self, heisenberg_setup):
        """With alpha=0, the two-site state is unchanged."""
        mps_t, mpo_t, left_envs, _ = heisenberg_setup
        i = 1
        M = mps_t[i].todense()
        B_orig = mps_t[i + 1].todense()
        A, B = expand_and_truncate_dense(
            M,
            B_orig,
            left_envs[i],
            mpo_t[i],
            alpha=0.0,
            max_bond_dim=M.shape[-1],
            direction="left_to_right",
        )
        theta_orig = jnp.einsum("apd,dqf->apqf", M, B_orig)
        theta_new = jnp.einsum("apd,dqf->apqf", A, B)
        np.testing.assert_allclose(theta_orig, theta_new, atol=1e-10)


class TestAdaptAlpha:
    def test_increases_when_truncation_small(self):
        new = adapt_alpha(alpha=1e-3, delta_e_opt=-1.0, delta_e_trunc=0.01)
        assert new > 1e-3

    def test_decreases_when_truncation_large(self):
        new = adapt_alpha(alpha=1e-3, delta_e_opt=-1.0, delta_e_trunc=0.9)
        assert new < 1e-3

    def test_stable_at_target_ratio(self):
        new = adapt_alpha(alpha=1e-3, delta_e_opt=-1.0, delta_e_trunc=0.3)
        assert new == 1e-3

    def test_no_change_when_converged(self):
        new = adapt_alpha(alpha=1e-3, delta_e_opt=0.0, delta_e_trunc=0.0)
        assert new == 1e-3


class TestDMRG3SIntegration:
    """Test DMRG with subspace expansion wired into sweep loop."""

    def test_1site_with_expansion_runs(self):
        """1-site + DMRG3S should complete without error."""
        from tenax.algorithms.auto_mpo import build_auto_mpo
        from tenax.algorithms.dmrg import DMRGConfig, dmrg
        from tenax.core.mps import FiniteMPS

        L, d, chi = 6, 2, 8
        key = jax.random.PRNGKey(0)
        mps = FiniteMPS.random(L, d, chi, key=key)
        mpo = build_auto_mpo(_heisenberg_terms(L), L)
        result = dmrg(
            mpo,
            mps,
            DMRGConfig(
                max_bond_dim=chi,
                num_sweeps=4,
                two_site=False,
                subspace_expansion=True,
                mixing_factor=1e-3,
            ),
        )
        assert result.energy < 0  # should find a negative energy ground state

    def test_expansion_improves_over_plain_1site(self):
        """1-site + DMRG3S should get closer to 2-site energy."""
        from tenax.algorithms.auto_mpo import build_auto_mpo
        from tenax.algorithms.dmrg import DMRGConfig, dmrg
        from tenax.core.mps import FiniteMPS

        L, d, chi = 8, 2, 8
        key = jax.random.PRNGKey(42)
        mps = FiniteMPS.random(L, d, chi, key=key)
        mpo = build_auto_mpo(_heisenberg_terms(L), L)

        r_2site = dmrg(
            mpo, mps, DMRGConfig(max_bond_dim=chi, num_sweeps=10, two_site=True)
        )
        r_1site = dmrg(
            mpo, mps, DMRGConfig(max_bond_dim=chi, num_sweeps=10, two_site=False)
        )
        r_3s = dmrg(
            mpo,
            mps,
            DMRGConfig(
                max_bond_dim=chi,
                num_sweeps=10,
                two_site=False,
                subspace_expansion=True,
                mixing_factor=1e-3,
            ),
        )

        gap_plain = abs(r_1site.energy - r_2site.energy)
        gap_3s = abs(r_3s.energy - r_2site.energy)
        assert gap_3s < gap_plain or gap_3s < 1e-6

    def test_subspace_expansion_with_two_site_raises(self):
        """subspace_expansion=True + two_site=True should raise."""
        from tenax.algorithms.auto_mpo import build_auto_mpo
        from tenax.algorithms.dmrg import DMRGConfig, dmrg
        from tenax.core.mps import FiniteMPS

        key = jax.random.PRNGKey(0)
        mps = FiniteMPS.random(4, 2, 4, key=key)
        mpo = build_auto_mpo(_heisenberg_terms(4), 4)
        with pytest.raises(ValueError, match="subspace_expansion"):
            dmrg(mpo, mps, DMRGConfig(two_site=True, subspace_expansion=True))


@pytest.mark.slow
class TestDMRG3SBenchmark:
    """Benchmark: DMRG3S convergence improvement over plain 1-site."""

    def test_3s_no_worse_than_plain_1site(self):
        """L=12 Heisenberg: DMRG3S should not degrade 1-site energy."""
        from tenax.algorithms.auto_mpo import build_auto_mpo
        from tenax.algorithms.dmrg import DMRGConfig, dmrg
        from tenax.core.mps import FiniteMPS

        L, d, chi = 12, 2, 16
        key = jax.random.PRNGKey(42)
        mpo = build_auto_mpo(_heisenberg_terms(L), L)

        mps = FiniteMPS.random(L, d, chi, key=key)
        e_1site = dmrg(
            mpo,
            mps,
            DMRGConfig(max_bond_dim=chi, num_sweeps=15, two_site=False),
        ).energy

        mps = FiniteMPS.random(L, d, chi, key=key)
        e_3s = dmrg(
            mpo,
            mps,
            DMRGConfig(
                max_bond_dim=chi,
                num_sweeps=15,
                two_site=False,
                subspace_expansion=True,
                mixing_factor=1e-2,
            ),
        ).energy

        # DMRG3S should be at least as good as plain 1-site (within noise)
        assert e_3s <= e_1site + 1e-6
