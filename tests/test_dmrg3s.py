"""Tests for the DMRG3S subspace expansion module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.dmrg3s import build_expansion_tensor_dense


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
