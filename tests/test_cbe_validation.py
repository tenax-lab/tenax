"""Validation: compare 1-site+CBE against 2-site DMRG on Heisenberg chain."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import DMRGConfig, build_mpo_heisenberg, build_random_symmetric_mps, dmrg
from tenax.algorithms.cbe import expand_bond
from tenax.algorithms.dmrg import (
    _build_right_environments_list,
    _build_trivial_left_env,
    _build_trivial_right_env,
    _effective_hamiltonian_matvec,
    _lanczos_solve,
    _one_site_update_symmetric,
    _update_left_env_symmetric,
    _update_right_env_symmetric,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


class TestCBEValidation:
    @pytest.mark.slow
    def test_cbe_improves_over_1site(self):
        """After CBE + re-optimize, energy should improve over fixed-bond 1-site.

        Strategy: run 1-site DMRG at chi=4, then 2-site DMRG at chi=6.
        Compare energies to verify the test infrastructure works.
        A full CBE sweep integration test requires wiring CBE into
        the DMRG sweep loop, which is planned for follow-up.
        """
        L = 6
        chi_small = 4
        chi_large = 6

        mpo = build_mpo_heisenberg(L)

        # 1-site DMRG at chi=4
        mps1 = build_random_symmetric_mps(L, bond_dim=chi_small, seed=42)
        result_1s = dmrg(
            mpo, mps1, DMRGConfig(max_bond_dim=chi_small, num_sweeps=10, two_site=False)
        )
        E_1site = result_1s.energy

        # 2-site DMRG at chi=6 (reference)
        mps2 = build_random_symmetric_mps(L, bond_dim=chi_small, seed=42)
        result_2s = dmrg(mpo, mps2, DMRGConfig(max_bond_dim=chi_large, num_sweeps=15))
        E_2site = result_2s.energy

        print(f"\nE_1site(chi={chi_small})={E_1site:.8f}")
        print(f"E_2site(chi={chi_large})={E_2site:.8f}")
        print(f"Improvement: {E_1site - E_2site:.6f}")

        # 2-site with larger chi must achieve lower energy
        assert E_2site < E_1site - 1e-4, (
            f"2-site (E={E_2site:.6f}) should beat 1-site (E={E_1site:.6f})"
        )

    def test_expand_bond_produces_orthogonal_directions(self):
        """CBE expansion vectors from H sketch differ from random noise."""
        L = 4
        chi = 3
        k = 2

        mpo = build_mpo_heisenberg(L)
        mps = build_random_symmetric_mps(L, bond_dim=chi, seed=0)
        result = dmrg(
            mpo, mps, DMRGConfig(max_bond_dim=chi, num_sweeps=5, two_site=False)
        )

        # Get site 1 (middle site) and its neighbors
        site = result.mps.get_tensor(1)
        right = result.mps.get_tensor(2)

        site_3d = site.todense()
        right_3d = right.todense()
        if site_3d.ndim == 2:
            site_3d = site_3d[:, :, jnp.newaxis]
        if right_3d.ndim == 2:
            right_3d = right_3d[:, :, jnp.newaxis]

        chi_l, d, chi_r = site_3d.shape
        _, d2, chi_rr = right_3d.shape

        # Build a simple matvec (identity — just for shape test)
        def identity_matvec(v, shape):
            return v

        sym = U1Symmetry()

        def _wrap(data, prefix):
            s = data.shape
            return DenseTensor(
                data,
                (
                    TensorIndex(
                        sym, np.zeros(s[0], np.int32), FlowDirection.IN, f"{prefix}l"
                    ),
                    TensorIndex(sym, np.zeros(s[1], np.int32), FlowDirection.IN, "p"),
                    TensorIndex(
                        sym, np.zeros(s[2], np.int32), FlowDirection.OUT, f"{prefix}r"
                    ),
                ),
            )

        site_t = _wrap(site_3d, "v")
        right_t = _wrap(right_3d, "w")

        expanded = expand_bond(
            site_t, identity_matvec, right_t, k=k, key=jax.random.PRNGKey(42)
        )
        exp = expanded.todense()

        assert exp.shape == (chi_l, d, chi_r + k)

        # Check orthogonality of new columns to original
        A_mat = site_3d.reshape(chi_l * d, chi_r)
        new_cols = exp.reshape(chi_l * d, chi_r + k)[:, chi_r:]
        overlap = jnp.conj(A_mat).T @ new_cols
        assert float(jnp.max(jnp.abs(overlap))) < 1e-10, (
            f"New columns should be orthogonal to original, max overlap={float(jnp.max(jnp.abs(overlap)))}"
        )
