"""Test numpy Lanczos eigensolver (_lanczos_solve_np) against JAX Lanczos."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tenax.algorithms._block_array import ba_to_symmetric, symmetric_to_ba
from tenax.algorithms.dmrg import (
    _blockwise_contract,
    _build_trivial_left_env_symmetric,
    _build_trivial_right_env_symmetric,
    _lanczos_solve_np,
    _lanczos_solve_tensor,
    _precompute_block_plan,
    _to_np_blocks,
    _update_right_env_symmetric,
    build_mpo_heisenberg,
)
from tenax.contraction.contractor import contract
from tenax.core.mps import FiniteMPS
from tenax.core.symmetry import U1Symmetry


class TestLanczosNp:
    """Compare _lanczos_solve_np (BlockArray) vs _lanczos_solve_tensor (Tensor)."""

    def test_np_lanczos_matches_jax_lanczos(self):
        """Numpy Lanczos finds the same ground-state energy as JAX Lanczos."""
        L = 4
        sym = U1Symmetry()

        # Build symmetric Heisenberg MPO
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0, hz=0.0)

        # Build random symmetric MPS
        key = jax.random.PRNGKey(0)
        mps = FiniteMPS.random(
            L,
            d=2,
            chi=4,
            key=key,
            symmetric=True,
            symmetry=sym,
            target_charge=0,
        )

        # Build trivial boundary environments
        l_env = _build_trivial_left_env_symmetric()
        r_env = _build_trivial_right_env_symmetric()

        # Build right environments by sweeping from right
        right_envs = [None] * L
        right_envs[L - 1] = r_env
        for i in range(L - 1, 0, -1):
            right_envs[i - 1] = _update_right_env_symmetric(
                right_envs[i], mps[i], mpo.get_tensor(i)
            )

        # 2-site theta = contract(mps[0], mps[1])
        theta = contract(mps[0], mps[1])

        # Shared caches
        _subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        _plan = _precompute_block_plan(
            [l_env, theta, mpo.get_tensor(0), mpo.get_tensor(1), right_envs[1]],
            _subs,
        )
        _expr_cache_jax = {}
        _expr_cache_np = {}

        # Pre-convert env blocks for reuse
        _env_np = [
            _to_np_blocks(l_env),
            None,  # theta slot — converted each call
            _to_np_blocks(mpo.get_tensor(0)),
            _to_np_blocks(mpo.get_tensor(1)),
            _to_np_blocks(right_envs[1]),
        ]

        # JAX matvec (returns SymmetricTensor)
        def matvec_jax(v):
            return _blockwise_contract(
                [l_env, v, mpo.get_tensor(0), mpo.get_tensor(1), right_envs[1]],
                _subs,
                output_indices=v.indices,
                expr_cache=_expr_cache_jax,
                block_plan=_plan,
                np_blocks_cache=_env_np,
            )

        # numpy matvec (BlockArray in, BlockArray out)
        def matvec_np(ba):
            v = ba_to_symmetric(ba)
            result = _blockwise_contract(
                [l_env, v, mpo.get_tensor(0), mpo.get_tensor(1), right_envs[1]],
                _subs,
                output_indices=v.indices,
                expr_cache=_expr_cache_np,
                block_plan=_plan,
                np_blocks_cache=_env_np,
                return_ba=True,
            )
            return result

        # Run JAX Lanczos
        e_jax, _ = _lanczos_solve_tensor(matvec_jax, theta, 20, 1e-12)

        # Run numpy Lanczos
        theta_ba = symmetric_to_ba(theta)
        e_np, _ = _lanczos_solve_np(matvec_np, theta_ba, 20, 1e-12)

        np.testing.assert_allclose(e_np, e_jax, rtol=1e-10)
