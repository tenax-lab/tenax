"""Tests for JIT-compatible dense environment update functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._jit_sweep import (
    _build_initial_left_envs,
    _build_initial_right_envs,
    _scan_left_to_right,
    _scan_right_to_left,
    effective_ham_matvec_blocks,
    effective_ham_matvec_dense,
    jit_dmrg_sweep_dense,
    lanczos_ground_state_blocks,
    lanczos_ground_state_dense,
    update_left_env_dense_jit,
    update_right_env_dense_jit,
)
from tenax.algorithms.auto_mpo import build_auto_mpo
from tenax.algorithms.dmrg import (
    DMRGConfig,
    _build_trivial_left_env,
    _build_trivial_right_env,
    _effective_hamiltonian_matvec,
    _update_left_env,
    _update_right_env,
    build_random_mps,
    dmrg,
)


def _build_dense_heisenberg(L: int):
    """Build a dense (non-symmetric) Heisenberg MPO as a TensorNetwork."""
    terms = []
    for i in range(L - 1):
        terms.append((1.0, "Sz", i, "Sz", i + 1))
        terms.append((0.5, "Sp", i, "Sm", i + 1))
        terms.append((0.5, "Sm", i, "Sp", i + 1))
    mpo_tn = build_auto_mpo(terms, L=L, symmetric=False)
    return mpo_tn


def _build_test_system(L: int = 6, bond_dim: int = 8, seed: int = 42):
    """Build an MPS + MPO test system and return raw arrays + Tensor objects."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=bond_dim, seed=seed)
    mpo_tn = _build_dense_heisenberg(L)

    mps_tensors = [mps_tn.get_tensor(i) for i in range(L)]
    mpo_tensors = [mpo_tn.get_tensor(i) for i in range(L)]

    return mps_tensors, mpo_tensors


class TestDenseLeftEnvUpdate:
    """Test update_left_env_dense_jit matches _update_left_env from dmrg.py."""

    def test_dense_left_env_update_matches_python(self):
        """Padded JIT left env update must match the existing Python implementation."""
        L = 6
        bond_dim = 8
        chi_max = 12  # larger than bond_dim to test padding

        mps_tensors, mpo_tensors = _build_test_system(L=L, bond_dim=bond_dim)

        # Start from trivial left env
        left_env_tensor = _build_trivial_left_env(dtype=jnp.float64)

        # Walk a few sites to build up a non-trivial environment
        for site_idx in range(3):
            # Python reference
            new_left_tensor = _update_left_env(
                left_env_tensor, mps_tensors[site_idx], mpo_tensors[site_idx]
            )

            # JIT version on raw arrays
            L_env_raw = left_env_tensor.todense()
            A_raw = mps_tensors[site_idx].todense()
            W_raw = mpo_tensors[site_idx].todense()
            D_w = W_raw.shape[3]  # output MPO bond dimension

            new_left_padded = update_left_env_dense_jit(
                L_env_raw, A_raw, W_raw, chi_max
            )

            # Check output shape is padded
            assert new_left_padded.shape == (chi_max, D_w, chi_max), (
                f"Expected shape ({chi_max}, {D_w}, {chi_max}), "
                f"got {new_left_padded.shape}"
            )

            # Extract the unpadded region and compare
            ref = new_left_tensor.todense()
            chi_r = ref.shape[0]
            actual = new_left_padded[:chi_r, :D_w, :chi_r]
            np.testing.assert_allclose(
                actual,
                ref,
                atol=1e-12,
                err_msg=f"Left env mismatch at site {site_idx}",
            )

            # Check padding is zero
            # Rows beyond chi_r
            assert jnp.allclose(new_left_padded[chi_r:, :, :], 0.0), (
                "Padding rows should be zero"
            )
            # Cols beyond chi_r
            assert jnp.allclose(new_left_padded[:, :, chi_r:], 0.0), (
                "Padding cols should be zero"
            )

            # Advance for next iteration
            left_env_tensor = new_left_tensor

    def test_left_env_jit_compilable(self):
        """update_left_env_dense_jit must be JIT-compilable."""
        L = 4
        bond_dim = 4
        chi_max = 8
        mps_tensors, mpo_tensors = _build_test_system(L=L, bond_dim=bond_dim)

        left_env_tensor = _build_trivial_left_env(dtype=jnp.float64)
        L_env_raw = left_env_tensor.todense()
        A_raw = mps_tensors[0].todense()
        W_raw = mpo_tensors[0].todense()

        # Should compile without error
        jit_fn = jax.jit(update_left_env_dense_jit, static_argnums=(3,))
        result = jit_fn(L_env_raw, A_raw, W_raw, chi_max)
        assert result.shape[0] == chi_max

    def test_left_env_no_padding_needed(self):
        """When chi_max equals actual bond dim, no padding is needed."""
        L = 6
        bond_dim = 8
        chi_max = bond_dim  # exact match

        mps_tensors, mpo_tensors = _build_test_system(L=L, bond_dim=bond_dim)
        left_env_tensor = _build_trivial_left_env(dtype=jnp.float64)

        # Update at site 1 (interior site with full bond dim on both sides)
        left_env_tensor = _update_left_env(
            left_env_tensor, mps_tensors[0], mpo_tensors[0]
        )
        # Now left_env has shape (bond_dim, D_w, bond_dim) at site 1

        L_env_raw = left_env_tensor.todense()
        A_raw = mps_tensors[1].todense()
        W_raw = mpo_tensors[1].todense()
        D_w = W_raw.shape[3]

        new_left_padded = update_left_env_dense_jit(L_env_raw, A_raw, W_raw, chi_max)

        ref = _update_left_env(
            left_env_tensor, mps_tensors[1], mpo_tensors[1]
        ).todense()
        chi_r = ref.shape[0]
        actual = new_left_padded[:chi_r, :D_w, :chi_r]
        np.testing.assert_allclose(actual, ref, atol=1e-12)


class TestDenseRightEnvUpdate:
    """Test update_right_env_dense_jit matches _update_right_env from dmrg.py."""

    def test_dense_right_env_update_matches_python(self):
        """Padded JIT right env update must match the existing Python implementation."""
        L = 6
        bond_dim = 8
        chi_max = 12

        mps_tensors, mpo_tensors = _build_test_system(L=L, bond_dim=bond_dim)

        # Start from trivial right env
        right_env_tensor = _build_trivial_right_env(dtype=jnp.float64)

        # Walk from right boundary inward
        for site_idx in range(L - 1, L - 4, -1):
            # Python reference
            new_right_tensor = _update_right_env(
                right_env_tensor, mps_tensors[site_idx], mpo_tensors[site_idx]
            )

            # JIT version on raw arrays
            R_env_raw = right_env_tensor.todense()
            B_raw = mps_tensors[site_idx].todense()
            W_raw = mpo_tensors[site_idx].todense()
            D_w = W_raw.shape[0]  # left MPO bond dimension (output)

            new_right_padded = update_right_env_dense_jit(
                R_env_raw, B_raw, W_raw, chi_max
            )

            # Check output shape is padded
            assert new_right_padded.shape == (chi_max, D_w, chi_max), (
                f"Expected shape ({chi_max}, {D_w}, {chi_max}), "
                f"got {new_right_padded.shape}"
            )

            # Extract unpadded region and compare
            ref = new_right_tensor.todense()
            chi_l = ref.shape[0]
            actual = new_right_padded[:chi_l, :D_w, :chi_l]
            np.testing.assert_allclose(
                actual,
                ref,
                atol=1e-12,
                err_msg=f"Right env mismatch at site {site_idx}",
            )

            # Check padding is zero
            assert jnp.allclose(new_right_padded[chi_l:, :, :], 0.0), (
                "Padding rows should be zero"
            )
            assert jnp.allclose(new_right_padded[:, :, chi_l:], 0.0), (
                "Padding cols should be zero"
            )

            # Advance for next iteration
            right_env_tensor = new_right_tensor

    def test_right_env_jit_compilable(self):
        """update_right_env_dense_jit must be JIT-compilable."""
        L = 4
        bond_dim = 4
        chi_max = 8
        mps_tensors, mpo_tensors = _build_test_system(L=L, bond_dim=bond_dim)

        right_env_tensor = _build_trivial_right_env(dtype=jnp.float64)
        R_env_raw = right_env_tensor.todense()
        B_raw = mps_tensors[L - 1].todense()
        W_raw = mpo_tensors[L - 1].todense()

        jit_fn = jax.jit(update_right_env_dense_jit, static_argnums=(3,))
        result = jit_fn(R_env_raw, B_raw, W_raw, chi_max)
        assert result.shape[0] == chi_max

    def test_right_env_no_padding_needed(self):
        """When chi_max equals actual bond dim, no padding is needed."""
        L = 6
        bond_dim = 8
        chi_max = bond_dim

        mps_tensors, mpo_tensors = _build_test_system(L=L, bond_dim=bond_dim)
        right_env_tensor = _build_trivial_right_env(dtype=jnp.float64)

        # Update at site L-2 (interior site)
        right_env_tensor = _update_right_env(
            right_env_tensor, mps_tensors[L - 1], mpo_tensors[L - 1]
        )

        R_env_raw = right_env_tensor.todense()
        B_raw = mps_tensors[L - 2].todense()
        W_raw = mpo_tensors[L - 2].todense()
        D_w = W_raw.shape[0]

        new_right_padded = update_right_env_dense_jit(R_env_raw, B_raw, W_raw, chi_max)

        ref = _update_right_env(
            right_env_tensor, mps_tensors[L - 2], mpo_tensors[L - 2]
        ).todense()
        chi_l = ref.shape[0]
        actual = new_right_padded[:chi_l, :D_w, :chi_l]
        np.testing.assert_allclose(actual, ref, atol=1e-12)


class TestPaddedEnvWithPaddedInput:
    """Test that padded env updates work when the input is already padded."""

    def test_left_env_padded_input(self):
        """Left env update with a pre-padded input env produces correct results."""
        L = 6
        bond_dim = 8
        chi_max = 16

        mps_tensors, mpo_tensors = _build_test_system(L=L, bond_dim=bond_dim)

        # Build reference left env at site 0
        left_env_tensor = _build_trivial_left_env(dtype=jnp.float64)
        ref_1 = _update_left_env(left_env_tensor, mps_tensors[0], mpo_tensors[0])

        # Create padded version of site-0 env
        W0 = mpo_tensors[0].todense()
        padded_L = update_left_env_dense_jit(
            left_env_tensor.todense(),
            mps_tensors[0].todense(),
            W0,
            chi_max,
        )

        # Now use the padded env as input for site 1
        A1 = mps_tensors[1].todense()
        W1 = mpo_tensors[1].todense()
        D_w_1 = W1.shape[3]

        # Pad A1 to chi_max on both virtual dims
        A1_padded = jnp.zeros((chi_max, A1.shape[1], chi_max), dtype=A1.dtype)
        A1_padded = A1_padded.at[: A1.shape[0], :, : A1.shape[2]].set(A1)

        padded_L2 = update_left_env_dense_jit(padded_L, A1_padded, W1, chi_max)

        # Reference
        ref_2 = _update_left_env(ref_1, mps_tensors[1], mpo_tensors[1]).todense()
        chi_r = ref_2.shape[0]
        actual = padded_L2[:chi_r, :D_w_1, :chi_r]
        np.testing.assert_allclose(actual, ref_2, atol=1e-12)

    def test_right_env_padded_input(self):
        """Right env update with a pre-padded input env produces correct results."""
        L = 6
        bond_dim = 8
        chi_max = 16

        mps_tensors, mpo_tensors = _build_test_system(L=L, bond_dim=bond_dim)

        # Build reference right env at site L-1
        right_env_tensor = _build_trivial_right_env(dtype=jnp.float64)
        ref_1 = _update_right_env(
            right_env_tensor, mps_tensors[L - 1], mpo_tensors[L - 1]
        )

        # Create padded version of site L-1 env
        W_last = mpo_tensors[L - 1].todense()
        padded_R = update_right_env_dense_jit(
            right_env_tensor.todense(),
            mps_tensors[L - 1].todense(),
            W_last,
            chi_max,
        )

        # Now use the padded env as input for site L-2
        B = mps_tensors[L - 2].todense()
        W = mpo_tensors[L - 2].todense()
        D_w = W.shape[0]

        # Pad B to chi_max on both virtual dims
        B_padded = jnp.zeros((chi_max, B.shape[1], chi_max), dtype=B.dtype)
        B_padded = B_padded.at[: B.shape[0], :, : B.shape[2]].set(B)

        padded_R2 = update_right_env_dense_jit(padded_R, B_padded, W, chi_max)

        # Reference
        ref_2 = _update_right_env(
            ref_1, mps_tensors[L - 2], mpo_tensors[L - 2]
        ).todense()
        chi_l = ref_2.shape[0]
        actual = padded_R2[:chi_l, :D_w, :chi_l]
        np.testing.assert_allclose(actual, ref_2, atol=1e-12)


# ------------------------------------------------------------------ #
# Effective Hamiltonian matvec tests (Task 5)                         #
# ------------------------------------------------------------------ #


class TestEffectiveHamiltonianMatvecDense:
    """Test effective_ham_matvec_dense matches _effective_hamiltonian_matvec."""

    def test_matvec_matches_python(self):
        """Padded dense matvec must match the existing Python implementation."""
        L = 6
        bond_dim = 8
        chi_max = 12

        mps_tensors, mpo_tensors = _build_test_system(L=L, bond_dim=bond_dim)

        left_env = _build_trivial_left_env(dtype=jnp.float64)
        for i in range(2):
            left_env = _update_left_env(left_env, mps_tensors[i], mpo_tensors[i])

        right_env = _build_trivial_right_env(dtype=jnp.float64)
        for i in range(L - 1, 3, -1):
            right_env = _update_right_env(right_env, mps_tensors[i], mpo_tensors[i])

        from tenax.contraction.contractor import contract

        theta = contract(mps_tensors[2], mps_tensors[3])
        theta_dense = theta.todense()
        theta_shape = theta_dense.shape
        theta_flat = theta_dense.ravel()
        chi_l, d_l, d_r, chi_r = theta_shape

        L_arr = left_env.todense()
        R_arr = right_env.todense()
        W_l_arr = mpo_tensors[2].todense()
        W_r_arr = mpo_tensors[3].todense()

        ref = _effective_hamiltonian_matvec(
            theta_flat, theta_shape, L_arr, W_l_arr, W_r_arr, R_arr
        )

        D_w_l = W_l_arr.shape[0]
        D_w_r = W_r_arr.shape[3]

        theta_padded = jnp.zeros((chi_max, d_l, d_r, chi_max), dtype=theta_dense.dtype)
        theta_padded = theta_padded.at[:chi_l, :, :, :chi_r].set(theta_dense)

        L_padded = jnp.zeros((chi_max, D_w_l, chi_max), dtype=L_arr.dtype)
        L_padded = L_padded.at[: L_arr.shape[0], :, : L_arr.shape[2]].set(L_arr)

        R_padded = jnp.zeros((chi_max, D_w_r, chi_max), dtype=R_arr.dtype)
        R_padded = R_padded.at[: R_arr.shape[0], :, : R_arr.shape[2]].set(R_arr)

        result = effective_ham_matvec_dense(
            theta_padded.ravel(), L_padded, W_l_arr, W_r_arr, R_padded, chi_max
        )

        result_4d = result.reshape(chi_max, d_l, d_r, chi_max)
        ref_4d = ref.reshape(theta_shape)
        actual = result_4d[:chi_l, :, :, :chi_r]
        np.testing.assert_allclose(actual, ref_4d, atol=1e-11)

    def test_matvec_is_jittable(self):
        """effective_ham_matvec_dense must work inside jax.jit."""
        L = 4
        bond_dim = 4
        chi_max = 8
        d = 2

        mps_tensors, mpo_tensors = _build_test_system(L=L, bond_dim=bond_dim)

        left_env = _build_trivial_left_env(dtype=jnp.float64)
        right_env = _build_trivial_right_env(dtype=jnp.float64)
        for i in range(L - 1, 1, -1):
            right_env = _update_right_env(right_env, mps_tensors[i], mpo_tensors[i])

        L_arr = left_env.todense()
        R_arr = right_env.todense()
        W_l_arr = mpo_tensors[0].todense()
        W_r_arr = mpo_tensors[1].todense()
        D_w_l = W_l_arr.shape[0]
        D_w_r = W_r_arr.shape[3]

        L_padded = jnp.zeros((chi_max, D_w_l, chi_max), dtype=L_arr.dtype)
        L_padded = L_padded.at[: L_arr.shape[0], :, : L_arr.shape[2]].set(L_arr)
        R_padded = jnp.zeros((chi_max, D_w_r, chi_max), dtype=R_arr.dtype)
        R_padded = R_padded.at[: R_arr.shape[0], :, : R_arr.shape[2]].set(R_arr)

        theta_flat = jnp.zeros(chi_max * d * d * chi_max, dtype=jnp.float64)

        jit_fn = jax.jit(effective_ham_matvec_dense, static_argnums=(5,))
        result = jit_fn(theta_flat, L_padded, W_l_arr, W_r_arr, R_padded, chi_max)
        assert result.shape == theta_flat.shape


# ------------------------------------------------------------------ #
# Lanczos eigensolver tests (Task 5)                                  #
# ------------------------------------------------------------------ #


class TestDenseLanczos:
    """Test lanczos_ground_state_dense finds correct ground states."""

    def test_dense_lanczos_finds_ground_state(self):
        """Lanczos on a small explicit Hamiltonian finds the correct eigenvalue."""
        H = jnp.array(
            [
                [0.25, 0.0, 0.0, 0.0],
                [0.0, -0.25, 0.5, 0.0],
                [0.0, 0.5, -0.25, 0.0],
                [0.0, 0.0, 0.0, 0.25],
            ],
            dtype=jnp.float64,
        )

        v0 = jax.random.normal(jax.random.PRNGKey(42), (4,), dtype=jnp.float64)
        eigenvalue, eigenvector = lanczos_ground_state_dense(
            lambda v: H @ v, v0, max_iter=10
        )

        np.testing.assert_allclose(float(eigenvalue), -0.75, atol=1e-12)
        Hv = H @ eigenvector
        np.testing.assert_allclose(Hv, float(eigenvalue) * eigenvector, atol=1e-10)

    def test_lanczos_larger_matrix(self):
        """Lanczos finds the ground state of a larger random symmetric matrix."""
        n = 32
        key = jax.random.PRNGKey(123)
        A = jax.random.normal(key, (n, n), dtype=jnp.float64)
        H = (A + A.T) / 2

        v0 = jax.random.normal(jax.random.PRNGKey(0), (n,), dtype=jnp.float64)
        eigenvalue, _ = lanczos_ground_state_dense(lambda v: H @ v, v0, max_iter=32)

        eigvals_exact = jnp.linalg.eigvalsh(H)
        np.testing.assert_allclose(
            float(eigenvalue), float(eigvals_exact[0]), atol=1e-10
        )

    def test_lanczos_is_jittable(self):
        """lanczos_ground_state_dense works inside jax.jit."""
        H = jnp.array(
            [
                [0.25, 0.0, 0.0, 0.0],
                [0.0, -0.25, 0.5, 0.0],
                [0.0, 0.5, -0.25, 0.0],
                [0.0, 0.0, 0.0, 0.25],
            ],
            dtype=jnp.float64,
        )

        def solve(v0):
            return lanczos_ground_state_dense(lambda v: H @ v, v0, max_iter=10)

        v0 = jax.random.normal(jax.random.PRNGKey(7), (4,), dtype=jnp.float64)
        eigenvalue, _ = jax.jit(solve)(v0)
        np.testing.assert_allclose(float(eigenvalue), -0.75, atol=1e-12)

    def test_lanczos_with_effective_hamiltonian(self):
        """Integration: Lanczos + effective_ham_matvec_dense on DMRG system."""
        L = 6
        bond_dim = 4
        chi_max = 8

        mps_tensors, mpo_tensors = _build_test_system(L=L, bond_dim=bond_dim, seed=99)

        left_env = _build_trivial_left_env(dtype=jnp.float64)
        for i in range(2):
            left_env = _update_left_env(left_env, mps_tensors[i], mpo_tensors[i])

        right_env = _build_trivial_right_env(dtype=jnp.float64)
        for i in range(L - 1, 3, -1):
            right_env = _update_right_env(right_env, mps_tensors[i], mpo_tensors[i])

        from tenax.contraction.contractor import contract

        theta = contract(mps_tensors[2], mps_tensors[3])
        theta_dense = theta.todense()
        theta_shape = theta_dense.shape
        chi_l, d_l, d_r, chi_r = theta_shape

        L_arr = left_env.todense()
        R_arr = right_env.todense()
        W_l_arr = mpo_tensors[2].todense()
        W_r_arr = mpo_tensors[3].todense()
        D_w_l = W_l_arr.shape[0]
        D_w_r = W_r_arr.shape[3]

        L_padded = jnp.zeros((chi_max, D_w_l, chi_max), dtype=L_arr.dtype)
        L_padded = L_padded.at[: L_arr.shape[0], :, : L_arr.shape[2]].set(L_arr)
        R_padded = jnp.zeros((chi_max, D_w_r, chi_max), dtype=R_arr.dtype)
        R_padded = R_padded.at[: R_arr.shape[0], :, : R_arr.shape[2]].set(R_arr)

        theta_padded = jnp.zeros((chi_max, d_l, d_r, chi_max), dtype=theta_dense.dtype)
        theta_padded = theta_padded.at[:chi_l, :, :, :chi_r].set(theta_dense)

        def matvec(v):
            return effective_ham_matvec_dense(
                v, L_padded, W_l_arr, W_r_arr, R_padded, chi_max
            )

        E_padded, _ = lanczos_ground_state_dense(
            matvec, theta_padded.ravel(), max_iter=20
        )

        # Reference
        from tenax.algorithms.dmrg import _lanczos_solve_jit

        def ref_matvec(v):
            return _effective_hamiltonian_matvec(
                v, theta_shape, L_arr, W_l_arr, W_r_arr, R_arr
            )

        E_ref, _ = _lanczos_solve_jit(ref_matvec, theta_dense.ravel(), 20)

        np.testing.assert_allclose(float(E_padded), float(E_ref), atol=1e-8)


# ------------------------------------------------------------------ #
# Full lax.scan DMRG sweep tests (Task 6)                             #
# ------------------------------------------------------------------ #


class TestDenseJitSweep:
    """Test jit_dmrg_sweep_dense produces correct energies."""

    def test_dense_jit_sweep_matches_python_sweep(self):
        """JIT sweep energy must match existing Python DMRG energy for Heisenberg L=6 chi=8."""
        L = 6
        chi_max = 8
        num_sweeps = 6

        # Build Heisenberg MPO
        mpo_tn = _build_dense_heisenberg(L)
        mpo_tensors_obj = [mpo_tn.get_tensor(i) for i in range(L)]

        # Reference: run Python DMRG
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=42)
        config = DMRGConfig(
            max_bond_dim=chi_max,
            num_sweeps=num_sweeps,
            two_site=True,
            lanczos_max_iter=20,
            verbose=False,
        )
        result = dmrg(mpo_tn, mps_tn, config)
        E_python = result.energy

        # JIT sweep: extract raw arrays from a fresh random MPS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn2 = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=42)
        mps_raw = [mps_tn2.get_tensor(i).todense() for i in range(L)]
        mpo_raw = [t.todense() for t in mpo_tensors_obj]

        energies, _ = jit_dmrg_sweep_dense(
            mps_raw, mpo_raw, chi_max, num_sweeps=num_sweeps, lanczos_max_iter=20
        )

        E_jit = energies[-1]

        # Both should converge to approximately the same ground state energy.
        # The algorithms differ slightly (padding, env rebuild strategy), so
        # we allow a tolerance of 1e-4 for the final energy.
        np.testing.assert_allclose(
            E_jit,
            E_python,
            atol=1e-4,
            err_msg=(f"JIT sweep E={E_jit:.8f} vs Python DMRG E={E_python:.8f}"),
        )

        # Also verify monotonic decrease (each sweep should lower or maintain energy)
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i - 1] + 1e-8, (
                f"Energy not monotonically decreasing: sweep {i - 1}={energies[i - 1]:.8f}, "
                f"sweep {i}={energies[i]:.8f}"
            )

    def test_jit_sweep_compiles_once(self):
        """Second call with same shapes should reuse compiled code (same result)."""
        L = 4
        chi_max = 4
        num_sweeps = 2

        mpo_tn = _build_dense_heisenberg(L)
        mpo_raw = [mpo_tn.get_tensor(i).todense() for i in range(L)]

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=7)
        mps_raw = [mps_tn.get_tensor(i).todense() for i in range(L)]

        # First call (triggers compilation)
        energies_1, _ = jit_dmrg_sweep_dense(
            mps_raw, mpo_raw, chi_max, num_sweeps=num_sweeps, lanczos_max_iter=10
        )

        # Second call (should reuse compiled XLA program)
        energies_2, _ = jit_dmrg_sweep_dense(
            mps_raw, mpo_raw, chi_max, num_sweeps=num_sweeps, lanczos_max_iter=10
        )

        # Results must be identical (exact same computation)
        for i in range(num_sweeps):
            np.testing.assert_allclose(energies_1[i], energies_2[i], atol=1e-14)

    def test_jit_sweep_heisenberg_exact_energy(self):
        """JIT sweep should get close to exact Heisenberg L=4 ground state energy."""
        L = 4
        chi_max = 8  # exact for L=4
        num_sweeps = 8

        mpo_tn = _build_dense_heisenberg(L)
        mpo_raw = [mpo_tn.get_tensor(i).todense() for i in range(L)]

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=123)
        mps_raw = [mps_tn.get_tensor(i).todense() for i in range(L)]

        energies, _ = jit_dmrg_sweep_dense(
            mps_raw, mpo_raw, chi_max, num_sweeps=num_sweeps, lanczos_max_iter=20
        )

        # Exact ground state energy of Heisenberg L=4: E = -1.61603...
        # (from exact diagonalization of 4-site chain)
        E_exact = -1.6160254037844388
        E_jit = energies[-1]

        np.testing.assert_allclose(
            E_jit,
            E_exact,
            atol=1e-4,
            err_msg=(f"JIT sweep E={E_jit:.8f} vs exact E={E_exact:.8f}"),
        )


# ------------------------------------------------------------------ #
# Block-sparse effective Hamiltonian matvec tests (Task 8d)            #
# ------------------------------------------------------------------ #


from tenax.algorithms._padded_block_array import (
    MultiContractionPlan,
    PaddedBlockArray,
    contract_multi_padded,
)
from tenax.core.mps import FiniteMPS
from tenax.core.symmetry import U1Symmetry


def _build_symmetric_heisenberg_mpo(L: int):
    """Build a fully symmetric (SymmetricTensor) Heisenberg MPO via AutoMPO."""
    terms = []
    for i in range(L - 1):
        terms.append((1.0, "Sz", i, "Sz", i + 1))
        terms.append((0.5, "Sp", i, "Sm", i + 1))
        terms.append((0.5, "Sm", i, "Sp", i + 1))
    return build_auto_mpo(terms, L=L, symmetric=True)


def _build_dmrg_env_tensors_symmetric(L=4, chi=4, target_charge=0, seed=42):
    """Build symmetric MPS, MPO, and environment tensors for testing.

    Returns:
        Tuple of (mps_tensors, mpo_tensors, left_envs, right_envs)
        where each is a list of SymmetricTensor objects.
    """
    from tenax.algorithms.dmrg import (
        _build_trivial_left_env_symmetric,
        _build_trivial_right_env_symmetric,
        _update_left_env_symmetric,
        _update_right_env_symmetric,
    )

    mpo_net = _build_symmetric_heisenberg_mpo(L)
    mps = FiniteMPS.random(
        L,
        d=2,
        chi=chi,
        key=jax.random.PRNGKey(seed),
        symmetric=True,
        symmetry=U1Symmetry(),
        target_charge=target_charge,
    )

    mps_tensors = [mps.get_tensor(i) for i in range(L)]
    mpo_tensors = [mpo_net.get_tensor(i) for i in range(L)]

    # Build left environments
    left_envs = [_build_trivial_left_env_symmetric()]
    for i in range(L - 1):
        new_l = _update_left_env_symmetric(
            left_envs[-1], mps_tensors[i], mpo_tensors[i]
        )
        left_envs.append(new_l)

    # Build right environments
    right_envs = [None] * (L + 1)
    right_envs[L] = _build_trivial_right_env_symmetric()
    for i in range(L - 1, 0, -1):
        right_envs[i] = _update_right_env_symmetric(
            right_envs[i + 1], mps_tensors[i], mpo_tensors[i]
        )

    return mps_tensors, mpo_tensors, left_envs, right_envs


class TestBlockSparseMatvec:
    """Test effective_ham_matvec_blocks matches _blockwise_contract reference."""

    def test_matvec_matches_symmetric_reference(self):
        """Block-sparse matvec via PBA matches _blockwise_contract reference."""
        L = 6
        chi = 4
        mps_tensors, mpo_tensors, left_envs, right_envs = (
            _build_dmrg_env_tensors_symmetric(L=L, chi=chi)
        )

        from tenax.algorithms.dmrg import _blockwise_contract
        from tenax.contraction.contractor import contract

        # Pick bond between sites 2 and 3
        site_l = 2
        site_r = 3
        l_env = left_envs[site_l]
        r_env = right_envs[site_r + 1]
        W_l = mpo_tensors[site_l]
        W_r = mpo_tensors[site_r]

        # Build theta
        theta = contract(mps_tensors[site_l], mps_tensors[site_r])

        # Reference: _blockwise_contract
        subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        ref_result = _blockwise_contract(
            [l_env, theta, W_l, W_r, r_env],
            subs,
            output_indices=theta.indices,
        )

        # Block-sparse PBA path
        l_pba = PaddedBlockArray.from_symmetric(l_env)
        theta_pba = PaddedBlockArray.from_symmetric(theta)
        wl_pba = PaddedBlockArray.from_symmetric(W_l)
        wr_pba = PaddedBlockArray.from_symmetric(W_r)
        r_pba = PaddedBlockArray.from_symmetric(r_env)

        plan = MultiContractionPlan.build(
            [l_env, theta, W_l, W_r, r_env],
            subs,
            output_indices=theta.indices,
        )

        result_pba = effective_ham_matvec_blocks(
            plan, theta_pba, l_pba, wl_pba, wr_pba, r_pba
        )
        result_sym = result_pba.to_symmetric()

        np.testing.assert_allclose(
            np.array(result_sym.todense()),
            np.array(ref_result.todense()),
            atol=1e-10,
        )

    def test_matvec_larger_bond_dim(self):
        """Block-sparse matvec with chi=8 for more block diversity."""
        L = 6
        chi = 8
        mps_tensors, mpo_tensors, left_envs, right_envs = (
            _build_dmrg_env_tensors_symmetric(L=L, chi=chi, seed=99)
        )

        from tenax.algorithms.dmrg import _blockwise_contract
        from tenax.contraction.contractor import contract

        site_l = 2
        site_r = 3
        l_env = left_envs[site_l]
        r_env = right_envs[site_r + 1]
        W_l = mpo_tensors[site_l]
        W_r = mpo_tensors[site_r]

        theta = contract(mps_tensors[site_l], mps_tensors[site_r])

        subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        ref_result = _blockwise_contract(
            [l_env, theta, W_l, W_r, r_env],
            subs,
            output_indices=theta.indices,
        )

        plan = MultiContractionPlan.build(
            [l_env, theta, W_l, W_r, r_env],
            subs,
            output_indices=theta.indices,
        )
        result_pba = effective_ham_matvec_blocks(
            plan,
            PaddedBlockArray.from_symmetric(theta),
            PaddedBlockArray.from_symmetric(l_env),
            PaddedBlockArray.from_symmetric(W_l),
            PaddedBlockArray.from_symmetric(W_r),
            PaddedBlockArray.from_symmetric(r_env),
        )
        result_sym = result_pba.to_symmetric()

        np.testing.assert_allclose(
            np.array(result_sym.todense()),
            np.array(ref_result.todense()),
            atol=1e-10,
        )

    def test_matvec_jit_compatible(self):
        """effective_ham_matvec_blocks works inside jax.jit."""
        L = 4
        chi = 4
        mps_tensors, mpo_tensors, left_envs, right_envs = (
            _build_dmrg_env_tensors_symmetric(L=L, chi=chi)
        )

        from tenax.algorithms._padded_block_array import pba_expand_blocks
        from tenax.contraction.contractor import contract

        l_env = left_envs[1]
        r_env = right_envs[3]
        W_l = mpo_tensors[1]
        W_r = mpo_tensors[2]
        theta = contract(mps_tensors[1], mps_tensors[2])

        subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        plan = MultiContractionPlan.build(
            [l_env, theta, W_l, W_r, r_env],
            subs,
            output_indices=theta.indices,
        )

        # Expand theta to match the plan's output block structure
        theta_pba = PaddedBlockArray.from_symmetric(theta)
        theta_pba = pba_expand_blocks(
            theta_pba,
            plan.output_charges,
            plan.output_shapes,
            plan.output_tensor_indices,
        )

        l_pba = PaddedBlockArray.from_symmetric(l_env)
        wl_pba = PaddedBlockArray.from_symmetric(W_l)
        wr_pba = PaddedBlockArray.from_symmetric(W_r)
        r_pba = PaddedBlockArray.from_symmetric(r_env)

        @jax.jit
        def jit_matvec(tp, lp, wlp, wrp, rp):
            return effective_ham_matvec_blocks(plan, tp, lp, wlp, wrp, rp)

        result = jit_matvec(theta_pba, l_pba, wl_pba, wr_pba, r_pba)
        # Output should match the plan's output structure
        assert result.data.shape[0] == plan.num_output_blocks


# ------------------------------------------------------------------ #
# Block-sparse Lanczos eigensolver tests (Task 8d)                     #
# ------------------------------------------------------------------ #


class TestBlockSparseLanczos:
    """Test lanczos_ground_state_blocks finds correct ground states."""

    def test_lanczos_finds_ground_state(self):
        """Block-sparse Lanczos finds the same ground state energy as dense."""
        L = 6
        chi = 4
        mps_tensors, mpo_tensors, left_envs, right_envs = (
            _build_dmrg_env_tensors_symmetric(L=L, chi=chi, seed=42)
        )

        from tenax.algorithms._padded_block_array import pba_expand_blocks
        from tenax.algorithms.dmrg import _blockwise_contract
        from tenax.contraction.contractor import contract

        site_l = 2
        site_r = 3
        l_env = left_envs[site_l]
        r_env = right_envs[site_r + 1]
        W_l = mpo_tensors[site_l]
        W_r = mpo_tensors[site_r]

        theta = contract(mps_tensors[site_l], mps_tensors[site_r])

        # Build plan and PBAs
        subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        plan = MultiContractionPlan.build(
            [l_env, theta, W_l, W_r, r_env],
            subs,
            output_indices=theta.indices,
        )

        l_pba = PaddedBlockArray.from_symmetric(l_env)
        # Expand theta to match the plan's output block structure so that
        # matvec input/output shapes are consistent for Lanczos iteration.
        theta_pba = pba_expand_blocks(
            PaddedBlockArray.from_symmetric(theta),
            plan.output_charges,
            plan.output_shapes,
            plan.output_tensor_indices,
        )
        wl_pba = PaddedBlockArray.from_symmetric(W_l)
        wr_pba = PaddedBlockArray.from_symmetric(W_r)
        r_pba = PaddedBlockArray.from_symmetric(r_env)

        def matvec(v_pba):
            return effective_ham_matvec_blocks(
                plan, v_pba, l_pba, wl_pba, wr_pba, r_pba
            )

        E_blocks, eigvec_pba = lanczos_ground_state_blocks(
            matvec, theta_pba, max_iter=20
        )

        # Reference: dense Lanczos on the same problem
        theta_dense = theta.todense()
        theta_shape = theta_dense.shape
        chi_l, d_l, d_r, chi_r = theta_shape
        L_arr = l_env.todense()
        R_arr = r_env.todense()
        W_l_arr = W_l.todense()
        W_r_arr = W_r.todense()

        def ref_matvec(v):
            return _effective_hamiltonian_matvec(
                v, theta_shape, L_arr, W_l_arr, W_r_arr, R_arr
            )

        E_ref, _ = lanczos_ground_state_dense(
            ref_matvec, theta_dense.ravel(), max_iter=20
        )

        # Both should converge to the same energy
        np.testing.assert_allclose(float(E_blocks), float(E_ref), atol=1e-8)

        # Verify eigenvector is an approximate eigenvector
        Hv = matvec(eigvec_pba)
        from tenax.algorithms._padded_block_array import pba_inner, pba_norm

        rayleigh = pba_inner(eigvec_pba, Hv).real
        norm_v = pba_norm(eigvec_pba)
        # Rayleigh quotient should be close to eigenvalue
        np.testing.assert_allclose(
            float(rayleigh) / float(norm_v) ** 2,
            float(E_blocks),
            atol=1e-8,
        )

    def test_lanczos_larger_bond_dim(self):
        """Block-sparse Lanczos with chi=8 for more block diversity."""
        L = 6
        chi = 8
        mps_tensors, mpo_tensors, left_envs, right_envs = (
            _build_dmrg_env_tensors_symmetric(L=L, chi=chi, seed=99)
        )

        from tenax.algorithms._padded_block_array import pba_expand_blocks
        from tenax.contraction.contractor import contract

        site_l = 2
        site_r = 3
        l_env = left_envs[site_l]
        r_env = right_envs[site_r + 1]
        W_l = mpo_tensors[site_l]
        W_r = mpo_tensors[site_r]

        theta = contract(mps_tensors[site_l], mps_tensors[site_r])

        subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        plan = MultiContractionPlan.build(
            [l_env, theta, W_l, W_r, r_env],
            subs,
            output_indices=theta.indices,
        )

        l_pba = PaddedBlockArray.from_symmetric(l_env)
        theta_pba = pba_expand_blocks(
            PaddedBlockArray.from_symmetric(theta),
            plan.output_charges,
            plan.output_shapes,
            plan.output_tensor_indices,
        )
        wl_pba = PaddedBlockArray.from_symmetric(W_l)
        wr_pba = PaddedBlockArray.from_symmetric(W_r)
        r_pba = PaddedBlockArray.from_symmetric(r_env)

        def matvec(v_pba):
            return effective_ham_matvec_blocks(
                plan, v_pba, l_pba, wl_pba, wr_pba, r_pba
            )

        E_blocks, _ = lanczos_ground_state_blocks(matvec, theta_pba, max_iter=20)

        # Reference: dense Lanczos
        theta_dense = theta.todense()
        theta_shape = theta_dense.shape
        L_arr = l_env.todense()
        R_arr = r_env.todense()

        def ref_matvec(v):
            return _effective_hamiltonian_matvec(
                v, theta_shape, L_arr, W_l.todense(), W_r.todense(), R_arr
            )

        E_ref, _ = lanczos_ground_state_dense(
            ref_matvec, theta_dense.ravel(), max_iter=20
        )

        np.testing.assert_allclose(float(E_blocks), float(E_ref), atol=1e-8)

    def test_lanczos_jit_compatible(self):
        """Block-sparse Lanczos works inside jax.jit."""
        L = 4
        chi = 4
        mps_tensors, mpo_tensors, left_envs, right_envs = (
            _build_dmrg_env_tensors_symmetric(L=L, chi=chi)
        )

        from tenax.algorithms._padded_block_array import pba_expand_blocks
        from tenax.contraction.contractor import contract

        l_env = left_envs[1]
        r_env = right_envs[3]
        W_l = mpo_tensors[1]
        W_r = mpo_tensors[2]
        theta = contract(mps_tensors[1], mps_tensors[2])

        subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        plan = MultiContractionPlan.build(
            [l_env, theta, W_l, W_r, r_env],
            subs,
            output_indices=theta.indices,
        )

        l_pba = PaddedBlockArray.from_symmetric(l_env)
        theta_pba = pba_expand_blocks(
            PaddedBlockArray.from_symmetric(theta),
            plan.output_charges,
            plan.output_shapes,
            plan.output_tensor_indices,
        )
        wl_pba = PaddedBlockArray.from_symmetric(W_l)
        wr_pba = PaddedBlockArray.from_symmetric(W_r)
        r_pba = PaddedBlockArray.from_symmetric(r_env)

        def matvec(v_pba):
            return effective_ham_matvec_blocks(
                plan, v_pba, l_pba, wl_pba, wr_pba, r_pba
            )

        @jax.jit
        def jit_lanczos(v0):
            return lanczos_ground_state_blocks(matvec, v0, max_iter=10)

        E, eigvec = jit_lanczos(theta_pba)
        # Should return a scalar energy and a PBA
        assert E.shape == ()
        assert eigvec.data.shape == theta_pba.data.shape


# ------------------------------------------------------------------ #
# Block-sparse full sweep tests (Task 8e)                              #
# ------------------------------------------------------------------ #


class TestBlockSparseJITSweep:
    """Test jit_dmrg_sweep_symmetric produces correct energies."""

    def test_symmetric_jit_sweep_matches_python(self):
        """JIT sweep on U(1) symmetric Heisenberg matches Python sweep."""
        from tenax.algorithms._jit_sweep import jit_dmrg_sweep_symmetric

        L = 6
        chi = 8
        num_sweeps = 4

        mpo_net = _build_symmetric_heisenberg_mpo(L)
        mps = FiniteMPS.random(
            L,
            d=2,
            chi=chi,
            key=jax.random.PRNGKey(42),
            symmetric=True,
            symmetry=U1Symmetry(),
            target_charge=0,
        )

        mps_tensors = [mps.get_tensor(i) for i in range(L)]
        mpo_tensors = [mpo_net.get_tensor(i) for i in range(L)]

        # Run JIT symmetric sweep
        energies_jit, mps_out = jit_dmrg_sweep_symmetric(
            mps_tensors,
            mpo_tensors,
            chi_max=chi,
            num_sweeps=num_sweeps,
            lanczos_max_iter=20,
        )

        # Run Python DMRG (accelerator="off") as reference
        mps2 = FiniteMPS.random(
            L,
            d=2,
            chi=chi,
            key=jax.random.PRNGKey(42),
            symmetric=True,
            symmetry=U1Symmetry(),
            target_charge=0,
        )
        config = DMRGConfig(
            max_bond_dim=chi,
            num_sweeps=num_sweeps,
            two_site=True,
            lanczos_max_iter=20,
            verbose=False,
            accelerator="off",
        )
        result_py = dmrg(mpo_net, mps2, config)

        # Both should converge to approximately the same ground state energy.
        # The algorithms differ slightly (Lanczos implementation, env rebuild
        # strategy), so we allow a tolerance of 1e-3 for the final energy.
        np.testing.assert_allclose(
            energies_jit[-1],
            result_py.energy,
            atol=1e-3,
            err_msg=(
                f"JIT sweep E={energies_jit[-1]:.8f} vs "
                f"Python DMRG E={result_py.energy:.8f}"
            ),
        )

        # Verify energy is decreasing across sweeps
        for i in range(1, len(energies_jit)):
            assert energies_jit[i] <= energies_jit[i - 1] + 1e-6, (
                f"Energy not monotonically decreasing: "
                f"sweep {i - 1}={energies_jit[i - 1]:.8f}, "
                f"sweep {i}={energies_jit[i]:.8f}"
            )

        # Output MPS tensors should all be SymmetricTensors
        from tenax.core.tensor import SymmetricTensor

        for t in mps_out:
            assert isinstance(t, SymmetricTensor)

    def test_symmetric_jit_sweep_small_exact(self):
        """JIT sweep on L=4 produces correct ground state energy."""
        from tenax.algorithms._jit_sweep import jit_dmrg_sweep_symmetric

        L = 4
        chi = 8  # exact for L=4
        num_sweeps = 6

        mpo_net = _build_symmetric_heisenberg_mpo(L)
        mps = FiniteMPS.random(
            L,
            d=2,
            chi=chi,
            key=jax.random.PRNGKey(123),
            symmetric=True,
            symmetry=U1Symmetry(),
            target_charge=0,
        )

        mps_tensors = [mps.get_tensor(i) for i in range(L)]
        mpo_tensors = [mpo_net.get_tensor(i) for i in range(L)]

        energies, _ = jit_dmrg_sweep_symmetric(
            mps_tensors,
            mpo_tensors,
            chi_max=chi,
            num_sweeps=num_sweeps,
            lanczos_max_iter=20,
        )

        # Exact ground state energy of Heisenberg L=4: E = -1.61603...
        E_exact = -1.6160254037844388
        np.testing.assert_allclose(
            energies[-1],
            E_exact,
            atol=1e-4,
            err_msg=(f"JIT sweep E={energies[-1]:.8f} vs exact E={E_exact:.8f}"),
        )

    def test_symmetric_jit_dispatch_via_dmrg(self):
        """dmrg() with accelerator='jit' dispatches to the symmetric JIT path."""
        L = 4
        chi = 6
        num_sweeps = 3

        mpo_net = _build_symmetric_heisenberg_mpo(L)
        mps = FiniteMPS.random(
            L,
            d=2,
            chi=chi,
            key=jax.random.PRNGKey(7),
            symmetric=True,
            symmetry=U1Symmetry(),
            target_charge=0,
        )

        config = DMRGConfig(
            max_bond_dim=chi,
            num_sweeps=num_sweeps,
            two_site=True,
            lanczos_max_iter=20,
            verbose=False,
            accelerator="jit",
        )
        result = dmrg(mpo_net, mps, config)

        # Should produce a reasonable energy (below zero for Heisenberg)
        assert result.energy < 0.0, f"Energy should be negative, got {result.energy}"

        # Output MPS should be FiniteMPS with symmetric tensors
        from tenax.core.tensor import SymmetricTensor

        for t in result.mps.tensors:
            assert isinstance(t, SymmetricTensor)


# ------------------------------------------------------------------ #
# Warmup-to-JIT transition tests (Task 9)                             #
# ------------------------------------------------------------------ #


class TestWarmupToJIT:
    """Test that the warmup-to-JIT transition produces correct energies.

    When initial bond dimensions are smaller than chi_max, DMRG should run
    Python sweeps to grow bonds, then switch to the JIT-compiled lax.scan
    sweep once all bonds have reached chi_max.
    """

    def test_warmup_then_jit_dense(self):
        """Start with small chi, grow via Python sweeps, switch to JIT."""
        import warnings

        L = 6
        num_sweeps = 10

        mpo_tn = _build_dense_heisenberg(L)

        # Start with bond_dim=2, grow up to chi_max=8
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=2, seed=7)

        config = DMRGConfig(
            max_bond_dim=8,
            num_sweeps=num_sweeps,
            lanczos_max_iter=20,
            convergence_tol=1e-10,
            accelerator="jit",
        )
        result = dmrg(mpo_tn, mps_tn, config)

        assert np.isfinite(result.energy)
        assert result.energy < 0.0

        # Compare with pure Python path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn2 = build_random_mps(L=L, physical_dim=2, bond_dim=2, seed=7)

        config_py = DMRGConfig(
            max_bond_dim=8,
            num_sweeps=num_sweeps,
            lanczos_max_iter=20,
            convergence_tol=1e-10,
            accelerator="off",
        )
        result_py = dmrg(mpo_tn, mps_tn2, config_py)

        assert abs(result.energy - result_py.energy) < 1e-4, (
            f"Warmup+JIT E={result.energy:.8f} vs Python E={result_py.energy:.8f}"
        )

    def test_warmup_skipped_when_saturated(self):
        """When initial chi == chi_max, no warmup sweeps should be needed."""
        import warnings

        L = 4
        chi_max = 4
        num_sweeps = 4

        mpo_tn = _build_dense_heisenberg(L)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=42)

        config = DMRGConfig(
            max_bond_dim=chi_max,
            num_sweeps=num_sweeps,
            lanczos_max_iter=20,
            convergence_tol=1e-10,
            accelerator="jit",
        )
        result = dmrg(mpo_tn, mps_tn, config)

        assert np.isfinite(result.energy)
        assert result.energy < 0.0
        # When chi == chi_max, all sweeps go through JIT (no warmup truncation
        # errors recorded since JIT path doesn't track them)
        assert len(result.energies_per_sweep) == num_sweeps

    def test_warmup_energy_monotone_decreasing(self):
        """Energies should decrease (or stay flat) across warmup + JIT phases."""
        import warnings

        L = 6
        num_sweeps = 8

        mpo_tn = _build_dense_heisenberg(L)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=2, seed=13)

        config = DMRGConfig(
            max_bond_dim=8,
            num_sweeps=num_sweeps,
            lanczos_max_iter=20,
            convergence_tol=1e-12,
            accelerator="jit",
        )
        result = dmrg(mpo_tn, mps_tn, config)

        energies = result.energies_per_sweep
        assert len(energies) >= 2
        # Energy should decrease (allow small numerical noise)
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i - 1] + 1e-6, (
                f"Energy not decreasing: sweep {i - 1}={energies[i - 1]:.8f}, "
                f"sweep {i}={energies[i]:.8f}"
            )


# ------------------------------------------------------------------ #
# Integration tests (Task 10)                                          #
# ------------------------------------------------------------------ #


class TestIntegration:
    """End-to-end integration tests for JIT-accelerated DMRG."""

    @staticmethod
    def _build_heisenberg_matrix(L):
        """Build full Heisenberg Hamiltonian matrix for ED comparison."""
        d = 2
        N = d**L
        Sz = np.array([[0.5, 0], [0, -0.5]])
        Sp = np.array([[0, 1], [0, 0]])
        Sm = np.array([[0, 0], [1, 0]])
        eye = np.eye(d)
        H = np.zeros((N, N))
        for i in range(L - 1):
            # S_i . S_{i+1} = Sz_i Sz_{i+1} + 0.5*(Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
            for op_a, op_b, coeff in [(Sz, Sz, 1.0), (Sp, Sm, 0.5), (Sm, Sp, 0.5)]:
                term = np.eye(1)
                for j in range(L):
                    if j == i:
                        term = np.kron(term, op_a)
                    elif j == i + 1:
                        term = np.kron(term, op_b)
                    else:
                        term = np.kron(term, eye)
                H += coeff * term
        return H

    def test_heisenberg_l6_dense_jit_vs_exact(self):
        """Dense JIT DMRG on L=6 Heisenberg matches ED within 1e-4."""
        import warnings

        L = 6
        chi_max = 16  # more than enough for exact L=6 spin-1/2
        num_sweeps = 12

        # Exact energy from ED
        H = self._build_heisenberg_matrix(L)
        e_exact = float(np.linalg.eigvalsh(H)[0])

        # Run DMRG with JIT accelerator
        mpo_tn = _build_dense_heisenberg(L)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=42)

        config = DMRGConfig(
            max_bond_dim=chi_max,
            num_sweeps=num_sweeps,
            two_site=True,
            lanczos_max_iter=20,
            convergence_tol=1e-10,
            accelerator="jit",
            verbose=False,
        )
        result = dmrg(mpo_tn, mps_tn, config)

        np.testing.assert_allclose(
            result.energy,
            e_exact,
            atol=1e-4,
            err_msg=(f"JIT DMRG E={result.energy:.8f} vs exact E={e_exact:.8f}"),
        )

    def test_dmrg_accelerator_jit_on_cpu_runs(self):
        """Smoke test: accelerator='jit' runs on CPU and gives finite negative energy."""
        import warnings

        L = 4
        chi_max = 4
        num_sweeps = 4

        mpo_tn = _build_dense_heisenberg(L)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=77)

        config = DMRGConfig(
            max_bond_dim=chi_max,
            num_sweeps=num_sweeps,
            two_site=True,
            lanczos_max_iter=20,
            accelerator="jit",
            verbose=False,
        )
        result = dmrg(mpo_tn, mps_tn, config)

        assert np.isfinite(result.energy), (
            f"Energy should be finite, got {result.energy}"
        )
        assert result.energy < 0.0, (
            f"Heisenberg ground state should be negative, got {result.energy}"
        )
        assert len(result.energies_per_sweep) == num_sweeps

    def test_convergence_detection_in_jit_path(self):
        """JIT dense path detects convergence (converged=True) after enough sweeps."""
        import warnings

        L = 4
        chi_max = 8
        num_sweeps = 50

        mpo_tn = _build_dense_heisenberg(L)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=42)

        config = DMRGConfig(
            max_bond_dim=chi_max,
            num_sweeps=num_sweeps,
            two_site=True,
            lanczos_max_iter=20,
            convergence_tol=1e-8,
            accelerator="jit",
            verbose=False,
        )
        result = dmrg(mpo_tn, mps_tn, config)

        # With chi=8 on L=4, DMRG converges quickly.
        # The JIT path runs all sweeps via lax.fori_loop (no early stopping
        # inside JIT), but convergence is detected after the JIT call returns.
        assert result.converged, (
            f"Expected convergence, but converged={result.converged}. "
            f"Last energies: {result.energies_per_sweep[-3:]}"
        )

    def test_symmetric_jit_vs_exact(self):
        """Symmetric JIT DMRG on L=6 Heisenberg with U(1) matches ED within 1e-4."""
        L = 6
        chi = 8
        num_sweeps = 10

        # Exact energy from ED
        H = self._build_heisenberg_matrix(L)
        e_exact = float(np.linalg.eigvalsh(H)[0])

        # Build symmetric MPO and MPS
        mpo_net = _build_symmetric_heisenberg_mpo(L)
        mps = FiniteMPS.random(
            L,
            d=2,
            chi=chi,
            key=jax.random.PRNGKey(42),
            symmetric=True,
            symmetry=U1Symmetry(),
            target_charge=0,
        )

        config = DMRGConfig(
            max_bond_dim=chi,
            num_sweeps=num_sweeps,
            two_site=True,
            lanczos_max_iter=20,
            convergence_tol=1e-10,
            accelerator="jit",
            verbose=False,
        )
        result = dmrg(mpo_net, mps, config)

        np.testing.assert_allclose(
            result.energy,
            e_exact,
            atol=1e-4,
            err_msg=(
                f"Symmetric JIT DMRG E={result.energy:.8f} vs exact E={e_exact:.8f}"
            ),
        )


# ------------------------------------------------------------------ #
# Multi-device sharded DMRG sweep tests                               #
# ------------------------------------------------------------------ #


class TestShardedSweep:
    """Test multi-device sharded DMRG sweep."""

    def test_sharded_sweep_matches_single_device(self):
        """Sharded sweep must produce same energy as single-device JIT."""
        from tenax.algorithms._jit_sweep import (
            jit_dmrg_sweep_dense,
            jit_dmrg_sweep_dense_sharded,
        )

        L = 6
        chi_max = 8
        num_sweeps = 4

        mpo_tn = _build_dense_heisenberg(L)
        mpo_raw = [mpo_tn.get_tensor(i).todense() for i in range(L)]

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=42)
        mps_raw = [mps_tn.get_tensor(i).todense() for i in range(L)]

        # Single-device JIT
        energies_jit, _ = jit_dmrg_sweep_dense(
            mps_raw, mpo_raw, chi_max, num_sweeps=num_sweeps, lanczos_max_iter=20
        )

        # Sharded (may use 1 or more devices depending on hardware)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn2 = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=42)
        mps_raw2 = [mps_tn2.get_tensor(i).todense() for i in range(L)]

        energies_sharded, _ = jit_dmrg_sweep_dense_sharded(
            mps_raw2,
            mpo_raw,
            chi_max,
            num_sweeps=num_sweeps,
            lanczos_max_iter=20,
        )

        for i in range(len(energies_jit)):
            np.testing.assert_allclose(
                energies_sharded[i],
                energies_jit[i],
                atol=1e-6,
                err_msg=(
                    f"Sweep {i}: sharded={energies_sharded[i]:.8f}"
                    f" vs jit={energies_jit[i]:.8f}"
                ),
            )
