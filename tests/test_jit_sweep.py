"""Tests for JIT-compatible dense environment update functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._jit_sweep import (
    effective_ham_matvec_dense,
    lanczos_ground_state_dense,
    update_left_env_dense_jit,
    update_right_env_dense_jit,
)
from tenax.algorithms.auto_mpo import build_auto_mpo
from tenax.algorithms.dmrg import (
    _build_trivial_left_env,
    _build_trivial_right_env,
    _effective_hamiltonian_matvec,
    _update_left_env,
    _update_right_env,
    build_random_mps,
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
