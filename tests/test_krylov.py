"""Tests for the Lanczos-based Krylov matrix exponential."""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._krylov import krylov_expm


def _random_hermitian(n: int, seed: int = 0) -> np.ndarray:
    """Generate a random n x n Hermitian matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return (A + A.conj().T) / 2


def _make_matvec(H_jax):
    """Return a matvec closure for the given matrix."""

    def matvec(x):
        return H_jax @ x

    return matvec


class TestKrylovExpm:
    """Tests for krylov_expm."""

    def test_krylov_expm_real_time_small_matrix(self):
        """Compare against scipy.linalg.expm for 8x8 Hermitian matrix with dt=-0.1j."""
        n = 8
        H = _random_hermitian(n, seed=42)
        dt = -0.1j
        v0 = np.random.default_rng(7).standard_normal(n).astype(np.float64)

        # Reference: scipy dense expm
        expected = scipy.linalg.expm(dt * H) @ v0

        # Krylov approximation
        H_jax = jnp.array(H)
        v0_jax = jnp.array(v0)
        result = krylov_expm(_make_matvec(H_jax), v0_jax, dt, krylov_dim=20, tol=1e-12)

        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_krylov_expm_imaginary_time(self):
        """Compare against scipy.linalg.expm for 8x8 Hermitian matrix with dt=-0.1."""
        n = 8
        H = _random_hermitian(n, seed=42)
        dt = -0.1
        v0 = np.random.default_rng(7).standard_normal(n).astype(np.float64)

        expected = scipy.linalg.expm(dt * H) @ v0

        H_jax = jnp.array(H)
        v0_jax = jnp.array(v0)
        result = krylov_expm(_make_matvec(H_jax), v0_jax, dt, krylov_dim=20, tol=1e-12)

        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_krylov_expm_preserves_norm_unitary(self):
        """Real-time evolution with Hermitian H preserves norm."""
        n = 8
        H = _random_hermitian(n, seed=99)
        dt = -0.3j  # purely imaginary => unitary evolution
        v0 = np.random.default_rng(11).standard_normal(n).astype(np.float64)
        v0 = v0 / np.linalg.norm(v0)

        H_jax = jnp.array(H)
        v0_jax = jnp.array(v0)
        result = krylov_expm(_make_matvec(H_jax), v0_jax, dt, krylov_dim=20, tol=1e-12)

        np.testing.assert_allclose(jnp.linalg.norm(result), 1.0, atol=1e-10)

    def test_krylov_expm_zero_dt(self):
        """dt=0 returns the input vector unchanged."""
        n = 8
        H = _random_hermitian(n, seed=42)
        v0 = np.random.default_rng(7).standard_normal(n).astype(np.float64)

        H_jax = jnp.array(H)
        v0_jax = jnp.array(v0)
        result = krylov_expm(_make_matvec(H_jax), v0_jax, 0.0, krylov_dim=20, tol=1e-12)

        np.testing.assert_allclose(result, v0, atol=1e-14)
