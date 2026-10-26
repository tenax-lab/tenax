"""The CPU SVD falls back to ``gesvd`` when LAPACK ``gesdd`` does not converge.

Apple Accelerate's ``gesdd`` (numpy, scipy and jaxlib on macOS arm64) fails to
converge on some finite, ill-conditioned matrices, and JAX fills the result
with NaN.  The CPU path then had no fallback: a CTM projector cut went NaN and
the whole environment followed (found on #1144's macOS leg).

Linux ``gesdd`` converges on the captured matrix, so the mechanism tests fake
the failure.  The captured-matrix test is the real regression on macOS.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

import tenax.linalg as tl

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    jax.default_backend() == "gpu", reason="the CPU SVD path only"
)

_REAL_GESDD = jnp.linalg.svd


def _x(shape=(7, 5), seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


@pytest.fixture
def failing_gesdd(monkeypatch):
    """``jnp.linalg.svd`` returns NaN for the matrices whose [0, 0] entry is > 0."""

    def fake(m, full_matrices=True, compute_uv=True):
        r = _REAL_GESDD(m, full_matrices=full_matrices, compute_uv=compute_uv)
        bad = m[..., 0, 0] > 0

        def poison(x):
            mask = bad.reshape(bad.shape + (1,) * (x.ndim - bad.ndim))
            return jnp.where(mask, jnp.nan, x)

        return tuple(poison(x) for x in r) if compute_uv else poison(r)

    monkeypatch.setattr(jnp.linalg, "svd", fake)


def _assert_is_the_svd_of(m, u, s, vh):
    m = np.asarray(m)
    np.testing.assert_allclose(s, np.linalg.svd(m, compute_uv=False), rtol=1e-12)
    np.testing.assert_allclose(
        (np.asarray(u) * np.asarray(s)[..., None, :]) @ vh, m, atol=1e-12
    )


def test_a_converged_gesdd_result_is_returned_unchanged():
    x = _x()
    got = tl._dense_svd(x)
    for a, b in zip(got, _REAL_GESDD(x, full_matrices=False), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_a_nan_gesdd_result_is_replaced_by_gesvd(failing_gesdd, mode):
    x = _x().at[0, 0].set(1.0)
    f = tl._dense_svd if mode == "eager" else jax.jit(tl._dense_svd)
    _assert_is_the_svd_of(x, *f(x))


def test_the_singular_values_only_call_falls_back_too(failing_gesdd):
    x = _x().at[0, 0].set(1.0)
    s = tl._dense_svd(x, compute_uv=False)
    np.testing.assert_allclose(
        s, np.linalg.svd(np.asarray(x), compute_uv=False), rtol=1e-12
    )


def test_a_failed_matrix_in_a_vmap_batch_is_recomputed(failing_gesdd):
    xs = jnp.stack([_x(seed=k) for k in range(3)])
    xs = xs.at[:, 0, 0].set(jnp.array([-1.0, 1.0, -1.0]))  # only [1] fails
    u, s, vh = jax.vmap(tl._dense_svd)(xs)
    for k in range(3):
        _assert_is_the_svd_of(xs[k], u[k], s[k], vh[k])


def test_the_gradient_goes_through_the_fallback(failing_gesdd):
    x = _x().at[0, 0].set(1.0)

    def loss(m):
        u, s, vh = tl._dense_svd(m)
        return jnp.sum(s**2) + jnp.sum(s)

    # d(sum s)/dM = U Vh and d(sum s^2)/dM = 2M, from the real SVD.
    u, _, vh = _REAL_GESDD(x, full_matrices=False)
    np.testing.assert_allclose(jax.grad(loss)(x), 2 * x + u @ vh, atol=1e-10)


def test_a_non_finite_input_stays_non_finite():
    s = tl._dense_svd(_x().at[0, 0].set(jnp.nan), compute_uv=False)
    assert not np.all(np.isfinite(s))


def test_the_matrix_accelerate_gesdd_does_not_converge_on():
    # A 54x54 bottom-projector matrix from #1144's frozen-layout test, cond 1e8,
    # smallest singular values 1.5e-9 and 1.2e-9.  On macOS arm64 numpy and
    # scipy gesdd raise "SVD did not converge" and jnp.linalg.svd returns NaN.
    m = np.load(
        Path(__file__).parent / "data" / "accelerate_gesdd_nonconvergent_54x54.npy"
    )
    u, s, vh = tl._dense_svd(jnp.asarray(m))
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(vh))
    # gesvd as the reference: numpy's gesdd is the routine that fails on macOS.
    ref = scipy.linalg.svd(m, compute_uv=False, lapack_driver="gesvd")
    np.testing.assert_allclose(s, ref, rtol=1e-6, atol=1e-14)
    np.testing.assert_allclose((np.asarray(u) * np.asarray(s)) @ vh, m, atol=1e-12)
