"""The CPU SVD falls back to ``gesvd`` when LAPACK ``gesdd`` does not converge.

Apple Accelerate's ``gesdd`` on the ``macos-26-arm64`` runner fails to
converge on some finite, ill-conditioned matrices, and JAX fills the result
with NaN.  The CPU path then had no fallback: a CTM projector cut went NaN and
the whole environment followed (found on #1144's macOS leg).

Linux ``gesdd`` converges on the captured matrix, so the mechanism tests fake
the failure.  The fake runs the real ``gesdd`` on a NaN-poisoned input, so the
failed factors *and their derivative* are NaN, as a real non-convergence is.
The captured-matrix test is the real regression only where ``gesdd`` fails.
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
    jax.default_backend() != "cpu", reason="the CPU SVD path only"
)

_REAL_GESDD = jnp.linalg.svd


def _x(shape=(7, 5), seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


def _fails(m):
    return m.at[..., 0, 0].set(1.0)


@pytest.fixture
def failing_gesdd(monkeypatch):
    """``gesdd`` returns NaN for the matrices whose [0, 0] entry is > 0."""

    def fake(m, full_matrices=True, compute_uv=True):
        # Add a stop-gradient NaN, not jnp.where(bad, nan, m): where's
        # transpose would zero the NaN cotangent and hide the derivative bug.
        bad = (m[..., 0, 0] > 0)[..., None, None]
        nan = jax.lax.stop_gradient(jnp.where(bad, jnp.nan, 0.0))
        return _REAL_GESDD(
            m + nan,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
        )

    # jax.jit(f) reuses an earlier trace of f, which would skip the fake.
    jax.clear_caches()
    monkeypatch.setattr(jnp.linalg, "svd", fake)
    yield
    jax.clear_caches()


needs_gesvd = pytest.mark.skipif(
    not tl._cpu_has_gesvd(), reason="this JAX has no CPU gesvd; no fallback"
)


def _assert_is_the_svd_of(m, u, s, vh):
    m = np.asarray(m)
    np.testing.assert_allclose(s, np.linalg.svd(m, compute_uv=False), rtol=1e-12)
    np.testing.assert_allclose(
        (np.asarray(u) * np.asarray(s)[..., None, :]) @ vh, m, atol=1e-12
    )


def test_a_converged_gesdd_result_is_returned_unchanged():
    x = _x()
    for f in (tl._dense_svd, jax.jit(tl._dense_svd)):
        got = f(x)
        for a, b in zip(got, _REAL_GESDD(x, full_matrices=False), strict=True):
            np.testing.assert_array_equal(a, b)


@needs_gesvd
@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_a_nan_gesdd_result_is_replaced_by_gesvd(failing_gesdd, mode):
    x = _fails(_x())
    f = tl._dense_svd if mode == "eager" else jax.jit(tl._dense_svd)
    _assert_is_the_svd_of(x, *f(x))


@needs_gesvd
def test_the_singular_values_only_call_falls_back_too(failing_gesdd):
    x = _fails(_x())
    s = tl._dense_svd(x, compute_uv=False)
    np.testing.assert_allclose(
        s, np.linalg.svd(np.asarray(x), compute_uv=False), rtol=1e-12
    )


@needs_gesvd
def test_a_failed_matrix_in_a_vmap_batch_is_recomputed(failing_gesdd):
    xs = jnp.stack([_x(seed=k) for k in range(3)])
    xs = xs.at[:, 0, 0].set(jnp.array([-1.0, 1.0, -1.0]))  # only [1] fails
    u, s, vh = jax.vmap(tl._dense_svd)(xs)
    for k in range(3):
        _assert_is_the_svd_of(xs[k], u[k], s[k], vh[k])


def _loss(m):
    u, s, vh = tl._dense_svd(m)
    # Gauge-invariant in u and vh: sum s + |u diag(s) vh|^2 (= sum s^2).
    return jnp.sum(s) + jnp.sum(((u * s[None, :]) @ vh) ** 2)


def _expected_grad(x):
    u, _, vh = _REAL_GESDD(x, full_matrices=False)
    return u @ vh + 2 * x  # d(sum s)/dM = U Vh, d|M|^2/dM = 2M


@needs_gesvd
@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_the_gradient_through_a_failed_gesdd_is_finite_and_right(failing_gesdd, mode):
    # Differentiating the failed gesdd itself gives NaN (0 * NaN in the
    # transpose) even when the forward result is the gesvd one.
    x = _fails(_x())
    g = jax.grad(_loss) if mode == "eager" else jax.jit(jax.grad(_loss))
    np.testing.assert_allclose(g(x), _expected_grad(x), atol=1e-10)


@needs_gesvd
@pytest.mark.parametrize(
    ("shape", "dtype", "compute_uv"),
    [
        ((7, 5), np.float64, True),
        ((5, 7), np.float64, True),
        ((6, 6), np.float64, True),
        ((6, 4), np.complex128, True),
        ((7, 5), np.float64, False),
    ],
)
def test_the_custom_jvp_equals_jaxs_svd_jvp(shape, dtype, compute_uv):
    rng = np.random.default_rng(1)

    def rnd():
        r = rng.standard_normal(shape)
        if dtype == np.complex128:
            r = r + 1j * rng.standard_normal(shape)
        return jnp.asarray(r.astype(dtype))

    x, dx = rnd(), rnd()
    ours = jax.jvp(lambda m: tl._dense_svd(m, compute_uv=compute_uv), (x,), (dx,))
    jaxs = jax.jvp(
        lambda m: _REAL_GESDD(m, full_matrices=False, compute_uv=compute_uv),
        (x,),
        (dx,),
    )
    for a, b in zip(jax.tree.leaves(ours), jax.tree.leaves(jaxs), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


@needs_gesvd
def test_an_eager_call_traces_no_cond(monkeypatch):
    # An eager lax.cond builds new branch closures per call, so JAX re-traces
    # and re-compiles it every time (#1087's no-recompile test caught this).
    def no_cond(*a, **k):
        raise AssertionError("eager SVD went through lax.cond")

    monkeypatch.setattr(jax.lax, "cond", no_cond)
    x = _x()
    tl._dense_svd(x)
    jax.grad(_loss)(x)


def test_a_jax_without_cpu_gesvd_keeps_plain_gesdd(monkeypatch, failing_gesdd):
    # The 0.4.3x floor cannot lower SvdAlgorithm.QR on CPU.
    monkeypatch.setattr(tl, "_cpu_has_gesvd", lambda: False)
    monkeypatch.setattr(tl, "_gesvd", lambda *a: pytest.fail("gesvd was used"))
    assert not np.all(np.isfinite(tl._dense_svd(_fails(_x()), compute_uv=False)))


@pytest.fixture
def fresh_probe():
    tl._cpu_has_gesvd.cache_clear()
    yield
    tl._cpu_has_gesvd.cache_clear()


def test_the_cpu_gesvd_probe_runs_eagerly_inside_a_jit_trace(monkeypatch, fresh_probe):
    # Old JAX fails only when QR is lowered for CPU: a staged probe inside the
    # caller's jit trace would "succeed" and cache True.  This fake gesvd
    # traces fine and raises only when run on concrete values.
    def gesvd(x, full_matrices, compute_uv):
        if not isinstance(x, jax.core.Tracer):
            raise NotImplementedError("no CPU gesvd")
        return _REAL_GESDD(x, full_matrices=full_matrices, compute_uv=compute_uv)

    monkeypatch.setattr(tl, "_gesvd", gesvd)
    seen = []

    def f(y):
        seen.append(tl._cpu_has_gesvd())
        return y

    jax.jit(f)(_x())
    assert seen == [False]


def test_a_non_cpu_backend_skips_the_fallback(monkeypatch):
    # TPU supports only SvdAlgorithm.POLAR, so the QR branch must not lower there.
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(tl, "_cpu_svd", lambda *a: pytest.fail("_cpu_svd was used"))
    tl._dense_svd(_x())


def test_a_non_finite_input_stays_non_finite():
    s = tl._dense_svd(_x().at[0, 0].set(jnp.nan), compute_uv=False)
    assert not np.all(np.isfinite(s))


@needs_gesvd
def test_the_matrix_accelerate_gesdd_does_not_converge_on():
    # A 54x54 bottom-projector matrix from #1144's frozen-layout test, cond 1e8,
    # smallest singular values 1.5e-9 and 1.2e-9.  On the macos-26-arm64 runner
    # numpy and scipy gesdd raise "SVD did not converge" and jnp.linalg.svd
    # returns NaN; macOS 27 (M2 Pro) and Linux converge.
    m = np.load(
        Path(__file__).parent / "data" / "accelerate_gesdd_nonconvergent_54x54.npy"
    )
    u, s, vh = tl._dense_svd(jnp.asarray(m))
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(vh))
    # gesvd as the reference: numpy's gesdd is the routine that fails there.
    ref = scipy.linalg.svd(m, compute_uv=False, lapack_driver="gesvd")
    np.testing.assert_allclose(s, ref, rtol=1e-6, atol=1e-14)
    np.testing.assert_allclose((np.asarray(u) * np.asarray(s)) @ vh, m, atol=1e-12)
