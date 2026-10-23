"""On GPU, ``_dense_svd`` picks the SVD algorithm by size: polar from 128, QR below.

``gesvd`` launches ~250 tiny kernels per call, which made the A100 CTM sweep
launch-bound; polar (``gesvdp``) was 1.3-1.4x faster on the t-V sweep at
D=4/5 but about 2x slower below ~100.  Jacobi stays excluded (#691).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.linalg as tl

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((200, 200), "POLAR"),
        ((128, 300), "POLAR"),
        ((127, 400), "QR"),
        ((50, 50), "QR"),
        ((3, 200, 200), "POLAR"),
    ],
)
def test_gpu_branch_selects_polar_from_128(monkeypatch, shape, expected):
    if not hasattr(jax.lax.linalg.SvdAlgorithm, "POLAR"):
        pytest.skip("this JAX has no SvdAlgorithm.POLAR; the GPU path stays on QR")
    seen = []

    def record(matrix, *, full_matrices, compute_uv, algorithm):
        seen.append(algorithm)
        return jnp.linalg.svd(
            matrix, full_matrices=full_matrices, compute_uv=compute_uv
        )

    monkeypatch.setattr(tl.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(tl, "_lax_svd", record)
    m = jnp.asarray(np.random.default_rng(0).standard_normal(shape))
    u, s, vh = tl._dense_svd(m)
    assert seen == [getattr(jax.lax.linalg.SvdAlgorithm, expected)]
    np.testing.assert_allclose((u * s[..., None, :]) @ vh, m, atol=1e-10)


def test_cpu_branch_is_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(tl, "_lax_svd", lambda *a, **k: calls.append(k))
    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only check")
    tl._dense_svd(jnp.eye(200))
    assert calls == []


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a CUDA device")
def test_gpu_compiles_to_gesvdp_from_128():
    def targets(n):
        # Exact custom-call targets: HLO metadata carries Python frame names,
        # so a substring search would match this test's own name.
        hlo = jax.jit(lambda x: tl._dense_svd(x)).lower(jnp.eye(n)).compile().as_text()
        tag = 'custom_call_target="'
        return sorted(
            {ln.split(tag)[1].split('"')[0] for ln in hlo.splitlines() if tag in ln}
        )

    assert targets(200) == ["cusolver_gesvdp_ffi"]
    assert targets(50) == ["cusolver_gesvd_ffi"]
