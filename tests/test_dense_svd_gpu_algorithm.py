"""On GPU, ``_dense_svd`` picks the SVD algorithm by size: polar from 128, QR below,
per lowering platform (a CPU operand in a GPU process takes the CPU path).

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
def test_gpu_algorithm_is_polar_from_128(shape, expected):
    if not hasattr(jax.lax.linalg.SvdAlgorithm, "POLAR"):
        pytest.skip("this JAX has no SvdAlgorithm.POLAR; the GPU path stays on QR")
    assert tl._gpu_svd_algorithm(shape) == getattr(
        jax.lax.linalg.SvdAlgorithm, expected
    )


def test_cpu_branch_is_unchanged():
    # CPU keeps gesdd; gesvd is only its non-convergence fallback
    # (test_cpu_svd_gesdd_fallback.py).
    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only check")
    x = jnp.asarray(np.random.default_rng(0).standard_normal((200, 200)))
    for a, b in zip(
        tl._dense_svd(x), jnp.linalg.svd(x, full_matrices=False), strict=True
    ):
        np.testing.assert_array_equal(a, b)


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


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a CUDA device")
@pytest.mark.parametrize("mode", ["eager", "jit", "grad", "default_device"])
def test_cpu_operand_in_a_gpu_process_takes_the_cpu_path(mode):
    # The algorithm follows the operand's lowering platform, not the process
    # default: polar has no CPU lowering (NotImplementedError before the fix).
    cpu = jax.devices("cpu")[0]
    x = np.random.default_rng(0).standard_normal((200, 200))
    svals = lambda m: tl._dense_svd(m)[1]  # noqa: E731
    if mode == "eager":
        s = svals(jax.device_put(x, cpu))
    elif mode == "jit":
        s = jax.jit(svals)(jax.device_put(x, cpu))
    elif mode == "grad":
        s = jax.grad(lambda m: jnp.sum(svals(m)))(jax.device_put(x, cpu))
        assert next(iter(s.devices())).platform == "cpu"
        return
    else:
        with jax.default_device(cpu):
            s = svals(jnp.asarray(x))
    assert next(iter(s.devices())).platform == "cpu"
    np.testing.assert_allclose(s, np.linalg.svd(x, compute_uv=False), rtol=1e-12)
