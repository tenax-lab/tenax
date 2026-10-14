"""The fused and unfused pipelines of ``block_sparse_jit`` agree.

Kept out of ``test_block_sparse_jit.py`` (``core``): this compiles a real
block-sparse CTM backward twice, which takes about a minute on CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _block_sparse_jit
from tenax.algorithms._block_sparse_jit import block_sparse_jit
from tenax.algorithms._ctm_python_loop import _make_jit_ctm_step
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

jax.config.update("jax_enable_x64", True)

pytestmark = [
    pytest.mark.skipif(
        jax.default_backend() != "cpu", reason="the no-fusion variant is CPU-only"
    ),
    pytest.mark.skipif(
        not _block_sparse_jit.JIT_TAKES_COMPILER_OPTIONS,
        reason="jax.jit takes compiler_options from JAX 0.4.36",
    ),
]


def _norm2(t):
    return sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(t))


def test_a_sweep_gradient_is_the_same_with_and_without_fusion():
    """A real block-sparse backward program, compiled both ways."""
    A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
    sites = {(0, 0): A, (1, 0): B}
    envs = {c: initialize_ctm_tensor_env(t, 8) for c, t in sites.items()}
    sweep = _make_jit_ctm_step(CHECKERBOARD_NEIGHBORS, "2x2").__wrapped__

    def loss(sites):
        new_envs, _, _ = sweep(
            sites, envs, chi=8, projector_method="svd", renormalize=True
        )
        return _norm2(new_envs)

    grad = block_sparse_jit(jax.grad(loss))
    a = jax.tree.leaves(grad.block_sparse(sites))
    b = jax.tree.leaves(grad.dense(sites))
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-11, atol=1e-13)
