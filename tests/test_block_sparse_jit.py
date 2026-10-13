"""Block-sparse CPU programs compile without XLA's fusion pass.

On a SymmetricTensor CTM-AD backward program the CPU fusion pass is
superlinear: the D=3 U(1) ``dE/denv`` program compiled in 8569 s with it and
905 s without, and ran faster without it (see ``_block_sparse_jit``).  The
forward sweep is the other way round, so it keeps the fused pipeline.  These
tests pin the dispatch and that the two pipelines compute the same numbers.
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
from tenax.core.tensor import DenseTensor

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


@pytest.fixture(scope="module")
def u1_pair():
    return heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))


def _norm2(t):
    return sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(t))


@pytest.fixture
def any_size(monkeypatch):
    """Drop the size threshold, so the D=2 fixture dispatches as a large state."""
    monkeypatch.setattr(_block_sparse_jit, "NO_FUSION_MIN_BLOCKS", 1)


# Each test jits a fresh function: jit objects over one Python function share
# a trace cache, so ``_cache_size()`` would count another test's traces.
def test_a_symmetric_call_takes_the_no_fusion_variant(u1_pair, any_size):
    f = block_sparse_jit(lambda t: _norm2(t))
    A, _ = u1_pair
    f(A)
    assert f.block_sparse._cache_size() == 1
    assert f.dense._cache_size() == 0


def test_a_small_symmetric_state_keeps_the_default_pipeline(u1_pair):
    """A D=2 U(1) 2-site state is 10 blocks: fusion still wins there."""
    f = block_sparse_jit(lambda a, b: _norm2(a) + _norm2(b))
    A, B = u1_pair
    assert len(A.blocks) + len(B.blocks) < _block_sparse_jit.NO_FUSION_MIN_BLOCKS
    f(A, B)
    assert f.dense._cache_size() == 1
    assert f.block_sparse._cache_size() == 0


def test_a_dense_call_keeps_the_default_pipeline(u1_pair, any_size):
    f = block_sparse_jit(lambda t: _norm2(t))
    A, _ = u1_pair
    f(DenseTensor(A.todense(), A.indices))
    f(jnp.ones(3))
    assert f.dense._cache_size() == 2
    assert f.block_sparse._cache_size() == 0


def test_a_call_inside_another_trace_takes_the_default_variant(u1_pair, any_size):
    """JAX refuses ``compiler_options`` on a nested jit, so under an outer
    ``jax.jit`` or ``jax.grad`` the inner call must not ask for them."""
    f = block_sparse_jit(lambda t: _norm2(t))
    A, _ = u1_pair
    outer = float(jax.jit(f)(A))
    g = jax.grad(lambda t: f(t))(A)
    assert np.isclose(outer, float(_norm2(A)))
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(g))
    assert f.block_sparse._cache_size() == 0


def test_a_closure_inside_an_outer_jit_takes_the_default_variant(u1_pair, any_size):
    """An outer ``jax.jit`` that closes over a concrete tensor passes no
    tracer in, so the check reads the trace state, not the arguments."""
    f = block_sparse_jit(lambda t: _norm2(t))
    A, _ = u1_pair
    out = jax.jit(lambda x: f(A) + x)(1.0)
    assert np.isclose(float(out), float(_norm2(A)) + 1.0)
    assert f.block_sparse._cache_size() == 0


def test_a_custom_vjp_backward_takes_the_no_fusion_variant(u1_pair, any_size):
    """The implicit-AD backward programs run inside a ``custom_vjp`` backward.
    ``jax.grad`` calls it outside any trace, so it gets the unfused variant."""
    inner = block_sparse_jit(lambda t: _norm2(t))

    @jax.custom_vjp
    def energy(t):
        return _norm2(t)

    def fwd(t):
        return _norm2(t), t

    def bwd(t, g):
        inner(t)
        return (jax.tree.map(lambda x: 2 * g * x, t),)

    energy.defvjp(fwd, bwd)
    A, _ = u1_pair
    jax.grad(energy)(A)
    assert inner.block_sparse._cache_size() == 1
    assert inner.dense._cache_size() == 0


def test_without_compiler_options_every_call_keeps_the_default(
    u1_pair, any_size, monkeypatch
):
    """JAX 0.4.30-0.4.35 have no ``compiler_options`` on ``jax.jit``."""
    monkeypatch.setattr(_block_sparse_jit, "JIT_TAKES_COMPILER_OPTIONS", False)
    f = block_sparse_jit(lambda t: _norm2(t))
    A, _ = u1_pair
    f(A)
    assert f.block_sparse is f.dense
    assert f.dense._cache_size() == 1


def test_static_argnames_pass_through(u1_pair):
    @block_sparse_jit(static_argnames=("k",))
    def f(t, *, k):
        return k * _norm2(t)

    A, _ = u1_pair
    np.testing.assert_allclose(float(f(A, k=3)), 3 * float(_norm2(A)), rtol=1e-14)


def test_the_forward_sweep_keeps_the_fused_pipeline():
    """Fusion helps the forward sweep at every size measured: the D=3 U(1)
    chi=9 sweep (225k StableHLO ops) compiled in 145 s fused and 517 s
    unfused, and ran 15.9 ms against 19.8 ms."""
    step = _make_jit_ctm_step(CHECKERBOARD_NEIGHBORS, "2x2")
    assert not hasattr(step, "block_sparse")


def test_a_sweep_gradient_is_the_same_with_and_without_fusion(u1_pair):
    """A real block-sparse backward program, compiled both ways."""
    A, B = u1_pair
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
