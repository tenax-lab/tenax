"""``SymmetricTensor.blocks`` never caches a tracer on a concrete tensor (#1103).

The block dict is cached on the tensor the first time it is read.  If that
first read happens inside a ``jax.jit`` that closes over a concrete tensor,
the slices of ``_data`` are tracers, and caching them left tracers on a
tensor that outlives the trace: the next eager use raised
``UnexpectedTracerError``.  It hit ``ctm_energy_implicit`` with a
``SymmetricTensor`` gate, which the compiled energy programs close over.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import ZnSymmetry
from tenax.core.tensor import SymmetricTensor

jax.config.update("jax_enable_x64", True)


def _tensor(seed=0):
    sym = ZnSymmetry(2)
    q = np.array([0, 1, 1], dtype=np.int32)
    return SymmetricTensor.random_normal(
        (
            TensorIndex.from_charges(sym, q, FlowDirection.IN, label="a"),
            TensorIndex.from_charges(sym, q, FlowDirection.OUT, label="b"),
        ),
        jax.random.PRNGKey(seed),
    )


def _block_sum(t):
    return sum(jnp.sum(b) for b in t.blocks.values())


@pytest.mark.parametrize(
    "transform",
    [
        lambda f: jax.jit(f)(1.0),
        lambda f: jax.grad(f)(1.0),
        lambda f: jax.vmap(f)(jnp.ones(2)),
        lambda f: jax.jit(jax.grad(f))(1.0),
    ],
    ids=["jit", "grad", "vmap", "jit-grad"],
)
def test_a_first_read_inside_a_trace_leaves_no_tracer(transform):
    A = _tensor()
    expected = np.asarray(_tensor().todense())
    transform(lambda x: x * _block_sum(A))
    assert not any(isinstance(b, jax.core.Tracer) for b in A.blocks.values())
    np.testing.assert_array_equal(np.asarray(A.todense()), expected)


def test_the_cache_still_serves_eager_reads():
    """The fix must not drop the cache for concrete tensors read eagerly."""
    A = _tensor()
    assert A.blocks is A.blocks


def test_a_cache_filled_eagerly_is_reused_inside_a_trace():
    A = _tensor()
    blocks = A.blocks
    out = jax.jit(lambda x: x * _block_sum(A))(2.0)
    assert A.blocks is blocks
    np.testing.assert_allclose(float(out), 2.0 * float(_block_sum(A)), rtol=1e-14)


def test_a_tensor_built_inside_a_trace_reads_its_blocks():
    """A tensor whose data is itself traced may cache within its trace."""

    def f(x):
        T = _tensor() * x
        return _block_sum(T) + _block_sum(T)

    out = jax.jit(f)(3.0)
    np.testing.assert_allclose(
        float(out), 6.0 * float(_block_sum(_tensor())), rtol=1e-12
    )
