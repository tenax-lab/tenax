"""Verify SymmetricTensor works as custom_vjp input/output in JAX.

This is a JAX capability validation: SymmetricTensor is a registered pytree
whose single leaf is a flat 1D data buffer (_data).  We confirm that
jax.custom_vjp can accept SymmetricTensor arguments, propagate cotangents
through the flat buffer, and produce SymmetricTensor gradients.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor


def _make_test_symmetric(dtype=jnp.float64):
    """Small U(1)-symmetric 2-leg tensor with charges [0, 1]."""
    sym = U1Symmetry()
    charges = np.array([0, 1], dtype=np.int32)
    idx_in = TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="a")
    idx_out = TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="b")
    return SymmetricTensor.random_normal(
        (idx_in, idx_out), jax.random.PRNGKey(0), dtype=dtype
    )


def test_custom_vjp_with_symmetric_tensor_leaf():
    """SymmetricTensor._data is the pytree leaf; custom_vjp should work."""
    t = _make_test_symmetric()

    @jax.custom_vjp
    def f(tensor):
        # Forward: sum of squares over the flat data buffer.
        return jnp.sum(tensor._data**2)

    def f_fwd(tensor):
        y = f(tensor)
        return y, tensor  # residuals

    def f_bwd(tensor, g):
        # Analytic gradient: d/dx sum(x^2) = 2x, broadcast by g.
        # Build a new SymmetricTensor with the same structure but grad values.
        grad_blocks = {k: 2 * g * tensor.blocks[k] for k in tensor.blocks}
        return (SymmetricTensor(grad_blocks, tensor.indices),)

    f.defvjp(f_fwd, f_bwd)

    grad_fn = jax.grad(lambda tensor: f(tensor))
    gt = grad_fn(t)

    assert isinstance(gt, SymmetricTensor)
    # Gradient of sum(x^2) = 2x for each block.
    for k in t.blocks:
        np.testing.assert_allclose(gt.blocks[k], 2 * t.blocks[k], rtol=1e-12)


def test_custom_vjp_pytree_round_trip_preserves_structure():
    """The gradient SymmetricTensor has the same block keys and indices."""
    t = _make_test_symmetric()

    @jax.custom_vjp
    def f(tensor):
        return jnp.sum(tensor._data**2)

    def f_fwd(tensor):
        return f(tensor), tensor

    def f_bwd(tensor, g):
        grad_blocks = {k: 2 * g * tensor.blocks[k] for k in tensor.blocks}
        return (SymmetricTensor(grad_blocks, tensor.indices),)

    f.defvjp(f_fwd, f_bwd)

    gt = jax.grad(lambda tensor: f(tensor))(t)

    assert gt._block_keys == t._block_keys
    assert gt._block_shapes == t._block_shapes
    assert gt.indices == t.indices


def test_custom_vjp_jit_compatible():
    """custom_vjp + SymmetricTensor works under jax.jit."""
    t = _make_test_symmetric()

    @jax.custom_vjp
    def f(tensor):
        return jnp.sum(tensor._data**2)

    def f_fwd(tensor):
        return f(tensor), tensor

    def f_bwd(tensor, g):
        grad_blocks = {k: 2 * g * tensor.blocks[k] for k in tensor.blocks}
        return (SymmetricTensor(grad_blocks, tensor.indices),)

    f.defvjp(f_fwd, f_bwd)

    grad_fn = jax.jit(jax.grad(lambda tensor: f(tensor)))
    gt = grad_fn(t)

    assert isinstance(gt, SymmetricTensor)
    for k in t.blocks:
        np.testing.assert_allclose(gt.blocks[k], 2 * t.blocks[k], rtol=1e-12)
