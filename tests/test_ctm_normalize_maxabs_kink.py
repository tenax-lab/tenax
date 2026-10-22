"""The CTM max-abs renormalisation: max-abs in value, smooth in gradient.

``max`` has a kink wherever the largest ``|entry|`` is tied, and the
double-layer CTM environment ties it exactly (entries related by the
ket<->bra swap).  Differentiated there, the backward depended on which tied
entry rounding picked: a 1e-15 relative change to a D=3 U(1) chi=9
environment moved the implicit-AD ``J^T v`` by 15%, and JAX and torch
evaluating the same program disagreed by as much.

Freezing the scale instead (``stop_gradient``) removes the kink but also the
projection of the scale direction out of the CTM Jacobian, which the adjoint
solve needs: GMRES on ``tests/test_adjoint_convergence_gate.py`` went from
converged to a 7e-4 residual.  So both properties are pinned here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import (
    _normalize_tensor as _normalize_convergence,
)
from tenax.algorithms._ctm_tensor_moves import _normalize_tensor as _normalize_moves
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

jax.config.update("jax_enable_x64", True)


def _tied_symmetric():
    u1 = U1Symmetry()
    q = np.array([-1, 0, 0, 1], dtype=np.int32)
    idx = (
        TensorIndex.from_charges(u1, q, FlowDirection.IN, label="a"),
        TensorIndex.from_charges(u1, u1.dual(q), FlowDirection.OUT, label="b"),
    )
    t = SymmetricTensor.random_normal(idx, jax.random.PRNGKey(0))
    leaves, treedef = jax.tree_util.tree_flatten(t)
    (data,) = leaves
    m = 2.0 * float(jnp.max(jnp.abs(data)))
    data = data.at[0].set(m).at[-1].set(-m)  # exact tie in |entry|, opposite signs
    return data, treedef


def _tied_dense():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4))
    m = 2.0 * np.max(np.abs(a))
    a[0, 0], a[2, 3] = m, -m
    idx = (
        TensorIndex.from_charges(
            U1Symmetry(), np.zeros(3, np.int32), FlowDirection.IN, label="a"
        ),
        TensorIndex.from_charges(
            U1Symmetry(), np.zeros(4, np.int32), FlowDirection.OUT, label="b"
        ),
    )
    leaves, treedef = jax.tree_util.tree_flatten(DenseTensor(jnp.asarray(a), idx))
    (data,) = leaves
    return data, treedef


@pytest.mark.parametrize("normalize", [_normalize_moves, _normalize_convergence])
@pytest.mark.parametrize("make", [_tied_symmetric, _tied_dense])
def test_normalize_is_smooth_and_scale_invariant_at_a_max_tie(normalize, make):
    data, treedef = make()

    def f(x):
        (out,) = jax.tree_util.tree_leaves(
            normalize(jax.tree_util.tree_unflatten(treedef, [x]))
        )
        return out

    # The forward is exactly the max-abs normalisation.
    m = jnp.max(jnp.abs(data))
    np.testing.assert_array_equal(f(data), data * (1.0 / (m + EPS)))

    # The VJP does not jump across the tie.  Break the tie each way by nudging
    # one tied entry up; a generic cotangent has a radial part, which is where
    # the ``d max / dT`` one-hot term lived.
    flat = data.ravel()
    tied = np.flatnonzero(np.abs(np.asarray(flat)) == float(m))
    assert tied.size == 2
    sides = [flat.at[k].multiply(1 + 1e-14).reshape(data.shape) for k in tied]
    v = jax.random.normal(jax.random.PRNGKey(2), data.shape)
    grads = [jax.vjp(f, x)[1](v)[0] for x in (data, *sides)]
    scale = float(jnp.max(jnp.abs(grads[0])))
    for g in grads[1:]:
        assert float(jnp.max(jnp.abs(g - grads[0]))) <= 1e-12 * scale

    # The scale direction is still projected out (J T = 0); the implicit-AD
    # adjoint solve depends on it.
    _, tangent = jax.jvp(f, (data,), (data,))
    assert float(jnp.max(jnp.abs(tangent))) <= 1e-12 * float(jnp.max(jnp.abs(f(data))))
