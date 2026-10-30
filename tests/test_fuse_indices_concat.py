"""Block-sparse ``fuse_indices``/``split_index`` against the dense reference.

``_fuse_indices_symmetric`` joins the sub-blocks of each fused block with one
``concatenate`` and, only for legs whose charges are not grouped by sector, one
permutation (#566).  What it must get right is the layout: ``fuse(T).todense()``
is ``T.todense()`` with the two legs transposed next to each other (sorted, at
the smaller axis) and reshaped row-major into one -- values, AD and ``jvp``
included -- and ``split_index`` inverts it.

The 2026-10-26 test-trim pass replaced the verbatim copy of the pre-#566
scatter implementation this file used as its reference with that dense
reference (stricter: it does not trust any block-sparse code), cut the
parametrisation to three axis pairs per (symmetry, grouping), and deleted the
two jaxpr op-count tests (``no scatter`` in the forward / one ``scatter-add``
per permutation in the backward): those pinned how #566 is implemented, not
what ``fuse_indices`` returns.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._tensor_utils import (
    _compute_fused_charges,
    fuse_indices,
    split_index,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry, ZnSymmetry
from tenax.core.tensor import SymmetricTensor

jax.config.update("jax_enable_x64", True)

IN, OUT = FlowDirection.IN, FlowDirection.OUT


# --- fixtures ---------------------------------------------------------------


def _leg(sym, charges, flow, label):
    return TensorIndex.from_charges(sym, np.asarray(charges, np.int32), flow, label)


def _legs(sym, grouped):
    """Four legs; ``grouped=False`` interleaves charges inside each leg."""
    if isinstance(sym, U1Symmetry):
        base = [[-1, 0, 0, 1], [0, 1, 1, -1, 0], [1, 0, -1], [0, 0, 1, -1, 2]]
    else:
        n = 2 if isinstance(sym, FermionParity) else sym.n
        base = [[0, 1, 1], [1, 0, 0, 1], [0, 1, 0], [1, 1, 0, 0, 1]]
        base = [[q % n for q in leg] for leg in base]
    if grouped:
        base = [sorted(leg) for leg in base]
    flows = (IN, OUT, IN, OUT)
    return tuple(
        _leg(sym, q, f, lab) for q, f, lab in zip(base, flows, ("a", "b", "c", "d"))
    )


SYMS = {
    "u1": U1Symmetry(),
    "z3": ZnSymmetry(3),
    "parity": FermionParity(),
}
CASES = [
    (name, grouped, ax)
    for name in SYMS
    for grouped in (True, False)
    # Adjacent in order, non-adjacent in order, non-adjacent reversed (the
    # fused leg lands at 0 and the legs between shift).  Six axis pairs per
    # (symmetry, grouping) until the 2026-10-26 test-trim pass.
    for ax in [(0, 1), (1, 3), (3, 0)]
]


def _tensor(name, grouped, seed=0):
    return SymmetricTensor.random_normal(
        _legs(SYMS[name], grouped), jax.random.PRNGKey(seed)
    )


def _dense_fuse(dense, ax):
    """The dense reference: legs ``ax`` transposed adjacent (sorted, at
    ``min(ax)``) and merged row-major."""
    a = min(ax)
    order = [i for i in range(dense.ndim) if i not in ax]
    order[a:a] = sorted(ax)
    d = jnp.transpose(dense, order)
    shape = list(d.shape)
    shape[a : a + 2] = [shape[a] * shape[a + 1]]
    return d.reshape(shape)


def _assert_fuses_like_dense(T, ax, flow):
    got = fuse_indices(T, ax[0], ax[1], "f", flow)
    np.testing.assert_array_equal(
        np.asarray(got.todense()), np.asarray(_dense_fuse(T.todense(), ax))
    )
    # The fused leg's metadata: flow as asked, charges the product charges.
    a, b = sorted(ax)
    fused = got.indices[a]
    assert fused.flow == flow
    np.testing.assert_array_equal(
        np.asarray(fused.charges),
        _compute_fused_charges(T.indices[a], T.indices[b], flow, fused.symmetry),
    )
    others = [i for i in range(T.ndim) if i not in ax]
    rest = [ix for i, ix in enumerate(got.indices) if i != a]
    assert all(
        np.array_equal(x.charges, T.indices[i].charges) and x.flow == T.indices[i].flow
        for x, i in zip(rest, others)
    )
    return got


# --- tests ------------------------------------------------------------------


@pytest.mark.parametrize("name,grouped,ax", CASES)
@pytest.mark.parametrize("flow", [IN, OUT])
def test_fuse_matches_the_dense_reference(name, grouped, ax, flow):
    _assert_fuses_like_dense(_tensor(name, grouped), ax, flow)


@pytest.mark.parametrize("name,grouped,ax", CASES)
def test_split_inverts_fuse(name, grouped, ax):
    T = _tensor(name, grouped)
    fused = fuse_indices(T, ax[0], ax[1], "f", OUT)
    a = min(ax)
    got = split_index(fused, a)
    # split(fuse(T)) is T with the two legs adjacent at ``a``, in sorted order.
    order = [i for i in range(T.ndim) if i not in ax]
    order[a:a] = sorted(ax)
    np.testing.assert_array_equal(
        np.asarray(got.todense()), np.transpose(np.asarray(T.todense()), order)
    )
    for x, i in zip(got.indices, order):
        assert np.array_equal(x.charges, T.indices[i].charges)
        assert x.flow == T.indices[i].flow


def test_partially_empty_fused_block_fills_zeros():
    """A sub-block absent from the input is zeros in the fused block."""
    T = _tensor("u1", True)
    dropped = next(iter(T.blocks))
    blocks = {k: v for k, v in T.blocks.items() if k != dropped}
    T2 = SymmetricTensor(blocks, T.indices)
    _assert_fuses_like_dense(T2, (0, 1), IN)


@pytest.mark.parametrize("grouped", [True, False])
def test_fuse_forward_mode_matches_the_dense_reference(grouped):
    """``jvp`` works through the permutation (a ``custom_vjp`` would refuse)."""
    T = _tensor("u1", grouped)
    dT = _tensor("u1", grouped, seed=3)

    def f(t):
        g = fuse_indices(t, 1, 3, "f", OUT)
        return sum(jnp.sum(jnp.sin(b)) for b in g.blocks.values())

    def f_dense(t):
        # sin(0) = 0, so the zeros outside the blocks add nothing.
        return jnp.sum(jnp.sin(_dense_fuse(t.todense(), (1, 3))))

    _, t_new = jax.jvp(f, (T,), (dT,))
    _, t_ref = jax.jvp(f_dense, (T,), (dT,))
    np.testing.assert_allclose(float(t_new), float(t_ref), rtol=1e-13)


@pytest.mark.parametrize("grouped", [True, False])
def test_fuse_gradient_matches_the_dense_reference(grouped):
    T = _tensor("u1", grouped)
    w = fuse_indices(_tensor("u1", grouped, seed=1), 0, 2, "f", IN)
    w_dense = w.todense()

    def loss(t):
        g = fuse_indices(t, 0, 2, "f", IN)
        return sum(jnp.vdot(w.blocks[k], g.blocks[k]) for k in g.blocks)

    def loss_dense(t):
        return jnp.vdot(w_dense, _dense_fuse(t.todense(), (0, 2)))

    g_new = jax.grad(loss)(T)
    g_ref = jax.grad(loss_dense)(T)
    leaves_new, leaves_ref = jax.tree.leaves(g_new), jax.tree.leaves(g_ref)
    assert len(leaves_new) == len(leaves_ref) and leaves_new
    for x, y in zip(leaves_new, leaves_ref):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-14)
    assert any(np.any(np.asarray(x) != 0) for x in leaves_new)  # not vacuous


def _same_blocks(x, y):
    assert x.indices == y.indices or all(
        np.array_equal(i.charges, j.charges) and i.flow == j.flow
        for i, j in zip(x.indices, y.indices)
    )
    assert set(x.blocks) == set(y.blocks)
    for k in x.blocks:
        np.testing.assert_array_equal(np.asarray(x.blocks[k]), np.asarray(y.blocks[k]))


def test_fuse_inside_jit_matches_eager():
    T = _tensor("z3", False)
    eager = fuse_indices(T, 0, 3, "f", IN)
    jitted = jax.jit(lambda t: fuse_indices(t, 0, 3, "f", IN))(T)
    _same_blocks(jitted, eager)
