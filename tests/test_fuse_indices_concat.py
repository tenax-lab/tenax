"""Block-sparse ``fuse_indices``/``split_index`` emit no scatter (#566).

``_fuse_indices_symmetric`` used to build each fused block with one
``.at[].set`` scatter per input block.  It now joins the sub-blocks with one
``concatenate`` and, only for legs whose charges are not grouped by sector, one
permutation.  The previous implementation is kept below, verbatim, as the
reference: the new one must produce the same blocks bit for bit.
"""

from __future__ import annotations

import itertools

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._tensor_utils import (
    _compute_fused_charges,
    _fused_sector_map,
    fuse_indices,
    split_index,
)
from tenax.core.index import FlowDirection, FuseInfo, Label, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry, ZnSymmetry
from tenax.core.tensor import BlockKey, SymmetricTensor

jax.config.update("jax_enable_x64", True)

IN, OUT = FlowDirection.IN, FlowDirection.OUT


# --- reference: the implementation before #566 op-count reduction ----------


def _reference_fuse(
    T: SymmetricTensor,
    axis_a: int,
    axis_b: int,
    fused_label: Label,
    fused_flow: FlowDirection,
) -> SymmetricTensor:
    """Fuse two axes of a SymmetricTensor.

    Computes product charges, then for each block, reshapes the two
    fused axes into one and places the data at the correct position
    in the fused block.
    """
    a, b = sorted([axis_a, axis_b])
    ndim = T.ndim
    idx_a = T.indices[a]
    idx_b = T.indices[b]
    sym = idx_a.symmetry

    # Build fused TensorIndex with FuseInfo.  ``_compute_fused_charges`` returns
    # canonical representatives so the cached dense charges and the sector table
    # agree -- __post_init__ canonicalises ``sectors`` (#734), and a charge with
    # no matching sector loses its blocks silently.
    fused_charges = _compute_fused_charges(idx_a, idx_b, fused_flow, sym)
    sectors, mults = np.unique(fused_charges, return_counts=True)
    fuse_info = FuseInfo(parent_indices=(idx_a, idx_b), fused_flow=fused_flow)
    fused_idx = TensorIndex(
        sym,
        sectors.astype(np.int32),
        mults.astype(np.int32),
        fused_flow,
        label=fused_label,
        fuse_info=fuse_info,
    )
    object.__setattr__(fused_idx, "_charges_cache", fused_charges)

    # Compute where each (i, j) element lands in the fused block.
    #
    # todense() scatters block data to positions np.where(charges == q_f)
    # in ascending order, so block[k] must hold the data for the k-th
    # smallest position with charge q_f.  The fused charges array has
    # charge q_f at positions {i*db + j : charges_a[i]=qa, charges_b[j]=qb}
    # for each (qa, qb) pair mapping to q_f.  We must interleave the
    # (qa, qb) sub-blocks by ascending i*db+j order, NOT group them
    # contiguously.

    db = len(idx_b.charges)
    unique_qa = idx_a.sectors
    unique_qb = idx_b.sectors

    # For each (qa, qb) pair, find which positions in idx_a/idx_b have those charges
    positions_a: dict[int, np.ndarray] = {}
    for q in unique_qa:
        positions_a[int(q)] = np.where(idx_a.charges == q)[0]
    positions_b: dict[int, np.ndarray] = {}
    for q in unique_qb:
        positions_b[int(q)] = np.where(idx_b.charges == q)[0]

    # Group (qa, qb) pairs by fused charge q_f.  The map is built once and then
    # reused when rewriting the block keys below, so the grouping (which sizes
    # and lays out the fused blocks) and the keys (which name them) cannot
    # disagree -- they used to be two independent copies of the same wrong
    # ``flow * q`` arithmetic (#734).
    qf_of = _fused_sector_map(idx_a, idx_b, fused_flow, sym)
    fused_groups: dict[int, list[tuple[int, int]]] = {}
    for qa in unique_qa:
        for qb in unique_qb:
            fused_groups.setdefault(qf_of[(int(qa), int(qb))], []).append(
                (int(qa), int(qb))
            )

    # For each fused charge q_f, compute:
    #   fused_dim[q_f]: total number of elements in this block
    #   scatter_map[(qa, qb)]: array of target offsets within the fused block
    #     for each element of the (qa, qb) sub-block (in row-major order
    #     over positions_a[qa] x positions_b[qb]).
    fused_dim: dict[int, int] = {}
    scatter_map: dict[tuple[int, int], np.ndarray] = {}

    for q_f, pairs in fused_groups.items():
        # Collect all (i*db + j) positions for this q_f, along with which
        # (qa, qb) pair and local index each belongs to.
        all_positions: list[
            tuple[int, int, int, int]
        ] = []  # (flat_pos, qa, qb, local_idx)
        for qa, qb in pairs:
            local_idx = 0
            for i in positions_a[qa]:
                for j in positions_b[qb]:
                    all_positions.append((int(i) * db + int(j), qa, qb, local_idx))
                    local_idx += 1

        # Sort by flat position (ascending) — this is the order todense() expects
        all_positions.sort(key=lambda x: x[0])
        fused_dim[q_f] = len(all_positions)

        # Build scatter arrays: for each (qa, qb) pair, scatter_map[(qa, qb)][local]
        # = target offset in the fused block
        tmp: dict[
            tuple[int, int], list[tuple[int, int]]
        ] = {}  # (qa,qb) -> [(local, target)]
        for target_offset, (_, qa, qb, local_idx) in enumerate(all_positions):
            tmp.setdefault((qa, qb), []).append((local_idx, target_offset))

        for qa, qb in pairs:
            if (qa, qb) in tmp:
                entries = tmp[(qa, qb)]
                entries.sort(key=lambda x: x[0])  # sort by local_idx
                scatter_map[(qa, qb)] = np.array(
                    [t for _, t in entries], dtype=np.int64
                )
            else:
                scatter_map[(qa, qb)] = np.array([], dtype=np.int64)

    # Build new indices list (axes a and b replaced by fused_idx at position a)
    other_axes = [i for i in range(ndim) if i not in (a, b)]
    new_indices = list(T.indices[i] for i in other_axes)
    new_indices.insert(a, fused_idx)
    new_indices = tuple(new_indices)

    # Reassemble blocks
    new_blocks: dict[BlockKey, jax.Array] = {}

    for key, block in T.blocks.items():
        qa = int(key[a])
        qb = int(key[b])

        # Same map that built ``fused_groups``/``scatter_map`` above.
        q_f = qf_of[(qa, qb)]

        # Transpose block to bring axes a and b adjacent
        other_block_axes = [i for i in range(ndim) if i not in (a, b)]
        perm = other_block_axes[:a] + [a, b] + other_block_axes[a:]
        block_t = jnp.transpose(block, perm)

        # Reshape: merge the two axes into one (at position a)
        shape = list(block_t.shape)
        new_shape = shape[:a] + [shape[a] * shape[a + 1]] + shape[a + 2 :]
        block_flat = block_t.reshape(new_shape)

        # Build new key
        other_charges = [key[i] for i in other_axes]
        new_key = tuple(other_charges[:a]) + (q_f,) + tuple(other_charges[a:])

        # Get the scatter offsets for this (qa, qb) pair
        offsets = scatter_map[(qa, qb)]

        if new_key not in new_blocks:
            full_shape = list(new_shape)
            full_shape[a] = fused_dim[q_f]
            new_blocks[new_key] = jnp.zeros(full_shape, dtype=block.dtype)

        # Scatter every sub-block slice along axis a in ONE vectorized op:
        #   existing[..., offsets[k], ...] = block_flat[..., k, ...]  for all k.
        # The offsets are distinct (each local index maps to a unique target),
        # so there are no scatter collisions; and different (qa, qb) pairs own
        # disjoint offset ranges within the fused block, so successive `.set()`
        # calls accumulate correctly without clobbering. This replaces a
        # per-element Python loop (one `.at[].set()` primitive per element —
        # hundreds at large bond dim, which bloats the jaxpr/XLA compile) with
        # a single scatter primitive (#569, Milestone A). Numerically identical
        # to the loop (same data movement) and differentiable (scatter VJP).
        existing = new_blocks[new_key]
        dst = [slice(None)] * len(new_shape)
        dst[a] = jnp.asarray(offsets)
        new_blocks[new_key] = existing.at[tuple(dst)].set(block_flat)

    obj = object.__new__(SymmetricTensor)
    obj._indices = new_indices
    obj._init_flat_buffer(new_blocks)
    return obj


def _reference_split(T: SymmetricTensor, axis: int) -> SymmetricTensor:
    """Split a fused axis of a SymmetricTensor back into two axes.

    Inverts the scatter logic from ``_fuse_indices_symmetric``: for each
    fused block, extracts sub-blocks for each (qa, qb) pair.
    """
    fused_idx = T.indices[axis]
    parent_a, parent_b = fused_idx.fuse_info.parent_indices
    sym = parent_a.symmetry
    ndim = T.ndim

    # Use the flow recorded at fusion time so the inverse mapping is correct
    # even if the fused leg's flow was later flipped (e.g. by bar/flip_flow,
    # which keep the data layout but change the leg's flow label).
    split_flow = (
        fused_idx.fuse_info.fused_flow
        if fused_idx.fuse_info.fused_flow is not None
        else fused_idx.flow
    )

    # Reconstruct the scatter map from fuse (same logic as _fuse_indices_symmetric)
    db = parent_b.dim
    positions_a: dict[int, np.ndarray] = {}
    for q in parent_a.sectors:
        positions_a[int(q)] = np.where(parent_a.charges == q)[0]
    positions_b: dict[int, np.ndarray] = {}
    for q in parent_b.sectors:
        positions_b[int(q)] = np.where(parent_b.charges == q)[0]

    # Group (qa, qb) pairs by fused charge q_f -- through the same helper the
    # forward direction uses, so the split inverts the fuse exactly (#734).
    qf_of = _fused_sector_map(parent_a, parent_b, split_flow, sym)
    fused_groups: dict[int, list[tuple[int, int]]] = {}
    for qa in parent_a.sectors:
        for qb in parent_b.sectors:
            fused_groups.setdefault(qf_of[(int(qa), int(qb))], []).append(
                (int(qa), int(qb))
            )

    # Rebuild scatter_map (same as fuse)
    scatter_map: dict[tuple[int, int], np.ndarray] = {}
    for q_f, pairs in fused_groups.items():
        all_positions: list[tuple[int, int, int, int]] = []
        for qa, qb in pairs:
            local_idx = 0
            for i in positions_a[qa]:
                for j in positions_b[qb]:
                    all_positions.append((int(i) * db + int(j), qa, qb, local_idx))
                    local_idx += 1
        all_positions.sort(key=lambda x: x[0])
        tmp: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for target_offset, (_, qa, qb, local_idx) in enumerate(all_positions):
            tmp.setdefault((qa, qb), []).append((local_idx, target_offset))
        for qa, qb in pairs:
            if (qa, qb) in tmp:
                entries = tmp[(qa, qb)]
                entries.sort(key=lambda x: x[0])
                scatter_map[(qa, qb)] = np.array(
                    [t for _, t in entries], dtype=np.int64
                )
            else:
                scatter_map[(qa, qb)] = np.array([], dtype=np.int64)

    # Build new indices: replace fused axis with (parent_a, parent_b)
    other_axes = [i for i in range(ndim) if i != axis]
    new_indices = (
        tuple(T.indices[:axis]) + (parent_a, parent_b) + tuple(T.indices[axis + 1 :])
    )

    # Gather blocks: for each fused block, extract sub-blocks for each (qa, qb)
    new_blocks: dict[BlockKey, jax.Array] = {}

    for key, block in T.blocks.items():
        q_f = int(key[axis])
        other_charges = tuple(key[i] for i in other_axes)

        # Which (qa, qb) pairs produce this q_f?
        if q_f not in fused_groups:
            continue
        for qa, qb in fused_groups[q_f]:
            offsets = scatter_map[(qa, qb)]
            if len(offsets) == 0:
                continue

            ma = len(positions_a[qa])
            mb = len(positions_b[qb])

            # Gather every sub-block slice along the fused axis in ONE
            # vectorized op: sub[..., k, ...] = block[..., offsets[k], ...].
            # A single advanced-index gather (the gathered dim stays at `axis`
            # since it is the only advanced index) replaces a per-element
            # Python gather + jnp.stack (#569). Numerically identical and
            # differentiable (gather VJP = scatter-add).
            src = [slice(None)] * ndim
            src[axis] = jnp.asarray(offsets)
            sub_block = block[tuple(src)]
            shape = list(sub_block.shape)
            new_shape = shape[:axis] + [ma, mb] + shape[axis + 1 :]
            sub_block = sub_block.reshape(new_shape)

            # Build new key
            new_key = other_charges[:axis] + (qa, qb) + other_charges[axis:]
            new_blocks[new_key] = sub_block

    obj = object.__new__(SymmetricTensor)
    obj._indices = new_indices
    obj._init_flat_buffer(new_blocks)
    return obj


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
    for ax in [(0, 1), (1, 0), (0, 2), (1, 3), (2, 3), (3, 0)]
]


def _tensor(name, grouped, seed=0):
    return SymmetricTensor.random_normal(
        _legs(SYMS[name], grouped), jax.random.PRNGKey(seed)
    )


def _same_blocks(x, y):
    assert x.indices == y.indices or all(
        np.array_equal(i.charges, j.charges) and i.flow == j.flow
        for i, j in zip(x.indices, y.indices)
    )
    assert set(x.blocks) == set(y.blocks)
    for k in x.blocks:
        np.testing.assert_array_equal(np.asarray(x.blocks[k]), np.asarray(y.blocks[k]))


# --- tests ------------------------------------------------------------------


@pytest.mark.parametrize("name,grouped,ax", CASES)
@pytest.mark.parametrize("flow", [IN, OUT])
def test_fuse_matches_reference_bit_for_bit(name, grouped, ax, flow):
    T = _tensor(name, grouped)
    got = fuse_indices(T, ax[0], ax[1], "f", flow)
    want = _reference_fuse(T, ax[0], ax[1], "f", flow)
    _same_blocks(got, want)


@pytest.mark.parametrize("name,grouped,ax", CASES)
def test_split_matches_reference_and_inverts_fuse(name, grouped, ax):
    T = _tensor(name, grouped)
    fused = fuse_indices(T, ax[0], ax[1], "f", OUT)
    a = min(ax)
    got = split_index(fused, a)
    _same_blocks(got, _reference_split(fused, a))
    # split(fuse(T)) is T with the two legs adjacent at ``a``, in sorted order.
    order = [i for i in range(T.ndim) if i not in ax]
    order[a:a] = sorted(ax)
    np.testing.assert_array_equal(
        np.asarray(got.todense()), np.transpose(np.asarray(T.todense()), order)
    )


def test_partially_empty_fused_block_fills_zeros():
    """A sub-block absent from the input is zeros in the fused block."""
    T = _tensor("u1", True)
    dropped = next(iter(T.blocks))
    blocks = {k: v for k, v in T.blocks.items() if k != dropped}
    T2 = SymmetricTensor(blocks, T.indices)
    _same_blocks(fuse_indices(T2, 0, 1, "f", IN), _reference_fuse(T2, 0, 1, "f", IN))


def _primitives(fn, *args):
    jaxpr = jax.make_jaxpr(fn)(*args)
    names = []

    def walk(jx):
        for eqn in jx.eqns:
            names.append(eqn.primitive.name)
            for sub in jax.extend.core.jaxprs_in_params(eqn.params):
                walk(sub)

    walk(jaxpr.jaxpr)
    return names


@pytest.mark.parametrize("grouped", [True, False])
def test_fuse_and_its_gradient_emit_no_scatter(grouped):
    T = _tensor("u1", grouped)

    def loss(t):
        f = fuse_indices(t, 1, 3, "f", OUT)
        return sum(jnp.sum(jnp.sin(b)) for b in f.blocks.values())

    prims = _primitives(jax.grad(loss), T)
    assert not [p for p in prims if p.startswith("scatter")], prims
    # Charge-grouped legs need no permutation; interleaved ones need one per
    # reordered block, which this fixture exercises.
    assert ("gather" in prims) is (not grouped)


@pytest.mark.parametrize("grouped", [True, False])
def test_fuse_gradient_matches_reference(grouped):
    T = _tensor("u1", grouped)
    w = fuse_indices(_tensor("u1", grouped, seed=1), 0, 2, "f", IN)

    def loss(fuse):
        def f(t):
            g = fuse(t, 0, 2, "f", IN)
            return sum(jnp.vdot(w.blocks[k], g.blocks[k]) for k in g.blocks)

        return f

    g_new = jax.grad(loss(fuse_indices))(T)
    g_ref = jax.grad(loss(_reference_fuse))(T)
    for x, y in zip(jax.tree.leaves(g_new), jax.tree.leaves(g_ref)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-14)


def test_fuse_inside_jit_matches_eager():
    T = _tensor("z3", False)
    eager = fuse_indices(T, 0, 3, "f", IN)
    jitted = jax.jit(lambda t: fuse_indices(t, 0, 3, "f", IN))(T)
    _same_blocks(jitted, eager)
