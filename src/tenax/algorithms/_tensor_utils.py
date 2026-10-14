"""Shared tensor utilities for algorithms.

Functions here work polymorphically on both DenseTensor and SymmetricTensor
via the Tensor protocol.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core._tensor_utils import (  # noqa: F401 -- re-export for backward compat
    _scale_bond_axis_symmetric,
    scale_bond_axis,
)
from tenax.core.index import FlowDirection, FuseInfo, Label, TensorIndex
from tenax.core.symmetry import BaseSymmetry
from tenax.core.tensor import BlockKey, DenseTensor, SymmetricTensor, Tensor


def safe_inv_lambda(lam: jax.Array, rel_tol: float = 1e-12) -> jax.Array:
    """Pseudo-inverse of a simple-update bond-weight vector.

    Returns ``1/lam`` for entries non-negligible relative to the largest weight
    (``lam > rel_tol * max(lam)``) and 0 otherwise. The naive ``1/(lam + EPS)``
    used in simple-update canonical-form bookkeeping amplifies a vanishing
    Schmidt sector to ~``1/EPS``; under the absorb-then-remove round-trip plus a
    global re-normalization that flips sector dominance and degenerates the whole
    bond (all weights -> 0, or NaN). Dropping dead sectors keeps the update
    numerically stable. This is a regularized/thresholded inverse, NOT the
    inversion-free canonical update of Hastings (arXiv:0903.3253), which avoids
    forming ``lambda^-1`` altogether and uses no pseudo-inverse. Mirrors
    ``pess._safe_inv``.
    """
    lam = jnp.asarray(lam)
    cutoff = rel_tol * jnp.max(lam)
    keep = lam > cutoff
    return jnp.where(keep, 1.0 / jnp.where(keep, lam, 1.0), 0.0)


def max_abs_normalize(T: Tensor) -> tuple[Tensor, jax.Array]:
    """Normalize tensor by its max absolute value.

    Args:
        T: Input tensor.

    Returns:
        (T_normalized, log_norm) where T_normalized = T / max_abs(T)
        and log_norm = log(max_abs(T)).
    """
    from tenax.core import LOG_EPS

    norm = T.max_abs()
    log_norm = jnp.log(norm + LOG_EPS)
    T_norm = T * (1.0 / (norm + LOG_EPS))
    return T_norm, log_norm


def absorb_sqrt_singular_values(
    U: Tensor,
    s: jax.Array,
    Vh: Tensor,
    bond_label: Label,
) -> tuple[Tensor, Tensor]:
    """Absorb sqrt(s) into both U and Vh along their shared bond.

    Args:
        U:          Left factor from SVD with bond_label as a leg.
        s:          1D singular values.
        Vh:         Right factor from SVD with bond_label as a leg.
        bond_label: Label of the SVD bond on both U and Vh.

    Returns:
        (F_left, F_right) with sqrt(s) absorbed into each.
    """
    # Backward-safe sqrt.  The adjoint of ``jnp.sqrt`` is ``0.5 / sqrt(s)``,
    # which is ``+inf`` at ``s == 0``.  Rank-deficient SVD bonds carry exact
    # zero singular values — e.g. the lossless split-CTM edge SVD padded to
    # ``chi_I > rank`` (exposed once the env corner is the rank-1
    # variPEPS ``chi_init=1`` seed) — so the naive sqrt NaNs the gradient.
    # Mask sub-threshold entries with the standard double-``where`` guard: the
    # forward value ``sqrt(0) = 0`` is unchanged (those bond entries carry no
    # weight), but the masked adjoint is ``0`` instead of ``inf``.  Mirrors the
    # ``S^{-1/2}`` guard in ``_compute_projector_tensor`` (#463).
    if s.shape[0] == 0:
        sqrt_s = s
    else:
        cutoff = 1e-14 * (jnp.max(s) + 1e-30)
        mask = s > cutoff
        s_safe = jnp.where(mask, s, 1.0)
        sqrt_s = jnp.where(mask, jnp.sqrt(s_safe), 0.0)
    F_left = scale_bond_axis(U, bond_label, sqrt_s)
    F_right = scale_bond_axis(Vh, bond_label, sqrt_s)
    return F_left, F_right


def fuse_indices(
    tensor: Tensor,
    axis_a: int,
    axis_b: int,
    fused_label: Label,
    fused_flow: FlowDirection,
) -> Tensor:
    """Fuse two adjacent tensor legs into a single leg.

    For DenseTensor: transpose to bring axes adjacent and reshape.
    For SymmetricTensor: compute product charges and reassemble blocks.

    The two axes are merged into one at the position of ``axis_a``
    (or ``axis_b``, whichever comes first). The fused dimension equals
    ``dim_a * dim_b``.

    Args:
        tensor:      Input tensor.
        axis_a:      First axis to fuse.
        axis_b:      Second axis to fuse.
        fused_label: Label for the resulting fused leg.
        fused_flow:  Flow direction for the fused leg.

    Returns:
        Tensor with one fewer leg; the two fused legs replaced by one.
    """
    if isinstance(tensor, SymmetricTensor):
        return _fuse_indices_symmetric(tensor, axis_a, axis_b, fused_label, fused_flow)
    return _fuse_indices_dense(tensor, axis_a, axis_b, fused_label, fused_flow)


def _fuse_indices_dense(
    T: DenseTensor,
    axis_a: int,
    axis_b: int,
    fused_label: Label,
    fused_flow: FlowDirection,
) -> DenseTensor:
    """Fuse two axes of a DenseTensor via transpose + reshape."""
    ndim = T.ndim
    a, b = sorted([axis_a, axis_b])

    # Transpose to bring axes a and b adjacent (a, b contiguous)
    other_axes = [i for i in range(ndim) if i not in (a, b)]
    perm = other_axes[:a] + [a, b] + other_axes[a:]
    data = jnp.transpose(T.todense(), perm)
    indices_perm = [T.indices[i] for i in perm]

    # Reshape: merge axes at position a
    shape = list(data.shape)
    new_shape = shape[:a] + [shape[a] * shape[a + 1]] + shape[a + 2 :]
    data = data.reshape(new_shape)

    # Build fused index with FuseInfo
    idx_a, idx_b = indices_perm[a], indices_perm[a + 1]
    sym = idx_a.symmetry
    # ``_compute_fused_charges`` returns canonical representatives.  That matters
    # because TensorIndex.__post_init__ rewrites ``sectors`` to canonical form
    # (#734): uncanonicalised charges would leave ``charges`` naming sectors the
    # index does not have, and blocks on those charges are dropped silently.
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
    # Preserve original charges ordering for from_dense/todense compat
    object.__setattr__(fused_idx, "_charges_cache", fused_charges)
    new_indices = tuple(indices_perm[:a]) + (fused_idx,) + tuple(indices_perm[a + 2 :])
    return DenseTensor(data, new_indices)


def _fused_charge_array(
    charges_a: np.ndarray,
    charges_b: np.ndarray,
    flow_a: int,
    flow_b: int,
    fused_flow: int,
    sym: BaseSymmetry,
) -> np.ndarray:
    """Charge carried by the fused leg for element-wise paired parent charges.

    This is the single charge-arithmetic site of leg fusion, and the fusion
    counterpart of :func:`~tenax.core.index._net_charge`: every ``q_f`` in this
    module is derived here so the sector table, the cached dense charges, the
    ``(qa, qb) -> q_f`` grouping and the rewritten block keys cannot drift apart.

    The defining property is that fusing must not change a block's net charge —
    the fused leg has to contribute exactly what its two parents did::

        flow_charge(fused_flow, q_f) == fuse(flow_charge(flow_a, qa),
                                             flow_charge(flow_b, qb))

    which inverts to the expression below because ``flow_charge`` is an
    involution on canonical representatives.

    The former ``(flow_a * qa + flow_b * qb) * fused_sign``, reduced by a
    hand-rolled ``% n_values()``, was wrong for
    :class:`~tenax.core.symmetry.ProductSymmetry` twice over (#734): negating a
    bit-packed charge is not the component-wise group inverse, and
    ``n_values()`` is a group cardinality (``2 * 3 == 6``) rather than a modulus
    for the packed integer.  Measured on ``ProductSymmetry(Z2, Z3)``, fusing two
    IN legs produced block keys ``{0, 1, 3, 4, 5}`` against a fused sector table
    of ``{0, 1}`` — keys naming charges the leg does not carry, whose blocks are
    then silently dropped by every downstream contraction.

    Args:
        charges_a:  Parent-a charges, broadcastable against ``charges_b``.
        charges_b:  Parent-b charges, broadcastable against ``charges_a``.
        flow_a:     Flow of parent leg a (``+1`` IN / ``-1`` OUT).
        flow_b:     Flow of parent leg b.
        fused_flow: Flow the fused leg is being given.
        sym:        Symmetry governing both parents.

    Returns:
        Canonical int32 fused charges, one per input pair.
    """
    effective = sym.fuse(
        sym.flow_charge(flow_a, np.asarray(charges_a, dtype=np.int32)),
        sym.flow_charge(flow_b, np.asarray(charges_b, dtype=np.int32)),
    )
    return sym.canonicalize_charges(sym.flow_charge(fused_flow, effective)).astype(
        np.int32
    )


def _fused_charge_grid(
    sectors_a: np.ndarray,
    sectors_b: np.ndarray,
    flow_a: int,
    flow_b: int,
    fused_flow: int,
    sym: BaseSymmetry,
) -> np.ndarray:
    """Fused charge of every ``(qa, qb)`` pair, flattened in ``i * nb + j`` order."""
    qa = np.asarray(sectors_a, dtype=np.int32)
    qb = np.asarray(sectors_b, dtype=np.int32)
    return _fused_charge_array(
        np.repeat(qa, len(qb)),
        np.tile(qb, len(qa)),
        flow_a,
        flow_b,
        fused_flow,
        sym,
    )


def _fused_sector_map(
    idx_a: TensorIndex,
    idx_b: TensorIndex,
    fused_flow: FlowDirection,
    sym: BaseSymmetry,
) -> dict[tuple[int, int], int]:
    """Map each ``(qa, qb)`` sector pair to the sector it fuses into."""
    grid = _fused_charge_grid(
        idx_a.sectors,
        idx_b.sectors,
        int(idx_a.flow),
        int(idx_b.flow),
        int(fused_flow),
        sym,
    )
    nb = len(idx_b.sectors)
    return {
        (int(qa), int(qb)): int(grid[i * nb + j])
        for i, qa in enumerate(idx_a.sectors)
        for j, qb in enumerate(idx_b.sectors)
    }


def _compute_fused_charges(
    idx_a: TensorIndex,
    idx_b: TensorIndex,
    fused_flow: FlowDirection,
    sym: BaseSymmetry,
) -> np.ndarray:
    """Compute the charges array for a fused index.

    For each ``(i, j)`` pair of basis states from legs a and b, the fused charge
    is :func:`_fused_charge_array` of the two parent charges, stored at
    ``i * db + j``.  The result is canonical, so it can be handed straight to
    ``TensorIndex`` (which canonicalises ``sectors``) without the two halves
    naming different representatives of the same sector (#734).
    """
    return _fused_charge_grid(
        idx_a.charges,
        idx_b.charges,
        int(idx_a.flow),
        int(idx_b.flow),
        int(fused_flow),
        sym,
    )


def _compute_fused_sectors(
    idx_a: TensorIndex,
    idx_b: TensorIndex,
    fused_flow: FlowDirection,
    sym: BaseSymmetry,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sectors and multiplicities for a fused index at O(n_sectors^2).

    For each pair of sectors ``(qa, qb)`` the fused charge comes from
    :func:`_fused_charge_array`, and the fused multiplicity is ``m_a * m_b``
    (summed over pairs giving the same ``q_f``).

    Returns:
        (sectors, multiplicities) — sorted int32 arrays.
    """
    q_f = _fused_charge_grid(
        idx_a.sectors,
        idx_b.sectors,
        int(idx_a.flow),
        int(idx_b.flow),
        int(fused_flow),
        sym,
    )
    nb = len(idx_b.sectors)
    # Accumulate in int64: the pairwise product of two int32 multiplicities can
    # overflow int32 before the per-sector sum is taken.
    mults = np.repeat(idx_a.multiplicities.astype(np.int64), nb) * np.tile(
        idx_b.multiplicities.astype(np.int64), len(idx_a.sectors)
    )

    sectors, inverse = np.unique(q_f, return_inverse=True)
    multiplicities = np.zeros(len(sectors), dtype=np.int64)
    np.add.at(multiplicities, inverse.ravel(), mults)
    return sectors.astype(np.int32), multiplicities.astype(np.int32)


class _FuseLayout(NamedTuple):
    """Where each ``(qa, qb)`` sub-block lands inside its fused block.

    ``todense()`` puts the elements of fused sector ``q_f`` at the positions
    ``i * db + j`` with that charge, in ascending order.  The ``(qa, qb)``
    sub-blocks of one ``q_f`` own disjoint sets of those positions.

    Attributes:
        qf_of:   ``(qa, qb) -> q_f``, from :func:`_fused_sector_map`.
        pieces:  ``q_f -> ((qa, qb), ...)`` in concatenation order: by the
                 smallest fused position each sub-block owns.
        perm:    ``q_f -> perm`` with ``fused[k] = concat[perm[k]]``, or
                 ``None`` when the concatenation already is the fused order.
                 For charge-grouped legs it is ``None``: each row ``i`` of
                 ``qa`` meets exactly one ``qb`` of a given ``q_f``.
        offsets: ``(qa, qb) -> offsets``, the fused-block position of each
                 element of the row-major sub-block.
        fused_dim: ``q_f -> size`` of the fused block along the fused axis.
    """

    qf_of: dict[tuple[int, int], int]
    pieces: dict[int, tuple[tuple[int, int], ...]]
    perm: dict[int, np.ndarray | None]
    offsets: dict[tuple[int, int], np.ndarray]
    fused_dim: dict[int, int]


def _fuse_layout(
    idx_a: TensorIndex,
    idx_b: TensorIndex,
    fused_flow: FlowDirection,
    sym: BaseSymmetry,
) -> _FuseLayout:
    """Compute the :class:`_FuseLayout` of fusing ``idx_a`` with ``idx_b``."""
    db = len(idx_b.charges)
    positions_a = {int(q): np.flatnonzero(idx_a.charges == q) for q in idx_a.sectors}
    positions_b = {int(q): np.flatnonzero(idx_b.charges == q) for q in idx_b.sectors}

    # The same map names the fused blocks and groups the pairs, so the two
    # cannot disagree (#734).
    qf_of = _fused_sector_map(idx_a, idx_b, fused_flow, sym)
    flat_of: dict[int, list[tuple[tuple[int, int], np.ndarray]]] = {}
    for qa in idx_a.sectors:
        for qb in idx_b.sectors:
            pair = (int(qa), int(qb))
            flat = (positions_a[pair[0]][:, None] * db + positions_b[pair[1]]).ravel()
            if flat.size:
                flat_of.setdefault(qf_of[pair], []).append((pair, flat))

    pieces: dict[int, tuple[tuple[int, int], ...]] = {}
    perm: dict[int, np.ndarray | None] = {}
    offsets: dict[tuple[int, int], np.ndarray] = {}
    fused_dim: dict[int, int] = {}
    for q_f, entries in flat_of.items():
        entries.sort(key=lambda e: int(e[1][0]))
        cat = np.concatenate([flat for _, flat in entries])
        order = np.argsort(cat, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        start = 0
        for pair, flat in entries:
            offsets[pair] = rank[start : start + flat.size]
            start += flat.size
        pieces[q_f] = tuple(pair for pair, _ in entries)
        perm[q_f] = None if np.array_equal(order, np.arange(order.size)) else order
        fused_dim[q_f] = int(cat.size)
    return _FuseLayout(qf_of, pieces, perm, offsets, fused_dim)


def _permute_axis(x: jax.Array, perm: np.ndarray, axis: int) -> jax.Array:
    """``jnp.take(x, perm, axis)`` for a permutation ``perm``.

    The VJP takes the inverse permutation.  A plain ``take`` transposes to a
    scatter-add, which is one more scatter in every backward program.
    """
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    fwd_idx, bwd_idx = jnp.asarray(perm), jnp.asarray(inv)

    def take(y, idx):
        return jnp.take(y, idx, axis=axis, mode="clip", unique_indices=True)

    @jax.custom_vjp
    def permute(y):
        return take(y, fwd_idx)

    def permute_fwd(y):
        return take(y, fwd_idx), None

    def permute_bwd(_, g):
        return (take(g, bwd_idx),)

    permute.defvjp(permute_fwd, permute_bwd)
    return permute(x)


def _fuse_indices_symmetric(
    T: SymmetricTensor,
    axis_a: int,
    axis_b: int,
    fused_label: Label,
    fused_flow: FlowDirection,
) -> SymmetricTensor:
    """Fuse two axes of a SymmetricTensor.

    Each block is transposed so the two axes are adjacent and reshaped to
    merge them.  The sub-blocks of one fused sector are then joined with one
    ``concatenate`` (absent sub-blocks are zeros) and, only for legs whose
    charges are not grouped by sector, reordered with one permutation.  This
    emits no scatter: the old per-block ``.at[].set`` was one scatter per
    input block, ~63k of them in a D=3 U(1) backward (#566).
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

    layout = _fuse_layout(idx_a, idx_b, fused_flow, sym)

    other_axes = [i for i in range(ndim) if i not in (a, b)]
    new_indices = list(T.indices[i] for i in other_axes)
    new_indices.insert(a, fused_idx)
    new_indices = tuple(new_indices)
    perm_axes = other_axes[:a] + [a, b] + other_axes[a:]

    # Merge the two axes of every block, grouped by the fused block it joins.
    # Dict order follows the first input block of each fused block.
    parts: dict[BlockKey, dict[tuple[int, int], jax.Array]] = {}
    for key, block in T.blocks.items():
        pair = (int(key[a]), int(key[b]))
        q_f = layout.qf_of[pair]
        other_charges = [key[i] for i in other_axes]
        new_key = tuple(other_charges[:a]) + (q_f,) + tuple(other_charges[a:])
        block_t = jnp.transpose(block, perm_axes)
        shape = list(block_t.shape)
        merged = shape[:a] + [shape[a] * shape[a + 1]] + shape[a + 2 :]
        parts.setdefault(new_key, {})[pair] = block_t.reshape(merged)

    new_blocks: dict[BlockKey, jax.Array] = {}
    for new_key, have in parts.items():
        q_f = int(new_key[a])
        like = next(iter(have.values()))
        seq = []
        for pair in layout.pieces[q_f]:
            piece = have.get(pair)
            if piece is None:
                shape = list(like.shape)
                shape[a] = layout.offsets[pair].size
                piece = jnp.zeros(shape, dtype=like.dtype)
            seq.append(piece)
        fused = seq[0] if len(seq) == 1 else jnp.concatenate(seq, axis=a)
        if layout.perm[q_f] is not None:
            fused = _permute_axis(fused, layout.perm[q_f], a)
        new_blocks[new_key] = fused

    obj = object.__new__(SymmetricTensor)
    obj._indices = new_indices
    obj._init_flat_buffer(new_blocks)
    return obj


def split_index(tensor: Tensor, axis: int) -> Tensor:
    """Split a fused leg back into its parent legs.

    Inverse of ``fuse_indices``. The leg at *axis* must have ``fuse_info``
    (i.e. it must have been produced by ``fuse_indices``).

    Args:
        tensor: Input tensor with a fused leg.
        axis:   Position of the fused leg to split.

    Returns:
        Tensor with the fused leg replaced by its two parent legs.
    """
    fused_idx = tensor.indices[axis]
    if fused_idx.fuse_info is None:
        raise ValueError(
            f"Cannot split index at axis {axis}: fuse_info is None "
            f"(this leg was not created by fuse_indices)"
        )
    if isinstance(tensor, SymmetricTensor):
        return _split_index_symmetric(tensor, axis)
    return _split_index_dense(tensor, axis)


def _split_index_dense(T: DenseTensor, axis: int) -> DenseTensor:
    """Split a fused axis of a DenseTensor back into two axes."""
    fused_idx = T.indices[axis]
    parent_a, parent_b = fused_idx.fuse_info.parent_indices
    sym = parent_a.symmetry

    # The fuse step computed charges in (i,j) order then stored them as
    # _charges_cache. We need to invert: un-permute so positions map back
    # to the original (i*db + j) layout, then reshape.
    fused_charges = fused_idx.charges  # original (i,j) order from _charges_cache
    # Compute what fuse would have produced for the parent charges, using the
    # flow recorded at fusion time (robust to a later flow flip).
    _split_flow = (
        fused_idx.fuse_info.fused_flow
        if fused_idx.fuse_info.fused_flow is not None
        else fused_idx.flow
    )
    expected_charges = _compute_fused_charges(parent_a, parent_b, _split_flow, sym)

    if not np.array_equal(fused_charges, expected_charges):
        # If charges were reordered, compute and apply inverse permutation.
        # This shouldn't happen when fuse_indices produced the tensor, but
        # handle it for safety.
        sort_perm = np.argsort(expected_charges, kind="stable")
        inv_perm = np.argsort(sort_perm)
        data = jnp.take(T.todense(), inv_perm, axis=axis)
    else:
        data = T.todense()

    # Reshape: split fused axis into (dim_a, dim_b)
    shape = list(data.shape)
    new_shape = shape[:axis] + [parent_a.dim, parent_b.dim] + shape[axis + 1 :]
    data = data.reshape(new_shape)

    # Build new indices
    new_indices = (
        tuple(T.indices[:axis]) + (parent_a, parent_b) + tuple(T.indices[axis + 1 :])
    )
    return DenseTensor(data, new_indices)


def _split_index_symmetric(T: SymmetricTensor, axis: int) -> SymmetricTensor:
    """Split a fused axis of a SymmetricTensor back into two axes.

    Inverts :func:`_fuse_indices_symmetric` with the same
    :func:`_fuse_layout`: each ``(qa, qb)`` sub-block is a static slice of the
    fused block, or for legs whose charges are not grouped by sector, one
    gather of its offsets.
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
    layout = _fuse_layout(parent_a, parent_b, split_flow, sym)
    mult_a = {int(q): int(m) for q, m in zip(parent_a.sectors, parent_a.multiplicities)}
    mult_b = {int(q): int(m) for q, m in zip(parent_b.sectors, parent_b.multiplicities)}

    other_axes = [i for i in range(ndim) if i != axis]
    new_indices = (
        tuple(T.indices[:axis]) + (parent_a, parent_b) + tuple(T.indices[axis + 1 :])
    )

    new_blocks: dict[BlockKey, jax.Array] = {}
    for key, block in T.blocks.items():
        q_f = int(key[axis])
        other_charges = tuple(key[i] for i in other_axes)
        for qa, qb in layout.pieces.get(q_f, ()):
            offsets = layout.offsets[(qa, qb)]
            start = int(offsets[0])
            if np.array_equal(offsets, np.arange(start, start + offsets.size)):
                sub_block = jax.lax.slice_in_dim(
                    block, start, start + offsets.size, axis=axis
                )
            else:
                # Gather VJP = scatter-add; only legs whose charges are not
                # grouped by sector get here.
                src = [slice(None)] * ndim
                src[axis] = jnp.asarray(offsets)
                sub_block = block[tuple(src)]
            shape = list(sub_block.shape)
            new_shape = shape[:axis] + [mult_a[qa], mult_b[qb]] + shape[axis + 1 :]
            new_key = other_charges[:axis] + (qa, qb) + other_charges[axis:]
            new_blocks[new_key] = sub_block.reshape(new_shape)

    obj = object.__new__(SymmetricTensor)
    obj._indices = new_indices
    obj._init_flat_buffer(new_blocks)
    return obj


def double_layer_tensor(A: Tensor) -> Tensor:
    """Build the double-layer tensor a = A * conj(A) with physical index traced.

    Contracts tensor A with its conjugate over the physical leg, then fuses
    ket/bra pairs into single legs for each spatial direction.

    Input A has 5 legs: (up, down, left, right, phys).
    Output has 4 legs: (up, down, left, right) with dimensions D².

    Args:
        A: Site tensor with labels ("up", "down", "left", "right", "phys").

    Returns:
        Double-layer tensor with fused legs.
    """
    if isinstance(A, DenseTensor):
        return _double_layer_dense(A)

    return _double_layer_symmetric(A)


def _double_layer_dense(A: DenseTensor) -> DenseTensor:
    """Double-layer tensor for DenseTensor via einsum + reshape."""
    data = A.todense()
    # a[u,d,l,r,U,D,L,R] = sum_s A[u,d,l,r,s] * conj(A[U,D,L,R,s])
    dl = jnp.einsum("udlrs,UDLRs->uUdDlLrR", data, jnp.conj(data))
    D = data.shape[0]
    dl_fused = dl.reshape(D * D, D * D, D * D, D * D)

    sym = A.indices[0].symmetry
    fused_charges = np.zeros(D * D, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, fused_charges, FlowDirection.IN, label="up"),
        TensorIndex.from_charges(sym, fused_charges, FlowDirection.OUT, label="down"),
        TensorIndex.from_charges(sym, fused_charges, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(sym, fused_charges, FlowDirection.OUT, label="right"),
    )
    return DenseTensor(dl_fused, indices)


def _double_layer_symmetric(A: SymmetricTensor) -> DenseTensor:
    """Double-layer tensor for SymmetricTensor via dense computation.

    The physical trace sum_s A[...,s]*conj(A[...,s]) pairs positions by index,
    not by charge value.  This is incompatible with the block-sparse contraction
    framework (which pairs by charge).  Since the double-layer tensor is only
    used by CTM (currently dense-only), we compute via the dense path.
    """
    return _double_layer_dense(DenseTensor(A.todense(), A.indices))
