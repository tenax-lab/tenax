"""Shared tensor utilities for algorithms.

Functions here work polymorphically on both DenseTensor and SymmetricTensor
via the Tensor protocol.
"""

from __future__ import annotations

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
    sqrt_s = jnp.sqrt(s)
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
    fused_charges = _compute_fused_charges(idx_a, idx_b, fused_flow, sym)
    sectors, mults = np.unique(fused_charges, return_counts=True)
    fuse_info = FuseInfo(parent_indices=(idx_a, idx_b))
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


def _compute_fused_charges(
    idx_a: TensorIndex,
    idx_b: TensorIndex,
    fused_flow: FlowDirection,
    sym: object,
) -> np.ndarray:
    """Compute the charges array for a fused index.

    For each (i, j) pair of basis states from legs a and b, the fused
    charge is: q_f = (flow_a * q_a[i] + flow_b * q_b[j]) * fused_flow_sign.

    The ordering is lexicographic over unique charge pairs (q_a, q_b),
    with states within each charge sector ordered contiguously.
    """
    da = len(idx_a.charges)
    db = len(idx_b.charges)
    fused = np.empty(da * db, dtype=np.int32)

    flow_a_sign = int(idx_a.flow)
    flow_b_sign = int(idx_b.flow)
    fused_sign = int(fused_flow)

    for i in range(da):
        for j in range(db):
            # Raw charge contribution: flow_a * q_a + flow_b * q_b
            raw = flow_a_sign * int(idx_a.charges[i]) + flow_b_sign * int(
                idx_b.charges[j]
            )
            # Map to fused charge: q_f such that fused_flow * q_f = raw
            q_f = raw * fused_sign  # since fused_sign^2 = 1
            # For Zn: reduce mod n
            n = sym.n_values() if hasattr(sym, "n_values") else None
            if n is not None:
                q_f = q_f % n
            fused[i * db + j] = q_f

    return fused


def _compute_fused_sectors(
    idx_a: TensorIndex,
    idx_b: TensorIndex,
    fused_flow: FlowDirection,
    sym: BaseSymmetry,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sectors and multiplicities for a fused index at O(n_sectors^2).

    For each pair of sectors (qa, qb), the fused charge is
    ``(flow_a * qa + flow_b * qb) * fused_flow_sign``, and the fused
    multiplicity is ``m_a * m_b`` (summed over pairs giving the same q_f).

    Returns:
        (sectors, multiplicities) — sorted int32 arrays.
    """
    flow_a = int(idx_a.flow)
    flow_b = int(idx_b.flow)
    fused_sign = int(fused_flow)
    n_vals = sym.n_values()

    fused_mults: dict[int, int] = {}
    for i, qa in enumerate(idx_a.sectors):
        ma = int(idx_a.multiplicities[i])
        for j, qb in enumerate(idx_b.sectors):
            mb = int(idx_b.multiplicities[j])
            raw = flow_a * int(qa) + flow_b * int(qb)
            q_f = raw * fused_sign
            if n_vals is not None:
                q_f = q_f % n_vals
            fused_mults[q_f] = fused_mults.get(q_f, 0) + ma * mb

    sectors = np.array(sorted(fused_mults.keys()), dtype=np.int32)
    multiplicities = np.array([fused_mults[int(q)] for q in sectors], dtype=np.int32)
    return sectors, multiplicities


def _fuse_indices_symmetric(
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

    # Build fused TensorIndex with FuseInfo
    fused_charges = _compute_fused_charges(idx_a, idx_b, fused_flow, sym)
    sectors, mults = np.unique(fused_charges, return_counts=True)
    fuse_info = FuseInfo(parent_indices=(idx_a, idx_b))
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

    flow_a_sign = int(idx_a.flow)
    flow_b_sign = int(idx_b.flow)
    fused_sign = int(fused_flow)
    n_vals = sym.n_values()

    # For each (qa, qb) pair, find which positions in idx_a/idx_b have those charges
    positions_a: dict[int, np.ndarray] = {}
    for q in unique_qa:
        positions_a[int(q)] = np.where(idx_a.charges == q)[0]
    positions_b: dict[int, np.ndarray] = {}
    for q in unique_qb:
        positions_b[int(q)] = np.where(idx_b.charges == q)[0]

    # Group (qa, qb) pairs by fused charge q_f
    fused_groups: dict[int, list[tuple[int, int]]] = {}
    for qa in unique_qa:
        for qb in unique_qb:
            raw = flow_a_sign * int(qa) + flow_b_sign * int(qb)
            q_f = raw * fused_sign
            if n_vals is not None:
                q_f = q_f % n_vals
            fused_groups.setdefault(q_f, []).append((int(qa), int(qb)))

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

        # Compute q_f for this (qa, qb) pair
        raw = flow_a_sign * qa + flow_b_sign * qb
        q_f = raw * fused_sign
        if n_vals is not None:
            q_f = q_f % n_vals

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

        # Scatter sub-block elements to correct positions in fused block
        existing = new_blocks[new_key]
        sub_size = block_flat.shape[a]
        for local_idx in range(sub_size):
            target = int(offsets[local_idx])
            # Extract slice along axis a at local_idx
            slc_src = [slice(None)] * len(new_shape)
            slc_src[a] = local_idx
            slc_dst = [slice(None)] * len(new_shape)
            slc_dst[a] = target
            existing = existing.at[tuple(slc_dst)].set(block_flat[tuple(slc_src)])
        new_blocks[new_key] = existing

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
    # Compute what fuse would have produced for the parent charges
    expected_charges = _compute_fused_charges(parent_a, parent_b, fused_idx.flow, sym)

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

    Inverts the scatter logic from ``_fuse_indices_symmetric``: for each
    fused block, extracts sub-blocks for each (qa, qb) pair.
    """
    fused_idx = T.indices[axis]
    parent_a, parent_b = fused_idx.fuse_info.parent_indices
    sym = parent_a.symmetry
    ndim = T.ndim

    flow_a_sign = int(parent_a.flow)
    flow_b_sign = int(parent_b.flow)
    fused_sign = int(fused_idx.flow)
    n_vals = sym.n_values()

    # Reconstruct the scatter map from fuse (same logic as _fuse_indices_symmetric)
    db = parent_b.dim
    positions_a: dict[int, np.ndarray] = {}
    for q in parent_a.sectors:
        positions_a[int(q)] = np.where(parent_a.charges == q)[0]
    positions_b: dict[int, np.ndarray] = {}
    for q in parent_b.sectors:
        positions_b[int(q)] = np.where(parent_b.charges == q)[0]

    # Group (qa, qb) pairs by fused charge q_f
    fused_groups: dict[int, list[tuple[int, int]]] = {}
    for qa in parent_a.sectors:
        for qb in parent_b.sectors:
            raw = flow_a_sign * int(qa) + flow_b_sign * int(qb)
            q_f = raw * fused_sign
            if n_vals is not None:
                q_f = q_f % n_vals
            fused_groups.setdefault(q_f, []).append((int(qa), int(qb)))

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

            # Gather elements from fused block at scatter offsets
            sub_size = ma * mb
            # Build new block by gathering from fused axis
            gathered_slices = []
            for local_idx in range(sub_size):
                target = int(offsets[local_idx])
                slc = [slice(None)] * ndim
                slc[axis] = target
                gathered_slices.append(block[tuple(slc)])

            if not gathered_slices:
                continue

            # Stack along fused axis, then reshape to (ma, mb)
            sub_block = jnp.stack(gathered_slices, axis=axis)
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
