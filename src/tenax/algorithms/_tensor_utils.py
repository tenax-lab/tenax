"""Shared tensor utilities for algorithms.

Functions here work polymorphically on both DenseTensor and SymmetricTensor
via the Tensor protocol.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.index import FlowDirection, Label, TensorIndex
from tenax.core.tensor import BlockKey, DenseTensor, SymmetricTensor, Tensor


def scale_bond_axis(T: Tensor, label: Label, scale: jax.Array) -> Tensor:
    """Scale a tensor along a labeled axis by a diagonal vector.

    For DenseTensor: broadcasts scale along the named axis.
    For SymmetricTensor: delegates to block-wise scaling.

    Args:
        T:     Input tensor.
        label: Label of the axis to scale along.
        scale: 1D JAX array of length matching the axis dimension.

    Returns:
        New tensor with the specified axis scaled.
    """
    labels = T.labels()
    axis = labels.index(label)

    if isinstance(T, SymmetricTensor):
        return _scale_bond_axis_symmetric(T, axis, scale)

    # DenseTensor path
    data = T.todense()
    shape = [1] * T.ndim
    shape[axis] = data.shape[axis]
    return DenseTensor(data * scale.reshape(shape), T.indices)


def _scale_bond_axis_symmetric(
    T: SymmetricTensor, axis: int, scale: jax.Array
) -> SymmetricTensor:
    """Block-wise scaling for SymmetricTensor (same logic as fermionic_ipeps)."""
    new_blocks = {}
    idx = T.indices[axis]
    for key, block in T._blocks.items():
        charge_val = key[axis]
        positions = np.where(idx.charges == charge_val)[0]
        block_size = block.shape[axis]
        scale_slice = scale[positions[:block_size]]
        shape = [1] * T.ndim
        shape[axis] = block_size
        new_blocks[key] = block * scale_slice.reshape(shape)

    obj = object.__new__(SymmetricTensor)
    obj._indices = T._indices
    obj._blocks = new_blocks
    return obj


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

    # Build fused index
    idx_a, idx_b = indices_perm[a], indices_perm[a + 1]
    fused_charges = _compute_fused_charges(idx_a, idx_b, fused_flow, idx_a.symmetry)
    fused_idx = TensorIndex(
        idx_a.symmetry, fused_charges, fused_flow, label=fused_label
    )
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

    # Build fused TensorIndex
    fused_charges = _compute_fused_charges(idx_a, idx_b, fused_flow, sym)
    fused_idx = TensorIndex(sym, fused_charges, fused_flow, label=fused_label)

    # Compute offsets: for each (q_a, q_b) pair, where does it sit in the
    # fused dimension?
    # unique_qa and unique_qb are the unique charges on each leg
    unique_qa = np.unique(idx_a.charges)
    unique_qb = np.unique(idx_b.charges)

    # For each fused charge q_f, track the offset for each contributing (q_a, q_b) pair
    # offset_map[(q_a, q_b)] = (q_f, start_position_in_fused_dim)
    offset_map: dict[tuple[int, int], tuple[int, int]] = {}
    # Compute number of states per charge
    dim_a: dict[int, int] = {}
    for q in unique_qa:
        dim_a[int(q)] = int(np.sum(idx_a.charges == q))
    dim_b: dict[int, int] = {}
    for q in unique_qb:
        dim_b[int(q)] = int(np.sum(idx_b.charges == q))

    # Group (q_a, q_b) pairs by fused charge q_f, track offsets
    fused_groups: dict[int, list[tuple[int, int]]] = {}
    flow_a_sign = int(idx_a.flow)
    flow_b_sign = int(idx_b.flow)
    fused_sign = int(fused_flow)
    n_vals = sym.n_values()

    for qa in unique_qa:
        for qb in unique_qb:
            raw = flow_a_sign * int(qa) + flow_b_sign * int(qb)
            q_f = raw * fused_sign
            if n_vals is not None:
                q_f = q_f % n_vals
            fused_groups.setdefault(q_f, []).append((int(qa), int(qb)))

    # Compute fused dimension per fused charge and offsets
    fused_dim: dict[int, int] = {}
    for q_f, pairs in fused_groups.items():
        offset = 0
        for qa, qb in pairs:
            offset_map[(qa, qb)] = (q_f, offset)
            offset += dim_a[qa] * dim_b[qb]
        fused_dim[q_f] = offset

    # Build new indices list (axes a and b replaced by fused_idx at position a)
    other_axes = [i for i in range(ndim) if i not in (a, b)]
    new_indices = list(T.indices[i] for i in other_axes)
    new_indices.insert(a, fused_idx)
    new_indices = tuple(new_indices)

    # Reassemble blocks
    # If b is not adjacent to a, we need to transpose the block first
    new_blocks: dict[BlockKey, jax.Array] = {}

    for key, block in T._blocks.items():
        qa = key[a]
        qb = key[b]
        q_f, offset = offset_map[(int(qa), int(qb))]

        # Transpose block to bring axes a and b adjacent
        other_block_axes = [i for i in range(ndim) if i not in (a, b)]
        perm = other_block_axes[:a] + [a, b] + other_block_axes[a:]
        block_t = jnp.transpose(block, perm)

        # Reshape: merge the two axes
        shape = list(block_t.shape)
        new_shape = shape[:a] + [shape[a] * shape[a + 1]] + shape[a + 2 :]
        block_flat = block_t.reshape(new_shape)

        # Build new key
        other_charges = [key[i] for i in other_axes]
        new_key = tuple(other_charges[:a]) + (q_f,) + tuple(other_charges[a:])

        if new_key in new_blocks:
            # Multiple (q_a, q_b) pairs contribute to the same fused block.
            # Place this sub-block at the correct offset within the fused dim.
            existing = new_blocks[new_key]
            # Use dynamic_update_slice to place block_flat at the offset
            start_indices = [0] * len(new_shape)
            start_indices[a] = offset
            new_blocks[new_key] = jax.lax.dynamic_update_slice(
                existing, block_flat, tuple(start_indices)
            )
        else:
            # Create the full fused block (zeros) and place this sub-block
            full_shape = list(new_shape)
            full_shape[a] = fused_dim[q_f]
            full_block = jnp.zeros(full_shape, dtype=block.dtype)
            start_indices = [0] * len(new_shape)
            start_indices[a] = offset
            new_blocks[new_key] = jax.lax.dynamic_update_slice(
                full_block, block_flat, tuple(start_indices)
            )

    obj = object.__new__(SymmetricTensor)
    obj._indices = new_indices
    obj._blocks = new_blocks
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
        TensorIndex(sym, fused_charges, FlowDirection.IN, label="up"),
        TensorIndex(sym, fused_charges, FlowDirection.OUT, label="down"),
        TensorIndex(sym, fused_charges, FlowDirection.IN, label="left"),
        TensorIndex(sym, fused_charges, FlowDirection.OUT, label="right"),
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
