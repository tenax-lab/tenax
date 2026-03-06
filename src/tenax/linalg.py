r"""Linear algebra decompositions for Tenax tensors.

Public API::

    svd(tensor, left_labels, right_labels, ...) -> (U, s, Vh, s_full)
    qr(tensor, left_labels, right_labels, ...) -> (Q, R)
    eigh(tensor, left_labels, right_labels, ...) -> (V, eigenvalues)

Each function dispatches to a block-sparse path for ``SymmetricTensor``
(decomposing each charge sector independently) or falls through to
dense ``jnp.linalg`` for ``DenseTensor``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.index import FlowDirection, Label, TensorIndex
from tenax.core.tensor import (
    BlockKey,
    DenseTensor,
    SymmetricTensor,
    Tensor,
    _koszul_sign,
)

# ---------- Shared helpers ----------


def _group_blocks_by_bond_charge(
    tensor: SymmetricTensor,
    left_leg_positions: list[int],
    right_leg_positions: list[int],
) -> dict[int, list[tuple[BlockKey, BlockKey, jax.Array]]]:
    """Group tensor blocks by their bond charge sector.

    For each block, the "bond charge" is determined by fusing the flow-weighted
    charges of the left legs.  Blocks sharing the same bond charge belong to
    the same diagonal block in the matrix representation.

    Args:
        tensor:              SymmetricTensor to decompose.
        left_leg_positions:  Axis positions belonging to the left (U / Q) factor.
        right_leg_positions: Axis positions belonging to the right (Vh / R) factor.

    Returns:
        Dict mapping bond charge ``q`` to a list of
        ``(left_subkey, right_subkey, block_array)`` tuples.
    """
    sym = tensor.indices[0].symmetry
    grouped: dict[int, list[tuple[BlockKey, BlockKey, jax.Array]]] = {}

    for key, block in tensor.blocks.items():
        # Compute bond charge from left legs
        effective = [
            np.array([int(tensor.indices[i].flow) * int(key[i])], dtype=np.int32)
            for i in left_leg_positions
        ]
        q = int(sym.fuse_many(effective)[0])

        left_subkey = tuple(key[i] for i in left_leg_positions)
        right_subkey = tuple(key[i] for i in right_leg_positions)
        grouped.setdefault(q, []).append((left_subkey, right_subkey, block))

    return grouped


# ---------- Block-sparse SVD ----------


def _truncated_svd_symmetric(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    max_singular_values: int | None,
    max_truncation_err: float | None,
    new_bond_label: Label,
    normalize: bool,
) -> tuple[SymmetricTensor, jax.Array, SymmetricTensor, jax.Array]:
    """Block-diagonal SVD for SymmetricTensor.

    Each charge sector is decomposed independently, then singular values
    are merged and truncated globally.

    Returns ``(U, s_truncated, Vh, s_full)`` where *s_full* contains all
    singular values (sorted descending) before truncation.
    """
    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    # Check if fermionic signs are needed for leg reordering
    sym = tensor.indices[0].symmetry
    is_fermionic = sym.is_fermionic
    # The permutation from original leg order to (left_axes, right_axes)
    decomp_perm = tuple(left_axes + right_axes)

    # For each charge sector, we need to know the row/col dimensions of the
    # block-diagonal matrix.  Rows are indexed by unique left_subkeys within
    # the sector; columns by unique right_subkeys.

    # Per-sector SVD results
    sector_results: dict[
        int,
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            list[BlockKey],
            list[BlockKey],
            list[int],
            list[int],
        ],
    ] = {}

    for q, entries in grouped.items():
        # Collect unique left / right subkeys (preserving order for determinism)
        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
        for lk, rk, _ in entries:
            if lk not in left_subkeys_seen:
                left_subkeys_seen[lk] = len(left_subkeys_seen)
            if rk not in right_subkeys_seen:
                right_subkeys_seen[rk] = len(right_subkeys_seen)

        left_subkeys = list(left_subkeys_seen.keys())
        right_subkeys = list(right_subkeys_seen.keys())

        # Determine row size per left_subkey and col size per right_subkey
        # by computing the product of charge-multiplicities along each leg.
        left_row_sizes: list[int] = []
        for lk in left_subkeys:
            size = 1
            for leg_pos, charge_val in zip(left_axes, lk):
                idx = tensor.indices[leg_pos]
                size *= int(np.sum(idx.charges == charge_val))
            left_row_sizes.append(size)

        right_col_sizes: list[int] = []
        for rk in right_subkeys:
            size = 1
            for leg_pos, charge_val in zip(right_axes, rk):
                idx = tensor.indices[leg_pos]
                size *= int(np.sum(idx.charges == charge_val))
            right_col_sizes.append(size)

        total_rows = sum(left_row_sizes)
        total_cols = sum(right_col_sizes)

        if total_rows == 0 or total_cols == 0:
            continue

        # Assemble the block matrix for this charge sector
        matrix = jnp.zeros((total_rows, total_cols), dtype=tensor.dtype)
        for lk, rk, block in entries:
            li = left_subkeys_seen[lk]
            ri = right_subkeys_seen[rk]
            row_start = sum(left_row_sizes[:li])
            col_start = sum(right_col_sizes[:ri])
            flat_block = block.reshape(left_row_sizes[li], right_col_sizes[ri])
            # Apply Koszul sign for leg reordering (original -> left+right)
            if is_fermionic:
                full_key = [0] * len(tensor.indices)
                for ax, ch in zip(left_axes, lk):
                    full_key[ax] = ch
                for ax, ch in zip(right_axes, rk):
                    full_key[ax] = ch
                parities = tuple(
                    int(sym.parity(np.array([full_key[i]]))[0])
                    for i in range(len(full_key))
                )
                ksign = _koszul_sign(parities, decomp_perm)
                if ksign < 0:
                    flat_block = -flat_block
            matrix = matrix.at[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ].set(flat_block)

        # SVD this sector
        U_q, s_q, Vh_q = jnp.linalg.svd(matrix, full_matrices=False)
        sector_results[q] = (
            U_q,
            s_q,
            Vh_q,
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
        )

    # Global truncation: merge all singular values across sectors
    all_sv_pairs: list[
        tuple[float, int, int]
    ] = []  # (value, sector_q, index_in_sector)
    for q, (_, s_q, _, _, _, _, _) in sector_results.items():
        s_np = np.array(s_q)
        for i, val in enumerate(s_np):
            all_sv_pairs.append((float(val), q, i))

    # Sort descending by singular value
    all_sv_pairs.sort(key=lambda x: -x[0])

    # Preserve the full singular-value spectrum before truncation
    s_full = jnp.array([v for v, _, _ in all_sv_pairs])

    # Determine global keep count
    n_total = len(all_sv_pairs)
    n_keep = n_total

    if max_truncation_err is not None and n_total > 0:
        total_sq = sum(x[0] ** 2 for x in all_sv_pairs)
        if total_sq > 0:
            trunc_sq = 0.0
            for i in range(n_total - 1, 0, -1):
                trunc_sq += all_sv_pairs[i][0] ** 2
                if trunc_sq / total_sq > max_truncation_err**2:
                    n_keep = i + 1
                    break
            else:
                n_keep = n_total

    if max_singular_values is not None:
        n_keep = min(n_keep, max_singular_values)

    n_keep = max(1, min(n_keep, n_total))

    # Count per-sector keep
    kept = all_sv_pairs[:n_keep]
    sector_keep_count: dict[int, int] = {}
    for _, q, _ in kept:
        sector_keep_count[q] = sector_keep_count.get(q, 0) + 1

    # Build the bond index charges: one entry per kept singular value,
    # charge = q for the sector it belongs to.
    # We need to order them: iterate sectors in sorted order.
    bond_charges_list: list[int] = []
    # Collect the final singular values in the same order
    final_sv_list: list[float] = []

    for q in sorted(sector_keep_count.keys()):
        n_q = sector_keep_count[q]
        bond_charges_list.extend([q] * n_q)
        s_q_np = np.array(sector_results[q][1])
        final_sv_list.extend(s_q_np[:n_q].tolist())

    bond_charges = np.array(bond_charges_list, dtype=np.int32)
    s_final = jnp.array(final_sv_list)

    if normalize and jnp.sum(s_final) > 0:
        s_final = s_final / jnp.sum(s_final)

    sym = tensor.indices[0].symmetry

    bond_index_out = TensorIndex(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    # Reconstruct U blocks: keys are (left_subkey..., bond_charge_q)
    # U has indices: (left_indices..., bond_index_out)
    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices

    U_blocks: dict[BlockKey, jax.Array] = {}
    Vh_blocks: dict[BlockKey, jax.Array] = {}

    for q in sorted(sector_keep_count.keys()):
        U_q, _, Vh_q, left_subkeys, right_subkeys, left_row_sizes, right_col_sizes = (
            sector_results[q]
        )
        n_q = sector_keep_count[q]

        # Slice U_q and Vh_q to keep only n_q singular vectors
        U_q_trunc = U_q[:, :n_q]
        Vh_q_trunc = Vh_q[:n_q, :]

        # Split U_q rows back into individual left_subkey blocks
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            u_slice = U_q_trunc[row_offset : row_offset + n_rows, :]
            # Reshape: (prod(left_shape_for_lk), n_q) -> (left_shape_for_lk..., n_q)
            left_shape = tuple(
                int(np.sum(tensor.indices[ax].charges == ch))
                for ax, ch in zip(left_axes, lk)
            )
            u_block = u_slice.reshape(left_shape + (n_q,))
            block_key = lk + (q,)
            U_blocks[block_key] = u_block
            row_offset += n_rows

        # Split Vh_q cols back into individual right_subkey blocks
        col_offset = 0
        for ri, rk in enumerate(right_subkeys):
            n_cols = right_col_sizes[ri]
            vh_slice = Vh_q_trunc[:, col_offset : col_offset + n_cols]
            right_shape = tuple(
                int(np.sum(tensor.indices[ax].charges == ch))
                for ax, ch in zip(right_axes, rk)
            )
            vh_block = vh_slice.reshape((n_q,) + right_shape)
            block_key = (q,) + rk
            Vh_blocks[block_key] = vh_block
            col_offset += n_cols

    # Check if input tensor has a non-identity target (e.g. boundary MPS
    # tensor targeting Sz != 0).  If so, the output tensors may also have
    # non-identity targets and need to bypass standard validation.
    input_target = 0
    if tensor.blocks:
        key0 = next(iter(tensor.blocks))
        input_target = sum(
            int(idx.flow) * int(q) for idx, q in zip(tensor.indices, key0)
        )

    if input_target != 0:
        # Bypass validation for non-identity targets
        U_tensor = object.__new__(SymmetricTensor)
        U_tensor._indices = U_indices
        U_tensor._blocks = U_blocks
        Vh_tensor = object.__new__(SymmetricTensor)
        Vh_tensor._indices = Vh_indices
        Vh_tensor._blocks = Vh_blocks
    else:
        U_tensor = SymmetricTensor(U_blocks, U_indices)
        Vh_tensor = SymmetricTensor(Vh_blocks, Vh_indices)

    return U_tensor, s_final, Vh_tensor, s_full


# ---------- Block-sparse QR ----------


def _qr_symmetric(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Block-diagonal QR decomposition for SymmetricTensor.

    Each charge sector is decomposed independently; the bond index carries
    the sector charge with multiplicity = min(left_dim, right_dim) per sector.
    """
    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    # Check if fermionic signs are needed for leg reordering
    sym = tensor.indices[0].symmetry
    is_fermionic = sym.is_fermionic
    decomp_perm = tuple(left_axes + right_axes)

    # Per-sector QR results
    sector_results: dict[
        int,
        tuple[
            jax.Array,
            jax.Array,
            list[BlockKey],
            list[BlockKey],
            list[int],
            list[int],
            int,
        ],
    ] = {}

    bond_charges_list: list[int] = []

    for q in sorted(grouped.keys()):
        entries = grouped[q]

        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
        for lk, rk, _ in entries:
            if lk not in left_subkeys_seen:
                left_subkeys_seen[lk] = len(left_subkeys_seen)
            if rk not in right_subkeys_seen:
                right_subkeys_seen[rk] = len(right_subkeys_seen)

        left_subkeys = list(left_subkeys_seen.keys())
        right_subkeys = list(right_subkeys_seen.keys())

        left_row_sizes: list[int] = []
        for lk in left_subkeys:
            size = 1
            for leg_pos, charge_val in zip(left_axes, lk):
                idx = tensor.indices[leg_pos]
                size *= int(np.sum(idx.charges == charge_val))
            left_row_sizes.append(size)

        right_col_sizes: list[int] = []
        for rk in right_subkeys:
            size = 1
            for leg_pos, charge_val in zip(right_axes, rk):
                idx = tensor.indices[leg_pos]
                size *= int(np.sum(idx.charges == charge_val))
            right_col_sizes.append(size)

        total_rows = sum(left_row_sizes)
        total_cols = sum(right_col_sizes)

        if total_rows == 0 or total_cols == 0:
            continue

        # Assemble block matrix
        matrix = jnp.zeros((total_rows, total_cols), dtype=tensor.dtype)
        for lk, rk, block in entries:
            li = left_subkeys_seen[lk]
            ri = right_subkeys_seen[rk]
            row_start = sum(left_row_sizes[:li])
            col_start = sum(right_col_sizes[:ri])
            flat_block = block.reshape(left_row_sizes[li], right_col_sizes[ri])
            # Apply Koszul sign for leg reordering (original -> left+right)
            if is_fermionic:
                full_key = [0] * len(tensor.indices)
                for ax, ch in zip(left_axes, lk):
                    full_key[ax] = ch
                for ax, ch in zip(right_axes, rk):
                    full_key[ax] = ch
                parities = tuple(
                    int(sym.parity(np.array([full_key[i]]))[0])
                    for i in range(len(full_key))
                )
                ksign = _koszul_sign(parities, decomp_perm)
                if ksign < 0:
                    flat_block = -flat_block
            matrix = matrix.at[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ].set(flat_block)

        Q_q, R_q = jnp.linalg.qr(matrix)
        bond_dim_q = Q_q.shape[1]

        bond_charges_list.extend([q] * bond_dim_q)
        sector_results[q] = (
            Q_q,
            R_q,
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
            bond_dim_q,
        )

    bond_charges = np.array(bond_charges_list, dtype=np.int32)
    sym = tensor.indices[0].symmetry

    bond_index_out = TensorIndex(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    Q_indices = left_indices + (bond_index_out,)
    R_indices = (bond_index_in,) + right_indices

    Q_blocks: dict[BlockKey, jax.Array] = {}
    R_blocks: dict[BlockKey, jax.Array] = {}

    for q, (
        Q_q,
        R_q,
        left_subkeys,
        right_subkeys,
        left_row_sizes,
        right_col_sizes,
        bond_dim_q,
    ) in sector_results.items():
        # Split Q rows back into left_subkey blocks
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            q_slice = Q_q[row_offset : row_offset + n_rows, :]
            left_shape = tuple(
                int(np.sum(tensor.indices[ax].charges == ch))
                for ax, ch in zip(left_axes, lk)
            )
            q_block = q_slice.reshape(left_shape + (bond_dim_q,))
            Q_blocks[lk + (q,)] = q_block
            row_offset += n_rows

        # Split R cols back into right_subkey blocks
        col_offset = 0
        for ri, rk in enumerate(right_subkeys):
            n_cols = right_col_sizes[ri]
            r_slice = R_q[:, col_offset : col_offset + n_cols]
            right_shape = tuple(
                int(np.sum(tensor.indices[ax].charges == ch))
                for ax, ch in zip(right_axes, rk)
            )
            r_block = r_slice.reshape((bond_dim_q,) + right_shape)
            R_blocks[(q,) + rk] = r_block
            col_offset += n_cols

    Q_tensor = SymmetricTensor(Q_blocks, Q_indices)
    R_tensor = SymmetricTensor(R_blocks, R_indices)

    return Q_tensor, R_tensor


# ---------- Block-sparse eigh ----------


def _eigh_symmetric(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label,
    max_eigenvalues: int | None,
) -> tuple[SymmetricTensor, jax.Array]:
    """Block-diagonal Hermitian eigendecomposition for SymmetricTensor.

    Each charge sector is eigendecomposed independently, then eigenvalues
    are merged and truncated globally (keeping the largest).
    """
    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    # Per-sector eigh results: (eigvecs, eigvals, left_subkeys, left_row_sizes)
    sector_results: dict[
        int,
        tuple[jax.Array, jax.Array, list[BlockKey], list[int]],
    ] = {}

    for q, entries in grouped.items():
        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
        for lk, rk, _ in entries:
            if lk not in left_subkeys_seen:
                left_subkeys_seen[lk] = len(left_subkeys_seen)
            if rk not in right_subkeys_seen:
                right_subkeys_seen[rk] = len(right_subkeys_seen)

        left_subkeys = list(left_subkeys_seen.keys())

        left_row_sizes: list[int] = []
        for lk in left_subkeys:
            size = 1
            for leg_pos, charge_val in zip(left_axes, lk):
                idx = tensor.indices[leg_pos]
                size *= int(np.sum(idx.charges == charge_val))
            left_row_sizes.append(size)

        right_col_sizes: list[int] = []
        right_subkeys = list(right_subkeys_seen.keys())
        for rk in right_subkeys:
            size = 1
            for leg_pos, charge_val in zip(right_axes, rk):
                idx = tensor.indices[leg_pos]
                size *= int(np.sum(idx.charges == charge_val))
            right_col_sizes.append(size)

        total_rows = sum(left_row_sizes)
        total_cols = sum(right_col_sizes)

        if total_rows == 0 or total_cols == 0:
            continue

        # Assemble the block matrix
        matrix = jnp.zeros((total_rows, total_cols), dtype=tensor.dtype)
        for lk, rk, block in entries:
            li = left_subkeys_seen[lk]
            ri = right_subkeys_seen[rk]
            row_start = sum(left_row_sizes[:li])
            col_start = sum(right_col_sizes[:ri])
            flat_block = block.reshape(left_row_sizes[li], right_col_sizes[ri])
            matrix = matrix.at[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ].set(flat_block)

        # Symmetrize for numerical stability
        matrix = 0.5 * (matrix + matrix.conj().T)
        eigvals_q, eigvecs_q = jnp.linalg.eigh(matrix)
        sector_results[q] = (eigvecs_q, eigvals_q, left_subkeys, left_row_sizes)

    # Global truncation: merge eigenvalues across sectors, keep top-k
    all_eig_pairs: list[tuple[float, int, int]] = []
    for q, (_, eigvals_q, _, _) in sector_results.items():
        ev_np = np.array(eigvals_q)
        for i, val in enumerate(ev_np):
            all_eig_pairs.append((float(val), q, i))

    # Sort descending by eigenvalue, then descending by index to match
    # the dense convention of taking eigvecs[:, -k:] for degenerate eigenvalues.
    all_eig_pairs.sort(key=lambda x: (-x[0], -x[2]))

    n_total = len(all_eig_pairs)
    n_keep = n_total
    if max_eigenvalues is not None:
        n_keep = min(n_keep, max_eigenvalues)
    n_keep = max(1, min(n_keep, n_total))

    kept = all_eig_pairs[:n_keep]

    # Eigenvalues in descending order
    eigenvalues = jnp.array([v for v, _, _ in kept])

    # Per-sector keep count and which indices to keep
    sector_keep: dict[int, list[int]] = {}
    for _, q, idx_in_sector in kept:
        sector_keep.setdefault(q, []).append(idx_in_sector)

    # Build bond index charges
    bond_charges_list: list[int] = []
    for q in sorted(sector_keep.keys()):
        bond_charges_list.extend([q] * len(sector_keep[q]))

    bond_charges = np.array(bond_charges_list, dtype=np.int32)
    sym = tensor.indices[0].symmetry

    bond_index_out = TensorIndex(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )

    V_indices = left_indices + (bond_index_out,)
    V_blocks: dict[BlockKey, jax.Array] = {}

    for q in sorted(sector_keep.keys()):
        eigvecs_q, _, left_subkeys, left_row_sizes = sector_results[q]
        keep_indices = sorted(sector_keep[q])
        n_q = len(keep_indices)

        # Select kept eigenvectors
        V_q = eigvecs_q[:, keep_indices]

        # Split rows back into left_subkey blocks
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            v_slice = V_q[row_offset : row_offset + n_rows, :]
            left_shape = tuple(
                int(np.sum(tensor.indices[ax].charges == ch))
                for ax, ch in zip(left_axes, lk)
            )
            v_block = v_slice.reshape(left_shape + (n_q,))
            V_blocks[lk + (q,)] = v_block
            row_offset += n_rows

    V_tensor = SymmetricTensor(V_blocks, V_indices)
    return V_tensor, eigenvalues


# ---------- Public API ----------


def svd(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
    max_singular_values: int | None = None,
    max_truncation_err: float | None = None,
    normalize: bool = False,
) -> tuple[Tensor, jax.Array, Tensor, jax.Array]:
    """Reshape tensor into matrix, compute SVD, truncate, reshape back.

    The tensor is first reshaped into a matrix by grouping left_labels as
    rows and right_labels as columns. After SVD and truncation, the result
    is reshaped back.

    The new bond leg (connecting U and Vh factors) is given label
    new_bond_label, making it immediately usable in label-based contractions.

    Output labels::

        U:  (left_labels..., new_bond_label)
        Vh: (new_bond_label, right_labels...)

    Note:
        This function is not JIT-able as a whole because the truncation
        cutoff is determined dynamically from singular values (dynamic shape).
        Apply ``@jax.jit`` to the inner SVD step only; call this at Python level.

    Args:
        tensor:               Tensor to decompose.
        left_labels:          Labels forming the "left" (U) factor.
        right_labels:         Labels forming the "right" (Vh) factor.
        new_bond_label:       Label for the new virtual bond.
        max_singular_values:  Hard cap on bond dimension after truncation.
        max_truncation_err:   Truncate until relative truncation error <= this.
        normalize:            Normalize singular values to sum to 1.

    Returns:
        ``(U_tensor, singular_values, Vh_tensor, singular_values_full)``
        -- U has labels ``(left_labels..., new_bond_label)``.
        Vh has labels ``(new_bond_label, right_labels...)``.
        singular_values is a 1-D JAX float array (truncated).
        singular_values_full is a 1-D JAX float array containing **all**
        singular values before truncation (length = min(left_dim, right_dim)),
        useful for computing truncation error without a second SVD.

    Raises:
        ValueError: If left_labels + right_labels don't cover all tensor labels.
    """
    all_labels = tensor.labels()
    all_labels_set = set(all_labels)
    left_set = set(left_labels)
    right_set = set(right_labels)

    if left_set | right_set != all_labels_set:
        raise ValueError(
            f"left_labels {list(left_labels)} + right_labels {list(right_labels)} "
            f"must cover all tensor labels {list(all_labels)}"
        )
    if left_set & right_set:
        raise ValueError(
            f"left_labels and right_labels must be disjoint, "
            f"got overlap: {left_set & right_set}"
        )

    # Dispatch to block-sparse path for SymmetricTensor
    if isinstance(tensor, SymmetricTensor):
        return _truncated_svd_symmetric(
            tensor,
            left_labels,
            right_labels,
            max_singular_values,
            max_truncation_err,
            new_bond_label,
            normalize,
        )

    # Build axis ordering: left labels first, then right labels
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]

    # Get dense representation and reshape
    dense = tensor.todense()
    perm = left_axes + right_axes
    dense_perm = jnp.transpose(dense, perm)

    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)
    left_dim = int(np.prod([idx.dim for idx in left_indices]))
    right_dim = int(np.prod([idx.dim for idx in right_indices]))

    matrix = dense_perm.reshape(left_dim, right_dim)

    # SVD (not JIT-able at this level due to dynamic truncation)
    U, s, Vh = jnp.linalg.svd(matrix, full_matrices=False)

    # Preserve the full singular-value spectrum before truncation
    s_full = s

    # Determine truncation cutoff
    s_np = np.array(s)
    n_keep = len(s_np)

    if max_truncation_err is not None:
        # Keep singular values until truncation error <= max_truncation_err
        total_sq = float(np.sum(s_np**2))
        trunc_sq = 0.0
        for i in range(len(s_np) - 1, -1, -1):
            trunc_sq += float(s_np[i] ** 2)
            if trunc_sq / total_sq > max_truncation_err**2:
                n_keep = i + 1
                break
        else:
            n_keep = len(s_np)

    if max_singular_values is not None:
        n_keep = min(n_keep, max_singular_values)

    n_keep = max(1, n_keep)  # always keep at least one

    # Truncate
    U = U[:, :n_keep]
    s = s[:n_keep]
    Vh = Vh[:n_keep, :]

    if normalize:
        s = s / jnp.sum(s)

    # Reshape back and build output tensors
    left_shape = tuple(idx.dim for idx in left_indices)
    right_shape = tuple(idx.dim for idx in right_indices)

    U_dense = U.reshape(left_shape + (n_keep,))
    Vh_dense = Vh.reshape((n_keep,) + right_shape)

    # Build new bond index
    bond_charges_out = np.zeros(n_keep, dtype=np.int32)
    if left_indices:
        sym = left_indices[0].symmetry
    elif right_indices:
        sym = right_indices[0].symmetry
    else:
        from tenax.core.symmetry import U1Symmetry

        sym = U1Symmetry()

    bond_index_out = TensorIndex(
        sym, bond_charges_out, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex(
        sym, bond_charges_out, FlowDirection.IN, label=new_bond_label
    )

    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices

    U_tensor = DenseTensor(U_dense, U_indices)
    Vh_tensor = DenseTensor(Vh_dense, Vh_indices)

    return U_tensor, s, Vh_tensor, s_full


def qr(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
) -> tuple[Tensor, Tensor]:
    """QR decomposition of a tensor for canonical form in DMRG.

    Reshapes tensor into a matrix, performs QR, then reshapes back.

    Output labels::

        Q: (left_labels..., new_bond_label)
        R: (new_bond_label, right_labels...)

    Args:
        tensor:          Tensor to decompose.
        left_labels:     Labels forming the Q (isometric) factor.
        right_labels:    Labels forming the R (upper triangular) factor.
        new_bond_label:  Label for the new virtual bond.

    Returns:
        (Q_tensor, R_tensor) where Q is isometric (Q^dag Q = I).
    """
    # Dispatch to block-sparse path for SymmetricTensor
    if isinstance(tensor, SymmetricTensor):
        return _qr_symmetric(tensor, left_labels, right_labels, new_bond_label)

    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]

    dense = tensor.todense()
    perm = left_axes + right_axes
    dense_perm = jnp.transpose(dense, perm)

    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)
    left_dim = int(np.prod([idx.dim for idx in left_indices]))
    right_dim = int(np.prod([idx.dim for idx in right_indices]))

    matrix = dense_perm.reshape(left_dim, right_dim)
    Q, R = jnp.linalg.qr(matrix)

    bond_dim = Q.shape[1]
    left_shape = tuple(idx.dim for idx in left_indices)
    right_shape = tuple(idx.dim for idx in right_indices)

    Q_dense = Q.reshape(left_shape + (bond_dim,))
    R_dense = R.reshape((bond_dim,) + right_shape)

    bond_charges = np.zeros(bond_dim, dtype=np.int32)
    if left_indices:
        sym = left_indices[0].symmetry
    else:
        from tenax.core.symmetry import U1Symmetry

        sym = U1Symmetry()

    bond_index_out = TensorIndex(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    Q_indices = left_indices + (bond_index_out,)
    R_indices = (bond_index_in,) + right_indices

    Q_tensor = DenseTensor(Q_dense, Q_indices)
    R_tensor = DenseTensor(R_dense, R_indices)

    return Q_tensor, R_tensor


def eigh(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
    max_eigenvalues: int | None = None,
) -> tuple[Tensor, jax.Array]:
    """Eigendecompose a Hermitian tensor.

    Reshapes the tensor into a square matrix (left_labels vs right_labels),
    computes the eigendecomposition, and returns eigenvectors as a Tensor.

    Eigenvalues are sorted in descending order. If ``max_eigenvalues`` is
    given, only the top-k eigenvalues (and corresponding eigenvectors) are
    kept.

    Output labels::

        V: (left_labels..., new_bond_label)

    Args:
        tensor:           Hermitian tensor to decompose.
        left_labels:      Labels forming the row side of the matrix.
        right_labels:     Labels forming the column side of the matrix.
        new_bond_label:   Label for the eigenvector bond index.
        max_eigenvalues:  Keep only the top-k eigenvalues.

    Returns:
        ``(V, eigenvalues)`` where V has labels ``(left_labels..., new_bond_label)``
        and eigenvalues is a 1-D JAX array sorted descending.
    """
    # Dispatch to block-sparse path for SymmetricTensor
    if isinstance(tensor, SymmetricTensor):
        return _eigh_symmetric(
            tensor, left_labels, right_labels, new_bond_label, max_eigenvalues
        )

    # Dense path
    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]

    dense = tensor.todense()
    perm = left_axes + right_axes
    dense_perm = jnp.transpose(dense, perm)

    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)
    left_dim = int(np.prod([idx.dim for idx in left_indices]))
    right_dim = int(np.prod([idx.dim for idx in right_indices]))

    matrix = dense_perm.reshape(left_dim, right_dim)
    matrix = 0.5 * (matrix + matrix.conj().T)  # symmetrize
    eigvals, eigvecs = jnp.linalg.eigh(matrix)

    # eigh returns ascending; reverse for descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # Truncate
    n_total = len(eigvals)
    n_keep = n_total
    if max_eigenvalues is not None:
        n_keep = min(n_keep, max_eigenvalues)
    n_keep = max(1, n_keep)

    eigvals = eigvals[:n_keep]
    eigvecs = eigvecs[:, :n_keep]

    # Reshape back
    left_shape = tuple(idx.dim for idx in left_indices)
    V_dense = eigvecs.reshape(left_shape + (n_keep,))

    bond_charges = np.zeros(n_keep, dtype=np.int32)
    if left_indices:
        sym = left_indices[0].symmetry
    else:
        from tenax.core.symmetry import U1Symmetry

        sym = U1Symmetry()

    bond_index = TensorIndex(sym, bond_charges, FlowDirection.OUT, label=new_bond_label)
    V_indices = left_indices + (bond_index,)
    V_tensor = DenseTensor(V_dense, V_indices)

    return V_tensor, eigvals
