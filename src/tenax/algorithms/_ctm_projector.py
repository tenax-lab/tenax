"""CTM projector computation for the Tensor protocol.

Provides ``_compute_projector_tensor`` which computes isometric projectors
from two grown corner tensors, with block-sparse support for SymmetricTensor.

Shared by both the standard CTM (``_ctm_tensor.py``) and the split CTM
(``_split_ctm_tensor.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

OUT = FlowDirection.OUT


# ------------------------------------------------------------------ #
# Block-sparse projector helpers                                       #
# ------------------------------------------------------------------ #


def _build_symmetric_projector(
    proj_blocks: dict[tuple[int, ...], jax.Array],
    chi_new_charges: list[int],
    fused_idx: TensorIndex,
    dtype: jnp.dtype,
) -> SymmetricTensor:
    """Assemble a projector SymmetricTensor from per-sector blocks.

    Args:
        proj_blocks: Map ``(fq, fq) -> V_q`` of eigenvector blocks.
        chi_new_charges: Flat list of chi_new charges (one per kept vector).
        fused_idx: The fused TensorIndex (row index of the projector).
        dtype: Data type for the output tensor.

    Returns:
        SymmetricTensor with indices ``(fused_idx, chi_new_idx)``.
    """
    sym = fused_idx.symmetry
    chi_new_idx = TensorIndex(
        sym, np.array(chi_new_charges, dtype=np.int32), OUT, label="chi_new"
    )
    obj = object.__new__(SymmetricTensor)
    obj._indices = (fused_idx, chi_new_idx)
    sorted_keys = sorted(proj_blocks.keys())
    if sorted_keys:
        flat_parts = [proj_blocks[k].ravel() for k in sorted_keys]
        obj._data = jnp.concatenate(flat_parts)
    else:
        obj._data = jnp.zeros(0, dtype=dtype)
    obj._block_keys = tuple(sorted_keys)
    shapes = [proj_blocks[k].shape for k in sorted_keys]
    obj._block_shapes = tuple(shapes)
    offsets: list[int] = []
    offset = 0
    for s in shapes:
        size = 1
        for d in s:
            size *= d
        offsets.append(offset)
        offset += size
    obj._block_offsets = tuple(offsets)
    return obj


def _eigh_projector_symmetric(
    C1g: SymmetricTensor,
    C4g: SymmetricTensor,
    chi: int,
) -> SymmetricTensor:
    r"""Block-sparse projector via per-sector density matrix eigh.

    For each charge sector *q* of the ``fused`` leg, accumulates
    :math:`\rho_q = M_1 M_1^\dagger + M_4 M_4^\dagger` where :math:`M_i`
    is the sector's dense block of C_ig, then eigendecomposes :math:`\rho_q`.
    Eigenvalues are merged across sectors and globally truncated to *chi*.

    The projector is constructed directly as a SymmetricTensor with
    correct per-sector charges on the ``chi_new`` index (charge = fused
    charge of the originating sector), preserving full block-sparse structure.

    Args:
        C1g: Grown corner SymmetricTensor ``(fused, col1)``.
        C4g: Grown corner SymmetricTensor ``(fused, col2)``.
        chi: Target bond dimension.

    Returns:
        Projector ``P`` with labels ``(fused, chi_new)``, flows ``(IN, OUT)``.
    """
    fused_pos = C1g.labels().index("fused")
    col_pos = 1 - fused_pos  # the other leg
    fused_idx = C1g.indices[fused_pos]

    # Group blocks by fused charge for each corner
    def _group_by_fused(Cg: SymmetricTensor) -> dict[int, list[tuple[int, jax.Array]]]:
        """Map fused_charge -> list of (col_charge, block)."""
        grouped: dict[int, list[tuple[int, jax.Array]]] = {}
        for key, block in Cg.blocks.items():
            fq = int(key[fused_pos])
            cq = int(key[col_pos])
            grouped.setdefault(fq, []).append((cq, block))
        return grouped

    c1_groups = _group_by_fused(C1g)
    c4_groups = _group_by_fused(C4g)

    all_fused_charges = sorted(set(c1_groups.keys()) | set(c4_groups.keys()))

    # Per-sector eigh results: q -> (eigvecs, eigvals, fused_dim, row_offset)
    sector_results: dict[int, tuple[jax.Array, jax.Array, int, int]] = {}

    # Build a map from fused charge to its row indices in the dense fused index
    charges_arr = np.asarray(fused_idx.charges)
    charge_rows: dict[int, np.ndarray] = {}
    for fq in all_fused_charges:
        charge_rows[fq] = np.where(charges_arr == fq)[0]

    for fq in all_fused_charges:
        fused_dim = int(len(charge_rows.get(fq, [])))
        if fused_dim == 0:
            continue

        # Accumulate rho = sum of M @ M^dagger for both corners
        rho = jnp.zeros((fused_dim, fused_dim), dtype=C1g.dtype)

        for entries in [c1_groups.get(fq, []), c4_groups.get(fq, [])]:
            for _cq, block in entries:
                if fused_pos == 0:
                    M = block.reshape(fused_dim, -1)
                else:
                    M = block.reshape(-1, fused_dim).T
                rho = rho + M @ M.conj().T

        rho = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = jnp.linalg.eigh(rho)
        sector_results[fq] = (eigvecs, eigvals, fused_dim, charge_rows[fq])

    # Global truncation: merge eigenvalues, keep top-chi
    all_eig_pairs: list[tuple[float, int, int]] = []  # (value, fused_charge, index)
    for fq, (_, eigvals, _, _) in sector_results.items():
        for i, val in enumerate(np.array(eigvals)):
            all_eig_pairs.append((float(val), fq, i))

    # Sort descending by eigenvalue, then descending by index to match
    # the dense convention of taking eigvecs[:, -k:] (highest indices first
    # among degenerate eigenvalues).
    all_eig_pairs.sort(key=lambda x: (-x[0], -x[2]))
    n_keep = min(chi, len(all_eig_pairs))

    # Count per-sector keeps
    sector_keep: dict[int, list[int]] = {}
    for _, fq, idx in all_eig_pairs[:n_keep]:
        sector_keep.setdefault(fq, []).append(idx)

    # Build projector blocks directly with correct per-sector charges.
    # Each kept eigenvector from sector fq gets chi_new charge = fq
    # (conservation: flow_fused*fq + flow_chi_new*q_chi = 0 → q_chi = fq).
    chi_new_charges: list[int] = []
    proj_blocks: dict[tuple[int, ...], jax.Array] = {}

    for fq in sorted(sector_keep.keys()):
        keep_indices = sorted(sector_keep[fq], reverse=True)
        n_q = len(keep_indices)
        chi_new_charges.extend([fq] * n_q)

        eigvecs, _, fused_dim, row_idx = sector_results[fq]
        V_q = eigvecs[:, keep_indices]  # (fused_dim, n_q)
        V_q = jax.lax.stop_gradient(V_q)
        proj_blocks[(fq, fq)] = V_q

    return _build_symmetric_projector(
        proj_blocks, chi_new_charges, fused_idx, C1g.dtype
    )


def _qr_projector_symmetric(
    C1g: SymmetricTensor,
    C4g: SymmetricTensor,
    chi: int,
) -> SymmetricTensor:
    r"""Block-sparse projector via per-sector QR + small eigh.

    For each charge sector *q* of the ``fused`` leg, concatenates the
    column-blocks from both grown corners to form :math:`M_q`, then
    QR-factors :math:`M_q = Q_q R_q`.  The reduced eigenproblem
    :math:`R_q R_q^\dagger` is cheaper than the full
    :math:`\rho_q = M_q M_q^\dagger` when the column dimension is small.

    Eigenvalues are merged across sectors and globally truncated to *chi*.

    Args:
        C1g: Grown corner SymmetricTensor ``(fused, col1)``.
        C4g: Grown corner SymmetricTensor ``(fused, col2)``.
        chi: Target bond dimension.

    Returns:
        Projector ``P`` with labels ``(fused, chi_new)``, flows ``(IN, OUT)``.
    """
    fused_pos = C1g.labels().index("fused")
    col_pos = 1 - fused_pos
    fused_idx = C1g.indices[fused_pos]

    # Group blocks by fused charge for each corner
    def _group_by_fused(
        Cg: SymmetricTensor,
    ) -> dict[int, list[tuple[int, jax.Array]]]:
        grouped: dict[int, list[tuple[int, jax.Array]]] = {}
        for key, block in Cg.blocks.items():
            fq = int(key[fused_pos])
            cq = int(key[col_pos])
            grouped.setdefault(fq, []).append((cq, block))
        return grouped

    c1_groups = _group_by_fused(C1g)
    c4_groups = _group_by_fused(C4g)
    all_fused_charges = sorted(set(c1_groups.keys()) | set(c4_groups.keys()))

    charges_arr = np.asarray(fused_idx.charges)
    charge_rows: dict[int, np.ndarray] = {}
    for fq in all_fused_charges:
        charge_rows[fq] = np.where(charges_arr == fq)[0]

    # Per-sector QR + small eigh
    sector_results: dict[int, tuple[jax.Array, jax.Array]] = {}  # fq -> (P_q, eigvals)

    for fq in all_fused_charges:
        fused_dim = int(len(charge_rows.get(fq, [])))
        if fused_dim == 0:
            continue

        # Collect column blocks from both corners for this fused charge
        col_blocks: list[jax.Array] = []
        for entries in [c1_groups.get(fq, []), c4_groups.get(fq, [])]:
            for _cq, block in entries:
                if fused_pos == 0:
                    col_blocks.append(block.reshape(fused_dim, -1))
                else:
                    col_blocks.append(block.reshape(-1, fused_dim).T)

        if not col_blocks:
            continue

        M_q = jnp.concatenate(col_blocks, axis=1)  # (fused_dim, total_col)
        Q_q, R_q = jnp.linalg.qr(M_q)

        rho_small = R_q @ R_q.conj().T
        rho_small = 0.5 * (rho_small + rho_small.conj().T)
        eigvals, eigvecs = jnp.linalg.eigh(rho_small)

        # P_q = Q @ eigvecs maps from reduced space back to fused space
        P_q = Q_q @ eigvecs  # (fused_dim, min(fused_dim, total_col))
        sector_results[fq] = (P_q, eigvals)

    # Global truncation across sectors
    all_eig_pairs: list[tuple[float, int, int]] = []
    for fq, (_, eigvals) in sector_results.items():
        for i, val in enumerate(np.array(eigvals)):
            all_eig_pairs.append((float(val), fq, i))

    all_eig_pairs.sort(key=lambda x: (-x[0], -x[2]))
    n_keep = min(chi, len(all_eig_pairs))

    sector_keep: dict[int, list[int]] = {}
    for _, fq, idx in all_eig_pairs[:n_keep]:
        sector_keep.setdefault(fq, []).append(idx)

    # Build projector blocks
    chi_new_charges: list[int] = []
    proj_blocks: dict[tuple[int, ...], jax.Array] = {}

    for fq in sorted(sector_keep.keys()):
        keep_indices = sorted(sector_keep[fq], reverse=True)
        n_q = len(keep_indices)
        chi_new_charges.extend([fq] * n_q)

        P_q, _ = sector_results[fq]
        V_q = P_q[:, keep_indices]  # (fused_dim, n_q)
        V_q = jax.lax.stop_gradient(V_q)
        proj_blocks[(fq, fq)] = V_q

    return _build_symmetric_projector(
        proj_blocks, chi_new_charges, fused_idx, C1g.dtype
    )


# ------------------------------------------------------------------ #
# Main projector entry point                                           #
# ------------------------------------------------------------------ #


def _compute_projector_tensor(
    C1g: Tensor,
    C4g: Tensor,
    chi: int,
    projector_method: str = "eigh",
) -> Tensor:
    r"""Compute isometric projector P as a Tensor.

    When ``projector_method == "eigh"``, forms the full density matrix
    :math:`\rho = C_{1g} C_{1g}^\dagger + C_{4g} C_{4g}^\dagger`,
    eigendecomposes, then wraps the top-k eigenvectors as a Tensor.

    When ``projector_method == "qr"``, QR-factors the concatenated corners
    ``[C1g, C4g]`` to reduce to a small ``(2*col, 2*col)`` eigenproblem,
    following the approach in arXiv:2505.00494.

    For SymmetricTensor inputs (both ``"eigh"`` and ``"qr"``), uses
    per-charge-sector decomposition to avoid dense round-trip.

    Args:
        C1g: Grown corner with labels ``(fused, <col1>)``.
        C4g: Grown corner with labels ``(fused, <col2>)``.
        chi: Target bond dimension.
        projector_method: ``"eigh"`` or ``"qr"``.

    Returns:
        Projector ``P`` with labels ``(fused, chi_new)``,
        flows ``(IN, OUT)``.  Wrapped in ``stop_gradient``.

    Raises:
        ValueError: If ``projector_method`` is not ``"eigh"`` or ``"qr"``.
    """
    if projector_method not in ("eigh", "qr"):
        raise ValueError(
            f"Unknown projector_method={projector_method!r}; expected 'eigh' or 'qr'."
        )

    # --- QR path ---
    if projector_method == "qr":
        # Block-sparse QR for SymmetricTensor (non-tracer)
        if isinstance(C1g, SymmetricTensor) and isinstance(C4g, SymmetricTensor):
            has_tracers = isinstance(C1g._data, jax.core.Tracer) or isinstance(
                C4g._data, jax.core.Tracer
            )
            if not has_tracers and C1g.n_blocks > 0 and C4g.n_blocks > 0:
                return _qr_projector_symmetric(C1g, C4g, chi)

        # Dense QR fallback
        C1g_dense = C1g.todense()
        C4g_dense = C4g.todense()

        M = jnp.concatenate([C1g_dense, C4g_dense], axis=1)
        Q, R = jnp.linalg.qr(M)

        rho_small = R @ R.conj().T
        rho_small = 0.5 * (rho_small + rho_small.conj().T)
        eigvals, eigvecs = jnp.linalg.eigh(rho_small)

        k = min(chi, len(eigvals))
        V = eigvecs[:, -k:][:, ::-1]
        P_dense = Q @ V
        P_dense = jax.lax.stop_gradient(P_dense)

        fused_idx = C1g.indices[C1g.labels().index("fused")]
        chi_new_idx = TensorIndex(
            fused_idx.symmetry,
            np.zeros(k, dtype=np.int32),
            OUT,
            label="chi_new",
        )
        if isinstance(C1g, SymmetricTensor):
            return SymmetricTensor.from_dense(
                P_dense, (fused_idx, chi_new_idx), tol=float("inf")
            )
        return DenseTensor(P_dense, (fused_idx, chi_new_idx))

    # --- eigh path ---
    # Use block-sparse path for SymmetricTensor unless blocks contain
    # JAX tracers (during AD), in which case fall back to dense path
    # since eigenvalue sorting requires concrete values.
    if isinstance(C1g, SymmetricTensor) and isinstance(C4g, SymmetricTensor):
        has_tracers = isinstance(C1g._data, jax.core.Tracer) or isinstance(
            C4g._data, jax.core.Tracer
        )
        if not has_tracers and C1g.n_blocks > 0 and C4g.n_blocks > 0:
            return _eigh_projector_symmetric(C1g, C4g, chi)

    C1g_dense = C1g.todense()
    C4g_dense = C4g.todense()

    rho = C1g_dense @ C1g_dense.conj().T + C4g_dense @ C4g_dense.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(rho)
    k = min(chi, len(eigvals))
    P_dense = eigvecs[:, -k:][:, ::-1]
    P_dense = jax.lax.stop_gradient(P_dense)

    # Wrap as Tensor with fused index from C1g and new chi_new bond
    fused_idx = C1g.indices[C1g.labels().index("fused")]
    chi_new_idx = TensorIndex(
        fused_idx.symmetry,
        np.zeros(k, dtype=np.int32),
        OUT,
        label="chi_new",
    )
    if isinstance(C1g, SymmetricTensor):
        return SymmetricTensor.from_dense(
            P_dense, (fused_idx, chi_new_idx), tol=float("inf")
        )
    return DenseTensor(P_dense, (fused_idx, chi_new_idx))
