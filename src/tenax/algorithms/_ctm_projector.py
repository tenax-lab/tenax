"""CTM projector computation for the Tensor protocol.

Provides ``_compute_projector_tensor`` which computes isometric projectors
from two grown corner tensors, with block-sparse support for SymmetricTensor.

Shared by both the standard CTM (``_ctm_tensor.py``) and the split CTM
(``_split_ctm_tensor.py``).

Dense fallback audit (issue #282)
----------------------------------
This module contains ``todense()`` calls in the projector computation.

**Block-sparse paths (no todense):**

- ``_eigh_projector_symmetric``: Fully block-sparse per-sector density
  matrix eigh with global eigenvalue truncation.

- ``_qr_projector_symmetric``: Fully block-sparse per-sector density
  matrix eigh (identical to eigh path; "qr" label retained for API compat).

- ``_svd_projector_symmetric``: Fully block-sparse per-sector Fishman
  two-projector SVD with global singular value truncation.

**Dense fallback paths (used during AD tracing or for DenseTensor):**

- SVD path (``projector_method="svd"``): Falls back to dense only when
  SymmetricTensor blocks contain JAX tracers (AD) or for DenseTensor.

- QR path (``projector_method="qr"``): Falls back to dense only when
  SymmetricTensor blocks contain JAX tracers (AD) or for DenseTensor.

- eigh path (``projector_method="eigh"``): Falls back to dense only when
  SymmetricTensor blocks contain JAX tracers (AD) or for DenseTensor.

During AD, projectors are evaluated with JAX tracers, forcing the dense
fallback.  However, projectors at the fixed point are ``stop_gradient``
constants, so the dense path affects only the backward step function.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_truncation_error import compute_truncation_error
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

OUT = FlowDirection.OUT


def _tensor_matrix_data(t: Tensor | jax.Array) -> jax.Array:
    """Return matrix data for dense-fallback projector math."""
    if isinstance(t, DenseTensor):
        return t._data
    if isinstance(t, SymmetricTensor):
        return t.todense()
    return t.todense() if hasattr(t, "todense") else t


def _make_chi_new_index(
    fused_idx: TensorIndex,
    k: int,
    base_charges: np.ndarray | None = None,
) -> TensorIndex:
    """Build ``chi_new`` index for dense-fallback projector outputs."""
    if base_charges is not None:
        from tenax.algorithms._ctm_utils import _derive_charges

        chi_charges = _derive_charges(base_charges, k)
    else:
        chi_charges = np.zeros(k, dtype=np.int32)
    return TensorIndex.from_charges(
        fused_idx.symmetry,
        chi_charges,
        OUT,
        label="chi_new",
    )


def _wrap_dense_projector(
    P_dense: jax.Array,
    fused_idx: TensorIndex,
    chi_new_idx: TensorIndex,
    as_symmetric: bool,
) -> Tensor:
    """Wrap dense projector matrix as DenseTensor or SymmetricTensor."""
    if as_symmetric:
        return SymmetricTensor.from_dense(
            P_dense, (fused_idx, chi_new_idx), tol=float("inf")
        )
    return DenseTensor(P_dense, (fused_idx, chi_new_idx))


def _wrap_dense_projector_pair(
    P1_dense: jax.Array,
    P2_dense: jax.Array,
    fused_idx: TensorIndex,
    chi_new_idx: TensorIndex,
    as_symmetric: bool,
) -> tuple[Tensor, Tensor]:
    """Wrap dense Fishman projector pair as DenseTensor/SymmetricTensor."""
    if as_symmetric:
        P1 = SymmetricTensor.from_dense(
            P1_dense, (fused_idx, chi_new_idx), tol=float("inf")
        )
        P2 = SymmetricTensor.from_dense(
            P2_dense, (fused_idx, chi_new_idx), tol=float("inf")
        )
        return P1, P2
    return (
        DenseTensor(P1_dense, (fused_idx, chi_new_idx)),
        DenseTensor(P2_dense, (fused_idx, chi_new_idx)),
    )


# ------------------------------------------------------------------ #
# Block-sparse projector helpers                                       #
# ------------------------------------------------------------------ #


def _build_unified_fused_idx(idx1: TensorIndex, idx2: TensorIndex) -> TensorIndex:
    """Build a fused TensorIndex covering all charge sectors from both inputs.

    For each unique charge, allocates ``max(count_in_idx1, count_in_idx2)``
    entries.  The resulting index may be larger than either input.
    """
    c1 = np.asarray(idx1.charges)
    c2 = np.asarray(idx2.charges)
    all_charges = sorted(set(int(q) for q in c1) | set(int(q) for q in c2))
    unified: list[int] = []
    for q in all_charges:
        count = max(int(np.sum(c1 == q)), int(np.sum(c2 == q)))
        unified.extend([q] * count)
    return TensorIndex.from_charges(
        idx1.symmetry,
        np.array(unified, dtype=np.int32),
        idx1.flow,
        label=idx1.label,
    )


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
    chi_new_idx = TensorIndex.from_charges(
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


def _infer_eigvec_charges(P: np.ndarray, fused_charges: np.ndarray) -> np.ndarray:
    """Infer the charge of each eigenvector column from its non-zero rows.

    For a block-diagonal rho (arising from symmetric tensors), each
    eigenvector belongs to exactly one charge sector. The charge of
    column j equals the fused charge of the row with maximum absolute
    value in that column.

    Args:
        P: Dense projector matrix ``(fused_dim, k)``.
        fused_charges: 1-D int32 array of length ``fused_dim``.

    Returns:
        1-D int32 array of length ``k`` with the inferred charge for
        each column.
    """
    k = P.shape[1]
    chi_new_charges = np.zeros(k, dtype=np.int32)
    for j in range(k):
        col = np.abs(P[:, j])
        max_row = int(np.argmax(col))
        chi_new_charges[j] = fused_charges[max_row]
    return chi_new_charges


def _reembed_fused(
    T: SymmetricTensor, target_fused_idx: TensorIndex
) -> SymmetricTensor:
    """Re-embed *T* so its ``fused`` leg uses *target_fused_idx*.

    For each charge sector *q*, if the target index has more states than
    T's current fused index, the block is zero-padded along the fused axis.
    Sectors present in *target_fused_idx* but absent from *T* produce
    no blocks (they would be all-zero anyway).

    If T's fused index already equals *target_fused_idx*, returns *T* as-is.
    """
    fused_pos = T.labels().index("fused")
    current_fused_idx = T.indices[fused_pos]

    if np.array_equal(current_fused_idx.charges, target_fused_idx.charges):
        return T

    # Precompute per-charge dimensions in both current and target
    cur_charges = np.asarray(current_fused_idx.charges)
    tgt_charges = np.asarray(target_fused_idx.charges)

    new_blocks: dict[tuple[int, ...], jax.Array] = {}
    for key, block in T.blocks.items():
        fq = int(key[fused_pos])
        cur_dim = int(np.sum(cur_charges == fq))
        tgt_dim = int(np.sum(tgt_charges == fq))
        if tgt_dim == 0:
            # Charge sector absent from target — drop the block
            continue
        elif tgt_dim == cur_dim:
            new_blocks[key] = block
        elif tgt_dim > cur_dim:
            # Pad along fused axis
            pad_width = [(0, 0)] * T.ndim
            pad_width[fused_pos] = (0, tgt_dim - cur_dim)
            new_blocks[key] = jnp.pad(block, pad_width)
        else:
            # Target is smaller — truncate
            slices = [slice(None)] * T.ndim
            slices[fused_pos] = slice(0, tgt_dim)
            new_blocks[key] = block[tuple(slices)]

    # Build new indices
    new_indices = list(T.indices)
    new_indices[fused_pos] = target_fused_idx
    new_indices = tuple(new_indices)

    obj = object.__new__(SymmetricTensor)
    obj._indices = new_indices
    obj._init_flat_buffer(new_blocks)
    return obj


def _eigh_projector_symmetric(
    C1g: SymmetricTensor,
    C4g: SymmetricTensor,
    chi: int,
    fused_idx: TensorIndex | None = None,
    base_charges: np.ndarray | None = None,
) -> tuple[SymmetricTensor, jax.Array]:
    r"""Block-sparse projector via per-sector density matrix eigh.

    For each charge sector *q* of the ``fused`` leg, accumulates
    :math:`\rho_q = M_1 M_1^\dagger + M_4 M_4^\dagger` where :math:`M_i`
    is the sector's dense block of C_ig, then eigendecomposes :math:`\rho_q`.
    Eigenvalues are merged across sectors and globally truncated to *chi*.

    When ``base_charges`` is provided, the chi budget is distributed
    per sector to match ``_derive_charges(base_charges, chi)`` — this
    prevents cascading charge-sector loss across CTM sweeps.


    The projector is constructed directly as a SymmetricTensor with
    correct per-sector charges on the ``chi_new`` index (charge = fused
    charge of the originating sector), preserving full block-sparse structure.

    Args:
        C1g: Grown corner SymmetricTensor ``(fused, col1)``.
        C4g: Grown corner SymmetricTensor ``(fused, col2)``.
        chi: Target bond dimension.
        fused_idx: Optional unified fused TensorIndex covering both
            corners' charge sectors.  When ``None``, uses C1g's fused
            index (safe when both corners have matching fused indices).
        base_charges: Bond charges from the iPEPS tensor A.  When provided,
            per-sector allocation via ``_derive_charges`` is used instead
            of purely global eigenvalue truncation.

    Returns:
        ``(P, eps_T)`` where ``P`` has labels ``(fused, chi_new)``, flows
        ``(IN, OUT)`` and ``eps_T`` is the variPEPS §2.8.2 truncation error
        ``‖λ[n_keep:]‖ / ‖λ‖`` over the merged per-sector eigenvalue spectrum
        (matches the dense eigh-path convention in
        ``_compute_projector_tensor``).
    """

    fused_pos = C1g.labels().index("fused")
    col_pos = 1 - fused_pos  # the other leg
    if fused_idx is None:
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

    # Per-sector eigh results: q -> (eigvecs, eigvals, fused_dim)
    sector_results: dict[int, tuple[jax.Array, jax.Array, int]] = {}

    # Build a map from fused charge to its dimension in the *unified* fused index.
    # Include ALL charges in fused_idx, not just those with data,
    # so seed eigenvectors can be created for absent-but-valid sectors.
    charges_arr = np.asarray(fused_idx.charges)
    charge_dim: dict[int, int] = {}
    for fq in set(int(q) for q in charges_arr):
        charge_dim[fq] = int(np.sum(charges_arr == fq))

    for fq in all_fused_charges:
        fused_dim = charge_dim.get(fq, 0)
        if fused_dim == 0:
            continue

        # Accumulate rho = sum of M @ M^dagger for both corners
        rho = jnp.zeros((fused_dim, fused_dim), dtype=C1g.dtype)

        for Cg, entries in [(C1g, c1_groups.get(fq, [])), (C4g, c4_groups.get(fq, []))]:
            # Determine this corner's own fused dimension for the sector
            cg_fused_charges = np.asarray(Cg.indices[fused_pos].charges)
            cg_fused_dim = int(np.sum(cg_fused_charges == fq))
            for _cq, block in entries:
                if fused_pos == 0:
                    M = block.reshape(cg_fused_dim, -1)
                else:
                    M = block.reshape(-1, cg_fused_dim).T
                # If this corner's sector is smaller than the unified dim,
                # pad M with zeros so it fits rho's shape.
                if cg_fused_dim < fused_dim:
                    pad_rows = fused_dim - cg_fused_dim
                    M = jnp.pad(M, ((0, pad_rows), (0, 0)))
                rho = rho + M @ M.conj().T

        rho = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = jnp.linalg.eigh(rho)
        sector_results[fq] = (eigvecs, eigvals, fused_dim)

    # Truncation: merge eigenvalues and select which eigenvectors to keep.
    all_eig_pairs: list[tuple[float, int, int]] = []  # (value, fused_charge, index)
    for fq, (_, eigvals, _) in sector_results.items():
        for i, val in enumerate(np.array(eigvals)):
            all_eig_pairs.append((float(val), fq, i))

    # Sort descending by eigenvalue, then descending by index to match
    # the dense convention of taking eigvecs[:, -k:] (highest indices first
    # among degenerate eigenvalues).
    all_eig_pairs.sort(key=lambda x: (-x[0], -x[2]))
    n_keep = min(chi, len(all_eig_pairs))

    sector_keep: dict[int, list[int]] = {}

    if base_charges is not None:
        # Per-sector allocation matching _derive_charges distribution.
        # This prevents cascading charge-sector loss across CTM sweeps
        # by ensuring every charge from A's bond is represented.
        from tenax.algorithms._ctm_utils import _derive_charges

        target_charges = _derive_charges(base_charges, n_keep)
        target_count: dict[int, int] = {}
        for q in target_charges:
            target_count[int(q)] = target_count.get(int(q), 0) + 1

        # Allocate per sector: take top eigenvalues within each sector.
        # For sectors absent from the data, create identity-like seed
        # eigenvectors to preserve charge structure (matching dense eigh
        # behavior where zero-eigenvalue sectors still get eigenvectors).
        for fq in sorted(target_count.keys()):
            n_want = target_count[fq]
            if fq in sector_results:
                eigvals_arr = np.array(sector_results[fq][1])
                n_avail = len(eigvals_arr)
                n_take = min(n_want, n_avail)
                top_indices = list(np.argsort(eigvals_arr)[-n_take:][::-1])
                sector_keep[fq] = top_indices
            else:
                # Sector absent from data — create seed eigenvectors
                # (identity columns) so the charge sector is preserved.
                fused_dim = charge_dim.get(fq, 0)
                if fused_dim > 0:
                    n_take = min(n_want, fused_dim)
                    seed_vecs = jnp.eye(fused_dim, dtype=C1g.dtype)[:, :n_take]
                    seed_vals = jnp.zeros(n_take, dtype=jnp.float64)
                    sector_results[fq] = (seed_vecs, seed_vals, fused_dim)
                    sector_keep[fq] = list(range(n_take))

        # If some target sectors had no data AND no fused_dim,
        # redistribute budget to existing sectors via global ranking
        all_eig_pairs = []
        for fq, (_, eigvals, _) in sector_results.items():
            for i, val in enumerate(np.array(eigvals)):
                all_eig_pairs.append((float(val), fq, i))
        all_eig_pairs.sort(key=lambda x: (-x[0], -x[2]))

        used = sum(len(v) for v in sector_keep.values())
        remaining = n_keep - used
        if remaining > 0:
            reserved = {(fq, idx) for fq, idxs in sector_keep.items() for idx in idxs}
            for _, fq, idx in all_eig_pairs:
                if remaining <= 0:
                    break
                if (fq, idx) not in reserved:
                    sector_keep.setdefault(fq, []).append(idx)
                    reserved.add((fq, idx))
                    remaining -= 1
    else:
        # Pure global truncation (no base_charges)
        for _, fq, idx in all_eig_pairs[:n_keep]:
            sector_keep.setdefault(fq, []).append(idx)

    # variPEPS §2.8.2 ε_T from the merged per-sector spectrum, computed
    # against the COMPLEMENT of the actually kept (fused_charge, index)
    # pairs in sector_keep. When base_charges is supplied the kept set is
    # a per-sector allocation that can differ from the global top-n_keep
    # — using the global top-n cutoff here would under-report truncation
    # whenever the forced allocation keeps a small value from a required
    # sector while discarding a larger value from another sector.
    # (Codex review on PR #430.)
    kept_pairs = {
        (int(fq), int(idx)) for fq, idxs in sector_keep.items() for idx in idxs
    }
    discarded_sq = 0.0
    total_sq = 0.0
    for fq, (_, eigvals, _) in sector_results.items():
        for idx, val in enumerate(np.array(eigvals)):
            v2 = float(val) ** 2
            total_sq += v2
            if (int(fq), idx) not in kept_pairs:
                discarded_sq += v2
    eps_T = jnp.asarray(
        np.sqrt(discarded_sq / total_sq) if total_sq > 0.0 else 0.0,
        dtype=jnp.float64,
    )

    # Build projector blocks directly with correct per-sector charges.
    # Each kept eigenvector from sector fq gets chi_new charge = fq
    # (conservation: flow_fused*fq + flow_chi_new*q_chi = 0 → q_chi = fq).
    chi_new_charges: list[int] = []
    proj_blocks: dict[tuple[int, ...], jax.Array] = {}

    for fq in sorted(sector_keep.keys()):
        keep_indices = sorted(sector_keep[fq], reverse=True)
        n_q = len(keep_indices)
        chi_new_charges.extend([fq] * n_q)

        eigvecs, _, fused_dim = sector_results[fq]
        V_q = eigvecs[:, keep_indices]  # (fused_dim, n_q)
        V_q = jax.lax.stop_gradient(V_q)
        proj_blocks[(fq, fq)] = V_q

    P = _build_symmetric_projector(proj_blocks, chi_new_charges, fused_idx, C1g.dtype)
    return P, eps_T


def _qr_projector_symmetric(
    C1g: SymmetricTensor,
    C4g: SymmetricTensor,
    chi: int,
) -> SymmetricTensor:
    r"""Block-sparse projector via per-sector density matrix eigh.

    For each charge sector *q* of the ``fused`` leg, concatenates the
    column-blocks from both grown corners to form :math:`M_q`, then
    eigendecomposes :math:`\rho_q = M_q M_q^\dagger`.

    This matches ``_eigh_projector_symmetric`` exactly, so the
    ``"qr"`` and ``"eigh"`` projector methods produce identical
    subspaces.  The ``"qr"`` label is retained for API compatibility.

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

    # Per-sector density matrix eigh
    sector_results: dict[
        int, tuple[jax.Array, jax.Array]
    ] = {}  # fq -> (eigvecs, eigvals)

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

        # Compute the full density matrix rho_q = M_q M_q^†  so that
        # the eigenvectors — including those in the null space of M_q —
        # match exactly what _eigh_projector_symmetric produces.
        #
        # The earlier QR trick (rho_small = R_q R_q†, then P_q = Q @ eigvecs)
        # only spans the column space of M_q, so when chi > rank(M_q) the
        # zero-eigenvector selections diverge from eigh's ordering.  Using
        # the same rho_q computation as _eigh_projector_symmetric guarantees
        # identical truncation at any chi.
        rho_q = M_q @ M_q.conj().T
        rho_q = 0.5 * (rho_q + rho_q.conj().T)
        eigvals, eigvecs = jnp.linalg.eigh(rho_q)

        sector_results[fq] = (eigvecs, eigvals)

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

        eigvecs_q, _ = sector_results[fq]
        V_q = eigvecs_q[:, keep_indices]  # (fused_dim, n_q)
        V_q = jax.lax.stop_gradient(V_q)
        proj_blocks[(fq, fq)] = V_q

    return _build_symmetric_projector(
        proj_blocks, chi_new_charges, fused_idx, C1g.dtype
    )


def _svd_projector_symmetric(
    C1g: SymmetricTensor,
    C4g: SymmetricTensor,
    chi: int,
    fused_idx: TensorIndex | None = None,
    base_charges: np.ndarray | None = None,
) -> tuple[SymmetricTensor, SymmetricTensor, jax.Array]:
    r"""Block-sparse two-projector Fishman SVD for SymmetricTensor.

    For each charge sector *q* of the ``fused`` leg, forms the
    cross-product :math:`M_q = C_{1g,q}^\dagger C_{4g,q}`, then
    SVD-decomposes :math:`M_q = U_q S_q V_q^\dagger`.  Two projectors
    are built:

    .. math::

        P_{1,q} = C_{4g,q} V_q S_q^{-1/2}, \quad
        P_{2,q} = C_{1g,q} U_q S_q^{-1/2}

    Singular values are merged across sectors and globally truncated
    to *chi*.

    Args:
        C1g: Grown corner SymmetricTensor ``(fused, col1)``.
        C4g: Grown corner SymmetricTensor ``(fused, col2)``.
        chi: Target bond dimension.
        fused_idx: Optional unified fused TensorIndex.
        base_charges: Bond charges for per-sector allocation.

    Returns:
        ``(P1, P2, eps_T)`` where ``P1``, ``P2`` each have labels
        ``(fused, chi_new)`` and ``eps_T`` is the variPEPS §2.8.2
        truncation error ``‖S[n_keep:]‖ / ‖S‖`` over the merged
        per-sector singular-value spectrum.
    """
    from tenax.algorithms._ad_primitives import _fix_svd_signs

    fused_pos = C1g.labels().index("fused")
    col_pos = 1 - fused_pos
    if fused_idx is None:
        fused_idx = C1g.indices[fused_pos]

    # Group blocks by fused charge
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
    charge_dim: dict[int, int] = {}
    for fq in set(int(q) for q in charges_arr):
        charge_dim[fq] = int(np.sum(charges_arr == fq))

    # Per-sector SVD results:
    #   fq -> (U_M, S_M, V_M, C1g_mat, C4g_mat)
    sector_results: dict[
        int, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    ] = {}

    for fq in all_fused_charges:
        fused_dim = charge_dim.get(fq, 0)
        if fused_dim == 0:
            continue

        # Collect column blocks for each corner
        def _collect_cols(
            Cg: SymmetricTensor,
            entries: list[tuple[int, jax.Array]],
            fd: int,
        ) -> jax.Array:
            cols: list[jax.Array] = []
            cg_fused_charges = np.asarray(Cg.indices[fused_pos].charges)
            cg_fused_dim = int(np.sum(cg_fused_charges == fq))
            for _cq, block in entries:
                if fused_pos == 0:
                    M = block.reshape(cg_fused_dim, -1)
                else:
                    M = block.reshape(-1, cg_fused_dim).T
                if cg_fused_dim < fd:
                    M = jnp.pad(M, ((0, fd - cg_fused_dim), (0, 0)))
                cols.append(M)
            if not cols:
                return jnp.zeros((fd, 0), dtype=Cg.dtype)
            return jnp.concatenate(cols, axis=1)

        C1g_mat = _collect_cols(C1g, c1_groups.get(fq, []), fused_dim)
        C4g_mat = _collect_cols(C4g, c4_groups.get(fq, []), fused_dim)

        # Cross-product: M_q = C1g_q^H @ C4g_q  (ncol1 x ncol2)
        M_q = C1g_mat.conj().T @ C4g_mat

        if M_q.size == 0:
            continue

        U_M, S_M, Vh_M = jnp.linalg.svd(M_q, full_matrices=False)
        U_M, S_M, Vh_M = _fix_svd_signs(U_M, S_M, Vh_M)
        V_M = Vh_M.conj().T

        sector_results[fq] = (U_M, S_M, V_M, C1g_mat, C4g_mat)

    # Global truncation: collect singular values across sectors
    all_sv_pairs: list[tuple[float, int, int]] = []
    for fq, (_, S_M, _, _, _) in sector_results.items():
        for i, val in enumerate(np.array(S_M)):
            all_sv_pairs.append((float(val), fq, i))

    all_sv_pairs.sort(key=lambda x: (-x[0], -x[2]))
    n_keep = min(chi, len(all_sv_pairs))

    sector_keep: dict[int, list[int]] = {}

    if base_charges is not None:
        from tenax.algorithms._ctm_utils import _derive_charges

        target_charges = _derive_charges(base_charges, n_keep)
        target_count: dict[int, int] = {}
        for q in target_charges:
            target_count[int(q)] = target_count.get(int(q), 0) + 1

        for fq in sorted(target_count.keys()):
            n_want = target_count[fq]
            if fq in sector_results:
                sv_arr = np.array(sector_results[fq][1])
                n_avail = len(sv_arr)
                n_take = min(n_want, n_avail)
                top_indices = list(np.argsort(sv_arr)[-n_take:][::-1])
                sector_keep[fq] = top_indices

        used = sum(len(v) for v in sector_keep.values())
        remaining = n_keep - used
        if remaining > 0:
            reserved = {(fq, idx) for fq, idxs in sector_keep.items() for idx in idxs}
            for _, fq, idx in all_sv_pairs:
                if remaining <= 0:
                    break
                if (fq, idx) not in reserved:
                    sector_keep.setdefault(fq, []).append(idx)
                    reserved.add((fq, idx))
                    remaining -= 1
    else:
        for _, fq, idx in all_sv_pairs[:n_keep]:
            sector_keep.setdefault(fq, []).append(idx)

    # variPEPS §2.8.2 ε_T from the merged per-sector singular-value
    # spectrum, computed against the COMPLEMENT of the actually kept
    # (fused_charge, index) pairs in sector_keep. When base_charges is
    # supplied the kept set is a per-sector allocation that can differ
    # from the global top-n_keep — using the global top-n cutoff here
    # would under-report truncation whenever the forced allocation keeps
    # a small singular value from a required sector while discarding a
    # larger one from another sector. (Codex review on PR #430.)
    kept_pairs = {
        (int(fq), int(idx)) for fq, idxs in sector_keep.items() for idx in idxs
    }
    discarded_sq = 0.0
    total_sq = 0.0
    for fq, (_, S_M, _, _, _) in sector_results.items():
        for idx, val in enumerate(np.array(S_M)):
            v2 = float(val) ** 2
            total_sq += v2
            if (int(fq), idx) not in kept_pairs:
                discarded_sq += v2
    eps_T = jnp.asarray(
        np.sqrt(discarded_sq / total_sq) if total_sq > 0.0 else 0.0,
        dtype=jnp.float64,
    )

    # Build both projector block dicts
    chi_new_charges: list[int] = []
    p1_blocks: dict[tuple[int, ...], jax.Array] = {}
    p2_blocks: dict[tuple[int, ...], jax.Array] = {}

    for fq in sorted(sector_keep.keys()):
        # SVD returns singular values in descending order, so kept indices
        # in ascending order = top singular values first. This matches the
        # dense path's ``[:k]`` convention and keeps the chi_new gauge
        # consistent across the four corner moves. (Cf. eigh path, which
        # uses descending-index because jnp.linalg.eigh returns ascending
        # eigenvalues — the opposite of SVD.)
        keep_indices = sorted(sector_keep[fq])
        n_q = len(keep_indices)
        chi_new_charges.extend([fq] * n_q)

        U_M, S_M, V_M, C1g_mat, C4g_mat = sector_results[fq]

        # Safe S^{-1/2}
        ki = jnp.array(keep_indices)
        S_k = S_M[ki]
        _cutoff = 1e-14 * (S_M[0] + 1e-30)
        _mask = S_k > _cutoff
        S_safe = jnp.where(_mask, S_k, 1.0)
        S_rsqrt = jnp.where(_mask, 1.0 / jnp.sqrt(S_safe), 0.0)

        U_k = U_M[:, ki]  # (ncol1, n_q)
        V_k = V_M[:, ki]  # (ncol2, n_q)

        # P1 = C4g @ V @ S^{-1/2}  (fused_dim, n_q)
        P1_q = C4g_mat @ (V_k * S_rsqrt[None, :])
        # P2 = C1g @ U @ S^{-1/2}  (fused_dim, n_q)
        P2_q = C1g_mat @ (U_k * S_rsqrt[None, :])

        P1_q = jax.lax.stop_gradient(P1_q)
        P2_q = jax.lax.stop_gradient(P2_q)

        p1_blocks[(fq, fq)] = P1_q
        p2_blocks[(fq, fq)] = P2_q

    P1 = _build_symmetric_projector(p1_blocks, chi_new_charges, fused_idx, C1g.dtype)
    P2 = _build_symmetric_projector(p2_blocks, chi_new_charges, fused_idx, C1g.dtype)
    return P1, P2, eps_T


# ------------------------------------------------------------------ #
# Main projector entry point                                           #
# ------------------------------------------------------------------ #


def _compute_projector_tensor(
    C1g: Tensor,
    C4g: Tensor,
    chi: int,
    projector_method: str = "svd",
    base_charges: np.ndarray | None = None,
    projector_backward: str = "auto",
) -> tuple[Tensor, Tensor, jax.Array]:
    r"""Compute projector pair (P_1, P_2) and truncation error ε_T as Tensors.

    Returns a triple ``(P_1, P_2, eps_T)`` where the projectors satisfy
    :math:`P_1^\dagger P_2 = I` (bi-orthogonality).  For ``"eigh"`` and
    ``"qr"`` methods, both projectors are identical isometries
    :math:`P_1 = P_2 = P`.  For ``"svd"`` (Fishman), they are distinct
    cross-projectors:

    .. math::
        P_1 = C_{4g} V S^{-1/2}, \quad P_2 = C_{1g} U S^{-1/2}

    where :math:`M = C_{1g}^\dagger C_{4g} = U S V^\dagger`.

    ``P_1`` is applied to the C1g side (first corner, left/top of edge)
    and ``P_2`` to the C4g side (second corner, right/bottom of edge).
    See arXiv:2502.10298 Eq. 10 for the two-projector formulation.

    Args:
        C1g: Grown corner with labels ``(fused, <col1>)``.
        C4g: Grown corner with labels ``(fused, <col2>)``.
        chi: Target bond dimension.
        projector_method: ``"svd"`` (Fishman, default), ``"eigh"``, or ``"qr"``.
            On the **dense non-tracer forward** path, ``"qr"`` is the
            reduced-corner QR isometry (issue #570, Phase 1): the dispatch
            routes through the canonical ``_reduced_qr_projector`` (concat
            both corners → QR → ``diag(R) >= 0`` gauge fix → tiny ``2χ×2χ``
            ``eigh`` → top-χ).  The AD-tracer sub-branch (regularized-SVD)
            and the ``SymmetricTensor`` non-tracer sub-branch
            (``_qr_projector_symmetric``) are unchanged here and are handled
            in later phases (Phase 2 = AD, Phase 3 = block-sparse).
        base_charges: Bond charges for per-sector allocation.

    Returns:
        ``(P_1, P_2, eps_T)`` where ``P_1``, ``P_2`` each have labels
        ``(fused, chi_new)``, flows ``(IN, OUT)`` (for eigh/qr, ``P_1 is
        P_2``), and ``eps_T`` is a scalar JAX array giving the variPEPS
        §2.8.2 truncation error ``‖S[χ:]‖ / ‖S‖``.

        **ε_T is meaningful only for the dense non-tracer SVD path.**  All
        other paths return ``jnp.asarray(0.0)`` as a placeholder:

        - *Symmetric-tensor SVD path*: block-sparse per-sector truncation;
          global ε_T extraction is out of scope for v1 (follow-up issue).
        - *Dense SVD with JAX tracers*: the AD backward (GMRES matvec) only
          sees a truncated spectrum via ``truncated_svd_ad``; ε_T is
          consumed only by the outer-loop auto-bump decision, not AD.
        - *eigh and qr paths*: truncation-error definition for these methods
          needs separate design; out of scope for the variPEPS §2.8.2
          auto-bump in v1.

    Raises:
        ValueError: If ``projector_method`` is not ``"eigh"``, ``"qr"``, or ``"svd"``.
    """
    if projector_method not in ("eigh", "qr", "svd"):
        raise ValueError(
            f"Unknown projector_method={projector_method!r}; expected 'eigh', 'qr', or 'svd'."
        )

    # --- SVD path (Fishman projector, standard CTMRG) ---
    #
    # Forms the half-corner cross-product M = C1g^T @ C4g (col1 × col2),
    # SVD(M) = U S V†, then constructs the isometric projector as:
    #
    #   P = C4g @ V @ S^{-1/2}      (fused_dim × chi)
    #
    # This is the standard CTMRG projector used by YASTN (`proj_corners`)
    # and variPEPS (Fishman method).  It's equivalent to the eigh approach
    # for the symmetric density matrix, but uses the cross-product SVD
    # which has different numerical properties.
    #
    # For improved conditioning, we first QR-factor each half-corner to
    # reduce to a small (col × col) SVD, then reconstruct.
    #
    # Reference: Fishman et al., PRB 98, 235148 (2018);
    #            YASTN proj_corners + regularize_1site_corners.
    if projector_method == "svd":
        # Block-sparse SVD for SymmetricTensor (non-tracer).
        # The dense fallback below is intended only for DenseTensor inputs
        # and the AD-traced backward path; for any non-traced SymmetricTensor
        # pair, _svd_projector_symmetric handles per-sector SVD correctly,
        # *including* the n_blocks==0 case (returns properly-shaped zero
        # projectors via the M_q.size==0 skip). Earlier this condition was
        # ``C1g.n_blocks > 0 and C4g.n_blocks > 0``, which dropped into the
        # dense fallback whenever one corner was zero-truncated. Then
        # ``C1g.todense() @ C4g.todense()`` could fail shape compatibility
        # because the fused TensorIndex charges (and therefore dense sizes)
        # differ between the two corners — producing the ``dot_general
        # (24,) vs (20,)`` shape error reported in issue #408.
        if isinstance(C1g, SymmetricTensor) and isinstance(C4g, SymmetricTensor):
            has_tracers = isinstance(C1g._data, jax.core.Tracer) or isinstance(
                C4g._data, jax.core.Tracer
            )
            if not has_tracers:
                fused_pos_chk = C1g.labels().index("fused")
                c1_charges = C1g.indices[fused_pos_chk].charges
                c4_charges = C4g.indices[fused_pos_chk].charges
                if np.array_equal(c1_charges, c4_charges):
                    P_1, P_2, eps_T = _svd_projector_symmetric(
                        C1g, C4g, chi, base_charges=base_charges
                    )
                    return P_1, P_2, eps_T
                # Mismatched fused charges: build unified fused index
                unified_fused_idx = _build_unified_fused_idx(
                    C1g.indices[fused_pos_chk], C4g.indices[fused_pos_chk]
                )
                C1g_re = _reembed_fused(C1g, unified_fused_idx)
                C4g_re = _reembed_fused(C4g, unified_fused_idx)
                P_1, P_2, eps_T = _svd_projector_symmetric(
                    C1g_re,
                    C4g_re,
                    chi,
                    fused_idx=unified_fused_idx,
                    base_charges=base_charges,
                )
                return P_1, P_2, eps_T

        # Dense fallback for SVD projector.  Reached in two cases:
        #   1. DenseTensor inputs (no block-sparse structure).
        #   2. SymmetricTensor blocks contain JAX tracers (AD backward).
        #
        # Design decision (Task 2.2): We intentionally keep the dense
        # fallback for the AD-traced path rather than implementing a
        # block-sparse AD-traced SVD.  Rationale:
        #   - The forward CTM sweep uses a Python loop (not JIT'd), so
        #     _svd_projector_symmetric handles it block-sparse already.
        #   - The backward pass (JIT'd GMRES via _jit_apply_Jt /
        #     _jit_gmres_solve) traces through the projector; JAX
        #     tracing requires static shapes per sector (k_q must be a
        #     Python int), making per-sector truncated_svd_ad complex.
        #   - The backward runs the projector only once per GMRES matvec,
        #     so the performance impact of densifying is small compared
        #     to the forward sweep which runs many CTM iterations.
        #   - The dense path is well-tested and numerically validated.
        C1g_dense = _tensor_matrix_data(C1g)
        C4g_dense = _tensor_matrix_data(C4g)

        _has_tracers = isinstance(C1g_dense, jax.core.Tracer) or isinstance(
            C4g_dense, jax.core.Tracer
        )

        # Direct cross-product: M = C1g^H @ C4g  (col1 × col2)
        # No QR pre-factoring — QR backward is unstable for rank-deficient matrices.
        M = C1g_dense.conj().T @ C4g_dense

        if _has_tracers:
            from tenax.algorithms._ad_primitives import truncated_svd_ad

            U_M, S_M, Vh_M = truncated_svd_ad(M, chi)
            # Tracer path: eps_T is consumed only by outer-loop auto-bump
            # (not AD); return placeholder so backward stays clean.
            _eps_T = jnp.asarray(0.0)
        else:
            U_M_full, S_full, Vh_full = jnp.linalg.svd(M, full_matrices=False)
            k = min(chi, S_full.shape[0])
            from tenax.algorithms._ad_primitives import _fix_svd_signs

            U_M_full, S_full, Vh_full = _fix_svd_signs(U_M_full, S_full, Vh_full)
            # Compute ε_T from the full spectrum before truncation.
            _eps_T = compute_truncation_error(S_full, k)
            S_M = S_full[:k]
            U_M = U_M_full[:, :k]
            Vh_M = Vh_full[:k, :]

        k = S_M.shape[0]
        V_M = Vh_M.conj().T  # (col2, k)

        # S^{-1/2} weighting: differentiable so the optimizer sees the
        # full gradient through the Fishman projector (VariPEPS-like).
        # Sanitize input before division so JAX backward never evaluates
        # 1/sqrt(0) (jnp.where evaluates both branches during AD).
        _cutoff = 1e-14 * (S_M[0] + 1e-30)
        _mask = S_M > _cutoff
        S_safe = jnp.where(_mask, S_M, 1.0)
        S_rsqrt = jnp.where(_mask, 1.0 / jnp.sqrt(S_safe), 0.0)

        # Two-projector Fishman (arXiv:2502.10298 Eq. 10):
        #   P_1 = C4g @ V @ S^{-1/2}  (applied to C1g side)
        #   P_2 = C1g @ U @ S^{-1/2}  (applied to C4g side)
        # Bi-orthogonality: P_1† @ P_2 = I
        P1_dense = C4g_dense @ (V_M * S_rsqrt[None, :])  # (fused_dim, k)
        P2_dense = C1g_dense @ (U_M * S_rsqrt[None, :])  # (fused_dim, k)

        fused_pos = C1g.labels().index("fused")
        fused_idx = C1g.indices[fused_pos]

        chi_new_idx = _make_chi_new_index(fused_idx, k, base_charges)
        P_1, P_2 = _wrap_dense_projector_pair(
            P1_dense,
            P2_dense,
            fused_idx,
            chi_new_idx,
            as_symmetric=isinstance(C1g, SymmetricTensor),
        )
        return P_1, P_2, _eps_T

    # --- QR path ---
    #
    # Sub-branch dispatch:
    #   (a) SymmetricTensor non-tracer  -> _qr_projector_symmetric
    #       (block-sparse; Phase 3 will replace this with a block-sparse
    #       reduced-corner QR).
    #   (b) Dense / SymmetricTensor-with-tracers (AD backward) -> dense
    #       regularized-SVD path (Phase 2 handles AD).
    #   (c) Dense non-tracer FORWARD -> the reduced-corner QR isometry
    #       (issue #570, Phase 1).  Consolidated to call the canonical,
    #       energy-validated _reduced_qr_projector so there is exactly one
    #       implementation of the Candidate-C construction.
    if projector_method == "qr":
        # Block-sparse QR for SymmetricTensor (non-tracer) -- sub-branch (a).
        if isinstance(C1g, SymmetricTensor) and isinstance(C4g, SymmetricTensor):
            has_tracers = isinstance(C1g._data, jax.core.Tracer) or isinstance(
                C4g._data, jax.core.Tracer
            )
            if not has_tracers and C1g.n_blocks > 0 and C4g.n_blocks > 0:
                P = _qr_projector_symmetric(C1g, C4g, chi)
                return P, P, jnp.asarray(0.0)

        _has_tracers = isinstance(
            _tensor_matrix_data(C1g), jax.core.Tracer
        ) or isinstance(_tensor_matrix_data(C4g), jax.core.Tracer)

        if not _has_tracers:
            # Sub-branch (c): dense non-tracer forward.  The reduced-corner
            # QR isometry (concat -> QR -> gauge fix -> tiny 2χ×2χ eigh ->
            # top-χ -> P = Q @ V).  Routed through the canonical function so
            # the dispatch runs exactly the energy-validated construction.
            P = _reduced_qr_projector(C1g, C4g, chi, base_charges=base_charges)
            return P, P, jnp.asarray(0.0)

        # Sub-branch (b): dense QR fallback for the AD-traced path
        # (blocks contain JAX tracers, or DenseTensor tracer inputs).
        # Differentiable via regularized SVD.  Phase 2 handles AD; leave
        # this byte-identical.  (issue #282)
        C1g_dense = _tensor_matrix_data(C1g)
        C4g_dense = _tensor_matrix_data(C4g)

        M = jnp.concatenate([C1g_dense, C4g_dense], axis=1)
        Q, R = jnp.linalg.qr(M)
        # QR sign-fix: force diag(R) >= 0 so the factorization is unique.
        # Without this, JAX's QR makes a sign choice that depends on the
        # specific values in M, and tiny perturbations of M can flip the
        # sign — making the projector forward non-smooth and the AD
        # gradient correspond to the wrong branch.  The product Q @ U
        # below is mathematically invariant under simultaneous sign flips
        # of (Q, R), so applying this fix has no effect on the forward
        # output but makes the autodiff gradient consistent across
        # arbitrarily small perturbations of M.
        # For complex matrices, `diag_R >= 0` is undefined; rotate by the
        # phase of diag(R) so diag(R) becomes real and non-negative. This
        # reduces to the ±1 sign convention in the real case. When a
        # diagonal entry is exactly zero the gauge is unconstrained, so
        # use phase=1 (leave the column/row untouched) rather than
        # phase=0 which would zero out that column of Q and row of R.
        Q, R = _gauge_fix_qr_dense(Q, R)
        from tenax.algorithms._ad_primitives import regularized_svd

        U_R, S_R, _Vh_R = regularized_svd(R)
        k = min(chi, len(S_R))
        P_dense = Q @ U_R[:, :k]

        fused_idx = C1g.indices[C1g.labels().index("fused")]
        chi_new_idx = _make_chi_new_index(fused_idx, k, base_charges)
        P = _wrap_dense_projector(
            P_dense,
            fused_idx,
            chi_new_idx,
            as_symmetric=isinstance(C1g, SymmetricTensor),
        )
        return P, P, jnp.asarray(0.0)

    # --- eigh path ---
    # Use block-sparse path for SymmetricTensor unless blocks contain
    # JAX tracers (during AD), in which case fall back to dense path
    # since eigenvalue sorting requires concrete values.
    if isinstance(C1g, SymmetricTensor) and isinstance(C4g, SymmetricTensor):
        has_tracers = isinstance(C1g._data, jax.core.Tracer) or isinstance(
            C4g._data, jax.core.Tracer
        )
        if not has_tracers and (C1g.n_blocks > 0 or C4g.n_blocks > 0):
            fused_pos = C1g.labels().index("fused")
            c1_charges = C1g.indices[fused_pos].charges
            c4_charges = C4g.indices[fused_pos].charges
            if np.array_equal(c1_charges, c4_charges):
                P, eps_T = _eigh_projector_symmetric(
                    C1g, C4g, chi, base_charges=base_charges
                )
                return P, P, eps_T
            # Mismatched fused charges (e.g. split CTM with different
            # D-leg flows): build a unified fused index covering both
            # corners' charge sectors, re-embed, and use block-sparse eigh.
            unified_fused_idx = _build_unified_fused_idx(
                C1g.indices[fused_pos], C4g.indices[fused_pos]
            )
            C1g_re = _reembed_fused(C1g, unified_fused_idx)
            C4g_re = _reembed_fused(C4g, unified_fused_idx)
            P, eps_T = _eigh_projector_symmetric(
                C1g_re,
                C4g_re,
                chi,
                fused_idx=unified_fused_idx,
                base_charges=base_charges,
            )
            return P, P, eps_T

    # Dense eigh fallback — only reached when SymmetricTensor blocks contain
    # JAX tracers (AD backward) or for DenseTensor inputs.  Block-sparse
    # _eigh_projector_symmetric handles the non-tracer case above.
    # Uses regularized_eigh when AD-traced for stable backward.
    # (issue #282)
    fused_pos = C1g.labels().index("fused")
    fused_idx = C1g.indices[fused_pos]

    C1g_dense = _tensor_matrix_data(C1g)
    C4g_dense = _tensor_matrix_data(C4g)
    _has_tracers = isinstance(C1g_dense, jax.core.Tracer) or isinstance(
        C4g_dense, jax.core.Tracer
    )

    rho = C1g_dense @ C1g_dense.conj().T + C4g_dense @ C4g_dense.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    _eps_T_eigh = jnp.asarray(0.0)
    if _has_tracers:
        if projector_backward == "lorentzian":
            # Lorentzian-regularized truncated-eigh backward (Francuz et al.
            # 2023, Phys. Rev. Research 7, 013237).  Using the kernel as
            # ``truncated_eigh_regularized(-rho, k)`` turns "smallest of -ρ"
            # into "largest of ρ" and returns the top-k eigenvectors in
            # descending eigenvalue order — matching the baseline column
            # order produced by ``eigvecs[:, -k:][:, ::-1]``.
            from tenax.algorithms._lorentzian_eigh import (
                truncated_eigh_regularized,
            )

            k = min(chi, rho.shape[0])
            _, P_dense = truncated_eigh_regularized(-rho, k)
        else:
            # projector_backward in {"auto", "standard"} — fall through to the
            # existing regularized_eigh backward.  Task 8 will resolve "auto"
            # to "standard" or "lorentzian" at the optimizer layer before it
            # reaches the projector; for Task 7, both map to "standard" here.
            from tenax.algorithms._ad_primitives import regularized_eigh

            eigvals, eigvecs = regularized_eigh(rho)
            k = min(chi, len(eigvals))
            P_dense = eigvecs[:, -k:][:, ::-1]
    else:
        eigvals, eigvecs = jnp.linalg.eigh(rho)
        k = min(chi, len(eigvals))
        P_dense = eigvecs[:, -k:][:, ::-1]
        P_dense = jax.lax.stop_gradient(P_dense)
        # Compute ε_T from the eigenvalue spectrum: the discarded eigenvalues
        # represent information that chi dimensions cannot capture.  This is
        # the variPEPS §2.8.2 truncation-error indicator (dense non-tracer path).
        _eps_T_eigh = compute_truncation_error(jnp.flip(jnp.abs(eigvals)), k)

    chi_new_idx = _make_chi_new_index(fused_idx, k, base_charges)
    P = _wrap_dense_projector(
        P_dense,
        fused_idx,
        chi_new_idx,
        as_symmetric=isinstance(C1g, SymmetricTensor),
    )
    return P, P, _eps_T_eigh


def _gauge_fix_qr_dense(Q, R):
    """Rephase QR factors so diag(R) is real-nonnegative (zero-diagonal -> phase 1),
    preserving Q @ R.  Makes the QR gauge continuous across CTM iterations (jnp.linalg.qr's
    sign choice is input-dependent; tiny perturbations otherwise flip Q's column signs)."""
    diag_R = jnp.diag(R)
    abs_diag = jnp.abs(diag_R)
    phase = jnp.where(
        abs_diag > 0, diag_R / jnp.where(abs_diag > 0, abs_diag, 1.0), jnp.ones_like(diag_R)
    ).astype(R.dtype)
    Q = Q * phase[None, :]
    R = R * jnp.conj(phase)[:, None]
    return Q, R


def _reduced_qr_projector(
    C1g: Tensor,
    C4g: Tensor,
    chi: int,
    base_charges: np.ndarray | None = None,
) -> Tensor:
    """Dense reduced-corner QR-CTMRG projector (issue #570, Phase 1).

    Productionizes the spike-validated "Candidate C" construction of the
    reduced-corner QR-CTMRG scheme (Yang/Zhang/Corboz, arXiv:2505.00494): a
    faithful, truncation-free projector that performs **no** large ``chi D^2``
    SVD.  The only truncating decomposition is a tiny ``2*chi x 2*chi``
    Hermitian eigendecomposition.

    Construction (operating on the dense ``(fused | cut)`` corner matrices,
    each shape ``(fused, chi)`` where ``fused = chi D^2`` and the cut leg
    ``t1_r`` / ``t3_l`` is already ``chi`` — the reduced corner):

    1. Concatenate both corners along the cut axis -> ``M : (fused, 2*chi)``.
       Using BOTH corners keeps the projector correct off the C4v fixed point.
    2. Unpivoted thin QR -> ``Q : (fused, 2*chi)`` with a ``diag(R) >= 0``
       gauge fix (phase = d/|d|, 1 if |d| == 0) for forward smoothness and
       uniqueness.
    3. Build the small Hermitian ``R R^dagger`` and ``eigh`` it; keep the
       top-``chi`` eigenvectors ``V``; the projector is ``P = Q @ V``.

    This both-corners form provably spans the eigh density-matrix subspace via
    ``R R^dagger``.  Candidate A (pure ``QR(C1g)`` alone) is the C4v
    single-corner special case; the multisite / projector-pair generalization
    is Phase 2.

    Phase 1 is dense + forward only.  Returns a single ``DenseTensor`` ``P``
    with labels ``(fused, chi_new)`` and flows ``(IN, OUT)`` — matching the
    eigh/qr single-isometry contract.  The dispatch (a later task) returns
    ``(P, P, jnp.asarray(0.0))``.
    """
    C1g_dense = _tensor_matrix_data(C1g)
    C4g_dense = _tensor_matrix_data(C4g)

    M = jnp.concatenate([C1g_dense, C4g_dense], axis=1)  # (fused, 2*chi)
    Q, R = jnp.linalg.qr(M)
    # diag(R) >= 0 gauge fix (uniqueness + forward smoothness).  For a zero
    # diagonal entry the gauge is unconstrained, so phase = 1 (leave that
    # column of Q untouched) rather than 0 (which would zero the column).
    # Mirror of the dense "qr"/"eigh" branch sign-fix above.
    Q, R = _gauge_fix_qr_dense(Q, R)

    # Tiny 2*chi x 2*chi Hermitian eigendecomposition selects the chi dominant
    # directions (top-chi eigenvectors, descending).  No large SVD anywhere.
    rho_small = R @ R.conj().T
    rho_small = 0.5 * (rho_small + rho_small.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(rho_small)
    k = min(chi, len(eigvals))
    V = eigvecs[:, -k:][:, ::-1]
    P_dense = Q @ V  # (fused, k)

    fused_idx = C1g.indices[C1g.labels().index("fused")]
    chi_new_idx = _make_chi_new_index(fused_idx, k, base_charges)
    return _wrap_dense_projector(
        P_dense,
        fused_idx,
        chi_new_idx,
        as_symmetric=isinstance(C1g, SymmetricTensor),
    )
