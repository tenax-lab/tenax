"""Split CTM with Tensor protocol — move helpers and directional moves."""

from __future__ import annotations

__all__ = [
    "_FORCE_CLOSED_EDGE",
    "_apply_projector",
    "_ensure_corner_flows",
    "_ensure_edge_flows",
    "_ensure_tensor_flows",
    "_doublelayer_grown_corner",
    "_factorize_projector",
    "_fused_charge_permutation",
    "_grow_and_project_bounded",
    "_grow_and_project_bounded_lr",
    "_grow_and_project_edge",
    "_grow_edge_halves",
    "_grow_edge_no_double_layer",
    "_precombine_projector_pair",
    "_project_grown_edge_tensor",
    "_project_grown_edge_tensor_lr",
    "_reembed_target_for_projector",
    "_select_bond_entries",
    "_split_ctm_move_bottom",
    "_split_ctm_move_left",
    "_split_ctm_move_right",
    "_split_ctm_move_top",
    "_svd_split_edge_tensor",
    "_truncate_svd_per_sector",
]

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_projector import _compute_projector_tensor, _reembed_fused
from tenax.algorithms._ctm_utils import _CORNER_SPECS, _derive_charges
from tenax.algorithms._split_ctm_tensor_init import (
    _EDGE_BRA_SPECS,
    _EDGE_KET_SPECS,
    SplitCTMTensorEnv,
)
from tenax.algorithms._tensor_utils import (
    absorb_sqrt_singular_values,
    fuse_indices,
    max_abs_normalize,
    split_index,
)
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor
from tenax.linalg import svd as tensor_svd

# When True, the four split moves grow the FULL closed chi^2*D^6 double-layer
# edge and project it with :func:`_project_grown_edge_tensor_lr` (the Task-4
# correctness-first path).  When False (default), they route through the
# memory-bounded :func:`_grow_and_project_bounded_lr` (chi^2*D^4) path, which
# reproduces the closed result to machine precision.  Tests flip this flag to
# prove bounded == closed; production always uses the bounded default.
_FORCE_CLOSED_EDGE = False

# ------------------------------------------------------------------ #
# Double-layer corner-pair projector + factorization helpers           #
# ------------------------------------------------------------------ #


def _doublelayer_grown_corner(C, T_ket, T_bra, c_relabel, ket_I, bra_I, fuse_labels):
    """Grow a corner with BOTH ket and bra edges, joined over the interlayer.

    Mirrors the fused move's grown corner but keeps it as a double layer:
    fused leg = (env, u_ket, u_bra) of dim chi*D^2; the remaining leg is the
    next-corner env bond. Returns (C_grown_fused, remaining_label).
    """
    C_r = C.relabel(*c_relabel)  # align bond label to the ket edge
    Cg = contract(C_r, T_ket)  # (env, u_ket, ket_I)
    Cg = contract(Cg.relabel(ket_I, bra_I), T_bra)  # (env, u_ket, u_bra, bra_r)
    labels = Cg.labels()
    # fuse the three to-truncate legs into 'fused' (env first, then u_ket, u_bra)
    Cg = fuse_indices(
        Cg,
        labels.index(fuse_labels[0]),
        labels.index(fuse_labels[1]),
        "fused",
        FlowDirection.IN,
    )
    labels = Cg.labels()
    Cg = fuse_indices(
        Cg,
        labels.index("fused"),
        labels.index(fuse_labels[2]),
        "fused",
        FlowDirection.IN,
    )
    remaining = [lbl for lbl in Cg.labels() if lbl != "fused"][0]
    return Cg, remaining


def _unfuse_projector_fused(P, env_dim, D, env_label, ketD_label, braD_label):
    """Reshape a corner projector's single ``fused`` leg into three legs.

    The corner projector ``P`` from :func:`_compute_projector_tensor` has legs
    ``(fused, chi_new)`` where the ``fused`` leg was built by
    :func:`_doublelayer_grown_corner` fusing ``(env, u_ket, u_bra)`` in two IN
    steps — ``fuse(env, u_ket)`` then ``fuse(., u_bra)`` — so the fused
    dimension is laid out row-major as ``env`` (slowest), ``u_ket``,
    ``u_bra`` (fastest), of size ``env_dim * D * D``.

    The unified projector index built by ``_build_unified_fused_idx`` carries no
    ``fuse_info`` (it is a fresh ``from_charges`` index), so ``split_index``
    cannot invert the fuse.  For the dense bounded path we reshape the fused
    axis directly into ``(env_label, ketD_label, braD_label)`` with U(1)-trivial
    (zero-charge) indices, matching the rest of the dense split-CTM path.

    Returns a 4-leg DenseTensor ``(env_label, ketD_label, braD_label,
    chi_new)``.  DenseTensor only.
    """
    fused_pos = P.labels().index("fused")
    chi_pos = P.labels().index("chi_new")
    # Bring to (fused, chi_new) order then reshape the fused axis.
    if fused_pos != 0:
        P = P.transpose((fused_pos, chi_pos))
    data = P.todense()  # (fused_dim, chi_new)
    chi_new_dim = data.shape[1]
    data = data.reshape(env_dim, D, D, chi_new_dim)

    sym = P.indices[chi_pos if fused_pos == 0 else 0].symmetry
    z_env = np.zeros(env_dim, dtype=np.int32)
    z_D = np.zeros(D, dtype=np.int32)
    chi_new_idx = P.indices[1]  # chi_new index (now at axis 1 after transpose)
    new_indices = (
        TensorIndex.from_charges(sym, z_env, FlowDirection.IN, label=env_label),
        TensorIndex.from_charges(sym, z_D, FlowDirection.IN, label=ketD_label),
        TensorIndex.from_charges(sym, z_D, FlowDirection.IN, label=braD_label),
        chi_new_idx,
    )
    return DenseTensor(data, new_indices)


def _factorize_projector(P, env_label, ketD_label, braD_label, chi_label):
    """Factorize a projector P[(env,ketD),(braD,chi)] -> P_first . P_second.

    SVD across (env, ketD) | (braD, chi); factorization bond m <= env*ketD.
    No truncation (exact rewrite). Returns (P_first, P_second, m).
    P_first: (env, ketD, _fac), P_second: (_fac, braD, chi).
    """
    U, s, Vh, _ = tensor_svd(
        P,
        left_labels=[env_label, ketD_label],
        right_labels=[braD_label, chi_label],
        new_bond_label="_fac",
        max_singular_values=None,
    )
    P_first, P_second = absorb_sqrt_singular_values(U, s, Vh, "_fac")
    m = s.shape[0]
    return P_first, P_second, m


# ------------------------------------------------------------------ #
# No-double-layer edge growth                                          #
# ------------------------------------------------------------------ #

_VIRTUAL_LEGS = ("u", "d", "l", "r")


def _grow_edge_halves(
    T_ket: Tensor,
    T_bra: Tensor,
    A: Tensor,
    A_bar: Tensor,
    contracted_leg: str,
    ket_I_label: str,
    bra_I_label: str,
) -> tuple[Tensor, Tensor]:
    """Build the two open half-edges of a T-edge without joining them.

    Each half is one boundary T contracted with its copy of ``A`` (ket) or
    ``A.bar()`` (bra).  Both halves share the ``_I`` (interlayer) and ``phys``
    labels, so a later ``contract(ket_half, bra_half)`` traces them.

    Returns ``(ket_half, bra_half)``, each a 6-leg Tensor (``chi``, ``_I``,
    and four ``D``/``phys`` legs).  Peak per half is ``chi * chi_I * D^3 * d``.
    """
    ket_D_label = f"{contracted_leg}_ket"
    bra_D_label = f"{contracted_leg}_bra"

    # --- Ket side ---
    A_ket = A.relabel(contracted_leg, ket_D_label)
    ket_half = contract(T_ket, A_ket).relabel(ket_I_label, "_I")

    # --- Bra side: relabel virtual legs to uppercase ---
    bra_mapping: dict[str, str] = {contracted_leg: bra_D_label}
    for v in _VIRTUAL_LEGS:
        if v != contracted_leg:
            bra_mapping[v] = v.upper()
    A_bra = A_bar.relabels(bra_mapping)
    bra_half = contract(T_bra, A_bra).relabel(bra_I_label, "_I")

    return ket_half, bra_half


def _grow_edge_no_double_layer(
    T_ket: Tensor,
    T_bra: Tensor,
    A: Tensor,
    A_bar: Tensor,
    contracted_leg: str,
    ket_I_label: str,
    bra_I_label: str,
    output_labels: tuple[str, ...],
) -> Tensor:
    """Grow a T-edge by contracting ket/bra layers separately.

    Instead of building a closed double-layer tensor, this contracts each
    half-edge with its copy of A (ket) and A.bar() (bra), then traces the
    physical and interlayer indices via label-based contraction.

    Returns an 8-leg Tensor with labels matching *output_labels*.

    NOTE: this joins the two half-edges into a single ``chi^2 * D^6`` tensor.
    The bounded path (:func:`_grow_and_project_edge`) avoids forming it; this
    closed form is retained for the SymmetricTensor (fermionic) projection
    path, whose order-dependent Koszul signs the bounded reorder does not yet
    reproduce (issue #641 / #463 Phase 2-4).
    """
    ket_half, bra_half = _grow_edge_halves(
        T_ket, T_bra, A, A_bar, contracted_leg, ket_I_label, bra_I_label
    )
    return contract(ket_half, bra_half, output_labels=output_labels)


# ------------------------------------------------------------------ #
# SVD helper                                                            #
# ------------------------------------------------------------------ #


def _svd_split_edge_tensor(
    T: Tensor,
    left_labels: list[str],
    right_labels: list[str],
    chi_I: int,
    ket_relabels: dict[str, str],
    bra_relabels: dict[str, str],
    base_charges: np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """SVD-split a 4-leg projected edge into ket/bra halves.

    Transposes *T* so left labels come first and right labels last
    (required for correct block-sparse SVD), then splits via SVD,
    absorbs sqrt(s) into both factors, relabels and normalizes.

    When *base_charges* is provided and the input is a SymmetricTensor,
    uses per-sector truncation (via ``_derive_charges``) instead of
    global truncation to prevent charge-sector loss.
    """
    # Ensure axes are in (left..., right...) order for block-sparse SVD
    labels = T.labels()
    perm = tuple(labels.index(lbl) for lbl in left_labels + right_labels)
    if perm != tuple(range(len(labels))):
        T = T.transpose(perm)

    if isinstance(T, SymmetricTensor) and base_charges is not None:
        # Per-sector SVD truncation: full SVD then sector-aware selection
        U_t, s, Vh_t, _s_full = tensor_svd(
            T,
            left_labels=left_labels,
            right_labels=right_labels,
            new_bond_label="_svd_bond",
            max_singular_values=None,  # get all singular values
        )
        U_t, s, Vh_t = _truncate_svd_per_sector(
            U_t, s, Vh_t, "_svd_bond", chi_I, base_charges
        )
    elif isinstance(T, DenseTensor):
        # Bare jnp.linalg.svd's adjoint NaN's on rank-deficient inputs
        # (split-CTM at small D after PR #399 flipped the projector default
        # to "svd"). Route through truncated_svd_symmetric_ad, whose backward
        # is the Lorentzian-regularized + rank-aware kernel from _ad_primitives.
        # TODO: add SymmetricTensor block-sparse regularized SVD as a follow-up
        # so the SymmetricTensor branches below also get a finite adjoint on
        # rank-deficient blocks.
        from tenax.algorithms._ad_primitives import truncated_svd_symmetric_ad

        U_t, s, Vh_t = truncated_svd_symmetric_ad(
            T,
            left_labels=left_labels,
            right_labels=right_labels,
            chi=chi_I,
            new_bond_label="_svd_bond",
        )
    else:
        U_t, s, Vh_t, _s_full = tensor_svd(
            T,
            left_labels=left_labels,
            right_labels=right_labels,
            new_bond_label="_svd_bond",
            max_singular_values=chi_I,
        )
    T_ket, T_bra = absorb_sqrt_singular_values(U_t, s, Vh_t, "_svd_bond")
    T_ket = T_ket.relabels(ket_relabels)
    T_bra = T_bra.relabels(bra_relabels)
    T_ket, _ = max_abs_normalize(T_ket)
    T_bra, _ = max_abs_normalize(T_bra)
    return T_ket, T_bra


def _truncate_svd_per_sector(
    U_t: SymmetricTensor,
    s: jax.Array,
    Vh_t: SymmetricTensor,
    bond_label: str,
    chi_I: int,
    base_charges: np.ndarray,
) -> tuple[SymmetricTensor, jax.Array, SymmetricTensor]:
    """Per-sector SVD truncation matching ``_derive_charges`` allocation.

    Selects singular values so that each charge sector from *base_charges*
    gets its fair share of the ``chi_I`` budget, preventing cascading
    charge-sector loss across CTM sweeps.
    """
    target = _derive_charges(base_charges, chi_I)
    target_count: dict[int, int] = {}
    for q in target:
        target_count[int(q)] = target_count.get(int(q), 0) + 1

    bond_pos_U = U_t.labels().index(bond_label)
    current_charges = np.asarray(U_t.indices[bond_pos_U].charges)

    # Walk through globally-sorted singular values, allocating per sector.
    # s is in descending order; current_charges[i] is the sector for s[i].
    keep_mask = np.zeros(len(s), dtype=bool)
    sector_allocated: dict[int, int] = {}
    for i, q in enumerate(current_charges):
        q_int = int(q)
        allocated = sector_allocated.get(q_int, 0)
        if allocated < target_count.get(q_int, 0):
            keep_mask[i] = True
            sector_allocated[q_int] = allocated + 1

    # If some target sectors had no data, fill budget from remaining
    total_kept = int(np.sum(keep_mask))
    if total_kept < chi_I:
        for i in range(len(s)):
            if total_kept >= chi_I:
                break
            if not keep_mask[i]:
                keep_mask[i] = True
                total_kept += 1

    keep_indices = np.where(keep_mask)[0]
    s_new = s[keep_indices]
    new_charges = current_charges[keep_mask]

    # Build per-sector entry mapping: which entries within each sector to keep
    sector_entry_map: dict[int, list[int]] = {}
    cur_sector_idx: dict[int, int] = {}
    for i, q in enumerate(current_charges):
        q_int = int(q)
        idx_in_sector = cur_sector_idx.get(q_int, 0)
        cur_sector_idx[q_int] = idx_in_sector + 1
        if keep_mask[i]:
            sector_entry_map.setdefault(q_int, []).append(idx_in_sector)

    sym = U_t.indices[bond_pos_U].symmetry
    new_bond_out = TensorIndex.from_charges(
        sym,
        np.asarray(new_charges, dtype=np.int32),
        FlowDirection.OUT,
        label=bond_label,
    )
    new_bond_in = TensorIndex.from_charges(
        sym,
        np.asarray(new_charges, dtype=np.int32),
        FlowDirection.IN,
        label=bond_label,
    )

    # Rebuild U (bond is last axis)
    U_new = _select_bond_entries(U_t, bond_pos_U, sector_entry_map, new_bond_out)
    # Rebuild Vh (bond is first axis)
    bond_pos_Vh = Vh_t.labels().index(bond_label)
    Vh_new = _select_bond_entries(Vh_t, bond_pos_Vh, sector_entry_map, new_bond_in)

    return U_new, s_new, Vh_new


def _select_bond_entries(
    T: SymmetricTensor,
    bond_pos: int,
    sector_entry_map: dict[int, list[int]],
    new_bond_idx: TensorIndex,
) -> SymmetricTensor:
    """Select specific entries from a bond axis of a SymmetricTensor."""
    new_blocks: dict[tuple[int, ...], jax.Array] = {}
    for key, block in T.blocks.items():
        q_bond = int(key[bond_pos])
        if q_bond not in sector_entry_map:
            continue
        kept = sector_entry_map[q_bond]
        idx = [slice(None)] * T.ndim
        idx[bond_pos] = jnp.array(kept)
        new_blocks[key] = block[tuple(idx)]

    new_indices = list(T.indices)
    new_indices[bond_pos] = new_bond_idx
    obj = object.__new__(SymmetricTensor)
    obj._indices = tuple(new_indices)
    obj._init_flat_buffer(new_blocks)
    return obj


def _fused_charge_permutation(
    source_charges: np.ndarray, target_charges: np.ndarray
) -> list[int] | None:
    """Compute a permutation mapping *source_charges* order to *target_charges*.

    Returns a list ``perm`` such that ``source_charges[perm[i]] == target_charges[i]``
    for every *i*, or ``None`` if no such mapping exists (different charge sets).

    This is needed when the projector P and the tensor being projected have
    different fused charge orderings (due to different leg flow conventions),
    so we must reorder the tensor's fused axis before dense matrix multiply.
    """
    if len(source_charges) != len(target_charges):
        return None

    # Build a map: charge -> list of indices in source
    source_map: dict[int, list[int]] = {}
    for i, q in enumerate(source_charges):
        source_map.setdefault(int(q), []).append(i)

    perm: list[int] = []
    used: dict[int, int] = {}  # charge -> next index to use from source_map
    for q in target_charges:
        q_int = int(q)
        if q_int not in source_map:
            return None
        pos = used.get(q_int, 0)
        if pos >= len(source_map[q_int]):
            return None
        perm.append(source_map[q_int][pos])
        used[q_int] = pos + 1

    return perm


def _ensure_tensor_flows(
    T: Tensor, expected_flows: tuple[FlowDirection, ...]
) -> Tensor:
    """No-op: let tensors keep whatever flows the SVD / projection gave them.

    Previously this rebuilt the tensor via from_dense with corrected flows,
    but that destroys charge sectors.  Since all contractions are
    label-based, the flows propagate naturally.
    """
    return T


def _ensure_corner_flows(corner: Tensor, corner_name: str) -> Tensor:
    """Rebuild a SymmetricTensor corner so its flows match _CORNER_SPECS."""
    _, _, expected_flow_a, expected_flow_b, _ = _CORNER_SPECS[corner_name]
    return _ensure_tensor_flows(corner, (expected_flow_a, expected_flow_b))


def _ensure_edge_flows(
    T_ket: Tensor, T_bra: Tensor, edge_name: str
) -> tuple[Tensor, Tensor]:
    """Rebuild SymmetricTensor edge halves so flows match _EDGE_*_SPECS."""
    _, _, _, fk1, fk2, fk3, _, _ = _EDGE_KET_SPECS[edge_name]
    _, _, _, fb1, fb2, fb3, _, _ = _EDGE_BRA_SPECS[edge_name]
    T_ket = _ensure_tensor_flows(T_ket, (fk1, fk2, fk3))
    T_bra = _ensure_tensor_flows(T_bra, (fb1, fb2, fb3))
    return T_ket, T_bra


def _reembed_target_for_projector(P: Tensor, Tg: Tensor) -> Tensor:
    """Re-embed Tg's 'fused' index to match P's if they differ.

    The projector has a unified fused index (covering both corners'
    charge sectors).  Re-embedding the target tensor UP to match P
    zero-pads any missing charge sectors, preserving all information.
    """
    if not isinstance(P, SymmetricTensor) or not isinstance(Tg, SymmetricTensor):
        return Tg
    p_fused_pos = P.labels().index("fused")
    tg_fused_pos = Tg.labels().index("fused")
    if not np.array_equal(
        P.indices[p_fused_pos].charges, Tg.indices[tg_fused_pos].charges
    ):
        return _reembed_fused(Tg, P.indices[p_fused_pos])
    return Tg


def _project_grown_edge_tensor(
    Tg: Tensor,
    P_first: Tensor,
    P_second: Tensor,
    left_fuse: tuple[str, str, str],
    right_fuse: tuple[str, str, str],
) -> Tensor:
    """Apply four projectors to a grown edge via Tensor protocol.

    Sequential application avoids todense: for each side (left/right),
    fuse two legs and contract with P_first†, then fuse the result with
    the third leg and contract with P_second†.

    When the fused charges after ``fuse_indices`` differ from the
    projector's unified fused index, ``_reembed_fused`` aligns them
    before contraction.

    Args:
        Tg:         Grown edge tensor (8 legs).
        P_first:    First projector (ket or bra, depending on move).
        P_second:   Second projector (bra or ket, depending on move).
        left_fuse:  (a, b, c) — fuse a+b for P_first, then chi+c for P_second.
        right_fuse: (a, b, c) — same for right side.

    Returns:
        4-leg Tensor ``(left_chi, mid1, mid2, right_chi)``.
    """
    la, lb, lc = left_fuse
    ra, rb, rc = right_fuse

    # --- Left side ---
    labels = Tg.labels()
    Tg = fuse_indices(Tg, labels.index(la), labels.index(lb), "fused", FlowDirection.IN)
    # Projector .bar() (no Koszul) for paired absorb contractions; see
    # _ctm_tensor_moves note for why this stays bosonic-bar.
    Tg = _reembed_target_for_projector(P_first, Tg)
    Tg = contract(P_first.bar(), Tg)  # "fused" contracted → "chi_new" created
    labels = Tg.labels()
    Tg = fuse_indices(
        Tg, labels.index("chi_new"), labels.index(lc), "fused", FlowDirection.IN
    )
    Tg = _reembed_target_for_projector(P_second, Tg)
    Tg = contract(P_second.bar(), Tg)
    Tg = Tg.relabel("chi_new", "left_chi")

    # --- Right side ---
    labels = Tg.labels()
    Tg = fuse_indices(Tg, labels.index(ra), labels.index(rb), "fused", FlowDirection.IN)
    Tg = _reembed_target_for_projector(P_first, Tg)
    Tg = contract(P_first.bar(), Tg)
    labels = Tg.labels()
    Tg = fuse_indices(
        Tg, labels.index("chi_new"), labels.index(rc), "fused", FlowDirection.IN
    )
    Tg = _reembed_target_for_projector(P_second, Tg)
    Tg = contract(P_second.bar(), Tg)
    Tg = Tg.relabel("chi_new", "right_chi")

    return Tg


def _project_grown_edge_tensor_lr(
    Tg: Tensor,
    P_1: Tensor,
    P_2: Tensor,
    left_fuse: tuple[str, str, str],
    right_fuse: tuple[str, str, str],
) -> Tensor:
    """Apply a biorthogonal projector pair to a closed grown edge.

    Mirrors the fused move's edge sandwich
    :math:`T' = P_1^\\dagger\\, T_g\\, P_2` (see
    :func:`tenax.algorithms._ctm_tensor_moves._apply_projector_tensor`),
    adapted to the split env's closed double-layer grown edge.

    Each side's three legs ``(env_chi, ket_D, bra_D)`` are hard-fused (two
    IN fuses, same order as :func:`_doublelayer_grown_corner`) into a single
    ``fused`` leg, re-embedded to the projector's unified fused index, then:

    - the **left** side (C1g side) is contracted with ``P_1.bar()`` (i.e.
      ``P_1^†``) — the new ``chi_new`` becomes ``left_chi``;
    - the **right** side (C4g side) is contracted with ``P_2`` (no dagger) —
      the new ``chi_new`` becomes ``right_chi``.

    Args:
        Tg:         Closed grown edge (8 legs).
        P_1:        Projector for the C1g side, labels ``(fused, chi_new)``.
        P_2:        Projector for the C4g side, labels ``(fused, chi_new)``.
        left_fuse:  (env_chi, ket_D, bra_D) labels of the left (C1) end.
        right_fuse: (env_chi, ket_D, bra_D) labels of the right (C4) end.

    Returns:
        4-leg Tensor ``(left_chi, mid_ket, mid_bra, right_chi)``.
    """
    la, lb, lc = left_fuse
    ra, rb, rc = right_fuse

    # --- Left side: fuse (env, ket_D), then (.., bra_D) -> P_1.bar() ---
    labels = Tg.labels()
    Tg = fuse_indices(Tg, labels.index(la), labels.index(lb), "fused", FlowDirection.IN)
    labels = Tg.labels()
    Tg = fuse_indices(
        Tg, labels.index("fused"), labels.index(lc), "fused", FlowDirection.IN
    )
    Tg = _reembed_target_for_projector(P_1, Tg)
    Tg = contract(P_1.bar(), Tg)  # "fused" contracted → "chi_new"
    Tg = Tg.relabel("chi_new", "left_chi")

    # --- Right side: fuse (env, ket_D), then (.., bra_D) -> P_2 (no dagger) ---
    # Right fuse is OUT (matches the fused move's ``fr``): P_2's fused leg is
    # IN, so the edge's right fused leg must be OUT to contract.
    labels = Tg.labels()
    Tg = fuse_indices(
        Tg, labels.index(ra), labels.index(rb), "fused", FlowDirection.OUT
    )
    labels = Tg.labels()
    Tg = fuse_indices(
        Tg, labels.index("fused"), labels.index(rc), "fused", FlowDirection.OUT
    )
    Tg = _reembed_target_for_projector(P_2, Tg)
    P2_relabeled = P_2.relabel("chi_new", "right_chi")
    Tg = contract(P2_relabeled, Tg)  # "fused" contracted → "right_chi"

    return Tg


def _apply_projector(
    P: Tensor,
    Cg_fused: Tensor,
    base_charges: np.ndarray | None = None,  # noqa: ARG001 — kept for API compat
) -> Tensor:
    """Apply projector P to a fused corner/edge tensor via Tensor protocol.

    Computes ``P^dagger @ Cg`` by contracting over the shared ``fused`` leg.
    Uses ``.bar()`` (conjugate + flip flows) so the contraction engine
    handles both DenseTensor and SymmetricTensor without todense.

    When P's fused index (from the unified projector) differs from Cg's
    fused index, re-embeds Cg to match P before contracting.

    Args:
        P:            Projector tensor with labels ``(fused, chi_new)``.
        Cg_fused:     Fused corner/edge with ``fused`` as one of its legs.
        base_charges: Unused (kept for call-site compatibility during transition).
    """
    # Re-embed Cg to match P's fused index if they differ.
    # This zero-pads charge sectors absent from Cg, ensuring the
    # contraction engine sees matching indices on the contracted leg.
    Cg_fused = _reembed_target_for_projector(P, Cg_fused)
    # Projector .bar() (no Koszul) — see _ctm_tensor_moves note.
    result = contract(P.bar(), Cg_fused)  # contracts over "fused" → (chi_new, ...)
    return result


def _precombine_projector_pair(
    P_first: Tensor,
    P_second: Tensor,
    chi_leg: str,
    D1_leg: str,
    D2_leg: str,
    out_label: str,
) -> Tensor:
    """Fuse a sequential projector pair into one 4-leg operator (dense path).

    Reproduces the two-step projection that :func:`_project_grown_edge_tensor`
    applies to one side of a grown edge — ``fuse(chi_leg, D1) -> P_first.bar()
    -> m`` then ``fuse(m, D2) -> P_second.bar() -> out`` — but as a standalone
    operator ``op(chi_leg, D1_leg, D2_leg, out_label)``.  Contracting ``op``
    into a half-edge over the two legs that live there absorbs the projectors
    *before* the interlayer join, so the ``chi^2 * D^6`` edge never forms.

    Both grown-corner projectors carry a ``fused`` leg whose parents are
    always ``(chi-type, D-type)`` (the env bond first, the ``A`` virtual leg
    second).  Splitting the ``fused`` leg therefore exposes the env-bond and
    ``D`` components in a known order; relabelling them to the edge's legs
    and contracting over the shared ``chi_new`` bond rebuilds the paired
    operator.

    DenseTensor only: ``split_index`` after ``.bar()`` does not preserve the
    order-dependent Koszul signs that the closed-edge contraction encodes for
    SymmetricTensor (issue #641 / #463 Phase 2-4), so the SymmetricTensor
    projection stays on the closed ``_project_grown_edge_tensor`` path.
    """
    Pf = P_first.bar().relabel("chi_new", "_pcomb_m")
    Pf = split_index(Pf, Pf.labels().index("fused"))
    pf_labels = Pf.labels()  # (parent_chi, parent_D, _pcomb_m)
    Pf = Pf.relabels({pf_labels[0]: chi_leg, pf_labels[1]: D1_leg})

    Ps = P_second.bar().relabel("chi_new", out_label)
    Ps = split_index(Ps, Ps.labels().index("fused"))
    ps_labels = Ps.labels()  # (parent_chi == _pcomb_m, parent_D, out_label)
    Ps = Ps.relabels({ps_labels[0]: "_pcomb_m", ps_labels[1]: D2_leg})

    return contract(Pf, Ps)  # -> (chi_leg, D1_leg, D2_leg, out_label)


def _grow_and_project_bounded(
    T_ket: Tensor,
    T_bra: Tensor,
    A: Tensor,
    A_bar: Tensor,
    P_first: Tensor,
    P_second: Tensor,
    contracted_leg: str,
    ket_I_label: str,
    bra_I_label: str,
    left_fuse: tuple[str, str, str],
    right_fuse: tuple[str, str, str],
) -> Tensor:
    """Grow + project an edge without forming the ``chi^2 * D^6`` tensor.

    Builds the two open half-edges, precombines each projector pair into a
    4-leg operator, absorbs each operator into the half-edge holding the
    majority of its legs, then joins.  Peak intermediate is ``chi^2 * D^3 * d``
    (vs the closed path's ``chi^2 * D^6``), so the forward CTM convergence is
    ``chi^2 * D^4``-bounded (issue #641).

    Returns the 4-leg projected edge ``(left_chi, mid_ket, mid_bra,
    right_chi)`` — identical (to machine precision) to
    ``_project_grown_edge_tensor(_grow_edge_no_double_layer(...), ...)``.

    DenseTensor only; see :func:`_precombine_projector_pair`.
    """
    ket_half, bra_half = _grow_edge_halves(
        T_ket, T_bra, A, A_bar, contracted_leg, ket_I_label, bra_I_label
    )

    # left_fuse is (chi, D, D); right_fuse is (D, chi, D) — the env bond sits
    # at position 0 of left_fuse and position 1 of right_fuse for every move.
    la, lb, lc = left_fuse
    ra, rb, rc = right_fuse
    P_left = _precombine_projector_pair(P_first, P_second, la, lb, lc, "left_chi")
    P_right = _precombine_projector_pair(P_first, P_second, rb, ra, rc, "right_chi")

    def _absorb(op: Tensor) -> Tensor:
        op_labels = set(op.labels())
        in_ket = len(op_labels & set(ket_half.labels()))
        in_bra = len(op_labels & set(bra_half.labels()))
        return contract(op, ket_half) if in_ket >= in_bra else contract(op, bra_half)

    # Join over (_I, phys, and the two cross-legs shared between the reduced
    # halves) -> (left_chi, mid_ket, mid_bra, right_chi).
    return contract(_absorb(P_left), _absorb(P_right))


def _precombine_factorized_pair(
    P_a: Tensor,
    P_b: Tensor,
    env_label: str,
    ketD_label: str,
    braD_label: str,
    out_chi: str,
    side: str,
) -> Tensor:
    """Fuse a factorized projector pair into one 4-leg edge operator.

    ``P_a``/``P_b`` come from :func:`_factorize_projector` applied to one of
    the biorthogonal corner projectors ``P_1`` (C1-side) or ``P_2`` (C4-side):

    - ``P_a`` has legs ``(env, ketD, _fac)``;
    - ``P_b`` has legs ``(_fac, braD, chi_new)``.

    The closed :func:`_project_grown_edge_tensor_lr` applies the **left** side
    with ``P_1.bar()`` and the **right** side with ``P_2`` (no dagger).  Since
    ``P = contract(P_a, P_b)`` (over ``_fac``) and ``bar`` distributes over a
    dense contraction, ``P.bar() = contract(P_a.bar(), P_b.bar())`` and
    ``P = contract(P_a, P_b)``.  This builds the corresponding precombined
    operator and relabels its three contracted legs to the half-edge labels
    ``env_label``/``ketD_label``/``braD_label`` and its open leg to
    ``out_chi``.

    Args:
        side: ``"left"`` → use ``.bar()`` factors (mirrors ``P_1.bar()``);
              ``"right"`` → use bare factors (mirrors un-daggered ``P_2``).

    Returns a 4-leg operator ``(env_label, ketD_label, braD_label, out_chi)``.

    DenseTensor only; ``bar`` distribution over contraction does not preserve
    the order-dependent Koszul signs for SymmetricTensor (#641 / #463 Ph 2-4).
    """
    if side == "left":
        Pa = P_a.bar()
        Pb = P_b.bar()
    else:
        Pa = P_a
        Pb = P_b
    op = contract(Pa, Pb)  # -> (env, ketD, braD, chi_new) up to leg order
    return op.relabels(
        {
            "env": env_label,
            "ketD": ketD_label,
            "braD": braD_label,
            "chi_new": out_chi,
        }
    )


def _grow_and_project_bounded_lr(
    T_ket: Tensor,
    T_bra: Tensor,
    A: Tensor,
    A_bar: Tensor,
    P_left_first: Tensor,
    P_left_second: Tensor,
    P_right_first: Tensor,
    P_right_second: Tensor,
    contracted_leg: str,
    ket_I_label: str,
    bra_I_label: str,
    left_fuse: tuple[str, str, str],
    right_fuse: tuple[str, str, str],
) -> Tensor:
    """Memory-bounded biorthogonal edge application (chi^2 * D^4).

    Like :func:`_grow_and_project_bounded` but uses the **left** factorized
    pair ``(P_left_first, P_left_second)`` for the C1-side end and the
    **right** factorized pair ``(P_right_first, P_right_second)`` for the
    C4-side end — the double-layer biorthogonal pair ``(P_1, P_2)`` factorized
    via :func:`_factorize_projector`.

    Reproduces :func:`_project_grown_edge_tensor_lr` applied to the closed
    :func:`_grow_edge_no_double_layer` edge to machine precision, but never
    forms the ``chi^2 * D^6`` closed edge: each precombined 4-leg operator is
    absorbed into the open half-edge that holds the majority of its legs before
    the interlayer join (peak ``chi^2 * D^3 * d``).

    ``left_fuse`` / ``right_fuse`` are ``(env_chi, ketD, braD)`` triples — the
    same tuples the closed ``_lr`` path fuses.  The factorized pairs carry the
    generic ``env``/``ketD``/``braD``/``chi_new`` labels from
    :func:`_factorize_projector`.

    Returns the 4-leg projected edge ``(left_chi, mid_ket, mid_bra,
    right_chi)``.  DenseTensor only.
    """
    ket_half, bra_half = _grow_edge_halves(
        T_ket, T_bra, A, A_bar, contracted_leg, ket_I_label, bra_I_label
    )

    la, lb, lc = left_fuse
    ra, rb, rc = right_fuse
    # Left end: mirrors P_1.bar(); right end: mirrors un-daggered P_2.
    P_left = _precombine_factorized_pair(
        P_left_first, P_left_second, la, lb, lc, "left_chi", "left"
    )
    P_right = _precombine_factorized_pair(
        P_right_first, P_right_second, ra, rb, rc, "right_chi", "right"
    )

    def _absorb(op: Tensor) -> Tensor:
        op_labels = set(op.labels())
        in_ket = len(op_labels & set(ket_half.labels()))
        in_bra = len(op_labels & set(bra_half.labels()))
        return contract(op, ket_half) if in_ket >= in_bra else contract(op, bra_half)

    return contract(_absorb(P_left), _absorb(P_right))


def _grow_and_project_edge(
    T_ket: Tensor,
    T_bra: Tensor,
    A: Tensor,
    A_bar: Tensor,
    P_first: Tensor,
    P_second: Tensor,
    contracted_leg: str,
    ket_I_label: str,
    bra_I_label: str,
    grow_output_labels: tuple[str, ...],
    left_fuse: tuple[str, str, str],
    right_fuse: tuple[str, str, str],
) -> Tensor:
    """Grow + project an edge, choosing the memory-bounded path when possible.

    For DenseTensor inputs, routes through :func:`_grow_and_project_bounded`
    (``chi^2 * D^4``-bounded, issue #641).  For SymmetricTensor inputs, falls
    back to the closed ``chi^2 * D^6`` grow + project, which preserves the
    fermionic Koszul-sign convention (bounded SymmetricTensor support is the
    #463 Phase 2-4 follow-up).
    """
    if isinstance(T_ket, DenseTensor) and isinstance(T_bra, DenseTensor):
        return _grow_and_project_bounded(
            T_ket,
            T_bra,
            A,
            A_bar,
            P_first,
            P_second,
            contracted_leg,
            ket_I_label,
            bra_I_label,
            left_fuse,
            right_fuse,
        )
    Tg = _grow_edge_no_double_layer(
        T_ket,
        T_bra,
        A,
        A_bar,
        contracted_leg,
        ket_I_label,
        bra_I_label,
        grow_output_labels,
    )
    return _project_grown_edge_tensor(
        Tg, P_first, P_second, left_fuse=left_fuse, right_fuse=right_fuse
    )


# ------------------------------------------------------------------ #
# Directional CTM moves                                                #
# ------------------------------------------------------------------ #


def _split_ctm_move_left(
    env: SplitCTMTensorEnv,
    A: Tensor,
    A_bar: Tensor,
    chi: int,
    chi_I: int,
) -> SplitCTMTensorEnv:
    """Left move: double-layer corner-pair projector (closed edge)."""
    base_charges = A.indices[0].charges if isinstance(A, SymmetricTensor) else None

    # --- Phase A: Double-layer grown corners + biorthogonal projector pair ---

    # Double-layer grown corners (C1 with T1_ket+T1_bra; C4 with T3_ket+T3_bra)
    C1g, c1_rem = _doublelayer_grown_corner(
        env.C1,
        env.T1_ket,
        env.T1_bra,
        ("c1_r", "t1k_l"),
        "t1k_I",
        "t1b_I",
        ("c1_d", "u_ket", "u_bra"),
    )  # fused=(c1_d,u_ket,u_bra); remaining=t1b_r
    C4g, c4_rem = _doublelayer_grown_corner(
        env.C4,
        env.T3_ket,
        env.T3_bra,
        ("c4_r", "t3k_r"),
        "t3k_I",
        "t3b_I",
        ("c4_u", "d_ket", "d_bra"),
    )  # fused=(c4_u,d_ket,d_bra); remaining=t3b_l

    # Fishman biorthogonal pair (P_1 for C1 side, P_2 for C4 side)
    P_1, P_2, _eps_t = _compute_projector_tensor(
        C1g, C4g, chi, base_charges=base_charges
    )

    # New corners: P_1 -> C1g, P_2 -> C4g
    C1_new = _apply_projector(P_1, C1g, base_charges).relabels(
        {"chi_new": "c1_d", c1_rem: "c1_r"}
    )
    C4_new = _apply_projector(P_2, C4g, base_charges).relabels(
        {"chi_new": "c4_r", c4_rem: "c4_u"}
    )
    C1_new = _ensure_corner_flows(C1_new, "C1")
    C4_new = _ensure_corner_flows(C4_new, "C4")
    C1_new, _ = max_abs_normalize(C1_new)
    C4_new, _ = max_abs_normalize(C4_new)

    # --- Phase B: project the grown edge with the biorthogonal pair ---
    left_fuse = ("t4k_d", "u", "U")
    right_fuse = ("t4b_u", "d", "D")
    if _FORCE_CLOSED_EDGE or not isinstance(A, DenseTensor):
        Tg = _grow_edge_no_double_layer(
            env.T4_ket,
            env.T4_bra,
            A,
            A_bar,
            "l",
            "t4k_I",
            "t4b_I",
            ("t4k_d", "u", "U", "r", "R", "t4b_u", "d", "D"),
        )
        T4g = _project_grown_edge_tensor_lr(
            Tg, P_1, P_2, left_fuse=left_fuse, right_fuse=right_fuse
        )
    else:
        D = A.indices[0].dim
        P_1u = _unfuse_projector_fused(
            P_1, P_1.indices[0].dim // (D * D), D, "env", "ketD", "braD"
        )
        P_2u = _unfuse_projector_fused(
            P_2, P_2.indices[0].dim // (D * D), D, "env", "ketD", "braD"
        )
        P1f, P1s, _ = _factorize_projector(P_1u, "env", "ketD", "braD", "chi_new")
        P2f, P2s, _ = _factorize_projector(P_2u, "env", "ketD", "braD", "chi_new")
        T4g = _grow_and_project_bounded_lr(
            env.T4_ket,
            env.T4_bra,
            A,
            A_bar,
            P1f,
            P1s,
            P2f,
            P2s,
            "l",
            "t4k_I",
            "t4b_I",
            left_fuse,
            right_fuse,
        )
    # T4g now: (left_chi, r, R, right_chi)

    # --- Phase C: SVD split into ket/bra ---
    T4_ket_new, T4_bra_new = _svd_split_edge_tensor(
        T4g,
        left_labels=["left_chi", "r"],
        right_labels=["R", "right_chi"],
        chi_I=chi_I,
        ket_relabels={"left_chi": "t4k_d", "r": "l_ket", "_svd_bond": "t4k_I"},
        bra_relabels={"_svd_bond": "t4b_I", "R": "l_bra", "right_chi": "t4b_u"},
        base_charges=base_charges,
    )
    T4_ket_new, T4_bra_new = _ensure_edge_flows(T4_ket_new, T4_bra_new, "T4")

    return env._replace(
        C1=C1_new,
        C4=C4_new,
        T4_ket=T4_ket_new,
        T4_bra=T4_bra_new,
    )


def _split_ctm_move_right(
    env: SplitCTMTensorEnv,
    A: Tensor,
    A_bar: Tensor,
    chi: int,
    chi_I: int,
) -> SplitCTMTensorEnv:
    """Right move: double-layer corner-pair projector (closed edge)."""
    base_charges = A.indices[0].charges if isinstance(A, SymmetricTensor) else None

    # --- Phase A: Double-layer grown corners + biorthogonal projector pair ---

    # C2 connects to the top edge via its bra-right chi (t1b_r); grow bra first
    # then the ket layer over the interlayer, but fuse in (env, ket, bra) order.
    C2g, c2_rem = _doublelayer_grown_corner(
        env.C2,
        env.T1_bra,
        env.T1_ket,
        ("c2_l", "t1b_r"),
        "t1b_I",
        "t1k_I",
        ("c2_d", "u_ket", "u_bra"),
    )  # fused=(c2_d,u_ket,u_bra); remaining=t1k_l
    C3g, c3_rem = _doublelayer_grown_corner(
        env.C3,
        env.T3_bra,
        env.T3_ket,
        ("c3_l", "t3b_l"),
        "t3b_I",
        "t3k_I",
        ("c3_u", "d_ket", "d_bra"),
    )  # fused=(c3_u,d_ket,d_bra); remaining=t3k_r

    # Fishman biorthogonal pair (P_1 for C2 side, P_2 for C3 side)
    P_1, P_2, _eps_t = _compute_projector_tensor(
        C2g, C3g, chi, base_charges=base_charges
    )

    C2_new = _apply_projector(P_1, C2g, base_charges).relabels(
        {"chi_new": "c2_l", c2_rem: "c2_d"}
    )
    C3_new = _apply_projector(P_2, C3g, base_charges).relabels(
        {"chi_new": "c3_u", c3_rem: "c3_l"}
    )
    C2_new = _ensure_corner_flows(C2_new, "C2")
    C3_new = _ensure_corner_flows(C3_new, "C3")
    C2_new, _ = max_abs_normalize(C2_new)
    C3_new, _ = max_abs_normalize(C3_new)

    # --- Phase B: project the grown edge with the biorthogonal pair ---
    left_fuse = ("t2k_u", "u", "U")
    right_fuse = ("t2b_d", "d", "D")
    if _FORCE_CLOSED_EDGE or not isinstance(A, DenseTensor):
        Tg = _grow_edge_no_double_layer(
            env.T2_ket,
            env.T2_bra,
            A,
            A_bar,
            "r",
            "t2k_I",
            "t2b_I",
            ("t2k_u", "u", "U", "l", "L", "t2b_d", "d", "D"),
        )
        T2g = _project_grown_edge_tensor_lr(
            Tg, P_1, P_2, left_fuse=left_fuse, right_fuse=right_fuse
        )
    else:
        D = A.indices[0].dim
        P_1u = _unfuse_projector_fused(
            P_1, P_1.indices[0].dim // (D * D), D, "env", "ketD", "braD"
        )
        P_2u = _unfuse_projector_fused(
            P_2, P_2.indices[0].dim // (D * D), D, "env", "ketD", "braD"
        )
        P1f, P1s, _ = _factorize_projector(P_1u, "env", "ketD", "braD", "chi_new")
        P2f, P2s, _ = _factorize_projector(P_2u, "env", "ketD", "braD", "chi_new")
        T2g = _grow_and_project_bounded_lr(
            env.T2_ket,
            env.T2_bra,
            A,
            A_bar,
            P1f,
            P1s,
            P2f,
            P2s,
            "r",
            "t2k_I",
            "t2b_I",
            left_fuse,
            right_fuse,
        )
    # T2g now: (left_chi, l, L, right_chi)

    # --- Phase C: SVD split into ket/bra ---
    T2_ket_new, T2_bra_new = _svd_split_edge_tensor(
        T2g,
        left_labels=["left_chi", "l"],
        right_labels=["L", "right_chi"],
        chi_I=chi_I,
        ket_relabels={"left_chi": "t2k_u", "l": "r_ket", "_svd_bond": "t2k_I"},
        bra_relabels={"_svd_bond": "t2b_I", "L": "r_bra", "right_chi": "t2b_d"},
        base_charges=base_charges,
    )
    T2_ket_new, T2_bra_new = _ensure_edge_flows(T2_ket_new, T2_bra_new, "T2")

    return env._replace(
        C2=C2_new,
        C3=C3_new,
        T2_ket=T2_ket_new,
        T2_bra=T2_bra_new,
    )


def _split_ctm_move_top(
    env: SplitCTMTensorEnv,
    A: Tensor,
    A_bar: Tensor,
    chi: int,
    chi_I: int,
) -> SplitCTMTensorEnv:
    """Top move: double-layer corner-pair projector (closed edge)."""
    base_charges = A.indices[0].charges if isinstance(A, SymmetricTensor) else None

    # --- Phase A: Double-layer grown corners + biorthogonal projector pair ---

    # C1 connects to the left edge via its ket end (t4k_d); C2 to the right
    # edge via its ket end (t2k_u).
    C1g, c1_rem = _doublelayer_grown_corner(
        env.C1,
        env.T4_ket,
        env.T4_bra,
        ("c1_d", "t4k_d"),
        "t4k_I",
        "t4b_I",
        ("c1_r", "l_ket", "l_bra"),
    )  # fused=(c1_r,l_ket,l_bra); remaining=t4b_u
    C2g, c2_rem = _doublelayer_grown_corner(
        env.C2,
        env.T2_ket,
        env.T2_bra,
        ("c2_d", "t2k_u"),
        "t2k_I",
        "t2b_I",
        ("c2_l", "r_ket", "r_bra"),
    )  # fused=(c2_l,r_ket,r_bra); remaining=t2b_d

    # Fishman biorthogonal pair (P_1 for C1 side, P_2 for C2 side)
    P_1, P_2, _eps_t = _compute_projector_tensor(
        C1g, C2g, chi, base_charges=base_charges
    )

    C1_new = _apply_projector(P_1, C1g, base_charges).relabels(
        {"chi_new": "c1_d", c1_rem: "c1_r"}
    )
    C2_new = _apply_projector(P_2, C2g, base_charges).relabels(
        {"chi_new": "c2_l", c2_rem: "c2_d"}
    )
    C1_new = _ensure_corner_flows(C1_new, "C1")
    C2_new = _ensure_corner_flows(C2_new, "C2")
    C1_new, _ = max_abs_normalize(C1_new)
    C2_new, _ = max_abs_normalize(C2_new)

    # --- Phase B: project the grown edge with the biorthogonal pair ---
    left_fuse = ("t1k_l", "l", "L")
    right_fuse = ("t1b_r", "r", "R")
    if _FORCE_CLOSED_EDGE or not isinstance(A, DenseTensor):
        Tg = _grow_edge_no_double_layer(
            env.T1_ket,
            env.T1_bra,
            A,
            A_bar,
            "u",
            "t1k_I",
            "t1b_I",
            ("t1k_l", "l", "L", "d", "D", "t1b_r", "r", "R"),
        )
        T1g = _project_grown_edge_tensor_lr(
            Tg, P_1, P_2, left_fuse=left_fuse, right_fuse=right_fuse
        )
    else:
        D = A.indices[0].dim
        P_1u = _unfuse_projector_fused(
            P_1, P_1.indices[0].dim // (D * D), D, "env", "ketD", "braD"
        )
        P_2u = _unfuse_projector_fused(
            P_2, P_2.indices[0].dim // (D * D), D, "env", "ketD", "braD"
        )
        P1f, P1s, _ = _factorize_projector(P_1u, "env", "ketD", "braD", "chi_new")
        P2f, P2s, _ = _factorize_projector(P_2u, "env", "ketD", "braD", "chi_new")
        T1g = _grow_and_project_bounded_lr(
            env.T1_ket,
            env.T1_bra,
            A,
            A_bar,
            P1f,
            P1s,
            P2f,
            P2s,
            "u",
            "t1k_I",
            "t1b_I",
            left_fuse,
            right_fuse,
        )
    # T1g now: (left_chi, d, D, right_chi)

    # --- Phase C: SVD split into ket/bra ---
    T1_ket_new, T1_bra_new = _svd_split_edge_tensor(
        T1g,
        left_labels=["left_chi", "d"],
        right_labels=["D", "right_chi"],
        chi_I=chi_I,
        ket_relabels={"left_chi": "t1k_l", "d": "u_ket", "_svd_bond": "t1k_I"},
        bra_relabels={"_svd_bond": "t1b_I", "D": "u_bra", "right_chi": "t1b_r"},
        base_charges=base_charges,
    )
    T1_ket_new, T1_bra_new = _ensure_edge_flows(T1_ket_new, T1_bra_new, "T1")

    return env._replace(
        C1=C1_new,
        C2=C2_new,
        T1_ket=T1_ket_new,
        T1_bra=T1_bra_new,
    )


def _split_ctm_move_bottom(
    env: SplitCTMTensorEnv,
    A: Tensor,
    A_bar: Tensor,
    chi: int,
    chi_I: int,
) -> SplitCTMTensorEnv:
    """Bottom move: double-layer corner-pair projector (closed edge)."""
    base_charges = A.indices[0].charges if isinstance(A, SymmetricTensor) else None

    # --- Phase A: Double-layer grown corners + biorthogonal projector pair ---

    # C4 connects to the left edge via its bra end (t4b_u); C3 to the right
    # edge via its bra end (t2b_d).  Grow bra first, fuse in (env, ket, bra).
    C4g, c4_rem = _doublelayer_grown_corner(
        env.C4,
        env.T4_bra,
        env.T4_ket,
        ("c4_r", "t4b_u"),
        "t4b_I",
        "t4k_I",
        ("c4_u", "l_ket", "l_bra"),
    )  # fused=(c4_u,l_ket,l_bra); remaining=t4k_d
    C3g, c3_rem = _doublelayer_grown_corner(
        env.C3,
        env.T2_bra,
        env.T2_ket,
        ("c3_l", "t2b_d"),
        "t2b_I",
        "t2k_I",
        ("c3_u", "r_ket", "r_bra"),
    )  # fused=(c3_u,r_ket,r_bra); remaining=t2k_u

    # Fishman biorthogonal pair (P_1 for C4 side, P_2 for C3 side)
    P_1, P_2, _eps_t = _compute_projector_tensor(
        C4g, C3g, chi, base_charges=base_charges
    )

    C4_new = _apply_projector(P_1, C4g, base_charges).relabels(
        {"chi_new": "c4_r", c4_rem: "c4_u"}
    )
    C3_new = _apply_projector(P_2, C3g, base_charges).relabels(
        {"chi_new": "c3_u", c3_rem: "c3_l"}
    )
    C4_new = _ensure_corner_flows(C4_new, "C4")
    C3_new = _ensure_corner_flows(C3_new, "C3")
    C4_new, _ = max_abs_normalize(C4_new)
    C3_new, _ = max_abs_normalize(C3_new)

    # --- Phase B: project the grown edge with the biorthogonal pair ---
    left_fuse = ("t3k_r", "l", "L")
    right_fuse = ("t3b_l", "r", "R")
    if _FORCE_CLOSED_EDGE or not isinstance(A, DenseTensor):
        Tg = _grow_edge_no_double_layer(
            env.T3_ket,
            env.T3_bra,
            A,
            A_bar,
            "d",
            "t3k_I",
            "t3b_I",
            ("t3k_r", "l", "L", "u", "U", "t3b_l", "r", "R"),
        )
        T3g = _project_grown_edge_tensor_lr(
            Tg, P_1, P_2, left_fuse=left_fuse, right_fuse=right_fuse
        )
    else:
        D = A.indices[0].dim
        P_1u = _unfuse_projector_fused(
            P_1, P_1.indices[0].dim // (D * D), D, "env", "ketD", "braD"
        )
        P_2u = _unfuse_projector_fused(
            P_2, P_2.indices[0].dim // (D * D), D, "env", "ketD", "braD"
        )
        P1f, P1s, _ = _factorize_projector(P_1u, "env", "ketD", "braD", "chi_new")
        P2f, P2s, _ = _factorize_projector(P_2u, "env", "ketD", "braD", "chi_new")
        T3g = _grow_and_project_bounded_lr(
            env.T3_ket,
            env.T3_bra,
            A,
            A_bar,
            P1f,
            P1s,
            P2f,
            P2s,
            "d",
            "t3k_I",
            "t3b_I",
            left_fuse,
            right_fuse,
        )
    # T3g now: (left_chi, u, U, right_chi)

    # --- Phase C: SVD split into ket/bra ---
    T3_ket_new, T3_bra_new = _svd_split_edge_tensor(
        T3g,
        left_labels=["left_chi", "u"],
        right_labels=["U", "right_chi"],
        chi_I=chi_I,
        ket_relabels={"left_chi": "t3k_r", "u": "d_ket", "_svd_bond": "t3k_I"},
        bra_relabels={"_svd_bond": "t3b_I", "U": "d_bra", "right_chi": "t3b_l"},
        base_charges=base_charges,
    )
    T3_ket_new, T3_bra_new = _ensure_edge_flows(T3_ket_new, T3_bra_new, "T3")

    return env._replace(
        C3=C3_new,
        C4=C4_new,
        T3_ket=T3_ket_new,
        T3_bra=T3_bra_new,
    )
