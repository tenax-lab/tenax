"""Split CTM with Tensor protocol — move helpers and directional moves."""

from __future__ import annotations

__all__ = [
    "_apply_projector",
    "_ensure_corner_flows",
    "_ensure_edge_flows",
    "_ensure_tensor_flows",
    "_fused_charge_permutation",
    "_grow_edge_no_double_layer",
    "_project_grown_edge_tensor",
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
)
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor
from tenax.linalg import svd as tensor_svd

# ------------------------------------------------------------------ #
# No-double-layer edge growth                                          #
# ------------------------------------------------------------------ #

_VIRTUAL_LEGS = ("u", "d", "l", "r")


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
    """
    ket_D_label = f"{contracted_leg}_ket"
    bra_D_label = f"{contracted_leg}_bra"

    # --- Ket side ---
    A_ket = A.relabel(contracted_leg, ket_D_label)
    ket_half = contract(T_ket, A_ket)

    # --- Bra side: relabel virtual legs to uppercase ---
    bra_mapping: dict[str, str] = {contracted_leg: bra_D_label}
    for v in _VIRTUAL_LEGS:
        if v != contracted_leg:
            bra_mapping[v] = v.upper()
    A_bra = A_bar.relabels(bra_mapping)
    bra_half = contract(T_bra, A_bra)

    # --- Match interlayer labels, then contract (traces _I + phys) ---
    ket_half = ket_half.relabel(ket_I_label, "_I")
    bra_half = bra_half.relabel(bra_I_label, "_I")
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
    """Left move: ket first (C1/C4 connect to T1/T3 ket chi bonds)."""
    base_charges = A.indices[0].charges if isinstance(A, SymmetricTensor) else None

    # --- Phase A: Per-layer projectors and new corners ---

    # Grow C1 with T1_ket
    C1_r = env.C1.relabel("c1_r", "t1k_l")
    C1g_ket = contract(C1_r, env.T1_ket)  # (c1_d, u_ket, t1k_I)
    C1g_ket_fused = fuse_indices(C1g_ket, 0, 1, "fused", FlowDirection.IN)

    # Grow C4 with T3_ket
    C4_r = env.C4.relabel("c4_r", "t3k_r")
    C4g_ket = contract(C4_r, env.T3_ket)  # (c4_u, d_ket, t3k_I)
    C4g_ket_fused = fuse_indices(C4g_ket, 0, 1, "fused", FlowDirection.IN)

    # Ket projector
    P_ket, _, _eps_t = _compute_projector_tensor(
        C1g_ket_fused, C4g_ket_fused, chi, base_charges=base_charges
    )

    # Mid-corners: project ket grown corners
    C1_mid = _apply_projector(P_ket, C1g_ket_fused, base_charges)  # (chi_new, t1k_I)
    C4_mid = _apply_projector(P_ket, C4g_ket_fused, base_charges)  # (chi_new, t3k_I)

    # Grow mid-corners with bra edges
    C1g_bra = contract(C1_mid.relabel("t1k_I", "t1b_I"), env.T1_bra)
    C1g_bra_fused = fuse_indices(C1g_bra, 0, 1, "fused", FlowDirection.IN)

    C4g_bra = contract(C4_mid.relabel("t3k_I", "t3b_I"), env.T3_bra)
    C4g_bra_fused = fuse_indices(C4g_bra, 0, 1, "fused", FlowDirection.IN)

    # Bra projector
    P_bra, _, _eps_t = _compute_projector_tensor(
        C1g_bra_fused, C4g_bra_fused, chi, base_charges=base_charges
    )

    # New corners
    C1_new = _apply_projector(P_bra, C1g_bra_fused, base_charges)  # (chi_new, t1b_r)
    C4_new = _apply_projector(P_bra, C4g_bra_fused, base_charges)  # (chi_new, t3b_l)

    C1_new = C1_new.relabels({"chi_new": "c1_d", "t1b_r": "c1_r"})
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t3b_l": "c4_u"})
    C1_new = _ensure_corner_flows(C1_new, "C1")
    C4_new = _ensure_corner_flows(C4_new, "C4")
    C1_new, _ = max_abs_normalize(C1_new)
    C4_new, _ = max_abs_normalize(C4_new)

    # --- Phase B: Sequential projector application to grown edge ---

    T4g = _grow_edge_no_double_layer(
        env.T4_ket,
        env.T4_bra,
        A,
        A_bar,
        "l",
        "t4k_I",
        "t4b_I",
        ("t4k_d", "u", "U", "r", "R", "t4b_u", "d", "D"),
    )

    T4g = _project_grown_edge_tensor(
        T4g,
        P_ket,
        P_bra,
        left_fuse=("t4k_d", "u", "U"),
        right_fuse=("d", "t4b_u", "D"),
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
    """Right move: bra first (C2/C3 connect to T1/T3 bra chi bonds)."""
    base_charges = A.indices[0].charges if isinstance(A, SymmetricTensor) else None

    # --- Phase A: Per-layer projectors and new corners ---

    # Grow C2 with T1_bra (bra first)
    C2_l = env.C2.relabel("c2_l", "t1b_r")
    C2g_bra = contract(C2_l, env.T1_bra)  # (c2_d, t1b_I, u_bra)
    C2g_bra_fused = fuse_indices(C2g_bra, 0, 2, "fused", FlowDirection.IN)

    # Grow C3 with T3_bra
    C3_l = env.C3.relabel("c3_l", "t3b_l")
    C3g_bra = contract(C3_l, env.T3_bra)  # (c3_u, t3b_I, d_bra)
    C3g_bra_fused = fuse_indices(C3g_bra, 0, 2, "fused", FlowDirection.IN)

    # Bra projector
    P_bra, _, _eps_t = _compute_projector_tensor(
        C2g_bra_fused, C3g_bra_fused, chi, base_charges=base_charges
    )

    # Mid-corners: project bra grown corners
    C2_mid = _apply_projector(P_bra, C2g_bra_fused, base_charges)  # (chi_new, t1b_I)
    C3_mid = _apply_projector(P_bra, C3g_bra_fused, base_charges)  # (chi_new, t3b_I)

    # Grow mid-corners with ket edges
    C2g_ket = contract(C2_mid.relabel("t1b_I", "t1k_I"), env.T1_ket)
    C2g_ket_fused = fuse_indices(C2g_ket, 0, 2, "fused", FlowDirection.IN)

    C3g_ket = contract(C3_mid.relabel("t3b_I", "t3k_I"), env.T3_ket)
    C3g_ket_fused = fuse_indices(C3g_ket, 0, 2, "fused", FlowDirection.IN)

    # Ket projector
    P_ket, _, _eps_t = _compute_projector_tensor(
        C2g_ket_fused, C3g_ket_fused, chi, base_charges=base_charges
    )

    # New corners
    C2_new = _apply_projector(P_ket, C2g_ket_fused, base_charges)  # (chi_new, t1k_l)
    C3_new = _apply_projector(P_ket, C3g_ket_fused, base_charges)  # (chi_new, t3k_r)

    C2_new = C2_new.relabels({"chi_new": "c2_l", "t1k_l": "c2_d"})
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t3k_r": "c3_l"})
    C2_new = _ensure_corner_flows(C2_new, "C2")
    C3_new = _ensure_corner_flows(C3_new, "C3")
    C2_new, _ = max_abs_normalize(C2_new)
    C3_new, _ = max_abs_normalize(C3_new)

    # --- Phase B: Sequential projector application to grown edge ---

    T2g = _grow_edge_no_double_layer(
        env.T2_ket,
        env.T2_bra,
        A,
        A_bar,
        "r",
        "t2k_I",
        "t2b_I",
        ("t2k_u", "u", "U", "l", "L", "t2b_d", "d", "D"),
    )
    T2g = _project_grown_edge_tensor(
        T2g,
        P_bra,
        P_ket,
        left_fuse=("t2k_u", "U", "u"),
        right_fuse=("D", "t2b_d", "d"),
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
    """Top move: ket first (C1/C2 connect to T4/T2 ket chi bonds)."""
    base_charges = A.indices[0].charges if isinstance(A, SymmetricTensor) else None

    # --- Phase A: Per-layer projectors and new corners ---

    # Grow C1 with T4_ket
    C1_d = env.C1.relabel("c1_d", "t4k_d")
    C1g_ket = contract(C1_d, env.T4_ket)  # (c1_r, l_ket, t4k_I)
    C1g_ket_fused = fuse_indices(C1g_ket, 0, 1, "fused", FlowDirection.IN)

    # Grow C2 with T2_ket
    C2_d = env.C2.relabel("c2_d", "t2k_u")
    C2g_ket = contract(C2_d, env.T2_ket)  # (c2_l, r_ket, t2k_I)
    C2g_ket_fused = fuse_indices(C2g_ket, 0, 1, "fused", FlowDirection.IN)

    # Ket projector
    P_ket, _, _eps_t = _compute_projector_tensor(
        C1g_ket_fused, C2g_ket_fused, chi, base_charges=base_charges
    )

    # Mid-corners: project ket grown corners
    C1_mid = _apply_projector(P_ket, C1g_ket_fused, base_charges)  # (chi_new, t4k_I)
    C2_mid = _apply_projector(P_ket, C2g_ket_fused, base_charges)  # (chi_new, t2k_I)

    # Grow mid-corners with bra edges
    C1g_bra = contract(C1_mid.relabel("t4k_I", "t4b_I"), env.T4_bra)
    C1g_bra_fused = fuse_indices(C1g_bra, 0, 1, "fused", FlowDirection.IN)

    C2g_bra = contract(C2_mid.relabel("t2k_I", "t2b_I"), env.T2_bra)
    C2g_bra_fused = fuse_indices(C2g_bra, 0, 1, "fused", FlowDirection.IN)

    # Bra projector
    P_bra, _, _eps_t = _compute_projector_tensor(
        C1g_bra_fused, C2g_bra_fused, chi, base_charges=base_charges
    )

    # New corners
    C1_new = _apply_projector(P_bra, C1g_bra_fused, base_charges)  # (chi_new, t4b_u)
    C2_new = _apply_projector(P_bra, C2g_bra_fused, base_charges)  # (chi_new, t2b_d)

    C1_new = C1_new.relabels({"chi_new": "c1_d", "t4b_u": "c1_r"})
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t2b_d": "c2_d"})
    C1_new = _ensure_corner_flows(C1_new, "C1")
    C2_new = _ensure_corner_flows(C2_new, "C2")
    C1_new, _ = max_abs_normalize(C1_new)
    C2_new, _ = max_abs_normalize(C2_new)

    # --- Phase B: Sequential projector application to grown edge ---

    T1g = _grow_edge_no_double_layer(
        env.T1_ket,
        env.T1_bra,
        A,
        A_bar,
        "u",
        "t1k_I",
        "t1b_I",
        ("t1k_l", "l", "L", "d", "D", "t1b_r", "r", "R"),
    )
    T1g = _project_grown_edge_tensor(
        T1g,
        P_ket,
        P_bra,
        left_fuse=("t1k_l", "l", "L"),
        right_fuse=("r", "t1b_r", "R"),
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
    """Bottom move: bra first (C4/C3 connect to T4/T2 bra chi bonds)."""
    base_charges = A.indices[0].charges if isinstance(A, SymmetricTensor) else None

    # --- Phase A: Per-layer projectors and new corners ---

    # Grow C4 with T4_bra (bra first)
    C4_u = env.C4.relabel("c4_u", "t4b_u")
    C4g_bra = contract(C4_u, env.T4_bra)  # (c4_r, t4b_I, l_bra)
    C4g_bra_fused = fuse_indices(C4g_bra, 0, 2, "fused", FlowDirection.IN)

    # Grow C3 with T2_bra
    C3_u = env.C3.relabel("c3_u", "t2b_d")
    C3g_bra = contract(C3_u, env.T2_bra)  # (c3_l, t2b_I, r_bra)
    C3g_bra_fused = fuse_indices(C3g_bra, 0, 2, "fused", FlowDirection.IN)

    # Bra projector
    P_bra, _, _eps_t = _compute_projector_tensor(
        C4g_bra_fused, C3g_bra_fused, chi, base_charges=base_charges
    )

    # Mid-corners: project bra grown corners
    C4_mid = _apply_projector(P_bra, C4g_bra_fused, base_charges)  # (chi_new, t4b_I)
    C3_mid = _apply_projector(P_bra, C3g_bra_fused, base_charges)  # (chi_new, t2b_I)

    # Grow mid-corners with ket edges
    C4g_ket = contract(C4_mid.relabel("t4b_I", "t4k_I"), env.T4_ket)
    C4g_ket_fused = fuse_indices(C4g_ket, 0, 2, "fused", FlowDirection.IN)

    C3g_ket = contract(C3_mid.relabel("t2b_I", "t2k_I"), env.T2_ket)
    C3g_ket_fused = fuse_indices(C3g_ket, 0, 2, "fused", FlowDirection.IN)

    # Ket projector
    P_ket, _, _eps_t = _compute_projector_tensor(
        C4g_ket_fused, C3g_ket_fused, chi, base_charges=base_charges
    )

    # New corners
    C4_new = _apply_projector(P_ket, C4g_ket_fused, base_charges)  # (chi_new, t4k_d)
    C3_new = _apply_projector(P_ket, C3g_ket_fused, base_charges)  # (chi_new, t2k_u)

    C4_new = C4_new.relabels({"chi_new": "c4_r", "t4k_d": "c4_u"})
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t2k_u": "c3_l"})
    C4_new = _ensure_corner_flows(C4_new, "C4")
    C3_new = _ensure_corner_flows(C3_new, "C3")
    C4_new, _ = max_abs_normalize(C4_new)
    C3_new, _ = max_abs_normalize(C3_new)

    # --- Phase B: Sequential projector application to grown edge ---

    T3g = _grow_edge_no_double_layer(
        env.T3_ket,
        env.T3_bra,
        A,
        A_bar,
        "d",
        "t3k_I",
        "t3b_I",
        ("t3k_r", "l", "L", "u", "U", "t3b_l", "r", "R"),
    )
    T3g = _project_grown_edge_tensor(
        T3g,
        P_bra,
        P_ket,
        left_fuse=("t3k_r", "L", "l"),
        right_fuse=("R", "t3b_l", "r"),
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
