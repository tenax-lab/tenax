"""Split CTM with Tensor protocol — move helpers and directional moves."""

from __future__ import annotations

__all__ = [
    "_FORCE_CLOSED_EDGE",
    "_apply_projector",
    "_build_split_enlarged_corner",
    "_compute_split_plaquette_projector_pair",
    "_ensure_corner_flows",
    "_ensure_edge_flows",
    "_ensure_tensor_flows",
    "_doublelayer_grown_corner",
    "_factorize_projector",
    "_fused_charge_permutation",
    "_grow_and_project_bounded_lr",
    "_grow_and_project_edge_lr",
    "_grow_edge_halves",
    "_grow_edge_no_double_layer",
    "_project_grown_edge_tensor_lr",
    "_reembed_target_for_projector",
    "_select_bond_entries",
    "_split_ctm_absorb_bottom_2plaq",
    "_split_ctm_absorb_left_2plaq",
    "_split_ctm_absorb_right_2plaq",
    "_split_ctm_absorb_top_2plaq",
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
from tenax.algorithms._ctm_tensor_moves import (
    _apply_proj_unfused,
    _half_to_chi_new_bot,
    _half_to_chi_new_top,
)
from tenax.algorithms._ctm_tensor_projector_2x2 import _compute_2x2_projector
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

# When True, the four split moves grow the FULL closed chi^2*D^6 double-layer
# edge and project it with :func:`_project_grown_edge_tensor_lr` (the Task-4
# correctness-first path).  When False (default), they route through the
# memory-bounded :func:`_grow_and_project_bounded_lr` (chi^2*D^4) path, which
# reproduces the closed result to machine precision.  Tests flip this flag to
# prove bounded == closed; production always uses the bounded default.
_FORCE_CLOSED_EDGE = False

# ------------------------------------------------------------------ #
# Split enlarged corner (parity with fused _build_enlarged_corner)      #
# ------------------------------------------------------------------ #


def _fuse_ket_bra(
    T: Tensor,
    ket_label: str,
    bra_label: str,
    fused_label: str,
    fused_flow: FlowDirection,
) -> Tensor:
    """Fuse a (ket, bra) virtual-leg pair KET-SLOW into one D^2 seam.

    ``_build_double_layer_tensor`` fuses each virtual pair with the ket leg
    slow-varying (``a8 = contract(A, A_bar)`` puts A's legs first, so ``u2 =
    u_ket * D + u_bra``).  :func:`fuse_indices` makes the LOWER-indexed axis
    slow, so we transpose the ket leg to sit immediately before the bra leg
    before fusing, guaranteeing the same ket-slow layout regardless of the
    incoming axis order.
    """
    labels = list(T.labels())
    ket_pos = labels.index(ket_label)
    bra_pos = labels.index(bra_label)
    # Bring ket immediately before bra (ket slow, bra fast).
    rest = [i for i in range(len(labels)) if i not in (ket_pos, bra_pos)]
    perm = tuple(
        [ket_pos, bra_pos] + rest
    )  # ket at 0 (slow), bra at 1 (fast), then rest
    T = T.transpose(perm)
    return fuse_indices(T, 0, 1, fused_label, fused_flow)


def _build_split_enlarged_corner(
    C: Tensor,
    T_h_ket: Tensor,
    T_h_bra: Tensor,
    T_v_ket: Tensor,
    T_v_bra: Tensor,
    A: Tensor,
    A_bar: Tensor,
    *,
    position: str,
) -> Tensor:
    """Split enlarged corner. Returns the SAME rank-4 object as the fused
    :func:`_build_enlarged_corner`, assembled from ket/bra split edges + a
    physical double layer (A ket, A_bar bra) with the phys index traced.

    Output free legs match the fused recipe's LABEL SET exactly (axis order is
    not significant — ``_compute_2x2_projector`` indexes by label):
      top_left     -> {chi_R, r2, chi_B, d2}
      top_right    -> {chi_L, l2, chi_B, d2}
      bottom_left  -> {chi_T, u2, chi_R, r2}
      bottom_right -> {chi_L, l2, chi_T, u2}

    The physical double layer is built by contracting the ket edges with the
    ket virtual legs of ``A`` and the bra edges with the bra virtual legs of
    ``A_bar``, tracing the shared ``phys``.  Each surviving ket/bra
    virtual-leg pair is fused KET-SLOW (matching
    :func:`_build_double_layer_tensor`) into the D^2 seam.
    """
    if position == "top_left":
        # Ket layer: C1.c1_r <-> T1k.t1k_l ; C1.c1_d <-> T4k.t4k_d ;
        # A.u <-> u_ket ; A.l <-> l_ket.  Bra layer joins via interlayer bonds.
        ket = contract(C.relabel("c1_r", "t1k_l"), T_h_ket)  # (c1_d,u_ket,t1k_I)
        ket = contract(
            ket.relabel("c1_d", "t4k_d"), T_v_ket
        )  # (u_ket,t1k_I,l_ket,t4k_I)
        A_ket = A.relabels({"u": "u_ket", "l": "l_ket"})
        ket = contract(ket, A_ket)  # (t1k_I, t4k_I, d, r, phys)
        # Bra layer joins interlayer bonds; A_bar supplies u_bra/l_bra + D_bra/R_bra.
        ket = contract(ket.relabel("t1k_I", "t1b_I"), T_h_bra)  # +(u_bra,t1b_r)
        ket = contract(ket.relabel("t4k_I", "t4b_I"), T_v_bra)  # +(l_bra,t4b_u)
        A_bra = A_bar.relabels({"u": "u_bra", "l": "l_bra", "d": "D_bra", "r": "R_bra"})
        Q = contract(ket, A_bra)  # (t1b_r, t4b_u, d, r, D_bra, R_bra)
        Q = _fuse_ket_bra(Q, "d", "D_bra", "d2", FlowDirection.OUT)
        Q = _fuse_ket_bra(Q, "r", "R_bra", "r2", FlowDirection.OUT)
        return Q.relabels({"t1b_r": "chi_R", "t4b_u": "chi_B"})

    if position == "top_right":
        # C2.c2_l <-> T1's right = t1b_r (BRA) ; C2.c2_d <-> T2's top = t2k_u (KET).
        # Output: t1k_l -> chi_L ; t2b_d -> chi_B.  A.u/r are the contracted legs.
        ket = contract(C.relabel("c2_l", "t1b_r"), T_h_bra)  # (c2_d,u_bra,t1b_I)
        ket = contract(
            ket.relabel("c2_d", "t2k_u"), T_v_ket
        )  # (u_bra,t1b_I,r_ket,t2k_I)
        # ket layer contributions: T1_ket (u_ket) joins via interlayer; T2_ket (r_ket) here.
        ket = contract(ket.relabel("t1b_I", "t1k_I"), T_h_ket)  # +(t1k_l,u_ket)
        A_ket = A.relabels({"u": "u_ket", "r": "r_ket"})
        ket = contract(ket, A_ket)  # trace u_ket,r_ket -> (u_bra,t2k_I,t1k_l,d,l,phys)
        ket = contract(ket.relabel("t2k_I", "t2b_I"), T_v_bra)  # +(r_bra,t2b_d)
        A_bra = A_bar.relabels({"u": "u_bra", "r": "r_bra", "d": "D_bra", "l": "L_bra"})
        Q = contract(ket, A_bra)  # (t1k_l, d, l, t2b_d, D_bra, L_bra)
        Q = _fuse_ket_bra(Q, "d", "D_bra", "d2", FlowDirection.OUT)
        Q = _fuse_ket_bra(Q, "l", "L_bra", "l2", FlowDirection.IN)
        return Q.relabels({"t1k_l": "chi_L", "t2b_d": "chi_B"})

    if position == "bottom_left":
        # C4.c4_r <-> T4's up = t4b_u (BRA) ; C4.c4_u <-> T3's right = t3k_r (KET).
        # Output: t4k_d -> chi_T ; t3b_l -> chi_R.  A.l (T4) / A.d (T3) contracted.
        ket = contract(C.relabel("c4_r", "t4b_u"), T_v_bra)  # (c4_u,l_bra,t4b_I)
        ket = contract(
            ket.relabel("c4_u", "t3k_r"), T_h_ket
        )  # (l_bra,t4b_I,d_ket,t3k_I)
        ket = contract(ket.relabel("t4b_I", "t4k_I"), T_v_ket)  # +(t4k_d,l_ket)
        A_ket = A.relabels({"l": "l_ket", "d": "d_ket"})
        ket = contract(ket, A_ket)  # trace l_ket,d_ket -> (l_bra,t3k_I,t4k_d,u,r,phys)
        ket = contract(ket.relabel("t3k_I", "t3b_I"), T_h_bra)  # +(d_bra,t3b_l)
        A_bra = A_bar.relabels({"l": "l_bra", "d": "d_bra", "u": "U_bra", "r": "R_bra"})
        Q = contract(ket, A_bra)  # (t4k_d, u, r, t3b_l, U_bra, R_bra)
        Q = _fuse_ket_bra(Q, "u", "U_bra", "u2", FlowDirection.IN)
        Q = _fuse_ket_bra(Q, "r", "R_bra", "r2", FlowDirection.OUT)
        return Q.relabels({"t4k_d": "chi_T", "t3b_l": "chi_R"})

    if position == "bottom_right":
        # C3.c3_l <-> T3's left = t3b_l (BRA) ; C3.c3_u <-> T2's bottom = t2b_d (BRA).
        # Output: t3k_r -> chi_L ; t2k_u -> chi_T.  A.d (T3) / A.r (T2) contracted.
        ket = contract(C.relabel("c3_l", "t3b_l"), T_h_bra)  # (c3_u,d_bra,t3b_I)
        ket = contract(
            ket.relabel("c3_u", "t2b_d"), T_v_bra
        )  # (d_bra,t3b_I,r_bra,t2b_I)
        ket = contract(ket.relabel("t3b_I", "t3k_I"), T_h_ket)  # +(t3k_r,d_ket)
        ket = contract(ket.relabel("t2b_I", "t2k_I"), T_v_ket)  # +(t2k_u,r_ket)
        A_ket = A.relabels({"d": "d_ket", "r": "r_ket"})
        ket = contract(
            ket, A_ket
        )  # trace d_ket,r_ket -> (d_bra,r_bra,t3k_r,t2k_u,u,l,phys)
        A_bra = A_bar.relabels({"d": "d_bra", "r": "r_bra", "u": "U_bra", "l": "L_bra"})
        Q = contract(ket, A_bra)  # (t3k_r, t2k_u, u, l, U_bra, L_bra)
        Q = _fuse_ket_bra(Q, "u", "U_bra", "u2", FlowDirection.IN)
        Q = _fuse_ket_bra(Q, "l", "L_bra", "l2", FlowDirection.IN)
        return Q.relabels({"t3k_r": "chi_L", "t2k_u": "chi_T"})

    raise ValueError(f"unsupported position={position!r}")


# ------------------------------------------------------------------ #
# Split plaquette projector pair (parity with fused)                   #
# ------------------------------------------------------------------ #


def _compute_split_plaquette_projector_pair(
    env_TL: SplitCTMTensorEnv,
    env_TR: SplitCTMTensorEnv,
    env_BL: SplitCTMTensorEnv,
    env_BR: SplitCTMTensorEnv,
    A_TL: Tensor,
    Abar_TL: Tensor,
    A_TR: Tensor,
    Abar_TR: Tensor,
    A_BL: Tensor,
    Abar_BL: Tensor,
    A_BR: Tensor,
    Abar_BR: Tensor,
    chi: int,
    direction: str,
    base_charges: np.ndarray | None = None,
) -> tuple[Tensor, Tensor, jax.Array, jax.Array]:
    """Split twin of :func:`_compute_plaquette_projector_pair`.

    Builds the four split enlarged corners (identical rank-4 objects to the
    fused path via :func:`_build_split_enlarged_corner`) and feeds them to the
    reused Fishman cross-projector :func:`_compute_2x2_projector` verbatim.

    Returns ``(P_top, P_bot, eps_T, smallest_S)`` with the same layout as
    :func:`_compute_plaquette_projector_pair`:

    - ``P_top`` labels ``("fused", "chi_new")``, flow IN on ``"fused"``.
    - ``P_bot`` labels ``("chi_new", "fused")``, flow IN on ``"fused"``.
    """
    Q_TL = _build_split_enlarged_corner(
        env_TL.C1,
        env_TL.T1_ket,
        env_TL.T1_bra,
        env_TL.T4_ket,
        env_TL.T4_bra,
        A_TL,
        Abar_TL,
        position="top_left",
    )
    Q_TR = _build_split_enlarged_corner(
        env_TR.C2,
        env_TR.T1_ket,
        env_TR.T1_bra,
        env_TR.T2_ket,
        env_TR.T2_bra,
        A_TR,
        Abar_TR,
        position="top_right",
    )
    Q_BL = _build_split_enlarged_corner(
        env_BL.C4,
        env_BL.T3_ket,
        env_BL.T3_bra,
        env_BL.T4_ket,
        env_BL.T4_bra,
        A_BL,
        Abar_BL,
        position="bottom_left",
    )
    Q_BR = _build_split_enlarged_corner(
        env_BR.C3,
        env_BR.T3_ket,
        env_BR.T3_bra,
        env_BR.T2_ket,
        env_BR.T2_bra,
        A_BR,
        Abar_BR,
        position="bottom_right",
    )
    P_top_raw, P_bot_raw, eps_T, smallest_S = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi, direction=direction, base_charges=base_charges
    )
    return (
        _half_to_chi_new_top(P_top_raw),
        _half_to_chi_new_bot(P_bot_raw),
        eps_T,
        smallest_S,
    )


# ------------------------------------------------------------------ #
# 2x2 split absorption (four directional twins of the fused            #
# _ctm_tensor_absorb_*_2plaq, on ket/bra split edges)                  #
# ------------------------------------------------------------------ #


def _grow_split_corner_2x2(
    C,
    c_leg,
    join_to,
    T_first,
    T_second,
    first_I,
    second_I,
    D_ket,
    D_bra,
    d2_label,
    d2_flow,
):
    """Grow a 2x2 corner from split ket/bra edges, keeping the env chi legs
    SEPARATE and fusing only the ``(D_ket, D_bra)`` virtual pair into a single
    D^2 seam ``d2_label`` (ket-slow, matching the fused double layer).

    ``C.c_leg`` joins ``T_first.join_to``; the second layer joins over the
    interlayer bond ``first_I -> second_I``.  Returns the 3-leg grown corner
    (mirrors the fused ``contract(C_r, T)`` result, with the env legs unfused).
    """
    Cg = contract(C.relabel(c_leg, join_to), T_first)
    Cg = contract(Cg.relabel(first_I, second_I), T_second)
    return _fuse_ket_bra(Cg, D_ket, D_bra, d2_label, d2_flow)


def _grow_split_edge_2x2(
    T_ket,
    T_bra,
    A,
    A_bar,
    *,
    absorbed,
    ket_I,
    bra_I,
    surv,
    proj_a,
    proj_a_label,
    proj_a_flow,
    proj_b,
    proj_b_label,
    proj_b_flow,
):
    """Grow an edge double layer (ket = ``T_ket . A``, bra = ``T_bra . A_bar``),
    keeping the surviving virtual pair SPLIT and fusing the two projected
    seams (``proj_a``, ``proj_b``) into D^2 labels for leg-by-leg projection.

    ``absorbed`` is the A virtual direction already carried by the edge (traced
    into ``T_*``'s ``{absorbed}_ket`` / ``{absorbed}_bra`` leg); ``surv`` is the
    virtual kept split (ket label ``surv``, bra label ``{surv}_bra``).  Returns
    the grown edge with legs
    ``(env_ket_outer, surv, proj_a_label, env_bra_outer, {surv}_bra, proj_b_label)``.
    """
    A_k = A.relabels({absorbed: f"{absorbed}_ket"})
    Tk = contract(T_ket, A_k)  # joins {absorbed}_ket; frees surv/proj_a/proj_b/phys
    A_b = A_bar.relabels(
        {
            absorbed: f"{absorbed}_bra",
            surv: f"{surv}_bra",
            proj_a: f"{proj_a}_bra",
            proj_b: f"{proj_b}_bra",
        }
    )
    Tb = contract(T_bra, A_b)  # joins {absorbed}_bra
    Tg = contract(Tk.relabel(ket_I, bra_I), Tb)  # join interlayer + trace phys
    Tg = _fuse_ket_bra(Tg, proj_a, f"{proj_a}_bra", proj_a_label, proj_a_flow)
    Tg = _fuse_ket_bra(Tg, proj_b, f"{proj_b}_bra", proj_b_label, proj_b_flow)
    return Tg


def _split_ctm_absorb_bottom_2plaq(
    env_src, A, A_bar, P_top_left, P_bot_left, P_top_curr, P_bot_curr, chi_I
):
    """Split BOTTOM absorption -> (C4_new, T3_ket_new, T3_bra_new, C3_new).

    Twin of the fused ``_ctm_tensor_absorb_bottom_2plaq`` (``C3.c3_u<->T2.t2_d``
    convention, #670/#674): projector halves applied via ``_apply_proj_unfused``,
    ket/bra edge halves grown then SVD-split over ``chi_I``.
    """
    C4g = _grow_split_corner_2x2(
        env_src.C4,
        "c4_r",
        "t4b_u",
        env_src.T4_bra,
        env_src.T4_ket,
        "t4b_I",
        "t4k_I",
        "l_ket",
        "l_bra",
        "l2",
        FlowDirection.IN,
    )
    C4_new = _apply_proj_unfused(P_bot_left, C4g, "c4_u", "l2")
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t4k_d": "c4_u"})

    C3g = _grow_split_corner_2x2(
        env_src.C3,
        "c3_u",
        "t2b_d",
        env_src.T2_bra,
        env_src.T2_ket,
        "t2b_I",
        "t2k_I",
        "r_ket",
        "r_bra",
        "r2",
        FlowDirection.OUT,
    )
    C3_new = _apply_proj_unfused(P_top_curr, C3g, "c3_l", "r2")
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t2k_u": "c3_l"})

    T3g = _grow_split_edge_2x2(
        env_src.T3_ket,
        env_src.T3_bra,
        A,
        A_bar,
        absorbed="d",
        ket_I="t3k_I",
        bra_I="t3b_I",
        surv="u",
        proj_a="l",
        proj_a_label="l2",
        proj_a_flow=FlowDirection.IN,
        proj_b="r",
        proj_b_label="r2",
        proj_b_flow=FlowDirection.OUT,
    )
    step = _apply_proj_unfused(P_top_left, T3g, "t3k_r", "l2")
    T3g = _apply_proj_unfused(
        P_bot_curr, step, "t3b_l", "r2", chi_new="chi_new_r", env_first=True
    )
    T3_ket_new, T3_bra_new = _svd_split_edge_tensor(
        T3g,
        left_labels=["chi_new", "u"],
        right_labels=["u_bra", "chi_new_r"],
        chi_I=chi_I,
        ket_relabels={"chi_new": "t3k_r", "u": "d_ket", "_svd_bond": "t3k_I"},
        bra_relabels={"_svd_bond": "t3b_I", "u_bra": "d_bra", "chi_new_r": "t3b_l"},
    )
    C4_new = _ensure_corner_flows(C4_new, "C4")
    C3_new = _ensure_corner_flows(C3_new, "C3")
    T3_ket_new, T3_bra_new = _ensure_edge_flows(T3_ket_new, T3_bra_new, "T3")
    return C4_new, T3_ket_new, T3_bra_new, C3_new


def _split_ctm_absorb_left_2plaq(
    env_src, A, A_bar, P_top_above, P_bot_above, P_top_curr, P_bot_curr, chi_I
):
    """Split LEFT absorption -> (C1_new, T4_ket_new, T4_bra_new, C4_new)."""
    C1g = _grow_split_corner_2x2(
        env_src.C1,
        "c1_r",
        "t1k_l",
        env_src.T1_ket,
        env_src.T1_bra,
        "t1k_I",
        "t1b_I",
        "u_ket",
        "u_bra",
        "u2",
        FlowDirection.IN,
    )
    C1_new = _apply_proj_unfused(P_bot_above, C1g, "c1_d", "u2")
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t1b_r": "c1_r"})

    C4g = _grow_split_corner_2x2(
        env_src.C4,
        "c4_u",
        "t3k_r",
        env_src.T3_ket,
        env_src.T3_bra,
        "t3k_I",
        "t3b_I",
        "d_ket",
        "d_bra",
        "d2",
        FlowDirection.OUT,
    )
    C4_new = _apply_proj_unfused(P_top_curr, C4g, "c4_r", "d2")
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t3b_l": "c4_u"})

    T4g = _grow_split_edge_2x2(
        env_src.T4_ket,
        env_src.T4_bra,
        A,
        A_bar,
        absorbed="l",
        ket_I="t4k_I",
        bra_I="t4b_I",
        surv="r",
        proj_a="u",
        proj_a_label="u2",
        proj_a_flow=FlowDirection.IN,
        proj_b="d",
        proj_b_label="d2",
        proj_b_flow=FlowDirection.OUT,
    )
    step = _apply_proj_unfused(P_top_above, T4g, "t4k_d", "u2")
    T4g = _apply_proj_unfused(
        P_bot_curr, step, "t4b_u", "d2", chi_new="chi_new_r", env_first=True
    )
    T4_ket_new, T4_bra_new = _svd_split_edge_tensor(
        T4g,
        left_labels=["chi_new", "r"],
        right_labels=["r_bra", "chi_new_r"],
        chi_I=chi_I,
        ket_relabels={"chi_new": "t4k_d", "r": "l_ket", "_svd_bond": "t4k_I"},
        bra_relabels={"_svd_bond": "t4b_I", "r_bra": "l_bra", "chi_new_r": "t4b_u"},
    )
    C1_new = _ensure_corner_flows(C1_new, "C1")
    C4_new = _ensure_corner_flows(C4_new, "C4")
    T4_ket_new, T4_bra_new = _ensure_edge_flows(T4_ket_new, T4_bra_new, "T4")
    return C1_new, T4_ket_new, T4_bra_new, C4_new


def _split_ctm_absorb_right_2plaq(
    env_src, A, A_bar, P_top_above, P_bot_above, P_top_curr, P_bot_curr, chi_I
):
    """Split RIGHT absorption -> (C2_new, T2_ket_new, T2_bra_new, C3_new)."""
    C2g = _grow_split_corner_2x2(
        env_src.C2,
        "c2_l",
        "t1b_r",
        env_src.T1_bra,
        env_src.T1_ket,
        "t1b_I",
        "t1k_I",
        "u_ket",
        "u_bra",
        "u2",
        FlowDirection.IN,
    )
    C2_new = _apply_proj_unfused(P_top_above, C2g, "c2_d", "u2")
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t1k_l": "c2_d"})

    C3g = _grow_split_corner_2x2(
        env_src.C3,
        "c3_u",
        "t3b_l",
        env_src.T3_bra,
        env_src.T3_ket,
        "t3b_I",
        "t3k_I",
        "d_ket",
        "d_bra",
        "d2",
        FlowDirection.OUT,
    )
    C3_new = _apply_proj_unfused(P_bot_curr, C3g, "c3_l", "d2")
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t3k_r": "c3_l"})

    T2g = _grow_split_edge_2x2(
        env_src.T2_ket,
        env_src.T2_bra,
        A,
        A_bar,
        absorbed="r",
        ket_I="t2k_I",
        bra_I="t2b_I",
        surv="l",
        proj_a="u",
        proj_a_label="u2",
        proj_a_flow=FlowDirection.IN,
        proj_b="d",
        proj_b_label="d2",
        proj_b_flow=FlowDirection.OUT,
    )
    step = _apply_proj_unfused(P_bot_above, T2g, "t2k_u", "u2")
    T2g = _apply_proj_unfused(
        P_top_curr, step, "t2b_d", "d2", chi_new="chi_new_r", env_first=True
    )
    T2_ket_new, T2_bra_new = _svd_split_edge_tensor(
        T2g,
        left_labels=["chi_new", "l"],
        right_labels=["l_bra", "chi_new_r"],
        chi_I=chi_I,
        ket_relabels={"chi_new": "t2k_u", "l": "r_ket", "_svd_bond": "t2k_I"},
        bra_relabels={"_svd_bond": "t2b_I", "l_bra": "r_bra", "chi_new_r": "t2b_d"},
    )
    C2_new = _ensure_corner_flows(C2_new, "C2")
    C3_new = _ensure_corner_flows(C3_new, "C3")
    T2_ket_new, T2_bra_new = _ensure_edge_flows(T2_ket_new, T2_bra_new, "T2")
    return C2_new, T2_ket_new, T2_bra_new, C3_new


def _split_ctm_absorb_top_2plaq(
    env_src, A, A_bar, P_top_left, P_bot_left, P_top_curr, P_bot_curr, chi_I
):
    """Split TOP absorption -> (C1_new, T1_ket_new, T1_bra_new, C2_new)."""
    C1g = _grow_split_corner_2x2(
        env_src.C1,
        "c1_d",
        "t4k_d",
        env_src.T4_ket,
        env_src.T4_bra,
        "t4k_I",
        "t4b_I",
        "l_ket",
        "l_bra",
        "l2",
        FlowDirection.IN,
    )
    C1_new = _apply_proj_unfused(P_top_left, C1g, "c1_r", "l2")
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t4b_u": "c1_r"})

    C2g = _grow_split_corner_2x2(
        env_src.C2,
        "c2_d",
        "t2k_u",
        env_src.T2_ket,
        env_src.T2_bra,
        "t2k_I",
        "t2b_I",
        "r_ket",
        "r_bra",
        "r2",
        FlowDirection.OUT,
    )
    C2_new = _apply_proj_unfused(P_bot_curr, C2g, "c2_l", "r2")
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t2b_d": "c2_d"})

    T1g = _grow_split_edge_2x2(
        env_src.T1_ket,
        env_src.T1_bra,
        A,
        A_bar,
        absorbed="u",
        ket_I="t1k_I",
        bra_I="t1b_I",
        surv="d",
        proj_a="l",
        proj_a_label="l2",
        proj_a_flow=FlowDirection.IN,
        proj_b="r",
        proj_b_label="r2",
        proj_b_flow=FlowDirection.OUT,
    )
    step = _apply_proj_unfused(P_bot_left, T1g, "t1k_l", "l2")
    T1g = _apply_proj_unfused(
        P_top_curr, step, "t1b_r", "r2", chi_new="chi_new_r", env_first=True
    )
    T1_ket_new, T1_bra_new = _svd_split_edge_tensor(
        T1g,
        left_labels=["chi_new", "d"],
        right_labels=["d_bra", "chi_new_r"],
        chi_I=chi_I,
        ket_relabels={"chi_new": "t1k_l", "d": "u_ket", "_svd_bond": "t1k_I"},
        bra_relabels={"_svd_bond": "t1b_I", "d_bra": "u_bra", "chi_new_r": "t1b_r"},
    )
    C1_new = _ensure_corner_flows(C1_new, "C1")
    C2_new = _ensure_corner_flows(C2_new, "C2")
    T1_ket_new, T1_bra_new = _ensure_edge_flows(T1_ket_new, T1_bra_new, "T1")
    return C1_new, T1_ket_new, T1_bra_new, C2_new


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

    # Layout is now (fused, chi_new); symmetry is uniform across all indices.
    sym = P.indices[0].symmetry
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
        # TODO(#463 Phase 2-4): add SymmetricTensor block-sparse regularized SVD
        # as a follow-up so the SymmetricTensor branches below also get a finite
        # adjoint on rank-deficient blocks.
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


def _projector_edge_operator(
    P_u: Tensor,
    env_label: str,
    ketD_label: str,
    braD_label: str,
    out_chi: str,
    side: str,
) -> Tensor:
    """Relabel an unfused 4-leg projector into a half-edge operator.

    ``P_u`` is the unfused biorthogonal corner projector (output of
    :func:`_unfuse_projector_fused`) with legs ``(env, ketD, braD, chi_new)``.
    The closed :func:`_project_grown_edge_tensor_lr` applies the **left**
    (C1-side) end with ``P_1.bar()`` and the **right** (C4-side) end with
    ``P_2`` (no dagger); this builds the matching operator and relabels its
    three contracted legs to the half-edge labels ``env_label`` /
    ``ketD_label`` / ``braD_label`` and its open leg to ``out_chi``.

    No SVD factorization is performed.  The earlier
    :func:`_factorize_projector` → recombine round-trip was a memory-layout
    artifact (the factors were contracted straight back into ``P``), and its
    ``sqrt(s)`` / rank-deficient-SVD backward divided by the projector's
    **zero** singular values — making the split explicit-AD gradient
    non-finite on the degenerate 1-site double-layer corner (#463 split
    backward AD-stability).  Applying ``P_u`` directly is forward-identical,
    cheaper (no per-move SVD), and degeneracy-safe.

    Args:
        side: ``"left"`` → ``P_u.bar()`` (mirrors ``P_1.bar()``);
              ``"right"`` → bare ``P_u`` (mirrors un-daggered ``P_2``).

    Returns a 4-leg operator ``(env_label, ketD_label, braD_label, out_chi)``.

    DenseTensor only; ``bar`` over the SymmetricTensor projector does not
    preserve the order-dependent Koszul signs (#641 / #463 Ph 2-4).
    """
    op = P_u.bar() if side == "left" else P_u
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
    P_1u: Tensor,
    P_2u: Tensor,
    contracted_leg: str,
    ket_I_label: str,
    bra_I_label: str,
    left_fuse: tuple[str, str, str],
    right_fuse: tuple[str, str, str],
) -> Tensor:
    """Memory-bounded biorthogonal edge application (chi^2 * D^4).

    Applies the unfused biorthogonal projector pair ``(P_1u, P_2u)`` — the
    C1-side ``P_1`` and C4-side ``P_2`` corner projectors unfused into 4-leg
    ``(env, ketD, braD, chi_new)`` form via :func:`_unfuse_projector_fused`.

    Reproduces :func:`_project_grown_edge_tensor_lr` applied to the closed
    :func:`_grow_edge_no_double_layer` edge to machine precision, but never
    forms the ``chi^2 * D^6`` closed edge: each 4-leg projector operator is
    absorbed into the open half-edge that holds the majority of its legs before
    the interlayer join (peak ``chi^2 * D^3 * d``).

    ``left_fuse`` / ``right_fuse`` are ``(env_chi, ketD, braD)`` triples — the
    same tuples the closed ``_lr`` path fuses.  The projectors carry the
    generic ``env``/``ketD``/``braD``/``chi_new`` labels from
    :func:`_unfuse_projector_fused`.

    The projectors are applied directly (no :func:`_factorize_projector` SVD
    round-trip): that factorization was a vestigial memory-layout step whose
    rank-deficient backward made the split explicit-AD gradient non-finite on
    the degenerate 1-site corner (see :func:`_projector_edge_operator`).

    Returns the 4-leg projected edge ``(left_chi, mid_ket, mid_bra,
    right_chi)``.  DenseTensor only.
    """
    ket_half, bra_half = _grow_edge_halves(
        T_ket, T_bra, A, A_bar, contracted_leg, ket_I_label, bra_I_label
    )

    la, lb, lc = left_fuse
    ra, rb, rc = right_fuse
    # Left end: mirrors P_1.bar(); right end: mirrors un-daggered P_2.
    P_left = _projector_edge_operator(P_1u, la, lb, lc, "left_chi", "left")
    P_right = _projector_edge_operator(P_2u, ra, rb, rc, "right_chi", "right")

    def _absorb(op: Tensor) -> Tensor:
        op_labels = set(op.labels())
        in_ket = len(op_labels & set(ket_half.labels()))
        in_bra = len(op_labels & set(bra_half.labels()))
        return contract(op, ket_half) if in_ket >= in_bra else contract(op, bra_half)

    return contract(_absorb(P_left), _absorb(P_right))


def _grow_and_project_edge_lr(
    T_ket: Tensor,
    T_bra: Tensor,
    A: Tensor,
    A_bar: Tensor,
    P_1: Tensor,
    P_2: Tensor,
    contracted_leg: str,
    ket_I_label: str,
    bra_I_label: str,
    grow_output_labels: tuple[str, ...],
    left_fuse: tuple[str, str, str],
    right_fuse: tuple[str, str, str],
) -> Tensor:
    """Grow + project a T-edge with a biorthogonal pair ``(P_1, P_2)``.

    Owns the whole Phase-B path shared by the four directional moves, mirroring
    how the fused path uses a single
    :func:`tenax.algorithms._ctm_tensor_moves._apply_projector_with_reembed`:

    - For SymmetricTensor inputs (or when :data:`_FORCE_CLOSED_EDGE` is set),
      builds the closed ``chi^2 * D^6`` double-layer edge and projects it with
      :func:`_project_grown_edge_tensor_lr`, preserving the fermionic
      Koszul-sign convention.
    - For DenseTensor inputs, unfuses each projector and routes through the
      memory-bounded :func:`_grow_and_project_bounded_lr` (``chi^2 * D^4``-
      bounded, issue #641), which reproduces the closed result to machine
      precision without forming the ``chi^2 * D^6`` edge.

    Returns the 4-leg projected edge ``(left_chi, mid_ket, mid_bra,
    right_chi)``.
    """
    if _FORCE_CLOSED_EDGE or not isinstance(A, DenseTensor):
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
        return _project_grown_edge_tensor_lr(
            Tg, P_1, P_2, left_fuse=left_fuse, right_fuse=right_fuse
        )

    D = A.indices[0].dim
    P_1u = _unfuse_projector_fused(
        P_1, P_1.indices[0].dim // (D * D), D, "env", "ketD", "braD"
    )
    P_2u = _unfuse_projector_fused(
        P_2, P_2.indices[0].dim // (D * D), D, "env", "ketD", "braD"
    )
    return _grow_and_project_bounded_lr(
        T_ket,
        T_bra,
        A,
        A_bar,
        P_1u,
        P_2u,
        contracted_leg,
        ket_I_label,
        bra_I_label,
        left_fuse,
        right_fuse,
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
    T4g = _grow_and_project_edge_lr(
        env.T4_ket,
        env.T4_bra,
        A,
        A_bar,
        P_1,
        P_2,
        "l",
        "t4k_I",
        "t4b_I",
        ("t4k_d", "u", "U", "r", "R", "t4b_u", "d", "D"),
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
    T2g = _grow_and_project_edge_lr(
        env.T2_ket,
        env.T2_bra,
        A,
        A_bar,
        P_1,
        P_2,
        "r",
        "t2k_I",
        "t2b_I",
        ("t2k_u", "u", "U", "l", "L", "t2b_d", "d", "D"),
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
    T1g = _grow_and_project_edge_lr(
        env.T1_ket,
        env.T1_bra,
        A,
        A_bar,
        P_1,
        P_2,
        "u",
        "t1k_I",
        "t1b_I",
        ("t1k_l", "l", "L", "d", "D", "t1b_r", "r", "R"),
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
    T3g = _grow_and_project_edge_lr(
        env.T3_ket,
        env.T3_bra,
        A,
        A_bar,
        P_1,
        P_2,
        "d",
        "t3k_I",
        "t3b_I",
        ("t3k_r", "l", "L", "u", "U", "t3b_l", "r", "R"),
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
