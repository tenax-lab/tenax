"""2x2 plaquette enlarged-corner builder for the multisite CTM projector.

Implements the standard CTMRG enlarged-corner construction (Corboz, Penc,
Mila, Lauchli, PRB 84, 041108(R) (2011)). For each plaquette quarter,
contracts one corner C, two adjacent edges T_h and T_v, and the double-
layer site tensor a into a rank-4 tensor with two seam legs (the chi
and D^2 legs that connect to the adjacent quarter in the 2x2).

Used by ``_ctm_tensor_move_*_2x2`` in ``_ctm_tensor_moves.py``.
"""

from __future__ import annotations

from tenax.contraction.contractor import contract
from tenax.core.tensor import Tensor

__all__ = ["_build_enlarged_corner"]


def _build_enlarged_corner(
    C: Tensor,
    T_h: Tensor,
    T_v: Tensor,
    a: Tensor,
    *,
    position: str,
) -> Tensor:
    """Enlarged corner Q = C . T_h . T_v . a for one plaquette quarter.

    For ``position="top_left"``:
      C   = C1   (labels: c1_d, c1_r)
      T_h = T1   (labels: t1_l, u2, t1_r)
      T_v = T4   (labels: t4_d, l2, t4_u)
      a   = double-layer tensor (labels: u2, d2, l2, r2)

    Contractions (auto-pair by shared label):
      C1.c1_r <-> T1.t1_l    (top-left corner connects to T1's left)
      C1.c1_d <-> T4.t4_d    (top-left corner connects to T4's top)
      T1.u2   <-> a.u2       (T1 absorbs a's top virtual)
      T4.l2   <-> a.l2       (T4 absorbs a's left virtual)

    Output free legs (rank-4):
      t1_r -> relabel chi_R   (right seam to Q_TR)
      r2                       (right D^2 seam, original label kept)
      t4_u -> relabel chi_B   (bottom seam to Q_BL)
      d2                       (bottom D^2 seam, original label kept)

    Analogous recipes apply for the other three positions:

      ``"top_right"``:
        C   = C2   (c2_l, c2_d)
        T_h = T1   (t1_l, u2, t1_r)
        T_v = T2   (t2_u, r2, t2_d)
        seams: t1_l -> chi_L, t2_d -> chi_B; l2 / d2 are open D^2 seams.

      ``"bottom_left"``:
        C   = C4   (c4_r, c4_u)
        T_h = T3   (t3_r, d2, t3_l)
        T_v = T4   (t4_d, l2, t4_u)
        seams: t4_d -> chi_T, t3_l -> chi_R; u2 / r2 are open D^2 seams.

      ``"bottom_right"``:
        C   = C3   (c3_u, c3_l)
        T_h = T3   (t3_r, d2, t3_l)
        T_v = T2   (t2_u, r2, t2_d)
        seams: t3_r -> chi_L, t2_u -> chi_T; l2 / u2 are open D^2 seams.
    """
    if position == "top_left":
        # C1.c1_r <-> T1.t1_l
        C_r = C.relabel("c1_r", "t1_l")
        CT_h = contract(C_r, T_h)  # -> (c1_d, u2, t1_r)
        # C1.c1_d <-> T4.t4_d
        T_v_r = T_v.relabel("t4_d", "c1_d")
        CTT = contract(CT_h, T_v_r)  # -> (u2, t1_r, l2, t4_u)
        # T1.u2 <-> a.u2 ; T4.l2 <-> a.l2
        Q = contract(CTT, a)  # -> (t1_r, t4_u, d2, r2) free legs
        # Relabel seams to chi_R, chi_B; r2 / d2 keep original labels.
        return Q.relabels({"t1_r": "chi_R", "t4_u": "chi_B"})

    if position == "top_right":
        # C2.c2_l <-> T1.t1_r
        C_r = C.relabel("c2_l", "t1_r")
        CT_h = contract(C_r, T_h)  # -> (c2_d, t1_l, u2)
        # C2.c2_d <-> T2.t2_u
        T_v_r = T_v.relabel("t2_u", "c2_d")
        CTT = contract(CT_h, T_v_r)  # -> (t1_l, u2, r2, t2_d)
        # T1.u2 <-> a.u2 ; T2.r2 <-> a.r2
        Q = contract(CTT, a)  # -> (t1_l, t2_d, l2, d2) free legs
        return Q.relabels({"t1_l": "chi_L", "t2_d": "chi_B"})

    if position == "bottom_left":
        # C4.c4_u <-> T4.t4_u
        C_r = C.relabel("c4_u", "t4_u")
        CT_v = contract(C_r, T_v)  # -> (c4_r, t4_d, l2)
        # C4.c4_r <-> T3.t3_r
        T_h_r = T_h.relabel("t3_r", "c4_r")
        CTT = contract(CT_v, T_h_r)  # -> (t4_d, l2, d2, t3_l)
        # T3.d2 <-> a.d2 ; T4.l2 <-> a.l2
        Q = contract(CTT, a)  # -> (t4_d, t3_l, u2, r2) free legs
        return Q.relabels({"t4_d": "chi_T", "t3_l": "chi_R"})

    if position == "bottom_right":
        # C3.c3_l <-> T3.t3_l
        C_r = C.relabel("c3_l", "t3_l")
        CT_h = contract(C_r, T_h)  # -> (c3_u, t3_r, d2)
        # C3.c3_u <-> T2.t2_d
        T_v_r = T_v.relabel("t2_d", "c3_u")
        CTT = contract(CT_h, T_v_r)  # -> (t3_r, d2, t2_u, r2)
        # T3.d2 <-> a.d2 ; T2.r2 <-> a.r2
        Q = contract(CTT, a)  # -> (t3_r, t2_u, l2, u2) free legs
        return Q.relabels({"t3_r": "chi_L", "t2_u": "chi_T"})

    raise ValueError(f"unsupported position={position!r}")
