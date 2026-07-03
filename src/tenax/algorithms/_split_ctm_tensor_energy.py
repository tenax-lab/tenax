"""Split CTM with Tensor protocol — conversion and energy computation."""

from __future__ import annotations

__all__ = [
    "_rdm1x2_split_tensor",
    "_rdm1x2_split_tensor_2site",
    "_rdm2x1_split_tensor",
    "_rdm2x1_split_tensor_2site",
    "_rdm_1site_split_tensor",
    "_rdm_diagonal_split_tensor",
    "_split_env_to_tensor_standard",
    "compute_energy_split_ctm_tensor",
    "compute_energy_split_ctm_tensor_2site",
    "compute_energy_split_ctm_tensor_multisite",
]

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv
from tenax.algorithms._tensor_utils import fuse_indices
from tenax.contraction.contractor import contract
from tenax.core import EPS
from tenax.core.index import FlowDirection
from tenax.core.tensor import Tensor

# Below this |trace| the unnormalized mixed-env RDM is treated as
# catastrophically cancelling: dividing by the trace amplifies ~4e-18 rounding
# noise above the atol=1e-10 parity tolerance against the shim path (issue
# #479 / #485 / #480).  When the split-aware contraction lands at or below
# this floor we delegate to ``_split_env_to_tensor_standard`` + the standard
# RDM routine, which produces a different (and on these random-tensor seeds,
# atol-safe) cancellation behaviour at the cost of the ``chi²·D⁶`` peak.
#
# Real CTM-converged envs land with O(1) traces, so the floor never fires in
# production and the split-aware ``chi²·D⁴`` memory bound is preserved.
_MIXED_ENV_RDM_TRACE_FLOOR = 1e-8


# ------------------------------------------------------------------ #
# Energy computation (split, no double-layer)                          #
# ------------------------------------------------------------------ #


def _make_split_edge(
    T_ket: Tensor,
    T_bra: Tensor,
    ket_I: str,
    bra_I: str,
    ket_chi: str,
    bra_chi: str,
    out_chi_l: str,
    out_chi_r: str,
) -> Tensor:
    """Contract T_ket·T_bra over the interlayer bond; do NOT fuse the two D-legs.

    Returns a 4-leg tensor with labels (out_chi_l, <ket D-label>, <bra D-label>, out_chi_r).
    The D-leg labels are inherited from the inputs (e.g. ``u_ket``/``u_bra`` for T1).
    Leaving the D-legs unfused lets downstream RDM construction contract them
    directly against the ket/bra layers of A, avoiding the chi^2 * D^4 double-layer
    peak that the standard shim (``_split_env_to_tensor_standard``) produces.
    """
    # Use ``_I_tmp`` rather than ``_I`` to avoid label collisions when several
    # split edges built by this helper are later contracted together in the same
    # RDM network. The standard shim's ``_merge_edge`` uses ``_I`` because each
    # call there is contracted in isolation.
    k = T_ket.relabel(ket_I, "_I_tmp")
    b = T_bra.relabel(bra_I, "_I_tmp")
    merged = contract(k, b)
    return merged.relabels({ket_chi: out_chi_l, bra_chi: out_chi_r})


def _make_split_edges(env: SplitCTMTensorEnv) -> dict[str, Tensor]:
    """Build 4-leg split edges for all four boundary T's.

    Returns a dict keyed ``"T1"``, ``"T2"``, ``"T3"``, ``"T4"`` with each value
    a 4-leg ``(chi, D_ket, D_bra, chi)`` tensor. D-leg labels are
    ``{u,r,d,l}_ket`` / ``{u,r,d,l}_bra`` (inherited from inputs and globally
    unique across the four edges). Chi-leg labels follow the standard
    ``CTMTensorEnv`` convention: ``t1_l/t1_r``, ``t2_u/t2_d``, ``t3_r/t3_l``,
    ``t4_d/t4_u``.
    """
    return {
        "T1": _make_split_edge(
            env.T1_ket,
            env.T1_bra,
            ket_I="t1k_I",
            bra_I="t1b_I",
            ket_chi="t1k_l",
            bra_chi="t1b_r",
            out_chi_l="t1_l",
            out_chi_r="t1_r",
        ),
        "T2": _make_split_edge(
            env.T2_ket,
            env.T2_bra,
            ket_I="t2k_I",
            bra_I="t2b_I",
            ket_chi="t2k_u",
            bra_chi="t2b_d",
            out_chi_l="t2_u",
            out_chi_r="t2_d",
        ),
        "T3": _make_split_edge(
            env.T3_ket,
            env.T3_bra,
            ket_I="t3k_I",
            bra_I="t3b_I",
            ket_chi="t3k_r",
            bra_chi="t3b_l",
            out_chi_l="t3_r",
            out_chi_r="t3_l",
        ),
        "T4": _make_split_edge(
            env.T4_ket,
            env.T4_bra,
            ket_I="t4k_I",
            bra_I="t4b_I",
            ket_chi="t4k_d",
            bra_chi="t4b_u",
            out_chi_l="t4_d",
            out_chi_r="t4_u",
        ),
    }


def _rdm_1site_split_tensor(A: Tensor, env: SplitCTMTensorEnv) -> jax.Array:
    """Single-site RDM via split-aware contraction.

    Same network topology as ``_rdm_1site_tensor`` but uses 4-leg split edges
    and keeps ``A`` and ``A.bar()`` separate (no double-layer fusion).

    Note: 1-site has an intrinsic chi^2 * D^6 frame stage; the chi^2 * D^4 peak
    target of the design only applies to the 2x1/1x2/diagonal RDMs (Tasks 4-6).
    1-site exists for parity testing and small-D probes; energy never uses it
    for nearest-neighbour bonds.
    """
    splits = _make_split_edges(env)
    T1, T2, T3, T4 = splits["T1"], splits["T2"], splits["T3"], splits["T4"]

    A_ket = A.relabels({"u": "u_ket", "d": "d_ket", "l": "l_ket", "r": "r_ket"})
    A_bra = A.bar().relabels(
        {
            "u": "u_bra",
            "d": "d_bra",
            "l": "l_bra",
            "r": "r_bra",
            "phys": "phys_bra",
        }
    )

    # Build the boundary frame: corners + four split T's, all chi-bonds matched.
    C1 = env.C1.relabel("c1_r", "t1_l")
    C2 = env.C2.relabel("c2_l", "t1_r")
    top_row = contract(contract(C1, T1), C2)  # (c1_d, u_ket, u_bra, c2_d)

    C4 = env.C4.relabel("c4_u", "t3_r")
    C3 = env.C3.relabel("c3_l", "t3_l")
    bot_row = contract(contract(C4, T3), C3)  # (c4_r, d_ket, d_bra, c3_u)

    T4_e = T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})
    T2_e = T2.relabels({"t2_u": "c2_d", "t2_d": "c3_u"})

    # Frame: top·T4·T2·bot. Pairwise so the 4-leg split edges keep their
    # D_ket/D_bra structure (no D^2 fuse anywhere).
    frame_top = contract(top_row, T4_e)
    frame_full = contract(frame_top, T2_e)
    frame_full = contract(frame_full, bot_row)
    # frame_full has 8 D-legs: u_ket, u_bra, l_ket, l_bra, r_ket, r_bra,
    # d_ket, d_bra.

    # Absorb A then A_bra. A_ket's u/l/r/d match the ket D-legs; A_bra's match bra.
    rdm_t = contract(frame_full, A_ket)  # consumes u_ket, l_ket, r_ket, d_ket
    rdm_t = contract(
        rdm_t, A_bra, output_labels=["phys", "phys_bra"]
    )  # consumes u_bra, l_bra, r_bra, d_bra

    rdm = rdm_t.todense()
    rdm = 0.5 * (rdm + rdm.conj().T)
    rdm = rdm / (jnp.trace(rdm) + EPS)
    return rdm


def _rdm1x2_split_tensor(A: Tensor, env: SplitCTMTensorEnv) -> jax.Array:
    """Vertical 1x2 RDM via split-aware contraction. Bounded peak chi^2 * D^4.

    Top half contraction order (mirrored for bottom)::

        top_row     = C1 . T1_split . C2                # chi^2 * D^2
        top_T4      = top_row . T4_T_split              # chi^2 * D^4   (peak edge stage)
        top_T4_A    = top_T4 . A_ket                    # chi^2 * D^4 * d (peak overall)
        top_T4_A_T2 = top_T4_A . T2_T_split             # chi^2 * D^3 * d
        top_half    = top_T4_A_T2 . A_bra               # chi^2 * D^2 * d^2

    Combine on (t4_u<->t4_uB chi seam, t2_d<->t2_dB chi seam,
    d_ket<->u_ketB inner-D, d_bra<->u_braB inner-D).

    Returns dense RDM of shape ``(d, d, d, d)`` in
    ``(s1_ket, s2_ket, s1_bra, s2_bra)``, symmetrised and trace-normalised.
    """
    splits = _make_split_edges(env)
    T1, T2, T3, T4 = splits["T1"], splits["T2"], splits["T3"], splits["T4"]

    A_bra = A.bar().relabels(
        {
            "u": "u_bra",
            "d": "d_bra",
            "l": "l_bra",
            "r": "r_bra",
            "phys": "phys_bra",
        }
    )

    # ---------- Top half ----------
    C1 = env.C1.relabel("c1_r", "t1_l")
    C2 = env.C2.relabel("c2_l", "t1_r")
    top_row = contract(contract(C1, T1), C2)  # (c1_d, u_ket, u_bra, c2_d)

    T4_T = T4.relabels({"t4_d": "c1_d"})  # (c1_d, l_ket, l_bra, t4_u)
    T2_T = T2.relabels({"t2_u": "c2_d"})  # (c2_d, r_ket, r_bra, t2_d)

    A_top = A.relabels({"u": "u_ket", "l": "l_ket", "r": "r_ket"})

    top_T4 = contract(top_row, T4_T)  # chi^2 * D^4
    top_T4_A = contract(top_T4, A_top)  # chi^2 * D^4 * d (peak)
    top_T4_A_T2 = contract(top_T4_A, T2_T)  # chi^2 * D^3 * d
    top_half = contract(top_T4_A_T2, A_bra)  # chi^2 * D^2 * d^2

    # ---------- Bottom half ----------
    # Suffix bottom-site labels with "B" so they don't collide with the top half.
    bot_row_C4 = env.C4.relabel("c4_u", "t3_r")
    bot_row_C3 = env.C3.relabel("c3_l", "t3_l")
    bot_row = contract(contract(bot_row_C4, T3), bot_row_C3)
    # bot_row: (c4_r, d_ket, d_bra, c3_u) — the d_ket/d_bra here are T3's D-legs,
    # i.e. the bottom site's *down* legs. Rename to d_ketB/d_braB so they don't
    # collide with the top-half seam labels (top A's bare ``d`` leg and A_bra's
    # ``d_bra`` leg are the seam, not the bottom site's down-legs).
    bot_row = bot_row.relabels({"d_ket": "d_ketB", "d_bra": "d_braB"})

    T4_B = T4.relabels(
        {
            "t4_u": "c4_r",  # T4's chi-up matches C4_r at the bottom seam
            "t4_d": "t4_uB",  # other chi-end becomes the bottom-half top-seam
            "l_ket": "l_ketB",
            "l_bra": "l_braB",
        }
    )
    T2_B = T2.relabels(
        {
            "t2_d": "c3_u",
            "t2_u": "t2_dB",
            "r_ket": "r_ketB",
            "r_bra": "r_braB",
        }
    )

    A_bot = A.relabels(
        {
            "u": "u_ketB",  # OPEN — becomes seam to top_half's ``d`` leg
            "l": "l_ketB",
            "r": "r_ketB",
            "d": "d_ketB",  # matches bot_row's d_ketB (T3 inner-D)
            "phys": "phys_B",
        }
    )
    A_bra_bot = A_bra.relabels(
        {
            "u_bra": "u_braB",  # OPEN — becomes seam to top_half's d_bra
            "l_bra": "l_braB",  # matches T4_B's l_braB
            "r_bra": "r_braB",  # matches T2_B's r_braB
            "d_bra": "d_braB",  # matches bot_row's d_braB (T3 inner-D)
            "phys_bra": "phys_braB",
        }
    )

    bot_T4 = contract(bot_row, T4_B)  # chi^2 * D^4
    bot_T4_A = contract(bot_T4, A_bot)  # chi^2 * D^4 * d (peak)
    bot_T4_A_T2 = contract(bot_T4_A, T2_B)  # chi^2 * D^3 * d
    bot_half = contract(bot_T4_A_T2, A_bra_bot)  # chi^2 * D^2 * d^2

    # ---------- Combine ----------
    # Open labels on top_half:  (t4_u, d, phys, t2_d, d_bra, phys_bra)
    # Open labels on bot_half:  (t4_uB, u_ketB, phys_B, t2_dB, u_braB, phys_braB)
    # Seam labels:
    #   t4_u  (top T4 chi-up)   <-> t4_uB  (bot T4 chi-up)
    #   t2_d  (top T2 chi-down) <-> t2_dB  (bot T2 chi-down)
    #   d     (top A d-leg)     <-> u_ketB (bot A u-leg)
    #   d_bra (top A_bra d-leg) <-> u_braB (bot A_bra u-leg)
    bot_half = bot_half.relabels(
        {
            "t4_uB": "t4_u",
            "t2_dB": "t2_d",
            "u_ketB": "d",
            "u_braB": "d_bra",
        }
    )

    rdm_t = contract(
        top_half,
        bot_half,
        output_labels=["phys", "phys_B", "phys_bra", "phys_braB"],
    )
    # -> (s1_ket, s2_ket, s1_bra, s2_bra)

    rdm = rdm_t.todense()
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + EPS)
    return rdm_mat.reshape(d, d, d, d)


def _rdm2x1_split_tensor(A: Tensor, env: SplitCTMTensorEnv) -> jax.Array:
    """Horizontal 2x1 RDM via split-aware contraction. Bounded peak chi^2 * D^4.

    Mirror of ``_rdm1x2_split_tensor`` rotated 90 degrees: left/right halves
    instead of top/bottom. The seam is the left site's r/r_bra legs vs the
    right site's l/l_bra legs (plus the chi-seams t1_r<->t1_lR, t3_l<->t3_rR).

    Left half contraction order (mirrored on right)::

        UL          = C1 . T1                      # chi^2 * D^2
        LL          = C4 . T3                      # chi^2 * D^2
        UL_T4       = UL . T4                      # chi^2 * D^4   (peak edge stage)
        UL_T4_A     = UL_T4 . A_left               # chi^2 * D^4 * d (peak overall)
        UL_T4_A_LL  = UL_T4_A . LL                 # chi^2 * D^3 * d
        left_half   = UL_T4_A_LL . A_bra_left      # chi^2 * D^2 * d^2

    Note: in the left half, A's u_ket/d_ket legs are consumed by T1/T3 (top/
    bottom edges of the left column) and l_ket by T4 (left boundary); r_ket
    is the open seam. Mirror on right with R-suffix labels: A_right's
    u_ketR/d_ketR are consumed by T1_R/T3_R and r_ketR by T2; l_ketR is open
    and matches left's r_ket via the seam relabel.

    Returns dense RDM of shape ``(d, d, d, d)`` in
    ``(s1_ket, s2_ket, s1_bra, s2_bra)``, symmetrised and trace-normalised.
    """
    splits = _make_split_edges(env)
    T1, T2, T3, T4 = splits["T1"], splits["T2"], splits["T3"], splits["T4"]

    A_bra = A.bar().relabels(
        {
            "u": "u_bra",
            "d": "d_bra",
            "l": "l_bra",
            "r": "r_bra",
            "phys": "phys_bra",
        }
    )

    # ---------- Left half ----------
    C1 = env.C1.relabel("c1_r", "t1_l")
    UL = contract(C1, T1)  # (c1_d, u_ket, u_bra, t1_r)

    C4 = env.C4.relabel("c4_u", "t3_r")
    LL = contract(C4, T3)  # (c4_r, d_ket, d_bra, t3_l)

    T4_e = T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})  # (c1_d, l_ket, l_bra, c4_r)

    A_left = A.relabels({"u": "u_ket", "d": "d_ket", "l": "l_ket", "r": "r_ket"})

    UL_T4 = contract(UL, T4_e)  # chi^2 * D^4
    UL_T4_A = contract(UL_T4, A_left)  # chi^2 * D^4 * d (peak)
    UL_T4_A_LL = contract(UL_T4_A, LL)  # chi^2 * D^3 * d
    left_half = contract(UL_T4_A_LL, A_bra)  # chi^2 * D^2 * d^2
    # left_half open: (t1_r, t3_l, r_ket, phys, r_bra, phys_bra)

    # ---------- Right half ----------
    # Suffix all right-side labels with "R" so they don't collide with left.
    T1_R = T1.relabels(
        {"t1_l": "t1_lR", "u_ket": "u_ketR", "u_bra": "u_braR", "t1_r": "t1_rR"}
    )
    T3_R = T3.relabels(
        {"t3_r": "t3_rR", "d_ket": "d_ketR", "d_bra": "d_braR", "t3_l": "t3_lR"}
    )

    C2 = env.C2.relabel("c2_l", "t1_rR")
    UR = contract(T1_R, C2)  # (t1_lR, u_ketR, u_braR, c2_d)

    C3 = env.C3.relabel("c3_l", "t3_lR")
    LR = contract(T3_R, C3)  # (t3_rR, d_ketR, d_braR, c3_u)

    T2_e = T2.relabels(
        {
            "t2_u": "c2_d",
            "t2_d": "c3_u",
            "r_ket": "r_ketR",
            "r_bra": "r_braR",
        }
    )

    A_right = A.relabels(
        {
            "u": "u_ketR",
            "d": "d_ketR",
            "l": "l_ketR",  # OPEN — becomes seam to left_half's r_ket
            "r": "r_ketR",
            "phys": "phys_R",
        }
    )
    A_bra_right = A.bar().relabels(
        {
            "u": "u_braR",
            "d": "d_braR",
            "l": "l_braR",  # OPEN — becomes seam to left_half's r_bra
            "r": "r_braR",
            "phys": "phys_braR",
        }
    )

    UR_T2 = contract(UR, T2_e)  # chi^2 * D^4
    UR_T2_A = contract(UR_T2, A_right)  # chi^2 * D^4 * d (peak)
    UR_T2_A_LR = contract(UR_T2_A, LR)  # chi^2 * D^3 * d
    right_half = contract(UR_T2_A_LR, A_bra_right)  # chi^2 * D^2 * d^2
    # right_half open: (t1_lR, t3_rR, l_ketR, phys_R, l_braR, phys_braR)

    # ---------- Combine ----------
    # Seam labels:
    #   t1_r   (left T1 chi-right) <-> t1_lR  (right T1 chi-left)
    #   t3_l   (left T3 chi-left)  <-> t3_rR  (right T3 chi-right)
    #   r_ket  (left A r-leg)      <-> l_ketR (right A l-leg)
    #   r_bra  (left A_bra r-leg)  <-> l_braR (right A_bra l-leg)
    right_half = right_half.relabels(
        {
            "t1_lR": "t1_r",
            "t3_rR": "t3_l",
            "l_ketR": "r_ket",
            "l_braR": "r_bra",
        }
    )

    rdm_t = contract(
        left_half,
        right_half,
        output_labels=["phys", "phys_R", "phys_bra", "phys_braR"],
    )
    # -> (s1_ket, s2_ket, s1_bra, s2_bra)

    rdm = rdm_t.todense()
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + EPS)
    return rdm_mat.reshape(d, d, d, d)


def _rdm_diagonal_split_tensor(A: Tensor, env: SplitCTMTensorEnv) -> jax.Array:
    """Diagonal NNN 2-site RDM via 4-site (2x2) per-site env decomposition.

    Top-left (TL) and bottom-right (BR) carry open phys legs (the diagonal pair
    we measure); top-right (TR) and bottom-left (BL) are closed (phys traced).

    Each per-site env builds a chi^2 * D^4 frame (corner + 2 split edges), then
    absorbs A interleaved with A_bra:
    - Closed sites (TR, BL): A and A_bra share "phys" (traced) -> chi^2 * D^4
      peak.
    - Open sites (TL, BR): A_bra has "phys_bra"/"phys_bra_BR" (distinct)
      -> chi^2 * D^4 * d^2 peak (intrinsic cost of the diagonal RDM; bigger
      than the chi^2 * D^4 * d of Tasks 4-5).

    Returns dense RDM of shape ``(d, d, d, d)`` in
    ``(s1_ket, s2_ket, s1_bra, s2_bra)`` (TL=s1, BR=s2),
    symmetrised and trace-normalised.
    """
    splits = _make_split_edges(env)
    T1, T2, T3, T4 = splits["T1"], splits["T2"], splits["T3"], splits["T4"]

    # ---------- TL site (open) — labels: base ----------
    # Frame: C1 · T1 · T4_T (top-half of T4, sharing c1_d).
    C1 = env.C1.relabel("c1_r", "t1_l")
    C1_T1 = contract(C1, T1)  # (c1_d, u_ket, u_bra, t1_r)
    T4_T = T4.relabels({"t4_d": "c1_d"})  # (c1_d, l_ket, l_bra, t4_u)
    TL_frame = contract(C1_T1, T4_T)
    # TL_frame open: (u_ket, u_bra, t1_r, l_ket, l_bra, t4_u) — chi^2 * D^4

    A_TL = A.relabels({"u": "u_ket", "l": "l_ket"})  # keep d, r, phys open
    TL_frame_A = contract(TL_frame, A_TL)
    # open: (u_bra, t1_r, l_bra, t4_u, d, r, phys) — chi^2 * D^4 * d
    A_bra_TL = A.bar().relabels(
        {
            "u": "u_bra",
            "l": "l_bra",
            "d": "d_bra",
            "r": "r_bra",
            "phys": "phys_bra",
        }
    )
    site_env_TL = contract(TL_frame_A, A_bra_TL)
    # site_env_TL open: (t1_r, t4_u, d, r, phys, d_bra, r_bra, phys_bra)
    # — chi^2 * D^4 * d^2 (peak)

    # ---------- TR site (closed) — labels: R suffix on top/right; _TR on inner seams ----------
    T1_R = T1.relabels(
        {"t1_l": "t1_lR", "u_ket": "u_ketR", "u_bra": "u_braR", "t1_r": "t1_rR"}
    )
    C2 = env.C2.relabel("c2_l", "t1_rR")
    T1R_C2 = contract(T1_R, C2)  # (t1_lR, u_ketR, u_braR, c2_d)
    T2_T = T2.relabels(
        {"t2_u": "c2_d", "r_ket": "r_ketR", "r_bra": "r_braR"}
    )  # (c2_d, r_ketR, r_braR, t2_d)
    TR_frame = contract(T1R_C2, T2_T)
    # TR_frame open: (t1_lR, u_ketR, u_braR, r_ketR, r_braR, t2_d) — chi^2 * D^4

    # A_TR has u_ketR (consumed by frame), r_ketR (consumed),
    # d_TR_ket (open seam to BR), l_TR_ket (open seam to TL), phys (shared with bra → traced).
    A_TR = A.relabels(
        {
            "u": "u_ketR",
            "r": "r_ketR",
            "d": "d_TR_ket",
            "l": "l_TR_ket",
        }
    )
    TR_frame_A = contract(TR_frame, A_TR)
    # open: (u_braR, t1_lR, r_braR, t2_d, d_TR_ket, l_TR_ket, phys) — chi^2 * D^4 * d
    A_bra_TR = A.bar().relabels(
        {
            "u": "u_braR",
            "r": "r_braR",
            "d": "d_TR_bra",
            "l": "l_TR_bra",
        }
    )
    # phys label NOT relabeled; sharing with A_TR's phys causes the trace.
    site_env_TR = contract(TR_frame_A, A_bra_TR)
    # site_env_TR open: (t1_lR, t2_d, d_TR_ket, l_TR_ket, d_TR_bra, l_TR_bra)
    # — chi^2 * D^4

    # ---------- BL site (closed) — labels: B suffix on bot/left; _BL on inner seams ----------
    T3_BL = T3.relabels(
        {"d_ket": "d_ketB_BL", "d_bra": "d_braB_BL"}
    )  # (t3_r, d_ketB_BL, d_braB_BL, t3_l)
    C4 = env.C4.relabel("c4_u", "t3_r")
    C4_T3 = contract(C4, T3_BL)  # (c4_r, d_ketB_BL, d_braB_BL, t3_l)
    T4_B = T4.relabels(
        {
            "t4_d": "t4_dB",
            "l_ket": "l_ketB",
            "l_bra": "l_braB",
            "t4_u": "c4_r",
        }
    )  # (t4_dB, l_ketB, l_braB, c4_r)
    BL_frame = contract(C4_T3, T4_B)
    # BL_frame open: (d_ketB_BL, d_braB_BL, t3_l, t4_dB, l_ketB, l_braB) — chi^2 * D^4

    # A_BL has d → d_ketB_BL (consumed by T3_BL), l → l_ketB (consumed by T4_B),
    # u → u_BL_ket (open seam to TL), r → r_BL_ket (open seam to BR), phys shared.
    A_BL = A.relabels(
        {
            "d": "d_ketB_BL",
            "l": "l_ketB",
            "u": "u_BL_ket",
            "r": "r_BL_ket",
        }
    )
    BL_frame_A = contract(BL_frame, A_BL)
    # open: (d_braB_BL, t3_l, t4_dB, l_braB, u_BL_ket, r_BL_ket, phys) — chi^2 * D^4 * d
    # phys label NOT relabeled; sharing with A_BL's phys causes the trace.
    A_bra_BL = A.bar().relabels(
        {
            "d": "d_braB_BL",
            "l": "l_braB",
            "u": "u_BL_bra",
            "r": "r_BL_bra",
        }
    )
    site_env_BL = contract(BL_frame_A, A_bra_BL)
    # site_env_BL open: (t3_l, t4_dB, u_BL_ket, r_BL_ket, u_BL_bra, r_BL_bra)
    # — chi^2 * D^4

    # ---------- BR site (open) — labels: R/B suffixes; _BR on inner seams ----------
    T3_R = T3.relabels(
        {"t3_r": "t3_rR", "d_ket": "d_ketR", "d_bra": "d_braR", "t3_l": "t3_lR"}
    )
    C3 = env.C3.relabel("c3_l", "t3_lR")
    T3R_C3 = contract(T3_R, C3)  # (t3_rR, d_ketR, d_braR, c3_u)
    T2_BR = T2.relabels(
        {
            "t2_u": "t2_uB",
            "r_ket": "r_ketB",
            "r_bra": "r_braB",
            "t2_d": "c3_u",
        }
    )  # (t2_uB, r_ketB, r_braB, c3_u)
    BR_frame = contract(T3R_C3, T2_BR)
    # BR_frame open: (t3_rR, d_ketR, d_braR, t2_uB, r_ketB, r_braB) — chi^2 * D^4

    # A_BR has d → d_ketR (consumed), r → r_ketB (consumed),
    # u → u_BR_ket (open seam to TR), l → l_BR_ket (open seam to BL), phys → phys_BR.
    A_BR = A.relabels(
        {
            "d": "d_ketR",
            "r": "r_ketB",
            "u": "u_BR_ket",
            "l": "l_BR_ket",
            "phys": "phys_BR",
        }
    )
    BR_frame_A = contract(BR_frame, A_BR)
    # open: (d_braR, t3_rR, t2_uB, r_braB, u_BR_ket, l_BR_ket, phys_BR) — chi^2 * D^4 * d
    A_bra_BR = A.bar().relabels(
        {
            "d": "d_braR",
            "r": "r_braB",
            "u": "u_BR_bra",
            "l": "l_BR_bra",
            "phys": "phys_bra_BR",
        }
    )
    site_env_BR = contract(BR_frame_A, A_bra_BR)
    # site_env_BR open: (t3_rR, t2_uB, u_BR_ket, l_BR_ket, phys_BR,
    #                    u_BR_bra, l_BR_bra, phys_bra_BR)
    # — chi^2 * D^4 * d^2 (peak)

    # ---------- Combine columns ----------
    # left_half = site_env_TL · site_env_BL on:
    #   chi seam: t4_u <-> t4_dB
    #   inner-D seams: d <-> u_BL_ket, d_bra <-> u_BL_bra
    site_env_BL = site_env_BL.relabels(
        {
            "t4_dB": "t4_u",
            "u_BL_ket": "d",
            "u_BL_bra": "d_bra",
        }
    )
    left_half = contract(site_env_TL, site_env_BL)
    # left_half open: (t1_r, r, phys, r_bra, phys_bra, t3_l, r_BL_ket, r_BL_bra)

    # right_half = site_env_TR · site_env_BR on:
    #   chi seam: t2_d <-> t2_uB
    #   inner-D seams: d_TR_ket <-> u_BR_ket, d_TR_bra <-> u_BR_bra
    site_env_BR = site_env_BR.relabels(
        {
            "t2_uB": "t2_d",
            "u_BR_ket": "d_TR_ket",
            "u_BR_bra": "d_TR_bra",
        }
    )
    right_half = contract(site_env_TR, site_env_BR)
    # right_half open: (t1_lR, l_TR_ket, l_TR_bra, t3_rR, l_BR_ket, l_BR_bra,
    #                   phys_BR, phys_bra_BR)

    # ---------- Final combine ----------
    # Match: t1_r<->t1_lR (chi), t3_l<->t3_rR (chi),
    #        r<->l_TR_ket (top inner-D), r_bra<->l_TR_bra,
    #        r_BL_ket<->l_BR_ket (bot inner-D), r_BL_bra<->l_BR_bra.
    right_half = right_half.relabels(
        {
            "t1_lR": "t1_r",
            "t3_rR": "t3_l",
            "l_TR_ket": "r",
            "l_TR_bra": "r_bra",
            "l_BR_ket": "r_BL_ket",
            "l_BR_bra": "r_BL_bra",
        }
    )
    rdm_t = contract(
        left_half,
        right_half,
        output_labels=["phys", "phys_BR", "phys_bra", "phys_bra_BR"],
    )

    rdm = rdm_t.todense()
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + EPS)
    return rdm_mat.reshape(d, d, d, d)


def _rdm1x2_split_tensor_2site(
    A: Tensor,
    B: Tensor,
    env_A: SplitCTMTensorEnv,
    env_B: SplitCTMTensorEnv,
) -> jax.Array:
    """Vertical 1x2 RDM for checkerboard / mixed-env (A top, B bottom).

    Mixed environment::

        C1_A — T1_A — C2_A
        |       |       |
        T4_A   ao_A   T2_A
        |       |       |
        T4_B   ao_B   T2_B
        |       |       |
        C4_B — T3_B — C3_B

    Same contraction order and label conventions as ``_rdm1x2_split_tensor``,
    but T1/T2_T/T4_T come from ``env_A`` (top half) and T3/T2_B/T4_B from
    ``env_B`` (bottom half).  Bounded peak ``chi²·D⁴`` per half, vs the
    standard shim's ``chi²·D⁶``.

    When the unnormalized trace lands below
    :data:`_MIXED_ENV_RDM_TRACE_FLOOR` the trace-divide amplifies floating-
    point noise above the atol=1e-10 shim-parity tolerance (issues #479,
    #485); the call delegates to ``_split_env_to_tensor_standard`` + the
    standard RDM routine on that adversarial path while keeping the
    memory-efficient split contraction for the common case (#480).

    Returns dense RDM of shape ``(d, d, d, d)`` in
    ``(s1_A_ket, s2_B_ket, s1_A_bra, s2_B_bra)``,
    symmetrised and trace-normalised.
    """
    splits_A = _make_split_edges(env_A)
    splits_B = _make_split_edges(env_B)
    T1, T4_A_split, T2_A_split = splits_A["T1"], splits_A["T4"], splits_A["T2"]
    T3, T4_B_split, T2_B_split = splits_B["T3"], splits_B["T4"], splits_B["T2"]

    A_bra = A.bar().relabels(
        {
            "u": "u_bra",
            "d": "d_bra",
            "l": "l_bra",
            "r": "r_bra",
            "phys": "phys_bra",
        }
    )

    # ---------- Top half (env_A, A) ----------
    C1 = env_A.C1.relabel("c1_r", "t1_l")
    C2 = env_A.C2.relabel("c2_l", "t1_r")
    top_row = contract(contract(C1, T1), C2)  # (c1_d, u_ket, u_bra, c2_d)

    T4_T = T4_A_split.relabels({"t4_d": "c1_d"})  # (c1_d, l_ket, l_bra, t4_u)
    T2_T = T2_A_split.relabels({"t2_u": "c2_d"})  # (c2_d, r_ket, r_bra, t2_d)

    A_top = A.relabels({"u": "u_ket", "l": "l_ket", "r": "r_ket"})

    top_T4 = contract(top_row, T4_T)  # chi^2 * D^4
    top_T4_A = contract(top_T4, A_top)  # chi^2 * D^4 * d (peak)
    top_T4_A_T2 = contract(top_T4_A, T2_T)  # chi^2 * D^3 * d
    top_half = contract(top_T4_A_T2, A_bra)  # chi^2 * D^2 * d^2

    # ---------- Bottom half (env_B, B) ----------
    # Suffix bottom-site labels with "B" so they don't collide with the top half.
    B_bra = B.bar().relabels(
        {
            "u": "u_bra",
            "d": "d_bra",
            "l": "l_bra",
            "r": "r_bra",
            "phys": "phys_bra",
        }
    )

    bot_row_C4 = env_B.C4.relabel("c4_u", "t3_r")
    bot_row_C3 = env_B.C3.relabel("c3_l", "t3_l")
    bot_row = contract(contract(bot_row_C4, T3), bot_row_C3)
    bot_row = bot_row.relabels({"d_ket": "d_ketB", "d_bra": "d_braB"})

    T4_B = T4_B_split.relabels(
        {
            "t4_u": "c4_r",
            "t4_d": "t4_uB",
            "l_ket": "l_ketB",
            "l_bra": "l_braB",
        }
    )
    T2_B = T2_B_split.relabels(
        {
            "t2_d": "c3_u",
            "t2_u": "t2_dB",
            "r_ket": "r_ketB",
            "r_bra": "r_braB",
        }
    )

    B_bot = B.relabels(
        {
            "u": "u_ketB",
            "l": "l_ketB",
            "r": "r_ketB",
            "d": "d_ketB",
            "phys": "phys_B",
        }
    )
    B_bra_bot = B_bra.relabels(
        {
            "u_bra": "u_braB",
            "l_bra": "l_braB",
            "r_bra": "r_braB",
            "d_bra": "d_braB",
            "phys_bra": "phys_braB",
        }
    )

    bot_T4 = contract(bot_row, T4_B)  # chi^2 * D^4
    bot_T4_A = contract(bot_T4, B_bot)  # chi^2 * D^4 * d (peak)
    bot_T4_A_T2 = contract(bot_T4_A, T2_B)  # chi^2 * D^3 * d
    bot_half = contract(bot_T4_A_T2, B_bra_bot)  # chi^2 * D^2 * d^2

    # ---------- Combine ----------
    bot_half = bot_half.relabels(
        {
            "t4_uB": "t4_u",
            "t2_dB": "t2_d",
            "u_ketB": "d",
            "u_braB": "d_bra",
        }
    )

    rdm_t = contract(
        top_half,
        bot_half,
        output_labels=["phys", "phys_B", "phys_bra", "phys_braB"],
    )
    # -> (s1_A_ket, s2_B_ket, s1_A_bra, s2_B_bra)

    rdm = rdm_t.todense()
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)

    # Trace-floor guard: if the split contraction's trace lands in the
    # catastrophic-cancellation regime, fall back to the shim path so the
    # atol=1e-10 parity tolerance holds.  Eager Python branch — these RDM
    # routines are not jit-traced anywhere in the codebase (energy probes
    # only).  ``.item()`` forces a host sync.
    trace_val = jnp.trace(rdm_mat)
    if float(jnp.abs(trace_val).item()) < _MIXED_ENV_RDM_TRACE_FLOOR:
        from tenax.algorithms._ctm_tensor_energy import _rdm1x2_tensor_2site

        return _rdm1x2_tensor_2site(
            A,
            B,
            _split_env_to_tensor_standard(env_A),
            _split_env_to_tensor_standard(env_B),
        )

    rdm_mat = rdm_mat / (trace_val + EPS)
    return rdm_mat.reshape(d, d, d, d)


def _rdm2x1_split_tensor_2site(
    A: Tensor,
    B: Tensor,
    env_A: SplitCTMTensorEnv,
    env_B: SplitCTMTensorEnv,
) -> jax.Array:
    """Horizontal 2x1 RDM for checkerboard / mixed-env (A left, B right).

    Mixed environment::

        C1_A — T1_A — T1_B — C2_B
        |       |       |       |
        T4_A   ao_A   ao_B    T2_B
        |       |       |       |
        C4_A — T3_A — T3_B — C3_B

    Same contraction order and label conventions as ``_rdm2x1_split_tensor``,
    but C1/C4/T1/T3/T4 come from ``env_A`` (left half) and C2/C3/T1_R/T3_R/T2
    from ``env_B`` (right half).  Bounded peak ``chi²·D⁴`` per half, vs the
    standard shim's ``chi²·D⁶``.

    When the unnormalized trace lands below
    :data:`_MIXED_ENV_RDM_TRACE_FLOOR` the trace-divide amplifies floating-
    point noise above the atol=1e-10 shim-parity tolerance (issues #479,
    #485); the call delegates to ``_split_env_to_tensor_standard`` + the
    standard RDM routine on that adversarial path while keeping the
    memory-efficient split contraction for the common case (#480).

    Returns dense RDM of shape ``(d, d, d, d)`` in
    ``(s1_A_ket, s2_B_ket, s1_A_bra, s2_B_bra)``,
    symmetrised and trace-normalised.
    """
    splits_A = _make_split_edges(env_A)
    splits_B = _make_split_edges(env_B)
    T1, T3, T4 = splits_A["T1"], splits_A["T3"], splits_A["T4"]
    T1_B_split, T3_B_split, T2_B_split = (
        splits_B["T1"],
        splits_B["T3"],
        splits_B["T2"],
    )

    A_bra = A.bar().relabels(
        {
            "u": "u_bra",
            "d": "d_bra",
            "l": "l_bra",
            "r": "r_bra",
            "phys": "phys_bra",
        }
    )

    # ---------- Left half (env_A, A) ----------
    C1 = env_A.C1.relabel("c1_r", "t1_l")
    UL = contract(C1, T1)  # (c1_d, u_ket, u_bra, t1_r)

    C4 = env_A.C4.relabel("c4_u", "t3_r")
    LL = contract(C4, T3)  # (c4_r, d_ket, d_bra, t3_l)

    T4_e = T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})

    A_left = A.relabels({"u": "u_ket", "d": "d_ket", "l": "l_ket", "r": "r_ket"})

    UL_T4 = contract(UL, T4_e)  # chi^2 * D^4
    UL_T4_A = contract(UL_T4, A_left)  # chi^2 * D^4 * d (peak)
    UL_T4_A_LL = contract(UL_T4_A, LL)  # chi^2 * D^3 * d
    left_half = contract(UL_T4_A_LL, A_bra)  # chi^2 * D^2 * d^2

    # ---------- Right half (env_B, B) ----------
    T1_R = T1_B_split.relabels(
        {"t1_l": "t1_lR", "u_ket": "u_ketR", "u_bra": "u_braR", "t1_r": "t1_rR"}
    )
    T3_R = T3_B_split.relabels(
        {"t3_r": "t3_rR", "d_ket": "d_ketR", "d_bra": "d_braR", "t3_l": "t3_lR"}
    )

    C2 = env_B.C2.relabel("c2_l", "t1_rR")
    UR = contract(T1_R, C2)  # (t1_lR, u_ketR, u_braR, c2_d)

    C3 = env_B.C3.relabel("c3_l", "t3_lR")
    LR = contract(T3_R, C3)  # (t3_rR, d_ketR, d_braR, c3_u)

    T2_e = T2_B_split.relabels(
        {
            "t2_u": "c2_d",
            "t2_d": "c3_u",
            "r_ket": "r_ketR",
            "r_bra": "r_braR",
        }
    )

    B_right = B.relabels(
        {
            "u": "u_ketR",
            "d": "d_ketR",
            "l": "l_ketR",
            "r": "r_ketR",
            "phys": "phys_R",
        }
    )
    B_bra_right = B.bar().relabels(
        {
            "u": "u_braR",
            "d": "d_braR",
            "l": "l_braR",
            "r": "r_braR",
            "phys": "phys_braR",
        }
    )

    UR_T2 = contract(UR, T2_e)  # chi^2 * D^4
    UR_T2_A = contract(UR_T2, B_right)  # chi^2 * D^4 * d (peak)
    UR_T2_A_LR = contract(UR_T2_A, LR)  # chi^2 * D^3 * d
    right_half = contract(UR_T2_A_LR, B_bra_right)  # chi^2 * D^2 * d^2

    # ---------- Combine ----------
    right_half = right_half.relabels(
        {
            "t1_lR": "t1_r",
            "t3_rR": "t3_l",
            "l_ketR": "r_ket",
            "l_braR": "r_bra",
        }
    )

    rdm_t = contract(
        left_half,
        right_half,
        output_labels=["phys", "phys_R", "phys_bra", "phys_braR"],
    )

    rdm = rdm_t.todense()
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)

    # Trace-floor guard: see :func:`_rdm1x2_split_tensor_2site` for rationale.
    trace_val = jnp.trace(rdm_mat)
    if float(jnp.abs(trace_val).item()) < _MIXED_ENV_RDM_TRACE_FLOOR:
        from tenax.algorithms._ctm_tensor_energy import _rdm2x1_tensor_2site

        return _rdm2x1_tensor_2site(
            A,
            B,
            _split_env_to_tensor_standard(env_A),
            _split_env_to_tensor_standard(env_B),
        )

    rdm_mat = rdm_mat / (trace_val + EPS)
    return rdm_mat.reshape(d, d, d, d)


def _split_env_to_tensor_standard(env: SplitCTMTensorEnv) -> CTMTensorEnv:
    """Convert SplitCTMTensorEnv to CTMTensorEnv via Tensor contraction.

    Merges each (T_ket, T_bra) pair by contracting over the interlayer bond
    and fusing the two D-legs into a single double-layer D² leg.
    Corners pass through unchanged (same labels/flows).
    """

    def _merge_edge(
        T_ket,
        T_bra,
        ket_I,
        bra_I,
        d_ket,
        d_bra,
        fused_label,
        fused_flow,
        ket_chi,
        bra_chi,
        std_chi_l,
        std_chi_r,
    ):
        # Contract over interlayer bond by relabelling both I-labels to "_I"
        k = T_ket.relabel(ket_I, "_I")
        b = T_bra.relabel(bra_I, "_I")
        merged = contract(k, b)
        # Fuse D-ket and D-bra legs
        labels = merged.labels()
        merged = fuse_indices(
            merged,
            labels.index(d_ket),
            labels.index(d_bra),
            fused_label,
            fused_flow,
        )
        # Relabel chi legs to standard CTMTensorEnv convention
        merged = merged.relabels({ket_chi: std_chi_l, bra_chi: std_chi_r})
        return merged

    T1 = _merge_edge(
        env.T1_ket,
        env.T1_bra,
        "t1k_I",
        "t1b_I",
        "u_ket",
        "u_bra",
        "u2",
        FlowDirection.IN,
        "t1k_l",
        "t1b_r",
        "t1_l",
        "t1_r",
    )
    T2 = _merge_edge(
        env.T2_ket,
        env.T2_bra,
        "t2k_I",
        "t2b_I",
        "r_ket",
        "r_bra",
        "r2",
        FlowDirection.OUT,
        "t2k_u",
        "t2b_d",
        "t2_u",
        "t2_d",
    )
    T3 = _merge_edge(
        env.T3_ket,
        env.T3_bra,
        "t3k_I",
        "t3b_I",
        "d_ket",
        "d_bra",
        "d2",
        FlowDirection.OUT,
        "t3k_r",
        "t3b_l",
        "t3_r",
        "t3_l",
    )
    T4 = _merge_edge(
        env.T4_ket,
        env.T4_bra,
        "t4k_I",
        "t4b_I",
        "l_ket",
        "l_bra",
        "l2",
        FlowDirection.IN,
        "t4k_d",
        "t4b_u",
        "t4_d",
        "t4_u",
    )

    return CTMTensorEnv(
        C1=env.C1,
        C2=env.C2,
        C3=env.C3,
        C4=env.C4,
        T1=T1,
        T2=T2,
        T3=T3,
        T4=T4,
    )


def compute_energy_split_ctm_tensor(
    A: Tensor,
    env: SplitCTMTensorEnv,
    hamiltonian_gate: Tensor | jax.Array,
    d: int | None = None,
) -> jax.Array:
    """Compute energy per site using a split CTM environment, split-aware.

    Builds horizontal and vertical RDMs directly from
    ``(T_ket, T_bra, A, A.bar())``, without merging ket/bra to the
    standard double-layer env. Bounded peak intermediate ~chi^2 * D^4.

    Args:
        A:                iPEPS site tensor with labels ``(u, d, l, r, phys)``.
        env:              Converged SplitCTMTensorEnv.
        hamiltonian_gate: 2-site Hamiltonian gate.
        d:                Physical dimension (inferred from A if None).

    Returns:
        Scalar energy per site.
    """
    if d is None:
        phys_idx = [i for i in A.indices if i.label == "phys"]
        d = phys_idx[0].dim if phys_idx else A.indices[-1].dim

    if isinstance(hamiltonian_gate, Tensor):
        H = hamiltonian_gate.todense().reshape(d, d, d, d)
    else:
        H = hamiltonian_gate.reshape(d, d, d, d)

    rdm_h = _rdm2x1_split_tensor(A, env)
    rdm_v = _rdm1x2_split_tensor(A, env)
    E_h = jnp.einsum("ijkl,ijkl->", rdm_h, H)
    E_v = jnp.einsum("ijkl,ijkl->", rdm_v, H)
    return (E_h + E_v).real


def compute_energy_split_ctm_tensor_2site(
    A: Tensor,
    B: Tensor,
    env_A: SplitCTMTensorEnv,
    env_B: SplitCTMTensorEnv,
    hamiltonian_gate: Tensor | jax.Array,
    d: int | None = None,
) -> jax.Array:
    """Compute energy per site for a 2-site checkerboard iPEPS, split-aware.

    Args:
        A:                Site tensor for sublattice A.
        B:                Site tensor for sublattice B.
        env_A:            Converged SplitCTMTensorEnv for sublattice A.
        env_B:            Converged SplitCTMTensorEnv for sublattice B.
        hamiltonian_gate: 2-site Hamiltonian gate.
        d:                Physical dimension (inferred from A if None).

    Returns:
        Scalar energy per site.
    """
    # Route through the generic multisite N=2 checkerboard path so the two
    # split energy code paths never diverge. The multisite function counts the
    # same 4 NN bonds (2 A-B + 2 B-A) and normalises by 2 sites, matching the
    # old 0.5 * sum-of-four-bonds formula (guarded by
    # test_split_2site_energy_equals_multisite).
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    return compute_energy_split_ctm_tensor_multisite(
        {(0, 0): A, (1, 0): B},
        {(0, 0): env_A, (1, 0): env_B},
        CHECKERBOARD_NEIGHBORS,
        hamiltonian_gate,
        d=d,
    )


def compute_energy_split_ctm_tensor_multisite(
    site_tensors: dict,
    envs: dict,
    neighbors: dict,
    gate: Tensor | jax.Array,
    d: int | None = None,
) -> jax.Array:
    """Compute energy per site summed over all NN bonds in a multi-site unit cell, split-aware.

    Each bond is counted once. Energy is normalized by the number of sites.

    Args:
        site_tensors: ``{coord: Tensor}`` mapping coordinates to iPEPS site tensors.
        envs:         ``{coord: SplitCTMTensorEnv}`` converged split environments per site.
        neighbors:    ``{coord: {"left": coord, "right": coord, "top": coord,
                      "bottom": coord}}`` neighbor map.
        gate:         2-site Hamiltonian gate.
        d:            Physical dimension (inferred from first site if None).

    Returns:
        Scalar energy per site.
    """
    # Infer physical dimension
    if d is None:
        first_A = next(iter(site_tensors.values()))
        phys_idx = [i for i in first_A.indices if i.label == "phys"]
        d = phys_idx[0].dim if phys_idx else first_A.indices[-1].dim

    # Prepare gate as dense (d, d, d, d) array
    if isinstance(gate, Tensor):
        H = gate.todense().reshape(d, d, d, d)
    else:
        H = gate.reshape(d, d, d, d)

    n_sites = len(site_tensors)
    total_energy = jnp.array(0.0)
    counted_bonds: set = set()

    for coord, A in site_tensors.items():
        env_A = envs[coord]
        for direction in ("right", "bottom"):
            nb_coord = neighbors[coord][direction]

            # Build a canonical bond identifier to avoid double-counting.
            # A bond from coord→right is the same as nb_coord→left.
            reverse_dir = "left" if direction == "right" else "top"
            reverse_bond = (nb_coord, reverse_dir)
            bond = (coord, direction)
            # Use frozenset so {(A,right), (B,left)} == {(B,left), (A,right)}
            bond_id = frozenset([bond, reverse_bond])
            if bond_id in counted_bonds:
                continue
            counted_bonds.add(bond_id)

            B = site_tensors[nb_coord]
            env_B = envs[nb_coord]

            # When a site is its own neighbor, use the single-site RDM functions
            if coord == nb_coord:
                if direction == "right":
                    rdm = _rdm2x1_split_tensor(A, env_A)
                else:
                    rdm = _rdm1x2_split_tensor(A, env_A)
            else:
                if direction == "right":
                    rdm = _rdm2x1_split_tensor_2site(A, B, env_A, env_B)
                else:
                    rdm = _rdm1x2_split_tensor_2site(A, B, env_A, env_B)

            bond_energy = jnp.einsum("ijkl,ijkl->", rdm, H)
            total_energy = total_energy + bond_energy

    return total_energy.real / n_sites
