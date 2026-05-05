"""Standard CTM with Tensor protocol — RDM construction and energy computation."""

from __future__ import annotations

__all__ = [
    "_rdm1x2_tensor",
    "_rdm1x2_tensor_2site",
    "_rdm2x1_tensor",
    "_rdm2x1_tensor_2site",
    "_rdm_1site_tensor",
    "_rdm_diagonal_tensor",
    "_rdm_diagonal_tensor_4site",
    "compute_energy_ctm_tensor",
    "compute_energy_ctm_tensor_2site",
    "compute_energy_ctm_tensor_multisite",
]

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_open_tensor,
    _build_double_layer_tensor,
)
from tenax.contraction.contractor import contract
from tenax.core import EPS
from tenax.core.tensor import Tensor


def _rdm_1site_tensor(A: Tensor, env: CTMTensorEnv) -> jax.Array:
    """Single-site RDM using label-based Tensor contractions.

    Contracts the network::

        C1 — T1 — C2
        |      |      |
        T4   ao     T2
        |      |      |
        C4 — T3 — C3

    Where ``ao = _build_double_layer_open_tensor(A)`` has labels
    ``(u2, d2, l2, r2, phys, phys_bra)``.

    Returns dense RDM of shape ``(d, d)`` in ``(phys, phys_bra)`` convention,
    symmetrised and trace-normalised.
    """
    ao = _build_double_layer_open_tensor(A)  # (u2, d2, l2, r2, phys, phys_bra)

    # Step 1: top row = C1 · T1 · C2
    C1 = env.C1.relabel("c1_r", "t1_l")
    C2 = env.C2.relabel("c2_l", "t1_r")
    C1_T1 = contract(C1, env.T1)  # (c1_d, u2, t1_r)
    top_row = contract(C1_T1, C2)  # (c1_d, u2, c2_d)

    # Step 2: bottom row = C4 · T3 · C3
    C4 = env.C4.relabel("c4_u", "t3_r")
    C3 = env.C3.relabel("c3_l", "t3_l")
    C4_T3 = contract(C4, env.T3)  # (c4_r, d2, t3_l)
    bot_row = contract(C4_T3, C3)  # (c4_r, d2, c3_u)

    # Step 3: add T4 — connects c1_d ↔ t4_d, t4_u ↔ c4_r
    T4 = env.T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})
    top_T4 = contract(top_row, T4)  # (u2, c2_d, l2, c4_r)

    # Step 4: add T2 — connects c2_d ↔ t2_u, t2_d ↔ c3_u
    T2 = env.T2.relabels({"t2_u": "c2_d", "t2_d": "c3_u"})
    top_T4_T2 = contract(top_T4, T2)  # (u2, l2, c4_r, r2, c3_u)

    # Step 5: contract with bottom row (shared: c4_r, c3_u; also d2)
    env_full = contract(top_T4_T2, bot_row)  # (u2, l2, r2, d2)

    # Step 6: contract full environment with ao (shared: u2, d2, l2, r2)
    rdm_t = contract(
        env_full,
        ao,
        output_labels=["phys", "phys_bra"],
    )

    rdm = rdm_t.todense()
    rdm = 0.5 * (rdm + rdm.conj().T)
    rdm = rdm / (jnp.trace(rdm) + EPS)
    return rdm


def _rdm_diagonal_tensor(A: Tensor, env: CTMTensorEnv) -> jax.Array:
    r"""Diagonal (NNN) 2-site RDM from a 2×2 plaquette contraction.

    Top-left (TL) and bottom-right (BR) sites have physical legs open;
    top-right (TR) and bottom-left (BL) have physical legs traced (closed).

    Returns dense RDM of shape ``(d, d, d, d)`` in
    ``(s1_ket, s2_ket, s1_bra, s2_bra)`` convention (TL=s1, BR=s2),
    symmetrised and trace-normalised.

    Uses a per-site env decomposition: each of the four sites first absorbs
    its corner-pair, edges, and ``ao``/``ac`` factor independently, then
    pairs are joined column-wise (TL+BL, TR+BR) and the two halves are
    combined.  This keeps every intermediate at the irreducible
    ``chi² · D⁴ · d²`` floor, instead of accumulating an extra ``D²`` from
    contracting the row edges before consuming ``ao``.
    """
    ao = _build_double_layer_open_tensor(A)  # (u2, d2, l2, r2, phys, phys_bra)
    ac = _build_double_layer_tensor(A)  # (u2, d2, l2, r2)

    # Right-column copies (suffix R)
    T1_R = env.T1.relabels({"t1_l": "t1_lR", "u2": "u2R", "t1_r": "t1_rR"})
    T3_R = env.T3.relabels({"t3_r": "t3_rR", "d2": "d2R", "t3_l": "t3_lR"})

    # Bottom-row copies (suffix B)
    T4_B = env.T4.relabels({"t4_d": "t4_dB", "l2": "l2B", "t4_u": "t4_uB"})
    T2_B = env.T2.relabels({"t2_u": "t2_uB", "r2": "r2B", "t2_d": "t2_dB"})

    # Site-internal labels.  Each ao/ac instance carries a unique label set
    # so its inner bonds (the ones shared with the diagonal partner site)
    # are explicit:
    #   - TL: keeps base names (u2, d2, l2, r2, phys, phys_bra).
    #   - TR (closed): inner-left bond ↔ TL.r2, inner-down bond ↔ BR.u2_BR.
    #   - BL (closed): inner-up bond ↔ TL.d2, inner-right bond ↔ BR.l2_BR.
    #   - BR (open):   inner-left bond ↔ BL.r2_BL, inner-up bond ↔ TR.d2_TR.
    ac_TR = ac.relabels({"u2": "u2R", "d2": "d2_TR", "l2": "l2_TR", "r2": "r2R"})
    ac_BL = ac.relabels({"u2": "u2_BL", "d2": "d2B_BL", "l2": "l2B", "r2": "r2_BL"})
    ao_BR = ao.relabels(
        {
            "u2": "u2_BR",
            "d2": "d2R",
            "l2": "l2_BR",
            "r2": "r2B",
            "phys": "phys_BR",
            "phys_bra": "phys_bra_BR",
        }
    )

    # ---------- Per-site envs (one ao/ac per site, contracted in place) ----------
    # site_env_TL:  C1·T1·T4_T·ao_TL  →  (t1_r, t4_u, d2, r2, phys, phys_bra)
    C1 = env.C1.relabel("c1_r", "t1_l")
    C1_T1 = contract(C1, env.T1)  # (c1_d, u2, t1_r)
    T4_T = env.T4.relabels({"t4_d": "c1_d"})  # (c1_d, l2, t4_u)
    TL_env = contract(C1_T1, T4_T)  # (u2, t1_r, l2, t4_u)
    site_env_TL = contract(TL_env, ao)  # (t1_r, t4_u, d2, r2, phys, phys_bra)

    # site_env_TR:  T1_R·C2·T2_T·ac_TR  →  (t1_lR, t2_d, l2_TR, d2_TR)
    C2 = env.C2.relabel("c2_l", "t1_rR")
    T1R_C2 = contract(T1_R, C2)  # (t1_lR, u2R, c2_d)
    T2_T = env.T2.relabels({"t2_u": "c2_d", "r2": "r2R"})  # (c2_d, r2R, t2_d)
    TR_env = contract(T1R_C2, T2_T)  # (t1_lR, u2R, r2R, t2_d)
    site_env_TR = contract(TR_env, ac_TR)  # (t1_lR, t2_d, l2_TR, d2_TR)

    # site_env_BL:  C4·T3·T4_B·ac_BL  →  (t3_l, t4_dB, u2_BL, r2_BL)
    C4 = env.C4.relabel("c4_u", "t3_r")
    T3_BL = env.T3.relabel("d2", "d2B_BL")  # (t3_r, d2B_BL, t3_l)
    C4_T3 = contract(C4, T3_BL)  # (c4_r, d2B_BL, t3_l)
    T4_BL = T4_B.relabel("t4_uB", "c4_r")  # (t4_dB, l2B, c4_r)
    BL_env = contract(C4_T3, T4_BL)  # (d2B_BL, t3_l, t4_dB, l2B)
    site_env_BL = contract(BL_env, ac_BL)  # (t3_l, t4_dB, u2_BL, r2_BL)

    # site_env_BR:  T3_R·C3·T2_B·ao_BR  →  (t3_rR, t2_uB, u2_BR, l2_BR, phys_BR, phys_bra_BR)
    C3 = env.C3.relabel("c3_l", "t3_lR")
    T3R_C3 = contract(T3_R, C3)  # (t3_rR, d2R, c3_u)
    T2_BR = T2_B.relabel("t2_dB", "c3_u")  # (t2_uB, r2B, c3_u)
    BR_env = contract(T3R_C3, T2_BR)  # (t3_rR, d2R, t2_uB, r2B)
    site_env_BR = contract(BR_env, ao_BR)
    # (t3_rR, t2_uB, u2_BR, l2_BR, phys_BR, phys_bra_BR)

    # ---------- Combine columns ----------
    # left_half = site_env_TL · site_env_BL  on (t4_u↔t4_dB, d2↔u2_BL)
    site_env_BL = site_env_BL.relabels({"t4_dB": "t4_u", "u2_BL": "d2"})
    left_half = contract(site_env_TL, site_env_BL)
    # (t1_r, r2, phys, phys_bra, t3_l, r2_BL)

    # right_half = site_env_TR · site_env_BR  on (t2_d↔t2_uB, d2_TR↔u2_BR)
    site_env_BR = site_env_BR.relabels({"t2_uB": "t2_d", "u2_BR": "d2_TR"})
    right_half = contract(site_env_TR, site_env_BR)
    # (t1_lR, l2_TR, t3_rR, l2_BR, phys_BR, phys_bra_BR)

    # Final combine: shared (t1_r↔t1_lR, t3_l↔t3_rR, r2↔l2_TR, r2_BL↔l2_BR)
    right_half = right_half.relabels(
        {"t1_lR": "t1_r", "l2_TR": "r2", "l2_BR": "r2_BL", "t3_rR": "t3_l"}
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


def _rdm_diagonal_tensor_4site(
    A_TL: Tensor,
    A_TR: Tensor,
    A_BL: Tensor,
    A_BR: Tensor,
    env_TL: CTMTensorEnv,
    env_TR: CTMTensorEnv,
    env_BL: CTMTensorEnv,
    env_BR: CTMTensorEnv,
) -> jax.Array:
    r"""Diagonal (NNN) 2-site RDM from a 2×2 multisite plaquette contraction.

    Generalizes :func:`_rdm_diagonal_tensor` to four distinct
    ``(site_tensor, env)`` pairs at the four plaquette corners.

    TL and BR sites have physical legs open; TR and BL are traced (closed).
    Returns dense RDM of shape ``(d, d, d, d)`` in
    ``(s_TL_ket, s_BR_ket, s_TL_bra, s_BR_bra)`` convention,
    symmetrised and trace-normalised.

    For the single-site case (all four arguments identical), this function
    returns the same result as :func:`_rdm_diagonal_tensor`.
    """
    # Open double-layer tensors for the corners with open phys legs (TL, BR).
    ao_TL = _build_double_layer_open_tensor(A_TL)
    ao_BR_base = _build_double_layer_open_tensor(A_BR)
    # Closed double-layer tensors for the traced corners (TR, BL).
    ac_TR_base = _build_double_layer_tensor(A_TR)
    ac_BL_base = _build_double_layer_tensor(A_BL)

    # Right-column copies (suffix R) of the right-column edges.
    T1_R = env_TR.T1.relabels({"t1_l": "t1_lR", "u2": "u2R", "t1_r": "t1_rR"})
    T3_R = env_BR.T3.relabels({"t3_r": "t3_rR", "d2": "d2R", "t3_l": "t3_lR"})

    # Bottom-row copies (suffix B) of the bottom-row edges.
    T4_B = env_BL.T4.relabels({"t4_d": "t4_dB", "l2": "l2B", "t4_u": "t4_uB"})
    T2_B = env_BR.T2.relabels({"t2_u": "t2_uB", "r2": "r2B", "t2_d": "t2_dB"})

    # Site-internal labels (positional, identical to the single-site version
    # — applied to the per-position ao/ac built from the per-position A).
    #   - TL: keeps base names (u2, d2, l2, r2, phys, phys_bra).
    #   - TR (closed): inner-left bond ↔ TL.r2, inner-down bond ↔ BR.u2_BR.
    #   - BL (closed): inner-up bond ↔ TL.d2, inner-right bond ↔ BR.l2_BR.
    #   - BR (open):   inner-left bond ↔ BL.r2_BL, inner-up bond ↔ TR.d2_TR.
    ac_TR = ac_TR_base.relabels(
        {"u2": "u2R", "d2": "d2_TR", "l2": "l2_TR", "r2": "r2R"}
    )
    ac_BL = ac_BL_base.relabels(
        {"u2": "u2_BL", "d2": "d2B_BL", "l2": "l2B", "r2": "r2_BL"}
    )
    ao_BR = ao_BR_base.relabels(
        {
            "u2": "u2_BR",
            "d2": "d2R",
            "l2": "l2_BR",
            "r2": "r2B",
            "phys": "phys_BR",
            "phys_bra": "phys_bra_BR",
        }
    )

    # ---------- Per-site envs (one ao/ac per site, contracted in place) ----------
    # site_env_TL:  C1·T1·T4_T·ao_TL  →  (t1_r, t4_u, d2, r2, phys, phys_bra)
    C1 = env_TL.C1.relabel("c1_r", "t1_l")
    C1_T1 = contract(C1, env_TL.T1)  # (c1_d, u2, t1_r)
    T4_T = env_TL.T4.relabels({"t4_d": "c1_d"})  # (c1_d, l2, t4_u)
    TL_env = contract(C1_T1, T4_T)  # (u2, t1_r, l2, t4_u)
    site_env_TL = contract(TL_env, ao_TL)  # (t1_r, t4_u, d2, r2, phys, phys_bra)

    # site_env_TR:  T1_R·C2·T2_T·ac_TR  →  (t1_lR, t2_d, l2_TR, d2_TR)
    C2 = env_TR.C2.relabel("c2_l", "t1_rR")
    T1R_C2 = contract(T1_R, C2)  # (t1_lR, u2R, c2_d)
    T2_T = env_TR.T2.relabels({"t2_u": "c2_d", "r2": "r2R"})  # (c2_d, r2R, t2_d)
    TR_env = contract(T1R_C2, T2_T)  # (t1_lR, u2R, r2R, t2_d)
    site_env_TR = contract(TR_env, ac_TR)  # (t1_lR, t2_d, l2_TR, d2_TR)

    # site_env_BL:  C4·T3·T4_B·ac_BL  →  (t3_l, t4_dB, u2_BL, r2_BL)
    C4 = env_BL.C4.relabel("c4_u", "t3_r")
    T3_BL = env_BL.T3.relabel("d2", "d2B_BL")  # (t3_r, d2B_BL, t3_l)
    C4_T3 = contract(C4, T3_BL)  # (c4_r, d2B_BL, t3_l)
    T4_BL = T4_B.relabel("t4_uB", "c4_r")  # (t4_dB, l2B, c4_r)
    BL_env = contract(C4_T3, T4_BL)  # (d2B_BL, t3_l, t4_dB, l2B)
    site_env_BL = contract(BL_env, ac_BL)  # (t3_l, t4_dB, u2_BL, r2_BL)

    # site_env_BR:  T3_R·C3·T2_B·ao_BR
    #   →  (t3_rR, t2_uB, u2_BR, l2_BR, phys_BR, phys_bra_BR)
    C3 = env_BR.C3.relabel("c3_l", "t3_lR")
    T3R_C3 = contract(T3_R, C3)  # (t3_rR, d2R, c3_u)
    T2_BR = T2_B.relabel("t2_dB", "c3_u")  # (t2_uB, r2B, c3_u)
    BR_env = contract(T3R_C3, T2_BR)  # (t3_rR, d2R, t2_uB, r2B)
    site_env_BR = contract(BR_env, ao_BR)
    # (t3_rR, t2_uB, u2_BR, l2_BR, phys_BR, phys_bra_BR)

    # ---------- Combine columns ----------
    # left_half = site_env_TL · site_env_BL  on (t4_u↔t4_dB, d2↔u2_BL)
    site_env_BL = site_env_BL.relabels({"t4_dB": "t4_u", "u2_BL": "d2"})
    left_half = contract(site_env_TL, site_env_BL)
    # (t1_r, r2, phys, phys_bra, t3_l, r2_BL)

    # right_half = site_env_TR · site_env_BR  on (t2_d↔t2_uB, d2_TR↔u2_BR)
    site_env_BR = site_env_BR.relabels({"t2_uB": "t2_d", "u2_BR": "d2_TR"})
    right_half = contract(site_env_TR, site_env_BR)
    # (t1_lR, l2_TR, t3_rR, l2_BR, phys_BR, phys_bra_BR)

    # Final combine: shared (t1_r↔t1_lR, t3_l↔t3_rR, r2↔l2_TR, r2_BL↔l2_BR)
    right_half = right_half.relabels(
        {"t1_lR": "t1_r", "l2_TR": "r2", "l2_BR": "r2_BL", "t3_rR": "t3_l"}
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


def _rdm2x1_tensor(A: Tensor, env: CTMTensorEnv) -> jax.Array:
    """Horizontal 2×1 RDM using label-based Tensor contractions.

    Contracts the network::

        C1 — T1_L — T1_R — C2
        |      |       |      |
        T4   ao_L    ao_R    T2
        |      |       |      |
        C4 — T3_L — T3_R — C3

    Returns dense RDM of shape ``(d, d, d, d)`` in
    ``(s1_ket, s2_ket, s1_bra, s2_bra)`` convention, symmetrised and
    normalised.

    Bond connectivity (verified against the dense CTMEnvironment convention):
        C1[0]=c1_d ↔ T4[0]=t4_d,  C1[1]=c1_r ↔ T1[0]=t1_l
        T1[2]=t1_r ↔ C2[0]=c2_l,  C2[1]=c2_d ↔ T2[0]=t2_u
        C4[1]=c4_u ↔ T3[0]=t3_r,  C4[0]=c4_r ↔ T4[2]=t4_u
        T3[2]=t3_l ↔ C3[1]=c3_l,  T2[2]=t2_d ↔ C3[0]=c3_u
    """
    ao = _build_double_layer_open_tensor(A)  # (u2, d2, l2, r2, phys, phys_bra)

    # Right-side copies: relabel to avoid conflicts
    T1_R = env.T1.relabels({"t1_l": "t1_lR", "u2": "u2R", "t1_r": "t1_rR"})
    T3_R = env.T3.relabels({"t3_r": "t3_rR", "d2": "d2R", "t3_l": "t3_lR"})
    ao_R = ao.relabels(
        {
            "u2": "u2R",
            "d2": "d2R",
            "l2": "l2R",
            "r2": "r2R",
            "phys": "phys_R",
            "phys_bra": "phys_braR",
        }
    )

    # Step 1: UL = C1·T1_L  — contract c1_r ↔ t1_l
    C1 = env.C1.relabel("c1_r", "t1_l")
    UL = contract(C1, env.T1)  # (c1_d, u2, t1_r)

    # Step 2: UR = T1_R·C2  — contract t1_rR ↔ c2_l
    C2 = env.C2.relabel("c2_l", "t1_rR")
    UR = contract(T1_R, C2)  # (t1_lR, u2R, c2_d)

    # Step 3: LL = C4·T3_L  — contract c4_u ↔ t3_r
    C4 = env.C4.relabel("c4_u", "t3_r")
    LL = contract(C4, env.T3)  # (c4_r, d2, t3_l)

    # Step 4: LR = T3_R·C3  — contract t3_lR ↔ c3_l
    C3 = env.C3.relabel("c3_l", "t3_lR")
    LR = contract(T3_R, C3)  # (t3_rR, d2R, c3_u)

    # Step 5: Lenv = UL·T4·LL (pairwise for SymmetricTensor safety)
    #   c1_d ↔ t4_d,  t4_u ↔ c4_r
    T4 = env.T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})
    UL_T4 = contract(UL, T4)  # (u2, t1_r, l2, c4_r)
    Lenv = contract(UL_T4, LL)  # (u2, t1_r, l2, d2, t3_l)

    # Step 6: Renv = UR·T2·LR (pairwise)
    #   c2_d ↔ t2_u,  t2_d ↔ c3_u,  r2 → r2R to match ao_R
    T2 = env.T2.relabels({"t2_u": "c2_d", "r2": "r2R", "t2_d": "c3_u"})
    UR_T2 = contract(UR, T2)  # (t1_lR, u2R, r2R, c3_u)
    Renv = contract(UR_T2, LR)  # (t1_lR, u2R, r2R, t3_rR, d2R)

    # Step 7: contract Lenv with ao_L  — shared: u2, d2, l2
    Lenv_ao = contract(Lenv, ao)  # (t1_r, r2, t3_l, phys, phys_bra)

    # Step 8: contract Renv with ao_R  — shared: u2R, d2R, r2R
    Renv_ao = contract(Renv, ao_R)  # (t1_lR, t3_rR, l2R, phys_R, phys_braR)

    # Step 9: contract Lenv_ao with Renv_ao
    #   Match: t1_r ↔ t1_lR, t3_l ↔ t3_rR, r2 ↔ l2R
    Renv_ao = Renv_ao.relabels({"t1_lR": "t1_r", "t3_rR": "t3_l", "l2R": "r2"})
    rdm_t = contract(
        Lenv_ao,
        Renv_ao,
        output_labels=["phys", "phys_R", "phys_bra", "phys_braR"],
    )
    # → (s1_ket, s2_ket, s1_bra, s2_bra)

    # RDM is d^2 x d^2 (physical dimension) — always small, todense OK.
    rdm = rdm_t.todense()
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + EPS)
    return rdm_mat.reshape(d, d, d, d)


def _rdm1x2_tensor(A: Tensor, env: CTMTensorEnv) -> jax.Array:
    """Vertical 1×2 RDM using label-based Tensor contractions.

    Contracts the network::

        C1  — T1  — C2
        |      |      |
        T4_T  ao_T   T2_T
        |      |      |
        T4_B  ao_B   T2_B
        |      |      |
        C4  — T3  — C3

    Returns dense RDM of shape ``(d, d, d, d)`` in
    ``(s1_ket, s2_ket, s1_bra, s2_bra)`` convention.

    Mirrors :func:`_rdm2x1_tensor`'s left/right-halves contraction order
    (top/bottom halves here): each half consumes its ``ao`` factor while
    three boundary D² legs are still attached, so the peak intermediate
    is bounded by ``(chi · chi · D² · d · d)`` instead of the
    ``(chi² · D⁴ · d²)`` blow-up that an "all-rows-then-ao" path produces.
    """
    ao = _build_double_layer_open_tensor(A)

    # Bottom copies: relabel with "B" suffix
    T4_B = env.T4.relabels({"t4_d": "t4_dB", "l2": "l2B", "t4_u": "t4_uB"})
    T2_B = env.T2.relabels({"t2_u": "t2_uB", "r2": "r2B", "t2_d": "t2_dB"})
    ao_B = ao.relabels(
        {
            "u2": "u2B",
            "d2": "d2B",
            "l2": "l2B",
            "r2": "r2B",
            "phys": "phys_B",
            "phys_bra": "phys_braB",
        }
    )

    # ---------- Top half ----------
    # Step T1: top_row = C1·T1·C2  (chi, D², chi)
    C1 = env.C1.relabel("c1_r", "t1_l")
    C2 = env.C2.relabel("c2_l", "t1_r")
    C1_T1 = contract(C1, env.T1)  # (c1_d, u2, t1_r)
    top_row = contract(C1_T1, C2)  # (c1_d, u2, c2_d)

    # Step T2: env_top = top_row·T4_T·T2_T
    T4_T = env.T4.relabels({"t4_d": "c1_d"})  # (c1_d, l2, t4_u)
    T2_T = env.T2.relabels({"t2_u": "c2_d"})  # (c2_d, r2, t2_d)
    top_T4 = contract(top_row, T4_T)  # (u2, c2_d, l2, t4_u)
    env_top = contract(top_T4, T2_T)  # (u2, l2, t4_u, r2, t2_d)

    # Step T3: top_half = env_top · ao  — consume u2, l2, r2 simultaneously
    top_half = contract(env_top, ao)  # (t4_u, t2_d, d2, phys, phys_bra)

    # ---------- Bottom half ----------
    # Step B1: bot_row = C4·T3·C3  (chi, D², chi)
    C4 = env.C4.relabel("c4_u", "t3_r")
    T3_B = env.T3.relabel("d2", "d2B")
    C3 = env.C3.relabel("c3_l", "t3_l")
    C4_T3 = contract(C4, T3_B)  # (c4_r, d2B, t3_l)
    bot_row = contract(C4_T3, C3)  # (c4_r, d2B, c3_u)

    # Step B2: env_bot = bot_row·T4_B·T2_B
    T4_B_t = T4_B.relabel("t4_uB", "c4_r")  # (t4_dB, l2B, c4_r)
    T2_B_t = T2_B.relabel("t2_dB", "c3_u")  # (t2_uB, r2B, c3_u)
    bot_T4 = contract(bot_row, T4_B_t)  # (d2B, c3_u, t4_dB, l2B)
    env_bot = contract(bot_T4, T2_B_t)  # (d2B, t4_dB, l2B, r2B, t2_uB)

    # Step B3: bot_half = env_bot · ao_B  — consume d2B, l2B, r2B simultaneously
    bot_half = contract(env_bot, ao_B)  # (t4_dB, t2_uB, u2B, phys_B, phys_braB)

    # ---------- Combine ----------
    # Match t4_u ↔ t4_dB (T4 chi), t2_d ↔ t2_uB (T2 chi), d2 ↔ u2B (inner ao bond)
    bot_half = bot_half.relabels({"t4_dB": "t4_u", "t2_uB": "t2_d", "u2B": "d2"})
    rdm_t = contract(
        top_half,
        bot_half,
        output_labels=["phys", "phys_B", "phys_bra", "phys_braB"],
    )

    # RDM is d^2 x d^2 (physical dimension) — always small, todense OK.
    rdm = rdm_t.todense()
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + EPS)
    return rdm_mat.reshape(d, d, d, d)


def compute_energy_ctm_tensor(
    A: Tensor,
    env: CTMTensorEnv,
    hamiltonian_gate: Tensor | jax.Array,
    d: int | None = None,
) -> jax.Array:
    """Compute energy per site using a standard Tensor-protocol CTM environment.

    Uses native Tensor contractions for the RDM computation, avoiding
    densification of the full chi-dimensional environment.

    Args:
        A:                iPEPS site tensor with labels ``(u, d, l, r, phys)``.
        env:              Converged CTMTensorEnv.
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

    rdm_h = _rdm2x1_tensor(A, env)
    rdm_v = _rdm1x2_tensor(A, env)
    E_h = jnp.einsum("ijkl,ijkl->", rdm_h, H)
    E_v = jnp.einsum("ijkl,ijkl->", rdm_v, H)
    return (E_h + E_v).real


def _rdm2x1_tensor_2site(
    A: Tensor,
    B: Tensor,
    env_A: CTMTensorEnv,
    env_B: CTMTensorEnv,
) -> jax.Array:
    """Horizontal 2×1 RDM for checkerboard (A left, B right).

    Mixed environment::

        C1_A — T1_A — T1_B — C2_B
        |        |       |       |
        T4_A   ao_A    ao_B    T2_B
        |        |       |       |
        C4_A — T3_A — T3_B — C3_B
    """
    ao_A = _build_double_layer_open_tensor(A)
    ao_B = _build_double_layer_open_tensor(B)

    # Right-side copies from env_B
    T1_R = env_B.T1.relabels({"t1_l": "t1_lR", "u2": "u2R", "t1_r": "t1_rR"})
    T3_R = env_B.T3.relabels({"t3_r": "t3_rR", "d2": "d2R", "t3_l": "t3_lR"})
    ao_R = ao_B.relabels(
        {
            "u2": "u2R",
            "d2": "d2R",
            "l2": "l2R",
            "r2": "r2R",
            "phys": "phys_R",
            "phys_bra": "phys_braR",
        }
    )

    # Left boundary from env_A
    C1 = env_A.C1.relabel("c1_r", "t1_l")
    UL = contract(C1, env_A.T1)
    C4 = env_A.C4.relabel("c4_u", "t3_r")
    LL = contract(C4, env_A.T3)
    T4 = env_A.T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})
    UL_T4 = contract(UL, T4)
    Lenv = contract(UL_T4, LL)

    # Right boundary from env_B
    C2 = env_B.C2.relabel("c2_l", "t1_rR")
    UR = contract(T1_R, C2)
    C3 = env_B.C3.relabel("c3_l", "t3_lR")
    LR = contract(T3_R, C3)
    T2 = env_B.T2.relabels({"t2_u": "c2_d", "r2": "r2R", "t2_d": "c3_u"})
    UR_T2 = contract(UR, T2)
    Renv = contract(UR_T2, LR)

    # Contract with site tensors
    Lenv_ao = contract(Lenv, ao_A)
    Renv_ao = contract(Renv, ao_R)
    Renv_ao = Renv_ao.relabels({"t1_lR": "t1_r", "t3_rR": "t3_l", "l2R": "r2"})
    rdm_t = contract(
        Lenv_ao,
        Renv_ao,
        output_labels=["phys", "phys_R", "phys_bra", "phys_braR"],
    )

    # RDM is d^2 x d^2 (physical dimension) — always small, todense OK.
    rdm = rdm_t.todense()
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + EPS)
    return rdm_mat.reshape(d, d, d, d)


def _rdm1x2_tensor_2site(
    A: Tensor,
    B: Tensor,
    env_A: CTMTensorEnv,
    env_B: CTMTensorEnv,
) -> jax.Array:
    """Vertical 1×2 RDM for checkerboard (A top, B bottom).

    Mixed environment::

        C1_A — T1_A — C2_A
        |        |       |
        T4_A   ao_A    T2_A
        |        |       |
        T4_B   ao_B    T2_B
        |        |       |
        C4_B — T3_B — C3_B

    Mirrors :func:`_rdm2x1_tensor_2site`'s left/right-halves order
    (top/bottom halves here): each half consumes its ``ao`` factor while
    three boundary D² legs are still attached, so the peak intermediate
    is bounded by ``(chi · chi · D² · d · d)``.
    """
    ao_A = _build_double_layer_open_tensor(A)
    ao_B = _build_double_layer_open_tensor(B)

    # Bottom copies from env_B
    T4_B = env_B.T4.relabels({"t4_d": "t4_dB", "l2": "l2B", "t4_u": "t4_uB"})
    T2_B = env_B.T2.relabels({"t2_u": "t2_uB", "r2": "r2B", "t2_d": "t2_dB"})
    ao_Br = ao_B.relabels(
        {
            "u2": "u2B",
            "d2": "d2B",
            "l2": "l2B",
            "r2": "r2B",
            "phys": "phys_B",
            "phys_bra": "phys_braB",
        }
    )

    # ---------- Top half (env_A) ----------
    C1 = env_A.C1.relabel("c1_r", "t1_l")
    C2 = env_A.C2.relabel("c2_l", "t1_r")
    C1_T1 = contract(C1, env_A.T1)
    top_row = contract(C1_T1, C2)  # (c1_d, u2, c2_d)

    T4_T = env_A.T4.relabels({"t4_d": "c1_d"})  # (c1_d, l2, t4_u)
    T2_T = env_A.T2.relabels({"t2_u": "c2_d"})  # (c2_d, r2, t2_d)
    top_T4 = contract(top_row, T4_T)  # (u2, c2_d, l2, t4_u)
    env_top = contract(top_T4, T2_T)  # (u2, l2, t4_u, r2, t2_d)

    top_half = contract(env_top, ao_A)  # (t4_u, t2_d, d2, phys, phys_bra)

    # ---------- Bottom half (env_B) ----------
    C4 = env_B.C4.relabel("c4_u", "t3_r")
    T3_BL = env_B.T3.relabel("d2", "d2B")
    C3 = env_B.C3.relabel("c3_l", "t3_l")
    C4_T3 = contract(C4, T3_BL)  # (c4_r, d2B, t3_l)
    bot_row = contract(C4_T3, C3)  # (c4_r, d2B, c3_u)

    T4_B_t = T4_B.relabel("t4_uB", "c4_r")  # (t4_dB, l2B, c4_r)
    T2_B_t = T2_B.relabel("t2_dB", "c3_u")  # (t2_uB, r2B, c3_u)
    bot_T4 = contract(bot_row, T4_B_t)  # (d2B, c3_u, t4_dB, l2B)
    env_bot = contract(bot_T4, T2_B_t)  # (d2B, t4_dB, l2B, r2B, t2_uB)

    bot_half = contract(env_bot, ao_Br)
    # (t4_dB, t2_uB, u2B, phys_B, phys_braB)

    # ---------- Combine ----------
    bot_half = bot_half.relabels({"t4_dB": "t4_u", "t2_uB": "t2_d", "u2B": "d2"})
    rdm_t = contract(
        top_half,
        bot_half,
        output_labels=["phys", "phys_B", "phys_bra", "phys_braB"],
    )

    # RDM is d^2 x d^2 (physical dimension) — always small, todense OK.
    rdm = rdm_t.todense()
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + EPS)
    return rdm_mat.reshape(d, d, d, d)


def compute_energy_ctm_tensor_2site(
    A: Tensor,
    B: Tensor,
    env_A: CTMTensorEnv,
    env_B: CTMTensorEnv,
    hamiltonian_gate: Tensor | jax.Array,
    d: int | None = None,
) -> jax.Array:
    """Compute energy per site for a 2-site checkerboard iPEPS.

    Uses native Tensor contractions for the RDM computation, avoiding
    densification of the chi-dimensional environment.

    Args:
        A:                Site tensor for sublattice A.
        B:                Site tensor for sublattice B.
        env_A:            Converged CTMTensorEnv for sublattice A.
        env_B:            Converged CTMTensorEnv for sublattice B.
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

    rdm_h = _rdm2x1_tensor_2site(A, B, env_A, env_B)
    rdm_v = _rdm1x2_tensor_2site(A, B, env_A, env_B)
    E_h = jnp.einsum("ijkl,ijkl->", rdm_h, H)
    E_v = jnp.einsum("ijkl,ijkl->", rdm_v, H)
    return (E_h + E_v).real


def compute_energy_ctm_tensor_multisite(
    site_tensors: dict,
    envs: dict,
    neighbors: dict,
    gate: Tensor | jax.Array,
    d: int | None = None,
) -> jax.Array:
    """Compute energy per site summed over all NN bonds in a multi-site unit cell.

    Each bond is counted once. Energy is normalized by the number of sites.

    Args:
        site_tensors: ``{coord: Tensor}`` mapping coordinates to iPEPS site tensors.
        envs:         ``{coord: CTMTensorEnv}`` converged environments per site.
        neighbors:    ``{coord: {"left": coord, "right": coord, "top": coord,
                      "bottom": coord}}`` neighbor map defining the unit cell topology.
        gate:         2-site Hamiltonian gate (dense array or Tensor).
        d:            Physical dimension (inferred from first site tensor if None).

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
                    rdm = _rdm2x1_tensor(A, env_A)
                else:
                    rdm = _rdm1x2_tensor(A, env_A)
            else:
                if direction == "right":
                    rdm = _rdm2x1_tensor_2site(A, B, env_A, env_B)
                else:
                    rdm = _rdm1x2_tensor_2site(A, B, env_A, env_B)

            bond_energy = jnp.einsum("ijkl,ijkl->", rdm, H)
            total_energy = total_energy + bond_energy

    return total_energy.real / n_sites
