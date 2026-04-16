"""Standard CTM with Tensor protocol — RDM construction and energy computation."""

from __future__ import annotations

__all__ = [
    "_rdm_1site_tensor",
    "_rdm_diagonal_tensor",
    "_rdm1x2_tensor",
    "_rdm1x2_tensor_2site",
    "_rdm2x1_tensor",
    "_rdm2x1_tensor_2site",
    "compute_energy_ctm_tensor",
    "compute_energy_ctm_tensor_2site",
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

    Bond connectivity (same as 2x1/1x2 conventions):
        C1[0]=c1_d ↔ T4[0]=t4_d,  C1[1]=c1_r ↔ T1[0]=t1_l
        T1[2]=t1_r ↔ C2[0]=c2_l,  C2[1]=c2_d ↔ T2[0]=t2_u
        C4[1]=c4_u ↔ T3[0]=t3_r,  C4[0]=c4_r ↔ T4[2]=t4_u
        T3[2]=t3_l ↔ C3[1]=c3_l,  T2[2]=t2_d ↔ C3[0]=c3_u
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

    # Step 7: symmetrise and trace-normalise
    rdm = rdm_t.todense()
    rdm = 0.5 * (rdm + rdm.conj().T)
    rdm = rdm / (jnp.trace(rdm) + EPS)
    return rdm


def _rdm_diagonal_tensor(A: Tensor, env: CTMTensorEnv) -> jax.Array:
    r"""Diagonal (NNN) 2-site RDM from a 2×2 plaquette contraction.

    Contracts the network::

        C1 — T1_L — T1_R — C2
        |      |       |      |
        T4_T  ao_TL   ac_TR  T2_T
        |      |       |      |
        T4_B  ac_BL   ao_BR  T2_B
        |      |       |      |
        C4 — T3_L — T3_R — C3

    Top-left (TL) and bottom-right (BR) sites have physical legs open;
    top-right (TR) and bottom-left (BL) have physical legs traced (closed).

    Returns dense RDM of shape ``(d, d, d, d)`` in
    ``(s1_ket, s2_ket, s1_bra, s2_bra)`` convention (TL=s1, BR=s2),
    symmetrised and trace-normalised.
    """
    ao = _build_double_layer_open_tensor(A)  # (u2, d2, l2, r2, phys, phys_bra)
    ac = _build_double_layer_tensor(A)  # (u2, d2, l2, r2)

    # --- Right-column copies (suffix R) ---
    T1_R = env.T1.relabels({"t1_l": "t1_lR", "u2": "u2R", "t1_r": "t1_rR"})
    T3_R = env.T3.relabels({"t3_r": "t3_rR", "d2": "d2R", "t3_l": "t3_lR"})

    # --- Bottom-row copies (suffix B) ---
    T4_B = env.T4.relabels({"t4_d": "t4_dB", "l2": "l2B", "t4_u": "t4_uB"})
    T2_B = env.T2.relabels({"t2_u": "t2_uB", "r2": "r2B", "t2_d": "t2_dB"})

    # --- Site tensors ---
    # ao_TL: top-left, open.  Keep default labels (u2, d2, l2, r2, phys, phys_bra)
    # ac_TR: top-right, closed.  Suffix R on virtual labels.
    ac_TR = ac.relabels({"u2": "u2R", "d2": "d2_TR", "l2": "l2_TR", "r2": "r2R"})
    # ac_BL: bottom-left, closed.  Suffix B on vertical, _BL on horizontal.
    ac_BL = ac.relabels({"u2": "u2_BL", "d2": "d2B_BL", "l2": "l2B", "r2": "r2_BL"})
    # ao_BR: bottom-right, open.  Suffix BR on virtual, distinct phys labels.
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

    # ===============================================================
    # Left half: C1 · T1_L · T4_T · ao_TL · T4_B · ac_BL · C4 · T3_L
    # ===============================================================

    # Step 1: top-left corner: C1 · T1_L
    C1 = env.C1.relabel("c1_r", "t1_l")
    C1_T1L = contract(C1, env.T1)  # (c1_d, u2, t1_r)

    # Step 2: add T4_T (connects c1_d ↔ t4_d)
    T4_T = env.T4.relabels({"t4_d": "c1_d"})
    left_top = contract(C1_T1L, T4_T)  # (u2, t1_r, l2, t4_u)

    # Step 3: contract with ao_TL (shared: u2, l2)
    left_top_ao = contract(left_top, ao)
    # (t1_r, t4_u, d2, r2, phys, phys_bra)

    # Step 4: add T4_B (connects t4_u ↔ t4_dB)
    T4_B = T4_B.relabel("t4_dB", "t4_u")
    left_mid = contract(left_top_ao, T4_B)
    # (t1_r, d2, r2, phys, phys_bra, l2B, t4_uB)

    # Step 5: contract with ac_BL (shared: d2 ↔ u2_BL, l2B)
    ac_BL = ac_BL.relabel("u2_BL", "d2")
    left_site = contract(left_mid, ac_BL)
    # (t1_r, r2, phys, phys_bra, t4_uB, d2B_BL, r2_BL)

    # Step 6: bottom-left corner: C4 · T3_L
    C4 = env.C4.relabel("c4_u", "t3_r")
    C4_T3L = contract(C4, env.T3)  # (c4_r, d2, t3_l)
    # Connect: t4_uB ↔ c4_r, d2B_BL ↔ d2
    C4_T3L = C4_T3L.relabels({"c4_r": "t4_uB", "d2": "d2B_BL"})
    left_half = contract(left_site, C4_T3L)
    # (t1_r, r2, phys, phys_bra, r2_BL, t3_l)

    # ===============================================================
    # Right half: C2 · T1_R · T2_T · ac_TR · T2_B · ao_BR · C3 · T3_R
    # ===============================================================

    # Step 7: top-right corner: T1_R · C2
    C2 = env.C2.relabel("c2_l", "t1_rR")
    T1R_C2 = contract(T1_R, C2)  # (t1_lR, u2R, c2_d)

    # Step 8: add T2_T (connects c2_d ↔ t2_u)
    T2_T = env.T2.relabels({"t2_u": "c2_d"})
    right_top = contract(T1R_C2, T2_T)  # (t1_lR, u2R, r2, t2_d)

    # Step 9: contract with ac_TR (shared: u2R, r2R→r2 already named r2R in ac_TR)
    # ac_TR has labels (u2R, d2_TR, l2_TR, r2R). right_top has r2 (not r2R).
    # Relabel ac_TR's r2R to match right_top's r2:
    ac_TR = ac_TR.relabel("r2R", "r2")
    right_top_ac = contract(right_top, ac_TR)
    # (t1_lR, t2_d, d2_TR, l2_TR)

    # Step 10: add T2_B (connects t2_d ↔ t2_uB)
    T2_B = T2_B.relabel("t2_uB", "t2_d")
    right_mid = contract(right_top_ac, T2_B)
    # (t1_lR, d2_TR, l2_TR, r2B, t2_dB)

    # Step 11: contract with ao_BR (shared: d2_TR ↔ u2_BR, r2B)
    ao_BR = ao_BR.relabel("u2_BR", "d2_TR")
    right_site = contract(right_mid, ao_BR)
    # (t1_lR, l2_TR, t2_dB, d2R, l2_BR, phys_BR, phys_bra_BR)

    # Step 12: bottom-right corner: T3_R · C3
    C3 = env.C3.relabel("c3_l", "t3_lR")
    T3R_C3 = contract(T3_R, C3)  # (t3_rR, d2R, c3_u)
    # Connect: t2_dB ↔ c3_u, d2R shared
    T3R_C3 = T3R_C3.relabel("c3_u", "t2_dB")
    right_half = contract(right_site, T3R_C3)
    # (t1_lR, l2_TR, l2_BR, phys_BR, phys_bra_BR, t3_rR)

    # ===============================================================
    # Combine left and right halves
    # ===============================================================
    # Match bonds between left and right:
    #   t1_r ↔ t1_lR  (top chi bond between columns)
    #   r2 ↔ l2_TR    (TL-to-TR virtual bond)
    #   r2_BL ↔ l2_BR (BL-to-BR virtual bond)
    #   t3_l ↔ t3_rR  (bottom chi bond between columns)
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

    # Step 1: top_row = C1·T1·C2
    C1 = env.C1.relabel("c1_r", "t1_l")
    C2 = env.C2.relabel("c2_l", "t1_r")
    C1_T1 = contract(C1, env.T1)  # (c1_d, u2, t1_r)
    top_row = contract(C1_T1, C2)  # (c1_d, u2, c2_d)

    # Step 2: env_row1 = top_row·T4_T·T2_T (pairwise)
    T4_T = env.T4.relabels({"t4_d": "c1_d"})
    T2_T = env.T2.relabels({"t2_u": "c2_d"})
    top_T4 = contract(top_row, T4_T)  # (u2, c2_d, l2, t4_u)
    env_row1 = contract(top_T4, T2_T)  # (u2, l2, t4_u, r2, t2_d)

    # Step 3: site1 = env_row1·ao_T  — shared: u2, l2, r2
    site1 = contract(env_row1, ao)  # (d2, t4_u, t2_d, phys, phys_bra)

    # Step 4: T4_ao_B = T4_B·ao_B  — shared: l2B
    T4_ao_B = contract(T4_B, ao_B)
    # (t4_dB, t4_uB, u2B, d2B, r2B, phys_B, phys_braB)

    # Step 5: site12 = site1·T4_ao_B
    #   d2 ↔ u2B (vertical bond), t4_u ↔ t4_dB (T4 chi)
    T4_ao_B = T4_ao_B.relabels({"u2B": "d2", "t4_dB": "t4_u"})
    site12 = contract(site1, T4_ao_B)
    # (t2_d, phys, phys_bra, t4_uB, d2B, r2B, phys_B, phys_braB)

    # Step 6: site12_r = site12·T2_B
    #   t2_d ↔ t2_uB (T2 chi),  r2B shared
    T2_B = T2_B.relabel("t2_uB", "t2_d")
    site12_r = contract(site12, T2_B)
    # (phys, phys_bra, t4_uB, d2B, phys_B, phys_braB, t2_dB)

    # Step 7: bot_row = C4·T3·C3  — T3.d2 relabeled to d2B for matching
    C4 = env.C4.relabel("c4_u", "t3_r")
    T3 = env.T3.relabel("d2", "d2B")
    C3 = env.C3.relabel("c3_l", "t3_l")
    C4_T3 = contract(C4, T3)  # (c4_r, d2B, t3_l)
    bot_row = contract(C4_T3, C3)  # (c4_r, d2B, c3_u)

    # Step 8: rdm = site12_r·bot_row
    #   t4_uB ↔ c4_r, d2B shared, t2_dB ↔ c3_u
    bot_row = bot_row.relabels({"c4_r": "t4_uB", "c3_u": "t2_dB"})
    rdm_t = contract(
        site12_r,
        bot_row,
        output_labels=["phys", "phys_B", "phys_bra", "phys_braB"],
    )

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

    # Top row from env_A
    C1 = env_A.C1.relabel("c1_r", "t1_l")
    C2 = env_A.C2.relabel("c2_l", "t1_r")
    C1_T1 = contract(C1, env_A.T1)
    top_row = contract(C1_T1, C2)

    # Top env row from env_A
    T4_T = env_A.T4.relabels({"t4_d": "c1_d"})
    T2_T = env_A.T2.relabels({"t2_u": "c2_d"})
    top_T4 = contract(top_row, T4_T)
    env_row1 = contract(top_T4, T2_T)

    # Contract with ao_A (top site)
    site1 = contract(env_row1, ao_A)

    # Bottom: T4_B · ao_B
    T4_ao_B = contract(T4_B, ao_Br)
    T4_ao_B = T4_ao_B.relabels({"u2B": "d2", "t4_dB": "t4_u"})
    site12 = contract(site1, T4_ao_B)

    # T2_B
    T2_B = T2_B.relabel("t2_uB", "t2_d")
    site12_r = contract(site12, T2_B)

    # Bottom row from env_B
    C4 = env_B.C4.relabel("c4_u", "t3_r")
    T3 = env_B.T3.relabel("d2", "d2B")
    C3 = env_B.C3.relabel("c3_l", "t3_l")
    C4_T3 = contract(C4, T3)
    bot_row = contract(C4_T3, C3)

    bot_row = bot_row.relabels({"c4_r": "t4_uB", "c3_u": "t2_dB"})
    rdm_t = contract(
        site12_r,
        bot_row,
        output_labels=["phys", "phys_B", "phys_bra", "phys_braB"],
    )

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
