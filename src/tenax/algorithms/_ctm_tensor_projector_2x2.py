"""2x2 plaquette enlarged-corner builder for the multisite CTM projector.

Implements the standard CTMRG enlarged-corner construction (Corboz, Penc,
Mila, Lauchli, PRB 84, 041108(R) (2011)). For each plaquette quarter,
contracts one corner C, two adjacent edges T_h and T_v, and the double-
layer site tensor a into a rank-4 tensor with two seam legs (the chi
and D^2 legs that connect to the adjacent quarter in the 2x2).

Used by ``_ctm_tensor_move_*_2x2`` in ``_ctm_tensor_moves.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor

__all__ = ["_build_enlarged_corner", "_compute_2x2_projector"]


def _gauge_fixed_svd(
    M: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reconstruction-preserving gauge-fixed SVD for the 2x2 projector.

    Returns ``(U, s, Vh)`` with each column of ``U`` and matching row of
    ``Vh`` rephased so the row of largest ``|U|`` is real-positive. Uses the
    variPEPS / YASTN convention of putting ``conj(phase)`` on ``U`` and the
    *bare* ``phase`` on ``Vh``, which preserves the SVD reconstruction
    ``U @ diag(s) @ Vh == M`` even for complex inputs.

    The shared :func:`tenax.algorithms._ad_primitives._fix_svd_signs` puts
    ``conj(phase)`` on both factors, so ``U @ diag(s) @ Vh`` picks up a
    ``conj(phase)**2`` factor. That is fine for the 1x1 Fishman closure
    ``P1^H @ M @ P2 = I`` because the middle ``M`` absorbs the phase
    mismatch, but it breaks the 2x2 closure ``P_bot @ P_top = I`` which
    has no intervening matrix.
    """
    U, s, Vh = jnp.linalg.svd(M, full_matrices=False)
    max_idx = jnp.argmax(jnp.abs(U), axis=0)  # (k,)
    diag = U[max_idx, jnp.arange(U.shape[1])]
    phases = jnp.where(jnp.abs(diag) > 0, diag / jnp.abs(diag), 1.0)
    U = U * jnp.conj(phases)[None, :]
    Vh = Vh * phases[:, None]
    return U, s, Vh


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


# ---------------------------------------------------------------- #
# Fishman 2x2 plaquette cross-projector                              #
# ---------------------------------------------------------------- #


def _fishman_truncate_S(S: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Zero out singular values whose ratio to the largest falls below eps.

    Mirrors the Fishman SVD truncation used in the 1x1 projector path
    (`_ctm_projector.py`).
    """
    s_max = S[0]
    return jnp.where(S / (s_max + 1e-30) >= eps, S, 0.0)


def _compute_2x2_projector(
    Q_TL: Tensor,
    Q_TR: Tensor,
    Q_BL: Tensor,
    Q_BR: Tensor,
    chi: int,
    *,
    direction: str = "left",
) -> tuple[Tensor, Tensor]:
    r"""Fishman 2x2 plaquette cross-projector for the multisite CTM move.

    Implements the two-projector recipe of Corboz, Penc, Mila, Lauchli
    (PRB 84, 041108(R) (2011)) on the four enlarged corners returned by
    :func:`_build_enlarged_corner`.  For ``direction="left"`` we cut the
    LEFT column of the 2x2 plaquette: form the top row matrix
    ``top_M = Q_TL.Q_TR`` (contracting the top seam) and the bottom row
    matrix ``bot_M = Q_BR.Q_BL`` (contracting the bottom seam, with the
    reversed ordering so that the LEFT side of ``bot_M`` corresponds to
    Q_BL's top seam).  Fishman SVD on each row and a small SVD of
    ``bot_half @ top_half`` produce the cross-projector pair
    ``(P_top, P_bot)`` satisfying ``P_bot.P_top = I`` (closure) on the
    truncated chi_new subspace.

    Args:
        Q_TL: Top-left enlarged corner (chi_R, r2, chi_B, d2).
        Q_TR: Top-right enlarged corner (chi_L, l2, chi_B, d2).
        Q_BL: Bottom-left enlarged corner (chi_R, r2, chi_T, u2).
        Q_BR: Bottom-right enlarged corner (chi_L, l2, chi_T, u2).
        chi:  Target bond dimension of the new chi_new leg.
        direction: Only ``"left"`` is implemented.  Other directions are
            a Task 7 follow-up.

    Returns:
        Pair ``(P_top, P_bot)`` of rank-3 :class:`DenseTensor` projectors:

        - ``P_top`` axes ``("chi_outer", "fused_D2", "chi_new_top")``,
          dims ``(chi, D**2, chi_new)``, flows ``(IN, IN, OUT)``.
        - ``P_bot`` axes ``("chi_new_bot", "chi_outer", "fused_D2")``,
          dims ``(chi_new, chi, D**2)``, flows ``(IN, OUT, OUT)``.

        ``chi_outer`` and ``fused_D2`` share names so :func:`contract`
        auto-pairs them to give the closure tensor on the free legs
        ``(chi_new_top, chi_new_bot)``.

    Note:
        This implementation currently supports trivial-charge tensors only
        (``DenseTensor`` or ``SymmetricTensor`` whose every leg has a single
        sector ``[0]``).  The output charge bookkeeping hard-codes zeros, so
        non-trivial U(1) sectors would silently be discarded.  A runtime
        guard at function entry raises ``NotImplementedError`` when the
        inputs carry non-trivial charges; symmetric support is a follow-up.
        See ``docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md``.

    Raises:
        NotImplementedError: For ``direction in {"right", "top", "bottom"}``,
            or when any input tensor carries non-trivial charge sectors.
        ValueError: For unrecognized ``direction``.

    References:
        Fishman, White, Stoudenmire, PRB 98, 235148 (2018).
        Corboz, Penc, Mila, Lauchli, PRB 84, 041108(R) (2011).
        variPEPS (Naumann et al.) Fishman two-projector implementation.
    """
    if direction in ("right", "top", "bottom"):
        raise NotImplementedError(f"direction={direction!r} not implemented yet")
    if direction != "left":
        raise ValueError(f"unsupported direction={direction!r}")

    # Trivial-charge guard: this routine wraps the projector outputs with
    # hard-coded zero-charge TensorIndex objects on the chi_outer / fused_D2
    # / chi_new legs (see Step 6 below).  If any input tensor carries a
    # non-trivial U(1) sector structure, that bookkeeping would silently
    # discard the symmetry information, producing a wrong projector.  Until
    # symmetric support lands as a follow-up, refuse to run on non-trivial
    # inputs.
    for name, tensor in (
        ("Q_TL", Q_TL),
        ("Q_TR", Q_TR),
        ("Q_BL", Q_BL),
        ("Q_BR", Q_BR),
    ):
        for axis, idx in enumerate(tensor.indices):
            if idx.n_sectors != 1 or int(idx.sectors[0]) != 0:
                raise NotImplementedError(
                    "_compute_2x2_projector currently supports trivial-charge "
                    "tensors only; symmetric support is a follow-up. "
                    "See docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md."
                    f" (Got {name}.indices[{axis}] with sectors={idx.sectors.tolist()}.)"
                )

    # ---- Step 1: form top_M and bot_M (dense). ----
    # Q_TL labels: (chi_R, r2, chi_B, d2). Q_TR labels: (chi_L, l2, chi_B, d2).
    # Disambiguate the OUTER bottom seams of Q_TL/Q_TR before contracting.
    Q_TL_relab = Q_TL.relabels({"chi_B": "chi_B_TL", "d2": "d2_TL"})
    Q_TR_relab = Q_TR.relabels(
        {"chi_L": "chi_R", "l2": "r2", "chi_B": "chi_B_TR", "d2": "d2_TR"}
    )
    # Auto-pair (chi_R, r2). Free legs: (chi_B_TL, d2_TL, chi_B_TR, d2_TR).
    top_T = contract(Q_TL_relab, Q_TR_relab)
    # Order axes so rows = (chi_B_TL, d2_TL), cols = (chi_B_TR, d2_TR).
    top_order = ("chi_B_TL", "d2_TL", "chi_B_TR", "d2_TR")
    top_axes = tuple(top_T.labels().index(lbl) for lbl in top_order)
    top_T = top_T.transpose(top_axes)
    chi_TL, D2_TL, chi_TR, D2_TR = (idx.dim for idx in top_T.indices)
    top_M = jnp.asarray(top_T.todense()).reshape(chi_TL * D2_TL, chi_TR * D2_TR)

    # Q_BR · Q_BL (note reversed order). Q_BR labels: (chi_L, l2, chi_T, u2).
    # Q_BL labels: (chi_R, r2, chi_T, u2).
    # Contract Q_BR.chi_L <-> Q_BL.chi_R and Q_BR.l2 <-> Q_BL.r2.
    Q_BR_relab = Q_BR.relabels(
        {"chi_L": "chi_R", "l2": "r2", "chi_T": "chi_T_BR", "u2": "u2_BR"}
    )
    Q_BL_relab = Q_BL.relabels({"chi_T": "chi_T_BL", "u2": "u2_BL"})
    bot_T = contract(Q_BR_relab, Q_BL_relab)
    # Rows = RIGHT side (Q_BR top seam), cols = LEFT side (Q_BL top seam).
    bot_order = ("chi_T_BR", "u2_BR", "chi_T_BL", "u2_BL")
    bot_axes = tuple(bot_T.labels().index(lbl) for lbl in bot_order)
    bot_T = bot_T.transpose(bot_axes)
    chi_TR_b, D2_TR_b, chi_TL_b, D2_TL_b = (idx.dim for idx in bot_T.indices)
    bot_M = jnp.asarray(bot_T.todense()).reshape(chi_TR_b * D2_TR_b, chi_TL_b * D2_TL_b)

    # ---- Step 2: Fishman SVD on both row matrices. ----
    # Gauge-fix each SVD via _gauge_fixed_svd: rotates U/Vh columns so the
    # row of largest |U| is real-positive (variPEPS convention, preserves
    # reconstruction even for complex inputs).  This is critical for AD —
    # raw jnp.linalg.svd's gauge has tiny sign flips across iterations
    # which produce non-smooth gradients (mirrors the 1x1 path in
    # _ctm_projector.py, which uses _fix_svd_signs there).
    eps = 1e-12
    top_U, top_S, top_Vh = _gauge_fixed_svd(top_M)
    top_S = _fishman_truncate_S(top_S, eps)
    bot_U, bot_S, bot_Vh = _gauge_fixed_svd(bot_M)
    bot_S = _fishman_truncate_S(bot_S, eps)

    # ---- Step 3: form half-matrices (Fishman recipe). ----
    # top_half rows index the LEFT side of top_M (Q_TL bottom seam,
    # which is the chi seam of the LEFT column at the top).
    top_sqrtS = jnp.sqrt(top_S)
    bot_sqrtS = jnp.sqrt(bot_S)
    top_half = top_U * top_sqrtS[None, :]  # (chi_TL*D2_TL, kept_top)
    # bot_half cols index the LEFT side of bot_M (Q_BL top seam).
    bot_half = bot_sqrtS[:, None] * bot_Vh  # (kept_bot, chi_TL_b*D2_TL_b)

    # variPEPS-style normalization for stability.
    top_norm = jnp.linalg.norm(top_half) + 1e-30
    bot_norm = jnp.linalg.norm(bot_half) + 1e-30
    top_half = top_half / top_norm
    bot_half = bot_half / bot_norm

    # ---- Step 4: small SVD of M_prime = bot_half @ top_half. ----
    # Gauge-fix this SVD too so the truncated U_M / V_M_h are smooth
    # under AD (see Step 2 comment).
    M_prime = bot_half @ top_half  # (kept_bot, kept_top)
    U_M, S_M, V_M_h = _gauge_fixed_svd(M_prime)
    k = min(chi, S_M.shape[0])
    U_M = U_M[:, :k]
    S_M = S_M[:k]
    V_M_h = V_M_h[:k, :]

    # S^{-1/2} with safe guard against zeros (for AD-friendliness).
    s_max = S_M[0]
    cutoff = eps * (s_max + 1e-30)
    mask = S_M > cutoff
    S_safe = jnp.where(mask, S_M, 1.0)
    S_inv_sqrt = jnp.where(mask, 1.0 / jnp.sqrt(S_safe), 0.0)

    # ---- Step 5: form Fishman cross-projectors. ----
    # P_top = top_half @ V_M @ S_inv_sqrt  shape (chi_TL*D2_TL, k)
    V_M = V_M_h.conj().T  # (kept_top, k)
    P_top_dense = top_half @ V_M * S_inv_sqrt[None, :]
    # P_bot = S_inv_sqrt @ U_M^dagger @ bot_half  shape (k, chi_TL_b*D2_TL_b)
    P_bot_dense = (S_inv_sqrt[:, None] * U_M.conj().T) @ bot_half

    # ---- Step 6: reshape and wrap. ----
    chi_new = k
    sym = U1Symmetry()
    chi_charges = np.zeros(chi_TL, dtype=np.int32)
    D2_charges = np.zeros(D2_TL, dtype=np.int32)
    new_top_charges = np.zeros(chi_new, dtype=np.int32)
    new_bot_charges = np.zeros(chi_new, dtype=np.int32)

    # Shared legs (chi_outer, fused_D2): pair via opposite flows on the
    # two projectors so contract() succeeds for both Dense and Symmetric
    # paths.
    P_top_idx = (
        TensorIndex.from_charges(
            sym, chi_charges.copy(), FlowDirection.IN, label="chi_outer"
        ),
        TensorIndex.from_charges(
            sym, D2_charges.copy(), FlowDirection.IN, label="fused_D2"
        ),
        TensorIndex.from_charges(
            sym, new_top_charges.copy(), FlowDirection.OUT, label="chi_new_top"
        ),
    )
    P_bot_idx = (
        TensorIndex.from_charges(
            sym, new_bot_charges.copy(), FlowDirection.IN, label="chi_new_bot"
        ),
        TensorIndex.from_charges(
            sym, chi_charges.copy(), FlowDirection.OUT, label="chi_outer"
        ),
        TensorIndex.from_charges(
            sym, D2_charges.copy(), FlowDirection.OUT, label="fused_D2"
        ),
    )

    P_top_arr = P_top_dense.reshape(chi_TL, D2_TL, chi_new)
    P_bot_arr = P_bot_dense.reshape(chi_new, chi_TL_b, D2_TL_b)

    P_top = DenseTensor(P_top_arr, P_top_idx)
    P_bot = DenseTensor(P_bot_arr, P_bot_idx)
    return P_top, P_bot
