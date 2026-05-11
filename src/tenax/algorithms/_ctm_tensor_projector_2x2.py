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
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

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


def _gauge_fix_symmetric_svd(
    U_T: SymmetricTensor, Vh_T: SymmetricTensor
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Per-sector gauge fix for SymmetricTensor SVD outputs.

    Mirrors :func:`_gauge_fixed_svd` (the dense 2x2 gauge convention) at the
    block level: for each kept singular vector j, finds the entry of largest
    ``|U[:, j]|`` across all U-blocks that share its bond charge, rotates U's
    column and Vh's row by ``conj(phase)`` / ``phase`` so that
    ``U @ diag(s) @ Vh == M`` is preserved.  Critical for the 2x2 closure
    ``P_bot · P_top = I`` (no intervening matrix to absorb a ``conj(phase)**2``
    factor — see the docstring of :func:`_gauge_fixed_svd`).
    """
    bond_idx = U_T.indices[-1]  # last leg of U is the SVD bond
    bond_charges = np.asarray(bond_idx.charges, dtype=np.int32)

    # Per global column j, find its (charge, in-sector local index).
    # The sector ordering matches _truncated_svd_symmetric (bond charges
    # are listed in descending-SV global order); within a sector, the
    # local indices are 0..n_q-1 in the order they appear in bond_charges.
    local_index_of: dict[int, dict[int, int]] = {}  # q -> {global_j: local_idx}
    counter: dict[int, int] = {}
    for j, q in enumerate(bond_charges):
        q_int = int(q)
        local_index_of.setdefault(q_int, {})[j] = counter.get(q_int, 0)
        counter[q_int] = counter.get(q_int, 0) + 1

    # Collect U-blocks indexed by bond charge (last key entry).
    u_blocks_by_q: dict[int, list[tuple[tuple[int, ...], jax.Array]]] = {}
    for key, block in U_T.blocks.items():
        q = int(key[-1])
        u_blocks_by_q.setdefault(q, []).append((key, block))

    # Collect Vh-blocks indexed by bond charge (FIRST key entry).
    vh_blocks_by_q: dict[int, list[tuple[tuple[int, ...], jax.Array]]] = {}
    for key, block in Vh_T.blocks.items():
        q = int(key[0])
        vh_blocks_by_q.setdefault(q, []).append((key, block))

    new_u_blocks: dict[tuple[int, ...], jax.Array] = {
        key: block for key, block in U_T.blocks.items()
    }
    new_vh_blocks: dict[tuple[int, ...], jax.Array] = {
        key: block for key, block in Vh_T.blocks.items()
    }

    # For each global column j, compute its phase and write it back.
    for j, q in enumerate(bond_charges):
        q_int = int(q)
        local = local_index_of[q_int][j]
        u_entries = u_blocks_by_q.get(q_int, [])

        # Find max-abs entry across all U-blocks' local-column `local`.
        best_abs = -1.0
        best_value: complex | float = 1.0
        for _key, block in u_entries:
            # block shape: (left_dims..., n_q); take the slice block[..., local]
            col_slice = block[..., local]
            col_flat = jnp.reshape(col_slice, (-1,))
            local_max_idx = int(jnp.argmax(jnp.abs(col_flat)))
            local_max_val = complex(col_flat[local_max_idx])
            local_max_abs = abs(local_max_val)
            if local_max_abs > best_abs:
                best_abs = local_max_abs
                best_value = local_max_val

        if best_abs <= 0.0:
            phase = 1.0 + 0.0j
        else:
            phase = best_value / abs(best_value)

        # If the imaginary part is exactly zero (real inputs), keep the phase
        # as a real scalar so multiplying float64 blocks does not trigger the
        # JAX complex-to-real cast warning.
        if phase.imag == 0.0:
            conj_phase = jnp.asarray(phase.real)
            bare_phase = jnp.asarray(phase.real)
        else:
            conj_phase = jnp.asarray(complex(phase).conjugate())
            bare_phase = jnp.asarray(complex(phase))

        # Apply conj(phase) to column `local` of every matching U-block.
        for key, _block in u_entries:
            new_block = new_u_blocks[key]
            new_block = new_block.at[..., local].multiply(conj_phase)
            new_u_blocks[key] = new_block

        # Apply phase to row `local` of every matching Vh-block.
        for key, _block in vh_blocks_by_q.get(q_int, []):
            new_block = new_vh_blocks[key]
            new_block = new_block.at[local, ...].multiply(bare_phase)
            new_vh_blocks[key] = new_block

    U_out = SymmetricTensor._from_blocks_unchecked(new_u_blocks, U_T.indices)
    Vh_out = SymmetricTensor._from_blocks_unchecked(new_vh_blocks, Vh_T.indices)
    return U_out, Vh_out


def _scale_bond_by_diag(
    T: SymmetricTensor, diag: jax.Array, bond_label: str
) -> SymmetricTensor:
    """Multiply each block of ``T`` along its ``bond_label`` axis by ``diag``.

    The bond's TensorIndex charges encode each slot's sector; ``diag[j]`` is
    applied to slot ``j`` of every block whose bond key matches sector
    ``bond_charges[j]``.

    Used to express ``T @ diag(s)`` (or ``diag(s) @ T``) in the symmetric
    pipeline without densifying.
    """
    bond_axis = T.labels().index(bond_label)
    bond_idx = T.indices[bond_axis]
    bond_charges = np.asarray(bond_idx.charges, dtype=np.int32)

    new_blocks: dict[tuple[int, ...], jax.Array] = {}
    for key, block in T.blocks.items():
        q = int(key[bond_axis])
        positions = [j for j, cq in enumerate(bond_charges) if int(cq) == q]
        diag_slice = jnp.asarray(diag)[jnp.array(positions, dtype=np.int32)]
        shape = [1] * block.ndim
        shape[bond_axis] = len(positions)
        new_blocks[key] = block * diag_slice.reshape(shape)
    return SymmetricTensor._from_blocks_unchecked(new_blocks, T.indices)


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


# ---- Design note: ket/bra fuse convention vs variPEPS ----------------------
#
# Tenax uses a UNIFORM (ket, bra) ket-slow fused convention for the virtual
# D^2 legs of the enlarged corners (`d2 = ket*D + bra` everywhere), threaded
# through the precomputed double-layer `a` tensor.  variPEPS uses
# PER-QUADRANT axis ordering on its rank-6 quarter tensors -- e.g.
# `ctmrg_top_left` outputs (ket_d, bra_d, ..., ket_r, bra_r) while
# `ctmrg_top_right` outputs (..., bra_d, ket_d) -- so that variPEPS's
# M_prime row/column bases have *deliberately mismatched* ket-bra orders
# that align with the natural Z2 of the absorption maps.
#
# Tenax's M_prime is related to variPEPS's by a constant permutation.  Same
# spectrum, identical results outside degenerate subspaces.  But for
# PAIRED-DEGENERATE singular values the LAPACK SVD picks a basis within the
# 2D subspace that depends on the matrix entries; the permutation rotates
# that basis relative to variPEPS's, and the rotation is not a symmetry of
# the iteration map.  Compounds across iterations and breaks Z2 pairing
# from rank-D padded-identity init (visible at iter 2).
#
# This is documented in issue #425 (closed as known limitation).  The bug
# only manifests on Z2-symmetric inits; post-bug-3a Tenax's standard init
# is rank-1, which never triggers it.  Fixing it would require either
# per-quadrant `a` tensors, a rank-6 "lightly split" rewrite of standard
# CTM, or pivoting to split-CTM as canonical (#392 first).
# -----------------------------------------------------------------------------


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
        direction: One of ``"left"``, ``"right"``, ``"top"``, ``"bottom"``
            selecting which seam of the 2x2 plaquette is truncated:

            - ``"left"``  truncates the LEFT-column chi seam (T4's chi),
            - ``"right"`` truncates the RIGHT-column chi seam (T2's chi),
            - ``"top"``   truncates the TOP-row chi seam (T1's chi),
            - ``"bottom"`` truncates the BOTTOM-row chi seam (T3's chi).

    Returns:
        Pair ``(P_top, P_bot)`` of rank-3 :class:`DenseTensor` projectors:

        - ``P_top`` axes ``("chi_outer", "fused_D2", "chi_new_top")``,
          dims ``(chi, D**2, chi_new)``, flows ``(IN, IN, OUT)``.
        - ``P_bot`` axes ``("chi_new_bot", "chi_outer", "fused_D2")``,
          dims ``(chi_new, chi, D**2)``, flows ``(IN, OUT, OUT)``.

        ``chi_outer`` and ``fused_D2`` share names so :func:`contract`
        auto-pairs them to give the closure tensor on the free legs
        ``(chi_new_top, chi_new_bot)``.  For ``direction="top"``/``"bottom"``
        the same shared label names are reused (the two projectors still
        truncate to the same chi_new), even though the seam being cut is
        physically a vertical row rather than a horizontal column.  Callers
        of :func:`_compute_2x2_projector` are responsible for absorbing the
        projectors into the appropriate sublattice edges.

    Note:
        This implementation currently supports trivial-charge tensors only
        (``DenseTensor`` or ``SymmetricTensor`` whose every leg has a single
        sector ``[0]``).  The output charge bookkeeping hard-codes zeros, so
        non-trivial U(1) sectors would silently be discarded.  A runtime
        guard at function entry raises ``NotImplementedError`` when the
        inputs carry non-trivial charges; symmetric support is a follow-up.
        See ``docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md``.

    Raises:
        NotImplementedError: When any input tensor carries non-trivial
            charge sectors.
        ValueError: For unrecognized ``direction``.

    References:
        Fishman, White, Stoudenmire, PRB 98, 235148 (2018).
        Corboz, Penc, Mila, Lauchli, PRB 84, 041108(R) (2011).
        variPEPS (Naumann et al.) Fishman two-projector implementation.
    """
    if direction not in ("left", "right", "top", "bottom"):
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

    # ---- Step 1: form the two halves (M1, M2) for the chosen direction. ----
    # Each direction contracts one OUTER seam between two adjacent quarters,
    # producing a 2D matrix whose rows/cols index the two remaining outer
    # seams (chi+D^2 each).  The variPEPS analogues are:
    #   "left"  -> M1=top row (TL.TR), M2=bottom row (BR.BL)
    #   "right" -> M1=top row (TL.TR), M2=bottom row (BR.BL)  [different SVD halves]
    #   "top"   -> M1=left  col (BL.TL), M2=right col (TR.BR)
    #   "bottom"-> M1=left  col (BL.TL), M2=right col (TR.BR) [different SVD halves]
    if direction in ("left", "right"):
        # Top row product: Q_TL.right <-> Q_TR.left
        # rows of M1 = TL.bottom (chi_B_TL, d2_TL); cols = TR.bottom (chi_B_TR, d2_TR).
        Q_TL_relab = Q_TL.relabels({"chi_B": "chi_B_TL", "d2": "d2_TL"})
        Q_TR_relab = Q_TR.relabels(
            {"chi_L": "chi_R", "l2": "r2", "chi_B": "chi_B_TR", "d2": "d2_TR"}
        )
        top_T = contract(Q_TL_relab, Q_TR_relab)
        top_order = ("chi_B_TL", "d2_TL", "chi_B_TR", "d2_TR")
        top_axes = tuple(top_T.labels().index(lbl) for lbl in top_order)
        top_T = top_T.transpose(top_axes)
        chi_M1_row, D2_M1_row, chi_M1_col, D2_M1_col = (
            idx.dim for idx in top_T.indices
        )
        M1 = jnp.asarray(top_T.todense()).reshape(
            chi_M1_row * D2_M1_row, chi_M1_col * D2_M1_col
        )

        # Bottom row product: Q_BR.left <-> Q_BL.right (REVERSED ordering).
        # rows of M2 = BR.top (chi_T_BR, u2_BR); cols = BL.top (chi_T_BL, u2_BL).
        Q_BR_relab = Q_BR.relabels(
            {"chi_L": "chi_R", "l2": "r2", "chi_T": "chi_T_BR", "u2": "u2_BR"}
        )
        Q_BL_relab = Q_BL.relabels({"chi_T": "chi_T_BL", "u2": "u2_BL"})
        bot_T = contract(Q_BR_relab, Q_BL_relab)
        bot_order = ("chi_T_BR", "u2_BR", "chi_T_BL", "u2_BL")
        bot_axes = tuple(bot_T.labels().index(lbl) for lbl in bot_order)
        bot_T = bot_T.transpose(bot_axes)
        chi_M2_row, D2_M2_row, chi_M2_col, D2_M2_col = (
            idx.dim for idx in bot_T.indices
        )
        M2 = jnp.asarray(bot_T.todense()).reshape(
            chi_M2_row * D2_M2_row, chi_M2_col * D2_M2_col
        )
    else:
        # direction in ("top", "bottom"): vertical cut.
        # Left column product: Q_BL.top <-> Q_TL.bottom (analogue of variPEPS
        # left_matrix = bottom_left_matrix @ top_left_matrix).
        # rows of M1 = BL.right (chi_R_BL, r2_BL); cols = TL.right (chi_R_TL, r2_TL).
        Q_BL_relab = Q_BL.relabels(
            {"chi_T": "chi_B", "u2": "d2", "chi_R": "chi_R_BL", "r2": "r2_BL"}
        )
        Q_TL_relab = Q_TL.relabels({"chi_R": "chi_R_TL", "r2": "r2_TL"})
        left_T = contract(Q_BL_relab, Q_TL_relab)
        left_order = ("chi_R_BL", "r2_BL", "chi_R_TL", "r2_TL")
        left_axes = tuple(left_T.labels().index(lbl) for lbl in left_order)
        left_T = left_T.transpose(left_axes)
        chi_M1_row, D2_M1_row, chi_M1_col, D2_M1_col = (
            idx.dim for idx in left_T.indices
        )
        M1 = jnp.asarray(left_T.todense()).reshape(
            chi_M1_row * D2_M1_row, chi_M1_col * D2_M1_col
        )

        # Right column product: Q_TR.bottom <-> Q_BR.top (analogue of variPEPS
        # right_matrix = top_right_matrix @ bottom_right_matrix).
        # rows of M2 = TR.left (chi_L_TR, l2_TR); cols = BR.left (chi_L_BR, l2_BR).
        Q_TR_relab = Q_TR.relabels(
            {"chi_B": "chi_T", "d2": "u2", "chi_L": "chi_L_TR", "l2": "l2_TR"}
        )
        Q_BR_relab = Q_BR.relabels({"chi_L": "chi_L_BR", "l2": "l2_BR"})
        right_T = contract(Q_TR_relab, Q_BR_relab)
        right_order = ("chi_L_TR", "l2_TR", "chi_L_BR", "l2_BR")
        right_axes = tuple(right_T.labels().index(lbl) for lbl in right_order)
        right_T = right_T.transpose(right_axes)
        chi_M2_row, D2_M2_row, chi_M2_col, D2_M2_col = (
            idx.dim for idx in right_T.indices
        )
        M2 = jnp.asarray(right_T.todense()).reshape(
            chi_M2_row * D2_M2_row, chi_M2_col * D2_M2_col
        )

    # ---- Step 2: Fishman SVD on both halves. ----
    # Gauge-fix each SVD via _gauge_fixed_svd: rotates U/Vh columns so the
    # row of largest |U| is real-positive (variPEPS convention, preserves
    # reconstruction even for complex inputs).  This is critical for AD —
    # raw jnp.linalg.svd's gauge has tiny sign flips across iterations
    # which produce non-smooth gradients (mirrors the 1x1 path in
    # _ctm_projector.py, which uses _fix_svd_signs there).
    eps = 1e-12
    M1_U, M1_S, M1_Vh = _gauge_fixed_svd(M1)
    M1_S = _fishman_truncate_S(M1_S, eps)
    M2_U, M2_S, M2_Vh = _gauge_fixed_svd(M2)
    M2_S = _fishman_truncate_S(M2_S, eps)

    # ---- Step 3: pick which side of each Fishman SVD becomes the half. ----
    # Each direction selects one side of M1 and one side of M2 such that the
    # contracted axis of M_prime corresponds to the cut seam.
    #
    #   direction  | M1 half (rows / cols of M_prime)  | M2 half          | M_prime
    #   -----------+-----------------------------------+------------------+--------
    #   left       | M1_U sqrt(S)  rows = TL.bottom    | sqrt(S) M2_Vh    | M2 @ M1
    #              |   (LEFT-col-top side)             |   cols = BL.top  | (cuts LEFT col)
    #              |                                   |   (LEFT-col-bot) |
    #   right      | sqrt(S) M1_Vh cols = TR.bottom    | M2_U sqrt(S)     | M1 @ M2
    #              |   (RIGHT-col-top side)            |   rows = BR.top  | (cuts RIGHT col)
    #              |                                   |   (RIGHT-col-bot)|
    #   top        | sqrt(S) M1_Vh cols = TL.right     | M2_U sqrt(S)     | M1 @ M2
    #              |   (TOP-row-left side)             |   rows = TR.left | (cuts TOP row)
    #              |                                   |   (TOP-row-right)|
    #   bottom     | M1_U sqrt(S)  rows = BL.right     | sqrt(S) M2_Vh    | M2 @ M1
    #              |   (BOT-row-left side)             |   cols = BR.left | (cuts BOTTOM row)
    #              |                                   |   (BOT-row-right)|
    M1_sqrtS = jnp.sqrt(M1_S)
    M2_sqrtS = jnp.sqrt(M2_S)
    if direction in ("left", "bottom"):
        # M_prime = M2 @ M1: P_first from M1's row side, P_second from M2's col side.
        first_half = M1_U * M1_sqrtS[None, :]  # (rows_M1, k1)
        second_half = M2_sqrtS[:, None] * M2_Vh  # (k2, cols_M2)
        first_size = (chi_M1_row, D2_M1_row)
        second_size = (chi_M2_col, D2_M2_col)
        prime_order = "second_first"  # M_prime = second_half @ first_half
    else:
        # direction in ("right", "top"): M_prime = M1 @ M2.
        first_half = M1_sqrtS[:, None] * M1_Vh  # (k1, cols_M1)
        second_half = M2_U * M2_sqrtS[None, :]  # (rows_M2, k2)
        first_size = (chi_M1_col, D2_M1_col)
        second_size = (chi_M2_row, D2_M2_row)
        prime_order = "first_second"  # M_prime = first_half @ second_half

    # variPEPS-style normalization for stability.
    first_norm = jnp.linalg.norm(first_half) + 1e-30
    second_norm = jnp.linalg.norm(second_half) + 1e-30
    first_half = first_half / first_norm
    second_half = second_half / second_norm

    # ---- Step 4: small SVD of M_prime. ----
    # Gauge-fix this SVD too so the truncated U_M / V_M_h are smooth
    # under AD (see Step 2 comment).
    if prime_order == "second_first":
        # M_prime = second @ first. SVD: M_prime = U_M S_M V_M_h.
        # P_first  ~ first  @ V_M S^-1/2  (row side -> chi_new)
        # P_second ~ U_M^H S^-1/2 @ second (chi_new -> col side)
        M_prime = second_half @ first_half
    else:
        # M_prime = first @ second. SVD: M_prime = U_M S_M V_M_h.
        # P_first  ~ U_M^H S^-1/2 @ first (chi_new -> col side)
        # P_second ~ second @ V_M S^-1/2 (row side -> chi_new)
        M_prime = first_half @ second_half

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
    if prime_order == "second_first":
        # P_first ~ first @ V_M S^-1/2 with shape (rows_M1_size, k);
        # P_second ~ S^-1/2 U_M^H @ second with shape (k, cols_M2_size).
        V_M = V_M_h.conj().T  # (kept, k)
        P_first_dense = first_half @ V_M * S_inv_sqrt[None, :]
        P_second_dense = (S_inv_sqrt[:, None] * U_M.conj().T) @ second_half
    else:
        # P_first ~ S^-1/2 U_M^H @ first with shape (k, cols_M1_size);
        # P_second ~ second @ V_M S^-1/2 with shape (rows_M2_size, k).
        V_M = V_M_h.conj().T
        P_first_dense = (S_inv_sqrt[:, None] * U_M.conj().T) @ first_half
        P_second_dense = second_half @ V_M * S_inv_sqrt[None, :]

    # ---- Step 6: reshape and wrap. ----
    # Per-direction packing: each of (P_first, P_second) corresponds to one
    # of the two halves of the cut seam.  The labels chi_outer / fused_D2 are
    # the SAME on both projectors so contract() auto-pairs them in the closure
    # check.  P_top wraps with leading (chi_outer, fused_D2) and trailing
    # chi_new_top; P_bot wraps with leading chi_new_bot and trailing
    # (chi_outer, fused_D2).  Naming is "top"/"bot" regardless of direction
    # for label uniformity (the closure test only checks ranks + identity).
    chi_new = k
    if prime_order == "second_first":
        # P_first has shape (rows_size, k) -- wrap as P_top (legs first).
        # P_second has shape (k, cols_size) -- wrap as P_bot (k first).
        P_top_dense = P_first_dense
        P_bot_dense = P_second_dense
        chi_top, D2_top = first_size
        chi_bot, D2_bot = second_size
    else:
        # P_first has shape (k, cols_size); P_second has shape (rows_size, k).
        # We want P_top to lead with (chi_outer, fused_D2) so wrap P_second
        # there.  P_bot leads with k so wrap P_first there.
        P_top_dense = P_second_dense  # (rows_size, k)
        P_bot_dense = P_first_dense  # (k, cols_size)
        chi_top, D2_top = second_size
        chi_bot, D2_bot = first_size

    sym = U1Symmetry()
    chi_top_charges = np.zeros(chi_top, dtype=np.int32)
    D2_top_charges = np.zeros(D2_top, dtype=np.int32)
    chi_bot_charges = np.zeros(chi_bot, dtype=np.int32)
    D2_bot_charges = np.zeros(D2_bot, dtype=np.int32)
    new_top_charges = np.zeros(chi_new, dtype=np.int32)
    new_bot_charges = np.zeros(chi_new, dtype=np.int32)

    # Shared legs (chi_outer, fused_D2): pair via opposite flows on the
    # two projectors so contract() succeeds for both Dense and Symmetric
    # paths.
    P_top_idx = (
        TensorIndex.from_charges(
            sym, chi_top_charges, FlowDirection.IN, label="chi_outer"
        ),
        TensorIndex.from_charges(
            sym, D2_top_charges, FlowDirection.IN, label="fused_D2"
        ),
        TensorIndex.from_charges(
            sym, new_top_charges, FlowDirection.OUT, label="chi_new_top"
        ),
    )
    P_bot_idx = (
        TensorIndex.from_charges(
            sym, new_bot_charges, FlowDirection.IN, label="chi_new_bot"
        ),
        TensorIndex.from_charges(
            sym, chi_bot_charges, FlowDirection.OUT, label="chi_outer"
        ),
        TensorIndex.from_charges(
            sym, D2_bot_charges, FlowDirection.OUT, label="fused_D2"
        ),
    )

    P_top_arr = P_top_dense.reshape(chi_top, D2_top, chi_new)
    P_bot_arr = P_bot_dense.reshape(chi_new, chi_bot, D2_bot)

    P_top = DenseTensor(P_top_arr, P_top_idx)
    P_bot = DenseTensor(P_bot_arr, P_bot_idx)
    return P_top, P_bot


def _compute_2x2_projector_symmetric(
    Q_TL: SymmetricTensor,
    Q_TR: SymmetricTensor,
    Q_BL: SymmetricTensor,
    Q_BR: SymmetricTensor,
    chi: int,
    *,
    direction: str,
    base_charges: np.ndarray | None = None,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Block-sparse 2x2 Fishman projector for SymmetricTensor inputs.

    Mirrors the dense pipeline in :func:`_compute_2x2_projector` stage-for-stage
    via :func:`tenax.linalg.svd` and the per-sector gauge-fix helper
    :func:`_gauge_fix_symmetric_svd`.

    See ``docs/superpowers/specs/2026-05-11-2x2-projector-symmetric-design.md``
    for the cut-seam relabel rationale and flow conventions.

    Args:
        Q_TL, Q_TR, Q_BL, Q_BR: 4-leg enlarged corners.  Must all be SymmetricTensor.
        chi: Target bond dimension of the new chi_new leg.
        direction: One of ``"left"``, ``"right"``, ``"top"``, ``"bottom"``.
        base_charges: Optional 1-D ``np.ndarray`` of charges. When supplied,
            chi_new is allocated per sector (added in Task 5; ignored in Task 2).

    Returns:
        ``(P_top, P_bot)`` SymmetricTensor projectors with the same label /
        flow conventions as the dense path's output.
    """
    from tenax.contraction.contractor import contract
    from tenax.linalg import svd as tensor_svd

    if direction not in ("left", "right", "top", "bottom"):
        raise ValueError(f"unsupported direction={direction!r}")

    # ---- Stage 1: form M1, M2 as 4-leg SymmetricTensors. ----
    if direction in ("left", "right"):
        Q_TL_relab = Q_TL.relabels({"chi_B": "chi_B_TL", "d2": "d2_TL"})
        Q_TR_relab = Q_TR.relabels(
            {"chi_L": "chi_R", "l2": "r2", "chi_B": "chi_B_TR", "d2": "d2_TR"}
        )
        M1_T = contract(Q_TL_relab, Q_TR_relab)
        m1_left_labels = ("chi_B_TL", "d2_TL")
        m1_right_labels = ("chi_B_TR", "d2_TR")

        Q_BR_relab = Q_BR.relabels(
            {"chi_L": "chi_R", "l2": "r2", "chi_T": "chi_T_BR", "u2": "u2_BR"}
        )
        Q_BL_relab = Q_BL.relabels({"chi_T": "chi_T_BL", "u2": "u2_BL"})
        M2_T = contract(Q_BR_relab, Q_BL_relab)
        m2_left_labels = ("chi_T_BR", "u2_BR")
        m2_right_labels = ("chi_T_BL", "u2_BL")
    else:  # "top", "bottom"
        Q_BL_relab = Q_BL.relabels(
            {"chi_T": "chi_B", "u2": "d2", "chi_R": "chi_R_BL", "r2": "r2_BL"}
        )
        Q_TL_relab = Q_TL.relabels({"chi_R": "chi_R_TL", "r2": "r2_TL"})
        M1_T = contract(Q_BL_relab, Q_TL_relab)
        m1_left_labels = ("chi_R_BL", "r2_BL")
        m1_right_labels = ("chi_R_TL", "r2_TL")

        Q_TR_relab = Q_TR.relabels(
            {"chi_B": "chi_T", "d2": "u2", "chi_L": "chi_L_TR", "l2": "l2_TR"}
        )
        Q_BR_relab = Q_BR.relabels({"chi_L": "chi_L_BR", "l2": "l2_BR"})
        M2_T = contract(Q_TR_relab, Q_BR_relab)
        m2_left_labels = ("chi_L_TR", "l2_TR")
        m2_right_labels = ("chi_L_BR", "l2_BR")

    # ---- Stage 2: SVDs of M1, M2 with per-sector gauge fix. ----
    U_M1_T, M1_S, Vh_M1_T, _ = tensor_svd(
        M1_T,
        left_labels=m1_left_labels,
        right_labels=m1_right_labels,
        new_bond_label="m1_bond",
        max_singular_values=None,
    )
    U_M1_T, Vh_M1_T = _gauge_fix_symmetric_svd(U_M1_T, Vh_M1_T)
    M1_S = _fishman_truncate_S(M1_S, eps=1e-12)

    U_M2_T, M2_S, Vh_M2_T, _ = tensor_svd(
        M2_T,
        left_labels=m2_left_labels,
        right_labels=m2_right_labels,
        new_bond_label="m2_bond",
        max_singular_values=None,
    )
    U_M2_T, Vh_M2_T = _gauge_fix_symmetric_svd(U_M2_T, Vh_M2_T)
    M2_S = _fishman_truncate_S(M2_S, eps=1e-12)

    M1_sqrtS = jnp.sqrt(M1_S)
    M2_sqrtS = jnp.sqrt(M2_S)

    # ---- Stage 3: form halves, normalize. ----
    if direction in ("left", "bottom"):
        first_half = _scale_bond_by_diag(U_M1_T, M1_sqrtS, bond_label="m1_bond")
        second_half = _scale_bond_by_diag(Vh_M2_T, M2_sqrtS, bond_label="m2_bond")
        prime_order = "second_first"
        first_outer_labels = m1_left_labels
        second_outer_labels = m2_right_labels
    else:  # "right", "top"
        first_half = _scale_bond_by_diag(Vh_M1_T, M1_sqrtS, bond_label="m1_bond")
        second_half = _scale_bond_by_diag(U_M2_T, M2_sqrtS, bond_label="m2_bond")
        prime_order = "first_second"
        first_outer_labels = m1_right_labels
        second_outer_labels = m2_left_labels

    first_norm = jnp.sqrt(jnp.sum(M1_S) + 1e-30)
    second_norm = jnp.sqrt(jnp.sum(M2_S) + 1e-30)
    first_half = _scale_bond_by_diag(
        first_half, jnp.ones_like(M1_S) / first_norm, bond_label="m1_bond"
    )
    second_half = _scale_bond_by_diag(
        second_half, jnp.ones_like(M2_S) / second_norm, bond_label="m2_bond"
    )

    # ---- Stage 4: form M_prime by renaming cut seam + contract; SVD M_prime. ----
    if prime_order == "second_first":
        cut_relabel = dict(zip(m1_left_labels, m2_right_labels))
        first_half_for_mp = first_half.relabels(cut_relabel)
        M_prime_T = contract(second_half, first_half_for_mp)
        mp_left_labels = ("m2_bond",)
        mp_right_labels = ("m1_bond",)
    else:
        cut_relabel = dict(zip(m1_right_labels, m2_left_labels))
        first_half_for_mp = first_half.relabels(cut_relabel)
        M_prime_T = contract(first_half_for_mp, second_half)
        mp_left_labels = ("m1_bond",)
        mp_right_labels = ("m2_bond",)

    U_Mp_T, S_Mp, Vh_Mp_T, _ = tensor_svd(
        M_prime_T,
        left_labels=mp_left_labels,
        right_labels=mp_right_labels,
        new_bond_label="chi_new",
        max_singular_values=chi,
    )
    U_Mp_T, Vh_Mp_T = _gauge_fix_symmetric_svd(U_Mp_T, Vh_Mp_T)

    # ---- Stage 5: cross-projectors via bar() for the SVD adjoint. ----
    # tensor_svd returns S_Mp in global-descending order, so S_Mp[0] is the max
    # (mirrors the dense path at _compute_2x2_projector). Keeping this as a
    # traced JAX scalar avoids TracerArrayConversionError under jax.jit.
    s_max = S_Mp[0]
    cutoff = 1e-12 * (s_max + 1e-30)
    mask = S_Mp > cutoff
    S_safe = jnp.where(mask, S_Mp, 1.0)
    S_inv_sqrt = jnp.where(mask, 1.0 / jnp.sqrt(S_safe), 0.0)

    if prime_order == "second_first":
        # P_first  = first_half · V_Mp · S^{-1/2}   = contract(first_half, Vh_Mp.bar())
        # P_second = S^{-1/2} · U_Mp^† · second_half = contract(U_Mp.bar(), second_half)
        P_first_unscaled = contract(first_half, Vh_Mp_T.bar())
        P_second_unscaled = contract(U_Mp_T.bar(), second_half)
    else:  # "first_second"
        P_first_unscaled = contract(U_Mp_T.bar(), first_half)
        P_second_unscaled = contract(second_half, Vh_Mp_T.bar())

    P_first = _scale_bond_by_diag(P_first_unscaled, S_inv_sqrt, bond_label="chi_new")
    P_second = _scale_bond_by_diag(P_second_unscaled, S_inv_sqrt, bond_label="chi_new")

    # ---- Stage 6: relabel and reorder axes to match the dense path's output. ----
    if prime_order == "second_first":
        P_top_unwrapped, top_outer = P_first, first_outer_labels
        P_bot_unwrapped, bot_outer = P_second, second_outer_labels
    else:
        P_top_unwrapped, top_outer = P_second, second_outer_labels
        P_bot_unwrapped, bot_outer = P_first, first_outer_labels

    chi_lbl_top, D2_lbl_top = top_outer
    chi_lbl_bot, D2_lbl_bot = bot_outer

    P_top = P_top_unwrapped.relabels(
        {
            chi_lbl_top: "chi_outer",
            D2_lbl_top: "fused_D2",
            "chi_new": "chi_new_top",
        }
    )
    P_bot = P_bot_unwrapped.relabels(
        {
            chi_lbl_bot: "chi_outer",
            D2_lbl_bot: "fused_D2",
            "chi_new": "chi_new_bot",
        }
    )

    P_top = P_top.transpose(
        tuple(
            P_top.labels().index(lbl)
            for lbl in ("chi_outer", "fused_D2", "chi_new_top")
        )
    )
    P_bot = P_bot.transpose(
        tuple(
            P_bot.labels().index(lbl)
            for lbl in ("chi_new_bot", "chi_outer", "fused_D2")
        )
    )
    return P_top, P_bot
