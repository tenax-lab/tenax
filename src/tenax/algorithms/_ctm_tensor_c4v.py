"""C4v-symmetric CTM with Tensor protocol.

Uses a single corner C and single edge T to exploit the full C4v
point-group symmetry of the square lattice iPEPS.  One sweep performs
a single "down" move instead of four directional moves, which avoids
the charge-sector mismatch problem that arises when independent
projectors produce chi legs with different block structures.
"""

from __future__ import annotations

__all__ = [
    "_c4v_sweep",
    "_c4v_to_full_env",
    "ctm_tensor_c4v",
]

import jax.numpy as jnp

from tenax.algorithms._ctm_projector import _compute_projector_tensor
from tenax.algorithms._ctm_tensor_convergence import _ctm_sv_diff
from tenax.algorithms._ctm_tensor_init import (
    IN,
    OUT,
    CTMTensorEnv,
    _build_double_layer_tensor,
    _fuse_pair_by_label,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_moves import _flip_leg_flow
from tenax.contraction.contractor import contract
from tenax.core.index import TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor


def _c4v_sweep(
    C: Tensor,
    T: Tensor,
    a: Tensor,
    chi: int,
    projector_method: str = "eigh",
) -> tuple[Tensor, Tensor]:
    """One C4v CTM sweep (single "down" move).

    Args:
        C: Corner tensor with labels ``(c_a=IN, c_b=OUT)``.
        T: Edge tensor with labels ``(t_l=IN, D2=IN, t_r=OUT)``.
        a: Double-layer tensor with labels ``(u2, d2, l2, r2)``.
        chi: Environment bond dimension.
        projector_method: ``"eigh"`` or ``"qr"``.

    Returns:
        ``(C_new, T_new)`` with the same label conventions.
    """
    # 1. Grow corner: Cg = C · T  (connect C.c_b to T.t_l)
    C_conn = C.relabel("c_b", "t_l")
    Cg = contract(C_conn, T)  # (c_a, D2, t_r)
    Cg = _fuse_pair_by_label(Cg, "c_a", "D2", "fused", IN)  # (fused, t_r)

    # 2. Grow edge: Tg = T · a  (connect T.D2 to a.u2)
    T_conn = T.relabel("D2", "u2")
    T_with_a = contract(T_conn, a)  # (t_l, t_r, d2, l2, r2)
    Tg = _fuse_pair_by_label(T_with_a, "t_l", "l2", "fl", IN)
    Tg = _fuse_pair_by_label(Tg, "t_r", "r2", "fr", OUT)  # (fl, d2, fr)

    # 3. Projector from Cg alone (C4v: both corners are equivalent)
    #    Use Cg for both corner slots — the density matrix is ρ = 2 * Cg · Cg†
    #    No base_charges: the C4v sweep uses a single corner, so there is no
    #    charge-sector drift between independent projectors.
    P, _ = _compute_projector_tensor(Cg, Cg, chi, projector_method)

    # 4. Apply projector to edge: T_new = P† · Tg · P (paired projector
    # contraction; see _ctm_tensor_moves note for why this stays .bar())
    P_bar = P.bar()
    P_left = P_bar.relabel("fused", "fl")
    step = contract(P_left, Tg)  # (chi_new, d2, fr)
    P_right = P.relabels({"fused": "fr", "chi_new": "chi_new_r"})
    T_new = contract(step, P_right)  # (chi_new, d2, chi_new_r)
    T_new = T_new.relabels({"chi_new": "t_l", "d2": "D2", "chi_new_r": "t_r"})

    # 5. New corner: C_new = (P† · Cg) · (P† · Cg)†
    #    Compute half = P† · Cg, then form C_new = half · half† in the
    #    dense domain.  The previous approach (contracting Cg with bar(Cg)
    #    as SymmetricTensor) introduced spurious Koszul signs for fermionic
    #    symmetries, making the density matrix non-PSD and preventing
    #    convergence.  The dense half · conj(half).T is PSD by construction.
    half = contract(P_bar, Cg)  # (chi_new, t_r)
    half_dense = half.todense()
    C_new_dense = half_dense @ jnp.conj(half_dense).T  # (chi_new, chi_new)

    # Build corner indices with the same charge distribution as the
    # projector's chi_new leg, ensuring compatibility with the edges.
    chi_new_idx = P.indices[P.labels().index("chi_new")]
    c_a_idx = TensorIndex.from_charges(
        chi_new_idx.symmetry, chi_new_idx.charges.copy(), IN, label="c_a"
    )
    c_b_idx = TensorIndex.from_charges(
        chi_new_idx.symmetry, chi_new_idx.charges.copy(), OUT, label="c_b"
    )
    if isinstance(Cg, SymmetricTensor):
        C_new = SymmetricTensor.from_dense(
            C_new_dense, (c_a_idx, c_b_idx), tol=float("inf")
        )
    else:
        C_new = DenseTensor(C_new_dense, (c_a_idx, c_b_idx))

    # 7. Normalize
    C_norm = C_new.max_abs()
    C_new = C_new * (1.0 / (C_norm + 1e-15))
    T_norm = T_new.max_abs()
    T_new = T_new * (1.0 / (T_norm + 1e-15))

    return C_new, T_new


def _c4v_to_full_env(C: Tensor, T: Tensor) -> CTMTensorEnv:
    """Expand one corner + one edge to the full 8-tensor CTMTensorEnv.

    C has labels ``(c_a=IN, c_b=OUT)``.
    T has labels ``(t_l=IN, D2=IN, t_r=OUT)``.

    The full environment label/flow conventions are:
        C1: (c1_d=IN, c1_r=OUT)
        C2: (c2_l=IN, c2_d=OUT)
        C3: (c3_u=OUT, c3_l=IN)
        C4: (c4_r=OUT, c4_u=IN)
        T1: (t1_l=IN, u2=IN, t1_r=OUT)
        T2: (t2_u=OUT, r2=OUT, t2_d=IN)
        T3: (t3_r=OUT, d2=OUT, t3_l=IN)
        T4: (t4_d=IN, l2=IN, t4_u=OUT)
    """
    # C1: c_a(IN)->c1_d(IN), c_b(OUT)->c1_r(OUT) — flows match
    C1 = C.relabels({"c_a": "c1_d", "c_b": "c1_r"})

    # C2: needs (c2_l=IN, c2_d=OUT). Map c_b(OUT)->c2_l, c_a(IN)->c2_d
    # But c_b has OUT and c2_l needs IN, c_a has IN and c2_d needs OUT → flip both
    C2 = C.relabels({"c_b": "c2_l", "c_a": "c2_d"})
    C2 = _flip_leg_flow(C2, "c2_l")
    C2 = _flip_leg_flow(C2, "c2_d")

    # C3: needs (c3_u=OUT, c3_l=IN). Map c_b(OUT)->c3_u, c_a(IN)->c3_l
    # c_b(OUT)->c3_u(OUT) OK, c_a(IN)->c3_l(IN) OK — but wait, the position
    # matters. Let's think about which legs connect where.
    # Actually for C3, c3_u connects to T2.t2_d and c3_l connects to T3.t3_l.
    # C3: c_b(OUT)->c3_u(OUT), c_a(IN)->c3_l(IN) — flows match!
    C3 = C.relabels({"c_b": "c3_u", "c_a": "c3_l"})

    # C4: needs (c4_r=OUT, c4_u=IN). Map c_a(IN)->c4_r, c_b(OUT)->c4_u
    # c_a(IN)->c4_r(OUT) MISMATCH, c_b(OUT)->c4_u(IN) MISMATCH → flip both
    C4 = C.relabels({"c_a": "c4_r", "c_b": "c4_u"})
    C4 = _flip_leg_flow(C4, "c4_r")
    C4 = _flip_leg_flow(C4, "c4_u")

    # T1: t_l(IN)->t1_l(IN), D2(IN)->u2(IN), t_r(OUT)->t1_r(OUT) — all match
    T1 = T.relabels({"t_l": "t1_l", "D2": "u2", "t_r": "t1_r"})

    # T2: needs (t2_u=OUT, r2=OUT, t2_d=IN)
    # Map t_l(IN)->t2_d(IN), D2(IN)->r2(OUT), t_r(OUT)->t2_u(OUT)
    # D2(IN)->r2(OUT) MISMATCH → flip r2
    T2 = T.relabels({"t_l": "t2_d", "D2": "r2", "t_r": "t2_u"})
    T2 = _flip_leg_flow(T2, "r2")

    # T3: needs (t3_r=OUT, d2=OUT, t3_l=IN)
    # Map t_l(IN)->t3_l(IN), D2(IN)->d2(OUT), t_r(OUT)->t3_r(OUT)
    # D2(IN)->d2(OUT) MISMATCH → flip d2
    T3 = T.relabels({"t_l": "t3_l", "D2": "d2", "t_r": "t3_r"})
    T3 = _flip_leg_flow(T3, "d2")

    # T4: needs (t4_d=IN, l2=IN, t4_u=OUT)
    # Map t_l(IN)->t4_d(IN), D2(IN)->l2(IN), t_r(OUT)->t4_u(OUT) — all match
    T4 = T.relabels({"t_l": "t4_d", "D2": "l2", "t_r": "t4_u"})

    return CTMTensorEnv(C1=C1, C2=C2, C3=C3, C4=C4, T1=T1, T2=T2, T3=T3, T4=T4)


def ctm_tensor_c4v(
    A: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-10,
    renormalize: bool = True,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
    """Run C4v-symmetric CTM to convergence.

    Uses a single corner and single edge, performing one move per sweep
    to maintain identical charge-sector structure across all environment
    tensors.

    Args:
        A:                 iPEPS site tensor with 5 legs ``(u, d, l, r, phys)``.
        chi:               Environment bond dimension.
        max_iter:          Maximum CTM iterations.
        conv_tol:          Convergence tolerance on corner singular values.
        renormalize:       (unused, kept for API compatibility).
        projector_method:  ``"eigh"`` or ``"qr"``.

    Returns:
        Converged CTMTensorEnv (full 8-tensor environment).
    """
    # Fermionic SymmetricTensors: densify before running C4v CTM.
    # The C4v expansion (_c4v_to_full_env) flips flows on C2/C4 corners,
    # which changes Koszul signs during RDM contraction and causes
    # cancellation to zero.  Densifying removes the fermionic grading
    # from the CTM loop; the energy computation still works correctly
    # because the open double-layer is numerically identical for
    # DenseTensor and SymmetricTensor.
    if isinstance(A, SymmetricTensor):
        from tenax.core.symmetry import BraidingStyle

        if A.indices[0].symmetry.braiding_style == BraidingStyle.FERMIONIC:
            A = DenseTensor(A.todense(), A.indices)

    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    # Extract C1 and T1 with generic C4v labels
    C = env.C1.relabels({"c1_d": "c_a", "c1_r": "c_b"})
    T = env.T1.relabels({"t1_l": "t_l", "u2": "D2", "t1_r": "t_r"})

    prev_sv = None
    for _ in range(max_iter):
        C, T = _c4v_sweep(C, T, a, chi, projector_method)

        current_sv = jnp.linalg.svd(C.todense(), compute_uv=False)
        if prev_sv is not None:
            diff = _ctm_sv_diff(current_sv, prev_sv)
            if float(diff) < conv_tol:
                break
        prev_sv = current_sv

    return _c4v_to_full_env(C, T)
