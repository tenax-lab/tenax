"""Raw-array CTM moves — einsum-only, no Tensor protocol.

Each CTM move operates on raw ``jnp.ndarray`` via explicit einsum
subscripts.  This produces a minimal XLA graph suitable for JIT tracing
in AD backward passes, avoiding the label-resolution and Tensor-wrapping
overhead that dominates compile time for the label-based moves.

The leg ordering conventions match ``_ctm_tensor_init.CTMTensorEnv``:

    Corners (2-leg, chi × chi):
      C1[c1_d, c1_r]   C2[c2_l, c2_d]   C3[c3_u, c3_l]   C4[c4_r, c4_u]

    Edges (3-leg, chi × D² × chi):
      T1[t1_l, u2, t1_r]   T2[t2_u, r2, t2_d]
      T3[t3_r, d2, t3_l]   T4[t4_d, l2, t4_u]

    Double-layer (4-leg, D² × D² × D² × D²):
      a[u2, d2, l2, r2]
"""

from __future__ import annotations

import jax.numpy as jnp

from tenax.algorithms.ad_utils import _fix_svd_signs, truncated_svd_ad

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _phase_fix_normalize_raw(arr: jnp.ndarray) -> jnp.ndarray:
    """Frobenius-normalize + phase-fix a raw array (matches variPEPS)."""
    norm = jnp.linalg.norm(arr)
    arr = arr / (norm + 1e-30)
    flat = arr.ravel()
    idx = jnp.argmax(jnp.abs(flat))
    phase = flat[idx] / (jnp.abs(flat[idx]) + 1e-30)
    return arr * jnp.conj(phase)


def _build_double_layer_raw(A: jnp.ndarray) -> jnp.ndarray:
    """Build 4-leg double-layer tensor from raw site tensor A[u,d,l,r,phys].

    Returns a[u2, d2, l2, r2] with D² legs, fused with ket-slow/bra-fast
    convention matching ``_build_double_layer_tensor``.
    """
    # Contract over physical index, arrange as (u,U,d,D,l,L,r,R)
    a = jnp.einsum("udlrp,UDLRp->uUdDlLrR", A, A.conj())
    D = A.shape[0]
    return a.reshape(D * D, D * D, D * D, D * D)


def _svd_projector_raw(
    C1g: jnp.ndarray,
    C4g: jnp.ndarray,
    chi: int,
    *,
    has_tracers: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute two-projector Fishman SVD projectors on raw arrays.

    Matches ``_compute_projector_tensor`` with ``projector_method="svd"``.

    Args:
        C1g: Grown corner, shape (fused_dim, col1).
        C4g: Grown corner, shape (fused_dim, col2).
        chi: Target bond dimension.
        has_tracers: Whether arrays contain JAX tracers (AD backward).

    Returns:
        (P1, P2) each of shape (fused_dim, k).
        P1 is applied to the C1g side, P2 to the C4g side.
    """
    # Cross-product: M = C1g^H @ C4g
    M = C1g.conj().T @ C4g

    if has_tracers:
        U_M, S_M, Vh_M = truncated_svd_ad(M, chi)
    else:
        U_full, S_full, Vh_full = jnp.linalg.svd(M, full_matrices=False)
        U_full, S_full, Vh_full = _fix_svd_signs(U_full, S_full, Vh_full)
        k = min(chi, S_full.shape[0])
        S_M = S_full[:k]
        U_M = U_full[:, :k]
        Vh_M = Vh_full[:k, :]

    V_M = Vh_M.conj().T  # (col2, k)

    # S^{-1/2} with safe division
    _cutoff = 1e-14 * (S_M[0] + 1e-30)
    _mask = S_M > _cutoff
    S_safe = jnp.where(_mask, S_M, 1.0)
    S_rsqrt = jnp.where(_mask, 1.0 / jnp.sqrt(S_safe), 0.0)

    # Two-projector Fishman (arXiv:2502.10298 Eq. 10)
    P1 = C4g @ (V_M * S_rsqrt[None, :])  # (fused_dim, k)
    P2 = C1g @ (U_M * S_rsqrt[None, :])  # (fused_dim, k)
    return P1, P2


def _apply_projector_raw(
    P1: jnp.ndarray,
    P2: jnp.ndarray,
    C1g: jnp.ndarray,
    C4g: jnp.ndarray,
    Tg: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply projector pair to grown corners and edge (raw arrays).

    Args:
        P1: Shape (fused_dim, k) — applied to C1g side.
        P2: Shape (fused_dim, k) — applied to C4g side.
        C1g: Shape (fused_dim, col1).
        C4g: Shape (fused_dim, col2).
        Tg:  Shape (fused_dim, fused_dim, D2) — grown edge.

    Returns:
        (C1_new, C4_new, T_new):
        C1_new: (k, col1), C4_new: (k, col2), T_new: (k, D2, k).
    """
    P1H = P1.conj().T  # (k, fused_dim)
    P2H = P2.conj().T

    C1_new = P1H @ C1g  # (k, col1)
    C4_new = P2H @ C4g  # (k, col2)

    # T_new = P1^H @ Tg @ P2: (k, D2, k)
    step = jnp.tensordot(P1.conj(), Tg, axes=([0], [0]))  # (k, fused_dim, D2)
    T_new = jnp.tensordot(step, P2, axes=([1], [0]))  # (k, D2, k)
    return C1_new, C4_new, T_new


# ------------------------------------------------------------------ #
# Directional moves                                                    #
# ------------------------------------------------------------------ #


def _compiled_move_left(
    env_self: tuple[jnp.ndarray, ...],
    env_nb: tuple[jnp.ndarray, ...],
    a: jnp.ndarray,
    chi: int,
    has_tracers: bool = False,
) -> tuple[jnp.ndarray, ...]:
    """Left CTM move on raw arrays.  Updates C1, T4, C4.

    Args:
        env_self:  (C1, C2, C3, C4, T1, T2, T3, T4) raw arrays.
        env_nb:    Same tuple, from neighbor site.
        a:         Double-layer tensor (D², D², D², D²).
        chi:       Target bond dimension.

    Returns:
        Updated env tuple (C1', C2, C3, C4', T1, T2, T3, T4').
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env_self
    _, _, _, _, T1_nb, _, T3_nb, _ = env_nb

    chi_dim = C1.shape[0]
    D2 = a.shape[0]

    # Corner growth: C1 · T1_nb
    # C1[c1_d, c1_r] · T1_nb[t1_l, u2, t1_r] (c1_r=t1_l)
    C1g = jnp.einsum("ij,jkl->ikl", C1, T1_nb)  # (chi, D2, chi)
    C1g = C1g.reshape(chi_dim * D2, -1)  # fuse (c1_d, u2) -> fused

    # Corner growth: C4 · T3_nb
    # C4[c4_r, c4_u] · T3_nb[t3_r, d2, t3_l] (c4_u=t3_r)
    C4g = jnp.einsum("ij,jkl->ikl", C4, T3_nb)  # (chi, D2, chi)
    C4g = C4g.reshape(chi_dim * D2, -1)  # fuse (c4_r, d2) -> fused

    # Edge growth: T4 · a
    # T4[t4_d, l2, t4_u] · a[u2, d2, l2, r2] (shared l2)
    T4_a = jnp.einsum("ijk,lmjn->iklmn", T4, a)  # (chi, chi, D2, D2, D2)
    # Fuse (t4_d=0, u2=2): transpose [0,2,1,3,4], reshape
    T4_a = T4_a.transpose(0, 2, 1, 3, 4)  # (chi, D2, chi, D2, D2)
    # Fuse (chi,D2) -> fl and (chi,D2) -> fr
    T4g = T4_a.reshape(chi_dim * D2, chi_dim * D2, D2)  # (fl, fr, r2)

    # Projector
    P1, P2 = _svd_projector_raw(C1g, C4g, chi, has_tracers=has_tracers)
    C1_new, C4_new, T4_new = _apply_projector_raw(P1, P2, C1g, C4g, T4g)

    # Phase fix + normalize
    C1_new = _phase_fix_normalize_raw(C1_new)
    C4_new = _phase_fix_normalize_raw(C4_new)
    T4_new = _phase_fix_normalize_raw(T4_new)

    return (C1_new, C2, C3, C4_new, T1, T2, T3, T4_new)


def _compiled_move_right(
    env_self: tuple[jnp.ndarray, ...],
    env_nb: tuple[jnp.ndarray, ...],
    a: jnp.ndarray,
    chi: int,
    has_tracers: bool = False,
) -> tuple[jnp.ndarray, ...]:
    """Right CTM move on raw arrays.  Updates C2, T2, C3.

    C2[c2_l, c2_d] · T1_nb[t1_l, u2, t1_r] (c2_l=t1_r)
    C3[c3_u, c3_l] · T3_nb[t3_r, d2, t3_l] (c3_l=t3_l) — wait, need to check.

    Actually from the Tensor code:
    C2.relabel("c2_l", "t1_r") then contract with T1_nb → C2_d shares nothing
    but "t1_r" is now relabeled. Let me re-derive.
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env_self
    _, _, _, _, T1_nb, _, T3_nb, _ = env_nb

    chi_dim = C2.shape[0]
    D2 = a.shape[0]

    # C2g: C2[c2_l, c2_d] · T1_nb[t1_l, u2, t1_r] (c2_l=t1_r → axis 0 of C2, axis 2 of T1)
    # C2[i,j] · T1_nb[k,l,i] -> C2g[j,k,l]
    C2g = jnp.einsum("ij,klj->ikl", C2, T1_nb)  # (c2_d, t1_l, u2)
    # Fuse (c2_d, u2) = axes (0, 2): transpose [0,2,1], reshape
    C2g = C2g.transpose(0, 2, 1)  # (c2_d, u2, t1_l)
    C2g = C2g.reshape(chi_dim * D2, -1)  # fuse -> (fused, t1_l)

    # C3g: C3[c3_u, c3_l] · T3_nb[t3_r, d2, t3_l] (c3_l=t3_l → axis 1 of C3, axis 2 of T3)
    # C3[i,j] · T3_nb[k,l,j] -> C3g[i,k,l]
    C3g = jnp.einsum("ij,klj->ikl", C3, T3_nb)  # (c3_u, t3_r, d2)
    # Fuse (c3_l→c3_u? No: fuse (c3_l, d2))
    # Wait, from the Tensor code:
    #   C3_u = env_self.C3.relabel("c3_u", "t3_l")
    #   C3g = contract(C3_u, env_neighbor.T3)  # (c3_l, t3_r, d2)
    #   C3g = _fuse_pair_by_label(C3g, "c3_l", "d2", "fused", IN)  # (fused, t3_r)
    # So C3_u relabels c3_u -> t3_l. C3 labels become (t3_l, c3_l).
    # Contract with T3(t3_r, d2, t3_l) on t3_l → result (c3_l, t3_r, d2)
    # Then fuse (c3_l=0, d2=2) -> (fused, t3_r)
    # In raw: C3[c3_u→t3_l, c3_l] axes=(0=t3_l,1=c3_l)
    # T3_nb[t3_r, d2, t3_l] axes=(0=t3_r, 1=d2, 2=t3_l)
    # Contract axis 0 of C3 with axis 2 of T3_nb
    # C3[i=c3_u=t3_l, j=c3_l], T3_nb[k=t3_r, l=d2, i=t3_l]
    # C3[i,j] · T3_nb[k,l,i] -> [j,k,l] = (c3_l, t3_r, d2)
    C3g = jnp.einsum("ij,kli->jkl", C3, T3_nb)  # (c3_l, t3_r, d2)
    # Fuse (c3_l=0, d2=2): transpose [0,2,1], reshape
    C3g = C3g.transpose(0, 2, 1)  # (c3_l, d2, t3_r)
    C3g = C3g.reshape(chi_dim * D2, -1)  # fuse -> (fused, t3_r)

    # T2g: T2[t2_u, r2, t2_d] · a[u2, d2, l2, r2] (shared r2)
    # T2 axis 1 = r2, a axis 3 = r2
    T2_a = jnp.einsum("ijk,lmnj->iklmn", T2, a)  # (t2_u, t2_d, u2, d2, l2)
    # Fuse (t2_u=0, u2=2): transpose [0,2,1,3,4], reshape
    T2_a = T2_a.transpose(0, 2, 1, 3, 4)  # (t2_u, u2, t2_d, d2, l2)
    # Fuse into (fl, fr, l2)
    T2g = T2_a.reshape(chi_dim * D2, chi_dim * D2, D2)

    # Projector
    P1, P2 = _svd_projector_raw(C2g, C3g, chi, has_tracers=has_tracers)
    C2_new, C3_new, T2_new = _apply_projector_raw(P1, P2, C2g, C3g, T2g)

    # Phase fix + normalize
    C2_new = _phase_fix_normalize_raw(C2_new)
    C3_new = _phase_fix_normalize_raw(C3_new)
    T2_new = _phase_fix_normalize_raw(T2_new)

    return (C1, C2_new, C3_new, C4, T1, T2_new, T3, T4)


def _compiled_move_top(
    env_self: tuple[jnp.ndarray, ...],
    env_nb: tuple[jnp.ndarray, ...],
    a: jnp.ndarray,
    chi: int,
    has_tracers: bool = False,
) -> tuple[jnp.ndarray, ...]:
    """Top CTM move on raw arrays.  Updates C1, T1, C2.

    From Tensor code:
      C1.relabel("c1_d","t4_d") -> C1[c1_r, t4_d], contract T4_nb[t4_d, l2, t4_u]
        -> C1g[c1_r, l2, t4_u], fuse (c1_r, l2) -> (fused, t4_u)
      C2.relabel("c2_d","t2_u") -> C2[c2_l, t2_u], contract T2_nb[t2_u, r2, t2_d]
        -> C2g[c2_l, r2, t2_d], fuse (c2_l, r2) -> (fused, t2_d)
      T1 · a: T1[t1_l, u2, t1_r] · a[u2, d2, l2, r2], shared u2
        -> T1_a[t1_l, t1_r, d2, l2, r2]
        fuse (t1_l, l2) -> fl, (t1_r, r2) -> fr -> T1g[fl, fr, d2]
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env_self
    _, _, _, _, _, T2_nb, _, T4_nb = env_nb

    chi_dim = C1.shape[0]
    D2 = a.shape[0]

    # C1g: C1[c1_d, c1_r], relabel c1_d -> t4_d
    # C1 axis 0 = c1_d = t4_d, T4_nb[t4_d, l2, t4_u]
    # C1[i,j] · T4_nb[i,k,l] -> [j,k,l] = (c1_r, l2, t4_u)
    C1g = jnp.einsum("ij,ikl->jkl", C1, T4_nb)  # (c1_r, l2, t4_u)
    # Fuse (c1_r=0, l2=1): already adjacent
    C1g = C1g.reshape(chi_dim * D2, -1)  # (fused, t4_u)

    # C2g: C2[c2_l, c2_d], relabel c2_d -> t2_u
    # C2 axis 1 = c2_d = t2_u, T2_nb[t2_u, r2, t2_d]
    # C2[i,j] · T2_nb[j,k,l] -> [i,k,l] = (c2_l, r2, t2_d)
    C2g = jnp.einsum("ij,jkl->ikl", C2, T2_nb)  # (c2_l, r2, t2_d)
    # Fuse (c2_l=0, r2=1): already adjacent
    C2g = C2g.reshape(chi_dim * D2, -1)  # (fused, t2_d)

    # T1g: T1[t1_l, u2, t1_r] · a[u2, d2, l2, r2], shared u2
    # T1 axis 1 = u2, a axis 0 = u2
    # T1[i,j,k] · a[j,l,m,n] -> [i,k,l,m,n] = (t1_l, t1_r, d2, l2, r2)
    T1_a = jnp.einsum("ijk,jlmn->iklmn", T1, a)  # (t1_l, t1_r, d2, l2, r2)
    # Fuse (t1_l=0, l2=3): transpose [0,3,1,4,2], reshape
    T1_a = T1_a.transpose(0, 3, 1, 4, 2)  # (t1_l, l2, t1_r, r2, d2)
    # Fuse (t1_l,l2) -> fl, (t1_r,r2) -> fr
    T1g = T1_a.reshape(chi_dim * D2, chi_dim * D2, D2)  # (fl, fr, d2)

    # Projector
    P1, P2 = _svd_projector_raw(C1g, C2g, chi, has_tracers=has_tracers)
    C1_new, C2_new, T1_new = _apply_projector_raw(P1, P2, C1g, C2g, T1g)

    # Phase fix + normalize
    C1_new = _phase_fix_normalize_raw(C1_new)
    C2_new = _phase_fix_normalize_raw(C2_new)
    T1_new = _phase_fix_normalize_raw(T1_new)

    return (C1_new, C2_new, C3, C4, T1_new, T2, T3, T4)


def _compiled_move_bottom(
    env_self: tuple[jnp.ndarray, ...],
    env_nb: tuple[jnp.ndarray, ...],
    a: jnp.ndarray,
    chi: int,
    has_tracers: bool = False,
) -> tuple[jnp.ndarray, ...]:
    """Bottom CTM move on raw arrays.  Updates C4, T3, C3.

    From Tensor code:
      C4.relabel("c4_r","t4_u") -> C4[t4_u, c4_u], contract T4_nb[t4_d, l2, t4_u]
        -> C4g[c4_u, t4_d, l2], fuse (c4_u, l2) -> (fused, t4_d)
      C3.relabel("c3_l","t2_d") -> C3[c3_u, t2_d], contract T2_nb[t2_u, r2, t2_d]
        -> C3g[c3_u, t2_u, r2], fuse (c3_u, r2) -> (fused, t2_u)
      T3 · a: T3[t3_r, d2, t3_l] · a[u2, d2, l2, r2], shared d2
        -> T3_a[t3_r, t3_l, u2, l2, r2]
        fuse (t3_r, l2) -> fl, (t3_l, r2) -> fr -> T3g[fl, fr, u2]
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env_self
    _, _, _, _, _, T2_nb, _, T4_nb = env_nb

    chi_dim = C4.shape[0]
    D2 = a.shape[0]

    # C4g: C4[c4_r, c4_u], relabel c4_r -> t4_u
    # C4 axis 0 = c4_r = t4_u, T4_nb[t4_d, l2, t4_u] axis 2 = t4_u
    # C4[i,j] · T4_nb[k,l,i] -> [j,k,l] = (c4_u, t4_d, l2)
    C4g = jnp.einsum("ij,kli->jkl", C4, T4_nb)  # (c4_u, t4_d, l2)
    # Fuse (c4_u=0, l2=2): transpose [0,2,1], reshape
    C4g = C4g.transpose(0, 2, 1)  # (c4_u, l2, t4_d)
    C4g = C4g.reshape(chi_dim * D2, -1)  # (fused, t4_d)

    # C3g: C3[c3_u, c3_l], relabel c3_l -> t2_d
    # C3 axis 1 = c3_l = t2_d, T2_nb[t2_u, r2, t2_d] axis 2 = t2_d
    # C3[i,j] · T2_nb[k,l,j] -> [i,k,l] = (c3_u, t2_u, r2)
    C3g = jnp.einsum("ij,klj->ikl", C3, T2_nb)  # (c3_u, t2_u, r2)
    # Fuse (c3_u=0, r2=2): transpose [0,2,1], reshape
    C3g = C3g.transpose(0, 2, 1)  # (c3_u, r2, t2_u)
    C3g = C3g.reshape(chi_dim * D2, -1)  # (fused, t2_u)

    # T3g: T3[t3_r, d2, t3_l] · a[u2, d2, l2, r2], shared d2
    # T3 axis 1 = d2, a axis 1 = d2
    # T3[i,j,k] · a[l,j,m,n] -> [i,k,l,m,n] = (t3_r, t3_l, u2, l2, r2)
    T3_a = jnp.einsum("ijk,ljmn->iklmn", T3, a)  # (t3_r, t3_l, u2, l2, r2)
    # Fuse (t3_r=0, l2=3): transpose [0,3,1,4,2], reshape
    T3_a = T3_a.transpose(0, 3, 1, 4, 2)  # (t3_r, l2, t3_l, r2, u2)
    T3g = T3_a.reshape(chi_dim * D2, chi_dim * D2, D2)  # (fl, fr, u2)

    # Projector
    P1, P2 = _svd_projector_raw(C4g, C3g, chi, has_tracers=has_tracers)
    C4_new, C3_new, T3_new = _apply_projector_raw(P1, P2, C4g, C3g, T3g)

    # Phase fix + normalize
    C4_new = _phase_fix_normalize_raw(C4_new)
    C3_new = _phase_fix_normalize_raw(C3_new)
    T3_new = _phase_fix_normalize_raw(T3_new)

    return (C1, C2, C3_new, C4_new, T1, T2, T3_new, T4)


# ------------------------------------------------------------------ #
# Direction dispatch                                                   #
# ------------------------------------------------------------------ #

_DIRECTION_MOVES_RAW = [
    ("left", _compiled_move_left),
    ("right", _compiled_move_right),
    ("top", _compiled_move_top),
    ("bottom", _compiled_move_bottom),
]


def _sort_coords_for_direction(coords, direction):
    """Sort coordinates for cascading order (matches _ctm_tensor_convergence)."""
    if direction == "left":
        return sorted(coords, key=lambda c: c[0])
    elif direction == "right":
        return sorted(coords, key=lambda c: -c[0])
    elif direction == "top":
        return sorted(coords, key=lambda c: c[1])
    elif direction == "bottom":
        return sorted(coords, key=lambda c: -c[1])
    return sorted(coords)


# ------------------------------------------------------------------ #
# Full sweep                                                           #
# ------------------------------------------------------------------ #

Coord = tuple[int, int]


def compiled_ctm_sweep_raw(
    site_arrays: dict[Coord, jnp.ndarray],
    env_arrays: dict[Coord, tuple[jnp.ndarray, ...]],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    has_tracers: bool = False,
) -> dict[Coord, tuple[jnp.ndarray, ...]]:
    """Full CTM sweep on raw arrays.

    Each env is a tuple of 8 arrays (C1, C2, C3, C4, T1, T2, T3, T4).
    Site arrays are raw site tensors A[u, d, l, r, phys].

    Args:
        site_arrays:  Map coord -> raw site tensor.
        env_arrays:   Map coord -> (C1, C2, ..., T4) raw arrays.
        neighbors:    Map coord -> {direction: neighbor_coord}.
        chi:          Target bond dimension.
        has_tracers:  Whether arrays contain JAX tracers (AD backward).

    Returns:
        Updated env_arrays dict.
    """
    # Build double layers
    double_layers = {c: _build_double_layer_raw(A) for c, A in site_arrays.items()}

    # Shallow copy
    envs = dict(env_arrays)
    all_coords = list(envs.keys())

    for direction, move_fn in _DIRECTION_MOVES_RAW:
        for coord in _sort_coords_for_direction(all_coords, direction):
            nb = neighbors[coord][direction]
            envs[coord] = move_fn(
                envs[coord],
                envs[nb],
                double_layers[nb],
                chi,
                has_tracers=has_tracers,
            )

    return envs


# ------------------------------------------------------------------ #
# Outer gauge fix (matches ad_utils._phase_fix_ctm_tensor)             #
# ------------------------------------------------------------------ #

_EPS_PHASE = 0.1  # threshold fraction for "large" element (variPEPS default)


def _frob_phase_fix_raw(arr: jnp.ndarray) -> jnp.ndarray:
    """Frobenius-normalize and fix phase of a dense array.

    Uses the threshold-based approach from ``_phase_fix_ctm_tensor``
    (first element with |x| >= 0.1 * max), NOT the simpler argmax(abs)
    used in per-absorption normalization.
    """
    norm = jnp.linalg.norm(arr)
    arr = arr / (norm + 1e-30)
    flat = arr.ravel()
    abs_flat = jnp.abs(flat)
    abs_max = jnp.max(abs_flat)
    threshold = _EPS_PHASE * abs_max
    mask = abs_flat >= threshold
    idx = jnp.argmax(mask)
    val = flat[idx]
    phase = val / (jnp.abs(val) + 1e-30)
    return arr * jnp.conj(phase)


def phase_fix_env_raw(
    env_tuple: tuple[jnp.ndarray, ...],
) -> tuple[jnp.ndarray, ...]:
    """Apply outer gauge fix to all 8 env arrays (matches _phase_fix_ctm_tensor)."""
    return tuple(_frob_phase_fix_raw(arr) for arr in env_tuple)


# ------------------------------------------------------------------ #
# Flat-leaves ↔ per-site dict conversion                               #
# ------------------------------------------------------------------ #


def leaves_to_env_arrays(
    env_leaves: tuple[jnp.ndarray, ...],
    coords: list[Coord],
) -> dict[Coord, tuple[jnp.ndarray, ...]]:
    """Convert flat env leaves (8 per coord) to per-site dict."""
    result = {}
    for i, c in enumerate(coords):
        result[c] = tuple(env_leaves[i * 8 : (i + 1) * 8])
    return result


def env_arrays_to_leaves(
    env_arrays: dict[Coord, tuple[jnp.ndarray, ...]],
    coords: list[Coord],
) -> tuple[jnp.ndarray, ...]:
    """Convert per-site dict to flat env leaves (8 per coord)."""
    leaves: list[jnp.ndarray] = []
    for c in coords:
        leaves.extend(env_arrays[c])
    return tuple(leaves)
