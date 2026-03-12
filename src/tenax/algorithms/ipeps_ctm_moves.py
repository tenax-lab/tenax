"""CTM algorithm for iPEPS — projector dispatch and directional moves."""

from __future__ import annotations

__all__ = [
    "_ctm_move_eigh",
    "_ctm_move_qr",
    "_ctm_move",
    "_ctm_left_move",
    "_ctm_right_move",
    "_ctm_top_move",
    "_ctm_bottom_move",
    "_ctm_left_move_2site",
    "_ctm_right_move_2site",
    "_ctm_top_move_2site",
    "_ctm_bottom_move_2site",
    "_split_ctm_projector",
    "_svd_split_edge",
    "_split_ctm_move",
]

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import (
    CTMEnvironment,
    SplitCTMEnvironment,
)
from tenax.algorithms.ipeps_ctm_init import _build_double_layer


def _ctm_move_eigh(
    C1g: jax.Array,
    C2g: jax.Array,
    Tg: jax.Array,
    chi: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Projector-based CTM truncation using eigh (standard method).

    Given two grown corners ``C1g`` and ``C2g`` (each a 2-D matrix whose row
    dimension is ``chi * D2``) and a grown edge tensor ``Tg`` of shape
    ``(chi*D2, D2, chi*D2)``, compute a single isometric projector from the
    combined half-system density matrix (Corboz et al., PRB 90, 165127, 2014),
    then truncate both corners and the edge to bond dimension ``chi``.

    The projector ``P`` is obtained from the eigendecomposition of the
    half-system density matrix ``rho = C1g @ C1g^H + C2g @ C2g^H``.
    Using ``eigh`` (symmetric eigendecomposition) is more numerically
    stable than SVD of the concatenated corners when ``chi * D2 == 2 * chi``
    (square matrix case), avoiding spurious sign oscillations in the
    projector that prevent convergence.

    Returns ``(C1_new, C2_new, T_new)`` with shapes ``(chi', col1)``,
    ``(chi', col2)``, ``(chi', D2, chi')`` where ``chi' <= chi``.
    """
    # Half-system density matrix (Corboz et al. 2014).
    # rho = C1g @ C1g^T + C2g @ C2g^T is positive semi-definite.
    # Its leading eigenvectors form the optimal isometric projector.
    rho = C1g @ C1g.conj().T + C2g @ C2g.conj().T
    rho = 0.5 * (rho + rho.conj().T)  # enforce exact Hermiticity

    eigvals, eigvecs = jnp.linalg.eigh(rho)
    # eigh returns eigenvalues in ascending order; take the top chi.
    k = min(chi, len(eigvals))
    P = eigvecs[:, -k:][:, ::-1]  # (n, chi'), largest first

    # Project corners — stop gradient through P for AD stability.
    # The implicit fixed-point differentiation (ctm_converge) handles
    # the overall response; differentiating through the projector
    # eigenvectors causes gradient blowup from degenerate eigenvalues.
    P_sg = jax.lax.stop_gradient(P)
    C1_new = P_sg.conj().T @ C1g  # (chi', col1)
    C2_new = P_sg.conj().T @ C2g  # (chi', col2)
    T_new = jnp.einsum("ia,idj,jb->adb", P_sg, Tg, P_sg)
    return C1_new, C2_new, T_new


def _ctm_move_qr(
    C1g: jax.Array,
    C2g: jax.Array,
    Tg: jax.Array,
    chi: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Projector-based CTM truncation using QR + small eigh.

    Instead of forming the full density matrix rho of size (chi*D^2)^2 and
    diagonalizing at O((chi*D^2)^3), QR-factor the concatenated corners
    [C1g, C2g] to reduce to a small (2*chi, 2*chi) problem, then eigh
    the small matrix (arXiv:2505.00494).

    Total cost: O(chi^2 * D^2 * chi) for QR + O(chi^3) for small eigh,
    vs O(chi^3 * D^6) for direct eigh of rho.

    Returns ``(C1_new, C2_new, T_new)`` with shapes ``(chi', col1)``,
    ``(chi', col2)``, ``(chi', D2, chi')`` where ``chi' <= chi``.
    """
    M = jnp.concatenate([C1g, C2g], axis=1)  # (n, 2*m)
    Q, R = jnp.linalg.qr(M)  # Q: (n, 2m), R: (2m, 2m)

    # Small eigh on R @ R^H (size 2*chi x 2*chi)
    rho_small = R @ R.conj().T
    rho_small = 0.5 * (rho_small + rho_small.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(rho_small)

    k = min(chi, len(eigvals))
    V = eigvecs[:, -k:][:, ::-1]  # leading eigenvectors
    P = Q @ V  # (n, k)

    P_sg = jax.lax.stop_gradient(P)
    C1_new = P_sg.conj().T @ C1g
    C2_new = P_sg.conj().T @ C2g
    T_new = jnp.einsum("ia,idj,jb->adb", P_sg, Tg, P_sg)
    return C1_new, C2_new, T_new


def _ctm_move(
    C1g: jax.Array,
    C2g: jax.Array,
    Tg: jax.Array,
    chi: int,
    projector_method: str = "eigh",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Dispatch CTM truncation to eigh or QR projector method."""
    if projector_method == "qr":
        return _ctm_move_qr(C1g, C2g, Tg, chi)
    return _ctm_move_eigh(C1g, C2g, Tg, chi)


def _ctm_left_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    projector_method: str = "eigh",
) -> CTMEnvironment:
    """Projector-based CTM left move: updates C1, T4, C4.

    Grows C1 with T1, C4 with T3, T4 with ``a``, then truncates with
    consistent projectors derived from the grown corners.
    """
    D2 = a.shape[0]
    # Grow corners
    C1g = jnp.einsum("ab,buc->auc", env.C1, env.T1).reshape(-1, env.T1.shape[2])
    C4g = jnp.einsum("gh,hdi->gdi", env.C4, env.T3).reshape(-1, env.T3.shape[2])
    # Grow edge: T4[a,l,g] * a[u,d,l,r] -> (a,u,g,d,r)
    T4g = jnp.einsum("alg,udlr->augdr", env.T4, a)
    T4g = T4g.transpose(0, 1, 4, 2, 3).reshape(C1g.shape[0], D2, C4g.shape[0])

    C1_new, C4_new, T4_new = _ctm_move(C1g, C4g, T4g, chi, projector_method)
    return CTMEnvironment(
        C1=C1_new,
        C2=env.C2,
        C3=env.C3,
        C4=C4_new,
        T1=env.T1,
        T2=env.T2,
        T3=env.T3,
        T4=T4_new,
    )


def _ctm_right_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    projector_method: str = "eigh",
) -> CTMEnvironment:
    """Projector-based CTM right move: updates C2, T2, C3."""
    D2 = a.shape[0]
    # Grow corners
    C2g = jnp.einsum("ce,buc->eub", env.C2, env.T1).reshape(-1, env.T1.shape[0])
    C3g = jnp.einsum("im,hdi->mdh", env.C3, env.T3).reshape(-1, env.T3.shape[0])
    # Grow edge: T2[e,r,m] * a[u,d,l,r] -> (e,u,m,d,l)
    T2g = jnp.einsum("erm,udlr->eumdl", env.T2, a)
    T2g = T2g.transpose(0, 1, 4, 2, 3).reshape(C2g.shape[0], D2, C3g.shape[0])

    C2_new, C3_new, T2_new = _ctm_move(C2g, C3g, T2g, chi, projector_method)
    return CTMEnvironment(
        C1=env.C1,
        C2=C2_new,
        C3=C3_new,
        C4=env.C4,
        T1=env.T1,
        T2=T2_new,
        T3=env.T3,
        T4=env.T4,
    )


def _ctm_top_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    projector_method: str = "eigh",
) -> CTMEnvironment:
    """Projector-based CTM top move: updates C1, T1, C2."""
    D2 = a.shape[0]
    # Grow corners
    C1g = jnp.einsum("ab,alg->blg", env.C1, env.T4).reshape(-1, env.T4.shape[2])
    C2g = jnp.einsum("ce,erm->crm", env.C2, env.T2).reshape(-1, env.T2.shape[2])
    # Grow edge: T1[b,u,c] * a[u,d,l,r] -> (b,c,d,l,r)
    T1g = jnp.einsum("buc,udlr->bcdlr", env.T1, a)
    T1g = T1g.transpose(0, 3, 2, 1, 4).reshape(C1g.shape[0], D2, C2g.shape[0])

    C1_new, C2_new, T1_new = _ctm_move(C1g, C2g, T1g, chi, projector_method)
    return CTMEnvironment(
        C1=C1_new,
        C2=C2_new,
        C3=env.C3,
        C4=env.C4,
        T1=T1_new,
        T2=env.T2,
        T3=env.T3,
        T4=env.T4,
    )


def _ctm_bottom_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    projector_method: str = "eigh",
) -> CTMEnvironment:
    """Projector-based CTM bottom move: updates C4, T3, C3."""
    D2 = a.shape[0]
    # Grow corners
    C4g = jnp.einsum("gh,alg->hal", env.C4, env.T4)
    C4g = C4g.transpose(0, 2, 1).reshape(-1, env.T4.shape[0])
    C3g = jnp.einsum("im,erm->ire", env.C3, env.T2).reshape(-1, env.T2.shape[0])
    # Grow edge: T3[h,d,i] * a[u,d,l,r] -> (h,i,u,l,r)
    T3g = jnp.einsum("hdi,udlr->hiulr", env.T3, a)
    T3g = T3g.transpose(0, 3, 2, 1, 4).reshape(C4g.shape[0], D2, C3g.shape[0])

    C4_new, C3_new, T3_new = _ctm_move(C4g, C3g, T3g, chi, projector_method)
    return CTMEnvironment(
        C1=env.C1,
        C2=env.C2,
        C3=C3_new,
        C4=C4_new,
        T1=env.T1,
        T2=env.T2,
        T3=T3_new,
        T4=env.T4,
    )


def _ctm_left_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Projector-based 2-site CTM left move."""
    D2 = a_neighbor.shape[0]
    C1g = jnp.einsum("ab,buc->auc", env_self.C1, env_neighbor.T1).reshape(
        -1, env_neighbor.T1.shape[2]
    )
    C4g = jnp.einsum("gh,hdi->gdi", env_self.C4, env_neighbor.T3).reshape(
        -1, env_neighbor.T3.shape[2]
    )
    T4g = jnp.einsum("alg,udlr->augdr", env_self.T4, a_neighbor)
    T4g = T4g.transpose(0, 1, 4, 2, 3).reshape(C1g.shape[0], D2, C4g.shape[0])
    C1_new, C4_new, T4_new = _ctm_move(C1g, C4g, T4g, chi)
    return CTMEnvironment(
        C1=C1_new,
        C2=env_self.C2,
        C3=env_self.C3,
        C4=C4_new,
        T1=env_self.T1,
        T2=env_self.T2,
        T3=env_self.T3,
        T4=T4_new,
    )


def _ctm_right_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Projector-based 2-site CTM right move."""
    D2 = a_neighbor.shape[0]
    C2g = jnp.einsum("ce,buc->eub", env_self.C2, env_neighbor.T1).reshape(
        -1, env_neighbor.T1.shape[0]
    )
    C3g = jnp.einsum("im,hdi->mdh", env_self.C3, env_neighbor.T3).reshape(
        -1, env_neighbor.T3.shape[0]
    )
    T2g = jnp.einsum("erm,udlr->eumdl", env_self.T2, a_neighbor)
    T2g = T2g.transpose(0, 1, 4, 2, 3).reshape(C2g.shape[0], D2, C3g.shape[0])
    C2_new, C3_new, T2_new = _ctm_move(C2g, C3g, T2g, chi)
    return CTMEnvironment(
        C1=env_self.C1,
        C2=C2_new,
        C3=C3_new,
        C4=env_self.C4,
        T1=env_self.T1,
        T2=T2_new,
        T3=env_self.T3,
        T4=env_self.T4,
    )


def _ctm_top_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Projector-based 2-site CTM top move."""
    D2 = a_neighbor.shape[0]
    C1g = jnp.einsum("ab,alg->blg", env_self.C1, env_neighbor.T4).reshape(
        -1, env_neighbor.T4.shape[2]
    )
    C2g = jnp.einsum("ce,erm->crm", env_self.C2, env_neighbor.T2).reshape(
        -1, env_neighbor.T2.shape[2]
    )
    T1g = jnp.einsum("buc,udlr->bcdlr", env_self.T1, a_neighbor)
    T1g = T1g.transpose(0, 3, 2, 1, 4).reshape(C1g.shape[0], D2, C2g.shape[0])
    C1_new, C2_new, T1_new = _ctm_move(C1g, C2g, T1g, chi)
    return CTMEnvironment(
        C1=C1_new,
        C2=C2_new,
        C3=env_self.C3,
        C4=env_self.C4,
        T1=T1_new,
        T2=env_self.T2,
        T3=env_self.T3,
        T4=env_self.T4,
    )


def _ctm_bottom_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Projector-based 2-site CTM bottom move."""
    D2 = a_neighbor.shape[0]
    C4g = jnp.einsum("gh,alg->hal", env_self.C4, env_neighbor.T4)
    C4g = C4g.transpose(0, 2, 1).reshape(-1, env_neighbor.T4.shape[0])
    C3g = jnp.einsum("im,erm->ire", env_self.C3, env_neighbor.T2).reshape(
        -1, env_neighbor.T2.shape[0]
    )
    T3g = jnp.einsum("hdi,udlr->hiulr", env_self.T3, a_neighbor)
    T3g = T3g.transpose(0, 3, 2, 1, 4).reshape(C4g.shape[0], D2, C3g.shape[0])
    C4_new, C3_new, T3_new = _ctm_move(C4g, C3g, T3g, chi)
    return CTMEnvironment(
        C1=env_self.C1,
        C2=env_self.C2,
        C3=C3_new,
        C4=C4_new,
        T1=env_self.T1,
        T2=env_self.T2,
        T3=T3_new,
        T4=env_self.T4,
    )


def _split_ctm_projector(
    C1g: jax.Array,
    C2g: jax.Array,
    chi: int,
) -> jax.Array:
    """Compute projector from grown corners for split-CTMRG.

    Uses eigh-based projector on the smaller (chi*D) matrices instead of
    the (chi*D^2) matrices used in standard CTMRG.
    """
    rho = C1g @ C1g.conj().T + C2g @ C2g.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(rho)
    k = min(chi, len(eigvals))
    P = eigvecs[:, -k:][:, ::-1]
    return jax.lax.stop_gradient(P)


def _svd_split_edge(
    T_full: jax.Array,
    chi_I: int,
) -> tuple[jax.Array, jax.Array]:
    """Split standard edge tensor into ket/bra pair via SVD.

    ``T_full`` of shape ``(chi, D^2, chi)`` is reshaped to
    ``(chi, D, D, chi)`` and split into ``T_ket(chi, D, chi_I)`` and
    ``T_bra(chi_I, D, chi)`` by SVD of the (chi*D, D*chi) matrix.
    """
    chi = T_full.shape[0]
    D2 = T_full.shape[1]
    D = int(round(D2**0.5))
    T_4d = T_full.reshape(chi, D, D, chi)
    T_mat = T_4d.reshape(chi * D, D * chi)

    U, s, Vh = jnp.linalg.svd(T_mat, full_matrices=False)
    k = min(chi_I, len(s))
    sqrt_s = jnp.sqrt(s[:k])
    T_ket = (U[:, :k] * sqrt_s[None, :]).reshape(chi, D, k)
    T_bra = (sqrt_s[:, None] * Vh[:k, :]).reshape(k, D, chi)
    return T_ket, T_bra


def _split_ctm_move(
    env: SplitCTMEnvironment,
    A: jax.Array,
    chi: int,
    chi_I: int,
    direction: str,
) -> SplitCTMEnvironment:
    """Perform one split-CTMRG directional move.

    For left/top moves the corner connects to the edge's ket side (chi bond),
    so ket projectors are computed first.  For right/bottom moves the corner
    connects to the edge's bra side, so bra projectors come first.

    Steps:
    1. Compute first-layer projectors from grown corners (size chi*D).
    2. Compute second-layer projectors from mid-corners (size chi*D).
    3. Combine into full ``(chi*D^2, chi)`` projector.
    4. Apply to the standard grown edge, then SVD-split into ket/bra.
    """
    D = A.shape[0]
    a = _build_double_layer(A)
    if a.ndim == 8:
        a = a.reshape(D**2, D**2, D**2, D**2)

    if direction == "left":
        # --- ket first (C1/C4 connect to T1/T3 left bond = ket chi) ---
        C1g_ket = jnp.einsum("ab,buc->auc", env.C1, env.T1_ket).reshape(-1, chi_I)
        C4g_ket = jnp.einsum("gh,hdi->gdi", env.C4, env.T3_ket).reshape(-1, chi_I)
        P_ket = _split_ctm_projector(C1g_ket, C4g_ket, chi)
        C1_mid = P_ket.conj().T @ C1g_ket  # (chi, chi_I)
        C4_mid = P_ket.conj().T @ C4g_ket

        C1g_bra = jnp.einsum("ac,cdb->adb", C1_mid, env.T1_bra).reshape(-1, chi)
        C4g_bra = jnp.einsum("ac,cdb->adb", C4_mid, env.T3_bra).reshape(-1, chi)
        P_bra = _split_ctm_projector(C1g_bra, C4g_bra, chi)
        C1_new = P_bra.conj().T @ C1g_bra  # (chi, chi)
        C4_new = P_bra.conj().T @ C4g_bra

        # Combined projector: P_full[(a,u,U), b] = sum_J P_ket[a,u,J] * P_bra[J,U,b]
        P_ket_3d = P_ket.reshape(chi, D, -1)  # (chi, D, chi_k)
        chi_k = P_ket_3d.shape[2]
        P_bra_3d = P_bra.reshape(chi_k, D, -1)  # (chi_k, D, chi)
        P_full = jnp.einsum("auJ,JUb->auUb", P_ket_3d, P_bra_3d)
        chi_new = P_full.shape[3]
        P_full = P_full.reshape(chi * D * D, chi_new)

        # Standard grown edge for T4
        T4_full = jnp.einsum("alc,cLg->alLg", env.T4_ket, env.T4_bra)
        T4_full = T4_full.reshape(chi, D * D, chi)
        T4g = jnp.einsum("alg,udlr->augdr", T4_full, a)
        T4g = T4g.transpose(0, 1, 4, 2, 3).reshape(chi * D * D, D * D, chi * D * D)
        T4_new_full = jnp.einsum("ia,idj,jb->adb", P_full, T4g, P_full)
        T4_ket_new, T4_bra_new = _svd_split_edge(T4_new_full, chi_I)

        return SplitCTMEnvironment(
            C1=C1_new,
            C2=env.C2,
            C3=env.C3,
            C4=C4_new,
            T1_ket=env.T1_ket,
            T1_bra=env.T1_bra,
            T2_ket=env.T2_ket,
            T2_bra=env.T2_bra,
            T3_ket=env.T3_ket,
            T3_bra=env.T3_bra,
            T4_ket=T4_ket_new,
            T4_bra=T4_bra_new,
        )

    elif direction == "right":
        # --- bra first (C2/C3 connect to T1/T3 right bond = bra chi) ---
        # C2 absorbs T1_bra: C2[c,e] * T1_bra[f,U,c] -> (e, U, f) = (chi, D, chi_I)
        C2g_bra = jnp.einsum("ce,fUc->eUf", env.C2, env.T1_bra).reshape(-1, chi_I)
        # C3 absorbs T3_bra: C3[i,m] * T3_bra[f,d,i] -> (m, d, f) = (chi, D, chi_I)
        C3g_bra = jnp.einsum("im,fdi->mdf", env.C3, env.T3_bra).reshape(-1, chi_I)
        P_bra = _split_ctm_projector(C2g_bra, C3g_bra, chi)
        C2_mid = P_bra.conj().T @ C2g_bra  # (chi, chi_I)
        C3_mid = P_bra.conj().T @ C3g_bra

        # Absorb ket via interlayer: C_mid[a,f] * T_ket[b,u,f] -> (a, u, b)
        C2g_ket = jnp.einsum("af,buf->aub", C2_mid, env.T1_ket).reshape(-1, chi)
        C3g_ket = jnp.einsum("af,hdf->adh", C3_mid, env.T3_ket).reshape(-1, chi)
        P_ket = _split_ctm_projector(C2g_ket, C3g_ket, chi)
        C2_new = P_ket.conj().T @ C2g_ket  # (chi, chi)
        C3_new = P_ket.conj().T @ C3g_ket

        # Combined: P_full[(a,u,U), b] = sum_J P_bra[a,U,J] * P_ket[J,u,b]
        P_bra_3d = P_bra.reshape(chi, D, -1)  # (chi, D_bra, chi_k)
        chi_k = P_bra_3d.shape[2]
        P_ket_3d = P_ket.reshape(chi_k, D, -1)  # (chi_k, D_ket, chi)
        P_full = jnp.einsum("aUJ,Jub->auUb", P_bra_3d, P_ket_3d)
        chi_new = P_full.shape[3]
        P_full = P_full.reshape(chi * D * D, chi_new)

        # Standard grown edge for T2
        T2_full = jnp.einsum("alc,cLg->alLg", env.T2_ket, env.T2_bra)
        T2_full = T2_full.reshape(chi, D * D, chi)
        T2g = jnp.einsum("erm,udlr->eumdl", T2_full, a)
        T2g = T2g.transpose(0, 1, 4, 2, 3).reshape(chi * D * D, D * D, chi * D * D)
        T2_new_full = jnp.einsum("ia,idj,jb->adb", P_full, T2g, P_full)
        T2_ket_new, T2_bra_new = _svd_split_edge(T2_new_full, chi_I)

        return SplitCTMEnvironment(
            C1=env.C1,
            C2=C2_new,
            C3=C3_new,
            C4=env.C4,
            T1_ket=env.T1_ket,
            T1_bra=env.T1_bra,
            T2_ket=T2_ket_new,
            T2_bra=T2_bra_new,
            T3_ket=env.T3_ket,
            T3_bra=env.T3_bra,
            T4_ket=env.T4_ket,
            T4_bra=env.T4_bra,
        )

    elif direction == "top":
        # --- ket first (C1/C2 connect to T4/T2 top bond = ket chi) ---
        C1g_ket = jnp.einsum("ab,alg->blg", env.C1, env.T4_ket).reshape(-1, chi_I)
        C2g_ket = jnp.einsum("ce,erm->crm", env.C2, env.T2_ket).reshape(-1, chi_I)
        P_ket = _split_ctm_projector(C1g_ket, C2g_ket, chi)
        C1_mid = P_ket.conj().T @ C1g_ket  # (chi, chi_I)
        C2_mid = P_ket.conj().T @ C2g_ket

        C1g_bra = jnp.einsum("ac,cdb->adb", C1_mid, env.T4_bra).reshape(-1, chi)
        C2g_bra = jnp.einsum("ac,cdb->adb", C2_mid, env.T2_bra).reshape(-1, chi)
        P_bra = _split_ctm_projector(C1g_bra, C2g_bra, chi)
        C1_new = P_bra.conj().T @ C1g_bra  # (chi, chi)
        C2_new = P_bra.conj().T @ C2g_bra

        # Combined: P_full[(a,u,U), b] = sum_J P_ket[a,u,J] * P_bra[J,U,b]
        P_ket_3d = P_ket.reshape(chi, D, -1)
        chi_k = P_ket_3d.shape[2]
        P_bra_3d = P_bra.reshape(chi_k, D, -1)
        P_full = jnp.einsum("auJ,JUb->auUb", P_ket_3d, P_bra_3d)
        chi_new = P_full.shape[3]
        P_full = P_full.reshape(chi * D * D, chi_new)

        # Standard grown edge for T1
        T1_full = jnp.einsum("alc,cLg->alLg", env.T1_ket, env.T1_bra)
        T1_full = T1_full.reshape(chi, D * D, chi)
        T1g = jnp.einsum("buc,udlr->bcdlr", T1_full, a)
        T1g = T1g.transpose(0, 3, 2, 1, 4).reshape(chi * D * D, D * D, chi * D * D)
        T1_new_full = jnp.einsum("ia,idj,jb->adb", P_full, T1g, P_full)
        T1_ket_new, T1_bra_new = _svd_split_edge(T1_new_full, chi_I)

        return SplitCTMEnvironment(
            C1=C1_new,
            C2=C2_new,
            C3=env.C3,
            C4=env.C4,
            T1_ket=T1_ket_new,
            T1_bra=T1_bra_new,
            T2_ket=env.T2_ket,
            T2_bra=env.T2_bra,
            T3_ket=env.T3_ket,
            T3_bra=env.T3_bra,
            T4_ket=env.T4_ket,
            T4_bra=env.T4_bra,
        )

    else:  # bottom
        # --- bra first (C4/C3 connect to T4/T2 bottom bond = bra chi) ---
        # C4 absorbs T4_bra: C4[g,h] * T4_bra[f,L,g] -> (h, L, f) = (chi, D, chi_I)
        C4g_bra = jnp.einsum("gh,fLg->hLf", env.C4, env.T4_bra).reshape(-1, chi_I)
        # C3 absorbs T2_bra: C3[i,m] * T2_bra[f,r,m] -> (i, r, f) = (chi, D, chi_I)
        C3g_bra = jnp.einsum("im,frm->irf", env.C3, env.T2_bra).reshape(-1, chi_I)
        P_bra = _split_ctm_projector(C4g_bra, C3g_bra, chi)
        C4_mid = P_bra.conj().T @ C4g_bra  # (chi, chi_I)
        C3_mid = P_bra.conj().T @ C3g_bra

        # Absorb ket via interlayer
        # C4_mid[a,f] * T4_ket[b,l,f] -> (a, l, b) = (chi, D, chi)
        C4g_ket = jnp.einsum("af,blf->alb", C4_mid, env.T4_ket).reshape(-1, chi)
        # C3_mid[a,f] * T2_ket[e,r,f] -> (a, r, e) = (chi, D, chi)
        C3g_ket = jnp.einsum("af,erf->are", C3_mid, env.T2_ket).reshape(-1, chi)
        P_ket = _split_ctm_projector(C4g_ket, C3g_ket, chi)
        C4_new = P_ket.conj().T @ C4g_ket  # (chi, chi)
        C3_new = P_ket.conj().T @ C3g_ket

        # Combined: P_full[(a,u,U), b] = sum_J P_bra[a,U,J] * P_ket[J,u,b]
        P_bra_3d = P_bra.reshape(chi, D, -1)
        chi_k = P_bra_3d.shape[2]
        P_ket_3d = P_ket.reshape(chi_k, D, -1)
        P_full = jnp.einsum("aUJ,Jub->auUb", P_bra_3d, P_ket_3d)
        chi_new = P_full.shape[3]
        P_full = P_full.reshape(chi * D * D, chi_new)

        # Standard grown edge for T3
        T3_full = jnp.einsum("alc,cLg->alLg", env.T3_ket, env.T3_bra)
        T3_full = T3_full.reshape(chi, D * D, chi)
        T3g = jnp.einsum("hdi,udlr->hiulr", T3_full, a)
        T3g = T3g.transpose(0, 3, 2, 1, 4).reshape(chi * D * D, D * D, chi * D * D)
        T3_new_full = jnp.einsum("ia,idj,jb->adb", P_full, T3g, P_full)
        T3_ket_new, T3_bra_new = _svd_split_edge(T3_new_full, chi_I)

        return SplitCTMEnvironment(
            C1=env.C1,
            C2=env.C2,
            C3=C3_new,
            C4=C4_new,
            T1_ket=env.T1_ket,
            T1_bra=env.T1_bra,
            T2_ket=env.T2_ket,
            T2_bra=env.T2_bra,
            T3_ket=T3_ket_new,
            T3_bra=T3_bra_new,
            T4_ket=env.T4_ket,
            T4_bra=env.T4_bra,
        )
