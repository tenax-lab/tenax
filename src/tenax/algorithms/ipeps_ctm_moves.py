"""CTM algorithm for iPEPS — the split-CTM projector and directional moves."""

from __future__ import annotations

__all__ = [
    "_split_ctm_projector",
    "_svd_split_edge",
    "_split_ctm_move",
]

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import (
    SplitCTMEnvironment,
)
from tenax.algorithms.ipeps_ctm_init import _build_double_layer
from tenax.linalg import _dense_svd


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

    U, s, Vh = _dense_svd(T_mat, full_matrices=False)
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
