"""RDM construction and energy computation for iPEPS.

Reduced density matrix (RDM) building from CTM environments and
energy-per-site evaluation for 1-site and 2-site unit cells.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import (
    CTMEnvironment,
    SplitCTMEnvironment,
)
from tenax.algorithms.ipeps_ctm import (
    _split_env_to_standard,
)


def _build_double_layer_open(A: jax.Array) -> jax.Array:
    """Double-layer tensor with physical indices left open.

    Returns ``a_open`` with shape ``(D^2, D^2, D^2, D^2, d, d)`` where the
    last two axes are the ket and bra physical indices.
    """
    if A.ndim == 5:
        # A[u,d,l,r,s], conj(A)[U,D,L,R,t]
        # -> a_open[uU, dD, lL, rR, s, t]
        ao = jnp.einsum("udlrs,UDLRt->uUdDlLrRst", A, jnp.conj(A))
        D = A.shape[0]
        d = A.shape[4]
        return ao.reshape(D**2, D**2, D**2, D**2, d, d)
    raise ValueError("_build_double_layer_open requires a 5-leg tensor")


def _rdm2x1(A: jax.Array, env: CTMEnvironment, d: int) -> jax.Array:
    """Horizontal 2-site reduced density matrix from CTM environment.

    Contracts the network:

    .. code-block::

        C1 — T1 — T1 — C2
        |     |     |    |
        T4  ao_1 — ao_2  T2
        |     |     |    |
        C4 — T3 — T3 — C3

    Returns RDM with shape ``(d, d, d, d)`` — ``(s1, s2, s1', s2')``
    (ket indices first, bra indices second), so that
    ``rdm.reshape(d*d, d*d)`` is a proper density matrix.
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env
    a_open = _build_double_layer_open(A)  # (D2, D2, D2, D2, d, d)

    # Step-by-step contraction (small intermediates):
    # UL = C1 · T1  →  (chi, D2, chi)
    UL = jnp.einsum("ab,buc->auc", C1, T1)
    # UR = T1 · C2  →  (chi, D2, chi)
    UR = jnp.einsum("cuf,fg->cug", T1, C2)
    # LL = C4 · T3  →  (chi, D2, chi)
    LL = jnp.einsum("gi,idj->gdj", C4, T3)
    # LR = T3 · C3  →  (chi, D2, chi)
    LR = jnp.einsum("jdk,mk->jdm", T3, C3)

    # Left env: UL[a,u1,c] · T4[a,l1,g] · LL[g,d1,j]
    # Contract a between UL and T4, g between T4 and LL
    Lenv = jnp.einsum("auc,axg,gdj->ucxdj", UL, T4, LL)
    # shape: (D2, chi, D2, D2, chi) → (u1, c, l1, d1, j)

    # Right env: UR[c,u2,f] · T2[f,r2,m] · LR[j,d2,m]
    # Contract f between UR and T2, m between T2 and LR
    Renv = jnp.einsum("cuf,frm,jdm->curjd", UR, T2, LR)
    # shape: (chi, D2, D2, chi, D2) → (c, u2, r2, j, d2)

    # Contract Lenv with ao1[u1, d1, l1, r1, s, sp]:
    # Match: u1=u, d1=d, l1=x  → free: c, r1, j, s, sp
    Lenv_ao1 = jnp.einsum("ucxdj,udxrst->crjst", Lenv, a_open)
    # shape: (chi, D2, chi, d, d) → (c, r1, j, s1, s1')

    # Contract Renv with ao2[u2, d2, l2, r2, t, tp]:
    # Match: u2=u, d2=d, r2=r  → free: c, l2, j, t, tp
    Renv_ao2 = jnp.einsum("curjd,udlrtv->cjltv", Renv, a_open)
    # shape: (chi, chi, D2, d, d) → (c, j, l2, s2, s2')

    # Final: contract Lenv_ao1 with Renv_ao2
    # Match: c=c, j=j, r1=l2  → free: s1, s1', s2, s2'
    rdm = jnp.einsum("crjst,cjruv->stuv", Lenv_ao1, Renv_ao2)
    # rdm has convention (s1_ket, s1_bra, s2_ket, s2_bra).
    # Transpose to (s1_ket, s2_ket, s1_bra, s2_bra) so that
    # reshape(d*d, d*d) yields a proper density matrix with rows =
    # ket and columns = bra, matching the Hamiltonian convention.
    rdm = rdm.transpose(0, 2, 1, 3)

    # Symmetrize and normalize
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def _rdm1x2(A: jax.Array, env: CTMEnvironment, d: int) -> jax.Array:
    """Vertical 2-site reduced density matrix from CTM environment.

    Contracts the network:

    .. code-block::

        C1  — T1  — C2
        |      |      |
        T4 — ao_1 — T2
        |      |      |
        T4 — ao_2 — T2
        |      |      |
        C4  — T3  — C3

    Returns RDM with shape ``(d, d, d, d)`` — ``(s1, s2, s1', s2')``
    (ket indices first, bra indices second), so that
    ``rdm.reshape(d*d, d*d)`` is a proper density matrix.
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env
    a_open = _build_double_layer_open(A)

    # Top row: C1·T1·C2 → (chi, D2, chi)  indices: (a, u, e)
    top_row = jnp.einsum("ab,buc,ce->aue", C1, T1, C2)

    # Contract top_row with T4 (site-1 left) and T2 (site-1 right):
    # top_row[a, u1, e]  T4[a, l1, f]  T2[e, r1, g]
    # Contract a, e → env_row1[u1, l1, f, r1, g]
    env_row1 = jnp.einsum("aue,alf,erg->ulfrg", top_row, T4, T2)
    # (D2, D2, chi, D2, chi) → (u1, l1, f, r1, g)

    # Contract with ao1[u1, d1, l1, r1, s, sp]:  match u1, l1, r1
    site1 = jnp.einsum("ulfrg,udlrst->dfgst", env_row1, a_open)
    # (D2, chi, chi, d, d) → (d1, f, g, s1, s1')

    # Step A: contract T4[f,l2,h] with ao2[d1, d2, l2, r2, t, tp]  match l2
    # Use unique index letters: a_open → (p, q, m, n, w, x)
    #   p=d1_ao2=u, q=d2, m=l2, n=r2, w=s2, x=s2'
    T4_ao2 = jnp.einsum("fmh,pqmnwx->fhpqnwx", T4, a_open)
    # (chi, chi, D2, D2, D2, d, d) → (f, h, p=d1, q=d2, n=r2, w, x)

    # Step B: contract site1[d1, f, g, s, t] with T4_ao2[f, h, d1, d2, r2, w, x]
    # Match: d1 and f  (use a=d1, b=f)
    # site1:  a(D2) b(chi) c(chi) s(d) t(d)
    # T4_ao2: b(chi) h(chi) a(D2) q(D2) n(D2) w(d) x(d)
    site12 = jnp.einsum("abcst,bhaqnwx->chqnstwx", site1, T4_ao2)
    # (chi, chi, D2, D2, d, d, d, d) → (c=g, h, q=d2, n=r2, s1, s1', s2, s2')

    # Contract T2[g, r2, i]: match g=c, r2=n
    site12_r = jnp.einsum("chqnstwx,cni->hqistwx", site12, T2)
    # (chi, D2, chi, d, d, d, d) → (h, q=d2, i, s1, s1', s2, s2')

    # Bottom row: C4·T3·C3 → (chi, D2, chi)  indices: (h, d2, i)
    bot_row = jnp.einsum("hj,jqk,ik->hqi", C4, T3, C3)

    # Final: contract site12_r with bot_row  match h, q=d2, i
    rdm = jnp.einsum("hqistwx,hqi->stwx", site12_r, bot_row)
    # rdm has convention (s1_ket, s1_bra, s2_ket, s2_bra).
    # Transpose to (s1_ket, s2_ket, s1_bra, s2_bra) so that
    # reshape(d*d, d*d) yields a proper density matrix.
    rdm = rdm.transpose(0, 2, 1, 3)

    # Symmetrize and normalize
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def compute_energy_ctm(
    A: jax.Array,
    env: CTMEnvironment,
    hamiltonian_gate: jax.Array,
    d: int,
) -> jax.Array:
    """Compute energy per site using CTM environment and 2-site RDMs.

    Constructs horizontal and vertical two-site reduced density matrices
    from the CTM environment and contracts each with the Hamiltonian to
    obtain the energy per site.

    Args:
        A:                 PEPS site tensor of shape ``(D, D, D, D, d)``.
        env:               Converged CTMEnvironment.
        hamiltonian_gate:  2-site Hamiltonian, shape ``(d, d, d, d)``.
        d:                 Physical dimension.

    Returns:
        Scalar energy per site.
    """
    if A.ndim != 5:
        raise ValueError(
            f"compute_energy_ctm requires a 5-leg tensor (D, D, D, D, d), "
            f"got ndim={A.ndim}"
        )

    rdm_h = _rdm2x1(A, env, d)
    rdm_v = _rdm1x2(A, env, d)
    H = hamiltonian_gate.reshape(d, d, d, d)
    E_h = jnp.einsum("ijkl,ijkl->", rdm_h, H)
    E_v = jnp.einsum("ijkl,ijkl->", rdm_v, H)
    return (E_h + E_v).real


def _rdm2x1_2site(
    A: jax.Array,
    B: jax.Array,
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    d: int,
) -> jax.Array:
    """Horizontal 2-site RDM for a checkerboard unit cell (A left, B right).

    Uses mixed environment:
        C1_A — T1_A — T1_B — C2_B
        |       |       |       |
        T4_A  ao_A   ao_B    T2_B
        |       |       |       |
        C4_A — T3_A — T3_B — C3_B
    """
    ao_A = _build_double_layer_open(A)
    ao_B = _build_double_layer_open(B)

    # Left boundary from env_A
    UL = jnp.einsum("ab,buc->auc", env_A.C1, env_A.T1)
    LL = jnp.einsum("gi,idj->gdj", env_A.C4, env_A.T3)
    Lenv = jnp.einsum("auc,axg,gdj->ucxdj", UL, env_A.T4, LL)

    # Right boundary from env_B
    UR = jnp.einsum("cuf,fg->cug", env_B.T1, env_B.C2)
    LR = jnp.einsum("jdk,mk->jdm", env_B.T3, env_B.C3)
    Renv = jnp.einsum("cuf,frm,jdm->curjd", UR, env_B.T2, LR)

    # Contract left env with ao_A
    Lenv_ao = jnp.einsum("ucxdj,udxrst->crjst", Lenv, ao_A)
    # Contract right env with ao_B
    Renv_ao = jnp.einsum("curjd,udlrtv->cjltv", Renv, ao_B)
    # Final contraction
    rdm = jnp.einsum("crjst,cjruv->stuv", Lenv_ao, Renv_ao)
    # Transpose from (s1_ket, s1_bra, s2_ket, s2_bra) to
    # (s1_ket, s2_ket, s1_bra, s2_bra) for proper density matrix convention.
    rdm = rdm.transpose(0, 2, 1, 3)

    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def _rdm1x2_2site(
    A: jax.Array,
    B: jax.Array,
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    d: int,
) -> jax.Array:
    """Vertical 2-site RDM for a checkerboard unit cell (A top, B bottom).

    Uses mixed environment:
        C1_A — T1_A — C2_A
        |       |       |
        T4_A  ao_A    T2_A
        |       |       |
        T4_B  ao_B    T2_B
        |       |       |
        C4_B — T3_B — C3_B
    """
    ao_A = _build_double_layer_open(A)
    ao_B = _build_double_layer_open(B)

    # Top row from env_A
    top_row = jnp.einsum("ab,buc,ce->aue", env_A.C1, env_A.T1, env_A.C2)
    # Env row 1: contract with T4_A (left) and T2_A (right)
    env_row1 = jnp.einsum("aue,alf,erg->ulfrg", top_row, env_A.T4, env_A.T2)
    # Contract with ao_A
    site1 = jnp.einsum("ulfrg,udlrst->dfgst", env_row1, ao_A)

    # Step A: T4_B with ao_B
    T4_ao2 = jnp.einsum("fmh,pqmnwx->fhpqnwx", env_B.T4, ao_B)
    # Step B: contract site1 with T4_ao2
    site12 = jnp.einsum("abcst,bhaqnwx->chqnstwx", site1, T4_ao2)
    # Contract T2_B
    site12_r = jnp.einsum("chqnstwx,cni->hqistwx", site12, env_B.T2)
    # Bottom row from env_B
    bot_row = jnp.einsum("hj,jqk,ik->hqi", env_B.C4, env_B.T3, env_B.C3)
    # Final
    rdm = jnp.einsum("hqistwx,hqi->stwx", site12_r, bot_row)
    # Transpose from (s1_ket, s1_bra, s2_ket, s2_bra) to
    # (s1_ket, s2_ket, s1_bra, s2_bra) for proper density matrix convention.
    rdm = rdm.transpose(0, 2, 1, 3)

    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def compute_energy_ctm_2site(
    A: jax.Array,
    B: jax.Array,
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    hamiltonian_gate: jax.Array,
    d: int,
) -> jax.Array:
    """Compute energy per site for a 2-site checkerboard iPEPS.

    E/site = E_horizontal + E_vertical (one bond of each type per site).
    """
    H = hamiltonian_gate.reshape(d, d, d, d)
    rdm_h = _rdm2x1_2site(A, B, env_A, env_B, d)
    rdm_v = _rdm1x2_2site(A, B, env_A, env_B, d)
    E_h = jnp.einsum("ijkl,ijkl->", rdm_h, H)
    E_v = jnp.einsum("ijkl,ijkl->", rdm_v, H)
    return (E_h + E_v).real


def compute_energy_split_ctm(
    A: jax.Array,
    env: SplitCTMEnvironment,
    hamiltonian_gate: jax.Array,
    d: int,
) -> jax.Array:
    """Compute energy using split CTM environment.

    Converts to standard CTMEnvironment then delegates to
    :func:`compute_energy_ctm`.
    """
    std_env = _split_env_to_standard(env)
    return compute_energy_ctm(A, std_env, hamiltonian_gate, d)
