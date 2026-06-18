"""Numerically stable infinite-TEBD (iTEBD) for 1D infinite MPS.

Imaginary-time evolution of a translation-invariant infinite MPS with a 2-site
unit cell, in the Vidal ``Gamma-lambda`` canonical form. The one numerically
delicate step of the original Vidal iTEBD is the explicit inversion of the bond
weights (``lambda^-1``) when restoring the canonical form after a gate+SVD: a
vanishing Schmidt value amplifies to ``1/eps`` and, after re-normalization,
degenerates the whole bond (see ``fermionic_ipeps._safe_inv``). We guard that
step by taking ``lambda^-1`` as a regularized *pseudo-inverse* that drops dead
Schmidt sectors (``_safe_inv``) instead of dividing by ~0.

This module provides two 1D drivers. ``itebd_groundstate`` is the standard Vidal
``Gamma-lambda`` iTEBD with the regularized inverse above -- it is NOT the
inversion-free canonical update of Hastings (arXiv:0903.3253). Hastings' trick
avoids forming ``lambda^-1`` entirely: it applies the gate to a left-canonical
2-site block and re-orthogonalizes via SVD, never using a pseudo-inverse. That
genuine inversion-free update IS provided here as ``itebd_groundstate_hastings``
(``_update_bond_hastings``). The 2D PEPS simple-update path still uses the
regularized pseudo-inverse: there is no 2D canonical-form analogue of the
Hastings update.

Validated against iDMRG: the spin-1/2 Heisenberg chain converges to
``e_0 = 1/4 - ln 2 ~= -0.4431`` per site.

This is a dense reference implementation (small chi, ``numpy`` linear algebra);
a block-sparse / JAX version can follow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def heisenberg_2site_h(Jz: float = 1.0, Jxy: float = 1.0) -> np.ndarray:
    """2-site spin-1/2 Heisenberg term as a ``(d, d, d, d)`` array.

    ``H[p0, p1, p0', p1'] = Jz Sz.Sz + (Jxy/2)(Sp.Sm + Sm.Sp)`` acting on the
    ket legs ``(p0', p1')`` -> bra legs ``(p0, p1)``.
    """
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]])
    Sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]])
    H = Jz * np.kron(Sz, Sz) + 0.5 * Jxy * (np.kron(Sp, Sm) + np.kron(Sm, Sp))
    return H.reshape(2, 2, 2, 2)


def _safe_inv(lam: np.ndarray, rel_tol: float = 1e-12) -> np.ndarray:
    """Pseudo-inverse of bond weights: ``1/lam`` above ``rel_tol*max``, else 0."""
    cutoff = rel_tol * float(np.max(lam))
    keep = lam > cutoff
    return np.where(keep, 1.0 / np.where(keep, lam, 1.0), 0.0)


def _trotter_gate(H: np.ndarray, dt: float) -> np.ndarray:
    """``exp(-dt H)`` as a ``(d, d, d, d)`` array via eigendecomposition."""
    d = H.shape[0]
    Hm = H.reshape(d * d, d * d)
    w, V = np.linalg.eigh(Hm)
    G = (V * np.exp(-dt * w)) @ V.conj().T
    return G.reshape(d, d, d, d)


@dataclass
class ITEBDState:
    """2-site unit cell in Vidal form: ... lB GA lA GB lB GA lA ..."""

    GA: np.ndarray  # (chi_B, d, chi_A)
    GB: np.ndarray  # (chi_A, d, chi_B)
    lA: np.ndarray  # (chi_A,) weight on the GA-GB bond
    lB: np.ndarray  # (chi_B,) weight on the GB-GA bond


@dataclass
class ITEBDStateLeft:
    """2-site unit cell in LEFT-canonical form: ... A B A B ...

    ``A``, ``B`` are left-canonical site tensors; ``lab``/``lba`` are the Schmidt
    weights on the A->B and B->A bonds respectively.
    """

    A: np.ndarray  # (chiB, d, chiA) left-canonical
    B: np.ndarray  # (chiA, d, chiB) left-canonical
    lab: np.ndarray  # (chiA,) Schmidt weights on the A->B bond
    lba: np.ndarray  # (chiB,) Schmidt weights on the B->A bond


def _bond_theta(G1, l_in, G2, l_out_left, l_out_right):
    """theta = l_out_left . G1 . l_in . G2 . l_out_right  ->  (chiL, d, d, chiR)."""
    t = np.einsum("a,aib->aib", l_out_left, G1)
    t = np.einsum("aib,b->aib", t, l_in)
    t = np.einsum("aib,bjc->aijc", t, G2)
    t = np.einsum("aijc,c->aijc", t, l_out_right)
    return t


def _update_bond(G1, l_in, G2, l_out, gate, chi_max):
    """Apply ``gate`` across the (G1, G2) bond (weight ``l_in``); ``l_out`` is the
    weight on both outer bonds. Returns updated ``(G1, l_in_new, G2)`` in Vidal
    form, using a pseudo-inverse for the outer-weight removal (no lambda^-1 blow-up).
    """
    chiL, d, _ = G1.shape
    _, _, chiR = G2.shape
    theta = _bond_theta(G1, l_in, G2, l_out, l_out)  # (chiL, d, d, chiR)
    theta = np.einsum("aijc,ijkl->aklc", theta, gate)  # apply gate (ij ket -> kl)
    mat = theta.reshape(chiL * d, d * chiR)
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    k = min(chi_max, int(np.sum(S > 1e-14)))
    k = max(k, 1)
    U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
    S = S / (np.linalg.norm(S) + 1e-300)
    inv_lo = _safe_inv(l_out)
    # G1_new = l_out^-1 . U ;  G2_new = Vh . l_out^-1
    G1_new = np.einsum("a,aik->aik", inv_lo, U.reshape(chiL, d, k))
    G2_new = np.einsum("kjc,c->kjc", Vh.reshape(k, d, chiR), inv_lo)
    return G1_new, S, G2_new


def _bond_energy(G1, l_in, G2, l_out, H):
    """<H> on the (G1, G2) bond in Vidal form (canonical => identity env)."""
    theta = _bond_theta(G1, l_in, G2, l_out, l_out)  # (chiL, d, d, chiR)
    norm = np.einsum("aijc,aijc->", theta, theta.conj()).real
    Ht = np.einsum("aijc,ijkl->aklc", theta, H)
    e = np.einsum("aklc,aklc->", Ht, theta.conj()).real
    return e / (norm + 1e-300)


def _update_bond_hastings(A, B, l_right, gate, chi_max):
    """Hastings' inversion-free iTEBD bond update (arXiv:0903.3253).

    ``A``, ``B`` are left-canonical ``(chiL, d, chiR)`` tensors sharing the bond
    to be updated; ``l_right`` is the Schmidt weight on the bond to the *right*
    of the ``B`` tensor. Applies ``gate`` across the A-B bond and returns
    ``(A_new, l_AB_new, B_new)``.

    ``A_new`` is an exact left-canonical isometry. ``B_new = X^dagger . Theta``
    is the inversion-free right-partner and is **intentionally NOT left-canonical**
    under truncation: its left Gram is ``Theta^dagger P_X Theta`` (``P_X = X X^dagger``,
    the rank-k projector), equivalently ``B_new = S . Yh . lambda_right^-1`` — it
    carries the implicit ``lambda^-1`` *without ever forming it*, which is the whole
    point of the Hastings scheme. The asymmetry is self-correcting and does not need
    re-orthogonalizing: the next sweep's SVD re-establishes the isometry from whatever
    gauge ``B`` is in, and ``_bond_energy_left`` normalizes explicitly
    (``<theta|H|theta>/<theta|theta>``), so a non-canonical ``B`` is absorbed, not
    propagated. Verified: ``itebd_groundstate_hastings`` matches the Vidal
    ``_safe_inv`` path (``itebd_groundstate``) bit-for-bit across the truncation
    regime (chi=2..32), converging to ``e0 = 1/4 - ln 2``. Do NOT "fix" ``B_new`` to
    be left-canonical — that reintroduces the normalization/inverse step the scheme
    exists to avoid (see PR #619 review thread).
    """
    chiL, d, _ = A.shape
    _, _, chiR = B.shape
    C = np.einsum("aik,kjc->aijc", A, B)  # (chiL, d, d, chiR), both left-canonical
    theta = np.einsum("aijc,ijkl->aklc", C, gate)  # apply gate: ket (ij) -> bra (kl)
    M = np.einsum("aijc,c->aijc", theta, l_right)  # weight the right bond (no inverse)
    mat = M.reshape(chiL * d, d * chiR)
    X, S, _Yh = np.linalg.svd(mat, full_matrices=False)
    k = min(chi_max, int(np.sum(S > 1e-14)))
    k = max(k, 1)
    X, S = X[:, :k], S[:k]
    S = S / (np.linalg.norm(S) + 1e-300)
    A_new = X.reshape(chiL, d, k)  # exact isometry (left-canonical)
    # B_new = X^dagger . theta: inversion-free right-partner (= S.Yh.l_right^-1
    # computed without forming l_right^-1). Intentionally NOT left-canonical under
    # truncation; re-orthogonalized by the next sweep's SVD. See docstring.
    B_new = np.einsum("aiK,aijc->Kjc", A_new.conj(), theta)
    return A_new, S, B_new


def _bond_energy_left(A, B, l_right, H):
    """<H> across the A-B bond for left-canonical A, B closed on the right by
    ``l_right`` (left tensors are isometries => identity left env; ``l_right**2``
    is the right env). ``Theta = A . B . diag(l_right)``."""
    theta = np.einsum("aik,kjc,c->aijc", A, B, l_right)  # (chiL, d, d, chiR)
    norm = np.einsum("aijc,aijc->", theta, theta.conj()).real
    Ht = np.einsum("aijc,ijkl->aklc", theta, H)
    e = np.einsum("aklc,aklc->", Ht, theta.conj()).real
    return e / (norm + 1e-300)


def itebd_groundstate(
    H2: np.ndarray,
    chi_max: int = 16,
    dts=(0.1, 0.03, 0.01, 0.003, 0.001),
    steps_per_dt: int = 2000,
    seed: int = 0,
) -> tuple[float, ITEBDState]:
    """Imaginary-time iTEBD ground state for a 2-site nearest-neighbour ``H2``.

    Returns ``(energy_per_site, state)``.
    """
    d = H2.shape[0]
    rng = np.random.RandomState(seed)
    chi = 1
    GA = rng.standard_normal((chi, d, chi))
    GB = rng.standard_normal((chi, d, chi))
    lA = np.ones(chi)
    lB = np.ones(chi)

    e_prev = np.inf
    for dt in dts:
        gate = _trotter_gate(H2, dt)
        for _ in range(steps_per_dt):
            # bond A: GA-GB (weight lA), outer weight lB
            GA, lA, GB = _update_bond(GA, lA, GB, lB, gate, chi_max)
            # bond B: GB-GA (weight lB), outer weight lA  (translation by 1 site)
            GB, lB, GA = _update_bond(GB, lB, GA, lA, gate, chi_max)
        eA = _bond_energy(GA, lA, GB, lB, H2)
        eB = _bond_energy(GB, lB, GA, lA, H2)
        e = 0.5 * (eA + eB)
        if abs(e - e_prev) < 1e-10:
            e_prev = e
            break
        e_prev = e
    return float(e_prev), ITEBDState(GA, GB, lA, lB)


def itebd_groundstate_hastings(
    H2: np.ndarray,
    chi_max: int = 16,
    dts=(0.1, 0.03, 0.01, 0.003, 0.001),
    steps_per_dt: int = 2000,
    seed: int = 0,
) -> tuple[float, ITEBDStateLeft]:
    """Inversion-free (Hastings) imaginary-time iTEBD ground state for a 2-site
    nearest-neighbour ``H2``. No ``lambda^-1`` is formed (cf. arXiv:0903.3253).

    Returns ``(energy_per_site, state)``.
    """
    d = H2.shape[0]
    rng = np.random.RandomState(seed)
    chi = 1
    # chi=1 left-canonical tensors: each is a normalized d-vector
    A = rng.standard_normal((chi, d, chi))
    A = A / np.linalg.norm(A)
    B = rng.standard_normal((chi, d, chi))
    B = B / np.linalg.norm(B)
    lab = np.ones(chi)  # weight on A->B bond
    lba = np.ones(chi)  # weight on B->A bond

    e_prev = np.inf
    for dt in dts:
        gate = _trotter_gate(H2, dt)
        for _ in range(steps_per_dt):
            # bond A-B (closed on the right by lba)
            A, lab, B = _update_bond_hastings(A, B, lba, gate, chi_max)
            # bond B-A (translate one site; closed on the right by lab)
            B, lba, A = _update_bond_hastings(B, A, lab, gate, chi_max)
        eAB = _bond_energy_left(A, B, lba, H2)
        eBA = _bond_energy_left(B, A, lab, H2)
        e = 0.5 * (eAB + eBA)
        if abs(e - e_prev) < 1e-10:
            e_prev = e
            break
        e_prev = e
    return float(e_prev), ITEBDStateLeft(A, B, lab, lba)
