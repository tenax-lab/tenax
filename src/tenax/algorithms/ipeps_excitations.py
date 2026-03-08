"""AD-based iPEPS quasiparticle excitations.

Implements the method from Ponsioen, Assaad & Corboz, SciPost Phys. 12, 006
(2022): construct the effective Hamiltonian and norm matrices for iPEPS
excitations using JAX automatic differentiation, then solve the generalized
eigenvalue problem for the excitation spectrum.

Key components:
1. Mixed double-layer tensors (A/B substitutions in ket/bra)
2. Mixed RDM contractions for excitation energy/norm functionals
3. H_eff(k) and N(k) matrix construction via AD
4. Generalized eigenvalue solver with null-space projection
5. High-level ``compute_excitations()`` for the dispersion relation
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps import CTMEnvironment

# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExcitationConfig:
    """Configuration for iPEPS excitation calculation.

    Attributes:
        chi:              CTM bond dimension.
        ctm_max_iter:     Maximum CTM iterations.
        ctm_conv_tol:     CTM convergence tolerance.
        num_excitations:  Number of lowest excitation energies to return.
        null_space_tol:   Threshold for filtering null-space of the norm
                          matrix (eigenvalues below this fraction of the
                          maximum are discarded).
    """

    chi: int = 20
    ctm_max_iter: int = 100
    ctm_conv_tol: float = 1e-8
    num_excitations: int = 3
    null_space_tol: float = 1e-2


@dataclass
class ExcitationResult:
    """Result of an iPEPS excitation calculation.

    Attributes:
        energies:             Excitation energies of shape
                              ``(num_k, num_excitations)``.
        momenta:              Momentum points ``(num_k, 2)``.
        ground_state_energy:  Ground state energy per site.
    """

    energies: np.ndarray
    momenta: np.ndarray
    ground_state_energy: float


# ---------------------------------------------------------------------------
# Mixed double-layer tensors
# ---------------------------------------------------------------------------


def _build_mixed_double_layer(
    A: jax.Array,
    B: jax.Array,
    position: str,
) -> jax.Array:
    """Build double-layer tensor with B substituted at *position*.

    Traces out the physical index to produce a closed double-layer tensor.

    Args:
        A: Ground state site tensor ``(D, D, D, D, d)``.
        B: Excitation perturbation tensor ``(D, D, D, D, d)``.
        position:
            ``"ket"`` — B in ket, A* in bra:
                ``a_mixed[uU,dD,lL,rR] = B[u,d,l,r,s] * conj(A[U,D,L,R,s])``
            ``"bra"`` — A in ket, B* in bra:
                ``a_mixed[uU,dD,lL,rR] = A[u,d,l,r,s] * conj(B[U,D,L,R,s])``

    Returns:
        Mixed double-layer tensor of shape ``(D^2, D^2, D^2, D^2)``.
    """
    D = A.shape[0]
    if position == "ket":
        # B in ket layer, A* in bra layer
        ao = jnp.einsum("udlrs,UDLRs->uUdDlLrR", B, jnp.conj(A))
    elif position == "bra":
        # A in ket layer, B* in bra layer
        ao = jnp.einsum("udlrs,UDLRs->uUdDlLrR", A, jnp.conj(B))
    else:
        raise ValueError(f"position must be 'ket' or 'bra', got {position!r}")
    return ao.reshape(D**2, D**2, D**2, D**2)


def _build_mixed_double_layer_open(
    A: jax.Array,
    B: jax.Array,
    position: str,
) -> jax.Array:
    """Build double-layer tensor with B substituted, physical indices open.

    Args:
        A: Ground state tensor ``(D, D, D, D, d)``.
        B: Excitation tensor ``(D, D, D, D, d)``.
        position: ``"ket"`` (B in ket, A* in bra) or ``"bra"`` (A in ket, B* in bra).

    Returns:
        Shape ``(D^2, D^2, D^2, D^2, d, d)`` with last two axes
        being ket and bra physical indices.
    """
    D = A.shape[0]
    d = A.shape[4]
    if position == "ket":
        ao = jnp.einsum("udlrs,UDLRt->uUdDlLrRst", B, jnp.conj(A))
    elif position == "bra":
        ao = jnp.einsum("udlrs,UDLRt->uUdDlLrRst", A, jnp.conj(B))
    else:
        raise ValueError(f"position must be 'ket' or 'bra', got {position!r}")
    return ao.reshape(D**2, D**2, D**2, D**2, d, d)


def _build_double_layer_BB_open(
    B: jax.Array,
) -> jax.Array:
    """Double-layer tensor with B in both ket and bra, physical indices open.

    Returns shape ``(D^2, D^2, D^2, D^2, d, d)``.
    """
    D = B.shape[0]
    d = B.shape[4]
    ao = jnp.einsum("udlrs,UDLRt->uUdDlLrRst", B, jnp.conj(B))
    return ao.reshape(D**2, D**2, D**2, D**2, d, d)


# ---------------------------------------------------------------------------
# Mixed RDM contractions
# ---------------------------------------------------------------------------


def _rdm2x1_with_open_tensors(
    ao1: jax.Array,
    ao2: jax.Array,
    env: CTMEnvironment,
    d: int,
) -> jax.Array:
    """Horizontal 2-site RDM from two open double-layer tensors and CTM env.

    Reuses the contraction structure from ``_rdm2x1`` but accepts
    pre-built open double-layer tensors (which may contain B substitutions).

    Args:
        ao1: Left site open double-layer ``(D^2, D^2, D^2, D^2, d, d)``.
        ao2: Right site open double-layer ``(D^2, D^2, D^2, D^2, d, d)``.
        env: CTM environment.
        d:   Physical dimension.

    Returns:
        RDM of shape ``(d, d, d, d)``.
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env

    UL = jnp.einsum("ab,buc->auc", C1, T1)
    UR = jnp.einsum("cuf,fg->cug", T1, C2)
    LL = jnp.einsum("gi,idj->gdj", C4, T3)
    LR = jnp.einsum("jdk,mk->jdm", T3, C3)

    Lenv = jnp.einsum("auc,axg,gdj->ucxdj", UL, T4, LL)
    Renv = jnp.einsum("cuf,frm,jdm->curjd", UR, T2, LR)

    Lenv_ao1 = jnp.einsum("ucxdj,udxrst->crjst", Lenv, ao1)
    Renv_ao2 = jnp.einsum("curjd,udlrtv->cjltv", Renv, ao2)

    rdm = jnp.einsum("crjst,cjruv->stuv", Lenv_ao1, Renv_ao2)

    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    trace = jnp.trace(rdm_mat)
    rdm_mat = rdm_mat / (trace + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def _rdm1x2_with_open_tensors(
    ao1: jax.Array,
    ao2: jax.Array,
    env: CTMEnvironment,
    d: int,
) -> jax.Array:
    """Vertical 2-site RDM from two open double-layer tensors and CTM env.

    Args:
        ao1: Top site open double-layer ``(D^2, D^2, D^2, D^2, d, d)``.
        ao2: Bottom site open double-layer ``(D^2, D^2, D^2, D^2, d, d)``.
        env: CTM environment.
        d:   Physical dimension.

    Returns:
        RDM of shape ``(d, d, d, d)``.
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env

    top_row = jnp.einsum("ab,buc,ce->aue", C1, T1, C2)
    env_row1 = jnp.einsum("aue,alf,erg->ulfrg", top_row, T4, T2)
    site1 = jnp.einsum("ulfrg,udlrst->dfgst", env_row1, ao1)

    T4_ao2 = jnp.einsum("fmh,pqmnwx->fhpqnwx", T4, ao2)
    site12 = jnp.einsum("abcst,bhaqnwx->chqnstwx", site1, T4_ao2)
    site12_r = jnp.einsum("chqnstwx,cni->hqistwx", site12, T2)

    bot_row = jnp.einsum("hj,jqk,ik->hqi", C4, T3, C3)
    rdm = jnp.einsum("hqistwx,hqi->stwx", site12_r, bot_row)

    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    trace = jnp.trace(rdm_mat)
    rdm_mat = rdm_mat / (trace + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def _rdm2x1_mixed(
    A: jax.Array,
    B: jax.Array,
    env: CTMEnvironment,
    d: int,
    sub_left: tuple[str, str],
    sub_right: tuple[str, str],
) -> jax.Array:
    """Horizontal 2-site RDM with specified ket/bra substitutions.

    Args:
        A: Ground state tensor.
        B: Excitation tensor.
        env: CTM environment.
        d: Physical dimension.
        sub_left:  ``(ket, bra)`` for left site, each ``"A"`` or ``"B"``.
        sub_right: ``(ket, bra)`` for right site, each ``"A"`` or ``"B"``.

    Returns:
        RDM of shape ``(d, d, d, d)``.
    """
    ao1 = _make_open_tensor(A, B, sub_left)
    ao2 = _make_open_tensor(A, B, sub_right)
    return _rdm2x1_with_open_tensors(ao1, ao2, env, d)


def _rdm1x2_mixed(
    A: jax.Array,
    B: jax.Array,
    env: CTMEnvironment,
    d: int,
    sub_top: tuple[str, str],
    sub_bottom: tuple[str, str],
) -> jax.Array:
    """Vertical 2-site RDM with specified ket/bra substitutions.

    Args:
        A: Ground state tensor.
        B: Excitation tensor.
        env: CTM environment.
        d: Physical dimension.
        sub_top:    ``(ket, bra)`` for top site.
        sub_bottom: ``(ket, bra)`` for bottom site.

    Returns:
        RDM of shape ``(d, d, d, d)``.
    """
    ao1 = _make_open_tensor(A, B, sub_top)
    ao2 = _make_open_tensor(A, B, sub_bottom)
    return _rdm1x2_with_open_tensors(ao1, ao2, env, d)


def _make_open_tensor(
    A: jax.Array,
    B: jax.Array,
    sub: tuple[str, str],
) -> jax.Array:
    """Build open double-layer tensor for given (ket, bra) substitution.

    Args:
        A: Ground state tensor.
        B: Excitation tensor.
        sub: ``(ket_type, bra_type)`` where each is ``"A"`` or ``"B"``.

    Returns:
        Open double-layer tensor ``(D^2, D^2, D^2, D^2, d, d)``.
    """
    from tenax.algorithms.ipeps import _build_double_layer_open

    ket_type, bra_type = sub
    if ket_type == "A" and bra_type == "A":
        return _build_double_layer_open(A)
    elif ket_type == "B" and bra_type == "A":
        return _build_mixed_double_layer_open(A, B, "ket")
    elif ket_type == "A" and bra_type == "B":
        return _build_mixed_double_layer_open(A, B, "bra")
    elif ket_type == "B" and bra_type == "B":
        return _build_double_layer_BB_open(B)
    else:
        raise ValueError(f"Invalid substitution: {sub}")


# ---------------------------------------------------------------------------
# Norm and energy functionals
# ---------------------------------------------------------------------------


def _compute_norm(
    A: jax.Array,
    B: jax.Array,
    env: CTMEnvironment,
    k: jax.Array,
    d: int,
) -> jax.Array:
    r"""Compute :math:`\langle\Phi_k(B)|\Phi_k(B)\rangle` — the norm of the excitation state.

    For a 1x1 unit cell, the dominant contribution is the on-site term
    (B at the same position in ket and bra). Off-diagonal contributions
    from B at neighboring sites enter with momentum phases
    :math:`e^{i k \cdot r}`.

    The norm is bilinear in B and B*, so ``jax.grad`` of this w.r.t. B
    at ``B = e_m`` gives the m-th column of the norm matrix N.
    """
    from tenax.algorithms.ipeps import _build_double_layer_open

    # On-site term: B in ket, B* in bra at same site, A elsewhere
    ao_BB = _build_double_layer_BB_open(B)
    ao_AA = _build_double_layer_open(A)

    # Horizontal on-site: (BB, AA) and (AA, BB)
    rdm_h_onsite = _rdm2x1_with_open_tensors(ao_BB, ao_AA, env, d)
    rdm_h_onsite2 = _rdm2x1_with_open_tensors(ao_AA, ao_BB, env, d)
    # Vertical on-site
    rdm_v_onsite = _rdm1x2_with_open_tensors(ao_BB, ao_AA, env, d)
    rdm_v_onsite2 = _rdm1x2_with_open_tensors(ao_AA, ao_BB, env, d)

    # Identity operator for computing norm from RDMs
    Id = jnp.eye(d)
    Id4 = jnp.einsum("ij,kl->ijkl", Id, Id)  # (d,d,d,d)

    norm_onsite = (
        jnp.einsum("ijkl,ijkl->", rdm_h_onsite, Id4)
        + jnp.einsum("ijkl,ijkl->", rdm_h_onsite2, Id4)
        + jnp.einsum("ijkl,ijkl->", rdm_v_onsite, Id4)
        + jnp.einsum("ijkl,ijkl->", rdm_v_onsite2, Id4)
    )

    # Off-site terms: B at neighboring sites with momentum phases
    # Horizontal: B_ket at left, B*_bra at right with phase e^{-ik_x}
    ao_Bket = _build_mixed_double_layer_open(A, B, "ket")
    ao_Bbra = _build_mixed_double_layer_open(A, B, "bra")

    phase_x = jnp.exp(1j * k[0])
    phase_y = jnp.exp(1j * k[1])

    rdm_h_off = _rdm2x1_with_open_tensors(ao_Bket, ao_Bbra, env, d)
    rdm_h_off_rev = _rdm2x1_with_open_tensors(ao_Bbra, ao_Bket, env, d)

    rdm_v_off = _rdm1x2_with_open_tensors(ao_Bket, ao_Bbra, env, d)
    rdm_v_off_rev = _rdm1x2_with_open_tensors(ao_Bbra, ao_Bket, env, d)

    norm_offsite = (
        phase_x * jnp.einsum("ijkl,ijkl->", rdm_h_off, Id4)
        + jnp.conj(phase_x) * jnp.einsum("ijkl,ijkl->", rdm_h_off_rev, Id4)
        + phase_y * jnp.einsum("ijkl,ijkl->", rdm_v_off, Id4)
        + jnp.conj(phase_y) * jnp.einsum("ijkl,ijkl->", rdm_v_off_rev, Id4)
    )

    return (norm_onsite + norm_offsite).real


def _compute_excitation_energy(
    A: jax.Array,
    B: jax.Array,
    env: CTMEnvironment,
    k: jax.Array,
    hamiltonian_gate: jax.Array,
    E_gs: float,
    d: int,
) -> jax.Array:
    r"""Compute :math:`\langle\Phi_k(B)|(H - E_{gs})|\Phi_k(B)\rangle`.

    Uses the shifted Hamiltonian ``H' = H - (E_gs / n_bonds) * I`` per
    bond so that excitation eigenvalues are directly the excitation gaps.

    Contracts 2-site RDMs with B substituted in various positions,
    weighted by momentum phases.
    """
    from tenax.algorithms.ipeps import _build_double_layer_open

    H = hamiltonian_gate.reshape(d, d, d, d)
    # Shift Hamiltonian: subtract E_gs/2 per bond (2 bonds per site)
    Id = jnp.eye(d)
    Id4 = jnp.einsum("ij,kl->ijkl", Id, Id)
    H_shifted = H - (E_gs / 2.0) * Id4

    ao_AA = _build_double_layer_open(A)
    ao_BB = _build_double_layer_BB_open(B)
    ao_Bket = _build_mixed_double_layer_open(A, B, "ket")
    ao_Bbra = _build_mixed_double_layer_open(A, B, "bra")

    phase_x = jnp.exp(1j * k[0])
    phase_y = jnp.exp(1j * k[1])

    energy = jnp.array(0.0 + 0.0j)

    # --- On-site contributions: B at same site in ket and bra ---
    # Horizontal bonds
    rdm = _rdm2x1_with_open_tensors(ao_BB, ao_AA, env, d)
    energy = energy + jnp.einsum("ijkl,ijkl->", rdm, H_shifted)

    rdm = _rdm2x1_with_open_tensors(ao_AA, ao_BB, env, d)
    energy = energy + jnp.einsum("ijkl,ijkl->", rdm, H_shifted)

    # Vertical bonds
    rdm = _rdm1x2_with_open_tensors(ao_BB, ao_AA, env, d)
    energy = energy + jnp.einsum("ijkl,ijkl->", rdm, H_shifted)

    rdm = _rdm1x2_with_open_tensors(ao_AA, ao_BB, env, d)
    energy = energy + jnp.einsum("ijkl,ijkl->", rdm, H_shifted)

    # --- Off-site: B_ket at one site, B*_bra at neighbor with phase ---
    # Horizontal
    rdm = _rdm2x1_with_open_tensors(ao_Bket, ao_Bbra, env, d)
    energy = energy + phase_x * jnp.einsum("ijkl,ijkl->", rdm, H_shifted)

    rdm = _rdm2x1_with_open_tensors(ao_Bbra, ao_Bket, env, d)
    energy = energy + jnp.conj(phase_x) * jnp.einsum("ijkl,ijkl->", rdm, H_shifted)

    # Vertical
    rdm = _rdm1x2_with_open_tensors(ao_Bket, ao_Bbra, env, d)
    energy = energy + phase_y * jnp.einsum("ijkl,ijkl->", rdm, H_shifted)

    rdm = _rdm1x2_with_open_tensors(ao_Bbra, ao_Bket, env, d)
    energy = energy + jnp.conj(phase_y) * jnp.einsum("ijkl,ijkl->", rdm, H_shifted)

    return energy.real


# ---------------------------------------------------------------------------
# H_eff and N matrix construction via AD
# ---------------------------------------------------------------------------


def _make_basis(D: int, d: int) -> list[jax.Array]:
    """Generate orthonormal basis vectors for B tensor space.

    Returns a list of ``D^4 * d`` basis tensors, each of shape
    ``(D, D, D, D, d)``.
    """
    basis_size = D**4 * d
    basis = []
    for i in range(basis_size):
        b = jnp.zeros(basis_size).at[i].set(1.0)
        basis.append(b.reshape(D, D, D, D, d))
    return basis


def _build_H_and_N(
    A: jax.Array,
    env: CTMEnvironment,
    k: jax.Array,
    hamiltonian_gate: jax.Array,
    E_gs: float,
    d: int,
    config: ExcitationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build H_eff(k) and N(k) matrices using automatic differentiation.

    For basis vector :math:`e_m` (m-th unit vector in B-parameter space):

    .. math::

        N_{:,m} = \nabla_{B^*} \langle\Phi_k(B)|\Phi_k(B)\rangle\big|_{B=e_m}

        H_{:,m} = \nabla_{B^*} \langle\Phi_k(B)|(H-E_{gs})|\Phi_k(B)\rangle\big|_{B=e_m}

    Since the functionals are bilinear in B and B*, the gradient w.r.t. B*
    at ``B = e_m`` gives the m-th column.

    Args:
        A:                Optimized ground state tensor.
        env:              Converged CTM environment.
        k:                Momentum vector ``(kx, ky)``.
        hamiltonian_gate: 2-site Hamiltonian.
        E_gs:             Ground state energy per site.
        d:                Physical dimension.
        config:           ExcitationConfig.

    Returns:
        ``(H_eff, N_mat)`` each of shape ``(basis_size, basis_size)``.
    """
    D = A.shape[0]
    basis_size = D**4 * d
    basis = _make_basis(D, d)

    # Stack basis tensors into a single JAX array: (basis_size, D, D, D, D, d)
    B_stacked = jnp.stack(basis)

    # Gradient of energy functional w.r.t. B
    def energy_fn(B):
        return _compute_excitation_energy(A, B, env, k, hamiltonian_gate, E_gs, d)

    # Gradient of norm functional w.r.t. B
    def norm_fn(B):
        return _compute_norm(A, B, env, k, d)

    # Batch-compute all gradients using vmap instead of a Python loop.
    # Each row of the output contains the gradient for the corresponding
    # basis vector.  Transposing gives the matrix whose m-th column is
    # the gradient for the m-th basis vector (matching the original API).
    H_grads = jax.vmap(jax.grad(energy_fn))(B_stacked)  # (basis_size, D, D, D, D, d)
    N_grads = jax.vmap(jax.grad(norm_fn))(B_stacked)  # (basis_size, D, D, D, D, d)

    # Reshape to (basis_size, basis_size) and transpose so that column m
    # corresponds to the gradient for basis vector m, then transfer to host.
    H_eff = np.array(H_grads.reshape(basis_size, basis_size).T)
    N_mat = np.array(N_grads.reshape(basis_size, basis_size).T)

    return H_eff, N_mat


# ---------------------------------------------------------------------------
# Generalized eigenvalue problem
# ---------------------------------------------------------------------------


def _solve_excitations(
    H_eff: np.ndarray,
    N_mat: np.ndarray,
    num_excitations: int,
    null_tol: float = 1e-3,
) -> np.ndarray:
    """Solve generalized eigenvalue problem ``H v = omega N v``.

    Steps:
    1. Symmetrize H and N.
    2. Eigendecompose N to find and project out null space.
    3. Solve reduced GEV in the non-null subspace.
    4. Return lowest excitation energies.

    Args:
        H_eff:            Effective Hamiltonian matrix.
        N_mat:            Norm matrix.
        num_excitations:  Number of excitation energies to return.
        null_tol:         Threshold for null-space filtering (relative to
                          largest N eigenvalue).

    Returns:
        Array of the lowest *num_excitations* excitation energies.
    """
    # Symmetrize
    H_eff = 0.5 * (H_eff + H_eff.conj().T)
    N_mat = 0.5 * (N_mat + N_mat.conj().T)

    # Single eigendecomposition of N — use eigenvectors directly for N^{-1/2}
    # transform (avoids redundant projection + re-eigendecomposition which can
    # introduce numerical errors on ill-conditioned norm matrices).
    eigvals_N, P = np.linalg.eigh(N_mat)

    # eigh returns sorted eigenvalues; largest is last
    max_eigval = eigvals_N[-1] if len(eigvals_N) > 0 else 1.0
    if max_eigval < 1e-15:
        return np.zeros(num_excitations)

    safe = eigvals_N > null_tol * max_eigval
    if not np.any(safe):
        return np.zeros(num_excitations)

    # N^{-1/2} regularised ordinary eigenvalue problem
    P_safe = P[:, safe]
    inv_sqrt = np.diag(1.0 / np.sqrt(eigvals_N[safe]))
    H_tilde = inv_sqrt @ P_safe.conj().T @ H_eff @ P_safe @ inv_sqrt
    H_tilde = 0.5 * (H_tilde + H_tilde.conj().T)
    eigvals = np.linalg.eigvalsh(H_tilde)

    # Always return exactly num_excitations values (pad with zeros if the
    # safe subspace is smaller than num_excitations).
    result = np.zeros(num_excitations)
    n_fill = min(num_excitations, len(eigvals))
    result[:n_fill] = eigvals[:n_fill]
    return result


# ---------------------------------------------------------------------------
# Momentum path utilities
# ---------------------------------------------------------------------------


def make_momentum_path(
    path_type: str = "brillouin",
    num_points: int = 20,
) -> list[tuple[float, float]]:
    r"""Generate momentum path through the Brillouin zone.

    For a square lattice with lattice constant 1:

    ``path_type="brillouin"``:
        :math:`\Gamma(0,0) \to X(\pi,0) \to M(\pi,\pi) \to \Gamma(0,0)`

    ``path_type="diagonal"``:
        :math:`\Gamma(0,0) \to M(\pi,\pi)`

    Args:
        path_type: Type of momentum path.
        num_points: Total number of momentum points.

    Returns:
        List of ``(kx, ky)`` tuples.
    """
    if path_type == "brillouin":
        # Three segments: Gamma->X, X->M, M->Gamma
        n1 = num_points // 3
        n2 = num_points // 3
        n3 = num_points - n1 - n2

        path = []
        # Gamma -> X: (0,0) -> (pi,0)
        for i in range(n1):
            t = i / max(n1, 1)
            path.append((t * np.pi, 0.0))

        # X -> M: (pi,0) -> (pi,pi)
        for i in range(n2):
            t = i / max(n2, 1)
            path.append((np.pi, t * np.pi))

        # M -> Gamma: (pi,pi) -> (0,0)
        for i in range(n3):
            t = i / max(n3, 1)
            path.append(((1 - t) * np.pi, (1 - t) * np.pi))

        return path

    elif path_type == "diagonal":
        path = []
        for i in range(num_points):
            t = i / max(num_points - 1, 1)
            path.append((t * np.pi, t * np.pi))
        return path

    else:
        raise ValueError(f"Unknown path_type: {path_type!r}")


# ---------------------------------------------------------------------------
# Main excitation function
# ---------------------------------------------------------------------------


def compute_excitations(
    A: jax.Array,
    env: CTMEnvironment,
    hamiltonian_gate: jax.Array,
    E_gs: float,
    momenta: list[tuple[float, float]],
    config: ExcitationConfig,
) -> ExcitationResult:
    """Compute excitation spectrum at given momentum points.

    For each momentum point, constructs the effective Hamiltonian and norm
    matrices using AD (Ponsioen et al. 2022), then solves the generalized
    eigenvalue problem for the lowest excitation energies.

    Args:
        A:                 Optimized ground state tensor ``(D, D, D, D, d)``.
        env:               Converged CTM environment for A.
        hamiltonian_gate:  2-site Hamiltonian ``(d, d, d, d)``.
        E_gs:              Ground state energy per site.
        momenta:           List of ``(kx, ky)`` momentum points.
        config:            ExcitationConfig.

    Returns:
        ExcitationResult with energies and momenta.
    """
    d = A.shape[-1]

    all_energies = []
    for kx, ky in momenta:
        k = jnp.array([kx, ky])
        H_eff, N_mat = _build_H_and_N(
            A,
            env,
            k,
            hamiltonian_gate,
            E_gs,
            d,
            config,
        )
        excitation_energies = _solve_excitations(
            H_eff,
            N_mat,
            config.num_excitations,
            config.null_space_tol,
        )
        all_energies.append(excitation_energies)

    return ExcitationResult(
        energies=np.array(all_energies),
        momenta=np.array(momenta),
        ground_state_energy=E_gs,
    )
