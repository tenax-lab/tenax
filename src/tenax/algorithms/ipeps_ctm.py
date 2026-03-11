"""CTM algorithm functions for iPEPS.

Corner Transfer Matrix (CTM) methods for computing the environment
of an infinite PEPS with 1x1 and 2-site unit cells, including the
split-CTMRG variant (arXiv:2502.10298).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import (
    CTMConfig,
    CTMEnvironment,
    SplitCTMEnvironment,
)
from tenax.core import EPS


def _ctm_sweep(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    renormalize: bool,
    projector_method: str = "eigh",
) -> CTMEnvironment:
    """One full CTM sweep: left, right, top, bottom moves + optional renormalize."""
    env = _ctm_left_move(env, a, chi, projector_method)
    env = _ctm_right_move(env, a, chi, projector_method)
    env = _ctm_top_move(env, a, chi, projector_method)
    env = _ctm_bottom_move(env, a, chi, projector_method)
    if renormalize:
        env = _renormalize_env(env)
    return env


def _ctm_sv_diff(sv_new: jax.Array, sv_old: jax.Array) -> jax.Array:
    """Compute max absolute difference between normalized singular value vectors."""
    sv1 = sv_new / (jnp.sum(sv_new) + 1e-15)
    sv2 = sv_old / (jnp.sum(sv_old) + 1e-15)
    return jnp.max(jnp.abs(sv1 - sv2))


def ctm(
    A: jax.Array,
    config: CTMConfig,
    initial_env: CTMEnvironment | None = None,
) -> CTMEnvironment:
    """Compute CTM environment for a PEPS with 1x1 unit cell.

    Runs the CTM algorithm (Corboz/Orus scheme) until convergence.
    The input A is the double-layer tensor A * A^* combined, or the
    single-layer A from which the doubled tensor is computed.

    The iteration loop uses ``jax.lax.while_loop`` so that the entire
    convergence procedure can be JIT-compiled without host sync.

    Args:
        A:           Site tensor (single layer) of PEPS.
        config:      CTMConfig.
        initial_env: Optional starting environment for warm start.

    Returns:
        Converged CTMEnvironment.
    """
    chi = config.chi

    # Build the double-layer tensor a = sum_s A[s,...] * conj(A[s,...])
    # For a simple 1x1 cell: a[u,d,l,r, U,D,L,R] = sum_s A[u,d,l,r,s]*A*[U,D,L,R,s]
    # The physical index is traced over.
    a = _build_double_layer(A)  # shape (D, D, D, D, D, D, D, D)
    # Reshape to (D^2, D^2, D^2, D^2) for CTM
    if a.ndim == 8:
        D_phys = a.shape[0]
        a = a.reshape(D_phys**2, D_phys**2, D_phys**2, D_phys**2)
    elif a.ndim == 4:
        pass  # already (D^2, D^2, D^2, D^2)

    # Initialize environment tensors
    if initial_env is not None:
        env = initial_env
    else:
        env = _initialize_ctm_env(a, chi)

    max_iter = config.max_iter
    conv_tol = config.conv_tol
    renormalize = config.renormalize
    projector_method = config.projector_method

    # QR warm-up: run a few eigh iterations before switching to QR
    if projector_method == "qr" and config.qr_warmup_steps > 0:
        warmup_steps = min(config.qr_warmup_steps, max_iter)
        for _ in range(warmup_steps):
            env = _ctm_sweep(env, a, chi, renormalize, "eigh")
        max_iter = max_iter - warmup_steps

    # Initial singular values (zeros — first iteration never converges)
    prev_sv = jnp.zeros(min(chi, env.C1.shape[0]), dtype=env.C1.dtype)

    # Carry: (env, prev_sv, iteration, converged)
    init_carry = (env, prev_sv, jnp.array(0, dtype=jnp.int32), jnp.bool_(False))

    def cond_fn(carry):
        _, _, iteration, converged = carry
        return ~converged & (iteration < max_iter)

    def body_fn(carry):
        env_i, prev_sv_i, iteration, _ = carry
        env_i = _ctm_sweep(env_i, a, chi, renormalize, projector_method)
        current_sv = jnp.linalg.svd(env_i.C1, compute_uv=False)
        diff = _ctm_sv_diff(current_sv, prev_sv_i)
        converged = diff < conv_tol
        return (env_i, current_sv, iteration + 1, converged)

    env, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return env


def _build_double_layer(A: jax.Array) -> jax.Array:
    """Build the double-layer tensor from a PEPS site tensor.

    For a tensor A with shape (D,...,d) where d is the physical dimension
    and D's are virtual bond dimensions, the double-layer tensor is:
    a[virtual...] = sum_s A[virtual..., s] * conj(A[virtual..., s])

    This traces out the physical index.
    """
    if A.ndim == 5:
        # A[u, d, l, r, s] — fuse ket/bra pairs per spatial direction
        return jnp.einsum("udlrs,UDLRs->uUdDlLrR", A, jnp.conj(A))
    elif A.ndim == 3:
        # A[l, r, s] — simplified 2D
        return jnp.einsum("lrs,LRs->lrLR", A, jnp.conj(A))
    else:
        # Generic: assume last index is physical
        # Squeeze to remove degenerate dims
        s_idx = "".join(chr(97 + i) for i in range(A.ndim))
        phys = s_idx[-1]
        virt1 = s_idx[:-1]
        virt2 = virt1.upper()
        return jnp.einsum(f"{s_idx},{virt2}{phys}->{virt1}{virt2}", A, jnp.conj(A))


def _initialize_ctm_env(a: jax.Array, chi: int) -> CTMEnvironment:
    """Initialize CTM environment tensors from the PEPS double-layer tensor.

    Uses a simple initialization: corners and edges built from partial traces
    of the double-layer tensor.

    Args:
        a:   Double-layer tensor of shape (D2, D2, D2, D2).
        chi: Environment bond dimension.
    """
    D2 = a.shape[0]
    dtype = a.dtype

    # Initialize corners as identity matrices (chi x chi)
    C = jnp.eye(min(chi, D2), dtype=dtype)
    C_small = jnp.zeros((chi, chi), dtype=dtype)
    C_small = C_small.at[: C.shape[0], : C.shape[1]].set(
        C[: min(chi, C.shape[0]), : min(chi, C.shape[1])]
    )

    # Initialize edges as a slice of the double-layer tensor
    # T[chi, D2, chi] — use first chi values
    T_chi = min(chi, D2)
    T_init = jnp.zeros((chi, D2, chi), dtype=dtype)
    # Fill with identity-like structure
    for i in range(min(T_chi, chi)):
        T_init = T_init.at[i, :, i].add(jnp.ones(D2))

    return CTMEnvironment(
        C1=C_small,
        C2=C_small,
        C3=C_small,
        C4=C_small,
        T1=T_init,
        T2=T_init,
        T3=T_init,
        T4=T_init,
    )


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


def _renormalize_env(env: CTMEnvironment) -> CTMEnvironment:
    """Normalize environment tensors to prevent exponential growth."""

    def normalize(x: jax.Array) -> jax.Array:
        norm = jnp.max(jnp.abs(x))
        return x / (norm + EPS)

    return CTMEnvironment(
        C1=normalize(env.C1),
        C2=normalize(env.C2),
        C3=normalize(env.C3),
        C4=normalize(env.C4),
        T1=normalize(env.T1),
        T2=normalize(env.T2),
        T3=normalize(env.T3),
        T4=normalize(env.T4),
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


def _ctm_2site_sweep(
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    a_A: jax.Array,
    a_B: jax.Array,
    chi: int,
    renormalize: bool,
) -> tuple[CTMEnvironment, CTMEnvironment]:
    """One full 2-site CTM sweep: L/R/T/B moves for both sublattices + renormalize."""
    # Left moves
    env_A = _ctm_left_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_left_move_2site(env_B, env_A, a_A, chi)
    # Right moves
    env_A = _ctm_right_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_right_move_2site(env_B, env_A, a_A, chi)
    # Top moves
    env_A = _ctm_top_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_top_move_2site(env_B, env_A, a_A, chi)
    # Bottom moves
    env_A = _ctm_bottom_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_bottom_move_2site(env_B, env_A, a_A, chi)
    if renormalize:
        env_A = _renormalize_env(env_A)
        env_B = _renormalize_env(env_B)
    return env_A, env_B


def ctm_2site(
    A: jax.Array,
    B: jax.Array,
    config: CTMConfig,
) -> tuple[CTMEnvironment, CTMEnvironment]:
    """Compute CTM environments for a 2-site checkerboard unit cell.

    On a checkerboard, all neighbors of A are B and vice versa. Each
    absorption move for env_A uses B's double-layer tensor and T's from
    env_B, and vice versa.

    The iteration loop uses ``jax.lax.while_loop`` so that the entire
    convergence procedure can be JIT-compiled without host sync.

    Args:
        A: Site tensor for sublattice A, shape (D, D, D, D, d).
        B: Site tensor for sublattice B, shape (D, D, D, D, d).
        config: CTMConfig.

    Returns:
        (env_A, env_B) — converged CTM environments for each sublattice.
    """
    chi = config.chi

    a_A = _build_double_layer(A)
    a_B = _build_double_layer(B)
    D_A = A.shape[0]
    D_B = B.shape[0]
    if a_A.ndim == 8:
        a_A = a_A.reshape(D_A**2, D_A**2, D_A**2, D_A**2)
    if a_B.ndim == 8:
        a_B = a_B.reshape(D_B**2, D_B**2, D_B**2, D_B**2)

    env_A = _initialize_ctm_env(a_A, chi)
    env_B = _initialize_ctm_env(a_B, chi)

    max_iter = config.max_iter
    conv_tol = config.conv_tol
    renormalize = config.renormalize

    # Initial singular values (zeros — first iteration never converges)
    sv_size_A = min(chi, env_A.C1.shape[0])
    sv_size_B = min(chi, env_B.C1.shape[0])
    prev_sv_A = jnp.zeros(sv_size_A, dtype=env_A.C1.dtype)
    prev_sv_B = jnp.zeros(sv_size_B, dtype=env_B.C1.dtype)

    # Carry: (env_A, env_B, prev_sv_A, prev_sv_B, iteration, converged)
    init_carry = (
        env_A,
        env_B,
        prev_sv_A,
        prev_sv_B,
        jnp.array(0, dtype=jnp.int32),
        jnp.bool_(False),
    )

    def cond_fn(carry):
        _, _, _, _, iteration, converged = carry
        return ~converged & (iteration < max_iter)

    def body_fn(carry):
        eA, eB, psA, psB, iteration, _ = carry
        eA, eB = _ctm_2site_sweep(eA, eB, a_A, a_B, chi, renormalize)
        sv_A = jnp.linalg.svd(eA.C1, compute_uv=False)
        sv_B = jnp.linalg.svd(eB.C1, compute_uv=False)
        diff_A = _ctm_sv_diff(sv_A, psA)
        diff_B = _ctm_sv_diff(sv_B, psB)
        converged = jnp.maximum(diff_A, diff_B) < conv_tol
        return (eA, eB, sv_A, sv_B, iteration + 1, converged)

    env_A, env_B, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return env_A, env_B


# ---------------------------------------------------------------------------
# Split-CTMRG: ket/bra layers kept separate (arXiv:2502.10298)
# ---------------------------------------------------------------------------


def _initialize_split_ctm_env(
    A: jax.Array,
    chi: int,
    chi_I: int,
) -> SplitCTMEnvironment:
    """Initialize a SplitCTMEnvironment from the PEPS site tensor.

    Args:
        A:     Site tensor of shape ``(D, D, D, D, d)``.
        chi:   Environment bond dimension.
        chi_I: Interlayer bond dimension.
    """
    D = A.shape[0]
    dtype = A.dtype

    # Corners: identity-like (chi x chi)
    C = jnp.eye(min(chi, D), dtype=dtype)
    C_pad = jnp.zeros((chi, chi), dtype=dtype)
    C_pad = C_pad.at[: C.shape[0], : C.shape[1]].set(C)

    # Split edges: identity-like structure
    chi_D = min(chi, D)
    chi_I_D = min(chi_I, D)

    T_ket = jnp.zeros((chi, D, chi_I), dtype=dtype)
    for i in range(min(chi_D, chi_I_D)):
        T_ket = T_ket.at[i, :, i].set(jnp.ones(D))

    T_bra = jnp.zeros((chi_I, D, chi), dtype=dtype)
    for i in range(min(chi_I_D, chi_D)):
        T_bra = T_bra.at[i, :, i].set(jnp.ones(D))

    return SplitCTMEnvironment(
        C1=C_pad,
        C2=C_pad,
        C3=C_pad,
        C4=C_pad,
        T1_ket=T_ket,
        T1_bra=T_bra,
        T2_ket=T_ket,
        T2_bra=T_bra,
        T3_ket=T_ket,
        T3_bra=T_bra,
        T4_ket=T_ket,
        T4_bra=T_bra,
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
        # C2 absorbs T1_bra: C2[c,e] * T1_bra[f,U,c] → (e, U, f) = (chi, D, chi_I)
        C2g_bra = jnp.einsum("ce,fUc->eUf", env.C2, env.T1_bra).reshape(-1, chi_I)
        # C3 absorbs T3_bra: C3[i,m] * T3_bra[f,d,i] → (m, d, f) = (chi, D, chi_I)
        C3g_bra = jnp.einsum("im,fdi->mdf", env.C3, env.T3_bra).reshape(-1, chi_I)
        P_bra = _split_ctm_projector(C2g_bra, C3g_bra, chi)
        C2_mid = P_bra.conj().T @ C2g_bra  # (chi, chi_I)
        C3_mid = P_bra.conj().T @ C3g_bra

        # Absorb ket via interlayer: C_mid[a,f] * T_ket[b,u,f] → (a, u, b)
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
        # C4 absorbs T4_bra: C4[g,h] * T4_bra[f,L,g] → (h, L, f) = (chi, D, chi_I)
        C4g_bra = jnp.einsum("gh,fLg->hLf", env.C4, env.T4_bra).reshape(-1, chi_I)
        # C3 absorbs T2_bra: C3[i,m] * T2_bra[f,r,m] → (i, r, f) = (chi, D, chi_I)
        C3g_bra = jnp.einsum("im,frm->irf", env.C3, env.T2_bra).reshape(-1, chi_I)
        P_bra = _split_ctm_projector(C4g_bra, C3g_bra, chi)
        C4_mid = P_bra.conj().T @ C4g_bra  # (chi, chi_I)
        C3_mid = P_bra.conj().T @ C3g_bra

        # Absorb ket via interlayer
        # C4_mid[a,f] * T4_ket[b,l,f] → (a, l, b) = (chi, D, chi)
        C4g_ket = jnp.einsum("af,blf->alb", C4_mid, env.T4_ket).reshape(-1, chi)
        # C3_mid[a,f] * T2_ket[e,r,f] → (a, r, e) = (chi, D, chi)
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


def _split_ctm_sweep(
    env: SplitCTMEnvironment,
    A: jax.Array,
    chi: int,
    chi_I: int,
    renormalize: bool,
) -> SplitCTMEnvironment:
    """One full split-CTM sweep: L/R/T/B moves + optional renormalize."""
    env = _split_ctm_move(env, A, chi, chi_I, "left")
    env = _split_ctm_move(env, A, chi, chi_I, "right")
    env = _split_ctm_move(env, A, chi, chi_I, "top")
    env = _split_ctm_move(env, A, chi, chi_I, "bottom")
    if renormalize:

        def normalize(x: jax.Array) -> jax.Array:
            norm = jnp.max(jnp.abs(x))
            return x / (norm + EPS)

        def normalize_pair(
            T_ket: jax.Array, T_bra: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            """Normalize ket/bra pair using a shared factor.

            Uses the geometric mean of the max-abs norms to preserve the
            relative scaling set by the SVD split.
            """
            nk = jnp.max(jnp.abs(T_ket))
            nb = jnp.max(jnp.abs(T_bra))
            shared = jnp.sqrt(nk * nb) + EPS
            return T_ket / shared, T_bra / shared

        T1k, T1b = normalize_pair(env.T1_ket, env.T1_bra)
        T2k, T2b = normalize_pair(env.T2_ket, env.T2_bra)
        T3k, T3b = normalize_pair(env.T3_ket, env.T3_bra)
        T4k, T4b = normalize_pair(env.T4_ket, env.T4_bra)

        env = SplitCTMEnvironment(
            C1=normalize(env.C1),
            C2=normalize(env.C2),
            C3=normalize(env.C3),
            C4=normalize(env.C4),
            T1_ket=T1k,
            T1_bra=T1b,
            T2_ket=T2k,
            T2_bra=T2b,
            T3_ket=T3k,
            T3_bra=T3b,
            T4_ket=T4k,
            T4_bra=T4b,
        )
    return env


def _split_env_to_standard(
    env: SplitCTMEnvironment,
) -> CTMEnvironment:
    """Convert SplitCTMEnvironment to standard CTMEnvironment.

    Contracts each ``(T_ket, T_bra)`` pair over the interlayer bond::

        T_full[a, (uU), b] = sum_c T_ket[a, u, c] * T_bra[c, U, b]
    """
    chi = env.C1.shape[0]

    def merge(T_ket, T_bra):
        D = T_ket.shape[1]
        T = jnp.einsum("auc,cUb->auUb", T_ket, T_bra)
        return T.reshape(chi, D * D, chi)

    return CTMEnvironment(
        C1=env.C1,
        C2=env.C2,
        C3=env.C3,
        C4=env.C4,
        T1=merge(env.T1_ket, env.T1_bra),
        T2=merge(env.T2_ket, env.T2_bra),
        T3=merge(env.T3_ket, env.T3_bra),
        T4=merge(env.T4_ket, env.T4_bra),
    )


def ctm_split(
    A: jax.Array,
    config: CTMConfig,
) -> SplitCTMEnvironment:
    """Compute split-CTM environment for a PEPS with 1x1 unit cell.

    Uses the split-CTMRG algorithm (arXiv:2502.10298) where ket and bra
    layers are kept separate, reducing projector cost from O(chi^3 * D^6)
    to O(chi^3 * D^3).

    Args:
        A:      Site tensor of shape ``(D, D, D, D, d)``.
        config: CTMConfig with ``chi_I`` set.

    Returns:
        Converged SplitCTMEnvironment.
    """
    chi = config.chi
    chi_I = config.chi_I if config.chi_I is not None else chi

    env = _initialize_split_ctm_env(A, chi, chi_I)

    prev_sv = None
    for _ in range(config.max_iter):
        env = _split_ctm_sweep(env, A, chi, chi_I, config.renormalize)

        current_sv = jnp.linalg.svd(env.C1, compute_uv=False)
        if prev_sv is not None:
            diff = _ctm_sv_diff(current_sv, prev_sv)
            if float(diff) < config.conv_tol:
                break
        prev_sv = current_sv

    return env
