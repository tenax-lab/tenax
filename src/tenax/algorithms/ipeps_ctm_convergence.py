"""CTM algorithm for iPEPS — sweep loops, convergence, and entry points."""

from __future__ import annotations

__all__ = [
    "_ctm_sweep",
    "_ctm_sv_diff",
    "ctm",
    "_renormalize_env",
    "_ctm_2site_sweep",
    "ctm_2site",
    "_split_ctm_sweep",
    "_split_env_to_standard",
    "ctm_split",
]

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import (
    CTMConfig,
    CTMEnvironment,
    SplitCTMEnvironment,
)
from tenax.algorithms.ipeps_ctm_init import (
    _build_double_layer,
    _initialize_ctm_env,
    _initialize_split_ctm_env,
)
from tenax.algorithms.ipeps_ctm_moves import (
    _ctm_bottom_move,
    _ctm_bottom_move_2site,
    _ctm_left_move,
    _ctm_left_move_2site,
    _ctm_right_move,
    _ctm_right_move_2site,
    _ctm_top_move,
    _ctm_top_move_2site,
    _split_ctm_move,
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
