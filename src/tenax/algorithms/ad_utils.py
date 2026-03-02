"""Stable automatic differentiation utilities for iPEPS.

Implements the solutions from Francuz et al., Phys. Rev. Research 7, 013237
(2025) for stable AD through CTM:

1. Custom truncated SVD with Lorentzian regularization for degenerate singular
   values and the full truncation correction term.
2. CTM fixed-point implicit differentiation (avoids storing all CTM iterations).
3. Gauge fixing for element-wise CTM convergence.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from tenax.algorithms.ipeps import CTMConfig, CTMEnvironment

# ---------------------------------------------------------------------------
# 1. Truncated SVD with stable backward pass
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def truncated_svd_ad(
    M: jax.Array,
    chi: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Truncated SVD with correct and stable backward pass.

    Forward: standard SVD truncated to *chi* singular values.
    Backward: Lorentzian-regularized F-matrix + truncation correction.

    Args:
        M:   2-D matrix of shape ``(m, n)``.
        chi: Number of singular values/vectors to keep.

    Returns:
        ``(U, s, Vh)`` truncated to *chi*.
    """
    U, s, Vh = jnp.linalg.svd(M, full_matrices=False)
    k = min(chi, s.shape[0])
    return U[:, :k], s[:k], Vh[:k, :]


def _truncated_svd_ad_fwd(
    M: jax.Array,
    chi: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple]:
    """Forward pass — store full SVD for backward."""
    U_full, s_full, Vh_full = jnp.linalg.svd(M, full_matrices=False)
    k = min(chi, s_full.shape[0])
    U = U_full[:, :k]
    s = s_full[:k]
    Vh = Vh_full[:k, :]
    residuals = (U_full, s_full, Vh_full, M, k)
    return (U, s, Vh), residuals


def _truncated_svd_ad_bwd(
    chi: int,
    residuals: tuple,
    g: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array]:
    """Backward pass with Lorentzian regularization and truncation term.

    Implements the stable SVD adjoint from Francuz et al. PRR 7, 013237:
    - Lorentzian broadening ``s_i^2 - s_j^2 / ((s_i^2-s_j^2)^2 + eps^2)``
      prevents divergences from degenerate singular values.
    - Full truncation correction accounts for coupling between kept and
      discarded subspaces (the dominant error source identified by Francuz
      et al.).
    """
    U_full, s_full, Vh_full, M, k = residuals
    dU, ds, dVh = g

    eps = 1e-12  # Lorentzian broadening parameter

    # Kept subspace
    U = U_full[:, :k]
    s = s_full[:k]
    V = Vh_full[:k, :].conj().T  # (n, k)

    # --- Lorentzian-regularized F-matrix ---
    # F_ij = (s_i^2 - s_j^2) / ((s_i^2 - s_j^2)^2 + eps^2)
    # Prevents divergences from degenerate singular values.
    s2 = s**2
    diff = s2[:, None] - s2[None, :]
    F = diff / (diff**2 + eps**2)
    # Zero diagonal (gauge freedom)
    F = F - jnp.diag(jnp.diag(F))

    # Antisymmetric parts of projected cotangents
    UtdU = U.conj().T @ dU  # (k, k)
    VtdV = V.conj().T @ dVh.conj().T  # (k, k)
    UtdU_anti = 0.5 * (UtdU - UtdU.conj().T)
    VtdV_anti = 0.5 * (VtdV - VtdV.conj().T)

    # Inverse singular values (safe)
    s_inv = jnp.where(s > eps, 1.0 / s, 0.0)

    # Projectors onto complements of kept subspaces
    proj_U_perp = jnp.eye(M.shape[0]) - U @ U.conj().T
    proj_V_perp = jnp.eye(M.shape[1]) - V @ V.conj().T

    # Assemble gradient (Wan & Narayanan 2023 / Francuz et al.):
    dM = jnp.zeros_like(M)

    # 1. Diagonal part from ds
    dM = dM + U @ jnp.diag(ds) @ Vh_full[:k, :]

    # 2. Off-diagonal from dU (within kept subspace)
    dM = dM + U @ (F * UtdU_anti) @ jnp.diag(s) @ Vh_full[:k, :]

    # 3. Off-diagonal from dVh (within kept subspace)
    dM = dM + U @ jnp.diag(s) @ (F * VtdV_anti) @ Vh_full[:k, :]

    # 4. Truncation correction from dU (kept-truncated coupling)
    dM = dM + proj_U_perp @ dU @ jnp.diag(s_inv) @ Vh_full[:k, :]

    # 5. Truncation correction from dVh (kept-truncated coupling)
    dM = dM + U @ jnp.diag(s_inv) @ dVh @ proj_V_perp

    return (dM,)


truncated_svd_ad.defvjp(_truncated_svd_ad_fwd, _truncated_svd_ad_bwd)


# ---------------------------------------------------------------------------
# 2. CTM fixed-point implicit differentiation
# ---------------------------------------------------------------------------


def _ctm_step(A: jax.Array, env: CTMEnvironment, config: CTMConfig) -> CTMEnvironment:
    """One full CTM iteration (left, right, top, bottom moves + renormalize).

    Imports CTM move functions from ipeps module to avoid circular imports.
    """
    from tenax.algorithms.ipeps import (
        _build_double_layer,
        _ctm_bottom_move,
        _ctm_left_move,
        _ctm_right_move,
        _ctm_top_move,
        _renormalize_env,
    )

    a = _build_double_layer(A)
    if a.ndim == 8:
        D = a.shape[0]
        a = a.reshape(D**2, D**2, D**2, D**2)

    chi = config.chi
    pm = config.projector_method
    env = _ctm_left_move(env, a, chi, pm)
    env = _ctm_right_move(env, a, chi, pm)
    env = _ctm_top_move(env, a, chi, pm)
    env = _ctm_bottom_move(env, a, chi, pm)

    if config.renormalize:
        env = _renormalize_env(env)

    return env


def _env_to_flat(env: CTMEnvironment) -> jax.Array:
    """Flatten CTMEnvironment into a single 1-D array."""
    arrays = [t.ravel() for t in env]
    return jnp.concatenate(arrays)


def _flat_to_env(flat: jax.Array, env_template: CTMEnvironment) -> CTMEnvironment:
    """Reconstruct CTMEnvironment from a flat array using template shapes."""
    from tenax.algorithms.ipeps import CTMEnvironment as CTMEnv

    arrays = []
    offset = 0
    for t in env_template:
        size = t.size
        arrays.append(flat[offset : offset + size].reshape(t.shape))
        offset += size
    return CTMEnv(*arrays)


def _gauge_fix_ctm(env: CTMEnvironment) -> CTMEnvironment:
    """Fix gauge of CTM tensors via QR decomposition of corners.

    Ensures unique element-wise convergence needed for fixed-point
    implicit differentiation (Francuz et al. PRR 7, 013237).

    Each corner C is replaced by R from its QR decomposition (C = Q R),
    and the corresponding Q factors are absorbed into the adjacent edge
    tensors. This removes the gauge freedom in the CTM environment.
    """
    from tenax.algorithms.ipeps import CTMEnvironment as CTMEnv

    C1, C2, C3, C4, T1, T2, T3, T4 = env

    # C1 = Q1 @ R1 -> C1_new = R1, T1_new = Q1^H @ T1, T4_new = Q1^H @ T4
    Q1, R1 = jnp.linalg.qr(C1)
    C1_new = R1
    # Absorb Q1^H into top edge (T1's left leg) and left edge (T4's left leg)
    T1_new = jnp.einsum("ab,bdc->adc", Q1.conj().T, T1)
    T4_new = jnp.einsum("ab,bdc->adc", Q1.conj().T, T4)

    # C2 = Q2 @ R2 -> C2_new = R2
    Q2, R2 = jnp.linalg.qr(C2)
    C2_new = R2
    T1_new = jnp.einsum("adb,bc->adc", T1_new, Q2)
    T2_new = jnp.einsum("ab,bdc->adc", Q2.conj().T, T2)

    # C3 = Q3 @ R3 -> C3_new = R3
    Q3, R3 = jnp.linalg.qr(C3)
    C3_new = R3
    T2_new = jnp.einsum("adb,bc->adc", T2_new, Q3)
    T3_new = jnp.einsum("adb,bc->adc", T3, Q3)

    # C4 = Q4 @ R4 -> C4_new = R4
    Q4, R4 = jnp.linalg.qr(C4)
    C4_new = R4
    T3_new = jnp.einsum("ab,bdc->adc", Q4.conj().T, T3_new)
    T4_new = jnp.einsum("adb,bc->adc", T4_new, Q4)

    return CTMEnv(C1_new, C2_new, C3_new, C4_new, T1_new, T2_new, T3_new, T4_new)


def ctm_fixed_point(
    A: jax.Array,
    config: CTMConfig,
    initial_env: CTMEnvironment | None = None,
) -> CTMEnvironment:
    """CTM with implicit differentiation at fixed point.

    Forward: run CTM to convergence (standard iteration).
    Backward: solve ``(I - J^T) lambda = v`` for the VJP via fixed-point
    iteration of the transpose CTM Jacobian.

    This avoids storing all intermediate CTM iterations and gives exact
    gradients at convergence.

    Args:
        A:           PEPS site tensor of shape ``(D, D, D, D, d)``.
        config:      CTMConfig.
        initial_env: Optional warm-start environment.

    Returns:
        Converged CTMEnvironment.
    """
    return _ctm_fixed_point_impl(A, config, initial_env)


def _ctm_fixed_point_impl(
    A: jax.Array,
    config: CTMConfig,
    initial_env: CTMEnvironment | None = None,
) -> CTMEnvironment:
    """Implementation of CTM with custom VJP for implicit differentiation."""
    from tenax.algorithms.ipeps import (
        _build_double_layer,
        _initialize_ctm_env,
    )

    a = _build_double_layer(A)
    if a.ndim == 8:
        D_phys = a.shape[0]
        a = a.reshape(D_phys**2, D_phys**2, D_phys**2, D_phys**2)

    if initial_env is not None:
        env = initial_env
    else:
        env = _initialize_ctm_env(a, config.chi)

    # Run CTM to convergence
    prev_sv = None
    for _ in range(config.max_iter):
        env = _ctm_step(A, env, config)
        env = _gauge_fix_ctm(env)

        current_sv = jnp.linalg.svd(env.C1, compute_uv=False)
        if prev_sv is not None:
            sv1 = current_sv / (jnp.sum(current_sv) + 1e-15)
            sv2 = prev_sv / (jnp.sum(prev_sv) + 1e-15)
            min_len = min(len(sv1), len(sv2))
            diff = float(jnp.max(jnp.abs(sv1[:min_len] - sv2[:min_len])))
            if diff < config.conv_tol:
                break
        prev_sv = current_sv

    return env


def _config_from_tuple(config_tuple: tuple):
    """Reconstruct CTMConfig from a packed tuple."""
    from tenax.algorithms.ipeps import CTMConfig

    pm_map = {0: "eigh", 1: "qr"}
    pm_int = config_tuple[4] if len(config_tuple) > 4 else 0
    return CTMConfig(
        chi=config_tuple[0],
        max_iter=config_tuple[1],
        conv_tol=config_tuple[2],
        renormalize=bool(config_tuple[3]),
        projector_method=pm_map.get(pm_int, "eigh"),
    )


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def ctm_converge(A: jax.Array, config_tuple: tuple) -> tuple[jax.Array, ...]:
    """CTM convergence with custom VJP for implicit differentiation.

    Wraps the CTM iteration as a differentiable function
    ``A -> env_flat`` with implicit fixed-point backward pass.

    Args:
        A:            PEPS site tensor.
        config_tuple: CTMConfig fields packed as tuple for JAX tracing.
                      ``(chi, max_iter, conv_tol, renormalize[, projector_method_int])``.

    Returns:
        Flat tuple of environment tensors (C1, C2, ..., T4).
    """
    config = _config_from_tuple(config_tuple)
    env = _ctm_fixed_point_impl(A, config)
    return tuple(env)


def _ctm_converge_fwd(
    A: jax.Array,
    config_tuple: tuple,
) -> tuple[tuple[jax.Array, ...], tuple]:
    """Forward pass — run CTM, cache result for backward."""
    config = _config_from_tuple(config_tuple)
    env = _ctm_fixed_point_impl(A, config)
    env_tuple = tuple(env)
    residuals = (A, env_tuple)
    return env_tuple, residuals


def _ctm_converge_bwd(
    config_tuple: tuple,
    residuals: tuple,
    g: tuple[jax.Array, ...],
) -> tuple[jax.Array]:
    """Backward pass via implicit differentiation of CTM fixed point.

    Solves ``(I - J^T) lambda = g`` using GMRES (Francuz et al. PRR 7,
    013237), where J = d(ctm_step)/d(env) is the Jacobian of one CTM step.
    Then ``dA = d(ctm_step)/dA^T @ lambda``.
    """
    from tenax.algorithms.ipeps import CTMEnvironment

    A, env_tuple = residuals
    config = _config_from_tuple(config_tuple)

    # g is a tuple of cotangents for (C1, C2, ..., T4)

    # Define one CTM step as function of (A, env_flat)
    def step_fn(A_in, env_in_tuple):
        env_in = CTMEnvironment(*env_in_tuple)
        env_out = _ctm_step(A_in, env_in, config)
        env_out = _gauge_fix_ctm(env_out)
        return tuple(env_out)

    # Solve (I - J_env^T) lambda = g via GMRES (Francuz et al. PRR 7, 013237).
    # GMRES converges superlinearly (Krylov acceleration) and directly
    # monitors the residual norm, unlike the Neumann series which converges
    # only geometrically with rate equal to the spectral radius.
    from jax.scipy.sparse.linalg import gmres as jax_gmres

    def apply_I_minus_Jt(v):
        """Apply (I - J_env^T) to vector v (a tuple of arrays)."""
        _, vjp_fn = jax.vjp(lambda e: step_fn(A, e), env_tuple)
        Jt_v = vjp_fn(v)[0]
        return tuple(vi - ji for vi, ji in zip(v, Jt_v))

    max_fp_iter = min(config.max_iter, 50)
    lam, info = jax_gmres(
        apply_I_minus_Jt,
        g,
        x0=g,
        tol=config.conv_tol,
        maxiter=max_fp_iter,
    )

    # Now compute dA = d(step)/dA^T @ lambda
    _, vjp_A_fn = jax.vjp(lambda a: step_fn(a, env_tuple), A)
    dA = vjp_A_fn(lam)[0]

    return (dA,)


ctm_converge.defvjp(_ctm_converge_fwd, _ctm_converge_bwd)


# ---------------------------------------------------------------------------
# 4. 2-site CTM fixed-point implicit differentiation
# ---------------------------------------------------------------------------


def _ctm_step_2site(
    A: jax.Array,
    B: jax.Array,
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    config: CTMConfig,
) -> tuple[CTMEnvironment, CTMEnvironment]:
    """One full 2-site CTM iteration + gauge fixing.

    Imports 2-site CTM sweep from ipeps module to avoid circular imports.
    """
    from tenax.algorithms.ipeps import (
        _build_double_layer,
        _ctm_2site_sweep,
    )

    a_A = _build_double_layer(A)
    a_B = _build_double_layer(B)
    if a_A.ndim == 8:
        D = a_A.shape[0]
        a_A = a_A.reshape(D**2, D**2, D**2, D**2)
    if a_B.ndim == 8:
        D = a_B.shape[0]
        a_B = a_B.reshape(D**2, D**2, D**2, D**2)

    env_A, env_B = _ctm_2site_sweep(
        env_A, env_B, a_A, a_B, config.chi, config.renormalize
    )
    return env_A, env_B


def _ctm_2site_fixed_point_impl(
    A: jax.Array,
    B: jax.Array,
    config: CTMConfig,
) -> tuple[CTMEnvironment, CTMEnvironment]:
    """Run 2-site CTM to convergence with gauge fixing."""
    from tenax.algorithms.ipeps import (
        _build_double_layer,
        _initialize_ctm_env,
    )

    a_A = _build_double_layer(A)
    a_B = _build_double_layer(B)
    if a_A.ndim == 8:
        D = a_A.shape[0]
        a_A = a_A.reshape(D**2, D**2, D**2, D**2)
    if a_B.ndim == 8:
        D = a_B.shape[0]
        a_B = a_B.reshape(D**2, D**2, D**2, D**2)

    env_A = _initialize_ctm_env(a_A, config.chi)
    env_B = _initialize_ctm_env(a_B, config.chi)

    prev_sv_A = None
    prev_sv_B = None
    for _ in range(config.max_iter):
        env_A, env_B = _ctm_step_2site(A, B, env_A, env_B, config)
        env_A = _gauge_fix_ctm(env_A)
        env_B = _gauge_fix_ctm(env_B)

        sv_A = jnp.linalg.svd(env_A.C1, compute_uv=False)
        sv_B = jnp.linalg.svd(env_B.C1, compute_uv=False)

        if prev_sv_A is not None:
            sv1_A = sv_A / (jnp.sum(sv_A) + 1e-15)
            sv2_A = prev_sv_A / (jnp.sum(prev_sv_A) + 1e-15)
            sv1_B = sv_B / (jnp.sum(sv_B) + 1e-15)
            sv2_B = prev_sv_B / (jnp.sum(prev_sv_B) + 1e-15)
            min_len_A = min(len(sv1_A), len(sv2_A))
            min_len_B = min(len(sv1_B), len(sv2_B))
            diff_A = float(jnp.max(jnp.abs(sv1_A[:min_len_A] - sv2_A[:min_len_A])))
            diff_B = float(jnp.max(jnp.abs(sv1_B[:min_len_B] - sv2_B[:min_len_B])))
            if max(diff_A, diff_B) < config.conv_tol:
                break
        prev_sv_A = sv_A
        prev_sv_B = sv_B

    return env_A, env_B


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def ctm_converge_2site(
    A: jax.Array,
    B: jax.Array,
    config_tuple: tuple,
) -> tuple[jax.Array, ...]:
    """2-site CTM convergence with custom VJP for implicit differentiation.

    Args:
        A:            Site tensor for sublattice A.
        B:            Site tensor for sublattice B.
        config_tuple: CTMConfig fields packed as tuple.

    Returns:
        Flat tuple ``(*env_A_tensors, *env_B_tensors)`` (16 arrays total).
    """
    config = _config_from_tuple(config_tuple)
    env_A, env_B = _ctm_2site_fixed_point_impl(A, B, config)
    return tuple(env_A) + tuple(env_B)


def _ctm_converge_2site_fwd(
    A: jax.Array,
    B: jax.Array,
    config_tuple: tuple,
) -> tuple[tuple[jax.Array, ...], tuple]:
    """Forward pass — run 2-site CTM, cache result."""
    config = _config_from_tuple(config_tuple)
    env_A, env_B = _ctm_2site_fixed_point_impl(A, B, config)
    out = tuple(env_A) + tuple(env_B)
    residuals = (A, B, out)
    return out, residuals


def _ctm_converge_2site_bwd(
    config_tuple: tuple,
    residuals: tuple,
    g: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array]:
    """Backward pass via implicit differentiation of 2-site CTM fixed point.

    Solves ``(I - J^T) lambda = g`` using GMRES where the state vector
    spans both sublattice environments ``(env_A, env_B)``.
    """
    from tenax.algorithms.ipeps import CTMEnvironment

    A, B, env_flat = residuals
    config = _config_from_tuple(config_tuple)

    # Split env_flat and g into A and B parts
    env_A_tuple = env_flat[:8]
    env_B_tuple = env_flat[8:]
    g_A = g[:8]
    g_B = g[8:]

    def step_fn(A_in, B_in, env_combined):
        eA = CTMEnvironment(*env_combined[:8])
        eB = CTMEnvironment(*env_combined[8:])
        eA_out, eB_out = _ctm_step_2site(A_in, B_in, eA, eB, config)
        eA_out = _gauge_fix_ctm(eA_out)
        eB_out = _gauge_fix_ctm(eB_out)
        return tuple(eA_out) + tuple(eB_out)

    env_combined = env_A_tuple + env_B_tuple
    g_combined = g_A + g_B

    from jax.scipy.sparse.linalg import gmres as jax_gmres

    def apply_I_minus_Jt(v):
        _, vjp_fn = jax.vjp(lambda e: step_fn(A, B, e), env_combined)
        Jt_v = vjp_fn(v)[0]
        return tuple(vi - ji for vi, ji in zip(v, Jt_v))

    max_fp_iter = min(config.max_iter, 50)
    lam, info = jax_gmres(
        apply_I_minus_Jt,
        g_combined,
        x0=g_combined,
        tol=config.conv_tol,
        maxiter=max_fp_iter,
    )

    # Compute dA and dB
    _, vjp_AB_fn = jax.vjp(lambda a, b: step_fn(a, b, env_combined), A, B)
    dA, dB = vjp_AB_fn(lam)

    return (dA, dB)


ctm_converge_2site.defvjp(_ctm_converge_2site_fwd, _ctm_converge_2site_bwd)
