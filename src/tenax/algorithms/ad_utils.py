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

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres as jax_gmres

from tenax.algorithms._ctm_tensor import (
    CHECKERBOARD_NEIGHBORS,
    CTMTensorEnv,
    _build_double_layer_tensor,
    _ctm_tensor_sweep,
    _ctm_tensor_sweep_multisite,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor import (
    _ctm_sv_diff as _ctm_sv_diff_tensor,
)
from tenax.algorithms._ctm_tensor_convergence import (
    _ctm_tensor_sweep_paired,
)
from tenax.algorithms._split_ctm_tensor import (
    _split_ctm_tensor_sweep,
    ctm_split_tensor,
)
from tenax.algorithms.ipeps_config import CTMConfig

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
# 2. Config tuple helpers (shared by all CTM AD paths)
# ---------------------------------------------------------------------------


_PM_STR_TO_INT = {"eigh": 0, "qr": 1}
_PM_INT_TO_STR = {0: "eigh", 1: "qr"}


def _config_to_tuple(config) -> tuple:
    """Pack CTMConfig into a hashable tuple for JAX tracing."""
    return (
        config.chi,
        config.max_iter,
        config.conv_tol,
        int(config.renormalize),
        _PM_STR_TO_INT.get(config.projector_method, 0),
        config.min_iter,
    )


def _config_from_tuple(config_tuple: tuple):
    """Reconstruct CTMConfig from a packed tuple."""
    pm_int = config_tuple[4] if len(config_tuple) > 4 else 0
    min_iter = config_tuple[5] if len(config_tuple) > 5 else 10
    return CTMConfig(
        chi=config_tuple[0],
        max_iter=config_tuple[1],
        conv_tol=config_tuple[2],
        renormalize=bool(config_tuple[3]),
        projector_method=_PM_INT_TO_STR.get(pm_int, "eigh"),
        min_iter=min_iter,
    )


# ---------------------------------------------------------------------------
# 3. Standard CTM (Tensor protocol) fixed-point implicit differentiation
# ---------------------------------------------------------------------------


def _gauge_fix_ctm_tensor(env):
    """Fix gauge of CTMTensorEnv via QR decomposition of corners.

    Performs dense QR on corner and edge arrays, then wraps results back
    into Tensor objects preserving the original index structure.  All
    dense operations (``todense()``, ``jnp.linalg.qr``, ``jnp.einsum``)
    are JAX-differentiable.

    For SymmetricTensor with trivial charges (all zeros), the dense
    round-trip is cheap (single block).  For non-trivial charges, the
    ``from_dense(..., tol=inf)`` wrapping preserves the charge layout.
    """
    from tenax.core.tensor import SymmetricTensor

    # Extract dense arrays — for SymmetricTensor with trivial charges
    # this is essentially free (single block covers the full tensor).
    C1, C2, C3, C4 = (c.todense() for c in (env.C1, env.C2, env.C3, env.C4))
    T1, T2, T3, T4 = (t.todense() for t in (env.T1, env.T2, env.T3, env.T4))

    # C1 = Q1 @ R1 → C1_new = R1, absorb Q1^H into T1 (left) and T4 (left)
    Q1, R1 = jnp.linalg.qr(C1)
    C1_new = R1
    T1_new = jnp.einsum("ab,bdc->adc", Q1.conj().T, T1)
    T4_new = jnp.einsum("ab,bdc->adc", Q1.conj().T, T4)

    # C2 = Q2 @ R2 → C2_new = R2, absorb Q2 into T1 (right) and Q2^H into T2 (top)
    Q2, R2 = jnp.linalg.qr(C2)
    C2_new = R2
    T1_new = jnp.einsum("adb,bc->adc", T1_new, Q2)
    T2_new = jnp.einsum("ab,bdc->adc", Q2.conj().T, T2)

    # C3 = Q3 @ R3 → C3_new = R3, absorb Q3 into T2 (bottom) and T3 (right)
    Q3, R3 = jnp.linalg.qr(C3)
    C3_new = R3
    T2_new = jnp.einsum("adb,bc->adc", T2_new, Q3)
    T3_new = jnp.einsum("adb,bc->adc", T3, Q3)

    # C4 = Q4 @ R4 → C4_new = R4, absorb Q4^H into T3 (left) and Q4 into T4 (bottom)
    Q4, R4 = jnp.linalg.qr(C4)
    C4_new = R4
    T3_new = jnp.einsum("ab,bdc->adc", Q4.conj().T, T3_new)
    T4_new = jnp.einsum("adb,bc->adc", T4_new, Q4)

    # Wrap back into Tensor objects preserving original index structure
    def _wrap(data, original):
        if isinstance(original, SymmetricTensor):
            return SymmetricTensor.from_dense(data, original.indices, tol=float("inf"))
        return type(original)(data, original.indices)

    return CTMTensorEnv(
        C1=_wrap(C1_new, env.C1),
        C2=_wrap(C2_new, env.C2),
        C3=_wrap(C3_new, env.C3),
        C4=_wrap(C4_new, env.C4),
        T1=_wrap(T1_new, env.T1),
        T2=_wrap(T2_new, env.T2),
        T3=_wrap(T3_new, env.T3),
        T4=_wrap(T4_new, env.T4),
    )


def _needs_paired_sweep(A) -> bool:
    """Check if A is a SymmetricTensor with non-trivial virtual charges."""
    from tenax.core.tensor import SymmetricTensor

    if not isinstance(A, SymmetricTensor):
        return False
    import numpy as _np

    virtual_charges = [_np.sort(A.indices[i].charges) for i in range(4)]
    has_nontrivial = any(not _np.all(vc == 0) for vc in virtual_charges)
    all_same = all(
        _np.array_equal(virtual_charges[0], virtual_charges[i]) for i in range(1, 4)
    )
    return has_nontrivial and all_same


def _ctm_tensor_step(
    A_leaves: tuple[jax.Array, ...],
    env_leaves: tuple[jax.Array, ...],
    chi: int,
    renormalize: bool,
    projector_method: str,
    A_treedef,
    env_treedef,
    use_paired: bool = False,
) -> tuple[jax.Array, ...]:
    """One CTM tensor sweep + gauge fix, mapping flat leaves to flat leaves."""
    A = jax.tree.unflatten(A_treedef, A_leaves)
    env = jax.tree.unflatten(env_treedef, list(env_leaves))

    a = _build_double_layer_tensor(A)
    sweep_fn = _ctm_tensor_sweep_paired if use_paired else _ctm_tensor_sweep
    env_new = sweep_fn(env, a, chi, renormalize, projector_method)
    env_new = _gauge_fix_ctm_tensor(env_new)

    return tuple(jax.tree.leaves(env_new))


def _ctm_tensor_fixed_point_impl(A, config, env_init=None):
    """Run standard Tensor-protocol CTM to convergence with gauge fixing."""
    a = _build_double_layer_tensor(A)
    env = env_init if env_init is not None else initialize_ctm_tensor_env(A, config.chi)
    sweep_fn = _ctm_tensor_sweep_paired if _needs_paired_sweep(A) else _ctm_tensor_sweep

    prev_sv = None
    for i in range(config.max_iter):
        env = sweep_fn(env, a, config.chi, config.renormalize, config.projector_method)
        env = _gauge_fix_ctm_tensor(env)

        if i + 1 < config.min_iter:
            continue

        current_sv = jnp.linalg.svd(env.C1.todense(), compute_uv=False)
        if prev_sv is not None:
            diff = _ctm_sv_diff_local(current_sv, prev_sv)
            if float(diff) < config.conv_tol:
                break
        prev_sv = current_sv

    return env


def _ctm_sv_diff_local(sv_new, sv_old):
    """Compute max abs diff between normalized SVs."""
    return _ctm_sv_diff_tensor(sv_new, sv_old)


def _unflatten_env_init(env_init_leaves, A, chi):
    """Unflatten env_init_leaves into a CTMTensorEnv, or return None."""
    if env_init_leaves is None:
        return None
    template = initialize_ctm_tensor_env(A, chi)
    env_treedef = jax.tree.structure(template)
    return jax.tree.unflatten(env_treedef, list(env_init_leaves))


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def ctm_tensor_converge(
    A,
    env_init_leaves,
    config_tuple: tuple,
) -> tuple[jax.Array, ...]:
    """Standard Tensor-protocol CTM with implicit differentiation.

    Args:
        A:              iPEPS site tensor (DenseTensor or SymmetricTensor).
        env_init_leaves: Flat tuple of env leaf arrays for warm-start, or None.
        config_tuple:   CTMConfig fields packed as tuple for JAX tracing.

    Returns:
        Flat tuple of environment pytree leaf arrays.
    """
    config = _config_from_tuple(config_tuple)
    env_init = _unflatten_env_init(env_init_leaves, A, config.chi)
    env = _ctm_tensor_fixed_point_impl(A, config, env_init=env_init)
    return tuple(jax.tree.leaves(env))


def _ctm_tensor_converge_fwd(A, env_init_leaves, config_tuple):
    """Forward pass — run Tensor CTM, cache A and env for backward."""
    config = _config_from_tuple(config_tuple)
    env_init = _unflatten_env_init(env_init_leaves, A, config.chi)
    env = _ctm_tensor_fixed_point_impl(A, config, env_init=env_init)
    env_leaves = tuple(jax.tree.leaves(env))
    residuals = (A, env)
    return env_leaves, residuals


def _ctm_tensor_converge_bwd(config_tuple, residuals, g):
    """Backward pass via implicit differentiation of Tensor CTM fixed point."""
    A, env = residuals
    config = _config_from_tuple(config_tuple)

    A_treedef = jax.tree.structure(A)
    env_treedef = jax.tree.structure(env)
    env_leaves = tuple(jax.tree.leaves(env))

    paired = _needs_paired_sweep(A)

    def step_fn(A_in, env_in_leaves):
        return _ctm_tensor_step(
            tuple(jax.tree.leaves(A_in)),
            env_in_leaves,
            config.chi,
            config.renormalize,
            config.projector_method,
            A_treedef,
            env_treedef,
            use_paired=paired,
        )

    def apply_I_minus_Jt(v):
        _, vjp_fn = jax.vjp(lambda e: step_fn(A, e), env_leaves)
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

    _, vjp_A_fn = jax.vjp(lambda a: step_fn(a, env_leaves), A)
    dA = vjp_A_fn(lam)[0]

    # Zero gradient for env_init — it's an initialization hint only
    d_env_init = tuple(jnp.zeros_like(x) for x in env_leaves)
    return (dA, d_env_init)


ctm_tensor_converge.defvjp(_ctm_tensor_converge_fwd, _ctm_tensor_converge_bwd)


# ---------------------------------------------------------------------------
# 4. 2-site Tensor-protocol CTM fixed-point implicit differentiation
# ---------------------------------------------------------------------------


def _ctm_tensor_multisite_fixed_point(site_tensors, neighbors, config, envs_init=None):
    """Run multisite Tensor-protocol CTM to convergence with gauge fixing."""
    double_layers = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
    envs = (
        envs_init
        if envs_init is not None
        else {
            c: initialize_ctm_tensor_env(A, config.chi) for c, A in site_tensors.items()
        }
    )

    prev_svs = {}
    for i in range(config.max_iter):
        envs = _ctm_tensor_sweep_multisite(
            envs,
            double_layers,
            neighbors,
            config.chi,
            config.renormalize,
            config.projector_method,
        )
        envs = {c: _gauge_fix_ctm_tensor(e) for c, e in envs.items()}

        if i + 1 < config.min_iter:
            continue

        converged = True
        for c in sorted(envs):
            sv = jnp.linalg.svd(envs[c].C1.todense(), compute_uv=False)
            if c in prev_svs:
                if float(_ctm_sv_diff_local(sv, prev_svs[c])) >= config.conv_tol:
                    converged = False
                    prev_svs[c] = sv
                    break
            else:
                converged = False
            prev_svs[c] = sv
        if converged:
            break

    return envs


def _ctm_tensor_step_2site(
    A_leaves,
    B_leaves,
    env_leaves,
    chi,
    renormalize,
    projector_method,
    A_treedef,
    B_treedef,
    env_A_treedef,
    n_env_A_leaves,
    double_layers=None,
):
    """One 2-site CTM tensor sweep + gauge fix, flat leaves → flat leaves.

    If *double_layers* is provided, it is used directly (avoids redundant
    recomputation when A/B are constant, e.g. in the GMRES backward pass).
    """
    A = jax.tree.unflatten(A_treedef, A_leaves)
    B = jax.tree.unflatten(B_treedef, B_leaves)
    env_A = jax.tree.unflatten(env_A_treedef, list(env_leaves[:n_env_A_leaves]))
    env_B = jax.tree.unflatten(env_A_treedef, list(env_leaves[n_env_A_leaves:]))

    if double_layers is None:
        double_layers = {
            (0, 0): _build_double_layer_tensor(A),
            (1, 0): _build_double_layer_tensor(B),
        }
    envs = {(0, 0): env_A, (1, 0): env_B}
    envs = _ctm_tensor_sweep_multisite(
        envs, double_layers, CHECKERBOARD_NEIGHBORS, chi, renormalize, projector_method
    )
    envs = {c: _gauge_fix_ctm_tensor(e) for c, e in envs.items()}

    return tuple(jax.tree.leaves(envs[(0, 0)])) + tuple(jax.tree.leaves(envs[(1, 0)]))


def _unflatten_2site_env_init(A, B, env_init_leaves, chi):
    """Unflatten 2-site env_init_leaves, or return None."""
    if env_init_leaves is None:
        return None
    template = initialize_ctm_tensor_env(A, chi)
    treedef = jax.tree.structure(template)
    n = len(jax.tree.leaves(template))
    env_A = jax.tree.unflatten(treedef, list(env_init_leaves[:n]))
    env_B = jax.tree.unflatten(treedef, list(env_init_leaves[n:]))
    return {(0, 0): env_A, (1, 0): env_B}


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def ctm_tensor_converge_2site(
    A,
    B,
    env_init_leaves,
    config_tuple: tuple,
) -> tuple[jax.Array, ...]:
    """2-site Tensor-protocol CTM with implicit differentiation.

    Args:
        A:              iPEPS site tensor A (DenseTensor or SymmetricTensor).
        B:              iPEPS site tensor B.
        env_init_leaves: Flat tuple of env leaf arrays for warm-start, or None.
        config_tuple:   CTMConfig fields packed as tuple for JAX tracing.

    Returns:
        Flat tuple ``(*env_A_leaves, *env_B_leaves)``.
    """
    config = _config_from_tuple(config_tuple)
    envs_init = _unflatten_2site_env_init(A, B, env_init_leaves, config.chi)
    envs = _ctm_tensor_multisite_fixed_point(
        {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, config, envs_init=envs_init
    )
    return tuple(jax.tree.leaves(envs[(0, 0)])) + tuple(jax.tree.leaves(envs[(1, 0)]))


def _ctm_tensor_converge_2site_fwd(A, B, env_init_leaves, config_tuple):
    """Forward pass — run 2-site Tensor CTM, cache A, B, envs."""
    config = _config_from_tuple(config_tuple)
    envs_init = _unflatten_2site_env_init(A, B, env_init_leaves, config.chi)
    envs = _ctm_tensor_multisite_fixed_point(
        {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, config, envs_init=envs_init
    )
    env_A, env_B = envs[(0, 0)], envs[(1, 0)]
    out = tuple(jax.tree.leaves(env_A)) + tuple(jax.tree.leaves(env_B))
    residuals = (A, B, env_A, env_B)
    return out, residuals


def _ctm_tensor_converge_2site_bwd(config_tuple, residuals, g):
    """Backward pass via implicit differentiation of 2-site Tensor CTM."""
    A, B, env_A, env_B = residuals
    config = _config_from_tuple(config_tuple)

    A_treedef = jax.tree.structure(A)
    B_treedef = jax.tree.structure(B)
    env_A_treedef = jax.tree.structure(env_A)

    env_A_leaves = tuple(jax.tree.leaves(env_A))
    env_B_leaves = tuple(jax.tree.leaves(env_B))
    n_env_A_leaves = len(env_A_leaves)
    env_leaves = env_A_leaves + env_B_leaves

    # Precompute double layers — A and B are constant during GMRES.
    cached_dls = {
        (0, 0): _build_double_layer_tensor(A),
        (1, 0): _build_double_layer_tensor(B),
    }

    def step_fn(A_in, B_in, env_in_leaves, double_layers=None):
        return _ctm_tensor_step_2site(
            tuple(jax.tree.leaves(A_in)),
            tuple(jax.tree.leaves(B_in)),
            env_in_leaves,
            config.chi,
            config.renormalize,
            config.projector_method,
            A_treedef,
            B_treedef,
            env_A_treedef,
            n_env_A_leaves,
            double_layers=double_layers,
        )

    def apply_I_minus_Jt(v):
        _, vjp_fn = jax.vjp(
            lambda e: step_fn(A, B, e, double_layers=cached_dls), env_leaves
        )
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

    _, vjp_AB_fn = jax.vjp(lambda a, b: step_fn(a, b, env_leaves), A, B)
    dA, dB = vjp_AB_fn(lam)

    # Zero gradient for env_init
    d_env_init = tuple(jnp.zeros_like(x) for x in env_leaves)
    return (dA, dB, d_env_init)


ctm_tensor_converge_2site.defvjp(
    _ctm_tensor_converge_2site_fwd, _ctm_tensor_converge_2site_bwd
)


# ---------------------------------------------------------------------------
# 5. Split CTM (Tensor protocol) fixed-point implicit differentiation
# ---------------------------------------------------------------------------


def _split_ctm_tensor_step(
    A_flat: jax.Array,
    env_tuple: tuple[jax.Array, ...],
    chi: int,
    chi_I: int,
    renormalize: bool,
    A_template,
    env_template,
) -> tuple[jax.Array, ...]:
    """One split-CTM sweep as function of (A_flat, env_flat).

    Reconstructs Tensor objects from flat arrays using templates,
    runs one sweep, and returns the flattened environment.
    """
    # Reconstruct A from flat
    A = jax.tree.unflatten(jax.tree.structure(A_template), (A_flat,))

    # Reconstruct env from tuple of arrays
    env_leaves = list(env_tuple)
    env = jax.tree.unflatten(jax.tree.structure(env_template), env_leaves)

    env_new = _split_ctm_tensor_sweep(env, A, chi, chi_I, renormalize)

    return tuple(jax.tree.leaves(env_new))


def ctm_split_tensor_fixed_point(
    A,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
):
    """Split-CTM with implicit differentiation at fixed point.

    Forward: run split-CTM to convergence.
    Backward: solve ``(I - J^T) lambda = g`` for the VJP via GMRES.

    Args:
        A:          iPEPS site tensor (DenseTensor or SymmetricTensor).
        chi:        Environment bond dimension.
        max_iter:   Maximum CTM iterations.
        conv_tol:   Convergence tolerance.
        chi_I:      Interlayer bond dimension.
        renormalize: Renormalize environment at each step.

    Returns:
        Converged SplitCTMTensorEnv.
    """
    return ctm_split_tensor(A, chi, max_iter, conv_tol, chi_I, renormalize)
