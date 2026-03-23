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
from tenax.contraction.contractor import contract

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
    )


def _config_from_tuple(config_tuple: tuple):
    """Reconstruct CTMConfig from a packed tuple."""
    pm_int = config_tuple[4] if len(config_tuple) > 4 else 0
    return CTMConfig(
        chi=config_tuple[0],
        max_iter=config_tuple[1],
        conv_tol=config_tuple[2],
        renormalize=bool(config_tuple[3]),
        projector_method=_PM_INT_TO_STR.get(pm_int, "eigh"),
    )


# ---------------------------------------------------------------------------
# 3. Standard CTM (Tensor protocol) fixed-point implicit differentiation
# ---------------------------------------------------------------------------


def _gauge_fix_ctm_tensor(env):
    """Fix gauge of CTMTensorEnv via QR decomposition of corners.

    Uses block-sparse ``tenax.linalg.qr`` directly on Tensor objects,
    avoiding ``todense()``/``from_dense()`` round-trips.  This preserves
    sparsity and gives cleaner gradients during AD.

    Each corner C is QR-decomposed; Q is absorbed into the two adjacent
    edge tensors, and C is replaced by R.
    """
    from tenax.linalg import qr as tensor_qr

    # C1(c1_d, c1_r) = Q1(c1_d, q1) @ R1(q1, c1_r)
    # C1_new = R1, absorb Q1^bar into T1's left (t1_l) and T4's top (t4_d)
    Q1, R1 = tensor_qr(
        env.C1, left_labels=["c1_d"], right_labels=["c1_r"], new_bond_label="q1"
    )
    C1_new = R1.relabel("q1", "c1_d")
    # Q1^bar: (q1_OUT, c1_d_IN) — relabel c1_d to match edge's chi label
    Q1b = Q1.bar()
    T1_new = contract(Q1b.relabel("c1_d", "t1_l"), env.T1)  # (q1, u2, t1_r)
    T1_new = T1_new.relabel("q1", "t1_l")  # restore label
    T4_new = contract(Q1b.relabel("c1_d", "t4_d"), env.T4)  # (q1, l2, t4_u)
    T4_new = T4_new.relabel("q1", "t4_d")  # restore label

    # C2(c2_l, c2_d) = Q2(c2_l, q2) @ R2(q2, c2_d)
    # C2_new = R2, absorb Q2 into T1's right (t1_r) and Q2^bar into T2's top (t2_u)
    Q2, R2 = tensor_qr(
        env.C2, left_labels=["c2_l"], right_labels=["c2_d"], new_bond_label="q2"
    )
    C2_new = R2.relabel("q2", "c2_l")
    T1_new = contract(T1_new, Q2.relabel("c2_l", "t1_r"))  # (t1_l, u2, q2)
    T1_new = T1_new.relabel("q2", "t1_r")  # restore label
    Q2b = Q2.bar()
    T2_new = contract(Q2b.relabel("c2_l", "t2_u"), env.T2)  # (q2, r2, t2_d)
    T2_new = T2_new.relabel("q2", "t2_u")  # restore label

    # C3(c3_u, c3_l) = Q3(c3_u, q3) @ R3(q3, c3_l)
    # C3_new = R3, absorb Q3 into T2's bottom (t2_d) and T3's right (t3_r)
    Q3, R3 = tensor_qr(
        env.C3, left_labels=["c3_u"], right_labels=["c3_l"], new_bond_label="q3"
    )
    C3_new = R3.relabel("q3", "c3_u")
    T2_new = contract(T2_new, Q3.relabel("c3_u", "t2_d"))  # (t2_u, r2, q3)
    T2_new = T2_new.relabel("q3", "t2_d")  # restore label
    T3_new = contract(env.T3, Q3.relabel("c3_u", "t3_r"))  # (t3_r→q3 side)
    T3_new = T3_new.relabel("q3", "t3_r")  # restore label

    # C4(c4_r, c4_u) = Q4(c4_r, q4) @ R4(q4, c4_u)
    # C4_new = R4, absorb Q4^bar into T3's left (t3_l) and Q4 into T4's bottom (t4_u)
    Q4, R4 = tensor_qr(
        env.C4, left_labels=["c4_r"], right_labels=["c4_u"], new_bond_label="q4"
    )
    C4_new = R4.relabel("q4", "c4_r")
    Q4b = Q4.bar()
    T3_new = contract(Q4b.relabel("c4_r", "t3_l"), T3_new)  # (q4, d2, t3_r)
    T3_new = T3_new.relabel("q4", "t3_l")  # restore label
    T4_new = contract(T4_new, Q4.relabel("c4_r", "t4_u"))  # (t4_d, l2, q4)
    T4_new = T4_new.relabel("q4", "t4_u")  # restore label

    return CTMTensorEnv(
        C1=C1_new,
        C2=C2_new,
        C3=C3_new,
        C4=C4_new,
        T1=T1_new,
        T2=T2_new,
        T3=T3_new,
        T4=T4_new,
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


def _ctm_tensor_fixed_point_impl(A, config):
    """Run standard Tensor-protocol CTM to convergence with gauge fixing."""
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, config.chi)
    sweep_fn = _ctm_tensor_sweep_paired if _needs_paired_sweep(A) else _ctm_tensor_sweep

    prev_sv = None
    for _ in range(config.max_iter):
        env = sweep_fn(env, a, config.chi, config.renormalize, config.projector_method)
        env = _gauge_fix_ctm_tensor(env)

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


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def ctm_tensor_converge(
    A,
    config_tuple: tuple,
) -> tuple[jax.Array, ...]:
    """Standard Tensor-protocol CTM with implicit differentiation.

    Args:
        A:            iPEPS site tensor (DenseTensor or SymmetricTensor).
        config_tuple: CTMConfig fields packed as tuple for JAX tracing.

    Returns:
        Flat tuple of environment pytree leaf arrays.
    """
    config = _config_from_tuple(config_tuple)
    env = _ctm_tensor_fixed_point_impl(A, config)
    return tuple(jax.tree.leaves(env))


def _ctm_tensor_converge_fwd(A, config_tuple):
    """Forward pass — run Tensor CTM, cache A and env for backward."""
    config = _config_from_tuple(config_tuple)
    env = _ctm_tensor_fixed_point_impl(A, config)
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

    return (dA,)


ctm_tensor_converge.defvjp(_ctm_tensor_converge_fwd, _ctm_tensor_converge_bwd)


# ---------------------------------------------------------------------------
# 4. 2-site Tensor-protocol CTM fixed-point implicit differentiation
# ---------------------------------------------------------------------------


def _ctm_tensor_multisite_fixed_point(site_tensors, neighbors, config):
    """Run multisite Tensor-protocol CTM to convergence with gauge fixing."""
    double_layers = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
    envs = {
        c: initialize_ctm_tensor_env(A, config.chi) for c, A in site_tensors.items()
    }

    prev_svs = {}
    for _ in range(config.max_iter):
        envs = _ctm_tensor_sweep_multisite(
            envs,
            double_layers,
            neighbors,
            config.chi,
            config.renormalize,
            config.projector_method,
        )
        envs = {c: _gauge_fix_ctm_tensor(e) for c, e in envs.items()}

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


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def ctm_tensor_converge_2site(
    A,
    B,
    config_tuple: tuple,
) -> tuple[jax.Array, ...]:
    """2-site Tensor-protocol CTM with implicit differentiation.

    Args:
        A:            iPEPS site tensor A (DenseTensor or SymmetricTensor).
        B:            iPEPS site tensor B.
        config_tuple: CTMConfig fields packed as tuple for JAX tracing.

    Returns:
        Flat tuple ``(*env_A_leaves, *env_B_leaves)``.
    """
    config = _config_from_tuple(config_tuple)
    envs = _ctm_tensor_multisite_fixed_point(
        {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, config
    )
    return tuple(jax.tree.leaves(envs[(0, 0)])) + tuple(jax.tree.leaves(envs[(1, 0)]))


def _ctm_tensor_converge_2site_fwd(A, B, config_tuple):
    """Forward pass — run 2-site Tensor CTM, cache A, B, envs."""
    config = _config_from_tuple(config_tuple)
    envs = _ctm_tensor_multisite_fixed_point(
        {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, config
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

    return (dA, dB)


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
