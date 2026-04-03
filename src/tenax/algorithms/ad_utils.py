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
    CTMTensorEnv,
    _build_double_layer_tensor,
    _ctm_tensor_sweep_multisite,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor import (
    _ctm_sv_diff as _ctm_sv_diff_tensor,
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


def _svd_sector_backward(
    U: jax.Array,
    s: jax.Array,
    Vh: jax.Array,
    dU: jax.Array,
    ds: jax.Array,
    dVh: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """Lorentzian-regularized SVD backward for one dense matrix sector.

    Computes the gradient of the input matrix from the gradients of the
    truncated SVD factors, using the Lorentzian broadening from Francuz
    et al., PRR 7, 013237 to regularize degenerate singular values.

    *U*, *s*, *Vh* are the **full** (untruncated) SVD factors of the
    sector matrix.  *dU*, *ds*, *dVh* are the incoming gradients,
    which may be truncated to *k* values (``k <= len(s)``).

    Args:
        U:   Left singular vectors, shape ``(m, p)`` where ``p = min(m, n)``.
        s:   Singular values, shape ``(p,)``.
        Vh:  Right singular vectors, shape ``(p, n)``.
        dU:  Gradient w.r.t. truncated U, shape ``(m, k)``.
        ds:  Gradient w.r.t. truncated s, shape ``(k,)``.
        dVh: Gradient w.r.t. truncated Vh, shape ``(k, n)``.
        eps: Lorentzian broadening parameter.

    Returns:
        dM: Gradient w.r.t. the input matrix, shape ``(m, n)``.
    """
    k = ds.shape[0]
    m = U.shape[0]
    n = Vh.shape[1]

    # Kept subspace
    U_k = U[:, :k]
    s_k = s[:k]
    V_k = Vh[:k, :].conj().T  # (n, k)

    # --- Lorentzian-regularized F-matrix ---
    s2 = s_k**2
    diff = s2[:, None] - s2[None, :]
    F = diff / (diff**2 + eps**2)
    F = F - jnp.diag(jnp.diag(F))

    # Antisymmetric parts of projected cotangents
    UtdU = U_k.conj().T @ dU  # (k, k)
    VtdV = V_k.conj().T @ dVh.conj().T  # (k, k)
    UtdU_anti = 0.5 * (UtdU - UtdU.conj().T)
    VtdV_anti = 0.5 * (VtdV - VtdV.conj().T)

    # Inverse singular values (safe)
    s_inv = jnp.where(s_k > eps, 1.0 / s_k, 0.0)

    # Projectors onto complements of kept subspaces
    proj_U_perp = jnp.eye(m) - U_k @ U_k.conj().T
    proj_V_perp = jnp.eye(n) - V_k @ V_k.conj().T

    # Assemble gradient (Wan & Narayanan 2023 / Francuz et al.):
    Vh_k = Vh[:k, :]
    dM = jnp.zeros((m, n), dtype=U.dtype)

    # 1. Diagonal part from ds
    dM = dM + U_k @ jnp.diag(ds) @ Vh_k

    # 2. Off-diagonal from dU (within kept subspace)
    dM = dM + U_k @ (F * UtdU_anti) @ jnp.diag(s_k) @ Vh_k

    # 3. Off-diagonal from dVh (within kept subspace)
    dM = dM + U_k @ jnp.diag(s_k) @ (F * VtdV_anti) @ Vh_k

    # 4. Truncation correction from dU (kept-truncated coupling)
    dM = dM + proj_U_perp @ dU @ jnp.diag(s_inv) @ Vh_k

    # 5. Truncation correction from dVh (kept-truncated coupling)
    dM = dM + U_k @ jnp.diag(s_inv) @ dVh @ proj_V_perp

    return dM


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
    dM = _svd_sector_backward(U_full, s_full, Vh_full, dU, ds, dVh)
    return (dM,)


truncated_svd_ad.defvjp(_truncated_svd_ad_fwd, _truncated_svd_ad_bwd)


# ---------------------------------------------------------------------------
# 1b. Truncated SVD with stable backward for SymmetricTensor
# ---------------------------------------------------------------------------


def truncated_svd_symmetric_ad(
    M,
    left_labels,
    right_labels,
    chi: int,
    new_bond_label: str = "bond",
):
    """Truncated SVD of a SymmetricTensor with Lorentzian-regularized backward.

    This is a convenience wrapper that densifies the SymmetricTensor, applies
    :func:`truncated_svd_ad` on the dense matrix, and reconstructs the result
    as a ``DenseTensor``.  The Lorentzian regularization from Francuz et al.
    (PRR 7, 013237) prevents NaN gradients from degenerate singular values
    within charge sectors.

    While a native per-sector custom_vjp would be more efficient (avoiding
    the dense round-trip), this implementation is correct and much simpler.
    The round-trip cost is acceptable for the moderate tensor sizes used in
    current AD iPEPS calculations.

    Args:
        M:               SymmetricTensor (or DenseTensor) to decompose.
        left_labels:     Labels forming the left (U) factor.
        right_labels:    Labels forming the right (Vh) factor.
        chi:             Number of singular values to keep.
        new_bond_label:  Label for the new virtual bond.

    Returns:
        ``(U, s, Vh)`` where U and Vh are ``DenseTensor`` objects and
        s is a 1-D ``jax.Array`` of length ``min(chi, min(m, n))``.
    """
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.tensor import DenseTensor

    # Resolve label ordering: left_labels then right_labels
    all_labels = list(left_labels) + list(right_labels)
    all_indices = []
    perm = []
    current_labels = M.labels()
    for lbl in all_labels:
        ax = current_labels.index(lbl)
        perm.append(ax)
        all_indices.append(M.indices[ax])

    # Densify and permute to (left_labels..., right_labels...)
    dense = M.todense()
    dense = jnp.transpose(dense, perm)

    # Reshape to matrix
    left_shape = tuple(dense.shape[i] for i in range(len(left_labels)))
    right_shape = tuple(
        dense.shape[i] for i in range(len(left_labels), len(all_labels))
    )
    m = 1
    for s in left_shape:
        m *= s
    n = 1
    for s in right_shape:
        n *= s
    matrix = dense.reshape(m, n)

    # Apply regularized SVD
    U_mat, s_vals, Vh_mat = truncated_svd_ad(matrix, chi)
    k = s_vals.shape[0]

    # Reshape back to tensor form
    U_data = U_mat.reshape(left_shape + (k,))
    Vh_data = Vh_mat.reshape((k,) + right_shape)

    # Build indices for the output tensors
    left_indices = tuple(all_indices[i] for i in range(len(left_labels)))
    right_indices = tuple(
        all_indices[i] for i in range(len(left_labels), len(all_labels))
    )

    # Determine symmetry from input (if available)
    sym = M.indices[0].symmetry
    bond_charges = jnp.zeros(k, dtype=jnp.int32)
    bond_out = TensorIndex(sym, bond_charges, FlowDirection.OUT, label=new_bond_label)
    bond_in = TensorIndex(sym, bond_charges, FlowDirection.IN, label=new_bond_label)

    U_tensor = DenseTensor(U_data, left_indices + (bond_out,))
    Vh_tensor = DenseTensor(Vh_data, (bond_in,) + right_indices)

    return U_tensor, s_vals, Vh_tensor


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
        int(getattr(config, "ad_regularize_svd", True)),
        int(getattr(config, "gmres_precondition", True)),
    )


def _config_from_tuple(config_tuple: tuple):
    """Reconstruct CTMConfig from a packed tuple."""
    pm_int = config_tuple[4] if len(config_tuple) > 4 else 0
    min_iter = config_tuple[5] if len(config_tuple) > 5 else 10
    ad_regularize_svd = bool(config_tuple[6]) if len(config_tuple) > 6 else True
    gmres_precondition = bool(config_tuple[7]) if len(config_tuple) > 7 else True
    return CTMConfig(
        chi=config_tuple[0],
        max_iter=config_tuple[1],
        conv_tol=config_tuple[2],
        renormalize=bool(config_tuple[3]),
        projector_method=_PM_INT_TO_STR.get(pm_int, "eigh"),
        min_iter=min_iter,
        ad_regularize_svd=ad_regularize_svd,
        gmres_precondition=gmres_precondition,
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


def _ctm_sv_diff_local(sv_new, sv_old):
    """Compute max abs diff between normalized SVs."""
    return _ctm_sv_diff_tensor(sv_new, sv_old)


def _flatten_envs(envs):
    """Flatten ``{Coord: CTMTensorEnv}`` to a flat tuple of leaves in coord-sorted order."""
    result = ()
    for c in sorted(envs):
        result = result + tuple(jax.tree.leaves(envs[c]))
    return result


def _unflatten_envs_init(env_init_leaves, site_tensors, chi):
    """Unflatten env_init_leaves into ``{Coord: CTMTensorEnv}``, or None."""
    if env_init_leaves is None:
        return None
    # Use first site tensor as template
    first_tensor = next(iter(site_tensors.values()))
    template = initialize_ctm_tensor_env(first_tensor, chi)
    treedef = jax.tree.structure(template)
    n = len(jax.tree.leaves(template))
    envs = {}
    offset = 0
    for c in sorted(site_tensors):
        envs[c] = jax.tree.unflatten(
            treedef, list(env_init_leaves[offset : offset + n])
        )
        offset += n
    return envs


def _ctm_tensor_step_multisite(
    site_leaves,
    env_leaves,
    neighbors,
    chi,
    renormalize,
    projector_method,
    site_treedefs,
    env_treedef,
    n_env_per_site,
    double_layers=None,
):
    """One multisite CTM sweep + gauge fix, flat leaves to flat leaves.

    Works for any number of sites (1-site, 2-site, etc.).

    If *double_layers* is provided, it is used directly (avoids redundant
    recomputation when site tensors are constant, e.g. in the GMRES backward pass).
    """
    # Unflatten site tensors
    coords = sorted(site_treedefs)
    site_tensors = {}
    site_offset = 0
    for c in coords:
        td = site_treedefs[c]
        n_leaves = td.num_leaves
        site_tensors[c] = jax.tree.unflatten(
            td, site_leaves[site_offset : site_offset + n_leaves]
        )
        site_offset += n_leaves

    # Unflatten env tensors
    envs = {}
    env_offset = 0
    for c in coords:
        envs[c] = jax.tree.unflatten(
            env_treedef, list(env_leaves[env_offset : env_offset + n_env_per_site])
        )
        env_offset += n_env_per_site

    # Build double layers if not cached
    if double_layers is None:
        double_layers = {
            c: _build_double_layer_tensor(A) for c, A in site_tensors.items()
        }

    envs = _ctm_tensor_sweep_multisite(
        envs, double_layers, neighbors, chi, renormalize, projector_method
    )
    envs = {c: _gauge_fix_ctm_tensor(e) for c, e in envs.items()}

    return _flatten_envs(envs)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def ctm_tensor_converge(
    site_tensors,
    env_init_leaves,
    neighbors,
    config_tuple: tuple,
) -> tuple[jax.Array, ...]:
    """Unified multisite Tensor-protocol CTM with implicit differentiation.

    Handles 1-site (with ``SINGLE_SITE_NEIGHBORS``) and multi-site
    (e.g. ``CHECKERBOARD_NEIGHBORS``) unit cells through a single code path.

    Args:
        site_tensors:    Dict ``{Coord: Tensor}`` of iPEPS site tensors.
        env_init_leaves: Flat tuple of env leaf arrays for warm-start, or None.
        neighbors:       Neighbor map ``{Coord: {direction: Coord}}``.
        config_tuple:    CTMConfig fields packed as tuple for JAX tracing.

    Returns:
        Flat tuple of environment pytree leaf arrays (all sites, coord-sorted).
    """
    config = _config_from_tuple(config_tuple)
    envs_init = _unflatten_envs_init(env_init_leaves, site_tensors, config.chi)
    envs = _ctm_tensor_multisite_fixed_point(
        site_tensors, neighbors, config, envs_init=envs_init
    )
    return _flatten_envs(envs)


def _ctm_tensor_converge_fwd(site_tensors, env_init_leaves, neighbors, config_tuple):
    """Forward pass -- run multisite Tensor CTM, cache tensors and envs."""
    config = _config_from_tuple(config_tuple)
    envs_init = _unflatten_envs_init(env_init_leaves, site_tensors, config.chi)
    envs = _ctm_tensor_multisite_fixed_point(
        site_tensors, neighbors, config, envs_init=envs_init
    )
    out = _flatten_envs(envs)
    residuals = (site_tensors, envs, env_init_leaves)
    return out, residuals


def _ctm_tensor_converge_bwd(neighbors, config_tuple, residuals, g):
    """Backward pass via implicit differentiation of multisite CTM fixed point."""
    site_tensors, envs, env_init_leaves = residuals
    config = _config_from_tuple(config_tuple)

    coords = sorted(site_tensors)

    # Build treedefs for each site tensor
    site_treedefs = {c: jax.tree.structure(site_tensors[c]) for c in coords}

    # All envs share the same treedef
    env_treedef = jax.tree.structure(envs[coords[0]])
    n_env_per_site = len(jax.tree.leaves(envs[coords[0]]))

    # Flatten site and env leaves
    site_leaves = ()
    for c in coords:
        site_leaves = site_leaves + tuple(jax.tree.leaves(site_tensors[c]))
    env_leaves = _flatten_envs(envs)

    def step_fn(s_leaves, e_leaves):
        return _ctm_tensor_step_multisite(
            s_leaves,
            e_leaves,
            neighbors,
            config.chi,
            config.renormalize,
            config.projector_method,
            site_treedefs,
            env_treedef,
            n_env_per_site,
        )

    # Hoist VJP: compute forward residuals once, reuse for operator + preconditioner
    _, vjp_fn = jax.vjp(lambda e: step_fn(site_leaves, e), env_leaves)

    def apply_Jt(v):
        return vjp_fn(v)[0]

    def apply_I_minus_Jt(v):
        Jt_v = apply_Jt(v)
        return tuple(vi - ji for vi, ji in zip(v, Jt_v))

    # Neumann preconditioner: M^{-1} v = v + J^T v ≈ (I - J^T)^{-1} v
    precond = None
    if config.gmres_precondition:

        def apply_precond(v):
            Jt_v = apply_Jt(v)
            return tuple(vi + ji for vi, ji in zip(v, Jt_v))

        precond = apply_precond

    max_fp_iter = min(config.max_iter, 50)
    lam, info = jax_gmres(
        apply_I_minus_Jt,
        g,
        x0=g,
        tol=config.conv_tol,
        maxiter=max_fp_iter,
        M=precond,
    )

    # VJP w.r.t. site_leaves
    _, vjp_site_fn = jax.vjp(lambda s: step_fn(s, env_leaves), site_leaves)
    d_site_leaves = vjp_site_fn(lam)[0]

    # Unflatten site gradients back into dict matching site_tensors structure
    d_site_tensors = {}
    offset = 0
    for c in coords:
        td = site_treedefs[c]
        n = td.num_leaves
        d_site_tensors[c] = jax.tree.unflatten(td, d_site_leaves[offset : offset + n])
        offset += n

    # Zero gradient for env_init
    if env_init_leaves is None:
        d_env_init = None
    else:
        d_env_init = tuple(jnp.zeros_like(x) for x in env_init_leaves)
    return (d_site_tensors, d_env_init)


ctm_tensor_converge.defvjp(_ctm_tensor_converge_fwd, _ctm_tensor_converge_bwd)


# ---------------------------------------------------------------------------
# 4. Multisite Tensor-protocol CTM fixed-point loop (shared by all paths)
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
