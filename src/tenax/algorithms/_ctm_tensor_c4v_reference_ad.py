"""Reference-mode dense C4v CTM forward fixed-point utilities.

This module provides the forward fixed-point map used by the opt-in
reference-mode iPEPS AD path. Backward/implicit differentiation is added
separately.

References
----------
- Liao, Liu, Wang, Xiang, "Differentiable Programming Tensor Networks",
  Phys. Rev. X 9, 031041 (2019). arXiv:1903.09650.
  Original implicit-diff CTM recipe; notes the near-singular ``(I - J^T)``
  adjoint system and the need for regularization.
- Francuz, Schmoll, Rizzi, Eisert, Naumann, "Stable and efficient
  differentiation of tensor network algorithms",
  Phys. Rev. Research 7, 013237 (2025). arXiv:2311.11894.
  The paper this reference-mode C4v path implements (dense C4v fixed-point
  backward, truncated-eigh Lorentzian VJP).
- Naumann, Weerda, Rizzi, Eisert, Schmoll, "variPEPS — a versatile tensor
  network library for variational ground state simulations in two spatial
  dimensions", SciPost Phys. Codebases (arXiv:2308.12358). Similar Krylov
  adjoint with diagonal scaling; we mirror its damping strategy.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab as _krylov_bicgstab
from jax.scipy.sparse.linalg import gmres as _krylov_gmres

from tenax.algorithms._ctm_tensor_c4v import _c4v_sweep, _c4v_to_full_env
from tenax.algorithms._ctm_tensor_convergence import _ctm_sv_diff
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

__all__ = [
    "_truncated_eigh_lorentzian_backward",
    "_solve_linear_adjoint",
    "ctm_tensor_c4v_reference_backward",
    "ctm_tensor_c4v_reference_converge_reduced",
    "ctm_tensor_c4v_reference_fixed_point",
    "truncated_eigh_regularized",
]


def _truncated_eigh_lorentzian_backward(
    w_full: jax.Array,
    v_full: jax.Array,
    dw_k: jax.Array,
    dV_k: jax.Array,
    *,
    k: int,
    eps: float = 1e-12,
) -> jax.Array:
    """Regularized truncated eigendecomposition backward.

    Uses a Lorentzian-regularized eigen-gap inverse for both kept-kept and
    kept-discarded couplings. The kept-discarded term is the critical
    truncation correction missing in naive truncated-eigh adjoints.
    """
    n = int(w_full.shape[0])
    k = min(int(k), n)
    v_h = jnp.conj(v_full).T
    dtype = jnp.result_type(v_full.dtype, dV_k.dtype, dw_k.dtype)

    dw_full = jnp.concatenate([dw_k.astype(dtype), jnp.zeros((n - k,), dtype=dtype)])

    # Differential coefficients K_{j,i} = <v_j, dV_i> / (w_i - w_j), i in kept.
    # Symmetrizing this n x n matrix recovers both the kept-kept anti-gauge term
    # and the kept-discarded truncation correction from the Francuz et al.
    # regularized differential.
    inner = v_h @ dV_k.astype(dtype)  # (n, k)
    denom = w_full[:k][None, :] - w_full[:, None]  # (n, k): w_i - w_j
    inv_gap = denom / (denom**2 + eps**2)
    K = inner * inv_gap
    if k > 0:
        idx = jnp.arange(k)
        K = K.at[idx, idx].set(0.0)

    K_full = jnp.zeros((n, n), dtype=dtype).at[:, :k].set(K)
    C = jnp.diag(dw_full) + 0.5 * (K_full + jnp.conj(K_full).T)
    dM = v_full @ C @ v_h
    return 0.5 * (dM + jnp.conj(dM).T)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def truncated_eigh_regularized(
    M: jax.Array,
    chi: int,
) -> tuple[jax.Array, jax.Array]:
    """Truncated symmetric eigendecomposition with a regularized VJP."""
    w, v = jnp.linalg.eigh(M)
    k = min(int(chi), int(w.shape[0]))
    return w[:k], v[:, :k]


def _truncated_eigh_regularized_fwd(M: jax.Array, chi: int):
    M_h = 0.5 * (M + jnp.conj(M).T)
    w, v = jnp.linalg.eigh(M_h)
    k = min(int(chi), int(w.shape[0]))
    return (w[:k], v[:, :k]), (w, v, k)


def _truncated_eigh_regularized_bwd(chi: int, residuals, g):
    w, v, k = residuals
    dw_k, dV_k = g
    dM = _truncated_eigh_lorentzian_backward(w, v, dw_k, dV_k, k=k)
    return (dM,)


truncated_eigh_regularized.defvjp(
    _truncated_eigh_regularized_fwd, _truncated_eigh_regularized_bwd
)


def _tree_sub(a, b):
    return jax.tree.map(lambda x, y: x - y, a, b)


def _tree_l2_norm(tree) -> float:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return 0.0
    total = sum(jnp.sum(jnp.abs(x) ** 2) for x in leaves)
    return float(jnp.sqrt(total))


def _tree_all_finite(tree) -> bool:
    leaves = jax.tree.leaves(tree)
    return all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)


def _ctm_tensor_c4v_reference_fixed_point_reduced(
    A: Tensor,
    config: CTMConfig,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Run dense C4v CTM to a fixed point (reduced C/T representation)."""
    if isinstance(A, SymmetricTensor):
        raise TypeError(
            "ctm_ad_mode='c4v_reference' currently supports dense tensors only."
        )
    if not isinstance(A, DenseTensor):
        raise TypeError(
            f"Expected DenseTensor for reference C4v mode, got {type(A).__name__}."
        )

    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, config.chi)

    C = env.C1.relabels({"c1_d": "c_a", "c1_r": "c_b"})
    T = env.T1.relabels({"t1_l": "t_l", "u2": "D2", "t1_r": "t_r"})

    prev_sv = None
    residual = float("inf")
    converged = False
    iters = 0
    min_iter = max(int(getattr(config, "min_iter", 1)), 1)

    for it in range(int(config.max_iter)):
        C, T = _c4v_sweep(C, T, a, config.chi, config.projector_method)
        iters = it + 1
        current_sv = jnp.linalg.svd(C.todense(), compute_uv=False)

        if prev_sv is not None:
            residual = float(_ctm_sv_diff(current_sv, prev_sv))
            if iters >= min_iter and residual < float(config.conv_tol):
                converged = True
                break
        prev_sv = current_sv

    meta = {"iters": iters, "residual": residual, "converged": converged}
    return C, T, meta


def ctm_tensor_c4v_reference_fixed_point(
    A: Tensor,
    config: CTMConfig,
) -> tuple[Any, dict[str, Any]]:
    """Run dense C4v CTM to a fixed point and return diagnostics."""
    C, T, meta = _ctm_tensor_c4v_reference_fixed_point_reduced(A, config)
    env_full = _c4v_to_full_env(C, T)
    return env_full, meta


def _build_env_diag_preconditioner(
    C: Tensor,
    T: Tensor,
    diag_shift: float,
):
    c_scale = 1.0 / (jnp.mean(jnp.abs(C.todense())) + diag_shift)
    t_scale = 1.0 / (jnp.mean(jnp.abs(T.todense())) + diag_shift)

    def precondition(v):
        v_c, v_t = v
        return (v_c * c_scale, v_t * t_scale)

    return precondition


def _solve_linear_adjoint(
    apply_i_minus_jt,
    rhs,
    config: CTMConfig,
    *,
    preconditioner=None,
) -> tuple[Any, dict[str, Any]]:
    """Solve ``(I - J^T) lambda = rhs`` with reference-mode Krylov policy."""
    maxiter = int(config.adjoint_maxiter)
    tol = float(config.adjoint_tol)
    requested = str(config.adjoint_solver)
    residual_target = max(tol * 10.0, 1e-12)

    def _solve_bicg(x0):
        lam, info = _krylov_bicgstab(
            apply_i_minus_jt,
            rhs,
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            M=preconditioner,
        )
        return lam, info

    def _solve_gmres(x0):
        lam, info = _krylov_gmres(
            apply_i_minus_jt,
            rhs,
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            restart=min(maxiter, 20),
            M=preconditioner,
        )
        return lam, info

    def _residual_norm(lam):
        return _tree_l2_norm(_tree_sub(apply_i_minus_jt(lam), rhs))

    def _is_success(lam, info, residual, *, solver: str) -> bool:
        if not _tree_all_finite(lam) or not jnp.isfinite(residual):
            return False
        if solver == "gmres" and int(info) != 0:
            return False
        return residual <= residual_target

    used_fallback = False
    if requested == "gmres":
        lam, info = _solve_gmres(rhs)
        solver_used = "gmres"
        residual = _residual_norm(lam)
    else:
        lam, info = _solve_bicg(rhs)
        solver_used = "bicgstab"
        residual = _residual_norm(lam)
        if not _is_success(lam, info, residual, solver="bicgstab"):
            lam, info = _solve_gmres(lam)
            solver_used = "gmres"
            residual = _residual_norm(lam)
            used_fallback = True

    converged = _is_success(lam, info, residual, solver=solver_used)
    meta = {
        "solver_requested": requested,
        "solver_used": solver_used,
        "used_fallback": used_fallback,
        "residual": residual,
        "tol": tol,
        "maxiter": maxiter,
        "info": None if info is None else int(info),
        "converged": bool(converged),
    }
    if not converged:
        raise RuntimeError(
            f"Reference-mode Krylov adjoint solver ({solver_used}) failed to "
            f"converge: residual={float(residual):.3e}, target<={residual_target:.3e}, "
            f"info={meta['info']}. Using a non-solution would corrupt gradients. "
            f"Increase adjoint_maxiter or loosen adjoint_tol."
        )
    return lam, meta


def _ctm_tensor_c4v_reference_implicit_backward(
    A: DenseTensor,
    C: Tensor,
    T: Tensor,
    g: tuple[Tensor, Tensor],
    config: CTMConfig,
) -> tuple[DenseTensor, dict[str, Any]]:
    """Matrix-free implicit backward for reference-mode C4v CTM."""
    a = _build_double_layer_tensor(A)
    g_c, g_t = g

    def _step_env(env_pair):
        c_in, t_in = env_pair
        return _c4v_sweep(c_in, t_in, a, config.chi, config.projector_method)

    _, vjp_env_fn = jax.vjp(_step_env, (C, T))

    def _step_site(A_in):
        a_in = _build_double_layer_tensor(A_in)
        return _c4v_sweep(C, T, a_in, config.chi, config.projector_method)

    _, vjp_site_fn = jax.vjp(_step_site, A)

    # Tikhonov-damped adjoint matvec. Near a CTM fixed point, ``J`` has
    # eigenvalues approaching 1 along the slowest-decaying modes, so
    # ``(I - J^T)`` is near-singular and Krylov solvers stall. Adding
    # ``tau * I`` shifts every eigenvalue by ``+tau``, bounding the condition
    # number at ``(sigma_max + tau) / tau`` at the cost of an O(tau) bias in
    # the gradient along the near-null directions. See the module docstring
    # for references (Liao et al. 2019; Francuz et al. 2025; variPEPS).
    tikhonov = float(getattr(config, "adjoint_tikhonov", 0.0) or 0.0)

    if tikhonov > 0.0:

        def _apply_i_minus_jt(v):
            jtv = vjp_env_fn(v)[0]
            base = _tree_sub(v, jtv)
            return jax.tree.map(lambda b, w: b + tikhonov * w, base, v)
    else:

        def _apply_i_minus_jt(v):
            jtv = vjp_env_fn(v)[0]
            return _tree_sub(v, jtv)

    precond = _build_env_diag_preconditioner(
        C, T, float(getattr(config, "adjoint_diag_shift", 1e-12))
    )
    lam, solve_meta = _solve_linear_adjoint(
        _apply_i_minus_jt,
        (g_c, g_t),
        config,
        preconditioner=precond,
    )
    solve_meta["tikhonov"] = tikhonov
    dA = vjp_site_fn(lam)[0]
    return dA, solve_meta


def ctm_tensor_c4v_reference_backward(
    A: DenseTensor,
    C: Tensor,
    T: Tensor,
    g_c: Tensor,
    g_t: Tensor,
    config: CTMConfig,
) -> tuple[DenseTensor, dict[str, Any]]:
    """Run one implicit-adjoint solve for diagnostics/testing."""
    return _ctm_tensor_c4v_reference_implicit_backward(A, C, T, (g_c, g_t), config)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def ctm_tensor_c4v_reference_converge_reduced(
    A: DenseTensor,
    config: CTMConfig,
) -> tuple[Tensor, Tensor]:
    """Fixed-point converger returning reduced C4v env ``(C, T)``."""
    C, T, _meta = _ctm_tensor_c4v_reference_fixed_point_reduced(A, config)
    return C, T


def _ctm_tensor_c4v_reference_converge_reduced_fwd(A: DenseTensor, config: CTMConfig):
    C, T, _meta = _ctm_tensor_c4v_reference_fixed_point_reduced(A, config)
    return (C, T), (A, C, T)


def _ctm_tensor_c4v_reference_converge_reduced_bwd(
    config: CTMConfig,
    residuals,
    g,
):
    A, C, T = residuals
    g_c, g_t = g
    dA, _solve_meta = _ctm_tensor_c4v_reference_implicit_backward(
        A, C, T, (g_c, g_t), config
    )
    return (dA,)


ctm_tensor_c4v_reference_converge_reduced.defvjp(
    _ctm_tensor_c4v_reference_converge_reduced_fwd,
    _ctm_tensor_c4v_reference_converge_reduced_bwd,
)
