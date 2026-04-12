"""Paper-mode dense C4v CTM forward fixed-point utilities.

This module provides the forward fixed-point map used by the opt-in
paper-faithful iPEPS AD path. Backward/implicit differentiation is added
separately.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_c4v import _c4v_sweep, _c4v_to_full_env
from tenax.algorithms._ctm_tensor_convergence import _ctm_sv_diff
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

__all__ = [
    "_appendix_c_truncated_eigh_backward",
    "ctm_tensor_c4v_paper_fixed_point",
    "truncated_eigh_appendix_c",
]


def _appendix_c_truncated_eigh_backward(
    w_full: jax.Array,
    v_full: jax.Array,
    dw_k: jax.Array,
    dV_k: jax.Array,
    *,
    k: int,
    eps: float = 1e-12,
) -> jax.Array:
    """Appendix-C-style truncated eigendecomposition backward.

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
    # and the kept-discarded truncation correction from Appendix C.
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
def truncated_eigh_appendix_c(
    M: jax.Array,
    chi: int,
) -> tuple[jax.Array, jax.Array]:
    """Truncated symmetric eigendecomposition with Appendix-C-style VJP."""
    w, v = jnp.linalg.eigh(M)
    k = min(int(chi), int(w.shape[0]))
    return w[:k], v[:, :k]


def _truncated_eigh_appendix_c_fwd(M: jax.Array, chi: int):
    M_h = 0.5 * (M + jnp.conj(M).T)
    w, v = jnp.linalg.eigh(M_h)
    k = min(int(chi), int(w.shape[0]))
    return (w[:k], v[:, :k]), (w, v, k)


def _truncated_eigh_appendix_c_bwd(chi: int, residuals, g):
    w, v, k = residuals
    dw_k, dV_k = g
    dM = _appendix_c_truncated_eigh_backward(w, v, dw_k, dV_k, k=k)
    return (dM,)


truncated_eigh_appendix_c.defvjp(
    _truncated_eigh_appendix_c_fwd, _truncated_eigh_appendix_c_bwd
)


def ctm_tensor_c4v_paper_fixed_point(
    A: Tensor,
    config: CTMConfig,
) -> tuple[Any, dict[str, Any]]:
    """Run dense C4v CTM to a fixed point and return diagnostics.

    Args:
        A:      Dense 1-site iPEPS tensor.
        config: CTM configuration (chi/max_iter/conv_tol/projector/min_iter).

    Returns:
        ``(env, meta)`` where ``env`` is a full 8-tensor CTM environment and
        ``meta`` contains ``iters``, ``residual``, and ``converged``.
    """
    if isinstance(A, SymmetricTensor):
        raise TypeError(
            "paper_ctm_ad='c4v_appendix_cf' currently supports dense tensors only."
        )
    if not isinstance(A, DenseTensor):
        raise TypeError(
            f"Expected DenseTensor for paper C4v mode, got {type(A).__name__}."
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

    env_full = _c4v_to_full_env(C, T)
    meta = {
        "iters": iters,
        "residual": residual,
        "converged": converged,
    }
    return env_full, meta
