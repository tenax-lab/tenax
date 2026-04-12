"""Paper-mode dense C4v CTM forward fixed-point utilities.

This module provides the forward fixed-point map used by the opt-in
paper-faithful iPEPS AD path. Backward/implicit differentiation is added
separately.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_c4v import _c4v_sweep, _c4v_to_full_env
from tenax.algorithms._ctm_tensor_convergence import _ctm_sv_diff
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

__all__ = ["ctm_tensor_c4v_paper_fixed_point"]


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
