"""Truncation-error metric for CTM projectors.

Implements the variPEPS §2.8.2 indicator that gates auto-χ_E bumps.
"""

from __future__ import annotations

import jax.numpy as jnp


def compute_truncation_error(s: jnp.ndarray, chi: int) -> jnp.ndarray:
    """Normalized L2 norm of discarded singular values.

    ε_T = ‖s[χ:]‖_2 / ‖s‖_2, where s is the full SV vector returned by the
    SVD inside CTM projector construction. variPEPS §2.8.2 (SciPost Lect.
    Notes 86) recommends bumping χ_E whenever this exceeds ~1e-5.

    Returns a JAX scalar so it composes inside `jit`-compiled CTM sweeps.
    """
    s_full_norm_sq = jnp.sum(s**2)
    discarded = s[chi:]
    discarded_norm_sq = jnp.sum(discarded**2)
    safe_total = jnp.where(s_full_norm_sq > 0.0, s_full_norm_sq, 1.0)
    eps = jnp.sqrt(discarded_norm_sq / safe_total)
    return jnp.where(s_full_norm_sq > 0.0, eps, jnp.array(0.0, dtype=eps.dtype))
