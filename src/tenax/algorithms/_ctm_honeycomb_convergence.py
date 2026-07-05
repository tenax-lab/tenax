"""Honeycomb CTM convergence checks.

Two methods, mirroring :mod:`tenax.algorithms._ctm_tensor_convergence`:

* ``"elementwise"`` -- max absolute element-wise difference across all
  18 fields of the env (9 fields × 2 sublattices). Cheap and tight; the
  default for new code per memory ``project_elementwise_convergence_svd_proj.md``.
* ``"svd"`` -- per-corner singular-value diff (normalized to unit sum,
  then max-abs across the spectrum). More tolerant of phase / gauge
  drift; useful for the A=B isometric path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_honeycomb_env import HoneycombCTMEnv
from tenax.algorithms._ctm_honeycomb_topology import Coord
from tenax.linalg import _dense_svd

__all__ = ["check_honeycomb_convergence"]


def _max_env_leaf_diff(
    envs_old: dict[Coord, HoneycombCTMEnv],
    envs_new: dict[Coord, HoneycombCTMEnv],
) -> float:
    """Maximum |element-wise diff| across all 9 fields × 2 sublattices."""
    max_diff = 0.0
    for coord in envs_old:
        env_old = envs_old[coord]
        env_new = envs_new[coord]
        for field in env_old._fields:
            a = getattr(env_old, field).todense()
            b = getattr(env_new, field).todense()
            diff = float(jnp.max(jnp.abs(b - a)))
            if diff > max_diff:
                max_diff = diff
    return max_diff


def _corner_sv_diff(
    envs_old: dict[Coord, HoneycombCTMEnv],
    envs_new: dict[Coord, HoneycombCTMEnv],
) -> float:
    """Max-abs diff of normalized corner singular values across 6 corners.

    For each of the 6 corners (3 directions × 2 sublattices), compute
    ``svd_vals / sum(svd_vals)`` for both old and new, then take
    ``max|sv_new - sv_old|`` and aggregate the worst across all 6.
    """
    worst = 0.0
    for coord in envs_old:
        env_old = envs_old[coord]
        env_new = envs_new[coord]
        for alpha in (0, 1, 2):
            C_old = getattr(env_old, f"C{alpha}").todense()
            C_new = getattr(env_new, f"C{alpha}").todense()
            sv_old = _dense_svd(C_old, compute_uv=False)
            sv_new = _dense_svd(C_new, compute_uv=False)
            sv_old = sv_old / (jnp.sum(sv_old) + 1e-15)
            sv_new = sv_new / (jnp.sum(sv_new) + 1e-15)
            diff = float(jnp.max(jnp.abs(sv_new - sv_old)))
            if diff > worst:
                worst = diff
    return worst


def check_honeycomb_convergence(
    envs_old: dict[Coord, HoneycombCTMEnv],
    envs_new: dict[Coord, HoneycombCTMEnv],
    *,
    method: str = "elementwise",
    tol: float,
) -> bool:
    """Return ``True`` if ``envs_old`` and ``envs_new`` agree within ``tol``.

    Args:
        envs_old, envs_new: per-sublattice env dicts to compare.
        method: ``"elementwise"`` (max-abs over all fields) or ``"svd"``
            (per-corner normalized singular-value diff).
        tol: convergence tolerance.

    Raises:
        ValueError: unknown ``method``.
    """
    if method == "elementwise":
        return _max_env_leaf_diff(envs_old, envs_new) < tol
    if method == "svd":
        return _corner_sv_diff(envs_old, envs_new) < tol
    raise ValueError(
        f"Unknown convergence method: {method!r}; expected 'elementwise' or 'svd'."
    )


# Silence unused-import warning when SymmetricTensor support is added later.
_ = jax
