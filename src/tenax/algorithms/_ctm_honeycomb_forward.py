"""Forward CTM convergence for honeycomb iPEPS — Python-loop, JIT'd sweep.

Mirrors :mod:`tenax.algorithms._ctm_python_loop` but for the rank-4,
6-corner, 3-direction, 2-sublattice honeycomb topology. Each sweep is
JIT-compiled; convergence checking happens in Python so changing tolerance
or method does not trigger recompilation.

Emits :func:`warnings.warn` (not :class:`RuntimeError`) when the iteration
budget is exhausted without meeting ``conv_tol`` — matches the AD-side
contract from PR #343 / memory ``project_python_level_ctm_ad.md``.
"""

from __future__ import annotations

__all__ = [
    "HoneycombConvergeInfo",
    "honeycomb_ctm_run",
]

import warnings
from functools import partial
from typing import NamedTuple

import jax

from tenax.algorithms._ctm_honeycomb_convergence import check_honeycomb_convergence
from tenax.algorithms._ctm_honeycomb_env import HoneycombCTMEnv
from tenax.algorithms._ctm_honeycomb_init import initialize_honeycomb_env
from tenax.algorithms._ctm_honeycomb_moves import honeycomb_ctm_step
from tenax.algorithms._ctm_honeycomb_topology import Coord
from tenax.core.tensor import Tensor


class HoneycombConvergeInfo(NamedTuple):
    """Convergence diagnostics for :func:`honeycomb_ctm_run`."""

    converged: bool
    iterations: int


@partial(
    jax.jit,
    static_argnames=("chi", "projector_method", "forward_gauge", "renormalize"),
)
def _jit_honeycomb_step(
    envs: dict[Coord, HoneycombCTMEnv],
    sites: dict[Coord, Tensor],
    *,
    chi: int,
    projector_method: str,
    forward_gauge: str,
    renormalize: bool,
) -> dict[Coord, HoneycombCTMEnv]:
    return honeycomb_ctm_step(
        envs,
        sites,
        chi=chi,
        projector_method=projector_method,
        forward_gauge=forward_gauge,
        renormalize=renormalize,
    )


def honeycomb_ctm_run(
    sites: dict[Coord, Tensor],
    *,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    projector_method: str = "biorthogonal",
    forward_gauge: str = "phase",
    renormalize: bool = True,
    conv_method: str = "elementwise",
    min_iter: int = 4,
    chi_ramp: list[tuple[int, int | None]] | None = None,
    env_init: dict[Coord, HoneycombCTMEnv] | None = None,
    seed: int = 42,
) -> tuple[dict[Coord, HoneycombCTMEnv], HoneycombConvergeInfo]:
    """Run honeycomb CTM to convergence using a Python-loop over JIT'd sweeps.

    Args:
        sites: ``{(0, 0): A, (1, 0): B}`` rank-4 honeycomb site tensors.
        chi: Final environment bond dimension.
        max_iter: Maximum CTM iterations (per chi-ramp stage).
        conv_tol: Convergence tolerance — see ``conv_method``.
        projector_method: ``"biorthogonal"`` (default), ``"eigh"``, or
            ``"svd"`` (the latter two are isometric A=B opt-in paths).
        forward_gauge: ``"phase"`` (default — phase fix per sublattice
            update inside the projector build) or ``"sigma"`` /
            ``"none"`` reserved for the isometric path.
        renormalize: Inf-norm normalize all 9 fields after each sweep.
        conv_method: ``"elementwise"`` (max-abs leaf diff, default) or
            ``"svd"`` (per-corner singular-value diff). The two map onto
            the two methods of :func:`check_honeycomb_convergence`.
        min_iter: Minimum sweeps before the first convergence check.
        chi_ramp: Optional ``[(stage_chi, stage_sweeps), ...]`` schedule.
            Last stage uses ``max_iter`` if ``stage_sweeps is None``.
        env_init: Optional initial envs; otherwise random-init at the
            (initial-stage) chi via :func:`initialize_honeycomb_env`.
        seed: PRNG seed for env initialization.

    Returns:
        ``(envs, info)`` — converged per-sublattice env dict and a
        :class:`HoneycombConvergeInfo` with ``converged`` and
        ``iterations`` fields.
    """
    if chi_ramp is not None:
        return _honeycomb_chi_ramp(
            sites,
            chi=chi,
            max_iter=max_iter,
            conv_tol=conv_tol,
            projector_method=projector_method,
            forward_gauge=forward_gauge,
            renormalize=renormalize,
            conv_method=conv_method,
            min_iter=min_iter,
            chi_ramp=chi_ramp,
            env_init=env_init,
            seed=seed,
        )

    envs = (
        env_init
        if env_init is not None
        else initialize_honeycomb_env(sites, chi_init=chi, seed=seed)
    )

    prev_envs: dict[Coord, HoneycombCTMEnv] | None = None
    converged = False
    total_iter = 0

    for i in range(max_iter):
        prev_envs = envs
        envs = _jit_honeycomb_step(
            envs,
            sites,
            chi=chi,
            projector_method=projector_method,
            forward_gauge=forward_gauge,
            renormalize=renormalize,
        )
        total_iter = i + 1
        if total_iter < min_iter:
            continue
        if check_honeycomb_convergence(
            prev_envs, envs, method=conv_method, tol=conv_tol
        ):
            converged = True
            break

    if not converged:
        warnings.warn(
            f"honeycomb_ctm_run did not converge within max_iter={max_iter} "
            f"(method={conv_method!r}, tol={conv_tol:g}, chi={chi}); "
            "returning the last env. Increase max_iter or loosen conv_tol "
            "if this is unexpected.",
            RuntimeWarning,
            stacklevel=2,
        )

    return envs, HoneycombConvergeInfo(converged=converged, iterations=total_iter)


def _honeycomb_chi_ramp(
    sites: dict[Coord, Tensor],
    *,
    chi: int,
    max_iter: int,
    conv_tol: float,
    projector_method: str,
    forward_gauge: str,
    renormalize: bool,
    conv_method: str,
    min_iter: int,
    chi_ramp: list[tuple[int, int | None]],
    env_init: dict[Coord, HoneycombCTMEnv] | None,
    seed: int,
) -> tuple[dict[Coord, HoneycombCTMEnv], HoneycombConvergeInfo]:
    """Multi-stage chi ramp: re-init env between stages when chi changes."""
    envs = env_init
    prev_chi: int | None = None
    info = HoneycombConvergeInfo(converged=False, iterations=0)

    for stage_idx, (stage_chi, stage_sweeps) in enumerate(chi_ramp):
        is_last = stage_idx == len(chi_ramp) - 1
        stage_max = stage_sweeps if stage_sweeps is not None else max_iter
        # Only the last stage applies the user-supplied conv_tol; intermediate
        # stages exhaust their iteration budget so the env warms up at each chi.
        stage_tol = conv_tol if is_last else 0.0

        if prev_chi is not None and stage_chi != prev_chi:
            envs = None  # force re-init at the new chi

        envs, info = honeycomb_ctm_run(
            sites,
            chi=stage_chi,
            max_iter=stage_max,
            conv_tol=stage_tol,
            projector_method=projector_method,
            forward_gauge=forward_gauge,
            renormalize=renormalize,
            conv_method=conv_method,
            min_iter=min_iter,
            chi_ramp=None,
            env_init=envs,
            seed=seed,
        )
        prev_chi = stage_chi

    return envs, info
