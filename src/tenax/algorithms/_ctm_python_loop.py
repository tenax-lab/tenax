"""Python-loop CTM convergence — JIT single sweeps, Python convergence loop.

Instead of JIT-tracing the entire CTM convergence loop (which takes 20-90+
minutes to compile), we JIT only single CTM sweeps and run the convergence
loop in Python.  This gives the same converged environment with seconds of
compile time.
"""

from __future__ import annotations

__all__ = [
    "CTMConvergeInfo",
    "_make_jit_ctm_step",
    "python_loop_ctm_converge",
]

from functools import partial
from typing import NamedTuple

import jax

from tenax.algorithms._ctm_tensor_convergence import (
    Coord,
    _corner_singular_values,
    _ctm_sv_diff,
    _ctm_tensor_sweep_multisite,
    _max_env_leaf_diff,
)
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.core.tensor import Tensor


class CTMConvergeInfo(NamedTuple):
    """Convergence information from python_loop_ctm_converge."""

    converged: bool
    iterations: int
    sv_diff: float
    max_truncation_error: float = 0.0  # variPEPS §2.8.2 indicator (last sweep)


def _make_jit_ctm_step(
    neighbors: dict[Coord, dict[str, Coord]],
):
    """Create a JIT-compiled CTM step function for a given neighbor topology.

    The ``neighbors`` dict is captured in the closure so it is not traced
    by JAX (it contains only Python-level coordinate tuples, not arrays).

    Returns:
        A JIT-compiled function with signature::

            step(site_tensors, envs, *, chi, projector_method,
                 renormalize, projector_backward)
                 -> tuple[dict[Coord, CTMTensorEnv], jax.Array]

        The returned tuple is ``(new_envs, max_truncation_error)`` where
        ``max_truncation_error`` is the largest singular-value truncation
        error (ε_T) observed across all projector computations in the sweep.
    """

    @partial(
        jax.jit,
        static_argnames=(
            "chi",
            "projector_method",
            "renormalize",
            "projector_backward",
        ),
    )
    def _step(
        site_tensors: dict[Coord, Tensor],
        envs: dict[Coord, CTMTensorEnv],
        *,
        chi: int,
        projector_method: str = "svd",
        renormalize: bool = True,
        projector_backward: str = "auto",
    ) -> tuple[dict[Coord, CTMTensorEnv], jax.Array]:
        double_layers = {
            c: _build_double_layer_tensor(A) for c, A in site_tensors.items()
        }
        return _ctm_tensor_sweep_multisite(
            envs,
            double_layers,
            neighbors,
            chi,
            renormalize,
            projector_method,
            projector_backward=projector_backward,
        )

    return _step


def python_loop_ctm_converge(
    site_tensors: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    *,
    chi: int,
    max_iter: int = 100,
    min_iter: int = 10,
    conv_tol: float = 1e-8,
    conv_method: str = "sv",
    renormalize: bool = True,
    projector_method: str = "svd",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
    chi_ramp: list[tuple[int, int | None]] | None = None,
    env_init: dict[Coord, CTMTensorEnv] | None = None,
    gauge_fix_fn=None,
) -> tuple[dict[Coord, CTMTensorEnv], CTMConvergeInfo]:
    """Run CTM to convergence using a Python for-loop over JIT'd sweeps.

    Each sweep is JIT-compiled via ``_make_jit_ctm_step``; convergence
    checking happens in Python so no recompilation is needed when
    tolerances change.

    Args:
        site_tensors:      Map from coordinate to iPEPS site tensor.
        neighbors:         Map from coordinate to direction->neighbor coordinate.
        chi:               Final environment bond dimension.
        max_iter:          Maximum CTM iterations (per chi-ramp stage).
        min_iter:          Minimum iterations before checking convergence.
        conv_tol:          Convergence tolerance on corner singular values.
        conv_method:       Convergence method: ``"sv"`` (corner singular
                           values) or ``"elementwise"`` (max element-wise
                           difference across all env tensors).
        renormalize:       Renormalize environments after each sweep.
        projector_method:  ``"svd"`` (Fishman, default), ``"eigh"``, or ``"qr"``.
        qr_warmup_steps:   Number of eigh warm-up sweeps before QR kicks in.
        projector_backward: Backward mode for projector.
        chi_ramp:          Optional chi-ramp schedule as list of
                           ``(stage_chi, max_sweeps)`` tuples.  The last
                           stage uses ``max_iter`` if ``max_sweeps is None``.
        env_init:          Optional initial environments.  If ``None``,
                           identity-initialized environments are created.

    Returns:
        ``(envs, CTMConvergeInfo)`` — converged environments and info.
    """
    if chi_ramp is not None:
        return _python_loop_chi_ramp(
            site_tensors,
            neighbors,
            chi=chi,
            max_iter=max_iter,
            min_iter=min_iter,
            conv_tol=conv_tol,
            conv_method=conv_method,
            renormalize=renormalize,
            projector_method=projector_method,
            qr_warmup_steps=qr_warmup_steps,
            projector_backward=projector_backward,
            chi_ramp=chi_ramp,
            env_init=env_init,
            gauge_fix_fn=gauge_fix_fn,
        )

    # Build the JIT'd step function (captures neighbors in closure)
    jit_step = _make_jit_ctm_step(neighbors)

    # Initialize environments
    envs = (
        env_init
        if env_init is not None
        else {c: initialize_ctm_tensor_env(A, chi) for c, A in site_tensors.items()}
    )

    # QR warm-up: run a few eigh iterations before switching to QR
    warmup = 0
    if projector_method == "qr" and qr_warmup_steps > 0:
        warmup = min(qr_warmup_steps, max_iter)
        for _ in range(warmup):
            envs, _ = jit_step(
                site_tensors,
                envs,
                chi=chi,
                projector_method="eigh",
                renormalize=renormalize,
                projector_backward=projector_backward,
            )

    remaining = max_iter - warmup
    prev_svs: dict[Coord, jax.Array] = {}
    prev_envs: dict[Coord, CTMTensorEnv] | None = None
    final_diff = float("inf")
    last_max_eps: float = 0.0

    for i in range(remaining):
        envs, _max_eps = jit_step(
            site_tensors,
            envs,
            chi=chi,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )
        last_max_eps = float(_max_eps)

        # Apply gauge fix if provided (e.g., phase fix for element-wise convergence)
        if gauge_fix_fn is not None:
            envs = {c: gauge_fix_fn(envs[c]) for c in envs}

        # Only check convergence after min_iter total iterations
        total_iter = warmup + i + 1
        if total_iter < min_iter:
            # Still track SVs / prev_envs for the first convergence check
            if conv_method == "sv":
                for c in sorted(envs):
                    prev_svs[c] = _corner_singular_values(envs[c].C1)
            else:
                prev_envs = {c: envs[c] for c in envs}
            continue

        if conv_method == "elementwise":
            # Element-wise convergence: max absolute difference across all
            # env tensor leaves.
            if prev_envs is None:
                prev_envs = {c: envs[c] for c in envs}
                continue
            max_diff = 0.0
            for c in sorted(envs):
                max_diff = max(max_diff, _max_env_leaf_diff(prev_envs[c], envs[c]))
            converged = max_diff < conv_tol
            final_diff = max_diff
            prev_envs = {c: envs[c] for c in envs}
        else:
            # SV convergence (default): corner singular value difference
            converged = True
            max_diff = 0.0
            for c in sorted(envs):
                sv = _corner_singular_values(envs[c].C1)
                if c in prev_svs:
                    diff = float(_ctm_sv_diff(sv, prev_svs[c]))
                    max_diff = max(max_diff, diff)
                    if diff >= conv_tol:
                        converged = False
                else:
                    converged = False
                prev_svs[c] = sv
            final_diff = max_diff

        if converged:
            return envs, CTMConvergeInfo(
                converged=True,
                iterations=total_iter,
                sv_diff=final_diff,
                max_truncation_error=last_max_eps,
            )

    return envs, CTMConvergeInfo(
        converged=False,
        iterations=max_iter,
        sv_diff=final_diff,
        max_truncation_error=last_max_eps,
    )


def _python_loop_chi_ramp(
    site_tensors: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    *,
    chi: int,
    max_iter: int,
    min_iter: int,
    conv_tol: float,
    conv_method: str,
    renormalize: bool,
    projector_method: str,
    qr_warmup_steps: int,
    projector_backward: str,
    chi_ramp: list[tuple[int, int | None]],
    env_init: dict[Coord, CTMTensorEnv] | None,
    gauge_fix_fn=None,
) -> tuple[dict[Coord, CTMTensorEnv], CTMConvergeInfo]:
    """Run CTM with chi-ramp schedule."""
    envs = env_init
    prev_chi: int | None = None
    info = CTMConvergeInfo(
        converged=False, iterations=0, sv_diff=float("inf"), max_truncation_error=0.0
    )

    for stage_idx, (stage_chi, stage_sweeps) in enumerate(chi_ramp):
        is_last = stage_idx == len(chi_ramp) - 1

        # Determine iteration budget for this stage
        if stage_sweeps is not None:
            stage_max = stage_sweeps
        else:
            stage_max = max_iter

        # Determine convergence tolerance: only converge on last stage
        stage_tol = conv_tol if is_last else 0.0

        # Re-initialize when chi changes
        if prev_chi is not None and stage_chi != prev_chi:
            envs = None

        envs, info = python_loop_ctm_converge(
            site_tensors,
            neighbors,
            chi=stage_chi,
            max_iter=stage_max,
            min_iter=min_iter,
            conv_tol=stage_tol,
            conv_method=conv_method,
            renormalize=renormalize,
            projector_method=projector_method,
            qr_warmup_steps=qr_warmup_steps if stage_idx == 0 else 0,
            projector_backward=projector_backward,
            chi_ramp=None,
            env_init=envs,
            gauge_fix_fn=gauge_fix_fn,
        )
        prev_chi = stage_chi

    return envs, info
