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

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax

if TYPE_CHECKING:
    from jax.sharding import Mesh

from tenax.algorithms._ctm_loop_core import (
    _run_ctm_loop_with_bump,
    _validate_chi_bump_args,
)
from tenax.algorithms._ctm_tensor_convergence import (
    Coord,
    _ctm_tensor_sweep_multisite,
    _warn_recipe_1x1_deprecated,
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
    iterations: int  # CTM sweeps actually performed (#781)
    sv_diff: float
    max_truncation_error: float = 0.0  # variPEPS §2.8.2 indicator (last sweep)
    max_smallest_S: float = 0.0  # variPEPS norm_smallest_S indicator (#492)
    final_chi: int = 0  # final chi after any in-CTM bumps (#492); 0 ⇒ unchanged
    # Sweep index whose environment is returned.  Equals ``iterations``
    # except on the ``plateau_patience`` bail, where the best-metric env is
    # handed back and this trails ``iterations`` by ``plateau_patience``.
    # ``sv_diff`` is the metric of *this* sweep, not of ``iterations``.
    best_iteration: int = 0


# Process-lifetime cache so repeat calls with the same neighbors dict reuse
# the same compiled @jit'd ``_step`` function.  Without this, every callsite
# (forward CTM convergence, implicit-AD f_bwd, line search, etc.) creates a
# fresh ``_step`` closure with its own JIT cache and pays redundant compile
# cost.  Diagnosed in docs/plans/2026-05-09-ipeps-ad-jit-cost-diagnosis.md.
# Keyed by id(neighbors) — safe because neighbors dicts are constructed once
# per optimizer invocation and stay alive throughout.
# Key is (id(neighbors), recipe, device_mesh, ctm_chunk_size): a 4-tuple of an
# int, a str, a ``jax.sharding.Mesh | None`` and an ``int | None`` (Mesh
# imported only under TYPE_CHECKING; the annotation is a string here thanks to
# ``from __future__ import annotations``).
_JIT_STEP_CACHE: dict[tuple[int, str, Mesh | None, int | None], Callable] = {}


def _make_jit_ctm_step(
    neighbors: dict[Coord, dict[str, Coord]],
    recipe: str = "2x2",
    device_mesh=None,
    ctm_chunk_size: int | None = None,
):
    """Create a JIT-compiled CTM step function for a given neighbor topology.

    The ``neighbors`` dict is captured in the closure so it is not traced
    by JAX (it contains only Python-level coordinate tuples, not arrays).

    Memoised by ``id(neighbors)`` so all callsites in a single optimizer
    invocation share one ``_step`` function and its JIT cache.

    Returns:
        A JIT-compiled function with signature::

            step(site_tensors, envs, *, chi, projector_method,
                 renormalize, projector_backward, chunk_size)
                 -> tuple[dict[Coord, CTMTensorEnv], jax.Array]

        The returned tuple is ``(new_envs, max_truncation_error)`` where
        ``max_truncation_error`` is the largest singular-value truncation
        error (ε_T) observed across all projector computations in the sweep.
    """
    # Include ``recipe`` in the cache key so the "1x1" and "2x2" sweeps get
    # distinct compiled ``_step`` closures even when they share the same
    # ``neighbors`` dict (which is reused across optimizer steps).
    # ``device_mesh`` is captured (static, non-traced) in the closure, so it
    # must participate in the cache key: the sharded and single-device steps
    # are different compiled functions.
    # ``ctm_chunk_size`` is static: different chunk sizes yield differently
    # shaped lax.map computations that must not share a JIT cache entry.
    # Asymmetric keying: ``neighbors`` is an unhashable dict so we fall back to
    # ``id()``; ``device_mesh`` is a hashable ``jax.sharding.Mesh`` (or None)
    # with a value ``__eq__``, so it is used directly — equal meshes share a
    # cache entry and we avoid id-reuse hazards after GC.
    cache_key = (id(neighbors), recipe, device_mesh, ctm_chunk_size)
    cached = _JIT_STEP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    @partial(
        jax.jit,
        static_argnames=(
            "chi",
            "projector_method",
            "renormalize",
            "projector_backward",
            "chunk_size",
        ),
    )
    def _step(
        site_tensors: dict[Coord, Tensor],
        envs: dict[Coord, CTMTensorEnv],
        *,
        chi: int,
        projector_method: str = "svd",
        renormalize: bool = False,
        projector_backward: str = "auto",
        chunk_size: int | None = None,
    ) -> tuple[dict[Coord, CTMTensorEnv], jax.Array, jax.Array]:
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
            recipe=recipe,
            device_mesh=device_mesh,
            chunk_size=chunk_size,
        )

    _JIT_STEP_CACHE[cache_key] = _step
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
    renormalize: bool = False,
    projector_method: str = "svd",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
    chi_ramp: list[tuple[int, int | None]] | None = None,
    env_init: dict[Coord, CTMTensorEnv] | None = None,
    gauge_fix_fn=None,
    plateau_patience: int | None = 20,
    ctmrg_heuristic_increase_chi: bool = False,
    ctmrg_heuristic_increase_chi_threshold: float = 1e-6,
    ctmrg_heuristic_increase_chi_step_size: int = 2,
    chi_max: int | None = None,
    recipe: str = "2x2",
    device_mesh=None,
    ctm_chunk_size: int | None = None,
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
                           difference across all env tensors).  Prefer
                           ``"sv"``: the environment is defined only up to a
                           gauge on each chi-bond, so ``"elementwise"``
                           measures gauge motion as well as convergence and
                           plateaus far above any usable ``conv_tol`` on a
                           physical state (#780).  Note this default does NOT
                           reach config-driven callers -- ``ctm_converge_kwargs``
                           always emits ``CTMConfig.ctm_conv_method``, which
                           is ``"elementwise"`` for the AD path (#351).
        renormalize:       Renormalize environments after each sweep.
        projector_method:  ``"svd"`` (Fishman, default), ``"eigh"``, or ``"qr"``.
        qr_warmup_steps:   Number of eigh warm-up sweeps before QR kicks in.
        projector_backward: Backward mode for projector.
        chi_ramp:          Optional chi-ramp schedule as list of
                           ``(stage_chi, max_sweeps)`` tuples.  The last
                           stage uses ``max_iter`` if ``max_sweeps is None``.
        env_init:          Optional initial environments.  If ``None``,
                           identity-initialized environments are created.
        plateau_patience:  Early-bail when the running minimum of the
                           convergence metric (``sv_diff`` or elementwise
                           ``max_diff``) has not improved over the last
                           ``plateau_patience`` iterations.  The loop
                           returns the env that achieved the best metric
                           and ``CTMConvergeInfo.converged=False`` — the
                           bail is a stop-loss, not a fixed point.  On that
                           path ``iterations`` counts the sweeps performed
                           (the bail sweep) while ``best_iteration`` locates
                           the returned env, ``plateau_patience`` sweeps
                           earlier; divide elapsed time by ``iterations``
                           for a per-sweep cost (#781).
                           Default ``20`` is a sane stop-loss against the
                           SU/random-init CTM plateau tracked in #425/#426
                           (memory ``project_tenax_ctm_doesnt_converge_random_init``):
                           a healthy converging run never accumulates 20
                           non-improving iterations because each better
                           ``best_diff`` resets the counter, while a true
                           plateau bails an order of magnitude faster than
                           burning the full ``max_iter`` budget.  Set to
                           ``None`` to restore the pre-2026-05-11 "run to
                           ``max_iter``" behavior.

    Returns:
        ``(envs, CTMConvergeInfo)`` — converged environments and info.
    """
    # #911: warned here rather than in ``_make_jit_ctm_step`` or
    # ``_python_loop_chi_ramp`` -- this is the once-per-convergence boundary,
    # those are once-per-sweep and once-per-ramp-stage.  Before the chi_ramp
    # early-return below, or a ramped call would never reach it.
    if recipe == "1x1":
        _warn_recipe_1x1_deprecated("python_loop_ctm_converge")

    # Reject ``ctmrg_heuristic_increase_chi`` + ``chi_ramp`` BEFORE the
    # chi_ramp early-return, otherwise the bump flag is silently ignored
    # by ``_python_loop_chi_ramp`` (which doesn't accept the new knobs).
    # CTMConfig also enforces this; defense-in-depth for direct callers
    # (codex review on PR #513).
    if ctmrg_heuristic_increase_chi and chi_ramp is not None:
        raise ValueError(
            "ctmrg_heuristic_increase_chi and chi_ramp are mutually "
            "exclusive: chi_ramp is a deterministic schedule applied "
            "across stages, while ctmrg_heuristic_increase_chi is reactive "
            "inside a single CTM convergence call."
        )
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
            plateau_patience=plateau_patience,
            recipe=recipe,
            device_mesh=device_mesh,
            ctm_chunk_size=ctm_chunk_size,
        )

    # Build the JIT'd step function (captures neighbors + device_mesh in closure)
    jit_step = _make_jit_ctm_step(
        neighbors, recipe, device_mesh=device_mesh, ctm_chunk_size=ctm_chunk_size
    )
    # Bind chunk_size so callers that don't know about it (e.g.
    # _run_ctm_loop_with_bump) pass it transparently.
    jit_step = partial(jit_step, chunk_size=ctm_chunk_size)

    # chi may grow during the loop when ``ctmrg_heuristic_increase_chi``
    # is enabled (variPEPS-style in-CTM bump; Issue #492).  ``chi_current``
    # is the live value used by JIT'd sweeps; the new chi is static_argname,
    # so each distinct chi value will retrace once on first use.
    #
    # Validation (chi_max required, step_size > 0, env_init chi vs chi_max,
    # chi_max >= chi_current) and env_init-driven chi_current promotion are
    # centralised in ``_validate_chi_bump_args`` so all three forward-CTM
    # entry points (this function, _sigma_gauged_ctm_converge, and
    # ctm_energy_explicit) raise identical errors.
    chi_current = _validate_chi_bump_args(
        chi=chi,
        chi_max=chi_max,
        env_init=env_init,
        bump_enabled=ctmrg_heuristic_increase_chi,
        bump_step_size=ctmrg_heuristic_increase_chi_step_size,
    )

    # Build gauge_fix_fn pair adapter
    if gauge_fix_fn is not None:
        _user_gauge = gauge_fix_fn

        def _gauge_pair(envs_new, envs_old):
            return {c: _user_gauge(envs_new[c]) for c in envs_new}
    else:
        _gauge_pair = None

    # Initialize envs
    envs = (
        env_init
        if env_init is not None
        else {
            c: initialize_ctm_tensor_env(A, chi_current)
            for c, A in site_tensors.items()
        }
    )

    # GSPMD: commit the initial envs onto the device mesh (edges D²-sharded,
    # corners replicated) so the JIT'd sweep's inputs already carry the
    # sharding the per-move constraints chain off.  No-op when device_mesh
    # is None (single-device path is byte-for-byte unchanged).
    if device_mesh is not None:
        from tenax.algorithms.ctm_sharding import commit_env

        envs = {coord: commit_env(env, device_mesh) for coord, env in envs.items()}

    # QR warm-up: run a few eigh iterations before switching to QR
    warmup = 0
    if projector_method == "qr" and qr_warmup_steps > 0:
        warmup = min(qr_warmup_steps, max_iter)
        for _ in range(warmup):
            envs, _, _ = jit_step(
                site_tensors,
                envs,
                chi=chi_current,
                projector_method="eigh",
                renormalize=renormalize,
                projector_backward=projector_backward,
            )

    # Run the bump-aware loop via shared helper.
    result = _run_ctm_loop_with_bump(
        jit_step,
        site_tensors,
        envs,
        chi_current=chi_current,
        chi_max=chi_max,
        bump_enabled=ctmrg_heuristic_increase_chi,
        bump_threshold=ctmrg_heuristic_increase_chi_threshold,
        bump_step_size=ctmrg_heuristic_increase_chi_step_size,
        projector_method=projector_method,
        renormalize=renormalize,
        projector_backward=projector_backward,
        gauge_fix_fn=_gauge_pair,
        max_iter=max_iter - warmup,
        min_iter=max(0, min_iter - warmup),
        conv_tol=conv_tol,
        conv_method=conv_method,
        plateau_patience=plateau_patience,
    )

    return result.envs, CTMConvergeInfo(
        converged=result.converged,
        iterations=warmup + result.iterations,
        sv_diff=result.sv_diff,
        max_truncation_error=result.max_truncation_error,
        max_smallest_S=result.max_smallest_S,
        final_chi=result.final_chi,
        best_iteration=warmup + result.best_iteration,
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
    plateau_patience: int | None = None,
    recipe: str = "2x2",
    device_mesh=None,
    ctm_chunk_size: int | None = None,
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

        # Disable plateau early-bail when the user pinned an explicit
        # warm-up budget for a non-final stage: a ramp like
        # ``[(8, 100), (16, None)]`` is a contract to spend exactly 100
        # sweeps at chi=8 before moving on, so the next stage sees a
        # consistent warm-start env regardless of the plateau metric
        # (codex review on PR #439).  The final stage and any stage with
        # ``stage_sweeps=None`` still honor ``plateau_patience``.
        stage_patience = (
            None if (not is_last and stage_sweeps is not None) else plateau_patience
        )

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
            plateau_patience=stage_patience,
            recipe=recipe,
            device_mesh=device_mesh,
            ctm_chunk_size=ctm_chunk_size,
        )
        prev_chi = stage_chi

    return envs, info
