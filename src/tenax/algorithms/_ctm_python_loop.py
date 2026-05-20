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
import numpy as np

from tenax.algorithms._ctm_env_pad import pad_dense_env_chi
from tenax.algorithms._ctm_tensor_convergence import (
    Coord,
    _corner_singular_values,
    _ctm_sv_diff,
    _ctm_tensor_sweep_multisite,
    _get_base_charges,
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
    max_smallest_S: float = 0.0  # variPEPS norm_smallest_S indicator (#492)
    final_chi: int = 0  # final chi after any in-CTM bumps (#492); 0 ⇒ unchanged


# Process-lifetime cache so repeat calls with the same neighbors dict reuse
# the same compiled @jit'd ``_step`` function.  Without this, every callsite
# (forward CTM convergence, implicit-AD f_bwd, line search, etc.) creates a
# fresh ``_step`` closure with its own JIT cache and pays redundant compile
# cost.  Diagnosed in docs/plans/2026-05-09-ipeps-ad-jit-cost-diagnosis.md.
# Keyed by id(neighbors) — safe because neighbors dicts are constructed once
# per optimizer invocation and stay alive throughout.
_JIT_STEP_CACHE: dict[int, callable] = {}


def _make_jit_ctm_step(
    neighbors: dict[Coord, dict[str, Coord]],
):
    """Create a JIT-compiled CTM step function for a given neighbor topology.

    The ``neighbors`` dict is captured in the closure so it is not traced
    by JAX (it contains only Python-level coordinate tuples, not arrays).

    Memoised by ``id(neighbors)`` so all callsites in a single optimizer
    invocation share one ``_step`` function and its JIT cache.

    Returns:
        A JIT-compiled function with signature::

            step(site_tensors, envs, *, chi, projector_method,
                 renormalize, projector_backward)
                 -> tuple[dict[Coord, CTMTensorEnv], jax.Array]

        The returned tuple is ``(new_envs, max_truncation_error)`` where
        ``max_truncation_error`` is the largest singular-value truncation
        error (ε_T) observed across all projector computations in the sweep.
    """
    cache_key = id(neighbors)
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
        plateau_patience:  Early-bail when the running minimum of the
                           convergence metric (``sv_diff`` or elementwise
                           ``max_diff``) has not improved over the last
                           ``plateau_patience`` iterations.  The loop
                           returns the env that achieved the best metric
                           and ``CTMConvergeInfo.converged=False`` — the
                           bail is a stop-loss, not a fixed point.
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
        )

    # Build the JIT'd step function (captures neighbors in closure)
    jit_step = _make_jit_ctm_step(neighbors)

    # chi may grow during the loop when ``ctmrg_heuristic_increase_chi``
    # is enabled (variPEPS-style in-CTM bump; Issue #492).  ``chi_current``
    # is the live value used by JIT'd sweeps; the new chi is static_argname,
    # so each distinct chi value will retrace once on first use.
    chi_current = chi
    # When the bump is enabled, require an explicit ``chi_max`` ceiling;
    # otherwise ``chi_max_eff == chi_current`` would make the growth guard
    # always False and the feature would silently no-op (codex review on
    # PR #513).  The CTMConfig path catches this at config construction;
    # this is defense-in-depth for direct callers of
    # ``python_loop_ctm_converge``.
    if ctmrg_heuristic_increase_chi and chi_max is None:
        raise ValueError(
            "ctmrg_heuristic_increase_chi=True requires chi_max to be set; "
            "without an explicit ceiling the in-CTM bump would silently "
            "no-op (chi can never grow above its initial value)."
        )
    chi_max_eff = chi_max if chi_max is not None else chi_current

    # Pre-compute base_charges from any site tensor for the SymmetricTensor
    # env-pad path; on dense envs ``pad_dense_env_chi`` ignores this arg.
    bump_base_charges: np.ndarray | None = None
    if ctmrg_heuristic_increase_chi:
        for A in site_tensors.values():
            bump_base_charges = _get_base_charges(_build_double_layer_tensor(A))
            if bump_base_charges is not None:
                break

    # Initialize environments
    envs = (
        env_init
        if env_init is not None
        else {
            c: initialize_ctm_tensor_env(A, chi_current)
            for c, A in site_tensors.items()
        }
    )

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

    remaining = max_iter - warmup
    prev_svs: dict[Coord, jax.Array] = {}
    prev_envs: dict[Coord, CTMTensorEnv] | None = None
    final_diff = float("inf")
    last_max_eps: float = 0.0
    last_max_smallest_S: float = 0.0
    best_diff = float("inf")
    best_envs: dict[Coord, CTMTensorEnv] | None = None
    best_iter = 0
    iters_since_best = 0

    for i in range(remaining):
        envs, _max_eps, _max_S = jit_step(
            site_tensors,
            envs,
            chi=chi_current,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )
        last_max_eps = float(_max_eps)
        last_max_smallest_S = float(_max_S)

        # variPEPS-style in-CTM χ-bump (Issue #492).  If the projector
        # SVD's normalised smallest kept SV is still above threshold, the
        # truncation cut hasn't reached a clean gap — grow chi, zero-pad
        # the env to the new shape, and immediately re-sweep at the new
        # chi so the env we hold is converged-at-current-chi rather than
        # a half-formed zero-padded one.  Without the post-bump sweep, a
        # bump on the last iteration (``i == remaining - 1``) would exit
        # with a zero-padded env that's never been swept at the new chi
        # (codex review on PR #513).
        if (
            ctmrg_heuristic_increase_chi
            and last_max_smallest_S > ctmrg_heuristic_increase_chi_threshold
            and chi_current < chi_max_eff
        ):
            chi_current = min(
                chi_current + ctmrg_heuristic_increase_chi_step_size,
                chi_max_eff,
            )
            envs = {
                c: pad_dense_env_chi(
                    envs[c], chi_current, base_charges=bump_base_charges
                )
                for c in envs
            }
            # Force-sweep once at the new chi so the returned env is
            # always swept at chi_current, even if this is the last iter.
            envs, _max_eps, _max_S = jit_step(
                site_tensors,
                envs,
                chi=chi_current,
                projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
            last_max_eps = float(_max_eps)
            last_max_smallest_S = float(_max_S)
            if gauge_fix_fn is not None:
                envs = {c: gauge_fix_fn(envs[c]) for c in envs}
            # Reset plateau tracking — diff metrics across a chi-bump are
            # not meaningful.  Best-tracking restarts at the new chi.
            prev_svs = {}
            prev_envs = None
            best_diff = float("inf")
            best_envs = None
            iters_since_best = 0
            continue

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

        # ``plateau_metric_valid`` flags whether ``final_diff`` was computed
        # from a real comparison this iter (vs left at the sentinel value
        # because no baseline was available yet).  Without this guard the
        # plateau block would treat the SV first-iter sentinel ``0.0`` as
        # ``best_diff``, then bail because no real positive diff can ever
        # improve on zero (codex review on PR #439).
        plateau_metric_valid = False

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
            plateau_metric_valid = True
        else:
            # SV convergence (default): corner singular value difference.
            # On the very first SV check (no entry in ``prev_svs`` yet —
            # possible when ``min_iter <= 1``), we have no real baseline,
            # so the plateau block must skip tracking this iter.
            have_prev_svs = bool(prev_svs)
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
            if have_prev_svs:
                final_diff = max_diff
                plateau_metric_valid = True
            # else: final_diff retains its previous value (inf on entry),
            # converged is False (no baseline), and the plateau block
            # below is skipped via plateau_metric_valid.

        if converged:
            return envs, CTMConvergeInfo(
                converged=True,
                iterations=total_iter,
                sv_diff=final_diff,
                max_truncation_error=last_max_eps,
                max_smallest_S=last_max_smallest_S,
                final_chi=chi_current,
            )

        if plateau_patience is not None and plateau_metric_valid:
            if final_diff < best_diff:
                best_diff = final_diff
                best_envs = {c: envs[c] for c in envs}
                best_iter = total_iter
                iters_since_best = 0
            else:
                iters_since_best += 1
                if iters_since_best >= plateau_patience:
                    return best_envs or envs, CTMConvergeInfo(
                        converged=False,
                        iterations=best_iter or total_iter,
                        sv_diff=best_diff,
                        max_truncation_error=last_max_eps,
                        max_smallest_S=last_max_smallest_S,
                        final_chi=chi_current,
                    )

    return envs, CTMConvergeInfo(
        converged=False,
        iterations=max_iter,
        sv_diff=final_diff,
        max_truncation_error=last_max_eps,
        max_smallest_S=last_max_smallest_S,
        final_chi=chi_current,
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
        )
        prev_chi = stage_chi

    return envs, info
