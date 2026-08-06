"""Shared bump-aware CTM convergence loop.

Consumed by python_loop_ctm_converge, _sigma_gauged_ctm_converge (implicit-AD
forward), and ctm_energy_explicit warmup.  Centralizing the bump pad+resweep
sequence keeps the variPEPS-style growth contract (#492) in one place across
all three forward CTM paths (#514).
"""

from __future__ import annotations

__all__ = [
    "CTMLoopResult",
    "_run_ctm_loop_with_bump",
    "_validate_chi_bump_args",
]

from typing import NamedTuple

from tenax.algorithms._ctm_env_pad import pad_dense_env_chi
from tenax.algorithms._ctm_tensor_convergence import (
    Coord,
    _corner_singular_values,
    _ctm_sv_diff,
    _get_base_charges,
    _max_env_leaf_diff,
)
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
)


def _validate_chi_bump_args(
    *,
    chi: int,
    chi_max: int | None,
    env_init,
    bump_enabled: bool,
    bump_step_size: int,
) -> int:
    """Validate bump-related args and return the finalized ``chi_current``.

    Centralises the validation that originally lived in three sibling
    forward-CTM modules (#514 follow-up de-dup).  ``chi_current`` equals
    ``chi`` by default, but when the in-CTM bump is enabled and
    ``env_init`` carries a larger χ, it is promoted to that env's χ so a
    warm-start round-trip does not silently down-truncate the env.

    Raises ``ValueError`` on:

    * ``bump_enabled`` and ``chi_max is None`` — without an explicit
      ceiling the in-CTM bump would silently no-op (``chi_max_eff``
      defaults to ``chi_current`` and the growth guard is always False).
    * ``bump_enabled`` and ``bump_step_size <= 0`` — would either stall
      (``== 0``: bump fires every iter with chi unchanged → infinite
      loop) or attempt an invalid shrink (``< 0``).
    * ``bump_enabled`` and ``env_init`` carrying χ above ``chi_max`` —
      a warm-start env above the configured ceiling is a
      misconfiguration; we surface it rather than silently clamp.
    * ``chi_max < chi_current`` after the env_init finalize — defense in
      depth for direct callers bypassing :class:`CTMConfig`'s constructor.
    """
    if bump_enabled and chi_max is None:
        raise ValueError(
            "ctmrg_heuristic_increase_chi=True requires chi_max to be set; "
            "without an explicit ceiling the in-CTM bump would silently "
            "no-op (chi can never grow above its initial value)."
        )
    if bump_enabled and bump_step_size <= 0:
        raise ValueError(
            "ctmrg_heuristic_increase_chi_step_size must be a positive "
            f"integer, got {bump_step_size}"
        )

    chi_current = chi
    if bump_enabled and env_init:
        try:
            sample_env = next(iter(env_init.values()))
            env_chi = int(sample_env.C1.indices[0].dim)
        except (StopIteration, AttributeError, IndexError):
            env_chi = None  # malformed env_init; let downstream raise
        if env_chi is not None:
            if chi_max is not None and env_chi > chi_max:
                raise ValueError(
                    f"env_init has chi={env_chi} which exceeds the "
                    f"configured chi_max={chi_max}. Either raise chi_max "
                    "or supply a warm-start env that respects the ceiling."
                )
            if env_chi > chi_current:
                chi_current = env_chi
    if chi_max is not None and chi_max < chi_current:
        raise ValueError(
            f"chi_max ({chi_max}) must be >= chi_current ({chi_current}). "
            "chi_current is the max of the input ``chi`` and (when the "
            "in-CTM bump is enabled) env_init's actual chi; chi_max is "
            "the ceiling and must not be smaller."
        )
    # chi_current is promoted by env_init when warm-start env exceeded the
    # requested chi (so warm-start round-trips preserve grown chi).
    return chi_current


class CTMLoopResult(NamedTuple):
    """Outcome of one bump-aware CTM convergence loop run."""

    envs: dict[Coord, CTMTensorEnv]
    converged: bool
    iterations: int
    sv_diff: float
    max_truncation_error: float
    max_smallest_S: float
    final_chi: int
    bump_extra_sweeps: int
    # Sweep index whose environment is the one returned in ``envs``.  Equal
    # to ``iterations`` on the converged and budget-exhausted paths; on the
    # ``plateau_patience`` bail it is the best-metric sweep, which trails
    # ``iterations`` by exactly ``plateau_patience``.  Split out from
    # ``iterations`` in #781, which reported the best-metric index as the
    # sweep count and so inflated every ``total_s / iterations`` per-sweep
    # timing derived from a bailed run.
    best_iteration: int = 0


def _run_ctm_loop_with_bump(
    jit_step,
    site_tensors,
    envs_init,
    *,
    chi_current: int,
    chi_max: int | None,
    bump_enabled: bool,
    bump_threshold: float,
    bump_step_size: int,
    projector_method: str,
    renormalize: bool,
    projector_backward: str,
    gauge_fix_fn,
    max_iter: int,
    min_iter: int,
    conv_tol: float,
    conv_method: str,
    plateau_patience: int | None,
) -> CTMLoopResult:
    """Run CTM sweeps with optional variPEPS-style in-CTM chi-bump.

    Mirrors the loop in python_loop_ctm_converge (lines 299-519 prior to
    extraction).  Caller is responsible for warmup, env_init validation,
    and (chi_max, chi_current) constraints.

    gauge_fix_fn:
        Callable (envs_new, envs_old) -> envs, or None.  Phase gauge wraps a
        single-arg phase fix; sigma gauge uses both args.  None disables.
    """
    # Compute base_charges for the symmetric env-pad path; ignored by dense
    # envs.  Cost is one D⁴ contraction per CTM-converge invocation — same
    # total work as before the helper consolidation (was previously done
    # once per direct-caller callsite).
    bump_base_charges = None
    if bump_enabled:
        for A in site_tensors.values():
            bump_base_charges = _get_base_charges(_build_double_layer_tensor(A))
            if bump_base_charges is not None:
                break

    chi_max_eff = chi_max if chi_max is not None else chi_current
    envs = envs_init
    remaining = max_iter

    prev_svs: dict = {}
    prev_envs: dict | None = None
    final_diff = float("inf")
    last_max_eps = 0.0
    last_max_smallest_S = 0.0
    best_diff = float("inf")
    best_envs: dict | None = None
    best_iter = 0
    iters_since_best = 0
    bump_extra_sweeps = 0

    for i in range(remaining):
        if i + bump_extra_sweeps >= remaining:
            break
        # Capture start-of-iter env: sigma-gauge alignment requires the prior
        # iteration's env as the second arg to gauge_fix_fn (transfer-matrix
        # eigenvector reference).  Phase gauge ignores the second arg.
        envs_at_iter_start = envs
        envs_new, _max_eps, _max_S = jit_step(
            site_tensors,
            envs,
            chi=chi_current,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )
        last_max_eps = float(_max_eps)
        last_max_smallest_S = float(_max_S)

        bump_would_fire = (
            bump_enabled
            and last_max_smallest_S > bump_threshold
            and chi_current < chi_max_eff
        )
        if bump_would_fire and (i + 1 + bump_extra_sweeps < remaining):
            chi_current = min(chi_current + bump_step_size, chi_max_eff)
            envs = {
                c: pad_dense_env_chi(
                    envs_new[c], chi_current, base_charges=bump_base_charges
                )
                for c in envs_new
            }
            envs, _max_eps, _max_S = jit_step(
                site_tensors,
                envs,
                chi=chi_current,
                projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
            bump_extra_sweeps += 1
            last_max_eps = float(_max_eps)
            last_max_smallest_S = float(_max_S)
            if gauge_fix_fn is not None:
                envs = gauge_fix_fn(envs, envs_at_iter_start)
            prev_svs = {}
            prev_envs = None
            best_diff = float("inf")
            best_envs = None
            iters_since_best = 0
            continue

        if gauge_fix_fn is not None:
            envs = gauge_fix_fn(envs_new, envs_at_iter_start)
        else:
            envs = envs_new

        total_iter = i + 1 + bump_extra_sweeps
        if total_iter < min_iter:
            if conv_method == "sv":
                for c in sorted(envs):
                    prev_svs[c] = _corner_singular_values(envs[c].C1)
            else:
                prev_envs = {c: envs[c] for c in envs}
            continue

        plateau_metric_valid = False
        if conv_method == "elementwise":
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

        if converged:
            return CTMLoopResult(
                envs=envs,
                converged=True,
                iterations=total_iter,
                sv_diff=final_diff,
                max_truncation_error=last_max_eps,
                max_smallest_S=last_max_smallest_S,
                final_chi=chi_current,
                bump_extra_sweeps=bump_extra_sweeps,
                best_iteration=total_iter,
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
                    return CTMLoopResult(
                        envs=best_envs or envs,
                        converged=False,
                        # Sweeps performed, not the best-metric index: the
                        # bail happens ``plateau_patience`` sweeps after the
                        # last improvement and callers divide elapsed time by
                        # this to get a per-sweep cost (#781).
                        iterations=total_iter,
                        sv_diff=best_diff,
                        max_truncation_error=last_max_eps,
                        max_smallest_S=last_max_smallest_S,
                        final_chi=chi_current,
                        bump_extra_sweeps=bump_extra_sweeps,
                        best_iteration=best_iter or total_iter,
                    )

    return CTMLoopResult(
        envs=envs,
        converged=False,
        iterations=remaining,
        sv_diff=final_diff,
        max_truncation_error=last_max_eps,
        max_smallest_S=last_max_smallest_S,
        final_chi=chi_current,
        bump_extra_sweeps=bump_extra_sweeps,
        best_iteration=remaining,
    )
