"""Shared bump-aware CTM convergence loop.

Consumed by python_loop_ctm_converge, _sigma_gauged_ctm_converge (implicit-AD
forward), and ctm_energy_explicit warmup.  Centralizing the bump pad+resweep
sequence keeps the variPEPS-style growth contract (#492) in one place across
all three forward CTM paths (#514).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

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
    _forced_corner_rank,
    _get_base_charges,
    _max_env_leaf_diff,
    _max_virtual_bond_dim,
    _nan_safe_max,
    _tensor_leaf_data,
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
    # Signed estimate of the dominant multiplier of the gauged CTM step on the
    # last two sweeps, ``Re<d_n, d_{n-1}> / |d_{n-1}|^2`` with
    # ``d_n = gauge(step(e_n)) - e_n`` (#1060).  Near ``+rho`` (0 < rho < 1)
    # for a slow contraction; near ``-1`` for a two-state cycle around an
    # unstable fixed point, which ``mixing`` can stabilize.  NaN when it is
    # not measurable (``conv_method="sv"``, fewer than two measured sweeps,
    # or a block-layout change between sweeps).
    step_multiplier: float = float("nan")


def _env_leaf_data(envs: dict) -> list:
    """Numeric buffers of every env leaf, in a fixed coordinate order."""
    return [
        _tensor_leaf_data(leaf)
        for c in sorted(envs)
        for leaf in jax.tree.leaves(envs[c])
    ]


def _step_multiplier(start_n, fixed_n, start_prev, fixed_prev) -> float:
    """Signed multiplier ``Re<d_n, d_{n-1}> / |d_{n-1}|^2`` (#1060).

    ``d_k = fixed_k - start_k`` is the undamped residual of sweep ``k``.  With
    ``d_n = rho d_{n-1}`` this returns ``rho`` exactly; its sign separates a
    two-state cycle (rho near -1) from a slow contraction (0 < rho < 1),
    which the unsigned element-wise residual cannot.  Returns NaN when the
    four environments do not share one leaf layout.
    """
    legs = [_env_leaf_data(e) for e in (fixed_n, start_n, fixed_prev, start_prev)]
    if len({len(x) for x in legs}) != 1 or any(
        len({jnp.shape(a) for a in group}) != 1 for group in zip(*legs)
    ):
        return float("nan")
    # Reduce on device and move only the two scalars to the host: this runs
    # on every loop exit, mixing or not (Codex P2 on #1061).
    num = jnp.zeros(())
    den = jnp.zeros(())
    for fn, sn, fp, sp in zip(*legs):
        cur = fn - sn
        prev = fp - sp
        num = num + jnp.real(jnp.vdot(prev, cur))
        den = den + jnp.real(jnp.vdot(prev, prev))
    num, den = float(num), float(den)
    return num / den if den > 0.0 else float("nan")


def _mix_envs(fixed: dict, start: dict, mixing: float) -> dict:
    """``(1 - mixing) * fixed + mixing * start``, leaf by leaf (#1060)."""
    return {
        c: jax.tree.map(
            lambda a, b: (1.0 - mixing) * a + mixing * b, fixed[c], start[c]
        )
        for c in fixed
    }


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
    mixing: float = 0.0,
) -> CTMLoopResult:
    """Run CTM sweeps with optional variPEPS-style in-CTM chi-bump.

    Mirrors the loop in python_loop_ctm_converge (lines 299-519 prior to
    extraction).  Caller is responsible for warmup, env_init validation,
    and (chi_max, chi_current) constraints.

    gauge_fix_fn:
        Callable (envs_new, envs_old) -> envs, or None.  Phase gauge wraps a
        single-arg phase fix; sigma gauge uses both args.  None disables.

    mixing:
        Linear mixing ``beta`` in ``[0, 1)`` (#1060).  ``0`` (default) is the
        plain iteration.  ``beta > 0`` iterates
        ``e <- (1 - beta) * gauge(step(e)) + beta * e``, which has the same
        fixed points as the plain map but turns a step multiplier ``lambda``
        into ``(1 - beta) * lambda + beta``: a two-state cycle around an
        unstable fixed point (``lambda < -1``) contracts once ``beta`` is
        large enough.  Convergence is still certified on the *undamped*
        residual ``|gauge(step(e)) - e|``, so a converged result is a fixed
        point of the plain gauged step, the premise of the implicit adjoint.
        Requires ``conv_method="elementwise"`` and a ``gauge_fix_fn``: mixing
        is element-wise, so it is meaningful only between gauge-aligned
        environments.
    """
    if not 0.0 <= mixing < 1.0:
        raise ValueError(f"mixing must be in [0, 1), got {mixing!r}")
    if mixing > 0.0 and conv_method != "elementwise":
        raise ValueError(
            "mixing > 0 requires conv_method='elementwise': the convergence "
            "test must measure the undamped element-wise residual"
        )
    if mixing > 0.0 and gauge_fix_fn is None:
        raise ValueError(
            "mixing > 0 requires a gauge fix (e.g. forward_gauge='bond_phase'): "
            "element-wise mixing of environments in unrelated gauges is "
            "meaningless"
        )
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
    # (start, gauge(step(start))) of the last two measured sweeps, for the
    # signed step multiplier (#1060).
    last_pair: tuple[dict, dict] | None = None
    prev_pair: tuple[dict, dict] | None = None
    # Last gauge(step(start)).  With mixing the loop variable ``envs`` holds
    # the mixed iterate, so an exhausted budget returns this instead: a plain
    # CTM output, not a blend no step produced.
    envs_fixed = envs_init

    def _multiplier() -> float:
        if last_pair is None or prev_pair is None:
            return float("nan")
        mu = _step_multiplier(*last_pair, *prev_pair)
        # Under mixing the residual ratio is the mixed map's multiplier
        # (1 - beta) lam + beta; report the plain step's lam, so the number
        # means the same thing whether or not mixing is on.
        return (mu - mixing) / (1.0 - mixing)

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
            # The bump changes chi: residuals across it share no layout.
            last_pair = prev_pair = None
            envs_fixed = envs
            continue

        if gauge_fix_fn is not None:
            envs = gauge_fix_fn(envs_new, envs_at_iter_start)
        else:
            envs = envs_new
        # ``envs`` is now gauge(step(start)); keep the pair for the multiplier.
        envs_fixed = envs
        if conv_method == "elementwise":
            prev_pair, last_pair = last_pair, (envs_at_iter_start, envs_fixed)

        total_iter = i + 1 + bump_extra_sweeps
        if total_iter < min_iter:
            if conv_method == "sv":
                for c in sorted(envs):
                    prev_svs[c] = _corner_singular_values(envs[c].C1)
            else:
                prev_envs = {c: envs[c] for c in envs}
            if mixing > 0.0:
                envs = _mix_envs(envs_fixed, envs_at_iter_start, mixing)
            continue

        plateau_metric_valid = False
        if conv_method == "elementwise":
            if prev_envs is None and mixing == 0.0:
                prev_envs = {c: envs[c] for c in envs}
                continue
            # With mixing the stored iterate is not gauge(step(start)), so the
            # residual is measured directly against this sweep's start: the
            # undamped |gauge(step(e)) - e|.  Without mixing ``prev_envs`` is
            # that same start, so both branches measure one quantity.
            reference = envs_at_iter_start if mixing > 0.0 else prev_envs
            max_diff = 0.0
            for c in sorted(envs):
                max_diff = _nan_safe_max(
                    max_diff, _max_env_leaf_diff(reference[c], envs[c])
                )
            converged = max_diff < conv_tol
            final_diff = max_diff
            prev_envs = {c: envs[c] for c in envs}
            # Same rule on the elementwise metric: a NaN leaf makes ``max_diff``
            # non-finite, and ``inf < best_diff`` is False just as
            # ``nan < best_diff`` is, so both would silently count as
            # "no improvement" and burn the patience budget.
            plateau_metric_valid = math.isfinite(max_diff)
        else:
            have_prev_svs = bool(prev_svs)
            # #903 P1: rank 1 is a collapse only if more was reachable.
            # Per coordinate, not per cell (#903 review): a cell-wide
            # aggregate fails open with `min` and wrongly closed with `max`.
            # #903 P1: a corner can be built from a NEIGHBOUR's tensor in the
            # 2x2 recipe, so a bound keyed on the storage coordinate can accept
            # a collapsed corner.  `neighbors` is not in scope here, so this
            # takes the max over every site -- a strict superset of any
            # coordinate's contributors, and therefore conservative: a larger
            # bound only makes the exemption harder to obtain.
            _mr_all = _forced_corner_rank(
                max(_max_virtual_bond_dim(A) ** 2 for A in site_tensors.values())
            )
            converged = True
            max_diff = 0.0
            for c in sorted(envs):
                sv = _corner_singular_values(envs[c].C1)
                if c in prev_svs:
                    diff = float(_ctm_sv_diff(sv, prev_svs[c], max_rank=_mr_all))
                    max_diff = _nan_safe_max(max_diff, diff)
                    if diff >= conv_tol:
                        converged = False
                else:
                    converged = False
                prev_svs[c] = sv
            if have_prev_svs:
                final_diff = max_diff
                # #903 review P1: a blind corner makes ``_ctm_sv_diff`` return
                # ``inf``, which is the criterion refusing to certify -- NOT a
                # measurement that the environment stopped improving.  Feeding
                # it to the plateau counter reads "no improvement" every sweep,
                # so ``iters_since_best`` reaches ``plateau_patience`` and the
                # loop bails with the collapsed environment it was supposed to
                # keep sweeping past.  #898's own fixture recovers at sweep 41
                # and the default patience is 20, so the bail fires first and
                # the fix defeats itself.
                #
                # An unmeasurable sweep is ineligible, not unimproved: the
                # budget runs on and ``max_iter`` decides.
                plateau_metric_valid = math.isfinite(max_diff)

        # The residual just measured belongs to this sweep's start.  Without
        # mixing that start is the previous output and ``envs`` is its own
        # contraction, so returning ``envs`` is conventional; with mixing the
        # plain step may expand (|lam| > 1 is what mixing is for), so return
        # the iterate that was actually certified (Codex P2 on #1061).
        certified = envs_at_iter_start if mixing > 0.0 else envs
        if converged:
            return CTMLoopResult(
                envs=certified,
                converged=True,
                iterations=total_iter,
                sv_diff=final_diff,
                max_truncation_error=last_max_eps,
                max_smallest_S=last_max_smallest_S,
                final_chi=chi_current,
                bump_extra_sweeps=bump_extra_sweeps,
                best_iteration=total_iter,
                step_multiplier=_multiplier(),
            )

        if plateau_patience is not None and plateau_metric_valid:
            if final_diff < best_diff:
                best_diff = final_diff
                best_envs = {c: certified[c] for c in certified}
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
                        step_multiplier=_multiplier(),
                    )

        if mixing > 0.0:
            envs = _mix_envs(envs_fixed, envs_at_iter_start, mixing)

    return CTMLoopResult(
        envs=envs_fixed if mixing > 0.0 else envs,
        converged=False,
        iterations=remaining,
        sv_diff=final_diff,
        max_truncation_error=last_max_eps,
        max_smallest_S=last_max_smallest_S,
        final_chi=chi_current,
        bump_extra_sweeps=bump_extra_sweeps,
        best_iteration=remaining,
        step_multiplier=_multiplier(),
    )
