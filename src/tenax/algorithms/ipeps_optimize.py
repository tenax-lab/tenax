"""AD-based ground state optimization for iPEPS.

Extracts optimize_gs_ad and related helpers from ipeps.py.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time as _time
from dataclasses import replace as _replace

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_energy_ad import invalidate_implicit_ad_warm_start
from tenax.algorithms._ctm_env_pad import pad_dense_env_chi
from tenax.algorithms.ipeps_ad_policy import (
    build_ad_ctm_config,
    resolve_projector_backward,
    use_reference_c4v_path,
)
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.lattice import Lattice
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

_logger = logging.getLogger(__name__)

Coord = tuple[int, int]


def _env_dict_chi(envs) -> int | None:
    """Extract chi from a ``coord -> CTMTensorEnv`` mapping.

    Returns ``None`` if the input is falsy or malformed (missing
    ``C1.indices[0].dim``).  Used by the finalize-pad sites to derive
    the *actual* chi an env carries — the in-CTM χ-bump
    (``ctmrg_heuristic_increase_chi=True``, #492/#514) grows chi inside
    ``python_loop_ctm_converge`` without updating the optimizer's local
    ``ctm_cfg.chi``.

    Accepts the bare env dict (``{coord: CTMTensorEnv}``), not an
    env_cache wrapper — callers should index ``env_cache["envs"]``
    themselves so both live cache and snapshot dicts can be queried
    with the same helper (Codex P2 on PR #542: a cleared live cache
    must not mask a bumped best-env snapshot).
    """
    if not envs:
        return None
    try:
        sample = next(iter(envs.values()))
        return int(sample.C1.indices[0].dim)
    except (AttributeError, IndexError, StopIteration):
        return None


def _restore_env_cache_after_line_search(env_cache: dict, snapshot: tuple) -> None:
    """Revert ``env_cache["envs"]`` to the pre-line-search snapshot.

    ``loss_fn_fwd`` writes its converged env to ``env_cache["envs"]`` to
    enable φ/dφ env sharing (#502).  Those writes are visible to nested
    probes within the line search but must not persist past the block —
    a rejected last probe (or a probe ending at a different α than the
    accepted one) would otherwise leak an under-converged env into the
    next iteration's ``value_and_grad`` warm-start (Codex P1 on PR
    #538).

    ``snapshot`` is the ``(had_envs, envs)`` pair captured before the
    line search via ``("envs" in env_cache, env_cache.get("envs"))``.
    """
    had_envs, envs = snapshot
    if had_envs:
        env_cache["envs"] = envs
    else:
        env_cache.pop("envs", None)


def _drop_env_cache_for_reset(env_cache: dict) -> None:
    """Clear the env warm-start cache AND the implicit-AD λ warm-start seed.

    Called from stall-recovery rollbacks (``params = best_params`` paths)
    and checkpoint restore.  After these events the env that the next CTM
    sweep produces decouples from whatever the implicit-AD backward
    cached, so the Neumann seed in ``_VJP_CACHE`` is no longer a good
    initial guess (issue #501).  Bump-driven env shape changes are
    handled by ``_apply_chi_bump`` directly.
    """
    env_cache.clear()
    invalidate_implicit_ad_warm_start()


def _apply_chi_bump(
    ctm_cfg: CTMConfig,
    env_cache: dict,
    chi_new: int,
    *,
    base_charges: np.ndarray | None = None,
) -> tuple[CTMConfig, dict]:
    """Pure mechanism: bump logical χ and pad cached envs in-place.

    Used by both the reactive auto-χ_E bump (``_maybe_bump_chi``,
    variPEPS §2.8.2) and the scheduled bump driven by
    ``gs_chi_schedule_steps`` (``_advance_chi_stage_if_due``, issue #455).
    No policy here — callers decide *whether* and *to what* to bump.

    ``env_cache`` is mutated in-place so closures that captured the
    dict reference (notably ``env_cache`` inside ``make_ctm_energy_fn``
    in ``optimize_gs_ad``) see the padded envs without rebinding.

    For SymmetricTensor envs, ``base_charges`` should be the bond
    charges of the iPEPS A tensor (the same ``base_charges`` the
    symmetric projector consumes).  Ignored on the dense path.
    """
    new_cfg = dataclasses.replace(ctm_cfg, chi=chi_new)
    if "envs" in env_cache:
        env_cache["envs"] = {
            c: pad_dense_env_chi(
                env_cache["envs"][c], chi_new, base_charges=base_charges
            )
            for c in env_cache["envs"]
        }
    # Issue #501: env shape just changed, so the cached Neumann warm-start
    # λ lives in the wrong vector space.  The shape-mismatch guard inside
    # ``f_bwd`` already drops it on the next call, but doing so here is
    # cheap and defensive (covers symmetric-tensor padding paths whose
    # leaf shape may not change even though the logical χ did).
    invalidate_implicit_ad_warm_start()
    return new_cfg, env_cache


def _maybe_bump_chi(
    ctm_cfg: CTMConfig,
    env_cache: dict,
    last_eps_t: float,
    *,
    last_smallest_S: float | None = None,
    base_charges: np.ndarray | None = None,
) -> tuple[CTMConfig, dict]:
    """variPEPS §2.8.2 reactive χ_E bump.

    When ``ctm_cfg.chi_auto_bump`` is enabled and the chosen indicator
    (``last_eps_t`` or ``last_smallest_S`` per
    ``ctm_cfg.chi_auto_bump_metric``) exceeds ``ctm_cfg.chi_auto_bump_eps``,
    return a new ``(ctm_cfg, env_cache)`` pair with χ raised by
    ``chi_auto_bump_step`` (capped at ``chi_max`` if set).  The cached env
    is zero-padded to the new χ.  Otherwise the input pair is returned
    unchanged.

    ``last_smallest_S`` is only consulted when
    ``ctm_cfg.chi_auto_bump_metric == "norm_smallest_S"`` (Tenax extension
    matching variPEPS's literal trigger).  When the metric is selected
    but the caller has not plumbed the indicator (``last_smallest_S is
    None``), the bump silently falls back to the ``last_eps_t`` (default
    ``eps_T``) signal so non-2-site paths still grow χ reactively.  This
    matches the ``CTMConfig.chi_auto_bump_metric`` docstring contract.

    See ``_apply_chi_bump`` for the in-place mutation contract.

    For SymmetricTensor envs, ``base_charges`` should be the bond
    charges of the iPEPS A tensor (the same ``base_charges`` the
    symmetric projector consumes).  Ignored on the dense path.
    """
    if not ctm_cfg.chi_auto_bump:
        return ctm_cfg, env_cache
    metric = ctm_cfg.chi_auto_bump_metric
    if metric == "norm_smallest_S" and last_smallest_S is not None:
        # variPEPS-literal trigger when the caller supplies the indicator.
        indicator = float(last_smallest_S)
        label = "norm_smallest_S"
    else:
        # Default ``eps_T`` path and fallback for callers (non-2-site paths)
        # that have not been plumbed for ``norm_smallest_S`` yet — see the
        # ``CTMConfig.chi_auto_bump_metric`` doc.  Mapping a missing
        # indicator to 0.0 would suppress all bumps on unsupported paths.
        indicator = last_eps_t
        label = "eps_T"
    _logger.debug(
        "_maybe_bump_chi: chi=%d %s=%.3e threshold=%.3e",
        ctm_cfg.chi,
        label,
        indicator,
        ctm_cfg.chi_auto_bump_eps,
    )
    if indicator <= ctm_cfg.chi_auto_bump_eps:
        return ctm_cfg, env_cache
    chi_new = ctm_cfg.chi + ctm_cfg.chi_auto_bump_step
    if ctm_cfg.chi_max is not None:
        chi_new = min(chi_new, ctm_cfg.chi_max)
    if chi_new <= ctm_cfg.chi:
        _logger.info(
            "_maybe_bump_chi: at chi_max=%d ceiling; %s=%.3e exceeded threshold %.3e but no headroom",
            ctm_cfg.chi,
            label,
            indicator,
            ctm_cfg.chi_auto_bump_eps,
        )
        return ctm_cfg, env_cache  # at ceiling
    _logger.info(
        "_maybe_bump_chi: reactive bump chi %d -> %d (%s=%.3e > threshold=%.3e)",
        ctm_cfg.chi,
        chi_new,
        label,
        indicator,
        ctm_cfg.chi_auto_bump_eps,
    )
    return _apply_chi_bump(ctm_cfg, env_cache, chi_new, base_charges=base_charges)


def _advance_chi_stage_if_due(
    ctm_cfg: CTMConfig,
    env_cache: dict,
    *,
    chi_schedule: list[tuple[int, int]] | None,
    current_stage_idx: int,
    steps_in_stage: int,
    config: iPEPSConfig,
    grad_norm: float,
    delta_energy: float,
    stall_count: int,
    base_charges: np.ndarray | None = None,
) -> tuple[CTMConfig, dict, int, bool, bool]:
    """Decide whether to advance to the next χ stage and apply it (#455).

    Three signals trigger an advance at non-final stages:
        - ``steps_in_stage >= max_steps`` (budget; existing).
        - ``_converged_outer(config, delta_energy, grad_norm)``
          (NEW PR 2 — reuses user's gs_conv_criterion).
        - ``stall_count > config.gs_stall_recovery_retries`` AND
          ``config.gs_stall_recovery == "reset"`` (NEW PR 2 —
          gated to reset path; noise path has its own retries).
          Mirrors the existing reset-budget exit logic (the third
          retry only counts as exhausted on the *fourth* failed
          attempt, not the third — codex review #467).

    At the final stage all three trigger ``should_break=True`` with
    no bump (matches existing exit semantics).
    """
    if not chi_schedule:
        return ctm_cfg, env_cache, current_stage_idx, False, False

    _, stage_max_steps = chi_schedule[current_stage_idx]
    budget_exhausted = steps_in_stage >= stage_max_steps

    converged = _converged_outer(config, delta_energy, grad_norm)

    stall_exhausted = (
        config.gs_stall_recovery == "reset"
        and stall_count > config.gs_stall_recovery_retries
    )

    should_advance = budget_exhausted or converged or stall_exhausted
    if not should_advance:
        return ctm_cfg, env_cache, current_stage_idx, False, False

    has_next = (current_stage_idx + 1) < len(chi_schedule)
    if not has_next:
        return ctm_cfg, env_cache, current_stage_idx, False, True

    next_chi, _ = chi_schedule[current_stage_idx + 1]
    if ctm_cfg.chi_max is not None:
        next_chi = min(next_chi, ctm_cfg.chi_max)

    if next_chi <= ctm_cfg.chi:
        return ctm_cfg, env_cache, current_stage_idx + 1, False, False

    new_ctm_cfg, new_env_cache = _apply_chi_bump(
        ctm_cfg, env_cache, next_chi, base_charges=base_charges
    )
    return new_ctm_cfg, new_env_cache, current_stage_idx + 1, True, False


def _lattice_to_neighbors(
    lattice: Lattice,
) -> tuple[dict[Coord, dict[str, Coord]], dict[str, Coord], dict[Coord, str]]:
    """Convert a Lattice neighbor_map to coordinate-keyed dicts.

    Returns (neighbors, name_to_coord, coord_to_name).
    """
    name_to_coord: dict[str, Coord] = {
        name: (i, 0) for i, name in enumerate(lattice.sites)
    }
    coord_to_name: dict[Coord, str] = {v: k for k, v in name_to_coord.items()}
    neighbors: dict[Coord, dict[str, Coord]] = {
        name_to_coord[name]: {
            direction: name_to_coord[nb]
            for direction, nb in lattice.neighbor_map[name].items()
        }
        for name in lattice.sites
    }
    return neighbors, name_to_coord, coord_to_name


def _resolve_projector_backward(config: iPEPSConfig) -> iPEPSConfig:
    """Compatibility wrapper around the shared AD policy helper."""
    return resolve_projector_backward(config, logger=_logger)


def _warn_implicit_ad_variational_caveat(config: iPEPSConfig, *, path: str) -> None:
    """Emit a warning when implicit AD's variational guarantee is at risk.

    Implicit AD through the CTM fixed point is variational *only* when the
    env passed to the gradient evaluation is a converged fixed point at the
    current chi.  Scheduled ``chi_ramp`` advances and end-of-outer-step
    ``chi_auto_bump`` events zero-pad newly-active env rows, which violate
    that precondition until CTM repopulates them.  See issue #511.

    Args:
        config: The iPEPSConfig being dispatched.
        path: Short label for the call site (used in the warning text).
    """
    if not config.gs_implicit_ad:
        return
    ctm_cfg = config.ctm
    in_ctm_bump = getattr(ctm_cfg, "ctmrg_heuristic_increase_chi", False)
    if in_ctm_bump:
        return
    chi_ramped = getattr(ctm_cfg, "chi_ramp", None) is not None
    end_of_step_bump = getattr(ctm_cfg, "chi_auto_bump", False)
    if not (chi_ramped or end_of_step_bump):
        return
    import warnings

    trigger = []
    if chi_ramped:
        trigger.append("scheduled chi_ramp")
    if end_of_step_bump:
        trigger.append("end-of-outer-step chi_auto_bump")
    warnings.warn(
        f"{path} AD with gs_implicit_ad=True is variational **only when "
        "the CTM environment is a converged fixed point at the current "
        f"chi**.  The current config enables {' and '.join(trigger)}, "
        "which zero-pads env rows for newly-active chi indices; several "
        "gradient evaluations may be required before CTM repopulates them, "
        "during which the optimizer can descend to a non-physical ghost "
        "minimum (see issue #511).  Prefer ctmrg_heuristic_increase_chi=True "
        "(in-CTM bump, #492/#514) to grow chi during CTM convergence and "
        "preserve the variational guarantee across stages.",
        stacklevel=3,
    )


def _normalize_stall_recovery(config, *, unit_cell: str):
    """Auto-default ``gs_stall_recovery`` based on unit cell when unset.

    The 1-site C4v production path requires the noise kick to break out
    of the SU-init plateau (gradient norms ~1e-10 trip ``gs_conv_tol``
    before the first real step), so the 1-site default is ``"noise"``.

    The 2-site default is ``"reset"`` because best-energy snapshot
    rollback dominates raw noise injection near convergence on this
    path.  Empirically verified in #520 (PR #551) on the post-#494
    corrected energy: under both modes the trajectory stayed above the
    QMC reference (so the original "non-variational drift" pathology
    from #298 did **not** reproduce), but a 10%-Frobenius noise kick
    fired close to a settled energy can perturb the L-BFGS state by
    enough that re-descent does not recover before ``gs_num_steps``
    runs out.  Reset's best-snapshot fallback dodges this failure mode
    by construction, so it wins on this path by ~2.6e-2 in the
    canonical D=2 χ=8 probe.

    Note: the older justification — "noise interacts pathologically
    with non-variational CTM regions on 2-site (see #298)" — was made
    on the pre-#494 broken 2-site bipartite energy and does not hold
    on the corrected loss landscape.  The default itself is unchanged.
    """
    from dataclasses import replace

    if config.gs_stall_recovery is not None:
        return config
    # CG with map_fn optimizes a tuple of raw site tensors; the noise
    # injection path assumes a single tensor (calls .todense()/jnp.linalg.norm
    # on params), so default to "reset" for that case.
    cg_with_map_fn = (
        config.cg_gates is not None
        and getattr(config.cg_gates, "map_fn", None) is not None
    )
    if cg_with_map_fn:
        default = "reset"
    else:
        default = "noise" if unit_cell == "1x1" else "reset"
    return replace(config, gs_stall_recovery=default)


_STALL_NOISE_FLOOR: float = 1e-12
"""Absolute energy decrease below which a step is treated as numerical noise
rather than real progress.  See ``_is_real_decrease`` and issue #510.

Tied to the J=1 spin-Hamiltonian convention (energies are O(1)) so a 1e-12
threshold sits 4 orders of magnitude above double-precision round-off on a
typical -0.6 energy.  Not exposed as config: the existing
``gs_stall_count_threshold`` / ``gs_stall_recovery_retries`` knobs control
how the trigger interacts with recovery — the floor itself is a noise level,
not a tuning parameter.
"""


def _is_real_decrease(
    new_energy: float, old_energy: float, *, noise: float = _STALL_NOISE_FLOOR
) -> bool:
    """Return True iff ``new_energy`` is strictly below ``old_energy`` by more
    than ``noise``.

    Defends the stall safety-net against α-collapse line searches that
    return a step at α≈1e-14 with a floating-point-noise energy decrease.
    Without this guard the strict ``new < old`` comparison would reset
    ``stall_count`` on jitter alone and the optimizer could grind for
    hours without real progress (issue #510, v8b evidence).
    """
    if not math.isfinite(new_energy) or not math.isfinite(old_energy):
        return False
    return new_energy < old_energy - noise


def _should_accept_best(
    *,
    current_best: float,
    candidate: float,
    floor: float | None,
) -> bool:
    """Return True iff ``candidate`` should overwrite ``best_energy``.

    Rejects non-finite (NaN/inf) candidates, candidates not strictly
    below ``current_best``, and candidates at or below ``floor``
    (treated as non-variational CTM artifacts per issue #298).  A
    ``None`` floor disables the floor check.
    """
    if not math.isfinite(candidate):
        return False
    if candidate >= current_best:
        return False
    if floor is not None and candidate <= floor:
        return False
    return True


def _build_optimizer(config: iPEPSConfig):
    """Build optax optimizer from config."""
    import optax

    name = config.gs_optimizer.lower()
    if name == "adam":
        lr = config.gs_learning_rate
        if config.gs_num_steps > 20:
            # Cosine decay from lr to lr/10 over the optimization
            schedule = optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=config.gs_num_steps,
                alpha=0.1,
            )
        else:
            schedule = lr
        return optax.chain(
            optax.clip_by_global_norm(config.gs_max_grad_norm),
            optax.adam(schedule),
        )
    elif name == "lbfgs":
        return optax.chain(
            optax.scale_by_lbfgs(memory_size=10),
            optax.clip_by_global_norm(config.gs_max_grad_norm),
            optax.scale(-1.0),
        )
    elif name == "cg":
        # CG direction is computed manually; optax just provides identity.
        return None
    else:
        raise ValueError(
            f"Unknown gs_optimizer {config.gs_optimizer!r}, "
            "expected 'adam', 'lbfgs', or 'cg'"
        )


def _use_line_search(config: iPEPSConfig) -> bool:
    """Whether to use backtracking line search."""
    if config.gs_line_search is not None:
        return config.gs_line_search
    return config.gs_optimizer.lower() in ("lbfgs", "cg")


def _resolve_line_search_method(config: iPEPSConfig, ctm_cfg: CTMConfig) -> str:
    """Return the concrete line-search method ("armijo" or "hager_zhang").

    Honours ``config.gs_line_search_method``:

    * ``"armijo"`` / ``"hager_zhang"`` — returned as-is.
    * ``"auto"`` (#509) — picks Armijo backtracking when
      ``ctm_cfg.chi < config.gs_line_search_hz_chi_threshold`` and
      Hager-Zhang otherwise.  Rationale: iPEPS-AD's gradient at small chi
      carries CTM truncation noise of order ε_T that frequently exceeds
      HZ's strong-Wolfe curvature tolerance (σ ≈ 0.9); Armijo only checks
      sufficient decrease and dodges this entirely at ~5-10× lower
      per-probe cost.

    Each iPEPS optimizer (1-site / 2-site / multisite) calls this helper
    exactly once after ``ctm_cfg`` is built and caches the result in
    ``line_search_method``; the dispatch sites inside the optimizer loop
    consult the cached value, not this function.  That keeps the method
    stable when ``ctm_cfg.chi`` shifts mid-run under legacy ``chi_ramp``
    or ``chi_auto_bump`` paths (both deprecated by #512, but still
    functional — without caching, an "auto" run could silently flip
    Armijo↔HZ after a bump).  Users wanting stage-specific behaviour
    should wrap ``optimize_gs_ad`` per chi stage with an explicit method.
    """
    method = config.gs_line_search_method
    if method == "auto":
        if ctm_cfg.chi < config.gs_line_search_hz_chi_threshold:
            return "armijo"
        return "hager_zhang"
    return method


def _tree_dot(a, b) -> float:
    """Compute real dot product between two pytrees of arrays.

    Returns ``Re(sum_leaves <conj(a_leaf), b_leaf>)`` as a Python float.

    Implementation note: routes through host NumPy rather than JAX. The
    function is called by Python-level optimizer plumbing (L-BFGS slope
    queries, CG beta numerators, Hager-Zhang ``dphi`` callbacks), where
    every call already terminates in ``float(...)`` and so must materialise
    to host anyway. The previous JAX path issued ~3 XLA dispatches per
    leaf plus a ``block_until_ready`` for the final ``float()``; that
    dispatch overhead (~100-300 µs each on CPU JAX) showed up at 62 % of
    AD wall-clock in cProfile (``project_f3_landed_line_search_next.md``).
    Materialising via ``np.asarray`` consolidates the host transfer into
    a single per-leaf step and lets NumPy do the scalar reduction.
    """
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    if not leaves_a:
        return 0.0
    total = 0.0
    for la, lb in zip(leaves_a, leaves_b):
        a_np = np.asarray(la).ravel()
        b_np = np.asarray(lb).ravel()
        # ``np.vdot`` already takes conj of the first argument and returns
        # a NumPy scalar; ``.real`` gives the real part as a Python float.
        total += np.vdot(a_np, b_np).real
    return float(total)


def _tree_scale(tree, alpha: float):
    """Scale all leaves in a pytree by a scalar."""
    return jax.tree.map(lambda x: x * alpha, tree)


def _tree_add(a, b):
    """Add two pytrees element-wise."""
    return jax.tree.map(lambda x, y: x + y, a, b)


def _random_noise(key, shape, dtype):
    """Generate noise matching the dtype of the parameter (real or complex)."""
    if jnp.issubdtype(dtype, jnp.complexfloating):
        k1, k2 = jax.random.split(key)
        return jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)
    return jax.random.normal(key, shape)


def _normalize_params(params):
    """Normalize iPEPS site tensor(s)."""
    if isinstance(params, tuple):
        return tuple(_normalize_params(p) for p in params)
    if hasattr(params, "norm"):
        return params * (1.0 / (params.norm() + 1e-10))
    # Plain JAX array (e.g. C4v coefficients) — use jnp.linalg.norm
    return params / (jnp.linalg.norm(params) + 1e-10)


def _tangent_project_unit(direction, params):
    """Project ``direction`` onto the tangent space of the unit-norm
    constraint ``||p|| = 1`` at ``params``.

    For a constraint ``||p|| = 1``, the tangent space at ``p`` is
    ``{v : <p, v> = 0}`` and the orthogonal projection of ``v`` is
    ``v - (<p, v> / <p, p>) p``.  Applied element-wise for tuple
    parameter trees.

    Used by the 2-site iPEPS AD optimizer to kill the radial component
    of the search direction before the line search (issue #328).
    Without this, the Euclidean L-BFGS direction can have a large
    component along ``A`` (or ``B``) that takes the line search on a
    long chord into a non-variational CTM region before
    ``_normalize_params`` retracts the iterate back.  Stale curvature
    pairs computed on that chord then corrupt subsequent L-BFGS steps.
    """
    if isinstance(params, tuple):
        return tuple(_tangent_project_unit(d, p) for d, p in zip(direction, params))
    if hasattr(params, "todense"):
        p_flat = params.todense().reshape(-1)
        d_flat = direction.todense().reshape(-1)
        # Use ``vdot`` (Hermitian inner product ``p^H v``) rather than
        # ``dot`` (bilinear ``p^T v``) so the projection works correctly
        # for complex-valued iPEPS site tensors — matches the optimizer's
        # ``_tree_dot`` convention elsewhere.  For real tensors this is
        # equivalent to ``jnp.dot``.
        p_norm_sq = jnp.vdot(p_flat, p_flat).real + 1e-30
        coef = jnp.vdot(p_flat, d_flat) / p_norm_sq
        return direction - params * coef
    p_norm_sq = jnp.vdot(params, params).real + 1e-30
    coef = jnp.vdot(params, direction) / p_norm_sq
    return direction - coef * params


def _backtracking_line_search(
    params,
    direction,
    grad,
    energy,
    loss_fn_fwd,
    c1=1e-4,
    rho=0.5,
    max_steps=8,
):
    """Armijo backtracking line search (Python-level, not JIT-traced).

    Args:
        params: current parameters (pytree)
        direction: search direction (pytree, same structure as params)
        grad: gradient at current params (pytree)
        energy: loss value at current params
        loss_fn_fwd: forward-only loss function params -> scalar
        c1: sufficient decrease parameter
        rho: backtracking factor
        max_steps: maximum number of backtracks

    Returns:
        (new_params, new_energy, step_size)
    """
    slope = _tree_dot(grad, direction)
    if slope >= 0:
        # Direction is not a descent direction; fall back to negative gradient
        direction = jax.tree.map(lambda g: -g, grad)
        slope = -_tree_dot(grad, grad)

    # Scale initial step so ||alpha * d|| ~ 0.1 * ||params||
    dir_norm = math.sqrt(max(_tree_dot(direction, direction), 1e-30))
    param_norm = math.sqrt(max(_tree_dot(params, params), 1e-30))
    alpha = min(1.0, 0.1 * param_norm / dir_norm)

    best_trial, best_f, best_alpha = params, energy, 0.0
    for _ in range(max_steps):
        trial = _normalize_params(_tree_add(params, _tree_scale(direction, alpha)))
        f_trial = loss_fn_fwd(trial)
        if f_trial < best_f:
            best_trial, best_f, best_alpha = trial, f_trial, alpha
        if f_trial <= energy + c1 * alpha * slope:
            return trial, f_trial, alpha
        alpha *= rho

    # Return best trial seen (stays at current params if nothing improved)
    return best_trial, best_f, best_alpha


def _cg_beta_pr(grad_new, grad_old):
    """Polak-Ribiere+ beta for conjugate gradient."""
    # beta = max(0, g_new . (g_new - g_old) / (g_old . g_old))
    diff = jax.tree.map(lambda gn, go: gn - go, grad_new, grad_old)
    num = _tree_dot(grad_new, diff)
    den = _tree_dot(grad_old, grad_old)
    if den < 1e-30:
        return 0.0
    return max(0.0, num / den)


def _wrap_as_dense_tensor(arr: jax.Array) -> DenseTensor:
    """Wrap a raw ``jax.Array`` iPEPS site tensor as a ``DenseTensor``.

    Assumes shape ``(D, D, D, D, d)`` with trivial U(1) charges
    (all zeros), matching the convention used by DenseTensor tests.
    """
    arr = jnp.asarray(arr)
    D = arr.shape[0]
    d = arr.shape[4]
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    return DenseTensor(arr, indices)


def _should_log_step(step: int, num_steps: int, interval: int) -> bool:
    if step == 0 or step == num_steps - 1:
        return True
    return (step + 1) % interval == 0


def _log_ad_step(
    backend: str,
    step: int,
    num_steps: int,
    energy: float,
    delta_energy: float,
    best_energy: float,
    *,
    grad_norm: float | None = None,
) -> None:
    delta_str = "N/A" if not math.isfinite(delta_energy) else f"{delta_energy:.3e}"
    grad_str = "" if grad_norm is None else f" |g|={grad_norm:.3e}"
    print(
        f"[iPEPS-AD:{backend}] step {step + 1}/{num_steps} "
        f"E={energy:.10f} dE={delta_str}{grad_str} E_best={best_energy:.10f}",
        flush=True,
    )


def _log_ad_converged(
    backend: str,
    step: int,
    delta_energy: float,
    tol: float,
    *,
    grad_norm: float | None = None,
    grad_norm_tol: float | None = None,
    criterion: str = "dE",
) -> None:
    parts = [f"[iPEPS-AD:{backend}] converged at step {step + 1}"]
    if criterion == "dE":
        parts.append(f"(dE={delta_energy:.3e} < tol={tol:.3e})")
    elif criterion == "grad_norm":
        parts.append(
            f"(||grad||={grad_norm:.3e} < tol={grad_norm_tol:.3e}, "
            f"dE={delta_energy:.3e})"
        )
    elif criterion == "both":
        parts.append(
            f"(dE={delta_energy:.3e} < tol={tol:.3e} AND "
            f"||grad||={grad_norm:.3e} < tol={grad_norm_tol:.3e})"
        )
    else:  # defensive; validated in iPEPSConfig.__post_init__
        parts.append(f"(dE={delta_energy:.3e} < tol={tol:.3e})")
    print(" ".join(parts), flush=True)


def _converged_outer(
    config: iPEPSConfig, delta_energy: float, grad_norm: float | None
) -> bool:
    """Return True if the outer AD loop should exit at this step.

    Honors ``config.gs_conv_criterion``:

    - ``"dE"`` (default): legacy behaviour — exit on
      ``|dE| < gs_conv_tol``.
    - ``"grad_norm"``: exit on ``||grad||_2 < gs_grad_norm_tol``
      (variPEPS ``optimizer_convergence_eps`` analog, issue #448).
    - ``"both"``: require both to hold simultaneously.

    A ``None`` ``grad_norm`` defeats any criterion that needs it.
    """
    criterion = config.gs_conv_criterion
    de_ok = delta_energy < config.gs_conv_tol
    if criterion == "dE":
        return de_ok
    if grad_norm is None:
        return False
    g_ok = float(grad_norm) < config.gs_grad_norm_tol
    if criterion == "grad_norm":
        return g_ok
    return de_ok and g_ok  # "both"


def _grad_l2_norm(grads) -> float:
    """L2 norm of an optax gradient pytree, returned as a Python float."""
    leaves = jax.tree_util.tree_leaves(grads)
    if not leaves:
        return 0.0
    sq = sum(jnp.vdot(jnp.ravel(g), jnp.ravel(g)).real for g in leaves)
    return float(jnp.sqrt(sq))


def optimize_gs_ad_chi_schedule(
    hamiltonian_gate: jax.Array | Tensor,
    A_init: jax.Array | Tensor | tuple | None,
    config: iPEPSConfig,
    chi_schedule: list[tuple[int, int]],
):
    """AD optimization with chi-ramping schedule (unified -- #453).

    Runs ``optimize_gs_ad`` ONCE with envs padded to ``max(chi)`` from
    the first iteration; the logical chi is ramped via
    ``_advance_chi_stage_if_due`` at each stage's budget boundary.  The
    JIT-compiled CTM / energy / backward kernels therefore see a single
    fixed env shape across the whole run -- no per-stage retraces.

    Trade-off: stages running at logical chi < ``max(chi)`` contract
    ``max(chi)``-shaped envs (zeros in the unused rows), paying more
    FLOPs per CTM iteration than a per-stage cold-start would.  The
    recompile cost this avoids (issue #453) dominates in practice.

    Reference: Zhang, Yang & Corboz, arXiv:2505.00494 (2025).

    Args:
        hamiltonian_gate: 2-site Hamiltonian of shape ``(d, d, d, d)``.
        A_init:           Initial site tensor(s) or ``None``.
        config:           Base ``iPEPSConfig``.  ``ctm.chi``,
                          ``ctm.chi_max``, ``gs_num_steps``, and
                          ``gs_chi_schedule_steps`` are overridden by the
                          shim per the schedule.
        chi_schedule:     List of ``(chi, max_steps)`` pairs, e.g.
                          ``[(8, 100), (16, 50), (32, 30)]``.  Each pair
                          says "run up to max_steps optimizer iterations
                          at logical chi = chi, then advance to the next
                          stage".

                          Three signals advance a stage at non-final
                          stages (#455):
                              - the per-stage ``max_steps`` budget is
                                exhausted;
                              - the user's ``gs_conv_criterion`` (dE,
                                grad_norm, or both) is met;
                              - the L-BFGS reset-recovery stall cap
                                ``gs_stall_recovery_retries`` is hit.
                          Unused steps from an early-exiting stage are
                          discarded (each stage's max_steps is an
                          upper bound, not a fixed quota).

                          Note: when stall-cap triggers a non-final
                          advance, the next stage starts from the
                          rolled-back ``best_params`` — fresh landscape,
                          fresh retry budget (PR #464's intent
                          preserved).

    Returns:
        Same as ``optimize_gs_ad`` at the final chi level.
    """
    from dataclasses import replace

    chi_max = max(chi for chi, _ in chi_schedule)
    total_steps = sum(n for _, n in chi_schedule)

    # #455 PR 1: pass the per-stage schedule straight through. Each
    # stage's max_steps is now a per-stage budget (was cumulative).
    # The optimizer loop tracks current_stage_idx + stage_start_step
    # and advances via _advance_chi_stage_if_due.
    ctm_cfg = replace(config.ctm, chi=chi_schedule[0][0], chi_max=chi_max)
    step_cfg = replace(
        config,
        ctm=ctm_cfg,
        gs_num_steps=total_steps,
        gs_chi_schedule_steps=list(chi_schedule),
    )

    if config.gs_verbose:
        print(
            f"[chi-ramp] unified: chi_max={chi_max}, "
            f"total_steps={total_steps}, stages={list(chi_schedule)}",
            flush=True,
        )

    return optimize_gs_ad(hamiltonian_gate, A_init, step_cfg)


def optimize_gs_ad(
    hamiltonian_gate: jax.Array | Tensor,
    A_init: jax.Array | Tensor | tuple | dict | None,
    config: iPEPSConfig,
):
    """AD-based ground state optimization of iPEPS.

    Uses automatic differentiation through the CTM fixed-point equation
    (Francuz et al. PRR 7, 013237) to compute exact gradients of the
    energy with respect to the site tensor(s), then optimizes with optax.

    Supports 1-site (``unit_cell="1x1"``), 2-site (``unit_cell="2site"``),
    and multi-site (``unit_cell=Lattice(...)``) unit cells.  Accepts dense
    ``jax.Array`` or Tensor-protocol objects (``DenseTensor``,
    ``SymmetricTensor``).

    Args:
        hamiltonian_gate: 2-site Hamiltonian of shape ``(d, d, d, d)``.
        A_init:           Initial site tensor ``(D, D, D, D, d)`` for 1-site,
                          ``(A, B)`` tuple for 2-site, ``dict[str, Tensor]``
                          for multi-site, or ``None`` for random
                          initialization.  When ``None`` and
                          ``config.su_init`` is ``True``, the tensor(s) are
                          initialized via simple update (``ipeps()``).
        config:           iPEPSConfig with AD optimization settings.

    Returns:
        For 1-site:    ``(A_opt, env, E_gs)``
        For 2-site:    ``((A_opt, B_opt), (env_A, env_B), E_gs)``
        For multi-site: ``(dict[str, Tensor], dict[str, CTMTensorEnv], E_gs)``
    """
    if config.gs_log_interval < 1:
        raise ValueError(f"gs_log_interval must be >= 1, got {config.gs_log_interval}")
    if config.gs_num_steps < 0:
        raise ValueError(f"gs_num_steps must be >= 0, got {config.gs_num_steps}")

    # The reference-C4v sub-path has no bump logic; reject early so the user
    # gets a clear error rather than silent no-op.
    if config.ctm.chi_auto_bump and _use_reference_c4v_path(config):
        raise NotImplementedError(
            "chi_auto_bump is not supported on the reference-C4v AD path; "
            "tracked as a follow-up issue."
        )

    # Resolve projector_backward policy before dispatch so every downstream
    # helper (1-site, 2-site, reference-C4v) sees the same CTM config.  No
    # silent gauge promotion — explicit user choices are preserved.
    # See docs/plans/2026-04-13-multisite-c4v-reference-ad-plan.md Task 8.
    config = _resolve_projector_backward(config)

    # Checkpointing is wired for the 2-site and 1-site (incl. coarse-grained
    # cg_gates) paths via _optimize_gs_ad_tensor / _optimize_gs_ad_2site.
    # Generic Lattice multisite and the dense C4v-reference path
    # (_optimize_gs_ad_tensor_reference_c4v) are NOT wired — block them
    # explicitly so gs_checkpoint_path is never silently ignored (#497).
    if config.gs_checkpoint_path is not None and (
        isinstance(config.unit_cell, Lattice) or _use_reference_c4v_path(config)
    ):
        raise NotImplementedError(
            "iPEPSConfig.gs_checkpoint_path is not yet supported for generic "
            "Lattice multisite unit cells or the dense C4v-reference path; use "
            "unit_cell='2site' or '1x1' (non-C4v-reference, incl. cg_gates). "
            "Multisite / C4v-reference checkpoint wiring is a follow-up (#497)."
        )

    if isinstance(config.unit_cell, Lattice):
        return _optimize_gs_ad_multisite(hamiltonian_gate, A_init, config)

    if config.unit_cell == "2site":
        return _optimize_gs_ad_2site(hamiltonian_gate, A_init, config)
    if _use_reference_c4v_path(config):
        return _optimize_gs_ad_tensor_reference_c4v(hamiltonian_gate, A_init, config)

    # CG-with-map_fn: optimizer params are a tuple of raw site arrays.
    # The user can pass that tuple directly as A_init for warm-start /
    # restart; otherwise we generate it via cg_gates.init_fn here so the
    # downstream optimizer doesn't have to.  In both cases we also build a
    # contracted A_init for shape / index inference.
    cg_raw_params: tuple | None = None
    cg_with_map_fn = (
        config.cg_gates is not None
        and getattr(config.cg_gates, "map_fn", None) is not None
    )
    if cg_with_map_fn and isinstance(A_init, tuple):
        cg_raw_params = A_init
        cg_data = config.cg_gates.map_fn(*cg_raw_params)
        A_init = _wrap_as_dense_tensor(cg_data)
    elif cg_with_map_fn and A_init is not None:
        # A single Tensor / array is ambiguous: the optimizer would have to
        # invert map_fn to recover raw params, which isn't generally possible.
        # Require the user to pass the raw-params tuple instead.
        raise ValueError(
            "When cg_gates.map_fn is set, A_init must be either None "
            "(auto-init via cg_gates.init_fn) or a tuple matching "
            "cg_gates.init_fn's output (raw site tensors).  Got "
            f"{type(A_init).__name__}; pass the raw-params tuple."
        )

    # Wrap raw jax.Array as DenseTensor so we always use the Tensor-protocol path.
    if A_init is not None and not isinstance(A_init, Tensor):
        A_init = _wrap_as_dense_tensor(A_init)

    if A_init is None:
        from tenax.algorithms.ipeps import ipeps

        gate = (
            hamiltonian_gate.todense()
            if isinstance(hamiltonian_gate, Tensor)
            else jnp.array(hamiltonian_gate)
        )
        d_phys = gate.shape[0]
        D = config.max_bond_dim

        if config.su_init:
            _, (A_su, _B_su), _ = ipeps(gate, None, config)
            A_init = A_su
        elif cg_with_map_fn and config.cg_gates.init_fn is not None:
            key = jax.random.PRNGKey(0)
            cg_raw_params = config.cg_gates.init_fn(D, key)
            cg_data = config.cg_gates.map_fn(*cg_raw_params)
            A_init = _wrap_as_dense_tensor(cg_data)
        else:
            key = jax.random.PRNGKey(0)
            k1, k2 = jax.random.split(key)
            A_data = jax.random.normal(
                k1, (D, D, D, D, d_phys)
            ) + 1j * jax.random.normal(k2, (D, D, D, D, d_phys))
            A_init = _wrap_as_dense_tensor(A_data)

    return _optimize_gs_ad_tensor(
        hamiltonian_gate, A_init, config, _cg_raw_params=cg_raw_params
    )


def _use_reference_c4v_path(config: iPEPSConfig) -> bool:
    """Compatibility wrapper around the shared AD policy helper."""
    return use_reference_c4v_path(config)


def _optimize_gs_ad_tensor_reference_c4v(
    hamiltonian_gate: jax.Array | Tensor,
    A_init: jax.Array | Tensor | None,
    config: iPEPSConfig,
):
    """Reference-mode dense C4v path with implicit-AD CTM backward."""
    if config.return_history:
        raise NotImplementedError(
            "return_history is currently only supported for unit_cell='1x1' "
            "(non-C4v-reference) and unit_cell='2site'."
        )
    _warn_implicit_ad_variational_caveat(config, path="1-site dense C4v reference")
    import optax

    from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor
    from tenax.algorithms._ctm_tensor_c4v import _c4v_to_full_env
    from tenax.algorithms._ctm_tensor_c4v_reference_ad import (
        ctm_tensor_c4v_reference_converge_reduced,
        ctm_tensor_c4v_reference_fixed_point,
    )
    from tenax.algorithms.ad_utils import CTMRGGradientError
    from tenax.algorithms.ipeps import (
        build_c4v_basis,
        c4v_coeffs_from_tensor,
        c4v_tensor_from_coeffs,
        ipeps,
    )

    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d_phys = gate.shape[0]
    D = config.max_bond_dim

    if A_init is not None and not isinstance(A_init, Tensor):
        A = _wrap_as_dense_tensor(A_init)
    elif A_init is None:
        if config.su_init:
            _, (A_su, _), _ = ipeps(gate, None, config)
            A = A_su
        else:
            key = jax.random.PRNGKey(0)
            k1, k2 = jax.random.split(key)
            A_data = jax.random.normal(
                k1, (D, D, D, D, d_phys)
            ) + 1j * jax.random.normal(k2, (D, D, D, D, d_phys))
            A = _wrap_as_dense_tensor(A_data)
    else:
        A = A_init

    if not isinstance(A, DenseTensor):
        raise TypeError(
            "ctm_ad_mode='c4v_reference' currently supports DenseTensor inputs only."
        )

    A = A * (1.0 / (A.norm() + 1e-10))
    # Enforce C4v by projecting to the C4v basis and reconstructing.
    D_bond = A.todense().shape[0]
    d_loc = A.todense().shape[-1]
    tensor_shape = (D_bond, D_bond, D_bond, D_bond, d_loc)
    c4v_basis = jnp.array(build_c4v_basis(D_bond, d_loc))
    coeffs = c4v_coeffs_from_tensor(A.todense(), c4v_basis)
    A_c4v = c4v_tensor_from_coeffs(coeffs, c4v_basis, tensor_shape)
    A = DenseTensor(
        A_c4v / (jnp.linalg.norm(A_c4v) + 1e-10),
        A.indices,
    )

    from dataclasses import replace as _replace

    ctm_cfg = config.ctm
    if config.gs_projector_method is not None:
        ctm_cfg = _replace(ctm_cfg, projector_method=config.gs_projector_method)

    if config.gs_num_steps == 0:
        env, _meta = ctm_tensor_c4v_reference_fixed_point(A, ctm_cfg)
        E0 = float(compute_energy_ctm_tensor(A, env, gate, d_phys))
        return A, env, E0

    optimizer = _build_optimizer(config)
    use_cg = optimizer is None
    params = A.todense()
    opt_state = None if optimizer is None else optimizer.init(params)
    best_energy = float("inf")
    best_params = params
    prev_energy = float("inf")

    def _project_c4v_and_normalize(A_data: jax.Array) -> jax.Array:
        coeffs = c4v_coeffs_from_tensor(A_data, c4v_basis)
        A_proj = c4v_tensor_from_coeffs(coeffs, c4v_basis, tensor_shape)
        return A_proj / (jnp.linalg.norm(A_proj) + 1e-10)

    def _loss_fn(A_data):
        A_proj = _project_c4v_and_normalize(A_data)
        A_tensor = DenseTensor(A_proj, A.indices)
        C, T = ctm_tensor_c4v_reference_converge_reduced(A_tensor, ctm_cfg)
        env = _c4v_to_full_env(C, T)
        energy = compute_energy_ctm_tensor(A_tensor, env, gate, d_phys)
        return energy, (env, A_tensor)

    for _step in range(config.gs_num_steps):
        try:
            (energy_val, _aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                params
            )
        except CTMRGGradientError as exc:
            _logger.warning(
                "[iPEPS-AD] Arnoldi precheck: rho(J^T) = %.4f >= 1 at step %d — "
                "skipping step (c4v_reference)",
                exc.spectral_radius,
                _step,
            )
            if config.gs_verbose:
                print(
                    f"[iPEPS-AD:c4v_reference] step {_step + 1}/{config.gs_num_steps} "
                    f"rho(J^T)={exc.spectral_radius:.4f} — skipping",
                    flush=True,
                )
            continue
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        E = float(energy_val)

        # Score / convergence-check on the *pre-step* params.  ``energy_val``
        # and ``grads`` describe ``params`` before the optimizer update, so
        # ``best_params`` and the convergence break must use that snapshot —
        # not the post-update value.  The 1-site / 2-site / multisite
        # dispatchers check before stepping for the same reason (codex
        # follow-up on PR #449 / #451).
        if _should_accept_best(
            current_best=best_energy,
            candidate=E,
            floor=getattr(config, "gs_energy_floor", None),
        ):
            best_energy = E
            best_params = params

        # Outer convergence (issue #448).  Mirrors the other dispatchers
        # so ``gs_conv_criterion`` and ``gs_conv_tol`` are honoured on the
        # reference-C4v path too — until this commit they were silently
        # ignored and every run consumed ``gs_num_steps``.  Grad-norm is
        # computed only when the chosen criterion needs it.
        delta_energy = abs(E - prev_energy)
        prev_energy = E
        grad_norm_val = (
            _grad_l2_norm(grads)
            if config.gs_conv_criterion in ("grad_norm", "both")
            else None
        )
        if _converged_outer(config, delta_energy, grad_norm_val):
            if config.gs_verbose:
                _log_ad_converged(
                    "c4v_reference",
                    _step,
                    delta_energy,
                    config.gs_conv_tol,
                    grad_norm=grad_norm_val,
                    grad_norm_tol=config.gs_grad_norm_tol,
                    criterion=config.gs_conv_criterion,
                )
            break

        if use_cg:
            params = _normalize_params(params - config.gs_learning_rate * grads)
        else:
            assert optimizer is not None
            if config.gs_optimizer.lower() == "lbfgs":
                updates, opt_state = optimizer.update(
                    grads,
                    opt_state,
                    params,
                    value=energy_val,
                    grad=grads,
                    value_fn=lambda p: _loss_fn(p)[0],
                )
            else:
                updates, opt_state = optimizer.update(grads, opt_state, params)
            params = _normalize_params(optax.apply_updates(params, updates))

    final_energy, (final_env, final_A) = _loss_fn(best_params)
    return final_A, final_env, float(final_energy)


def _log_ad_compile_notice(config) -> None:
    """Emit a one-time, verbose-gated notice before the first gradient step.

    Step 1 traces and XLA-compiles the CTM backward.  For block-sparse
    (symmetric / fermionic) site tensors this trace+compile scales with the
    number of charge blocks and can take minutes at larger ``D``/``chi`` —
    a *one-time* cost per optimizer run (the implicit-AD backward is cached
    across steps).  Without this notice the first step looks like a silent
    hang, which is exactly how issue #565 was reported.  The underlying
    compile-time scaling is tracked in issue #566.
    """
    if config.gs_verbose:
        print(
            "[iPEPS-AD] Tracing + compiling the CTM backward for step 1 "
            "(one-time per run). For block-sparse symmetric/fermionic "
            "tensors this scales with charge-block count and can take "
            "minutes at larger D/chi — see issue #566.",
            flush=True,
        )


def _optimize_gs_ad_tensor(
    hamiltonian_gate: jax.Array,
    A_init: Tensor,
    config: iPEPSConfig,
    *,
    _cg_raw_params: tuple | None = None,
):
    """AD-based ground state optimization for Tensor-protocol iPEPS (1-site).

    Uses ``ctm_tensor_converge`` with implicit differentiation through
    the standard Tensor-protocol CTM.

    Private kwargs:
        _cg_raw_params: Initial raw site-tensor tuple for CG-with-map_fn
            warm-start.  When provided, used in place of ``cg_gates.init_fn``;
            ``optimize_gs_ad`` forwards a tuple ``A_init`` (or its own
            init_fn output) here so the user's starting state is honored.
    """
    config = _normalize_stall_recovery(config, unit_cell="1x1")
    _warn_implicit_ad_variational_caveat(config, path="1-site Tensor-protocol")
    import optax

    from tenax.algorithms._checkpoint import (
        _config_to_dict,
        cg_gates_fingerprint,
        checkpoint_exists,
        gate_fingerprint,
        load_checkpoint,
        save_checkpoint,
        validate_config,
    )
    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor
    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
    from tenax.algorithms._split_ctm_energy_ad import converge_split_env
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor,
    )
    from tenax.algorithms.ad_utils import CTMRGGradientError
    from tenax.algorithms.ipeps_ad_policy import (
        ctm_converge_kwargs,
        make_ctm_energy_fn,
        validate_split_ctm_config,
    )

    cg_gates = config.cg_gates
    _use_cg = cg_gates is not None
    if _use_cg:
        from tenax.algorithms.coarse_grain import compute_energy_cg

        _cg_d_eff = int(cg_gates.h_intra.shape[0])
        _cg_map_fn = cg_gates.map_fn
        _cg_init_fn = cg_gates.init_fn
    else:
        compute_energy_cg = None
        _cg_d_eff = None
        _cg_map_fn = None
        _cg_init_fn = None

    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d_phys = gate.shape[0]

    A = A_init
    A = A * (1.0 / (A.norm() + 1e-10))

    # Apply AD policy overrides (projector method + explicit-AD gauge promotion)
    # in one place so 1-site and 2-site paths stay consistent.
    ctm_cfg = build_ad_ctm_config(config)

    use_c4v = config.gs_c4v
    if use_c4v:
        from tenax.algorithms.ipeps import (
            build_c4v_basis,
            c4v_coeffs_from_tensor,
            c4v_tensor_from_coeffs,
        )

        D_bond = A.todense().shape[0]
        d_loc = A.todense().shape[-1]
        tensor_shape = (D_bond, D_bond, D_bond, D_bond, d_loc)
        c4v_basis = jnp.array(build_c4v_basis(D_bond, d_loc))
        # Project initial tensor into C4v subspace
        c4v_coeffs = c4v_coeffs_from_tensor(A.todense(), c4v_basis)
        A_sym_data = c4v_tensor_from_coeffs(c4v_coeffs, c4v_basis, tensor_shape)
        A = DenseTensor(A_sym_data, A.indices)
    else:
        c4v_basis = None
        c4v_coeffs = None
        tensor_shape = None

    # Env warm-start cache — replaces flat env_leaves threading.
    _env_cache: dict[str, dict] = {}

    # Split-CTM (fuse_virtual_legs=False) single-site path (#463 Phase 2).
    # Routes the AD energy AND the forward-only CTMs (warm-start, line-search
    # probe, final-env eval) through the split χ²·D⁴ forward so the memory win
    # is real and the line-search φ(α)/dφ(α) stay consistent.  Only the
    # simplest stable config is supported; reject the fused-env-coupled
    # accelerators (bump, ramp, schedules, metric, cg) up front rather than
    # silently feeding a SplitCTMTensorEnv to fused-only machinery.
    use_split = not ctm_cfg.fuse_virtual_legs
    if use_split:
        # Shared single source of truth for the CTMConfig-level rejects
        # (recipe + the three χ-changing knobs); see validate_split_ctm_config.
        validate_split_ctm_config(ctm_cfg, config.gs_recipe)
        # iPEPSConfig-level rejects the shared helper can't see.
        if _use_cg:
            raise NotImplementedError(
                "fuse_virtual_legs=False (split CTM) does not support cg_gates; "
                "use fuse_virtual_legs=True."
            )
        if (
            config.gs_ctm_conv_tol_schedule is not None
            or config.gs_ctm_max_iter_schedule is not None
            or config.gs_plateau_patience_schedule is not None
        ):
            raise NotImplementedError(
                "CTM conv_tol/max_iter/plateau schedules are not supported on "
                "the split-CTM path; use fuse_virtual_legs=True."
            )

    use_explicit = not config.gs_implicit_ad
    explicit_steps = config.gs_explicit_ad_steps
    explicit_warmup = config.gs_explicit_ad_warmup
    explicit_backward_steps = config.gs_explicit_ad_backward_steps

    def _params_to_A_norm(params):
        """Convert raw optimizer params to a normalized DenseTensor."""
        if _use_cg and _cg_map_fn is not None:
            cg_data = _cg_map_fn(*params)
            cg_data = cg_data / (jnp.linalg.norm(cg_data) + 1e-10)
            return DenseTensor(cg_data, A.indices)
        if use_c4v:
            A_data = c4v_tensor_from_coeffs(params, c4v_basis, tensor_shape)
            A_norm_data = A_data / (jnp.linalg.norm(A_data) + 1e-10)
            return DenseTensor(A_norm_data, A.indices)
        return params * (1.0 / (params.norm() + 1e-10))

    if _use_cg:

        def _cg_energy_callable(site_tensors, envs, _gate):
            """energy_fn closure for ctm_energy_explicit/implicit (CG path)."""
            A_norm = site_tensors[(0, 0)]
            return compute_energy_cg(A_norm, envs[(0, 0)], cg_gates, _cg_d_eff)

        _energy_fn_kw = _cg_energy_callable
    else:
        _energy_fn_kw = None

    _ctm_energy_fn = make_ctm_energy_fn(
        neighbors=SINGLE_SITE_NEIGHBORS,
        gate=gate,
        # Resolved live so gs_ctm_conv_tol_schedule rebindings of
        # ``ctm_cfg`` propagate to the AD loss closure (codex P1, #382).
        get_ctm_cfg=lambda: ctm_cfg,
        env_cache=_env_cache,
        use_explicit=use_explicit,
        explicit_warmup=explicit_warmup,
        explicit_steps=explicit_steps,
        explicit_backward_steps=explicit_backward_steps,
        energy_fn=_energy_fn_kw,
        recipe=config.gs_recipe,
    )

    def loss_fn(params):
        A_norm = _params_to_A_norm(params)
        site_tensors = {(0, 0): A_norm}
        energy = _ctm_energy_fn(site_tensors)
        return energy

    def _split_forward(A_norm):
        """Forward-only gauge-fixed split-CTM converge (warm-start/probe/final).

        Reads ``ctm_cfg`` live so any in-loop rebinding is honored (schedules
        are rejected on the split path, so it is effectively static here).
        Lands on the same fixed point the implicit-AD loss differentiates.
        """
        return converge_split_env(
            A_norm,
            chi=ctm_cfg.chi,
            max_iter=ctm_cfg.max_iter,
            conv_tol=ctm_cfg.conv_tol,
            chi_I=ctm_cfg.chi_I,
            renormalize=ctm_cfg.renormalize,
            min_iter=ctm_cfg.min_iter,
        )

    def _update_env_cache(params):
        """Re-run forward CTM (no grad) to warm-start next step."""
        A_norm = _params_to_A_norm(params)
        if use_split:
            # χ²·D⁴ split forward — no fused double layer is ever built.
            # eps_T is unused (auto-bump is rejected on the split path).
            _env_cache["envs"] = {(0, 0): _split_forward(A_norm)}
            _env_cache["max_truncation_error"] = 0.0
            return
        site_tensors = {(0, 0): A_norm}
        envs, info = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            **ctm_converge_kwargs(ctm_cfg, env_init=_env_cache.get("envs", None)),
        )
        _env_cache["envs"] = envs
        # ``info.max_truncation_error`` comes from the JIT-compiled CTM step,
        # which sets eps_T = 0.0 for any input that is a JAX tracer during
        # JIT compilation.  For the auto-bump path we need a real eps_T from
        # a non-JIT-compiled sweep so we can compare against the threshold.
        #
        # We use the ``"eigh"`` projector for this measurement regardless of
        # the optimizer's configured projector_method: eigh builds the full
        # density matrix rho = C1g @ C1g^H + C4g @ C4g^H (shape chi*D² × chi*D²)
        # and discards chi*D² - chi eigenvalues, giving a meaningful eps_T > 0
        # whenever chi < chi_eff = D² * chi (i.e. always when D > 1).  The
        # SVD cross-product M = C1g^H @ C4g is chi×chi and retains all chi
        # singular values, so the SVD-path eps_T is identically 0.
        #
        # This measurement is NOT used for the gradient computation — it only
        # drives the bump decision, so the projector choice here does not
        # affect optimization accuracy.
        if ctm_cfg.chi_auto_bump and not _use_cg:
            from tenax.algorithms._ctm_tensor_convergence import _ctm_tensor_sweep
            from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor

            a_dl = _build_double_layer_tensor(A_norm)
            _, real_eps_t = _ctm_tensor_sweep(
                envs[(0, 0)],
                a_dl,
                ctm_cfg.chi,
                ctm_cfg.renormalize,
                "eigh",  # eigh gives eps_T > 0 for chi < D²*chi; SVD always gives 0
                projector_backward="auto",
            )
            _env_cache["max_truncation_error"] = float(real_eps_t)
        else:
            _env_cache["max_truncation_error"] = float(info.max_truncation_error)

    # Metric L-BFGS preconditioning calls .todense() on grads/params (see
    # _metric_precond.py:146 and the metric branch below).  CG with map_fn
    # parameterizes the optimizer state as a tuple of raw site arrays, so the
    # metric path crashes with AttributeError.  Disable it for that case and
    # warn the user once so silent loss-of-preconditioning is visible.
    _cg_uses_tuple_params = _use_cg and _cg_map_fn is not None
    is_metric_lbfgs = (
        config.gs_metric_precond
        and config.gs_optimizer.lower() == "lbfgs"
        and not _cg_uses_tuple_params
        # Metric preconditioning builds its inner product from the fused
        # CTMTensorEnv (``env_for_metric`` below); the split path stores a
        # SplitCTMTensorEnv, so disable it there rather than feed an
        # incompatible env to the preconditioner.
        and not use_split
    )
    if (
        _cg_uses_tuple_params
        and config.gs_metric_precond
        and config.gs_optimizer.lower() == "lbfgs"
    ):
        import warnings

        warnings.warn(
            "gs_metric_precond=True is incompatible with cg_gates that "
            "supply a map_fn (params are a tuple of raw site tensors, but "
            "the metric path expects tensor-like params with .todense()). "
            "Falling back to non-preconditioned L-BFGS for this run.",
            stacklevel=3,
        )
    if (
        use_split
        and config.gs_metric_precond
        and config.gs_optimizer.lower() == "lbfgs"
    ):
        import warnings

        warnings.warn(
            "gs_metric_precond=True is not yet supported on the split-CTM path "
            "(fuse_virtual_legs=False); the metric inner product needs the "
            "fused CTMTensorEnv. Falling back to non-preconditioned L-BFGS.",
            stacklevel=3,
        )
    if _use_cg and _cg_map_fn is not None:
        # CG with map_fn: optimize raw site tensors, contract via map_fn in
        # _params_to_A_norm.  Use caller-supplied raw params when provided
        # (warm-start / restart); otherwise call init_fn.
        if _cg_raw_params is not None:
            params = _cg_raw_params
        elif _cg_init_fn is not None:
            params = _cg_init_fn(config.max_bond_dim, jax.random.PRNGKey(0))
        else:
            raise ValueError(
                "cg_gates.map_fn is set but neither cg_gates.init_fn nor a "
                "raw-params A_init was provided; cannot construct optimizer "
                "params."
            )
    elif use_c4v:
        params = c4v_coeffs
    else:
        params = A
    optimizer = None if is_metric_lbfgs else _build_optimizer(config)
    opt_state = optimizer.init(params) if optimizer is not None else None
    use_ls = _use_line_search(config)
    # Resolve ``gs_line_search_method`` once per dispatch (#509 / codex P2 on
    # #549).  ``ctm_cfg.chi`` can shift mid-run under legacy ``chi_ramp`` /
    # ``chi_auto_bump`` paths (deprecated by #512 but still functional), so
    # re-resolving inside the line-search step would silently flip Armijo↔HZ
    # after a bump.  Cache the initial-chi decision and keep it stable.
    line_search_method = _resolve_line_search_method(config, ctm_cfg)
    is_cg = config.gs_optimizer.lower() == "cg"

    best_energy = float("inf")
    best_params = params
    best_env_cache: dict[str, dict] = {}  # tracked for fresh-CTM warm-start (#317)
    prev_energy = float("inf")
    prev_grad = None
    cg_direction = None
    prev_precond_grad = None  # for preconditioned CG beta
    log_interval = config.gs_log_interval
    # L-BFGS history for metric-preconditioned path
    lbfgs_history: list = []
    prev_A_flat: jnp.ndarray | None = None
    prev_grad_flat: jnp.ndarray | None = None
    # Rolling gradient-norm window for the spike guard (mirrors the 2-site
    # path; #524).  Non-variational drift on the implicit-AD path shows up as
    # a >Nx gradient blowup; rolling back the step before the line search both
    # protects best-state tracking and avoids the line search thrashing on the
    # exploded gradient.  Off unless ``gs_grad_spike_ratio`` is set.
    recent_gnorms: list[float] = []

    from tenax.algorithms.ad_utils import (
        _wrap_tensor,
    )

    def loss_fn_fwd(p):
        """Forward-only loss for line search — warm-starts CTM from env cache."""
        A_norm = _params_to_A_norm(p)
        if use_split:
            # Same gauge-fixed split forward + split energy as the AD loss, so
            # φ(α) (this probe) and dφ/dα (the implicit-AD gradient) agree.
            env = _split_forward(A_norm)
            _env_cache["envs"] = {(0, 0): env}
            return float(compute_energy_split_ctm_tensor(A_norm, env, gate))
        site_tensors = {(0, 0): A_norm}
        envs, _ = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            **ctm_converge_kwargs(
                ctm_cfg,
                env_init=_env_cache.get("envs", None),
                for_probe=True,
            ),
        )
        # Issue #502: share the just-converged env with the subsequent
        # ``_dphi(α)`` call (and any nearby ``_phi(α')`` probe).  The
        # implicit-AD ``loss_fn`` warm-starts from the same ``_env_cache``
        # dict, so this turns the second CTM converge at the same α into
        # an effectively-no-op warm restart.  The line-search end-of-step
        # ``_update_env_cache(params)`` overwrites this with the
        # accepted-step env, so intermediate writes never persist past
        # the L-BFGS step boundary.
        _env_cache["envs"] = envs
        if _use_cg:
            return float(compute_energy_cg(A_norm, envs[(0, 0)], cg_gates, _cg_d_eff))
        return float(compute_energy_ctm_tensor(A_norm, envs[(0, 0)], gate, d_phys))

    stall_count = 0  # noise recovery: consecutive line search failures
    current_stage_idx = 0
    stage_start_step = 0

    # Optional trajectory capture (config.return_history).  Always allocated
    # but only populated/returned when the flag is set, so there is no
    # extra wall-clock cost when the flag is False (only the bool check
    # below is added per step).
    _history_energies: list[float] = []
    _history_step_times: list[float] = []
    _jit_compile_time: float = 0.0
    _first_step = True
    _converged = False

    # CTM conv_tol schedule: update ctm_cfg when tolerance changes
    _conv_tol_schedule = config.gs_ctm_conv_tol_schedule
    _current_conv_tol = ctm_cfg.conv_tol

    # CTM max_iter schedule (issue #507): same lookup contract as the
    # conv_tol schedule but caps the accepted-step CTM converge budget.
    _max_iter_schedule = config.gs_ctm_max_iter_schedule
    _current_max_iter = ctm_cfg.max_iter

    # CTM plateau-patience schedule: update ctm_cfg when patience changes.
    # Independent of the conv_tol schedule (intentionally — see
    # iPEPSConfig.gs_plateau_patience_schedule docstring on auto-tuning).
    _patience_schedule = config.gs_plateau_patience_schedule
    _current_patience = ctm_cfg.plateau_patience

    # variPEPS §2.8.2 auto-χ_E bump padding policy: for a SymmetricTensor
    # A, derive the same ``base_charges`` the projector uses so the bump
    # pads χ-leg charges to match the projector's per-sector allocation
    # rather than tiling the post-projector grouped pattern. A's bond
    # charges are fixed across optimization iterations, so we compute
    # this once outside the loop. (PR #430 codex review on
    # ``_ctm_env_pad.py``.)
    _bump_base_charges: np.ndarray | None = None
    if ctm_cfg.chi_auto_bump or config.gs_chi_schedule_steps is not None:
        _A_init = _params_to_A_norm(params)
        if isinstance(_A_init, SymmetricTensor):
            from tenax.algorithms._ctm_tensor_convergence import _get_base_charges
            from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor

            _bump_base_charges = _get_base_charges(_build_double_layer_tensor(_A_init))

    def _get_scheduled_conv_tol(step_idx, num_steps):
        """Look up conv_tol from schedule based on step fraction."""
        if _conv_tol_schedule is None:
            return _current_conv_tol
        frac = step_idx / max(num_steps, 1)
        tol = _conv_tol_schedule[0][1]  # default to first entry
        for threshold, t in _conv_tol_schedule:
            if frac >= threshold:
                tol = t
        return tol

    def _get_scheduled_max_iter(step_idx, num_steps):
        """Look up max_iter from schedule based on step fraction (issue #507)."""
        if _max_iter_schedule is None:
            return _current_max_iter
        frac = step_idx / max(num_steps, 1)
        mi = _max_iter_schedule[0][1]  # default to first entry
        for threshold, m in _max_iter_schedule:
            if frac >= threshold:
                mi = m
        return mi

    def _get_scheduled_plateau_patience(step_idx, num_steps):
        """Look up plateau_patience from schedule based on step fraction."""
        if _patience_schedule is None:
            return _current_patience
        frac = step_idx / max(num_steps, 1)
        patience = _patience_schedule[0][1]
        for threshold, p in _patience_schedule:
            if frac >= threshold:
                patience = p
        return patience

    _gate_fp = (
        gate_fingerprint(gate)
        if (config.gs_checkpoint_path is not None and gate is not None)
        else None
    )
    _cg_fp = (
        cg_gates_fingerprint(config.cg_gates)
        if (config.gs_checkpoint_path is not None and config.cg_gates is not None)
        else None
    )

    # ---- Checkpoint resume ----------------------------------------
    # When gs_resume=True and a checkpoint exists at gs_checkpoint_path,
    # restore optimizer state and pick up at step ``saved_step + 1``.
    # All other branches keep the fresh-init values set above.  These
    # statements reassign function locals (params, ctm_cfg, _env_cache
    # contents, ...) at function scope, so plain assignment is correct.
    start_step = 0
    if config.gs_resume:
        # Fail fast on a typo'd / deleted / never-written checkpoint
        # path rather than silently falling through to fresh init.
        if not checkpoint_exists(config.gs_checkpoint_path):
            raise FileNotFoundError(
                f"gs_resume=True but no checkpoint found at "
                f"{config.gs_checkpoint_path!r} (looked for 'ckpt.last.pkl'). "
                f"Either point gs_checkpoint_path at the directory of a prior "
                f"run, or set gs_resume=False to start fresh."
            )
        bundle = load_checkpoint(config.gs_checkpoint_path)
        validate_config(bundle.get("config", {}), config)

        saved_fp = bundle.get("hamiltonian_fingerprint")
        if saved_fp is not None and tuple(saved_fp) != _gate_fp:
            raise ValueError(
                "Cannot resume: the hamiltonian gate has changed since the "
                "checkpoint was written.\n"
                f"  saved fingerprint:   {tuple(saved_fp)!r}\n"
                f"  current fingerprint: {_gate_fp!r}\n"
                "If this is intentional, start a fresh run."
            )

        # Full inequality (not `is not None and ...`): a plain checkpoint
        # (saved_cg_fp is None) resumed with cg_gates set, or vice versa, or a
        # changed parameterization mode, are ALL fatal mismatches — restoring a
        # param tree that doesn't match the live (CG vs plain) path would crash
        # or silently run with incompatible dims.
        saved_cg_fp = bundle.get("cg_gates_fingerprint")
        if saved_cg_fp != _cg_fp:
            raise ValueError(
                "Cannot resume: the coarse-grained cg_gates (or parameterization "
                "mode / plain-vs-CG path) have changed since the checkpoint was "
                "written.\n"
                f"  saved cg fingerprint:   {saved_cg_fp!r}\n"
                f"  current cg fingerprint: {_cg_fp!r}\n"
                "If this is intentional, start a fresh run."
            )

        saved_cfg = bundle.get("config", {})
        _opt_compat = (
            saved_cfg.get("gs_optimizer") == config.gs_optimizer
            and saved_cfg.get("gs_metric_precond") == config.gs_metric_precond
        )

        params = bundle["params"]
        best_params = bundle["best_params"]
        best_energy = float(bundle["best_energy"])
        prev_energy = float(bundle["prev_energy"])
        # Clear the env warm-start cache AND the implicit-AD lambda seed on
        # restore (issue #501), matching the 2-site resume path. The restored
        # env below is a fresh starting point; the prior Neumann seed is stale.
        _drop_env_cache_for_reset(_env_cache)
        _env_cache.update(bundle.get("env_cache", {}))
        best_env_cache = dict(bundle.get("best_env_cache", {}))
        stall_count = int(bundle.get("stall_count", 0))

        if _opt_compat:
            if optimizer is not None and bundle.get("opt_state") is not None:
                opt_state = bundle["opt_state"]
            lbfgs_history = list(bundle.get("lbfgs_history") or [])
            prev_A_flat = bundle.get("prev_params_flat")
            prev_grad_flat = bundle.get("prev_grad_flat")
            cg_direction = bundle.get("cg_direction")
            prev_grad = bundle.get("prev_grad")
            prev_precond_grad = bundle.get("prev_precond_grad")
        else:
            import warnings as _warnings

            _warnings.warn(
                "Optimizer-defining config differs from checkpoint "
                f"(saved: gs_optimizer={saved_cfg.get('gs_optimizer')!r}, "
                f"gs_metric_precond={saved_cfg.get('gs_metric_precond')!r}; "
                f"current: gs_optimizer={config.gs_optimizer!r}, "
                f"gs_metric_precond={config.gs_metric_precond!r}). "
                "Restoring params/envs/energies but discarding saved "
                "optimizer history (curvature/momentum will restart fresh).",
                stacklevel=2,
            )

        current_stage_idx = int(bundle.get("current_stage_idx", 0))
        stage_start_step = int(bundle.get("stage_start_step", 0))
        saved_chi = bundle.get("ctm_cfg_chi")
        if saved_chi is not None and int(saved_chi) != ctm_cfg.chi:
            ctm_cfg = _replace(ctm_cfg, chi=int(saved_chi))
        saved_conv_tol = bundle.get("current_conv_tol")
        if saved_conv_tol is not None:
            _current_conv_tol = float(saved_conv_tol)
            ctm_cfg = _replace(ctm_cfg, conv_tol=_current_conv_tol)
        saved_patience = bundle.get("current_patience")
        if saved_patience is not None:
            _current_patience = int(saved_patience)
            ctm_cfg = _replace(ctm_cfg, plateau_patience=_current_patience)

        start_step = int(bundle["step"]) + 1
        if config.gs_verbose:
            print(
                f"[iPEPS-AD:1site-tensor] resumed from step {start_step} "
                f"(best E={best_energy:.10f}, chi={ctm_cfg.chi})",
                flush=True,
            )

    def _maybe_save_1s_checkpoint(step, chi_before, e_prev, *, force_last=False):
        if config.gs_checkpoint_path is None:
            return
        chi_changed = ctm_cfg.chi != chi_before
        is_new_best = best_energy < e_prev
        should_save_last = (
            force_last or chi_changed or (step + 1) % config.gs_checkpoint_every == 0
        )
        if not (should_save_last or is_new_best):
            return
        _ckpt_state = {
            "step": step,
            "config": _config_to_dict(config),
            "hamiltonian_fingerprint": _gate_fp,
            "cg_gates_fingerprint": _cg_fp,
            "params": params,
            "best_params": best_params,
            "best_energy": float(best_energy),
            "prev_energy": float(prev_energy),
            "env_cache": dict(_env_cache),
            "best_env_cache": dict(best_env_cache),
            "opt_state": opt_state,
            "lbfgs_history": list(lbfgs_history),
            "prev_params_flat": prev_A_flat,
            "prev_grad_flat": prev_grad_flat,
            "cg_direction": cg_direction,
            "prev_grad": prev_grad,
            "prev_precond_grad": prev_precond_grad,
            "stall_count": stall_count,
            "current_stage_idx": current_stage_idx,
            "stage_start_step": stage_start_step,
            "ctm_cfg_chi": ctm_cfg.chi,
            "current_conv_tol": _current_conv_tol,
            "current_patience": _current_patience,
        }
        if should_save_last:
            save_checkpoint(_ckpt_state, config.gs_checkpoint_path)
        if is_new_best:
            save_checkpoint(_ckpt_state, config.gs_checkpoint_path, is_best=True)

    _log_ad_compile_notice(config)
    for step in range(start_step, config.gs_num_steps):
        # Snapshots for checkpoint "did chi change / new best" detection.
        # ``best_energy`` only decreases, so a strict < comparison after the
        # step body identifies a freshly-accepted best.  Both snapshots are
        # passed to ``_maybe_save_1s_checkpoint`` at every save call site so
        # the helper can decide what to flush.
        _best_energy_at_step_start = best_energy
        _chi_at_step_start = ctm_cfg.chi
        # Update conv_tol if schedule is active
        if _conv_tol_schedule is not None:
            new_tol = _get_scheduled_conv_tol(step, config.gs_num_steps)
            if new_tol != _current_conv_tol:
                _current_conv_tol = new_tol
                ctm_cfg = _replace(ctm_cfg, conv_tol=new_tol)
        # Update max_iter if schedule is active (#507)
        if _max_iter_schedule is not None:
            new_max_iter = _get_scheduled_max_iter(step, config.gs_num_steps)
            if new_max_iter != _current_max_iter:
                _current_max_iter = new_max_iter
                ctm_cfg = _replace(ctm_cfg, max_iter=new_max_iter)
        # Update plateau_patience if schedule is active
        if _patience_schedule is not None:
            new_patience = _get_scheduled_plateau_patience(step, config.gs_num_steps)
            if new_patience != _current_patience:
                _current_patience = new_patience
                ctm_cfg = _replace(ctm_cfg, plateau_patience=new_patience)
        if config.return_history:
            _step_t0 = _time.perf_counter()
        try:
            energy_val, grads = jax.value_and_grad(loss_fn)(params)
        except CTMRGGradientError as exc:
            _logger.warning(
                "[iPEPS-AD] Arnoldi precheck: rho(J^T) = %.4f >= 1 at step %d — "
                "skipping, triggering stall recovery",
                exc.spectral_radius,
                step,
            )
            if config.gs_verbose:
                print(
                    f"[iPEPS-AD:1site-tensor] step {step + 1}/{config.gs_num_steps} "
                    f"rho(J^T)={exc.spectral_radius:.4f} — stall recovery",
                    flush=True,
                )
            stall_count += 1
            if (
                config.gs_stall_recovery == "noise"
                and stall_count <= config.gs_noise_recovery_retries
            ):
                noise_key = jax.random.PRNGKey(step * 1000 + stall_count)
                if use_c4v:
                    noise = config.gs_noise_amplitude * _random_noise(
                        noise_key, params.shape, params.dtype
                    )
                    params = params + noise * jnp.linalg.norm(params)
                    params = params / (jnp.linalg.norm(params) + 1e-10)
                else:
                    data = params.todense()
                    noise = config.gs_noise_amplitude * _random_noise(
                        noise_key, data.shape, data.dtype
                    )
                    noisy = data + noise * jnp.linalg.norm(data)
                    noisy = noisy / (jnp.linalg.norm(noisy) + 1e-10)
                    params = _wrap_tensor(noisy, params)
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_A_flat = None
                    prev_grad_flat = None
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
            elif config.gs_stall_recovery == "reset":
                # Cap on CTMRGGradientError-driven reset path (#454 follow-up,
                # codex review on PR #457).  Same retry budget as the
                # post-line-search reset block; without this, an Arnoldi
                # precheck failure loop could spin for all gs_num_steps.
                if stall_count > config.gs_stall_recovery_retries:
                    n_resets_done = stall_count - 1
                    if config.gs_verbose:
                        print(
                            f"[iPEPS-AD] CTM-error stall budget exhausted after "
                            f"{n_resets_done} resets, "
                            f"returning best E={best_energy:.10f}",
                            flush=True,
                        )
                    break
                params = best_params
                # #518: ``best_env_cache`` may be at a stale χ if a
                # reactive/scheduled bump fired after it was last
                # snapshotted.  Clear instead of restoring; the next
                # CTM call cold-starts at the current ctm_cfg.chi.
                _drop_env_cache_for_reset(_env_cache)
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_A_flat = None
                    prev_grad_flat = None
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
            continue
        energy_float = float(energy_val)
        grad_norm_val = _grad_l2_norm(grads)

        # Gradient-spike guard (ported from the 2-site path, #524): a
        # non-variational implicit-AD step shows up as a >Nx gradient blowup
        # (e.g. |g| jumping from ~0.2 to ~5e3) whose energy can dip below the
        # variational floor.  Roll back to best BEFORE the line search so the
        # bad step neither poisons best-state tracking nor makes the line
        # search thrash on the exploded gradient.  Off unless
        # ``gs_grad_spike_ratio`` is set.
        if (
            config.gs_grad_spike_ratio is not None
            and len(recent_gnorms) >= 1
            and best_energy < float("inf")
        ):
            spike_floor = max(float(np.median(recent_gnorms)), 1.0)
            if grad_norm_val > config.gs_grad_spike_ratio * spike_floor:
                params = best_params
                _drop_env_cache_for_reset(_env_cache)
                recent_gnorms.clear()
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_A_flat = None
                    prev_grad_flat = None
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                    opt_state = optimizer.init(params)
                _logger.warning(
                    "[iPEPS-AD:1site-tensor] step %d/%d gradient spike "
                    "|g|=%.3e > %.1fx median %.3e — rolling back to best",
                    step + 1,
                    config.gs_num_steps,
                    grad_norm_val,
                    config.gs_grad_spike_ratio,
                    spike_floor,
                )
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD:1site-tensor] step {step + 1}/"
                        f"{config.gs_num_steps} |g|={grad_norm_val:.3e} spike "
                        f"(>{config.gs_grad_spike_ratio:.1f}x median "
                        f"{spike_floor:.3e}) — rollback to best, clear history",
                        flush=True,
                    )
                # A rollback leaves params == best_params, so the NEXT step
                # re-evaluates the same state and sees dE == 0 — which the "dE"
                # convergence criterion would misread as convergence (the
                # gradient is still large).  Reset prev_energy so the
                # post-rollback step cannot false-converge.
                prev_energy = float("inf")
                continue
        recent_gnorms.append(grad_norm_val)
        if len(recent_gnorms) > config.gs_grad_spike_window:
            recent_gnorms.pop(0)

        if config.return_history:
            _step_dt = _time.perf_counter() - _step_t0
            if _first_step:
                _jit_compile_time = float(_step_dt)
                _first_step = False
            else:
                _history_step_times.append(float(_step_dt))
            _history_energies.append(energy_float)

        # Update env cache for warm-starting next step
        _update_env_cache(params)

        _accepted_best_this_iter = False
        if _should_accept_best(
            current_best=best_energy,
            candidate=energy_float,
            floor=config.gs_energy_floor,
        ):
            best_energy = energy_float
            best_params = params
            best_env_cache = dict(_env_cache)  # snapshot for warm-start (#317)
            _accepted_best_this_iter = True

        delta_energy = abs(energy_float - prev_energy)
        logged = False
        if config.gs_verbose and _should_log_step(
            step, config.gs_num_steps, log_interval
        ):
            _log_ad_step(
                "1site-tensor",
                step,
                config.gs_num_steps,
                energy_float,
                delta_energy,
                best_energy,
                grad_norm=grad_norm_val,
            )
            logged = True

        prev_energy = energy_float
        if _converged_outer(config, delta_energy, grad_norm_val):
            # Convergence break short-circuits the end-of-iter bump. If
            # the energy stalled because χ is too small (high eps_T from
            # _update_env_cache above), the user-requested auto-bump
            # would silently no-op and the final env would stay at the
            # old χ. Fire the bump here so the final state matches
            # ctm_cfg.chi at exit — same behaviour as before the #419
            # move-to-end refactor. (PR #432 codex review; not picked up
            # in #432's squash, so re-included here.)
            # Snapshot χ before either bump fires; the reset below triggers
            # on EITHER reactive (_maybe_bump_chi) or scheduled
            # (_advance_chi_stage_if_due) bump changing it — both are landscape
            # transitions (#464 codex review).
            chi_before_bump = ctm_cfg.chi
            last_eps_t = float(_env_cache.get("max_truncation_error", 0.0))
            ctm_cfg, _env_cache = _maybe_bump_chi(
                ctm_cfg,
                _env_cache,
                last_eps_t,
                base_charges=_bump_base_charges,
            )
            # Scheduled outer-loop χ bump (#453 / #455).  In PR 1
            # this still only fires on the budget-exhausted path —
            # PR 2 layers convergence/stall signals on top.
            if config.gs_chi_schedule_steps is not None:
                steps_in_stage = (step + 1) - stage_start_step
                _gn_for_bump = (
                    grad_norm_val
                    if grad_norm_val is not None
                    else (
                        _grad_l2_norm(grads)
                        if config.gs_conv_criterion != "dE"
                        else 0.0
                    )
                )
                ctm_cfg, _env_cache, new_stage_idx, bump_fired, _ = (
                    _advance_chi_stage_if_due(
                        ctm_cfg,
                        _env_cache,
                        chi_schedule=config.gs_chi_schedule_steps,
                        current_stage_idx=current_stage_idx,
                        steps_in_stage=steps_in_stage,
                        config=config,
                        grad_norm=_gn_for_bump,
                        delta_energy=delta_energy,
                        stall_count=stall_count,
                        base_charges=_bump_base_charges,
                    )
                )
                # Codex review (PR #467): the helper's idempotent-advance
                # branch returns ``bump_fired=False`` AND
                # ``new_stage_idx > current_stage_idx`` when a reactive
                # ε_T-bump already raised χ past the next stage's target.
                # The schedule index must STILL advance, and the
                # ``continue`` below must STILL fire — otherwise the
                # optimizer exits at the previous stage's converged
                # energy.  ``stage_advanced`` decouples the two signals.
                stage_advanced = new_stage_idx != current_stage_idx
                if stage_advanced:
                    current_stage_idx = new_stage_idx
                    stage_start_step = step + 1
            else:
                bump_fired = False
                stage_advanced = False
            if ctm_cfg.chi != chi_before_bump:
                # Reactive or scheduled bump fired — fresh landscape,
                # fresh stall budget (#464 codex review).
                stall_count = 0
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_A_flat = None
                    prev_grad_flat = None
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                    opt_state = optimizer.init(params)
            if _accepted_best_this_iter:
                best_env_cache = dict(_env_cache)
            if bump_fired or stage_advanced:
                # #455 PR2: converged at non-final stage → advance and
                # continue, NOT break.  The reset block above already
                # cleared stall_count, L-BFGS history, and opt_state at
                # the new χ; let the optimizer keep stepping on the
                # fresh landscape rather than exit prematurely.
                #
                # Codex review (PR #467): ``stage_advanced`` covers the
                # idempotent-advance branch where the reactive bump
                # already raised χ past the next stage's target — the
                # schedule index moved forward but ``bump_fired=False``.
                # Without the disjunction, we would fall through to
                # break and exit at stage N's converged energy without
                # ever running stage N+1.
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD step {step + 1}] converged at "
                        f"chi={chi_before_bump} → bumping to "
                        f"chi={ctm_cfg.chi} (#455 PR2)",
                        flush=True,
                    )
                # Force a checkpoint on stage advance before continuing so a
                # crash inside the next stage's first step doesn't roll back
                # across the boundary (mirror 2-site, Codex P2 #497).
                _maybe_save_1s_checkpoint(
                    step,
                    _chi_at_step_start,
                    _best_energy_at_step_start,
                    force_last=True,
                )
                continue
            if config.gs_verbose:
                if not logged:
                    _log_ad_step(
                        "1site-tensor",
                        step,
                        config.gs_num_steps,
                        energy_float,
                        delta_energy,
                        best_energy,
                        grad_norm=grad_norm_val,
                    )
                _log_ad_converged(
                    "1site-tensor",
                    step,
                    delta_energy,
                    config.gs_conv_tol,
                    grad_norm=grad_norm_val,
                    grad_norm_tol=config.gs_grad_norm_tol,
                    criterion=config.gs_conv_criterion,
                )
            _converged = True
            break

        # Compute search direction
        if is_cg:
            if config.gs_metric_precond and not use_c4v:
                from tenax.algorithms._metric_precond import precondition_gradient

                env_for_metric = _env_cache["envs"][(0, 0)]
                delta_metric = delta_energy if step > 0 else _tree_dot(grads, grads)
                z_dense = precondition_gradient(
                    A, env_for_metric, grads, delta_metric, config
                )
                z = _wrap_tensor(z_dense, grads)
                neg_z = jax.tree.map(lambda g: -g, z)
                if prev_precond_grad is not None and cg_direction is not None:
                    z_diff = jax.tree.map(lambda a, b: a - b, z, prev_precond_grad)
                    num = _tree_dot(grads, z_diff)
                    den = _tree_dot(prev_grad, prev_precond_grad)
                    beta = max(0.0, num / den) if den > 1e-30 else 0.0
                    cg_direction = _tree_add(neg_z, _tree_scale(cg_direction, beta))
                else:
                    cg_direction = neg_z
                prev_precond_grad = z
            else:
                neg_grad = jax.tree.map(lambda g: -g, grads)
                if prev_grad is not None and cg_direction is not None:
                    beta = _cg_beta_pr(grads, prev_grad)
                    cg_direction = _tree_add(neg_grad, _tree_scale(cg_direction, beta))
                else:
                    cg_direction = neg_grad
            prev_grad = grads
            direction = cg_direction
        elif is_metric_lbfgs:
            from tenax.algorithms._metric_precond import lbfgs_two_loop

            if use_c4v:
                # Plain L-BFGS in coefficient space (already reduced)
                g_flat = grads
                p_flat = params
            else:
                g_flat = grads.todense().reshape(-1)
                p_flat = A.todense().reshape(-1)

            # Update L-BFGS history
            if prev_A_flat is not None:
                s = p_flat - prev_A_flat
                y = g_flat - prev_grad_flat
                sy = float(jnp.real(jnp.vdot(s, y)))
                if sy > 1e-10:
                    rho = 1.0 / sy
                    lbfgs_history.append((s, y, rho))
                    if len(lbfgs_history) > 10:
                        lbfgs_history.pop(0)
            prev_A_flat = p_flat
            prev_grad_flat = g_flat

            if use_c4v:
                # Identity H0 in coefficient space (no metric preconditioning)
                direction = -lbfgs_two_loop(g_flat, lbfgs_history, lambda v: v)
            else:
                from tenax.algorithms._metric_precond import precondition_gradient

                env_for_metric = _env_cache["envs"][(0, 0)]
                delta_metric = (
                    delta_energy
                    if step > 0
                    else float(jnp.real(jnp.vdot(g_flat, g_flat)))
                )
                D_bond = A.todense().shape[0]
                d_loc = A.todense().shape[-1]

                def h0_matvec(v):
                    v_tensor = _wrap_tensor(
                        v.reshape(D_bond, D_bond, D_bond, D_bond, d_loc), A
                    )
                    result = precondition_gradient(
                        A, env_for_metric, v_tensor, delta_metric, config
                    )
                    return result.reshape(-1)

                direction_flat = lbfgs_two_loop(g_flat, lbfgs_history, h0_matvec)
                direction_dense = -direction_flat.reshape(
                    D_bond, D_bond, D_bond, D_bond, d_loc
                )
                direction = _wrap_tensor(direction_dense, A)
        elif optimizer is not None:
            updates, opt_state = optimizer.update(grads, opt_state, params)
            direction = updates
        else:
            direction = jax.tree.map(lambda g: -g, grads)

        if use_ls:
            # Snapshot for env-cache restore after the line search; see
            # ``_restore_env_cache_after_line_search`` (#502 Codex P1).
            _ls_env_snap = ("envs" in _env_cache, _env_cache.get("envs"))
            if line_search_method == "hager_zhang":
                from tenax.algorithms._line_search import hager_zhang_line_search

                slope = _tree_dot(grads, direction)
                if slope >= 0:
                    direction = jax.tree.map(lambda g: -g, grads)
                    slope = -_tree_dot(grads, grads)
                    if is_metric_lbfgs:
                        lbfgs_history.clear()

                hz_counter = {"phi": 0, "dphi": 0}

                def _phi(alpha):
                    hz_counter["phi"] += 1
                    trial = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    return loss_fn_fwd(trial)

                def _dphi(alpha):
                    hz_counter["dphi"] += 1
                    trial = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    _, g = jax.value_and_grad(loss_fn)(trial)
                    return _tree_dot(g, direction)

                dir_norm = math.sqrt(max(_tree_dot(direction, direction), 1e-30))
                param_norm = math.sqrt(max(_tree_dot(params, params), 1e-30))
                alpha0 = min(1.0, 0.1 * param_norm / dir_norm)

                alpha, f_alpha, converged = hager_zhang_line_search(
                    _phi,
                    _dphi,
                    energy_float,
                    slope,
                    alpha_init=alpha0,
                    rho=1.5,
                    max_step=2.0 * alpha0,
                    energy_bound=max(2.0, 2.0 * abs(best_energy)),
                    max_iter=config.gs_hz_max_iter,
                    bracket_only_phi=True,  # #504: skip dphi in bracket
                )
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD:1site-tensor] HZ probes phi={hz_counter['phi']} "
                        f"dphi={hz_counter['dphi']} alpha={alpha:.3e} "
                        f"converged={converged}",
                        flush=True,
                    )
                if _is_real_decrease(f_alpha, energy_float):
                    params = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                slope_bt = _tree_dot(grads, direction)
                if slope_bt >= 0:
                    direction = jax.tree.map(lambda g: -g, grads)
                    if is_metric_lbfgs:
                        lbfgs_history.clear()
                params, new_energy, step_size = _backtracking_line_search(
                    params,
                    direction,
                    grads,
                    energy_float,
                    loss_fn_fwd,
                    max_steps=config.gs_line_search_max_steps,
                )
                if _is_real_decrease(new_energy, energy_float):
                    stall_count = 0
                else:
                    stall_count += 1

            _restore_env_cache_after_line_search(_env_cache, _ls_env_snap)

            # Noise recovery on persistent stall (legacy; see issue #298).
            if (
                config.gs_stall_recovery == "noise"
                and stall_count > 0
                and stall_count <= config.gs_noise_recovery_retries
            ):
                noise_key = jax.random.PRNGKey(step * 1000 + stall_count)
                if use_c4v:
                    noise = config.gs_noise_amplitude * _random_noise(
                        noise_key, params.shape, params.dtype
                    )
                    params = params + noise * jnp.linalg.norm(params)
                    params = params / (jnp.linalg.norm(params) + 1e-10)
                else:
                    data = params.todense()
                    noise = config.gs_noise_amplitude * _random_noise(
                        noise_key, data.shape, data.dtype
                    )
                    noisy = data + noise * jnp.linalg.norm(data)
                    noisy = noisy / (jnp.linalg.norm(noisy) + 1e-10)
                    params = _wrap_tensor(noisy, params)
                if config.gs_verbose:
                    print(f"[iPEPS-AD] stall #{stall_count}, adding noise", flush=True)
                # Reset optimizer state
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_A_flat = None
                    prev_grad_flat = None
            elif config.gs_stall_recovery == "reset" and stall_count > 0:
                # Rollback to best on reset (#454). The CTM-error reset path
                # above (around L1206-1207) already does this for the
                # CTMRGGradientError branch; we extend the same pattern to the
                # Wolfe-failure path that was missed. #298's anti-rollback
                # evidence was on a pre-trifecta CTM stack (pre-PR #406 2x2
                # projector, pre-multisite-CTM rewrite, pre-PR #447 AD
                # stop_gradient) and no longer applies.
                if stall_count > config.gs_stall_recovery_retries:
                    # #455 PR2: at non-final χ stages, the stall-cap hit
                    # means "this χ is too small to make progress" —
                    # advance to the next stage and keep optimizing
                    # instead of returning best_energy.  Final stage
                    # falls through to the existing break.
                    if config.gs_chi_schedule_steps is not None:
                        steps_in_stage = (step + 1) - stage_start_step
                        _gn_for_bump = (
                            grad_norm_val
                            if grad_norm_val is not None
                            else (
                                _grad_l2_norm(grads)
                                if config.gs_conv_criterion != "dE"
                                else 0.0
                            )
                        )
                        (
                            ctm_cfg,
                            _env_cache,
                            new_stage_idx,
                            bump_fired,
                            _,
                        ) = _advance_chi_stage_if_due(
                            ctm_cfg,
                            _env_cache,
                            chi_schedule=config.gs_chi_schedule_steps,
                            current_stage_idx=current_stage_idx,
                            steps_in_stage=steps_in_stage,
                            config=config,
                            grad_norm=_gn_for_bump,
                            delta_energy=delta_energy,
                            stall_count=stall_count,
                            base_charges=_bump_base_charges,
                        )
                        # Codex review (PR #467): treat idempotent
                        # advance (bump_fired=False AND
                        # new_stage_idx>current_stage_idx) the same as
                        # a real bump for control-flow purposes —
                        # otherwise the stall-cap intercept would still
                        # break.  The reset block (params rollback,
                        # stall_count=0, L-BFGS clear, opt_state init)
                        # stays gated on bump_fired because it only
                        # makes sense when χ actually changed.
                        stage_advanced = new_stage_idx != current_stage_idx
                        if stage_advanced:
                            current_stage_idx = new_stage_idx
                            stage_start_step = step + 1
                        if bump_fired:
                            # Rollback params to best from the previous
                            # stage; _env_cache stays at the freshly
                            # padded post-bump state (the budget-path
                            # invariant — see _apply_chi_bump).
                            params = best_params
                            stall_count = 0
                            if is_metric_lbfgs:
                                lbfgs_history.clear()
                                prev_A_flat = None
                                prev_grad_flat = None
                            if is_cg:
                                cg_direction = None
                                prev_grad = None
                                prev_precond_grad = None
                            if (
                                optimizer is not None
                                and config.gs_optimizer.lower() == "lbfgs"
                            ):
                                opt_state = optimizer.init(params)
                            if config.gs_verbose:
                                print(
                                    f"[iPEPS-AD step {step + 1}] stall-cap at "
                                    f"chi={ctm_cfg.chi} → advancing to next "
                                    f"stage (#455 PR2)",
                                    flush=True,
                                )
                        if bump_fired or stage_advanced:
                            # Force a checkpoint on stall-cap stage advance
                            # before continuing (mirror 2-site, Codex P2 #497).
                            _maybe_save_1s_checkpoint(
                                step,
                                _chi_at_step_start,
                                _best_energy_at_step_start,
                                force_last=True,
                            )
                            continue
                    n_resets_done = stall_count - 1
                    if config.gs_verbose:
                        print(
                            f"[iPEPS-AD] stall budget exhausted after "
                            f"{n_resets_done} resets, "
                            f"returning best E={best_energy:.10f}",
                            flush=True,
                        )
                    break
                params = best_params
                # #518: ``best_env_cache`` may be at a stale χ if a
                # reactive/scheduled bump fired after it was last
                # snapshotted.  Clear instead of restoring; the next
                # CTM call cold-starts at the current ctm_cfg.chi.
                _drop_env_cache_for_reset(_env_cache)
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_A_flat = None
                    prev_grad_flat = None
                # Optax-backed L-BFGS stores curvature history in opt_state,
                # not in lbfgs_history.  Reinitialize it on the rolled-back
                # params so the next step really is steepest descent.
                if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                    opt_state = optimizer.init(params)
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD] stall #{stall_count}, reset L-BFGS history "
                        f"(rollback to best, retry "
                        f"{stall_count}/{config.gs_stall_recovery_retries})",
                        flush=True,
                    )
        else:
            params = optax.apply_updates(params, direction)
            if not use_c4v and not (_use_cg and _cg_map_fn is not None):
                params = params * (1.0 / (params.norm() + 1e-10))

        # variPEPS §2.8.2 auto-χ_E bump — must fire AFTER the line search
        # and parameter update so that ``value_and_grad`` (start of next
        # iteration), the metric preconditioner (when L-BFGS) and the
        # Hager-Zhang ``_phi`` / ``_dphi`` closures all evaluate at the
        # SAME χ. Otherwise the Wolfe sufficient-decrease and curvature
        # conditions compare ``f(0)`` / ``f'(0)`` at OLD χ against
        # ``f(α)`` at NEW χ — a valid step gets rejected (or an invalid
        # one accepted) purely because of the χ-induced discontinuity.
        # Issue #419.
        # Snapshot χ before either bump fires; the reset below triggers
        # on EITHER reactive or scheduled bump changing it (#464 codex review).
        chi_before_bump = ctm_cfg.chi
        last_eps_t = float(_env_cache.get("max_truncation_error", 0.0))
        ctm_cfg, _env_cache = _maybe_bump_chi(
            ctm_cfg,
            _env_cache,
            last_eps_t,
            base_charges=_bump_base_charges,
        )
        # Scheduled outer-loop χ bump (#453 / #455).  Composes with the
        # reactive bump above; ctm_cfg.chi_max caps both.  Per-stage
        # state (current_stage_idx, stage_start_step) drives the new
        # helper; #455 PR2 will add convergence/stall-cap triggers.
        if config.gs_chi_schedule_steps is not None:
            steps_in_stage = (step + 1) - stage_start_step
            _gn_for_bump = (
                grad_norm_val
                if grad_norm_val is not None
                else (_grad_l2_norm(grads) if config.gs_conv_criterion != "dE" else 0.0)
            )
            ctm_cfg, _env_cache, new_stage_idx, bump_fired, _should_break = (
                _advance_chi_stage_if_due(
                    ctm_cfg,
                    _env_cache,
                    chi_schedule=config.gs_chi_schedule_steps,
                    current_stage_idx=current_stage_idx,
                    steps_in_stage=steps_in_stage,
                    config=config,
                    grad_norm=_gn_for_bump,
                    delta_energy=delta_energy,
                    stall_count=stall_count,
                    base_charges=_bump_base_charges,
                )
            )
            stage_advanced = new_stage_idx != current_stage_idx
            if stage_advanced:
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD step {step + 1}] schedule advance: "
                        f"stage {current_stage_idx} → {new_stage_idx}, "
                        f"chi {chi_before_bump} → {ctm_cfg.chi} (#455)",
                        flush=True,
                    )
                current_stage_idx = new_stage_idx
                stage_start_step = step + 1
        if ctm_cfg.chi != chi_before_bump:
            # Reactive or scheduled bump fired — fresh landscape, fresh
            # stall budget (#464 codex review).
            stall_count = 0
            if is_metric_lbfgs:
                lbfgs_history.clear()
                prev_A_flat = None
                prev_grad_flat = None
            if is_cg:
                cg_direction = None
                prev_grad = None
                prev_precond_grad = None
            if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                opt_state = optimizer.init(params)
        # If best was accepted at this iter's pre-line-search params, refresh
        # the snapshot so its env matches the new ctm_cfg.chi. ``_env_cache``
        # still holds envs for those params (line search updates ``params``
        # but does not touch ``_env_cache``), padded to the new χ by the bump.
        if _accepted_best_this_iter:
            best_env_cache = dict(_env_cache)

        # End-of-step save: cadence-based + new-best detection.
        _maybe_save_1s_checkpoint(step, _chi_at_step_start, _best_energy_at_step_start)

    # Re-evaluate both final A and best_A with fully converged fresh CTM.
    # In-loop energies use warm-started CTM that can produce unphysical values
    # (non-variational at finite chi), so we compare fresh evaluations only.
    # Match in-loop CTM tolerances (#317) by reusing ctm_cfg directly.

    def _eval_fresh(p, env_init=None):
        """Evaluate energy with fully converged fresh CTM."""
        A_t = _params_to_A_norm(p)
        if use_split:
            # Final env is the split fixed point used by the gradient; return
            # the SplitCTMTensorEnv (not the fused CTMTensorEnv).
            env_ = _split_forward(A_t)
            E_ = float(compute_energy_split_ctm_tensor(A_t, env_, gate))
            return A_t, env_, E_
        envs, _ = python_loop_ctm_converge(
            {(0, 0): A_t},
            SINGLE_SITE_NEIGHBORS,
            **ctm_converge_kwargs(ctm_cfg, env_init=env_init),
        )
        env_ = envs[(0, 0)]
        if _use_cg:
            E_ = float(compute_energy_cg(A_t, env_, cg_gates, _cg_d_eff))
        else:
            E_ = float(compute_energy_ctm_tensor(A_t, env_, gate, d_phys))
        return A_t, env_, E_

    A_final, env_final, E_final = _eval_fresh(params, _env_cache.get("envs", None))

    # Pad best_env_cache envs to the current ctm_cfg.chi before use.
    # best_env_cache is a snapshot taken when best_energy was recorded, so its
    # env tensors may be at a smaller chi if the reactive auto-bump or the chi
    # schedule fired after that snapshot.  pad_dense_env_chi is a no-op when
    # chi_new == chi_old, so this is safe to call unconditionally.  Fixes #469.
    # NOTE: if a new _eval_fresh(best_params, best_env_cache["envs"]) call site
    # is added later, it must reapply this padding — there's no producer-side
    # guarantee that the snapshot is at ctm_cfg.chi.
    _best_env_init = best_env_cache.get("envs", None)
    if _best_env_init:
        # Issue #514: ``ctm_cfg.chi`` may be stale if the in-CTM χ-bump fired
        # during a gradient evaluation.  Pad to the larger of the static
        # config chi and the env_cache's actual chi so we never try to
        # shrink a bumped env.
        # Consult both the live cache *and* the best-env snapshot:
        # rollback paths may have cleared ``_env_cache`` after the
        # snapshot was taken, leaving the bumped chi reachable only via
        # ``_best_env_init`` itself (Codex P2 on PR #542).
        _target_chi = max(
            ctm_cfg.chi,
            _env_dict_chi(_env_cache.get("envs")) or 0,
            _env_dict_chi(_best_env_init) or 0,
        )
        _best_env_init = {
            c: pad_dense_env_chi(
                _best_env_init[c], _target_chi, base_charges=_bump_base_charges
            )
            for c in _best_env_init
        }

    if best_params is not params:
        _, env_best, E_best_fresh = _eval_fresh(best_params, _best_env_init)
    else:
        E_best_fresh = E_final

    if E_final <= E_best_fresh:
        env, E_gs = env_final, E_final
    else:
        A_final, _, _ = _eval_fresh(best_params, _best_env_init)
        env, E_gs = env_best, E_best_fresh
    if config.gs_verbose:
        print(f"[iPEPS-AD:1site-tensor] final E={E_gs:.10f}", flush=True)

    if config.return_history:
        history = {
            "energies": _history_energies,
            "step_times": _history_step_times,
            "jit_compile_time": _jit_compile_time,
            "num_steps": len(_history_energies),
            "converged": _converged,
        }
        return A_final, env, E_gs, history
    return A_final, env, E_gs


def _maybe_relabel_su_tensor(t: Tensor) -> Tensor:
    """Relabel simple-update tensor labels to standard iPEPS convention if needed."""
    _SU_LABEL_MAP = {"up": "u", "down": "d", "left": "l", "right": "r"}
    current_labels = {idx.label for idx in t.indices}
    if current_labels & {"up", "down", "left", "right"}:
        for old, new in _SU_LABEL_MAP.items():
            if old in current_labels:
                t = t.relabel(old, new)
    return t


def _optimize_gs_ad_2site(
    hamiltonian_gate: jax.Array,
    AB_init: tuple[jax.Array, jax.Array] | tuple[Tensor, Tensor] | None,
    config: iPEPSConfig,
):
    """AD-based ground state optimization for 2-site iPEPS unit cell.

    Uses implicit differentiation through the 2-site CTM fixed point
    to compute gradients of energy w.r.t. both site tensors (A, B).

    Always uses the Tensor-protocol path. Raw ``jax.Array`` inputs are
    automatically wrapped as ``DenseTensor`` with trivial U(1) charges.
    """
    if AB_init is not None:
        if not isinstance(AB_init, tuple) or len(AB_init) != 2:
            raise TypeError(
                "For unit_cell='2site', A_init must be None or a tuple (A, B)."
            )
        # Wrap raw arrays as DenseTensor
        AB_init = tuple(
            _wrap_as_dense_tensor(t) if not isinstance(t, Tensor) else t
            for t in AB_init
        )

    if AB_init is None:
        gate = (
            hamiltonian_gate.todense()
            if isinstance(hamiltonian_gate, Tensor)
            else jnp.array(hamiltonian_gate)
        )
        d_phys = gate.shape[0]
        D = config.max_bond_dim

        if config.su_init:
            from tenax.algorithms.ipeps import ipeps

            su_config = iPEPSConfig(
                max_bond_dim=D,
                num_imaginary_steps=config.num_imaginary_steps,
                dt=config.dt,
                ctm=config.ctm,
            )
            _, (A_su, B_su), _ = ipeps(gate, None, su_config)
            AB_init = (A_su, B_su)
        else:
            # Random complex128 initialization for 2-site AD (matches variPEPS)
            key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
            kA1, kA2 = jax.random.split(key_A)
            kB1, kB2 = jax.random.split(key_B)
            A_data = jax.random.normal(
                kA1, (D, D, D, D, d_phys)
            ) + 1j * jax.random.normal(kA2, (D, D, D, D, d_phys))
            B_data = jax.random.normal(
                kB1, (D, D, D, D, d_phys)
            ) + 1j * jax.random.normal(kB2, (D, D, D, D, d_phys))
            A = _wrap_as_dense_tensor(A_data)
            B = _wrap_as_dense_tensor(B_data)
            AB_init = (A, B)

    return _optimize_gs_ad_tensor_2site(hamiltonian_gate, AB_init, config)


def _optimize_gs_ad_tensor_2site(
    hamiltonian_gate: jax.Array,
    AB_init: tuple[Tensor, Tensor],
    config: iPEPSConfig,
):
    """AD-based ground state optimization for 2-site Tensor-protocol iPEPS.

    Uses ``ctm_tensor_converge`` with implicit differentiation through
    the 2-site Tensor-protocol CTM.

    With ``gs_c4v=True`` (recommended), optimizes a single C4v coefficient
    vector and derives B from A via sublattice rotation on the physical leg.
    This enforces A and B to be related by a spin-π rotation, preventing
    the A/B drift that plagues independent 2-site optimization.

    .. warning::

        Without ``gs_c4v=True``, the 2-site AD optimizer is
        unstable and produces unphysical energies.  For antiferromagnetic
        models, prefer ``gs_c4v=True`` or 1-site optimization with
        ``sublattice_rotate_gate()`` + ``gs_c4v=True``.
    """
    config = _normalize_stall_recovery(config, unit_cell="2site")
    use_c4v = config.gs_c4v
    if not use_c4v:
        import warnings

        if not config.gs_implicit_ad:
            # Issue #328: non-C4v 2-site *explicit* AD is non-variational at
            # finite chi for general models.  The forward-mode AD chain
            # through the finite-sweep CTM reliably finds unphysical
            # minima of <H>_ctm/<psi|psi>_ctm where the surrogate energy
            # falls well below the true ground state.  Bench evidence
            # (D=2 Heisenberg, chi=16, 30 AD steps, Armijo L-BFGS):
            #   gs_explicit_ad_steps=10 -> E = -0.06  (stuck)
            #   gs_explicit_ad_steps=30 -> E = -0.26  (descending, non-var)
            #   gs_explicit_ad_steps=60 -> E = -1.18  (below physical)
            # vs physical ground state E/site = -0.6694.  Implicit AD
            # (gs_implicit_ad=True) lands at E = -0.566 for the same
            # config — variational and close to physical.
            warnings.warn(
                "Non-C4v 2-site *explicit* AD (gs_c4v=False, "
                "gs_implicit_ad=False) is known to be non-variational at "
                "finite chi: the optimizer drifts below the physical "
                "ground state (see issue #328). Set gs_implicit_ad=True "
                "to use the implicit-AD path, which is variational at "
                "chi >= 16 for 2-site Heisenberg. For antiferromagnetic "
                "bipartite models, gs_c4v=True is also a stable option.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                "2-site AD with gs_c4v=False uses the implicit-AD path. "
                "This is variational **only when the CTM environment is a "
                "converged fixed point at the current chi**.  This condition "
                "is automatically satisfied for fixed chi (no ramp, no bump) "
                "and for runs using ctmrg_heuristic_increase_chi=True (in-CTM "
                "bump, #492/#514).  It is NOT satisfied immediately after a "
                "scheduled chi_ramp stage advance or an end-of-outer-step "
                "chi_auto_bump event, where env rows for newly-active chi "
                "indices are zero-initialized; several gradient evaluations "
                "may be required before CTM repopulates them, during which "
                "the optimizer can descend to a non-physical ghost minimum "
                "(see issue #511).  For chi-ramped runs prefer "
                "ctmrg_heuristic_increase_chi=True over scheduled chi_ramp "
                "or end-of-outer-step chi_auto_bump.  Without raising chi "
                "manually, chi >= 16 is typically needed for generic 2-site "
                "Heisenberg.  For antiferromagnetic bipartite models, "
                "consider gs_c4v=True or 1-site with sublattice_rotate_gate().",
                stacklevel=2,
            )
    import optax

    from tenax.algorithms._checkpoint import (
        _config_to_dict,
        checkpoint_exists,
        gate_fingerprint,
        load_checkpoint,
        save_checkpoint,
        validate_config,
    )
    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor import (
        compute_energy_ctm_tensor_2site,
    )
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    from tenax.algorithms.ad_utils import (
        CTMRGGradientError,
        _wrap_tensor,
    )
    from tenax.algorithms.ipeps_ad_policy import (
        ctm_converge_kwargs,
        make_ctm_energy_fn,
    )

    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d_phys = gate.shape[0]

    A, B = AB_init
    A = A * (1.0 / (A.norm() + 1e-10))
    B = B * (1.0 / (B.norm() + 1e-10))

    ctm_cfg_2s = build_ad_ctm_config(config)
    use_explicit = not config.gs_implicit_ad
    explicit_steps = config.gs_explicit_ad_steps
    explicit_warmup = config.gs_explicit_ad_warmup
    explicit_backward_steps = config.gs_explicit_ad_backward_steps

    # Env warm-start cache — replaces flat env_leaves threading.
    _env_cache_2s: dict[str, dict] = {}

    if use_c4v:
        from tenax.algorithms.ipeps import (
            build_c4v_basis,
            c4v_coeffs_from_tensor,
            c4v_tensor_from_coeffs,
        )

        D_bond = A.todense().shape[0]
        d_loc = A.todense().shape[-1]
        if d_loc != 2:
            raise ValueError(
                "2-site C4v shared-tensor path requires physical dimension "
                f"d=2 (spin-1/2), got d={d_loc}. The sublattice rotation "
                "U = e^{i π σ^y/2} is spin-1/2 specific; higher-spin models "
                "need a model-specific sublattice rotation."
            )
        if config.gs_stall_recovery == "noise":
            raise ValueError(
                "gs_stall_recovery='noise' is incompatible with "
                "gs_c4v=True on the 2-site path (noise recovery assumes "
                "tuple-of-DenseTensor params, not a C4v coefficient vector). "
                "Use gs_stall_recovery='reset' instead."
            )
        tensor_shape = (D_bond, D_bond, D_bond, D_bond, d_loc)
        c4v_basis = jnp.array(build_c4v_basis(D_bond, d_loc))
        c4v_coeffs = c4v_coeffs_from_tensor(A.todense(), c4v_basis)
        A_indices = A.indices
        # Sublattice rotation: B = A with physical leg rotated by U = e^{iπ σ^y/2}
        _U_sub = jnp.array([[0.0, 1.0], [-1.0, 0.0]], dtype=A.todense().dtype)
    else:
        c4v_basis = None
        tensor_shape = None
        A_indices = None
        _U_sub = None

    def _c4v_AB(coeffs_):
        """Build normalized (A, B) DenseTensors from a single C4v coeff vector."""
        A_data = c4v_tensor_from_coeffs(coeffs_, c4v_basis, tensor_shape)
        A_data = A_data / (jnp.linalg.norm(A_data) + 1e-10)
        B_data = jnp.einsum("luRDs,sS->luRDS", A_data, _U_sub)
        return DenseTensor(A_data, A_indices), DenseTensor(B_data, A_indices)

    def _energy_fn_2site(site_tensors, envs, gate_):
        """2-site energy from site_tensors dict and envs dict."""
        return compute_energy_ctm_tensor_2site(
            site_tensors[(0, 0)],
            site_tensors[(1, 0)],
            envs[(0, 0)],
            envs[(1, 0)],
            gate_,
            d_phys,
        )

    _ctm_energy_fn_2s = make_ctm_energy_fn(
        neighbors=CHECKERBOARD_NEIGHBORS,
        gate=gate,
        get_ctm_cfg=lambda: ctm_cfg_2s,
        env_cache=_env_cache_2s,
        use_explicit=use_explicit,
        explicit_warmup=explicit_warmup,
        explicit_steps=explicit_steps,
        explicit_backward_steps=explicit_backward_steps,
        energy_fn=_energy_fn_2site,
        recipe=config.gs_recipe,
    )

    def loss_fn(params):
        if use_c4v:
            A_norm, B_norm = _c4v_AB(params)
        else:
            A_p, B_p = params
            A_norm = A_p * (1.0 / (A_p.norm() + 1e-10))
            B_norm = B_p * (1.0 / (B_p.norm() + 1e-10))
        site_tensors = {(0, 0): A_norm, (1, 0): B_norm}
        energy = _ctm_energy_fn_2s(site_tensors)
        return energy

    def _update_env_cache_2s(params):
        """Re-run forward CTM (no grad) to warm-start next step."""
        if use_c4v:
            A_norm, B_norm = _c4v_AB(params)
        else:
            A_p, B_p = params
            A_norm = A_p * (1.0 / (A_p.norm() + 1e-10))
            B_norm = B_p * (1.0 / (B_p.norm() + 1e-10))
        site_tensors = {(0, 0): A_norm, (1, 0): B_norm}
        envs, info = python_loop_ctm_converge(
            site_tensors,
            CHECKERBOARD_NEIGHBORS,
            **ctm_converge_kwargs(ctm_cfg_2s, env_init=_env_cache_2s.get("envs", None)),
        )
        _env_cache_2s["envs"] = envs
        # Capture ``info.max_truncation_error`` so the end-of-step
        # ``_maybe_bump_chi`` reactive trigger (#472) has an ε_T to
        # compare against.  As of #474 the 2x2 plaquette projector
        # returns a real ε_T (previously a 0.0 placeholder), so reactive
        # bumps fire on the 2-site forward stack with the default recipe.
        _env_cache_2s["max_truncation_error"] = float(info.max_truncation_error)
        # Also capture ``info.max_smallest_S`` (smallest retained SV /
        # largest, per projector SVD) so ``chi_auto_bump_metric =
        # "norm_smallest_S"`` can match variPEPS's literal trigger.
        _env_cache_2s["max_smallest_S"] = float(info.max_smallest_S)

    params = c4v_coeffs if use_c4v else (A, B)
    is_metric_lbfgs = (
        config.gs_metric_precond and config.gs_optimizer.lower() == "lbfgs"
    )
    optimizer = None if is_metric_lbfgs else _build_optimizer(config)
    opt_state = optimizer.init(params) if optimizer is not None else None
    use_ls = _use_line_search(config)
    # Resolve ``gs_line_search_method`` once per dispatch (#509 / codex P2 on
    # #549) — see comment in the 1-site path.  2-site uses ``ctm_cfg_2s``.
    line_search_method = _resolve_line_search_method(config, ctm_cfg_2s)
    is_cg = config.gs_optimizer.lower() == "cg"

    best_energy = float("inf")
    best_params = params
    best_env_cache_2s: dict[str, dict] = {}  # tracked for fresh-CTM warm-start (#317)
    prev_energy = float("inf")
    prev_grad = None
    cg_direction = None
    prev_precond_grad = None
    log_interval = config.gs_log_interval
    lbfgs_history: list = []
    prev_params_flat: jnp.ndarray | None = None
    prev_grad_flat: jnp.ndarray | None = None
    stall_count = 0  # noise recovery: consecutive line search failures
    current_stage_idx = 0
    stage_start_step = 0
    # Rolling buffer of accepted ``||grad||_2`` for the gradient-spike
    # guard (config.gs_grad_spike_ratio).  Bad implicit-AD steps land in
    # a region where the gradient explodes by >10x relative to recent
    # magnitudes; rejecting those steps prevents the v9-style collapse
    # below the QMC floor.
    recent_gnorms_2s: list[float] = []
    # Consecutive "at chi_max with indicator above threshold" counter for
    # the variPEPS §2.8.2 bail-out (config.gs_chi_ceiling_bailout).  Reset
    # on chi bump or when the indicator dips below threshold.
    chi_ceiling_consecutive_2s = 0

    # Optional trajectory capture (config.return_history).  Always allocated
    # but only populated/returned when the flag is set.
    _history_energies: list[float] = []
    _history_step_times: list[float] = []
    _jit_compile_time: float = 0.0
    _first_step = True
    _converged = False

    # CTM conv_tol schedule (shared helper with 1-site optimizer)
    _conv_tol_schedule_2s = config.gs_ctm_conv_tol_schedule
    _current_conv_tol_2s = ctm_cfg_2s.conv_tol

    # CTM max_iter schedule (#507): same lookup contract as conv_tol.
    _max_iter_schedule_2s = config.gs_ctm_max_iter_schedule
    _current_max_iter_2s = ctm_cfg_2s.max_iter

    # CTM plateau-patience schedule (independent of conv_tol — see
    # iPEPSConfig.gs_plateau_patience_schedule docstring).
    _patience_schedule_2s = config.gs_plateau_patience_schedule
    _current_patience_2s = ctm_cfg_2s.plateau_patience

    def _get_scheduled_conv_tol_2s(step_idx, num_steps):
        if _conv_tol_schedule_2s is None:
            return _current_conv_tol_2s
        frac = step_idx / max(num_steps, 1)
        tol = _conv_tol_schedule_2s[0][1]
        for threshold, t in _conv_tol_schedule_2s:
            if frac >= threshold:
                tol = t
        return tol

    def _get_scheduled_max_iter_2s(step_idx, num_steps):
        """Look up max_iter from schedule based on step fraction (issue #507)."""
        if _max_iter_schedule_2s is None:
            return _current_max_iter_2s
        frac = step_idx / max(num_steps, 1)
        mi = _max_iter_schedule_2s[0][1]
        for threshold, m in _max_iter_schedule_2s:
            if frac >= threshold:
                mi = m
        return mi

    def _get_scheduled_plateau_patience_2s(step_idx, num_steps):
        if _patience_schedule_2s is None:
            return _current_patience_2s
        frac = step_idx / max(num_steps, 1)
        patience = _patience_schedule_2s[0][1]
        for threshold, p in _patience_schedule_2s:
            if frac >= threshold:
                patience = p
        return patience

    def loss_fn_fwd(params_):
        """Forward-only loss for line search — warm-starts CTM from env cache."""
        if use_c4v:
            A_norm, B_norm = _c4v_AB(params_)
        else:
            A_p, B_p = params_
            A_norm = A_p * (1.0 / (A_p.norm() + 1e-10))
            B_norm = B_p * (1.0 / (B_p.norm() + 1e-10))
        site_tensors = {(0, 0): A_norm, (1, 0): B_norm}
        envs, _ = python_loop_ctm_converge(
            site_tensors,
            CHECKERBOARD_NEIGHBORS,
            **ctm_converge_kwargs(
                ctm_cfg_2s,
                env_init=_env_cache_2s.get("envs", None),
                for_probe=True,
            ),
        )
        # Issue #502: see 1-site loss_fn_fwd above for rationale.
        _env_cache_2s["envs"] = envs
        return float(
            compute_energy_ctm_tensor_2site(
                A_norm,
                B_norm,
                envs[(0, 0)],
                envs[(1, 0)],
                gate,
                d_phys,
            )
        )

    # Base charges for SymmetricTensor env-padding (#453).  Same shape
    # infrastructure used by the 1-site auto-bump.  ``A``'s bond charges
    # are fixed across optimization steps, so compute once outside the
    # loop.  Only needed when either the reactive auto-bump or the
    # scheduled-bump (gs_chi_schedule_steps) may fire.  C4v always
    # works in dense coefficient space, so we only inspect A in the
    # non-C4v branch.
    _bump_base_charges_2s: np.ndarray | None = None
    if (ctm_cfg_2s.chi_auto_bump or config.gs_chi_schedule_steps is not None) and (
        not use_c4v
    ):
        _A_init = A  # 2-site non-C4v: ``A`` (the first tensor); both A and B
        # share the same bond charges in a uniform iPEPS.
        if isinstance(_A_init, SymmetricTensor):
            from tenax.algorithms._ctm_tensor_convergence import _get_base_charges
            from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor

            _bump_base_charges_2s = _get_base_charges(
                _build_double_layer_tensor(_A_init)
            )

    # ---- Checkpoint resume ----------------------------------------
    # When gs_resume=True and a checkpoint exists at gs_checkpoint_path,
    # restore optimizer state and pick up at step `saved_step + 1`.
    # All other branches keep the fresh-init values set above.
    _gate_fp = gate_fingerprint(gate) if config.gs_checkpoint_path is not None else None
    start_step = 0
    if config.gs_resume:
        # Fail fast on a typo'd / deleted / never-written checkpoint
        # path rather than silently falling through to fresh init —
        # the user explicitly asked to resume, and a silent fresh
        # start would discard their intended long-run state and begin
        # overwriting checkpoints in the wrong place (Codex P2 review
        # on PR #497).
        if not checkpoint_exists(config.gs_checkpoint_path):
            raise FileNotFoundError(
                f"gs_resume=True but no checkpoint found at "
                f"{config.gs_checkpoint_path!r} (looked for "
                f"'ckpt.last.pkl').  Either point gs_checkpoint_path "
                f"at the directory of a prior run, or set "
                f"gs_resume=False to start fresh."
            )
        bundle = load_checkpoint(config.gs_checkpoint_path)
        validate_config(bundle.get("config", {}), config)

        # Reject silent gate swaps: the checkpoint records the
        # fingerprint of the hamiltonian it was optimized against, so
        # resuming with a different gate (same shape but different
        # bytes — e.g. transverse-field model vs Heisenberg, or J=1
        # vs J=2) is a fatal config-style mismatch.
        saved_fp = bundle.get("hamiltonian_fingerprint")
        if saved_fp is not None and tuple(saved_fp) != _gate_fp:
            raise ValueError(
                "Cannot resume: the hamiltonian gate has changed since the "
                "checkpoint was written.\n"
                f"  saved fingerprint:   {tuple(saved_fp)!r}\n"
                f"  current fingerprint: {_gate_fp!r}\n"
                "If this is intentional, start a fresh run (delete the "
                "checkpoint or point gs_checkpoint_path elsewhere)."
            )

        # Optimizer-defining fields (gs_optimizer, gs_metric_precond)
        # are validated soft, but the saved ``opt_state`` /
        # L-BFGS curvature pytree are only valid for the optimizer
        # they came from.  If either changed across resume, restore
        # the model state (params, envs, energies, schedule) but
        # discard the optimizer history so the new optimizer starts
        # from a fresh state.
        saved_cfg = bundle.get("config", {})
        _opt_compat = (
            saved_cfg.get("gs_optimizer") == config.gs_optimizer
            and saved_cfg.get("gs_metric_precond") == config.gs_metric_precond
        )

        params = bundle["params"]
        best_params = bundle["best_params"]
        best_energy = float(bundle["best_energy"])
        prev_energy = float(bundle["prev_energy"])
        _drop_env_cache_for_reset(_env_cache_2s)
        _env_cache_2s.update(bundle.get("env_cache", {}))
        best_env_cache_2s = dict(bundle.get("best_env_cache", {}))
        stall_count = int(bundle.get("stall_count", 0))

        if _opt_compat:
            if optimizer is not None and bundle.get("opt_state") is not None:
                opt_state = bundle["opt_state"]
            lbfgs_history = list(bundle.get("lbfgs_history") or [])
            prev_params_flat = bundle.get("prev_params_flat")
            prev_grad_flat = bundle.get("prev_grad_flat")
            cg_direction = bundle.get("cg_direction")
            prev_grad = bundle.get("prev_grad")
            prev_precond_grad = bundle.get("prev_precond_grad")
        else:
            import warnings as _warnings

            _warnings.warn(
                "Optimizer-defining config differs from checkpoint "
                f"(saved: gs_optimizer={saved_cfg.get('gs_optimizer')!r}, "
                f"gs_metric_precond={saved_cfg.get('gs_metric_precond')!r}; "
                f"current: gs_optimizer={config.gs_optimizer!r}, "
                f"gs_metric_precond={config.gs_metric_precond!r}). "
                "Restoring params/envs/energies but discarding saved "
                "optimizer history (curvature/momentum will restart fresh).",
                stacklevel=2,
            )

        current_stage_idx = int(bundle.get("current_stage_idx", 0))
        stage_start_step = int(bundle.get("stage_start_step", 0))
        saved_chi = bundle.get("ctm_cfg_chi")
        if saved_chi is not None and int(saved_chi) != ctm_cfg_2s.chi:
            ctm_cfg_2s = _replace(ctm_cfg_2s, chi=int(saved_chi))
        saved_conv_tol = bundle.get("current_conv_tol")
        if saved_conv_tol is not None:
            _current_conv_tol_2s = float(saved_conv_tol)
            ctm_cfg_2s = _replace(ctm_cfg_2s, conv_tol=_current_conv_tol_2s)
        saved_patience = bundle.get("current_patience")
        if saved_patience is not None:
            _current_patience_2s = (
                int(saved_patience) if saved_patience is not None else None
            )
            ctm_cfg_2s = _replace(ctm_cfg_2s, plateau_patience=_current_patience_2s)

        start_step = int(bundle["step"]) + 1
        if config.gs_verbose:
            print(
                f"[iPEPS-AD:2site-tensor] resumed from step {start_step} "
                f"(best E={best_energy:.10f}, chi={ctm_cfg_2s.chi})",
                flush=True,
            )

    # Nested save helper so the three checkpoint-save call sites
    # (end-of-step, mid-loop convergence-bump intercept, mid-loop
    # stall-cap stage-advance) stay in sync.  Captures most state by
    # closure; step-local snapshots (``chi_before``, ``e_prev``) are
    # passed explicitly so the caller controls the "did chi change /
    # did we accept a new best this step" detection.
    def _maybe_save_2s_checkpoint(
        step: int,
        chi_before: int,
        e_prev: float,
        *,
        force_last: bool = False,
    ) -> None:
        if config.gs_checkpoint_path is None:
            return
        chi_changed = ctm_cfg_2s.chi != chi_before
        is_new_best = best_energy < e_prev
        should_save_last = (
            force_last or chi_changed or (step + 1) % config.gs_checkpoint_every == 0
        )
        if not (should_save_last or is_new_best):
            return
        _ckpt_state = {
            "step": step,
            "config": _config_to_dict(config),
            "hamiltonian_fingerprint": _gate_fp,
            "params": params,
            "best_params": best_params,
            "best_energy": float(best_energy),
            "prev_energy": float(prev_energy),
            "env_cache": dict(_env_cache_2s),
            "best_env_cache": dict(best_env_cache_2s),
            "opt_state": opt_state,
            "lbfgs_history": list(lbfgs_history),
            "prev_params_flat": prev_params_flat,
            "prev_grad_flat": prev_grad_flat,
            "cg_direction": cg_direction,
            "prev_grad": prev_grad,
            "prev_precond_grad": prev_precond_grad,
            "stall_count": stall_count,
            "current_stage_idx": current_stage_idx,
            "stage_start_step": stage_start_step,
            "ctm_cfg_chi": ctm_cfg_2s.chi,
            "current_conv_tol": _current_conv_tol_2s,
            "current_patience": _current_patience_2s,
        }
        if should_save_last:
            save_checkpoint(_ckpt_state, config.gs_checkpoint_path)
        if is_new_best:
            save_checkpoint(_ckpt_state, config.gs_checkpoint_path, is_best=True)

    # Enable per-backward ``||lam||`` extraction in the implicit-AD F3 path
    # only when a consumer is reading them (verbose logging in this loop).
    # Default-off saves one ``jax.device_get`` per env-leaf per gradient
    # call on production runs (codex PR #524 P2).  Restore via ``finally``
    # below so an exception in the loop or in the post-loop fresh-CTM eval
    # does not leak the diagnostics flag (codex PR #524 follow-up).
    if config.gs_implicit_ad and config.gs_verbose:
        from tenax.algorithms._ctm_energy_ad import (
            set_implicit_ad_norm_diagnostics,
        )

        _prev_norm_diag = set_implicit_ad_norm_diagnostics(True)
    else:
        set_implicit_ad_norm_diagnostics = None  # type: ignore[assignment]
        _prev_norm_diag = None

    try:
        _log_ad_compile_notice(config)
        for step in range(start_step, config.gs_num_steps):
            # Snapshots for checkpoint "did chi change / new best" detection.
            # ``best_energy`` only decreases, so a strict < comparison after
            # the step body identifies a freshly-accepted best.  Both
            # snapshots are passed to ``_maybe_save_2s_checkpoint`` at every
            # save call site so the helper can decide what to flush.
            _best_energy_at_step_start = best_energy
            _chi_at_step_start = ctm_cfg_2s.chi
            # Update conv_tol if schedule is active
            if _conv_tol_schedule_2s is not None:
                new_tol = _get_scheduled_conv_tol_2s(step, config.gs_num_steps)
                if new_tol != _current_conv_tol_2s:
                    _current_conv_tol_2s = new_tol
                    ctm_cfg_2s = _replace(ctm_cfg_2s, conv_tol=new_tol)
            # Update max_iter if schedule is active (#507)
            if _max_iter_schedule_2s is not None:
                new_max_iter = _get_scheduled_max_iter_2s(step, config.gs_num_steps)
                if new_max_iter != _current_max_iter_2s:
                    _current_max_iter_2s = new_max_iter
                    ctm_cfg_2s = _replace(ctm_cfg_2s, max_iter=new_max_iter)
            # Update plateau_patience if schedule is active
            if _patience_schedule_2s is not None:
                new_patience = _get_scheduled_plateau_patience_2s(
                    step, config.gs_num_steps
                )
                if new_patience != _current_patience_2s:
                    _current_patience_2s = new_patience
                    ctm_cfg_2s = _replace(ctm_cfg_2s, plateau_patience=new_patience)

            if config.return_history:
                _step_t0 = _time.perf_counter()
            try:
                energy_val, grads = jax.value_and_grad(loss_fn)(params)
            except CTMRGGradientError as exc:
                _logger.warning(
                    "[iPEPS-AD] Arnoldi precheck: rho(J^T) = %.4f >= 1 at step %d — "
                    "skipping, triggering stall recovery",
                    exc.spectral_radius,
                    step,
                )
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD:2site-tensor] step {step + 1}/{config.gs_num_steps} "
                        f"rho(J^T)={exc.spectral_radius:.4f} — stall recovery",
                        flush=True,
                    )
                stall_count += 1
                if (
                    config.gs_stall_recovery == "noise"
                    and stall_count <= config.gs_noise_recovery_retries
                ):
                    noise_key = jax.random.PRNGKey(step * 1000 + stall_count)
                    if use_c4v:
                        noise = config.gs_noise_amplitude * _random_noise(
                            noise_key, params.shape, params.dtype
                        )
                        params = params + noise * jnp.linalg.norm(params)
                        params = params / (jnp.linalg.norm(params) + 1e-10)
                    else:
                        noisy_params = []
                        for i, p in enumerate(params):
                            k = jax.random.fold_in(noise_key, i)
                            data = p.todense()
                            noise = config.gs_noise_amplitude * _random_noise(
                                k, data.shape, data.dtype
                            )
                            noisy = data + noise * jnp.linalg.norm(data)
                            noisy = noisy / (jnp.linalg.norm(noisy) + 1e-10)
                            noisy_params.append(_wrap_tensor(noisy, p))
                        params = tuple(noisy_params)
                    if is_metric_lbfgs:
                        lbfgs_history.clear()
                        prev_params_flat = None
                        prev_grad_flat = None
                    if is_cg:
                        cg_direction = None
                        prev_grad = None
                        prev_precond_grad = None
                elif config.gs_stall_recovery == "reset":
                    # Cap on CTMRGGradientError-driven reset path (#454 follow-up,
                    # codex review on PR #457).
                    if stall_count > config.gs_stall_recovery_retries:
                        n_resets_done = stall_count - 1
                        if config.gs_verbose:
                            print(
                                f"[iPEPS-AD] CTM-error stall budget exhausted after "
                                f"{n_resets_done} resets, "
                                f"returning best E={best_energy:.10f}",
                                flush=True,
                            )
                        break
                    params = best_params
                    # #518: ``best_env_cache_2s`` may be at a stale χ if a
                    # reactive/scheduled bump fired after it was last
                    # snapshotted.  Clear instead of restoring; the next
                    # CTM call cold-starts at the current ctm_cfg_2s.chi.
                    _drop_env_cache_for_reset(_env_cache_2s)
                    if is_metric_lbfgs:
                        lbfgs_history.clear()
                        prev_params_flat = None
                        prev_grad_flat = None
                    if is_cg:
                        cg_direction = None
                        prev_grad = None
                        prev_precond_grad = None
                    if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                        opt_state = optimizer.init(params)
                # Counter-reset contract (codex PR #524 follow-up): the
                # ``gs_chi_ceiling_bailout`` docstring states the streak
                # resets on stall recovery.  This branch is the
                # ``CTMRGGradientError`` recovery path, so clear the
                # counter before the ``continue`` — otherwise a
                # recovered CTM-error step would still count toward the
                # K-consecutive bail-out trigger.
                chi_ceiling_consecutive_2s = 0
                continue
            energy_float = float(energy_val)

            if config.return_history:
                _step_dt = _time.perf_counter() - _step_t0
                if _first_step:
                    _jit_compile_time = float(_step_dt)
                    _first_step = False
                else:
                    _history_step_times.append(float(_step_dt))
                _history_energies.append(energy_float)

            # Update env cache for warm-starting next step
            _update_env_cache_2s(params)

            grad_norm_val = _grad_l2_norm(grads)

            # Gradient-spike guard: reject steps whose new ||grad||_2 exceeds
            # ``gs_grad_spike_ratio * max(median(recent), 1.0)``.  See
            # project_v9_collapse_findings.md — non-variational drift shows up
            # as a >10x gradient blowup before the best_energy crosses the QMC
            # floor.  We check BEFORE updating best so the bad step never gets
            # accepted as the new best.
            if (
                config.gs_grad_spike_ratio is not None
                and len(recent_gnorms_2s) >= 1
                and best_energy < float("inf")
            ):
                # ``len >= 1`` (was ``>= 2``, codex PR #524 P2): with
                # ``gs_grad_spike_window == 1`` the buffer is trimmed back to
                # length 1 each step, so the previous ``>= 2`` guard silently
                # disabled the spike check for that valid config.  ``len == 1``
                # is enough: the median is just the prior step's gradient norm,
                # which is the right baseline for "is this step a >5x jump?".
                spike_floor = max(float(np.median(recent_gnorms_2s)), 1.0)
                if grad_norm_val > config.gs_grad_spike_ratio * spike_floor:
                    params = best_params
                    _drop_env_cache_for_reset(_env_cache_2s)
                    # Clear the rolling buffer too (codex PR #524 P1): if a chi
                    # bump or stall recovery has shifted the legitimate
                    # gradient scale upward, the stale median would keep
                    # tripping the guard on every subsequent step, locking the
                    # optimizer into perpetual rollback.  Re-seed from the
                    # rolled-back state instead.
                    recent_gnorms_2s.clear()
                    if is_metric_lbfgs:
                        lbfgs_history.clear()
                        prev_params_flat = None
                        prev_grad_flat = None
                    if is_cg:
                        cg_direction = None
                        prev_grad = None
                        prev_precond_grad = None
                    if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                        opt_state = optimizer.init(params)
                    _logger.warning(
                        "[iPEPS-AD:2site-tensor] step %d/%d gradient spike "
                        "|g|=%.3e > %.1fx median %.3e — rolling back to best",
                        step + 1,
                        config.gs_num_steps,
                        grad_norm_val,
                        config.gs_grad_spike_ratio,
                        spike_floor,
                    )
                    if config.gs_verbose:
                        print(
                            f"[iPEPS-AD:2site-tensor] step {step + 1}/{config.gs_num_steps} "
                            f"|g|={grad_norm_val:.3e} spike "
                            f"(>{config.gs_grad_spike_ratio:.1f}x median {spike_floor:.3e}) "
                            f"— rollback to best, clear L-BFGS history",
                            flush=True,
                        )
                    continue
            recent_gnorms_2s.append(grad_norm_val)
            if len(recent_gnorms_2s) > config.gs_grad_spike_window:
                recent_gnorms_2s.pop(0)

            if _should_accept_best(
                current_best=best_energy,
                candidate=energy_float,
                floor=config.gs_energy_floor,
            ):
                best_energy = energy_float
                best_params = params
                best_env_cache_2s = dict(
                    _env_cache_2s
                )  # snapshot for warm-start (#317)

            delta_energy = abs(energy_float - prev_energy)
            logged = False
            if config.gs_verbose and _should_log_step(
                step, config.gs_num_steps, log_interval
            ):
                _log_ad_step(
                    "2site-tensor",
                    step,
                    config.gs_num_steps,
                    energy_float,
                    delta_energy,
                    best_energy,
                    grad_norm=grad_norm_val,
                )
                # Implicit-AD adjoint diagnostics: surface ||lam||/||init_lam||
                # amplification so chi-ceiling collapse runs (e.g. v9d/v9h)
                # have a smoking-gun trace.  Populated by the fused fixed-point
                # backward; empty dict on the explicit-AD path.
                if config.gs_implicit_ad:
                    from tenax.algorithms._ctm_energy_ad import (
                        get_last_implicit_ad_diagnostics,
                    )

                    _diag = get_last_implicit_ad_diagnostics()
                    if _diag:
                        print(
                            f"[iPEPS-AD:2site-tensor] step {step + 1}/{config.gs_num_steps} "
                            f"GMRES n_iter={_diag.get('n_iter', '?')} "
                            f"converged={_diag.get('converged', '?')} "
                            f"diverged={_diag.get('diverged', '?')} "
                            f"||lam||={_diag.get('lam_norm', float('nan')):.3e} "
                            f"amp={_diag.get('amplification', float('nan')):.3e}",
                            flush=True,
                        )
                logged = True

            prev_energy = energy_float
            if _converged_outer(config, delta_energy, grad_norm_val):
                # #455 PR2: at non-final χ stages, treat convergence as a
                # signal to advance to the next stage rather than exit.
                # Mirrors the 1-site convergence-block intercept, including
                # the reactive ε_T bump (#472 codex review on PR #473):
                # converged-at-too-small-χ runs must apply the requested
                # auto-bump before exiting so the final env reflects the
                # bumped χ, and the bump can also unstick a stalled-at-χ_old
                # landscape via the ``continue`` below.
                chi_before_bump = ctm_cfg_2s.chi
                last_eps_t = float(_env_cache_2s.get("max_truncation_error", 0.0))
                ctm_cfg_2s, _env_cache_2s = _maybe_bump_chi(
                    ctm_cfg_2s,
                    _env_cache_2s,
                    last_eps_t,
                    base_charges=_bump_base_charges_2s,
                )
                if config.gs_chi_schedule_steps is not None:
                    steps_in_stage = (step + 1) - stage_start_step
                    _gn_for_bump = (
                        grad_norm_val
                        if grad_norm_val is not None
                        else (
                            _grad_l2_norm(grads)
                            if config.gs_conv_criterion != "dE"
                            else 0.0
                        )
                    )
                    (
                        ctm_cfg_2s,
                        _env_cache_2s,
                        new_stage_idx,
                        bump_fired,
                        _,
                    ) = _advance_chi_stage_if_due(
                        ctm_cfg_2s,
                        _env_cache_2s,
                        chi_schedule=config.gs_chi_schedule_steps,
                        current_stage_idx=current_stage_idx,
                        steps_in_stage=steps_in_stage,
                        config=config,
                        grad_norm=_gn_for_bump,
                        delta_energy=delta_energy,
                        stall_count=stall_count,
                        base_charges=_bump_base_charges_2s,
                    )
                    # Codex review (PR #467): decouple the schedule index
                    # advance from bump_fired so idempotent advances
                    # (bump_fired=False AND new_stage_idx>current_stage_idx)
                    # still continue rather than fall through to break.
                    stage_advanced = new_stage_idx != current_stage_idx
                    if stage_advanced:
                        current_stage_idx = new_stage_idx
                        stage_start_step = step + 1
                else:
                    bump_fired = False
                    stage_advanced = False
                if ctm_cfg_2s.chi != chi_before_bump:
                    # Reactive (#472) or scheduled (#455) bump fired —
                    # fresh landscape, fresh stall budget.  Gated on the
                    # χ delta so a reactive-only bump also clears state.
                    stall_count = 0
                    if is_metric_lbfgs:
                        lbfgs_history.clear()
                        prev_params_flat = None
                        prev_grad_flat = None
                    if is_cg:
                        cg_direction = None
                        prev_grad = None
                        prev_precond_grad = None
                    if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                        opt_state = optimizer.init(params)
                    if config.gs_verbose:
                        print(
                            f"[iPEPS-AD step {step + 1}] converged at "
                            f"chi={chi_before_bump} → bumping to "
                            f"chi={ctm_cfg_2s.chi} (#455 PR2 / #472)",
                            flush=True,
                        )
                if bump_fired or stage_advanced:
                    # Force a checkpoint on stage advance before continuing
                    # so a crash inside the next stage's first step doesn't
                    # roll back across the boundary (Codex P2 #497).
                    _maybe_save_2s_checkpoint(
                        step,
                        _chi_at_step_start,
                        _best_energy_at_step_start,
                        force_last=True,
                    )
                    continue
                if config.gs_verbose:
                    if not logged:
                        _log_ad_step(
                            "2site-tensor",
                            step,
                            config.gs_num_steps,
                            energy_float,
                            delta_energy,
                            best_energy,
                            grad_norm=grad_norm_val,
                        )
                    _log_ad_converged(
                        "2site-tensor",
                        step,
                        delta_energy,
                        config.gs_conv_tol,
                        grad_norm=grad_norm_val,
                        grad_norm_tol=config.gs_grad_norm_tol,
                        criterion=config.gs_conv_criterion,
                    )
                _converged = True
                break

            # Compute search direction
            if is_cg:
                if config.gs_metric_precond and not use_c4v:
                    from tenax.algorithms._metric_precond import (
                        precondition_gradient_multisite,
                    )

                    envs_cached = _env_cache_2s["envs"]
                    A_g, B_g = grads
                    envs_m = {(0, 0): envs_cached[(0, 0)], (1, 0): envs_cached[(1, 0)]}
                    sites_m = {(0, 0): params[0], (1, 0): params[1]}
                    grads_m = {(0, 0): A_g, (1, 0): B_g}
                    delta_metric = delta_energy if step > 0 else _tree_dot(grads, grads)
                    z_dict = precondition_gradient_multisite(
                        sites_m, envs_m, grads_m, delta_metric, config
                    )
                    z = (
                        _wrap_tensor(z_dict[(0, 0)], A_g),
                        _wrap_tensor(z_dict[(1, 0)], B_g),
                    )
                    neg_z = jax.tree.map(lambda g: -g, z)
                    if prev_precond_grad is not None and cg_direction is not None:
                        z_diff = jax.tree.map(lambda a, b: a - b, z, prev_precond_grad)
                        num = _tree_dot(grads, z_diff)
                        den = _tree_dot(prev_grad, prev_precond_grad)
                        beta = max(0.0, num / den) if den > 1e-30 else 0.0
                        cg_direction = _tree_add(neg_z, _tree_scale(cg_direction, beta))
                    else:
                        cg_direction = neg_z
                    prev_precond_grad = z
                else:
                    neg_grad = jax.tree.map(lambda g: -g, grads)
                    if prev_grad is not None and cg_direction is not None:
                        beta = _cg_beta_pr(grads, prev_grad)
                        cg_direction = _tree_add(
                            neg_grad, _tree_scale(cg_direction, beta)
                        )
                    else:
                        cg_direction = neg_grad
                prev_grad = grads
                direction = cg_direction
            elif is_metric_lbfgs:
                from tenax.algorithms._metric_precond import lbfgs_two_loop

                if use_c4v:
                    p_flat = params
                    g_flat = grads
                else:
                    from tenax.algorithms._metric_precond import (
                        precondition_gradient_multisite,
                    )

                    A_cur, B_cur = params
                    A_g, B_g = grads
                    p_flat = jnp.concatenate(
                        [
                            A_cur.todense().reshape(-1),
                            B_cur.todense().reshape(-1),
                        ]
                    )
                    g_flat = jnp.concatenate(
                        [
                            A_g.todense().reshape(-1),
                            B_g.todense().reshape(-1),
                        ]
                    )

                if prev_params_flat is not None:
                    s = p_flat - prev_params_flat
                    y = g_flat - prev_grad_flat
                    sy = float(jnp.real(jnp.vdot(s, y)))
                    if sy > 1e-10:
                        rho = 1.0 / sy
                        lbfgs_history.append((s, y, rho))
                        if len(lbfgs_history) > 10:
                            lbfgs_history.pop(0)
                prev_params_flat = p_flat
                prev_grad_flat = g_flat

                if use_c4v:
                    direction_flat = lbfgs_two_loop(g_flat, lbfgs_history, lambda v: v)
                    direction = -direction_flat
                else:
                    envs_cached = _env_cache_2s["envs"]
                    envs_m = {(0, 0): envs_cached[(0, 0)], (1, 0): envs_cached[(1, 0)]}
                    sites_m = {(0, 0): A_cur, (1, 0): B_cur}
                    delta_metric = (
                        delta_energy
                        if step > 0
                        else float(jnp.real(jnp.vdot(g_flat, g_flat)))
                    )
                    n_A = A_cur.todense().size

                    def h0_matvec(v):
                        v_A = v[:n_A]
                        v_B = v[n_A:]
                        D_b = A_cur.todense().shape[0]
                        d_l = A_cur.todense().shape[-1]
                        grads_v = {
                            (0, 0): _wrap_tensor(
                                v_A.reshape(D_b, D_b, D_b, D_b, d_l), A_cur
                            ),
                            (1, 0): _wrap_tensor(
                                v_B.reshape(D_b, D_b, D_b, D_b, d_l), B_cur
                            ),
                        }
                        z_dict = precondition_gradient_multisite(
                            sites_m, envs_m, grads_v, delta_metric, config
                        )
                        return jnp.concatenate(
                            [
                                z_dict[(0, 0)].reshape(-1),
                                z_dict[(1, 0)].reshape(-1),
                            ]
                        )

                    direction_flat = lbfgs_two_loop(g_flat, lbfgs_history, h0_matvec)
                    D_b = A_cur.todense().shape[0]
                    d_l = A_cur.todense().shape[-1]
                    dir_A = -direction_flat[:n_A].reshape(D_b, D_b, D_b, D_b, d_l)
                    dir_B = -direction_flat[n_A:].reshape(D_b, D_b, D_b, D_b, d_l)
                    direction = (
                        _wrap_tensor(dir_A, A_cur),
                        _wrap_tensor(dir_B, B_cur),
                    )
            elif optimizer is not None:
                updates, opt_state = optimizer.update(grads, opt_state, params)
                direction = updates
            else:
                direction = jax.tree.map(lambda g: -g, grads)

            # Issue #328: kill the radial component of the search direction
            # so the line-search chord stays on-manifold to first order.
            # ``_normalize_params`` projects the final iterate back to unit
            # norm, but the intermediate chord ``params + alpha * direction``
            # drifts off the sphere proportional to ``alpha * <params, dir>``
            # — for a Euclidean L-BFGS direction this can be large and push
            # the line search into non-variational CTM regions before
            # retraction.  Stale curvature pairs accumulated on that chord
            # then corrupt subsequent L-BFGS steps.
            if not use_c4v:
                direction = _tangent_project_unit(direction, params)

            if use_ls:
                # Snapshot for env-cache restore after the line search; see
                # ``_restore_env_cache_after_line_search`` (#502 Codex P1).
                _ls_env_snap = (
                    "envs" in _env_cache_2s,
                    _env_cache_2s.get("envs"),
                )
                if line_search_method == "hager_zhang":
                    from tenax.algorithms._line_search import hager_zhang_line_search

                    slope = _tree_dot(grads, direction)
                    if slope >= 0:
                        direction = jax.tree.map(lambda g: -g, grads)
                        if not use_c4v:
                            direction = _tangent_project_unit(direction, params)
                        slope = _tree_dot(grads, direction)
                        if is_metric_lbfgs:
                            lbfgs_history.clear()

                    hz_counter = {"phi": 0, "dphi": 0}

                    def _phi(alpha):
                        hz_counter["phi"] += 1
                        trial = _normalize_params(
                            _tree_add(params, _tree_scale(direction, alpha))
                        )
                        return loss_fn_fwd(trial)

                    def _dphi(alpha):
                        hz_counter["dphi"] += 1
                        trial = _normalize_params(
                            _tree_add(params, _tree_scale(direction, alpha))
                        )
                        _, g = jax.value_and_grad(loss_fn)(trial)
                        return _tree_dot(g, direction)

                    dir_norm = math.sqrt(max(_tree_dot(direction, direction), 1e-30))
                    param_norm = math.sqrt(max(_tree_dot(params, params), 1e-30))
                    alpha0 = min(1.0, 0.1 * param_norm / dir_norm)

                    alpha, f_alpha, converged = hager_zhang_line_search(
                        _phi,
                        _dphi,
                        energy_float,
                        slope,
                        alpha_init=alpha0,
                        rho=1.5,
                        max_step=2.0 * alpha0,
                        energy_bound=max(2.0, 2.0 * abs(best_energy)),
                        max_iter=config.gs_hz_max_iter,
                        bracket_only_phi=True,  # #504: skip dphi in bracket
                    )
                    if config.gs_verbose:
                        print(
                            f"[iPEPS-AD:2site-tensor] HZ probes phi={hz_counter['phi']} "
                            f"dphi={hz_counter['dphi']} alpha={alpha:.3e} "
                            f"converged={converged}",
                            flush=True,
                        )
                    if _is_real_decrease(f_alpha, energy_float):
                        params = _normalize_params(
                            _tree_add(params, _tree_scale(direction, alpha))
                        )
                        stall_count = 0
                    else:
                        stall_count += 1
                else:
                    slope_bt = _tree_dot(grads, direction)
                    if slope_bt >= 0:
                        direction = jax.tree.map(lambda g: -g, grads)
                        if not use_c4v:
                            direction = _tangent_project_unit(direction, params)
                        if is_metric_lbfgs:
                            lbfgs_history.clear()
                    params, new_energy, step_size = _backtracking_line_search(
                        params,
                        direction,
                        grads,
                        energy_float,
                        loss_fn_fwd,
                        max_steps=config.gs_line_search_max_steps,
                    )
                    if _is_real_decrease(new_energy, energy_float):
                        stall_count = 0
                    else:
                        stall_count += 1

                _restore_env_cache_after_line_search(_env_cache_2s, _ls_env_snap)

                # Noise recovery on persistent stall (legacy; see issue #298).
                # Only used by the non-C4v path; C4v defaults to "reset".
                if (
                    config.gs_stall_recovery == "noise"
                    and stall_count > 0
                    and stall_count <= config.gs_noise_recovery_retries
                ):
                    noise_key = jax.random.PRNGKey(step * 1000 + stall_count)
                    noisy_params = []
                    for i, p in enumerate(params):
                        k = jax.random.fold_in(noise_key, i)
                        data = p.todense()
                        noise = config.gs_noise_amplitude * _random_noise(
                            k, data.shape, data.dtype
                        )
                        noisy = data + noise * jnp.linalg.norm(data)
                        noisy = noisy / (jnp.linalg.norm(noisy) + 1e-10)
                        noisy_params.append(_wrap_tensor(noisy, p))
                    params = tuple(noisy_params)
                    if config.gs_verbose:
                        print(
                            f"[iPEPS-AD] stall #{stall_count}, adding noise", flush=True
                        )
                    # Reset optimizer state
                    if is_cg:
                        cg_direction = None
                        prev_grad = None
                        prev_precond_grad = None
                    if is_metric_lbfgs:
                        lbfgs_history.clear()
                        prev_params_flat = None
                        prev_grad_flat = None
                elif config.gs_stall_recovery == "reset" and stall_count > 0:
                    # Rollback to best on reset (#454). The CTM-error reset path
                    # above (around L1998-2002) already does this for the
                    # CTMRGGradientError branch; we extend the same pattern to the
                    # Wolfe-failure path that was missed. #298's anti-rollback
                    # evidence was on a pre-trifecta CTM stack (pre-PR #406 2x2
                    # projector, pre-multisite-CTM rewrite, pre-PR #447 AD
                    # stop_gradient) and no longer applies.
                    if stall_count > config.gs_stall_recovery_retries:
                        # #455 PR2: at non-final χ stages, treat stall-cap
                        # exhaustion as a signal to advance the χ schedule
                        # rather than exit with best_energy.  Mirrors the
                        # 1-site stall-cap intercept (commit 07ffe8e).
                        if config.gs_chi_schedule_steps is not None:
                            steps_in_stage = (step + 1) - stage_start_step
                            _gn_for_bump = (
                                grad_norm_val
                                if grad_norm_val is not None
                                else (
                                    _grad_l2_norm(grads)
                                    if config.gs_conv_criterion != "dE"
                                    else 0.0
                                )
                            )
                            (
                                ctm_cfg_2s,
                                _env_cache_2s,
                                new_stage_idx,
                                bump_fired,
                                _,
                            ) = _advance_chi_stage_if_due(
                                ctm_cfg_2s,
                                _env_cache_2s,
                                chi_schedule=config.gs_chi_schedule_steps,
                                current_stage_idx=current_stage_idx,
                                steps_in_stage=steps_in_stage,
                                config=config,
                                grad_norm=_gn_for_bump,
                                delta_energy=delta_energy,
                                stall_count=stall_count,
                                base_charges=_bump_base_charges_2s,
                            )
                            # Codex review (PR #467): see 1-site stall-cap
                            # intercept for rationale.  Decouple schedule
                            # advance from bump_fired; keep reset block
                            # gated on bump_fired.
                            stage_advanced = new_stage_idx != current_stage_idx
                            if stage_advanced:
                                current_stage_idx = new_stage_idx
                                stage_start_step = step + 1
                            if bump_fired:
                                # Rollback params to best from the previous
                                # stage; _env_cache_2s stays at the freshly
                                # padded post-bump state (the budget-path
                                # invariant — see _apply_chi_bump).
                                params = best_params
                                stall_count = 0
                                if is_metric_lbfgs:
                                    lbfgs_history.clear()
                                    prev_params_flat = None
                                    prev_grad_flat = None
                                if is_cg:
                                    cg_direction = None
                                    prev_grad = None
                                    prev_precond_grad = None
                                if (
                                    optimizer is not None
                                    and config.gs_optimizer.lower() == "lbfgs"
                                ):
                                    opt_state = optimizer.init(params)
                                if config.gs_verbose:
                                    print(
                                        f"[iPEPS-AD step {step + 1}] stall-cap at "
                                        f"chi={ctm_cfg_2s.chi} → advancing to next "
                                        f"stage (#455 PR2)",
                                        flush=True,
                                    )
                            if bump_fired or stage_advanced:
                                # Force a checkpoint on stall-cap stage
                                # advance before continuing (Codex P2 #497).
                                _maybe_save_2s_checkpoint(
                                    step,
                                    _chi_at_step_start,
                                    _best_energy_at_step_start,
                                    force_last=True,
                                )
                                continue
                        n_resets_done = stall_count - 1
                        if config.gs_verbose:
                            print(
                                f"[iPEPS-AD] stall budget exhausted after "
                                f"{n_resets_done} resets, "
                                f"returning best E={best_energy:.10f}",
                                flush=True,
                            )
                        break
                    params = best_params
                    # #518: ``best_env_cache_2s`` may be at a stale χ if a
                    # reactive/scheduled bump fired after it was last
                    # snapshotted.  Clear instead of restoring; the next
                    # CTM call cold-starts at the current ctm_cfg_2s.chi.
                    _drop_env_cache_for_reset(_env_cache_2s)
                    if is_cg:
                        cg_direction = None
                        prev_grad = None
                        prev_precond_grad = None
                    if is_metric_lbfgs:
                        lbfgs_history.clear()
                        prev_params_flat = None
                        prev_grad_flat = None
                    # Optax-backed L-BFGS stores curvature history in opt_state,
                    # not in lbfgs_history.  Reinitialize it on the rolled-back
                    # params so the next step really is steepest descent.
                    if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                        opt_state = optimizer.init(params)
                    if config.gs_verbose:
                        print(
                            f"[iPEPS-AD] stall #{stall_count}, reset L-BFGS history "
                            f"(rollback to best, retry "
                            f"{stall_count}/{config.gs_stall_recovery_retries})",
                            flush=True,
                        )
            else:
                params = optax.apply_updates(params, direction)
                params = _normalize_params(params)

            # Reactive ε_T auto-bump (variPEPS §2.8.2, #472) followed by the
            # scheduled outer-loop χ bump (#453 / #455).  Compose order
            # mirrors the 1-site path: reactive first so it can pre-empt the
            # schedule, scheduled second to advance the stage index even on
            # idempotent (post-reactive) bumps.  ``chi_before`` is snapshotted
            # before either fires so the post-bump reset block below catches
            # either trigger via ``ctm_cfg_2s.chi != chi_before``.
            chi_before = ctm_cfg_2s.chi
            last_eps_t = float(_env_cache_2s.get("max_truncation_error", 0.0))
            last_smallest_S = float(_env_cache_2s.get("max_smallest_S", 0.0))
            ctm_cfg_2s, _env_cache_2s = _maybe_bump_chi(
                ctm_cfg_2s,
                _env_cache_2s,
                last_eps_t,
                last_smallest_S=last_smallest_S,
                base_charges=_bump_base_charges_2s,
            )
            # Scheduled outer-loop χ bump (#453 / #455).  No-ops when
            # ``gs_chi_schedule_steps`` is None.  Fires at the step boundary
            # so the next iteration's value_and_grad sees the bumped χ — same
            # invariant as the 1-site path.  Per-stage state
            # (current_stage_idx, stage_start_step) drives the new helper;
            # #455 PR2 will add convergence/stall-cap triggers.
            if config.gs_chi_schedule_steps is not None:
                steps_in_stage = (step + 1) - stage_start_step
                _gn_for_bump = (
                    grad_norm_val
                    if grad_norm_val is not None
                    else (
                        _grad_l2_norm(grads)
                        if config.gs_conv_criterion != "dE"
                        else 0.0
                    )
                )
                ctm_cfg_2s, _env_cache_2s, new_stage_idx, bump_fired, _should_break = (
                    _advance_chi_stage_if_due(
                        ctm_cfg_2s,
                        _env_cache_2s,
                        chi_schedule=config.gs_chi_schedule_steps,
                        current_stage_idx=current_stage_idx,
                        steps_in_stage=steps_in_stage,
                        config=config,
                        grad_norm=_gn_for_bump,
                        delta_energy=delta_energy,
                        stall_count=stall_count,
                        base_charges=_bump_base_charges_2s,
                    )
                )
                stage_advanced = new_stage_idx != current_stage_idx
                if stage_advanced:
                    if config.gs_verbose:
                        print(
                            f"[iPEPS-AD step {step + 1}] schedule advance: "
                            f"stage {current_stage_idx} → {new_stage_idx}, "
                            f"chi {chi_before} → {ctm_cfg_2s.chi} (#455)",
                            flush=True,
                        )
                    current_stage_idx = new_stage_idx
                    stage_start_step = step + 1
            # Reset block must live OUTSIDE the schedule-only ``if`` so a
            # reactive-only bump (``chi_auto_bump=True`` with no schedule)
            # still clears stall_count, CG state, and L-BFGS history at the
            # new χ.  Gated solely on ``ctm_cfg_2s.chi != chi_before``, which
            # is set by EITHER reactive (#472) or scheduled (#455) bumps.
            # (Codex PR #473 review.)
            if ctm_cfg_2s.chi != chi_before:
                # χ bump fired: a new landscape begins.  Reset the stall
                # counter so the next stage gets a fresh retry budget;
                # also clear L-BFGS curvature history so the first step
                # at the new χ is plain steepest descent (curvature from
                # the previous χ landscape isn't valid here).
                stall_count = 0
                chi_ceiling_consecutive_2s = 0
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_params_flat = None
                    prev_grad_flat = None
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                    opt_state = optimizer.init(params)

            # variPEPS §2.8.2 chi-ceiling bail-out.  After the post-step bump
            # has been attempted, if χ is still at chi_max AND the indicator
            # remains above threshold for K consecutive steps, exit early and
            # return ``best_params``.  Counter resets on any bump (handled
            # above), any successful headroom recovery, or stall recovery.
            if (
                config.gs_chi_ceiling_bailout > 0
                and ctm_cfg_2s.chi_max is not None
                and ctm_cfg_2s.chi >= ctm_cfg_2s.chi_max
            ):
                # ``getattr`` fallback (codex PR #524 follow-up): the
                # ``chi_auto_bump_metric`` field is added by PR #525.  Until
                # that lands this PR must still resolve to a working
                # indicator instead of raising ``AttributeError`` when the
                # bail-out fires.  Default ``"eps_T"`` matches CTMConfig's
                # eventual default and the existing 2-site bump cadence.
                _bail_metric = getattr(ctm_cfg_2s, "chi_auto_bump_metric", "eps_T")
                if _bail_metric == "norm_smallest_S":
                    _bail_indicator = float(_env_cache_2s.get("max_smallest_S", 0.0))
                else:
                    _bail_indicator = float(
                        _env_cache_2s.get("max_truncation_error", 0.0)
                    )
                if _bail_indicator > ctm_cfg_2s.chi_auto_bump_eps:
                    chi_ceiling_consecutive_2s += 1
                    if chi_ceiling_consecutive_2s >= config.gs_chi_ceiling_bailout:
                        if config.gs_verbose:
                            print(
                                f"[iPEPS-AD:2site-tensor] step {step + 1}/"
                                f"{config.gs_num_steps} chi-ceiling bail-out: "
                                f"chi={ctm_cfg_2s.chi} at chi_max, "
                                f"{_bail_metric}={_bail_indicator:.3e} > "
                                f"{ctm_cfg_2s.chi_auto_bump_eps:.3e} for "
                                f"{chi_ceiling_consecutive_2s} consecutive steps "
                                f"— rolling back to best E={best_energy:.10f}",
                                flush=True,
                            )
                        params = best_params
                        _maybe_save_2s_checkpoint(
                            step, chi_before, _best_energy_at_step_start
                        )
                        break
                else:
                    chi_ceiling_consecutive_2s = 0
            else:
                chi_ceiling_consecutive_2s = 0

            # End-of-step save: cadence-based + new-best detection.
            _maybe_save_2s_checkpoint(step, chi_before, _best_energy_at_step_start)

        # Re-evaluate both final params and best_params with fully converged
        # fresh CTM.  In-loop energies use warm-started CTM that can produce
        # unphysical values, so we compare fresh evaluations only.
        # Match in-loop CTM tolerances (#317) by reusing ctm_cfg_2s directly.

        def _eval_fresh_2site(p, env_init=None):
            """Evaluate energy with fully converged fresh CTM."""
            if use_c4v:
                A_t, B_t = _c4v_AB(p)
            else:
                A_t, B_t = _normalize_params(p)
            st = {(0, 0): A_t, (1, 0): B_t}
            envs, _ = python_loop_ctm_converge(
                st,
                CHECKERBOARD_NEIGHBORS,
                **ctm_converge_kwargs(ctm_cfg_2s, env_init=env_init),
            )
            E_ = float(
                compute_energy_ctm_tensor_2site(
                    A_t, B_t, envs[(0, 0)], envs[(1, 0)], gate, d_phys
                )
            )
            return A_t, B_t, envs, E_

        A_last, B_last, envs_last, E_last = _eval_fresh_2site(
            params, _env_cache_2s.get("envs", None)
        )
        env_A_last, env_B_last = envs_last[(0, 0)], envs_last[(1, 0)]

        # Pad best_env_cache_2s envs to the current ctm_cfg_2s.chi before use.
        # Mirrors the same fix on the 1-site path — see comment there.  Fixes #469.
        # NOTE: if a new _eval_fresh_2site(best_params, best_env_cache_2s["envs"])
        # call site is added later, it must reapply this padding — there's no
        # producer-side guarantee that the snapshot is at ctm_cfg_2s.chi.
        _best_env_init_2s = best_env_cache_2s.get("envs", None)
        if _best_env_init_2s:
            # Issue #514: see 1-site finalize.  Pad to the larger of the
            # static ``ctm_cfg_2s.chi`` and the env_cache's actual chi
            # so a bumped env is never asked to shrink.
            # See 1-site finalize: consult both live cache and the
            # best-env snapshot (Codex P2 on PR #542).
            _target_chi_2s = max(
                ctm_cfg_2s.chi,
                _env_dict_chi(_env_cache_2s.get("envs")) or 0,
                _env_dict_chi(_best_env_init_2s) or 0,
            )
            _best_env_init_2s = {
                c: pad_dense_env_chi(
                    _best_env_init_2s[c],
                    _target_chi_2s,
                    base_charges=_bump_base_charges_2s,
                )
                for c in _best_env_init_2s
            }

        if best_params is not params:
            A_best, B_best, envs_best, E_best_fresh = _eval_fresh_2site(
                best_params, _best_env_init_2s
            )
            env_A_best = envs_best[(0, 0)]
            env_B_best = envs_best[(1, 0)]
        else:
            E_best_fresh = E_last

        # Pick whichever fresh evaluation is lower
        if E_last <= E_best_fresh:
            A_final, B_final = A_last, B_last
            env_A, env_B, E_gs = env_A_last, env_B_last, E_last
        else:
            A_final, B_final = A_best, B_best
            env_A, env_B, E_gs = env_A_best, env_B_best, E_best_fresh
        if config.gs_verbose:
            print(f"[iPEPS-AD:2site-tensor] final E={E_gs:.10f}", flush=True)

        if config.return_history:
            history = {
                "energies": _history_energies,
                "step_times": _history_step_times,
                "jit_compile_time": _jit_compile_time,
                "num_steps": len(_history_energies),
                "converged": _converged,
            }
            return (A_final, B_final), (env_A, env_B), E_gs, history
        return (A_final, B_final), (env_A, env_B), E_gs
    finally:
        if _prev_norm_diag is not None:
            set_implicit_ad_norm_diagnostics(_prev_norm_diag)


def _optimize_gs_ad_multisite(
    hamiltonian_gate: jax.Array | Tensor,
    A_init: dict[str, Tensor] | None,
    config: iPEPSConfig,
):
    """AD-based ground state optimization for multi-site iPEPS (Lattice unit cell).

    Uses implicit differentiation through the multisite CTM fixed point.
    Generalizes the 2-site Tensor-protocol optimizer to N sites on an
    arbitrary ``Lattice``.

    Returns ``(site_tensors_dict, envs_dict, E_gs)`` where the dicts are
    keyed by site name (e.g. ``"u"``, ``"v"``, ``"w"``).
    """
    config = _normalize_stall_recovery(config, unit_cell="multisite")
    _warn_implicit_ad_variational_caveat(config, path="Multisite Lattice")

    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor_energy import (
        compute_energy_ctm_tensor_multisite,
    )
    from tenax.algorithms.ad_utils import (
        CTMRGGradientError,
        _wrap_tensor,
    )
    from tenax.algorithms.ipeps_ad_policy import (
        ctm_converge_kwargs,
        make_ctm_energy_fn,
    )

    # ── Lattice → coordinate dicts ──────────────────────────────────────
    lattice: Lattice = config.unit_cell  # type: ignore[assignment]
    neighbors, name_to_coord, coord_to_name = _lattice_to_neighbors(lattice)
    coords = sorted(neighbors.keys())  # deterministic ordering

    # ── Gate ─────────────────────────────────────────────────────────────
    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d_phys = gate.shape[0]

    # ── Initial site tensors ─────────────────────────────────────────────
    if A_init is not None:
        # Convert name-keyed dict → coord-keyed, normalize
        site_tensors_init: dict[Coord, Tensor] = {}
        for name, tensor in A_init.items():
            c = name_to_coord[name]
            site_tensors_init[c] = tensor * (1.0 / (tensor.norm() + 1e-10))
        # Build ordered tuple for params
        params: tuple[Tensor, ...] = tuple(site_tensors_init[c] for c in coords)
    else:
        # Random complex128 initialization
        D = config.max_bond_dim
        tensors = []
        for i, c in enumerate(coords):
            key = jax.random.PRNGKey(42 + i)
            k1, k2 = jax.random.split(key)
            data = jax.random.normal(k1, (D, D, D, D, d_phys), dtype=jnp.float64) + (
                1j * jax.random.normal(k2, (D, D, D, D, d_phys), dtype=jnp.float64)
            )
            data = data / (jnp.linalg.norm(data) + 1e-10)
            tensors.append(_wrap_as_dense_tensor(data))
        params = tuple(tensors)

    # ── CTM config & AD mode ─────────────────────────────────────────────
    ctm_cfg = build_ad_ctm_config(config)
    use_explicit = not config.gs_implicit_ad
    explicit_steps = config.gs_explicit_ad_steps
    explicit_warmup = config.gs_explicit_ad_warmup
    explicit_backward_steps = config.gs_explicit_ad_backward_steps

    # Env warm-start cache
    _env_cache: dict[str, dict] = {}

    # ── Energy function for multisite ────────────────────────────────────
    def _energy_fn(site_tensors, envs, gate_):
        return compute_energy_ctm_tensor_multisite(
            site_tensors,
            envs,
            neighbors,
            gate_,
        )

    _ctm_energy_fn = make_ctm_energy_fn(
        neighbors=neighbors,
        gate=gate,
        get_ctm_cfg=lambda: ctm_cfg,
        env_cache=_env_cache,
        use_explicit=use_explicit,
        explicit_warmup=explicit_warmup,
        explicit_steps=explicit_steps,
        explicit_backward_steps=explicit_backward_steps,
        energy_fn=_energy_fn,
        recipe=config.gs_recipe,
    )

    def loss_fn(params_):
        site_tensors = {}
        for i, c in enumerate(coords):
            p = params_[i]
            site_tensors[c] = p * (1.0 / (p.norm() + 1e-10))
        return _ctm_energy_fn(site_tensors)

    def _update_env_cache(params_):
        site_tensors = {}
        for i, c in enumerate(coords):
            p = params_[i]
            site_tensors[c] = p * (1.0 / (p.norm() + 1e-10))
        envs, info = python_loop_ctm_converge(
            site_tensors,
            neighbors,
            **ctm_converge_kwargs(ctm_cfg, env_init=_env_cache.get("envs", None)),
        )
        _env_cache["envs"] = envs
        # See ``_update_env_cache_2s`` (#472, #474): the 2x2 plaquette
        # projector tracks ε_T directly so reactive bumps fire on the
        # multisite forward stack with the default recipe.
        _env_cache["max_truncation_error"] = float(info.max_truncation_error)

    def loss_fn_fwd(params_):
        """Forward-only loss for line search — warm-starts CTM from env cache."""
        site_tensors = {}
        for i, c in enumerate(coords):
            p = params_[i]
            site_tensors[c] = p * (1.0 / (p.norm() + 1e-10))
        envs, _ = python_loop_ctm_converge(
            site_tensors,
            neighbors,
            **ctm_converge_kwargs(
                ctm_cfg,
                env_init=_env_cache.get("envs", None),
                for_probe=True,
            ),
        )
        # Issue #502: see 1-site loss_fn_fwd above for rationale.
        _env_cache["envs"] = envs
        return float(
            compute_energy_ctm_tensor_multisite(
                site_tensors,
                envs,
                neighbors,
                gate,
            )
        )

    # ── Optimizer setup ──────────────────────────────────────────────────
    is_metric_lbfgs = (
        config.gs_metric_precond and config.gs_optimizer.lower() == "lbfgs"
    )
    optimizer = None if is_metric_lbfgs else _build_optimizer(config)
    opt_state = optimizer.init(params) if optimizer is not None else None
    use_ls = _use_line_search(config)
    # Resolve ``gs_line_search_method`` once per dispatch (#509 / codex P2 on
    # #549) — see comment in the 1-site path.
    line_search_method = _resolve_line_search_method(config, ctm_cfg)
    is_cg = config.gs_optimizer.lower() == "cg"

    best_energy = float("inf")
    best_params = params
    best_env_cache: dict[str, dict] = {}
    prev_energy = float("inf")
    prev_grad = None
    cg_direction = None
    prev_precond_grad = None
    log_interval = config.gs_log_interval
    lbfgs_history: list = []
    prev_params_flat: jnp.ndarray | None = None
    prev_grad_flat: jnp.ndarray | None = None
    stall_count = 0
    current_stage_idx = 0
    stage_start_step = 0

    # CTM conv_tol schedule
    _conv_tol_schedule = config.gs_ctm_conv_tol_schedule
    _current_conv_tol = ctm_cfg.conv_tol

    # CTM max_iter schedule (#507): same lookup contract as conv_tol.
    _max_iter_schedule = config.gs_ctm_max_iter_schedule
    _current_max_iter = ctm_cfg.max_iter

    # CTM plateau-patience schedule (independent of conv_tol — see
    # iPEPSConfig.gs_plateau_patience_schedule docstring).
    _patience_schedule = config.gs_plateau_patience_schedule
    _current_patience = ctm_cfg.plateau_patience

    # Optional trajectory capture (config.return_history).  Always allocated
    # so the post-loop return doesn't have to special-case the disabled path;
    # values are only populated under ``if config.return_history`` (#458).
    _history_energies: list[float] = []
    _history_step_times: list[float] = []
    _jit_compile_time: float = 0.0
    _first_step = True
    _converged = False

    def _get_scheduled_conv_tol(step_idx, num_steps):
        if _conv_tol_schedule is None:
            return _current_conv_tol
        frac = step_idx / max(num_steps, 1)
        tol = _conv_tol_schedule[0][1]
        for threshold, t in _conv_tol_schedule:
            if frac >= threshold:
                tol = t
        return tol

    def _get_scheduled_max_iter(step_idx, num_steps):
        """Look up max_iter from schedule based on step fraction (issue #507)."""
        if _max_iter_schedule is None:
            return _current_max_iter
        frac = step_idx / max(num_steps, 1)
        mi = _max_iter_schedule[0][1]
        for threshold, m in _max_iter_schedule:
            if frac >= threshold:
                mi = m
        return mi

    def _get_scheduled_plateau_patience(step_idx, num_steps):
        if _patience_schedule is None:
            return _current_patience
        frac = step_idx / max(num_steps, 1)
        patience = _patience_schedule[0][1]
        for threshold, p in _patience_schedule:
            if frac >= threshold:
                patience = p
        return patience

    # Base charges for SymmetricTensor env-padding (#453).  Same shape
    # infrastructure used by the 1-site auto-bump.  All sites in a uniform
    # iPEPS share the same bond charges; ``params[0]`` is representative.
    _bump_base_charges_multi: np.ndarray | None = None
    if ctm_cfg.chi_auto_bump or config.gs_chi_schedule_steps is not None:
        _A_init = params[0]
        if isinstance(_A_init, SymmetricTensor):
            from tenax.algorithms._ctm_tensor_convergence import _get_base_charges
            from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor

            _bump_base_charges_multi = _get_base_charges(
                _build_double_layer_tensor(_A_init)
            )

    # ── Optimization loop ────────────────────────────────────────────────
    for step in range(config.gs_num_steps):
        # Update conv_tol if schedule is active
        if _conv_tol_schedule is not None:
            new_tol = _get_scheduled_conv_tol(step, config.gs_num_steps)
            if new_tol != _current_conv_tol:
                _current_conv_tol = new_tol
                ctm_cfg = _replace(ctm_cfg, conv_tol=new_tol)
        # Update max_iter if schedule is active (#507)
        if _max_iter_schedule is not None:
            new_max_iter = _get_scheduled_max_iter(step, config.gs_num_steps)
            if new_max_iter != _current_max_iter:
                _current_max_iter = new_max_iter
                ctm_cfg = _replace(ctm_cfg, max_iter=new_max_iter)
        # Update plateau_patience if schedule is active
        if _patience_schedule is not None:
            new_patience = _get_scheduled_plateau_patience(step, config.gs_num_steps)
            if new_patience != _current_patience:
                _current_patience = new_patience
                ctm_cfg = _replace(ctm_cfg, plateau_patience=new_patience)

        if config.return_history:
            _step_t0 = _time.perf_counter()
        try:
            energy_val, grads = jax.value_and_grad(loss_fn)(params)
        except CTMRGGradientError as exc:
            _logger.warning(
                "[iPEPS-AD] Arnoldi precheck: rho(J^T) = %.4f >= 1 at step %d — "
                "skipping, triggering stall recovery",
                exc.spectral_radius,
                step,
            )
            if config.gs_verbose:
                print(
                    f"[iPEPS-AD:multisite] step {step + 1}/{config.gs_num_steps} "
                    f"rho(J^T)={exc.spectral_radius:.4f} — stall recovery",
                    flush=True,
                )
            stall_count += 1
            if config.gs_stall_recovery == "reset":
                # Cap on CTMRGGradientError-driven reset path (#454 follow-up,
                # codex review on PR #457).
                if stall_count > config.gs_stall_recovery_retries:
                    n_resets_done = stall_count - 1
                    if config.gs_verbose:
                        print(
                            f"[iPEPS-AD] CTM-error stall budget exhausted after "
                            f"{n_resets_done} resets, "
                            f"returning best E={best_energy:.10f}",
                            flush=True,
                        )
                    break
                params = best_params
                # #518: ``best_env_cache`` may be at a stale χ if a
                # reactive/scheduled bump fired after it was last
                # snapshotted.  Clear instead of restoring; the next
                # CTM call cold-starts at the current ctm_cfg.chi.
                _drop_env_cache_for_reset(_env_cache)
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_params_flat = None
                    prev_grad_flat = None
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                    opt_state = optimizer.init(params)
            elif (
                config.gs_stall_recovery == "noise"
                and stall_count <= config.gs_noise_recovery_retries
            ):
                noise_key = jax.random.PRNGKey(step * 1000 + stall_count)
                noisy_params = []
                for i, p in enumerate(params):
                    k = jax.random.fold_in(noise_key, i)
                    data = p.todense()
                    noise = config.gs_noise_amplitude * _random_noise(
                        k, data.shape, data.dtype
                    )
                    noisy = data + noise * jnp.linalg.norm(data)
                    noisy = noisy / (jnp.linalg.norm(noisy) + 1e-10)
                    noisy_params.append(_wrap_tensor(noisy, p))
                params = tuple(noisy_params)
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_params_flat = None
                    prev_grad_flat = None
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
            continue
        energy_float = float(energy_val)

        if config.return_history:
            _step_dt = _time.perf_counter() - _step_t0
            if _first_step:
                _jit_compile_time = float(_step_dt)
                _first_step = False
            else:
                _history_step_times.append(float(_step_dt))
            _history_energies.append(energy_float)

        # Update env cache for warm-starting next step
        _update_env_cache(params)

        if _should_accept_best(
            current_best=best_energy,
            candidate=energy_float,
            floor=config.gs_energy_floor,
        ):
            best_energy = energy_float
            best_params = params
            best_env_cache = dict(_env_cache)

        delta_energy = abs(energy_float - prev_energy)
        grad_norm_val = _grad_l2_norm(grads)
        logged = False
        if config.gs_verbose and _should_log_step(
            step, config.gs_num_steps, log_interval
        ):
            _log_ad_step(
                "multisite",
                step,
                config.gs_num_steps,
                energy_float,
                delta_energy,
                best_energy,
                grad_norm=grad_norm_val,
            )
            logged = True

        prev_energy = energy_float

        # Early-step / post-stall warmup gate.  For dE-based criteria the
        # gate protects against false ``dE ≈ 0`` exits on step 0
        # (``prev_energy = inf`` → ``delta_energy = inf`` actually fails
        # the dE check, but a stall reset can leave ``prev_energy ≈
        # current`` and produce a real false-zero) and right after a
        # stall recovery.  Grad-norm is variationally meaningful, so if
        # the user's initial multisite state already satisfies
        # ``||grad||_2 < tol`` we should respect that and exit on step 0
        # — gating it would silently force the optimizer through up to
        # ``gs_num_steps`` even when the user explicitly opted into a
        # loose ``gs_grad_norm_tol`` for early-stop (codex #449
        # follow-up).
        needs_warmup = config.gs_conv_criterion in ("dE", "both")
        warmup_ok = (not needs_warmup) or (step > 5 and stall_count == 0)
        if _converged_outer(config, delta_energy, grad_norm_val) and warmup_ok:
            # #455 PR2: at non-final χ stages, treat convergence as a
            # signal to advance to the next stage rather than exit.
            # Mirrors the 1-site / 2-site convergence-block intercepts,
            # including the reactive ε_T bump (#472 codex review on
            # PR #473): the reactive trigger must also fire here so a
            # converged-at-too-small-χ multisite run exits with the
            # bumped χ rather than the stale one.
            chi_before_bump = ctm_cfg.chi
            last_eps_t = float(_env_cache.get("max_truncation_error", 0.0))
            ctm_cfg, _env_cache = _maybe_bump_chi(
                ctm_cfg,
                _env_cache,
                last_eps_t,
                base_charges=_bump_base_charges_multi,
            )
            if config.gs_chi_schedule_steps is not None:
                steps_in_stage = (step + 1) - stage_start_step
                _gn_for_bump = (
                    grad_norm_val
                    if grad_norm_val is not None
                    else (
                        _grad_l2_norm(grads)
                        if config.gs_conv_criterion != "dE"
                        else 0.0
                    )
                )
                (
                    ctm_cfg,
                    _env_cache,
                    new_stage_idx,
                    bump_fired,
                    _,
                ) = _advance_chi_stage_if_due(
                    ctm_cfg,
                    _env_cache,
                    chi_schedule=config.gs_chi_schedule_steps,
                    current_stage_idx=current_stage_idx,
                    steps_in_stage=steps_in_stage,
                    config=config,
                    grad_norm=_gn_for_bump,
                    delta_energy=delta_energy,
                    stall_count=stall_count,
                    base_charges=_bump_base_charges_multi,
                )
                # Codex review (PR #467): see 1-site / 2-site
                # convergence intercept for rationale.  Decouple
                # schedule advance from bump_fired.
                stage_advanced = new_stage_idx != current_stage_idx
                if stage_advanced:
                    current_stage_idx = new_stage_idx
                    stage_start_step = step + 1
            else:
                bump_fired = False
                stage_advanced = False
            if ctm_cfg.chi != chi_before_bump:
                # Reactive (#472) or scheduled (#455) bump fired —
                # fresh landscape, fresh stall budget.  Gated on the
                # χ delta so a reactive-only bump also clears state.
                stall_count = 0
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_params_flat = None
                    prev_grad_flat = None
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                    opt_state = optimizer.init(params)
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD step {step + 1}] converged at "
                        f"chi={chi_before_bump} → bumping to "
                        f"chi={ctm_cfg.chi} (#455 PR2 / #472)",
                        flush=True,
                    )
            if bump_fired or stage_advanced:
                continue
            if config.gs_verbose:
                if not logged:
                    _log_ad_step(
                        "multisite",
                        step,
                        config.gs_num_steps,
                        energy_float,
                        delta_energy,
                        best_energy,
                        grad_norm=grad_norm_val,
                    )
                _log_ad_converged(
                    "multisite",
                    step,
                    delta_energy,
                    config.gs_conv_tol,
                    grad_norm=grad_norm_val,
                    grad_norm_tol=config.gs_grad_norm_tol,
                    criterion=config.gs_conv_criterion,
                )
            _converged = True
            break

        # ── Search direction ─────────────────────────────────────────────
        if is_cg:
            if config.gs_metric_precond:
                from tenax.algorithms._metric_precond import (
                    precondition_gradient_multisite,
                )

                envs_cached = _env_cache["envs"]
                sites_m = {c: params[i] for i, c in enumerate(coords)}
                grads_m = {c: grads[i] for i, c in enumerate(coords)}
                delta_metric = delta_energy if step > 0 else _tree_dot(grads, grads)
                z_dict = precondition_gradient_multisite(
                    sites_m, envs_cached, grads_m, delta_metric, config
                )
                z = tuple(
                    _wrap_tensor(z_dict[c], grads[i]) for i, c in enumerate(coords)
                )
                neg_z = jax.tree.map(lambda g: -g, z)
                if prev_precond_grad is not None and cg_direction is not None:
                    z_diff = jax.tree.map(lambda a, b: a - b, z, prev_precond_grad)
                    num = _tree_dot(grads, z_diff)
                    den = _tree_dot(prev_grad, prev_precond_grad)
                    beta = max(0.0, num / den) if den > 1e-30 else 0.0
                    cg_direction = _tree_add(neg_z, _tree_scale(cg_direction, beta))
                else:
                    cg_direction = neg_z
                prev_precond_grad = z
            else:
                neg_grad = jax.tree.map(lambda g: -g, grads)
                if prev_grad is not None and cg_direction is not None:
                    beta = _cg_beta_pr(grads, prev_grad)
                    cg_direction = _tree_add(neg_grad, _tree_scale(cg_direction, beta))
                else:
                    cg_direction = neg_grad
            prev_grad = grads
            direction = cg_direction
        elif is_metric_lbfgs:
            from tenax.algorithms._metric_precond import (
                lbfgs_two_loop,
                precondition_gradient_multisite,
            )

            # Flatten params and grads for L-BFGS
            p_flat = jnp.concatenate([p.todense().reshape(-1) for p in params])
            g_flat = jnp.concatenate([g.todense().reshape(-1) for g in grads])

            if prev_params_flat is not None:
                s = p_flat - prev_params_flat
                y = g_flat - prev_grad_flat
                sy = float(jnp.real(jnp.vdot(s, y)))
                if sy > 1e-10:
                    rho = 1.0 / sy
                    lbfgs_history.append((s, y, rho))
                    if len(lbfgs_history) > 10:
                        lbfgs_history.pop(0)
            prev_params_flat = p_flat
            prev_grad_flat = g_flat

            envs_cached = _env_cache["envs"]
            sites_m = {c: params[i] for i, c in enumerate(coords)}
            delta_metric = (
                delta_energy if step > 0 else float(jnp.real(jnp.vdot(g_flat, g_flat)))
            )
            # Compute per-site sizes for slicing
            sizes = [params[i].todense().size for i in range(len(coords))]
            shapes = [params[i].todense().shape for i in range(len(coords))]

            def h0_matvec(v):
                # Slice flat vector into per-site pieces
                grads_v = {}
                offset = 0
                for idx, c in enumerate(coords):
                    sz = sizes[idx]
                    grads_v[c] = _wrap_tensor(
                        v[offset : offset + sz].reshape(shapes[idx]),
                        params[idx],
                    )
                    offset += sz
                z_dict = precondition_gradient_multisite(
                    sites_m, envs_cached, grads_v, delta_metric, config
                )
                return jnp.concatenate([z_dict[c].reshape(-1) for c in coords])

            direction_flat = lbfgs_two_loop(g_flat, lbfgs_history, h0_matvec)
            # Unpack flat direction into per-site Tensors
            dir_parts = []
            offset = 0
            for idx, c in enumerate(coords):
                sz = sizes[idx]
                dir_parts.append(
                    _wrap_tensor(
                        -direction_flat[offset : offset + sz].reshape(shapes[idx]),
                        params[idx],
                    )
                )
                offset += sz
            direction = tuple(dir_parts)
        elif optimizer is not None:
            updates, opt_state = optimizer.update(grads, opt_state, params)
            direction = updates
        else:
            direction = jax.tree.map(lambda g: -g, grads)

        # Tangent projection (no C4v for multisite)
        direction = _tangent_project_unit(direction, params)

        # ── Line search / update ─────────────────────────────────────────
        if use_ls:
            # Snapshot for env-cache restore after the line search; see
            # ``_restore_env_cache_after_line_search`` (#502 Codex P1).
            _ls_env_snap = ("envs" in _env_cache, _env_cache.get("envs"))
            if line_search_method == "hager_zhang":
                from tenax.algorithms._line_search import hager_zhang_line_search

                slope = _tree_dot(grads, direction)
                if slope >= 0:
                    direction = jax.tree.map(lambda g: -g, grads)
                    direction = _tangent_project_unit(direction, params)
                    slope = _tree_dot(grads, direction)
                    if is_metric_lbfgs:
                        lbfgs_history.clear()

                hz_counter = {"phi": 0, "dphi": 0}

                def _phi(alpha):
                    hz_counter["phi"] += 1
                    trial = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    return loss_fn_fwd(trial)

                def _dphi(alpha):
                    hz_counter["dphi"] += 1
                    trial = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    _, g = jax.value_and_grad(loss_fn)(trial)
                    return _tree_dot(g, direction)

                dir_norm = math.sqrt(max(_tree_dot(direction, direction), 1e-30))
                param_norm = math.sqrt(max(_tree_dot(params, params), 1e-30))
                alpha0 = min(1.0, 0.1 * param_norm / dir_norm)

                alpha, f_alpha, converged = hager_zhang_line_search(
                    _phi,
                    _dphi,
                    energy_float,
                    slope,
                    alpha_init=alpha0,
                    rho=1.5,
                    max_step=2.0 * alpha0,
                    energy_bound=max(2.0, 2.0 * abs(best_energy)),
                    max_iter=config.gs_hz_max_iter,
                    bracket_only_phi=True,  # #504: skip dphi in bracket
                )
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD:multisite-tensor] HZ probes phi={hz_counter['phi']} "
                        f"dphi={hz_counter['dphi']} alpha={alpha:.3e} "
                        f"converged={converged}",
                        flush=True,
                    )
                if _is_real_decrease(f_alpha, energy_float):
                    params = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                slope_bt = _tree_dot(grads, direction)
                if slope_bt >= 0:
                    direction = jax.tree.map(lambda g: -g, grads)
                    direction = _tangent_project_unit(direction, params)
                    if is_metric_lbfgs:
                        lbfgs_history.clear()
                params, new_energy, step_size = _backtracking_line_search(
                    params,
                    direction,
                    grads,
                    energy_float,
                    loss_fn_fwd,
                    max_steps=config.gs_line_search_max_steps,
                )
                if _is_real_decrease(new_energy, energy_float):
                    stall_count = 0
                else:
                    stall_count += 1

            _restore_env_cache_after_line_search(_env_cache, _ls_env_snap)

            # Stall recovery
            if config.gs_stall_recovery == "reset" and stall_count > 0:
                # Rollback to best on reset (#454). The CTM-error reset path
                # above (around L2658-2659) already does this for the
                # CTMRGGradientError branch; we extend the same pattern to the
                # Wolfe-failure path that was missed. #298's anti-rollback
                # evidence was on a pre-trifecta CTM stack (pre-PR #406 2x2
                # projector, pre-multisite-CTM rewrite, pre-PR #447 AD
                # stop_gradient) and no longer applies.
                if stall_count > config.gs_stall_recovery_retries:
                    # #455 PR2: at non-final χ stages, treat stall-cap
                    # exhaustion as a signal to advance the χ schedule
                    # rather than exit with best_energy.  Mirrors the
                    # 1-site (07ffe8e) and 2-site (41e590f) intercepts.
                    if config.gs_chi_schedule_steps is not None:
                        steps_in_stage = (step + 1) - stage_start_step
                        _gn_for_bump = (
                            grad_norm_val
                            if grad_norm_val is not None
                            else (
                                _grad_l2_norm(grads)
                                if config.gs_conv_criterion != "dE"
                                else 0.0
                            )
                        )
                        (
                            ctm_cfg,
                            _env_cache,
                            new_stage_idx,
                            bump_fired,
                            _,
                        ) = _advance_chi_stage_if_due(
                            ctm_cfg,
                            _env_cache,
                            chi_schedule=config.gs_chi_schedule_steps,
                            current_stage_idx=current_stage_idx,
                            steps_in_stage=steps_in_stage,
                            config=config,
                            grad_norm=_gn_for_bump,
                            delta_energy=delta_energy,
                            stall_count=stall_count,
                            base_charges=_bump_base_charges_multi,
                        )
                        # Codex review (PR #467): see 1-site stall-cap
                        # intercept for rationale.  Decouple schedule
                        # advance from bump_fired; keep reset block
                        # gated on bump_fired.
                        stage_advanced = new_stage_idx != current_stage_idx
                        if stage_advanced:
                            current_stage_idx = new_stage_idx
                            stage_start_step = step + 1
                        if bump_fired:
                            # Rollback params to best from the previous
                            # stage; _env_cache stays at the freshly
                            # padded post-bump state (the budget-path
                            # invariant — see _apply_chi_bump).
                            params = best_params
                            stall_count = 0
                            if is_metric_lbfgs:
                                lbfgs_history.clear()
                                prev_params_flat = None
                                prev_grad_flat = None
                            if is_cg:
                                cg_direction = None
                                prev_grad = None
                                prev_precond_grad = None
                            if (
                                optimizer is not None
                                and config.gs_optimizer.lower() == "lbfgs"
                            ):
                                opt_state = optimizer.init(params)
                            if config.gs_verbose:
                                print(
                                    f"[iPEPS-AD step {step + 1}] stall-cap at "
                                    f"chi={ctm_cfg.chi} → advancing to next "
                                    f"stage (#455 PR2)",
                                    flush=True,
                                )
                        if bump_fired or stage_advanced:
                            continue
                    n_resets_done = stall_count - 1
                    if config.gs_verbose:
                        print(
                            f"[iPEPS-AD] stall budget exhausted after "
                            f"{n_resets_done} resets, "
                            f"returning best E={best_energy:.10f}",
                            flush=True,
                        )
                    break
                params = best_params
                # #518: ``best_env_cache`` may be at a stale χ if a
                # reactive/scheduled bump fired after it was last
                # snapshotted.  Clear instead of restoring; the next
                # CTM call cold-starts at the current ctm_cfg.chi.
                _drop_env_cache_for_reset(_env_cache)
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_params_flat = None
                    prev_grad_flat = None
                # Optax-backed L-BFGS stores curvature history in opt_state,
                # not in lbfgs_history.  Reinitialize it on the rolled-back
                # params so the next step really is steepest descent.
                if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                    opt_state = optimizer.init(params)
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD] stall #{stall_count}, reset L-BFGS history "
                        f"(rollback to best, retry "
                        f"{stall_count}/{config.gs_stall_recovery_retries})",
                        flush=True,
                    )
            elif (
                config.gs_stall_recovery == "noise"
                and stall_count > 0
                and stall_count <= config.gs_noise_recovery_retries
            ):
                noise_key = jax.random.PRNGKey(step * 1000 + stall_count)
                noisy_params = []
                for i, p in enumerate(params):
                    k = jax.random.fold_in(noise_key, i)
                    data = p.todense()
                    noise = config.gs_noise_amplitude * _random_noise(
                        k, data.shape, data.dtype
                    )
                    noisy = data + noise * jnp.linalg.norm(data)
                    noisy = noisy / (jnp.linalg.norm(noisy) + 1e-10)
                    noisy_params.append(_wrap_tensor(noisy, p))
                params = tuple(noisy_params)
                if config.gs_verbose:
                    print(f"[iPEPS-AD] stall #{stall_count}, adding noise", flush=True)
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_params_flat = None
                    prev_grad_flat = None
        else:
            import optax

            params = optax.apply_updates(params, direction)
            params = _normalize_params(params)

        # Reactive ε_T auto-bump (variPEPS §2.8.2, #472) followed by the
        # scheduled outer-loop χ bump (#453 / #455).  Compose order
        # mirrors the 1-site / 2-site paths: reactive first so it can
        # pre-empt the schedule; scheduled second to advance the stage
        # index even on idempotent (post-reactive) bumps.  The reactive
        # bump is NOT gated on ``warmup_ok`` — that gate exists to keep
        # the scheduled ``dE`` trigger from firing on false-convergence
        # signals early in the run, but reactive is driven by the CTM-side
        # ε_T which is well-defined from step 1.  ``chi_before`` is
        # snapshotted before either fires so the post-bump reset block
        # catches either trigger via ``ctm_cfg.chi != chi_before``.
        chi_before = ctm_cfg.chi
        last_eps_t = float(_env_cache.get("max_truncation_error", 0.0))
        ctm_cfg, _env_cache = _maybe_bump_chi(
            ctm_cfg,
            _env_cache,
            last_eps_t,
            base_charges=_bump_base_charges_multi,
        )
        # Scheduled outer-loop χ bump (#453 / #455).  No-ops when
        # ``gs_chi_schedule_steps`` is None.  Fires at the step boundary
        # so the next iteration's value_and_grad sees the bumped χ — same
        # invariant as the 1-site path.  Per-stage state
        # (current_stage_idx, stage_start_step) drives the new helper;
        # #455 PR2 will add convergence/stall-cap triggers.
        if config.gs_chi_schedule_steps is not None:
            steps_in_stage = (step + 1) - stage_start_step
            _gn_for_bump = (
                grad_norm_val
                if grad_norm_val is not None
                else (_grad_l2_norm(grads) if config.gs_conv_criterion != "dE" else 0.0)
            )
            # Codex P2 review on PR #467: the convergence-block above
            # is gated on ``warmup_ok`` to prevent ``dE ≈ 0`` early
            # steps and post-stall resets from acting on a false
            # convergence signal.  This end-of-step bump bypasses that
            # gate, so inject a synthetic large ``delta_energy`` when
            # ``not warmup_ok`` to keep the dE trigger inside
            # ``_advance_chi_stage_if_due`` from firing prematurely.
            # Grad-norm and stall-cap signals are unaffected.
            _dE_for_bump = delta_energy if warmup_ok else float("inf")
            ctm_cfg, _env_cache, new_stage_idx, bump_fired, _should_break = (
                _advance_chi_stage_if_due(
                    ctm_cfg,
                    _env_cache,
                    chi_schedule=config.gs_chi_schedule_steps,
                    current_stage_idx=current_stage_idx,
                    steps_in_stage=steps_in_stage,
                    config=config,
                    grad_norm=_gn_for_bump,
                    delta_energy=_dE_for_bump,
                    stall_count=stall_count,
                    base_charges=_bump_base_charges_multi,
                )
            )
            stage_advanced = new_stage_idx != current_stage_idx
            if stage_advanced:
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD step {step + 1}] schedule advance: "
                        f"stage {current_stage_idx} → {new_stage_idx}, "
                        f"chi {chi_before} → {ctm_cfg.chi} (#455)",
                        flush=True,
                    )
                current_stage_idx = new_stage_idx
                stage_start_step = step + 1
        # Reset block lives OUTSIDE the schedule-only ``if`` so a
        # reactive-only bump (no schedule) still clears stall_count,
        # CG state, and L-BFGS history at the new χ.  See the 2-site
        # twin block — same Codex PR #473 review.
        if ctm_cfg.chi != chi_before:
            # Bump fired — fresh landscape, fresh stall budget.
            stall_count = 0
            if is_metric_lbfgs:
                lbfgs_history.clear()
                prev_params_flat = None
                prev_grad_flat = None
            if is_cg:
                cg_direction = None
                prev_grad = None
                prev_precond_grad = None
            if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                opt_state = optimizer.init(params)

    # ── Final evaluation ─────────────────────────────────────────────────
    def _eval_fresh(p, env_init=None):
        site_tensors = {}
        p_normed = _normalize_params(p)
        for i, c in enumerate(coords):
            site_tensors[c] = p_normed[i]
        envs, _ = python_loop_ctm_converge(
            site_tensors,
            neighbors,
            **ctm_converge_kwargs(ctm_cfg, env_init=env_init),
        )
        E_ = float(
            compute_energy_ctm_tensor_multisite(
                site_tensors,
                envs,
                neighbors,
                gate,
            )
        )
        return site_tensors, envs, E_

    sites_last, envs_last, E_last = _eval_fresh(params, _env_cache.get("envs", None))

    # Pad best_env_cache envs to the current ctm_cfg.chi before use.
    # Mirrors the same fix on the 1-site and 2-site paths — see comment there.
    # Fixes #469.
    # NOTE: if a new _eval_fresh(best_params, best_env_cache["envs"]) call site
    # is added later, it must reapply this padding — there's no producer-side
    # guarantee that the snapshot is at ctm_cfg.chi.
    _best_env_init_multi = best_env_cache.get("envs", None)
    if _best_env_init_multi:
        # Issue #514: see 1-site finalize.  Pad to the larger of the
        # static ``ctm_cfg.chi`` and the env_cache's actual chi.
        # See 1-site finalize: consult both live cache and the
        # best-env snapshot (Codex P2 on PR #542).
        _target_chi_multi = max(
            ctm_cfg.chi,
            _env_dict_chi(_env_cache.get("envs")) or 0,
            _env_dict_chi(_best_env_init_multi) or 0,
        )
        _best_env_init_multi = {
            c: pad_dense_env_chi(
                _best_env_init_multi[c],
                _target_chi_multi,
                base_charges=_bump_base_charges_multi,
            )
            for c in _best_env_init_multi
        }

    if best_params is not params:
        sites_best, envs_best, E_best_fresh = _eval_fresh(
            best_params, _best_env_init_multi
        )
    else:
        E_best_fresh = E_last

    # Pick whichever fresh evaluation is lower
    if E_last <= E_best_fresh:
        out_sites, out_envs, E_gs = sites_last, envs_last, E_last
    else:
        out_sites, out_envs, E_gs = sites_best, envs_best, E_best_fresh

    if config.gs_verbose:
        print(f"[iPEPS-AD:multisite] final E={E_gs:.10f}", flush=True)

    # Map coord keys back to site names
    out_tensors = {coord_to_name[c]: out_sites[c] for c in coords}
    out_envs_named = {coord_to_name[c]: out_envs[c] for c in coords}

    if config.return_history:
        history = {
            "energies": _history_energies,
            "step_times": _history_step_times,
            "jit_compile_time": _jit_compile_time,
            "num_steps": len(_history_energies),
            "converged": _converged,
        }
        return out_tensors, out_envs_named, E_gs, history
    return out_tensors, out_envs_named, E_gs


def optimize_fpeps_ad(
    hamiltonian_gate: Tensor,
    A_init: Tensor | None,
    config: iPEPSConfig,
    fpeps_config=None,
) -> tuple:
    """AD-based ground state optimization of fermionic iPEPS.

    Uses automatic differentiation through the CTM fixed-point equation
    to compute exact gradients of the energy with respect to the
    fermionic site tensor, then optimizes with optax.

    Accepts either ``DenseTensor`` or ``SymmetricTensor`` (e.g.
    ``FermionParity`` symmetry) inputs — the optimizer shell is
    polymorphic over the Tensor protocol (#297) and returns a tensor
    of the same type as the input, preserving charges and flows.

    Args:
        hamiltonian_gate: 2-site Hamiltonian as a ``Tensor`` (typically
            a ``SymmetricTensor`` with ``FermionParity`` symmetry),
            shape ``(d, d, d, d)``.
        A_init:           Initial fPEPS site tensor ``(D, D, D, D, d)``
            with labels ``(u, d, l, r, phys)``.  If ``None``, a random
            tensor with ``FermionParity`` is created using
            *fpeps_config*.
        config:           ``iPEPSConfig`` with AD optimization settings
            (learning rate, number of steps, CTM config, etc.).
        fpeps_config:     ``FPEPSConfig`` used only when ``A_init`` is
            ``None`` to build the initial tensor (bond dimension D,
            physical dimension d=2, FermionParity charges).

    Returns:
        ``(A_opt, env, E_gs)`` where ``A_opt`` is the optimized site
        tensor (same type as ``A_init``), ``env`` is a ``CTMTensorEnv``,
        and ``E_gs`` is the ground-state energy per site.
    """
    if A_init is None:
        if fpeps_config is None:
            raise ValueError(
                "fpeps_config is required when A_init is None "
                "(needed to build the initial fPEPS tensor)."
            )
        from tenax.algorithms.fermionic_ipeps import _build_initial_fpeps_tensor

        A_init = _build_initial_fpeps_tensor(fpeps_config)

    return _optimize_gs_ad_tensor(hamiltonian_gate, A_init, config)
