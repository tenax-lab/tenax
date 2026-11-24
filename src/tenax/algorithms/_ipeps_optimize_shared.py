"""Helpers shared by the iPEPS AD optimizer entry points.

These functions used to live in ``ipeps_optimize``.  They were moved here to
break the ``ipeps_optimize`` <-> ``ipeps_optimize_root_implicit`` import cycle
(#790).

Both modules need them, and ``ipeps_optimize`` must be able to dispatch *into*
``ipeps_optimize_root_implicit``.  Importing them the other way -- even from
inside a function body, which is what the two modules did -- still forms a
cycle: ``tests/test_architecture_imports.py`` walks the full AST, so a
function-local import is an edge like any other.  Hoisting the shared leaves
into a module that imports neither is what actually breaks it.

The line-search and logging helpers (``_backtracking_line_search``,
``_is_real_decrease``, ``_normalize_stall_recovery``,
``_resolve_line_search_method``, ``_log_ad_step``,
``_log_ctm_not_converged_reset`` and the ``_tree_*`` leaves they use) moved
here for the same reason, with the same cycle shape: ``ipeps_optimize``
dispatches *into* ``ipeps_optimize_c4v_charged``, which needs them.  The
charged loop imports them from here and never from ``ipeps_optimize``.

``ipeps_optimize`` re-exports every name defined here, so
``from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor`` keeps
working for the tests and benchmarks that already use that path.
"""

from __future__ import annotations

import logging
import math

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

__all__ = [
    "_STALL_NOISE_FLOOR",
    "_backtracking_line_search",
    "_build_optimizer",
    "_converged_outer",
    "_euclidean_grads",
    "_grad_l2_norm",
    "_is_real_decrease",
    "_log_ad_converged",
    "_log_ad_step",
    "_log_ctm_not_converged_reset",
    "_normalize_params",
    "_normalize_stall_recovery",
    "_resolve_line_search_method",
    "_should_accept_best",
    "_tree_add",
    "_tree_dot",
    "_tree_scale",
    "_use_line_search",
    "_warn_implicit_ad_variational_caveat",
    "_wrap_as_dense_tensor",
]

# The reset warning keeps the logger name of its old home, which users filter on.
_logger = logging.getLogger("tenax.algorithms.ipeps_optimize")


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


def _should_accept_best(
    *,
    current_best: float,
    candidate: float,
    floor: float | None,
    remeasured: bool = False,
) -> bool:
    """Return True iff ``candidate`` should overwrite ``best_energy``.

    Rejects non-finite (NaN/inf) candidates, candidates not strictly
    below ``current_best`` (unless ``remeasured``, below), and candidates
    at or below ``floor`` (treated as non-variational CTM artifacts per
    issue #298).  A ``None`` floor disables the floor check.

    ``remeasured=True`` means ``candidate`` is a *certified* measurement at
    ``best_params`` itself: the caller passes ``params is best_params and
    _remeasure_certified(...)``.  ``params is best_params`` is what every
    rollback produces -- spike, CTM-error and stall resets, and the chi-bump /
    stage-advance rollbacks, so ``best_energy`` is re-based at the new chi
    there.  A certified re-measure replaces ``best_energy`` even when it is
    *higher* (#1095): a single in-loop forward that read low must not stay
    the best forever and rank every later candidate against a value the
    state does not have.  An uncertified re-measure (unconverged under
    ``on_unconverged="warn"``) is not ``remeasured``, so it can only lower
    the bar through the ordinary rule.  The finiteness and floor checks
    still apply; a floor-violating re-measure keeps the previous value.
    """
    if not math.isfinite(candidate):
        return False
    if floor is not None and candidate <= floor:
        return False
    if remeasured:
        return True
    if candidate >= current_best:
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


def _normalize_params(params):
    """Normalize iPEPS site tensor(s)."""
    if isinstance(params, tuple):
        return tuple(_normalize_params(p) for p in params)
    if hasattr(params, "norm"):
        return params * (1.0 / (params.norm() + 1e-10))
    # Plain JAX array (e.g. C4v coefficients) — use jnp.linalg.norm
    return params / (jnp.linalg.norm(params) + 1e-10)


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

    - ``"grad_norm"`` (default since v0.9.0): exit on
      ``||grad||_2 < gs_grad_norm_tol`` (variPEPS
      ``optimizer_convergence_eps`` analog, issue #448).
    - ``"dE"`` (default before v0.9.0, deprecated): exit on
      ``|dE| < gs_conv_tol``.
    - ``"both"``: require both to hold simultaneously.

    A ``None`` ``grad_norm`` defeats any criterion that needs it, and so
    does a NaN one (``nan < tol`` is False) -- keep the comparisons in the
    ``x < tol`` form; ``not (x >= tol)`` would let NaN through.
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


def _euclidean_grads(grads):
    """Convert a JAX cotangent tree to Euclidean (descent) gradients.

    For a real objective of complex parameters, JAX's cotangent pairs
    UNCONJUGATED: ``df = Re sum(g * dz)``.  The steepest-descent direction —
    and the vector every Euclidean consumer here expects (Optax updates,
    ``_tree_dot`` slopes, CG beta, L-BFGS curvature pairs, the metric
    preconditioner) — is therefore ``-conj(g)``, not ``-g``.  Feeding the
    raw cotangent to Optax made the update *ascend* along the imaginary
    coordinates and made the result depend on the global phase of the
    initial tensor (#957).  ``conj`` is the identity on real leaves, so the
    default real-tensor paths are bit-for-bit unchanged.

    Must be applied at every gradient production site: the main-loop
    ``value_and_grad`` of each dispatcher in ``ipeps_optimize``, the trial
    gradients inside the Hager-Zhang ``dphi`` callbacks, and the
    ``*_energy_and_grad`` results in ``ipeps_optimize_root_implicit`` —
    the root-implicit loop was missed on the first pass and found by review
    (the duplicate-implementation trap), which is why this lives in the
    shared module now: one definition, every optimizer loop.
    """
    return jax.tree.map(jnp.conj, grads)


def _grad_l2_norm(grads) -> float:
    """L2 norm of an optax gradient pytree, returned as a Python float."""
    leaves = jax.tree_util.tree_leaves(grads)
    if not leaves:
        return 0.0
    sq = sum(jnp.vdot(jnp.ravel(g), jnp.ravel(g)).real for g in leaves)
    return float(jnp.sqrt(sq))


def _su_start_guard_armed(init_from_su: bool, start_step: int) -> bool:
    """Whether to skip the convergence test on the first gradient evaluation.

    Under ``"grad_norm"`` the very first evaluation can pass the test, which
    ``"dE"`` never could (``prev_energy = inf``).  For a simple-update start
    that is a stationary point (a saddle such as the product state, or the SU
    plateau the 1-site ``"noise"`` default exists for) that would return the
    SU state as converged before the line search could fail and trigger stall
    recovery (Codex P1 on #1075).  So: skip the test once and take the
    optimizer step, letting the configured recovery act; a still-stationary
    next evaluation converges normally.  Only for an SU-derived start on a
    fresh run -- a user-supplied ``A_init`` that is already stationary is a
    warm start and converges at once, and so does a checkpoint resume.
    """
    return init_from_su and start_step == 0


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


def _normalize_stall_recovery(config, *, unit_cell: str):
    """Resolve an unset ``gs_stall_recovery`` to ``"reset"`` on every unit cell.

    ``"reset"`` rolls back to ``best_params`` and clears the L-BFGS / CG
    state; ``"noise"`` kicks the *current* params by ``gs_noise_amplitude``
    (10 % Frobenius) with no rollback.  The 1-site default used to be
    ``"noise"``, to break the 1-site C4v run off an SU-init plateau with
    gradient norms ~1e-10.  That plateau no longer reproduces: Heisenberg
    1x1 from the default SU start has |g| = 0.30 at D=2 chi=8 and 0.54 at
    D=3 chi=9, with and without ``gs_c4v``, and at D=2 ``"noise"`` and
    ``"reset"`` reach the same energy to 1e-10.  The kick is destructive near
    a settled energy instead: on D=3 chi=16
    Heisenberg (``forward_gauge="bond_phase"``, ``projector_backward="flow"``)
    a stall at E=-0.66819 -- fired by a Hager-Zhang approximate-Wolfe step
    that does not lower E by more than ``_STALL_NOISE_FLOOR`` -- kicked the
    state to E=-0.033, after which the CTM adjoint diverged (residual up to
    1e46) and the run never returned to its best energy; the same kick took
    the 1-site C4v D=3 run from -0.6668 to -0.50.  The 2-site default has
    been ``"reset"`` since #520 for the same reason.  ``unit_cell`` is kept
    for the call sites; an explicit user setting is never overridden.
    Should an SU start be stationary anyway, ``_su_start_guard_armed``
    keeps the ``"grad_norm"`` test from accepting it before the first
    optimizer step.
    """
    from dataclasses import replace

    if config.gs_stall_recovery is not None:
        return config
    del unit_cell
    return replace(config, gs_stall_recovery="reset")


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


def _log_ctm_not_converged_reset(
    exc, *, conv_tol, n_reset: int, retries: int, verbose: bool
) -> None:
    """Spec section 3 reset log line (logger always; stdout when verbose)."""
    from tenax.algorithms._ctm_convergence_policy import (
        _fmt,
        format_step_multiplier,
    )

    info = exc.info
    msg = (
        f"[iPEPS-AD] CTM forward not converged at step {exc.step} "
        f"(sweeps {_fmt(getattr(info, 'iterations', None), 'd')}, sv_diff "
        f"{_fmt(getattr(info, 'sv_diff', None), '.3g')} vs tol "
        f"{_fmt(conv_tol, 'g')}, step multiplier {format_step_multiplier(info)}) "
        f"— reset to best (#{n_reset}/{retries})"
    )
    _logger.warning(msg)
    if verbose:
        print(msg, flush=True)
