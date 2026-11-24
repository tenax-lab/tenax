"""``optimize_gs_ad`` for a charged C4v ``SymmetricTensor`` (spec 2, section 5).

Design: ``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``.

A charged C4v site tensor A (a virtual charge q with q != dual(q)) is the
2-sublattice state (A, negate(A)) (#1211).  This loop optimises it with
L-BFGS and the line searches of the 1x1 loop.  The gradient comes from
``charged_c4v_converge``: implicit AD through the #1211 sweep at the layout
of the cold start, which stays frozen for the run.  The parameter x is A's
flat block buffer; the site is A(x) = P x / |P x|, with P the C4v
projection of the buffer.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time

import jax
import jax.numpy as jnp

from tenax.algorithms import _ctm_tensor_c4v_charged_ad as _engine
from tenax.algorithms._ctm_convergence_policy import CTMNotConvergedError
from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout
from tenax.algorithms._ctm_tensor_c4v_staggered import (
    _c4v_staggered_enlarged_corner,
    _c4v_staggered_layout,
    _c4v_staggered_to_full_env,
)
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.algorithms._ipeps_optimize_shared import (
    _backtracking_line_search,
    _converged_outer,
    _euclidean_grads,
    _grad_l2_norm,
    _is_real_decrease,
    _log_ad_converged,
    _log_ad_step,
    _log_ctm_not_converged_reset,
    _normalize_params,
    _normalize_stall_recovery,
    _resolve_line_search_method,
    _should_accept_best,
)
from tenax.algorithms._line_search import LineSearchAborted, hager_zhang_line_search
from tenax.algorithms._metric_precond import lbfgs_two_loop
from tenax.algorithms.ipeps import symmetrize_c4v
from tenax.core.tensor import SymmetricTensor, Tensor

__all__ = [
    "_c4v_projector_matrix",
    "_check_charged_c4v_config",
    "_free_layout",
    "_keep_lbfgs_pair",
    "_trial_energy",
    "optimize_gs_ad_c4v_charged",
]

_logger = logging.getLogger(__name__)
_BACKEND = "1site-c4v-charged"
_LBFGS_MEMORY = 10  # curvature pairs kept, as in the 1x1 loop
_PAIR_RTOL = 1e-8  # a pair enters when s.y > _PAIR_RTOL |s| |y| (_keep_lbfgs_pair)


def _check_charged_c4v_config(config, ctm, A_init) -> None:
    """Refuse each setting this loop cannot honour or would ignore.

    The table of spec 2, section 5, and ruling 8 of the plan.  ``ctm`` is
    ``config.ctm`` after the ``gs_projector_method`` override.

    Raises:
        ValueError: naming the first refused setting and the reason.
    """

    def refuse(setting: str, reason: str) -> None:
        raise ValueError(f"charged C4v AD: {setting} is not supported: {reason}")

    if not config.gs_implicit_ad:
        refuse("gs_implicit_ad=False", "there is no explicit-AD path for it")
    if jnp.iscomplexobj(A_init._data):
        refuse("complex data", "the mirror pairing holds for real data only (#1211)")
    if ctm.projector_method not in ("svd", "eigh"):
        refuse(
            f"projector_method={ctm.projector_method!r}",
            "'svd' and 'eigh' both select the #1211 block projector",
        )
    if ctm.forward_gauge not in ("auto", "bond_phase"):
        refuse(
            f"forward_gauge={ctm.forward_gauge!r}",
            "the loop aligns one sign per chi index itself",
        )
    if ctm.ctm_conv_method != "elementwise":
        refuse(
            f"ctm_conv_method={ctm.ctm_conv_method!r}",
            "the adjoint needs an element-wise fixed point",
        )
    if ctm.projector_backward not in ("auto", "flow"):
        refuse(
            f"projector_backward={ctm.projector_backward!r}",
            "the gradient always includes the projector response",
        )
    if ctm.on_unconverged != "raise":
        refuse(
            f"on_unconverged={ctm.on_unconverged!r}",
            "without a fixed point there is no gradient",
        )
    for name in ("chi_ramp", "chi_auto_bump", "ctmrg_heuristic_increase_chi"):
        if getattr(ctm, name):
            refuse(name, "the layout is frozen for the run")
    if not ctm.fuse_virtual_legs:
        refuse("fuse_virtual_legs=False", "the split CTM is dense-only")
    if ctm.ctm_mixing > 0:
        refuse(
            f"ctm_mixing={ctm.ctm_mixing}",
            "a mixed sweep is not the map whose fixed point the adjoint solves",
        )
    if ctm.device_mesh is not None:
        refuse("device_mesh", "the charged sweep runs on one device")
    if config.gs_stall_recovery == "noise":
        refuse("gs_stall_recovery='noise'", "the noise kick assumes dense params")
    if config.cg_gates is not None:
        refuse("cg_gates", "the charged branch has no coarse-grained energy")
    if config.gs_optimizer.lower() != "lbfgs":
        refuse(f"gs_optimizer={config.gs_optimizer!r}", "the loop runs L-BFGS only")
    if config.gs_line_search is False:
        refuse("gs_line_search=False", "the L-BFGS step needs a line search")
    for name in (
        "gs_chi_schedule_steps",
        "gs_ctm_conv_tol_schedule",
        "gs_ctm_max_iter_schedule",
        "gs_plateau_patience_schedule",
    ):
        if getattr(config, name):
            refuse(name, "the loop runs one CTMConfig for the whole run")


def _c4v_projector_matrix(A) -> jax.Array:
    """The C4v projection of A's flat block buffer, as an n x n matrix P.

    P is the Jacobian of the linear map
    x -> buffer(from_dense(symmetrize_c4v(todense(A(x))))).  It is symmetric
    and idempotent.  A must hold every block that its charges allow, or P
    is not square.
    """
    treedef = jax.tree_util.tree_structure(A)

    def project(x):
        dense = jax.tree_util.tree_unflatten(treedef, [x]).todense()
        return SymmetricTensor.from_dense(symmetrize_c4v(dense), A.indices)._data

    return jax.jacfwd(project)(A._data)


def _trial_energy(A, warm, layout, ctm, gate, d) -> float:
    """The energy at a line-search trial: the element-wise phase from ``warm``.

    Returns +inf when the phase finds no fixed point or the energy is not
    finite.  A trial is a bad point, never an error (Review Focus 3).
    """
    try:
        C, T = _engine._elementwise_phase(A, warm, layout, ctm, {})
    except CTMNotConvergedError:
        return float("inf")
    env = _c4v_staggered_to_full_env(C, T)
    E = float(compute_energy_ctm_tensor(A, env, gate, d=d))
    return E if math.isfinite(E) else float("inf")


def _keep_lbfgs_pair(s, y) -> bool:
    """Whether the curvature pair (s, y) enters the L-BFGS history.

    Yes when s.y > ``_PAIR_RTOL`` |s| |y|: the angle of s and y decides,
    not their size.  A negative, a zero or a nearly orthogonal pair is
    dropped.
    """
    sy = float(jnp.vdot(s, y))
    return sy > _PAIR_RTOL * float(jnp.linalg.norm(s)) * float(jnp.linalg.norm(y))


def _free_layout(A, C, T, chi) -> BondLayout | None:
    """The layout that a free ranking keeps at e* = (C, T); None if it refuses."""
    Qf = _c4v_staggered_enlarged_corner(C, T, _build_double_layer_tensor(A))
    try:
        return _c4v_staggered_layout(Qf, chi)
    except ValueError:
        return None


def optimize_gs_ad_c4v_charged(hamiltonian_gate, A_init, config):
    """``optimize_gs_ad`` on a charged C4v ``SymmetricTensor``.

    Args:
        hamiltonian_gate: the 2-site gate, shape ``(d, d, d, d)``.
        A_init: a charged C4v ``SymmetricTensor``.  It need not be C4v or
            hold every block: the loop rebuilds it with every valid block
            and projects it to C4v.
        config: an ``iPEPSConfig`` that ``_check_charged_c4v_config`` accepts.

    Returns:
        ``(A, env, E)``, and a ``history`` dict last when
        ``config.return_history``.  ``env`` is
        ``_c4v_staggered_to_full_env(C*, T*)``, the env that
        ``ctm_tensor_c4v`` returns for charged input.

    Raises:
        ValueError: a refused setting, or an ``A_init`` with no C4v part.
        CTMNotConvergedError: a forward with no fixed point and no reset left.
        RuntimeError: a failed adjoint solve at an evaluated point.
    """
    ctm = config.ctm
    if config.gs_projector_method is not None:
        ctm = dataclasses.replace(ctm, projector_method=config.gs_projector_method)
    config = _normalize_stall_recovery(config, unit_cell="1x1")
    _check_charged_c4v_config(config, ctm, A_init)
    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.asarray(hamiltonian_gate)
    )
    d = gate.shape[0]

    # Rebuild with every valid block, so P is square (Review Focus 4).
    A0 = SymmetricTensor.from_dense(A_init.todense(), A_init.indices)
    treedef = jax.tree_util.tree_structure(A0)
    P = _c4v_projector_matrix(A0)
    x = P @ A0._data
    x_norm = float(jnp.linalg.norm(x))
    if x_norm == 0.0:
        raise ValueError("charged C4v AD: A_init has no C4v component")
    x = x / x_norm

    def site(x_):
        y = P @ x_
        return jax.tree_util.tree_unflatten(treedef, [y / jnp.linalg.norm(y)])

    record: dict = {}
    C, T, layout = _engine.charged_c4v_cold_start(site(x), ctm)
    warm = _engine._elementwise_phase(site(x), (C, T), layout, ctm, record)

    def loss(x_, warm_):
        A_ = site(x_)
        C_, T_ = _engine.charged_c4v_converge(A_, warm_, layout, ctm, record)
        env_ = _c4v_staggered_to_full_env(C_, T_)
        return compute_energy_ctm_tensor(A_, env_, gate, d=d), (C_, T_)

    value_and_grad = jax.value_and_grad(loss, has_aux=True)

    per_step = (
        "energies",
        "grad_norms",
        "step_times",
        "forward_sweeps",
        "adjoint_matvecs",
        "adjoint_residual",
    )
    hist: dict = {key: [] for key in per_step}
    mismatch_steps: list[int] = []
    state = {"converged": False}

    def finish(x_, warm_, E_):
        A_ = site(x_)
        env_ = _c4v_staggered_to_full_env(*warm_)
        if not config.return_history:
            return A_, env_, E_
        history = {
            **hist,
            **state,
            "num_steps": len(hist["energies"]),
            # Every recorded step had a converged forward: an unconverged
            # forward records nothing (it resets or raises).
            "ctm_converged": [True] * len(hist["energies"]),
            "layout": layout.dims,
            "layout_mismatch_steps": mismatch_steps,
        }
        return A_, env_, E_, history

    if config.gs_num_steps == 0:
        env0 = _c4v_staggered_to_full_env(*warm)
        E0 = float(compute_energy_ctm_tensor(site(x), env0, gate, d=d))
        return finish(x, warm, E0)

    line_search = _resolve_line_search_method(config, ctm)
    retries = config.gs_stall_recovery_retries
    best_E, best_x, best_warm = float("inf"), x, warm
    prev_E = float("inf")
    lbfgs_history: list = []
    prev_x = prev_g = None
    stall = 0

    for step in range(config.gs_num_steps):
        t0 = time.perf_counter()
        try:
            (E_val, e_star), g = value_and_grad(x, warm)
        except CTMNotConvergedError as exc:
            # The 1-site rule (#1059): reset to the best point within the
            # retry budget; re-raise when the failing point is the best one,
            # no best exists yet, or no retry is left.
            exc.step = step + 1
            if x is best_x or best_E == float("inf") or stall + 1 > retries:
                raise
            _log_ctm_not_converged_reset(
                exc,
                conv_tol=ctm.conv_tol,
                n_reset=stall + 1,
                retries=retries,
                verbose=config.gs_verbose,
            )
            stall += 1
            x, warm = best_x, best_warm
            lbfgs_history.clear()
            prev_x = prev_g = None
            prev_E = float("inf")
            continue
        warm = e_star

        # One free ranking per evaluation (spec 2, section 5, step 5).
        free = _free_layout(site(x), *warm, ctm.chi)
        if free != layout:
            if not mismatch_steps:
                _logger.warning(
                    "[iPEPS-AD:%s] step %d: a free ranking at e* keeps %s, not "
                    "the frozen layout %s; the run keeps the frozen layout "
                    "(logged once)",
                    _BACKEND,
                    step + 1,
                    None if free is None else free.dims,
                    layout.dims,
                )
            mismatch_steps.append(step + 1)

        n_nonfinite = int(jnp.sum(~jnp.isfinite(g)))
        g = _euclidean_grads(jnp.where(jnp.isfinite(g), g, 0.0))
        E = float(E_val)
        grad_norm = _grad_l2_norm(g)
        hist["energies"].append(E)
        hist["grad_norms"].append(grad_norm)
        for key in ("forward_sweeps", "adjoint_matvecs", "adjoint_residual"):
            hist[key].append(record.get(key))
        if _should_accept_best(
            current_best=best_E, candidate=E, floor=config.gs_energy_floor
        ):
            best_E, best_x, best_warm = E, x, warm
        delta = abs(E - prev_E)
        prev_E = E
        if config.gs_verbose:
            _log_ad_step(
                _BACKEND,
                step,
                config.gs_num_steps,
                E,
                delta,
                best_E,
                grad_norm=grad_norm,
            )
        if n_nonfinite:
            prev_E = float("inf")  # a masked gradient must not pass the exit test
        elif _converged_outer(config, delta, grad_norm):
            state["converged"] = True
            if config.gs_verbose:
                _log_ad_converged(
                    _BACKEND,
                    step,
                    delta,
                    config.gs_conv_tol,
                    grad_norm=grad_norm,
                    grad_norm_tol=config.gs_grad_norm_tol,
                    criterion=config.gs_conv_criterion,
                )
            hist["step_times"].append(time.perf_counter() - t0)
            break

        # The L-BFGS direction, identity H0.  The pair rule is relative
        # (_keep_lbfgs_pair).  The absolute floor s.y > 1e-10 of the dense
        # 1x1 loop drops every pair once |s| |y| ~ 1e-9.  At D=3, chi=36, in
        # a 16-step restart from a run's end state, it kept the step-2 pair
        # (s.y = 1.08e-10) and dropped steps 3-16 (s.y = 6.5e-11,
        # cos(s, y) = 0.07-0.08).  The loop then sat in a 2-cycle on that one
        # stale pair: |g| alternated 8.6e-6 / 2.2e-5, and E fell 3.3e-10 per
        # step.
        if prev_x is not None:
            s, y = x - prev_x, g - prev_g
            if _keep_lbfgs_pair(s, y):
                lbfgs_history.append((s, y, 1.0 / float(jnp.vdot(s, y))))
                if len(lbfgs_history) > _LBFGS_MEMORY:
                    lbfgs_history.pop(0)
        prev_x, prev_g = x, g
        direction = -lbfgs_two_loop(g, lbfgs_history, lambda v: v)
        slope = float(jnp.vdot(g, direction))
        if slope >= 0:
            direction, slope = -g, -float(jnp.vdot(g, g))
            lbfgs_history.clear()

        def trial_energy(x_trial):
            return _trial_energy(site(x_trial), warm, layout, ctm, gate, d)

        if line_search == "hager_zhang":

            def phi(alpha):
                return trial_energy(_normalize_params(x + alpha * direction))

            def dphi(alpha):
                trial = _normalize_params(x + alpha * direction)
                try:
                    _value, g_trial = value_and_grad(trial, warm)
                except CTMNotConvergedError as exc:  # first: a RuntimeError
                    raise LineSearchAborted from exc
                except RuntimeError as exc:  # HZ makes it a bad point
                    _logger.warning(
                        "[iPEPS-AD:%s] the adjoint solve failed at a "
                        "line-search trial (alpha=%.3e): %s",
                        _BACKEND,
                        alpha,
                        exc,
                    )
                    raise
                return float(jnp.vdot(_euclidean_grads(g_trial), direction))

            x_scale = float(jnp.linalg.norm(x))
            d_scale = max(float(jnp.linalg.norm(direction)), 1e-15)
            alpha0 = min(1.0, 0.1 * x_scale / d_scale)
            alpha, E_new, _wolfe = hager_zhang_line_search(
                phi,
                dphi,
                E,
                slope,
                alpha_init=alpha0,
                rho=1.5,
                max_step=2.0 * alpha0,
                energy_bound=max(2.0, 2.0 * abs(best_E)),
                max_iter=config.gs_hz_max_iter,
                bracket_only_phi=True,
            )
            x_new = _normalize_params(x + alpha * direction)
        else:
            x_new, E_new, _alpha = _backtracking_line_search(
                x,
                direction,
                g,
                E,
                trial_energy,
                max_steps=config.gs_line_search_max_steps,
            )

        if _is_real_decrease(E_new, E):
            x, stall = x_new, 0
        else:
            stall += 1
            if stall > retries:
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD:{_BACKEND}] stall budget exhausted after "
                        f"{stall - 1} resets, returning best E={best_E:.10f}",
                        flush=True,
                    )
                hist["step_times"].append(time.perf_counter() - t0)
                break
            x, warm = best_x, best_warm
            lbfgs_history.clear()
            prev_x = prev_g = None
            prev_E = float("inf")
        hist["step_times"].append(time.perf_counter() - t0)

    if best_E == float("inf"):  # gs_energy_floor refused every energy
        A_last = site(x)
        e_last = _engine._elementwise_phase(A_last, warm, layout, ctm, {})
        env_last = _c4v_staggered_to_full_env(*e_last)
        E_last = float(compute_energy_ctm_tensor(A_last, env_last, gate, d=d))
        return finish(x, e_last, E_last)
    return finish(best_x, best_warm, best_E)
