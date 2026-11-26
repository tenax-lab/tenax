"""``optimize_gs_ad`` for a charged C4v ``SymmetricTensor`` (spec 2, section 5).

Design: ``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``.

A charged C4v site tensor A (a virtual charge q with q != dual(q)) is the
2-sublattice state (A, negate(A)) (#1211).  This loop optimises it with
L-BFGS and the line searches of the 1x1 loop.  The gradient comes from
``charged_c4v_converge``: implicit AD through the #1211 sweep at the layout
of the cold start, which stays frozen for the run.  The parameter x is A's
flat block buffer; the site is A(x) = P x / |P x|, with P the C4v
projection of the buffer: the mean of x over the 8 permutations of the
buffer that the C4v elements make (``_c4v_permutations``).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms import _ctm_tensor_c4v_charged_ad as _engine
from tenax.algorithms._ctm_c4v_charges import _dual_charge
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
from tenax.core.tensor import SymmetricTensor, Tensor

__all__ = [
    "_CTM_CHECKED",
    "_CTM_IGNORED",
    "_CTM_IMPLIED",
    "_CTM_READ",
    "_c4v_permutations",
    "_check_charged_c4v_config",
    "_free_layout",
    "_keep_lbfgs_pair",
    "_trial_energy",
    "_with_every_block",
    "optimize_gs_ad_c4v_charged",
]

_logger = logging.getLogger(__name__)
_BACKEND = "1site-c4v-charged"
_LBFGS_MEMORY = 10  # curvature pairs kept, as in the 1x1 loop
_PAIR_RTOL = 1e-8  # a pair enters when s.y > _PAIR_RTOL |s| |y| (_keep_lbfgs_pair)

# What this path does with each CTMConfig field, so that no setting is ignored
# without a name in the docs.  Data only: no code reads these sets.  A test in
# tests/test_c4v_charged_ad.py keeps them complete: they are disjoint, their
# union is every field, and each _CTM_IGNORED name is in the optimize_gs_ad
# docstring and in Path 4b of docs/guide/algorithms/ipeps_ad_paths.md.
#
# Read by the cold start, the element-wise phase, the adjoint solve or the loop.
_CTM_READ = frozenset(
    {
        "chi",
        "max_iter",
        "conv_tol",
        "adjoint_tol",
        "adjoint_maxiter",
        "adjoint_tikhonov",
    }
)
# Refused or validated by _check_charged_c4v_config; use_charged_c4v_path
# takes the path only with ctm_ad_mode=None.
_CTM_CHECKED = frozenset(
    {
        "projector_method",
        "forward_gauge",
        "ctm_conv_method",
        "projector_backward",
        "on_unconverged",
        "chi_ramp",
        "chi_auto_bump",
        "ctmrg_heuristic_increase_chi",
        "fuse_virtual_legs",
        "ctm_mixing",
        "device_mesh",
        "ctm_ad_mode",
    }
)
# A user can set these for this path, and the path does not read them.  The
# docstring and the guide name each one.
_CTM_IGNORED = frozenset(
    {
        "adjoint_solver",
        "adjoint_method",
        "adjoint_chain_rule",
        "adjoint_degen_tol",
        "adjoint_diag_shift",
        "adjoint_arnoldi_precheck",
        "adjoint_arnoldi_threshold",
        "ad_backward_method",
        "gmres_tol",
        "gmres_restart",
        "gmres_maxiter",
        "gmres_precondition",
        "ad_regularize_svd",
        "min_iter",
        "renormalize",
        "plateau_patience",
        "probe_max_iter",
        "probe_conv_tol",
    }
)
# These act only through a refused setting or on another engine; one clause
# of the docs covers them.  The comment names the setting or the engine.
_CTM_IMPLIED = frozenset(
    {
        "chi_auto_bump_eps",  # chi_auto_bump
        "chi_auto_bump_step",  # chi_auto_bump
        "chi_auto_bump_metric",  # chi_auto_bump
        "ctmrg_heuristic_increase_chi_threshold",  # ctmrg_heuristic_increase_chi
        "ctmrg_heuristic_increase_chi_step_size",  # ctmrg_heuristic_increase_chi
        "chi_max",  # chi_auto_bump, ctmrg_heuristic_increase_chi
        "qr_warmup_steps",  # projector_method="qr"
        "chi_I",  # fuse_virtual_legs=False (the split CTM)
        "rel_floor",  # engine: ctm_ad_mode="root_implicit"
        "ctm_chunk_size",  # engine: the dense fused 1x1 forward
    }
)


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


# The 8 elements of C4v, in the order of ``symmetrize_c4v``.  Each maps a face
# to the face whose leg it takes: C4 there, transpose (3, 2, 0, 1, 4), is
# {u: r, d: l, l: u, r: d} here.
_C4V_ELEMENTS = (
    {"u": "u", "d": "d", "l": "l", "r": "r"},  # E
    {"u": "r", "d": "l", "l": "u", "r": "d"},  # C4
    {"u": "d", "d": "u", "l": "r", "r": "l"},  # C2
    {"u": "l", "d": "r", "l": "d", "r": "u"},  # C4^-1
    {"u": "u", "d": "d", "l": "r", "r": "l"},  # sigma_v
    {"u": "d", "d": "u", "l": "l", "r": "r"},  # sigma_h
    {"u": "r", "d": "l", "l": "d", "r": "u"},  # sigma_d
    {"u": "l", "d": "r", "l": "u", "r": "d"},  # sigma_d'
)


def _with_every_block(A: SymmetricTensor) -> SymmetricTensor:
    """A with every block that its charges allow; a block A lacks is zero.

    Reads the blocks only: no ``todense()``.
    """
    stored = A.blocks
    zeros = SymmetricTensor.zeros(A.indices, dtype=A.dtype).blocks
    return SymmetricTensor(
        {key: stored.get(key, zero) for key, zero in zeros.items()}, A.indices
    )


def _c4v_permutations(A: SymmetricTensor) -> np.ndarray:
    """The 8 elements of C4v as permutations of A's flat block buffer.

    Row g of the (8, n) result gives, for each entry of the buffer of g(A),
    the entry of A's buffer that it holds.  The C4v projection P of a buffer
    x is then ``x[perms].mean(axis=0)``, and P is never formed as a matrix.
    An element moves each block to the key of its permuted legs, with a
    charge dualised where the two legs have different flows, and transposes
    the block.  The legs are found by label, so any leg order works.  Cost
    O(n), with no ``todense()``.  A must hold every block that its charges
    allow (``_with_every_block``), and its four faces must carry equal
    flow-weighted charges at each position (``_is_charged_c4v``).
    """
    labels = A.labels()
    flows = [idx.flow for idx in A.indices]
    sym = A.indices[0].symmetry
    treedef = jax.tree_util.tree_structure(A)
    # The blocks of a buffer that holds the position of each entry.
    position = jax.tree_util.tree_unflatten(treedef, [np.arange(A._data.size)])
    rows = []
    for element in _C4V_ELEMENTS:
        axes = [labels.index(element.get(label, label)) for label in labels]
        moved = {}
        for key, block in position.blocks.items():
            new_key = tuple(
                key[a] if flows[a] == flows[i] else _dual_charge(sym, key[a])
                for i, a in enumerate(axes)
            )
            moved[new_key] = np.transpose(block, axes)
        rows.append(np.concatenate([moved[key].ravel() for key in position.blocks]))
    return np.stack(rows)


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
        A_init: a charged C4v ``SymmetricTensor``, its legs in any order.
            It need not be C4v or hold every block: the loop rebuilds it
            with every valid block and projects it to C4v.
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

    # Rebuild with every valid block, so that each C4v element permutes the
    # buffer (Review Focus 4).
    A0 = _with_every_block(A_init)
    treedef = jax.tree_util.tree_structure(A0)
    perms = jnp.asarray(_c4v_permutations(A0))
    x = jnp.mean(A0._data[perms], axis=0)
    x_norm = float(jnp.linalg.norm(x))
    if x_norm == 0.0:
        raise ValueError("charged C4v AD: A_init has no C4v component")
    x = x / x_norm

    def site(x_):
        y = jnp.mean(x_[perms], axis=0)
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
