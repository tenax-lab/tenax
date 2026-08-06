"""Ground-state optimization driven by root implicit AD (#715).

Wires the root-implicit ``*_energy_and_grad`` entry points into
``optimize_gs_ad`` behind ``CTMConfig.ctm_ad_mode="root_implicit"``.

Root implicit differentiation (arXiv:2607.15030) characterises the converged
environment as the root of an *algebraic* equation ``F(y, p) = 0`` and gets the
gradient from a single un-nested linear solve.  No SVD or eigh backward appears
anywhere in the gradient path, which is the point: #566 (block-sparse AD
compile wall) and #687 (symmetric CTM AD accuracy floor) are specifically those
VJPs inside the compiled backward.

.. important::
    **This is not a speed optimization.**  Per #715's own cost/benefit, the
    crossover where implicit beats fixed-point appears only at large ``D²χ``;
    at ``D = 3-4``, ``chi = 16-24`` we are on the wrong side of it, and our
    fixed-point baseline is *better* than the one the paper benchmarks against
    (they pay a nested Sylvester solve for sparse SVDs; we take a thin SVD with
    JAX's closed-form VJP).  The reasons to use this path are **stability**
    (it removes the degenerate-singular-value failure class outright) and
    block-sparse accuracy/compile cost.

Architectural note -- why this is a separate optimizer loop rather than a
backward swap.  The ``*_energy_and_grad`` entry points replace the *whole*
``value_and_grad``, not just the backward: each runs its own CTM convergence
(``max_iter``, ``conv_tol``, ``min_iter``, ``polish_steps``) and takes **no
warm-start environment**.  So every optimizer step reconverges CTM from cold,
and the knobs that ride on a warm-started environment -- chi auto-bump,
chi ramping, environment checkpointing -- have nothing to attach to.  Those
hard-error in :func:`validate_root_implicit_config` rather than silently
no-op; this codebase has been burned repeatedly by silent inapplicability
(#723, #760, #762).
"""

from __future__ import annotations

__all__ = [
    "optimize_gs_ad_root_implicit",
    "root_implicit_variant",
    "validate_root_implicit_config",
]

import logging
import time as _time
import warnings

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.core.tensor import DenseTensor, Tensor

_logger = logging.getLogger(__name__)

# ``ctm_ad_mode`` values handled here.  "root_implicit" auto-selects among the
# *dense* variants from the unit cell; the symmetric variant must be asked for
# by name so nobody lands on the 8.4 GB GMRES path (#731) by accident.  C4v
# (Phase 0) is deliberately not exposed -- it exists to validate the
# characteristic equations, not to run production.
ROOT_IMPLICIT_MODES = ("root_implicit", "root_implicit_symmetric")


def root_implicit_variant(config: iPEPSConfig) -> str:
    """Return which root-implicit engine ``config`` selects.

    ``"asym"`` (dense 1x1), ``"cell"`` (dense multisite) or ``"symmetric"``.
    """
    mode = getattr(config.ctm, "ctm_ad_mode", None)
    if mode == "root_implicit_symmetric":
        return "symmetric"
    if mode != "root_implicit":
        raise ValueError(f"not a root-implicit mode: {mode!r}")
    return "asym" if config.unit_cell == "1x1" else "cell"


def validate_root_implicit_config(config: iPEPSConfig) -> None:
    """Reject configurations the root-implicit path cannot honour.

    Every rejection here is a knob that would otherwise be **silently
    ignored**, because the ``*_energy_and_grad`` entry points own their CTM
    convergence and accept no warm-start environment.  Failing loudly at
    config time is the same policy as ``validate_split_ctm_config``.

    ``gs_metric_precond`` is deliberately *not* in this list even though it is
    equally unhonourable here (the metric inner product needs a per-step
    environment).  It defaults to ``True``, so rejecting it would make
    ``ctm_ad_mode="root_implicit"`` unusable without an unrelated opt-out.  It
    warns and falls back instead, which is exactly what the split-CTM path does
    with the same knob for the same reason.
    """
    ctm = config.ctm
    unsupported: list[tuple[str, str]] = []

    if ctm.chi_auto_bump:
        unsupported.append(
            ("chi_auto_bump", "the entry points converge their own environment")
        )
    if getattr(ctm, "ctmrg_heuristic_increase_chi", False):
        unsupported.append(
            (
                "ctmrg_heuristic_increase_chi",
                "in-CTM chi growth needs the forward loop this path replaces",
            )
        )
    if getattr(ctm, "chi_ramp", None) is not None:
        unsupported.append(("chi_ramp", "chi is fixed for the lifetime of the solve"))
    if not getattr(ctm, "fuse_virtual_legs", True):
        unsupported.append(
            (
                "fuse_virtual_legs=False",
                "root implicit AD is defined on the fused double layer",
            )
        )
    if config.gs_checkpoint_path is not None:
        unsupported.append(
            (
                "gs_checkpoint_path",
                "there is no warm-start environment to checkpoint",
            )
        )
    if config.cg_gates is not None:
        unsupported.append(
            ("cg_gates", "coarse-grained gates have no root-implicit objective")
        )
    # #792.  Both are opt-in -- you only get them by asking -- so an
    # unsatisfiable request is an error here.  ``gs_line_search`` is NOT in
    # this list: it is effectively default-on (``gs_optimizer`` defaults to
    # 'lbfgs' and ``gs_line_search=None`` means auto-on), so refusing it would
    # reject the path's own default configuration.  It warns instead, in
    # ``optimize_gs_ad_root_implicit``, the same way ``gs_metric_precond``
    # does.
    if config.gs_optimizer.lower() == "cg":
        unsupported.append(
            (
                "gs_optimizer='cg'",
                "the loop would run plain gradient descent under that name -- "
                "no retained previous gradient, no Polak-Ribiere coefficient, "
                "no conjugate direction. Use 'lbfgs' or 'adam'",
            )
        )
    if getattr(config, "gs_c4v", False):
        unsupported.append(
            (
                "gs_c4v",
                "nothing on this path projects into the C4v basis or "
                "optimises reduced coefficients, so the first update can "
                "leave the requested symmetry sector",
            )
        )
    if unsupported:
        lines = "\n".join(f"  - {k}: {why}" for k, why in unsupported)
        raise NotImplementedError(
            "ctm_ad_mode='root_implicit' does not support the following "
            f"settings, which would otherwise be silently ignored:\n{lines}\n"
            "Unset them, or use the default (fixed-point implicit) AD path."
        )


def _initial_tensor(hamiltonian_gate, A_init, config: iPEPSConfig, gate, d_phys):
    """Resolve the starting site tensor (explicit / simple-update / random)."""
    from tenax.algorithms.ipeps import ipeps
    from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor

    if A_init is not None:
        return A_init if isinstance(A_init, Tensor) else _wrap_as_dense_tensor(A_init)
    if config.su_init:
        _E, tensors, _envs = ipeps(gate, None, config)
        return tensors[0] if isinstance(tensors, (list, tuple)) else tensors
    D = config.max_bond_dim
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    data = jax.random.normal(k1, (D, D, D, D, d_phys)) + 1j * jax.random.normal(
        k2, (D, D, D, D, d_phys)
    )
    return _wrap_as_dense_tensor(data)


def optimize_gs_ad_root_implicit(
    hamiltonian_gate,
    A_init,
    config: iPEPSConfig,
):
    """Optimize a ground state using root-implicit gradients.

    Returns ``(A_opt, env, E_gs)``, matching ``optimize_gs_ad``.

    The returned environment is converged *separately* by the standard forward
    CTM after optimization finishes: the root-implicit entry points return
    ``(energy, grad)`` and do not surface the environment they built
    internally.  It is therefore the environment of ``A_opt``, converged with
    the ordinary machinery -- not a by-product of the last gradient step.
    """
    import optax

    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
    from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
    from tenax.algorithms.ipeps_optimize import (
        _build_optimizer,
        _converged_outer,
        _grad_l2_norm,
        _log_ad_converged,
        _normalize_params,
        _should_accept_best,
        _use_line_search,
        _warn_implicit_ad_variational_caveat,
    )

    validate_root_implicit_config(config)
    if getattr(config, "gs_metric_precond", False):
        # Default-on knob, so this warns rather than raising (see
        # validate_root_implicit_config).  Mirrors the split-CTM fallback.
        warnings.warn(
            "gs_metric_precond=True is not supported on the root-implicit AD "
            "path (ctm_ad_mode='root_implicit'): the metric inner product "
            "needs a per-step CTM environment, which the *_energy_and_grad "
            "entry points do not expose. Falling back to non-preconditioned "
            "optimization.",
            UserWarning,
            stacklevel=2,
        )
    if _use_line_search(config):
        # #792.  Default-on for lbfgs/cg, so this warns rather than raising --
        # refusing it would reject the path's own default configuration.
        #
        # No line search runs here: ``_build_optimizer`` returns a plain
        # ``optax.chain(scale_by_lbfgs, clip_by_global_norm, scale(-1))`` and
        # ``value_fn=`` is consumed only by ``optax.lbfgs(linesearch=...)``.
        # Passing it to that chain does nothing.
        warnings.warn(
            "gs_line_search is enabled (explicitly, or by default for "
            f"gs_optimizer={config.gs_optimizer!r}) but no line search runs on "
            "the root-implicit AD path (ctm_ad_mode='root_implicit'): the "
            "optimizer chain has no line-search transform, so gs_line_search, "
            "gs_line_search_method, gs_line_search_max_steps and gs_hz_max_iter "
            "are all no-ops. Steps are unconditional quasi-Newton steps and an "
            "energy-increasing one can be accepted. The best-so-far state is "
            "still tracked and returned. Set gs_line_search=False to silence "
            "this.",
            UserWarning,
            stacklevel=2,
        )
    variant = root_implicit_variant(config)

    if variant == "cell":
        raise NotImplementedError(
            "ctm_ad_mode='root_implicit' currently wires the dense 1x1 "
            "(asymmetric) engine only; got unit_cell="
            f"{config.unit_cell!r}. The multisite engine "
            "(cell_root_implicit_energy_and_grad, #715 Phase 2) is built and "
            "tested but not yet wired here -- its shifted-cell index tables "
            "are the whole risk of that phase and a wrong shift yields a "
            "silently wrong gradient, so it gets its own increment. Use "
            "unit_cell='1x1'."
        )
    if variant == "symmetric":
        raise NotImplementedError(
            "ctm_ad_mode='root_implicit_symmetric' is not wired yet: the "
            "symmetric adjoint peaks at 8.4 GB in the GMRES solve (#731), "
            "which needs resolving before it can be driven from an optimizer "
            "loop. The engine itself (sym_root_implicit_energy_and_grad) is "
            "built and tested."
        )

    _warn_implicit_ad_variational_caveat(config, path="1-site dense root-implicit")

    from tenax.algorithms._ctm_root_implicit_asym import (
        asym_root_implicit_energy_and_grad,
    )

    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.asarray(hamiltonian_gate)
    )
    d_phys = gate.shape[0]

    A = _initial_tensor(hamiltonian_gate, A_init, config, gate, d_phys)
    if not isinstance(A, DenseTensor):
        raise TypeError(
            "ctm_ad_mode='root_implicit' is dense-only (#715 Phase 3 covers "
            f"SymmetricTensor); got {type(A).__name__}."
        )
    A = A * (1.0 / (A.norm() + 1e-10))
    indices = A.indices
    ctm_cfg = config.ctm

    def _energy_and_grad(params):
        A_t = DenseTensor(params, indices)
        return asym_root_implicit_energy_and_grad(
            A_t,
            gate,
            chi=ctm_cfg.chi,
            max_iter=ctm_cfg.max_iter,
            conv_tol=ctm_cfg.conv_tol,
            min_iter=ctm_cfg.min_iter,
            rel_floor=ctm_cfg.rel_floor,
        )

    def _final_env(params):
        """Converge the reported environment with the ordinary forward CTM."""
        A_t = DenseTensor(params, indices)
        envs, _info = python_loop_ctm_converge(
            {(0, 0): A_t}, SINGLE_SITE_NEIGHBORS, **ctm_converge_kwargs(ctm_cfg)
        )
        return A_t, envs[(0, 0)]

    params = A.todense()

    # #792: return_history was accepted and dropped, so callers of the
    # documented 1x1 history API got a 3-tuple and lost the trajectory.  The
    # loop already computes every field; this only records them.  Same dict
    # shape as the 1-site tensor path in ipeps_optimize.
    history_energies: list[float] = []
    history_step_times: list[float] = []
    jit_compile_time = 0.0
    first_step = True

    def _with_history(A_t, env, E, *, converged):
        if not config.return_history:
            return A_t, env, E
        return (
            A_t,
            env,
            E,
            {
                "energies": history_energies,
                "step_times": history_step_times,
                "jit_compile_time": jit_compile_time,
                "num_steps": len(history_energies),
                "converged": converged,
            },
        )

    if config.gs_num_steps == 0:
        A_t, env = _final_env(params)
        return _with_history(
            A_t,
            env,
            float(compute_energy_ctm_tensor(A_t, env, gate, d_phys)),
            converged=False,
        )

    optimizer = _build_optimizer(config)
    use_cg = optimizer is None
    opt_state = None if optimizer is None else optimizer.init(params)
    best_energy = float("inf")
    best_params = params
    prev_energy = float("inf")
    nonfinite_grad_steps = 0
    fully_masked_steps = 0
    steps_run = 0

    converged_flag = False
    for step in range(config.gs_num_steps):
        steps_run = step + 1
        _step_t0 = _time.perf_counter()
        energy_val, grads = _energy_and_grad(params)
        n_nonfinite = int(jnp.sum(~jnp.isfinite(grads)))
        if n_nonfinite:
            nonfinite_grad_steps += 1
            # Masking every entry really is a no-op; masking *some* of them is
            # not, and must not be reported as one.  The surviving components
            # still drive a full optimizer update -- and they came out of the
            # same solve that produced the non-finite ones, so that step moves
            # the state along a direction already known to be contaminated.
            # That is a worse situation than a stalled step, not a milder one.
            if n_nonfinite == grads.size:
                fully_masked_steps += 1
                detail = (
                    "every entry was non-finite, so the step is a no-op and "
                    "apparent convergence here is not convergence"
                )
            else:
                detail = (
                    f"the remaining {grads.size - n_nonfinite} finite entries "
                    "still drive a full update, so this step is NOT a no-op -- "
                    "it moves the state along a gradient already known to be "
                    "contaminated"
                )
            warnings.warn(
                f"ctm_ad_mode='root_implicit': step {step} produced "
                f"{n_nonfinite} of {grads.size} non-finite gradient entries, "
                f"masked to zero so the best-so-far state survives; {detail}. "
                "Check usable_rank -- this is the #772 failure shape.",
                RuntimeWarning,
                stacklevel=2,
            )
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        E = float(jnp.real(energy_val))
        if config.return_history:
            _step_dt = float(_time.perf_counter() - _step_t0)
            if first_step:
                # The first evaluation is compile-dominated even though this
                # path has no top-level ``jax.jit``: the root solve and its
                # adjoint are jitted internally, so step 0 pays their lazy
                # compilation.  Charging it to ``step_times`` would hand the
                # two-point compile/steady-state fit a cold sample as if it
                # were warm.  Diverting it here is also what keeps
                # ``len(step_times) == max(num_steps - 1, 0)`` -- the shape
                # invariant asserted for the other AD paths in
                # ``tests/test_ipeps_ad_history.py``.
                jit_compile_time = _step_dt
                first_step = False
            else:
                history_step_times.append(_step_dt)
            history_energies.append(E)

        if _should_accept_best(
            current_best=best_energy,
            candidate=E,
            floor=getattr(config, "gs_energy_floor", None),
        ):
            best_energy = E
            best_params = params

        if config.gs_verbose:
            print(
                f"[iPEPS-AD:root_implicit] step {step + 1}/{config.gs_num_steps} "
                f"E={E:.12f}",
                flush=True,
            )

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
                    "root_implicit",
                    step,
                    delta_energy,
                    config.gs_conv_tol,
                    grad_norm=grad_norm_val,
                    grad_norm_tol=config.gs_grad_norm_tol,
                    criterion=config.gs_conv_criterion,
                )
            # Masking a non-finite gradient to zero deflates both criteria: a
            # fully masked step drives ``grad_norm_val`` to exactly 0.0, and it
            # is a no-op, so the *next* step re-evaluates identical params and
            # sees delta_energy == 0.0.  Either can fire ``_converged_outer``
            # on a run this loop has already warned is "not ... a converged
            # optimization".  The break itself is pre-existing behaviour and is
            # left alone here (#812); what must not happen is publishing that
            # as converged=True to a consumer that cannot see the warning.
            converged_flag = nonfinite_grad_steps == 0
            break

        if use_cg:
            params = _normalize_params(params - config.gs_learning_rate * grads)
        else:
            assert optimizer is not None
            if config.gs_optimizer.lower() == "lbfgs":
                # optax's L-BFGS line search re-evaluates the objective, which
                # on this path means a full root-implicit solve per probe.
                # Pass only the value function, never value_and_grad.
                updates, opt_state = optimizer.update(
                    grads,
                    opt_state,
                    params,
                    value=energy_val,
                    grad=grads,
                    value_fn=lambda p: _energy_and_grad(p)[0],
                )
            else:
                updates, opt_state = optimizer.update(grads, opt_state, params)
            params = _normalize_params(optax.apply_updates(params, updates))

    if nonfinite_grad_steps:
        partly_masked = nonfinite_grad_steps - fully_masked_steps
        warnings.warn(
            f"ctm_ad_mode='root_implicit': {nonfinite_grad_steps} of "
            f"{steps_run} optimizer steps had a non-finite gradient "
            f"({fully_masked_steps} fully masked, so genuinely no-ops; "
            f"{partly_masked} only partly masked, which still stepped -- on a "
            "gradient known to be contaminated). The reported energy is from "
            "the best state actually reached, not from a converged "
            "optimization.",
            RuntimeWarning,
            stacklevel=2,
        )

    A_opt, env = _final_env(best_params)
    return _with_history(
        A_opt,
        env,
        float(compute_energy_ctm_tensor(A_opt, env, gate, d_phys)),
        converged=converged_flag,
    )
