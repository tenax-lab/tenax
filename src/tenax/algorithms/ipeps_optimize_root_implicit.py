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

    if config.gs_num_steps == 0:
        A_t, env = _final_env(params)
        return A_t, env, float(compute_energy_ctm_tensor(A_t, env, gate, d_phys))

    optimizer = _build_optimizer(config)
    use_cg = optimizer is None
    opt_state = None if optimizer is None else optimizer.init(params)
    best_energy = float("inf")
    best_params = params
    prev_energy = float("inf")
    nonfinite_grad_steps = 0

    for step in range(config.gs_num_steps):
        energy_val, grads = _energy_and_grad(params)
        n_nonfinite = int(jnp.sum(~jnp.isfinite(grads)))
        if n_nonfinite:
            nonfinite_grad_steps += 1
            warnings.warn(
                f"ctm_ad_mode='root_implicit': step {step} produced "
                f"{n_nonfinite} non-finite gradient entries, masked to zero so "
                "the best-so-far state survives. The step is a no-op, so "
                "apparent convergence here is not convergence. Check "
                "usable_rank -- this is the #772 failure shape.",
                RuntimeWarning,
                stacklevel=2,
            )
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        E = float(jnp.real(energy_val))

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
        warnings.warn(
            f"ctm_ad_mode='root_implicit': {nonfinite_grad_steps} of "
            f"{config.gs_num_steps} optimizer steps made no progress because "
            "their gradient was masked. The reported energy is from the best "
            "state actually reached, not from a converged optimization.",
            RuntimeWarning,
            stacklevel=2,
        )

    A_opt, env = _final_env(best_params)
    return A_opt, env, float(compute_energy_ctm_tensor(A_opt, env, gate, d_phys))
