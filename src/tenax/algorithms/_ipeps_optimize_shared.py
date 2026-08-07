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

``ipeps_optimize`` re-exports every name defined here, so
``from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor`` keeps
working for the tests and benchmarks that already use that path.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

__all__ = [
    "_build_optimizer",
    "_converged_outer",
    "_grad_l2_norm",
    "_log_ad_converged",
    "_normalize_params",
    "_should_accept_best",
    "_use_line_search",
    "_warn_implicit_ad_variational_caveat",
    "_wrap_as_dense_tensor",
]


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
