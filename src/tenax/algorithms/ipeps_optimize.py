"""AD-based ground state optimization for iPEPS.

Extracts optimize_gs_ad and related helpers from ipeps.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace as _replace

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps_ad_policy import (
    build_ad_ctm_config,
    resolve_projector_backward,
    use_reference_c4v_path,
)
from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor

_logger = logging.getLogger(__name__)


def _resolve_projector_backward(config: iPEPSConfig) -> iPEPSConfig:
    """Compatibility wrapper around the shared AD policy helper."""
    return resolve_projector_backward(config, logger=_logger)


def _normalize_stall_recovery(config, *, unit_cell: str):
    """Auto-default ``gs_stall_recovery`` based on unit cell when unset.

    The 1-site C4v production path requires the noise kick to break out of the
    SU-init plateau (gradient norms ~1e-10 trip ``gs_conv_tol`` before the first
    real step).  The 2-site path's larger parameter space interacts
    pathologically with non-variational CTM regions under noise; see issue #298.
    """
    from dataclasses import replace

    if config.gs_stall_recovery is not None:
        return config
    default = "noise" if unit_cell == "1x1" else "reset"
    return replace(config, gs_stall_recovery=default)


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


def _tree_dot(a, b) -> float:
    """Compute real dot product between two pytrees of arrays."""
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    return float(
        jnp.real(sum(jnp.sum(jnp.conj(la) * lb) for la, lb in zip(leaves_a, leaves_b)))
    )


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
) -> None:
    delta_str = "N/A" if not math.isfinite(delta_energy) else f"{delta_energy:.3e}"
    print(
        f"[iPEPS-AD:{backend}] step {step + 1}/{num_steps} "
        f"E={energy:.10f} dE={delta_str} E_best={best_energy:.10f}",
        flush=True,
    )


def _log_ad_converged(backend: str, step: int, delta_energy: float, tol: float) -> None:
    print(
        f"[iPEPS-AD:{backend}] converged at step {step + 1} "
        f"(dE={delta_energy:.3e} < tol={tol:.3e})",
        flush=True,
    )


def optimize_gs_ad_chi_schedule(
    hamiltonian_gate: jax.Array | Tensor,
    A_init: jax.Array | Tensor | tuple | None,
    config: iPEPSConfig,
    chi_schedule: list[tuple[int, int]],
):
    """AD optimization with chi-ramping schedule.

    Runs ``optimize_gs_ad`` at each chi level in sequence, using the
    optimized tensor from the previous level as initialization for the
    next.  This avoids cold-starting at large chi and gives much better
    convergence.

    Reference: Zhang, Yang & Corboz, arXiv:2505.00494 (2025).

    Args:
        hamiltonian_gate: 2-site Hamiltonian of shape ``(d, d, d, d)``.
        A_init:           Initial site tensor(s) or ``None``.
        config:           Base iPEPSConfig (chi and gs_num_steps will be
                          overridden per schedule entry).
        chi_schedule:     List of ``(chi, num_steps)`` pairs, e.g.
                          ``[(8, 100), (16, 50), (32, 30)]``.

    Returns:
        Same as ``optimize_gs_ad`` at the final chi level.
    """
    from dataclasses import replace

    result = None
    current_init = A_init

    for chi, num_steps in chi_schedule:
        ctm_cfg = replace(config.ctm, chi=chi)
        step_cfg = replace(config, ctm=ctm_cfg, gs_num_steps=num_steps)

        if config.gs_verbose:
            print(
                f"[chi-ramp] chi={chi}, {num_steps} steps",
                flush=True,
            )

        result = optimize_gs_ad(hamiltonian_gate, current_init, step_cfg)

        # Extract optimized tensor for next level
        if config.unit_cell == "2site":
            (A_opt, B_opt), _, E = result
            current_init = (A_opt, B_opt)
        else:
            A_opt, _, E = result
            current_init = A_opt

        if config.gs_verbose:
            print(f"[chi-ramp] chi={chi} done, E={E:.10f}", flush=True)

    return result


def optimize_gs_ad(
    hamiltonian_gate: jax.Array | Tensor,
    A_init: jax.Array | Tensor | tuple | None,
    config: iPEPSConfig,
):
    """AD-based ground state optimization of iPEPS.

    Uses automatic differentiation through the CTM fixed-point equation
    (Francuz et al. PRR 7, 013237) to compute exact gradients of the
    energy with respect to the site tensor(s), then optimizes with optax.

    Supports both 1-site (``unit_cell="1x1"``) and 2-site
    (``unit_cell="2site"``) unit cells.  Accepts dense ``jax.Array`` or
    Tensor-protocol objects (``DenseTensor``, ``SymmetricTensor``).

    Args:
        hamiltonian_gate: 2-site Hamiltonian of shape ``(d, d, d, d)``.
        A_init:           Initial site tensor ``(D, D, D, D, d)`` for 1-site,
                          ``(A, B)`` tuple for 2-site, or ``None`` for random
                          initialization.  When ``None`` and
                          ``config.su_init`` is ``True``, the tensor(s) are
                          initialized via simple update (``ipeps()``).
        config:           iPEPSConfig with AD optimization settings.

    Returns:
        For 1-site dense:  ``(A_opt, env, E_gs)``
        For 1-site Tensor: ``(A_opt, env, E_gs)`` where A_opt is Tensor, env is CTMTensorEnv
        For 2-site: ``((A_opt, B_opt), (env_A, env_B), E_gs)``
    """
    if config.gs_log_interval < 1:
        raise ValueError(f"gs_log_interval must be >= 1, got {config.gs_log_interval}")
    if config.gs_num_steps < 0:
        raise ValueError(f"gs_num_steps must be >= 0, got {config.gs_num_steps}")

    # Auto-promote projector_backward before dispatch so every downstream
    # helper (1-site, 2-site, reference-C4v) sees the same CTM config.
    # Mirrors the forward_gauge "qr" -> "phase" promotion further below.
    # See docs/plans/2026-04-13-multisite-c4v-reference-ad-plan.md Task 8.
    config = _resolve_projector_backward(config)

    if config.unit_cell == "2site":
        return _optimize_gs_ad_2site(hamiltonian_gate, A_init, config)
    if _use_reference_c4v_path(config):
        return _optimize_gs_ad_tensor_reference_c4v(hamiltonian_gate, A_init, config)

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
        else:
            key = jax.random.PRNGKey(0)
            k1, k2 = jax.random.split(key)
            A_data = jax.random.normal(
                k1, (D, D, D, D, d_phys)
            ) + 1j * jax.random.normal(k2, (D, D, D, D, d_phys))
            A_init = _wrap_as_dense_tensor(A_data)

    return _optimize_gs_ad_tensor(hamiltonian_gate, A_init, config)


def _use_reference_c4v_path(config: iPEPSConfig) -> bool:
    """Compatibility wrapper around the shared AD policy helper."""
    return use_reference_c4v_path(config)


def _optimize_gs_ad_tensor_reference_c4v(
    hamiltonian_gate: jax.Array | Tensor,
    A_init: jax.Array | Tensor | None,
    config: iPEPSConfig,
):
    """Reference-mode dense C4v path with implicit-AD CTM backward."""
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
        E = float(energy_val)
        if _should_accept_best(
            current_best=best_energy,
            candidate=E,
            floor=getattr(config, "gs_energy_floor", None),
        ):
            best_energy = E
            best_params = params

    final_energy, (final_env, final_A) = _loss_fn(best_params)
    return final_A, final_env, float(final_energy)


def _optimize_gs_ad_tensor(
    hamiltonian_gate: jax.Array,
    A_init: Tensor,
    config: iPEPSConfig,
):
    """AD-based ground state optimization for Tensor-protocol iPEPS (1-site).

    Uses ``ctm_tensor_converge`` with implicit differentiation through
    the standard Tensor-protocol CTM.
    """
    config = _normalize_stall_recovery(config, unit_cell="1x1")
    import optax

    from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit, ctm_energy_implicit
    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor
    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
    from tenax.algorithms.ad_utils import CTMRGGradientError

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

    use_explicit = not config.gs_implicit_ad
    explicit_steps = config.gs_explicit_ad_steps
    explicit_warmup = config.gs_explicit_ad_warmup

    def _ctm_energy_fn(site_tensors):
        """Dispatch to implicit or explicit CTM energy."""
        env_init = _env_cache.get("envs", None)
        if use_explicit:
            return ctm_energy_explicit(
                site_tensors,
                SINGLE_SITE_NEIGHBORS,
                gate,
                chi=ctm_cfg.chi,
                warmup_steps=explicit_warmup,
                backprop_steps=explicit_steps,
                projector_method=ctm_cfg.projector_method,
                renormalize=ctm_cfg.renormalize,
                projector_backward=ctm_cfg.projector_backward,
                env_init=env_init,
            )
        else:
            return ctm_energy_implicit(
                site_tensors,
                SINGLE_SITE_NEIGHBORS,
                gate,
                chi=ctm_cfg.chi,
                max_iter=ctm_cfg.max_iter,
                conv_tol=ctm_cfg.conv_tol,
                projector_method=ctm_cfg.projector_method,
                renormalize=ctm_cfg.renormalize,
                projector_backward=ctm_cfg.projector_backward,
                qr_warmup_steps=ctm_cfg.qr_warmup_steps,
                chi_ramp=ctm_cfg.chi_ramp,
                env_init=env_init,
                forward_gauge=ctm_cfg.forward_gauge,
                conv_method=ctm_cfg.ctm_conv_method,
                min_iter=ctm_cfg.min_iter,
                gmres_tol=ctm_cfg.gmres_tol,
                gmres_maxiter=ctm_cfg.gmres_restart,
                gmres_restart=ctm_cfg.gmres_restart,
                arnoldi_precheck=False,
            )

    def loss_fn(params):
        if use_c4v:
            A_data = c4v_tensor_from_coeffs(params, c4v_basis, tensor_shape)
            A_norm_data = A_data / (jnp.linalg.norm(A_data) + 1e-10)
            A_norm = DenseTensor(A_norm_data, A.indices)
        else:
            A_norm = params * (1.0 / (params.norm() + 1e-10))
        site_tensors = {(0, 0): A_norm}
        energy = _ctm_energy_fn(site_tensors)
        return energy

    def _update_env_cache(params):
        """Re-run forward CTM (no grad) to warm-start next step."""
        if use_c4v:
            A_data = c4v_tensor_from_coeffs(params, c4v_basis, tensor_shape)
            A_data = A_data / (jnp.linalg.norm(A_data) + 1e-10)
            A_norm = DenseTensor(A_data, A.indices)
        else:
            A_norm = params * (1.0 / (params.norm() + 1e-10))
        site_tensors = {(0, 0): A_norm}
        envs, _ = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            chi=ctm_cfg.chi,
            max_iter=ctm_cfg.max_iter,
            conv_tol=ctm_cfg.conv_tol,
            renormalize=ctm_cfg.renormalize,
            projector_method=ctm_cfg.projector_method,
            qr_warmup_steps=ctm_cfg.qr_warmup_steps,
            projector_backward=ctm_cfg.projector_backward,
            chi_ramp=ctm_cfg.chi_ramp,
            env_init=_env_cache.get("envs", None),
        )
        _env_cache["envs"] = envs

    is_metric_lbfgs = (
        config.gs_metric_precond and config.gs_optimizer.lower() == "lbfgs"
    )
    params = c4v_coeffs if use_c4v else A
    optimizer = None if is_metric_lbfgs else _build_optimizer(config)
    opt_state = optimizer.init(params) if optimizer is not None else None
    use_ls = _use_line_search(config)
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

    from tenax.algorithms.ad_utils import (
        _wrap_tensor,
    )

    def loss_fn_fwd(p):
        """Forward-only loss for line search — warm-starts CTM from env cache."""
        if use_c4v:
            A_data = c4v_tensor_from_coeffs(p, c4v_basis, tensor_shape)
            A_norm_data = A_data / (jnp.linalg.norm(A_data) + 1e-10)
            A_norm = DenseTensor(A_norm_data, A.indices)
        else:
            A_norm = p * (1.0 / (p.norm() + 1e-10))
        site_tensors = {(0, 0): A_norm}
        envs, _ = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            chi=ctm_cfg.chi,
            max_iter=ctm_cfg.max_iter,
            conv_tol=ctm_cfg.conv_tol,
            renormalize=ctm_cfg.renormalize,
            projector_method=ctm_cfg.projector_method,
            qr_warmup_steps=ctm_cfg.qr_warmup_steps,
            projector_backward=ctm_cfg.projector_backward,
            chi_ramp=ctm_cfg.chi_ramp,
            env_init=_env_cache.get("envs", None),
        )
        return float(compute_energy_ctm_tensor(A_norm, envs[(0, 0)], gate, d_phys))

    stall_count = 0  # noise recovery: consecutive line search failures

    # CTM conv_tol schedule: update ctm_cfg when tolerance changes
    _conv_tol_schedule = config.gs_ctm_conv_tol_schedule
    _current_conv_tol = ctm_cfg.conv_tol

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

    for step in range(config.gs_num_steps):
        # Update conv_tol if schedule is active
        if _conv_tol_schedule is not None:
            new_tol = _get_scheduled_conv_tol(step, config.gs_num_steps)
            if new_tol != _current_conv_tol:
                _current_conv_tol = new_tol
                ctm_cfg = _replace(ctm_cfg, conv_tol=new_tol)
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
                params = best_params
                _env_cache.update(best_env_cache)
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

        # Update env cache for warm-starting next step
        _update_env_cache(params)

        if _should_accept_best(
            current_best=best_energy,
            candidate=energy_float,
            floor=config.gs_energy_floor,
        ):
            best_energy = energy_float
            best_params = params
            best_env_cache = dict(_env_cache)  # snapshot for warm-start (#317)

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
            )
            logged = True

        prev_energy = energy_float

        if delta_energy < config.gs_conv_tol:
            if config.gs_verbose:
                if not logged:
                    _log_ad_step(
                        "1site-tensor",
                        step,
                        config.gs_num_steps,
                        energy_float,
                        delta_energy,
                        best_energy,
                    )
                _log_ad_converged(
                    "1site-tensor", step, delta_energy, config.gs_conv_tol
                )
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
            if config.gs_line_search_method == "hager_zhang":
                from tenax.algorithms._line_search import hager_zhang_line_search

                slope = _tree_dot(grads, direction)
                if slope >= 0:
                    direction = jax.tree.map(lambda g: -g, grads)
                    slope = -_tree_dot(grads, grads)
                    if is_metric_lbfgs:
                        lbfgs_history.clear()

                def _phi(alpha):
                    trial = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    return loss_fn_fwd(trial)

                def _dphi(alpha):
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
                )
                if f_alpha < energy_float:
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
                if new_energy < energy_float:
                    stall_count = 0
                else:
                    stall_count += 1

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
                # variPEPS-style reset: clear L-BFGS / CG state so the next
                # step is plain steepest descent (or preconditioned steepest
                # descent) from the CURRENT iterate.  Do NOT roll back
                # params — issue #298's trajectory study shows "do nothing
                # and continue" is strictly better than rollback for the
                # L-BFGS + Hager-Zhang + metric-precond path.
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_A_flat = None
                    prev_grad_flat = None
                # Optax-backed L-BFGS stores curvature history in opt_state,
                # not in lbfgs_history.  Reinitialize it so the next step
                # really is steepest descent (reviewer feedback on #298).
                if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                    opt_state = optimizer.init(params)
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD] stall #{stall_count}, "
                        f"reset L-BFGS history (no rollback)",
                        flush=True,
                    )
        else:
            params = optax.apply_updates(params, direction)
            if not use_c4v:
                params = params * (1.0 / (params.norm() + 1e-10))

    # Re-evaluate both final A and best_A with fully converged fresh CTM.
    # In-loop energies use warm-started CTM that can produce unphysical values
    # (non-variational at finite chi), so we compare fresh evaluations only.
    # Match in-loop CTM tolerances (#317) by reusing ctm_cfg directly.

    def _eval_fresh(p, env_init=None):
        """Evaluate energy with fully converged fresh CTM."""
        if use_c4v:
            A_data = c4v_tensor_from_coeffs(p, c4v_basis, tensor_shape)
            A_data = A_data / (jnp.linalg.norm(A_data) + 1e-10)
            A_t = DenseTensor(A_data, A.indices)
        else:
            A_t = p * (1.0 / (p.norm() + 1e-10))
        envs, _ = python_loop_ctm_converge(
            {(0, 0): A_t},
            SINGLE_SITE_NEIGHBORS,
            chi=ctm_cfg.chi,
            max_iter=ctm_cfg.max_iter,
            min_iter=ctm_cfg.min_iter,
            conv_tol=ctm_cfg.conv_tol,
            conv_method=ctm_cfg.ctm_conv_method,
            renormalize=ctm_cfg.renormalize,
            projector_method=ctm_cfg.projector_method,
            qr_warmup_steps=ctm_cfg.qr_warmup_steps,
            projector_backward=ctm_cfg.projector_backward,
            chi_ramp=ctm_cfg.chi_ramp,
            env_init=env_init,
        )
        env_ = envs[(0, 0)]
        E_ = float(compute_energy_ctm_tensor(A_t, env_, gate, d_phys))
        return A_t, env_, E_

    A_final, env_final, E_final = _eval_fresh(params, _env_cache.get("envs", None))

    if best_params is not params:
        _, env_best, E_best_fresh = _eval_fresh(
            best_params, best_env_cache.get("envs", None)
        )
    else:
        E_best_fresh = E_final

    if E_final <= E_best_fresh:
        env, E_gs = env_final, E_final
    else:
        A_final, _, _ = _eval_fresh(best_params, best_env_cache.get("envs", None))
        env, E_gs = env_best, E_best_fresh
    if config.gs_verbose:
        print(f"[iPEPS-AD:1site-tensor] final E={E_gs:.10f}", flush=True)

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
                "This is variational at chi >= 16 for generic models but "
                "can be slower than C4v or 1-site optimization. For "
                "antiferromagnetic bipartite models, consider gs_c4v=True "
                "or 1-site with sublattice_rotate_gate().",
                stacklevel=2,
            )
    import optax

    from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit, ctm_energy_implicit
    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor import (
        compute_energy_ctm_tensor_2site,
    )
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    from tenax.algorithms.ad_utils import (
        CTMRGGradientError,
        _wrap_tensor,
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

    def _ctm_energy_fn_2s(site_tensors):
        """Dispatch to implicit or explicit CTM energy for 2-site."""
        env_init = _env_cache_2s.get("envs", None)
        if use_explicit:
            return ctm_energy_explicit(
                site_tensors,
                CHECKERBOARD_NEIGHBORS,
                gate,
                chi=ctm_cfg_2s.chi,
                warmup_steps=explicit_warmup,
                backprop_steps=explicit_steps,
                projector_method=ctm_cfg_2s.projector_method,
                renormalize=ctm_cfg_2s.renormalize,
                projector_backward=ctm_cfg_2s.projector_backward,
                env_init=env_init,
                energy_fn=_energy_fn_2site,
            )
        else:
            return ctm_energy_implicit(
                site_tensors,
                CHECKERBOARD_NEIGHBORS,
                gate,
                chi=ctm_cfg_2s.chi,
                max_iter=ctm_cfg_2s.max_iter,
                conv_tol=ctm_cfg_2s.conv_tol,
                projector_method=ctm_cfg_2s.projector_method,
                renormalize=ctm_cfg_2s.renormalize,
                projector_backward=ctm_cfg_2s.projector_backward,
                qr_warmup_steps=ctm_cfg_2s.qr_warmup_steps,
                chi_ramp=ctm_cfg_2s.chi_ramp,
                env_init=env_init,
                forward_gauge=ctm_cfg_2s.forward_gauge,
                conv_method=ctm_cfg_2s.ctm_conv_method,
                min_iter=ctm_cfg_2s.min_iter,
                gmres_tol=ctm_cfg_2s.gmres_tol,
                gmres_maxiter=ctm_cfg_2s.gmres_restart,
                gmres_restart=ctm_cfg_2s.gmres_restart,
                energy_fn=_energy_fn_2site,
                arnoldi_precheck=False,
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
        envs, _ = python_loop_ctm_converge(
            site_tensors,
            CHECKERBOARD_NEIGHBORS,
            chi=ctm_cfg_2s.chi,
            max_iter=ctm_cfg_2s.max_iter,
            conv_tol=ctm_cfg_2s.conv_tol,
            renormalize=ctm_cfg_2s.renormalize,
            projector_method=ctm_cfg_2s.projector_method,
            qr_warmup_steps=ctm_cfg_2s.qr_warmup_steps,
            projector_backward=ctm_cfg_2s.projector_backward,
            chi_ramp=ctm_cfg_2s.chi_ramp,
            env_init=_env_cache_2s.get("envs", None),
        )
        _env_cache_2s["envs"] = envs

    params = c4v_coeffs if use_c4v else (A, B)
    is_metric_lbfgs = (
        config.gs_metric_precond and config.gs_optimizer.lower() == "lbfgs"
    )
    optimizer = None if is_metric_lbfgs else _build_optimizer(config)
    opt_state = optimizer.init(params) if optimizer is not None else None
    use_ls = _use_line_search(config)
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

    # CTM conv_tol schedule (shared helper with 1-site optimizer)
    _conv_tol_schedule_2s = config.gs_ctm_conv_tol_schedule
    _current_conv_tol_2s = ctm_cfg_2s.conv_tol

    def _get_scheduled_conv_tol_2s(step_idx, num_steps):
        if _conv_tol_schedule_2s is None:
            return _current_conv_tol_2s
        frac = step_idx / max(num_steps, 1)
        tol = _conv_tol_schedule_2s[0][1]
        for threshold, t in _conv_tol_schedule_2s:
            if frac >= threshold:
                tol = t
        return tol

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
            chi=ctm_cfg_2s.chi,
            max_iter=ctm_cfg_2s.max_iter,
            conv_tol=ctm_cfg_2s.conv_tol,
            renormalize=ctm_cfg_2s.renormalize,
            projector_method=ctm_cfg_2s.projector_method,
            qr_warmup_steps=ctm_cfg_2s.qr_warmup_steps,
            projector_backward=ctm_cfg_2s.projector_backward,
            chi_ramp=ctm_cfg_2s.chi_ramp,
            env_init=_env_cache_2s.get("envs", None),
        )
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

    for step in range(config.gs_num_steps):
        # Update conv_tol if schedule is active
        if _conv_tol_schedule_2s is not None:
            new_tol = _get_scheduled_conv_tol_2s(step, config.gs_num_steps)
            if new_tol != _current_conv_tol_2s:
                _current_conv_tol_2s = new_tol
                ctm_cfg_2s = _replace(ctm_cfg_2s, conv_tol=new_tol)

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
                params = best_params
                _env_cache_2s.update(best_env_cache_2s)
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
            continue
        energy_float = float(energy_val)

        # Update env cache for warm-starting next step
        _update_env_cache_2s(params)

        if _should_accept_best(
            current_best=best_energy,
            candidate=energy_float,
            floor=config.gs_energy_floor,
        ):
            best_energy = energy_float
            best_params = params
            best_env_cache_2s = dict(_env_cache_2s)  # snapshot for warm-start (#317)

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
            )
            logged = True

        prev_energy = energy_float

        if delta_energy < config.gs_conv_tol:
            if config.gs_verbose:
                if not logged:
                    _log_ad_step(
                        "2site-tensor",
                        step,
                        config.gs_num_steps,
                        energy_float,
                        delta_energy,
                        best_energy,
                    )
                _log_ad_converged(
                    "2site-tensor", step, delta_energy, config.gs_conv_tol
                )
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
                    cg_direction = _tree_add(neg_grad, _tree_scale(cg_direction, beta))
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
            if config.gs_line_search_method == "hager_zhang":
                from tenax.algorithms._line_search import hager_zhang_line_search

                slope = _tree_dot(grads, direction)
                if slope >= 0:
                    direction = jax.tree.map(lambda g: -g, grads)
                    if not use_c4v:
                        direction = _tangent_project_unit(direction, params)
                    slope = _tree_dot(grads, direction)
                    if is_metric_lbfgs:
                        lbfgs_history.clear()

                def _phi(alpha):
                    trial = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    return loss_fn_fwd(trial)

                def _dphi(alpha):
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
                )
                if f_alpha < energy_float:
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
                if new_energy < energy_float:
                    stall_count = 0
                else:
                    stall_count += 1

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
                    print(f"[iPEPS-AD] stall #{stall_count}, adding noise", flush=True)
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
                # variPEPS-style reset: clear L-BFGS / CG state so the next
                # step is plain (preconditioned) steepest descent from the
                # CURRENT iterate.  Do NOT roll back params — see 1-site
                # branch comment and issue #298 trajectory study.
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_params_flat = None
                    prev_grad_flat = None
                # Optax-backed L-BFGS stores curvature history in opt_state,
                # not in lbfgs_history.  Reinitialize it so the next step
                # really is steepest descent (reviewer feedback on #298).
                if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                    opt_state = optimizer.init(params)
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD] stall #{stall_count}, "
                        f"reset L-BFGS history (no rollback)",
                        flush=True,
                    )
        else:
            params = optax.apply_updates(params, direction)
            params = _normalize_params(params)

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
            chi=ctm_cfg_2s.chi,
            max_iter=ctm_cfg_2s.max_iter,
            min_iter=ctm_cfg_2s.min_iter,
            conv_tol=ctm_cfg_2s.conv_tol,
            conv_method=ctm_cfg_2s.ctm_conv_method,
            renormalize=ctm_cfg_2s.renormalize,
            projector_method=ctm_cfg_2s.projector_method,
            qr_warmup_steps=ctm_cfg_2s.qr_warmup_steps,
            projector_backward=ctm_cfg_2s.projector_backward,
            chi_ramp=ctm_cfg_2s.chi_ramp,
            env_init=env_init,
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

    if best_params is not params:
        A_best, B_best, envs_best, E_best_fresh = _eval_fresh_2site(
            best_params, best_env_cache_2s.get("envs", None)
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

    return (A_final, B_final), (env_A, env_B), E_gs


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
