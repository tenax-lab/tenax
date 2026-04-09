"""AD-based ground state optimization for iPEPS.

Extracts optimize_gs_ad and related helpers from ipeps.py.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor


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


def _normalize_params(params):
    """Normalize iPEPS site tensor(s)."""
    if isinstance(params, tuple):
        return tuple(p * (1.0 / (p.norm() + 1e-10)) for p in params)
    if hasattr(params, "norm"):
        return params * (1.0 / (params.norm() + 1e-10))
    # Plain JAX array (e.g. C4v coefficients) — use jnp.linalg.norm
    return params / (jnp.linalg.norm(params) + 1e-10)


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

    if config.unit_cell == "2site":
        return _optimize_gs_ad_2site(hamiltonian_gate, A_init, config)

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
            A_init = _wrap_as_dense_tensor(jax.random.normal(key, (D, D, D, D, d_phys)))

    return _optimize_gs_ad_tensor(hamiltonian_gate, A_init, config)


def _optimize_gs_ad_tensor(
    hamiltonian_gate: jax.Array,
    A_init: Tensor,
    config: iPEPSConfig,
):
    """AD-based ground state optimization for Tensor-protocol iPEPS (1-site).

    Uses ``ctm_tensor_converge`` with implicit differentiation through
    the standard Tensor-protocol CTM.
    """
    import optax

    from tenax.algorithms._ctm_tensor import (
        compute_energy_ctm_tensor,
        initialize_ctm_tensor_env,
    )
    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
    from tenax.algorithms.ad_utils import (
        _config_to_tuple,
        ctm_tensor_converge,
        ctm_tensor_converge_explicit,
    )

    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d_phys = gate.shape[0]

    A = A_init
    A = A * (1.0 / (A.norm() + 1e-10))

    # Override projector method for AD if gs_projector_method is set
    ctm_cfg = config.ctm
    if config.gs_projector_method is not None:
        from dataclasses import replace as _replace

        ctm_cfg = _replace(ctm_cfg, projector_method=config.gs_projector_method)
    config_tuple = _config_to_tuple(ctm_cfg)

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

    _env_template = initialize_ctm_tensor_env(A, config.ctm.chi)
    env_treedef = jax.tree.structure(_env_template)
    prev_env_leaves = tuple(jax.tree.leaves(_env_template))

    use_explicit = config.gs_explicit_ad
    explicit_steps = config.gs_explicit_ad_steps
    explicit_warmup = config.gs_explicit_ad_warmup
    _ctm_converge = (
        ctm_tensor_converge_explicit if use_explicit else ctm_tensor_converge
    )

    def loss_fn(params, env_init_leaves):
        if use_c4v:
            A_data = c4v_tensor_from_coeffs(params, c4v_basis, tensor_shape)
        else:
            A_data = params.todense()
        A_norm_data = A_data / (jnp.linalg.norm(A_data) + 1e-10)
        A_norm = DenseTensor(A_norm_data, A.indices)
        site_tensors = {(0, 0): A_norm}
        if use_explicit:
            env_leaves = _ctm_converge(
                site_tensors,
                env_init_leaves,
                SINGLE_SITE_NEIGHBORS,
                config_tuple,
                explicit_steps,
                explicit_warmup,
            )
        else:
            env_leaves = _ctm_converge(
                site_tensors, env_init_leaves, SINGLE_SITE_NEIGHBORS, config_tuple
            )
        env = jax.tree.unflatten(env_treedef, env_leaves)
        energy = compute_energy_ctm_tensor(A_norm, env, gate, d_phys)
        return energy, env_leaves

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
    prev_energy = float("inf")
    prev_grad = None
    cg_direction = None
    prev_precond_grad = None  # for preconditioned CG beta
    log_interval = config.gs_log_interval
    # L-BFGS history for metric-preconditioned path
    lbfgs_history: list = []
    prev_A_flat: jnp.ndarray | None = None
    prev_grad_flat: jnp.ndarray | None = None

    # Forward-only loss for line search — fresh CTM (no warm-start)
    from tenax.algorithms.ad_utils import (
        _config_from_tuple,
        _ctm_tensor_multisite_fixed_point,
        _ctm_tensor_multisite_fixed_point_jit,
    )

    _fp_fn = (
        _ctm_tensor_multisite_fixed_point_jit
        if getattr(config.ctm, "jit_ctm", False)
        else _ctm_tensor_multisite_fixed_point
    )

    def loss_fn_fwd(p):
        """Forward-only loss for line search — warm-starts CTM from prev_env_leaves."""
        if use_c4v:
            A_data = c4v_tensor_from_coeffs(p, c4v_basis, tensor_shape)
        else:
            A_data = p.todense()
        A_norm_data = A_data / (jnp.linalg.norm(A_data) + 1e-10)
        A_norm = DenseTensor(A_norm_data, A.indices)
        site_tensors = {(0, 0): A_norm}
        env_leaves = ctm_tensor_converge(
            site_tensors, prev_env_leaves, SINGLE_SITE_NEIGHBORS, config_tuple
        )
        env_ = jax.tree.unflatten(env_treedef, env_leaves)
        return float(compute_energy_ctm_tensor(A_norm, env_, gate, d_phys))

    stall_count = 0  # noise recovery: consecutive line search failures

    for step in range(config.gs_num_steps):
        (energy_val, env_leaves), grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True
        )(params, prev_env_leaves)
        energy_float = float(energy_val)

        if energy_float < best_energy:
            best_energy = energy_float
            best_params = params

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
        prev_energy = energy_float
        prev_env_leaves = jax.tree.map(jax.lax.stop_gradient, env_leaves)

        # Compute search direction
        if is_cg:
            if config.gs_metric_precond and not use_c4v:
                from tenax.algorithms._metric_precond import precondition_gradient

                env_for_metric = jax.tree.unflatten(env_treedef, env_leaves)
                delta_metric = delta_energy if step > 0 else _tree_dot(grads, grads)
                z_dense = precondition_gradient(
                    A, env_for_metric, grads, delta_metric, config
                )
                z = type(grads)(z_dense, grads.indices)
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
                sy = float(jnp.dot(s, y))
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

                env_for_metric = jax.tree.unflatten(env_treedef, env_leaves)
                delta_metric = (
                    delta_energy if step > 0 else float(jnp.dot(g_flat, g_flat))
                )
                D_bond = A.todense().shape[0]
                d_loc = A.todense().shape[-1]

                def h0_matvec(v):
                    v_tensor = type(A)(
                        v.reshape(D_bond, D_bond, D_bond, D_bond, d_loc), A.indices
                    )
                    result = precondition_gradient(
                        A, env_for_metric, v_tensor, delta_metric, config
                    )
                    return result.reshape(-1)

                direction_flat = lbfgs_two_loop(g_flat, lbfgs_history, h0_matvec)
                direction_dense = -direction_flat.reshape(
                    D_bond, D_bond, D_bond, D_bond, d_loc
                )
                direction = type(A)(direction_dense, A.indices)
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

                def _phi(alpha):
                    trial = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    return loss_fn_fwd(trial)

                def _dphi(alpha):
                    trial = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    (_, _aux), g = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
                        trial, prev_env_leaves
                    )
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

            # Noise recovery on persistent stall
            if stall_count > 0 and stall_count <= config.gs_noise_recovery_retries:
                noise_key = jax.random.PRNGKey(step * 1000 + stall_count)
                if use_c4v:
                    noise = config.gs_noise_amplitude * jax.random.normal(
                        noise_key, params.shape
                    )
                    params = params + noise * jnp.linalg.norm(params)
                    params = params / (jnp.linalg.norm(params) + 1e-10)
                else:
                    data = params.todense()
                    noise = config.gs_noise_amplitude * jax.random.normal(
                        noise_key, data.shape
                    )
                    noisy = data + noise * jnp.linalg.norm(data)
                    params = type(params)(
                        noisy / (jnp.linalg.norm(noisy) + 1e-10), params.indices
                    )
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
        else:
            params = optax.apply_updates(params, direction)
            if not use_c4v:
                params = params * (1.0 / (params.norm() + 1e-10))

    # Re-evaluate both final A and best_A with fully converged fresh CTM.
    # In-loop energies use warm-started CTM that can produce unphysical values
    # (non-variational at finite chi), so we compare fresh evaluations only.
    _base_cfg = _config_from_tuple(config_tuple)
    eval_config = CTMConfig(
        chi=_base_cfg.chi,
        max_iter=max(_base_cfg.max_iter, 200),
        conv_tol=min(_base_cfg.conv_tol, 1e-10),
        min_iter=max(_base_cfg.min_iter, 30),
        renormalize=_base_cfg.renormalize,
        projector_method=_base_cfg.projector_method,
        jit_ctm=_base_cfg.jit_ctm,
        ctm_conv_method=_base_cfg.ctm_conv_method,
        forward_gauge=_base_cfg.forward_gauge,
    )

    def _eval_fresh(p):
        """Evaluate energy with fully converged fresh CTM."""
        if use_c4v:
            A_data = c4v_tensor_from_coeffs(p, c4v_basis, tensor_shape)
        else:
            A_data = p.todense()
        A_data = A_data / (jnp.linalg.norm(A_data) + 1e-10)
        A_t = DenseTensor(A_data, A.indices)
        envs = _fp_fn({(0, 0): A_t}, SINGLE_SITE_NEIGHBORS, eval_config)
        env_ = envs[(0, 0)]
        E_ = float(compute_energy_ctm_tensor(A_t, env_, gate, d_phys))
        return A_t, env_, E_

    A_final, env_final, E_final = _eval_fresh(params)

    if best_params is not params:
        _, env_best, E_best_fresh = _eval_fresh(best_params)
    else:
        E_best_fresh = E_final

    if E_final <= E_best_fresh:
        env, E_gs = env_final, E_final
    else:
        A_final, _, _ = _eval_fresh(best_params)
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
            # Random initialization for 2-site AD
            key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
            A_data = jax.random.normal(key_A, (D, D, D, D, d_phys))
            B_data = jax.random.normal(key_B, (D, D, D, D, d_phys))
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
    """
    import optax

    from tenax.algorithms._ctm_tensor import (
        compute_energy_ctm_tensor_2site,
        initialize_ctm_tensor_env,
    )
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    from tenax.algorithms.ad_utils import (
        _config_from_tuple,
        _config_to_tuple,
        _ctm_tensor_multisite_fixed_point,
        _ctm_tensor_multisite_fixed_point_jit,
        ctm_tensor_converge,
        ctm_tensor_converge_explicit,
    )

    _fp_fn_2site = (
        _ctm_tensor_multisite_fixed_point_jit
        if getattr(config.ctm, "jit_ctm", False)
        else _ctm_tensor_multisite_fixed_point
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

    ctm_cfg_2s = config.ctm
    if config.gs_projector_method is not None:
        from dataclasses import replace as _replace

        ctm_cfg_2s = _replace(ctm_cfg_2s, projector_method=config.gs_projector_method)
    config_tuple = _config_to_tuple(ctm_cfg_2s)
    use_explicit = config.gs_explicit_ad
    explicit_steps = config.gs_explicit_ad_steps
    explicit_warmup = config.gs_explicit_ad_warmup

    # Get env treedef from a template
    _env_template = initialize_ctm_tensor_env(A, config.ctm.chi)
    env_treedef = jax.tree.structure(_env_template)
    n_env_leaves = len(jax.tree.leaves(_env_template))
    _env_template_B = initialize_ctm_tensor_env(B, config.ctm.chi)
    prev_env_leaves = tuple(jax.tree.leaves(_env_template)) + tuple(
        jax.tree.leaves(_env_template_B)
    )

    _ctm_converge = (
        ctm_tensor_converge_explicit if use_explicit else ctm_tensor_converge
    )

    def loss_fn(params, env_init_leaves):
        A_p, B_p = params
        A_norm = A_p * (1.0 / (A_p.norm() + 1e-10))
        B_norm = B_p * (1.0 / (B_p.norm() + 1e-10))
        site_tensors = {(0, 0): A_norm, (1, 0): B_norm}
        if use_explicit:
            env_leaves = _ctm_converge(
                site_tensors,
                env_init_leaves,
                CHECKERBOARD_NEIGHBORS,
                config_tuple,
                explicit_steps,
                explicit_warmup,
            )
        else:
            env_leaves = _ctm_converge(
                site_tensors, env_init_leaves, CHECKERBOARD_NEIGHBORS, config_tuple
            )
        env_A = jax.tree.unflatten(env_treedef, env_leaves[:n_env_leaves])
        env_B = jax.tree.unflatten(env_treedef, env_leaves[n_env_leaves:])
        energy = compute_energy_ctm_tensor_2site(
            A_norm, B_norm, env_A, env_B, gate, d_phys
        )
        return energy, env_leaves

    params = (A, B)
    is_metric_lbfgs = (
        config.gs_metric_precond and config.gs_optimizer.lower() == "lbfgs"
    )
    optimizer = None if is_metric_lbfgs else _build_optimizer(config)
    opt_state = optimizer.init(params) if optimizer is not None else None
    use_ls = _use_line_search(config)
    is_cg = config.gs_optimizer.lower() == "cg"

    best_energy = float("inf")
    best_params = params
    prev_energy = float("inf")
    prev_grad = None
    cg_direction = None
    prev_precond_grad = None
    log_interval = config.gs_log_interval
    lbfgs_history: list = []
    prev_params_flat: jnp.ndarray | None = None
    prev_grad_flat: jnp.ndarray | None = None
    stall_count = 0  # noise recovery: consecutive line search failures

    # Forward-only loss for line search — warm-starts CTM from prev_env_leaves

    def loss_fn_fwd(params_):
        A_p, B_p = params_
        A_norm = A_p * (1.0 / (A_p.norm() + 1e-10))
        B_norm = B_p * (1.0 / (B_p.norm() + 1e-10))
        site_tensors = {(0, 0): A_norm, (1, 0): B_norm}
        env_leaves = ctm_tensor_converge(
            site_tensors, prev_env_leaves, CHECKERBOARD_NEIGHBORS, config_tuple
        )
        env_A_ = jax.tree.unflatten(env_treedef, env_leaves[:n_env_leaves])
        env_B_ = jax.tree.unflatten(env_treedef, env_leaves[n_env_leaves:])
        return float(
            compute_energy_ctm_tensor_2site(
                A_norm,
                B_norm,
                env_A_,
                env_B_,
                gate,
                d_phys,
            )
        )

    for step in range(config.gs_num_steps):
        (energy_val, env_leaves), grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True
        )(params, prev_env_leaves)
        energy_float = float(energy_val)
        env_leaves_sg = jax.tree.map(jax.lax.stop_gradient, env_leaves)

        if energy_float < best_energy:
            best_energy = energy_float
            best_params = params

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
        prev_energy = energy_float
        prev_env_leaves = env_leaves_sg

        # Compute search direction
        if is_cg:
            if config.gs_metric_precond:
                from tenax.algorithms._metric_precond import (
                    precondition_gradient_multisite,
                )

                env_A_m = jax.tree.unflatten(env_treedef, env_leaves_sg[:n_env_leaves])
                env_B_m = jax.tree.unflatten(env_treedef, env_leaves_sg[n_env_leaves:])
                A_g, B_g = grads
                envs_m = {(0, 0): env_A_m, (1, 0): env_B_m}
                sites_m = {(0, 0): params[0], (1, 0): params[1]}
                grads_m = {(0, 0): A_g, (1, 0): B_g}
                delta_metric = delta_energy if step > 0 else _tree_dot(grads, grads)
                z_dict = precondition_gradient_multisite(
                    sites_m, envs_m, grads_m, delta_metric, config
                )
                z = (
                    type(A_g)(z_dict[(0, 0)], A_g.indices),
                    type(B_g)(z_dict[(1, 0)], B_g.indices),
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

            A_cur, B_cur = params
            A_g, B_g = grads
            p_flat = jnp.concatenate(
                [A_cur.todense().reshape(-1), B_cur.todense().reshape(-1)]
            )
            g_flat = jnp.concatenate(
                [A_g.todense().reshape(-1), B_g.todense().reshape(-1)]
            )

            if prev_params_flat is not None:
                s = p_flat - prev_params_flat
                y = g_flat - prev_grad_flat
                sy = float(jnp.dot(s, y))
                if sy > 1e-10:
                    rho = 1.0 / sy
                    lbfgs_history.append((s, y, rho))
                    if len(lbfgs_history) > 10:
                        lbfgs_history.pop(0)
            prev_params_flat = p_flat
            prev_grad_flat = g_flat

            env_A_m = jax.tree.unflatten(env_treedef, env_leaves_sg[:n_env_leaves])
            env_B_m = jax.tree.unflatten(env_treedef, env_leaves_sg[n_env_leaves:])
            envs_m = {(0, 0): env_A_m, (1, 0): env_B_m}
            sites_m = {(0, 0): A_cur, (1, 0): B_cur}
            delta_metric = delta_energy if step > 0 else float(jnp.dot(g_flat, g_flat))
            n_A = A_cur.todense().size

            def h0_matvec(v):
                v_A = v[:n_A]
                v_B = v[n_A:]
                D_b = A_cur.todense().shape[0]
                d_l = A_cur.todense().shape[-1]
                grads_v = {
                    (0, 0): type(A_cur)(
                        v_A.reshape(D_b, D_b, D_b, D_b, d_l), A_cur.indices
                    ),
                    (1, 0): type(B_cur)(
                        v_B.reshape(D_b, D_b, D_b, D_b, d_l), B_cur.indices
                    ),
                }
                z_dict = precondition_gradient_multisite(
                    sites_m, envs_m, grads_v, delta_metric, config
                )
                return jnp.concatenate(
                    [z_dict[(0, 0)].reshape(-1), z_dict[(1, 0)].reshape(-1)]
                )

            direction_flat = lbfgs_two_loop(g_flat, lbfgs_history, h0_matvec)
            D_b = A_cur.todense().shape[0]
            d_l = A_cur.todense().shape[-1]
            dir_A = -direction_flat[:n_A].reshape(D_b, D_b, D_b, D_b, d_l)
            dir_B = -direction_flat[n_A:].reshape(D_b, D_b, D_b, D_b, d_l)
            direction = (
                type(A_cur)(dir_A, A_cur.indices),
                type(B_cur)(dir_B, B_cur.indices),
            )
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

                def _phi(alpha):
                    trial = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    return loss_fn_fwd(trial)

                def _dphi(alpha):
                    trial = _normalize_params(
                        _tree_add(params, _tree_scale(direction, alpha))
                    )
                    (_, _aux), g = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
                        trial, prev_env_leaves
                    )
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

            # Noise recovery on persistent stall
            if stall_count > 0 and stall_count <= config.gs_noise_recovery_retries:
                noise_key = jax.random.PRNGKey(step * 1000 + stall_count)
                noisy_params = []
                for i, p in enumerate(params):
                    k = jax.random.fold_in(noise_key, i)
                    data = p.todense()
                    noise = config.gs_noise_amplitude * jax.random.normal(k, data.shape)
                    noisy = data + noise * jnp.linalg.norm(data)
                    noisy_params.append(
                        type(p)(noisy / (jnp.linalg.norm(noisy) + 1e-10), p.indices)
                    )
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
        else:
            params = optax.apply_updates(params, direction)
            params = _normalize_params(params)

    # Re-evaluate both final params and best_params with fully converged
    # fresh CTM.  In-loop energies use warm-started CTM that can produce
    # unphysical values, so we compare fresh evaluations only.
    _base_cfg2 = _config_from_tuple(config_tuple)
    eval_config2 = CTMConfig(
        chi=_base_cfg2.chi,
        max_iter=max(_base_cfg2.max_iter, 200),
        conv_tol=min(_base_cfg2.conv_tol, 1e-10),
        min_iter=max(_base_cfg2.min_iter, 30),
        renormalize=_base_cfg2.renormalize,
        projector_method=_base_cfg2.projector_method,
        jit_ctm=_base_cfg2.jit_ctm,
        ctm_conv_method=_base_cfg2.ctm_conv_method,
        forward_gauge=_base_cfg2.forward_gauge,
    )

    def _eval_fresh_2site(p):
        A_t, B_t = _normalize_params(p)
        st = {(0, 0): A_t, (1, 0): B_t}
        envs = _fp_fn_2site(st, CHECKERBOARD_NEIGHBORS, eval_config2)
        E_ = float(
            compute_energy_ctm_tensor_2site(
                A_t, B_t, envs[(0, 0)], envs[(1, 0)], gate, d_phys
            )
        )
        return A_t, B_t, envs, E_

    A_last, B_last, envs_last, E_last = _eval_fresh_2site(params)
    env_A_last, env_B_last = envs_last[(0, 0)], envs_last[(1, 0)]

    if best_params is not params:
        _, _, envs_best, E_best_fresh = _eval_fresh_2site(best_params)
        env_A_best = envs_best[(0, 0)]
        env_B_best = envs_best[(1, 0)]
    else:
        E_best_fresh = E_last

    # Pick whichever fresh evaluation is lower
    if E_last <= E_best_fresh:
        A_final, B_final = A_last, B_last
        env_A, env_B, E_gs = env_A_last, env_B_last, E_last
    else:
        A_final, B_final = _normalize_params(best_params)
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

    The AD backward pass (GMRES implicit differentiation) currently
    requires ``DenseTensor`` leaves for stable gradients, so input
    ``SymmetricTensor`` tensors are automatically wrapped as
    ``DenseTensor`` (preserving the index structure including
    ``FermionParity`` symmetry charges and flow directions).  The
    returned ``A_opt`` is a ``DenseTensor`` with the same index
    metadata.

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
        ``(A_opt, env, E_gs)`` where ``A_opt`` is the optimized
        ``DenseTensor``, ``env`` is a ``CTMTensorEnv``, and ``E_gs``
        is the ground-state energy per site.
    """
    if A_init is None:
        if fpeps_config is None:
            raise ValueError(
                "fpeps_config is required when A_init is None "
                "(needed to build the initial fPEPS tensor)."
            )
        from tenax.algorithms.fermionic_ipeps import _build_initial_fpeps_tensor

        A_init = _build_initial_fpeps_tensor(fpeps_config)

    # Wrap SymmetricTensor as DenseTensor for stable AD backward pass.
    # The DenseTensor retains the original index metadata (symmetry,
    # charges, flows) so the CTM pipeline uses the correct labels.
    if isinstance(A_init, SymmetricTensor):
        A_init = DenseTensor(A_init.todense(), A_init.indices)

    return _optimize_gs_ad_tensor(hamiltonian_gate, A_init, config)
