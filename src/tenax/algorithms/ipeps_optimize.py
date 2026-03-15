"""AD-based ground state optimization for iPEPS.

Extracts optimize_gs_ad and related helpers from ipeps.py.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import (
    CTMEnvironment,
    iPEPSConfig,
)
from tenax.algorithms.ipeps_rdm import (
    compute_energy_ctm,
    compute_energy_ctm_2site,
)
from tenax.core.tensor import Tensor


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
        f"E={energy:.10f} dE={delta_str} E_best={best_energy:.10f}"
    )


def _log_ad_converged(backend: str, step: int, delta_energy: float, tol: float) -> None:
    print(
        f"[iPEPS-AD:{backend}] converged at step {step + 1} "
        f"(dE={delta_energy:.3e} < tol={tol:.3e})"
    )


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

    # Dispatch: Tensor-protocol path vs dense path
    if isinstance(A_init, Tensor):
        return _optimize_gs_ad_tensor(hamiltonian_gate, A_init, config)

    import optax

    from tenax.algorithms.ad_utils import _config_to_tuple, ctm_converge
    from tenax.algorithms.ipeps import ipeps

    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d_phys = gate.shape[0]
    D = config.max_bond_dim

    # Initialize site tensor
    if A_init is None:
        if config.su_init:
            _, su_peps, _ = ipeps(gate, None, config)
            A = su_peps.get_tensor((0, 0)).todense()
        else:
            key = jax.random.PRNGKey(0)
            A = jax.random.normal(key, (D, D, D, D, d_phys))
    else:
        A = jnp.array(A_init)
    A = A / (jnp.linalg.norm(A) + 1e-10)

    config_tuple = _config_to_tuple(config.ctm)

    # Define loss: A -> energy
    def loss_fn(A_param):
        A_norm = A_param / (jnp.linalg.norm(A_param) + 1e-10)
        env_tuple = ctm_converge(A_norm, config_tuple)
        env = CTMEnvironment(*env_tuple)
        energy = compute_energy_ctm(A_norm, env, gate, d_phys)
        return energy

    # Set up optimizer
    if config.gs_optimizer == "adam":
        optimizer = optax.adam(config.gs_learning_rate)
    else:
        optimizer = optax.adam(config.gs_learning_rate)

    opt_state = optimizer.init(A)

    best_energy = float("inf")
    best_A = A
    prev_energy = float("inf")
    log_interval = config.gs_log_interval

    for step in range(config.gs_num_steps):
        energy_val, grads = jax.value_and_grad(loss_fn)(A)
        energy_float = float(energy_val)

        if energy_float < best_energy:
            best_energy = energy_float
            best_A = A

        delta_energy = abs(energy_float - prev_energy)
        logged = False
        if config.gs_verbose and _should_log_step(
            step, config.gs_num_steps, log_interval
        ):
            _log_ad_step(
                "1site-dense",
                step,
                config.gs_num_steps,
                energy_float,
                delta_energy,
                best_energy,
            )
            logged = True

        # Check convergence
        if delta_energy < config.gs_conv_tol:
            if config.gs_verbose:
                if not logged:
                    _log_ad_step(
                        "1site-dense",
                        step,
                        config.gs_num_steps,
                        energy_float,
                        delta_energy,
                        best_energy,
                    )
                _log_ad_converged("1site-dense", step, delta_energy, config.gs_conv_tol)
            break
        prev_energy = energy_float

        updates, opt_state = optimizer.update(grads, opt_state, A)
        A = optax.apply_updates(A, updates)
        # Re-normalize
        A = A / (jnp.linalg.norm(A) + 1e-10)

    # Final CTM environment
    A_final = best_A / (jnp.linalg.norm(best_A) + 1e-10)
    env_tuple = ctm_converge(A_final, config_tuple)
    env = CTMEnvironment(*env_tuple)
    E_gs = float(compute_energy_ctm(A_final, env, gate, d_phys))
    if config.gs_verbose:
        print(f"[iPEPS-AD:1site-dense] final E={E_gs:.10f}")

    return A_final, env, E_gs


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
    from tenax.algorithms.ad_utils import _config_to_tuple, ctm_tensor_converge

    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d_phys = gate.shape[0]

    A = A_init
    A = A * (1.0 / (A.norm() + 1e-10))

    config_tuple = _config_to_tuple(config.ctm)

    _env_template = initialize_ctm_tensor_env(A, config.ctm.chi)
    env_treedef = jax.tree.structure(_env_template)

    def loss_fn(A_param):
        A_norm = A_param * (1.0 / (A_param.norm() + 1e-10))
        env_leaves = ctm_tensor_converge(A_norm, config_tuple)
        env = jax.tree.unflatten(env_treedef, env_leaves)
        energy = compute_energy_ctm_tensor(A_norm, env, gate, d_phys)
        return energy

    optimizer = optax.adam(config.gs_learning_rate)
    opt_state = optimizer.init(A)

    best_energy = float("inf")
    best_A = A
    prev_energy = float("inf")
    log_interval = config.gs_log_interval

    for step in range(config.gs_num_steps):
        energy_val, grads = jax.value_and_grad(loss_fn)(A)
        energy_float = float(energy_val)

        if energy_float < best_energy:
            best_energy = energy_float
            best_A = A

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

        updates, opt_state = optimizer.update(grads, opt_state, A)
        A = optax.apply_updates(A, updates)
        A = A * (1.0 / (A.norm() + 1e-10))

    A_final = best_A * (1.0 / (best_A.norm() + 1e-10))
    env_leaves = ctm_tensor_converge(A_final, config_tuple)
    env = jax.tree.unflatten(env_treedef, env_leaves)
    E_gs = float(compute_energy_ctm_tensor(A_final, env, gate, d_phys))
    if config.gs_verbose:
        print(f"[iPEPS-AD:1site-tensor] final E={E_gs:.10f}")

    return A_final, env, E_gs


def _optimize_gs_ad_2site(
    hamiltonian_gate: jax.Array,
    AB_init: tuple[jax.Array, jax.Array] | tuple[Tensor, Tensor] | None,
    config: iPEPSConfig,
):
    """AD-based ground state optimization for 2-site iPEPS unit cell.

    Uses implicit differentiation through the 2-site CTM fixed point
    to compute gradients of energy w.r.t. both site tensors (A, B).

    Accepts dense ``jax.Array`` or Tensor-protocol objects.
    """
    if AB_init is not None:
        if not isinstance(AB_init, tuple) or len(AB_init) != 2:
            raise TypeError(
                "For unit_cell='2site', A_init must be None or a tuple (A, B)."
            )
        has_tensor = any(isinstance(t, Tensor) for t in AB_init)
        if has_tensor and not all(isinstance(t, Tensor) for t in AB_init):
            raise TypeError(
                "For unit_cell='2site', A_init must be either "
                "(Tensor, Tensor) or (array, array); mixed tuples are not supported."
            )
        if has_tensor:
            return _optimize_gs_ad_tensor_2site(hamiltonian_gate, AB_init, config)

    import optax

    from tenax.algorithms.ad_utils import ctm_converge_2site

    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d_phys = gate.shape[0]
    D = config.max_bond_dim

    # Initialize site tensors
    if AB_init is None:
        if config.su_init:
            from tenax.algorithms.ipeps import ipeps

            su_config = iPEPSConfig(
                max_bond_dim=D,
                num_imaginary_steps=config.num_imaginary_steps,
                dt=config.dt,
                ctm=config.ctm,
                unit_cell="2site",
            )
            _, su_peps, _ = ipeps(gate, None, su_config)
            A = su_peps.get_tensor((0, 0)).todense()
            B = su_peps.get_tensor((1, 0)).todense()
        else:
            key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
            A = jax.random.normal(key_A, (D, D, D, D, d_phys))
            B = jax.random.normal(key_B, (D, D, D, D, d_phys))
    else:
        A, B = AB_init
        A = jnp.array(A)
        B = jnp.array(B)
    A = A / (jnp.linalg.norm(A) + 1e-10)
    B = B / (jnp.linalg.norm(B) + 1e-10)

    from tenax.algorithms.ad_utils import _config_to_tuple

    config_tuple = _config_to_tuple(config.ctm)

    def loss_fn(params):
        A_p, B_p = params
        A_norm = A_p / (jnp.linalg.norm(A_p) + 1e-10)
        B_norm = B_p / (jnp.linalg.norm(B_p) + 1e-10)
        env_tuple = ctm_converge_2site(A_norm, B_norm, config_tuple)
        env_A = CTMEnvironment(*env_tuple[:8])
        env_B = CTMEnvironment(*env_tuple[8:])
        energy = compute_energy_ctm_2site(A_norm, B_norm, env_A, env_B, gate, d_phys)
        return energy, env_tuple

    # optax.adam supports pytree params natively
    params = (A, B)
    if config.gs_optimizer == "adam":
        optimizer = optax.adam(config.gs_learning_rate)
    else:
        optimizer = optax.adam(config.gs_learning_rate)

    opt_state = optimizer.init(params)

    last_energy = float("inf")
    last_params = params
    last_env_tuple = None
    prev_energy = float("inf")
    log_interval = config.gs_log_interval

    for step in range(config.gs_num_steps):
        (energy_val, env_tuple), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )
        energy_float = float(energy_val)
        last_energy = energy_float
        last_params = params
        last_env_tuple = jax.tree.map(lambda x: jax.lax.stop_gradient(x), env_tuple)

        delta_energy = abs(energy_float - prev_energy)
        logged = False
        if config.gs_verbose and _should_log_step(
            step, config.gs_num_steps, log_interval
        ):
            _log_ad_step(
                "2site-dense",
                step,
                config.gs_num_steps,
                energy_float,
                delta_energy,
                energy_float,
            )
            logged = True

        if delta_energy < config.gs_conv_tol:
            if config.gs_verbose:
                if not logged:
                    _log_ad_step(
                        "2site-dense",
                        step,
                        config.gs_num_steps,
                        energy_float,
                        delta_energy,
                        energy_float,
                    )
                _log_ad_converged("2site-dense", step, delta_energy, config.gs_conv_tol)
            break
        prev_energy = energy_float

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        # Re-normalize
        A_p, B_p = params
        params = (
            A_p / (jnp.linalg.norm(A_p) + 1e-10),
            B_p / (jnp.linalg.norm(B_p) + 1e-10),
        )

    if last_env_tuple is None:
        energy_val, env_tuple = loss_fn(params)
        last_energy = float(energy_val)
        last_params = params
        last_env_tuple = jax.tree.map(lambda x: jax.lax.stop_gradient(x), env_tuple)

    # Use the last evaluated params and environment (not "best" which can
    # capture transient CTM artifacts at finite chi).
    A_final, B_final = last_params
    A_final = A_final / (jnp.linalg.norm(A_final) + 1e-10)
    B_final = B_final / (jnp.linalg.norm(B_final) + 1e-10)
    env_A = CTMEnvironment(*last_env_tuple[:8])
    env_B = CTMEnvironment(*last_env_tuple[8:])
    E_gs = last_energy
    if config.gs_verbose:
        print(f"[iPEPS-AD:2site-dense] final E={E_gs:.10f}")

    return (A_final, B_final), (env_A, env_B), E_gs


def _optimize_gs_ad_tensor_2site(
    hamiltonian_gate: jax.Array,
    AB_init: tuple[Tensor, Tensor],
    config: iPEPSConfig,
):
    """AD-based ground state optimization for 2-site Tensor-protocol iPEPS.

    Uses ``ctm_tensor_converge_2site`` with implicit differentiation through
    the 2-site Tensor-protocol CTM.
    """
    import optax

    from tenax.algorithms._ctm_tensor import (
        compute_energy_ctm_tensor_2site,
        initialize_ctm_tensor_env,
    )
    from tenax.algorithms.ad_utils import _config_to_tuple, ctm_tensor_converge_2site

    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d_phys = gate.shape[0]

    A, B = AB_init
    A = A * (1.0 / (A.norm() + 1e-10))
    B = B * (1.0 / (B.norm() + 1e-10))

    config_tuple = _config_to_tuple(config.ctm)

    # Get env treedef from a template
    _env_template = initialize_ctm_tensor_env(A, config.ctm.chi)
    env_treedef = jax.tree.structure(_env_template)
    n_env_leaves = len(jax.tree.leaves(_env_template))

    def loss_fn(params):
        A_p, B_p = params
        A_norm = A_p * (1.0 / (A_p.norm() + 1e-10))
        B_norm = B_p * (1.0 / (B_p.norm() + 1e-10))
        env_leaves = ctm_tensor_converge_2site(A_norm, B_norm, config_tuple)
        env_A = jax.tree.unflatten(env_treedef, env_leaves[:n_env_leaves])
        env_B = jax.tree.unflatten(env_treedef, env_leaves[n_env_leaves:])
        energy = compute_energy_ctm_tensor_2site(
            A_norm, B_norm, env_A, env_B, gate, d_phys
        )
        return energy, env_leaves

    params = (A, B)
    optimizer = optax.adam(config.gs_learning_rate)
    opt_state = optimizer.init(params)

    last_energy = float("inf")
    last_params = params
    last_env_leaves = None
    prev_energy = float("inf")
    log_interval = config.gs_log_interval

    for step in range(config.gs_num_steps):
        (energy_val, env_leaves), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )
        energy_float = float(energy_val)
        last_energy = energy_float
        last_params = params
        last_env_leaves = jax.tree.map(lambda x: jax.lax.stop_gradient(x), env_leaves)

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
                energy_float,
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
                        energy_float,
                    )
                _log_ad_converged(
                    "2site-tensor", step, delta_energy, config.gs_conv_tol
                )
            break
        prev_energy = energy_float

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        A_p, B_p = params
        params = (
            A_p * (1.0 / (A_p.norm() + 1e-10)),
            B_p * (1.0 / (B_p.norm() + 1e-10)),
        )

    if last_env_leaves is None:
        energy_val, env_leaves = loss_fn(params)
        last_energy = float(energy_val)
        last_params = params
        last_env_leaves = jax.tree.map(lambda x: jax.lax.stop_gradient(x), env_leaves)

    # Use last evaluated params and environment
    A_final, B_final = last_params
    A_final = A_final * (1.0 / (A_final.norm() + 1e-10))
    B_final = B_final * (1.0 / (B_final.norm() + 1e-10))
    env_A = jax.tree.unflatten(env_treedef, last_env_leaves[:n_env_leaves])
    env_B = jax.tree.unflatten(env_treedef, last_env_leaves[n_env_leaves:])
    E_gs = last_energy
    if config.gs_verbose:
        print(f"[iPEPS-AD:2site-tensor] final E={E_gs:.10f}")

    return (A_final, B_final), (env_A, env_B), E_gs
