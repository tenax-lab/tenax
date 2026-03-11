"""AD-based ground state optimization for iPEPS.

Extracts optimize_gs_ad and related helpers from ipeps.py.
"""

from __future__ import annotations

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


def optimize_gs_ad(
    hamiltonian_gate: jax.Array,
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
    if config.unit_cell == "2site":
        return _optimize_gs_ad_2site(hamiltonian_gate, A_init, config)

    # Dispatch: Tensor-protocol path vs dense path
    if isinstance(A_init, Tensor):
        return _optimize_gs_ad_tensor(hamiltonian_gate, A_init, config)

    import optax

    from tenax.algorithms.ad_utils import _config_to_tuple, ctm_converge
    from tenax.algorithms.ipeps import ipeps

    gate = jnp.array(hamiltonian_gate)
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

    for step in range(config.gs_num_steps):
        energy_val, grads = jax.value_and_grad(loss_fn)(A)
        energy_float = float(energy_val)

        if energy_float < best_energy:
            best_energy = energy_float
            best_A = A

        # Check convergence
        if abs(energy_float - prev_energy) < config.gs_conv_tol:
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

    gate = jnp.array(hamiltonian_gate)
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

    for step in range(config.gs_num_steps):
        energy_val, grads = jax.value_and_grad(loss_fn)(A)
        energy_float = float(energy_val)

        if energy_float < best_energy:
            best_energy = energy_float
            best_A = A

        if abs(energy_float - prev_energy) < config.gs_conv_tol:
            break
        prev_energy = energy_float

        updates, opt_state = optimizer.update(grads, opt_state, A)
        A = optax.apply_updates(A, updates)
        A = A * (1.0 / (A.norm() + 1e-10))

    A_final = best_A * (1.0 / (best_A.norm() + 1e-10))
    env_leaves = ctm_tensor_converge(A_final, config_tuple)
    env = jax.tree.unflatten(env_treedef, env_leaves)
    E_gs = float(compute_energy_ctm_tensor(A_final, env, gate, d_phys))

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
    if isinstance(AB_init, tuple) and any(isinstance(t, Tensor) for t in AB_init):
        return _optimize_gs_ad_tensor_2site(hamiltonian_gate, AB_init, config)

    import optax

    from tenax.algorithms.ad_utils import ctm_converge_2site

    gate = jnp.array(hamiltonian_gate)
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
        return energy

    # optax.adam supports pytree params natively
    params = (A, B)
    if config.gs_optimizer == "adam":
        optimizer = optax.adam(config.gs_learning_rate)
    else:
        optimizer = optax.adam(config.gs_learning_rate)

    opt_state = optimizer.init(params)

    best_energy = float("inf")
    best_params = params
    prev_energy = float("inf")

    for step in range(config.gs_num_steps):
        energy_val, grads = jax.value_and_grad(loss_fn)(params)
        energy_float = float(energy_val)

        if energy_float < best_energy:
            best_energy = energy_float
            best_params = params

        if abs(energy_float - prev_energy) < config.gs_conv_tol:
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

    # Final CTM environment
    A_final, B_final = best_params
    A_final = A_final / (jnp.linalg.norm(A_final) + 1e-10)
    B_final = B_final / (jnp.linalg.norm(B_final) + 1e-10)
    env_tuple = ctm_converge_2site(A_final, B_final, config_tuple)
    env_A = CTMEnvironment(*env_tuple[:8])
    env_B = CTMEnvironment(*env_tuple[8:])
    E_gs = float(compute_energy_ctm_2site(A_final, B_final, env_A, env_B, gate, d_phys))

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

    gate = jnp.array(hamiltonian_gate)
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
        return energy

    params = (A, B)
    optimizer = optax.adam(config.gs_learning_rate)
    opt_state = optimizer.init(params)

    best_energy = float("inf")
    best_params = params
    prev_energy = float("inf")

    for _ in range(config.gs_num_steps):
        energy_val, grads = jax.value_and_grad(loss_fn)(params)
        energy_float = float(energy_val)

        if energy_float < best_energy:
            best_energy = energy_float
            best_params = params

        if abs(energy_float - prev_energy) < config.gs_conv_tol:
            break
        prev_energy = energy_float

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        A_p, B_p = params
        params = (
            A_p * (1.0 / (A_p.norm() + 1e-10)),
            B_p * (1.0 / (B_p.norm() + 1e-10)),
        )

    # Final CTM environment
    A_final, B_final = best_params
    A_final = A_final * (1.0 / (A_final.norm() + 1e-10))
    B_final = B_final * (1.0 / (B_final.norm() + 1e-10))
    env_leaves = ctm_tensor_converge_2site(A_final, B_final, config_tuple)
    env_A = jax.tree.unflatten(env_treedef, env_leaves[:n_env_leaves])
    env_B = jax.tree.unflatten(env_treedef, env_leaves[n_env_leaves:])
    E_gs = float(
        compute_energy_ctm_tensor_2site(A_final, B_final, env_A, env_B, gate, d_phys)
    )

    return (A_final, B_final), (env_A, env_B), E_gs
