"""AD-based ground state optimization for iPEPS.

Extracts optimize_gs_ad and related helpers from ipeps.py.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor


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
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
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
    prev_env_leaves = tuple(jax.tree.leaves(_env_template))

    def loss_fn(A_param, env_init_leaves):
        A_norm = A_param * (1.0 / (A_param.norm() + 1e-10))
        site_tensors = {(0, 0): A_norm}
        env_leaves = ctm_tensor_converge(
            site_tensors, env_init_leaves, SINGLE_SITE_NEIGHBORS, config_tuple
        )
        env = jax.tree.unflatten(env_treedef, env_leaves)
        energy = compute_energy_ctm_tensor(A_norm, env, gate, d_phys)
        return energy, env_leaves

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gs_max_grad_norm),
        optax.adam(config.gs_learning_rate),
    )
    opt_state = optimizer.init(A)

    best_energy = float("inf")
    best_A = A
    prev_energy = float("inf")
    log_interval = config.gs_log_interval

    for step in range(config.gs_num_steps):
        (energy_val, env_leaves), grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True
        )(A, prev_env_leaves)
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
        prev_env_leaves = jax.tree.map(jax.lax.stop_gradient, env_leaves)

        updates, opt_state = optimizer.update(grads, opt_state, A)
        A = optax.apply_updates(A, updates)
        A = A * (1.0 / (A.norm() + 1e-10))

    A_final = best_A * (1.0 / (best_A.norm() + 1e-10))
    env_leaves = ctm_tensor_converge(
        {(0, 0): A_final}, prev_env_leaves, SINGLE_SITE_NEIGHBORS, config_tuple
    )
    env = jax.tree.unflatten(env_treedef, env_leaves)
    E_gs = float(compute_energy_ctm_tensor(A_final, env, gate, d_phys))
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
            key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
            A = _wrap_as_dense_tensor(jax.random.normal(key_A, (D, D, D, D, d_phys)))
            B = _wrap_as_dense_tensor(jax.random.normal(key_B, (D, D, D, D, d_phys)))
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
    from tenax.algorithms.ad_utils import _config_to_tuple, ctm_tensor_converge

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
    _env_template_B = initialize_ctm_tensor_env(B, config.ctm.chi)
    prev_env_leaves = tuple(jax.tree.leaves(_env_template)) + tuple(
        jax.tree.leaves(_env_template_B)
    )

    def loss_fn(params, env_init_leaves):
        A_p, B_p = params
        A_norm = A_p * (1.0 / (A_p.norm() + 1e-10))
        B_norm = B_p * (1.0 / (B_p.norm() + 1e-10))
        site_tensors = {(0, 0): A_norm, (1, 0): B_norm}
        env_leaves = ctm_tensor_converge(
            site_tensors, env_init_leaves, CHECKERBOARD_NEIGHBORS, config_tuple
        )
        env_A = jax.tree.unflatten(env_treedef, env_leaves[:n_env_leaves])
        env_B = jax.tree.unflatten(env_treedef, env_leaves[n_env_leaves:])
        energy = compute_energy_ctm_tensor_2site(
            A_norm, B_norm, env_A, env_B, gate, d_phys
        )
        return energy, env_leaves

    params = (A, B)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gs_max_grad_norm),
        optax.adam(config.gs_learning_rate),
    )
    opt_state = optimizer.init(params)

    last_energy = float("inf")
    last_params = params
    last_env_leaves = None
    prev_energy = float("inf")
    log_interval = config.gs_log_interval

    for step in range(config.gs_num_steps):
        (energy_val, env_leaves), grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True
        )(params, prev_env_leaves)
        energy_float = float(energy_val)
        last_energy = energy_float
        last_params = params
        last_env_leaves = jax.tree.map(jax.lax.stop_gradient, env_leaves)

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
        prev_env_leaves = jax.tree.map(jax.lax.stop_gradient, env_leaves)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        A_p, B_p = params
        params = (
            A_p * (1.0 / (A_p.norm() + 1e-10)),
            B_p * (1.0 / (B_p.norm() + 1e-10)),
        )

    if last_env_leaves is None:
        energy_val, env_leaves = loss_fn(params, prev_env_leaves)
        last_energy = float(energy_val)
        last_params = params
        last_env_leaves = jax.tree.map(jax.lax.stop_gradient, env_leaves)

    # Use last evaluated params and environment
    A_final, B_final = last_params
    A_final = A_final * (1.0 / (A_final.norm() + 1e-10))
    B_final = B_final * (1.0 / (B_final.norm() + 1e-10))
    env_A = jax.tree.unflatten(env_treedef, last_env_leaves[:n_env_leaves])
    env_B = jax.tree.unflatten(env_treedef, last_env_leaves[n_env_leaves:])
    E_gs = last_energy
    if config.gs_verbose:
        print(f"[iPEPS-AD:2site-tensor] final E={E_gs:.10f}", flush=True)

    return (A_final, B_final), (env_A, env_B), E_gs
