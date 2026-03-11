"""Infinite Projected Entangled Pair States (iPEPS) algorithm.

iPEPS is a variational ansatz for 2D quantum lattice models. The state is
represented as a PEPS (Projected Entangled Pair States) tensor network where
each site has a local tensor A[u,d,l,r,s] (up,down,left,right,physical).

For infinite systems, we use a unit cell (typically 1x1 for translationally
invariant states) and compute observables using the Corner Transfer Matrix (CTM)
method to approximate the infinite environment.

This module implements:
1. Simple update: fast imaginary time evolution optimization
2. CTM algorithm: environment computation for expectation values
3. Energy evaluation using CTM environment

Reference:
- Corboz et al., PRB 81, 165104 (2010) (CTM)
- Jiang et al., PRB 78, 134432 (2008) (simple update)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps_config import (
    CTMConfig,
    CTMEnvironment,
    SplitCTMEnvironment,
    iPEPSConfig,
)
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor
from tenax.algorithms.ipeps_simple_update import (
    _absorb_lambdas_tensor,
    _make_trotter_gate_tensor,
    _simple_update_1x1,
    _simple_update_2site_bond,
    _simple_update_2site_horizontal,
    _simple_update_2site_vertical,
    _simple_update_3leg,
    _simple_update_bond,
    _simple_update_horizontal,
    _simple_update_horizontal_tensor,
    _simple_update_vertical,
    _simple_update_vertical_tensor,
)
from tenax.algorithms.ipeps_rdm import (
    _build_double_layer_open,
    _rdm1x2,
    _rdm1x2_2site,
    _rdm2x1,
    _rdm2x1_2site,
    compute_energy_ctm,
    compute_energy_ctm_2site,
    compute_energy_split_ctm,
)
from tenax.network.network import TensorNetwork


def ipeps(
    hamiltonian_gate: jax.Array,
    initial_peps: TensorNetwork | jax.Array | Tensor | tuple | None,
    config: iPEPSConfig,
) -> tuple[float, TensorNetwork | Tensor, object]:
    """Run iPEPS simple update + CTM for a 2D quantum lattice model.

    Algorithm overview:

    1. Simple update (imaginary time evolution) -- apply ``exp(-dt * H_bond)``
       on each bond, SVD-truncate to D, update lambda matrices.
    2. CTM environment computation -- initialise and iteratively absorb
       rows/columns until convergence.
    3. Compute energy per site using the CTM environment.

    Args:
        hamiltonian_gate: The 2-site Hamiltonian as a 4-leg tensor of shape
                          (d, d, d, d) representing H on a bond.
        initial_peps:     TensorNetwork, raw JAX array, Tensor (DenseTensor or
                          SymmetricTensor), tuple for 2-site, or ``None``
                          for random initialization.
        config:           iPEPSConfig.

    Returns:
        (energy_per_site, optimized_peps, ctm_environment)
    """
    if config.unit_cell == "2site":
        init_2site = None
        if isinstance(initial_peps, tuple):
            init_2site = initial_peps
        return _ipeps_2site(hamiltonian_gate, init_2site, config)

    # Dispatch Tensor-protocol path (DenseTensor / SymmetricTensor)
    if isinstance(initial_peps, Tensor):
        return _ipeps_tensor(hamiltonian_gate, initial_peps, config)

    # Get site tensor
    if initial_peps is None:
        # Build random initial PEPS tensor
        key = jax.random.PRNGKey(0)
        D = config.max_bond_dim
        d_phys = hamiltonian_gate.shape[0]  # physical dimension from gate shape
        A_dense = jax.random.normal(key, (D, D, D, D, d_phys))
        A_dense = A_dense / (jnp.linalg.norm(A_dense) + 1e-10)
    elif isinstance(initial_peps, jax.Array):
        # Raw JAX array passed directly as the site tensor
        A_dense = initial_peps
        A_dense = A_dense / (jnp.linalg.norm(A_dense) + 1e-10)
    else:
        node_ids = initial_peps.node_ids()
        peps_tensors = {nid: initial_peps.get_tensor(nid) for nid in node_ids}

        # For simplicity, assume 1x1 unit cell with node_id (0,0)
        A_tensor = peps_tensors.get((0, 0))
        if A_tensor is None and len(peps_tensors) == 1:
            A_tensor = next(iter(peps_tensors.values()))

        if A_tensor is None:
            raise ValueError("iPEPS: could not find site tensor")

        A_dense = A_tensor.todense()
    gate = jnp.array(hamiltonian_gate)

    # Build Trotter gate: exp(-dt * H_bond)
    # Reshape gate (d,d,d,d) -> (d^2, d^2), diagonalize, exponentiate
    d = A_dense.shape[-1] if A_dense.ndim > 4 else 2  # physical dim
    d2 = d * d

    gate_matrix = gate.reshape(d2, d2)
    # Ensure Hermitian
    gate_matrix = 0.5 * (gate_matrix + gate_matrix.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(gate_matrix)
    trotter_gate_matrix = (
        eigvecs @ jnp.diag(jnp.exp(-config.dt * eigvals)) @ eigvecs.conj().T
    )
    trotter_gate = trotter_gate_matrix.reshape(d, d, d, d)

    # Initialize lambda matrices (identity = no environment approximation)
    D = config.max_bond_dim
    lambdas = {
        "horizontal": jnp.ones(D),
        "vertical": jnp.ones(D),
    }

    # Simple update iterations — alternate horizontal and vertical bonds
    for step in range(config.num_imaginary_steps):
        bond = "horizontal" if step % 2 == 0 else "vertical"
        A_dense, lambdas = _simple_update_1x1(
            A_dense,
            A_dense,
            lambdas,
            trotter_gate,
            config.max_bond_dim,
            bond=bond,
        )

    # Reconstruct PEPS tensor network with optimized tensor
    peps = _build_1x1_peps(A_dense, d, D)

    # CTM environment
    env = ctm(A_dense, config.ctm)

    # Compute energy
    energy = compute_energy_ctm(A_dense, env, gate, d)

    return float(energy), peps, env


def _ipeps_tensor(
    hamiltonian_gate: jax.Array,
    A_init: Tensor,
    config: iPEPSConfig,
) -> tuple[float, Tensor, object]:
    """Run iPEPS simple update + CTM for a Tensor-protocol site tensor.

    Works with DenseTensor and SymmetricTensor via polymorphic operations.

    Args:
        hamiltonian_gate: 2-site Hamiltonian (d,d,d,d).
        A_init:           Initial site tensor with labels (u, d, l, r, phys).
        config:           iPEPSConfig.

    Returns:
        (energy, A_opt, CTMTensorEnv)
    """
    from tenax.algorithms._ctm_tensor import (
        compute_energy_ctm_tensor,
        ctm_tensor,
    )

    D = config.max_bond_dim
    gate = _make_trotter_gate_tensor(hamiltonian_gate, config.dt, site_tensor=A_init)

    A = A_init
    norm_val = float(A.norm())
    if norm_val > EPS:
        A = A * (1.0 / norm_val)

    lam_h = jnp.ones(D)
    lam_v = jnp.ones(D)

    for step in range(config.num_imaginary_steps):
        if step % 2 == 0:
            A, lam_h = _simple_update_horizontal_tensor(A, gate, lam_h, lam_v, D)
        else:
            A, lam_v = _simple_update_vertical_tensor(A, gate, lam_h, lam_v, D)

    # Absorb lambdas for CTM
    A_abs = _absorb_lambdas_tensor(A, lam_h, lam_v)
    norm_val = float(A_abs.norm())
    if norm_val > EPS:
        A_abs = A_abs * (1.0 / norm_val)

    env = ctm_tensor(
        A_abs,
        chi=config.ctm.chi,
        max_iter=config.ctm.max_iter,
        conv_tol=config.ctm.conv_tol,
        renormalize=config.ctm.renormalize,
        projector_method=config.ctm.projector_method,
        qr_warmup_steps=config.ctm.qr_warmup_steps,
    )
    energy = compute_energy_ctm_tensor(A_abs, env, hamiltonian_gate)

    return float(energy), A, env


from tenax.algorithms.ipeps_ctm import (  # noqa: E402
    _build_double_layer,
    _ctm_2site_sweep,
    _ctm_bottom_move,
    _ctm_bottom_move_2site,
    _ctm_left_move,
    _ctm_left_move_2site,
    _ctm_move,
    _ctm_move_eigh,
    _ctm_move_qr,
    _ctm_right_move,
    _ctm_right_move_2site,
    _ctm_sv_diff,
    _ctm_sweep,
    _ctm_top_move,
    _ctm_top_move_2site,
    _initialize_ctm_env,
    _initialize_split_ctm_env,
    _renormalize_env,
    _split_ctm_move,
    _split_ctm_projector,
    _split_ctm_sweep,
    _split_env_to_standard,
    _svd_split_edge,
    ctm,
    ctm_2site,
    ctm_split,
)


def _build_1x1_peps(A: jax.Array, d: int, D: int) -> TensorNetwork:
    """Build a 1x1 unit cell PEPS TensorNetwork from a site tensor.

    Args:
        A: Site tensor.
        d: Physical dimension.
        D: Virtual bond dimension.

    Returns:
        TensorNetwork with a single node (0, 0).
    """
    sym = U1Symmetry()
    indices: tuple[TensorIndex, ...]

    if A.ndim == 3:
        # (D_l, D_r, d)
        D_l, D_r, d_actual = A.shape
        indices = (
            TensorIndex(
                sym, np.zeros(D_l, dtype=np.int32), FlowDirection.IN, label="left"
            ),
            TensorIndex(
                sym, np.zeros(D_r, dtype=np.int32), FlowDirection.OUT, label="right"
            ),
            TensorIndex(
                sym, np.zeros(d_actual, dtype=np.int32), FlowDirection.IN, label="phys"
            ),
        )
    elif A.ndim == 5:
        # (D_u, D_d, D_l, D_r, d)
        D_u, D_d, D_l, D_r, d_actual = A.shape
        indices = (
            TensorIndex(
                sym, np.zeros(D_u, dtype=np.int32), FlowDirection.IN, label="up"
            ),
            TensorIndex(
                sym, np.zeros(D_d, dtype=np.int32), FlowDirection.OUT, label="down"
            ),
            TensorIndex(
                sym, np.zeros(D_l, dtype=np.int32), FlowDirection.IN, label="left"
            ),
            TensorIndex(
                sym, np.zeros(D_r, dtype=np.int32), FlowDirection.OUT, label="right"
            ),
            TensorIndex(
                sym, np.zeros(d_actual, dtype=np.int32), FlowDirection.IN, label="phys"
            ),
        )
    else:
        # Generic fallback
        indices = tuple(
            TensorIndex(
                sym, np.zeros(s, dtype=np.int32), FlowDirection.IN, label=f"leg{i}"
            )
            for i, s in enumerate(A.shape)
        )

    peps = TensorNetwork(name="iPEPS_1x1")
    peps.add_node((0, 0), DenseTensor(A, indices))
    return peps


def _ipeps_2site(
    hamiltonian_gate: jax.Array,
    initial_peps: tuple[jax.Array, jax.Array] | None,
    config: iPEPSConfig,
) -> tuple[float, TensorNetwork, tuple[CTMEnvironment, CTMEnvironment]]:
    """Run iPEPS simple update + CTM for a 2-site checkerboard unit cell.

    Returns:
        (energy_per_site, peps_network, (env_A, env_B))
    """
    gate = jnp.array(hamiltonian_gate)
    d = gate.shape[0]
    D = config.max_bond_dim

    # Build Trotter gate
    d2 = d * d
    gate_matrix = gate.reshape(d2, d2)
    gate_matrix = 0.5 * (gate_matrix + gate_matrix.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(gate_matrix)
    trotter_gate = (
        eigvecs @ jnp.diag(jnp.exp(-config.dt * eigvals)) @ eigvecs.conj().T
    ).reshape(d, d, d, d)

    # Initialize A and B tensors
    if initial_peps is not None:
        A, B = initial_peps
        A = A / (jnp.linalg.norm(A) + 1e-10)
        B = B / (jnp.linalg.norm(B) + 1e-10)
    else:
        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        A = jax.random.normal(key_A, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)
        B = jax.random.normal(key_B, (D, D, D, D, d))
        B = B / (jnp.linalg.norm(B) + 1e-10)

    lambdas = {
        "horizontal": jnp.ones(D),
        "vertical": jnp.ones(D),
    }

    # Simple update iterations — alternate horizontal and vertical bonds
    for step in range(config.num_imaginary_steps):
        lam_h = lambdas["horizontal"]
        lam_v = lambdas["vertical"]
        if step % 2 == 0:
            A, B, lambdas = _simple_update_2site_horizontal(
                A,
                B,
                lam_h,
                lam_v,
                trotter_gate,
                D,
                lambdas,
            )
        else:
            A, B, lambdas = _simple_update_2site_vertical(
                A,
                B,
                lam_h,
                lam_v,
                trotter_gate,
                D,
                lambdas,
            )

    # Build PEPS TensorNetwork
    peps = TensorNetwork(name="iPEPS_2site")
    sym = U1Symmetry()
    for label, tensor in [((0, 0), A), ((1, 0), B)]:
        D_u, D_d, D_l, D_r, d_phys = tensor.shape
        indices = (
            TensorIndex(
                sym, np.zeros(D_u, dtype=np.int32), FlowDirection.IN, label="up"
            ),
            TensorIndex(
                sym, np.zeros(D_d, dtype=np.int32), FlowDirection.OUT, label="down"
            ),
            TensorIndex(
                sym, np.zeros(D_l, dtype=np.int32), FlowDirection.IN, label="left"
            ),
            TensorIndex(
                sym, np.zeros(D_r, dtype=np.int32), FlowDirection.OUT, label="right"
            ),
            TensorIndex(
                sym, np.zeros(d_phys, dtype=np.int32), FlowDirection.IN, label="phys"
            ),
        )
        peps.add_node(label, DenseTensor(tensor, indices))

    # CTM environment
    env_A, env_B = ctm_2site(A, B, config.ctm)

    # Compute energy
    energy = compute_energy_ctm_2site(A, B, env_A, env_B, gate, d)

    return float(energy), peps, (env_A, env_B)




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
