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

from tenax.algorithms.ipeps_config import CTMEnvironment, iPEPSConfig
from tenax.algorithms.ipeps_ctm import ctm, ctm_2site
from tenax.algorithms.ipeps_rdm import compute_energy_ctm, compute_energy_ctm_2site
from tenax.algorithms.ipeps_simple_update import (
    _absorb_lambdas_tensor,
    _make_trotter_gate_tensor,
    _simple_update_1x1,
    _simple_update_2site_horizontal,
    _simple_update_2site_vertical,
    _simple_update_horizontal_tensor,
    _simple_update_vertical_tensor,
)
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor
from tenax.network.network import TensorNetwork


def heisenberg_gate(dtype=jnp.float64) -> DenseTensor:
    """Build the 2-site Heisenberg Hamiltonian as a DenseTensor.

    ``H = Sz Sz + 0.5 (S+ S- + S- S+)`` on two spin-1/2 sites, returned
    as a 4-leg tensor with labels ``(si, sj, si_out, sj_out)``.
    """
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    sym = U1Symmetry()
    charges = np.zeros(2, dtype=np.int32)
    indices = (
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="si_out"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="sj_out"),
    )
    return DenseTensor(H.reshape(2, 2, 2, 2), indices)


def ipeps(
    hamiltonian_gate: Tensor | jax.Array,
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

    # Dense path: convert gate to JAX array if needed
    gate_arr = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )

    # Get site tensor
    if initial_peps is None:
        # Build random initial PEPS tensor
        key = jax.random.PRNGKey(0)
        D = config.max_bond_dim
        d_phys = gate_arr.shape[0]  # physical dimension from gate shape
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
    gate = gate_arr

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
    hamiltonian_gate: Tensor | jax.Array,
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
    hamiltonian_gate: Tensor | jax.Array,
    initial_peps: tuple[jax.Array, jax.Array] | None,
    config: iPEPSConfig,
) -> tuple[float, TensorNetwork, tuple[CTMEnvironment, CTMEnvironment]]:
    """Run iPEPS simple update + CTM for a 2-site checkerboard unit cell.

    Returns:
        (energy_per_site, peps_network, (env_A, env_B))
    """
    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
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
