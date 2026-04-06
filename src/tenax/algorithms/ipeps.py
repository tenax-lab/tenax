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
from tenax.algorithms.ipeps_ctm import ctm_2site
from tenax.algorithms.ipeps_rdm import compute_energy_ctm_2site
from tenax.algorithms.ipeps_simple_update import (
    _make_trotter_gate_tensor,
    _simple_update_2site_horizontal_tensor,
    _simple_update_2site_vertical_tensor,
)
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor


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
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="si_out"
        ),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="sj_out"
        ),
    )
    return DenseTensor(H.reshape(2, 2, 2, 2), indices)


def xxz_gate(delta: float = 1.0, dtype=jnp.float64) -> DenseTensor:
    """Build the 2-site XXZ Hamiltonian as a DenseTensor.

    ``H = delta * Sz Sz + 0.5 (S+ S- + S- S+)`` on two spin-1/2 sites.

    Args:
        delta: Anisotropy parameter. delta=1 is isotropic Heisenberg,
               delta=0 is XX model, delta->inf is Ising limit.
        dtype: Array dtype.

    Returns:
        4-leg DenseTensor with labels ``(si, sj, si_out, sj_out)``.
    """
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = delta * jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    sym = U1Symmetry()
    charges = np.zeros(2, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="si_out"
        ),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="sj_out"
        ),
    )
    return DenseTensor(H.reshape(2, 2, 2, 2), indices)


def _wrap_as_dense_tensor(arr: jax.Array) -> DenseTensor:
    """Wrap a raw ``jax.Array`` iPEPS site tensor as a ``DenseTensor``.

    Assumes shape ``(D, D, D, D, d)`` with trivial U(1) charges
    (all zeros) and labels ``(u, d, l, r, phys)``.
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


def neel_init(D: int, d: int = 2, seed: int = 0) -> tuple[jax.Array, jax.Array]:
    """Create Néel product state tensors for 2-site iPEPS.

    Sublattice A is initialized to spin-up and sublattice B to spin-down,
    with small random noise to break exact symmetry for simple update.

    Args:
        D: Bond dimension.
        d: Physical dimension (default 2 for spin-1/2).
        seed: Random seed for noise.

    Returns:
        ``(A, B)`` raw JAX arrays of shape ``(D, D, D, D, d)``.
    """
    key_A, key_B = jax.random.split(jax.random.PRNGKey(seed))
    noise = 0.01

    A = noise * jax.random.normal(key_A, (D, D, D, D, d))
    A = A.at[0, 0, 0, 0, 0].set(1.0)  # spin-up

    B = noise * jax.random.normal(key_B, (D, D, D, D, d))
    B = B.at[0, 0, 0, 0, 1].set(1.0)  # spin-down

    return A, B


def ipeps(
    hamiltonian_gate: Tensor | jax.Array,
    initial_peps: tuple[Tensor, Tensor] | tuple[jax.Array, jax.Array] | None,
    config: iPEPSConfig,
) -> tuple[float, tuple[Tensor, Tensor], tuple[CTMEnvironment, CTMEnvironment]]:
    """Run 2-site iPEPS simple update + CTM for a 2D quantum lattice model.

    Always uses the Tensor-protocol 2-site simple update path.  Raw JAX
    arrays are automatically wrapped as ``DenseTensor`` with trivial U(1)
    charges.

    Algorithm overview:

    1. Simple update (imaginary time evolution) -- apply ``exp(-dt * H_bond)``
       on each bond, SVD-truncate to D, update lambda matrices.
    2. CTM environment computation -- initialise and iteratively absorb
       rows/columns until convergence.
    3. Compute energy per site using the CTM environment.

    Args:
        hamiltonian_gate: The 2-site Hamiltonian as a 4-leg tensor of shape
                          ``(d, d, d, d)`` representing H on a bond.
        initial_peps:     Tuple ``(A, B)`` of site tensors (Tensor or raw
                          JAX arrays), or ``None`` for random initialization.
        config:           iPEPSConfig.

    Returns:
        ``(energy_per_site, (A, B), (env_A, env_B))``
    """
    if initial_peps is not None and not isinstance(initial_peps, tuple):
        raise TypeError(
            f"ipeps() requires initial_peps to be a tuple of two tensors "
            f"or None, got {type(initial_peps).__name__}"
        )

    # Convert gate to dense JAX array for CTM energy evaluation
    gate_dense = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d = gate_dense.shape[0]
    D = config.max_bond_dim

    # Initialize A and B tensors — default is Néel product state
    if initial_peps is not None:
        A_raw, B_raw = initial_peps
        A = A_raw if isinstance(A_raw, Tensor) else _wrap_as_dense_tensor(A_raw)
        B = B_raw if isinstance(B_raw, Tensor) else _wrap_as_dense_tensor(B_raw)
    else:
        A_data, B_data = neel_init(D, d)
        A = _wrap_as_dense_tensor(A_data)
        B = _wrap_as_dense_tensor(B_data)

    # Normalize
    norm_A = float(A.norm())
    if norm_A > EPS:
        A = A * (1.0 / norm_A)
    norm_B = float(B.norm())
    if norm_B > EPS:
        B = B * (1.0 / norm_B)

    # Build Trotter gate via Tensor protocol
    gate = _make_trotter_gate_tensor(hamiltonian_gate, config.dt, site_tensor=A)

    # Initialize lambdas from actual tensor bond dimensions
    _labels = A.labels()
    D_h = A.indices[_labels.index("r")].dim
    D_v = A.indices[_labels.index("d")].dim
    lam_h = jnp.ones(D_h)
    lam_v = jnp.ones(D_v)

    # Simple update iterations — alternate horizontal and vertical bonds
    for step in range(config.num_imaginary_steps):
        if step % 2 == 0:
            A, B, lam_h = _simple_update_2site_horizontal_tensor(
                A, B, gate, lam_h, lam_v, D
            )
        else:
            A, B, lam_v = _simple_update_2site_vertical_tensor(
                A, B, gate, lam_h, lam_v, D
            )

    # CTM environment (uses dense arrays)
    A_dense = A.todense()
    B_dense = B.todense()
    env_A, env_B = ctm_2site(A_dense, B_dense, config.ctm)

    # Compute energy
    energy = compute_energy_ctm_2site(A_dense, B_dense, env_A, env_B, gate_dense, d)

    return float(energy), (A, B), (env_A, env_B)
