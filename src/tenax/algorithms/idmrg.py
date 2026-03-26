"""Infinite Density Matrix Renormalization Group (iDMRG) algorithm.

Finds the ground-state energy per site of a translationally invariant 1D
Hamiltonian in the thermodynamic limit.  The Hamiltonian is specified by a
single bulk MPO tensor (the W-matrix repeated at every site).

The algorithm works on a 2-site unit cell, optimising a two-site wavefunction
at each step and growing the chain by two sites per iteration.  Left- and
right-canonical MPS tensors are obtained from the SVD of the optimised
wavefunction, and the environments are updated incrementally.

Architecture decisions mirror ``dmrg.py``:

- The outer loop is a Python for-loop (bond dimensions change after SVD).
- The effective-Hamiltonian matvec is JIT-compiled via the same helper used
  in finite DMRG.
- Environments are dense JAX arrays wrapped in ``DenseTensor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.dmrg import (
    _blockwise_contract,
    _lanczos_solve,
    _lanczos_solve_tensor,
    _update_left_env_symmetric,
    _update_right_env_symmetric,
)
from tenax.contraction.contractor import truncated_svd
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.mps import InfiniteMPS
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

# ---------------------------------------------------------------------------
# Config & Result
# ---------------------------------------------------------------------------


@dataclass
class iDMRGConfig:
    """Configuration for an iDMRG run.

    Attributes:
        max_bond_dim:    Maximum allowed bond dimension (chi).
        max_iterations:  Maximum number of sweeps (or growth steps for the
                         growing-chain variant).
        convergence_tol: Convergence threshold on energy per site.
        lanczos_max_iter: Maximum Lanczos iterations.
        lanczos_tol:     Lanczos convergence tolerance.
        svd_trunc_err:   Maximum SVD truncation error (None = use max_bond_dim).
        verbose:         Print per-step diagnostics.
        unit_cell_size:  Number of sites in the unit cell.
        two_site:        Use 2-site updates (True) or 1-site updates (False).
        arnoldi_tol:     Transfer matrix eigensolver tolerance.
        orthogonalize_interval: Number of sweeps between orthogonalization steps.
        mixing_factor:   DMRG3S subspace expansion mixing factor α (1-site only).
    """

    max_bond_dim: int = 100
    max_iterations: int = 200
    convergence_tol: float = 1e-8
    lanczos_max_iter: int = 50
    lanczos_tol: float = 1e-12
    svd_trunc_err: float | None = None
    verbose: bool = False
    unit_cell_size: int = 2
    two_site: bool = True
    arnoldi_tol: float = 1e-12
    orthogonalize_interval: int = 10
    mixing_factor: float = 0.05


class iDMRGResult(NamedTuple):
    """Result of an iDMRG run.

    Attributes:
        energy_per_site:    Converged energy per site.
        energies_per_step:  Energy-per-site estimate at each iteration.
        mps:                2-site unit cell as ``InfiniteMPS``.
        converged:          True if the run converged within tolerance.
    """

    energy_per_site: float
    energies_per_step: list[float]
    mps: InfiniteMPS
    converged: bool


# ---------------------------------------------------------------------------
# Bulk MPO builder
# ---------------------------------------------------------------------------


def build_bulk_mpo_heisenberg(
    Jz: float = 1.0,
    Jxy: float = 1.0,
    hz: float = 0.0,
    d: int = 2,
    dtype: Any = jnp.float64,
) -> DenseTensor:
    """Build a single bulk W-matrix for the spin-1/2 XXZ Heisenberg model.

    The returned tensor is the 5×d×d×5 MPO site tensor that is repeated at
    every site of an infinite chain.

    Args:
        Jz:    Ising coupling strength.
        Jxy:   XY coupling strength.
        hz:    Longitudinal magnetic field.
        d:     Physical dimension (must be 2).
        dtype: JAX dtype for the tensor data.

    Returns:
        ``DenseTensor`` with legs ``("w_l", "mpo_top", "mpo_bot", "w_r")``.
    """
    if d != 2:
        raise ValueError(f"build_bulk_mpo_heisenberg only supports d=2, got {d}")

    Sp = jnp.array([[0, 1], [0, 0]], dtype=dtype)
    Sm = jnp.array([[0, 0], [1, 0]], dtype=dtype)
    Sz = 0.5 * jnp.array([[1, 0], [0, -1]], dtype=dtype)
    I2 = jnp.eye(d, dtype=dtype)

    D_w = 5
    W = jnp.zeros((D_w, d, d, D_w), dtype=dtype)
    W = W.at[0, :, :, 0].set(I2)
    W = W.at[1, :, :, 0].set(Sp)
    W = W.at[2, :, :, 0].set(Sm)
    W = W.at[3, :, :, 0].set(Sz)
    W = W.at[4, :, :, 0].set(hz * Sz)
    W = W.at[4, :, :, 1].set((Jxy / 2) * Sm)
    W = W.at[4, :, :, 2].set((Jxy / 2) * Sp)
    W = W.at[4, :, :, 3].set(Jz * Sz)
    W = W.at[4, :, :, 4].set(I2)

    sym = U1Symmetry()
    bond_dw = np.zeros(D_w, dtype=np.int32)
    bond_d = np.zeros(d, dtype=np.int32)

    indices = (
        TensorIndex(sym, bond_dw, FlowDirection.IN, label="w_l"),
        TensorIndex(sym, bond_d, FlowDirection.IN, label="mpo_top"),
        TensorIndex(sym, bond_d, FlowDirection.OUT, label="mpo_bot"),
        TensorIndex(sym, bond_dw, FlowDirection.OUT, label="w_r"),
    )
    return DenseTensor(W, indices)


def build_bulk_mpo_heisenberg_cylinder(
    Ly: int,
    J: float = 1.0,
    dtype: Any = jnp.float64,
) -> DenseTensor:
    """Build a bulk W-matrix for the Heisenberg model on an infinite cylinder.

    Each "super-site" represents an entire ring of ``Ly`` spins (physical
    dimension ``d = 2**Ly``).  Within-ring Heisenberg bonds (periodic in y)
    become an on-site term, and between-ring bonds become nearest-neighbour
    MPO interactions.

    The resulting MPO tensor can be passed directly to :func:`idmrg`.

    Args:
        Ly:    Circumference of the cylinder (number of spins per ring).
        J:     Coupling constant (positive = antiferromagnetic).
        dtype: JAX dtype for the tensor data.

    Returns:
        ``DenseTensor`` with legs ``("w_l", "mpo_top", "mpo_bot", "w_r")``
        and shape ``(D_w, d, d, D_w)`` where ``D_w = 3*Ly + 2``, ``d = 2**Ly``.
    """
    if Ly < 1:
        raise ValueError(f"Ly must be >= 1, got {Ly}")
    if Ly % 2 != 0:
        raise ValueError(
            f"Ly must be even, got {Ly}. Odd circumference is incompatible "
            "with Néel (AFM) order on the square lattice because the periodic "
            "boundary creates frustrated odd-length cycles."
        )

    d = 2**Ly
    D_w = 3 * Ly + 2

    # --- Single-spin operators ---
    Sz_1 = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp_1 = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm_1 = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    I2 = jnp.eye(2, dtype=dtype)
    Id = jnp.eye(d, dtype=dtype)

    # --- Embed single-spin operator at position y in a ring of Ly spins ---
    def _embed(op_2x2: jax.Array, y: int) -> jax.Array:
        """Embed a 2x2 operator at position y via Kronecker products."""
        result = jnp.array([[1.0]], dtype=dtype)
        for k in range(Ly):
            result = jnp.kron(result, op_2x2 if k == y else I2)
        return result

    # Pre-compute embedded operators for each ring position
    Sz = [_embed(Sz_1, y) for y in range(Ly)]
    Sp = [_embed(Sp_1, y) for y in range(Ly)]
    Sm = [_embed(Sm_1, y) for y in range(Ly)]

    # --- Within-ring Heisenberg bonds (on-site term) ---
    # Each unique bond (y, y_next) is counted once.  For Ly=2 the wrap-around
    # (1→0) duplicates (0→1), so we track visited pairs to avoid overcounting.
    h_ring = jnp.zeros((d, d), dtype=dtype)
    seen_bonds: set[tuple[int, int]] = set()
    if Ly >= 2:
        for y in range(Ly):
            y_next = (y + 1) % Ly
            bond = (min(y, y_next), max(y, y_next))
            if bond in seen_bonds:
                continue
            seen_bonds.add(bond)
            h_ring = h_ring + J * (
                Sz[y] @ Sz[y_next] + 0.5 * (Sp[y] @ Sm[y_next] + Sm[y] @ Sp[y_next])
            )

    # --- Build MPO W-matrix ---
    # Layout: row 0 = "done", rows 1..Ly = Sz channels, rows Ly+1..2Ly = Sp
    # channels, rows 2Ly+1..3Ly = Sm channels, row D_w-1 = "vacuum".
    W = jnp.zeros((D_w, d, d, D_w), dtype=dtype)

    # done → done: identity
    W = W.at[0, :, :, 0].set(Id)
    # vacuum → vacuum: identity
    W = W.at[D_w - 1, :, :, D_w - 1].set(Id)
    # vacuum → done: within-ring Hamiltonian
    W = W.at[D_w - 1, :, :, 0].set(h_ring)

    for y in range(Ly):
        # Channel completions: channel → done
        W = W.at[y + 1, :, :, 0].set(Sz[y])  # Sz channel
        W = W.at[Ly + y + 1, :, :, 0].set(Sp[y])  # S+ channel (completes S-·S+)
        W = W.at[2 * Ly + y + 1, :, :, 0].set(Sm[y])  # S- channel (completes S+·S-)

        # Channel initiations: vacuum → channel
        W = W.at[D_w - 1, :, :, y + 1].set(J * Sz[y])  # vacuum → Sz
        W = W.at[D_w - 1, :, :, Ly + y + 1].set(
            (J / 2) * Sm[y]
        )  # vacuum → Sp (send Sm)
        W = W.at[D_w - 1, :, :, 2 * Ly + y + 1].set(
            (J / 2) * Sp[y]
        )  # vacuum → Sm (send Sp)

    # --- Wrap as DenseTensor ---
    sym = U1Symmetry()
    bond_dw = np.zeros(D_w, dtype=np.int32)
    bond_d = np.zeros(d, dtype=np.int32)

    indices = (
        TensorIndex(sym, bond_dw, FlowDirection.IN, label="w_l"),
        TensorIndex(sym, bond_d, FlowDirection.IN, label="mpo_top"),
        TensorIndex(sym, bond_d, FlowDirection.OUT, label="mpo_bot"),
        TensorIndex(sym, bond_dw, FlowDirection.OUT, label="w_r"),
    )
    return DenseTensor(W, indices)


def build_bulk_mpo_heisenberg_symmetric(
    Jz: float = 1.0,
    Jxy: float = 1.0,
    hz: float = 0.0,
    d: int = 2,
    dtype: Any = jnp.float64,
) -> SymmetricTensor:
    """Build a U(1) block-sparse bulk W-matrix for spin-1/2 XXZ Heisenberg.

    Same Hamiltonian as :func:`build_bulk_mpo_heisenberg` but returns a
    ``SymmetricTensor`` with proper U(1) charge assignments on every leg,
    enabling block-sparse iDMRG.

    MPO bond charges encode the finite automaton state:
    - State 0 (done):    charge  0
    - State 1 (Sp sent): charge -2
    - State 2 (Sm sent): charge +2
    - State 3 (Sz sent): charge  0
    - State 4 (vacuum):  charge  0

    Physical charges: spin up = +1, spin down = -1.

    Args:
        Jz, Jxy, hz, d, dtype: Same as :func:`build_bulk_mpo_heisenberg`.

    Returns:
        ``SymmetricTensor`` with legs ``("w_l", "mpo_top", "mpo_bot", "w_r")``.
    """
    if d != 2:
        raise ValueError(
            f"build_bulk_mpo_heisenberg_symmetric only supports d=2, got {d}"
        )

    Sp = jnp.array([[0, 1], [0, 0]], dtype=dtype)
    Sm = jnp.array([[0, 0], [1, 0]], dtype=dtype)
    Sz = 0.5 * jnp.array([[1, 0], [0, -1]], dtype=dtype)
    I2 = jnp.eye(d, dtype=dtype)

    D_w = 5
    W = jnp.zeros((D_w, d, d, D_w), dtype=dtype)
    W = W.at[0, :, :, 0].set(I2)
    W = W.at[1, :, :, 0].set(Sp)
    W = W.at[2, :, :, 0].set(Sm)
    W = W.at[3, :, :, 0].set(Sz)
    W = W.at[4, :, :, 0].set(hz * Sz)
    W = W.at[4, :, :, 1].set((Jxy / 2) * Sm)
    W = W.at[4, :, :, 2].set((Jxy / 2) * Sp)
    W = W.at[4, :, :, 3].set(Jz * Sz)
    W = W.at[4, :, :, 4].set(I2)

    sym = U1Symmetry()
    mpo_bond_charges = np.array([0, -2, 2, 0, 0], dtype=np.int32)
    phys_charges = np.array([1, -1], dtype=np.int32)
    indices = (
        TensorIndex(sym, mpo_bond_charges, FlowDirection.IN, label="w_l"),
        TensorIndex(sym, phys_charges, FlowDirection.IN, label="mpo_top"),
        TensorIndex(sym, phys_charges, FlowDirection.OUT, label="mpo_bot"),
        TensorIndex(sym, mpo_bond_charges, FlowDirection.OUT, label="w_r"),
    )
    return SymmetricTensor.from_dense(W, indices)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trivial_left_env(D_w: int, dtype: Any = jnp.float64) -> DenseTensor:
    """Trivial (1, D_w, 1) left environment for iDMRG."""
    sym = U1Symmetry()
    bond_mps = np.zeros(1, dtype=np.int32)
    bond_mpo = np.zeros(D_w, dtype=np.int32)
    data = jnp.zeros((1, D_w, 1), dtype=dtype)
    # Initialise: only the "vacuum" row (last index) is 1.
    # In the standard MPO convention, the vacuum state is the last row.
    data = data.at[0, D_w - 1, 0].set(1.0)
    indices = (
        TensorIndex(sym, bond_mps, FlowDirection.IN, label="env_mps_l"),
        TensorIndex(sym, bond_mpo, FlowDirection.IN, label="env_mpo_l"),
        TensorIndex(sym, bond_mps, FlowDirection.OUT, label="env_mps_conj_l"),
    )
    return DenseTensor(data, indices)


def _trivial_right_env(D_w: int, dtype: Any = jnp.float64) -> DenseTensor:
    """Trivial (1, D_w, 1) right environment for iDMRG."""
    sym = U1Symmetry()
    bond_mps = np.zeros(1, dtype=np.int32)
    bond_mpo = np.zeros(D_w, dtype=np.int32)
    data = jnp.zeros((1, D_w, 1), dtype=dtype)
    # Only the "done" row (index 0) is 1.
    data = data.at[0, 0, 0].set(1.0)
    indices = (
        TensorIndex(sym, bond_mps, FlowDirection.OUT, label="env_mps_r"),
        TensorIndex(sym, bond_mpo, FlowDirection.OUT, label="env_mpo_r"),
        TensorIndex(sym, bond_mps, FlowDirection.IN, label="env_mps_conj_r"),
    )
    return DenseTensor(data, indices)


def _idmrg_matvec(
    theta_flat: jax.Array,
    theta_shape: tuple[int, ...],
    L_env: jax.Array,
    W_l: jax.Array,
    W_r: jax.Array,
    R_env: jax.Array,
) -> jax.Array:
    """Apply effective Hamiltonian to 2-site wavefunction (iDMRG version)."""
    theta = theta_flat.reshape(theta_shape)
    result = jnp.einsum(
        "abc,apqd,bpse,eqtf,dfg->cstg",
        L_env,
        theta,
        W_l,
        W_r,
        R_env,
    )
    return result.ravel()


_idmrg_matvec_jit = jax.jit(_idmrg_matvec, static_argnums=(1,))


def _compute_local_energy(
    theta: jax.Array,
    W_bulk: jax.Array,
    d: int,
) -> float:
    """Compute the energy per site from the 2-site wavefunction.

    Evaluates ``<theta|H_bond|theta>`` where ``H_bond`` is the nearest-
    neighbour Hamiltonian extracted from the bulk MPO's vacuum→done
    transition.  For translationally invariant nearest-neighbour models,
    this equals the energy per bond = energy per site.

    Args:
        theta: Optimised 2-site wavefunction, shape (chi_l, d, d, chi_r).
        W_bulk: Bulk MPO tensor, shape (D_w, d, d, D_w).
        d: Physical dimension.

    Returns:
        Energy per site (float).
    """
    # Build 2-site Hamiltonian from the MPO: H[p,q,p',q'] = sum_e W[D-1,p,p',e] * W[e,q,q',0]
    # (vacuum row of left site → done column of right site).
    D_w = W_bulk.shape[0]
    W_left = W_bulk[D_w - 1, :, :, :]  # (d, d, D_w) — vacuum row
    W_right = W_bulk[:, :, :, 0]  # (D_w, d, d) — done column
    # H_2site[p, p', q, q'] = sum_e W_left[p, p', e] * W_right[e, q, q']
    H_2site = jnp.einsum("abe,ecd->abcd", W_left, W_right)
    # Contract indices: H_2site[p_top, p_bot, q_top, q_bot]

    # <theta|H_2site|theta> with theta[a, p, q, b]:
    # = sum_{a,b} sum_{p,q,p',q'} conj(theta[a,p',q',b]) * H[p',p,q',q] * theta[a,p,q,b]
    # Wait, let me be careful with bra vs ket indices.
    # H acts as: H|p,q> = sum_{p',q'} H[p',q',p,q] |p',q'>
    # <theta|H|theta> = sum_{a,b,p,q,p',q'} conj(theta[a,p',q',b]) * H[p',q',p,q] * theta[a,p,q,b]
    # H_2site[p_top, p_bot, q_top, q_bot]:
    #   p_top, q_top = bra (output) physical indices
    #   p_bot, q_bot = ket (input) physical indices
    energy = jnp.einsum(
        "asrb,PsQr,aPQb->",
        jnp.conj(theta),
        H_2site,
        theta,
    )
    norm = jnp.einsum("apqb,apqb->", jnp.conj(theta), theta)
    return float(energy / norm)


def _update_left_env_dense(
    L_env: jax.Array,
    A: jax.Array,
    W: jax.Array,
) -> jax.Array:
    """Update left environment (raw arrays, always 3-leg).

    L_env: (chi_l, D_w, chi_l)
    A:     (chi_l, d, chi_r)
    W:     (D_w_l, d, d, D_w_r)
    returns: (chi_r, D_w_r, chi_r)
    """
    return jnp.einsum("abc,apd,bpxe,cxf->def", L_env, A, W, jnp.conj(A))


def _update_right_env_dense(
    R_env: jax.Array,
    B: jax.Array,
    W: jax.Array,
) -> jax.Array:
    """Update right environment (raw arrays, always 3-leg).

    R_env: (chi_r, D_w, chi_r)
    B:     (chi_l, d, chi_r)
    W:     (D_w_l, d, d, D_w_r)
    returns: (chi_l, D_w_l, chi_l)
    """
    return jnp.einsum("abc,dpa,epxb,fxc->def", R_env, B, W, jnp.conj(B))


# ---------------------------------------------------------------------------
# Symmetric helpers
# ---------------------------------------------------------------------------


def _trivial_left_env_symmetric(
    mpo: SymmetricTensor, dtype: Any = jnp.float64
) -> SymmetricTensor:
    """Build trivial (1,D_w,1) left environment matching the MPO symmetry."""
    sym = mpo.indices[0].symmetry
    bond_mps = np.zeros(1, dtype=np.int32)
    mpo_charges = mpo.indices[0].charges  # w_l charges
    indices = (
        TensorIndex(sym, bond_mps, FlowDirection.IN, label="env_mps_l"),
        TensorIndex(sym, mpo_charges, FlowDirection.IN, label="env_mpo_l"),
        TensorIndex(sym, bond_mps, FlowDirection.OUT, label="env_mps_conj_l"),
    )
    # Only the vacuum row (last MPO state) is non-zero.
    # For Heisenberg: vacuum = state D_w-1, charge = 0.
    D_w = len(mpo_charges)
    data = jnp.zeros((1, D_w, 1), dtype=dtype)
    data = data.at[0, D_w - 1, 0].set(1.0)
    return SymmetricTensor.from_dense(data, indices)


def _trivial_right_env_symmetric(
    mpo: SymmetricTensor, dtype: Any = jnp.float64
) -> SymmetricTensor:
    """Build trivial (1,D_w,1) right environment matching the MPO symmetry."""
    sym = mpo.indices[3].symmetry
    bond_mps = np.zeros(1, dtype=np.int32)
    mpo_charges = mpo.indices[3].charges  # w_r charges
    indices = (
        TensorIndex(sym, bond_mps, FlowDirection.OUT, label="env_mps_r"),
        TensorIndex(sym, mpo_charges, FlowDirection.OUT, label="env_mpo_r"),
        TensorIndex(sym, bond_mps, FlowDirection.IN, label="env_mps_conj_r"),
    )
    # Only the "done" row (index 0) is non-zero.
    D_w = len(mpo_charges)
    data = jnp.zeros((1, D_w, 1), dtype=dtype)
    data = data.at[0, 0, 0].set(1.0)
    return SymmetricTensor.from_dense(data, indices)


def _idmrg_growing_chain_symmetric(
    bulk_mpo: SymmetricTensor,
    config: iDMRGConfig,
    d: int,
    dtype: Any,
) -> iDMRGResult:
    """Symmetric (block-sparse) growing-chain iDMRG implementation.

    Uses the same Lanczos + SVD + environment update approach as the dense
    path, but operates on SymmetricTensors throughout.  The block-sparse
    structure reduces computational cost when exploiting U(1) symmetry.
    """
    W_sym = bulk_mpo
    sym = W_sym.indices[0].symmetry
    phys_charges = np.array(W_sym.indices[1].charges)

    L_env: Tensor = _trivial_left_env_symmetric(W_sym, dtype=dtype)
    R_env: Tensor = _trivial_right_env_symmetric(W_sym, dtype=dtype)

    energies_per_step: list[float] = []
    converged = False
    key = jax.random.PRNGKey(0)
    E_prev: float | None = None

    # Virtual bond starts trivial: single charge-0 state.
    virt_charges = np.zeros(1, dtype=np.int32)

    # Store MPS tensors and previous theta for warm start
    A_L_tensor: Tensor | None = None
    A_R_tensor: Tensor | None = None
    s_vals = jnp.ones(1, dtype=dtype)
    theta_prev: Tensor | None = None

    for step in range(config.max_iterations):
        # ---- Build initial theta ----
        theta_indices = (
            TensorIndex(sym, virt_charges, FlowDirection.IN, label="v_l"),
            TensorIndex(sym, phys_charges, FlowDirection.IN, label="p_l"),
            TensorIndex(sym, phys_charges, FlowDirection.IN, label="p_r"),
            TensorIndex(sym, virt_charges, FlowDirection.OUT, label="v_r"),
        )

        if theta_prev is not None and np.array_equal(
            np.array(theta_prev.indices[0].charges), virt_charges
        ):
            # Warm start: reuse previous theta when charges match
            theta = theta_prev
        else:
            # Cold start: random init when bond dimension changed
            key, subkey = jax.random.split(key)
            theta = SymmetricTensor.random_normal(theta_indices, subkey, dtype=dtype)

        theta_norm = theta.norm()
        if theta_norm > 1e-15:
            theta = theta * (1.0 / theta_norm)

        # Shared cache for opt_einsum contraction expressions
        _matvec_cache: dict = {}

        def matvec(v: Tensor) -> Tensor:
            return _blockwise_contract(
                [L_env, v, W_sym, W_sym, R_env],
                "abc,apqd,bpse,eqtf,dfg->cstg",
                output_indices=v.indices,
                expr_cache=_matvec_cache,
            )

        E_total_val, theta_opt = _lanczos_solve_tensor(
            matvec, theta, config.lanczos_max_iter, config.lanczos_tol
        )
        E_total = float(E_total_val)
        theta_prev = theta_opt

        # ---- SVD and truncate ----
        U_t, s_full, Vh_t, _ = truncated_svd(
            theta_opt,
            left_labels=["v_l", "p_l"],
            right_labels=["p_r", "v_r"],
            new_bond_label="v_c",
            max_singular_values=config.max_bond_dim,
            max_truncation_err=config.svd_trunc_err,
        )
        # U_t: (v_l, p_l, v_c),  Vh_t: (v_c, p_r, v_r)
        s_vals = s_full
        s_norm = float(jnp.linalg.norm(s_vals))
        if s_norm > 1e-15:
            s_vals = s_vals / s_norm

        # A_L = U (left-isometric), A_R = Vh (right-isometric)
        A_L_tensor = U_t
        A_R_tensor = Vh_t

        # ---- Update environments ----
        L_env = _update_left_env_symmetric(L_env, A_L_tensor, W_sym)
        R_env = _update_right_env_symmetric(R_env, A_R_tensor, W_sym)

        # Update virtual charges for next step from the SVD bond
        virt_charges = np.array(A_L_tensor.indices[2].charges)

        # ---- Compute energy per site ----
        if E_prev is not None:
            e_per_site = (E_total - E_prev) / 2.0
        else:
            e_per_site = E_total / 2.0
        energies_per_step.append(e_per_site)

        if config.verbose:
            n_keep = len(s_vals)
            print(
                f"iDMRG step {step + 1}: E_total={E_total:.10f}, "
                f"e/site={e_per_site:.10f}, chi={n_keep}",
                flush=True,
            )

        # ---- Check convergence ----
        n_e = len(energies_per_step)
        if n_e >= 4:
            n_half = min(n_e // 2, 5)
            avg_recent = sum(energies_per_step[-n_half:]) / n_half
            avg_prev = sum(energies_per_step[-2 * n_half : -n_half]) / n_half
            if abs(avg_recent - avg_prev) < config.convergence_tol:
                converged = True
                if config.verbose:
                    print(f"Converged at step {step + 1}", flush=True)
                break

        E_prev = E_total

    n_avg = max(len(energies_per_step) // 2, 1)
    e_per_site_avg = sum(energies_per_step[-n_avg:]) / n_avg

    return iDMRGResult(
        energy_per_site=e_per_site_avg,
        energies_per_step=energies_per_step,
        mps=InfiniteMPS.from_tensors(
            [A_L_tensor, A_R_tensor],
            [s_vals, s_vals, s_vals],  # L+1 = 3 bonds for 2-site cell
        ),
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def _idmrg_growing_chain(
    bulk_mpo_dense: jax.Array,
    config: iDMRGConfig,
    d: int,
    dtype: Any,
) -> iDMRGResult:
    """Dense growing-chain iDMRG implementation.

    Grows the chain by two sites per iteration, optimising a two-site
    wavefunction via Lanczos and updating environments incrementally.

    Args:
        bulk_mpo_dense: Raw dense MPO array, shape (D_w, d, d, D_w).
        config: iDMRG configuration.
        d: Physical dimension.
        dtype: JAX dtype for computation.

    Returns:
        ``iDMRGResult`` with energy per site and diagnostic information.
    """
    W = bulk_mpo_dense
    D_w = W.shape[0]

    # ---- Initialise environments ----
    L_env = _trivial_left_env(D_w, dtype=dtype).todense()  # (1, D_w, 1)
    R_env = _trivial_right_env(D_w, dtype=dtype).todense()  # (1, D_w, 1)

    energies_per_step: list[float] = []
    e_per_site = 0.0
    converged = False
    chi_env = 1
    key = jax.random.PRNGKey(0)
    s_vals = jnp.ones(1, dtype=dtype)
    theta_prev: jax.Array | None = None
    E_prev: float | None = None  # previous Lanczos eigenvalue

    for step in range(config.max_iterations):
        # ---- Form initial two-site wavefunction theta ----
        if theta_prev is not None and theta_prev.shape == (chi_env, d, d, chi_env):
            theta = theta_prev
        elif theta_prev is not None:
            old_chi = theta_prev.shape[0]
            theta = jnp.zeros((chi_env, d, d, chi_env), dtype=dtype)
            theta = theta.at[:old_chi, :, :, :old_chi].set(theta_prev)
            key, subkey = jax.random.split(key)
            noise = 1e-3 * jax.random.normal(
                subkey, (chi_env, d, d, chi_env), dtype=dtype
            )
            theta = theta + noise
        else:
            key, subkey = jax.random.split(key)
            theta = jax.random.normal(subkey, (chi_env, d, d, chi_env), dtype=dtype)
        theta = theta / jnp.linalg.norm(theta)

        theta_shape = theta.shape
        theta_flat = theta.ravel()

        # ---- Solve eigenvalue problem via Lanczos ----
        _ts = theta_shape
        _le = L_env
        _re = R_env

        def matvec(v: jax.Array) -> jax.Array:
            return _idmrg_matvec_jit(v, _ts, _le, W, W, _re)

        E_total, theta_opt_flat = _lanczos_solve(
            matvec, theta_flat, config.lanczos_max_iter, config.lanczos_tol
        )
        E_total = float(E_total)

        theta_opt = theta_opt_flat.reshape(theta_shape)

        # ---- SVD and truncate ----
        chi_l, d_l, d_r, chi_r = theta_shape
        matrix = theta_opt.reshape(chi_l * d_l, d_r * chi_r)
        U, s_full, Vt = jnp.linalg.svd(matrix, full_matrices=False)

        n_keep = min(config.max_bond_dim, len(s_full))
        if config.svd_trunc_err is not None:
            total_sq = jnp.sum(s_full**2)
            cumul_sq = jnp.cumsum(s_full[::-1] ** 2)[::-1]
            mask = cumul_sq > (config.svd_trunc_err**2 * total_sq)
            n_by_err = max(int(jnp.sum(mask)), 1)
            n_keep = min(n_keep, n_by_err)

        U = U[:, :n_keep]
        s_vals = s_full[:n_keep]
        Vt = Vt[:n_keep, :]

        # Normalise singular values
        s_norm = jnp.linalg.norm(s_vals)
        if s_norm > 1e-15:
            s_vals = s_vals / s_norm

        # A_L: left-isometric (from U columns)
        A_L = U.reshape(chi_l, d_l, n_keep)
        # A_R_iso: right-isometric (from Vt rows, no singular values)
        A_R_iso = Vt.reshape(n_keep, d_r, chi_r)

        # ---- Update environments with isometric tensors ----
        L_env_new = _update_left_env_dense(L_env, A_L, W)
        R_env_new = _update_right_env_dense(R_env, A_R_iso, W)

        # ---- Compute energy per site via energy difference ----
        if E_prev is not None:
            e_per_site = (E_total - E_prev) / 2.0
        else:
            e_per_site = E_total / 2.0
        energies_per_step.append(e_per_site)

        if config.verbose:
            print(
                f"iDMRG step {step + 1}: E_total={E_total:.10f}, "
                f"e/site={e_per_site:.10f}, chi={n_keep}",
                flush=True,
            )

        # ---- Check convergence (rolling average to handle oscillation) ----
        n_e = len(energies_per_step)
        if n_e >= 4:
            n_half = min(n_e // 2, 5)
            avg_recent = sum(energies_per_step[-n_half:]) / n_half
            avg_prev = sum(energies_per_step[-2 * n_half : -n_half]) / n_half
            if abs(avg_recent - avg_prev) < config.convergence_tol:
                converged = True
                if config.verbose:
                    print(f"Converged at step {step + 1}", flush=True)
                break

        # ---- Prepare for next iteration ----
        E_prev = E_total
        theta_prev = theta_opt
        chi_env = n_keep
        L_env = L_env_new
        R_env = R_env_new

    # ---- Wrap final MPS tensors ----
    sym = U1Symmetry()

    def _wrap_mps(data: jax.Array, labels: tuple[str, ...]) -> DenseTensor:
        indices = tuple(
            TensorIndex(
                sym,
                np.zeros(data.shape[k], dtype=np.int32),
                FlowDirection.IN if k < data.ndim - 1 else FlowDirection.OUT,
                label=labels[k],
            )
            for k in range(data.ndim)
        )
        return DenseTensor(data, indices)

    A_L_tensor = _wrap_mps(A_L, ("v_l", "p_l", "v_c"))
    # Return A_R with singular values absorbed for a complete MPS
    A_R_sv = (jnp.diag(s_vals) @ Vt).reshape(n_keep, d, chi_r)
    A_R_tensor = _wrap_mps(A_R_sv, ("v_c", "p_r", "v_r"))

    # Report energy as average of last half of steps to smooth oscillation
    n_avg = max(len(energies_per_step) // 2, 1)
    e_per_site_avg = sum(energies_per_step[-n_avg:]) / n_avg

    return iDMRGResult(
        energy_per_site=e_per_site_avg,
        energies_per_step=energies_per_step,
        mps=InfiniteMPS.from_tensors(
            [A_L_tensor, A_R_tensor],
            [s_vals, s_vals, s_vals],  # L+1 = 3 bonds for 2-site cell
        ),
        converged=converged,
    )


def _orthogonalize_unit_cell_dense(
    A_L: np.ndarray,
    A_R: np.ndarray,
    s_vals: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-orthogonalize a 2-site unit cell MPS via QR decomposition.

    After many sweeps, truncation errors cause the MPS to drift from exact
    orthogonality.  This function restores left/right-canonical form by
    QR-decomposing each site tensor and absorbing the remainder into the
    centre bond, then re-SVD'ing to get a diagonal centre bond.

    All inputs/outputs are numpy arrays.

    Args:
        A_L: Left-canonical tensor, shape (chi_l, d, chi_r).
        A_R: Right-canonical tensor, shape (chi_l, d, chi_r).
        s_vals: Singular values at the centre bond, shape (chi,).
        tol: (unused, kept for API compatibility).

    Returns:
        Tuple of (A_L, A_R, s_vals) after orthogonalization.
    """
    chi_l, d, chi_r = A_L.shape
    if chi_r <= 1:
        return A_L, A_R, s_vals

    # ---- Make A_L left-canonical via QR ----
    # Reshape (chi_l, d, chi_r) -> (chi_l * d, chi_r)
    A_L_mat = A_L.reshape(chi_l * d, chi_r)
    Q_L, R_L = np.linalg.qr(A_L_mat)  # Q: (chi_l*d, chi_r), R: (chi_r, chi_r)
    A_L_new = Q_L.reshape(chi_l, d, chi_r)

    # ---- Make A_R right-canonical via RQ (= QR of transpose) ----
    chi_l_r, d_r, chi_r_r = A_R.shape
    A_R_mat = A_R.reshape(chi_l_r, d_r * chi_r_r)
    # RQ decomposition: A_R_mat = L_R @ Q_R
    # Use QR on transpose: A_R_mat^T = Q_R^T @ L_R^T
    Q_Rt, L_Rt = np.linalg.qr(A_R_mat.T)
    Q_R = Q_Rt.T  # (chi_l_r, d_r * chi_r_r)
    L_R = L_Rt.T  # (chi_l_r, chi_l_r)
    A_R_new = Q_R.reshape(chi_l_r, d_r, chi_r_r)

    # ---- Re-build centre bond: absorb R_L and L_R, then re-SVD ----
    # Original: ... A_L @ diag(s) @ A_R ...
    # New:      ... Q_L @ (R_L @ diag(s) @ L_R) @ Q_R ...
    S_matrix = R_L @ np.diag(s_vals) @ L_R
    U_s, s_new, Vt_s = np.linalg.svd(S_matrix, full_matrices=False)
    s_norm = np.linalg.norm(s_new)
    if s_norm > 1e-15:
        s_new = s_new / s_norm

    # Absorb U_s and Vt_s into A_L and A_R
    A_L_final = np.empty_like(A_L)
    for s in range(d):
        A_L_final[:, s, :] = A_L_new[:, s, :] @ U_s

    A_R_final = np.empty_like(A_R)
    for s in range(d_r):
        A_R_final[:, s, :] = Vt_s @ A_R_new[:, s, :]

    return A_L_final, A_R_final, s_new


def _idmrg_sweep(
    bulk_mpo: Tensor,
    config: iDMRGConfig,
    d: int,
    dtype: Any,
) -> iDMRGResult:
    """Sweep-based iDMRG for a 2-site unit cell (dense path).

    Instead of growing the chain each step, maintains a fixed 2-site unit cell
    with left/right environments that wrap around to represent the infinite
    half-chains. Each sweep optimises the 2-site wavefunction in L->R then R->L
    direction, updating environments after each half-sweep.

    Args:
        bulk_mpo: Bulk MPO tensor as a ``Tensor`` wrapper.
        config: iDMRG configuration.
        d: Physical dimension.
        dtype: JAX dtype for computation.

    Returns:
        ``iDMRGResult`` with energy per site and diagnostic information.
    """
    W = bulk_mpo.todense()
    D_w = W.shape[0]
    key = jax.random.PRNGKey(0)

    # ---- Phase 1: Grow bond dimension from chi=1 to max_bond_dim ----
    L_env = _trivial_left_env(D_w, dtype=dtype).todense()  # (1, D_w, 1)
    R_env = _trivial_right_env(D_w, dtype=dtype).todense()  # (1, D_w, 1)

    chi_env = 1
    s_center = jnp.ones(1, dtype=dtype)
    A_L: jax.Array | None = None
    A_R: jax.Array | None = None
    theta_prev: jax.Array | None = None
    E_prev: float | None = None

    warmup_steps = 0
    while chi_env < config.max_bond_dim:
        # Form initial theta
        if theta_prev is not None and theta_prev.shape == (chi_env, d, d, chi_env):
            theta = theta_prev
        elif theta_prev is not None:
            old_chi = theta_prev.shape[0]
            theta = jnp.zeros((chi_env, d, d, chi_env), dtype=dtype)
            theta = theta.at[:old_chi, :, :, :old_chi].set(theta_prev)
            key, subkey = jax.random.split(key)
            noise = 1e-3 * jax.random.normal(
                subkey, (chi_env, d, d, chi_env), dtype=dtype
            )
            theta = theta + noise
        else:
            key, subkey = jax.random.split(key)
            theta = jax.random.normal(subkey, (chi_env, d, d, chi_env), dtype=dtype)
        theta = theta / jnp.linalg.norm(theta)

        theta_shape = theta.shape
        theta_flat = theta.ravel()

        _ts = theta_shape
        _le = L_env
        _re = R_env

        def matvec(v: jax.Array) -> jax.Array:
            return _idmrg_matvec_jit(v, _ts, _le, W, W, _re)

        E_total, theta_opt_flat = _lanczos_solve(
            matvec, theta_flat, config.lanczos_max_iter, config.lanczos_tol
        )
        E_total = float(E_total)
        theta_opt = theta_opt_flat.reshape(theta_shape)

        chi_l, d_l, d_r, chi_r = theta_shape
        matrix = theta_opt.reshape(chi_l * d_l, d_r * chi_r)
        U, s_full, Vt = jnp.linalg.svd(matrix, full_matrices=False)

        n_keep = min(config.max_bond_dim, len(s_full))
        U = U[:, :n_keep]
        s_center = s_full[:n_keep]
        Vt = Vt[:n_keep, :]
        s_norm = jnp.linalg.norm(s_center)
        if s_norm > 1e-15:
            s_center = s_center / s_norm

        A_L = U.reshape(chi_l, d, n_keep)
        A_R = Vt.reshape(n_keep, d, chi_r)

        L_env = _update_left_env_dense(L_env, A_L, W)
        R_env = _update_right_env_dense(R_env, A_R, W)

        E_prev = E_total
        theta_prev = theta_opt
        chi_env = n_keep
        warmup_steps += 1

        if config.verbose:
            print(
                f"  Warmup step {warmup_steps}: chi={n_keep}, E_total={E_total:.10f}",
                flush=True,
            )

    # ---- Phase 2: First optimization at target chi + environment warmup ----
    # Do one optimization step at full chi to get square A_L/A_R tensors,
    # then run environment-only warmup sweeps for self-consistency.
    assert A_L is not None and A_R is not None

    # First optimization at target chi (A_L/A_R from growth may be non-square)
    old_chi = theta_prev.shape[0] if theta_prev is not None else 1
    theta = jnp.zeros((chi_env, d, d, chi_env), dtype=dtype)
    if theta_prev is not None:
        theta = theta.at[:old_chi, :, :, :old_chi].set(theta_prev)
    key, subkey = jax.random.split(key)
    noise = 1e-3 * jax.random.normal(subkey, (chi_env, d, d, chi_env), dtype=dtype)
    theta = theta + noise
    theta = theta / jnp.linalg.norm(theta)

    theta_shape = theta.shape
    _ts = theta_shape
    _le = L_env
    _re = R_env

    def _matvec_init(v: jax.Array) -> jax.Array:
        return _idmrg_matvec_jit(v, _ts, _le, W, W, _re)

    E_total, theta_opt_flat = _lanczos_solve(
        _matvec_init, theta.ravel(), config.lanczos_max_iter, config.lanczos_tol
    )
    E_total = float(E_total)
    theta_opt = theta_opt_flat.reshape(theta_shape)

    chi_l, d_l, d_r, chi_r = theta_shape
    matrix = theta_opt.reshape(chi_l * d_l, d_r * chi_r)
    U, s_full, Vt = jnp.linalg.svd(matrix, full_matrices=False)
    n_keep = min(config.max_bond_dim, len(s_full))
    U = U[:, :n_keep]
    s_center = s_full[:n_keep]
    Vt = Vt[:n_keep, :]
    s_norm = jnp.linalg.norm(s_center)
    if s_norm > 1e-15:
        s_center = s_center / s_norm
    A_L = U.reshape(chi_l, d, n_keep)
    A_R = Vt.reshape(n_keep, d, chi_r)
    L_env = _update_left_env_dense(L_env, A_L, W)
    R_env = _update_right_env_dense(R_env, A_R, W)
    chi_env = n_keep

    # Environment warmup: contract A_L/A_R through environments repeatedly
    # to build self-consistent infinite environments before optimization.
    n_env_warmup = 10
    for warmup in range(n_env_warmup):
        L_env = _update_left_env_dense(L_env, A_L, W)
        R_env = _update_right_env_dense(R_env, A_R, W)
        if config.verbose:
            print(
                f"  Env warmup {warmup + 1}/{n_env_warmup}: L_env shape={L_env.shape}",
                flush=True,
            )

    # ---- Phase 3: Self-consistent optimization sweeps ----
    energies_per_step: list[float] = []
    converged = False
    n_keep = chi_env
    E_prev = None

    for sweep in range(config.max_iterations):
        # Form theta from current MPS tensors
        theta = jnp.einsum("ija,a,akl->ijkl", A_L, s_center, A_R)
        theta = theta / jnp.linalg.norm(theta)

        theta_shape = theta.shape
        theta_flat = theta.ravel()

        _ts = theta_shape
        _le = L_env
        _re = R_env

        def matvec(v: jax.Array) -> jax.Array:
            return _idmrg_matvec_jit(v, _ts, _le, W, W, _re)

        E_total, theta_opt_flat = _lanczos_solve(
            matvec, theta_flat, config.lanczos_max_iter, config.lanczos_tol
        )
        E_total = float(E_total)
        theta_opt = theta_opt_flat.reshape(theta_shape)

        # ---- SVD and truncate ----
        chi_l, d_l, d_r, chi_r = theta_shape
        matrix = theta_opt.reshape(chi_l * d_l, d_r * chi_r)
        U, s_full, Vt = jnp.linalg.svd(matrix, full_matrices=False)

        n_keep = min(config.max_bond_dim, len(s_full))
        if config.svd_trunc_err is not None:
            total_sq = jnp.sum(s_full**2)
            cumul_sq = jnp.cumsum(s_full[::-1] ** 2)[::-1]
            mask = cumul_sq > (config.svd_trunc_err**2 * total_sq)
            n_by_err = max(int(jnp.sum(mask)), 1)
            n_keep = min(n_keep, n_by_err)

        U = U[:, :n_keep]
        s_center = s_full[:n_keep]
        Vt = Vt[:n_keep, :]

        s_norm = jnp.linalg.norm(s_center)
        if s_norm > 1e-15:
            s_center = s_center / s_norm

        A_L = U.reshape(chi_l, d, n_keep)
        A_R = Vt.reshape(n_keep, d, chi_r)

        # ---- Update both environments with optimized tensors ----
        L_env = _update_left_env_dense(L_env, A_L, W)
        R_env = _update_right_env_dense(R_env, A_R, W)

        # ---- Energy per site via energy difference ----
        # Each sweep adds 2 sites to the effective chain via environment updates,
        # so the energy per site is the change in total energy divided by 2.
        if E_prev is not None:
            e_per_site = (E_total - E_prev) / 2.0
        else:
            e_per_site = E_total / 2.0
        energies_per_step.append(e_per_site)

        # ---- Prepare for next sweep ----
        E_prev = E_total
        chi_env = n_keep

        if config.verbose:
            print(
                f"iDMRG sweep {sweep + 1}: E_total={E_total:.10f}, "
                f"e/site={e_per_site:.10f}, chi={n_keep}",
                flush=True,
            )

        # ---- Check convergence ----
        n_e = len(energies_per_step)
        if n_e >= 4:
            n_half = min(n_e // 2, 5)
            avg_recent = sum(energies_per_step[-n_half:]) / n_half
            avg_prev = sum(energies_per_step[-2 * n_half : -n_half]) / n_half
            if abs(avg_recent - avg_prev) < config.convergence_tol:
                converged = True
                if config.verbose:
                    print(f"Converged at sweep {sweep + 1}", flush=True)
                break

    # ---- Final orthogonalization ----
    if (
        config.orthogonalize_interval > 0
        and chi_env > 1
        and A_L.shape[0] == A_L.shape[2]
    ):
        A_L_np, A_R_np, s_np = _orthogonalize_unit_cell_dense(
            np.array(A_L),
            np.array(A_R),
            np.array(s_center),
            tol=config.arnoldi_tol,
        )
        A_L = jnp.array(A_L_np)
        A_R = jnp.array(A_R_np)
        s_center = jnp.array(s_np)

        if config.verbose:
            print("  Applied final orthogonalization", flush=True)

    # ---- Wrap final MPS tensors ----
    sym = U1Symmetry()

    def _wrap_mps(data: jax.Array, labels: tuple[str, ...]) -> DenseTensor:
        indices = tuple(
            TensorIndex(
                sym,
                np.zeros(data.shape[k], dtype=np.int32),
                FlowDirection.IN if k < data.ndim - 1 else FlowDirection.OUT,
                label=labels[k],
            )
            for k in range(data.ndim)
        )
        return DenseTensor(data, indices)

    A_L_tensor = _wrap_mps(A_L, ("v_l", "p_l", "v_c"))
    # Absorb singular values into A_R for a complete MPS
    A_R_sv = jnp.einsum("a,apb->apb", s_center, A_R)
    A_R_tensor = _wrap_mps(A_R_sv, ("v_c", "p_r", "v_r"))

    n_avg = max(len(energies_per_step) // 2, 1)
    e_per_site_avg = sum(energies_per_step[-n_avg:]) / n_avg

    return iDMRGResult(
        energy_per_site=e_per_site_avg,
        energies_per_step=energies_per_step,
        mps=InfiniteMPS.from_tensors(
            [A_L_tensor, A_R_tensor],
            [s_center, s_center, s_center],
        ),
        converged=converged,
    )


def _idmrg_sweep_symmetric(
    bulk_mpo: SymmetricTensor,
    config: iDMRGConfig,
    d: int,
    dtype: Any,
) -> iDMRGResult:
    """Sweep-based iDMRG for a 2-site unit cell (symmetric/block-sparse path).

    Mirrors ``_idmrg_sweep`` but operates on ``SymmetricTensor`` throughout,
    preserving block-sparse structure and U(1) quantum numbers.  Uses the same
    Lanczos + SVD + environment update approach as the growing-chain symmetric
    variant but in a sweep-based fashion.

    Args:
        bulk_mpo: Bulk MPO tensor as a ``SymmetricTensor``.
        config: iDMRG configuration.
        d: Physical dimension.
        dtype: JAX dtype for computation.

    Returns:
        ``iDMRGResult`` with energy per site and diagnostic information.
    """
    W_sym = bulk_mpo
    sym = W_sym.indices[0].symmetry
    phys_charges = np.array(W_sym.indices[1].charges)

    L_env: Tensor = _trivial_left_env_symmetric(W_sym, dtype=dtype)
    R_env: Tensor = _trivial_right_env_symmetric(W_sym, dtype=dtype)

    energies_per_step: list[float] = []
    converged = False
    key = jax.random.PRNGKey(0)
    E_prev: float | None = None

    # Virtual bond starts trivial: single charge-0 state.
    virt_charges = np.zeros(1, dtype=np.int32)

    # Store MPS tensors and previous theta for warm start
    A_L_tensor: Tensor | None = None
    A_R_tensor: Tensor | None = None
    s_vals = jnp.ones(1, dtype=dtype)
    theta_prev: Tensor | None = None

    for sweep in range(config.max_iterations):
        # ---- Build initial theta ----
        theta_indices = (
            TensorIndex(sym, virt_charges, FlowDirection.IN, label="v_l"),
            TensorIndex(sym, phys_charges, FlowDirection.IN, label="p_l"),
            TensorIndex(sym, phys_charges, FlowDirection.IN, label="p_r"),
            TensorIndex(sym, virt_charges, FlowDirection.OUT, label="v_r"),
        )

        if theta_prev is not None and np.array_equal(
            np.array(theta_prev.indices[0].charges), virt_charges
        ):
            # Warm start: reuse previous theta when charges match
            theta = theta_prev
        else:
            # Cold start: random init when bond dimension changed
            key, subkey = jax.random.split(key)
            theta = SymmetricTensor.random_normal(theta_indices, subkey, dtype=dtype)

        theta_norm = theta.norm()
        if theta_norm > 1e-15:
            theta = theta * (1.0 / theta_norm)

        # Shared cache for opt_einsum contraction expressions
        _matvec_cache: dict = {}

        def matvec(v: Tensor) -> Tensor:
            return _blockwise_contract(
                [L_env, v, W_sym, W_sym, R_env],
                "abc,apqd,bpse,eqtf,dfg->cstg",
                output_indices=v.indices,
                expr_cache=_matvec_cache,
            )

        E_total_val, theta_opt = _lanczos_solve_tensor(
            matvec, theta, config.lanczos_max_iter, config.lanczos_tol
        )
        E_total = float(E_total_val)
        theta_prev = theta_opt

        # ---- SVD and truncate ----
        U_t, s_full, Vh_t, _ = truncated_svd(
            theta_opt,
            left_labels=["v_l", "p_l"],
            right_labels=["p_r", "v_r"],
            new_bond_label="v_c",
            max_singular_values=config.max_bond_dim,
            max_truncation_err=config.svd_trunc_err,
        )
        # U_t: (v_l, p_l, v_c),  Vh_t: (v_c, p_r, v_r)
        s_vals = s_full
        s_norm = float(jnp.linalg.norm(s_vals))
        if s_norm > 1e-15:
            s_vals = s_vals / s_norm

        # A_L = U (left-isometric), A_R = Vh (right-isometric)
        A_L_tensor = U_t
        A_R_tensor = Vh_t

        # ---- Update environments ----
        L_env = _update_left_env_symmetric(L_env, A_L_tensor, W_sym)
        R_env = _update_right_env_symmetric(R_env, A_R_tensor, W_sym)

        # Update virtual charges for next step from the SVD bond
        virt_charges = np.array(A_L_tensor.indices[2].charges)

        # ---- Compute energy per site via energy difference ----
        if E_prev is not None:
            e_per_site = (E_total - E_prev) / 2.0
        else:
            e_per_site = E_total / 2.0
        energies_per_step.append(e_per_site)

        if config.verbose:
            n_keep = len(s_vals)
            print(
                f"iDMRG sweep {sweep + 1}: E_total={E_total:.10f}, "
                f"e/site={e_per_site:.10f}, chi={n_keep}",
                flush=True,
            )

        # ---- Check convergence ----
        n_e = len(energies_per_step)
        if n_e >= 4:
            n_half = min(n_e // 2, 5)
            avg_recent = sum(energies_per_step[-n_half:]) / n_half
            avg_prev = sum(energies_per_step[-2 * n_half : -n_half]) / n_half
            if abs(avg_recent - avg_prev) < config.convergence_tol:
                converged = True
                if config.verbose:
                    print(f"Converged at sweep {sweep + 1}", flush=True)
                break

        E_prev = E_total

    n_avg = max(len(energies_per_step) // 2, 1)
    e_per_site_avg = sum(energies_per_step[-n_avg:]) / n_avg

    return iDMRGResult(
        energy_per_site=e_per_site_avg,
        energies_per_step=energies_per_step,
        mps=InfiniteMPS.from_tensors(
            [A_L_tensor, A_R_tensor],
            [s_vals, s_vals, s_vals],  # L+1 = 3 bonds for 2-site cell
        ),
        converged=converged,
    )


def _idmrg_1site_matvec(
    theta_flat: jax.Array,
    theta_shape: tuple[int, ...],
    L_env: jax.Array,
    W: jax.Array,
    R_env: jax.Array,
) -> jax.Array:
    """Apply effective 1-site Hamiltonian to a site tensor.

    H_eff(M) = L[a,b,c] * M[a,p,d] * W[b,p,x,e] * R[d,e,f] → result[c,x,f]
    """
    M = theta_flat.reshape(theta_shape)
    result = jnp.einsum("abc,apd,bpxe,def->cxf", L_env, M, W, R_env)
    return result.ravel()


_idmrg_1site_matvec_jit = jax.jit(_idmrg_1site_matvec, static_argnums=(1,))


def _idmrg_1site_sweep(
    bulk_mpo: Tensor,
    config: iDMRGConfig,
    d: int,
    dtype: Any,
) -> iDMRGResult:
    """1-site iDMRG with DMRG3S subspace expansion (dense path).

    Uses 1-site Lanczos optimization followed by DMRG3S subspace expansion
    to grow the bond dimension.  The structure mirrors the 2-site sweep:

      1. Form 2-site theta from A_L, s_center, A_R.
      2. Solve the 2-site effective Hamiltonian via Lanczos (for total energy).
      3. SVD -> A_L, s, A_R.
      4. DMRG3S expansion (L->R): expand A_L's right bond, absorb into A_R.
      5. Update environments.

    The key difference from pure 2-site is the DMRG3S expansion step, which
    provides controlled bond growth beyond what the 2-site SVD gives.

    Args:
        bulk_mpo: Bulk MPO tensor as a ``Tensor`` wrapper.
        config: iDMRG configuration.
        d: Physical dimension.
        dtype: JAX dtype for computation.

    Returns:
        ``iDMRGResult`` with energy per site and diagnostic information.
    """
    from tenax.algorithms.dmrg3s import expand_and_truncate_dense

    W = bulk_mpo.todense()
    D_w = W.shape[0]
    key = jax.random.PRNGKey(0)

    # ---- Initialise ----
    L_env = _trivial_left_env(D_w, dtype=dtype).todense()
    R_env = _trivial_right_env(D_w, dtype=dtype).todense()

    chi_env = 1
    alpha = config.mixing_factor
    s_center = jnp.ones(1, dtype=dtype)
    theta_prev: jax.Array | None = None
    E_prev: float | None = None

    energies_per_step: list[float] = []
    converged = False
    n_keep = 1

    for sweep in range(config.max_iterations):
        # ---- Form initial 2-site theta ----
        if theta_prev is not None and theta_prev.shape == (chi_env, d, d, chi_env):
            theta = theta_prev
        elif theta_prev is not None:
            old_chi = theta_prev.shape[0]
            theta = jnp.zeros((chi_env, d, d, chi_env), dtype=dtype)
            theta = theta.at[:old_chi, :, :, :old_chi].set(theta_prev)
            key, subkey = jax.random.split(key)
            noise = 1e-3 * jax.random.normal(
                subkey, (chi_env, d, d, chi_env), dtype=dtype
            )
            theta = theta + noise
        else:
            key, subkey = jax.random.split(key)
            theta = jax.random.normal(subkey, (chi_env, d, d, chi_env), dtype=dtype)
        theta = theta / jnp.linalg.norm(theta)

        # ---- 2-site Lanczos for total energy ----
        theta_shape = theta.shape
        theta_flat = theta.ravel()

        _ts = theta_shape
        _le = L_env
        _re = R_env

        def matvec_2site(v: jax.Array) -> jax.Array:
            return _idmrg_matvec_jit(v, _ts, _le, W, W, _re)

        E_total, theta_opt_flat = _lanczos_solve(
            matvec_2site, theta_flat, config.lanczos_max_iter, config.lanczos_tol
        )
        E_total = float(E_total)
        theta_opt = theta_opt_flat.reshape(theta_shape)

        # ---- SVD to get A_L and A_R ----
        chi_l, d_l, d_r, chi_r = theta_shape
        matrix = theta_opt.reshape(chi_l * d_l, d_r * chi_r)
        U, s_full, Vt = jnp.linalg.svd(matrix, full_matrices=False)

        n_keep = min(config.max_bond_dim, len(s_full))
        if config.svd_trunc_err is not None:
            total_sq = jnp.sum(s_full**2)
            cumul_sq = jnp.cumsum(s_full[::-1] ** 2)[::-1]
            mask = cumul_sq > (config.svd_trunc_err**2 * total_sq)
            n_by_err = max(int(jnp.sum(mask)), 1)
            n_keep = min(n_keep, n_by_err)

        U = U[:, :n_keep]
        s_center = s_full[:n_keep]
        Vt = Vt[:n_keep, :]

        s_norm = jnp.linalg.norm(s_center)
        if s_norm > 1e-15:
            s_center = s_center / s_norm

        A_L = U.reshape(chi_l, d, n_keep)
        A_R = Vt.reshape(n_keep, d, chi_r)

        # ---- DMRG3S subspace expansion (L->R) ----
        # Expand A_L's right bond using the expansion tensor from L_env + MPO,
        # then SVD-truncate back to max_bond_dim.
        new_A_L, new_A_R = expand_and_truncate_dense(
            site=A_L,
            neighbor=jnp.einsum("a,apb->apb", s_center, A_R),
            env=L_env,
            mpo=W,
            alpha=alpha,
            max_bond_dim=config.max_bond_dim,
            direction="left_to_right",
            svd_trunc_err=config.svd_trunc_err,
        )
        A_L = new_A_L
        # new_A_R has s absorbed; extract s_center and right-canonical A_R
        chi_c = A_L.shape[2]
        mat_r = new_A_R.reshape(chi_c, d * chi_r)
        U_r, s_r, Vt_r = jnp.linalg.svd(mat_r, full_matrices=False)
        s_norm = jnp.linalg.norm(s_r)
        if s_norm > 1e-15:
            s_center = s_r / s_norm
        else:
            s_center = s_r
        # Absorb U_r into A_L to maintain left-isometry
        A_L = jnp.einsum("ija,ab->ijb", A_L, U_r)
        A_R = Vt_r.reshape(chi_c, d, chi_r)

        # ---- Update environments ----
        L_env = _update_left_env_dense(L_env, A_L, W)
        R_env = _update_right_env_dense(R_env, A_R, W)

        n_keep = chi_c

        # ---- Energy per site via energy difference ----
        if E_prev is not None:
            e_per_site = (E_total - E_prev) / 2.0
        else:
            e_per_site = E_total / 2.0
        energies_per_step.append(e_per_site)

        # ---- Prepare for next sweep ----
        E_prev = E_total
        theta_prev = theta_opt
        chi_env = n_keep

        if config.verbose:
            print(
                f"iDMRG 1-site sweep {sweep + 1}: E_total={E_total:.10f}, "
                f"e/site={e_per_site:.10f}, chi={n_keep}",
                flush=True,
            )

        # ---- Check convergence ----
        n_e = len(energies_per_step)
        if n_e >= 4:
            n_half = min(n_e // 2, 5)
            avg_recent = sum(energies_per_step[-n_half:]) / n_half
            avg_prev = sum(energies_per_step[-2 * n_half : -n_half]) / n_half
            if abs(avg_recent - avg_prev) < config.convergence_tol:
                converged = True
                if config.verbose:
                    print(f"Converged at sweep {sweep + 1}", flush=True)
                break

    # ---- Post-convergence orthogonalization ----
    if (
        config.orthogonalize_interval > 0
        and chi_env > 1
        and A_L.shape[0] == A_L.shape[2]
    ):
        A_L_np, A_R_np, s_np = _orthogonalize_unit_cell_dense(
            np.array(A_L),
            np.array(A_R),
            np.array(s_center),
            tol=config.arnoldi_tol,
        )
        A_L = jnp.array(A_L_np)
        A_R = jnp.array(A_R_np)
        s_center = jnp.array(s_np)

        if config.verbose:
            print("  Applied final orthogonalization", flush=True)

    # ---- Wrap final MPS tensors ----
    sym = U1Symmetry()

    def _wrap_mps(data: jax.Array, labels: tuple[str, ...]) -> DenseTensor:
        indices = tuple(
            TensorIndex(
                sym,
                np.zeros(data.shape[k], dtype=np.int32),
                FlowDirection.IN if k < data.ndim - 1 else FlowDirection.OUT,
                label=labels[k],
            )
            for k in range(data.ndim)
        )
        return DenseTensor(data, indices)

    A_L_tensor = _wrap_mps(A_L, ("v_l", "p_l", "v_c"))
    A_R_sv = jnp.einsum("a,apb->apb", s_center, A_R)
    A_R_tensor = _wrap_mps(A_R_sv, ("v_c", "p_r", "v_r"))

    n_avg = max(len(energies_per_step) // 2, 1)
    e_per_site_avg = sum(energies_per_step[-n_avg:]) / n_avg

    return iDMRGResult(
        energy_per_site=e_per_site_avg,
        energies_per_step=energies_per_step,
        mps=InfiniteMPS.from_tensors(
            [A_L_tensor, A_R_tensor],
            [s_center, s_center, s_center],
        ),
        converged=converged,
    )


def idmrg(
    bulk_mpo: Tensor,
    config: iDMRGConfig | None = None,
    d: int = 2,
    dtype: Any = jnp.float64,
) -> iDMRGResult:
    """Run infinite DMRG to find the ground-state energy per site.

    Args:
        bulk_mpo: Bulk MPO tensor (D_w, d, d, D_w) as a ``Tensor``.
                  Accepts both ``DenseTensor`` (dense path) and
                  ``SymmetricTensor`` (block-sparse path with U(1) charges).
        config:   iDMRG configuration. Uses defaults if *None*.
        d:        Physical dimension.
        dtype:    JAX dtype for computation.

    Returns:
        ``iDMRGResult`` with energy per site and diagnostic information.
    """
    if config is None:
        config = iDMRGConfig()

    if config.unit_cell_size != 2:
        raise NotImplementedError(
            f"unit_cell_size={config.unit_cell_size} is not yet supported; "
            "only unit_cell_size=2 is implemented."
        )

    if not config.two_site:
        if isinstance(bulk_mpo, SymmetricTensor):
            raise NotImplementedError(
                "1-site iDMRG with DMRG3S is not yet supported for "
                "SymmetricTensor; use two_site=True for symmetric tensors."
            )
        return _idmrg_1site_sweep(bulk_mpo, config, d, dtype)

    if isinstance(bulk_mpo, SymmetricTensor):
        return _idmrg_sweep_symmetric(bulk_mpo, config, d, dtype)
    return _idmrg_sweep(bulk_mpo, config, d, dtype)
