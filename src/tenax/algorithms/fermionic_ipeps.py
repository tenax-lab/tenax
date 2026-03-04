"""Fermionic infinite Projected Entangled Pair States (fPEPS) algorithm.

This module implements iPEPS for fermionic systems using FermionParity symmetry.
The state is represented as a PEPS with fermionic tensor structure, where
Koszul signs are automatically handled by SymmetricTensor operations.

Currently supports:
- Spinless fermion Hamiltonian: H = -t(c†c + h.c.) + V(n_i n_j)
- Trotter decomposition for imaginary time evolution
- fPEPS site tensor initialization with FermionParity
- Simple update (horizontal and vertical bonds)
- Full fPEPS optimization with CTM energy evaluation

Reference:
- Corboz et al., PRB 81, 165104 (2010)
- Barthel, Pineda, Eisert, PRA 80, 042333 (2009)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor, Tensor


@dataclass
class FPEPSConfig:
    """Configuration for fermionic iPEPS.

    Attributes:
        D:                    Virtual bond dimension.
        t:                    Hopping amplitude.
        V:                    Nearest-neighbour interaction strength.
        dt:                   Imaginary time step size for Trotter decomposition.
        num_imaginary_steps:  Number of imaginary time evolution steps.
        ctm_chi:              Bond dimension for CTM environment.
        ctm_max_iter:         Maximum CTM iterations.
        ctm_conv_tol:         CTM convergence tolerance.
    """

    D: int = 2
    t: float = 1.0
    V: float = 0.0
    dt: float = 0.01
    num_imaginary_steps: int = 200
    ctm_chi: int = 8
    ctm_max_iter: int = 50
    ctm_conv_tol: float = 1e-6


def spinless_fermion_gate(config: FPEPSConfig) -> SymmetricTensor:
    """Build the 2-site Hamiltonian H = -t(c†c + h.c.) + V(n_i n_j).

    The Hamiltonian acts on two spinless fermion sites with local
    Hilbert space ``{|0>, |1>}`` (empty, occupied). The fermionic
    anti-commutation relations are encoded via FermionParity symmetry.

    Args:
        config: FPEPSConfig with hopping t and interaction V.

    Returns:
        SymmetricTensor with 4 legs (si, sj, si_out, sj_out),
        shape (2, 2, 2, 2), using FermionParity symmetry.
    """
    t = config.t
    V = config.V

    # Build the dense 4x4 Hamiltonian matrix in the basis
    # |00>, |01>, |10>, |11> (site i tensor site j)
    #
    # c†_i c_j: |10><01| (with fermionic sign from Jordan-Wigner = +1 here)
    # c†_j c_i: |01><10|
    # n_i n_j:  |11><11|
    H = np.zeros((4, 4), dtype=np.float64)

    # Hopping: -t (c†_i c_j + c†_j c_i)
    # |01> -> |10>: c†_i c_j |01> = c†_i |00> = |10>, sign = +1
    # |10> -> |01>: c†_j c_i |10> = c†_j |00> = |01>, sign = +1
    H[2, 1] = -t  # <10|H|01>
    H[1, 2] = -t  # <01|H|10>

    # Interaction: V * n_i * n_j
    H[3, 3] = V  # <11|H|11>

    # Reshape to (2, 2, 2, 2): (si, sj, si_out, sj_out)
    H_4leg = H.reshape(2, 2, 2, 2)

    # Create TensorIndex objects with FermionParity
    sym = FermionParity()
    charges = np.array([0, 1], dtype=np.int32)

    indices = (
        TensorIndex(sym, charges, FlowDirection.IN, label="si"),
        TensorIndex(sym, charges, FlowDirection.IN, label="sj"),
        TensorIndex(sym, charges, FlowDirection.OUT, label="si_out"),
        TensorIndex(sym, charges, FlowDirection.OUT, label="sj_out"),
    )

    return SymmetricTensor.from_dense(jnp.array(H_4leg), indices)


def _trotter_gate(H: SymmetricTensor, dt: float) -> SymmetricTensor:
    """Compute the Trotter gate exp(-dt * H).

    Uses dense eigendecomposition: H = U diag(E) U†, then
    exp(-dt * H) = U diag(exp(-dt * E)) U†.

    Args:
        H:  2-site Hamiltonian as SymmetricTensor with 4 legs.
        dt: Imaginary time step (real-valued).

    Returns:
        SymmetricTensor with same indices as H, representing exp(-dt * H).
    """
    dense = H.todense().reshape(4, 4)
    dense_np = np.array(dense)

    # Eigendecomposition of the Hermitian matrix
    eigvals, eigvecs = np.linalg.eigh(dense_np)

    # Compute exp(-dt * H)
    exp_eigvals = np.exp(-dt * eigvals)
    gate = eigvecs @ np.diag(exp_eigvals) @ eigvecs.conj().T

    # Reshape back to (2, 2, 2, 2)
    gate_4leg = gate.reshape(2, 2, 2, 2)

    return SymmetricTensor.from_dense(jnp.array(gate_4leg), H.indices)


def _initialize_fpeps(config: FPEPSConfig, key: jax.Array) -> SymmetricTensor:
    """Create a random fPEPS site tensor A[u, d, l, r, phys].

    The tensor has FermionParity symmetry on all legs. Virtual bond
    charges alternate 0, 1, 0, 1, ... for bond dimension D.
    Physical charges are [0, 1] (empty, occupied).

    Flows:
        u = OUT, d = IN, l = OUT, r = IN, phys = IN

    Args:
        config: FPEPSConfig with bond dimension D.
        key:    JAX random key.

    Returns:
        SymmetricTensor with 5 legs (u, d, l, r, phys).
    """
    D = config.D
    sym = FermionParity()

    # Virtual charges: [i % 2 for i in range(D)]
    virt_charges = np.array([i % 2 for i in range(D)], dtype=np.int32)

    # Physical charges: [0, 1]
    phys_charges = np.array([0, 1], dtype=np.int32)

    indices = (
        TensorIndex(sym, virt_charges, FlowDirection.OUT, label="u"),
        TensorIndex(sym, virt_charges, FlowDirection.IN, label="d"),
        TensorIndex(sym, virt_charges, FlowDirection.OUT, label="l"),
        TensorIndex(sym, virt_charges, FlowDirection.IN, label="r"),
        TensorIndex(sym, phys_charges, FlowDirection.IN, label="phys"),
    )

    return SymmetricTensor.random_normal(indices, key)


def _normalize_tensor(T: SymmetricTensor) -> SymmetricTensor:
    """Normalize a SymmetricTensor to unit norm."""
    norm_val = float(T.norm())
    if norm_val <= EPS:
        return T
    return T * (1.0 / norm_val)


def _absorb_lambdas(
    A: SymmetricTensor, lam_h: jax.Array, lam_v: jax.Array
) -> SymmetricTensor:
    """Absorb lambda vectors into A on all virtual legs (block-level operation).

    Multiplies each virtual leg of A by its corresponding lambda vector:
      u ← lam_v, d ← lam_v, l ← lam_h, r ← lam_h

    Args:
        A:     Site tensor with labels (u, d, l, r, phys).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.

    Returns:
        SymmetricTensor with lambdas absorbed.
    """
    result = scale_bond_axis(A, "u", lam_v)
    result = scale_bond_axis(result, "d", lam_v)
    result = scale_bond_axis(result, "l", lam_h)
    result = scale_bond_axis(result, "r", lam_h)
    return result


def _fpeps_simple_update_horizontal(
    A: SymmetricTensor,
    gate: SymmetricTensor,
    lam_h: jax.Array,
    lam_v: jax.Array,
    max_D: int,
) -> tuple[SymmetricTensor, jax.Array]:
    """Simple update on the horizontal bond (A.r <-> B.l, B=A by periodicity).

    Uses SymmetricTensor operations (contract, truncated_svd, transpose) to
    correctly handle fermionic Koszul signs throughout.

    Args:
        A:     fPEPS site tensor with labels (u, d, l, r, phys).
        gate:  Trotter gate with labels (si, sj, si_out, sj_out).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.
        max_D: Maximum bond dimension after SVD.

    Returns:
        (A_new, lam_h_new) where A_new has labels (u, d, l, r, phys).
    """
    # 1. Absorb all lambdas into A
    A_abs = _absorb_lambdas(A, lam_h, lam_v)

    # 2. Create left and right tensors by relabeling
    A_left = A_abs.relabel("r", "shared")
    B_right = A_abs.relabels(
        {
            "u": "u_B",
            "d": "d_B",
            "l": "shared",
            "r": "r_B",
            "phys": "phys_B",
        }
    )

    # 3. Contract A_left and B_right over "shared"
    theta = contract(A_left, B_right)

    # 4. Apply gate: relabel phys->si, phys_B->sj, contract with gate
    theta = theta.relabel("phys", "si")
    theta = theta.relabel("phys_B", "sj")
    theta = contract(theta, gate)

    # 5. SVD
    U, sigma, Vh, s_full = truncated_svd(
        theta,
        left_labels=["u", "d", "l", "si_out"],
        right_labels=["u_B", "d_B", "r_B", "sj_out"],
        new_bond_label="r_new",
        max_singular_values=max_D,
    )
    # U has labels: (u, d, l, si_out, r_new)

    # 6. Normalize lambda
    lam_h_new = sigma / (jnp.max(sigma) + EPS)

    # 7. Transpose to (u, d, l, r_new, si_out) and relabel
    U_reordered = U.transpose((0, 1, 2, 4, 3))
    U_final = U_reordered.relabels({"r_new": "r", "si_out": "phys"})

    # 8. Absorb sqrt(sigma) into the bond axis "r"
    sqrt_sig = jnp.sqrt(sigma + EPS)
    U_final = scale_bond_axis(U_final, "r", sqrt_sig)

    # 9. Remove outer lambdas: u <- lam_v^{-1}, d <- lam_v^{-1}, l <- lam_h^{-1}
    inv_lam_v = 1.0 / (lam_v + EPS)
    inv_lam_h = 1.0 / (lam_h + EPS)
    U_final = scale_bond_axis(U_final, "u", inv_lam_v)
    U_final = scale_bond_axis(U_final, "d", inv_lam_v)
    U_final = scale_bond_axis(U_final, "l", inv_lam_h)

    # 10. Normalize
    U_final = _normalize_tensor(U_final)

    return U_final, lam_h_new


def _fpeps_simple_update_vertical(
    A: SymmetricTensor,
    gate: SymmetricTensor,
    lam_h: jax.Array,
    lam_v: jax.Array,
    max_D: int,
) -> tuple[SymmetricTensor, jax.Array]:
    """Simple update on the vertical bond (A.d <-> B.u, B=A by periodicity).

    Uses SymmetricTensor operations to correctly handle fermionic Koszul signs.

    Args:
        A:     fPEPS site tensor with labels (u, d, l, r, phys).
        gate:  Trotter gate with labels (si, sj, si_out, sj_out).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.
        max_D: Maximum bond dimension after SVD.

    Returns:
        (A_new, lam_v_new) where A_new has labels (u, d, l, r, phys).
    """
    # 1. Absorb all lambdas into A
    A_abs = _absorb_lambdas(A, lam_h, lam_v)

    # 2. Create top and bottom tensors by relabeling
    A_top = A_abs.relabel("d", "shared")
    B_bottom = A_abs.relabels(
        {
            "u": "shared",
            "d": "d_B",
            "l": "l_B",
            "r": "r_B",
            "phys": "phys_B",
        }
    )

    # 3. Contract A_top and B_bottom over "shared"
    theta = contract(A_top, B_bottom)

    # 4. Apply gate
    theta = theta.relabel("phys", "si")
    theta = theta.relabel("phys_B", "sj")
    theta = contract(theta, gate)

    # 5. SVD
    U, sigma, Vh, s_full = truncated_svd(
        theta,
        left_labels=["u", "l", "r", "si_out"],
        right_labels=["d_B", "l_B", "r_B", "sj_out"],
        new_bond_label="d_new",
        max_singular_values=max_D,
    )
    # U has labels: (u, l, r, si_out, d_new)

    # 6. Normalize lambda
    lam_v_new = sigma / (jnp.max(sigma) + EPS)

    # 7. Transpose to (u, d_new, l, r, si_out) and relabel
    U_reordered = U.transpose((0, 4, 1, 2, 3))
    U_final = U_reordered.relabels({"d_new": "d", "si_out": "phys"})

    # 8. Absorb sqrt(sigma) into the bond axis "d"
    sqrt_sig = jnp.sqrt(sigma + EPS)
    U_final = scale_bond_axis(U_final, "d", sqrt_sig)

    # 9. Remove outer lambdas: u <- lam_v^{-1}, l <- lam_h^{-1}, r <- lam_h^{-1}
    inv_lam_v = 1.0 / (lam_v + EPS)
    inv_lam_h = 1.0 / (lam_h + EPS)
    U_final = scale_bond_axis(U_final, "u", inv_lam_v)
    U_final = scale_bond_axis(U_final, "l", inv_lam_h)
    U_final = scale_bond_axis(U_final, "r", inv_lam_h)

    # 10. Normalize
    U_final = _normalize_tensor(U_final)

    return U_final, lam_v_new


def _fpeps_simple_update(
    A: SymmetricTensor,
    hamiltonian_gate: SymmetricTensor,
    max_D: int,
    dt: float,
    steps: int,
) -> tuple[SymmetricTensor, jax.Array, jax.Array]:
    """Run simple update for a given number of steps.

    Alternates horizontal and vertical bond updates.

    Args:
        A:                fPEPS site tensor.
        hamiltonian_gate: 2-site Hamiltonian (SymmetricTensor).
        max_D:            Maximum bond dimension.
        dt:               Imaginary time step.
        steps:            Number of simple update steps.

    Returns:
        (A_opt, lam_h, lam_v) after all steps.
    """
    gate = _trotter_gate(hamiltonian_gate, dt)
    lam_h = jnp.ones(max_D)
    lam_v = jnp.ones(max_D)

    for _ in range(steps):
        A, lam_h = _fpeps_simple_update_horizontal(A, gate, lam_h, lam_v, max_D)
        A, lam_v = _fpeps_simple_update_vertical(A, gate, lam_h, lam_v, max_D)

    return A, lam_h, lam_v


# ------------------------------------------------------------------ #
# CTM energy evaluation                                                #
# ------------------------------------------------------------------ #
#
# For FermionParity (Z₂), the twist phase in dagger() is always +1,
# so the fermionic double-layer tensor is identical to the bosonic one.
# We therefore convert to dense and delegate to the bosonic CTM for
# environment and energy computation.  The SymmetricTensor simple
# update (above) handles Koszul signs correctly for optimization.
# ------------------------------------------------------------------ #


def fermionic_ctm(A, config):
    """Run CTM to convergence for a fermionic PEPS site tensor.

    For FermionParity (Z₂), the twist phase in ``dagger()`` is always +1,
    so the fermionic double-layer tensor is identical to the bosonic one.
    This function converts *A* to a dense array and delegates to the
    bosonic CTM implementation.

    For symmetric CTM without densification, use
    :func:`tenax.algorithms._split_ctm_tensor.ctm_split_tensor` directly.

    Args:
        A:      fPEPS site tensor (SymmetricTensor with lambdas absorbed).
        config: FPEPSConfig.

    Returns:
        Converged bosonic ``CTMEnvironment``.
    """
    from tenax.algorithms.ipeps import CTMConfig, ctm

    A_dense = A.todense() if isinstance(A, Tensor) else A
    ctm_cfg = CTMConfig(
        chi=config.ctm_chi,
        max_iter=config.ctm_max_iter,
        conv_tol=config.ctm_conv_tol,
    )
    return ctm(A_dense, ctm_cfg)


def compute_energy_fermionic_ctm(A, env, hamiltonian_gate):
    """Compute energy per site using a CTM environment.

    Supports both ``SplitCTMTensorEnv`` (from split-CTM) and legacy
    ``CTMEnvironment`` (from dense CTM).

    Args:
        A:                fPEPS site tensor (SymmetricTensor or dense).
        env:              Converged environment from :func:`fermionic_ctm`.
        hamiltonian_gate: 2-site Hamiltonian (SymmetricTensor or dense array).

    Returns:
        Energy per site (float).
    """
    from tenax.algorithms.ipeps import compute_energy_ctm

    A_dense = A.todense() if isinstance(A, Tensor) else A
    d = A_dense.shape[-1]
    if isinstance(hamiltonian_gate, SymmetricTensor):
        H = hamiltonian_gate.todense().reshape(d, d, d, d)
    else:
        H = hamiltonian_gate.reshape(d, d, d, d)
    return float(compute_energy_ctm(A_dense, env, H, d))


def fpeps(
    hamiltonian_gate: SymmetricTensor,
    config: FPEPSConfig,
    initial_tensor: SymmetricTensor | None = None,
    key: jax.Array | None = None,
) -> tuple[float, SymmetricTensor, object]:
    """Run fPEPS: simple update optimization + CTM energy evaluation.

    Args:
        hamiltonian_gate: 2-site Hamiltonian as SymmetricTensor.
        config:           FPEPSConfig.
        initial_tensor:   Optional initial site tensor. If None, random init.
        key:              JAX random key (used if initial_tensor is None).

    Returns:
        (energy, A_opt, env) where energy is a scalar, A_opt is the
        optimized SymmetricTensor, and env is the CTMEnvironment.
    """
    if initial_tensor is not None:
        A = initial_tensor
    else:
        if key is None:
            key = jax.random.PRNGKey(0)
        A = _initialize_fpeps(config, key)

    A_opt, lam_h, lam_v = _fpeps_simple_update(
        A,
        hamiltonian_gate,
        max_D=config.D,
        dt=config.dt,
        steps=config.num_imaginary_steps,
    )

    A_abs = _absorb_lambdas(A_opt, lam_h, lam_v)
    A_abs = _normalize_tensor(A_abs)
    env = fermionic_ctm(A_abs, config)
    energy = compute_energy_fermionic_ctm(A_abs, env, hamiltonian_gate)

    return float(energy), A_opt, env
