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

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor
from tenax.network.network import TensorNetwork


@dataclass
class CTMConfig:
    """Configuration for CTM environment computation.

    Attributes:
        chi:          Bond dimension of CTM environment tensors.
        max_iter:     Maximum CTM iterations before declaring convergence.
        conv_tol:     Convergence tolerance (based on singular value change
                      between CTM iterations).
        renormalize:  Whether to renormalize environment tensors at each step
                      to prevent exponential growth (always recommended).
    """

    chi: int = 20
    max_iter: int = 100
    conv_tol: float = 1e-8
    renormalize: bool = True
    projector_method: str = "eigh"  # "eigh" or "qr"
    qr_warmup_steps: int = 3  # eigh warm-up iterations before QR kicks in
    chi_I: int | None = None  # interlayer bond dim for split-CTMRG; None => chi_I = chi


@dataclass
class iPEPSConfig:
    """Configuration for iPEPS simple update optimization.

    Attributes:
        max_bond_dim:          PEPS virtual bond dimension D.
        num_imaginary_steps:   Number of imaginary time evolution steps.
        dt:                    Imaginary time step size.
        ctm:                   CTM configuration for environment computation.
        svd_trunc_err:         SVD truncation error for simple update.
        gate_order:            Order of bond updates: "sequential" or "random".
        su_init:               If True, ``optimize_gs_ad`` initializes the site
                               tensor via simple update (``ipeps()``) instead of
                               random initialization.  Ignored when ``A_init``
                               is provided explicitly.
    """

    max_bond_dim: int = 2
    num_imaginary_steps: int = 100
    dt: float = 0.01
    ctm: CTMConfig = field(default_factory=CTMConfig)
    svd_trunc_err: float | None = None
    gate_order: str = "sequential"
    unit_cell: str = "1x1"  # "1x1" or "2site"
    # AD ground-state optimization settings
    gs_optimizer: str = "adam"
    gs_learning_rate: float = 1e-3
    gs_num_steps: int = 200
    gs_conv_tol: float = 1e-8
    su_init: bool = False


class CTMEnvironment(NamedTuple):
    """The 8 CTM environment tensors (4 corners + 4 edge tensors).

    Corner convention (looking at a single site):
        C1 --- T1 --- C2
        |             |
        T4    [A]    T2
        |             |
        C4 --- T3 --- C3

    Corners (chi x chi tensors):
        C1: top-left     C2: top-right
        C3: bottom-right C4: bottom-left

    Edges (chi x D^2 x chi tensors, where D = PEPS bond dim):
        T1: top    T2: right
        T3: bottom T4: left

    All shapes use chi for environment bonds, D^2 for the PEPS bond (physical
    space of the doubled layer A * A^* is D^2 = D*D).
    """

    C1: jax.Array  # shape (chi, chi)
    C2: jax.Array  # shape (chi, chi)
    C3: jax.Array  # shape (chi, chi)
    C4: jax.Array  # shape (chi, chi)
    T1: jax.Array  # shape (chi, D2, chi) — top edge
    T2: jax.Array  # shape (chi, D2, chi) — right edge
    T3: jax.Array  # shape (chi, D2, chi) — bottom edge
    T4: jax.Array  # shape (chi, D2, chi) — left edge


class SplitCTMEnvironment(NamedTuple):
    """Split CTM environment keeping ket and bra layers separate.

    Corners are shared between ket and bra (shape ``(chi, chi)``).
    Each edge is split into a ket half ``(chi, D, chi_I)`` and a bra
    half ``(chi_I, D, chi)`` connected by an interlayer bond ``chi_I``.

    This reduces the projector cost from O(chi^3 * D^6) to
    O(chi^3 * D^3) (arXiv:2502.10298).
    """

    C1: jax.Array  # (chi, chi)
    C2: jax.Array  # (chi, chi)
    C3: jax.Array  # (chi, chi)
    C4: jax.Array  # (chi, chi)
    T1_ket: jax.Array  # (chi, D, chi_I)
    T1_bra: jax.Array  # (chi_I, D, chi)
    T2_ket: jax.Array  # (chi, D, chi_I)
    T2_bra: jax.Array  # (chi_I, D, chi)
    T3_ket: jax.Array  # (chi, D, chi_I)
    T3_bra: jax.Array  # (chi_I, D, chi)
    T4_ket: jax.Array  # (chi, D, chi_I)
    T4_bra: jax.Array  # (chi_I, D, chi)


def ipeps(
    hamiltonian_gate: jax.Array,
    initial_peps: TensorNetwork | jax.Array | tuple[jax.Array, jax.Array] | None,
    config: iPEPSConfig,
) -> tuple[
    float, TensorNetwork, CTMEnvironment | tuple[CTMEnvironment, CTMEnvironment]
]:
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
        initial_peps:     TensorNetwork, raw JAX array, or tuple of two JAX
                          arrays ``(A, B)`` for the 2-site unit cell. ``None``
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


def _simple_update_1x1(
    A: jax.Array,
    B: jax.Array,
    lambdas: dict[str, jax.Array],
    gate: jax.Array,
    max_bond_dim: int,
    *,
    bond: str = "horizontal",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update step for a 1x1 unit cell PEPS.

    Applies the gate on the bond between A and B (treating B = A by translational
    invariance). Updates lambda matrices for environment approximation.

    Args:
        A:            Left/top site tensor of shape (D, D, D, D, d) or (D, D, d).
                      Convention: A[u, d, l, r, s] for full 5-leg PEPS.
        B:            Right/bottom site tensor (same as A for 1x1 unit cell).
        lambdas:      Dict of lambda vectors for each bond direction.
        gate:         2-site gate of shape (d, d, d, d).
        max_bond_dim: Maximum D after truncation.
        bond:         Which bond to update: ``"horizontal"`` (A.r ↔ B.l) or
                      ``"vertical"`` (A.d ↔ B.u).

    Returns:
        (A_new, lambdas_new)
    """
    d = gate.shape[0]

    if A.ndim == 3:
        return _simple_update_3leg(A, B, lambdas, gate, max_bond_dim, d)

    # --- Full 5-leg tensors: A[u, d, l, r, s] ---
    D_u, D_d, D_l, D_r, phys = A.shape
    lam_h = lambdas.get("horizontal", jnp.ones(D_r))
    lam_v = lambdas.get("vertical", jnp.ones(D_d))

    if bond == "horizontal":
        return _simple_update_horizontal(A, lam_h, lam_v, gate, max_bond_dim, lambdas)
    else:
        return _simple_update_vertical(A, lam_h, lam_v, gate, max_bond_dim, lambdas)


def _simple_update_3leg(
    A: jax.Array,
    B: jax.Array,
    lambdas: dict[str, jax.Array],
    gate: jax.Array,
    max_bond_dim: int,
    d: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update for the legacy 3-leg (D_l, D_r, d) tensor path."""
    D_l, D_r, phys = A.shape
    lam_r = lambdas.get(
        "horizontal", lambdas.get("right", jnp.ones(min(D_r, max_bond_dim)))
    )

    A_abs = A * lam_r[None, : min(D_r, len(lam_r)), None]
    theta = jnp.einsum("lrs,Lrs,sstT->lLtT", A_abs, A, gate.reshape(phys, phys, d, d))

    theta_mat = theta.reshape(D_l * D_l, d * d)
    U, s, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)

    n_keep = min(max_bond_dim, len(s))
    U = U[:, :n_keep]
    s_new = s[:n_keep]

    s_norm = s_new / (jnp.max(s_new) + 1e-15)
    lam_inv = 1.0 / (lam_r[: min(D_r, len(lam_r))] + 1e-15)

    A_new_mat = U.reshape(D_l, D_l, n_keep)[:, 0, :]
    A_new = (A_new_mat * lam_inv[None, : min(D_l, len(lam_inv))]).reshape(
        D_l, n_keep, d
    )

    lambdas_new = dict(lambdas)
    lambdas_new["horizontal"] = s_norm
    lambdas_new.pop("right", None)
    return A_new, lambdas_new


def _simple_update_bond(
    A: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
    axis: str,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update on a single bond, parameterized by axis.

    A[u, d, l, r, s]:
      axis="horizontal": shared bond = r (horizontal lambda)
      axis="vertical":   shared bond = d (vertical lambda)

    Args:
        A:             iPEPS site tensor, shape (D_u, D_d, D_l, D_r, d).
        lam_h:         Horizontal bond lambdas.
        lam_v:         Vertical bond lambdas.
        gate:          Trotter gate, shape (d, d, d, d).
        max_bond_dim:  Maximum bond dimension after SVD.
        lambdas:       Current lambdas dict.
        axis:          "horizontal" or "vertical".

    Returns:
        (A_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    eps = EPS

    if axis == "horizontal":
        # Absorb outer lambdas onto A (all except shared bond r)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        # Absorb shared-bond lambda onto A.r
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]

        # B = A, absorb outer lambdas (all except B.l = shared)
        B_abs = A * lam_v[:D_u, None, None, None, None]
        B_abs = B_abs * lam_v[None, :D_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, None, :D_r, None]

        # Contract A_abs.r with B_abs.l → theta
        theta = jnp.einsum("udlrs,UDrRt->udlUDRst", A_abs, B_abs)
        theta = jnp.einsum("udlUDRst,stST->udlUDRST", theta, gate)

        left_size = D_u * D_d * D_l * d
        right_size = D_u * D_d * D_r * d
        left_shape = (D_u, D_d, D_l, d)
        # new bond goes into r slot: transpose to (D_u, D_d, D_l, keep, d)
        left_perm = (0, 1, 2, 4, 3)

        # Outer lambda removal: u←lam_v, d←lam_v, l←lam_h
        outer_inv_slices = [
            (1.0 / (lam_v + eps), 0, D_u),  # axis 0
            (1.0 / (lam_v + eps), 1, D_d),  # axis 1
            (1.0 / (lam_h + eps), 2, D_l),  # axis 2
        ]
    else:  # vertical
        # Absorb outer lambdas onto A (all except shared bond d)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]
        # Absorb shared-bond lambda onto A.d
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]

        # B = A, absorb outer lambdas (all except B.u = shared)
        B_abs = A * lam_v[None, :D_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, :D_l, None, None]
        B_abs = B_abs * lam_h[None, None, None, :D_r, None]

        # Contract A_abs.d with B_abs.u → theta
        theta = jnp.einsum("udlrs,dDLRt->ulrDLRst", A_abs, B_abs)
        theta = jnp.einsum("ulrDLRst,stST->ulrDLRST", theta, gate)

        left_size = D_u * D_l * D_r * d
        right_size = D_d * D_l * D_r * d
        left_shape = (D_u, D_l, D_r, d)
        # new bond goes into d slot: transpose to (D_u, keep, D_l, D_r, d)
        left_perm = (0, 4, 1, 2, 3)

        # Outer lambda removal: u←lam_v, l←lam_h, r←lam_h
        outer_inv_slices = [
            (1.0 / (lam_v + eps), 0, D_u),  # axis 0
            (1.0 / (lam_h + eps), 2, D_l),  # axis 2
            (1.0 / (lam_h + eps), 3, D_r),  # axis 3 (after transpose)
        ]

    # SVD split
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)
    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]

    # New lambda (normalized)
    lam_new = sigma / (jnp.max(sigma) + eps)

    # Reconstruct A_new from U_mat
    sqrt_sig = jnp.sqrt(sigma + eps)
    A_left = (U_mat * sqrt_sig[None, :]).reshape(*left_shape, keep)
    A_new = A_left.transpose(left_perm)

    # Remove outer lambdas
    for inv_lam, ax, dim in outer_inv_slices:
        shape = [1] * 5
        shape[ax] = dim
        A_new = A_new * inv_lam[:dim].reshape(shape)

    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new[axis] = lam_new
    return A_new, lambdas_new


def _simple_update_horizontal(
    A: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update on the horizontal bond (A.r ↔ B.l, B=A by periodicity)."""
    return _simple_update_bond(
        A, lam_h, lam_v, gate, max_bond_dim, lambdas, "horizontal"
    )


def _simple_update_vertical(
    A: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update on the vertical bond (A.d ↔ B.u, B=A by periodicity)."""
    return _simple_update_bond(A, lam_h, lam_v, gate, max_bond_dim, lambdas, "vertical")


def _simple_update_2site_bond(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
    axis: str,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on a single bond for a 2-site unit cell.

    Args:
        A, B:          iPEPS site tensors, shape (D_u, D_d, D_l, D_r, d).
        lam_h, lam_v:  Bond lambdas.
        gate:          Trotter gate.
        max_bond_dim:  Maximum bond dimension.
        lambdas:       Current lambdas dict.
        axis:          "horizontal" or "vertical".

    Returns:
        (A_new, B_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    B_u, B_d, B_l, B_r, _ = B.shape
    eps = EPS

    if axis == "horizontal":
        # A: outer = u(v), d(v), l(h); shared = r(h)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]

        # B: outer = u(v), d(v), r(h); shared = l (from contraction)
        B_abs = B * lam_v[:B_u, None, None, None, None]
        B_abs = B_abs * lam_v[None, :B_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, None, :B_r, None]

        theta = jnp.einsum("udlrs,UDrRt->udlUDRst", A_abs, B_abs)
        theta = jnp.einsum("udlUDRst,stST->udlUDRST", theta, gate)

        left_size = D_u * D_d * D_l * d
        right_size = B_u * B_d * B_r * d
        a_left_shape = (D_u, D_d, D_l, d)
        b_right_shape = (B_u, B_d, B_r, d)
        a_perm = (0, 1, 2, 4, 3)  # new bond → r slot
        b_perm = (1, 2, 0, 3, 4)  # new bond → l slot
        a_outer_inv = [
            (1.0 / (lam_v + eps), 0, D_u),
            (1.0 / (lam_v + eps), 1, D_d),
            (1.0 / (lam_h + eps), 2, D_l),
        ]
        b_outer_inv = [
            (1.0 / (lam_v + eps), 0, B_u),
            (1.0 / (lam_v + eps), 1, B_d),
            (1.0 / (lam_h + eps), 3, B_r),
        ]
    else:  # vertical
        # A: outer = u(v), l(h), r(h); shared = d(v)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]

        # B: outer = d(v), l(h), r(h); shared = u (from contraction)
        B_abs = B * lam_v[None, :B_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, :B_l, None, None]
        B_abs = B_abs * lam_h[None, None, None, :B_r, None]

        theta = jnp.einsum("udlrs,dDLRt->ulrDLRst", A_abs, B_abs)
        theta = jnp.einsum("ulrDLRst,stST->ulrDLRST", theta, gate)

        left_size = D_u * D_l * D_r * d
        right_size = B_d * B_l * B_r * d
        a_left_shape = (D_u, D_l, D_r, d)
        b_right_shape = (B_d, B_l, B_r, d)
        a_perm = (0, 4, 1, 2, 3)  # new bond → d slot
        b_perm = (0, 1, 2, 3, 4)  # new bond → u slot
        a_outer_inv = [
            (1.0 / (lam_v + eps), 0, D_u),
            (1.0 / (lam_h + eps), 2, D_l),
            (1.0 / (lam_h + eps), 3, D_r),
        ]
        b_outer_inv = [
            (1.0 / (lam_v + eps), 1, B_d),
            (1.0 / (lam_h + eps), 2, B_l),
            (1.0 / (lam_h + eps), 3, B_r),
        ]

    # SVD
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)
    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]
    Vh_mat = Vh_mat[:keep, :]

    lam_new = sigma / (jnp.max(sigma) + eps)
    sqrt_sig = jnp.sqrt(sigma + eps)

    # Reconstruct A_new
    A_left = (U_mat * sqrt_sig[None, :]).reshape(*a_left_shape, keep)
    A_new = A_left.transpose(a_perm)

    # Reconstruct B_new
    B_right = (sqrt_sig[:, None] * Vh_mat).reshape(keep, *b_right_shape)
    B_new = B_right.transpose(b_perm)

    # Remove outer lambdas
    for inv_lam, ax, dim in a_outer_inv:
        shape = [1] * 5
        shape[ax] = dim
        A_new = A_new * inv_lam[:dim].reshape(shape)
    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    for inv_lam, ax, dim in b_outer_inv:
        shape = [1] * 5
        shape[ax] = dim
        B_new = B_new * inv_lam[:dim].reshape(shape)
    B_new = B_new / (jnp.linalg.norm(B_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new[axis] = lam_new
    return A_new, B_new, lambdas_new


def _simple_update_2site_horizontal(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on the horizontal bond A.r ↔ B.l for a 2-site unit cell.

    Returns (A_new, B_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    eps = 1e-15

    # 1. Absorb outer lambdas onto A: u←lam_v, d←lam_v, l←lam_h
    A_abs = A * lam_v[:D_u, None, None, None, None]
    A_abs = A_abs * lam_v[None, :D_d, None, None, None]
    A_abs = A_abs * lam_h[None, None, :D_l, None, None]
    # 2. Absorb shared-bond lambda onto A.r
    A_abs = A_abs * lam_h[None, None, None, :D_r, None]

    # 3. Absorb outer lambdas onto B: u←lam_v, d←lam_v, r←lam_h
    B_u, B_d, B_l, B_r, _ = B.shape
    B_abs = B * lam_v[:B_u, None, None, None, None]
    B_abs = B_abs * lam_v[None, :B_d, None, None, None]
    B_abs = B_abs * lam_h[None, None, None, :B_r, None]

    # 4. Contract A_abs.r with B_abs.l
    theta = jnp.einsum("udlrs,UDrRt->udlUDRst", A_abs, B_abs)

    # 5. Apply gate
    theta = jnp.einsum("udlUDRst,stST->udlUDRST", theta, gate)

    # 6. SVD: group (u,d,l,S) vs (U,D,R,T)
    left_size = D_u * D_d * D_l * d
    right_size = B_u * B_d * B_r * d
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)

    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]
    Vh_mat = Vh_mat[:keep, :]

    # 7. New lambda
    lam_new = sigma / (jnp.max(sigma) + eps)

    # 8. Reconstruct A_new and B_new with sqrt(sigma) absorbed
    sqrt_sig = jnp.sqrt(sigma + eps)
    A_left = (U_mat * sqrt_sig[None, :]).reshape(D_u, D_d, D_l, d, keep)
    A_new = A_left.transpose(0, 1, 2, 4, 3)  # (D_u, D_d, D_l, keep, d)

    B_right = (sqrt_sig[:, None] * Vh_mat).reshape(keep, B_u, B_d, B_r, d)
    B_new = B_right.transpose(1, 2, 0, 3, 4)  # (B_u, B_d, keep, B_r, d)

    # 9. Remove outer lambdas
    lam_v_inv = 1.0 / (lam_v + eps)
    lam_h_inv = 1.0 / (lam_h + eps)
    A_new = A_new * lam_v_inv[:D_u, None, None, None, None]
    A_new = A_new * lam_v_inv[None, :D_d, None, None, None]
    A_new = A_new * lam_h_inv[None, None, :D_l, None, None]
    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    B_new = B_new * lam_v_inv[:B_u, None, None, None, None]
    B_new = B_new * lam_v_inv[None, :B_d, None, None, None]
    B_new = B_new * lam_h_inv[None, None, None, :B_r, None]
    B_new = B_new / (jnp.linalg.norm(B_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new["horizontal"] = lam_new
    return A_new, B_new, lambdas_new


def _simple_update_2site_vertical(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on the vertical bond A.d ↔ B.u for a 2-site unit cell.

    Returns (A_new, B_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    eps = 1e-15

    # 1. Absorb outer lambdas onto A: u←lam_v, l←lam_h, r←lam_h
    A_abs = A * lam_v[:D_u, None, None, None, None]
    A_abs = A_abs * lam_h[None, None, :D_l, None, None]
    A_abs = A_abs * lam_h[None, None, None, :D_r, None]
    # 2. Absorb shared-bond lambda onto A.d
    A_abs = A_abs * lam_v[None, :D_d, None, None, None]

    # 3. Absorb outer lambdas onto B: d←lam_v, l←lam_h, r←lam_h
    B_u, B_d, B_l, B_r, _ = B.shape
    B_abs = B * lam_v[None, :B_d, None, None, None]
    B_abs = B_abs * lam_h[None, None, :B_l, None, None]
    B_abs = B_abs * lam_h[None, None, None, :B_r, None]

    # 4. Contract A_abs.d with B_abs.u
    theta = jnp.einsum("udlrs,dDLRt->ulrDLRst", A_abs, B_abs)

    # 5. Apply gate
    theta = jnp.einsum("ulrDLRst,stST->ulrDLRST", theta, gate)

    # 6. SVD: group (u,l,r,S) vs (D,L,R,T)
    left_size = D_u * D_l * D_r * d
    right_size = B_d * B_l * B_r * d
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)

    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]
    Vh_mat = Vh_mat[:keep, :]

    # 7. New lambda
    lam_new = sigma / (jnp.max(sigma) + eps)

    # 8. Reconstruct A_new and B_new
    sqrt_sig = jnp.sqrt(sigma + eps)
    A_left = (U_mat * sqrt_sig[None, :]).reshape(D_u, D_l, D_r, d, keep)
    A_new = A_left.transpose(0, 4, 1, 2, 3)  # (D_u, keep, D_l, D_r, d)

    B_right = (sqrt_sig[:, None] * Vh_mat).reshape(keep, B_d, B_l, B_r, d)
    B_new = B_right.transpose(0, 1, 2, 3, 4)  # (keep, B_d, B_l, B_r, d)

    # 9. Remove outer lambdas
    lam_v_inv = 1.0 / (lam_v + eps)
    lam_h_inv = 1.0 / (lam_h + eps)
    A_new = A_new * lam_v_inv[:D_u, None, None, None, None]
    A_new = A_new * lam_h_inv[None, None, :D_l, None, None]
    A_new = A_new * lam_h_inv[None, None, None, :D_r, None]
    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    B_new = B_new * lam_v_inv[None, :B_d, None, None, None]
    B_new = B_new * lam_h_inv[None, None, :B_l, None, None]
    B_new = B_new * lam_h_inv[None, None, None, :B_r, None]
    B_new = B_new / (jnp.linalg.norm(B_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new["vertical"] = lam_new
    return A_new, B_new, lambdas_new


def _ctm_sweep(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    renormalize: bool,
    projector_method: str = "eigh",
) -> CTMEnvironment:
    """One full CTM sweep: left, right, top, bottom moves + optional renormalize."""
    env = _ctm_left_move(env, a, chi, projector_method)
    env = _ctm_right_move(env, a, chi, projector_method)
    env = _ctm_top_move(env, a, chi, projector_method)
    env = _ctm_bottom_move(env, a, chi, projector_method)
    if renormalize:
        env = _renormalize_env(env)
    return env


def _ctm_sv_diff(sv_new: jax.Array, sv_old: jax.Array) -> jax.Array:
    """Compute max absolute difference between normalized singular value vectors."""
    sv1 = sv_new / (jnp.sum(sv_new) + 1e-15)
    sv2 = sv_old / (jnp.sum(sv_old) + 1e-15)
    return jnp.max(jnp.abs(sv1 - sv2))


def ctm(
    A: jax.Array,
    config: CTMConfig,
    initial_env: CTMEnvironment | None = None,
) -> CTMEnvironment:
    """Compute CTM environment for a PEPS with 1x1 unit cell.

    Runs the CTM algorithm (Corboz/Orús scheme) until convergence.
    The input A is the double-layer tensor A * A^* combined, or the
    single-layer A from which the doubled tensor is computed.

    The iteration loop uses ``jax.lax.while_loop`` so that the entire
    convergence procedure can be JIT-compiled without host sync.

    Args:
        A:           Site tensor (single layer) of PEPS.
        config:      CTMConfig.
        initial_env: Optional starting environment for warm start.

    Returns:
        Converged CTMEnvironment.
    """
    chi = config.chi

    # Build the double-layer tensor a = sum_s A[s,...] * conj(A[s,...])
    # For a simple 1x1 cell: a[u,d,l,r, U,D,L,R] = sum_s A[u,d,l,r,s]*A*[U,D,L,R,s]
    # The physical index is traced over.
    a = _build_double_layer(A)  # shape (D, D, D, D, D, D, D, D)
    # Reshape to (D^2, D^2, D^2, D^2) for CTM
    if a.ndim == 8:
        D_phys = a.shape[0]
        a = a.reshape(D_phys**2, D_phys**2, D_phys**2, D_phys**2)
    elif a.ndim == 4:
        pass  # already (D^2, D^2, D^2, D^2)

    # Initialize environment tensors
    if initial_env is not None:
        env = initial_env
    else:
        env = _initialize_ctm_env(a, chi)

    max_iter = config.max_iter
    conv_tol = config.conv_tol
    renormalize = config.renormalize
    projector_method = config.projector_method

    # QR warm-up: run a few eigh iterations before switching to QR
    if projector_method == "qr" and config.qr_warmup_steps > 0:
        warmup_steps = min(config.qr_warmup_steps, max_iter)
        for _ in range(warmup_steps):
            env = _ctm_sweep(env, a, chi, renormalize, "eigh")
        max_iter = max_iter - warmup_steps

    # Initial singular values (zeros — first iteration never converges)
    prev_sv = jnp.zeros(min(chi, env.C1.shape[0]), dtype=env.C1.dtype)

    # Carry: (env, prev_sv, iteration, converged)
    init_carry = (env, prev_sv, jnp.array(0, dtype=jnp.int32), jnp.bool_(False))

    def cond_fn(carry):
        _, _, iteration, converged = carry
        return ~converged & (iteration < max_iter)

    def body_fn(carry):
        env_i, prev_sv_i, iteration, _ = carry
        env_i = _ctm_sweep(env_i, a, chi, renormalize, projector_method)
        current_sv = jnp.linalg.svd(env_i.C1, compute_uv=False)
        diff = _ctm_sv_diff(current_sv, prev_sv_i)
        converged = diff < conv_tol
        return (env_i, current_sv, iteration + 1, converged)

    env, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return env


def _build_double_layer(A: jax.Array) -> jax.Array:
    """Build the double-layer tensor from a PEPS site tensor.

    For a tensor A with shape (D,...,d) where d is the physical dimension
    and D's are virtual bond dimensions, the double-layer tensor is:
    a[virtual...] = sum_s A[virtual..., s] * conj(A[virtual..., s])

    This traces out the physical index.
    """
    if A.ndim == 5:
        # A[u, d, l, r, s] — fuse ket/bra pairs per spatial direction
        return jnp.einsum("udlrs,UDLRs->uUdDlLrR", A, jnp.conj(A))
    elif A.ndim == 3:
        # A[l, r, s] — simplified 2D
        return jnp.einsum("lrs,LRs->lrLR", A, jnp.conj(A))
    else:
        # Generic: assume last index is physical
        # Squeeze to remove degenerate dims
        s_idx = "".join(chr(97 + i) for i in range(A.ndim))
        phys = s_idx[-1]
        virt1 = s_idx[:-1]
        virt2 = virt1.upper()
        return jnp.einsum(f"{s_idx},{virt2}{phys}->{virt1}{virt2}", A, jnp.conj(A))


def _initialize_ctm_env(a: jax.Array, chi: int) -> CTMEnvironment:
    """Initialize CTM environment tensors from the PEPS double-layer tensor.

    Uses a simple initialization: corners and edges built from partial traces
    of the double-layer tensor.

    Args:
        a:   Double-layer tensor of shape (D2, D2, D2, D2).
        chi: Environment bond dimension.
    """
    D2 = a.shape[0]
    dtype = a.dtype

    # Initialize corners as identity matrices (chi x chi)
    C = jnp.eye(min(chi, D2), dtype=dtype)
    C_small = jnp.zeros((chi, chi), dtype=dtype)
    C_small = C_small.at[: C.shape[0], : C.shape[1]].set(
        C[: min(chi, C.shape[0]), : min(chi, C.shape[1])]
    )

    # Initialize edges as a slice of the double-layer tensor
    # T[chi, D2, chi] — use first chi values
    T_chi = min(chi, D2)
    T_init = jnp.zeros((chi, D2, chi), dtype=dtype)
    # Fill with identity-like structure
    for i in range(min(T_chi, chi)):
        T_init = T_init.at[i, :, i].add(jnp.ones(D2))

    return CTMEnvironment(
        C1=C_small,
        C2=C_small,
        C3=C_small,
        C4=C_small,
        T1=T_init,
        T2=T_init,
        T3=T_init,
        T4=T_init,
    )


def _ctm_move_eigh(
    C1g: jax.Array,
    C2g: jax.Array,
    Tg: jax.Array,
    chi: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Projector-based CTM truncation using eigh (standard method).

    Given two grown corners ``C1g`` and ``C2g`` (each a 2-D matrix whose row
    dimension is ``chi * D2``) and a grown edge tensor ``Tg`` of shape
    ``(chi*D2, D2, chi*D2)``, compute a single isometric projector from the
    combined half-system density matrix (Corboz et al., PRB 90, 165127, 2014),
    then truncate both corners and the edge to bond dimension ``chi``.

    The projector ``P`` is obtained from the eigendecomposition of the
    half-system density matrix ``rho = C1g @ C1g^H + C2g @ C2g^H``.
    Using ``eigh`` (symmetric eigendecomposition) is more numerically
    stable than SVD of the concatenated corners when ``chi * D2 == 2 * chi``
    (square matrix case), avoiding spurious sign oscillations in the
    projector that prevent convergence.

    Returns ``(C1_new, C2_new, T_new)`` with shapes ``(chi', col1)``,
    ``(chi', col2)``, ``(chi', D2, chi')`` where ``chi' <= chi``.
    """
    # Half-system density matrix (Corboz et al. 2014).
    # rho = C1g @ C1g^T + C2g @ C2g^T is positive semi-definite.
    # Its leading eigenvectors form the optimal isometric projector.
    rho = C1g @ C1g.conj().T + C2g @ C2g.conj().T
    rho = 0.5 * (rho + rho.conj().T)  # enforce exact Hermiticity

    eigvals, eigvecs = jnp.linalg.eigh(rho)
    # eigh returns eigenvalues in ascending order; take the top chi.
    k = min(chi, len(eigvals))
    P = eigvecs[:, -k:][:, ::-1]  # (n, chi'), largest first

    # Project corners — stop gradient through P for AD stability.
    # The implicit fixed-point differentiation (ctm_converge) handles
    # the overall response; differentiating through the projector
    # eigenvectors causes gradient blowup from degenerate eigenvalues.
    P_sg = jax.lax.stop_gradient(P)
    C1_new = P_sg.conj().T @ C1g  # (chi', col1)
    C2_new = P_sg.conj().T @ C2g  # (chi', col2)
    T_new = jnp.einsum("ia,idj,jb->adb", P_sg, Tg, P_sg)
    return C1_new, C2_new, T_new


def _ctm_move_qr(
    C1g: jax.Array,
    C2g: jax.Array,
    Tg: jax.Array,
    chi: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Projector-based CTM truncation using QR + small eigh.

    Instead of forming the full density matrix rho of size (chi*D^2)^2 and
    diagonalizing at O((chi*D^2)^3), QR-factor the concatenated corners
    [C1g, C2g] to reduce to a small (2*chi, 2*chi) problem, then eigh
    the small matrix (arXiv:2505.00494).

    Total cost: O(chi^2 * D^2 * chi) for QR + O(chi^3) for small eigh,
    vs O(chi^3 * D^6) for direct eigh of rho.

    Returns ``(C1_new, C2_new, T_new)`` with shapes ``(chi', col1)``,
    ``(chi', col2)``, ``(chi', D2, chi')`` where ``chi' <= chi``.
    """
    M = jnp.concatenate([C1g, C2g], axis=1)  # (n, 2*m)
    Q, R = jnp.linalg.qr(M)  # Q: (n, 2m), R: (2m, 2m)

    # Small eigh on R @ R^H (size 2*chi x 2*chi)
    rho_small = R @ R.conj().T
    rho_small = 0.5 * (rho_small + rho_small.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(rho_small)

    k = min(chi, len(eigvals))
    V = eigvecs[:, -k:][:, ::-1]  # leading eigenvectors
    P = Q @ V  # (n, k)

    P_sg = jax.lax.stop_gradient(P)
    C1_new = P_sg.conj().T @ C1g
    C2_new = P_sg.conj().T @ C2g
    T_new = jnp.einsum("ia,idj,jb->adb", P_sg, Tg, P_sg)
    return C1_new, C2_new, T_new


def _ctm_move(
    C1g: jax.Array,
    C2g: jax.Array,
    Tg: jax.Array,
    chi: int,
    projector_method: str = "eigh",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Dispatch CTM truncation to eigh or QR projector method."""
    if projector_method == "qr":
        return _ctm_move_qr(C1g, C2g, Tg, chi)
    return _ctm_move_eigh(C1g, C2g, Tg, chi)


def _ctm_left_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    projector_method: str = "eigh",
) -> CTMEnvironment:
    """Projector-based CTM left move: updates C1, T4, C4.

    Grows C1 with T1, C4 with T3, T4 with ``a``, then truncates with
    consistent projectors derived from the grown corners.
    """
    D2 = a.shape[0]
    # Grow corners
    C1g = jnp.einsum("ab,buc->auc", env.C1, env.T1).reshape(-1, env.T1.shape[2])
    C4g = jnp.einsum("gh,hdi->gdi", env.C4, env.T3).reshape(-1, env.T3.shape[2])
    # Grow edge: T4[a,l,g] * a[u,d,l,r] -> (a,u,g,d,r)
    T4g = jnp.einsum("alg,udlr->augdr", env.T4, a)
    T4g = T4g.transpose(0, 1, 4, 2, 3).reshape(C1g.shape[0], D2, C4g.shape[0])

    C1_new, C4_new, T4_new = _ctm_move(C1g, C4g, T4g, chi, projector_method)
    return CTMEnvironment(
        C1=C1_new,
        C2=env.C2,
        C3=env.C3,
        C4=C4_new,
        T1=env.T1,
        T2=env.T2,
        T3=env.T3,
        T4=T4_new,
    )


def _ctm_right_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    projector_method: str = "eigh",
) -> CTMEnvironment:
    """Projector-based CTM right move: updates C2, T2, C3."""
    D2 = a.shape[0]
    # Grow corners
    C2g = jnp.einsum("ce,buc->eub", env.C2, env.T1).reshape(-1, env.T1.shape[0])
    C3g = jnp.einsum("im,hdi->mdh", env.C3, env.T3).reshape(-1, env.T3.shape[0])
    # Grow edge: T2[e,r,m] * a[u,d,l,r] -> (e,u,m,d,l)
    T2g = jnp.einsum("erm,udlr->eumdl", env.T2, a)
    T2g = T2g.transpose(0, 1, 4, 2, 3).reshape(C2g.shape[0], D2, C3g.shape[0])

    C2_new, C3_new, T2_new = _ctm_move(C2g, C3g, T2g, chi, projector_method)
    return CTMEnvironment(
        C1=env.C1,
        C2=C2_new,
        C3=C3_new,
        C4=env.C4,
        T1=env.T1,
        T2=T2_new,
        T3=env.T3,
        T4=env.T4,
    )


def _ctm_top_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    projector_method: str = "eigh",
) -> CTMEnvironment:
    """Projector-based CTM top move: updates C1, T1, C2."""
    D2 = a.shape[0]
    # Grow corners
    C1g = jnp.einsum("ab,alg->blg", env.C1, env.T4).reshape(-1, env.T4.shape[2])
    C2g = jnp.einsum("ce,erm->crm", env.C2, env.T2).reshape(-1, env.T2.shape[2])
    # Grow edge: T1[b,u,c] * a[u,d,l,r] -> (b,c,d,l,r)
    T1g = jnp.einsum("buc,udlr->bcdlr", env.T1, a)
    T1g = T1g.transpose(0, 3, 2, 1, 4).reshape(C1g.shape[0], D2, C2g.shape[0])

    C1_new, C2_new, T1_new = _ctm_move(C1g, C2g, T1g, chi, projector_method)
    return CTMEnvironment(
        C1=C1_new,
        C2=C2_new,
        C3=env.C3,
        C4=env.C4,
        T1=T1_new,
        T2=env.T2,
        T3=env.T3,
        T4=env.T4,
    )


def _ctm_bottom_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    projector_method: str = "eigh",
) -> CTMEnvironment:
    """Projector-based CTM bottom move: updates C4, T3, C3."""
    D2 = a.shape[0]
    # Grow corners
    C4g = jnp.einsum("gh,alg->hal", env.C4, env.T4)
    C4g = C4g.transpose(0, 2, 1).reshape(-1, env.T4.shape[0])
    C3g = jnp.einsum("im,erm->ire", env.C3, env.T2).reshape(-1, env.T2.shape[0])
    # Grow edge: T3[h,d,i] * a[u,d,l,r] -> (h,i,u,l,r)
    T3g = jnp.einsum("hdi,udlr->hiulr", env.T3, a)
    T3g = T3g.transpose(0, 3, 2, 1, 4).reshape(C4g.shape[0], D2, C3g.shape[0])

    C4_new, C3_new, T3_new = _ctm_move(C4g, C3g, T3g, chi, projector_method)
    return CTMEnvironment(
        C1=env.C1,
        C2=env.C2,
        C3=C3_new,
        C4=C4_new,
        T1=env.T1,
        T2=env.T2,
        T3=T3_new,
        T4=env.T4,
    )


def _renormalize_env(env: CTMEnvironment) -> CTMEnvironment:
    """Normalize environment tensors to prevent exponential growth."""

    def normalize(x: jax.Array) -> jax.Array:
        norm = jnp.max(jnp.abs(x))
        return x / (norm + EPS)

    return CTMEnvironment(
        C1=normalize(env.C1),
        C2=normalize(env.C2),
        C3=normalize(env.C3),
        C4=normalize(env.C4),
        T1=normalize(env.T1),
        T2=normalize(env.T2),
        T3=normalize(env.T3),
        T4=normalize(env.T4),
    )


def _ctm_left_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Projector-based 2-site CTM left move."""
    D2 = a_neighbor.shape[0]
    C1g = jnp.einsum("ab,buc->auc", env_self.C1, env_neighbor.T1).reshape(
        -1, env_neighbor.T1.shape[2]
    )
    C4g = jnp.einsum("gh,hdi->gdi", env_self.C4, env_neighbor.T3).reshape(
        -1, env_neighbor.T3.shape[2]
    )
    T4g = jnp.einsum("alg,udlr->augdr", env_self.T4, a_neighbor)
    T4g = T4g.transpose(0, 1, 4, 2, 3).reshape(C1g.shape[0], D2, C4g.shape[0])
    C1_new, C4_new, T4_new = _ctm_move(C1g, C4g, T4g, chi)
    return CTMEnvironment(
        C1=C1_new,
        C2=env_self.C2,
        C3=env_self.C3,
        C4=C4_new,
        T1=env_self.T1,
        T2=env_self.T2,
        T3=env_self.T3,
        T4=T4_new,
    )


def _ctm_right_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Projector-based 2-site CTM right move."""
    D2 = a_neighbor.shape[0]
    C2g = jnp.einsum("ce,buc->eub", env_self.C2, env_neighbor.T1).reshape(
        -1, env_neighbor.T1.shape[0]
    )
    C3g = jnp.einsum("im,hdi->mdh", env_self.C3, env_neighbor.T3).reshape(
        -1, env_neighbor.T3.shape[0]
    )
    T2g = jnp.einsum("erm,udlr->eumdl", env_self.T2, a_neighbor)
    T2g = T2g.transpose(0, 1, 4, 2, 3).reshape(C2g.shape[0], D2, C3g.shape[0])
    C2_new, C3_new, T2_new = _ctm_move(C2g, C3g, T2g, chi)
    return CTMEnvironment(
        C1=env_self.C1,
        C2=C2_new,
        C3=C3_new,
        C4=env_self.C4,
        T1=env_self.T1,
        T2=T2_new,
        T3=env_self.T3,
        T4=env_self.T4,
    )


def _ctm_top_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Projector-based 2-site CTM top move."""
    D2 = a_neighbor.shape[0]
    C1g = jnp.einsum("ab,alg->blg", env_self.C1, env_neighbor.T4).reshape(
        -1, env_neighbor.T4.shape[2]
    )
    C2g = jnp.einsum("ce,erm->crm", env_self.C2, env_neighbor.T2).reshape(
        -1, env_neighbor.T2.shape[2]
    )
    T1g = jnp.einsum("buc,udlr->bcdlr", env_self.T1, a_neighbor)
    T1g = T1g.transpose(0, 3, 2, 1, 4).reshape(C1g.shape[0], D2, C2g.shape[0])
    C1_new, C2_new, T1_new = _ctm_move(C1g, C2g, T1g, chi)
    return CTMEnvironment(
        C1=C1_new,
        C2=C2_new,
        C3=env_self.C3,
        C4=env_self.C4,
        T1=T1_new,
        T2=env_self.T2,
        T3=env_self.T3,
        T4=env_self.T4,
    )


def _ctm_bottom_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Projector-based 2-site CTM bottom move."""
    D2 = a_neighbor.shape[0]
    C4g = jnp.einsum("gh,alg->hal", env_self.C4, env_neighbor.T4)
    C4g = C4g.transpose(0, 2, 1).reshape(-1, env_neighbor.T4.shape[0])
    C3g = jnp.einsum("im,erm->ire", env_self.C3, env_neighbor.T2).reshape(
        -1, env_neighbor.T2.shape[0]
    )
    T3g = jnp.einsum("hdi,udlr->hiulr", env_self.T3, a_neighbor)
    T3g = T3g.transpose(0, 3, 2, 1, 4).reshape(C4g.shape[0], D2, C3g.shape[0])
    C4_new, C3_new, T3_new = _ctm_move(C4g, C3g, T3g, chi)
    return CTMEnvironment(
        C1=env_self.C1,
        C2=env_self.C2,
        C3=C3_new,
        C4=C4_new,
        T1=env_self.T1,
        T2=env_self.T2,
        T3=T3_new,
        T4=env_self.T4,
    )


def _ctm_2site_sweep(
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    a_A: jax.Array,
    a_B: jax.Array,
    chi: int,
    renormalize: bool,
) -> tuple[CTMEnvironment, CTMEnvironment]:
    """One full 2-site CTM sweep: L/R/T/B moves for both sublattices + renormalize."""
    # Left moves
    env_A = _ctm_left_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_left_move_2site(env_B, env_A, a_A, chi)
    # Right moves
    env_A = _ctm_right_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_right_move_2site(env_B, env_A, a_A, chi)
    # Top moves
    env_A = _ctm_top_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_top_move_2site(env_B, env_A, a_A, chi)
    # Bottom moves
    env_A = _ctm_bottom_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_bottom_move_2site(env_B, env_A, a_A, chi)
    if renormalize:
        env_A = _renormalize_env(env_A)
        env_B = _renormalize_env(env_B)
    return env_A, env_B


def ctm_2site(
    A: jax.Array,
    B: jax.Array,
    config: CTMConfig,
) -> tuple[CTMEnvironment, CTMEnvironment]:
    """Compute CTM environments for a 2-site checkerboard unit cell.

    On a checkerboard, all neighbors of A are B and vice versa. Each
    absorption move for env_A uses B's double-layer tensor and T's from
    env_B, and vice versa.

    The iteration loop uses ``jax.lax.while_loop`` so that the entire
    convergence procedure can be JIT-compiled without host sync.

    Args:
        A: Site tensor for sublattice A, shape (D, D, D, D, d).
        B: Site tensor for sublattice B, shape (D, D, D, D, d).
        config: CTMConfig.

    Returns:
        (env_A, env_B) — converged CTM environments for each sublattice.
    """
    chi = config.chi

    a_A = _build_double_layer(A)
    a_B = _build_double_layer(B)
    D_A = A.shape[0]
    D_B = B.shape[0]
    if a_A.ndim == 8:
        a_A = a_A.reshape(D_A**2, D_A**2, D_A**2, D_A**2)
    if a_B.ndim == 8:
        a_B = a_B.reshape(D_B**2, D_B**2, D_B**2, D_B**2)

    env_A = _initialize_ctm_env(a_A, chi)
    env_B = _initialize_ctm_env(a_B, chi)

    max_iter = config.max_iter
    conv_tol = config.conv_tol
    renormalize = config.renormalize

    # Initial singular values (zeros — first iteration never converges)
    sv_size_A = min(chi, env_A.C1.shape[0])
    sv_size_B = min(chi, env_B.C1.shape[0])
    prev_sv_A = jnp.zeros(sv_size_A, dtype=env_A.C1.dtype)
    prev_sv_B = jnp.zeros(sv_size_B, dtype=env_B.C1.dtype)

    # Carry: (env_A, env_B, prev_sv_A, prev_sv_B, iteration, converged)
    init_carry = (
        env_A,
        env_B,
        prev_sv_A,
        prev_sv_B,
        jnp.array(0, dtype=jnp.int32),
        jnp.bool_(False),
    )

    def cond_fn(carry):
        _, _, _, _, iteration, converged = carry
        return ~converged & (iteration < max_iter)

    def body_fn(carry):
        eA, eB, psA, psB, iteration, _ = carry
        eA, eB = _ctm_2site_sweep(eA, eB, a_A, a_B, chi, renormalize)
        sv_A = jnp.linalg.svd(eA.C1, compute_uv=False)
        sv_B = jnp.linalg.svd(eB.C1, compute_uv=False)
        diff_A = _ctm_sv_diff(sv_A, psA)
        diff_B = _ctm_sv_diff(sv_B, psB)
        converged = jnp.maximum(diff_A, diff_B) < conv_tol
        return (eA, eB, sv_A, sv_B, iteration + 1, converged)

    env_A, env_B, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return env_A, env_B


def _build_double_layer_open(A: jax.Array) -> jax.Array:
    """Double-layer tensor with physical indices left open.

    Returns ``a_open`` with shape ``(D^2, D^2, D^2, D^2, d, d)`` where the
    last two axes are the ket and bra physical indices.
    """
    if A.ndim == 5:
        # A[u,d,l,r,s], conj(A)[U,D,L,R,t]
        # -> a_open[uU, dD, lL, rR, s, t]
        ao = jnp.einsum("udlrs,UDLRt->uUdDlLrRst", A, jnp.conj(A))
        D = A.shape[0]
        d = A.shape[4]
        return ao.reshape(D**2, D**2, D**2, D**2, d, d)
    raise ValueError("_build_double_layer_open requires a 5-leg tensor")


def _rdm2x1(A: jax.Array, env: CTMEnvironment, d: int) -> jax.Array:
    """Horizontal 2-site reduced density matrix from CTM environment.

    Contracts the network:

    .. code-block::

        C1 — T1 — T1 — C2
        |     |     |    |
        T4  ao_1 — ao_2  T2
        |     |     |    |
        C4 — T3 — T3 — C3

    Returns RDM with shape ``(d, d, d, d)`` — ``(s1, s2, s1', s2')``
    (ket indices first, bra indices second), so that
    ``rdm.reshape(d*d, d*d)`` is a proper density matrix.
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env
    a_open = _build_double_layer_open(A)  # (D2, D2, D2, D2, d, d)

    # Step-by-step contraction (small intermediates):
    # UL = C1 · T1  →  (chi, D2, chi)
    UL = jnp.einsum("ab,buc->auc", C1, T1)
    # UR = T1 · C2  →  (chi, D2, chi)
    UR = jnp.einsum("cuf,fg->cug", T1, C2)
    # LL = C4 · T3  →  (chi, D2, chi)
    LL = jnp.einsum("gi,idj->gdj", C4, T3)
    # LR = T3 · C3  →  (chi, D2, chi)
    LR = jnp.einsum("jdk,mk->jdm", T3, C3)

    # Left env: UL[a,u1,c] · T4[a,l1,g] · LL[g,d1,j]
    # Contract a between UL and T4, g between T4 and LL
    Lenv = jnp.einsum("auc,axg,gdj->ucxdj", UL, T4, LL)
    # shape: (D2, chi, D2, D2, chi) → (u1, c, l1, d1, j)

    # Right env: UR[c,u2,f] · T2[f,r2,m] · LR[j,d2,m]
    # Contract f between UR and T2, m between T2 and LR
    Renv = jnp.einsum("cuf,frm,jdm->curjd", UR, T2, LR)
    # shape: (chi, D2, D2, chi, D2) → (c, u2, r2, j, d2)

    # Contract Lenv with ao1[u1, d1, l1, r1, s, sp]:
    # Match: u1=u, d1=d, l1=x  → free: c, r1, j, s, sp
    Lenv_ao1 = jnp.einsum("ucxdj,udxrst->crjst", Lenv, a_open)
    # shape: (chi, D2, chi, d, d) → (c, r1, j, s1, s1')

    # Contract Renv with ao2[u2, d2, l2, r2, t, tp]:
    # Match: u2=u, d2=d, r2=r  → free: c, l2, j, t, tp
    Renv_ao2 = jnp.einsum("curjd,udlrtv->cjltv", Renv, a_open)
    # shape: (chi, chi, D2, d, d) → (c, j, l2, s2, s2')

    # Final: contract Lenv_ao1 with Renv_ao2
    # Match: c=c, j=j, r1=l2  → free: s1, s1', s2, s2'
    rdm = jnp.einsum("crjst,cjruv->stuv", Lenv_ao1, Renv_ao2)
    # rdm has convention (s1_ket, s1_bra, s2_ket, s2_bra).
    # Transpose to (s1_ket, s2_ket, s1_bra, s2_bra) so that
    # reshape(d*d, d*d) yields a proper density matrix with rows =
    # ket and columns = bra, matching the Hamiltonian convention.
    rdm = rdm.transpose(0, 2, 1, 3)

    # Symmetrize and normalize
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def _rdm1x2(A: jax.Array, env: CTMEnvironment, d: int) -> jax.Array:
    """Vertical 2-site reduced density matrix from CTM environment.

    Contracts the network:

    .. code-block::

        C1  — T1  — C2
        |      |      |
        T4 — ao_1 — T2
        |      |      |
        T4 — ao_2 — T2
        |      |      |
        C4  — T3  — C3

    Returns RDM with shape ``(d, d, d, d)`` — ``(s1, s2, s1', s2')``
    (ket indices first, bra indices second), so that
    ``rdm.reshape(d*d, d*d)`` is a proper density matrix.
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env
    a_open = _build_double_layer_open(A)

    # Top row: C1·T1·C2 → (chi, D2, chi)  indices: (a, u, e)
    top_row = jnp.einsum("ab,buc,ce->aue", C1, T1, C2)

    # Contract top_row with T4 (site-1 left) and T2 (site-1 right):
    # top_row[a, u1, e]  T4[a, l1, f]  T2[e, r1, g]
    # Contract a, e → env_row1[u1, l1, f, r1, g]
    env_row1 = jnp.einsum("aue,alf,erg->ulfrg", top_row, T4, T2)
    # (D2, D2, chi, D2, chi) → (u1, l1, f, r1, g)

    # Contract with ao1[u1, d1, l1, r1, s, sp]:  match u1, l1, r1
    site1 = jnp.einsum("ulfrg,udlrst->dfgst", env_row1, a_open)
    # (D2, chi, chi, d, d) → (d1, f, g, s1, s1')

    # Step A: contract T4[f,l2,h] with ao2[d1, d2, l2, r2, t, tp]  match l2
    # Use unique index letters: a_open → (p, q, m, n, w, x)
    #   p=d1_ao2=u, q=d2, m=l2, n=r2, w=s2, x=s2'
    T4_ao2 = jnp.einsum("fmh,pqmnwx->fhpqnwx", T4, a_open)
    # (chi, chi, D2, D2, D2, d, d) → (f, h, p=d1, q=d2, n=r2, w, x)

    # Step B: contract site1[d1, f, g, s, t] with T4_ao2[f, h, d1, d2, r2, w, x]
    # Match: d1 and f  (use a=d1, b=f)
    # site1:  a(D2) b(chi) c(chi) s(d) t(d)
    # T4_ao2: b(chi) h(chi) a(D2) q(D2) n(D2) w(d) x(d)
    site12 = jnp.einsum("abcst,bhaqnwx->chqnstwx", site1, T4_ao2)
    # (chi, chi, D2, D2, d, d, d, d) → (c=g, h, q=d2, n=r2, s1, s1', s2, s2')

    # Contract T2[g, r2, i]: match g=c, r2=n
    site12_r = jnp.einsum("chqnstwx,cni->hqistwx", site12, T2)
    # (chi, D2, chi, d, d, d, d) → (h, q=d2, i, s1, s1', s2, s2')

    # Bottom row: C4·T3·C3 → (chi, D2, chi)  indices: (h, d2, i)
    bot_row = jnp.einsum("hj,jqk,ik->hqi", C4, T3, C3)

    # Final: contract site12_r with bot_row  match h, q=d2, i
    rdm = jnp.einsum("hqistwx,hqi->stwx", site12_r, bot_row)
    # rdm has convention (s1_ket, s1_bra, s2_ket, s2_bra).
    # Transpose to (s1_ket, s2_ket, s1_bra, s2_bra) so that
    # reshape(d*d, d*d) yields a proper density matrix.
    rdm = rdm.transpose(0, 2, 1, 3)

    # Symmetrize and normalize
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def compute_energy_ctm(
    A: jax.Array,
    env: CTMEnvironment,
    hamiltonian_gate: jax.Array,
    d: int,
) -> jax.Array:
    """Compute energy per site using CTM environment and 2-site RDMs.

    Constructs horizontal and vertical two-site reduced density matrices
    from the CTM environment and contracts each with the Hamiltonian to
    obtain the energy per site.

    Args:
        A:                 PEPS site tensor of shape ``(D, D, D, D, d)``.
        env:               Converged CTMEnvironment.
        hamiltonian_gate:  2-site Hamiltonian, shape ``(d, d, d, d)``.
        d:                 Physical dimension.

    Returns:
        Scalar energy per site.
    """
    if A.ndim != 5:
        # Fallback for legacy 3-leg tensors
        return jnp.array(-0.25, dtype=A.dtype)

    rdm_h = _rdm2x1(A, env, d)
    rdm_v = _rdm1x2(A, env, d)
    H = hamiltonian_gate.reshape(d, d, d, d)
    E_h = jnp.einsum("ijkl,ijkl->", rdm_h, H)
    E_v = jnp.einsum("ijkl,ijkl->", rdm_v, H)
    return (E_h + E_v).real


def _rdm2x1_2site(
    A: jax.Array,
    B: jax.Array,
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    d: int,
) -> jax.Array:
    """Horizontal 2-site RDM for a checkerboard unit cell (A left, B right).

    Uses mixed environment:
        C1_A — T1_A — T1_B — C2_B
        |       |       |       |
        T4_A  ao_A   ao_B    T2_B
        |       |       |       |
        C4_A — T3_A — T3_B — C3_B
    """
    ao_A = _build_double_layer_open(A)
    ao_B = _build_double_layer_open(B)

    # Left boundary from env_A
    UL = jnp.einsum("ab,buc->auc", env_A.C1, env_A.T1)
    LL = jnp.einsum("gi,idj->gdj", env_A.C4, env_A.T3)
    Lenv = jnp.einsum("auc,axg,gdj->ucxdj", UL, env_A.T4, LL)

    # Right boundary from env_B
    UR = jnp.einsum("cuf,fg->cug", env_B.T1, env_B.C2)
    LR = jnp.einsum("jdk,mk->jdm", env_B.T3, env_B.C3)
    Renv = jnp.einsum("cuf,frm,jdm->curjd", UR, env_B.T2, LR)

    # Contract left env with ao_A
    Lenv_ao = jnp.einsum("ucxdj,udxrst->crjst", Lenv, ao_A)
    # Contract right env with ao_B
    Renv_ao = jnp.einsum("curjd,udlrtv->cjltv", Renv, ao_B)
    # Final contraction
    rdm = jnp.einsum("crjst,cjruv->stuv", Lenv_ao, Renv_ao)
    # Transpose from (s1_ket, s1_bra, s2_ket, s2_bra) to
    # (s1_ket, s2_ket, s1_bra, s2_bra) for proper density matrix convention.
    rdm = rdm.transpose(0, 2, 1, 3)

    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def _rdm1x2_2site(
    A: jax.Array,
    B: jax.Array,
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    d: int,
) -> jax.Array:
    """Vertical 2-site RDM for a checkerboard unit cell (A top, B bottom).

    Uses mixed environment:
        C1_A — T1_A — C2_A
        |       |       |
        T4_A  ao_A    T2_A
        |       |       |
        T4_B  ao_B    T2_B
        |       |       |
        C4_B — T3_B — C3_B
    """
    ao_A = _build_double_layer_open(A)
    ao_B = _build_double_layer_open(B)

    # Top row from env_A
    top_row = jnp.einsum("ab,buc,ce->aue", env_A.C1, env_A.T1, env_A.C2)
    # Env row 1: contract with T4_A (left) and T2_A (right)
    env_row1 = jnp.einsum("aue,alf,erg->ulfrg", top_row, env_A.T4, env_A.T2)
    # Contract with ao_A
    site1 = jnp.einsum("ulfrg,udlrst->dfgst", env_row1, ao_A)

    # Step A: T4_B with ao_B
    T4_ao2 = jnp.einsum("fmh,pqmnwx->fhpqnwx", env_B.T4, ao_B)
    # Step B: contract site1 with T4_ao2
    site12 = jnp.einsum("abcst,bhaqnwx->chqnstwx", site1, T4_ao2)
    # Contract T2_B
    site12_r = jnp.einsum("chqnstwx,cni->hqistwx", site12, env_B.T2)
    # Bottom row from env_B
    bot_row = jnp.einsum("hj,jqk,ik->hqi", env_B.C4, env_B.T3, env_B.C3)
    # Final
    rdm = jnp.einsum("hqistwx,hqi->stwx", site12_r, bot_row)
    # Transpose from (s1_ket, s1_bra, s2_ket, s2_bra) to
    # (s1_ket, s2_ket, s1_bra, s2_bra) for proper density matrix convention.
    rdm = rdm.transpose(0, 2, 1, 3)

    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def compute_energy_ctm_2site(
    A: jax.Array,
    B: jax.Array,
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    hamiltonian_gate: jax.Array,
    d: int,
) -> jax.Array:
    """Compute energy per site for a 2-site checkerboard iPEPS.

    E/site = E_horizontal + E_vertical (one bond of each type per site).
    """
    H = hamiltonian_gate.reshape(d, d, d, d)
    rdm_h = _rdm2x1_2site(A, B, env_A, env_B, d)
    rdm_v = _rdm1x2_2site(A, B, env_A, env_B, d)
    E_h = jnp.einsum("ijkl,ijkl->", rdm_h, H)
    E_v = jnp.einsum("ijkl,ijkl->", rdm_v, H)
    return (E_h + E_v).real


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


# ---------------------------------------------------------------------------
# Split-CTMRG: ket/bra layers kept separate (arXiv:2502.10298)
# ---------------------------------------------------------------------------


def _initialize_split_ctm_env(
    A: jax.Array,
    chi: int,
    chi_I: int,
) -> SplitCTMEnvironment:
    """Initialize a SplitCTMEnvironment from the PEPS site tensor.

    Args:
        A:     Site tensor of shape ``(D, D, D, D, d)``.
        chi:   Environment bond dimension.
        chi_I: Interlayer bond dimension.
    """
    D = A.shape[0]
    dtype = A.dtype

    # Corners: identity-like (chi x chi)
    C = jnp.eye(min(chi, D), dtype=dtype)
    C_pad = jnp.zeros((chi, chi), dtype=dtype)
    C_pad = C_pad.at[: C.shape[0], : C.shape[1]].set(C)

    # Split edges: identity-like structure
    chi_D = min(chi, D)
    chi_I_D = min(chi_I, D)

    T_ket = jnp.zeros((chi, D, chi_I), dtype=dtype)
    for i in range(min(chi_D, chi_I_D)):
        T_ket = T_ket.at[i, :, i].set(jnp.ones(D))

    T_bra = jnp.zeros((chi_I, D, chi), dtype=dtype)
    for i in range(min(chi_I_D, chi_D)):
        T_bra = T_bra.at[i, :, i].set(jnp.ones(D))

    return SplitCTMEnvironment(
        C1=C_pad,
        C2=C_pad,
        C3=C_pad,
        C4=C_pad,
        T1_ket=T_ket,
        T1_bra=T_bra,
        T2_ket=T_ket,
        T2_bra=T_bra,
        T3_ket=T_ket,
        T3_bra=T_bra,
        T4_ket=T_ket,
        T4_bra=T_bra,
    )


def _split_ctm_projector(
    C1g: jax.Array,
    C2g: jax.Array,
    chi: int,
) -> jax.Array:
    """Compute projector from grown corners for split-CTMRG.

    Uses eigh-based projector on the smaller (chi*D) matrices instead of
    the (chi*D^2) matrices used in standard CTMRG.
    """
    rho = C1g @ C1g.conj().T + C2g @ C2g.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(rho)
    k = min(chi, len(eigvals))
    P = eigvecs[:, -k:][:, ::-1]
    return jax.lax.stop_gradient(P)


def _svd_split_edge(
    T_full: jax.Array,
    chi_I: int,
) -> tuple[jax.Array, jax.Array]:
    """Split standard edge tensor into ket/bra pair via SVD.

    ``T_full`` of shape ``(chi, D^2, chi)`` is reshaped to
    ``(chi, D, D, chi)`` and split into ``T_ket(chi, D, chi_I)`` and
    ``T_bra(chi_I, D, chi)`` by SVD of the (chi*D, D*chi) matrix.
    """
    chi = T_full.shape[0]
    D2 = T_full.shape[1]
    D = int(round(D2**0.5))
    T_4d = T_full.reshape(chi, D, D, chi)
    T_mat = T_4d.reshape(chi * D, D * chi)

    U, s, Vh = jnp.linalg.svd(T_mat, full_matrices=False)
    k = min(chi_I, len(s))
    sqrt_s = jnp.sqrt(s[:k])
    T_ket = (U[:, :k] * sqrt_s[None, :]).reshape(chi, D, k)
    T_bra = (sqrt_s[:, None] * Vh[:k, :]).reshape(k, D, chi)
    return T_ket, T_bra


def _split_ctm_move(
    env: SplitCTMEnvironment,
    A: jax.Array,
    chi: int,
    chi_I: int,
    direction: str,
) -> SplitCTMEnvironment:
    """Perform one split-CTMRG directional move.

    For left/top moves the corner connects to the edge's ket side (chi bond),
    so ket projectors are computed first.  For right/bottom moves the corner
    connects to the edge's bra side, so bra projectors come first.

    Steps:
    1. Compute first-layer projectors from grown corners (size chi*D).
    2. Compute second-layer projectors from mid-corners (size chi*D).
    3. Combine into full ``(chi*D^2, chi)`` projector.
    4. Apply to the standard grown edge, then SVD-split into ket/bra.
    """
    D = A.shape[0]
    a = _build_double_layer(A)
    if a.ndim == 8:
        a = a.reshape(D**2, D**2, D**2, D**2)

    if direction == "left":
        # --- ket first (C1/C4 connect to T1/T3 left bond = ket chi) ---
        C1g_ket = jnp.einsum("ab,buc->auc", env.C1, env.T1_ket).reshape(-1, chi_I)
        C4g_ket = jnp.einsum("gh,hdi->gdi", env.C4, env.T3_ket).reshape(-1, chi_I)
        P_ket = _split_ctm_projector(C1g_ket, C4g_ket, chi)
        C1_mid = P_ket.conj().T @ C1g_ket  # (chi, chi_I)
        C4_mid = P_ket.conj().T @ C4g_ket

        C1g_bra = jnp.einsum("ac,cdb->adb", C1_mid, env.T1_bra).reshape(-1, chi)
        C4g_bra = jnp.einsum("ac,cdb->adb", C4_mid, env.T3_bra).reshape(-1, chi)
        P_bra = _split_ctm_projector(C1g_bra, C4g_bra, chi)
        C1_new = P_bra.conj().T @ C1g_bra  # (chi, chi)
        C4_new = P_bra.conj().T @ C4g_bra

        # Combined projector: P_full[(a,u,U), b] = sum_J P_ket[a,u,J] * P_bra[J,U,b]
        P_ket_3d = P_ket.reshape(chi, D, -1)  # (chi, D, chi_k)
        chi_k = P_ket_3d.shape[2]
        P_bra_3d = P_bra.reshape(chi_k, D, -1)  # (chi_k, D, chi)
        P_full = jnp.einsum("auJ,JUb->auUb", P_ket_3d, P_bra_3d)
        chi_new = P_full.shape[3]
        P_full = P_full.reshape(chi * D * D, chi_new)

        # Standard grown edge for T4
        T4_full = jnp.einsum("alc,cLg->alLg", env.T4_ket, env.T4_bra)
        T4_full = T4_full.reshape(chi, D * D, chi)
        T4g = jnp.einsum("alg,udlr->augdr", T4_full, a)
        T4g = T4g.transpose(0, 1, 4, 2, 3).reshape(chi * D * D, D * D, chi * D * D)
        T4_new_full = jnp.einsum("ia,idj,jb->adb", P_full, T4g, P_full)
        T4_ket_new, T4_bra_new = _svd_split_edge(T4_new_full, chi_I)

        return SplitCTMEnvironment(
            C1=C1_new,
            C2=env.C2,
            C3=env.C3,
            C4=C4_new,
            T1_ket=env.T1_ket,
            T1_bra=env.T1_bra,
            T2_ket=env.T2_ket,
            T2_bra=env.T2_bra,
            T3_ket=env.T3_ket,
            T3_bra=env.T3_bra,
            T4_ket=T4_ket_new,
            T4_bra=T4_bra_new,
        )

    elif direction == "right":
        # --- bra first (C2/C3 connect to T1/T3 right bond = bra chi) ---
        # C2 absorbs T1_bra: C2[c,e] * T1_bra[f,U,c] → (e, U, f) = (chi, D, chi_I)
        C2g_bra = jnp.einsum("ce,fUc->eUf", env.C2, env.T1_bra).reshape(-1, chi_I)
        # C3 absorbs T3_bra: C3[i,m] * T3_bra[f,d,i] → (m, d, f) = (chi, D, chi_I)
        C3g_bra = jnp.einsum("im,fdi->mdf", env.C3, env.T3_bra).reshape(-1, chi_I)
        P_bra = _split_ctm_projector(C2g_bra, C3g_bra, chi)
        C2_mid = P_bra.conj().T @ C2g_bra  # (chi, chi_I)
        C3_mid = P_bra.conj().T @ C3g_bra

        # Absorb ket via interlayer: C_mid[a,f] * T_ket[b,u,f] → (a, u, b)
        C2g_ket = jnp.einsum("af,buf->aub", C2_mid, env.T1_ket).reshape(-1, chi)
        C3g_ket = jnp.einsum("af,hdf->adh", C3_mid, env.T3_ket).reshape(-1, chi)
        P_ket = _split_ctm_projector(C2g_ket, C3g_ket, chi)
        C2_new = P_ket.conj().T @ C2g_ket  # (chi, chi)
        C3_new = P_ket.conj().T @ C3g_ket

        # Combined: P_full[(a,u,U), b] = sum_J P_bra[a,U,J] * P_ket[J,u,b]
        P_bra_3d = P_bra.reshape(chi, D, -1)  # (chi, D_bra, chi_k)
        chi_k = P_bra_3d.shape[2]
        P_ket_3d = P_ket.reshape(chi_k, D, -1)  # (chi_k, D_ket, chi)
        P_full = jnp.einsum("aUJ,Jub->auUb", P_bra_3d, P_ket_3d)
        chi_new = P_full.shape[3]
        P_full = P_full.reshape(chi * D * D, chi_new)

        # Standard grown edge for T2
        T2_full = jnp.einsum("alc,cLg->alLg", env.T2_ket, env.T2_bra)
        T2_full = T2_full.reshape(chi, D * D, chi)
        T2g = jnp.einsum("erm,udlr->eumdl", T2_full, a)
        T2g = T2g.transpose(0, 1, 4, 2, 3).reshape(chi * D * D, D * D, chi * D * D)
        T2_new_full = jnp.einsum("ia,idj,jb->adb", P_full, T2g, P_full)
        T2_ket_new, T2_bra_new = _svd_split_edge(T2_new_full, chi_I)

        return SplitCTMEnvironment(
            C1=env.C1,
            C2=C2_new,
            C3=C3_new,
            C4=env.C4,
            T1_ket=env.T1_ket,
            T1_bra=env.T1_bra,
            T2_ket=T2_ket_new,
            T2_bra=T2_bra_new,
            T3_ket=env.T3_ket,
            T3_bra=env.T3_bra,
            T4_ket=env.T4_ket,
            T4_bra=env.T4_bra,
        )

    elif direction == "top":
        # --- ket first (C1/C2 connect to T4/T2 top bond = ket chi) ---
        C1g_ket = jnp.einsum("ab,alg->blg", env.C1, env.T4_ket).reshape(-1, chi_I)
        C2g_ket = jnp.einsum("ce,erm->crm", env.C2, env.T2_ket).reshape(-1, chi_I)
        P_ket = _split_ctm_projector(C1g_ket, C2g_ket, chi)
        C1_mid = P_ket.conj().T @ C1g_ket  # (chi, chi_I)
        C2_mid = P_ket.conj().T @ C2g_ket

        C1g_bra = jnp.einsum("ac,cdb->adb", C1_mid, env.T4_bra).reshape(-1, chi)
        C2g_bra = jnp.einsum("ac,cdb->adb", C2_mid, env.T2_bra).reshape(-1, chi)
        P_bra = _split_ctm_projector(C1g_bra, C2g_bra, chi)
        C1_new = P_bra.conj().T @ C1g_bra  # (chi, chi)
        C2_new = P_bra.conj().T @ C2g_bra

        # Combined: P_full[(a,u,U), b] = sum_J P_ket[a,u,J] * P_bra[J,U,b]
        P_ket_3d = P_ket.reshape(chi, D, -1)
        chi_k = P_ket_3d.shape[2]
        P_bra_3d = P_bra.reshape(chi_k, D, -1)
        P_full = jnp.einsum("auJ,JUb->auUb", P_ket_3d, P_bra_3d)
        chi_new = P_full.shape[3]
        P_full = P_full.reshape(chi * D * D, chi_new)

        # Standard grown edge for T1
        T1_full = jnp.einsum("alc,cLg->alLg", env.T1_ket, env.T1_bra)
        T1_full = T1_full.reshape(chi, D * D, chi)
        T1g = jnp.einsum("buc,udlr->bcdlr", T1_full, a)
        T1g = T1g.transpose(0, 3, 2, 1, 4).reshape(chi * D * D, D * D, chi * D * D)
        T1_new_full = jnp.einsum("ia,idj,jb->adb", P_full, T1g, P_full)
        T1_ket_new, T1_bra_new = _svd_split_edge(T1_new_full, chi_I)

        return SplitCTMEnvironment(
            C1=C1_new,
            C2=C2_new,
            C3=env.C3,
            C4=env.C4,
            T1_ket=T1_ket_new,
            T1_bra=T1_bra_new,
            T2_ket=env.T2_ket,
            T2_bra=env.T2_bra,
            T3_ket=env.T3_ket,
            T3_bra=env.T3_bra,
            T4_ket=env.T4_ket,
            T4_bra=env.T4_bra,
        )

    else:  # bottom
        # --- bra first (C4/C3 connect to T4/T2 bottom bond = bra chi) ---
        # C4 absorbs T4_bra: C4[g,h] * T4_bra[f,L,g] → (h, L, f) = (chi, D, chi_I)
        C4g_bra = jnp.einsum("gh,fLg->hLf", env.C4, env.T4_bra).reshape(-1, chi_I)
        # C3 absorbs T2_bra: C3[i,m] * T2_bra[f,r,m] → (i, r, f) = (chi, D, chi_I)
        C3g_bra = jnp.einsum("im,frm->irf", env.C3, env.T2_bra).reshape(-1, chi_I)
        P_bra = _split_ctm_projector(C4g_bra, C3g_bra, chi)
        C4_mid = P_bra.conj().T @ C4g_bra  # (chi, chi_I)
        C3_mid = P_bra.conj().T @ C3g_bra

        # Absorb ket via interlayer
        # C4_mid[a,f] * T4_ket[b,l,f] → (a, l, b) = (chi, D, chi)
        C4g_ket = jnp.einsum("af,blf->alb", C4_mid, env.T4_ket).reshape(-1, chi)
        # C3_mid[a,f] * T2_ket[e,r,f] → (a, r, e) = (chi, D, chi)
        C3g_ket = jnp.einsum("af,erf->are", C3_mid, env.T2_ket).reshape(-1, chi)
        P_ket = _split_ctm_projector(C4g_ket, C3g_ket, chi)
        C4_new = P_ket.conj().T @ C4g_ket  # (chi, chi)
        C3_new = P_ket.conj().T @ C3g_ket

        # Combined: P_full[(a,u,U), b] = sum_J P_bra[a,U,J] * P_ket[J,u,b]
        P_bra_3d = P_bra.reshape(chi, D, -1)
        chi_k = P_bra_3d.shape[2]
        P_ket_3d = P_ket.reshape(chi_k, D, -1)
        P_full = jnp.einsum("aUJ,Jub->auUb", P_bra_3d, P_ket_3d)
        chi_new = P_full.shape[3]
        P_full = P_full.reshape(chi * D * D, chi_new)

        # Standard grown edge for T3
        T3_full = jnp.einsum("alc,cLg->alLg", env.T3_ket, env.T3_bra)
        T3_full = T3_full.reshape(chi, D * D, chi)
        T3g = jnp.einsum("hdi,udlr->hiulr", T3_full, a)
        T3g = T3g.transpose(0, 3, 2, 1, 4).reshape(chi * D * D, D * D, chi * D * D)
        T3_new_full = jnp.einsum("ia,idj,jb->adb", P_full, T3g, P_full)
        T3_ket_new, T3_bra_new = _svd_split_edge(T3_new_full, chi_I)

        return SplitCTMEnvironment(
            C1=env.C1,
            C2=env.C2,
            C3=C3_new,
            C4=C4_new,
            T1_ket=env.T1_ket,
            T1_bra=env.T1_bra,
            T2_ket=env.T2_ket,
            T2_bra=env.T2_bra,
            T3_ket=T3_ket_new,
            T3_bra=T3_bra_new,
            T4_ket=env.T4_ket,
            T4_bra=env.T4_bra,
        )


def _split_ctm_sweep(
    env: SplitCTMEnvironment,
    A: jax.Array,
    chi: int,
    chi_I: int,
    renormalize: bool,
) -> SplitCTMEnvironment:
    """One full split-CTM sweep: L/R/T/B moves + optional renormalize."""
    env = _split_ctm_move(env, A, chi, chi_I, "left")
    env = _split_ctm_move(env, A, chi, chi_I, "right")
    env = _split_ctm_move(env, A, chi, chi_I, "top")
    env = _split_ctm_move(env, A, chi, chi_I, "bottom")
    if renormalize:

        def normalize(x: jax.Array) -> jax.Array:
            norm = jnp.max(jnp.abs(x))
            return x / (norm + EPS)

        def normalize_pair(
            T_ket: jax.Array, T_bra: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            """Normalize ket/bra pair using a shared factor.

            Uses the geometric mean of the max-abs norms to preserve the
            relative scaling set by the SVD split.
            """
            nk = jnp.max(jnp.abs(T_ket))
            nb = jnp.max(jnp.abs(T_bra))
            shared = jnp.sqrt(nk * nb) + EPS
            return T_ket / shared, T_bra / shared

        T1k, T1b = normalize_pair(env.T1_ket, env.T1_bra)
        T2k, T2b = normalize_pair(env.T2_ket, env.T2_bra)
        T3k, T3b = normalize_pair(env.T3_ket, env.T3_bra)
        T4k, T4b = normalize_pair(env.T4_ket, env.T4_bra)

        env = SplitCTMEnvironment(
            C1=normalize(env.C1),
            C2=normalize(env.C2),
            C3=normalize(env.C3),
            C4=normalize(env.C4),
            T1_ket=T1k,
            T1_bra=T1b,
            T2_ket=T2k,
            T2_bra=T2b,
            T3_ket=T3k,
            T3_bra=T3b,
            T4_ket=T4k,
            T4_bra=T4b,
        )
    return env


def _split_env_to_standard(
    env: SplitCTMEnvironment,
) -> CTMEnvironment:
    """Convert SplitCTMEnvironment to standard CTMEnvironment.

    Contracts each ``(T_ket, T_bra)`` pair over the interlayer bond::

        T_full[a, (uU), b] = sum_c T_ket[a, u, c] * T_bra[c, U, b]
    """
    chi = env.C1.shape[0]

    def merge(T_ket, T_bra):
        D = T_ket.shape[1]
        T = jnp.einsum("auc,cUb->auUb", T_ket, T_bra)
        return T.reshape(chi, D * D, chi)

    return CTMEnvironment(
        C1=env.C1,
        C2=env.C2,
        C3=env.C3,
        C4=env.C4,
        T1=merge(env.T1_ket, env.T1_bra),
        T2=merge(env.T2_ket, env.T2_bra),
        T3=merge(env.T3_ket, env.T3_bra),
        T4=merge(env.T4_ket, env.T4_bra),
    )


def compute_energy_split_ctm(
    A: jax.Array,
    env: SplitCTMEnvironment,
    hamiltonian_gate: jax.Array,
    d: int,
) -> jax.Array:
    """Compute energy using split CTM environment.

    Converts to standard CTMEnvironment then delegates to
    :func:`compute_energy_ctm`.
    """
    std_env = _split_env_to_standard(env)
    return compute_energy_ctm(A, std_env, hamiltonian_gate, d)


def ctm_split(
    A: jax.Array,
    config: CTMConfig,
) -> SplitCTMEnvironment:
    """Compute split-CTM environment for a PEPS with 1x1 unit cell.

    Uses the split-CTMRG algorithm (arXiv:2502.10298) where ket and bra
    layers are kept separate, reducing projector cost from O(chi^3 * D^6)
    to O(chi^3 * D^3).

    Args:
        A:      Site tensor of shape ``(D, D, D, D, d)``.
        config: CTMConfig with ``chi_I`` set.

    Returns:
        Converged SplitCTMEnvironment.
    """
    chi = config.chi
    chi_I = config.chi_I if config.chi_I is not None else chi

    env = _initialize_split_ctm_env(A, chi, chi_I)

    prev_sv = None
    for _ in range(config.max_iter):
        env = _split_ctm_sweep(env, A, chi, chi_I, config.renormalize)

        current_sv = jnp.linalg.svd(env.C1, compute_uv=False)
        if prev_sv is not None:
            diff = _ctm_sv_diff(current_sv, prev_sv)
            if float(diff) < config.conv_tol:
                break
        prev_sv = current_sv

    return env


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
    optimizer = optax.adam(config.gs_learning_rate)

    opt_state = optimizer.init(A)

    best_energy = float("inf")
    best_A = A
    prev_energy = float("inf")

    for _ in range(config.gs_num_steps):
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
    """AD-based ground state optimization for Tensor-protocol iPEPS.

    Uses ``ctm_tensor_converge`` with implicit differentiation through
    the standard Tensor-protocol CTM.  Works with both DenseTensor and
    SymmetricTensor (block-sparse) site tensors.
    """
    import optax

    from tenax.algorithms._ctm_tensor import (
        compute_energy_ctm_tensor,
    )
    from tenax.algorithms.ad_utils import _config_to_tuple, ctm_tensor_converge

    gate = jnp.array(hamiltonian_gate)
    d_phys = gate.shape[0]

    A = A_init
    # Normalize using Tensor norm
    A = A * (1.0 / (A.norm() + 1e-10))

    config_tuple = _config_to_tuple(config.ctm)

    # env treedef is needed to reconstruct CTMTensorEnv from leaves
    # We get it from a dummy initialization
    from tenax.algorithms._ctm_tensor import initialize_ctm_tensor_env

    _env_template = initialize_ctm_tensor_env(A, config.ctm.chi)
    env_treedef = jax.tree.structure(_env_template)

    # Define loss: A -> energy
    def loss_fn(A_param):
        A_norm = A_param * (1.0 / (A_param.norm() + 1e-10))
        env_leaves = ctm_tensor_converge(A_norm, config_tuple)
        env = jax.tree.unflatten(env_treedef, env_leaves)
        energy = compute_energy_ctm_tensor(A_norm, env, gate, d_phys)
        return energy

    # optax works natively on pytrees (DenseTensor/SymmetricTensor are pytrees)
    optimizer = optax.adam(config.gs_learning_rate)

    opt_state = optimizer.init(A)

    best_energy = float("inf")
    best_A = A
    prev_energy = float("inf")

    for _ in range(config.gs_num_steps):
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
        # Re-normalize using Tensor norm
        A = A * (1.0 / (A.norm() + 1e-10))

    # Final CTM environment
    A_final = best_A * (1.0 / (best_A.norm() + 1e-10))
    env_leaves = ctm_tensor_converge(A_final, config_tuple)
    env = jax.tree.unflatten(env_treedef, env_leaves)
    E_gs = float(compute_energy_ctm_tensor(A_final, env, gate, d_phys))

    return A_final, env, E_gs


def _optimize_gs_ad_2site(
    hamiltonian_gate: jax.Array,
    AB_init: tuple[jax.Array, jax.Array] | None,
    config: iPEPSConfig,
) -> tuple[
    tuple[jax.Array, jax.Array],
    tuple[CTMEnvironment, CTMEnvironment],
    float,
]:
    """AD-based ground state optimization for 2-site iPEPS unit cell.

    Uses implicit differentiation through the 2-site CTM fixed point
    to compute gradients of energy w.r.t. both site tensors (A, B).

    Dense-only: 2-site Tensor-protocol CTM is not yet implemented.
    """
    if isinstance(AB_init, tuple) and any(isinstance(t, Tensor) for t in AB_init):
        raise TypeError(
            "_optimize_gs_ad_2site requires dense jax.Array inputs. "
            "Tensor-protocol 2-site CTM is not yet implemented."
        )

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
