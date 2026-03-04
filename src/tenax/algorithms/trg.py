"""Tensor Renormalization Group (TRG) algorithm.

TRG is a coarse-graining algorithm for 2D classical partition functions
on a square lattice. Starting from a single-site tensor ``T_{udlr}`` (up, down,
left, right), TRG iteratively reduces the lattice by a factor of 2 in each
step via SVD splitting and tensor contraction.

Reference: Levin & Nave, PRL 99, 120601 (2007).

Algorithm per step::

    1. SVD split horizontally: T[u,d,l,r] -> F_l[u,l,k] * F_r[k,d,r]
    2. Similarly split vertically: T -> F_u[u,r,k] * F_d[k,d,l]
    3. Contract 4 half-tensors around a plaquette -> T_new

The partition function estimate grows exponentially; we track the
log-normalization at each step to compute ``log(Z)/N``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._tensor_utils import (
    absorb_sqrt_singular_values,
    max_abs_normalize,
)
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry, ZnSymmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor


@dataclass
class TRGConfig:
    """Configuration for TRG coarse-graining.

    Attributes:
        max_bond_dim:   Maximum bond dimension chi after each coarse-graining step.
        num_steps:      Number of coarse-graining iterations (number of times
                        lattice size is halved).
        svd_trunc_err:  Optional maximum truncation error per SVD step.
                        If set and more restrictive than max_bond_dim, takes precedence.
    """

    max_bond_dim: int = 16
    num_steps: int = 10
    svd_trunc_err: float | None = None


def trg(
    tensor: Tensor,
    config: TRGConfig,
) -> jax.Array:
    """TRG coarse-graining for a 2D square lattice partition function.

    The input tensor ``T_{u,d,l,r}`` (up, down, left, right legs) represents
    a single site tensor placed on every site of an infinite 2D square lattice.
    TRG iteratively coarse-grains the lattice by SVD splitting into
    half-tensors, then contracting four half-tensors around a plaquette
    to form the new coarse tensor.

    The partition function estimate is tracked via log normalization:
    ``log(Z)/N = sum_steps log(norm_step) / 2^step``.

    Args:
        tensor: Initial site tensor (DenseTensor or SymmetricTensor) with
                4 legs labeled ("up", "down", "left", "right").
        config: TRGConfig parameters.

    Returns:
        Scalar JAX array: estimated log(Z)/N (free energy per site up to sign).
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"trg() requires a Tensor, got {type(tensor).__name__}")

    T = tensor
    log_norm_total = jnp.zeros((), dtype=T.todense().dtype)

    for step in range(config.num_steps):
        T, log_norm = _trg_step(T, config.max_bond_dim, config.svd_trunc_err)
        # Each TRG step halves the number of tensors (45° rotation of the
        # lattice). The contribution to log(Z)/N from step k is log_norm / 2^(k+1).
        log_norm_total = log_norm_total + log_norm / (2.0 ** (step + 1))

    return log_norm_total


def _trg_step(
    T: Tensor,
    max_bond_dim: int,
    svd_trunc_err: float | None,
) -> tuple[Tensor, jax.Array]:
    """Single TRG coarse-graining step (polymorphic).

    Works identically for DenseTensor and SymmetricTensor via the
    ``truncated_svd`` / ``contract`` / ``max_abs_normalize`` infrastructure.

    Performs:
    1. Horizontal SVD split: T[u,d,l,r] -> F1[u,l,k] * F2[k,d,r]
    2. Vertical SVD split:   T[u,d,l,r] -> F3[u,r,m] * F4[m,d,l]
    3. Contract 4 half-tensors around a plaquette to get T_new.
    4. Normalize T_new and track log normalization.

    Args:
        T:              Site tensor with labels ("up", "down", "left", "right").
        max_bond_dim:   Maximum chi after truncation.
        svd_trunc_err:  Optional max truncation error.

    Returns:
        (T_new, log_norm) where T_new is the coarse-grained tensor and
        log_norm = log(||T_new|| before normalization).
    """
    # --- Horizontal split ---
    # Group (up, left) vs (down, right) → F1[up,left,k] * F2[k,down,right]
    U_h, s_h, Vh_h, _ = truncated_svd(
        T,
        left_labels=["up", "left"],
        right_labels=["down", "right"],
        new_bond_label="k",
        max_singular_values=max_bond_dim,
        max_truncation_err=svd_trunc_err,
    )
    F1, F2 = absorb_sqrt_singular_values(U_h, s_h, Vh_h, "k")
    # F1: (up, left, k),  F2: (k, down, right)

    # --- Vertical split ---
    # Group (up, right) vs (down, left) → F3[up,right,m] * F4[m,down,left]
    U_v, s_v, Vh_v, _ = truncated_svd(
        T,
        left_labels=["up", "right"],
        right_labels=["down", "left"],
        new_bond_label="m",
        max_singular_values=max_bond_dim,
        max_truncation_err=svd_trunc_err,
    )
    F3, F4 = absorb_sqrt_singular_values(U_v, s_v, Vh_v, "m")
    # F3: (up, right, m),  F4: (m, down, left)

    # --- Relabel bond indices to avoid collision, then contract ---
    # Plaquette arrangement:
    #   F3 (upper-right): [up, right, m]  — m becomes new "up"
    #   F1 (upper-left):  [up, left, k]   — k becomes new "left"
    #   F4 (lower-left):  [m', down, left] — m' becomes new "down"
    #   F2 (lower-right): [k', down, right] — k' becomes new "right"
    #
    # Shared indices for contraction:
    #   up:    F3 ↔ F1
    #   left:  F1 ↔ F4
    #   right: F3 ↔ F2
    #   down:  F4 ↔ F2
    F2 = F2.relabel("k", "K")  # (K, down, right)
    F4 = F4.relabel("m", "M")  # (M, down, left)

    T_new = contract(F3, F1, F4, F2, output_labels=("m", "M", "k", "K"))
    T_new = T_new.relabels({"m": "up", "M": "down", "k": "left", "K": "right"})

    # --- Normalize ---
    T_new, log_norm = max_abs_normalize(T_new)
    return T_new, log_norm


def compute_ising_tensor(
    beta: float,
    J: float = 1.0,
    symmetric: bool = False,
) -> Tensor:
    """Build the initial transfer matrix tensor for the 2D Ising model.

    Constructs the local tensor T_{udlr} for the partition function
    Z = Tr prod T_{s_i s_j s_k s_l} where the trace is over all spin configs.

    The tensor is derived from the Boltzmann weight
    T_{udlr} = sum_{s} sqrt(Q_{u,s}) sqrt(Q_{d,s}) sqrt(Q_{l,s}) sqrt(Q_{r,s})
    where Q_{a,b} = exp(beta*J*s_a*s_b) / 2 and s ∈ {-1, +1} (or {0, 1}).

    Args:
        beta:      Inverse temperature beta = 1/(k_B T).
        J:         Nearest-neighbor coupling constant (J>0 ferromagnet).
        symmetric: If True, return a SymmetricTensor with Z₂ symmetry
                   (spin-flip symmetry of the Ising model).

    Returns:
        Tensor of shape (2, 2, 2, 2) with legs ("up", "down", "left", "right").
        DenseTensor when symmetric=False, SymmetricTensor when symmetric=True.

    Note:
        At the 2D Ising critical temperature,
        beta_c = ln(1 + sqrt(2)) / (2*J) ≈ 0.4407 / J,
        TRG should reproduce the Onsager exact free energy.
    """
    # Q matrix: Q[i,j] = exp(beta * J * sigma_i * sigma_j)
    # spin values: sigma_0 = +1, sigma_1 = -1
    spins = jnp.array([1.0, -1.0])

    # Q[i, j] = exp(beta * J * sigma_i * sigma_j)
    Q = jnp.exp(beta * J * jnp.outer(spins, spins))

    # Matrix square root of Q (NOT element-wise sqrt).
    # Q is symmetric positive-definite, so sqrtQ @ sqrtQ = Q.
    evals, evecs = jnp.linalg.eigh(Q)
    sqrtQ = evecs @ jnp.diag(jnp.sqrt(evals)) @ evecs.T

    # Build T_{udlr} = sum_s sqrtQ[u,s] * sqrtQ[d,s] * sqrtQ[l,s] * sqrtQ[r,s]
    # This ensures the trace correctly gives Z.
    T = jnp.einsum("us,ds,ls,rs->udlr", sqrtQ, sqrtQ, sqrtQ, sqrtQ)

    if symmetric:
        # Z₂ eigenbasis: the Hadamard H diagonalizes Q, so the tensor
        # T' = (H⊗H⊗H⊗H) T in the eigenbasis has Z₂ block structure.
        # Elements with sum(charges) odd mod 2 are exactly zero.
        H = jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2.0)
        T_z2 = jnp.einsum("ua,vb,wc,xd,uvwx->abcd", H, H, H, H, T)
        sym = ZnSymmetry(2)
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex(sym, charges, FlowDirection.IN, label="up"),
            TensorIndex(sym, charges, FlowDirection.OUT, label="down"),
            TensorIndex(sym, charges, FlowDirection.IN, label="left"),
            TensorIndex(sym, charges, FlowDirection.OUT, label="right"),
        )
        return SymmetricTensor.from_dense(T_z2, indices)

    sym = U1Symmetry()
    bond_2 = np.zeros(2, dtype=np.int32)
    indices = (
        TensorIndex(sym, bond_2, FlowDirection.IN, label="up"),
        TensorIndex(sym, bond_2, FlowDirection.OUT, label="down"),
        TensorIndex(sym, bond_2, FlowDirection.IN, label="left"),
        TensorIndex(sym, bond_2, FlowDirection.OUT, label="right"),
    )
    return DenseTensor(T, indices)


def ising_free_energy_exact(beta: float, J: float = 1.0) -> float:
    """Compute the exact 2D Ising free energy per site via Onsager's formula.

    The exact result for ln(Z)/N on the square lattice is:

      ln(Z)/N = ln(2) + (1/(2*pi^2)) * int_0^pi int_0^pi
                ln(cosh^2(2K) - sinh(2K)*(cos t1 + cos t2)) dt1 dt2

    where K = beta*J. The inner integral over t2 is evaluated analytically
    via int_0^pi ln(A - B*cos(t)) dt = pi*ln((A + sqrt(A^2-B^2))/2),
    reducing to a single integral over t1.

    Reference: Onsager, Phys. Rev. 65, 117 (1944).

    Args:
        beta: Inverse temperature.
        J:    Coupling constant.

    Returns:
        Free energy per site f = -ln(Z)/(N*beta).
    """
    if beta == 0:
        return -float(jnp.log(2.0))

    K = beta * J
    c2K = float(jnp.cosh(2 * K))
    s2K = float(jnp.sinh(2 * K))

    # Single integral derived from the Onsager double integral.
    # Inner integral over t2 evaluated analytically:
    #   int_0^pi ln(A - s2K*cos(t2)) dt2 = pi*ln((A + sqrt(A^2 - s2K^2))/2)
    # where A = c2K^2 - s2K*cos(t1).
    # Use half-step offset grid to avoid the integrable log singularity at t1=0
    # when K = K_c (critical temperature).
    N_pts = 10000
    dt = np.pi / N_pts
    t1 = jnp.linspace(dt / 2, np.pi - dt / 2, N_pts)

    A = c2K**2 - s2K * jnp.cos(t1)
    disc = A**2 - s2K**2
    integrand = jnp.log((A + jnp.sqrt(jnp.maximum(disc, 0.0))) / 2)

    # integral = (1/pi) * int_0^pi integrand dt1
    integral = float(jnp.trapezoid(integrand, t1)) / np.pi
    # f = -ln(Z)/(N*beta) = -ln(2)/beta - integral/(2*beta)
    free_energy = -float(jnp.log(2.0)) / beta - integral / (2 * beta)
    return float(free_energy)
