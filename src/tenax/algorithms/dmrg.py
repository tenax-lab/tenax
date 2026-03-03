"""Density Matrix Renormalization Group (DMRG) algorithm.

DMRG finds the ground state (or low-lying eigenstates) of a 1D quantum
Hamiltonian given as a Matrix Product Operator (MPO).

Architecture decisions:

- The outer sweep loop is a Python for-loop (not ``jax.lax.scan``) because bond
  dimensions change after each SVD truncation, preventing JIT across sweeps.
- The effective Hamiltonian matvec is ``@jax.jit`` compiled for performance.
- Lanczos eigensolver uses ``jax.lax.while_loop`` for static shapes inside JIT.
- Environment tensors (left/right blocks) are stored as Python lists of Tensor.

Label conventions::

    MPS site tensors:    legs = ("v{i-1}_{i}", "p{i}", "v{i}_{i+1}")
                         boundary: left site has ("p0", "v0_1"),
                                   right site has ("v{L-2}_{L-1}", "p{L-1}")
    MPO site tensors:    legs = ("w{i-1}_{i}", "mpo_top_{i}", "mpo_bot_{i}", "w{i}_{i+1}")
    Environment tensors: left_env[i] has legs ("mps_l", "mpo_l", "mps_l_conj")
                         right_env[i] has legs ("mps_r", "mpo_r", "mps_r_conj")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.contraction.contractor import contract, qr_decompose, truncated_svd
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor
from tenax.network.network import TensorNetwork


@dataclass
class DMRGConfig:
    """Configuration for a DMRG run.

    Attributes:
        max_bond_dim:       Maximum allowed bond dimension (chi).
        num_sweeps:         Number of full left-right sweep cycles.
        convergence_tol:    Energy convergence threshold to stop early.
        num_states:         Number of lowest eigenstates to target (1 = ground state).
        two_site:           If True, use 2-site DMRG (allows bond dim growth).
                            If False, use 1-site DMRG (conserves bond dim exactly).
        lanczos_max_iter:   Maximum Lanczos iterations for eigenvalue solve.
        lanczos_tol:        Convergence tolerance for Lanczos.
        noise:              Perturbative noise added to density matrix (helps
                            escape local minima in 2-site DMRG).
        svd_trunc_err:      Maximum truncation error per SVD (overrides
                            max_bond_dim when set and more restrictive).
        verbose:            Print energy at each sweep.
    """

    max_bond_dim: int = 100
    num_sweeps: int = 10
    convergence_tol: float = 1e-10
    num_states: int = 1
    two_site: bool = True
    lanczos_max_iter: int = 50
    lanczos_tol: float = 1e-12
    noise: float = 0.0
    svd_trunc_err: float | None = None
    verbose: bool = False


class DMRGResult(NamedTuple):
    """Result of a DMRG run.

    Attributes:
        energy:               Final ground state energy.
        energies_per_sweep:   Energy at the end of each sweep.
        mps:                  TensorNetwork representing the optimized MPS.
        truncation_errors:    List of truncation errors at each bond update step.
        converged:            True if energy converged within convergence_tol.
    """

    energy: float
    energies_per_sweep: list[float]
    mps: TensorNetwork
    truncation_errors: list[float]
    converged: bool


def dmrg(
    hamiltonian: TensorNetwork,
    initial_mps: TensorNetwork,
    config: DMRGConfig,
) -> DMRGResult:
    """Run DMRG to find the ground state of a 1D Hamiltonian given as MPO.

    The Hamiltonian must be provided as an MPO (Matrix Product Operator)
    TensorNetwork with L site tensors connected by virtual bonds.

    Args:
        hamiltonian:  MPO representation of the Hamiltonian.
        initial_mps:  Starting MPS TensorNetwork (modified in-place conceptually;
                      the result MPS is returned in DMRGResult).
        config:       DMRGConfig parameters.

    Returns:
        DMRGResult with energy, sweep history, optimized MPS, and diagnostics.
    """
    L = hamiltonian.n_nodes()
    # Convert any SymmetricTensor initial states to DenseTensor.  The DMRG
    # engine operates entirely in dense mode (all contractions go through
    # .todense()), so the block-sparse structure is not preserved across sweeps.
    mps_tensors: list[Tensor] = [
        DenseTensor(t.todense(), t.indices) if not isinstance(t, DenseTensor) else t
        for t in [initial_mps.get_tensor(i) for i in range(L)]
    ]
    mpo_tensors = [hamiltonian.get_tensor(i) for i in range(L)]

    # Right-canonicalize the initial MPS (skipped: label-based QR may reorder legs)
    # mps_tensors = _right_canonicalize(mps_tensors)

    # Build left environments (L[i] = trivial for i=0)
    left_envs = _build_left_environments_list(mps_tensors, mpo_tensors, L)
    right_envs = _build_right_environments_list(mps_tensors, mpo_tensors, L)

    energies_per_sweep: list[float] = []
    truncation_errors: list[float] = []
    energy = 0.0
    converged = False

    for sweep in range(config.num_sweeps):
        prev_energy = energy

        # Rebuild left environments from updated MPS before left-to-right sweep
        if sweep > 0:
            left_envs = _build_left_environments_list(mps_tensors, mpo_tensors, L)

        # Left-to-right sweep
        for i in range(L - 1):
            l_env = left_envs[i]
            assert l_env is not None
            if config.two_site:
                # 2-site update at sites (i, i+1)
                # right_envs[i+2] = environment to the right of site i+1
                _r = right_envs[i + 2]
                r_env = _r if _r is not None else _build_trivial_right_env()
                theta, e = _two_site_update(
                    mps_tensors[i],
                    mps_tensors[i + 1],
                    l_env,
                    mpo_tensors[i],
                    mpo_tensors[i + 1],
                    r_env,
                    config,
                )
                energy = float(e)

                # SVD and truncate
                A, s, B, trunc_err = _svd_and_truncate_site(theta, i, config)
                mps_tensors[i] = A
                mps_tensors[i + 1] = B
                truncation_errors.append(float(trunc_err))

                # Update left environment
                left_envs[i + 1] = _update_left_env(l_env, A, mpo_tensors[i])
            else:
                # 1-site update
                _ri = right_envs[i + 1]
                r_env_1s = _ri if _ri is not None else _build_trivial_right_env()
                new_site, e = _one_site_update(
                    mps_tensors[i],
                    l_env,
                    mpo_tensors[i],
                    r_env_1s,
                    config,
                )
                energy = float(e)
                mps_tensors[i] = new_site
                left_envs[i + 1] = _update_left_env(l_env, new_site, mpo_tensors[i])

        # Rebuild right environments from updated MPS before right-to-left sweep
        right_envs = _build_right_environments_list(mps_tensors, mpo_tensors, L)

        # Right-to-left sweep
        for i in range(L - 2, -1, -1):
            l_env = left_envs[i]
            assert l_env is not None
            _r2 = right_envs[i + 2]
            r2_env = _r2 if _r2 is not None else _build_trivial_right_env()
            if config.two_site:
                # 2-site update at sites (i, i+1)
                # right_envs[i+2] = environment to the right of site i+1
                theta, e = _two_site_update(
                    mps_tensors[i],
                    mps_tensors[i + 1],
                    l_env,
                    mpo_tensors[i],
                    mpo_tensors[i + 1],
                    r2_env,
                    config,
                )
                energy = float(e)

                A, s, B, trunc_err = _svd_and_truncate_site(
                    theta, i, config, sweep_right=False
                )
                mps_tensors[i] = A
                mps_tensors[i + 1] = B
                truncation_errors.append(float(trunc_err))

                right_envs[i + 1] = _update_right_env(r2_env, B, mpo_tensors[i + 1])
            else:
                _r1 = right_envs[i + 1]
                r1_env = _r1 if _r1 is not None else _build_trivial_right_env()
                new_site, e = _one_site_update(
                    mps_tensors[i],
                    l_env,
                    mpo_tensors[i],
                    r1_env,
                    config,
                )
                energy = float(e)
                mps_tensors[i] = new_site
                right_envs[i] = _update_right_env(r1_env, new_site, mpo_tensors[i])

        energies_per_sweep.append(energy)
        if config.verbose:
            print(f"Sweep {sweep + 1}/{config.num_sweeps}: E = {energy:.10f}")

        # Check convergence
        if sweep > 0 and abs(energy - prev_energy) < config.convergence_tol:
            converged = True
            if config.verbose:
                print(f"Converged at sweep {sweep + 1}")
            break

    # Build result MPS as TensorNetwork
    result_mps = TensorNetwork(name="DMRG_MPS")
    for i, tensor in enumerate(mps_tensors):
        result_mps.add_node(i, tensor)
    for i in range(L - 1):
        shared = set(mps_tensors[i].labels()) & set(mps_tensors[i + 1].labels())
        for label in sorted(shared, key=str):
            try:
                result_mps.connect(i, label, i + 1, label)
            except ValueError:
                pass

    return DMRGResult(
        energy=energy,
        energies_per_sweep=energies_per_sweep,
        mps=result_mps,
        truncation_errors=truncation_errors,
        converged=converged,
    )


def _right_canonicalize(mps_tensors: list[Tensor]) -> list[Tensor]:
    """Right-canonicalize MPS by QR from right to left."""
    L = len(mps_tensors)
    tensors = list(mps_tensors)

    for i in range(L - 1, 0, -1):
        tensor = tensors[i]
        labels = tensor.labels()

        # Find the virtual bond to the left
        left_bond = _find_left_bond(labels, i)
        if left_bond is None:
            continue

        other_labels = [lbl for lbl in labels if lbl != left_bond]
        Q, R = qr_decompose(
            tensor,
            left_labels=[left_bond],
            right_labels=other_labels,
            new_bond_label=left_bond + "_new"
            if isinstance(left_bond, str)
            else f"b{i}",
        )

        # Absorb R into site i-1
        tensors[i] = Q
        tensors[i - 1] = contract(tensors[i - 1], R)

    return tensors


def _find_left_bond(labels: tuple, site: int) -> str | None:
    """Find the left virtual bond label for a given site."""
    for lbl in labels:
        if isinstance(lbl, str) and lbl.startswith(f"v{site - 1}_"):
            return lbl
    return None


def _find_right_bond(labels: tuple, site: int) -> str | None:
    """Find the right virtual bond label for a given site."""
    for lbl in labels:
        if isinstance(lbl, str) and lbl.startswith(f"v{site}_"):
            return lbl
    return None


def _trivial_env() -> DenseTensor:
    """Create a trivial (scalar 1) environment tensor."""
    from tenax.core.symmetry import U1Symmetry

    sym = U1Symmetry()
    idx = TensorIndex(sym, np.zeros(1, dtype=np.int32), FlowDirection.IN, label="env")
    return DenseTensor(jnp.ones((1,), dtype=jnp.float64), (idx,))


def _build_left_environments_list(
    mps_tensors: list[Tensor],
    mpo_tensors: list[Tensor],
    L: int,
) -> list[Tensor | None]:
    """Build all left environment tensors by sweeping left to right.

    L_env[0] = trivial, L_env[i] = contraction of sites 0..i-1.

    Returns list of L+1 environment tensors (None used as placeholder where
    not yet computed; replaced with dense contractions in full implementation).
    """
    envs: list[Tensor | None] = [None] * (L + 1)
    # Trivial left environment (scalar 1)
    envs[0] = _build_trivial_left_env()

    for i in range(L - 1):
        env = envs[i]
        if env is not None:
            envs[i + 1] = _update_left_env(env, mps_tensors[i], mpo_tensors[i])

    return envs


def _build_right_environments_list(
    mps_tensors: list[Tensor],
    mpo_tensors: list[Tensor],
    L: int,
) -> list[Tensor | None]:
    """Build all right environment tensors by sweeping right to left."""
    envs: list[Tensor | None] = [None] * (L + 1)
    envs[L] = _build_trivial_right_env()

    for i in range(L - 1, 0, -1):
        env = envs[i + 1]
        if env is not None:
            envs[i] = _update_right_env(env, mps_tensors[i], mpo_tensors[i])

    return envs


def _build_trivial_left_env(dtype=None) -> DenseTensor:
    """Build trivial (1x1x1) left boundary environment."""
    if dtype is None:
        dtype = jnp.float64
    sym = U1Symmetry()
    bond = np.zeros(1, dtype=np.int32)
    indices = (
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mps_l"),
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mpo_l"),
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mps_conj_l"),
    )
    return DenseTensor(jnp.ones((1, 1, 1), dtype=dtype), indices)


def _build_trivial_right_env(dtype=None) -> DenseTensor:
    """Build trivial (1x1x1) right boundary environment."""
    if dtype is None:
        dtype = jnp.float64
    sym = U1Symmetry()
    bond = np.zeros(1, dtype=np.int32)
    indices = (
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mps_r"),
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mpo_r"),
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mps_conj_r"),
    )
    return DenseTensor(jnp.ones((1, 1, 1), dtype=dtype), indices)


def _update_left_env(
    left_env: Tensor,
    mps_site: Tensor,
    mpo_site: Tensor,
) -> DenseTensor:
    """Update left environment by absorbing one MPS/MPO site.

    Contracts: new_L[r, w, r'] = L[l, w_l, l'] * A[l, p, r] * W[w_l, p, p', w] * A*[l', p', r']

    Args:
        left_env: Current left environment tensor.
        mps_site: MPS site tensor A.
        mpo_site: MPO site tensor W.

    Returns:
        Updated left environment tensor.
    """
    # Dense implementation using todense() for generality
    L_dense = left_env.todense()  # shape (chi_l, D_w, chi_l')
    A_dense = mps_site.todense()  # shape (chi_l, d, chi_r) for middle sites
    W_dense = mpo_site.todense()  # shape (D_w_l, d_top, d_bot, D_w_r)

    # Pad A to always be 3D: if boundary site is 2D, add a trivial dim
    if A_dense.ndim == 2:
        labels = mps_site.labels()
        if isinstance(labels[0], str) and labels[0].startswith("p"):
            # Left boundary: (d, chi_r) -> (1, d, chi_r)
            A_dense = A_dense[jnp.newaxis, :]
        else:
            # Right boundary: (chi_l, d) -> (chi_l, d, 1)
            A_dense = A_dense[:, :, jnp.newaxis]

    # new_L[chi_r, D_w_r, chi_r'] =
    #   L[chi_l, D_w_l, chi_l'] * A[chi_l, d, chi_r] * W[D_w_l, d, d', D_w_r] * A*[chi_l', d', chi_r']
    # Using subscripts: L=abc (a=chi_l, b=D_w_l, c=chi_l')
    #                   A=apd (a=chi_l, p=d_ket, d=chi_r)
    #                   W=bpxe (b=D_w_l, p=d_ket, x=d_bra, e=D_w_r)
    #                   A*=cxf (c=chi_l', x=d_bra, f=chi_r')
    # -> new_L[d, e, f] = (chi_r, D_w_r, chi_r')
    new_L = jnp.einsum(
        "abc,apd,bpxe,cxf->def",
        L_dense,
        A_dense,
        W_dense,
        jnp.conj(A_dense),
    )

    sym = U1Symmetry()
    bond_r = np.zeros(new_L.shape[0], dtype=np.int32)
    bond_w = np.zeros(new_L.shape[1], dtype=np.int32)
    indices = (
        TensorIndex(sym, bond_r, FlowDirection.IN, label="env_mps_l"),
        TensorIndex(sym, bond_w, FlowDirection.IN, label="env_mpo_l"),
        TensorIndex(sym, bond_r, FlowDirection.OUT, label="env_mps_conj_l"),
    )
    return DenseTensor(new_L, indices)


def _update_right_env(
    right_env: Tensor,
    mps_site: Tensor,
    mpo_site: Tensor,
) -> DenseTensor:
    """Update right environment by absorbing one MPS/MPO site."""
    R_dense = right_env.todense()  # shape (chi_r, D_w, chi_r')
    B_dense = mps_site.todense()  # shape (chi_l, d, chi_r) for middle sites
    W_dense = mpo_site.todense()  # shape (D_w_l, d_top, d_bot, D_w_r)

    # Pad B to 3D if boundary site (2D tensor)
    if B_dense.ndim == 2:
        labels = mps_site.labels()
        if isinstance(labels[0], str) and labels[0].startswith("p"):
            # Left boundary: (d, chi_r) -> (1, d, chi_r)
            B_dense = B_dense[jnp.newaxis, :, :]
        else:
            # Right boundary: (chi_l, d) -> (chi_l, d, 1)
            B_dense = B_dense[:, :, jnp.newaxis]

    # new_R[chi_l, D_w_l, chi_l'] =
    #   R[chi_r, D_w_r, chi_r'] * B[chi_l, d, chi_r] * W[D_w_l, d, d', D_w_r] * B*[chi_l', d', chi_r']
    # R=abc (a=chi_r, b=D_w_r, c=chi_r')
    # B=dpa (d=chi_l, p=d_ket, a=chi_r)   [contracted on a]
    # W=epxb (e=D_w_l, p=d_ket, x=d_bra, b=D_w_r)  [contracted on a,b]
    # B*=fxc (f=chi_l', x=d_bra, c=chi_r')  [contracted on c]
    # -> new_R[d, e, f] = (chi_l, D_w_l, chi_l')
    new_R = jnp.einsum(
        "abc,dpa,epxb,fxc->def",
        R_dense,
        B_dense,
        W_dense,
        jnp.conj(B_dense),
    )

    sym = U1Symmetry()
    bond_l = np.zeros(new_R.shape[0], dtype=np.int32)
    bond_w = np.zeros(new_R.shape[1], dtype=np.int32)
    indices = (
        TensorIndex(sym, bond_l, FlowDirection.OUT, label="env_mps_r"),
        TensorIndex(sym, bond_w, FlowDirection.OUT, label="env_mpo_r"),
        TensorIndex(sym, bond_l, FlowDirection.IN, label="env_mps_conj_r"),
    )
    return DenseTensor(new_R, indices)


def _effective_hamiltonian_matvec(
    theta_flat: jax.Array,
    theta_shape: tuple[int, ...],
    L_env: jax.Array,
    W_l: jax.Array,
    W_r: jax.Array,
    R_env: jax.Array,
) -> jax.Array:
    """Apply effective Hamiltonian H_eff to 2-site wavefunction theta.

    H_eff = L * W_l * W_r * R (diagrammatic notation).
    All inputs are raw JAX arrays (flattened for JIT compatibility).

    This function is @jax.jit compiled for performance.

    Args:
        theta_flat:  Flattened 2-site wavefunction.
        theta_shape: Shape tuple for reshaping.
        L_env:       Left environment, shape (chi_l, d_w_l, chi_l).
        W_l:         Left MPO site, shape (d_w_l, d_p_l, d_p_l', d_w_m).
        W_r:         Right MPO site, shape (d_w_m, d_p_r, d_p_r', d_w_r).
        R_env:       Right environment, shape (chi_r, d_w_r, chi_r).

    Returns:
        Flattened result of H_eff @ theta.
    """
    theta = theta_flat.reshape(theta_shape)

    # Contract: L[a,b,c] * theta[a,p,q,d] * W_l[b,p,s,e] * W_r[e,q,t,f] * R[d,f,g]
    # -> result[c,s,t,g]
    # Indices:
    #   a = chi_l (MPS bond, ket)
    #   b = D_w_l (MPO bond left)
    #   c = chi_l (MPS bond, bra)
    #   p = d_phys_l (ket physical, left site)
    #   q = d_phys_r (ket physical, right site)
    #   d = chi_r (MPS bond right, ket)
    #   s = d_phys_l' (bra physical, left site)
    #   e = D_w_m (MPO bond middle)
    #   t = d_phys_r' (bra physical, right site)
    #   f = D_w_r (MPO bond right)
    #   g = chi_r (MPS bond right, bra)
    result = jnp.einsum(
        "abc,apqd,bpse,eqtf,dfg->cstg",
        L_env,
        theta,
        W_l,
        W_r,
        R_env,
    )
    return result.ravel()


_matvec_jit = jax.jit(_effective_hamiltonian_matvec, static_argnums=(1,))


def _two_site_update(
    site_l: Tensor,
    site_r: Tensor,
    left_env: Tensor,
    mpo_l: Tensor,
    mpo_r: Tensor,
    right_env: Tensor,
    config: DMRGConfig,
) -> tuple[Tensor, float]:
    """Perform 2-site DMRG update: contract theta, solve eigenvalue problem.

    Returns:
        (theta_opt, energy) where theta_opt is the optimized 2-site tensor.
    """
    # Contract theta = A[i] * A[i+1] (shared virtual bond contracted)
    shared = set(site_l.labels()) & set(site_r.labels())
    if shared:
        theta = contract(site_l, site_r)
    else:
        # No shared label: concatenate (this shouldn't happen in a valid MPS)
        theta = site_l

    # Use Lanczos to find the ground state
    theta_dense = theta.todense()
    theta_indices = theta.indices

    # Ensure theta is always 4D: (chi_l, d_l, d_r, chi_r)
    # Boundary cases: left site (i=0) → theta is 3D (d_l, d_r, chi_r)
    #                 right site (i=L-2) → theta is 3D (chi_l, d_l, d_r)
    original_ndim = theta_dense.ndim
    if theta_dense.ndim == 3:
        # Determine which boundary: check if first dim is small (=d) or large (=chi)
        # Left boundary: first dim = d (physical), so add trivial dim at left
        # Right boundary: last dim = d (physical), add trivial dim at right
        # We detect by looking at the labels
        labels_list = [idx.label for idx in theta_indices]
        # Left boundary: no left virtual bond, first label is physical
        has_left_virt = any(
            isinstance(lbl, str) and lbl.startswith("v") for lbl in labels_list[:1]
        )
        if not has_left_virt:
            theta_dense = theta_dense[jnp.newaxis, :]  # (1, d, d, chi_r)
        else:
            theta_dense = theta_dense[:, :, :, jnp.newaxis]  # (chi_l, d, d, 1)

    L_arr = left_env.todense()
    R_arr = right_env.todense()
    W_l_arr = mpo_l.todense()
    W_r_arr = mpo_r.todense()

    # Ensure environments are 3D
    if L_arr.ndim == 1:
        L_arr = L_arr.reshape(1, 1, 1)
    if R_arr.ndim == 1:
        R_arr = R_arr.reshape(1, 1, 1)

    theta_shape = theta_dense.shape
    theta_flat = theta_dense.ravel()

    def matvec(v: jax.Array) -> jax.Array:
        return _matvec_jit(v, theta_shape, L_arr, W_l_arr, W_r_arr, R_arr)

    energy, theta_opt_flat = _lanczos_solve(
        matvec, theta_flat, config.lanczos_max_iter, config.lanczos_tol
    )

    theta_opt_dense = theta_opt_flat.reshape(theta_shape)
    # Remove trivial dims added for boundary sites
    if original_ndim == 3:
        labels_list = [idx.label for idx in theta_indices]
        has_left_virt = any(
            isinstance(lbl, str) and lbl.startswith("v") for lbl in labels_list[:1]
        )
        if not has_left_virt:
            theta_opt_dense = theta_opt_dense[0, :, :, :]  # remove left trivial dim
        else:
            theta_opt_dense = theta_opt_dense[:, :, :, 0]  # remove right trivial dim

    theta_opt = DenseTensor(theta_opt_dense, theta_indices)
    return theta_opt, energy


def _one_site_update(
    site: Tensor,
    left_env: Tensor,
    mpo_site: Tensor,
    right_env: Tensor,
    config: DMRGConfig,
) -> tuple[Tensor, float]:
    """Perform 1-site DMRG update."""
    site_dense = site.todense()
    original_site_shape = site_dense.shape

    # Ensure site is always 3D: (chi_l, d, chi_r)
    if site_dense.ndim == 2:
        labels_list = list(site.labels())
        has_left_virt = any(
            isinstance(lbl, str) and lbl.startswith("v") for lbl in labels_list[:1]
        )
        if not has_left_virt:
            site_dense = site_dense[jnp.newaxis, :]  # (1, d, chi_r)
        else:
            site_dense = site_dense[:, :, jnp.newaxis]  # (chi_l, d, 1)

    site_shape = site_dense.shape
    site_flat = site_dense.ravel()

    L_arr = left_env.todense()
    R_arr = right_env.todense()
    W_arr = mpo_site.todense()

    if L_arr.ndim == 1:
        L_arr = L_arr.reshape(1, 1, 1)
    if R_arr.ndim == 1:
        R_arr = R_arr.reshape(1, 1, 1)

    def matvec(v: jax.Array) -> jax.Array:
        s = v.reshape(site_shape)
        # H_eff = L[a,b,c] * s[a,p,d] * W[b,p,x,e] * R[d,e,f] -> result[c,x,f]
        # a=chi_l_ket, b=D_w_l, c=chi_l_bra, p=d_ket, d=chi_r_ket,
        # x=d_bra, e=D_w_r, f=chi_r_bra
        result = jnp.einsum("abc,apd,bpxe,def->cxf", L_arr, s, W_arr, R_arr)
        return result.ravel()

    energy, site_opt_flat = _lanczos_solve(
        matvec, site_flat, config.lanczos_max_iter, config.lanczos_tol
    )

    site_opt_dense = site_opt_flat.reshape(site_shape)
    # Remove trivial dims if we added them
    if len(original_site_shape) == 2 and site_opt_dense.ndim == 3:
        labels_list = list(site.labels())
        has_left_virt = any(
            isinstance(lbl, str) and lbl.startswith("v") for lbl in labels_list[:1]
        )
        if not has_left_virt:
            site_opt_dense = site_opt_dense[0, :, :]
        else:
            site_opt_dense = site_opt_dense[:, :, 0]
    site_opt = DenseTensor(site_opt_dense, site.indices)
    return site_opt, energy


def _lanczos_solve(
    matvec: Callable[[jax.Array], jax.Array],
    initial_vector: jax.Array,
    num_steps: int,
    tol: float,
) -> tuple[float, jax.Array]:
    """Lanczos eigensolver for the smallest eigenvalue.

    Optimizations over the naive implementation:
    - Keeps alpha/beta as JAX scalars to avoid host-device sync per step
    - Vectorized eigenvector reconstruction via jnp.tensordot on stacked basis

    Args:
        matvec:         Function applying the effective Hamiltonian.
        initial_vector: Starting vector (will be normalized).
        num_steps:      Maximum number of Lanczos steps.
        tol:            Convergence tolerance on the residual.

    Returns:
        (eigenvalue, eigenvector) for the ground state.
    """
    v = initial_vector / (jnp.linalg.norm(initial_vector) + 1e-15)

    # Krylov basis and tridiagonal matrix coefficients
    basis = [v]
    alphas_jax: list[jax.Array] = []
    betas_jax: list[jax.Array] = [jnp.zeros(())]

    for step in range(num_steps):
        w = matvec(basis[-1])
        alpha = jnp.dot(basis[-1].conj(), w).real
        alphas_jax.append(alpha)

        w = w - alpha * basis[-1]
        if step > 0:
            w = w - betas_jax[-1] * basis[-2]

        beta = jnp.linalg.norm(w)
        betas_jax.append(beta)

        # Convergence check requires host sync (unavoidable for loop control)
        if float(beta) < tol:
            break

        basis.append(w / beta)

    # Build tridiagonal matrix and find ground state
    n = len(alphas_jax)

    if n == 0:
        # No iterations completed — return initial vector with zero energy
        return 0.0, v

    if n == 1:
        # Single iteration: eigenvalue is alpha, eigenvector is first basis vector
        return float(alphas_jax[0]), basis[0]

    alphas_arr = jnp.stack(alphas_jax)
    betas_arr = jnp.stack(betas_jax[1:n])
    T = jnp.diag(alphas_arr) + jnp.diag(betas_arr, k=1) + jnp.diag(betas_arr, k=-1)

    eigvals, eigvecs = jnp.linalg.eigh(T)
    idx = jnp.argmin(eigvals)
    eigenvalue = float(eigvals[idx])
    krylov_coefs = eigvecs[:, idx]

    # Vectorized eigenvector reconstruction: stack basis and contract
    # basis may have n+1 entries (the last one was added but has no alpha);
    # krylov_coefs has length n, so slice basis to match.
    basis_stacked = jnp.stack(basis[:n], axis=0)  # (n, vec_dim)
    eigenvector = jnp.tensordot(krylov_coefs, basis_stacked, axes=1)
    eigenvector = eigenvector / (jnp.linalg.norm(eigenvector) + 1e-15)

    return eigenvalue, eigenvector


def _svd_and_truncate_site(
    theta: Tensor,
    site: int,
    config: DMRGConfig,
    sweep_right: bool = True,
) -> tuple[Tensor, jax.Array, Tensor, float]:
    """SVD of 2-site tensor and truncation.

    Computes SVD once via truncated_svd, then derives the truncation error
    from the full singular values returned by that same decomposition.

    Args:
        theta:       2-site wavefunction tensor.
        site:        Left site index.
        config:      DMRGConfig.
        sweep_right: If True, left site gets orthogonality center (A-form);
                     if False, right site gets it (B-form).

    Returns:
        (A_tensor, singular_values, B_tensor, truncation_error)
    """
    labels = theta.labels()

    # Find physical and virtual labels
    left_virt = f"v{site - 1}_{site}" if site > 0 else None
    right_virt = f"v{site + 1}_{site + 2}"
    left_phys = f"p{site}"
    right_phys = f"p{site + 1}"

    # Build actual left/right label splits based on what's available
    left_labels = [
        lbl for lbl in labels if lbl in (left_virt, left_phys) and lbl is not None
    ]
    right_labels = [
        lbl for lbl in labels if lbl in (right_virt, right_phys) and lbl is not None
    ]

    if not left_labels or not right_labels:
        # Fallback: split roughly in half
        n = len(labels)
        left_labels = list(labels[: n // 2])
        right_labels = list(labels[n // 2 :])

    bond_label = f"v{site}_{site + 1}"

    # Single SVD via truncated_svd (handles both Dense and Symmetric)
    A, s, B, s_full = truncated_svd(
        theta,
        left_labels=left_labels,
        right_labels=right_labels,
        new_bond_label=bond_label,
        max_singular_values=config.max_bond_dim,
        max_truncation_err=config.svd_trunc_err,
    )

    # Compute truncation error from the full singular-value spectrum
    # returned by truncated_svd (no second SVD needed).
    n_keep = len(s)
    if len(s_full) > n_keep:
        total_sq = jnp.sum(s_full**2)
        trunc_sq = jnp.sum(s_full[n_keep:] ** 2)
        trunc_err = float(jnp.sqrt(trunc_sq / (total_sq + 1e-15)))
    else:
        trunc_err = 0.0

    # Absorb singular values into the tensor moving away from the
    # orthogonality center so the MPS stays in canonical form.
    if sweep_right:
        # Left-to-right: A = U (left-canonical), absorb s into B
        B = scale_bond_axis(B, bond_label, s)
    else:
        # Right-to-left: B = Vh (right-canonical), absorb s into A
        A = scale_bond_axis(A, bond_label, s)

    return A, s, B, trunc_err


# ------------------------------------------------------------------ #
# MPO builders                                                        #
# ------------------------------------------------------------------ #


def build_mpo_heisenberg(
    L: int,
    Jz: float = 1.0,
    Jxy: float = 1.0,
    hz: float = 0.0,
    dtype: Any = jnp.float64,
) -> TensorNetwork:
    """Build the MPO for the spin-1/2 XXZ Heisenberg chain.

    H = Jz * sum_i Sz_i Sz_{i+1} + Jxy/2 * sum_i (S+_i S-_{i+1} + S-_i S+_{i+1})
        + hz * sum_i Sz_i

    The MPO is constructed using the standard 5x5 MPO representation
    with bond dimension 5 (I, S+, S-, Sz, I boundaries).

    Args:
        L:      Chain length (number of sites).
        Jz:     Ising coupling strength.
        Jxy:    XY coupling strength.
        hz:     Longitudinal magnetic field.
        dtype:  JAX dtype for MPO tensors.

    Returns:
        TensorNetwork representing the MPO with L site tensors connected
        by virtual bonds. Each site tensor has legs:
        ("w{i-1}_{i}", "mpo_top_{i}", "mpo_bot_{i}", "w{i}_{i+1}")
        Boundary sites have only 3 legs (one virtual bond removed).
    """
    # Spin-1/2 operators (physical dimension d=2)
    d = 2
    Sp = jnp.array([[0, 1], [0, 0]], dtype=dtype)  # S+ = |up><down|
    Sm = jnp.array([[0, 0], [1, 0]], dtype=dtype)  # S- = |down><up|
    Sz = 0.5 * jnp.array([[1, 0], [0, -1]], dtype=dtype)
    I2 = jnp.eye(d, dtype=dtype)

    # MPO bond dimension = 5
    # W = [[I,  0,  0,  0,  0 ],
    #      [S+, 0,  0,  0,  0 ],
    #      [S-, 0,  0,  0,  0 ],
    #      [Sz, 0,  0,  0,  0 ],
    #      [h*Sz, Jxy/2*Sm, Jxy/2*Sp, Jz*Sz, I]]
    # Shape: (D_w, d, d, D_w) where D_w = 5
    D_w = 5

    def make_bulk_W() -> jax.Array:
        W = jnp.zeros((D_w, d, d, D_w), dtype=dtype)
        # Row 0: I (pass-through left boundary)
        W = W.at[0, :, :, 0].set(I2)
        # Row 1: S+ (start Jxy/2 * S+ S- term)
        W = W.at[1, :, :, 0].set(Sp)
        # Row 2: S- (start Jxy/2 * S- S+ term)
        W = W.at[2, :, :, 0].set(Sm)
        # Row 3: Sz (start Jz * Sz Sz term)
        W = W.at[3, :, :, 0].set(Sz)
        # Row 4: complete terms + on-site field + pass-through right boundary
        W = W.at[4, :, :, 0].set(hz * Sz)
        W = W.at[4, :, :, 1].set((Jxy / 2) * Sm)
        W = W.at[4, :, :, 2].set((Jxy / 2) * Sp)
        W = W.at[4, :, :, 3].set(Jz * Sz)
        W = W.at[4, :, :, 4].set(I2)
        return W

    # Left boundary: shape (1, d, d, D_w) — last row of W only
    W_left = make_bulk_W()[D_w - 1 : D_w, :, :, :]  # shape (1, d, d, D_w)

    # Right boundary: shape (D_w, d, d, 1) — first column of W only
    W_right = make_bulk_W()[:, :, :, 0:1]  # shape (D_w, d, d, 1)

    W_bulk = make_bulk_W()

    sym = U1Symmetry()
    bond_1 = np.zeros(1, dtype=np.int32)
    bond_dw = np.zeros(D_w, dtype=np.int32)
    bond_d = np.zeros(d, dtype=np.int32)

    mpo = TensorNetwork(name=f"Heisenberg_MPO_L{L}")

    for i in range(L):
        if L == 1:
            W = make_bulk_W()[D_w - 1 : D_w, :, :, 0:1]
            left_bond = bond_1
            right_bond = bond_1
        elif i == 0:
            W = W_left
            left_bond = bond_1
            right_bond = bond_dw
        elif i == L - 1:
            W = W_right
            left_bond = bond_dw
            right_bond = bond_1
        else:
            W = W_bulk
            left_bond = bond_dw
            right_bond = bond_dw

        if i == 0:
            left_label = "w_left_0"
        else:
            left_label = f"w{i - 1}_{i}"

        if i == L - 1:
            right_label = "w_right"
        else:
            right_label = f"w{i}_{i + 1}"

        indices = (
            TensorIndex(sym, left_bond, FlowDirection.IN, label=left_label),
            TensorIndex(sym, bond_d, FlowDirection.IN, label=f"mpo_top_{i}"),
            TensorIndex(sym, bond_d, FlowDirection.OUT, label=f"mpo_bot_{i}"),
            TensorIndex(sym, right_bond, FlowDirection.OUT, label=right_label),
        )
        mpo.add_node(i, DenseTensor(W, indices))

    # Connect virtual MPO bonds
    for i in range(L - 1):
        shared = set(mpo.get_tensor(i).labels()) & set(mpo.get_tensor(i + 1).labels())
        for label in sorted(shared, key=str):
            try:
                mpo.connect(i, label, i + 1, label)
            except (ValueError, KeyError):
                pass

    return mpo


def build_random_symmetric_mps(
    L: int,
    bond_dim: int = 4,
    dtype: Any = jnp.float64,
    seed: int = 42,
) -> TensorNetwork:
    """Build a random block-sparse MPS with U(1) charge conservation.

    Physical dimension is 2 (spin-1/2). Charges represent accumulated Sz:
    spin up = +1, spin down = -1. Virtual bonds carry sectors that allow
    the total-Sz = 0 subspace.

    The resulting tensors are SymmetricTensor objects with non-trivial block
    structure. DMRG treats them identically to DenseTensor via .todense().

    Args:
        L:         Chain length.
        bond_dim:  Virtual bond dimension (must be >= 2; blocks distributed
                   across Sz = -1, 0, +1 sectors).
        dtype:     JAX dtype.
        seed:      Random seed for block initialisation.

    Returns:
        TensorNetwork representing the symmetric random MPS.
    """
    from tenax.core.tensor import SymmetricTensor

    sym = U1Symmetry()

    # Physical: spin up = +1, spin down = −1
    phys_charges = np.array([1, -1], dtype=np.int32)

    # Virtual bond: distribute bond_dim states across Sz = {-1, 0, +1}.
    # Ensures at least one state per sector so blocks are non-trivial.
    q_each = max(1, bond_dim // 4)
    q_zero = max(1, bond_dim - 2 * q_each)
    virt_charges = np.concatenate(
        [
            np.full(q_each, -1, dtype=np.int32),
            np.full(q_zero, 0, dtype=np.int32),
            np.full(q_each, 1, dtype=np.int32),
        ]
    )[:bond_dim]  # Trim to exact bond_dim

    mps = TensorNetwork(name=f"symmetric_MPS_L{L}")

    for i in range(L):
        key = jax.random.PRNGKey(seed + i)

        if i == 0:
            # Left boundary: (phys_IN, virt_right_OUT)
            indices: tuple[TensorIndex, ...] = (
                TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    sym, virt_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == L - 1:
            # Right boundary: (virt_left_IN, phys_IN)
            indices = (
                TensorIndex(sym, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"p{i}"),
            )
        else:
            # Middle: (virt_left_IN, phys_IN, virt_right_OUT)
            indices = (
                TensorIndex(sym, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    sym, virt_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )

        tensor = SymmetricTensor.random_normal(indices, key=key, dtype=dtype)
        mps.add_node(i, tensor)

    # Connect virtual bonds
    for i in range(L - 1):
        bond_label = f"v{i}_{i + 1}"
        mps.connect(i, bond_label, i + 1, bond_label)

    return mps


def build_random_mps(
    L: int,
    physical_dim: int = 2,
    bond_dim: int = 4,
    dtype: Any = jnp.float64,
    seed: int = 0,
) -> TensorNetwork:
    """Build a random MPS for use as initial state in DMRG.

    Args:
        L:            Chain length.
        physical_dim: Physical dimension per site.
        bond_dim:     Virtual bond dimension.
        dtype:        Data type.
        seed:         Random seed.

    Returns:
        TensorNetwork representing the random MPS.
    """
    sym = U1Symmetry()
    bond_d = np.zeros(physical_dim, dtype=np.int32)
    bond_chi = np.zeros(bond_dim, dtype=np.int32)

    mps = TensorNetwork(name=f"random_MPS_L{L}")

    shape: tuple[int, ...]
    indices: tuple[TensorIndex, ...]
    for i in range(L):
        key = jax.random.PRNGKey(seed + i)

        if i == 0:
            shape = (physical_dim, bond_dim)
            indices = (
                TensorIndex(sym, bond_d, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(sym, bond_chi, FlowDirection.OUT, label=f"v{i}_{i + 1}"),
            )
        elif i == L - 1:
            shape = (bond_dim, physical_dim)
            indices = (
                TensorIndex(sym, bond_chi, FlowDirection.IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, bond_d, FlowDirection.IN, label=f"p{i}"),
            )
        else:
            shape = (bond_dim, physical_dim, bond_dim)
            indices = (
                TensorIndex(sym, bond_chi, FlowDirection.IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, bond_d, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(sym, bond_chi, FlowDirection.OUT, label=f"v{i}_{i + 1}"),
            )

        data = jax.random.normal(key, shape, dtype=dtype)
        # Normalize
        data = data / jnp.linalg.norm(data)
        mps.add_node(i, DenseTensor(data, indices))

    # Connect virtual bonds
    for i in range(L - 1):
        bond_label = f"v{i}_{i + 1}"
        mps.connect(i, bond_label, i + 1, bond_label)

    return mps
