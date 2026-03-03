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
import opt_einsum

from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.algorithms.auto_mpo import build_auto_mpo
from tenax.contraction.contractor import contract, qr_decompose, truncated_svd
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor, inner
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
        target_charge:      Target total charge (e.g. 2*Sz for U(1)). If set,
                            validates MPS sector before and after each sweep.
                            Use with ``build_random_symmetric_mps(target_charge=...)``.
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
    target_charge: int | None = None
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


class SweepOps(NamedTuple):
    """Callback bundle holding all backend-specific operations for a DMRG sweep.

    Dense and symmetric backends each provide their own implementations.
    The sweep loop is backend-agnostic — it only calls through ``ops.*``.
    """

    build_trivial_left_env: Callable[..., Tensor]
    build_trivial_right_env: Callable[..., Tensor]
    update_left_env: Callable[[Tensor, Tensor, Tensor], Tensor]
    update_right_env: Callable[[Tensor, Tensor, Tensor], Tensor]
    two_site_update: Callable[..., tuple[Tensor, float]]
    one_site_update: Callable[..., tuple[Tensor, float]]


def _dense_ops() -> SweepOps:
    """Return the dense (existing) backend callbacks."""
    return SweepOps(
        build_trivial_left_env=_build_trivial_left_env,
        build_trivial_right_env=_build_trivial_right_env,
        update_left_env=_update_left_env,
        update_right_env=_update_right_env,
        two_site_update=_two_site_update,
        one_site_update=_one_site_update,
    )


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
    mps_tensors: list[Tensor] = [initial_mps.get_tensor(i) for i in range(L)]
    mpo_tensors = [hamiltonian.get_tensor(i) for i in range(L)]

    # Select backend: symmetric when both MPS and MPO are all SymmetricTensor
    all_mps_sym = all(isinstance(t, SymmetricTensor) for t in mps_tensors)
    all_mpo_sym = all(isinstance(t, SymmetricTensor) for t in mpo_tensors)
    use_symmetric = all_mps_sym and all_mpo_sym

    if use_symmetric:
        ops = _symmetric_ops()
    else:
        ops = _dense_ops()
        # Convert any SymmetricTensor to DenseTensor for the dense path
        mps_tensors = [
            DenseTensor(t.todense(), t.indices) if not isinstance(t, DenseTensor) else t
            for t in mps_tensors
        ]
        mpo_tensors = [
            DenseTensor(t.todense(), t.indices) if not isinstance(t, DenseTensor) else t
            for t in mpo_tensors
        ]

    # Validate initial MPS sector if target_charge is specified
    if config.target_charge is not None and use_symmetric:
        validate_mps_sector(mps_tensors, config.target_charge)

    # Build left environments (L[i] = trivial for i=0)
    left_envs = _build_left_environments_list(mps_tensors, mpo_tensors, L, ops)
    right_envs = _build_right_environments_list(mps_tensors, mpo_tensors, L, ops)

    energies_per_sweep: list[float] = []
    truncation_errors: list[float] = []
    energy = 0.0
    converged = False

    for sweep in range(config.num_sweeps):
        prev_energy = energy

        # Rebuild left environments from updated MPS before left-to-right sweep
        if sweep > 0:
            left_envs = _build_left_environments_list(mps_tensors, mpo_tensors, L, ops)

        # Left-to-right sweep
        for i in range(L - 1):
            l_env = left_envs[i]
            assert l_env is not None
            if config.two_site:
                _r = right_envs[i + 2]
                r_env = _r if _r is not None else ops.build_trivial_right_env()
                theta, e = ops.two_site_update(
                    mps_tensors[i],
                    mps_tensors[i + 1],
                    l_env,
                    mpo_tensors[i],
                    mpo_tensors[i + 1],
                    r_env,
                    config,
                )
                energy = float(e)

                A, s, B, trunc_err = _svd_and_truncate_site(theta, i, config)
                mps_tensors[i] = A
                mps_tensors[i + 1] = B
                truncation_errors.append(float(trunc_err))

                left_envs[i + 1] = ops.update_left_env(l_env, A, mpo_tensors[i])
            else:
                _ri = right_envs[i + 1]
                r_env_1s = _ri if _ri is not None else ops.build_trivial_right_env()
                new_site, e = ops.one_site_update(
                    mps_tensors[i],
                    l_env,
                    mpo_tensors[i],
                    r_env_1s,
                    config,
                )
                energy = float(e)
                mps_tensors[i] = new_site
                left_envs[i + 1] = ops.update_left_env(l_env, new_site, mpo_tensors[i])

        # Rebuild right environments from updated MPS before right-to-left sweep
        right_envs = _build_right_environments_list(mps_tensors, mpo_tensors, L, ops)

        # Right-to-left sweep
        for i in range(L - 2, -1, -1):
            l_env = left_envs[i]
            assert l_env is not None
            _r2 = right_envs[i + 2]
            r2_env = _r2 if _r2 is not None else ops.build_trivial_right_env()
            if config.two_site:
                theta, e = ops.two_site_update(
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

                right_envs[i + 1] = ops.update_right_env(r2_env, B, mpo_tensors[i + 1])
            else:
                _r1 = right_envs[i + 1]
                r1_env = _r1 if _r1 is not None else ops.build_trivial_right_env()
                new_site, e = ops.one_site_update(
                    mps_tensors[i],
                    l_env,
                    mpo_tensors[i],
                    r1_env,
                    config,
                )
                energy = float(e)
                mps_tensors[i] = new_site
                right_envs[i] = ops.update_right_env(r1_env, new_site, mpo_tensors[i])

        energies_per_sweep.append(energy)
        if config.verbose:
            print(f"Sweep {sweep + 1}/{config.num_sweeps}: E = {energy:.10f}")

        # Validate sector preservation after each sweep
        if config.target_charge is not None and use_symmetric:
            sector = compute_mps_sector(mps_tensors)
            if sector != config.target_charge:
                raise RuntimeError(
                    f"Sector drift detected after sweep {sweep + 1}: "
                    f"MPS sector={sector}, expected target_charge={config.target_charge}."
                )

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


def _build_left_environments_list(
    mps_tensors: list[Tensor],
    mpo_tensors: list[Tensor],
    L: int,
    ops: SweepOps | None = None,
) -> list[Tensor | None]:
    """Build all left environment tensors by sweeping left to right.

    L_env[0] = trivial, L_env[i] = contraction of sites 0..i-1.

    Returns list of L+1 environment tensors (None used as placeholder where
    not yet computed; replaced with dense contractions in full implementation).
    """
    if ops is None:
        ops = _dense_ops()
    envs: list[Tensor | None] = [None] * (L + 1)
    envs[0] = ops.build_trivial_left_env()

    for i in range(L - 1):
        env = envs[i]
        if env is not None:
            envs[i + 1] = ops.update_left_env(env, mps_tensors[i], mpo_tensors[i])

    return envs


def _build_right_environments_list(
    mps_tensors: list[Tensor],
    mpo_tensors: list[Tensor],
    L: int,
    ops: SweepOps | None = None,
) -> list[Tensor | None]:
    """Build all right environment tensors by sweeping right to left."""
    if ops is None:
        ops = _dense_ops()
    envs: list[Tensor | None] = [None] * (L + 1)
    envs[L] = ops.build_trivial_right_env()

    for i in range(L - 1, 0, -1):
        env = envs[i + 1]
        if env is not None:
            envs[i] = ops.update_right_env(env, mps_tensors[i], mpo_tensors[i])

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
# Symmetric (block-sparse) backend                                     #
# ------------------------------------------------------------------ #


def _build_trivial_left_env_symmetric(dtype=None) -> SymmetricTensor:
    """Build trivial (1x1x1) left boundary environment as SymmetricTensor."""
    if dtype is None:
        dtype = jnp.float64
    sym = U1Symmetry()
    bond = np.zeros(1, dtype=np.int32)
    indices = (
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mps_l"),
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mpo_l"),
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mps_conj_l"),
    )
    blocks: dict[tuple[int, ...], jax.Array] = {
        (0, 0, 0): jnp.ones((1, 1, 1), dtype=dtype)
    }
    return SymmetricTensor(blocks, indices)


def _build_trivial_right_env_symmetric(dtype=None) -> SymmetricTensor:
    """Build trivial (1x1x1) right boundary environment as SymmetricTensor."""
    if dtype is None:
        dtype = jnp.float64
    sym = U1Symmetry()
    bond = np.zeros(1, dtype=np.int32)
    indices = (
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mps_r"),
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mpo_r"),
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mps_conj_r"),
    )
    blocks: dict[tuple[int, ...], jax.Array] = {
        (0, 0, 0): jnp.ones((1, 1, 1), dtype=dtype)
    }
    return SymmetricTensor(blocks, indices)


def _pad_boundary_symmetric(t: SymmetricTensor, pad_left: bool) -> SymmetricTensor:
    """Pad a 2D boundary SymmetricTensor to 3D by adding a trivial dimension.

    Args:
        t:        2D SymmetricTensor (boundary MPS site).
        pad_left: If True, prepend trivial dim (left boundary: (p,v) -> (1,p,v)).
                  If False, append trivial dim (right boundary: (v,p) -> (v,p,1)).

    Returns:
        3D SymmetricTensor with a trivial charge-0 dimension added.
    """
    sym = U1Symmetry()
    trivial_bond = np.zeros(1, dtype=np.int32)

    if pad_left:
        trivial_idx = TensorIndex(sym, trivial_bond, FlowDirection.IN, label="_pad_l")
        new_indices = (trivial_idx,) + t.indices
        new_blocks = {(0,) + key: arr[np.newaxis, :] for key, arr in t._blocks.items()}
    else:
        trivial_idx = TensorIndex(sym, trivial_bond, FlowDirection.OUT, label="_pad_r")
        new_indices = t.indices + (trivial_idx,)
        new_blocks = {
            key + (0,): arr[:, :, np.newaxis] for key, arr in t._blocks.items()
        }

    obj = object.__new__(SymmetricTensor)
    obj._indices = new_indices
    obj._blocks = new_blocks
    return obj


def _unpad_boundary_symmetric(t: SymmetricTensor, pad_left: bool) -> SymmetricTensor:
    """Remove a trivial dimension added by _pad_boundary_symmetric.

    Args:
        t:        3D SymmetricTensor with a trivial padding dimension.
        pad_left: If True, remove first dim. If False, remove last dim.

    Returns:
        2D SymmetricTensor with the trivial dimension removed.
    """
    if pad_left:
        new_indices = t.indices[1:]
        new_blocks = {key[1:]: arr[0] for key, arr in t._blocks.items()}
    else:
        new_indices = t.indices[:-1]
        new_blocks = {key[:-1]: arr[:, :, 0] for key, arr in t._blocks.items()}

    obj = object.__new__(SymmetricTensor)
    obj._indices = new_indices
    obj._blocks = new_blocks
    return obj


def _lanczos_solve_tensor(
    matvec: Callable[[Tensor], Tensor],
    initial: Tensor,
    num_steps: int,
    tol: float,
) -> tuple[float, Tensor]:
    """Lanczos eigensolver operating on Tensor objects (dense or symmetric).

    Uses inner(), norm(), and Tensor arithmetic instead of flat JAX arrays.

    Args:
        matvec:   Function applying H_eff to a Tensor, returning a Tensor.
        initial:  Starting vector (Tensor).
        num_steps: Maximum Lanczos iterations.
        tol:      Convergence tolerance on the residual norm.

    Returns:
        (eigenvalue, eigenvector) for the ground state as Tensor.
    """
    v_norm = initial.norm()
    v = initial * (1.0 / (float(v_norm) + 1e-15))

    basis: list[Tensor] = [v]
    alphas: list[float] = []
    betas: list[float] = [0.0]

    for step in range(num_steps):
        w = matvec(basis[-1])
        alpha_val = float(inner(basis[-1], w).real)
        alphas.append(alpha_val)

        w = w - basis[-1] * alpha_val
        if step > 0:
            w = w - basis[-2] * betas[-1]

        beta_val = float(w.norm())
        betas.append(beta_val)

        if beta_val < tol:
            break

        basis.append(w * (1.0 / beta_val))

    n = len(alphas)
    if n == 0:
        return 0.0, v
    if n == 1:
        return alphas[0], basis[0]

    # Build tridiagonal matrix and diagonalize
    alphas_arr = jnp.array(alphas)
    betas_arr = jnp.array(betas[1:n])
    T = jnp.diag(alphas_arr) + jnp.diag(betas_arr, k=1) + jnp.diag(betas_arr, k=-1)

    eigvals, eigvecs = jnp.linalg.eigh(T)
    idx = int(jnp.argmin(eigvals))
    eigenvalue = float(eigvals[idx])
    krylov_coefs = eigvecs[:, idx]

    # Reconstruct eigenvector: sum(c_k * basis[k]) — can't stack SymmetricTensors
    eigenvector = basis[0] * float(krylov_coefs[0])
    for k in range(1, n):
        eigenvector = eigenvector + basis[k] * float(krylov_coefs[k])

    ev_norm = float(eigenvector.norm())
    eigenvector = eigenvector * (1.0 / (ev_norm + 1e-15))

    return eigenvalue, eigenvector


def _blockwise_contract(
    tensors: list[SymmetricTensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    expr_cache: dict[tuple[tuple[int, ...], ...], Any] | None = None,
) -> SymmetricTensor:
    """Contract multiple SymmetricTensors using block-level charge matching.

    Unlike ``_contract_symmetric`` in contractor.py, this handles multi-tensor
    contractions correctly by iterating over all compatible block combinations
    (with early pruning via charge matching) and does NOT filter output blocks
    by a conservation law — it trusts the contraction result.

    This is necessary for DMRG environment updates and matvec where contracted
    indices may have same-direction flows (ket-ket or bra-bra physical indices),
    which violates the opposite-flow assumption in ``_contract_symmetric``.

    Args:
        tensors:        List of SymmetricTensor inputs.
        subscripts:     Einsum subscript string (e.g., "abc,apd,bpxe,cxf->def").
        output_indices: TensorIndex metadata for the output legs.
        expr_cache:     Optional shared cache for opt_einsum contraction
                        expressions. Pass the same dict across calls (e.g.,
                        Lanczos iterations) to avoid recomputing paths.

    Returns:
        SymmetricTensor with contracted result (bypasses conservation validation).

    Note:
        Callers must validate inputs via ``_assert_symmetric`` before calling.
    """
    input_part, output_part = subscripts.split("->")
    input_subs = input_part.split(",")

    # Accumulate contributions per output key, then sum once at the end
    # to avoid repeated intermediate JAX array allocations.
    output_accum: dict[tuple[int, ...], list[jax.Array]] = {}

    # Cache for opt_einsum contraction expressions
    if expr_cache is None:
        expr_cache = {}

    # Backtracking state (mutated in-place to avoid per-branch allocations)
    combo_arrays: list[jax.Array] = []
    char_charges: dict[str, int] = {}

    def _recurse(tensor_idx: int) -> None:
        if tensor_idx == len(tensors):
            # All tensors matched — compute contraction
            output_key = tuple(char_charges.get(c, 0) for c in output_part)

            block_shapes = tuple(a.shape for a in combo_arrays)
            if block_shapes in expr_cache:
                expr = expr_cache[block_shapes]
            else:
                expr = opt_einsum.contract_expression(
                    subscripts, *block_shapes, optimize="auto"
                )
                expr_cache[block_shapes] = expr
            result_array = expr(*combo_arrays, backend="jax")

            output_accum.setdefault(output_key, []).append(result_array)
            return

        subs = input_subs[tensor_idx]
        for key, arr in tensors[tensor_idx].blocks.items():
            # Check charge compatibility and track new assignments
            added_chars: list[str] = []
            compatible = True
            for char, q in zip(subs, key):
                qi = int(q)
                if char in char_charges:
                    if char_charges[char] != qi:
                        compatible = False
                        break
                else:
                    char_charges[char] = qi
                    added_chars.append(char)

            if compatible:
                combo_arrays.append(arr)
                _recurse(tensor_idx + 1)
                combo_arrays.pop()

            # Restore char_charges (backtrack)
            for char in added_chars:
                del char_charges[char]

    _recurse(0)

    # Sum accumulated contributions per output key
    output_blocks: dict[tuple[int, ...], jax.Array] = {}
    for key, arrays in output_accum.items():
        total = arrays[0]
        for a in arrays[1:]:
            total = total + a
        output_blocks[key] = total

    # Build result bypassing SymmetricTensor validation (flows may not
    # satisfy the standard conservation law for environment tensors).
    obj = object.__new__(SymmetricTensor)
    obj._indices = output_indices
    obj._blocks = output_blocks
    return obj


def _assert_symmetric(*tensors: Tensor, context: str) -> None:
    """Assert all tensors are SymmetricTensor; raise TypeError otherwise."""
    for i, t in enumerate(tensors):
        if not isinstance(t, SymmetricTensor):
            raise TypeError(
                f"{context}: expected SymmetricTensor for input {i}, "
                f"got {type(t).__name__}. "
                f"The symmetric DMRG path must never fall back to dense."
            )


def _update_left_env_symmetric(
    left_env: Tensor,
    mps_site: Tensor,
    mpo_site: Tensor,
) -> SymmetricTensor:
    """Update left environment using block-sparse contraction.

    Contracts: new_L[d,e,f] = L[a,b,c] * A[a,p,d] * W[b,p,x,e] * A*[c,x,f]

    All inputs must be SymmetricTensor. The symmetric path must never
    fall back to dense operations.
    """
    _assert_symmetric(
        left_env, mps_site, mpo_site, context="_update_left_env_symmetric"
    )
    A = mps_site
    # Pad boundary sites to 3D
    if A.ndim == 2:
        labels = A.labels()
        is_left_boundary = isinstance(labels[0], str) and labels[0].startswith("p")
        A = _pad_boundary_symmetric(A, pad_left=is_left_boundary)

    A_conj = A.conj()

    # Build output indices from the free legs of the contraction:
    # d = A's right virtual, e = W's right bond, f = A_conj's right virtual
    # Use generic env labels so subsequent contractions find consistent metadata.
    out_indices = (A.indices[2], mpo_site.indices[3], A_conj.indices[2])
    result = _blockwise_contract(
        [left_env, A, mpo_site, A_conj],
        "abc,apd,bpxe,cxf->def",
        output_indices=out_indices,
    )
    return result


def _update_right_env_symmetric(
    right_env: Tensor,
    mps_site: Tensor,
    mpo_site: Tensor,
) -> SymmetricTensor:
    """Update right environment using block-sparse contraction.

    Contracts: new_R[d,e,f] = R[a,b,c] * B[d,p,a] * W[e,p,x,b] * B*[f,x,c]

    All inputs must be SymmetricTensor. The symmetric path must never
    fall back to dense operations.
    """
    _assert_symmetric(
        right_env, mps_site, mpo_site, context="_update_right_env_symmetric"
    )
    B = mps_site
    # Pad boundary sites to 3D
    if B.ndim == 2:
        labels = B.labels()
        is_left_boundary = isinstance(labels[0], str) and labels[0].startswith("p")
        B = _pad_boundary_symmetric(B, pad_left=is_left_boundary)

    B_conj = B.conj()

    # Output: d = B's left virtual, e = W's left bond, f = B_conj's left virtual
    out_indices = (B.indices[0], mpo_site.indices[0], B_conj.indices[0])
    result = _blockwise_contract(
        [right_env, B, mpo_site, B_conj],
        "abc,dpa,epxb,fxc->def",
        output_indices=out_indices,
    )
    return result


def _two_site_update_symmetric(
    site_l: Tensor,
    site_r: Tensor,
    left_env: Tensor,
    mpo_l: Tensor,
    mpo_r: Tensor,
    right_env: Tensor,
    config: DMRGConfig,
) -> tuple[Tensor, float]:
    """Perform 2-site DMRG update using block-sparse tensors.

    All tensor inputs must be SymmetricTensor. The symmetric path must never
    fall back to dense operations.
    """
    _assert_symmetric(
        site_l,
        site_r,
        left_env,
        mpo_l,
        mpo_r,
        right_env,
        context="_two_site_update_symmetric",
    )
    # Contract theta = A[i] * A[i+1]
    shared = set(site_l.labels()) & set(site_r.labels())
    if shared:
        theta = contract(site_l, site_r)
    else:
        theta = site_l

    # Pad boundary sites to 4D
    original_ndim = theta.ndim
    pad_left = False
    if theta.ndim == 3:
        labels_list = [idx.label for idx in theta.indices]
        has_left_virt = any(
            isinstance(lbl, str) and lbl.startswith("v") for lbl in labels_list[:1]
        )
        if not has_left_virt:
            pad_left = True
            theta = _pad_boundary_symmetric(theta, pad_left=True)
        else:
            theta = _pad_boundary_symmetric(theta, pad_left=False)

    # Shared cache for opt_einsum expressions across Lanczos iterations
    _matvec_cache: dict[tuple[tuple[int, ...], ...], Any] = {}

    def matvec(v: Tensor) -> Tensor:
        result = _blockwise_contract(
            [left_env, v, mpo_l, mpo_r, right_env],
            "abc,apqd,bpse,eqtf,dfg->cstg",
            output_indices=v.indices,
            expr_cache=_matvec_cache,
        )
        return result

    energy, theta_opt = _lanczos_solve_tensor(
        matvec, theta, config.lanczos_max_iter, config.lanczos_tol
    )

    # Unpad if we padded
    if original_ndim == 3:
        theta_opt = _unpad_boundary_symmetric(theta_opt, pad_left=pad_left)

    return theta_opt, energy


def _one_site_update_symmetric(
    site: Tensor,
    left_env: Tensor,
    mpo_site: Tensor,
    right_env: Tensor,
    config: DMRGConfig,
) -> tuple[Tensor, float]:
    """Perform 1-site DMRG update using block-sparse tensors.

    All tensor inputs must be SymmetricTensor. The symmetric path must never
    fall back to dense operations.
    """
    _assert_symmetric(
        site, left_env, mpo_site, right_env, context="_one_site_update_symmetric"
    )
    original_ndim = site.ndim
    pad_left = False

    if site.ndim == 2:
        labels_list = list(site.labels())
        has_left_virt = any(
            isinstance(lbl, str) and lbl.startswith("v") for lbl in labels_list[:1]
        )
        if not has_left_virt:
            pad_left = True
            site_3d = _pad_boundary_symmetric(site, pad_left=True)
        else:
            site_3d = _pad_boundary_symmetric(site, pad_left=False)
    else:
        site_3d = site

    # Shared cache for opt_einsum expressions across Lanczos iterations
    _matvec_cache: dict[tuple[tuple[int, ...], ...], Any] = {}

    def matvec(v: Tensor) -> Tensor:
        result = _blockwise_contract(
            [left_env, v, mpo_site, right_env],
            "abc,apd,bpxe,def->cxf",
            output_indices=v.indices,
            expr_cache=_matvec_cache,
        )
        return result

    energy, site_opt = _lanczos_solve_tensor(
        matvec, site_3d, config.lanczos_max_iter, config.lanczos_tol
    )

    # Unpad if we padded
    if original_ndim == 2:
        site_opt = _unpad_boundary_symmetric(site_opt, pad_left=pad_left)

    return site_opt, energy


def _symmetric_ops() -> SweepOps:
    """Return the block-sparse symmetric backend callbacks."""
    return SweepOps(
        build_trivial_left_env=_build_trivial_left_env_symmetric,
        build_trivial_right_env=_build_trivial_right_env_symmetric,
        update_left_env=_update_left_env_symmetric,
        update_right_env=_update_right_env_symmetric,
        two_site_update=_two_site_update_symmetric,
        one_site_update=_one_site_update_symmetric,
    )


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

    Returns a block-sparse (SymmetricTensor) MPO with U(1) charge conservation.
    Paired with a symmetric MPS (e.g. from ``build_random_symmetric_mps``),
    DMRG will use the fully block-sparse backend automatically.

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
    """
    terms: list[tuple[float, ...]] = []
    for i in range(L - 1):
        terms.append((Jz, "Sz", i, "Sz", i + 1))
        terms.append((Jxy / 2, "Sp", i, "Sm", i + 1))
        terms.append((Jxy / 2, "Sm", i, "Sp", i + 1))
    if hz != 0.0:
        for i in range(L):
            terms.append((hz, "Sz", i))
    if not terms:
        # L=1 with hz=0: add a zero on-site term so AutoMPO has at least one term
        terms.append((0.0, "Sz", 0))
    return build_auto_mpo(terms, L=L, symmetric=True)


def build_random_symmetric_mps(
    L: int,
    bond_dim: int = 4,
    dtype: Any = jnp.float64,
    seed: int = 42,
    target_charge: int = 0,
) -> TensorNetwork:
    """Build a random block-sparse MPS with U(1) charge conservation.

    Physical dimension is 2 (spin-1/2). Charges represent accumulated Sz:
    spin up = +1, spin down = -1. Virtual bonds carry sectors that allow
    the specified total-Sz subspace.

    Args:
        L:              Chain length.
        bond_dim:       Virtual bond dimension (must be >= 2; blocks distributed
                        across charge sectors).
        dtype:          JAX dtype.
        seed:           Random seed for block initialisation.
        target_charge:  Target total charge (2*Sz). Default 0 (Sz=0 sector).
                        Must satisfy parity: target_charge % 2 == L % 2
                        (each site contributes ±1).

    Returns:
        TensorNetwork representing the symmetric random MPS.

    Raises:
        ValueError: If target_charge has incompatible parity with L.
    """
    if target_charge % 2 != L % 2:
        raise ValueError(
            f"target_charge={target_charge} has parity {target_charge % 2} but "
            f"L={L} has parity {L % 2}. Each site contributes ±1, so total "
            f"charge must have the same parity as L."
        )

    sym = U1Symmetry()

    # Physical: spin up = +1, spin down = −1
    phys_charges = np.array([1, -1], dtype=np.int32)

    # Virtual bond: include charge sectors compatible with target propagation.
    # For total charge Q, the right boundary needs virt charges in {Q-1, Q+1}
    # (since phys charges are ±1 and we need virt + phys = Q).
    # Interior bonds need a range that connects left boundary (near 0) to
    # right boundary (near Q). We include charges from -1 to max(1, Q+1)
    # (or min(-1, Q-1) to 1 for negative Q).
    if target_charge == 0:
        required_charges = [-1, 0, 1]
    else:
        # Include range from 0 to target, plus margins for both boundaries
        lo = min(-1, target_charge - 1)
        hi = max(1, target_charge + 1)
        required_charges = list(range(lo, hi + 1))

    # Distribute bond_dim states across the required charge sectors
    n_sectors = len(required_charges)
    per_sector = max(1, bond_dim // n_sectors)
    arrays = [np.full(per_sector, q, dtype=np.int32) for q in required_charges]
    virt_charges = np.concatenate(arrays)[:bond_dim]
    # If bond_dim is larger, pad with the middle charge
    if len(virt_charges) < bond_dim:
        mid_q = required_charges[n_sectors // 2]
        pad = np.full(bond_dim - len(virt_charges), mid_q, dtype=np.int32)
        virt_charges = np.concatenate([virt_charges, pad])

    mps = TensorNetwork(name=f"symmetric_MPS_L{L}")

    for i in range(L):
        key = jax.random.PRNGKey(seed + i)

        # Right boundary tensor uses target_charge; all others use identity (0)
        site_target = target_charge if i == L - 1 else None

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

        tensor = SymmetricTensor.random_normal(
            indices, key=key, dtype=dtype, target=site_target
        )
        mps.add_node(i, tensor)

    # Connect virtual bonds
    for i in range(L - 1):
        bond_label = f"v{i}_{i + 1}"
        mps.connect(i, bond_label, i + 1, bond_label)

    return mps


def compute_mps_sector(mps_tensors: list[Tensor]) -> int | None:
    """Infer total charge sector of an MPS from its right boundary tensor.

    For an OBC MPS with U(1) symmetry, the conservation law on each tensor is
    ``sum(flow_i * charge_i) = 0`` (or ``= target`` for the boundary tensor).
    The total sector Q is determined by the right boundary tensor: for each
    block, ``sum(flow_i * charge_i)`` gives Q.

    Args:
        mps_tensors: List of SymmetricTensor MPS site tensors.

    Returns:
        The total charge if all blocks agree, or None if the MPS is in a
        mixed sector (or contains no SymmetricTensor).
    """
    right_site = mps_tensors[-1]
    if not isinstance(right_site, SymmetricTensor):
        return None
    if not right_site.blocks:
        return None

    sectors: set[int] = set()
    for key in right_site.blocks:
        total = 0
        for idx, q in zip(right_site.indices, key):
            total += int(idx.flow) * q
        sectors.add(total)

    if len(sectors) == 1:
        return sectors.pop()
    return None


def validate_mps_sector(mps_tensors: list[Tensor], target_charge: int) -> None:
    """Assert that an MPS is in the specified charge sector.

    Args:
        mps_tensors:   List of MPS site tensors (SymmetricTensor).
        target_charge: Expected total charge (e.g. 2*Sz for spin-1/2 U(1)).

    Raises:
        ValueError: If the MPS is not in the target sector.
    """
    sector = compute_mps_sector(mps_tensors)
    if sector is None:
        raise ValueError(
            f"Cannot determine MPS sector (mixed or no SymmetricTensor blocks). "
            f"Expected target_charge={target_charge}."
        )
    if sector != target_charge:
        raise ValueError(
            f"MPS sector {sector} does not match target_charge={target_charge}."
        )


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
