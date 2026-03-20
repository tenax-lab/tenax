"""Time-Dependent Variational Principle (TDVP) for MPS time evolution.

Implements 1-site and 2-site TDVP using a second-order Lie-Trotter
decomposition with Krylov subspace exponentiation.

Label conventions follow DMRG (see dmrg.py docstring).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._krylov import krylov_expm
from tenax.algorithms.dmrg import (
    _build_trivial_left_env,
    _build_trivial_right_env,
    _matvec_jit,
    _update_left_env,
    _update_right_env,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.mps import FiniteMPS
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor
from tenax.network.network import TensorNetwork


@dataclass
class TDVPConfig:
    """Configuration for a TDVP run.

    Attributes:
        mode:          "1site" or "2site".
        dt:            Time step size (real or complex).
        time_type:     "real", "imaginary", or "complex".
                       For "real": evolution is exp(-i dt H).
                       For "imaginary": evolution is exp(-dt H).
                       For "complex": dt is used as a complex time step,
                       evolution is exp(-i dt H) with complex dt.
                       This is useful for spectral functions, thermal
                       states, and Schwinger-Keldysh contour evolution.
        num_steps:     Number of TDVP steps.
        max_bond_dim:  Maximum bond dimension (used in 2-site truncation).
        svd_trunc_err: Maximum truncation error per SVD (optional).
        krylov_dim:    Maximum Krylov subspace dimension.
        krylov_tol:    Convergence tolerance for Krylov.
        verbose:       Print info during evolution.
    """

    mode: str = "1site"
    dt: float | complex = 0.05
    time_type: str = "real"
    num_steps: int = 100
    max_bond_dim: int = 64
    svd_trunc_err: float | None = None
    krylov_dim: int = 20
    krylov_tol: float = 1e-12
    verbose: bool = False


@dataclass
class TDVPResult:
    """Result of a TDVP run.

    Attributes:
        mps:          Final MPS as FiniteMPS.
        times:        Time values at each measurement.
        energies:     Energy at each measurement.
        observables:  Dictionary of observable names to measurement lists.
    """

    mps: FiniteMPS
    times: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    observables: dict[str, list[float]] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# 1-site effective Hamiltonian matvec                                  #
# ------------------------------------------------------------------ #


def _effective_hamiltonian_matvec_1site(
    theta_flat: jax.Array,
    theta_shape: tuple[int, ...],
    L_env: jax.Array,
    W: jax.Array,
    R_env: jax.Array,
) -> jax.Array:
    """Apply 1-site effective Hamiltonian to site tensor.

    Contracts: L[a,b,c] * theta[a,p,d] * W[b,p,s,e] * R[d,e,f] -> result[c,s,f]
    """
    theta = theta_flat.reshape(theta_shape)
    result = jnp.einsum("abc,apd,bpse,def->csf", L_env, theta, W, R_env)
    return result.ravel()


_matvec_1site_jit = jax.jit(_effective_hamiltonian_matvec_1site, static_argnums=(1,))


# ------------------------------------------------------------------ #
# Bond effective Hamiltonian matvec                                    #
# ------------------------------------------------------------------ #


def _bond_hamiltonian_matvec(
    bond_flat: jax.Array,
    bond_shape: tuple[int, ...],
    L_env: jax.Array,
    R_env: jax.Array,
) -> jax.Array:
    """Apply bond effective Hamiltonian for back-evolution.

    Contracts: L[a,b,c] * C[a,d] * R[d,b,e] -> result[c,e]
    """
    C = bond_flat.reshape(bond_shape)
    result = jnp.einsum("abc,ad,dbe->ce", L_env, C, R_env)
    return result.ravel()


_bond_matvec_jit = jax.jit(_bond_hamiltonian_matvec, static_argnums=(1,))


# ------------------------------------------------------------------ #
# MPS tensor helpers                                                   #
# ------------------------------------------------------------------ #


def _site_to_3d(site: Tensor) -> jax.Array:
    """Convert site tensor to 3D dense array (chi_l, d, chi_r).

    All MPS tensors are uniformly 3-leg, so this simply returns the dense array.
    """
    return site.todense()


def _make_site_tensor(
    data_3d: jax.Array,
    site_idx: int,
    L: int,
) -> DenseTensor:
    """Create a 3-leg DenseTensor for an MPS site with proper labels and indices.

    data_3d has shape (chi_l, d, chi_r).  All sites (including boundaries)
    are uniformly 3-leg with trivial dim-1 bonds at the edges.
    """
    sym = U1Symmetry()
    chi_l, d, chi_r = data_3d.shape

    left_label = f"v{site_idx - 1}_{site_idx}" if site_idx > 0 else "v_-1_0"
    right_label = f"v{site_idx}_{site_idx + 1}"

    indices = (
        TensorIndex(
            sym,
            np.zeros(chi_l, dtype=np.int32),
            FlowDirection.IN,
            label=left_label,
        ),
        TensorIndex(
            sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label=f"p{site_idx}"
        ),
        TensorIndex(
            sym,
            np.zeros(chi_r, dtype=np.int32),
            FlowDirection.OUT,
            label=right_label,
        ),
    )
    return DenseTensor(data_3d, indices)


def _right_canonicalize_dense(tensors_3d: list[jax.Array]) -> list[jax.Array]:
    """Right-canonicalize MPS working on raw 3D arrays.

    Each tensor has shape (chi_l, d, chi_r). After right-canonicalization,
    sites 1..L-1 are right-isometric and the orthogonality center is at site 0.
    """
    L = len(tensors_3d)
    result = [t for t in tensors_3d]

    for i in range(L - 1, 0, -1):
        chi_l, d, chi_r = result[i].shape
        # Reshape to (chi_l, d*chi_r) and do RQ decomposition
        mat = result[i].reshape(chi_l, d * chi_r)
        # RQ via QR of transpose: mat^T = Q R^T => mat = R Q^T
        Q_t, R_t = jnp.linalg.qr(mat.T)
        Q = Q_t.T  # (new_chi, d*chi_r)
        R = R_t.T  # (chi_l, new_chi)
        new_chi = Q.shape[0]

        result[i] = Q.reshape(new_chi, d, chi_r)

        # Absorb R into previous site
        prev_chi_l, prev_d, prev_chi_r = result[i - 1].shape
        # result[i-1] has shape (prev_chi_l, prev_d, prev_chi_r=chi_l)
        # R has shape (chi_l, new_chi) = (prev_chi_r, new_chi)
        prev_mat = result[i - 1].reshape(prev_chi_l * prev_d, prev_chi_r)
        new_prev = (prev_mat @ R).reshape(prev_chi_l, prev_d, new_chi)
        result[i - 1] = new_prev

    return result


def _tensors_to_network(mps_tensors: list[Tensor]) -> TensorNetwork:
    """Convert list of MPS tensors to TensorNetwork."""
    L = len(mps_tensors)
    result = TensorNetwork(name="TDVP_MPS")
    for i, tensor in enumerate(mps_tensors):
        result.add_node(i, tensor)
    for i in range(L - 1):
        shared = set(mps_tensors[i].labels()) & set(mps_tensors[i + 1].labels())
        for label in sorted(shared, key=str):
            try:
                result.connect(i, label, i + 1, label)
            except (ValueError, KeyError):
                # Bond indices may be incompatible after evolution changed
                # dimensions.  Skip — TensorNetwork is still usable via
                # shared labels.
                pass
    return result


# ------------------------------------------------------------------ #
# Compute energy                                                       #
# ------------------------------------------------------------------ #


def _compute_energy(
    mps_tensors: list[Tensor],
    mpo_tensors: list[Tensor],
) -> float:
    """Compute <psi|H|psi> by contracting environments left to right."""
    L = len(mps_tensors)
    left_env = _build_trivial_left_env()

    for i in range(L):
        left_env = _update_left_env(left_env, mps_tensors[i], mpo_tensors[i])

    return float(left_env.todense().real.ravel()[0])


def _compute_norm_sq(mps_tensors: list[Tensor]) -> float:
    """Compute <psi|psi> using identity MPO."""
    L = len(mps_tensors)
    # Build identity MPO tensors on the fly
    left_env = _build_trivial_left_env()

    for i in range(L):
        id_mpo = _identity_mpo_site(mps_tensors[i])
        left_env = _update_left_env(left_env, mps_tensors[i], id_mpo)

    return float(left_env.todense().real.ravel()[0])


def _identity_mpo_site(mps_site: Tensor) -> DenseTensor:
    """Create identity MPO site (1, d, d, 1)."""
    labels = mps_site.labels()
    dense = mps_site.todense()

    # All MPS tensors are 3-leg: (chi_l, d, chi_r)
    d = dense.shape[1]

    # Physical index is always at position 1 in a 3-leg MPS tensor
    site_idx = int(labels[1][1:])

    eye = jnp.eye(d, dtype=dense.dtype).reshape(1, d, d, 1)
    sym = U1Symmetry()
    bond_w = np.zeros(1, dtype=np.int32)
    bond_d = np.zeros(d, dtype=np.int32)

    indices = (
        TensorIndex(
            sym,
            bond_w,
            FlowDirection.IN,
            label=f"w{site_idx - 1}_{site_idx}"
            if site_idx > 0
            else f"w_left_{site_idx}",
        ),
        TensorIndex(sym, bond_d, FlowDirection.IN, label=f"mpo_top_{site_idx}"),
        TensorIndex(sym, bond_d, FlowDirection.OUT, label=f"mpo_bot_{site_idx}"),
        TensorIndex(
            sym, bond_w, FlowDirection.OUT, label=f"w{site_idx}_{site_idx + 1}"
        ),
    )
    return DenseTensor(eye, indices)


# ------------------------------------------------------------------ #
# Build environments from 3D arrays                                    #
# ------------------------------------------------------------------ #


def _build_right_envs(
    tensors_3d: list[jax.Array], mpo_tensors: list[Tensor], L: int
) -> list:
    """Build right environment DenseTensors from 3D MPS arrays."""
    envs: list[Tensor | None] = [None] * (L + 1)
    envs[L] = _build_trivial_right_env()

    for i in range(L - 1, 0, -1):
        # Wrap the 3D array as a DenseTensor for _update_right_env
        site_tensor = _make_site_tensor(tensors_3d[i], i, L)
        envs[i] = _update_right_env(envs[i + 1], site_tensor, mpo_tensors[i])

    return envs


def _build_left_envs_up_to(
    tensors_3d: list[jax.Array], mpo_tensors: list[Tensor], L: int, up_to: int
) -> list:
    """Build left environments from site 0 up to site up_to."""
    envs: list[Tensor | None] = [None] * (L + 1)
    envs[0] = _build_trivial_left_env()
    for i in range(min(up_to, L)):
        site_tensor = _make_site_tensor(tensors_3d[i], i, L)
        envs[i + 1] = _update_left_env(envs[i], site_tensor, mpo_tensors[i])
    return envs


# ------------------------------------------------------------------ #
# 1-site TDVP step                                                     #
# ------------------------------------------------------------------ #


def _tdvp_step_1site(
    mps: FiniteMPS | TensorNetwork,
    hamiltonian: TensorNetwork,
    config: TDVPConfig,
) -> FiniteMPS:
    """Perform one 1-site TDVP step with second-order Lie-Trotter splitting."""
    L = mps.n_nodes()
    mpo_tensors: list[Tensor] = [hamiltonian.get_tensor(i) for i in range(L)]

    # Convert MPS to 3D arrays
    tensors_3d: list[jax.Array] = []
    for i in range(L):
        arr = _site_to_3d(mps.get_tensor(i))
        tensors_3d.append(arr)

    # Determine dt factor: exp(dt_factor * H) is the evolution operator
    if config.time_type == "real":
        dt_factor = -1j * config.dt
    elif config.time_type == "imaginary":
        dt_factor = -config.dt
    else:  # "complex"
        dt_factor = -1j * config.dt
    dt_half = dt_factor / 2.0

    # Right-canonicalize
    tensors_3d = _right_canonicalize_dense(tensors_3d)

    # Build right environments
    R_envs = _build_right_envs(tensors_3d, mpo_tensors, L)

    # Initialize left environments
    L_envs: list[Tensor | None] = [None] * (L + 1)
    L_envs[0] = _build_trivial_left_env()

    # ---- Left-to-right sweep: sites 0 to L-2 ----
    for i in range(L - 1):
        l_env = L_envs[i]
        assert l_env is not None
        r_env = R_envs[i + 1]
        assert r_env is not None

        site_shape = tensors_3d[i].shape
        L_arr = l_env.todense()
        R_arr = r_env.todense()
        W_arr = mpo_tensors[i].todense()

        # Forward evolve site with dt/2
        def site_matvec(v, _shape=site_shape, _L=L_arr, _W=W_arr, _R=R_arr):
            return _matvec_1site_jit(v, _shape, _L, _W, _R)

        evolved_flat = krylov_expm(
            site_matvec,
            tensors_3d[i].ravel(),
            dt_half,
            config.krylov_dim,
            config.krylov_tol,
        )
        evolved_3d = evolved_flat.reshape(site_shape)

        # QR decomposition: (chi_l*d, chi_r) -> Q(chi_l*d, k) R(k, chi_r)
        chi_l, d_phys, chi_r = site_shape
        mat = evolved_3d.reshape(chi_l * d_phys, chi_r)
        Q_mat, R_mat = jnp.linalg.qr(mat)
        new_chi = Q_mat.shape[1]

        # Store Q as new site tensor (3D)
        tensors_3d[i] = Q_mat.reshape(chi_l, d_phys, new_chi)

        # Update left environment
        site_tensor_i = _make_site_tensor(tensors_3d[i], i, L)
        L_envs[i + 1] = _update_left_env(l_env, site_tensor_i, mpo_tensors[i])

        # Back-evolve bond matrix with -dt/2
        new_l_arr = L_envs[i + 1].todense()
        R_env_arr = r_env.todense()
        bond_shape = R_mat.shape

        def bond_mv(v, _bs=bond_shape, _Le=new_l_arr, _Re=R_env_arr):
            return _bond_matvec_jit(v, _bs, _Le, _Re)

        bond_evolved_flat = krylov_expm(
            bond_mv,
            R_mat.ravel(),
            -dt_half,
            config.krylov_dim,
            config.krylov_tol,
        )
        R_mat_evolved = bond_evolved_flat.reshape(bond_shape)

        # Absorb evolved bond matrix into next site: C @ next
        next_3d = tensors_3d[i + 1]
        next_chi_l, next_d, next_chi_r = next_3d.shape
        # R_mat_evolved: (new_chi, next_chi_l)
        next_mat = next_3d.reshape(next_chi_l, next_d * next_chi_r)
        new_next = (R_mat_evolved @ next_mat).reshape(new_chi, next_d, next_chi_r)
        tensors_3d[i + 1] = new_next

    # ---- Forward evolve site L-1 with dt/2 (no QR/back-evolve) ----
    i = L - 1
    l_env_last = L_envs[i]
    assert l_env_last is not None
    r_env_last = R_envs[i + 1]
    assert r_env_last is not None

    site_shape_last = tensors_3d[i].shape
    L_arr_last = l_env_last.todense()
    R_arr_last = r_env_last.todense()
    W_arr_last = mpo_tensors[i].todense()

    def site_mv_last(
        v, _shape=site_shape_last, _L=L_arr_last, _W=W_arr_last, _R=R_arr_last
    ):
        return _matvec_1site_jit(v, _shape, _L, _W, _R)

    evolved_last_flat = krylov_expm(
        site_mv_last,
        tensors_3d[i].ravel(),
        dt_half,
        config.krylov_dim,
        config.krylov_tol,
    )
    tensors_3d[i] = evolved_last_flat.reshape(site_shape_last)

    # ---- Right-to-left sweep: sites L-1 to 1 ----
    R_envs_new: list[Tensor | None] = [None] * (L + 1)
    R_envs_new[L] = _build_trivial_right_env()

    for i in range(L - 1, 0, -1):
        l_env = L_envs[i]
        assert l_env is not None
        r_env = R_envs_new[i + 1]
        assert r_env is not None

        site_shape = tensors_3d[i].shape
        L_arr = l_env.todense()
        R_arr = r_env.todense()
        W_arr = mpo_tensors[i].todense()

        # Forward evolve site with dt/2
        def site_mv_r(v, _shape=site_shape, _L=L_arr, _W=W_arr, _R=R_arr):
            return _matvec_1site_jit(v, _shape, _L, _W, _R)

        evolved_flat = krylov_expm(
            site_mv_r,
            tensors_3d[i].ravel(),
            dt_half,
            config.krylov_dim,
            config.krylov_tol,
        )
        evolved_3d = evolved_flat.reshape(site_shape)

        # RQ decomposition
        chi_l, d_phys, chi_r = site_shape
        mat = evolved_3d.reshape(chi_l, d_phys * chi_r)
        Q_t, R_t = jnp.linalg.qr(mat.T)
        Q = Q_t.T  # (new_chi, d*chi_r)
        R_mat = R_t.T  # (chi_l, new_chi)
        new_chi = Q.shape[0]

        tensors_3d[i] = Q.reshape(new_chi, d_phys, chi_r)

        # Update right environment
        site_tensor_i = _make_site_tensor(tensors_3d[i], i, L)
        R_envs_new[i] = _update_right_env(r_env, site_tensor_i, mpo_tensors[i])

        # Back-evolve bond matrix with -dt/2
        new_r_arr = R_envs_new[i].todense()
        L_env_arr = l_env.todense()
        bond_shape_r = R_mat.shape

        def bond_mv_r(v, _bs=bond_shape_r, _Le=L_env_arr, _Re=new_r_arr):
            return _bond_matvec_jit(v, _bs, _Le, _Re)

        bond_evolved_flat = krylov_expm(
            bond_mv_r,
            R_mat.ravel(),
            -dt_half,
            config.krylov_dim,
            config.krylov_tol,
        )
        R_mat_evolved = bond_evolved_flat.reshape(bond_shape_r)

        # Absorb into previous site: prev @ C
        prev_3d = tensors_3d[i - 1]
        prev_chi_l, prev_d, prev_chi_r = prev_3d.shape
        # R_mat_evolved: (prev_chi_r, new_chi) = (chi_l, new_chi)
        prev_mat = prev_3d.reshape(prev_chi_l * prev_d, prev_chi_r)
        new_prev = (prev_mat @ R_mat_evolved).reshape(prev_chi_l, prev_d, new_chi)
        tensors_3d[i - 1] = new_prev

    # ---- Final: evolve site 0 with dt/2 ----
    r_env_0 = R_envs_new[1]
    assert r_env_0 is not None
    l_env_0 = L_envs[0]
    assert l_env_0 is not None

    site0_shape = tensors_3d[0].shape
    L0_arr = l_env_0.todense()
    R0_arr = r_env_0.todense()
    W0_arr = mpo_tensors[0].todense()

    def site0_mv(v, _shape=site0_shape, _L=L0_arr, _W=W0_arr, _R=R0_arr):
        return _matvec_1site_jit(v, _shape, _L, _W, _R)

    evolved0_flat = krylov_expm(
        site0_mv,
        tensors_3d[0].ravel(),
        dt_half,
        config.krylov_dim,
        config.krylov_tol,
    )
    tensors_3d[0] = evolved0_flat.reshape(site0_shape)

    # Normalize for imaginary time
    if config.time_type == "imaginary":
        tensors_3d = _normalize_3d(tensors_3d, L)

    # Convert back to DenseTensors
    mps_tensors = [_make_site_tensor(tensors_3d[i], i, L) for i in range(L)]
    return FiniteMPS.from_tensors(mps_tensors, orth_center=0)


# ------------------------------------------------------------------ #
# 2-site TDVP step                                                     #
# ------------------------------------------------------------------ #


def _tdvp_step_2site(
    mps: FiniteMPS | TensorNetwork,
    hamiltonian: TensorNetwork,
    config: TDVPConfig,
) -> FiniteMPS:
    """Perform one 2-site TDVP step."""
    L = mps.n_nodes()
    mpo_tensors: list[Tensor] = [hamiltonian.get_tensor(i) for i in range(L)]

    tensors_3d: list[jax.Array] = []
    for i in range(L):
        arr = _site_to_3d(mps.get_tensor(i))
        tensors_3d.append(arr)

    if config.time_type == "real":
        dt_factor = -1j * config.dt
    else:
        dt_factor = -config.dt
    dt_half = dt_factor / 2.0

    # Right-canonicalize
    tensors_3d = _right_canonicalize_dense(tensors_3d)

    # Build right environments
    R_envs = _build_right_envs(tensors_3d, mpo_tensors, L)

    L_envs: list[Tensor | None] = [None] * (L + 1)
    L_envs[0] = _build_trivial_left_env()

    # ---- Left-to-right sweep ----
    for i in range(L - 1):
        l_env = L_envs[i]
        assert l_env is not None
        r_env = (
            R_envs[i + 2] if R_envs[i + 2] is not None else _build_trivial_right_env()
        )

        # Form 2-site tensor: theta[chi_l, d_l, d_r, chi_r]
        chi_l, d_l, chi_m = tensors_3d[i].shape
        chi_m2, d_r, chi_r = tensors_3d[i + 1].shape
        # chi_m should equal chi_m2
        theta_4d = jnp.einsum("abc,cde->abde", tensors_3d[i], tensors_3d[i + 1])
        theta_shape = theta_4d.shape

        L_arr = l_env.todense()
        R_arr = r_env.todense()
        W_l_arr = mpo_tensors[i].todense()
        W_r_arr = mpo_tensors[i + 1].todense()

        def two_mv(v, _ts=theta_shape, _L=L_arr, _Wl=W_l_arr, _Wr=W_r_arr, _R=R_arr):
            return _matvec_jit(v, _ts, _L, _Wl, _Wr, _R)

        evolved_flat = krylov_expm(
            two_mv,
            theta_4d.ravel(),
            dt_half,
            config.krylov_dim,
            config.krylov_tol,
        )
        evolved_4d = evolved_flat.reshape(theta_shape)

        # SVD truncate
        chi_l_dim, dl, dr, chi_r_dim = theta_shape
        mat = evolved_4d.reshape(chi_l_dim * dl, dr * chi_r_dim)
        U, s, Vh = jnp.linalg.svd(mat, full_matrices=False)

        max_bond = config.max_bond_dim
        if config.svd_trunc_err is not None:
            total = jnp.sum(s**2)
            cumulative = jnp.cumsum(s[::-1] ** 2)[::-1]
            keep = int(jnp.sum(cumulative / total > config.svd_trunc_err**2))
            keep = max(keep, 1)
            max_bond = min(max_bond, keep)
        n_keep = min(max_bond, len(s))

        U = U[:, :n_keep]
        s_trunc = s[:n_keep]
        Vh = Vh[:n_keep, :]

        # Left sweep: A is left-canonical, absorb s into B
        A_3d = U.reshape(chi_l_dim, dl, n_keep)
        sVh = jnp.diag(s_trunc) @ Vh
        B_3d = sVh.reshape(n_keep, dr, chi_r_dim)

        tensors_3d[i] = A_3d
        tensors_3d[i + 1] = B_3d

        # Update left environment
        site_i = _make_site_tensor(tensors_3d[i], i, L)
        L_envs[i + 1] = _update_left_env(l_env, site_i, mpo_tensors[i])

    # ---- Right-to-left sweep ----
    R_envs_new: list[Tensor | None] = [None] * (L + 1)
    R_envs_new[L] = _build_trivial_right_env()

    for i in range(L - 2, -1, -1):
        l_env = L_envs[i]
        assert l_env is not None
        r_env = (
            R_envs_new[i + 2]
            if R_envs_new[i + 2] is not None
            else _build_trivial_right_env()
        )

        chi_l, d_l, chi_m = tensors_3d[i].shape
        chi_m2, d_r, chi_r = tensors_3d[i + 1].shape
        theta_4d = jnp.einsum("abc,cde->abde", tensors_3d[i], tensors_3d[i + 1])
        theta_shape = theta_4d.shape

        L_arr = l_env.todense()
        R_arr = r_env.todense()
        W_l_arr = mpo_tensors[i].todense()
        W_r_arr = mpo_tensors[i + 1].todense()

        def two_mv_r(v, _ts=theta_shape, _L=L_arr, _Wl=W_l_arr, _Wr=W_r_arr, _R=R_arr):
            return _matvec_jit(v, _ts, _L, _Wl, _Wr, _R)

        evolved_flat = krylov_expm(
            two_mv_r,
            theta_4d.ravel(),
            dt_half,
            config.krylov_dim,
            config.krylov_tol,
        )
        evolved_4d = evolved_flat.reshape(theta_shape)

        chi_l_dim, dl, dr, chi_r_dim = theta_shape
        mat = evolved_4d.reshape(chi_l_dim * dl, dr * chi_r_dim)
        U, s, Vh = jnp.linalg.svd(mat, full_matrices=False)

        max_bond = config.max_bond_dim
        if config.svd_trunc_err is not None:
            total = jnp.sum(s**2)
            cumulative = jnp.cumsum(s[::-1] ** 2)[::-1]
            keep = int(jnp.sum(cumulative / total > config.svd_trunc_err**2))
            keep = max(keep, 1)
            max_bond = min(max_bond, keep)
        n_keep = min(max_bond, len(s))

        U = U[:, :n_keep]
        s_trunc = s[:n_keep]
        Vh = Vh[:n_keep, :]

        # Right sweep: B is right-canonical, absorb s into A
        Us = U @ jnp.diag(s_trunc)
        A_3d = Us.reshape(chi_l_dim, dl, n_keep)
        B_3d = Vh.reshape(n_keep, dr, chi_r_dim)

        tensors_3d[i] = A_3d
        tensors_3d[i + 1] = B_3d

        # Update right environment
        site_j = _make_site_tensor(tensors_3d[i + 1], i + 1, L)
        R_envs_new[i + 1] = _update_right_env(r_env, site_j, mpo_tensors[i + 1])

    # Normalize for imaginary time
    if config.time_type == "imaginary":
        tensors_3d = _normalize_3d(tensors_3d, L)

    mps_tensors = [_make_site_tensor(tensors_3d[i], i, L) for i in range(L)]
    return FiniteMPS.from_tensors(mps_tensors, orth_center=0)


# ------------------------------------------------------------------ #
# Normalization                                                        #
# ------------------------------------------------------------------ #


def _normalize_3d(tensors_3d: list[jax.Array], L: int) -> list[jax.Array]:
    """Normalize MPS (3D arrays) so that <psi|psi> = 1."""
    mps_tensors = [_make_site_tensor(tensors_3d[i], i, L) for i in range(L)]
    norm_sq = _compute_norm_sq(mps_tensors)
    norm = np.sqrt(abs(norm_sq))
    if norm > 1e-15:
        tensors_3d[0] = tensors_3d[0] / norm
    return tensors_3d


# ------------------------------------------------------------------ #
# Dispatchers                                                          #
# ------------------------------------------------------------------ #


def tdvp_step(
    mps: FiniteMPS | TensorNetwork,
    hamiltonian: TensorNetwork,
    config: TDVPConfig,
) -> FiniteMPS:
    """Perform one TDVP time step.

    Args:
        mps:          MPS as FiniteMPS or TensorNetwork (backward compat).
        hamiltonian:  MPO Hamiltonian as TensorNetwork.
        config:       TDVPConfig.

    Returns:
        Updated MPS as FiniteMPS.
    """
    # Convert TensorNetwork to FiniteMPS if needed
    if isinstance(mps, TensorNetwork):
        _tensors = [mps.get_tensor(i) for i in range(mps.n_nodes())]
        mps = FiniteMPS.from_tensors(_tensors)

    if config.mode == "1site":
        return _tdvp_step_1site(mps, hamiltonian, config)
    elif config.mode == "2site":
        return _tdvp_step_2site(mps, hamiltonian, config)
    else:
        raise ValueError(f"Unknown TDVP mode: {config.mode!r}. Use '1site' or '2site'.")


def tdvp(
    mps: FiniteMPS | TensorNetwork,
    hamiltonian: TensorNetwork,
    config: TDVPConfig,
    measure: Callable[[TensorNetwork, int], dict[str, float]] | None = None,
) -> TDVPResult:
    """Run multi-step TDVP time evolution.

    Args:
        mps:          Initial MPS as FiniteMPS or TensorNetwork (backward compat).
        hamiltonian:  MPO Hamiltonian.
        config:       TDVPConfig.
        measure:      Optional callback(mps, step) -> dict of observables.
                      Receives a TensorNetwork for backward compatibility.

    Returns:
        TDVPResult with final MPS as FiniteMPS, times, energies, and observables.
    """
    L = hamiltonian.n_nodes()
    mpo_tensors = [hamiltonian.get_tensor(i) for i in range(L)]

    # Convert TensorNetwork to FiniteMPS if needed
    if isinstance(mps, TensorNetwork):
        _tensors = [mps.get_tensor(i) for i in range(mps.n_nodes())]
        mps = FiniteMPS.from_tensors(_tensors)

    current_mps: FiniteMPS = mps
    times: list[float] = []
    energies: list[float] = []
    observables: dict[str, list[float]] = {}

    def _record(step: int, current: FiniteMPS) -> None:
        t = step * config.dt
        times.append(t)

        cur_tensors = list(current.tensors)
        e = _compute_energy(cur_tensors, mpo_tensors)
        energies.append(e)

        if measure is not None:
            # Convert to TensorNetwork for backward-compatible callback
            mps_tn = _tensors_to_network(cur_tensors)
            obs = measure(mps_tn, step)
            for key, val in obs.items():
                if key not in observables:
                    observables[key] = []
                observables[key].append(val)

    _record(0, current_mps)

    for step in range(1, config.num_steps + 1):
        current_mps = tdvp_step(current_mps, hamiltonian, config)
        _record(step, current_mps)

        if config.verbose:
            print(f"TDVP step {step}/{config.num_steps}: E = {energies[-1]:.10f}")

    return TDVPResult(
        mps=current_mps,
        times=times,
        energies=energies,
        observables=observables,
    )
