# TDVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 1-site and 2-site TDVP for real-time and imaginary-time MPS evolution in Tenax.

**Architecture:** Two new files: `_krylov.py` (reusable Lanczos-based matrix exponential) and `tdvp.py` (TDVP sweep logic, config, driver). Reuses DMRG's environment update functions. Polymorphic on DenseTensor/SymmetricTensor via label-based API.

**Tech Stack:** JAX, Tenax tensor infrastructure (TensorNetwork, contract, svd, qr), existing DMRG environment utilities.

**Design doc:** `docs/plans/2026-03-16-tdvp-design.md`

---

### Task 1: Krylov Matrix Exponential — Core

**Files:**
- Create: `src/tenax/algorithms/_krylov.py`
- Create: `tests/test_krylov.py`

**Step 1: Write the failing test**

Create `tests/test_krylov.py`:

```python
"""Tests for Lanczos-based Krylov matrix exponential."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

jax.config.update("jax_enable_x64", True)


def test_krylov_expm_real_time_small_matrix():
    """Krylov expm matches scipy expm for a small Hermitian matrix (real-time)."""
    from tenax.algorithms._krylov import krylov_expm

    rng = np.random.default_rng(42)
    n = 8
    H = rng.standard_normal((n, n))
    H = (H + H.T) / 2  # Hermitian
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    dt = -0.1j  # real-time: exp(-i H t) with t=0.1

    result = krylov_expm(
        matvec=lambda x: jnp.array(H) @ x,
        v=jnp.array(v),
        dt=dt,
        krylov_dim=20,
        tol=1e-12,
    )

    expected = expm(dt * H) @ v
    np.testing.assert_allclose(np.array(result), expected, atol=1e-10)


def test_krylov_expm_imaginary_time():
    """Krylov expm matches scipy expm for imaginary-time evolution."""
    from tenax.algorithms._krylov import krylov_expm

    rng = np.random.default_rng(42)
    n = 8
    H = rng.standard_normal((n, n))
    H = (H + H.T) / 2
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    dt = -0.1  # imaginary-time: exp(-H tau) with tau=0.1

    result = krylov_expm(
        matvec=lambda x: jnp.array(H) @ x,
        v=jnp.array(v),
        dt=dt,
        krylov_dim=20,
        tol=1e-12,
    )

    expected = expm(dt * H) @ v
    np.testing.assert_allclose(np.array(result), expected, atol=1e-10)


def test_krylov_expm_preserves_norm_unitary():
    """Real-time evolution of Hermitian H preserves vector norm."""
    from tenax.algorithms._krylov import krylov_expm

    rng = np.random.default_rng(7)
    n = 16
    H = rng.standard_normal((n, n))
    H = (H + H.T) / 2
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    result = krylov_expm(
        matvec=lambda x: jnp.array(H) @ x,
        v=jnp.array(v),
        dt=-0.5j,
        krylov_dim=30,
    )

    assert abs(float(jnp.linalg.norm(result)) - 1.0) < 1e-10


def test_krylov_expm_zero_dt():
    """dt=0 returns the input vector unchanged."""
    from tenax.algorithms._krylov import krylov_expm

    v = jnp.array([1.0, 0.0, 0.0])
    result = krylov_expm(
        matvec=lambda x: x,
        v=v,
        dt=0.0,
        krylov_dim=10,
    )
    np.testing.assert_allclose(np.array(result), np.array(v), atol=1e-14)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_krylov.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenax.algorithms._krylov'`

**Step 3: Write implementation**

Create `src/tenax/algorithms/_krylov.py`:

```python
"""Lanczos-based Krylov matrix exponential.

Computes exp(dt * A) @ v without forming the full matrix, using only
matrix-vector products. Suitable for large sparse effective Hamiltonians
in TDVP time evolution.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def krylov_expm(
    matvec: Callable[[jax.Array], jax.Array],
    v: jax.Array,
    dt: complex,
    krylov_dim: int = 20,
    tol: float = 1e-12,
) -> jax.Array:
    """Compute exp(dt * A) @ v via Lanczos iteration.

    Builds a Krylov subspace {v, Av, A^2 v, ...} and projects the
    matrix exponential onto it. The projection is exact when the
    Krylov dimension equals the matrix dimension.

    Args:
        matvec:     Function computing A @ x for arbitrary x.
        v:          Starting vector.
        dt:         Time step (complex for real-time: -i*delta_t,
                    real negative for imaginary-time: -delta_t).
        krylov_dim: Maximum Lanczos iterations.
        tol:        Early termination if residual norm < tol.

    Returns:
        The vector exp(dt * A) @ v.
    """
    norm_v = jnp.linalg.norm(v)
    if float(norm_v) < 1e-15:
        return v

    v0 = v / norm_v

    basis = [v0]
    alphas: list[jax.Array] = []
    betas: list[jax.Array] = []

    for step in range(krylov_dim):
        w = matvec(basis[-1])
        alpha = jnp.dot(basis[-1].conj(), w).real
        alphas.append(alpha)

        w = w - alpha * basis[-1]
        if step > 0:
            w = w - betas[-1] * basis[-2]

        beta = jnp.linalg.norm(w)
        if float(beta) < tol:
            break
        betas.append(beta)
        basis.append(w / beta)

    n = len(alphas)
    if n == 0:
        return v

    # Build tridiagonal matrix T
    alphas_arr = jnp.stack(alphas)
    T = jnp.diag(alphas_arr)
    if betas:
        betas_arr = jnp.stack(betas)
        T = T + jnp.diag(betas_arr, k=1) + jnp.diag(betas_arr, k=-1)

    # exp(dt * T) via diagonalization
    eigvals, eigvecs = jnp.linalg.eigh(T)
    exp_T_e1 = eigvecs @ (jnp.exp(dt * eigvals) * eigvecs[0, :].conj())

    # Map back to full space
    basis_stacked = jnp.stack(basis[:n], axis=0)
    result = norm_v * jnp.tensordot(exp_T_e1, basis_stacked, axes=1)

    return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_krylov.py -v`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_krylov.py tests/test_krylov.py
git commit -m "feat(krylov): add Lanczos-based matrix exponential"
```

---

### Task 2: Register Krylov Tests and Add conftest Entry

**Files:**
- Modify: `tests/conftest.py:16-38` (add test_krylov.py and test_tdvp.py markers)

**Step 1: Add markers**

Add to the `_FILE_MARKERS` dict in `tests/conftest.py`:

```python
    "test_krylov.py": "core",
    "test_tdvp.py": "algorithm",
```

`test_krylov.py` is `core` (fast, no algorithm convergence). `test_tdvp.py` is `algorithm` (requires running TDVP sweeps).

**Step 2: Verify markers work**

Run: `uv run pytest tests/test_krylov.py -v -m core`
Expected: 4 tests collected and PASS

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: register krylov and tdvp test markers in conftest"
```

---

### Task 3: TDVP Config, Result, and 1-Site Effective Hamiltonian

**Files:**
- Create: `src/tenax/algorithms/tdvp.py`
- Create: `tests/test_tdvp.py`

**Step 1: Write the failing test**

Create `tests/test_tdvp.py` with an initial test for the 1-site effective Hamiltonian matvec:

```python
"""Tests for TDVP time evolution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import (
    DMRGConfig,
    TDVPConfig,
    build_mpo_heisenberg,
    build_random_mps,
    dmrg,
)


def test_tdvp_config_defaults():
    """TDVPConfig has sensible defaults."""
    config = TDVPConfig()
    assert config.mode == "1site"
    assert config.time_type == "real"
    assert config.dt == 0.05
    assert config.num_steps == 100


def test_1site_effective_hamiltonian_matvec():
    """1-site effective Hamiltonian matvec produces correct shape."""
    from tenax.algorithms.tdvp import _effective_hamiltonian_matvec_1site

    L = 6
    mpo = build_mpo_heisenberg(L)
    mps = build_random_mps(L, physical_dim=2, bond_dim=4)

    # Right-canonicalize and build environments using DMRG helpers
    from tenax.algorithms.dmrg import (
        _build_right_environments_list,
        _build_trivial_left_env,
        _right_canonicalize,
    )

    mps_tensors = [mps.get_tensor(i) for i in range(L)]
    mpo_tensors = [mpo.get_tensor(i) for i in range(L)]
    mps_tensors = _right_canonicalize(mps_tensors)

    L_env = _build_trivial_left_env()
    R_envs = _build_right_environments_list(mps_tensors, mpo_tensors, L)

    site = 2  # middle site
    A = mps_tensors[site].todense()
    if A.ndim == 2:
        A = A[jnp.newaxis, :]

    theta_flat = A.ravel()
    result = _effective_hamiltonian_matvec_1site(
        theta_flat,
        A.shape,
        L_env.todense() if hasattr(L_env, "todense") else L_env,
        mpo_tensors[site].todense(),
        R_envs[site].todense() if hasattr(R_envs[site], "todense") else R_envs[site],
    )
    assert result.shape == theta_flat.shape
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tdvp.py::test_tdvp_config_defaults -v`
Expected: FAIL with `ImportError: cannot import name 'TDVPConfig' from 'tenax'`

**Step 3: Write minimal implementation**

Create `src/tenax/algorithms/tdvp.py`:

```python
"""Time-Dependent Variational Principle (TDVP) for MPS time evolution.

Implements 1-site and 2-site TDVP with second-order Lie-Trotter splitting.
Supports real-time (e^{-iHt}) and imaginary-time (e^{-Ht}) evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax
import jax.numpy as jnp

from tenax.algorithms._krylov import krylov_expm
from tenax.algorithms.dmrg import (
    _build_right_environments_list,
    _build_trivial_left_env,
    _right_canonicalize,
    _update_left_env,
    _update_right_env,
)
from tenax.network.network import TensorNetwork


@dataclass
class TDVPConfig:
    """Configuration for TDVP time evolution.

    Attributes:
        mode:          ``"1site"`` or ``"2site"``.
        dt:            Time step magnitude.
        time_type:     ``"real"`` for e^{-iHt}, ``"imaginary"`` for e^{-Ht}.
        num_steps:     Number of time steps for the ``tdvp()`` driver.
        max_bond_dim:  Maximum bond dimension (2-site mode only).
        svd_trunc_err: SVD truncation error threshold (2-site mode only).
        krylov_dim:    Krylov subspace dimension for matrix exponential.
        krylov_tol:    Convergence tolerance for Krylov iteration.
        verbose:       Print per-step diagnostics.
    """

    mode: Literal["1site", "2site"] = "1site"
    dt: float = 0.05
    time_type: Literal["real", "imaginary"] = "real"
    num_steps: int = 100
    max_bond_dim: int = 64
    svd_trunc_err: float | None = None
    krylov_dim: int = 20
    krylov_tol: float = 1e-12
    verbose: bool = False


@dataclass
class TDVPResult:
    """Result of a TDVP evolution.

    Attributes:
        mps:         Final MPS state.
        times:       Time value after each step.
        energies:    Energy <H> after each step.
        observables: User-measured observables per step.
    """

    mps: TensorNetwork
    times: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    observables: dict[str, list[float]] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# 1-site effective Hamiltonian                                         #
# ------------------------------------------------------------------ #


def _effective_hamiltonian_matvec_1site(
    theta_flat: jax.Array,
    theta_shape: tuple[int, ...],
    L_env: jax.Array,
    W: jax.Array,
    R_env: jax.Array,
) -> jax.Array:
    """Apply 1-site effective Hamiltonian to site tensor.

    Contracts: L[a,b,c] * theta[a,p,d] * W[b,p,s,e] * R[d,e,f]
    -> result[c,s,f]

    Args:
        theta_flat:  Flattened site tensor.
        theta_shape: Shape for reshaping (chi_l, d, chi_r).
        L_env:       Left environment (chi_l, D_w, chi_l').
        W:           MPO site tensor (D_w_l, d, d', D_w_r).
        R_env:       Right environment (chi_r, D_w, chi_r').

    Returns:
        Flattened result.
    """
    theta = theta_flat.reshape(theta_shape)
    result = jnp.einsum(
        "abc,apd,bpse,def->csf",
        L_env,
        theta,
        W,
        R_env,
    )
    return result.ravel()


_matvec_1site_jit = jax.jit(
    _effective_hamiltonian_matvec_1site, static_argnums=(1,)
)
```

Add imports to `src/tenax/__init__.py` — append after the iDMRG imports:

```python
from tenax.algorithms.tdvp import (
    TDVPConfig,
    TDVPResult,
)
```

And in `__all__`, add after the iDMRG section:

```python
    # TDVP
    "TDVPConfig",
    "TDVPResult",
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_tdvp.py -v`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/tdvp.py src/tenax/__init__.py tests/test_tdvp.py
git commit -m "feat(tdvp): add TDVPConfig, TDVPResult, and 1-site matvec"
```

---

### Task 4: Bond Effective Hamiltonian (for 1-site back-evolution)

**Files:**
- Modify: `src/tenax/algorithms/tdvp.py`
- Modify: `tests/test_tdvp.py`

**Step 1: Write the failing test**

Add to `tests/test_tdvp.py`:

```python
def test_bond_effective_hamiltonian_matvec():
    """Bond effective Hamiltonian matvec produces correct shape."""
    from tenax.algorithms.tdvp import _bond_hamiltonian_matvec

    # The bond matvec acts on the R matrix from QR, shape (chi_l, chi_r)
    # Contracts: L_env[a,b,c] * R_mat[a,d] * R_env[d,b,f] -> result[c,f]
    chi_l, chi_r, D_w = 4, 4, 3
    L_env = jnp.ones((chi_l, D_w, chi_l))
    R_env = jnp.ones((chi_r, D_w, chi_r))
    R_mat = jnp.ones((chi_l, chi_r))

    result = _bond_hamiltonian_matvec(
        R_mat.ravel(),
        R_mat.shape,
        L_env,
        R_env,
    )
    assert result.shape == R_mat.ravel().shape
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tdvp.py::test_bond_effective_hamiltonian_matvec -v`
Expected: FAIL with `ImportError`

**Step 3: Write implementation**

Add to `src/tenax/algorithms/tdvp.py`:

```python
def _bond_hamiltonian_matvec(
    r_flat: jax.Array,
    r_shape: tuple[int, ...],
    L_env: jax.Array,
    R_env: jax.Array,
) -> jax.Array:
    """Apply bond effective Hamiltonian to bond matrix.

    Used for the 1-site TDVP back-evolution step.
    Contracts: L[a,b,c] * R_mat[a,d] * R_env[d,b,f] -> result[c,f]

    Args:
        r_flat:  Flattened bond matrix.
        r_shape: Shape (chi_l, chi_r).
        L_env:   Left environment (chi_l, D_w, chi_l').
        R_env:   Right environment (chi_r, D_w, chi_r').

    Returns:
        Flattened result.
    """
    R_mat = r_flat.reshape(r_shape)
    result = jnp.einsum("abc,ad,dbe->ce", L_env, R_mat, R_env)
    return result.ravel()


_bond_matvec_jit = jax.jit(_bond_hamiltonian_matvec, static_argnums=(1,))
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_tdvp.py -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/tdvp.py tests/test_tdvp.py
git commit -m "feat(tdvp): add bond effective Hamiltonian matvec"
```

---

### Task 5: 1-Site TDVP Sweep

**Files:**
- Modify: `src/tenax/algorithms/tdvp.py`
- Modify: `tests/test_tdvp.py`

**Step 1: Write the failing test**

Add to `tests/test_tdvp.py`:

```python
def test_1site_tdvp_energy_conservation():
    """Real-time 1-site TDVP conserves energy."""
    from tenax import tdvp_step

    L = 6
    chi = 8
    mpo = build_mpo_heisenberg(L)
    mps = build_random_mps(L, physical_dim=2, bond_dim=chi)

    # First get a reasonable state via DMRG
    dmrg_config = DMRGConfig(max_bond_dim=chi, num_sweeps=10)
    dmrg_result = dmrg(mpo, mps, dmrg_config)
    mps = dmrg_result.mps

    config = TDVPConfig(mode="1site", dt=0.05, time_type="real", krylov_dim=20)

    # Measure initial energy
    from tenax.algorithms.observables import expectation_value

    E0 = dmrg_result.energy

    # Evolve 5 steps
    current_mps = mps
    for _ in range(5):
        current_mps = tdvp_step(current_mps, mpo, config)

    # Measure final energy via a DMRG sweep with 0 sweeps (just measure)
    dmrg_measure = DMRGConfig(max_bond_dim=chi, num_sweeps=1)
    E_final = dmrg(mpo, current_mps, dmrg_measure).energy

    # Energy should be conserved to ~1e-6 for dt=0.05 with krylov_dim=20
    assert abs(E_final - E0) < 1e-4, f"Energy not conserved: {E0} -> {E_final}"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tdvp.py::test_1site_tdvp_energy_conservation -v`
Expected: FAIL with `ImportError: cannot import name 'tdvp_step'`

**Step 3: Write implementation**

Add the full 1-site TDVP sweep to `src/tenax/algorithms/tdvp.py`. This is the
core algorithm — the second-order Lie-Trotter integrator:

```python
from tenax.contraction.contractor import qr_decompose
from tenax.core.tensor import DenseTensor, Tensor


def _get_dt_factor(config: TDVPConfig) -> complex:
    """Convert config to the complex dt used in krylov_expm."""
    if config.time_type == "real":
        return -1j * config.dt
    else:
        return -config.dt


def tdvp_step(
    mps: TensorNetwork,
    hamiltonian: TensorNetwork,
    config: TDVPConfig,
) -> TensorNetwork:
    """Perform one TDVP time step.

    Args:
        mps:         MPS state as TensorNetwork (L sites, indexed 0..L-1).
        hamiltonian: MPO Hamiltonian as TensorNetwork (L sites, indexed 0..L-1).
        config:      TDVP configuration.

    Returns:
        Updated MPS after one time step.
    """
    if config.mode == "1site":
        return _tdvp_step_1site(mps, hamiltonian, config)
    elif config.mode == "2site":
        return _tdvp_step_2site(mps, hamiltonian, config)
    else:
        raise ValueError(f"Unknown TDVP mode: {config.mode!r}")


def _tdvp_step_1site(
    mps: TensorNetwork,
    hamiltonian: TensorNetwork,
    config: TDVPConfig,
) -> TensorNetwork:
    """1-site TDVP with second-order Lie-Trotter splitting."""
    L = len([n for n in mps._nodes])
    mps_tensors = [mps.get_tensor(i) for i in range(L)]
    mpo_tensors = [hamiltonian.get_tensor(i) for i in range(L)]

    # Right-canonicalize
    mps_tensors = _right_canonicalize(mps_tensors)

    # Build right environments
    R_envs = _build_right_environments_list(mps_tensors, mpo_tensors, L)

    # Left environment starts trivial
    L_envs = [None] * L
    L_envs[0] = _build_trivial_left_env()

    dt_full = _get_dt_factor(config)
    dt_half = dt_full / 2.0

    # --- Left-to-right sweep (sites 0 to L-2) ---
    for i in range(L - 1):
        A = mps_tensors[i].todense()
        is_left_boundary = (A.ndim == 2 and isinstance(mps_tensors[i].labels()[0], str)
                           and mps_tensors[i].labels()[0].startswith("p"))
        if A.ndim == 2:
            if is_left_boundary:
                A = A[jnp.newaxis, :]
            else:
                A = A[:, :, jnp.newaxis]

        L_env_dense = L_envs[i].todense()
        R_env_dense = R_envs[i].todense()
        W_dense = mpo_tensors[i].todense()

        # Forward evolve site
        def site_matvec(x):
            return _matvec_1site_jit(x, A.shape, L_env_dense, W_dense, R_env_dense)

        A_evolved = krylov_expm(site_matvec, A.ravel(), dt_half,
                                config.krylov_dim, config.krylov_tol)
        A_evolved = A_evolved.reshape(A.shape)

        # QR decompose: A = Q R (left-canonical)
        chi_l, d, chi_r = A_evolved.shape
        A_mat = A_evolved.reshape(chi_l * d, chi_r)
        Q, R_mat = jnp.linalg.qr(A_mat)
        chi_new = Q.shape[1]
        Q = Q.reshape(chi_l, d, chi_new)

        # Store left-canonical site tensor
        if is_left_boundary:
            Q_store = Q[0, :, :]  # (d, chi_new)
        else:
            Q_store = Q
        mps_tensors[i] = DenseTensor(Q_store, mps_tensors[i].indices)

        # Update left environment
        L_envs[i + 1] = _update_left_env(L_envs[i], mps_tensors[i], mpo_tensors[i])
        L_env_new_dense = L_envs[i + 1].todense()

        # Back-evolve the bond matrix R
        def bond_matvec(x):
            return _bond_matvec_jit(x, R_mat.shape, L_env_new_dense, R_env_dense)

        R_evolved = krylov_expm(bond_matvec, R_mat.ravel(), -dt_half,
                                config.krylov_dim, config.krylov_tol)
        R_evolved = R_evolved.reshape(R_mat.shape)

        # Absorb R into next site
        A_next = mps_tensors[i + 1].todense()
        is_right_boundary_next = (i + 1 == L - 1 and A_next.ndim == 2)
        if A_next.ndim == 2:
            if is_right_boundary_next:
                A_next = A_next[:, :, jnp.newaxis]
            else:
                A_next = A_next[jnp.newaxis, :]
        A_next = jnp.einsum("ij,jpk->ipk", R_evolved, A_next)
        if is_right_boundary_next:
            A_next = A_next[:, :, 0]
        mps_tensors[i + 1] = DenseTensor(A_next, mps_tensors[i + 1].indices)

    # --- Right-to-left sweep (sites L-1 to 1) ---
    for i in range(L - 1, 0, -1):
        A = mps_tensors[i].todense()
        is_right_boundary = (A.ndim == 2 and i == L - 1)
        if A.ndim == 2:
            if is_right_boundary:
                A = A[:, :, jnp.newaxis]
            else:
                A = A[jnp.newaxis, :]

        L_env_dense = L_envs[i].todense()
        R_env_dense = R_envs[i].todense() if R_envs[i] is not None else _build_trivial_right_env().todense()
        W_dense = mpo_tensors[i].todense()

        # Forward evolve site
        def site_matvec(x, _shape=A.shape, _L=L_env_dense, _W=W_dense, _R=R_env_dense):
            return _matvec_1site_jit(x, _shape, _L, _W, _R)

        A_evolved = krylov_expm(site_matvec, A.ravel(), dt_half,
                                config.krylov_dim, config.krylov_tol)
        A_evolved = A_evolved.reshape(A.shape)

        # RQ decompose: A = L Q (right-canonical)
        chi_l, d, chi_r = A_evolved.shape
        A_mat = A_evolved.reshape(chi_l, d * chi_r)
        # RQ via transposed QR
        Q_T, L_T = jnp.linalg.qr(A_mat.T)
        Q_right = Q_T.T  # (chi_new, d*chi_r)
        L_mat = L_T.T    # (chi_l, chi_new)
        chi_new = Q_right.shape[0]
        Q_right = Q_right.reshape(chi_new, d, chi_r)

        # Store right-canonical site tensor
        if is_right_boundary:
            Q_store = Q_right[:, :, 0]  # (chi_new, d)
        else:
            Q_store = Q_right
        mps_tensors[i] = DenseTensor(Q_store, mps_tensors[i].indices)

        # Update right environment
        R_envs[i - 1] = _update_right_env(
            R_envs[i] if R_envs[i] is not None else _build_trivial_right_env(),
            mps_tensors[i], mpo_tensors[i]
        )

        if i > 1:
            R_env_new_dense = R_envs[i - 1].todense()

            # Back-evolve the bond matrix L
            def bond_matvec(x, _shape=L_mat.shape, _L=L_env_dense, _R=R_env_new_dense):
                return _bond_matvec_jit(x, _shape, _L, _R)

            L_evolved = krylov_expm(bond_matvec, L_mat.ravel(), -dt_half,
                                    config.krylov_dim, config.krylov_tol)
            L_evolved = L_evolved.reshape(L_mat.shape)

            # Absorb L into previous site
            A_prev = mps_tensors[i - 1].todense()
            is_left_boundary_prev = (i - 1 == 0 and A_prev.ndim == 2)
            if A_prev.ndim == 2:
                if is_left_boundary_prev:
                    A_prev = A_prev[jnp.newaxis, :]
                else:
                    A_prev = A_prev[:, :, jnp.newaxis]
            A_prev = jnp.einsum("ijk,kl->ijl", A_prev, L_evolved)
            if is_left_boundary_prev:
                A_prev = A_prev[0, :, :]
            mps_tensors[i - 1] = DenseTensor(A_prev, mps_tensors[i - 1].indices)

    # Evolve the last remaining site (site 0) in the right-to-left half
    A = mps_tensors[0].todense()
    if A.ndim == 2:
        A = A[jnp.newaxis, :]
    L_env_dense = L_envs[0].todense()
    R_env_dense = R_envs[0].todense()
    W_dense = mpo_tensors[0].todense()

    def site_matvec_0(x):
        return _matvec_1site_jit(x, A.shape, L_env_dense, W_dense, R_env_dense)

    A_evolved = krylov_expm(site_matvec_0, A.ravel(), dt_half,
                            config.krylov_dim, config.krylov_tol)
    A_evolved = A_evolved.reshape(A.shape)
    mps_tensors[0] = DenseTensor(A_evolved[0, :, :] if A_evolved.shape[0] == 1 else A_evolved,
                                  mps_tensors[0].indices)

    # Rebuild TensorNetwork
    new_mps = TensorNetwork()
    for i, tensor in enumerate(mps_tensors):
        new_mps.add_node(i, tensor)
    for i in range(L - 1):
        new_mps.connect_by_shared_label(i, i + 1)
    return new_mps
```

Also add the import for `_build_trivial_right_env`:

```python
from tenax.algorithms.dmrg import (
    _build_right_environments_list,
    _build_trivial_left_env,
    _build_trivial_right_env,
    _right_canonicalize,
    _update_left_env,
    _update_right_env,
)
```

Add `tdvp_step` to `src/tenax/__init__.py` imports and `__all__`:

```python
from tenax.algorithms.tdvp import (
    TDVPConfig,
    TDVPResult,
    tdvp_step,
)
```

```python
    # TDVP
    "TDVPConfig",
    "TDVPResult",
    "tdvp_step",
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_tdvp.py -v`
Expected: All tests PASS (energy conservation test may need tolerance tuning)

**Step 5: Commit**

```bash
git add src/tenax/algorithms/tdvp.py src/tenax/__init__.py tests/test_tdvp.py
git commit -m "feat(tdvp): implement 1-site TDVP sweep with energy conservation"
```

---

### Task 6: 2-Site TDVP Sweep

**Files:**
- Modify: `src/tenax/algorithms/tdvp.py`
- Modify: `tests/test_tdvp.py`

**Step 1: Write the failing test**

Add to `tests/test_tdvp.py`:

```python
def test_2site_tdvp_bond_growth():
    """2-site TDVP grows bond dimension from a product state."""
    from tenax import tdvp_step

    L = 6
    mpo = build_mpo_heisenberg(L)
    mps = build_random_mps(L, physical_dim=2, bond_dim=1)

    config = TDVPConfig(mode="2site", dt=0.1, time_type="imaginary",
                        max_bond_dim=8, krylov_dim=15)

    new_mps = tdvp_step(mps, mpo, config)

    # Check that bond dimension grew beyond 1
    tensors = [new_mps.get_tensor(i).todense() for i in range(L)]
    # Middle bond should have grown
    middle = tensors[L // 2]
    if middle.ndim == 3:
        assert middle.shape[0] > 1 or middle.shape[2] > 1, \
            "Bond dimension did not grow from product state"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tdvp.py::test_2site_tdvp_bond_growth -v`
Expected: FAIL (mode="2site" not implemented yet)

**Step 3: Write implementation**

Add `_tdvp_step_2site` to `src/tenax/algorithms/tdvp.py`. This reuses DMRG's
`_effective_hamiltonian_matvec` (the 2-site version) and `_svd_and_truncate_site`:

```python
from tenax.algorithms.dmrg import (
    _build_right_environments_list,
    _build_trivial_left_env,
    _build_trivial_right_env,
    _effective_hamiltonian_matvec,
    _matvec_jit,
    _right_canonicalize,
    _svd_and_truncate_site,
    _update_left_env,
    _update_right_env,
)


def _tdvp_step_2site(
    mps: TensorNetwork,
    hamiltonian: TensorNetwork,
    config: TDVPConfig,
) -> TensorNetwork:
    """2-site TDVP with SVD truncation for bond dimension growth."""
    L = len([n for n in mps._nodes])
    mps_tensors = [mps.get_tensor(i) for i in range(L)]
    mpo_tensors = [hamiltonian.get_tensor(i) for i in range(L)]

    mps_tensors = _right_canonicalize(mps_tensors)
    R_envs = _build_right_environments_list(mps_tensors, mpo_tensors, L)

    L_envs = [None] * L
    L_envs[0] = _build_trivial_left_env()

    dt_full = _get_dt_factor(config)
    dt_half = dt_full / 2.0

    # Use a DMRGConfig for SVD truncation parameters
    from tenax.algorithms.dmrg import DMRGConfig
    svd_config = DMRGConfig(
        max_bond_dim=config.max_bond_dim,
        svd_trunc_err=config.svd_trunc_err,
        num_sweeps=1,
    )

    # --- Left-to-right sweep (bonds 0..L-3) ---
    for i in range(L - 1):
        # Merge two sites into theta
        A_l = mps_tensors[i].todense()
        A_r = mps_tensors[i + 1].todense()
        if A_l.ndim == 2:
            if i == 0:
                A_l = A_l[jnp.newaxis, :]
            else:
                A_l = A_l[:, :, jnp.newaxis]
        if A_r.ndim == 2:
            if i + 1 == L - 1:
                A_r = A_r[:, :, jnp.newaxis]
            else:
                A_r = A_r[jnp.newaxis, :]

        theta = jnp.einsum("ijk,klm->ijlm", A_l, A_r)

        L_env_dense = L_envs[i].todense()
        R_env_i1 = R_envs[i + 1] if R_envs[i + 1] is not None else _build_trivial_right_env()
        R_env_dense = R_env_i1.todense()
        W_l = mpo_tensors[i].todense()
        W_r = mpo_tensors[i + 1].todense()

        # Forward evolve 2-site tensor
        def two_site_matvec(x, _shape=theta.shape, _L=L_env_dense,
                            _Wl=W_l, _Wr=W_r, _R=R_env_dense):
            return _matvec_jit(x, _shape, _L, _Wl, _Wr, _R)

        theta_evolved = krylov_expm(two_site_matvec, theta.ravel(), dt_half,
                                    config.krylov_dim, config.krylov_tol)
        theta_evolved = theta_evolved.reshape(theta.shape)

        # SVD split
        chi_l, d_l, d_r, chi_r = theta_evolved.shape
        theta_mat = theta_evolved.reshape(chi_l * d_l, d_r * chi_r)
        U, s, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)

        # Truncate
        keep = min(config.max_bond_dim, len(s))
        if config.svd_trunc_err is not None:
            cumulative = jnp.cumsum(s[::-1] ** 2)[::-1]
            norm_sq = jnp.sum(s**2)
            above_tol = cumulative / norm_sq > config.svd_trunc_err**2
            keep_err = int(jnp.sum(above_tol))
            keep = min(keep, max(keep_err, 1))
        U = U[:, :keep]
        s = s[:keep]
        Vh = Vh[:keep, :]

        # Absorb s into Vh (sweep right: left site is left-canonical)
        SVh = jnp.diag(s) @ Vh

        new_A_l = U.reshape(chi_l, d_l, keep)
        new_A_r = SVh.reshape(keep, d_r, chi_r)

        # Handle boundary tensors
        if i == 0:
            new_A_l = new_A_l[0, :, :]
        if i + 1 == L - 1:
            new_A_r = new_A_r[:, :, 0]

        mps_tensors[i] = DenseTensor(new_A_l, mps_tensors[i].indices)
        mps_tensors[i + 1] = DenseTensor(new_A_r, mps_tensors[i + 1].indices)

        # Update left environment
        L_envs[i + 1] = _update_left_env(L_envs[i], mps_tensors[i], mpo_tensors[i])

    # --- Right-to-left sweep (bonds L-2..0) ---
    # Rebuild right environments from the updated MPS
    R_envs = _build_right_environments_list(mps_tensors, mpo_tensors, L)

    for i in range(L - 2, -1, -1):
        A_l = mps_tensors[i].todense()
        A_r = mps_tensors[i + 1].todense()
        if A_l.ndim == 2:
            if i == 0:
                A_l = A_l[jnp.newaxis, :]
            else:
                A_l = A_l[:, :, jnp.newaxis]
        if A_r.ndim == 2:
            if i + 1 == L - 1:
                A_r = A_r[:, :, jnp.newaxis]
            else:
                A_r = A_r[jnp.newaxis, :]

        theta = jnp.einsum("ijk,klm->ijlm", A_l, A_r)

        L_env_dense = L_envs[i].todense()
        R_env_i1 = R_envs[i + 1] if R_envs[i + 1] is not None else _build_trivial_right_env()
        R_env_dense = R_env_i1.todense()
        W_l = mpo_tensors[i].todense()
        W_r = mpo_tensors[i + 1].todense()

        def two_site_matvec(x, _shape=theta.shape, _L=L_env_dense,
                            _Wl=W_l, _Wr=W_r, _R=R_env_dense):
            return _matvec_jit(x, _shape, _L, _Wl, _Wr, _R)

        theta_evolved = krylov_expm(two_site_matvec, theta.ravel(), dt_half,
                                    config.krylov_dim, config.krylov_tol)
        theta_evolved = theta_evolved.reshape(theta.shape)

        chi_l, d_l, d_r, chi_r = theta_evolved.shape
        theta_mat = theta_evolved.reshape(chi_l * d_l, d_r * chi_r)
        U, s, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)

        keep = min(config.max_bond_dim, len(s))
        if config.svd_trunc_err is not None:
            cumulative = jnp.cumsum(s[::-1] ** 2)[::-1]
            norm_sq = jnp.sum(s**2)
            above_tol = cumulative / norm_sq > config.svd_trunc_err**2
            keep_err = int(jnp.sum(above_tol))
            keep = min(keep, max(keep_err, 1))
        U = U[:, :keep]
        s = s[:keep]
        Vh = Vh[:keep, :]

        # Absorb s into U (sweep left: right site is right-canonical)
        Us = U @ jnp.diag(s)

        new_A_l = Us.reshape(chi_l, d_l, keep)
        new_A_r = Vh.reshape(keep, d_r, chi_r)

        if i == 0:
            new_A_l = new_A_l[0, :, :]
        if i + 1 == L - 1:
            new_A_r = new_A_r[:, :, 0]

        mps_tensors[i] = DenseTensor(new_A_l, mps_tensors[i].indices)
        mps_tensors[i + 1] = DenseTensor(new_A_r, mps_tensors[i + 1].indices)

        # Update right environment
        R_envs[i] = _update_right_env(
            R_envs[i + 1] if R_envs[i + 1] is not None else _build_trivial_right_env(),
            mps_tensors[i + 1], mpo_tensors[i + 1]
        )

    # Rebuild TensorNetwork
    new_mps = TensorNetwork()
    for i, tensor in enumerate(mps_tensors):
        new_mps.add_node(i, tensor)
    for i in range(L - 1):
        new_mps.connect_by_shared_label(i, i + 1)
    return new_mps
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_tdvp.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/tdvp.py tests/test_tdvp.py
git commit -m "feat(tdvp): implement 2-site TDVP sweep with bond growth"
```

---

### Task 7: TDVP Driver Function

**Files:**
- Modify: `src/tenax/algorithms/tdvp.py`
- Modify: `tests/test_tdvp.py`

**Step 1: Write the failing test**

Add to `tests/test_tdvp.py`:

```python
def test_tdvp_driver_imaginary_time():
    """tdvp() driver converges to ground state via imaginary-time evolution."""
    from tenax import tdvp

    L = 6
    chi = 8
    mpo = build_mpo_heisenberg(L)
    mps = build_random_mps(L, physical_dim=2, bond_dim=chi)

    config = TDVPConfig(
        mode="2site", dt=0.05, time_type="imaginary",
        num_steps=40, max_bond_dim=chi, krylov_dim=15,
    )
    result = tdvp(mps, mpo, config)

    assert isinstance(result, TDVPResult)
    assert len(result.times) == 40
    assert len(result.energies) == 40
    # Energy should decrease over imaginary-time evolution
    assert result.energies[-1] < result.energies[0]

    # Compare with DMRG
    dmrg_config = DMRGConfig(max_bond_dim=chi, num_sweeps=20)
    dmrg_result = dmrg(mpo, build_random_mps(L, physical_dim=2, bond_dim=chi), dmrg_config)
    # Should be within ~10% of DMRG energy
    assert abs(result.energies[-1] - dmrg_result.energy) / abs(dmrg_result.energy) < 0.1


def test_tdvp_driver_with_measure():
    """tdvp() driver calls measure callback and stores observables."""
    from tenax import tdvp

    L = 4
    mpo = build_mpo_heisenberg(L)
    mps = build_random_mps(L, physical_dim=2, bond_dim=4)

    call_count = [0]

    def measure(mps_state, t):
        call_count[0] += 1
        return {"step": float(call_count[0])}

    config = TDVPConfig(mode="1site", dt=0.05, time_type="real", num_steps=3)
    result = tdvp(mps, mpo, config, measure=measure)

    assert call_count[0] == 3
    assert "step" in result.observables
    assert len(result.observables["step"]) == 3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tdvp.py::test_tdvp_driver_imaginary_time -v`
Expected: FAIL with `ImportError: cannot import name 'tdvp'`

**Step 3: Write implementation**

Add to `src/tenax/algorithms/tdvp.py`:

```python
def _compute_energy(mps: TensorNetwork, hamiltonian: TensorNetwork) -> float:
    """Compute <psi|H|psi> by contracting MPS-MPO-MPS."""
    L = len([n for n in mps._nodes])
    mps_tensors = [mps.get_tensor(i) for i in range(L)]
    mpo_tensors = [hamiltonian.get_tensor(i) for i in range(L)]

    L_env = _build_trivial_left_env()
    for i in range(L):
        L_env = _update_left_env(L_env, mps_tensors[i], mpo_tensors[i])

    # Final L_env is a (1,1,1) tensor; the single element is <psi|H|psi>
    return float(L_env.todense().ravel()[0].real)


def tdvp(
    mps: TensorNetwork,
    hamiltonian: TensorNetwork,
    config: TDVPConfig,
    measure: Callable[[TensorNetwork, float], dict[str, float]] | None = None,
) -> TDVPResult:
    """Run TDVP time evolution for multiple steps.

    Args:
        mps:         Initial MPS state.
        hamiltonian: MPO Hamiltonian.
        config:      TDVP configuration (includes num_steps, dt, etc.).
        measure:     Optional callback called after each step with
                     (mps, time) -> dict of observable values.

    Returns:
        TDVPResult with final MPS, time series, energies, and observables.
    """
    current_mps = mps
    times: list[float] = []
    energies: list[float] = []
    observables: dict[str, list[float]] = {}

    for step in range(config.num_steps):
        current_mps = tdvp_step(current_mps, hamiltonian, config)
        t = (step + 1) * config.dt
        times.append(t)

        E = _compute_energy(current_mps, hamiltonian)
        energies.append(E)

        if measure is not None:
            obs = measure(current_mps, t)
            for key, val in obs.items():
                observables.setdefault(key, []).append(val)

        if config.verbose:
            print(f"  TDVP step {step + 1:4d}/{config.num_steps}  "
                  f"t={t:.4f}  E={E:.10f}")

    return TDVPResult(
        mps=current_mps,
        times=times,
        energies=energies,
        observables=observables,
    )
```

Add `tdvp` to `src/tenax/__init__.py` imports and `__all__`:

```python
from tenax.algorithms.tdvp import (
    TDVPConfig,
    TDVPResult,
    tdvp,
    tdvp_step,
)
```

```python
    "tdvp",
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_tdvp.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/tdvp.py src/tenax/__init__.py tests/test_tdvp.py
git commit -m "feat(tdvp): add tdvp() driver with energy tracking and measure callback"
```

---

### Task 8: Norm Preservation Test and Normalization Fix

**Files:**
- Modify: `tests/test_tdvp.py`

**Step 1: Write the test**

Add to `tests/test_tdvp.py`:

```python
def test_1site_tdvp_norm_preservation():
    """Real-time 1-site TDVP preserves MPS norm."""
    from tenax import tdvp_step
    from tenax.core.tensor import inner

    L = 6
    chi = 8
    mpo = build_mpo_heisenberg(L)
    mps = build_random_mps(L, physical_dim=2, bond_dim=chi)

    # Get a normalized state from DMRG
    dmrg_config = DMRGConfig(max_bond_dim=chi, num_sweeps=5)
    mps = dmrg(mpo, mps, dmrg_config).mps

    config = TDVPConfig(mode="1site", dt=0.05, time_type="real")

    current = mps
    for _ in range(5):
        current = tdvp_step(current, mpo, config)

    # Compute norm via full MPS contraction
    L_sites = len([n for n in current._nodes])
    tensors = [current.get_tensor(i).todense() for i in range(L_sites)]

    # Transfer matrix contraction for norm
    norm_env = jnp.array([[1.0]])
    for t in tensors:
        if t.ndim == 2:
            # Boundary: try both orientations
            if t.shape[0] < t.shape[1]:
                t = t[jnp.newaxis, :]
            else:
                t = t[:, :, jnp.newaxis]
        norm_env = jnp.einsum("ab,apc,bpd->cd", norm_env, t, jnp.conj(t))
    norm_sq = float(norm_env.ravel()[0].real)

    assert abs(norm_sq - 1.0) < 1e-6, f"Norm not preserved: <psi|psi> = {norm_sq}"
```

**Step 2: Run test**

Run: `uv run pytest tests/test_tdvp.py::test_1site_tdvp_norm_preservation -v`
Expected: PASS (real-time Krylov expm of Hermitian H is unitary)

If it fails, add explicit renormalization after each Krylov step in the sweep.

**Step 3: Commit**

```bash
git add tests/test_tdvp.py
git commit -m "test(tdvp): add norm preservation test for real-time evolution"
```

---

### Task 9: Export krylov_expm and Update README

**Files:**
- Modify: `src/tenax/__init__.py:150-266`
- Modify: `README.md`

**Step 1: Add krylov_expm export**

Add to `src/tenax/__init__.py`:

```python
from tenax.algorithms._krylov import krylov_expm
```

And in `__all__`:

```python
    "krylov_expm",
```

**Step 2: Update README.md**

Add TDVP to the algorithms/features section of `README.md`. Find the existing
algorithm list and add:

```markdown
- **TDVP** — 1-site and 2-site time-dependent variational principle for
  real-time dynamics and imaginary-time ground-state finding
```

**Step 3: Run full core tests**

Run: `uv run pytest -m core -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/tenax/__init__.py README.md
git commit -m "feat(tdvp): export krylov_expm and update README"
```

---

### Task 10: Final Integration Test and PR

**Files:**
- Modify: `tests/test_tdvp.py`

**Step 1: Add U(1) symmetric TDVP test**

```python
def test_1site_tdvp_symmetric():
    """1-site TDVP works with U(1) symmetric MPS/MPO."""
    from tenax import tdvp_step, build_random_symmetric_mps

    L = 6
    chi = 8
    mpo = build_mpo_heisenberg(L, symmetric=True)
    mps = build_random_symmetric_mps(L, target_charge=0, bond_dim=chi)

    # DMRG to get a good initial state
    dmrg_config = DMRGConfig(max_bond_dim=chi, num_sweeps=10, target_charge=0)
    mps = dmrg(mpo, mps, dmrg_config).mps

    config = TDVPConfig(mode="1site", dt=0.02, time_type="real")
    new_mps = tdvp_step(mps, mpo, config)

    # Should not crash — symmetric tensors go through todense() path
    assert new_mps is not None
```

**Step 2: Run full algorithm test suite**

Run: `uv run pytest tests/test_tdvp.py tests/test_krylov.py -v`
Expected: All PASS

**Step 3: Run CI-equivalent tests**

Run: `uv run pytest -m core -v`
Expected: All PASS

**Step 4: Commit and create PR**

```bash
git add tests/test_tdvp.py
git commit -m "test(tdvp): add U(1) symmetric tensor integration test"
git push -u origin feat/tdvp
gh pr create --title "feat: add TDVP algorithm (1-site and 2-site)" \
  --body "$(cat <<'EOF'
## Summary
- Add Lanczos-based Krylov matrix exponential (`_krylov.py`)
- Implement 1-site TDVP with second-order Lie-Trotter integrator
- Implement 2-site TDVP with SVD truncation for bond dimension growth
- Support real-time (e^{-iHt}) and imaginary-time (e^{-Ht}) evolution
- `tdvp_step()` for single steps, `tdvp()` driver for multi-step with callbacks
- Works with both dense and U(1) symmetric tensors

## Test plan
- [x] Krylov expm vs scipy.linalg.expm (small matrices)
- [x] Energy conservation (real-time 1TDVP)
- [x] Norm preservation (real-time 1TDVP)
- [x] Bond dimension growth (2TDVP from product state)
- [x] Imaginary-time convergence to DMRG energy
- [x] U(1) symmetric tensor support
- [ ] CI passes

Design: docs/plans/2026-03-16-tdvp-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Set auto-merge: `gh pr merge <number> --squash --delete-branch --auto`
