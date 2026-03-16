# TDVP Algorithm Design

**Date**: 2026-03-16
**Status**: Approved

## Overview

Implement the Time-Dependent Variational Principle (TDVP) for MPS time
evolution in Tenax. Supports both real-time dynamics (`e^{-iHt}`) and
imaginary-time ground-state finding (`e^{-Hτ}`), with 1-site and 2-site
variants.

## Scope

- 1-site TDVP (fixed bond dimension, second-order Lie-Trotter integrator)
- 2-site TDVP (adaptive bond dimension via SVD truncation)
- Real-time and imaginary-time evolution
- Dense and symmetric (U(1), Z_n) tensor support from day one
- Lanczos-based Krylov matrix exponential (no dense expm)

## File Layout

```
src/tenax/algorithms/
  _krylov.py     # Lanczos-based matrix exponential (reusable)
  tdvp.py        # TDVP config, step, driver, result types
```

## Public API

```python
@dataclass
class TDVPConfig:
    mode: Literal["1site", "2site"] = "1site"
    dt: float = 0.05
    time_type: Literal["real", "imaginary"] = "real"
    num_steps: int = 100
    max_bond_dim: int = 64              # 2-site only
    svd_trunc_err: float | None = None  # 2-site only
    krylov_dim: int = 20
    krylov_tol: float = 1e-12
    verbose: bool = False

@dataclass
class TDVPResult:
    mps: TensorNetwork
    times: list[float]
    energies: list[float]
    observables: dict[str, list[float]]

def tdvp_step(
    mps: TensorNetwork,
    hamiltonian: TensorNetwork,
    config: TDVPConfig,
) -> TensorNetwork

def tdvp(
    mps: TensorNetwork,
    hamiltonian: TensorNetwork,
    config: TDVPConfig,
    measure: Callable[[TensorNetwork, float], dict[str, float]] | None = None,
) -> TDVPResult
```

## Krylov Matrix Exponential

**File**: `_krylov.py`

```python
def krylov_expm(
    matvec: Callable[[jax.Array], jax.Array],
    v: jax.Array,
    dt: complex,
    krylov_dim: int = 20,
    tol: float = 1e-12,
) -> jax.Array
```

Algorithm:
1. Lanczos iteration: build orthonormal basis `V` and tridiagonal `T`
   (up to `krylov_dim` steps, early termination if residual < `tol`)
2. Diagonalize `T` via `jnp.linalg.eigh(T)`
3. Exponentiate: `exp(dt * T) = P @ diag(exp(dt * λ)) @ P^†`
4. Map back: `result = ||v|| * V @ exp(dt * T) @ e_1`

The `dt` parameter is complex-valued to handle both real-time (`-i*δt`)
and imaginary-time (`-δt`) in one code path. Uses `eigh` on the small
tridiagonal matrix rather than dense `expm`.

Also provides a Tensor-aware wrapper `krylov_expm_tensor()` that handles
flatten/unflatten for the polymorphic Tensor interface.

## TDVP Sweep Logic

### 1-site TDVP

Second-order Lie-Trotter integrator. One full time step `dt`:

**Left-to-right** (sites 0 to L-2):
1. Build effective Hamiltonian `H_eff` from `L_env[i]`, `W[i]`, `R_env[i]`
2. Forward evolve: `A[i] ← exp(dt/2 · H_eff) · A[i]`
3. QR decompose: `A[i] = Q · R`, set `A[i] = Q` (left-canonical)
4. Build bond effective Hamiltonian `H_bond` from `L_env[i+1]`, `R_env[i]`
5. Back-evolve: `R ← exp(-dt/2 · H_bond) · R`
6. Absorb R into `A[i+1]`
7. Update `L_env[i+1]`

**Right-to-left** (sites L-1 to 1):
1. Build `H_eff`, forward evolve `A[i] ← exp(dt/2 · H_eff) · A[i]`
2. RQ decompose: `A[i] = L · Q`, set `A[i] = Q` (right-canonical)
3. Back-evolve: `L ← exp(-dt/2 · H_bond) · L`
4. Absorb L into `A[i-1]`
5. Update `R_env[i-1]`

The bond back-evolution corrects the Lie-Trotter splitting error, making
the integrator second-order accurate.

### 2-site TDVP

Same sweep structure but evolves a merged 2-site tensor `θ[i,i+1]`:
1. Merge `A[i]` and `A[i+1]` into `θ`
2. Forward evolve: `θ ← exp(dt/2 · H_eff_2site) · θ`
3. SVD truncate: `θ = U · S · V†`, keeping up to `max_bond_dim` values
4. Set `A[i] = U`, absorb `S` into `V†`, set `A[i+1] = S · V†`

No bond back-evolution needed — the SVD handles the bond directly.
Allows bond dimension growth from product states.

### Environment Management

- Pre-build all `R_env` at start (right-canonicalize MPS first)
- Update `L_env` incrementally during left-to-right sweep
- Update `R_env` incrementally during right-to-left sweep
- Reuse DMRG's `_update_left_env` / `_update_right_env` by importing
  from `dmrg.py`

## Integration

**Exports**: Add `TDVPConfig`, `TDVPResult`, `tdvp`, `tdvp_step`,
`krylov_expm` to `src/tenax/__init__.py` and `__all__`.

**Documentation**: Update `README.md` features list. Add example script
`examples/heisenberg_quench_tdvp.py` (domain wall quench dynamics).

## Testing

Tests in `tests/test_tdvp.py`:

1. **Krylov expm correctness** — compare against `scipy.linalg.expm`
   for small random matrices
2. **Energy conservation (real-time)** — Heisenberg chain, verify
   `<H>` constant to Krylov precision
3. **Imaginary-time ground state** — converge to DMRG energy
4. **2-site bond growth** — product state → verify bond dim grows
5. **1-site vs 2-site consistency** — same evolution from converged state
6. **Symmetric tensor support** — U(1) symmetric energy conservation
7. **Norm preservation** — `<ψ|ψ> = 1` after multiple real-time steps

Quick tests (small L, small chi) auto-marked `core` by filename
convention. Convergence tests marked `algorithm`.
