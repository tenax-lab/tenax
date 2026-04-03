# DMRG

The Density Matrix Renormalization Group (DMRG) finds the ground state of a 1D
quantum Hamiltonian given as a Matrix Product Operator (MPO).

## Background

DMRG variationally optimises a Matrix Product State (MPS) by sweeping through
the chain and updating one or two site tensors at a time. At each step it
solves a local eigenvalue problem (Lanczos) for the effective Hamiltonian
projected into the MPS subspace, then truncates via SVD to control the bond
dimension.

Key properties of the Tenax implementation:

- **2-site and 1-site** variants (`DMRGConfig.two_site`).
- **Cython BLAS fast path** — fused Lanczos solver and block-sparse
  contractions via direct BLAS calls with zero Python reentry. When Cython
  extensions are available, the entire Lanczos solve runs in a single Cython
  call, eliminating Python loop overhead.
- **GPU/TPU acceleration** — `DMRGConfig(accelerator="jit")` for JIT-compiled
  sweeps on GPU/TPU; `accelerator="sharded"` for multi-GPU runs via GSPMD.
- Outer sweep loop is a Python for-loop (bond dimensions change dynamically).
- The effective Hamiltonian matvec is `@jax.jit` compiled.

## Configuration

```python
from tenax import DMRGConfig

config = DMRGConfig(
    max_bond_dim=64,       # maximum MPS bond dimension
    num_sweeps=30,         # full left-right sweep cycles
    convergence_tol=1e-10, # stop early when |dE| < tol
    two_site=True,         # 2-site DMRG (allows bond growth)
    lanczos_max_iter=50,   # Lanczos iteration cap
    lanczos_tol=1e-12,     # Lanczos convergence tolerance
    noise=0.0,             # density-matrix perturbation
    verbose=True,
)
```

## Example -- Heisenberg chain

```python
import jax
from tenax import (
    DMRGConfig, dmrg,
    build_mpo_heisenberg, FiniteMPS,
)

L = 20
mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0, hz=0.0)
mps = FiniteMPS.random(L=L, d=2, chi=4, key=jax.random.PRNGKey(0))

config = DMRGConfig(max_bond_dim=64, num_sweeps=30, verbose=True)
result = dmrg(mpo, mps, config)

print(f"Energy:    {result.energy:.10f}")
print(f"Converged: {result.converged}")
print(f"Sweeps:    {len(result.energies_per_sweep)}")
```

## Result object

`dmrg()` returns a `DMRGResult` named tuple:

| Field | Type | Description |
|-------|------|-------------|
| `energy` | `float` | Final ground-state energy |
| `energies_per_sweep` | `list[float]` | Energy at the end of each sweep |
| `mps` | `TensorNetwork` | Optimised MPS |
| `truncation_errors` | `list[float]` | Truncation error at each bond update |
| `converged` | `bool` | Whether energy converged within tolerance |

## MPO construction

`build_mpo_heisenberg` builds the standard 5-state MPO for the XXZ
Heisenberg model:

$$H = J_z \sum_i S^z_i S^z_{i+1} + \frac{J_{xy}}{2} \sum_i (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) + h_z \sum_i S^z_i$$

For custom Hamiltonians, see the {doc}`auto_mpo` tutorial.

## Example -- 2D Heisenberg cylinder

For 2D systems, map the lattice to a 1D chain (column-major ordering) and
use `AutoMPO` to build the long-range MPO:

```python
import jax
from tenax import AutoMPO, DMRGConfig, FiniteMPS, dmrg

Lx, Ly, N = 8, 4, 32
auto = AutoMPO(L=N, d=2)
for x in range(Lx):
    for y in range(Ly):
        # Within-ring bond (periodic y-direction)
        i, j = x * Ly + y, x * Ly + (y + 1) % Ly
        auto += (1.0, "Sz", min(i,j), "Sz", max(i,j))
        auto += (0.5, "Sp", min(i,j), "Sm", max(i,j))
        auto += (0.5, "Sm", min(i,j), "Sp", max(i,j))
        # Between-ring bond (open x-direction)
        if x < Lx - 1:
            i, j = x * Ly + y, (x + 1) * Ly + y
            auto += (1.0, "Sz", i, "Sz", j)
            auto += (0.5, "Sp", i, "Sm", j)
            auto += (0.5, "Sm", i, "Sp", j)

mpo = auto.to_mpo(compress=True)
mps = FiniteMPS.random(L=N, d=2, chi=16, key=jax.random.PRNGKey(0))
config = DMRGConfig(max_bond_dim=200, num_sweeps=15, verbose=True)
result = dmrg(mpo, mps, config)
print(f"E/N = {result.energy / N:.8f}")
```

See `examples/heisenberg_cylinder.py` for a complete working example with
multiple cylinder sizes and exact-diagonalisation cross-checks.

## Label conventions

MPS and MPO tensors follow these leg-label conventions:

**MPS site tensors:**

- Left virtual bond: `v{i-1}_{i}`
- Physical leg: `p{i}`
- Right virtual bond: `v{i}_{i+1}`
- All sites are 3-leg; boundary sites have trivial dimension-1 bonds
  (`v_-1_0` at site 0, `v{L-1}_{L}` at site L-1)

**MPO site tensors:**

- Left MPO bond: `w{i-1}_{i}`
- Top physical (ket): `mpo_top_{i}`
- Bottom physical (bra): `mpo_bot_{i}`
- Right MPO bond: `w{i}_{i+1}`
