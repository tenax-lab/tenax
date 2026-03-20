---
name: tenax-migration-itensor
description: >
  Help users migrate tensor network code from ITensor (Julia/C++) to Tenax.
  Maps ITensor concepts (Index, ITensor, MPS, MPO, AutoMPO, siteinds) to
  Tenax equivalents. Use this skill when the user mentions ITensor, "coming
  from ITensor", "ITensor equivalent", "Julia tensor networks", or asks how
  Tenax compares to ITensor. Also trigger for "migrate from ITensor" or
  "I used to use ITensor".
---

# Migrating from ITensor to Tenax

Help ITensor users translate their mental model and code to Tenax. Both
libraries share key design ideas (label-based contraction, AutoMPO, built-in
DMRG), but differ in language (Julia/C++ vs Python/JAX) and tensor design.

## Quick Reference

| ITensor (Julia) | Tenax (Python) | Notes |
|-----------------|----------------|-------|
| `Index(dim, "label")` | `TensorIndex(sym, charges, flow, label)` | Tenax carries symmetry + flow |
| `ITensor(idx1, idx2)` | `DenseTensor(data, indices)` | Tenax requires explicit data |
| `randomITensor(idx1, idx2)` | `DenseTensor.random_normal(indices, key)` | JAX needs explicit RNG key |
| `Index(dim, "l"; tags="Link")` | `TensorIndex(..., label="l")` | Tenax uses labels, not tags |
| `dag(idx)` | `idx.dual()` | Flip FlowDirection |
| `A * B` | `contract(A, B)` | Both label-based |
| `svd(T, i1, i2)` | `truncated_svd(T, left_labels, right_labels)` | By labels, not Index objects |
| `qr(T, i1, i2)` | `qr_decompose(T, left_labels, right_labels)` | Same pattern |
| `AutoMPO()` | `AutoMPO(L, d)` | Very similar API |
| `dmrg(H, psi0, sweeps)` | `dmrg(mpo, mps, config)` | Config replaces Sweeps object |
| `siteinds("S=1/2", N)` | `FiniteMPS.random(L, d=2, chi, key)` | No site-type system in Tenax |
| `expect(psi, "Sz")` | Manual contraction (see observables skill) | No built-in expect function |

---

## Key Design Differences

### 1. No Tag System

**ITensor:** Indices carry tags ("Link", "Site", "l=3") for flexible matching
and filtering.

```julia
s = siteinds("S=1/2", 10)    # Tagged site indices
l = Index(4, "Link,l=3")     # Tagged link index
```

**Tenax:** Indices carry a single string label. Use descriptive labels instead
of tags:

```python
from tenax import TensorIndex, FlowDirection
import numpy as np

phys = TensorIndex(None, np.array([], dtype=np.int32), FlowDirection.IN, label="s3")
bond = TensorIndex(None, np.array([], dtype=np.int32), FlowDirection.OUT, label="l3")
```

### 2. Explicit Data, Explicit RNG

**ITensor:** Tensors can be created empty or with implicit random init.

```julia
A = randomITensor(i, j, k)   # Random tensor, global RNG
B = ITensor(i, j)             # Zero tensor
```

**Tenax:** JAX requires explicit data and RNG keys (no global state):

```python
import jax
A = DenseTensor.random_normal(indices=(i, j, k), key=jax.random.PRNGKey(0))
```

### 3. FlowDirection (Arrows)

**ITensor:** Indices can have arrows (QN mode) or not (dense mode). Arrows
are implicit in most operations.

**Tenax:** Every `TensorIndex` has an explicit `FlowDirection` (IN or OUT).
Contracted legs must form IN↔OUT pairs for `SymmetricTensor`.

### 4. AutoMPO: Nearly Identical

Both libraries have AutoMPO with very similar syntax:

**ITensor:**
```julia
ampo = AutoMPO()
for i in 1:N-1
    ampo += (1.0, "Sz", i, "Sz", i+1)
    ampo += (0.5, "S+", i, "S-", i+1)
    ampo += (0.5, "S-", i, "S+", i+1)
end
H = MPO(ampo, sites)
```

**Tenax:**
```python
from tenax import AutoMPO

auto = AutoMPO(L=N, d=2)
for i in range(N - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)  # "Sp"/"Sm" not "S+"/"S-"
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()
```

Key differences:
- Tenax uses `"Sp"` / `"Sm"` (not `"S+"` / `"S-"`)
- Tenax uses 0-based indexing
- Tenax's `AutoMPO` takes `L` and `d` upfront (no separate sites object)
- `to_mpo(symmetric=True)` for U(1) block-sparse MPO

### 5. DMRG Configuration

**ITensor:**
```julia
sweeps = Sweeps(10)
setmaxdim!(sweeps, 10, 20, 50, 100)
setnoise!(sweeps, 1e-5, 1e-6, 0.0)
energy, psi = dmrg(H, psi0, sweeps)
```

**Tenax:**
```python
from tenax import DMRGConfig, dmrg

config = DMRGConfig(
    max_bond_dim=100,
    num_sweeps=10,
    noise=0.0,
    verbose=True,
)
result = dmrg(mpo, mps, config)
# result.energy, result.mps, result.converged
```

Tenax uses a single `max_bond_dim` (no per-sweep schedule). For progressive
bond dimension growth, run DMRG multiple times with increasing `max_bond_dim`.

### 6. JAX Backend → Autodiff + JIT

**ITensor:** No automatic differentiation. Optimizations are algorithmic
(DMRG sweeps, TEBD gates).

**Tenax:** Pure JAX — everything is differentiable and JIT-compilable.
This enables AD-based iPEPS optimization (`optimize_gs_ad`) which has no
direct ITensor equivalent.

---

## Code Translation: Complete DMRG Example

**ITensor (Julia):**
```julia
using ITensors

N = 20
sites = siteinds("S=1/2", N)

ampo = AutoMPO()
for j in 1:N-1
    ampo += ("Sz", j, "Sz", j+1)
    ampo += (0.5, "S+", j, "S-", j+1)
    ampo += (0.5, "S-", j, "S+", j+1)
end
H = MPO(ampo, sites)

psi0 = randomMPS(sites, linkdims=10)
sweeps = Sweeps(10)
setmaxdim!(sweeps, 10, 20, 50, 100)

energy, psi = dmrg(H, psi0, sweeps)
println("Energy: $energy")
```

**Tenax (Python):**
```python
import jax
from tenax import AutoMPO, DMRGConfig, FiniteMPS, dmrg

L = 20
auto = AutoMPO(L=L, d=2)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()

key = jax.random.PRNGKey(0)
mps = FiniteMPS.random(L=L, d=2, chi=10, key=key)
config = DMRGConfig(max_bond_dim=100, num_sweeps=10, verbose=True)
result = dmrg(mpo, mps, config)
print(f"Energy: {result.energy:.10f}")
```

---

## Migration Checklist

1. **Replace `Index` with `TensorIndex`** — add symmetry, FlowDirection, label.
2. **Replace `ITensor` with `DenseTensor`/`SymmetricTensor`** — provide data explicitly.
3. **Replace `*` contraction with `contract()`** — same label-based semantics.
4. **Replace `svd`/`qr` with explicit label lists** — `left_labels` / `right_labels`.
5. **Translate AutoMPO** — `"S+"` → `"Sp"`, `"S-"` → `"Sm"`, 1-based → 0-based indexing.
6. **Replace `Sweeps` with `DMRGConfig`** — single dataclass instead of per-sweep settings.
7. **Add JAX RNG keys** — `jax.random.PRNGKey(seed)` for all random operations.
8. **No `expect()`** — compute observables manually via tensor contraction.

## What You Gain

- **Python ecosystem** — NumPy, SciPy, matplotlib, Jupyter integration
- **Autodiff** — `jax.grad` through any contraction
- **JIT compilation** — `jax.jit` for automatic optimization
- **GPU/TPU** — same code on CPU, CUDA, TPU, Metal
- **iPEPS + excitations** — built-in 2D algorithms beyond DMRG

## What You Lose

- **Tag system** — replaced by simple string labels
- **Per-sweep bond dimension schedule** — use multiple DMRG runs instead
- **`expect()` / `correlation_matrix()`** — no built-in observable functions
- **TEBD / TDVP** — not yet implemented in Tenax
- **Non-Abelian symmetry** — Tenax currently supports only Abelian (U(1), Z_n)
- **MPS/MPO as first-class types** — Tenax uses `TensorNetwork` (generic graph)
