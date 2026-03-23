---
name: tenax-migration-quimb
description: >
  Help users migrate tensor network code from quimb to Tenax. Maps quimb
  concepts (Tensor, TensorNetwork, DMRG, TEBD) to Tenax equivalents. Use
  this skill when the user mentions quimb, "coming from quimb", "quimb
  equivalent", or asks how Tenax compares to quimb. Also trigger for
  "migrate from quimb" or "I used to use quimb".
---

# Migrating from quimb to Tenax

Help quimb users translate their code to Tenax. Both libraries use
graph-based tensor network containers with label-based contraction, but
differ in backend (NumPy/autoray vs JAX), symmetry support, and algorithm
scope.

## Quick Reference

| quimb | Tenax | Notes |
|-------|-------|-------|
| `qtn.Tensor(data, inds, tags)` | `DenseTensor(data, indices)` | No tags; labels on TensorIndex |
| `qtn.TensorNetwork(...)` | `TensorNetwork()` | Similar graph container |
| `tn.contract()` | `tn.contract()` | Both contract full network |
| `tn ^ all` | `tn.contract()` | Tenax uses method, not operator |
| `A & B` | `contract(A, B)` | Pairwise contraction |
| `A.reindex({"old": "new"})` | `A.relabel("old", "new")` | Immutable in Tenax |
| `qtn.DMRG2(ham)` | `dmrg(mpo, mps, config)` | Functional API |
| `qtn.SpinHam1D(S=0.5)` | `AutoMPO(L, d=2)` | Similar builder pattern |
| `ham.build_mpo(L)` | `auto.to_mpo()` | Explicit L in AutoMPO constructor |
| `qtn.MPS_rand_state(L, bond_dim)` | `FiniteMPS.random(L, d, chi, key)` | Similar |
| `tensor_network.draw()` | — | No visualization in Tenax |

---

## Key Design Differences

### 1. Tags vs Labels

**quimb:** Tensors carry both `inds` (index names for contraction) and
`tags` (metadata for selection/grouping):

```python
import quimb.tensor as qtn

A = qtn.Tensor(data, inds=("k0", "k1", "b0"), tags={"MPS", "I0"})
# Select by tag:
mps_tensors = tn.select("MPS")
```

**Tenax:** No tag system. Each leg has a label (for contraction) and the
tensor itself has no metadata. Select tensors by node ID in TensorNetwork:

```python
from tenax import DenseTensor, TensorNetwork

tn = TensorNetwork()
tn.add_node("site_0", A)
tn.add_node("site_1", B)
tensor = tn.get_tensor("site_0")  # By node ID, not tags
```

### 2. Symmetry Support

**quimb:** No built-in symmetry-aware tensors. All tensors are dense.

**Tenax:** First-class `SymmetricTensor` with U(1), Z_n, and fermionic
symmetries. Block-sparse storage saves memory and compute:

```python
from tenax import U1Symmetry, SymmetricTensor

# Block-sparse MPO that exploits Sz conservation
mpo = auto.to_mpo(symmetric=True)
```

### 3. Backend

**quimb:** Uses `autoray` for backend flexibility (NumPy, TensorFlow, JAX,
PyTorch, etc.). Can opt into JAX but not designed around it.

**Tenax:** Pure JAX, all tensors are JAX pytrees. `jax.jit`, `jax.grad`,
`jax.vmap` work natively. This is a core design choice, not an optional
backend.

### 4. Hamiltonian Construction

**quimb:**
```python
builder = qtn.SpinHam1D(S=0.5)
builder += 1.0, "Z", "Z"
builder += 0.5, "+", "-"
builder += 0.5, "-", "+"
H = builder.build_mpo(L)
```

**Tenax:**
```python
from tenax import AutoMPO

auto = AutoMPO(L=L, d=2)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()
```

Key difference: quimb's `SpinHam1D` adds terms by operator pattern (applied
to all bonds). Tenax's `AutoMPO` adds terms by explicit site indices, giving
full control over geometry (ladders, cylinders, irregular lattices).

### 5. DMRG

**quimb:**
```python
dmrg = qtn.DMRG2(H, bond_dims=[10, 20, 50, 100], cutoffs=1e-10)
dmrg.solve(tol=1e-9, verbosity=1)
E = dmrg.energy
psi = dmrg.state
```

**Tenax:**
```python
from tenax import DMRGConfig, dmrg

config = DMRGConfig(max_bond_dim=100, num_sweeps=10, verbose=True)
result = dmrg(mpo, mps, config)
E = result.energy
psi = result.mps
```

### 6. Contraction and Optimization

**quimb:** Has sophisticated contraction path optimization via `opt_einsum`
and `cotengra`. Supports hyper-optimization of contraction order.

**Tenax:** Uses `opt_einsum` for path finding. `NetworkBlueprint` caches
the contraction path for reuse in inner loops.

---

## Code Translation: Complete DMRG Example

**quimb:**
```python
import quimb.tensor as qtn

builder = qtn.SpinHam1D(S=0.5)
builder += 1.0, "Z", "Z"
builder += 0.5, "+", "-"
builder += 0.5, "-", "+"
H = builder.build_mpo(20)

dmrg = qtn.DMRG2(H, bond_dims=[10, 20, 50, 100])
dmrg.solve(tol=1e-9)
print(f"Energy: {dmrg.energy:.10f}")
```

**Tenax:**
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

1. **Replace `qtn.Tensor` with `DenseTensor`** — move index names to
   `TensorIndex` labels. Drop tags.
2. **Replace `qtn.TensorNetwork` with `TensorNetwork`** — use `add_node()`
   with string IDs instead of tag-based selection.
3. **Replace `SpinHam1D` with `AutoMPO`** — explicit site indices instead
   of operator patterns.
4. **Replace `DMRG2` with `dmrg()`** — functional API, `DMRGConfig` dataclass.
5. **Replace `reindex()` with `relabel()`** — immutable, returns new tensor.
6. **Replace `tn ^ all` with `tn.contract()`** — no operator overloading.
7. **Add `TensorIndex` with FlowDirection** — required for `SymmetricTensor`.
8. **Add JAX RNG keys** — explicit `jax.random.PRNGKey(seed)`.

## What You Gain

- **Symmetry-aware tensors** — `SymmetricTensor` with U(1), Z_n (quimb has none)
- **JIT compilation** — built-in, not opt-in
- **Autodiff** — gradient-based iPEPS optimization
- **iPEPS + excitations** — built-in 2D algorithms
- **iDMRG** — infinite DMRG for thermodynamic limit
- **TRG / HOTRG** — classical stat mech algorithms
- **NetworkBlueprint** — reusable `.net` file contraction templates

## What You Lose

- **Tag system** — flexible tensor selection/grouping
- **`cotengra` integration** — advanced contraction path optimization
- **Visualization** — `tn.draw()` for tensor network diagrams
- **TEBD / TDVP** — not yet in Tenax
- **Backend flexibility** — Tenax is JAX-only; quimb supports multiple backends
- **Arbitrary geometry TN** — quimb's `TensorNetwork` is more flexible for
  non-standard topologies
