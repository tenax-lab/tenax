---
name: tenax-migration-cytnx
description: >
  Help users migrate tensor network code from Cytnx to Tenax. Maps Cytnx
  concepts (UniTensor, Bond, Network) to their Tenax equivalents (DenseTensor,
  SymmetricTensor, TensorIndex, TensorNetwork, NetworkBlueprint), translates
  code patterns, and explains key design differences. Use this skill when
  the user mentions Cytnx, UniTensor, Bond, "coming from Cytnx", "convert
  Cytnx code", "Cytnx vs Tenax", or asks how Tenax compares to Cytnx.
  Also trigger for "migrate from Cytnx", "Cytnx equivalent", or "I used
  to use Cytnx".
---

# Migrating from Cytnx to Tenax

Help Cytnx users translate their mental model and code to Tenax. The two
libraries share a label-based contraction philosophy and `.net` file
support, but differ in backend (C++ vs JAX), tensor design (UniTensor vs
DenseTensor/SymmetricTensor), and decomposition API.

## Quick Reference

| Cytnx | Tenax | Notes |
|-------|-------|-------|
| `UniTensor` | `DenseTensor` / `SymmetricTensor` | No row/col rank in Tenax |
| `Bond` | `TensorIndex` | Carries symmetry, charges, FlowDirection, label |
| `Bond.BD_IN` / `BD_OUT` | `FlowDirection.IN` / `OUT` | Same concept |
| `Bond(dim, BD_IN, [[charges]])` | `TensorIndex.from_charges(sym, charges, flow, label)` | Label is part of the index |
| `Network` | `NetworkBlueprint` | Same `.net` file format |
| `Contract(A, B)` | `contract(A, B)` | Both label-based |
| `Svd(T)` | `truncated_svd(T, left_labels, right_labels)` | Explicit label partition |
| `Qr(T)` | `qr_decompose(T, left_labels, right_labels)` | Explicit label partition |
| — | `AutoMPO`, `dmrg`, `idmrg`, `trg`, `ipeps` | Tenax has built-in algorithms |
| `T.labels()` | `T.labels()` | Same |
| `T.set_labels(...)` | `T.relabel(old, new)` / `T.relabels({...})` | Immutable in Tenax |
| `T.Conj()` | `T.conj()` | Returns new tensor |
| `T.Transpose(perm)` | `T.transpose(labels)` | By label, not index |
| `T.Norm()` | `T.norm()` | Same |

---

## Key Design Differences

### 1. No Row/Column Rank

**Cytnx:** `UniTensor` tracks which legs are "row" (bra) and which are
"column" (ket). SVD and QR use this partition implicitly.

```cpp
// Cytnx: row/col rank determines SVD partition
auto T = UniTensor({bond_a, bond_b, bond_c}, {}, 2);  // rowrank=2
auto [U, S, Vh] = Svd(T);  // Splits at rowrank boundary
```

**Tenax:** No row/column distinction. SVD and QR take explicit label lists:

```python
# Tenax: explicit label partition
U, S, Vh, _ = truncated_svd(
    T,
    left_labels=["a", "b"],    # These go into U
    right_labels=["c"],        # These go into Vh
    new_bond_label="bond",
)
```

**Why:** JAX's functional model requires immutable tensors. Storing mutable
row/column rank doesn't fit JAX pytrees. The explicit-labels approach is
also less error-prone — you see exactly what's being split.

### 2. Labels Live on TensorIndex

**Cytnx:** Labels are set separately from Bonds:

```cpp
auto T = UniTensor({bd_a, bd_b, bd_c});
T.set_labels({"left", "phys", "right"});
```

**Tenax:** Labels are part of the index definition:

```python
idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="left")
A = SymmetricTensor.random_normal(indices=(idx_left, idx_phys, idx_right), key=key)
# Labels are intrinsic — no separate set_labels step
```

### 3. Immutable Tensors

**Cytnx:** `UniTensor` is mutable — you can modify labels, reshape in place.

**Tenax:** Tensors are immutable (JAX pytrees). Operations return new tensors:

```python
# Cytnx style (mutable):
# T.set_labels(["a", "b", "c"])

# Tenax style (immutable):
T_new = T.relabels({"old_a": "a", "old_b": "b"})
```

### 4. JAX Backend → Autodiff + JIT

**Cytnx:** C++ backend with Python bindings. No automatic differentiation.

**Tenax:** Pure JAX. Everything is differentiable and JIT-compilable:

```python
import jax

# Differentiate through a tensor network contraction
@jax.jit
def energy(params):
    A = build_peps_from_params(params)
    return compute_energy(A)

grads = jax.grad(energy)(params)  # Works!
```

This enables AD-based iPEPS optimization (`optimize_gs_ad`) which has no
Cytnx equivalent.

### 5. Built-in Algorithms

Cytnx is a tensor library — DMRG, TRG, etc. are implemented by the user.
Tenax includes production-ready algorithms:

```python
from tenax import dmrg, idmrg, trg, hotrg, ipeps, optimize_gs_ad
from tenax import AutoMPO, build_auto_mpo
```

---

## Code Translation Examples

### Creating a symmetric tensor

**Cytnx:**
```cpp
auto bd_phys = Bond(2, BD_IN, {{Qs(-1), Qs(1)}});
auto bd_bond = Bond(3, BD_IN, {{Qs(-1), Qs(0), Qs(1)}});
auto T = UniTensor({bd_phys, bd_bond, bd_bond.redirect()}, {}, 1);
T.set_labels({"p", "l", "r"});
```

**Tenax:**
```python
from tenax import U1Symmetry, TensorIndex, FlowDirection, SymmetricTensor
import numpy as np, jax

u1 = U1Symmetry()
T = SymmetricTensor.random_normal(
    indices=(
        TensorIndex.from_charges(u1, np.array([-1, 1], dtype=np.int32), FlowDirection.IN,  label="p"),
        TensorIndex.from_charges(u1, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN,  label="l"),
        TensorIndex.from_charges(u1, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.OUT, label="r"),
    ),
    key=jax.random.PRNGKey(0),
)
```

### Contraction

**Cytnx:**
```cpp
auto result = Contract(A, B);
```

**Tenax:**
```python
from tenax import contract
result = contract(A, B)
```

Identical semantics: shared labels are contracted.

### .net file / NetworkBlueprint

**Cytnx:**
```cpp
auto net = Network("dmrg_eff_ham.net");
net.PutUniTensor("L", L_env);
net.PutUniTensor("W", W);
net.PutUniTensor("R", R_env);
auto result = net.Launch();
```

**Tenax:**
```python
from tenax import NetworkBlueprint

bp = NetworkBlueprint("dmrg_eff_ham.net")  # Same .net file format!
bp.put_tensor("L", L_env)
bp.put_tensor("W", W)
bp.put_tensor("R", R_env)
result = bp.launch()
```

The `.net` file format is compatible between Cytnx and Tenax. The one
difference: Cytnx uses a semicolon `;` in `TOUT:` to mark the row/column
boundary. Tenax ignores semicolons (no row/column rank).

### SVD

**Cytnx:**
```cpp
auto T = UniTensor({bd_a, bd_b, bd_c}, {}, 2);  // rowrank=2
auto [U, S, Vh] = Svd(T);
auto [U_t, S_t, Vh_t] = Svd_truncate(T, 16);
```

**Tenax:**
```python
from tenax import truncated_svd

U, S, Vh, S_full = truncated_svd(
    T,
    left_labels=["a", "b"],
    right_labels=["c"],
    new_bond_label="bond",
    max_singular_values=16,
)
```

### Building a Hamiltonian

**Cytnx:** Manual MPO construction (no AutoMPO equivalent).

**Tenax:**
```python
from tenax import AutoMPO

L = 20
auto = AutoMPO(L=L, d=2)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()
```

---

## Migration Checklist

When porting a Cytnx project to Tenax:

1. **Replace `Bond` with `TensorIndex`** — move labels into the index
   constructor. Map `BD_IN`/`BD_OUT` to `FlowDirection.IN`/`OUT`.

2. **Replace `UniTensor` with `DenseTensor` or `SymmetricTensor`** — use
   `SymmetricTensor` if you had symmetry-aware Bonds, `DenseTensor` otherwise.

3. **Replace `Svd`/`Qr` with explicit label partition** — identify which
   labels are "left" vs "right" (this was implicit via rowrank in Cytnx).

4. **Replace `Network` with `NetworkBlueprint`** — the `.net` files work
   as-is (ignore semicolons in `TOUT:`). Rename `PutUniTensor` → `put_tensor`,
   `Launch` → `launch`.

5. **Replace manual MPO construction with AutoMPO** — if you were building
   MPOs by hand in Cytnx, `AutoMPO` is much simpler.

6. **Replace manual DMRG loops with `dmrg()`/`idmrg()`** — Tenax has
   production-ready algorithm implementations.

7. **Remove `set_labels` calls** — labels are immutable in Tenax. Use
   `relabel()` or `relabels()` if you need to change them.

8. **Add `jax.random.PRNGKey`** — JAX requires explicit random keys
   (no global RNG state). Pass a key to `random_normal()` etc.

---

## What You Gain

- **Autodiff** — differentiate through any tensor network contraction
- **JIT compilation** — `jax.jit` for automatic optimization
- **Built-in algorithms** — DMRG, iDMRG, TRG, HOTRG, iPEPS, excitations
- **AutoMPO** — symbolic Hamiltonian construction
- **GPU/TPU** — same code runs on CPU, CUDA, TPU, and Metal

## What You Lose

- **C++ performance** for small tensors — JAX has JIT overhead
- **Row/column rank** semantics — replaced by explicit label arguments
- **Non-Abelian symmetry** — Tenax currently supports only Abelian (U(1), Z_n)
- **Richer linear algebra** — Cytnx has `Eig`, `Inv`, `Det` on UniTensor;
  in Tenax, use `jax.numpy.linalg` on the underlying arrays
