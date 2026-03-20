---
name: tenax-migration-tenpy
description: >
  Help users migrate tensor network code from TeNPy to Tenax. Maps TeNPy
  concepts (Site, MPS, MPO, Model, Engine) to Tenax equivalents. Use this
  skill when the user mentions TeNPy, "coming from TeNPy", "TeNPy equivalent",
  or asks how Tenax compares to TeNPy. Also trigger for "migrate from TeNPy",
  "I used to use TeNPy", or "tenpy".
---

# Migrating from TeNPy to Tenax

Help TeNPy users translate their object-oriented workflow to Tenax's
functional-style JAX-based approach.

## Quick Reference

| TeNPy | Tenax | Notes |
|-------|-------|-------|
| `SpinHalfSite()` | `spin_half_ops()` | Returns operator dict, no Site object |
| `MPS.from_lat_product_state(...)` | `FiniteMPS.random(L, d, chi, key)` | No lattice/product-state builder |
| `MPOModel` / `CouplingMPOModel` | `AutoMPO(L, d)` | Functional, not class-based |
| `model.calc_H_MPO()` | `auto.to_mpo()` | Direct construction |
| `TwoSiteDMRGEngine(psi, model, params)` | `dmrg(mpo, mps, config)` | Functional API |
| `eng.run()` | `result = dmrg(mpo, mps, config)` | Returns result dataclass |
| `psi.entanglement_entropy()` | Manual from singular values | No built-in method |
| `psi.correlation_function("Sz", "Sz")` | Manual contraction | No built-in correlations |
| `Array` (TeNPy's tensor) | `DenseTensor` / `SymmetricTensor` | Label-based in both |
| `npc.tensordot(A, B, axes)` | `contract(A, B)` | Tenax uses label matching |
| `npc.svd(A, inner_labels)` | `truncated_svd(A, left_labels, right_labels)` | Explicit partition |

---

## Key Design Differences

### 1. No Model / Site / Lattice Classes

**TeNPy** uses an object-oriented hierarchy: `Site` → `Lattice` → `Model` →
`MPOModel`. You define the physics through class inheritance and configuration
dicts.

```python
# TeNPy
from tenpy.models.xxz_chain import XXZChain
model_params = {"L": 20, "Jxx": 1.0, "Jz": 1.0, "hz": 0.0}
model = XXZChain(model_params)
```

**Tenax** is functional — build the Hamiltonian directly:

```python
# Tenax
from tenax import AutoMPO

L = 20
auto = AutoMPO(L=L, d=2)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()
```

No model classes, no parameter dicts — just explicit operator terms.

### 2. No Engine Pattern

**TeNPy:** Algorithms are engine objects with mutable state:

```python
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
eng = TwoSiteDMRGEngine(psi, model, dmrg_params)
E, psi = eng.run()
# Access internals: eng.sweep_stats, eng.trunc_err, ...
```

**Tenax:** Algorithms are pure functions:

```python
from tenax import dmrg, DMRGConfig

config = DMRGConfig(max_bond_dim=100, num_sweeps=10, verbose=True)
result = dmrg(mpo, mps, config)
# result.energy, result.mps, result.converged, result.energies_per_sweep
```

### 3. Charge Conservation (Similar Concept, Different API)

**TeNPy:** Charges are defined on `Site` objects. The `Array` class (TeNPy's
tensor) tracks charge legs automatically.

```python
site = SpinHalfSite(conserve="Sz")  # U(1) Sz conservation
```

**Tenax:** Charges are on `TensorIndex`. Use `SymmetricTensor` for block-sparse:

```python
from tenax import U1Symmetry, TensorIndex, FlowDirection, SymmetricTensor

u1 = U1Symmetry()
phys = TensorIndex(u1, np.array([-1, 1], dtype=np.int32), FlowDirection.IN, label="p")
```

Or simply use `auto.to_mpo(symmetric=True)` and the MPO handles it.

### 4. NumPy vs JAX

**TeNPy:** Pure NumPy/SciPy. No GPU, no autodiff. Mature and stable.

**Tenax:** Pure JAX. GPU/TPU support, JIT compilation, automatic
differentiation. Enables AD-based iPEPS optimization.

### 5. Observables

**TeNPy:** Rich built-in measurement tools:

```python
Sz = psi.expectation_value("Sz")
C = psi.correlation_function("Sz", "Sz")
S = psi.entanglement_entropy()
```

**Tenax:** Observables require manual contraction. The iDMRG result exposes
`singular_values` for entanglement entropy:

```python
import jax.numpy as jnp

S = result.singular_values
p = (S / jnp.linalg.norm(S))**2
entropy = -jnp.sum(p * jnp.log(p))
```

See the **observables skill** for full guidance.

---

## Code Translation: Complete DMRG Example

**TeNPy:**
```python
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine

model_params = {"L": 20, "Jxx": 1.0, "Jz": 1.0, "hz": 0.0, "bc_MPS": "finite"}
model = XXZChain(model_params)

psi = MPS.from_lat_product_state(model.lat, [["up"], ["down"]])

dmrg_params = {"trunc_params": {"chi_max": 100}, "mixer": True}
eng = TwoSiteDMRGEngine(psi, model, dmrg_params)
E, psi = eng.run()
print(f"Energy: {E:.10f}")
print(f"Entanglement entropy: {psi.entanglement_entropy()}")
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
mps = FiniteMPS.random(L=L, d=2, chi=16, key=key)
config = DMRGConfig(max_bond_dim=100, num_sweeps=10, verbose=True)
result = dmrg(mpo, mps, config)
print(f"Energy: {result.energy:.10f}")
```

---

## Migration Checklist

1. **Replace Model classes with AutoMPO** — explicitly add each coupling term.
2. **Replace Engine with function call** — `dmrg(mpo, mps, config)`.
3. **Replace `dmrg_params` dict with `DMRGConfig`** — typed dataclass.
4. **Replace `Site` with operator dicts** — `spin_half_ops()`, `spin_one_ops()`.
5. **Replace `Array` with `DenseTensor`/`SymmetricTensor`** — label-based.
6. **Replace `npc.tensordot` with `contract()`** — label matching, not axis lists.
7. **Replace `psi.expectation_value()` with manual contraction** — see observables skill.
8. **Add JAX RNG keys** — `jax.random.PRNGKey(seed)` for random initialization.

## What You Gain

- **GPU/TPU** — same code on all backends
- **JIT compilation** — significant speedup for large problems
- **Autodiff** — gradient-based iPEPS optimization
- **iPEPS + excitations** — built-in 2D algorithms
- **NetworkBlueprint** — reusable contraction templates (`.net` files)

## What You Lose

- **Rich model library** — TeNPy has dozens of pre-built models; Tenax requires
  manual AutoMPO construction
- **Built-in observables** — `expectation_value()`, `correlation_function()`,
  `entanglement_entropy()` are not available as methods
- **Product state initialization** — Tenax uses random MPS; no `from_lat_product_state`
- **TEBD / TDVP** — not yet in Tenax
- **Detailed documentation** — TeNPy has extensive user guides and tutorials
- **Per-sweep parameter schedules** — TeNPy supports chi ramp-up per sweep
