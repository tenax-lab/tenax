---
name: tenax-dmrg-workflow
description: >
  Guide graduate students through a complete DMRG ground-state calculation using
  Tenax, from defining the Hamiltonian through AutoMPO to analyzing results.
  Pedagogical and step-by-step. Covers finite DMRG, iDMRG (1D chain and infinite
  cylinder), and 2D cylinder DMRG. Use this skill whenever the user wants to run
  DMRG, find a ground state with tensor networks, set up an MPS calculation,
  build an MPO Hamiltonian, or asks about DMRG sweeps, bond dimensions, canonical
  forms, or convergence — especially in the context of Tenax or tenax. Also
  trigger for "ground state energy", "variational MPS", "finite DMRG", "infinite
  DMRG", "iDMRG", "cylinder DMRG", "AutoMPO", "Heisenberg chain", "build MPO",
  or "how do I use DMRG for [model]".
---

# DMRG Workflow — Guided Ground-State Calculations with Tenax

This skill walks graduate students through complete DMRG calculations using the
Tenax library. The tone is pedagogical: each step explains *what* we're doing,
*why* it works, and *what to watch out for*.

## How to Use This Skill

When the user wants to run a DMRG calculation, first determine which variant:

1. **Finite DMRG** — fixed chain of N sites, open boundaries (most common).
2. **iDMRG** — infinite 1D chain, directly in the thermodynamic limit.
3. **Cylinder DMRG** — 2D lattice wrapped on a cylinder (finite in x, periodic
   in y), mapped to a 1D chain via AutoMPO.
4. **Infinite cylinder iDMRG** — infinite cylinder using super-site MPO.

Then walk through the stages below. At each stage, explain the physics briefly,
show Tenax code, point out key parameters, and ask the student to run and
report before moving on.

---

## Stage 1: Define the Physical System

Before writing any code, make sure the student can answer:

- **What model?** (Heisenberg, transverse-field Ising, Hubbard, custom)
- **What geometry?** (1D chain, ladder, cylinder, infinite chain)
- **What local Hilbert space?** (spin-1/2 → d=2, spin-1 → d=3)
- **Boundary conditions?** (open for finite DMRG, periodic-y for cylinder)
- **What observable?** (ground state energy, correlations, entanglement entropy)

### Physics checkpoint

Ask: "What do you expect the ground state to look like? Is this system gapped
or gapless? Does it break any symmetries?" This builds intuition for
interpreting results later.

---

## Stage 2: Build the MPO Hamiltonian

Tenax provides two ways to build Hamiltonians.

### Option A: Built-in builders (simplest)

For standard models, Tenax has ready-made functions:

```python
from tenax.algorithms.dmrg import build_mpo_heisenberg

# Finite Heisenberg chain, L=20 sites
L = 20
mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
```

For iDMRG (infinite chain):
```python
from tenax import build_bulk_mpo_heisenberg

W = build_bulk_mpo_heisenberg(Jz=1.0, Jxy=1.0)
```

For infinite cylinders:
```python
from tenax import build_bulk_mpo_heisenberg_cylinder

# Ly=4 cylinder: super-site is a ring of 4 spins (d=16)
# Only even Ly supported (odd Ly frustrates AFM order)
W = build_bulk_mpo_heisenberg_cylinder(Ly=4)
```

### Option B: AutoMPO (flexible, any Hamiltonian)

AutoMPO lets you build any Hamiltonian from symbolic operator descriptions.
This is the recommended approach for custom models.

```python
from tenax import AutoMPO

L = 20
auto = AutoMPO(L=L, d=2)  # d=2 for spin-1/2

# Heisenberg model: H = Σ_i S_i · S_{i+1}
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)  # Note: 0.5, not 1.0
    auto += (0.5, "Sm", i, "Sp", i + 1)

mpo = auto.to_mpo()
```

**Key points to emphasize:**

- The Heisenberg exchange uses `S+ S- + S- S+` with a factor of **0.5** each
  (since S·S = Sz·Sz + (1/2)(S+S- + S-S+)). This is the most common
  AutoMPO mistake.
- For custom operators, you can pass a dictionary:
  ```python
  import numpy as np
  custom_ops = {
      "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
      "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
  }
  from tenax import build_auto_mpo
  terms = [(1.0, "Z", i, "Z", i + 1) for i in range(L - 1)]
  terms += [(0.5, "X", i) for i in range(L)]
  mpo = build_auto_mpo(terms, L=L, site_ops=custom_ops)
  ```
- For U(1) symmetric MPOs: `mpo = auto.to_mpo(symmetric=True)`
- For compressed MPOs (long-range terms): `mpo = auto.to_mpo(compress=True)`

### Building a cylinder Hamiltonian with AutoMPO

Map the 2D cylinder to a 1D chain using snake ordering:

```python
from tenax import AutoMPO

Lx, Ly = 6, 3
N = Lx * Ly  # 18 sites
auto = AutoMPO(L=N, d=2)

for x in range(Lx):
    for y in range(Ly):
        # Within-ring bond (periodic in y)
        i = x * Ly + y
        j = x * Ly + (y + 1) % Ly  # Periodic y!
        auto += (1.0, "Sz", min(i,j), "Sz", max(i,j))
        auto += (0.5, "Sp", min(i,j), "Sm", max(i,j))
        auto += (0.5, "Sm", min(i,j), "Sp", max(i,j))
        # Between-ring bond (open in x)
        if x < Lx - 1:
            i = x * Ly + y
            j = (x + 1) * Ly + y
            auto += (1.0, "Sz", i, "Sz", j)
            auto += (0.5, "Sp", i, "Sm", j)
            auto += (0.5, "Sm", i, "Sp", j)

mpo = auto.to_mpo(compress=True)
```

**Emphasize:** `min(i,j)` and `max(i,j)` ensure operators are ordered left-to-right,
which AutoMPO requires. The `compress=True` flag reduces MPO bond dimension for
the long-range y-periodic bonds.

---

## Stage 3: Initialize the MPS

```python
from tenax import FiniteMPS
import jax

key = jax.random.PRNGKey(0)
mps = FiniteMPS.random(L=20, d=2, chi=16, key=key)
```

Or using the legacy builder (still supported):
```python
from tenax import build_random_mps
mps = build_random_mps(20, physical_dim=2, bond_dim=16)
```

### Key points

- **Start with moderate χ.** `chi=16` or `32` for a first run. You can
  always increase in the DMRGConfig.
- **`max_bond_dim` in DMRGConfig controls truncation.** The initial MPS bond
  dimension just needs to be ≤ max_bond_dim. DMRG will grow it as needed
  (in two-site DMRG).
- **Symmetry sectors.** For symmetric MPS, pass `symmetric=True` and
  `target_charge`:
  ```python
  from tenax.core.symmetry import U1Symmetry
  mps = FiniteMPS.random(L=20, d=2, chi=16, key=key,
                         symmetric=True, symmetry=U1Symmetry(),
                         target_charge=0)  # Sz=0 sector
  ```
- **All sites are 3-leg** `(chi_l, d, chi_r)`, including boundaries (which
  have trivial dim-1 bonds).

For cylinders, the physical dimension matches the super-site:
```python
mps = FiniteMPS.random(L=Lx, d=2**Ly, chi=16, key=key)
```

For iDMRG, the initial MPS is handled internally — you just pass the config.

---

## Stage 4: Configure and Run DMRG

### Finite DMRG

```python
from tenax.algorithms.dmrg import dmrg, DMRGConfig

config = DMRGConfig(
    max_bond_dim=64,    # Maximum bond dimension after truncation
    num_sweeps=10,      # Number of left-right sweep pairs
    verbose=True,       # Print energy per sweep
)
result = dmrg(mpo, mps, config)

print(f"Ground state energy: {result.energy:.10f}")
print(f"Converged: {result.converged}")
```

### iDMRG (infinite chain)

```python
from tenax import idmrg, build_bulk_mpo_heisenberg, iDMRGConfig

W = build_bulk_mpo_heisenberg(Jz=1.0, Jxy=1.0)
config = iDMRGConfig(
    max_bond_dim=32,
    max_iterations=100,
    convergence_tol=1e-8,
)
result = idmrg(W, config)

print(f"Energy per site: {result.energy_per_site:.8f}")  # ≈ -0.4431
print(f"Converged: {result.converged}")
```

### iDMRG (infinite cylinder)

```python
from tenax import build_bulk_mpo_heisenberg_cylinder, iDMRGConfig, idmrg

Ly = 4
W = build_bulk_mpo_heisenberg_cylinder(Ly=Ly)
config = iDMRGConfig(
    max_bond_dim=200,
    max_iterations=200,
    convergence_tol=1e-4,
)
result = idmrg(W, config, d=2**Ly)  # d=16 for Ly=4

e_per_spin = result.energy_per_site / Ly
print(f"Energy per spin: {e_per_spin:.6f}")
```

### Key parameters explained

| Parameter | What it does | Too small | Too large |
|-----------|-------------|-----------|-----------|
| `max_bond_dim` | Controls entanglement captured | Energy plateau | Slow, memory-heavy |
| `num_sweeps` | Iterations over the chain | Not converged | Wasted compute |
| `convergence_tol` | Stopping criterion (ΔE) | Premature stop | (Fine, just slower) |

**Recommendation for students:** Start with `max_bond_dim=32, num_sweeps=5`.
If the energy is still changing, increase both. For gapless systems, you'll
typically need `max_bond_dim=128+` and `num_sweeps=15+`.

---

## Stage 5: Monitor Convergence

If `verbose=True`, Tenax prints energy per sweep. Guide students to watch for:

- **Monotonic decrease** — energy should drop every sweep (two-site DMRG) or
  every few sweeps.
- **Rapid initial drop, then slow convergence** — normal and expected.
- **Oscillations** — something is wrong. Check the Hamiltonian, symmetry sector,
  or initial MPS.
- **Plateau far above known value** — bond dimension too small.

### Manual convergence tracking

```python
# Run multiple configs to study χ-dependence
chis = [16, 32, 64, 128]
energies = []
for chi in chis:
    config = DMRGConfig(max_bond_dim=chi, num_sweeps=15)
    key = jax.random.PRNGKey(chi)
    mps = FiniteMPS.random(L=L, d=2, chi=16, key=key)
    result = dmrg(mpo, mps, config)
    energies.append(result.energy)
    print(f"χ = {chi:4d} → E = {result.energy:.10f}")

# Students should plot E vs 1/χ and extrapolate to χ → ∞
```

---

## Stage 6: Extract Physics from the Ground State

Once DMRG converges, extract physical observables from `result`.

### Ground state energy

Already in `result.energy`. Compare against:

| Model | N→∞ exact E/site | Notes |
|-------|-----------------|-------|
| Heisenberg S=1/2 | 1/4 − ln(2) ≈ −0.4431 | Bethe ansatz |
| Heisenberg S=1 | ≈ −1.401 | Haldane chain |
| TFI at h=J | −1/π ≈ −0.31831 | Ising critical point |

For finite chains, the energy per site approaches these values as N→∞.

### Entanglement entropy

The entanglement entropy across a bond reveals the nature of the state:
- **Area law** (constant S) → gapped system.
- **Logarithmic S(l) = (c/3) ln(l)** → critical system with central charge c.
- For the Heisenberg chain: c = 1 (Luttinger liquid).

### Correlation functions

Use the built-in `correlation()` function:

```python
from tenax.algorithms.observables import correlation
import numpy as np

Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
corr = [correlation(result.mps, Sz, 0, Sz, r) for r in range(L)]
```

Compute ⟨S^z_i S^z_j⟩ as a function of distance |i−j| to identify:
- **Algebraic decay** (1/r^α) → gapless / quasi-long-range order.
- **Exponential decay** (e^{−r/ξ}) → gapped, with correlation length ξ.

---

## Stage 7: Beyond Basic DMRG

Once the basic calculation works, guide students through extensions:

### Bond dimension scaling
Run at χ = 32, 64, 128, 256. Plot E vs 1/χ (or vs truncation error) and
extrapolate. This is how real research papers report DMRG energies.

### Finite-size scaling
Run at N = 20, 40, 80, 160 and extrapolate E/N to the thermodynamic limit.

### 2D systems on cylinders
Use AutoMPO to build cylinder Hamiltonians (Stage 2). Start with small Ly
(Ly=2 or 3) and increase. The MPO bond dimension grows with Ly, so cylinder
DMRG gets expensive fast. Typical research uses Ly=4–8 with χ=1000+.

### Symmetry-aware DMRG
Use `auto.to_mpo(symmetric=True)` for U(1) block-sparse MPOs. This exploits
conservation laws (e.g., total Sz) to reduce computational cost and ensure
the result is in the correct quantum number sector.

### Comparison with iPEPS for 2D
For truly 2D problems, iPEPS may be more natural than cylinder DMRG:
```python
from tenax import iPEPSConfig, CTMConfig, ipeps

config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.3,
    ctm=CTMConfig(chi=10, max_iter=40),
    unit_cell="2site",
)
energy, peps, envs = ipeps(gate, None, config)
```

---

## Common Pitfalls (Quick Reference)

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Energy too high, not improving | χ too small | Increase `max_bond_dim` |
| Energy oscillates | Bad initial state or Hamiltonian bug | Check AutoMPO terms |
| `UFuncOutputCastingError` | Complex ops + float64 array | Use `complex128` or Sp/Sm |
| Wrong energy for cylinder | Site ordering wrong | Check `min(i,j)`, modular y |
| iDMRG gives wrong E/site for cylinder | Forgot to divide by Ly | `e_per_spin = result.energy_per_site / Ly` |
| Symmetric MPO crashes | Initial MPS in wrong sector | Match quantum numbers |

---

## Pedagogical Notes

- **Don't skip the physics.** Every code block should be preceded by a brief
  explanation. The goal is a physicist who can write tensor network code.
- **Encourage experimentation.** "What happens if you set χ = 4? What if you
  double the chain length?"
- **Connect to coursework.** Link DMRG concepts to solid state physics:
  Bloch's theorem ↔ translational invariance in iDMRG, band gaps ↔ entanglement
  entropy scaling, Néel order ↔ why we need 2-site unit cells in iPEPS.
