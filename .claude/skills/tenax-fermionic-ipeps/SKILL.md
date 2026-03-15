---
name: tenax-fermionic-ipeps
description: >
  Guide users through fermionic iPEPS (fPEPS) calculations using Tenax's
  graded tensor formalism. Covers spinless fermions, the t-V model,
  FermionParity and FermionicU1 symmetries, and the fpeps() entry point.
  Use this skill when the user mentions fermionic PEPS, fPEPS, fermionic
  tensor networks, spinless fermions, t-V model, Hubbard model on a 2D
  lattice, FermionParity, FermionicU1, graded tensors, Koszul signs,
  fermionic exchange statistics, or Jordan-Wigner strings in 2D.
  Also trigger for "fermion sign problem in tensor networks",
  "anticommutation in PEPS", or "how do I handle fermions in 2D".
---

# Fermionic iPEPS (fPEPS) — 2D Fermionic Ground States with Tenax

This skill guides users through fermionic infinite Projected Entangled Pair
State calculations. Fermions in 2D tensor networks require careful treatment
of exchange statistics — Tenax handles this automatically via the graded
tensor formalism.

## When to Use fPEPS vs Regular iPEPS

| | iPEPS (bosonic) | fPEPS (fermionic) |
|---|---|---|
| **Statistics** | Bosonic (commuting) | Fermionic (anticommuting) |
| **Symmetry class** | `U1Symmetry`, `ZnSymmetry` | `FermionParity`, `FermionicU1` |
| **Sign handling** | None needed | Automatic via graded tensors |
| **Gate builder** | `heisenberg_gate()`, `xxz_gate()` | `spinless_fermion_gate()` |
| **Entry point** | `ipeps()` / `optimize_gs_ad()` | `fpeps()` |
| **Models** | Heisenberg, XXZ, J1-J2 | Spinless fermions, t-V, Hubbard |

Rule of thumb: if your Hamiltonian has creation/annihilation operators
(c†, c) rather than spin operators (S+, S-, Sz), use fPEPS.

---

## Background: Why Fermions Are Special in 2D

In 1D, fermionic signs are handled by Jordan-Wigner strings — a linear
ordering of sites turns anticommutation into phase factors along a chain.
In 2D there is no canonical linear ordering, so Jordan-Wigner strings
become path-dependent and cumbersome.

Tenax uses the **graded tensor formalism** instead: each tensor leg carries
a Z₂ parity (even=0, odd=1), and all tensor operations (transpose,
contraction, SVD) automatically insert the correct Koszul signs. This means:

- No explicit swap gates in the tensor network
- No Jordan-Wigner strings to track
- Standard CTM works unchanged — fermionic signs are built into the tensors
- The `SymmetricTensor` class handles everything internally

---

## Stage 1: Choose the Symmetry

### FermionParity (Z₂ grading)

The simplest fermionic symmetry. Charges are 0 (even/bosonic) and
1 (odd/fermionic). Exchanging two odd-parity objects yields a minus sign.

```python
from tenax import FermionParity

fp = FermionParity()
print(fp.exchange_sign(1, 1))  # -1 (two fermions anticommute)
print(fp.exchange_sign(0, 1))  # +1 (boson-fermion commute)
print(fp.is_fermionic)         # True
```

Use `FermionParity` when you only need to track the parity of particle
number (even vs odd), not the exact particle number.

### FermionicU1 (U(1) with fermionic grading)

For models where total particle number is conserved, use `FermionicU1`.
It combines U(1) charge conservation with fermionic exchange statistics.

```python
from tenax import FermionicU1

fu1 = FermionicU1()  # Default grading: abs(q) % 2
print(fu1.fuse(1, 1))            # 2 (U(1) addition)
print(fu1.exchange_sign(1, 1))   # -1 (both odd parity → anticommute)
print(fu1.exchange_sign(0, 2))   # +1 (both even parity → commute)
```

The grading function maps U(1) charges to Z₂ parity. The default
`"abs_mod_2"` maps charge q to |q| mod 2. Alternative: `"mod_2"` maps
q to q mod 2.

```python
fu1_alt = FermionicU1(grading_key="mod_2")
```

### Physics checkpoint

Ask the student: "Does your model conserve total particle number? If yes,
use `FermionicU1` for better block sparsity. If you only need even/odd
parity, `FermionParity` is simpler and sufficient."

---

## Stage 2: Build the Gate

### Pre-built: spinless fermion gate

Tenax provides a ready-made gate for the spinless fermion t-V model:

H = −t(c†ᵢcⱼ + c†ⱼcᵢ) + V nᵢnⱼ

```python
from tenax import FPEPSConfig, spinless_fermion_gate

config = FPEPSConfig(
    D=2,                       # Virtual bond dimension
    t=1.0,                     # Hopping amplitude
    V=0.0,                     # NN interaction (V=0 → free fermions)
    dt=0.01,                   # Trotter time step
    num_imaginary_steps=200,   # Number of evolution steps
    ctm_chi=8,                 # CTM environment bond dim
    ctm_max_iter=50,           # CTM convergence iterations
)

gate = spinless_fermion_gate(config)
print(gate.todense().shape)  # (2, 2, 2, 2)
print(gate.labels())         # ('si', 'sj', 'si_out', 'sj_out')
```

The gate is a `SymmetricTensor` with `FermionParity` symmetry. Physical
charges are [0, 1] (empty, occupied). The 4×4 matrix in the occupation
basis {|00⟩, |01⟩, |10⟩, |11⟩} is:

```
H = | 0   0   0   0 |
    | 0   0  -t   0 |
    | 0  -t   0   0 |
    | 0   0   0   V |
```

### Building a custom fermionic gate

For models not covered by `spinless_fermion_gate`, construct the gate
manually using `SymmetricTensor.from_dense()`:

```python
import jax.numpy as jnp
import numpy as np
from tenax import FermionParity, SymmetricTensor, TensorIndex, FlowDirection

fp = FermionParity()
phys_charges = np.array([0, 1], dtype=np.int32)  # |empty⟩, |occupied⟩

# Build 4x4 Hamiltonian matrix in occupation basis
H_mat = jnp.zeros((4, 4))
H_mat = H_mat.at[1, 2].set(-t)  # c†_i c_j
H_mat = H_mat.at[2, 1].set(-t)  # c†_j c_i
H_mat = H_mat.at[3, 3].set(V)   # n_i n_j
H_dense = H_mat.reshape(2, 2, 2, 2)

indices = (
    TensorIndex(fp, phys_charges.copy(), FlowDirection.IN,  label="si"),
    TensorIndex(fp, phys_charges.copy(), FlowDirection.IN,  label="sj"),
    TensorIndex(fp, phys_charges.copy(), FlowDirection.OUT, label="si_out"),
    TensorIndex(fp, phys_charges.copy(), FlowDirection.OUT, label="sj_out"),
)

gate = SymmetricTensor.from_dense(H_dense, indices)
```

### Verifying the gate

Always check Hermiticity and the spectrum:

```python
H_4x4 = gate.todense().reshape(4, 4)
assert jnp.allclose(H_4x4, H_4x4.T.conj())       # Hermitian
evals = jnp.linalg.eigvalsh(H_4x4)
print("Spectrum:", jnp.sort(evals))                 # Free: [-t, 0, 0, +t]
```

---

## Stage 3: Run fPEPS

The `fpeps()` function runs the full pipeline: random initialization →
simple update (imaginary time evolution) → CTM energy evaluation.

```python
from tenax import FPEPSConfig, fpeps, spinless_fermion_gate
import jax

config = FPEPSConfig(
    D=2,
    t=1.0,
    V=0.0,
    dt=0.01,
    num_imaginary_steps=200,
    ctm_chi=8,
    ctm_max_iter=50,
)

gate = spinless_fermion_gate(config)
energy, A_opt, env = fpeps(gate, config, key=jax.random.PRNGKey(42))
print(f"Energy per site: {energy:.6f}")
```

### Return values

| Value | Type | Description |
|-------|------|-------------|
| `energy` | float | Variational energy per site |
| `A_opt` | SymmetricTensor | Optimized site tensor, shape (D,D,D,D,2) |
| `env` | CTMTensorEnv | Converged CTM environment |

### Starting from an existing tensor

```python
energy, A_opt, env = fpeps(gate, config, initial_tensor=A_prev, key=key)
```

---

## Stage 4: Understanding the fPEPS Tensor

The site tensor `A` has 5 legs: `(u, d, l, r, phys)` corresponding to
up, down, left, right virtual bonds and the physical leg.

```
        u (OUT)
        |
l (OUT)—A—r (IN)
        |
        d (IN)
        |
      phys (IN)
```

- Virtual charges alternate parity: `[0, 1, 0, 1, ...]` up to D
- Physical charges: `[0, 1]` (empty, occupied)
- Flow directions ensure charge conservation in contractions

---

## Stage 5: Key Parameters

| Parameter | Role | Guidance |
|-----------|------|----------|
| `D` | Virtual bond dim | Start D=2, increase to 3,4 |
| `t` | Hopping amplitude | Sets energy scale |
| `V` | NN interaction | V=0 → free fermions, V>0 → repulsive |
| `dt` | Trotter step | 0.01–0.1; smaller = more accurate but slower |
| `num_imaginary_steps` | Evolution length | 200–500; increase if not converged |
| `ctm_chi` | CTM bond dim | χ ≥ D²; start 8–16 |
| `ctm_max_iter` | CTM iterations | 40–100 |
| `ctm_conv_tol` | CTM convergence | 1e-6 default |

---

## Stage 6: The Simple Update Algorithm

Under the hood, `fpeps()` performs imaginary time evolution with
alternating horizontal and vertical bond updates:

1. **Compute Trotter gate**: exp(−δτ H) via eigendecomposition of H
2. **Horizontal update**: Contract two sites across horizontal bond →
   apply gate → SVD to re-split → truncate to D → update bond weights
3. **Vertical update**: Same for the vertical bond
4. **Repeat** for `num_imaginary_steps` iterations
5. **Absorb bond weights** into site tensor → run CTM → compute energy

The bond weights (λ vectors) approximate the environment during updates
(like mean-field). This is why it's called "simple" update — the full
environment is only used at the end for energy evaluation.

### What to watch for

- **Energy not decreasing** → dt too large (Trotter error). Reduce dt.
- **NaN in tensors** → dt too large or gate not Hermitian.
- **Energy converged but too high** → D too small, or CTM χ too small.
- **Bond lambdas all equal** → system may not have converged; try more steps.

---

## Stage 7: The Graded Tensor Formalism

This is the key insight that makes fPEPS tractable. Explain to students:

"In the graded tensor formalism, every tensor index carries a Z₂ parity
label. When you transpose tensor legs or contract tensors, the formalism
automatically tracks which fermionic lines cross and inserts the correct
minus signs (Koszul signs). This is mathematically equivalent to placing
swap gates at every crossing, but it's handled inside the tensor
operations — you never see the signs explicitly.

The practical consequence: once your gate and site tensor use a fermionic
symmetry class (`FermionParity` or `FermionicU1`), you can use the
standard CTM algorithm unchanged. The double-layer CTM contractions will
produce the correct fermionic expectation values automatically."

### When signs matter

Signs appear in three operations inside `SymmetricTensor`:

1. **Transpose** — reordering legs of a fermionic tensor picks up signs
   from the permutation parity of odd-charge legs.
2. **Contraction** — summing over a shared fermionic leg includes the
   exchange sign from moving that leg past others.
3. **SVD** — decomposition respects the grading, ensuring U and Vh
   tensors have consistent fermionic charges.

---

## Stage 8: Convergence Studies

### D-scaling

```python
import jax
from tenax import FPEPSConfig, fpeps, spinless_fermion_gate

for D in [2, 3, 4]:
    config = FPEPSConfig(
        D=D, t=1.0, V=0.0, dt=0.01,
        num_imaginary_steps=300,
        ctm_chi=max(D**2, 8), ctm_max_iter=60,
    )
    gate = spinless_fermion_gate(config)
    E, _, _ = fpeps(gate, config, key=jax.random.PRNGKey(0))
    print(f"D={D}: E/site = {E:.8f}")
```

### V-scan (phase diagram)

```python
for V in [0.0, 0.5, 1.0, 2.0, 4.0]:
    config = FPEPSConfig(D=2, t=1.0, V=V, dt=0.01,
                         num_imaginary_steps=300, ctm_chi=8)
    gate = spinless_fermion_gate(config)
    E, _, _ = fpeps(gate, config, key=jax.random.PRNGKey(0))
    print(f"V={V:.1f}: E/site = {E:.8f}")
```

At large V/t, the system transitions from a metallic phase to a
charge-density-wave (CDW) insulator.

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| Wrong energy sign | Gate sign convention | Check H is Hermitian; verify spectrum |
| NaN after few steps | dt too large | Reduce dt to 0.001–0.01 |
| Energy too high | D or χ too small | Increase both; χ ≥ D² |
| Flat energy vs D | Trotter error dominates | Reduce dt first, then increase D |
| "Charge mismatch" error | Wrong FlowDirection | Check IN/OUT pairing on gate legs |
| Dense tensor where SymmetricTensor expected | Used jnp array directly | Wrap with `SymmetricTensor.from_dense()` |

---

## Pedagogical Notes

- **Connect to condensed matter:** fPEPS is a variational wavefunction
  for interacting fermions on a 2D lattice, analogous to a Gutzwiller-
  projected BCS state but with controlled entanglement truncation.
- **The sign problem:** Quantum Monte Carlo suffers from the sign problem
  for fermions; tensor networks avoid it entirely (at the cost of the
  bond dimension approximation).
- **Grading vs JW:** The graded tensor approach is equivalent to
  Jordan-Wigner but more elegant in 2D — it doesn't depend on an
  arbitrary choice of 1D path through the 2D lattice.
- **From free to interacting:** Start with V=0 (free fermions, exact
  solution known) to validate, then turn on V to study interactions.
