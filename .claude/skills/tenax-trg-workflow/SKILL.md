---
name: tenax-trg-workflow
description: >
  Guide graduate students through Tensor Renormalization Group (TRG) and HOTRG
  calculations for 2D classical statistical mechanics models using Tenax.
  Covers building the initial tensor from the partition function, running
  coarse-graining, extracting free energy and thermodynamic quantities, and
  studying phase transitions. Use this skill when the user mentions TRG, HOTRG,
  tensor renormalization group, classical partition function, 2D Ising model,
  transfer matrix, coarse-graining, real-space renormalization, free energy
  calculation, critical temperature, critical exponents, or "how to study phase
  transitions with tensor networks". Also trigger for compute_ising_tensor,
  TRGConfig, GILT, Gilt-TNR, or classical stat mech problems.
---

# TRG/HOTRG Workflow — Classical Stat Mech with Tenax

This skill guides students through tensor renormalization group calculations
for 2D classical lattice models. TRG is a real-space RG method that
systematically coarse-grains the partition function using tensor decompositions.

## Background for Students

The partition function of a 2D classical model can be written as a tensor
network: one tensor per lattice site, with shared indices representing the
Boltzmann weights of neighboring interactions. The free energy per site is:

    f = -(1/β) lim_{N→∞} (1/N) ln Z

TRG computes this by iteratively contracting and truncating the tensor network,
doubling the effective system size with each step.

---

## Stage 1: Build the Initial Tensor

### 2D Ising model

Tenax provides a built-in function:

```python
from tenax.algorithms.trg import compute_ising_tensor
import jax.numpy as jnp

beta = 0.44  # Inverse temperature (near T_c ≈ 2.269, β_c ≈ 0.4407)
T = compute_ising_tensor(beta)        # DenseTensor; symmetric=True gives Z2
print(T.labels())          # ('up', 'down', 'left', 'right')
print(T.todense().shape)   # (2, 2, 2, 2) — tensors have no .shape
```

The tensor `T` encodes the Boltzmann weights exp(βJ σ_i σ_j) for all spin
configurations on the four bonds meeting at a site.

### Custom models

For models beyond Ising, build the tensor from the Boltzmann weights manually.
The general recipe:

1. Write the energy as a sum over bonds: E = Σ_{⟨ij⟩} ε(σ_i, σ_j)
2. For each site, form a tensor T[u,d,l,r] (labels `up, down, left, right`,
   the order `trg` expects) = Π_{bonds} √(W(σ_bond)) where
   W(σ_i, σ_j) = exp(-β ε(σ_i, σ_j)) and the square root is split between
   the two sites sharing each bond.
3. For the Ising model with σ = ±1:
   W = [[exp(βJ), exp(-βJ)], [exp(-βJ), exp(βJ)]]
   and T is constructed by SVD of W.

### Physics checkpoint

Ask: "What temperature range are you interested in? Is the model exactly
solvable (for validation)? What's the expected critical temperature?"

For the 2D Ising model: T_c = 2/ln(1+√2) ≈ 2.269, β_c ≈ 0.4407.

---

## Stage 2: Run TRG

```python
from tenax.algorithms.trg import trg, TRGConfig

config = TRGConfig(
    max_bond_dim=16,   # Truncation bond dimension χ
    num_steps=20,      # Number of coarse-graining steps
)
log_z = trg(T, config)          # returns ln(Z)/N, NOT the free energy
f = -float(log_z) / beta        # f = -ln(Z)/(N β)
print(f"Free energy per site: {f:.8f}")
```

### What happens at each step

Each TRG step:
1. Decomposes the 4-index tensor T into pairs of 3-index tensors (via SVD).
2. Contracts neighboring pairs to form a new effective tensor T'.
3. Truncates to bond dimension χ.
4. The effective system size doubles: after n steps, the network represents
   2^n × 2^n sites.

After 20 steps: 2^20 ≈ 10^6 sites — effectively the thermodynamic limit.

### HOTRG (Higher-Order TRG)

HOTRG improves upon TRG by using a higher-order SVD (Tucker decomposition)
for the coarse-graining step. This preserves more information and gives
better accuracy at the same χ.

```python
from tenax import hotrg, HOTRGConfig  # HOTRG is a separate module

config = HOTRGConfig(max_bond_dim=16, num_steps=20)
f_hotrg = -float(hotrg(T, config)) / beta   # hotrg also returns ln(Z)/N
```

### GILT / Gilt-TNR (new in v0.9.0)

Plain TRG plateaus near criticality: corner-double-line (CDL) short-range
entanglement accumulates and a larger χ does not help. Gilt-TNR applies the
graph-independent local truncation (Hauru–Delcamp–Mizera) to each plaquette
before every TRG step. `gilt_tnr` is a drop-in counterpart of `trg` (same
input, same `ln(Z)/N` output):

```python
from tenax import gilt_tnr, GiltTNRConfig, GiltConfig

config = GiltTNRConfig(max_bond_dim=8, num_steps=20,
                       gilt=GiltConfig(gilt_eps=1e-6))
f_gilt = -float(gilt_tnr(T, config)) / beta
```

Measured at β_c (error in ln Z per site vs Onsager): χ=8 Gilt-TNR 5.5e-5 vs
plain TRG 2.1e-3; χ=12 4.8e-6 vs 2.1e-3. It is several times slower per step
than TRG (χ=8, 20 steps: ~44 s vs ~5 s on CPU). Works on `DenseTensor` and
bosonic `SymmetricTensor`; fermionic symmetries raise `NotImplementedError`.
Lower-level pieces: `gilt_plaquette`, `gilt_tnr_step`.

---

## Stage 3: Study the Phase Transition

### Temperature scan

```python
import numpy as np

betas = np.linspace(0.3, 0.6, 50)
free_energies = []
config = TRGConfig(max_bond_dim=24, num_steps=20)

for beta in betas:
    T = compute_ising_tensor(beta)
    f = -float(trg(T, config)) / beta
    free_energies.append(f)

# Plot f(β) — look for the non-analyticity at β_c
```

### Identifying T_c

The free energy itself is continuous at T_c (2D Ising is a second-order
transition), but its second derivative (specific heat) diverges.

```python
# Numerical second derivative of ln(Z)/N = -β f
log_z = -betas * np.array(free_energies)
C = betas**2 * np.gradient(np.gradient(log_z, betas), betas)
# The peak of C locates T_c
```

### Specific heat

C = β² ∂²(ln Z/N)/∂β² = -β² ∂²(βf)/∂β² (per site). The 2D Ising model has a logarithmic divergence:
C ∼ -ln|T - T_c|.

### Magnetization (order parameter)

To extract the magnetization, insert a modified tensor with a spin-weighted
trace. This is more involved — guide students through the impurity tensor
technique if they need it.

---

## Stage 4: Convergence Studies

### Bond dimension convergence

```python
beta = 0.4407  # At criticality — hardest case
for chi in [4, 8, 16, 32, 64]:
    config = TRGConfig(max_bond_dim=chi, num_steps=20)
    T = compute_ising_tensor(beta)
    f = -float(trg(T, config)) / beta
    print(f"χ = {chi:3d}: f = {f:.10f}")
```

Away from T_c, even χ=4 gives good results. At T_c, large χ is needed because
correlations are long-range and the truncation error is largest — and plain
TRG stalls there (measured: f error 4.8e-3 at both χ=8 and χ=16). That plateau
is the CDL accumulation; Gilt-TNR removes it.

### TRG vs HOTRG comparison

```python
for chi in [8, 16, 32]:
    T = compute_ising_tensor(0.4407)
    f_trg = -float(trg(T, TRGConfig(max_bond_dim=chi, num_steps=20))) / 0.4407
    f_hotrg = -float(hotrg(T, HOTRGConfig(max_bond_dim=chi, num_steps=20))) / 0.4407
    print(f"χ={chi}: TRG={f_trg:.10f}, HOTRG={f_hotrg:.10f}")
```

HOTRG should be more accurate at the same χ, especially near criticality
(measured at β_c, χ=8: f error 2.1e-3 vs TRG 4.8e-3; Gilt-TNR 1.2e-4).

### Exact result for validation

The exact free energy of the 2D Ising model (Onsager):

    f_exact(β) = -kT [ln(2) + (1/2π²) ∫∫ ln(cosh²(2βJ) - sinh(2βJ)(cos θ₁ + cos θ₂)) dθ₁ dθ₂]

At β_c = 0.4407: f ≈ -2.10961 (per site, J=1). Students should compare their TRG result.

Tenax provides a built-in exact solution for validation:

```python
from tenax import ising_free_energy_exact

f_exact = ising_free_energy_exact(beta=0.4407)   # f = -ln(Z)/(N β)
print(f"Exact free energy: {f_exact:.10f}")      # -2.1096089420
```

`ising_free_energy_exact` returns the free energy `f`, not `ln(Z)/N`: compare it
with `-trg(...) / beta`.

---

## Stage 5: Beyond the Ising Model

### Potts models (q states)

Build a q×q Boltzmann weight matrix and decompose into a 4-index tensor.
Z_q symmetry can be exploited.

### Clock models

Similar to Potts but with continuous-like angular variables discretized to
q orientations. Shows Berezinskii-Kosterlitz-Thouless transitions for q ≥ 5.

### Classical dimer models

The tensor encodes the constraint that each site is covered by exactly one dimer.

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| Free energy wrong far from T_c | Too few RG steps | Increase `num_steps` |
| Free energy wrong near T_c | χ too small, or the TRG CDL plateau | Increase `max_bond_dim`; if the error stops shrinking with χ, switch to `gilt_tnr` |
| "Free energy" ≈ +0.93 at β_c, off by a factor and sign | Printed `trg()`'s `ln(Z)/N` directly | `f = -trg(T, cfg) / beta` |
| NaN or Inf in free energy | Overflow in exp(βJ) at large β | Use log-space arithmetic |
| Specific heat peak at wrong T | Finite-χ shift | Increase χ and extrapolate |

## Pedagogical Notes

- TRG is the tensor network realization of Kadanoff's block-spin RG idea.
  Each step is literally a block-spin transformation.
- The bond dimension χ plays the role of the number of kept states — analogous
  to χ in DMRG but for 2D classical systems.
- Connect to the course: the 2D classical Ising model is equivalent (via
  transfer matrix) to the 1D quantum Ising model. TRG on the classical side
  ↔ DMRG on the quantum side.
