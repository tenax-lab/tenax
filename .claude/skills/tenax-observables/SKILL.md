---
name: tenax-observables
description: >
  Guide users through computing physical observables from Tenax ground states:
  local expectation values, correlation functions, entanglement entropy, and
  order parameters from MPS (DMRG/iDMRG) and iPEPS states. Use this skill
  when the user asks "how do I compute observables", "measure correlation
  function", "entanglement entropy", "expectation value", "order parameter",
  "magnetization", "structure factor", or "what can I measure from my ground
  state". Also trigger for "post-DMRG analysis", "reduced density matrix",
  or "how to extract physics from MPS".
---

# Observables and Measurements in Tenax

Guide users through extracting physical quantities from optimized tensor
network states. Tenax provides the ground state as a TensorNetwork (MPS)
or raw tensor (iPEPS); measuring observables requires contracting the
state with operator insertions.

## What's Available from Algorithm Results

### DMRG → DMRGResult

```python
result = dmrg(mpo, mps, config)
result.energy              # Total ground state energy (float)
result.energies_per_sweep  # Energy history (list[float])
result.mps                 # Optimized MPS (TensorNetwork)
result.truncation_errors   # Truncation errors per bond update
result.converged           # Whether DMRG converged (bool)
```

### iDMRG → iDMRGResult

```python
result = idmrg(W, config)
result.energy_per_site     # Energy per site (float)
result.energies_per_step   # Energy history
result.mps_tensors         # [A_L, A_R] unit cell tensors
result.singular_values     # Centre bond singular values (jax.Array)
result.converged           # bool
```

### iPEPS → (energy, peps, env)

```python
energy, peps, env = ipeps(gate, None, config)
# or
A_opt, env, E_gs = optimize_gs_ad(gate, None, config)
```

The CTM environment `env` enables observable computation via reduced
density matrices.

---

## Entanglement Entropy from iDMRG

The singular values on the centre bond give the entanglement entropy
directly — no extra contraction needed.

```python
import jax.numpy as jnp

S = result.singular_values
# Normalize
S_norm = S / jnp.linalg.norm(S)
# Von Neumann entropy
p = S_norm**2
S_ent = -jnp.sum(p * jnp.log(p))
print(f"Entanglement entropy: {S_ent:.6f}")

# For critical systems: S = (c/3) ln(xi) where c is the central charge
# and xi is the correlation length (related to bond dimension)
```

---

## Local Expectation Values from MPS

Tenax provides built-in functions for computing expectation values and
correlation functions from MPS ground states:

```python
import numpy as np
from tenax.algorithms.observables import expectation_value, correlation

# Standard spin-1/2 operators
Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
Sm = np.array([[0.0, 0.0], [1.0, 0.0]])

# ⟨Sz_i⟩ at site 0
sz_val = expectation_value(result.mps, Sz, site=0)
```

These functions work polymorphically on both `DenseTensor` and
`SymmetricTensor` MPS.

> **Note:** `expectation_value()` returns `float(Re(⟨O⟩))` and emits a
> warning if the imaginary part is non-negligible. This catches
> accidental use of non-Hermitian operators.

### Full state vector approach (small systems only)

For systems with L ≤ ~18, you can contract the MPS into a full state
vector and use standard linear algebra:

```python
import jax.numpy as jnp

psi = result.mps.contract().todense().flatten()
# Embed operator in full Hilbert space
Sz_full = build_full_operator(Sz, site=i, L=L)
expectation = jnp.real(psi.conj() @ Sz_full @ psi)
```

---

## Two-Point Correlation Functions

⟨S^z_i S^z_j⟩ measures spin-spin correlations and reveals the nature of
order in the ground state.

### Using the built-in `correlation()` function

```python
from tenax.algorithms.observables import correlation

# ⟨Sz_0 Sz_r⟩ for all r
correlations = [correlation(result.mps, Sz, 0, Sz, r) for r in range(L)]
```

### Fermionic correlators

For anticommuting operators (e.g., `⟨c†_i c_j⟩`), use `anticommute=True`
to get the correct sign when the function reorders operators internally:

```python
Cdag = np.array([[0.0, 0.0], [1.0, 0.0]])  # creation operator
C    = np.array([[0.0, 1.0], [0.0, 0.0]])  # annihilation operator

# ⟨c†_0 c_r⟩ — anticommute ensures correct sign regardless of site ordering
green_fn = [correlation(result.mps, Cdag, 0, C, r, anticommute=True) for r in range(L)]
```

Without `anticommute=True`, the default behavior swaps operators when
`site_i > site_j` without a sign change — correct for bosonic operators
like Sz but wrong for fermions.

### Full state vector approach (small systems)

```python
def correlation_function(psi, op1, op2, site_i, site_j, L, d=2):
    """Compute ⟨ψ|op1_i op2_j|ψ⟩ via exact state vector."""
    from scipy.sparse import kron, eye, csr_matrix

    I = eye(d)
    ops = [I] * L
    ops[site_i] = csr_matrix(op1)
    ops[site_j] = csr_matrix(op2)

    O = ops[0]
    for k in range(1, L):
        O = kron(O, ops[k])

    return float(jnp.real(psi.conj() @ O @ psi))
```

### Expected behavior

| Phase | ⟨S^z_0 S^z_r⟩ | Signature |
|-------|---------------|-----------|
| Antiferromagnetic (gapless, 1D) | ∼ (-1)^r / r | Algebraic decay |
| Gapped (e.g., Haldane) | ∼ e^{-r/ξ} | Exponential decay |
| Ferromagnetic | → const > 0 | Long-range order |
| Néel ordered (2D) | → (-1)^r m² | Staggered long-range order |

---

## iPEPS Observables via CTM

For iPEPS, observables are computed using the CTM environment and reduced
density matrices.

### Energy (built-in)

```python
from tenax import compute_energy_ctm_2site

# For 2-site unit cell
energy = compute_energy_ctm_2site(peps_a, peps_b, gate, env_a, env_b)
```

### Custom observables from RDMs

The CTM environment enables computing 2-site reduced density matrices
(RDMs). From a 2-site RDM ρ, any 2-site observable is:

    ⟨O⟩ = Tr(ρ · O)

```python
# Staggered magnetization for Néel order
Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]])
# m_s = (1/2)|⟨Sz_A⟩ - ⟨Sz_B⟩| for A-B sublattice
```

---

## Order Parameters

### Staggered magnetization (Néel order)

```python
# m_stag = (1/L) Σ (-1)^i ⟨Sz_i⟩
m_stag = sum((-1)**i * expect_Sz[i] for i in range(L)) / L
```

For the 1D Heisenberg chain, m_stag → 0 as L → ∞ (no Néel order in 1D,
Mermin-Wagner theorem). For 2D (iPEPS with 2-site unit cell), m_stag ≈ 0.307.

### String order parameter (Haldane phase)

For spin-1 chains, the Haldane phase has hidden topological order detected
by the string order parameter:

    O_string = ⟨S^z_i exp(iπ Σ_{k=i+1}^{j-1} S^z_k) S^z_j⟩

This requires inserting a non-local string of operators into the MPS
contraction — a good exercise for the NetworkBlueprint.

---

## Structure Factor

The static structure factor S(q) is the Fourier transform of the
real-space correlation function:

```python
import numpy as np

# From real-space correlations C(r) = ⟨Sz_0 Sz_r⟩
def structure_factor(correlations, q_values, L):
    """S(q) = Σ_r exp(iqr) C(r)"""
    S_q = []
    for q in q_values:
        s = sum(np.exp(1j * q * r) * correlations[r] for r in range(L))
        S_q.append(float(np.real(s)))
    return S_q

q_values = np.linspace(0, 2 * np.pi, 100)
S_q = structure_factor(correlations, q_values, L)
# Peak at q=π → antiferromagnetic correlations
```

---

## Practical Workflow

1. **Run DMRG/iDMRG/iPEPS** to get the ground state.
2. **Check energy** — compare against known exact results or extrapolate
   in bond dimension.
3. **Compute entanglement entropy** — from singular values (iDMRG) or
   by SVD of the MPS at a bond (finite DMRG).
4. **Measure correlations** — ⟨S^z_0 S^z_r⟩ to identify the phase.
5. **Compute order parameters** — staggered magnetization, string order, etc.
6. **Structure factor** — Fourier transform of correlations for momentum-space
   picture.

## Pedagogical Notes

- Observables are where the physics lives. The energy alone doesn't tell you
  what phase you're in — you need correlations and order parameters.
- For 1D systems, the entanglement entropy scaling (constant vs logarithmic)
  immediately tells you if the system is gapped or gapless.
- Always compare observables across multiple bond dimensions to assess
  convergence. If ⟨Sz_0 Sz_r⟩ changes significantly when you double χ,
  you need more bond dimension.
