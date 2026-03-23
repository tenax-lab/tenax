---
name: tenax-ed-comparator
description: >
  Run exact diagonalization (ED) and DMRG side-by-side on small quantum systems
  to teach students why DMRG works and where it breaks down. Compares ground
  state energies, wavefunctions, entanglement spectra, and correlation functions
  between the exact result and the DMRG approximation at various bond dimensions.
  Use this skill when the user wants to validate DMRG, compare exact vs
  approximate results, understand truncation error, study entanglement, or asks
  "is my DMRG result correct", "how accurate is DMRG at bond dimension χ",
  "what does the entanglement spectrum look like", or "show me why DMRG works".
  Also trigger for exact diagonalization, ED, Lanczos, full diagonalization,
  wavefunction overlap, fidelity, or Schmidt decomposition.
---

# Exact Diag ↔ DMRG Comparator

Run exact diagonalization and DMRG on the same small system to teach students
what DMRG does well, what it misses, and why bond dimension matters.

## When to Use This

- **Validating a new Hamiltonian** — check that AutoMPO gives the right answer
  before scaling to large systems.
- **Teaching DMRG** — show concretely how truncation affects accuracy.
- **Understanding entanglement** — compare the exact Schmidt spectrum with
  the truncated one from DMRG.

**Size constraint:** ED is limited to L ≤ 14 (spin-1/2) or L ≤ 10 (spin-1)
due to exponential memory scaling (2^L states). DMRG has no such limit.

---

## Stage 1: Build the Hamiltonian (Both Ways)

### For DMRG: use AutoMPO

```python
import jax
from tenax import AutoMPO, DMRGConfig, FiniteMPS, dmrg

L = 12  # Small enough for ED
auto = AutoMPO(L=L, d=2)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()
```

### For ED: build the full Hamiltonian matrix

```python
import numpy as np
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh

d = 2
Sz = csr_matrix(0.5 * np.array([[1, 0], [0, -1]]))
Sp = csr_matrix(np.array([[0, 1], [0, 0]], dtype=float))
Sm = csr_matrix(np.array([[0, 0], [1, 0]], dtype=float))
I = eye(d)

def build_full_hamiltonian(L):
    """Build the Heisenberg Hamiltonian as a sparse 2^L × 2^L matrix."""
    dim = d ** L
    H = csr_matrix((dim, dim))
    for i in range(L - 1):
        # Sz_i Sz_{i+1}
        op = embed_two_site(Sz, Sz, i, i + 1, L)
        H += op
        # 0.5 * (Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
        H += 0.5 * embed_two_site(Sp, Sm, i, i + 1, L)
        H += 0.5 * embed_two_site(Sm, Sp, i, i + 1, L)
    return H

def embed_two_site(op1, op2, i, j, L):
    """Embed a two-site operator at sites i, j into the full Hilbert space."""
    ops = [I] * L
    ops[i] = op1
    ops[j] = op2
    result = ops[0]
    for k in range(1, L):
        result = kron(result, ops[k])
    return result

H_full = build_full_hamiltonian(L)
E_exact, psi_exact = eigsh(H_full, k=1, which='SA')
E_exact = E_exact[0]
print(f"Exact ground state energy: {E_exact:.12f}")
```

---

## Stage 2: Run DMRG at Various Bond Dimensions

```python
chis = [2, 4, 8, 16, 32, 64]
results = {}

for chi in chis:
    key = jax.random.PRNGKey(chi)  # different key per bond dim
    mps = FiniteMPS.random(L=L, d=2, chi=min(chi, 4), key=key)
    config = DMRGConfig(max_bond_dim=chi, num_sweeps=15)
    result = dmrg(mpo, mps, config)
    results[chi] = result
    error = abs(result.energy - E_exact)
    print(f"χ = {chi:4d}: E = {result.energy:.12f}, "
          f"|ΔE| = {error:.2e}, "
          f"relative error = {error/abs(E_exact):.2e}")
```

### Expected behavior

- **χ = 2:** Poor energy, relative error ~10⁻².
- **χ = 4–8:** Rapid improvement; captures dominant entanglement.
- **χ = 16–32:** Near-exact for L=12 Heisenberg (the exact Schmidt rank at
  the midpoint is at most 2^(L/2) = 64).
- **χ ≥ 2^(L/2):** Exact — DMRG becomes equivalent to ED (the MPS can
  represent any state).

---

## Stage 3: Compare Entanglement Spectra

The entanglement spectrum (Schmidt values across the midpoint cut) reveals
what DMRG keeps and what it truncates.

### From ED

```python
# Reshape the exact wavefunction into a bipartite form
psi = psi_exact.reshape(d**(L//2), d**(L//2))
U, S_exact, Vh = np.linalg.svd(psi, full_matrices=False)
print("Exact Schmidt values:", S_exact[:10])
print(f"Number of nonzero Schmidt values: {np.sum(S_exact > 1e-14)}")
```

### From DMRG

The DMRG result's MPS contains the Schmidt values implicitly in its canonical
form. After convergence, the singular values at the midpoint bond give the
DMRG entanglement spectrum (truncated to χ values).

### Comparison

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogy(S_exact[:30], 'ko-', label='Exact')
for chi in [4, 8, 16]:
    # Plot the first chi Schmidt values from DMRG
    ax.semilogy(range(chi), S_dmrg[chi], 's-', label=f'DMRG χ={chi}')
ax.set_xlabel("Schmidt value index")
ax.set_ylabel("Schmidt value (log scale)")
ax.legend()
ax.set_title("Entanglement spectrum: Exact vs DMRG")
plt.savefig("entanglement_spectrum.png")
```

**Teaching point:** The Schmidt values decay rapidly for gapped systems
(exponentially) and slowly for gapless systems (algebraically). This is *why*
DMRG works: it keeps the largest Schmidt values, and if they decay fast, a
small χ suffices.

---

## Stage 4: Wavefunction Overlap (Fidelity)

For small systems, we can compute the overlap |⟨ψ_exact | ψ_DMRG⟩|² to
measure how close the DMRG state is to the true ground state.

```python
# Convert the DMRG MPS to a full state vector
# (only possible for small L due to exponential size)
psi_dmrg_full = mps_to_full_vector(result.mps)  # Custom helper needed

fidelity = abs(np.dot(psi_exact.conj().flatten(), psi_dmrg_full))**2
print(f"Fidelity: {fidelity:.10f}")
```

**Teaching point:** Fidelity can be high (>0.999) even when the energy error
is noticeable. This is because the energy is a sum over many local terms,
and small errors in the wavefunction accumulate. Conversely, having the exact
energy doesn't guarantee the exact wavefunction (degeneracies, excited states).

---

## Stage 5: Correlation Functions

Compare ⟨S^z_0 S^z_r⟩ from ED and DMRG:

### From ED

```python
correlations_exact = []
for r in range(L):
    Sz0_Szr = embed_two_site(Sz, Sz, 0, r, L) if r > 0 else embed_one_site(Sz @ Sz, 0, L)
    C = psi_exact.conj() @ Sz0_Szr @ psi_exact
    correlations_exact.append(float(C.real))
```

### From DMRG

Compute expectation values using the MPS structure (efficient — does not
require expanding to the full Hilbert space).

### Comparison

Plot both on the same axes. For the Heisenberg chain:
- ⟨S^z_0 S^z_r⟩ ∼ (-1)^r / r for large r (algebraic decay, gapless).
- At small χ, DMRG underestimates long-range correlations (truncation cuts
  off entanglement that mediates them).

---

## Stage 6: Where DMRG Breaks Down

Use the comparison to illustrate DMRG limitations:

### 1D gapless systems (logarithmic entanglement)

DMRG still works but needs larger χ. The entanglement entropy grows as
S = (c/3) ln(L), so the required χ grows polynomially with L.

### Critical systems in 2D

Cylinder DMRG struggles when Ly is large because entanglement across the
cylinder grows linearly with Ly (area law in 2D). Show this by comparing
ED and DMRG for a 4×3 cluster.

### Volume-law states

Highly excited states or thermal states have volume-law entanglement:
S ∝ L. MPS cannot efficiently represent these. Demonstrate by targeting
an excited state.

---

## Summary Table for Students

| Quantity | ED | DMRG (large χ) | DMRG (small χ) |
|----------|-----|----------------|-----------------|
| Energy | Exact | Near-exact | Variational upper bound |
| Wavefunction | Exact | High fidelity | Missing long-range entanglement |
| Schmidt spectrum | Full | Truncated but accurate top values | Missing tail |
| Correlations | Exact | Accurate at short range | Underestimates long-range |
| Cost | O(2^L) — exponential | O(χ³ L) — polynomial | Fast but approximate |

## Pedagogical Notes

- This comparison is the single most effective way to teach *why* DMRG works.
  Students who see the Schmidt spectrum understand truncation viscerally.
- Emphasize: DMRG is variational — the energy is always an upper bound. If
  DMRG gives a lower energy than someone else's calculation, either they made
  an error or they're using a worse method.
- The crossover point where DMRG beats ED is around L ≈ 14–18 for spin-1/2.
  Below that, ED is exact and faster. Above that, ED is impossible but DMRG
  scales gracefully.
