---
name: tenax-ipeps-workflow
description: >
  Guide graduate students through iPEPS calculations for 2D quantum lattice
  models using Tenax. Covers the full pipeline: simple update (imaginary time
  evolution), AD-based ground-state optimization via optimize_gs_ad with CTM
  environments, and quasiparticle excitation spectra via compute_excitations.
  Supports 1-site and 2-site unit cells. Use this skill when the user mentions
  iPEPS, PEPS, 2D tensor networks, projected entangled pair states, corner
  transfer matrix (CTM), simple update, imaginary time evolution for 2D,
  AD optimization of tensor networks, quasiparticle excitations, or Brillouin
  zone dispersion. Also trigger for "2D Heisenberg ground state", "Neel order",
  "2D phase diagram", or any request to go beyond 1D DMRG to 2D systems.
---

# iPEPS Workflow — 2D Ground States and Excitations with Tenax

This skill guides students through infinite Projected Entangled Pair State
(iPEPS) calculations using Tenax. iPEPS is the natural tensor network ansatz
for 2D quantum systems in the thermodynamic limit.

## When to Use iPEPS vs Cylinder DMRG

| | Cylinder DMRG | iPEPS |
|---|---|---|
| **Geometry** | Finite cylinder (open x, periodic y) | Infinite 2D plane |
| **Finite-size effects** | Yes (finite Lx and Ly) | No (infinite, but finite D) |
| **Best for** | Quasi-1D, moderate Ly | Truly 2D, ordered phases |
| **Main parameter** | Bond dim χ (MPS) | Bond dim D (PEPS) + χ (CTM) |
| **Tenax function** | `dmrg()` with cylinder MPO | `ipeps()` or `optimize_gs_ad()` |

Rule of thumb: if the system has clear 2D order (Néel, VBS) or you want the
thermodynamic limit directly, use iPEPS. If you need high accuracy for finite
systems or entanglement entropy, use cylinder DMRG.

---

## Stage 1: Define the Model and Build the Gate

iPEPS works with nearest-neighbor gates (two-site operators), not MPOs.

### Pre-built gates

Tenax provides ready-made gates for common models:

```python
from tenax import heisenberg_gate, xxz_gate

# Isotropic Heisenberg: H = Sz Sz + 0.5 (S+ S- + S- S+)
gate = heisenberg_gate()

# Anisotropic XXZ: H = delta * Sz Sz + 0.5 (S+ S- + S- S+)
gate_xxz = xxz_gate(delta=1.5)  # delta=1.0 recovers Heisenberg
```

Both return a `DenseTensor` with shape `(2, 2, 2, 2)` and labels
`(si, sj, si_out, sj_out)`. They also work as plain `jax.Array` gates.

### Building a custom gate

For models without a pre-built gate, construct the two-site operator manually:

```python
import jax.numpy as jnp

# Spin-1/2 operators
Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])

# Custom gate example: J1-J2 nearest-neighbor part
gate = jnp.einsum("ij,kl->ikjl", Sz, Sz) \
     + 0.5 * (jnp.einsum("ij,kl->ikjl", Sp, Sm)
             + jnp.einsum("ij,kl->ikjl", Sm, Sp))
```

The gate has shape `(d, d, d, d)` = `(2, 2, 2, 2)` for spin-1/2. The indices
are `(ket_i, ket_j, bra_i, bra_j)`.

### Physics checkpoint

Ask the student: "Is this a frustrated model? Does the ground state break a
symmetry (e.g., Néel order for Heisenberg)? This determines the unit cell size."

---

## Stage 2: Simple Update (Imaginary Time Evolution)

The simple update is the fastest way to get an initial iPEPS. It applies
imaginary time evolution gates e^{-δτ h} to evolve toward the ground state,
using a Trotter decomposition.

### 1-site unit cell

```python
from tenax import iPEPSConfig, CTMConfig, ipeps

config = iPEPSConfig(
    max_bond_dim=2,              # iPEPS bond dimension D
    num_imaginary_steps=200,     # Number of Trotter steps
    dt=0.3,                      # Imaginary time step δτ
    ctm=CTMConfig(chi=10, max_iter=40),  # CTM environment params
    unit_cell="1x1",             # Translationally invariant
)
energy, peps, envs = ipeps(gate, None, config)
print(f"Energy per site: {energy:.6f}")
```

### 2-site unit cell (checkerboard)

For states with broken translational symmetry (e.g., Néel order):

```python
config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.3,
    ctm=CTMConfig(chi=10, max_iter=40),
    unit_cell="2site",           # A-B checkerboard
)
energy, peps, (env_A, env_B) = ipeps(gate, None, config)
print(f"Energy per site: {energy:.6f}")  # ≈ -0.65 for Heisenberg
```

### Key parameters

| Parameter | Role | Guidance |
|-----------|------|----------|
| `max_bond_dim` (D) | iPEPS expressiveness | Start D=2, increase to 3,4,5 |
| `dt` | Trotter time step | Start 0.3, reduce to 0.1, 0.01 for accuracy |
| `num_imaginary_steps` | Evolution length | 200–500; increase if not converged |
| `ctm.chi` | CTM environment bond dim | χ ≥ D² for accuracy; start 10–20 |
| `ctm.max_iter` | CTM convergence iterations | 40–100 |
| `unit_cell` | Translational symmetry | "1x1" or "2site" |

### What to watch for

- **Energy not decreasing** → δτ too large (Trotter error dominates). Reduce dt.
- **Energy oscillating** → try a 2-site unit cell if using 1-site (maybe the
  ground state breaks translational symmetry).
- **Energy converged but too high** → D too small, or CTM χ too small.

---

## Stage 3: AD-Based Ground-State Optimization

The simple update gives a good initial state but is approximate (it ignores
the environment during updates). For higher accuracy, use automatic
differentiation (AD) to variationally optimize the iPEPS tensors directly.

Tenax implements the method of Francuz et al. (PRR 7, 013237): gradient
optimization via implicit differentiation through the CTM fixed point.

```python
from tenax import iPEPSConfig, CTMConfig, optimize_gs_ad

config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50),  # Higher χ for AD
    gs_num_steps=200,                     # Optimization steps
    gs_learning_rate=1e-3,                # Adam learning rate
)

# Can start from scratch or from simple-update result
A_opt, env, E_gs = optimize_gs_ad(gate, None, config)
print(f"Ground-state energy: {E_gs:.6f}")
```

### Key differences from simple update

- **More accurate** — optimizes the full variational energy, not a local
  approximation.
- **Slower** — each step requires CTM convergence + backward pass through it.
- **Sensitive to initialization** — starting from a simple-update result
  helps avoid bad local minima.
- **Learning rate matters** — if loss oscillates, reduce `gs_learning_rate`.
  If convergence is too slow, increase it.

### Physics insight

Explain to students: "The simple update is like a mean-field approximation
for the environment — each bond is updated independently. The AD optimization
treats the full 2D environment exactly (up to CTM truncation), which is why
it's more accurate but more expensive. This is analogous to going from
Hartree-Fock to a correlated method in quantum chemistry."

---

## Stage 4: Quasiparticle Excitations

Once you have an optimized ground state, Tenax can compute quasiparticle
excitation spectra at arbitrary Brillouin zone momenta, following
Ponsioen et al. (SciPost Phys. 12, 006, 2022).

```python
from tenax import ExcitationConfig, compute_excitations, make_momentum_path

# Define a path through the Brillouin zone
momenta = make_momentum_path("brillouin", num_points=20)

# Compute excitation energies
exc_config = ExcitationConfig(num_excitations=3, chi=16)
result = compute_excitations(A_opt, env, gate, E_gs, momenta, exc_config)

print(result.energies.shape)  # (num_momenta, num_excitations) = (20, 3)
```

### Interpreting results

- **Gapped spectrum** → the lowest excitation energy is > 0 everywhere in
  the BZ. Indicates a gapped phase (e.g., valence bond solid).
- **Gapless at specific k-points** → indicates spontaneous symmetry breaking
  (Goldstone modes) or a critical point.
- **For the Heisenberg antiferromagnet** → expect gapless magnon excitations
  at k = (π, π) (the antiferromagnetic wavevector).

### Plotting the dispersion

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for n in range(result.energies.shape[1]):
    ax.plot(result.energies[:, n], label=f"Band {n}")
ax.set_xlabel("Momentum path index")
ax.set_ylabel("Excitation energy")
ax.legend()
plt.savefig("dispersion.png")
```

---

## Stage 4b: Lattice Abstraction and Multi-Site CTM

For lattices beyond simple square (1-site) or checkerboard (2-site), Tenax
provides a declarative `Lattice` class with built-in geometries and a
`ctm_multisite()` entry point for 3+ site unit cells.

### Built-in lattices

```python
from tenax import square, checkerboard, honeycomb, triangular, kagome

lat = kagome()      # 3 sites (u, v, w) per unit cell
print(lat.sites)    # ('u', 'v', 'w')
print(lat.bonds)    # Bond connectivity
print(lat.neighbor_map)  # site -> {left/right/top/bottom -> neighbor}
```

Available factories: `square()` (1-site), `checkerboard()` (2-site),
`honeycomb()` (2-site), `triangular()` (1-site with diagonal bonds),
`kagome()` (3-site).

### Custom lattices

```python
from tenax import Lattice, Bond

my_lattice = Lattice(
    sites=("a", "b", "c"),
    bonds=(Bond("a", "b", "horizontal"), Bond("b", "c", "vertical")),
    neighbor_map={
        "a": {"left": "c", "right": "b", "top": "c", "bottom": "b"},
        "b": {"left": "a", "right": "c", "top": "a", "bottom": "c"},
        "c": {"left": "b", "right": "a", "top": "b", "bottom": "a"},
    },
)
```

### Multi-site CTM with ctm_multisite()

For 3+ site unit cells, use `ctm_multisite()` instead of `ctm()`.
Note: `ctm_2site()` is the legacy dense CTM used by simple update only;
for AD optimization with 2-site cells, use `optimize_gs_ad()` with
`unit_cell="2site"` which uses the Tensor-protocol multisite CTM.

`ctm_multisite()` accepts string-keyed site tensors and a `Lattice`:

```python
from tenax import ctm_multisite, kagome
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor, ctm_tensor_2site

lat = kagome()
# site_tensors: dict mapping site names to DenseTensor/SymmetricTensor
envs = ctm_multisite(site_tensors, lat, chi=16, max_iter=60)
# envs["u"], envs["v"], envs["w"] are the converged CTMTensorEnv objects
```

For 1-site and 2-site cells, prefer `ctm_tensor()` and `ctm_tensor_2site()`
which are optimized for those cases.

---

## Stage 4c: Split-CTM (Alternative CTM Method)

For large bond dimensions, the split-CTM algorithm keeps ket and bra layers
separate, reducing the projector cost from O(chi^3 D^6) to O(chi^3 D^3):

```python
from tenax import ctm_split, compute_energy_split_ctm, CTMConfig

config = CTMConfig(chi=20, chi_I=40, max_iter=60)
env_split = ctm_split(A, config)
E = compute_energy_split_ctm(A, env_split, gate, d)
```

The `chi_I` parameter controls the interlayer bond dimension. Set `chi_I >= chi`
(typical: `chi_I = 2 * chi`).

---

## Stage 5: Convergence Studies

Guide students through systematic convergence checks:

### D-scaling (iPEPS bond dimension)

```python
for D in [2, 3, 4, 5]:
    config = iPEPSConfig(
        max_bond_dim=D,
        ctm=CTMConfig(chi=D**2 + 4, max_iter=60),
        gs_num_steps=300,
        gs_learning_rate=1e-3,
    )
    A, env, E = optimize_gs_ad(gate, None, config)
    print(f"D={D}: E/site = {E:.8f}")
```

Rule of thumb: CTM χ should be at least D² for reliable results.

### χ-scaling (CTM environment)

Fix D and vary χ to check CTM convergence:
```python
D = 3
for chi in [10, 20, 40, 60]:
    config = iPEPSConfig(
        max_bond_dim=D,
        ctm=CTMConfig(chi=chi, max_iter=80),
        gs_num_steps=200,
        gs_learning_rate=1e-3,
    )
    A, env, E = optimize_gs_ad(gate, None, config)
    print(f"χ={chi}: E/site = {E:.8f}")
```

---

## Reference Values

| Model | iPEPS D=2 | iPEPS D→∞ (extrap.) | QMC/exact |
|-------|-----------|-------------------|-----------|
| 2D Heisenberg | ≈ −0.6548 | ≈ −0.6694 | −0.6694 |
| J1-J2 at J2/J1=0.5 | model-dependent | — | Debated |

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| Energy much too high | D or χ too small | Increase both |
| AD optimization diverges | Learning rate too high | Reduce `gs_learning_rate` |
| CTM not converging | χ too small or max_iter too low | Increase both |
| Wrong symmetry breaking | Wrong unit cell | Try "2site" for AFM order |
| NaN in gradients | SVD degeneracy | Reduce learning rate, check gate is Hermitian |

## Pedagogical Notes

- Connect to solid state: iPEPS is a variational wavefunction for the
  thermodynamic limit, like a Jastrow wavefunction but structured as a
  tensor network.
- The CTM is the 2D analog of the left/right environments in DMRG — it
  summarizes the infinite 2D surroundings of a local patch.
- Excitation spectra from iPEPS are the tensor-network analog of spin-wave
  theory, but non-perturbative.
