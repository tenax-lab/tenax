---
name: tenax-autompo
description: >
  Build Hamiltonian MPOs from natural-language model descriptions using Tenax's
  AutoMPO. Translates physics ("Heisenberg ladder with NNN coupling and staggered
  field") into correct AutoMPO code with proper prefactors, site ordering, custom
  operators, compress and symmetric flags. Use this skill whenever the user wants
  to build an MPO, construct a Hamiltonian, set up a spin model, define coupling
  terms, create an operator for DMRG, or asks about AutoMPO, build_auto_mpo,
  operator terms, Sp Sm Sz conventions, site ordering for cylinders/ladders, or
  "how do I write the Hamiltonian for [model]". Also trigger for custom operators,
  NNN interactions, long-range couplings, staggered fields, or multi-site unit cells.
---

# AutoMPO Builder — Hamiltonian Construction for Tenax

You help users build correct MPO Hamiltonians using Tenax's `AutoMPO` class and
`build_auto_mpo` functional interface. Your primary job is translating a physical
model description into bug-free AutoMPO code.

## Tenax AutoMPO API

### Class-based interface

```python
from tenax import AutoMPO

L = 20       # number of sites
auto = AutoMPO(L=L, d=2)  # d = local Hilbert space dimension

# Add terms: (coefficient, "Op1", site1, "Op2", site2, ...)
auto += (1.0, "Sz", 0, "Sz", 1)       # Two-site term
auto += (0.5, "Sz", 3)                  # Single-site term (custom names like
                                        # "X" need AutoMPO(..., site_ops=...))

mpo = auto.to_mpo()                     # Dense MPO
mpo = auto.to_mpo(symmetric=True)       # U(1) block-sparse MPO
mpo = auto.to_mpo(compress=True)        # Compressed (for long-range terms)
```

### Functional interface

```python
from tenax import build_auto_mpo
import numpy as np

custom_ops = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "Id": np.eye(2),
}
terms = [(1.0, "Z", i, "Z", i + 1) for i in range(L - 1)]
terms += [(0.5, "X", i) for i in range(L)]
mpo = build_auto_mpo(terms, L=L, site_ops=custom_ops)
```

### Built-in operators

Use `spin_half_ops()` (d=2) and `spin_one_ops()` (d=3) to get operator dictionaries:

```python
from tenax import spin_half_ops, spin_one_ops

ops = spin_half_ops()  # {"Sz", "Sp", "Sm", "Id"}
```

**Spin-1/2 (d=2):**

| Name | Matrix | Notes |
|------|--------|-------|
| `"Sz"` | diag(1/2, -1/2) | z-component |
| `"Sp"` | [[0,1],[0,0]] | Raising operator S+ |
| `"Sm"` | [[0,0],[1,0]] | Lowering operator S- |
| `"Id"` | eye(2) | Identity |

**Spin-1 (d=3):**

| Name | Matrix | Notes |
|------|--------|-------|
| `"Sz"` | diag(1, 0, -1) | z-component |
| `"Sp"` | 3×3 raising | S+ for spin-1 |
| `"Sm"` | 3×3 lowering | S- for spin-1 |
| `"Id"` | eye(3) | Identity |

Note: `"Sx"` and `"Sy"` are **not** built-in. If needed, define them as custom
operators or use the Sp/Sm decomposition (recommended to avoid complex numbers).

## Translation Rules

When the user describes a model, apply these rules:

### Heisenberg exchange: S_i · S_{i+1}

**Always decompose as:**
```python
auto += (Jz,  "Sz", i, "Sz", j)
auto += (Jxy/2, "Sp", i, "Sm", j)   # Factor of 0.5!
auto += (Jxy/2, "Sm", i, "Sp", j)   # Factor of 0.5!
```

The 0.5 comes from: S·S = Sz·Sz + (1/2)(S+S- + S-S+). This is the single
most common AutoMPO mistake. Always use 0.5 for the Sp/Sm terms unless the
user explicitly asks for XX+YY coupling (which would be 1.0 each).

For the isotropic Heisenberg model: Jz = Jxy = J.
For XXZ: Jz ≠ Jxy (anisotropic).

### Site ordering constraint

AutoMPO sorts each term's operators by site itself, so the order you write
them in does not matter. Writing `min(i,j)` / `max(i,j)` keeps the code
readable:

```python
auto += (J, "Sz", min(i,j), "Sz", max(i,j))
```

### Avoiding complex numbers

Prefer the Sp/Sm decomposition over Sx/Sy. Using "Sy" introduces complex
numbers, which can cause `UFuncOutputCastingError` with NumPy >= 2.0 and
forces complex128 throughout.

**Instead of:**
```python
auto += (Jx, "Sx", i, "Sx", j)
auto += (Jy, "Sy", i, "Sy", j)
```

**Use:**
```python
auto += (Jz, "Sz", i, "Sz", j)
auto += ((Jx + Jy)/2, "Sp", i, "Sm", j)  # = (Jx+Jy)/2 * (S+S- + S-S+)/2... 
auto += ((Jx + Jy)/2, "Sm", i, "Sp", j)  # Careful: only valid if Jx = Jy
```

For fully anisotropic XYZ models where Jx ≠ Jy, you may need to use Sx/Sy
directly with complex128 dtype, or decompose carefully:
Sx·Sx = (1/4)(S+S+ + S+S- + S-S+ + S-S-)
Sy·Sy = -(1/4)(S+S+ - S+S- - S-S+ + S-S-)

### Geometry mappings

**1D chain (open BC):**
```python
for i in range(L - 1):
    add_bond(i, i + 1)
```

**1D chain (periodic BC):**
```python
for i in range(L):
    add_bond(i, (i + 1) % L)
```
Note: periodic BC creates a long-range term (site 0 ↔ site L-1) so use
`compress=True`.

**Ladder (2-leg, open rungs):**
```python
# Sites: leg=0 → 0,2,4,...  leg=1 → 1,3,5,...
# Rung bond: (2*x, 2*x+1)
# Leg bond:  (2*x, 2*(x+1)), (2*x+1, 2*(x+1)+1)
for x in range(Lx):
    add_bond(2*x, 2*x + 1)          # rung
    if x < Lx - 1:
        add_bond(2*x, 2*(x+1))      # leg 0
        add_bond(2*x+1, 2*(x+1)+1)  # leg 1
```

**Cylinder (Lx × Ly, periodic in y, open in x):**
```python
Lx, Ly = 6, 3
N = Lx * Ly
for x in range(Lx):
    for y in range(Ly):
        i = x * Ly + y
        j = x * Ly + (y + 1) % Ly       # periodic y
        add_bond(min(i,j), max(i,j))     # within-ring
        if x < Lx - 1:
            k = (x + 1) * Ly + y
            add_bond(i, k)               # between-ring
```

### Flags

- **`symmetric=True`** — generates U(1) block-sparse MPO. Use when the
  Hamiltonian conserves total Sz (most spin models without transverse fields).
  The initial MPS must be in the correct Sz sector.

- **`compress=True`** — compresses the MPO via SVD. Essential for:
  periodic boundaries, long-range interactions, cylinders with large Ly.
  Without compression, the MPO bond dimension can be unnecessarily large.

## Workflow

1. Ask the user to describe the model (or identify it from context).
2. Identify: lattice geometry, coupling terms, symmetries, boundary conditions.
3. Generate the AutoMPO code with correct prefactors and site ordering.
4. Recommend `symmetric` and `compress` flags based on the model.
5. Suggest a sanity check: run DMRG on a small system and compare against
   exact diagonalization or known results.

## Common Models Quick Reference

### Transverse-field Ising
```python
# "Sx" is not built in: use Sx = (Sp + Sm)/2
for i in range(L - 1):
    auto += (-J, "Sz", i, "Sz", i + 1)
for i in range(L):
    auto += (-h / 2, "Sp", i)
    auto += (-h / 2, "Sm", i)
```

### XXZ model
```python
for i in range(L - 1):
    auto += (delta, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
```

### Spinless fermions
`fermion_site_ops()` gives `C`, `Cd`, `N`, `F`, `Id` (d=2). Declare the
fermionic names and AutoMPO inserts the Jordan-Wigner `F` strings between them:

```python
from tenax import AutoMPO, fermion_site_ops

auto = AutoMPO(L=L, d=2, site_ops=fermion_site_ops(), fermionic_ops={"C", "Cd"})
for i in range(L - 1):
    auto += (-t, "Cd", i, "C", i + 1)   # -t c†_i c_{i+1}
    auto += (-t, "Cd", i + 1, "C", i)   # -t c†_{i+1} c_i
```

Write every two-fermion term **creation-first**, `(coef, "Cd", i, "C", j)` for
c†_i c_j, in either site order. AutoMPO sorts the operators by site and inserts
the `F` strings between them, but it does not supply the sign from
anticommuting the operators themselves, so a term written annihilation-first
silently becomes a different Hamiltonian: `(+t, "C", j, "Cd", i)` is *not*
read as c_j c†_i = −c†_i c_j. Measured at L=6 with NN hopping 1.0, NNN hopping
0.6 and μ=0.3: creation-first reproduces the exact free-fermion energy
−4.496054; the same NNN terms written annihilation-first give −5.051679.
This applies to **hopping** terms: anticommute a c c† ordering into c† c
yourself. Pairing terms (c_i c_j, c†_i c†_j) cannot be put in that form,
since anticommuting changes the order of the operators, not their type.
They need their own construction: keep the operator types, supply the
left-endpoint parity factor and the ordering sign by hand (see the Hubbard
recipe below for the endpoint factor), and check against ED on a small
chain. This skill gives no verified pairing recipe.

### Hubbard (d=4: |0⟩, |↑⟩, |↓⟩, |↑↓⟩)
There is no built-in Hubbard builder; pass custom d=4 operators as `site_ops`.
Two signs are yours to supply, because AutoMPO only inserts `F` on the sites
strictly *between* the two operators:

1. **Within a site**, build the up/down order into the matrices (Jordan-Wigner
   over the two modes, up first): `c_dn` carries the up-parity factor.
2. **At the left endpoint**, for i < j the Jordan-Wigner string also covers
   site i: c†_i c_j = (c† F)_i … c_j and c†_j c_i = (F c)_i … c†_j. For
   spinless fermions c†F = c† and Fc = c, which is why the spinless recipe
   needs nothing extra; for d=4 they differ, so register the dressed
   operators and put them on the left site.

```python
import numpy as np
from tenax import AutoMPO

a, Z, I2 = np.array([[0., 1.], [0., 0.]]), np.diag([1., -1.]), np.eye(2)
P = np.eye(4)[[0, 2, 1, 3]]                 # (n_up, n_dn) order -> |0>,|up>,|dn>,|updn>
loc = lambda M: P @ M @ P.T
cu, cd = loc(np.kron(a, I2)), loc(np.kron(Z, a))   # c_dn carries the up parity
F = loc(np.kron(Z, Z)); nu, nd = cu.T @ cu, cd.T @ cd
ops = {"Cu": cu, "Cdu": cu.T, "Cd": cd, "Cdd": cd.T, "F": F, "Id": np.eye(4),
       "CduF": cu.T @ F, "CddF": cd.T @ F, "FCu": F @ cu, "FCd": F @ cd,
       "Onsite": U * nu @ nd - mu * (nu + nd)}
auto = AutoMPO(L=L, d=4, site_ops=ops,
               fermionic_ops={"Cu", "Cdu", "Cd", "Cdd", "CduF", "CddF", "FCu", "FCd"})
for i in range(L - 1):                      # i < j = i + 1
    for cdag, c in [("Cdu", "Cu"), ("Cdd", "Cd")]:
        auto += (-t, cdag + "F", i, c, i + 1)   # c†_i c_j
        auto += (-t, "F" + c, i, cdag, i + 1)   # c†_j c_i
for i in range(L):
    auto += (1.0, "Onsite", i)
```

Measured against ED at L=3 (NN t=1, NNN t=0.5, U=2, μ=0.4): this gives
−3.643050 (ED −3.643050); the same terms with bare `Cdu`/`Cu` on the left
site give −3.169031. Check any new fermionic model against ED on a small
chain the same way.

### Spin-1 (d=3)
```python
auto = AutoMPO(L=L, d=3)  # d=3 for spin-1
# Built-in Sz, Sp, Sm are 3×3 for d=3
```
