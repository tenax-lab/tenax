---
name: tenax-symmetry
description: >
  Teach graduate students Tenax's symmetry system from the ground up: U(1) and
  Z_n symmetries, TensorIndex with charges and FlowDirection, SymmetricTensor
  construction and operations, block-sparse SVD and QR, and how symmetries
  connect to physics (charge conservation, selection rules). Use this skill when
  the user asks about SymmetricTensor, U1Symmetry, ZnSymmetry, TensorIndex,
  FlowDirection, block-sparse tensors, charge sectors, symmetry-aware DMRG,
  "what is FlowDirection", "how do charges work", "why use symmetric tensors",
  or wants to understand the block structure of tensors in Tenax.
---

# Tenax Symmetry System — From Physics to Code

This skill teaches the symmetry-aware tensor infrastructure in Tenax,
connecting the abstract math to physical conservation laws and concrete code.

## Why Symmetries Matter

If a Hamiltonian commutes with a symmetry generator (e.g., [H, S^z_total] = 0),
then eigenstates can be labeled by the conserved quantum number. Exploiting this
in tensor networks:

1. **Reduces computation** — only symmetry-allowed blocks are stored and
   contracted, skipping zero blocks entirely.
2. **Targets specific sectors** — DMRG can find the ground state in the Sz=0
   sector directly, without variational competition from other sectors.
3. **Improves numerical stability** — block structure prevents mixing of sectors
   that should be decoupled.

## Stage 1: Symmetry Groups

Tenax supports Abelian symmetries with a clean interface.

### U(1) — continuous symmetry, integer charges

```python
from tenax import U1Symmetry
import numpy as np

u1 = U1Symmetry()
charges = np.array([-1, 0, 1], dtype=np.int32)

# Fusion: adding quantum numbers
print(u1.fuse(charges, charges))  # [-2, 0, 2]

# Dual: conjugate charges (bra vs ket)
print(u1.dual(charges))           # [1, 0, -1]
```

**Physics connection:** U(1) corresponds to conservation of a quantity like
total Sz, total particle number, or total charge. The integer labels are the
eigenvalues of the conserved operator.

### Z_n — discrete cyclic symmetry, charges mod n

```python
from tenax import ZnSymmetry

z3 = ZnSymmetry(3)
a = np.array([1, 2], dtype=np.int32)
b = np.array([2, 2], dtype=np.int32)

print(z3.fuse(a, b))  # [0, 1]  (addition mod 3)
```

**Physics connection:** Z_2 arises in Ising symmetry (spin flip). Z_3 appears
in clock models and some frustrated magnets.

### Fermionic symmetries and product symmetries

Tenax also supports fermionic statistics and composite symmetries:

```python
from tenax import FermionParity, FermionicU1, ProductSymmetry

# Z_2 parity with fermionic exchange signs
fp = FermionParity()

# U(1) particle number with fermionic statistics
fu1 = FermionicU1()

# Composite: e.g., charge U(1) × spin U(1)
prod = ProductSymmetry(U1Symmetry(), U1Symmetry())
```

- `FermionParity` — Z_2 symmetry with fermionic braiding (exchange signs).
- `FermionicU1` — U(1) with fermionic exchange statistics, for models with
  fermion number conservation.
- `ProductSymmetry` — tensor product of multiple symmetry groups.

### Choosing a symmetry

| Physical conservation law | Symmetry | Charges |
|--------------------------|----------|---------|
| Total Sz | U(1) | ..., -1, 0, 1, ... |
| Total particle number | U(1) | 0, 1, 2, ... |
| Fermion number | FermionicU1 | 0, 1, 2, ... |
| Spin-flip (Ising) | Z_2 | 0, 1 |
| Fermion parity | FermionParity | 0, 1 |
| Clock model | Z_n | 0, 1, ..., n-1 |
| Multiple conserved quantities | ProductSymmetry | tuples |
| None / unknown | Don't use SymmetricTensor | — |

---

## Stage 2: TensorIndex — Labeled Legs with Charges

Every leg of a `SymmetricTensor` is a `TensorIndex` that carries:

1. **Symmetry group** — which symmetry (U(1), Z_3, etc.)
2. **Charges** — which quantum numbers this leg can carry
3. **FlowDirection** — IN or OUT (bra vs ket)
4. **Label** — a string for automatic contraction

```python
from tenax import TensorIndex, FlowDirection

u1 = U1Symmetry()
phys_charges = np.array([-1, 1], dtype=np.int32)  # spin-1/2: Sz = ±1/2 (×2)
bond_charges = np.array([-1, 0, 1], dtype=np.int32)

# Physical leg (ket side — incoming)
phys = TensorIndex.from_charges(u1, phys_charges, FlowDirection.IN, label="phys")

# Bond leg (outgoing to the right)
bond_R = TensorIndex.from_charges(u1, bond_charges, FlowDirection.OUT, label="bond_R")

# Bond leg (incoming from the left)
bond_L = TensorIndex.from_charges(u1, bond_charges, FlowDirection.IN, label="bond_L")
```

### FlowDirection: the bra-ket convention

This is the most conceptually tricky part. Think of it as:

- **IN** = ket = |ψ⟩ side = "quantum number flows into the tensor"
- **OUT** = bra = ⟨ψ| side = "quantum number flows out of the tensor"

**Contraction rule:** When two legs contract, one must be IN and the other OUT.
This ensures charge conservation: what flows in must flow out.

```
  IN ──→ [Tensor A] ──→ OUT
                         ↕  (contraction: OUT connects to IN)
  IN ──→ [Tensor B] ──→ OUT
```

If both legs are IN (or both OUT), the contraction violates charge conservation
and will produce wrong results or errors.

The partner leg must have the **opposite flow and identical charges** — what
`idx.flip_flow()` (or `bar()` on a tensor) produces. `idx.dual()` also negates
the charges, and block-sparse contraction pairs blocks by charge *value* while
dense pairs by *position*, so mixing the two conventions gives a silently
representation-dependent answer (#834). To audit a path, run with
`TENAX_STRICT_CONTRACT=1`, which makes `contract()` raise `ValueError` when the
two representations would disagree (opt-in; not for production).

### Charge arrays

The charge array lists all quantum numbers that a leg can carry. For spin-1/2
with U(1) Sz symmetry:

- Physical leg: `[-1, 1]` (representing Sz = -1/2 and +1/2, scaled by 2)
- Bond leg: `[-2, -1, 0, 1, 2]` (all Sz values the bond can carry)

The charges must be `int32`. The convention for scaling (×1 or ×2) depends on
your model — just be consistent.

---

## Stage 3: SymmetricTensor — Block-Sparse Tensors

A `SymmetricTensor` only stores blocks where the charges satisfy the fusion
rule (total charge = 0 for the full tensor network, or a specified target).

### Creating a random symmetric tensor

```python
from tenax import SymmetricTensor
import jax

key = jax.random.PRNGKey(0)

A = SymmetricTensor.random_normal(
    indices=(
        TensorIndex.from_charges(u1, phys_charges, FlowDirection.IN,  label="p0"),
        TensorIndex.from_charges(u1, bond_charges, FlowDirection.IN,  label="left"),
        TensorIndex.from_charges(u1, bond_charges, FlowDirection.OUT, label="right"),
    ),
    key=key,
)

print(A.labels())  # ('p0', 'left', 'right')
```

### What's actually stored

Internally, only charge-allowed blocks are stored. For a tensor with three
U(1) legs, a block exists only when:

    charge(p0) + charge(left) - charge(right) = 0

(The sign flip on "right" comes from FlowDirection.OUT → dual charges.)

This means many potential blocks are zero and never allocated. For large bond
dimensions with many charge sectors, this saves enormous memory and compute.

### Label-based contraction

```python
from tenax import contract

B = SymmetricTensor.random_normal(
    indices=(
        TensorIndex.from_charges(u1, phys_charges, FlowDirection.IN,  label="p1"),
        TensorIndex.from_charges(u1, bond_charges, FlowDirection.IN,  label="right"),  # Matches A's "right"
        TensorIndex.from_charges(u1, bond_charges, FlowDirection.OUT, label="far_right"),
    ),
    key=jax.random.PRNGKey(1),
)

# "right" is shared → automatically contracted
result = contract(A, B)
print(result.labels())  # ('p0', 'left', 'p1', 'far_right')
```

Note that B's "right" leg is `FlowDirection.IN` while A's is `FlowDirection.OUT`
— they form a proper IN↔OUT pair for contraction.

---

## Stage 4: Block-Sparse Decompositions

Tenax provides symmetry-aware SVD and QR that preserve block structure.

### Block-sparse SVD

Used in DMRG truncation: each charge block is decomposed separately, and
truncation keeps the globally largest singular values, ranked across all
sectors (`tenax.linalg.svd` / `truncated_svd`). `bond_order="sector"` instead
returns the untruncated bond charge-grouped, which is the form that runs under
`jax.jit` (v0.9.0).

### Block-sparse QR

Used to bring MPS tensors into canonical form. Each charge block is QR-
decomposed independently.

**Key insight for students:** These decompositions never mix charge sectors.
A singular value in the Sz=0 sector is independent of singular values in
Sz=1. This is why symmetric tensors produce better-conditioned numerics.

---

## Stage 5: Using Symmetries in Practice

### Symmetric MPOs with AutoMPO

```python
from tenax import AutoMPO

auto = AutoMPO(L=20, d=2)
for i in range(19):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)

mpo = auto.to_mpo(symmetric=True)  # U(1) block-sparse MPO
```

The resulting MPO has `SymmetricTensor` entries with U(1) charges reflecting
that Sp carries charge +1, Sm carries charge -1, and Sz carries charge 0.

### When NOT to use symmetric tensors

- **Transverse-field Ising** (H includes Sx terms) — breaks Sz conservation.
  Use Z_2 symmetry instead, or dense tensors.
- **When you don't know the symmetry** — use dense tensors first, identify
  the symmetry from the spectrum, then switch to symmetric.
- **When D or χ is very small** — the overhead of block-sparse bookkeeping
  may outweigh the savings. Typically worthwhile for bond dim ≥ 20.
- **iPEPS-AD at D ≥ 3** — the block-sparse CTM-AD backward has a long first
  compile and dense is usually faster; symmetric U(1) iPEPS-AD at D ≥ 3 is
  not certified (#1101). Symmetric CTM results from before v0.9.0 zeroed the
  charged environment sectors (#905) and should be recomputed.
- **Avoid `todense()` on large symmetric tensors** — it defeats the point;
  use the block-sparse `tenax.linalg.svd` / `qr` / `eigh` instead.

---

## Pedagogical Notes

- **"Charges are quantum numbers"** — connect every code concept to the
  underlying physics. FlowDirection = bra/ket. Fusion = adding angular momenta.
  Block-sparsity = selection rules.
- **Draw the analogy to Clebsch-Gordan coefficients** — fusing two U(1) legs
  is like adding angular momenta; the allowed outputs are determined by the
  same rules.
- **The non-Abelian future** — Tenax has an extensible symmetry interface
  for future SU(2) support. The concepts taught here (charges, fusion,
  FlowDirection) generalize directly to non-Abelian multiplets, where charges
  become irreps and fusion becomes the Clebsch-Gordan series.
