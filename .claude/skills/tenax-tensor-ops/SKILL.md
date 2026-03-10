---
name: tenax-tensor-ops
description: >
  Teach users Tenax's core tensor operations: creating DenseTensor and
  SymmetricTensor, label-based contraction, truncated SVD, QR decomposition,
  and the TensorNetwork graph container. Use this skill when the user asks
  about basic tensor operations, "how does contraction work", "how to do SVD",
  "create a tensor", "what are labels", DenseTensor, contract, truncated_svd,
  qr_decompose, or wants to understand the building blocks before running
  algorithms. Also trigger for "tensor basics", "label-based contraction",
  or "how do I contract two tensors".
---

# Core Tensor Operations in Tenax

Teach users the fundamental building blocks: tensor creation, label-based
contraction, and decompositions. These primitives underlie every algorithm
in Tenax.

## Key Concepts

1. **Every tensor leg has a label** — contraction happens automatically on
   shared labels. No manual index bookkeeping.
2. **Two tensor types** — `DenseTensor` (full arrays) and `SymmetricTensor`
   (block-sparse with charge conservation). Same API for both.
3. **JAX pytrees** — both tensor types work with `jax.jit`, `jax.grad`,
   `jax.vmap` natively.

---

## Creating Tensors

### DenseTensor

```python
import jax
import numpy as np
from tenax import DenseTensor, TensorIndex, FlowDirection

# Define legs with labels
idx_left  = TensorIndex(None, np.array([], dtype=np.int32), FlowDirection.IN,  label="left")
idx_phys  = TensorIndex(None, np.array([], dtype=np.int32), FlowDirection.IN,  label="phys")
idx_right = TensorIndex(None, np.array([], dtype=np.int32), FlowDirection.OUT, label="right")

# Create from a JAX array
data = jax.random.normal(jax.random.PRNGKey(0), shape=(4, 2, 4))
A = DenseTensor(data, indices=(idx_left, idx_phys, idx_right))

print(A.labels())  # ('left', 'phys', 'right')
print(A.ndim)      # 3
```

For dense tensors without symmetry, the `charges` array in `TensorIndex` is
unused — pass an empty array. The `FlowDirection` still matters for
contractions with `SymmetricTensor`.

### DenseTensor.random_normal

```python
key = jax.random.PRNGKey(42)
A = DenseTensor.random_normal(
    indices=(idx_left, idx_phys, idx_right),
    key=key,
)
```

### SymmetricTensor

See the **symmetry skill** for full details. Quick example:

```python
from tenax import U1Symmetry, SymmetricTensor, TensorIndex, FlowDirection
import numpy as np, jax

u1 = U1Symmetry()
A = SymmetricTensor.random_normal(
    indices=(
        TensorIndex(u1, np.array([-1, 1], dtype=np.int32), FlowDirection.IN,  label="phys"),
        TensorIndex(u1, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN,  label="left"),
        TensorIndex(u1, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.OUT, label="right"),
    ),
    key=jax.random.PRNGKey(0),
)
```

---

## Label-Based Contraction

The central operation in tensor networks. Two tensors are contracted by
summing over legs that share the same label.

```python
from tenax import contract

# A has legs: (left, phys, bond)
# B has legs: (bond, phys2, right)
# "bond" is shared → contracted automatically
result = contract(A, B)
# result has legs: (left, phys, phys2, right)
```

### Rules

1. **Shared labels are contracted.** If A has label "x" and B has label "x",
   that pair is summed over.
2. **Unique labels are kept.** They appear in the output.
3. **FlowDirection pairing.** For `SymmetricTensor`, contracted legs must
   form IN↔OUT pairs (charge conservation).
4. **Multiple tensors.** `contract(A, B, C, ...)` contracts all shared labels.

### Controlling output label order

```python
result = contract(A, B, output_labels=["left", "right", "phys"])
```

### Multi-tensor contraction

```python
# Contracts all pairwise shared labels, uses opt_einsum for optimal order
result = contract(A, B, C, D)
```

### Relabeling

To contract legs that don't share a label, or to avoid unwanted contraction:

```python
B_relabeled = B.relabel("old_label", "new_label")

# Bulk relabeling
B_relabeled = B.relabels({"old1": "new1", "old2": "new2"})
```

---

## Truncated SVD

Decomposes a tensor into U · S · Vh, keeping only the largest singular values.
This is the core operation in DMRG (bond truncation) and TRG (coarse-graining).

```python
from tenax import truncated_svd

# Split tensor A along (left, phys) vs (right,)
U, S, Vh, S_full = truncated_svd(
    A,
    left_labels=["left", "phys"],
    right_labels=["right"],
    new_bond_label="bond",       # Label for the new bond leg
    max_singular_values=16,      # Keep at most 16 singular values
    max_truncation_err=1e-10,    # Or truncate by error threshold
)

# U has legs: (left, phys, bond)
# S is a 1D array of singular values
# Vh has legs: (bond, right)
# S_full is the full (untruncated) singular value array
```

### Key parameters

| Parameter | Purpose |
|-----------|---------|
| `left_labels` | Legs that go into U |
| `right_labels` | Legs that go into Vh |
| `new_bond_label` | Label for the new bond connecting U and Vh |
| `max_singular_values` | Hard cap on bond dimension |
| `max_truncation_err` | Discard singular values below this threshold |
| `normalize` | If True, normalize S so that sum(S^2) = 1 |

### Physics connection

The singular values S are the Schmidt coefficients of the bipartition.
The entanglement entropy is S_ent = -Σ s² ln(s²). Truncating small
singular values is the controlled approximation that makes DMRG work.

---

## QR Decomposition

Factorizes a tensor into Q (isometric) · R (upper triangular). Used for
canonical form in MPS algorithms.

```python
from tenax import qr_decompose

Q, R = qr_decompose(
    A,
    left_labels=["left", "phys"],
    right_labels=["right"],
    new_bond_label="bond",
)
# Q has legs: (left, phys, bond) — isometric
# R has legs: (bond, right)
```

Unlike SVD, QR does not truncate — it's exact. Use it when you need
canonical form without discarding information.

---

## TensorNetwork Container

For managing multiple tensors as a graph:

```python
from tenax import TensorNetwork

tn = TensorNetwork()
tn.add_node("A", tensor_A)
tn.add_node("B", tensor_B)
tn.connect_by_shared_label("A", "B")  # Connect legs with matching labels

result = tn.contract()  # Contract the whole network
```

### Building standard structures

```python
from tenax import build_mps, build_peps

mps = build_mps(tensors)                # 1D chain
peps = build_peps(tensors, Lx, Ly)      # 2D grid
```

---

## Subscript-Based Contraction

For cases where label-based contraction is awkward, use einsum-style
subscripts:

```python
from tenax import contract_with_subscripts

# Equivalent to np.einsum("ijk,jkl->il", A, B)
result = contract_with_subscripts(
    [A, B],
    subscripts="ijk,jkl->il",
    output_indices=(A.indices[0], B.indices[2]),
)
```

This is lower-level and rarely needed — prefer label-based `contract()`.

---

## Pedagogical Notes

- **Labels are the key abstraction.** They replace manual index tracking,
  making tensor network code readable and less error-prone.
- **Think diagrammatically.** Each tensor is a node, each leg is a line.
  Shared labels = connected lines. `contract()` sums over connected lines.
- **SVD = entanglement.** The singular values tell you how much entanglement
  crosses a cut. This is why SVD is central to tensor network algorithms.
