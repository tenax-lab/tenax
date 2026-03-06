# Contraction Semantics

This document describes the label-based contraction model used in Tenax.

---

## Overview

Tenax contracts tensors using **shared string labels** rather than explicit
einsum subscript strings. Two tensors that carry the same label on one of their
legs are automatically contracted over that pair of legs. This is identical in
spirit to the Cytnx / ITensor interface.

---

## Label Rules

| Rule | Description |
|------|-------------|
| **Shared label → contraction** | Any label that appears on exactly two tensors (or twice within one tensor) is contracted. |
| **Free label → output leg** | A label that appears on exactly one tensor leg in the full set of operands becomes a free (open) index in the result. |
| **Label multiplicity > 2 → error** | A label appearing on three or more legs raises `ValueError`. Labels must identify a unique pair. |

---

## `contract(*tensors, output_labels, optimize)`

```python
from tenax import contract

result = contract(t1, t2, t3, output_labels=["a", "c"], optimize="auto")
```

**Arguments**

- `*tensors` — One or more `Tensor` objects (`DenseTensor` or `SymmetricTensor`).
- `output_labels` — Optional list of free labels that determines the **leg order** of the
  result tensor. If `None`, free labels appear in the order they are first encountered
  while scanning tensor indices left-to-right.
- `optimize` — opt_einsum path optimizer (`"auto"`, `"greedy"`, `"optimal"`, etc.).
  Defaults to `"auto"`.

**Return value** — A single `DenseTensor` whose legs correspond to all free labels.

### How labels become subscripts

Internally, `_labels_to_subscripts()` in `contraction/contractor.py` assigns a
unique single-character symbol to each distinct label, then builds the standard
`opt_einsum` expression. The mapping is ephemeral (not exposed to users) and
changes each call.

```
t1 labels: ["a", "b"]   → subscript "ab"
t2 labels: ["b", "c"]   → subscript "bc"
shared: {"b"}            → contracted
free: {"a", "c"}         → output "ac"  (or "ca" if output_labels=["c","a"])
```

opt_einsum finds the optimal pairwise contraction order for the given operands
before JAX executes any computation.

---

## `TensorNetwork.contract(nodes, output_labels, optimize, cache)`

When tensors are stored in a `TensorNetwork`, contraction is done via the
network object:

```python
tn = TensorNetwork()
tn.add_node("A", t1)
tn.add_node("B", t2)
result = tn.contract(nodes=["A", "B"], output_labels=["a", "c"])
```

**Cache behaviour**

Results are cached by the key `(tuple(nodes), tuple(output_labels), optimize)`.
Node order matters: when `output_labels` is ``None``, the output leg order depends
on the order nodes are listed. The cache is invalidated automatically whenever the
graph structure changes (node added/removed, edge added/removed, tensor replaced).
Two calls with the same nodes but different `output_labels` or different `optimize`
strings receive independent cache entries.

---

## `svd(tensor, left_labels, right_labels, ...)`

SVD splits a tensor into `U, S, Vh, S_full` with an optional truncation of
singular values:

```python
U, S, Vh, S_full = svd(
    theta,
    left_labels=["v0_1", "p0"],
    right_labels=["p1", "v1_2"],
    new_bond_label="v0_2",
    max_singular_values=chi,
)
```

- `left_labels` — Legs of `theta` that go onto `U`.
- `right_labels` — Legs of `theta` that go onto `Vh`.
- `new_bond_label` — Label for the new shared bond between `U` and `Vh`.
- `max_singular_values` — Hard cap on bond dimension (discards smallest singular values).
- `max_truncation_err` — Discard singular values until cumulative truncation error
  exceeds this threshold (whichever is more restrictive with `max_singular_values`).

`S` contains the **truncated** singular values in descending order. `S_full`
contains all singular values before truncation, useful for computing truncation
error without a second SVD.

> The legacy name `truncated_svd` is still available for backward compatibility.

---

## `qr(tensor, left_labels, right_labels, new_bond_label)`

Thin QR decomposition. The orthogonality column of `Q` corresponds to
`left_labels`; `R` carries `right_labels`. Used in MPS right-canonicalization.

> The legacy name `qr_decompose` is still available for backward compatibility.

---

## `eigh(tensor, left_labels, right_labels, ...)`

Hermitian eigendecomposition. Reshapes the tensor into a square matrix
(left_labels vs right_labels), computes `jnp.linalg.eigh`, and returns
eigenvectors as a Tensor with labels `(left_labels..., new_bond_label)`.

Eigenvalues are sorted in descending order. `max_eigenvalues` keeps only
the top-k. Dispatches to a block-sparse path for `SymmetricTensor`.

---

## Label Conventions in DMRG

| Object | Label pattern |
|--------|---------------|
| MPS virtual bond between sites i and i+1 | `"v{i}_{i+1}"` |
| MPS physical leg at site i | `"p{i}"` |
| MPO virtual bond between sites i and i+1 | `"w{i}_{i+1}"` |
| MPO physical ket leg at site i | `"mpo_top_{i}"` |
| MPO physical bra leg at site i | `"mpo_bot_{i}"` |
| Left environment legs | `"env_mps_l"`, `"env_mpo_l"`, `"env_mps_conj_l"` |
| Right environment legs | `"env_mps_r"`, `"env_mpo_r"`, `"env_mps_conj_r"` |

---

## Design Rationale

**Why label-based?** Explicit einsum subscript management becomes error-prone as
networks grow. Labels self-document which physical degrees of freedom are being
contracted (virtual bond vs. physical leg vs. MPO bond), and the label-to-subscript
translation is handled once inside `_labels_to_subscripts()`.

**Why opt_einsum?** For networks with more than two tensors, the contraction order
dominates the computational cost. opt_einsum finds an order that minimises the
intermediate tensor size (flops), whereas naive left-to-right pairwise contraction
can be exponentially worse.

**Pairwise-only enforcement** (no trace mode): labels appearing more than twice
raise an error immediately. This prevents silent incorrect results when a label is
accidentally reused.
