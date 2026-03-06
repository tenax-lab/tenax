# Contraction and Decompositions

The contraction engine translates label-based tensor operations into optimised
einsum calls executed by JAX.

## Label-based contraction

The core idea: **legs with the same label across different tensors are
automatically summed over**.

```python
from tenax import contract

# A has legs ("i", "bond"), B has legs ("bond", "j")
C = contract(A, B)
# "bond" appears in both -> contracted
# Result has legs ("i", "j")
```

### Multi-tensor contraction

`contract` accepts any number of tensors. Internally it uses `opt_einsum`
to find the optimal contraction order:

```python
D = contract(A, B, C)  # three-tensor contraction
```

### Controlling output label order

By default, free labels appear in the order they are encountered. Use
`output_labels` to specify an explicit ordering:

```python
C = contract(A, B, output_labels=("j", "i"))
```

### Optimiser selection

The `optimize` parameter selects the opt_einsum strategy:

```python
C = contract(A, B, optimize="auto")       # default
C = contract(A, B, optimize="greedy")     # faster for large networks
C = contract(A, B, optimize="optimal")    # brute-force optimal
```

Decompositions live in `tenax.linalg` and are re-exported from top-level
`tenax`.

## SVD

`svd` decomposes a tensor into U, s, V^dagger with truncation:

```python
from tenax import svd

# Split tensor T with legs ("left", "phys", "right") along the cut
# left_labels vs right_labels
U, s, Vh, s_full = svd(
    T,
    left_labels=["left", "phys"],
    right_labels=["right"],
    new_bond_label="bond",
    max_singular_values=16,
)
# U has legs ("left", "phys", "bond")
# s is a 1D JAX array of truncated singular values
# Vh has legs ("bond", "right")
# s_full is the complete singular value spectrum before truncation
```

Parameters controlling truncation:

- `max_singular_values` -- hard cap on the bond dimension
- `max_truncation_err` -- discard smallest singular values until the
  relative truncation error exceeds this threshold

Both dense and symmetric tensors are supported. For `SymmetricTensor`,
the SVD is performed block-by-block within each charge sector.

> The legacy name `truncated_svd` is still available for backward
> compatibility.

## QR decomposition

`qr` splits a tensor into an orthogonal factor Q and an upper-
triangular factor R:

```python
from tenax import qr

Q, R = qr(
    T,
    left_labels=["left", "phys"],
    right_labels=["right"],
    new_bond_label="bond",
)
# Q has legs ("left", "phys", "bond")  -- isometric
# R has legs ("bond", "right")
```

QR is cheaper than SVD and is useful for canonicalising MPS tensors
during DMRG sweeps.

> The legacy name `qr_decompose` is still available for backward
> compatibility.

## Eigendecomposition (eigh)

`eigh` eigendecomposes a Hermitian tensor, returning eigenvectors and
eigenvalues sorted in descending order:

```python
from tenax import eigh

V, eigenvalues = eigh(
    T,
    left_labels=["left", "phys"],
    right_labels=["right"],
    new_bond_label="bond",
    max_eigenvalues=16,
)
# V has legs ("left", "phys", "bond")
# eigenvalues is a 1D JAX array (descending order)
```

Like SVD and QR, `eigh` dispatches to a block-sparse path for
`SymmetricTensor`.

## Lower-level API

For full control, `contract_with_subscripts` accepts explicit einsum
subscript strings:

```python
from tenax import contract_with_subscripts

result = contract_with_subscripts(
    [A, B],
    subscripts="ij,jk->ik",
    output_indices=(...),  # TensorIndex tuple for the result
)
```

This is mainly used internally by the `TensorNetwork` graph container.
