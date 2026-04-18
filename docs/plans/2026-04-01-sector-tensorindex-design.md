# Sector-based TensorIndex with fuse/unfuse

**Date:** 2026-04-01
**Status:** COMPLETED (PR #213) — sector-based TensorIndex with fuse/unfuse
**Goal:** Change TensorIndex from dense charges array to sector-based `{charge: multiplicity}` representation, and add `split_index` as the inverse of `fuse_indices`. This lays the foundation for non-abelian symmetries.

## Design

### 1. TensorIndex representation change

Replace `charges: np.ndarray` (one entry per basis state) with:

```python
@dataclass(frozen=True)
class TensorIndex:
    symmetry: BaseSymmetry
    sectors: np.ndarray         # sorted unique charges, shape (n_sectors,)
    multiplicities: np.ndarray  # multiplicity per sector, shape (n_sectors,)
    flow: FlowDirection
    label: str | int
    fuse_info: FuseInfo | None = None  # None for elementary legs
```

- `dim` = `sum(multiplicities)`
- Property `charges` reconstructs the dense array for backward compatibility
- `TensorIndex.from_charges(sym, charges, flow, label)` classmethod for old-style construction
- Performance: block operations already work at sector level — this makes it explicit

### 2. FuseInfo

```python
@dataclass(frozen=True)
class FuseInfo:
    parent_indices: tuple[TensorIndex, ...]
```

Recursive: parents may themselves have `fuse_info` (e.g., `((a,b), c)`). For future non-abelian symmetries, add CG coefficients field.

### 3. fuse_indices update

Computes fused sectors directly from parent sectors:
- For abelian: `q_f = q_a + q_b`, multiplicity = `m_a * m_b` (summed over pairs giving same q_f)
- Populates `fuse_info` on the fused TensorIndex
- O(n_sectors_a * n_sectors_b) instead of current O(dim_a * dim_b)

### 4. split_index — new

```python
def split_index(tensor: Tensor, axis: int) -> Tensor:
```

Reads `fuse_info` from `tensor.indices[axis]`, reconstructs parent legs, reshapes block data. Raises `ValueError` if `fuse_info is None`.

### 5. Migration

All code constructing `TensorIndex(sym, charges, flow, label)` changes to either:
- `TensorIndex(sym, sectors, multiplicities, flow, label)` for new code
- `TensorIndex.from_charges(sym, charges, flow, label)` for convenience

### Files affected

| Area | Files |
|------|-------|
| Core | `core/index.py` (TensorIndex, FuseInfo) |
| Core | `core/tensor.py` (SymmetricTensor block construction, todense, from_dense) |
| Algorithms | `algorithms/_tensor_utils.py` (fuse_indices, split_index) |
| Algorithms | `algorithms/_ctm_*` (CTM init, projectors — construct TensorIndex) |
| Algorithms | `algorithms/ipeps*.py`, `algorithms/dmrg.py` |
| Linalg | `linalg.py` (SVD, QR, eigh — output index construction) |
| Tests | All tests constructing TensorIndex |
