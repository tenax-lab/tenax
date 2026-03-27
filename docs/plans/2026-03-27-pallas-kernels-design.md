# Pallas Kernels for Block-Sparse Tensor Operations

**Date:** 2026-03-27
**Status:** Proposed
**Issue:** #195

## Problem

SymmetricTensor operations are 10x slower than TeNPy on CPU due to Python-level
block dispatch. On GPU, the situation is worse — 1.15M tiny kernel launches for
a chi=32 iDMRG run, making GPU *slower* than CPU for symmetric tensors.

Current bottleneck chain:
```
_blockwise_contract (Python loop)
  → for each block combination (Python)
    → opt_einsum (Python dispatch)
      → jax.lax.dot_general (GPU kernel launch per block)
```

## Goal

Replace the Python block loop with a single fused kernel that processes all
charge sectors in one launch. Target: symmetric GPU performance within 2x of
dense GPU.

## Architecture

### Phase 1: Padded Batched Contractions (pure JAX, no Pallas)

The simplest approach that eliminates the Python loop:

1. **Pad all blocks to max sector size** — for a tensor with sectors of sizes
   [3, 5, 4], pad all to 5. Stack into `(n_sectors, max_dim, max_dim, ...)`.

2. **Precompute contraction plan** — enumerate valid (sector_A, sector_B, ...)
   combinations. Store as index arrays.

3. **`vmap` the einsum** — one fused kernel for all combinations:
   ```python
   # Instead of:
   for qa, qb in valid_combos:
       result[qc] += einsum(A[qa], B[qb])

   # Do:
   padded_A = stack_and_pad(A.blocks)  # (n_combos, max_d, max_d)
   padded_B = stack_and_pad(B.blocks)  # (n_combos, max_d, max_d)
   results = vmap(einsum)(padded_A, padded_B)  # single kernel
   scatter_add(output, results, combo_indices)
   ```

4. **Scatter-add results** — accumulate into output sectors via `jax.ops.segment_sum`.

**Pros:** Pure JAX, works on all backends, JIT-able, no custom kernels.
**Cons:** Padding waste if sectors are very unbalanced.

### Phase 2: Pallas Kernels (GPU-optimized)

For maximum GPU performance, write custom Pallas kernels:

```python
@pl.kernel
def block_sparse_matvec(
    data_in: pl.Array,      # flat input data
    data_out: pl.Array,     # flat output data
    offsets: pl.Array,      # block offset table
    shapes: pl.Array,       # block shape table
    plan: pl.Array,         # precomputed contraction plan
):
    # Each GPU thread block processes one sector combination
    combo_idx = pl.program_id(0)
    # Load block data from flat buffer using offsets
    # Compute local einsum
    # Accumulate into output
```

**Pros:** No padding waste, optimal GPU utilization, custom memory access patterns.
**Cons:** Pallas API is still evolving, GPU-only (CPU fallback needed).

### Phase 3: Hybrid Dispatch

```python
def _blockwise_contract(tensors, subscripts, output_indices, ...):
    if all on GPU and n_blocks > threshold:
        return _batched_block_contract_gpu(...)  # Phase 1 or 2
    elif on CPU:
        return _numpy_block_contract_cpu(...)    # current NumPy path
    else:
        return _python_block_contract(...)       # fallback
```

## Data Layout (already correct)

The current `SymmetricTensor` flat `_data` layout is designed for this:

```python
class SymmetricTensor:
    _data: jax.Array           # flat buffer, all blocks concatenated
    _block_keys: tuple         # charge sector labels
    _block_shapes: tuple       # (d1, d2, ...) per block
    _block_offsets: tuple      # byte offset in _data per block
```

The flat buffer is GPU-friendly — one contiguous allocation. Block metadata
(`_block_keys`, `_block_shapes`, `_block_offsets`) becomes the kernel's
dispatch table.

## Implementation Plan

### Step 1: Padded batched matvec for Lanczos (Phase 1)

Focus on the iDMRG/DMRG Lanczos matvec `L·M·W·R`:

```python
def _batched_matvec(
    L_data, M_data, W_data, R_data,   # flat data arrays
    plan,                               # precomputed combo table
    L_meta, M_meta, W_meta, R_meta,   # offset/shape metadata
):
    # Extract and pad blocks according to plan
    # vmap the contraction
    # scatter-add results
```

This is the single highest-impact operation — called thousands of times
per iDMRG run. Getting this right makes symmetric GPU competitive.

### Step 2: Batched SVD/QR

Apply the same padding+vmap pattern to `tenax.linalg.svd` and `qr` for
SymmetricTensor. Currently these loop over sectors in Python.

### Step 3: Batched environment updates

`_update_left_env_symmetric` and `_update_right_env_symmetric` also do
per-sector contractions. Batch these.

### Step 4: Pallas kernels (optional, GPU-only)

If padding waste is too large (very unbalanced sectors), write custom
Pallas kernels that process variable-size blocks without padding.

## Estimated Impact

| Operation | Current (GPU) | Phase 1 (vmap) | Phase 2 (Pallas) |
|-----------|--------------|----------------|------------------|
| Lanczos matvec | 1.15M kernel launches | 1 kernel launch | 1 kernel launch |
| SVD per sector | N kernel launches | 1 batched SVD | 1 kernel |
| Env update | N kernel launches | 1 vmap | 1 kernel |

Expected: **10-50x speedup** for symmetric on GPU, bringing it within
2x of dense GPU performance.

## References

- [JAX Pallas documentation](https://jax.readthedocs.io/en/latest/pallas/)
- [Pallas GPU tutorial](https://jax.readthedocs.io/en/latest/pallas/tpu.html)
- Issue #195: SymmetricTensor performance profiling
