# Batched Matvec for Block-Sparse DMRG

## Problem

The numpy blockwise DMRG path spends ~42% of total time in the Python loop
inside `_blockwise_contract`, iterating over ~930K block combos per run.
An additional ~13% goes to redundant `numpy.transpose` calls on fixed
environment blocks that are re-transposed every Lanczos iteration.

Profile data (L=30, chi=64, 4 sweeps, Cython BLAS compiled):

| Bottleneck | Time | % |
|-----------|------|---|
| `_blockwise_contract` Python loop | 9.6s | 42% |
| `numpy.transpose` (4.8M calls) | 3.0s | 13% |
| `ba_inner` (reorthogonalization) | 2.7s | 12% |
| `build_blas_plan` + opt_einsum | 2.6s | 11% |

Root cause: at moderate chi with U(1) symmetry, blocks are small (~21x22
per charge sector). Each GEMM is cheap but Python dispatch overhead per
block combo dominates (~10us x 930K = ~9.3s).

## Design

### Change 1: Pre-transpose environment blocks

**Where:** `_two_site_update_symmetric_np` and `_one_site_update_symmetric_np`

In the 2-site matvec (`"abc,apqd,bpse,eqtf,dfg->cstg"`), four of five
tensors (left_env, mpo_l, mpo_r, right_env) are fixed across all ~40
Lanczos iterations. Only theta changes.

Currently `cython_execute_plan` re-transposes every block on every call.
Instead:

1. After computing the block plan, group combos by `block_shapes` (same
   BLAS plan).
2. For each shape group, pre-transpose the fixed env blocks to their
   GEMM-ready 2D layout once.
3. Store in a cache: `dict[(block_shapes, tensor_idx, block_key)] -> np.ndarray`
4. In the matvec, only transpose theta's blocks fresh; pass pre-transposed
   env blocks directly to the GEMM kernel.

This eliminates ~80% of the 4.8M transpose calls (~3s -> ~0.6s).

### Change 2: Batched Cython combo kernel

**Where:** New function in `_cython_blas.pyx`, called from `_blockwise_contract`

New function `cython_execute_combos` that processes all combos for a
matvec call in a single Python->C transition:

```
def cython_execute_combos(
    list combo_groups,      # list of (step_params, combo_block_arrays, output_indices)
    list output_buffers,    # pre-allocated output arrays, indexed by output slot
    int n_output_slots,
):
```

For each combo group (combos sharing the same BLAS plan):
- `combo_block_arrays`: list of lists of pre-transposed 2D numpy arrays
- `output_indices`: int array mapping each combo to its output slot
- The C loop iterates over combos, executes GEMM steps, accumulates
  into pre-allocated output buffers via `daxpy` (beta=1 GEMM or explicit add)

This replaces ~100 Python->Cython calls per matvec with 1.

### Integration in _blockwise_contract

Add a new code path when `batched=True` (opt-in flag):

```python
if batched and block_plan is not None:
    # Group combos by block_shapes
    groups = _group_combos_by_shape(block_plan, np_blocks_list)
    # Pre-transpose env blocks per group
    pretransposed = _pretranspose_fixed_blocks(groups, ...)
    # Single Cython call
    output_blocks = _cython_execute_combos(groups, pretransposed, ...)
```

The existing per-combo path remains as fallback.

## Files Changed

| File | Change |
|------|--------|
| `_cython_blas.pyx` | New `cython_execute_combos` function |
| `_blas_plan.py` | Helper `pretranspose_blocks_for_plan` |
| `dmrg.py` | Modified matvec setup in `_two_site_update_symmetric_np` and `_one_site_update_symmetric_np`; new `_group_combos_by_shape` helper |

## Expected Impact

- Eliminate ~9s of Python loop overhead -> ~0.5s C loop
- Eliminate ~2.4s of redundant transpose -> ~0s (pre-transposed)
- Total: ~23s -> ~12s (1.9x speedup) for L=30, chi=64

## Testing

- All existing `test_dmrg.py` tests must pass (correctness)
- Benchmark: compare wall time with `batched=True` vs `batched=False`
- Verify energy matches to machine precision between paths
