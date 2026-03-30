# Cython BLAS v2: Zero-Python-Reentry Kernel for DMRG

**Date:** 2026-03-29
**Status:** Approved

## Problem

The v1 Cython BLAS kernel (PR #205) gives only 2-3x speedup over pure Python. The 25-100x gap to TeNPy remains because the "Cython" kernel still re-enters Python for every operation inside the block-combo loop: `np.transpose`, `np.ascontiguousarray`, `scipy_blas.dgemm`, `dict.__getitem__`, `dict.__setitem__`.

Benchmark (Heisenberg, U(1), 2-site DMRG, float64):

| L | chi | TeNPy (s) | Tenax v1 (s) | Gap |
|---|-----|-----------|-------------|-----|
| 20 | 32 | 1.3 | 63 | 49x |
| 40 | 64 | 3.5 | 105 | 30x |
| 80 | 128 | 14 | 360 | 26x |

**Target:** < 3x gap to TeNPy (i.e. < 42s for L=80, chi=128).

## Design Decisions

- **Full C inner loop** — zero Python calls per block combo. All block lookup, GEMM, and accumulation happen in a tight Cython loop with typed memoryviews.
- **Multi-step GEMM** — keep opt_einsum's pairwise decomposition (4 GEMMs for 5-tensor contraction). Execute via raw `dgemm`/`zgemm` pointer calls.
- **Pre-transpose at setup** — all blocks pre-transposed and reshaped to GEMM-ready 2D C-contiguous layout before entering the C kernel. Inner loop does pure GEMM + accumulate.
- **Dtype dispatch via flag** — branch once at kernel entry on dtype enum (f64/f32/c128/c64), not per combo.
- **Fallback unchanged** — the existing opt_einsum + JAX path stays as-is for environments without Cython.

## Architecture: Three Phases

### Phase 1: Python setup (once per Lanczos solve)

A new `_prepare_blas_kernel_args()` function:

1. Convert JAX blocks to NumPy once per tensor
2. Group combos by shape signature, get cached `BlasExecPlan` per group
3. For each combo, pre-transpose and reshape each input block to GEMM-ready 2D layout (`np.ascontiguousarray`)
4. Pack into flat arrays: `combo_blocks` list, `step_m/n/k` int arrays, `combo_output_idx` mapping, pre-allocated output and work buffers
5. Return a `KernelArgs` dataclass

Called once at the same point `_precompute_block_plan` is called. For the Lanczos matvec, environment blocks are pre-transposed once; only the varying `v` tensor's blocks are re-transposed per iteration.

### Phase 2: C kernel (zero Python re-entry)

Rewritten `_cython_blas.pyx`:

```cython
def execute_blas_kernel(
    int dtype_code,                    # 0=f64, 1=f32, 2=c128, 3=c64
    int n_combos,
    int n_steps,
    int[:] step_m, int[:] step_n, int[:] step_k,
    int[:] step_left_buf, int[:] step_right_buf, int[:] step_out_buf,
    list combo_blocks,                 # pre-transposed 2D arrays
    int[:] input_mapping,
    int[:] combo_output_idx,
    list output_buffers,               # pre-allocated, zeroed
    list work_buffers,                 # pre-allocated intermediates
) -> None:
```

Inner loop (nogil where possible):

```
for each combo:
    load pre-transposed input pointers into buffer slots
    for each step:
        ptr_left  = buffer[step_left_buf[s]]
        ptr_right = buffer[step_right_buf[s]]
        ptr_out   = work_buffer or output_buffer
        dgemm/zgemm(m, n, k, alpha=1.0, ptr_left, ptr_right, beta, ptr_out)
    accumulate into output_buffers[combo_output_idx[combo]] with beta=1.0
```

Key properties:
- No Python objects inside the loop — typed memoryviews and raw pointers only
- Output accumulation uses `dgemm` with `beta=1.0` directly, no intermediate allocation
- Work buffers reused across combos (overwritten each iteration)

### Phase 3: Python wrap-up (once)

Apply final output permutation (if any) and wrap NumPy arrays back into `SymmetricTensor`.

## Integration Point

`_blockwise_contract` in `dmrg.py`, same location as v1:

```python
if block_plan is not None and CYTHON_BLAS_AVAILABLE:
    # Phase 1: prepare args (cached for env tensors, only v re-prepared)
    kernel_args = _prepare_blas_kernel_args(tensors, subscripts, block_plan)
    # Phase 2: C kernel
    execute_blas_kernel(**kernel_args)
    # Phase 3: wrap results
    ...
elif block_plan is not None:
    # Existing opt_einsum + JAX fallback (unchanged)
    ...
else:
    # Original backtracking (no precomputed plan)
    ...
```

The matvec callers (`_two_site_update_symmetric`, `_one_site_update_symmetric`) cache the environment portion of `KernelArgs` and only re-prepare the `v` tensor blocks per Lanczos iteration.

## Coverage

All DMRG symmetric hot paths go through `_blockwise_contract`:
- 2-site DMRG matvec (5-tensor, ~10-50x per sweep step) **— hottest**
- 1-site DMRG matvec (4-tensor)
- Left/right environment updates (4-tensor, once per sweep step)
- iDMRG (reuses same engine)

## File Changes

| File | Change |
|------|--------|
| `src/tenax/contraction/_cython_blas.pyx` | Rewrite: typed memoryviews, raw BLAS, nogil loop |
| `src/tenax/contraction/_blas_plan.py` | Add `_prepare_blas_kernel_args()`, `KernelArgs` dataclass |
| `src/tenax/algorithms/dmrg.py` | Update Cython branch in `_blockwise_contract`, cache env args in matvec callers |

Unchanged: `BlasExecPlan`, `GemmStep`, `build_blas_plan()`, fallback path, "Tests without Cython" CI.

## Testing

- **Correctness:** Existing DMRG symmetric tests (energies match to 1e-10)
- **Benchmark:** Cython path at least 10x faster than fallback for chi=64 2-site matvec
- **Fallback CI:** "Tests without Cython" job verifies fallback still works
