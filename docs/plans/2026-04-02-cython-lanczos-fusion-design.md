# Cython Fused Lanczos + Matvec Dispatch

**Date:** 2026-04-02
**Status:** Approved

## Goal

Match TeNPy at chi=32-64 by eliminating all Python round-trips in the DMRG
inner loop. Two components: (1) Cythonized matvec dispatch via `cdef` class,
(2) Cython Lanczos loop calling it without returning to Python.

## Current State

| Case | TeNPy | Tenax | Ratio |
|------|-------|-------|-------|
| L=20 chi=32 10sw | 2.6s | 5.7s | 2.2x |
| L=40 chi=128 5sw | 19.1s | 33.8s | 1.8x |
| L=80 chi=128 5sw | 49.6s | 84.1s | 1.7x |

The gap at small chi comes from Python per-combo dispatch (~26μs each vs
TeNPy's ~2μs Cython). At large chi both are BLAS-bound.

## Architecture

All new code in `src/tenax/contraction/_cython_blas.pyx` (already compiled,
already has BLAS imports).

### Three new constructs

```
cdef class MatvecOp:
    cdef dict apply(self, dict theta_blocks)

cdef class DMRGMatvec2Site(MatvecOp):
    # Holds: combo_descriptors, env_2d_blocks, theta_buf_idx,
    #        output_keys, output_shapes, theta_perms
    cdef dict apply(self, dict theta_blocks):
        # 1. Pre-transpose theta (C loop over unique theta keys)
        # 2. Execute GEMM combos (reuse existing _combo_loop_f64/z128)
        # 3. Assemble output dict
        # All nogil except dict operations

cdef class DMRGMatvec1Site(MatvecOp):
    # Same pattern, 4-tensor subscript "abc,apd,bpxe,def->cxf"

def cython_lanczos_ground(MatvecOp mv, dict v0_blocks, int max_iter,
                           double tol, list index_keys) -> tuple:
    # Full Lanczos loop:
    #   - calls mv.apply() per step (C-level)
    #   - calls cython_ba_inner (existing), cython_ba_axpy (existing)
    #   - cython_lanczos_reorth (existing)
    #   - ba_norm via ddot+sqrt (new cdef helper)
    #   - ba_scale via dscal (existing)
    #   - eigenvector reconstruction via daxpy loop
    #   - early termination on beta < tol
    # Returns: (eigenvalue, eigenvector_blocks)
```

### What stays in Python

- `_precompute_matvec_combos()` — runs once per site update, builds combo
  descriptors. Setup, not hot path.
- `_precompute_block_plan()` — charge backtracking, also setup-only.
- `DMRGMatvec2Site.__init__()` — Python-level construction (once per site update).
- SVD/QR after Lanczos — out of scope.

### What moves to C

| Current Python function | New Cython replacement | Calls/run (L=40 chi=128 5sw) |
|---|---|---|
| `_execute_matvec_combos` theta transpose loop | `DMRGMatvec2Site.apply()` | ~4000 |
| `_execute_matvec_combos` output dict assembly | `DMRGMatvec2Site.apply()` | ~4000 |
| `_lanczos_solve_np` outer loop | `cython_lanczos_ground` | ~200 |
| `ba_inner` in Lanczos | direct `cdef _ba_inner_impl` | ~4000 |
| `ba_norm` in Lanczos | `cdef _ba_norm_impl` (ddot+sqrt) | ~4000 |
| `ba_scale` in Lanczos | `cdef _ba_scale_impl` (dscal) | ~4000 |
| `ba_add` in eigvec reconstruction | `cdef _ba_axpy_impl` loop | ~4000 |

### Expected impact

At chi=32: per-combo dispatch drops from ~26μs to ~3-5μs (BLAS call + dict
lookup). Closes the 2.2x gap to ~1.1-1.2x.

At chi=128: dispatch overhead is smaller fraction but still helps. Expect
improvement from 1.7x to ~1.3-1.4x (remaining gap is SVD, out of scope).

## Integration

Single integration point in `_two_site_update_symmetric_np` (and 1-site
variant):

```python
if _USE_CYTHON_LANCZOS:
    mv = DMRGMatvec2Site(combo_descs, env_blocks, theta_buf_idx,
                         output_keys, output_shapes)
    energy, theta_opt_blocks = cython_lanczos_ground(
        mv, theta_ba.blocks, config.lanczos_max_iter, config.lanczos_tol,
        list(theta_ba.blocks.keys()),
    )
    theta_opt_ba = BlockArray(blocks=theta_opt_blocks, indices=_out_indices)
else:
    energy, theta_opt_ba = _lanczos_solve_np(matvec, theta_ba, ...)
```

Fallback: existing `_lanczos_solve_np` + `_execute_matvec_combos` stays as-is.

Feature flag: `_USE_CYTHON_LANCZOS` in `contraction/__init__.py`, detected at
import time like the existing flags.

## Testing

- **Unit: `DMRGMatvec2Site.apply()`** — compare against numpy einsum for
  known block combos (2-site and 1-site).
- **Unit: `cython_lanczos_ground`** — small tridiagonal system, compare
  eigenvalue against `np.linalg.eigh`.
- **Integration:** existing `test_dmrg_cython.py` DMRG energy convergence.
- **Benchmark:** extend `test_blas_benchmark.py` to time full Lanczos loop.

## Build

No new files or dependencies. `_cython_blas.pyx` already compiles with
`hatch-cython`; `cdef class` is standard Cython 3.0. The `.so` grows ~50KB.

## Scope excluded

- SVD/QR block assembly/reconstruction (separate effort if needed)
- General `_contract_symmetric` in contractor.py (stays JAX-based for CTM)
- GPU paths (Cython BLAS is CPU-only)
