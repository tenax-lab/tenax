# Multi-GPU Sharded DMRG Design

**Date:** 2026-03-31
**Status:** COMPLETED (PR #209) — multi-GPU sharding merged
**Branch:** `feat/tpu-gpu-dmrg`
**Prerequisite:** Tasks 1-11 (JIT-compiled DMRG sweep) — completed

## Goal

Shard the dense JIT DMRG sweep across 2 GPUs to enable chi=1000-2000 at
L=20-40. Use JAX GSPMD via `NamedSharding` — minimal code changes, XLA
handles communication.

## Scope

- Dense path only (`not use_symmetric`). Symmetric path deferred but the
  architecture supports it via `PaddedBlockArray` uniform shapes.
- Target: 2x RTX 4070 Ti SUPER (11.7 GB each), float64.
- chi=1000 (2.3 GB/GPU), chi=1500 (5.2 GB/GPU), chi=2000 (9.3 GB/GPU).

## Sharding Strategy

**Mesh:** 1D mesh with `N` devices, axis name `'chi'`.

```python
mesh = jax.sharding.Mesh(jax.devices()[:N], axis_names=('chi',))
```

**Tensor sharding:**

| Tensor | Shape | PartitionSpec |
|--------|-------|---------------|
| `mps_stack` | `(L, chi, d, chi)` | `P(None, 'chi', None, 'chi')` |
| `left_envs` | `(L+1, chi, D_w, chi)` | `P(None, 'chi', None, 'chi')` |
| `right_envs` | `(L+1, chi, D_w, chi)` | `P(None, 'chi', None, 'chi')` |
| `W_stack` | `(L, D_w, d, d, D_w)` | `P()` (replicated) |
| `energies` | `(num_sweeps,)` | `P()` (replicated) |

Both bond dimension axes of MPS and environments are sharded. XLA/GSPMD
automatically inserts all-reduce for einsums that contract along sharded
axes.

## API

**New function:** `jit_dmrg_sweep_dense_sharded` in `_jit_sweep.py`.

Thin wrapper around existing `_jit_sweep_loop`:
1. Create `Mesh` from available devices
2. Shard input arrays via `jax.device_put` + `NamedSharding`
3. Re-JIT `_jit_sweep_loop` with sharding annotations
4. Return unsharded results

**New dispatch option:** `DMRGConfig(accelerator="sharded")`.
- Auto-detect device count; fall back to `"jit"` if single device
- Warmup-to-JIT transition still applies
- Dense 2-site only (same guard as `"jit"`)

**No changes to `_jit_sweep_loop` internals.** Sharding is purely an
execution concern applied at the call boundary.

## SVD Handling

The padded SVD reshapes theta `(chi, d, d, chi)` → `(chi*d, d*chi)` then
calls `jnp.linalg.svd`. With chi sharded on both axes, XLA must handle a
sharded SVD.

**Primary:** Let XLA partition the SVD automatically. XLA has had sharded
SVD support since JAX ~0.4.20.

**Fallback:** If XLA gathers to one device for SVD, that's acceptable for
chi <= 2000. The SVD matrix at chi=2000 is (4000, 4000) = ~250 MB float64,
fits on one GPU. If this becomes a bottleneck, add explicit `all_gather`
before SVD and scatter afterward — a local change inside
`padded_svd_dense`.

## Testing

1. `test_sharded_sweep_matches_single_device` — same energy within 1e-6
2. `test_sharded_dispatch_via_dmrg` — end-to-end via `dmrg()` entry point
3. `test_sharded_fallback_single_device` — graceful fallback to `"jit"`

## Memory Budget (float64, 2 GPUs, L=40)

| chi  | Per-GPU | Status |
|------|---------|--------|
| 1000 | 2.3 GB  | OK     |
| 1500 | 5.2 GB  | OK     |
| 2000 | 9.3 GB  | OK     |

## Non-Goals

- Symmetric/block-sparse sharding (future work)
- Multi-node (single-node 2-GPU only)
- chi > 2000 (would need float32 or more GPUs)
- Pipeline parallelism across sweep sites
