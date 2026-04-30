# TPU/GPU Accelerated DMRG — Design Document

**Date**: 2026-03-30
**Status**: COMPLETED (PR #209) — JIT DMRG for GPU/TPU + multi-GPU sharding
**Branch**: TBD

## Motivation

Tenax's DMRG has two execution paths: a numpy/Cython block-sparse path optimized
for CPU (1.7-2.2x TeNPy at chi=128-256), and a JAX dense path that is
GPU/TPU-capable but not optimized for accelerators. The PRX Quantum paper
("Density Matrix Renormalization Group with Tensor Processing Units", PRX Quantum
4, 010317, 2023) demonstrates 100x speedups by JIT-compiling the entire DMRG sweep
with static-shape tensors on TPU.

This design brings those techniques to Tenax for both dense and block-sparse
(symmetric) tensors, targeting GPU and TPU equally.

## Scope

- **2-site DMRG only** (1-site may be added later)
- **Dense and symmetric tensors** on accelerator (padded vmap approach for symmetric)
- **Automatic dispatch**: CPU symmetric → existing numpy/Cython path; accelerator → JIT path
- **Single-device first**, with multi-device as a future extension point
- **No API changes** visible to users beyond a new `accelerator` field on `DMRGConfig`

## Design

### 1. Padded Tensor Representation

A new `PaddedBlockArray` class stores all charge-sector blocks as a single
`(num_blocks, M_max, N_max)` JAX array, plus a boolean mask of the same shape.

```python
class PaddedBlockArray:
    data: jax.Array        # (num_blocks, M_max, N_max) — zero-padded
    mask: jax.Array        # (num_blocks, M_max, N_max) — True for real data
    block_charges: tuple   # static: charge label per block
    block_shapes: tuple    # static: (m_i, n_i) actual shape per block
    indices: tuple         # TensorIndex metadata for reconstruction
```

- `data` and `mask` are JAX pytree leaves (traced during JIT).
- `block_charges`, `block_shapes`, `indices` are pytree aux (static).
- Conversion: `SymmetricTensor.to_padded()` / `PaddedBlockArray.to_symmetric()`.
- For `DenseTensor`: no `PaddedBlockArray` needed — pad the raw `jax.Array` to
  `chi_max` with zeros.

### 2. vmap Block-Sparse Contractions

Contractions on `PaddedBlockArray` use a static `PaddedContractionPlan`:

```python
class PaddedContractionPlan:
    left_block_indices: tuple[int, ...]
    right_block_indices: tuple[int, ...]
    output_block_indices: tuple[int, ...]
    subscripts: str
```

Execution: gather participating blocks by index, `jax.vmap` the per-block einsum,
scatter-add results to output:

```python
def contract_padded(plan, A, B):
    a_blocks = A.data[plan.left_block_indices]
    b_blocks = B.data[plan.right_block_indices]
    results = jax.vmap(lambda a, b: jnp.einsum(plan.subscripts, a, b))(a_blocks, b_blocks)
    output = jnp.zeros((num_out, M_max, N_max))
    output = output.at[plan.output_block_indices].add(results)
    return PaddedBlockArray(output, ...)
```

The plan is entirely static — block indices don't change within a sweep. JIT
compiles once, inner loop is pure XLA.

For dense path: direct `jnp.einsum`, no block indexing.

### 3. JIT-Fused Lanczos + SVD + Truncation

**Lanczos**: Extend `_lanczos_solve_jit` (existing `jax.lax.fori_loop`
implementation) to operate on `PaddedBlockArray` via `contract_padded` matvecs.

**SVD**: Per-block SVD via `jax.vmap(jnp.linalg.svd)`. Global truncation via
`jax.lax.top_k`:

```python
def padded_svd(pba, chi_max):
    U_all, s_all, Vh_all = jax.vmap(jnp.linalg.svd)(pba.data)
    s_flat = s_all.ravel()
    top_values, top_indices = jax.lax.top_k(s_flat, chi_max)
    # derive per-block keep counts, mask columns of U/Vh
    return U_padded, s_truncated, Vh_padded
```

`jax.lax.top_k` returns static-shape output, so the entire SVD+truncation is
JIT-compatible with no host-device sync. Per-block column selection uses masking
rather than dynamic slicing.

For dense path: `jnp.linalg.svd` on full matrix, keep first `chi_max` columns.

### 4. `lax.scan`-Based Full Sweep

Once chi saturates at `chi_max`, the sweep becomes a `jax.lax.scan` over sites:

```python
def sweep_step(carry, site_data):
    mps_tensors, left_envs, right_envs = carry
    W_l, W_r = site_data

    theta = contract_padded(mps[i], mps[i+1])
    theta_opt, energy = lanczos_padded(theta, left_envs[i], W_l, W_r, right_envs[i+2])
    U, s, Vh = padded_svd(theta_opt, chi_max)

    mps_tensors = mps_tensors.at[i].set(U)
    mps_tensors = mps_tensors.at[i+1].set(s @ Vh)
    left_envs = left_envs.at[i+1].set(update_left_env_padded(...))

    return (mps_tensors, left_envs, right_envs), energy

final_carry, energies = jax.lax.scan(sweep_step, init_carry, site_data)
```

All arrays are fixed shape (padded to `chi_max`). Left-to-right and right-to-left
are separate scans compiled into a single XLA program.

### 5. Automatic Dispatch and User API

New config field:

```python
class DMRGConfig:
    accelerator: str = "auto"  # "auto" | "jit" | "off"
```

Dispatch logic:
- `"off"`: existing Python sweep (unchanged)
- `"auto"`: CPU + symmetric → numpy/Cython path; GPU/TPU → JIT path;
  CPU + dense → JIT path (still benefits from fused sweep)
- `"jit"`: force JIT path regardless of device

Warmup → JIT transition (transparent):
1. Sweeps 1..N: Python loop, bond dimension grows → `chi_max`
2. Once chi saturated: convert to padded representation, run `lax.scan` sweeps
3. On completion: convert back to `SymmetricTensor`/`DenseTensor` for return

`DMRGResult` is unchanged. Multi-device extension point via optional
`device_mesh: Optional[jax.sharding.Mesh]` on `DMRGConfig` (future work).

### 6. Testing Strategy

1. **PaddedBlockArray round-trip**: `SymmetricTensor → to_padded() → from_padded()`
   is identity. Same for `DenseTensor` pad/unpad.
2. **contract_padded correctness**: compare vs `contract()` on `SymmetricTensor` for
   environment updates, multiple symmetries (U(1), Z2).
3. **padded_svd correctness**: compare global truncation vs `truncated_svd`, verify
   singular values and subspaces match.
4. **Full DMRG integration**: Heisenberg chain L=10 chi=16, compare Python path vs
   JIT path energy to 1e-10. Run for dense, U(1) symmetric, and Z2 symmetric.
5. **CI**: All tests run on CPU with `accelerator="jit"`. GPU/TPU correctness follows
   from JAX backend abstraction. Mark accelerator benchmarks as `@pytest.mark.slow`.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Block organization | Pad all to max, vmap | Simplest to shard across devices; memory overhead acceptable on accelerators |
| SVD truncation in JIT | `jax.lax.top_k` | Static output shape, no host-device sync |
| Sweep compilation | `jax.lax.scan` | Single XLA program, no Python loop overhead |
| Warmup strategy | Python then JIT | No wasted FLOPs while chi grows; full JIT once saturated |
| Scope | 2-site only | Simpler, covers main use case, matches PRX Quantum paper |
| Dispatch | Automatic by device | User doesn't need to know about internals |

## Non-Goals (Future Work)

- 1-site DMRG on accelerator
- Multi-device sharding (design supports it, implementation deferred)
- iDMRG on accelerator (different sweep structure)
- Subspace expansion / DMRG3S on accelerator
