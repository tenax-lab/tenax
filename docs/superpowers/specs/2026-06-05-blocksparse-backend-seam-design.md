# Block-sparse contraction backend seam (#200) — design

**Date:** 2026-06-05
**Issue:** #200 / #566 (symmetric iPEPS AD compile + runtime)
**Status:** design → implementation
**Supersedes:** the pure-JAX P1d deep-restructure path. Decision (2026-06-05): pure-JAX
contractions-only persistence measured 0.995× on the real sweep (the 96%-structural
fuse/construct cost is intrinsic to per-block JAX trace emission); the principled fix is a
single custom block-sparse contraction kernel that emits O(1) ops per contraction, fixing
*both* the compile wall (#566) and the tiny-kernel runtime (#195). cuTensorNet first (GPU),
behind a kernel-agnostic seam so Pallas/TPU can be added later without re-plumbing.

## Goal

A kernel-agnostic backend seam for block-sparse `SymmetricTensor` contraction:

- the **reusable ~70%** (plan + dispatch + VJP structure + accuracy harness) is pure-JAX,
  CPU-testable, and shared by every backend;
- **cuTensorNet** is the first GPU backend behind it;
- **Pallas/TPU** (later) is strictly additive — a new backend against the same seam, reusing
  the plan, the VJP convention, and the tests. Only the kernel code is hardware-specific.

The flat `_data` buffer + static block metadata (PR #87) is the data contract every backend
consumes; it is unchanged.

## What carries across backends (the seam) vs what doesn't (the kernel)

Reusable, built in Phase A: the dispatch seam; the flat-buffer→backend data contract; the
**block-contraction plan** (charge matching, valid-output filtering, block-combo grouping,
segment/accumulation structure, gather indices — the static logic already in
`_contract_symmetric_stacked`); the JAX-primitive + **VJP structure** (the backward of a
block-sparse contraction is transposed contractions via the same plan machinery); the
three-tier accuracy harness. NOT reusable (and never could be): the CUDA/cuTensorNet API code
vs Pallas/Mosaic kernel code.

## Components

### 1. `BlockContractPlan` (backend-agnostic, from static metadata only)

Extracted from `_contract_symmetric_stacked` (contractor.py). Computed once from block
keys/shapes (no tensor data):

```python
@dataclass(frozen=True)
class BlockContractPlan:
    out_indices: tuple[TensorIndex, ...]
    out_block_keys: tuple[BlockKey, ...]
    out_block_shapes: tuple[tuple[int, ...], ...]
    batched_subscripts: str                 # batch label prepended
    groups: tuple[PlanGroup, ...]           # per input-shape-group

@dataclass(frozen=True)
class PlanGroup:
    operand_rows: tuple[tuple[int, ...], ...]   # per operand: which stacked rows feed the batched einsum
    segment_ids: tuple[int, ...]                # output-key accumulation segments
    num_segments: int
    out_shape: tuple[int, ...]

def build_block_contract_plan(tensors, subscripts, output_indices) -> BlockContractPlan | None:
    # None => unsupported here (e.g. >2 tensors / multi-shape-group) -> caller falls back.
```

### 2. `BlockSparseContractBackend` protocol

```python
class BlockSparseContractBackend(Protocol):
    name: str
    def available(self) -> bool: ...                      # platform + library present
    def supports(self, tensors, plan) -> bool: ...        # dtype / symmetry / shape-uniformity
    def execute(self, operand_stacks, plan) -> jax.Array: ...   # stacked operands -> output _data
```

Backends:
- **`StackedJaxBackend`** — Task-3 pure-JAX stacked execution (gather rows + batched einsum +
  `segment_sum`). Available everywhere; supports 2-tensor single-shape-group. Naturally
  differentiable (jnp ops) → no custom VJP needed.
- **`CuTensorNetBackend`** (Phase B, GPU) — FFI `custom_call`; custom VJP via transposed plan.
- **(future) `PallasBackend`** (TPU) — Mosaic kernel; same VJP convention.
- The existing **per-block path** stays as the universal fallback (used when
  `build_block_contract_plan` returns None or no backend applies).

### 3. Dispatch

In `_contract_symmetric`, after parsing:
```python
plan = build_block_contract_plan(tensors, subscripts, output_indices)
if plan is not None:
    backend = select_backend(tensors, plan)   # env override TENAX_BLOCKSPARSE_BACKEND,
    if backend is not None:                    # else platform + available() + supports()
        return _execute_backend(backend, tensors, plan)   # AD-wrapped
# fall through to per-block path
```
`select_backend` precedence: explicit env (`stacked`/`cutensornet`/`perblock`/`auto`) →
on GPU prefer an available `CuTensorNetBackend` → else `StackedJaxBackend` if it supports →
else None (per-block fallback). Default remains **per-block** unless `TENAX_STACK_BLOCKSPARSE`/
backend env opts in (byte-identical default, as today).

### 4. VJP seam (so opaque FFI backends plug in)

A block-sparse contraction is multilinear in its operands; its VJP w.r.t. operand *i* is a
contraction of the cotangent with the other operand(s), expressible by a **transposed plan**.
The seam provides `transpose_plan(plan, wrt_operand) -> BlockContractPlan` so every backend's
backward reuses the same machinery:
- `StackedJaxBackend`: JAX differentiates its jnp ops directly (free).
- FFI backends (`CuTensorNetBackend`, `PallasBackend`): register a `jax.custom_vjp` whose
  backward calls `execute` on the transposed plan (on the same backend), so only the forward
  kernel is hardware-specific; the backward *structure* is shared.

## Accuracy contract (unchanged)

The three-tier comparator (tests/stacked/_harness.py) governs every backend: bit-identical for
data movement, bounded-fp (rtol/atol 1e-12) for reductions, **gauge-invariant only** for SVD.
Lesson from the P1d drift investigation: compare converged-sweep energies **only at full
convergence** (tol 1e-12), never at a fixed loose iteration — early-iteration gauge freedom in
the SVD projector makes raw env data differ by O(1) under 1e-16 input perturbations while the
gauge-invariant energy is identical at convergence.

## Phasing

- **Phase A (CPU-local):**
  - **A1** extract `BlockContractPlan` + `build_block_contract_plan` (pure refactor of
    `_contract_symmetric_stacked`); existing stacked tests stay green.
  - **A2** `BlockSparseContractBackend` protocol + `select_backend` + register
    `StackedJaxBackend`; wire `_contract_symmetric` dispatch; equivalence preserved.
  - **A3** VJP seam (`transpose_plan` + custom_vjp path) + a **`MockFFIBackend`** (a pure-JAX
    stand-in that mimics an opaque `custom_call` with a hand-written transposed-plan VJP),
    asserting value+grad match per-block at even D — **proves the seam hosts an opaque
    differentiable backend before any GPU exists.**
- **Phase B (A100/CUDA handoff):** `CuTensorNetBackend` — FFI `custom_call` (extend the
  existing eager `tenax/contraction/cutensor_blocksparse.py` to a tracer-safe differentiable
  op), register for GPU, slot in via the A3 VJP convention. Validate against the accuracy spine
  and `StackedJaxBackend`; then the compile/runtime measurement on A100.

## Non-goals / constraints

- Flat-buffer storage + single pytree leaf unchanged (#87 substrate).
- Default execution path unchanged (per-block) unless opted in; byte-identical default.
- No Pallas code in Phase A — only the seam it will plug into. No cuTensorNet-isms allowed to
  leak past the backend boundary (the lock-in risk): the contractor sees only the protocol.
- Even-D / 2-tensor single-shape-group scope for the plan initially (ragged is a later backend
  concern); per-block fallback covers everything else.
