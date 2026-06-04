# Symmetric CTM AD — stacked-block representation (even-D first)

**Date:** 2026-06-04
**Issue:** #566 (symmetric iPEPS AD compile + runtime cost)
**Status:** design, pending implementation plan
**Scope of this spec:** P0 (accuracy spine) → P1d (even-D stacked rep threaded through the
CTM sweep + A100 measurement gate). Ragged odd-D/U(1) (P1e) and the cuTensorNet GPU
kernel (P2) are a roadmap, **gated on P1d proving the efficiency gain**.

## Problem (measured, not assumed)

Block-sparse `SymmetricTensor` CTM-AD is dominated by **per-block structural op
emission**. The Python loop over blocks unrolls at trace time into O(n_blocks) XLA
primitives, and this is paid twice:

- **Compile** (one-time, the #566 headline): A100 fermionic `value_and_grad` compile
  = 206 / 2111 / 2379 s at D=2/3/4 (16 blocks held constant); ferm/dense ratio
  4× → 52× → 61×. Dense is flat (~40–60 s, χ- and D-independent — XLA dedups one block).
- **Runtime** (every optimizer step): fermionic warm step 4.5–9.5 s vs dense 0.5–2.3 s
  — the #195/#200 "1.15M tiny kernel launches" problem.

Root cause, localized by jaxpr op-histogram + monkeypatch counters on one traced forward
sweep (fermionic D=2):

- The op explosion is in the **forward** CTM step (21,920 ferm ops vs 1,472 dense = 15×).
  The backward only adds ×1.3 on top — it is huge *because the forward it differentiates
  is huge*.
- `dot_general` (contraction) is only **~4%** of the graph. The cost is **structural**:
  `_get_block` (slice+reshape off the flat buffer) ×1428 and the `.blocks` property ×766,
  plus per-combo contractor scaffolding (broadcast/select/transpose/charge-compares).
- Ruled out empirically: scan-fusion (compile already depth-flat), `TENAX_BATCH_BLOCKSPARSE`
  within-call batching (#571/2/3 — neutral/worse on A100, attacks the wrong 4%), #570 as a
  *compile* lever (dense is χ-flat).

### Even-D vs ragged (the reason this spec scopes to even-D)

For `FermionParity` with alternating virtual charges, **even D ⇒ all blocks of a tensor
share one shape** (D2/D4 site A: 16 blocks, 1 distinct shape → up to 16× collapse). **Odd D
and general U(1) ⇒ unequal sectors ⇒ blocks fragment into distinct shapes** (D3 site A: 16
blocks, 16 shapes → no collapse from grouping alone; contraction-level collapse only ~2.1×).
Even-D is the clean, decisive case and the right place to *prove the gain* before paying for
ragged handling.

## Approach

Add a **stacked working representation** over the **unchanged** flat `_data` buffer and
single-pytree-leaf storage (that storage is #87's deliberate JIT/grad + accelerator
substrate — it is **not** reverted).

- **Representation layer** (`tensor.py`): `stacked_blocks()` → `{block_shape: (group_keys,
  array[n, *block_shape])}` built by **one reshape/gather of `_data` per shape-group**
  (O(n_groups)); inverse `from_stacked_blocks(...)` writes back with **one scatter per
  group**. Grouping is pure static index math over `_block_keys`/`_block_shapes`/
  `_block_offsets` (trace-time, on static metadata); only the `_data` reshape/gather/scatter
  is a traced op.
- **Operation layer**: stacked-aware `contract`, `fuse`/`split`, `svd`/`qr`/`eigh`,
  `transpose` — each emits **O(n_groups)** ops via batched-einsum / `vmap` over the leading
  block axis + `segment_sum`. Reuses `_grouped_decomp_by_shape` (#572).
- **Backend dispatch behind one interface**: pure-JAX stacked (default, CPU/GPU/TPU) now;
  cuTensorNet `custom_call` (P2, GPU) later. Sweep/AD/tests are backend-agnostic.

**Load-bearing boundary decision:** the stacked rep **persists across the ~387 calls within
a sweep**; we pack back to `_data` only at sweep boundaries (where the pytree leaf matters
for jit/scan carry). Per-call re-pack/unpack is exactly what made within-call batching
net-neutral; persisting it is what unlocks the across-call collapse.

### Why this cuts both compile and runtime

For even-D, n_groups ≈ 1, so the per-block round-trip (`_get_block`/`.blocks`, ~96% of the
graph) collapses ~16× → fewer traced ops → super-linear compile win (op ratio 19× mapped to
compile ratio 52× at D3 in prior data), and one batched kernel replaces n_blocks tiny ones →
runtime win. The VJP of stacked ops is itself stacked ops (vmap/`segment_sum` differentiable),
so the backward graph shrinks *with* the forward.

## Components & boundaries

| Unit | Responsibility | Depends on |
|---|---|---|
| `StackedView` (new, `tensor.py`) | build groups from static metadata; gather `_data`→stacks; scatter stacks→`_data` | `_block_*`, `_data` |
| stacked contractor (`contractor.py`) | group block-pairs by `(lhs,rhs,out shape, subscripts)`; batched einsum + `segment_sum` | `StackedView`, static charge matching |
| stacked decomposition (`linalg`) | `_grouped_decomp_by_shape` as the stacked-path default; consistent gauge-fix | #572 |
| CTM sweep threading | carry stacked tensors through absorb/projector/RDM; pack to `_data` only at boundaries | all of the above |
| backend dispatch | select pure-JAX vs (P2) cuTensorNet behind one signature | — |

Each unit is independently testable: `StackedView` against the per-block accessors; the
stacked contractor/decomp against the per-block ops; the sweep against the energy/grad golden.

## Data flow — one CTM sweep

```
site tensors (_data) ─stacked_blocks()─▶ stacks
   └▶ absorb / projector contractions on stacks (batched einsum + segment_sum)
   └▶ projector SVD on stacked bond matrices (grouped, gauge-fixed)
   └▶ env tensors kept stacked across calls (no per-call repack)
   └▶ from_stacked_blocks() once at the sweep boundary ─▶ _data ─▶ energy
Backward: VJP of stacked ops = stacked ops ⇒ backward op-count shrinks with the forward.
```

## Accuracy contract (the guarantee)

Built **as P0, before any restructuring**: the current per-block path is frozen as the
golden reference; every later phase is asserted against it by a comparator with three
explicit tiers:

- **Bit-identical (`atol=0`):** `from_stacked(stacked(A)) == A`; contracted tensor *values*
  and fuse/split round-trips **when accumulation order is preserved**. We impose a canonical
  block ordering on `segment_sum` so the reduction matches sequential per-block accumulation
  wherever the segment structure allows.
- **Bounded-fp (`rtol ≈ 1e-12` in f64, stated & asserted):** energy, gradient, and any
  output where XLA may reassociate a reduction (the GPU `segment_sum` ~5e-7-class drift lives
  here — bounded, not hand-waved).
- **Gauge-invariant only (raw `U`/`Vh` exempt):** SVD/eigh/QR compared on singular values,
  eigenvalues, and reconstruction (`A=UΣVᴴ`, `QR=A`) — **never** raw factors, because
  `vmap`-ed LAPACK sign-/basis-flips degenerate subspaces (the #572 trap).

Golden coverage: fermionic + U(1) + dense, D∈{2,4} (even), **including a degenerate-SV case
and a rank-deficient sector**. (Odd D=3 is captured as a golden too, but its stacked path is
out of this spec's implementation scope — it stays on the per-block path until P1e.)

## Testing

- **P0 spine:** golden capture (per-block = reference) + tier-asserting comparator.
- **Unit:** stacked round-trip (bit-exact); stacked-vs-per-block contract (tiered); stacked
  decomp gauge-invariants; FD gradient + per-block-vs-stacked VJP agreement.
- **Integration:** full CTM-AD energy+grad vs golden, fermionic/U(1)/dense, even D.
- **Perf gate (A100, documented — not CI-asserted):** `vg_cmp` compile + warm-step, even D
  ∈{2,4} (extend to D=6 if it holds), stacked vs per-block. **Target: collapse the measured
  52–61× fermionic/dense compile ratio toward dense, plus a warm-step win.** This is P1d and
  the decision gate for P1e/P2.

## Phasing (committed scope = P0–P1d)

- **P0** — accuracy spine (golden + comparator + tier contract). Ships alone.
- **P1a** — `StackedView` + round-trip, no math change (byte-exact).
- **P1b** — stacked contractor, even-D, validated against golden tiers.
- **P1c** — stacked decomposition + fuse/split, even-D.
- **P1d** — thread stacked rep through the CTM absorb/RDM so it persists across calls;
  **A100 compile + runtime measurement = the gate.**

Roadmap, gated on P1d showing the gain:

- **P1e** — ragged odd-D/U(1) via group-by-exact-shape (+ optional pad-to-max behind a
  sub-flag where it provably wins).
- **P2** — cuTensorNet differentiable `custom_call` behind the same interface (GPU);
  supersedes the existing eager `TENAX_USE_CUTENSOR_BLOCKSPARSE` path with a tracer-safe,
  differentiable XLA op (fwd + hand-written transpose-contraction VJP).

## Risks / open questions

- **Forward→backward coupling (P1d):** the load-bearing assumption is that cutting forward
  op-count shrinks the backward graph. The op-histogram (backward = ×1.3 of forward) supports
  it, but P1d **measures** it before P1e/P2 are funded.
- **Even-D-only win in Phase 1:** odd-D/U(1) stay on the per-block path until P1e; ragged is
  largely P2's job. Expectation set explicitly so a strong even-D result is not over-read.
- **Blast radius:** threading the stacked rep touches the most-used paths (contractor, fuse,
  decomp, absorb, RDM). Mitigated by persist-at-sweep-boundaries, the P0 golden spine, and
  shipping P1a–P1d incrementally with byte-exact gates at each step.
- **`segment_sum` reduction order on GPU:** handled by the tier contract; the energy/grad
  bound must be re-verified to hold on A100, not only CPU.

## Non-goals

- No change to flat-buffer storage or the single-pytree-leaf layout (#87 substrate).
- No revert to list-of-blocks (undoes #87's JIT/grad win and the #200/#568/#569 substrate).
- No within-call contraction batching as a compile lever (`TENAX_BATCH_BLOCKSPARSE` —
  measured dead for compile; stays a harmless opt-in).
