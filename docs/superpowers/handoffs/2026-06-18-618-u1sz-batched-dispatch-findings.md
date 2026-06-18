# #618 — Batched block-sparse dispatch to close the D≥3 U(1)-Sz CTM-AD warm wall — Findings

**Date:** 2026-06-18
**Parent:** #566 (block-sparse AD cost scales with charge-block count). **Motivated by:** #615.
**Verdict:** **NO-GO (measured, not inferred).** The lever #618 proposes — "batch same-shape blocks
so eager cost goes O(n_blocks) → O(n_distinct_shapes)" — is (a) **already implemented** for the
contraction path and gated by the same `TENAX_BATCH_BLOCKSPARSE` flag step 1 measured, and (b)
**structurally incapable** of closing the 24× warm gap. The warm wall is **host-bound eager per-block
dispatch dominated by the fuse** (`_fuse_indices_symmetric`, 8 eager forward sweeps, ~19 s) — *not*
the contraction or the device. Batching the fuse = the #566 stacked-fuse lever (already NO-GO, within
1% noise); batching the contraction = the gate (already measured +14% warm). Both cap at the U(1)-Sz
same-shape collapse ceiling ~1.9× against a 24× gap. No `src/` change.

## Question (issue #618, steps 2–4)

After step 1 (existing `TENAX_BATCH_BLOCKSPARSE=1` helps compile ~21%, **hurts** warm ~14% at D=3
χ=12), step 2 asks: where does the u1sz warm-step eager time go (block-pack vs contraction vs
decomp)? Step 3: prototype batched contraction/block-pack dispatch. Step 4: does it cut the warm gap
materially toward dense?

## Answer: the premise is wrong, and the measured ceiling kills the lever.

### Premise correction (load-bearing): contraction IS already batched.

The issue states "Batching today covers decompositions; the CTM-AD warm-step hot path (eager
contraction + block-pack assembly) appears **not** to be batched." **This is false.** The same
`TENAX_BATCH_BLOCKSPARSE` flag routes symmetric contraction through `_contract_symmetric_batched`
(`src/tenax/contraction/contractor.py:785`), which groups surviving block-combos by input
block-shape signature, `jnp.stack`s same-shape blocks on a fresh batch axis, runs **one batched
`jnp.einsum` per shape group**, and `jax.ops.segment_sum`s combos sharing an output key
(`contractor.py:241–371`). That is *exactly* the step-3 lever ("O(n_blocks) → O(n_distinct_shapes)").
So step 3 is not a new build — it exists, and step 1 already measured it end-to-end.

**Tooling caveat that hid this:** `examples/probe_backward_jaxpr_566.py` compares gate off vs on
**in one process**. `jax.make_jaxpr` reuses the first (gate-off) trace for the second (gate-on) call
when avals are identical — the same `jax.jit`-cache contamination the #610 handoff flagged — so the
in-process probe reports a spurious **1.00× identical** histogram. The gate's effect is only visible
with **separate processes per gate value** (or `jax.clear_caches()` between). Verified directly: with
a cold trace and the gate on, `_contract_symmetric_batched` fires (call-counted = 1) and emits
`segment_sum`/batched einsum; with the contaminated in-process re-trace it does not.

### Step 2 — where the warm time goes (clean separate-process backward op histogram)

Full fused backward VJP (env-sweep + params-sweep), u1sz D=3 χ=12, 32 blocks, trace-only, **one
process per gate** (`/tmp/trace618.py`):

| bucket | gate OFF (per-block) | gate ON (batched) | ratio |
|---|---:|---:|---:|
| contraction (`dot_general`) | 8,890 | **2,807** | **0.32×** ✓ batched einsum collapses it |
| transpose | 13,112 | **4,210** | 0.32× ✓ |
| decomp (svd/eigh/qr) | 156 | 90 | 0.58× ✓ |
| **block-pack (slice/scatter/reshape)** | **58,656** | **66,871** | **1.14×** ✗ grows |
| broadcast_in_dim | 29,808 | 36,457 | 1.22× ✗ grows |
| charge-mask / index arith | 21,717 | 20,993 | 0.97× ~flat |
| elementwise | 31,014 | 31,984 | 1.03× ~flat |
| other | 37,899 | 59,328 | 1.57× ✗ grows |
| **TOTAL** | **171,444** | **186,283** | **1.087× — MORE ops** |

**Reading:** block-pack (slice/scatter/reshape) is the dominant cluster (34% of all ops), ≫
contraction (5%) ≫ decomp (0.1%). Batching the contraction *works* (it collapses contraction 3.2×
and transpose 3.1×), but contraction is only 5% of the graph, and the batching **machinery** (stack
into a batch axis, `segment_sum` scatter, broadcasts) **adds ~15k ops** to the dominant block-pack /
broadcast / "other" clusters. Net op count goes **up +8.7%** → more device work → the measured **+14%
slower warm** (step 1). The batchable cluster is minor; the dominant cluster grows.

### Step 2 — the warm wall is HOST-bound eager per-block dispatch, dominated by the FUSE

A100, D=3 χ=12 depth=8, implicit fixed-point AD, `examples/profile_warm_dispatch_618.py`:

| arm | n_blocks | warm_med | dispatch (host return) | sync | pipeline ×4 speedup |
|---|---:|---:|---:|---:|---:|
| dense | 1 | 778.9 ms | 775.1 ms (99.5%) | ~0 ms | 1.01× |
| u1sz | 32 | **18,482.6 ms** (~24× dense) | 18,498 ms (100.1%) | ~0 ms | 1.00× |

> ⚠️ **Caveat (PR #625 review):** the `dispatch`/`sync` split and the pipeline probe above are **not**
> clean host-vs-device discriminators. The production loss converts the convergence scalars to Python
> floats every sweep (`_ctm_loop_core.py:187-188`: `float(_max_eps)` / `float(_max_S)`), which blocks
> on the device once per `max_iter` *inside* `value_and_grad`. So `sync ≈ 0` is **forced by those
> internal syncs** and would look identical for genuinely device-bound work; treat those two columns
> as indicative only. The load-bearing host-bound evidence is the **cProfile** below, which measures
> host dispatch directly.

`value_and_grad` returns only after the work is essentially done — and the cProfile shows the host
cost directly: the forward CTM sweep runs **eagerly in Python**, not as one compiled unit. The warm
`value_and_grad` cProfile (tottime / cumtime, u1sz):

| frame | ncalls | tottime | cumtime |
|---|---:|---:|---:|
| `_tensor_utils.py:231 _fuse_indices_symmetric` | **8** | 0.245 s | **19.159 s** |
| `_ctm_loop_core.py:116 _run_ctm_loop_with_bump` | 1 | 0.457 s | 5.673 s |
| `jax .../lax.py _convert_element_type` | **73,748** | 0.359 s | 2.695 s |
| `jax .../array_constructors.py array` | **32,820** | 0.338 s | 2.408 s |
| `jax .../indexing.py _parse_indices` | **17,567** | 0.288 s | 0.596 s |
| `jax .../core.py __new__` | **62,392** | 0.211 s | — |
| `_ctm_energy_ad.py:1281 f_bwd` (jitted backward) | 1 | 0.132 s | **0.134 s** |

**Reading:** the dominant warm cost is `_fuse_indices_symmetric` — **cumtime 19.16 s across 8 eager
sweeps** (one per `max_iter`), driving **tens of thousands** of Python-dispatched JAX micro-ops
(73,748 `convert_element_type`, 32,820 array constructions, 62,392 aval `__new__`). The **fuse /
block-pack** path — slicing the flat buffer per block, transposing/reshaping, scattering into fused
output blocks — runs **per block, eagerly, in Python, every sweep**. By contrast the jitted
fixed-point backward (`f_bwd`) is **0.134 s** — the backward *is* compiled and fast; it is the
**eager forward** that is the 18.5 s wall. This is **host-bound eager per-block dispatch**, exactly
the mechanism #618 names — but the dominant cluster is the **fuse**, not contraction. The host-bound
verdict rests on **this cProfile** (tens of thousands of Python-dispatched micro-ops — a count a
device-bound compiled graph cannot produce, and far more than the ~8 per-step `float()` syncs could
account for), **not** on `dispatch ≈ warm / sync ≈ 0 / pipeline 1×`, which are contaminated by the
per-sweep scalar syncs noted in the caveat above and do not by themselves separate host- from
device-bound (PR #625 review). The per-op count scales ~linearly with block count, so dense (1 block)
pays ~nothing and u1sz (32 blocks) pays ~24×.

### Why batching cannot close the gap (the measured ceiling) — and the fuse lever is already a NO-GO

The dominant warm cost is `_fuse_indices_symmetric`. **Batching the fuse is precisely the #566
stacking spike's primary lever** ("a single targeted lever (default: stacked-aware `fuse`)") — which
was already measured **NO-GO**: stacked-fuse on/off was within **1% noise** at D=3 because the
same-shape collapse ceiling on the env tensors that the fuse touches is only ~1.9×.

The #566 stacking census **measured** that ceiling (`n_blocks / n_distinct_shapes`) for U(1)-Sz at
D=3: site 32× (one shape) but **env corners 1.67×, edges 1.9×, median 1.90×**. The fuse + contraction
work is dominated by the *env* tensors (C/T), so any same-shape batching of that work caps at
**~1.9×**. The warm gap to close is **~24–28×** — batching is off by **more than an order of
magnitude**. A perfect fuse-batching would cut warm ~18.5 s → ~10 s (still ~13× dense); and per the
contraction histogram above, for fragmented U(1)-Sz the stacking fixed cost (stack + segment_sum +
broadcast) isn't even amortized, so on the contraction cluster the net is **negative** (the measured
+14%). This is the same wall as the #566 stacking NO-GO and the #615 de-frag NO-GO, now confirmed
from the warm-step / eager-dispatch angle.

The only structural escape — jit the **whole forward sweep** as one unit so warm = 1 dispatch instead
of 8 eager sweeps × tens-of-thousands of micro-ops — is not a batching lever; it re-opens the #566
**compile** wall (the per-block emission that makes the symmetric forward compile minutes-long;
fwd_n_compiles ≈ 1362 today). That trade is #566's scan-fusion question, not #618's.

## Recommendation / next lever

**Close #618 NO-GO.** The warm wall is host-bound eager per-block dispatch dominated by the
**fuse** (`_fuse_indices_symmetric`, 8 eager sweeps, ~19 s). Batching the fuse = the #566 stacked-fuse
lever = already NO-GO (within 1% noise; ceiling ~1.9×). Batching the contraction = the existing
`TENAX_BATCH_BLOCKSPARSE` path = measured **+14% warm** (it *grows* the dominant block-pack/broadcast
clusters). Both are capped at ~1.9× against a 24× gap. Do not fund a fuse/block-pack-batching build:
the ceiling is **measured, not inferred**.

The compile win is the only real benefit of the flag (−21%, from batched **decomp**); it could be
worth wiring `TENAX_BATCH_BLOCKSPARSE` as a **compile-only** default for D≥3 U(1)-Sz if/when the
compile face of #566 is the binding cost — but that belongs to #566, not #618, and must NOT be the
warm-step default (it regresses warm +14%).

**For D≥3 U(1)-Sz today, dense remains pragmatic** (memory `u1sz-perf-study-d3-findings`,
`615-u1sz-uniform-sector-env`). The binding wall is #566 eager per-block dispatch of a graph whose
block count cannot be collapsed (fragmentation), not anything #618's batching can move.

## Artifacts (branch-local; no `src/` touched)

- `examples/profile_warm_dispatch_618.py` — warm-step host-vs-device attribution (dispatch/sync
  split, pipelining probe, cProfile).
- `profile_warm_dispatch_618_gateoff.json` — the warm-step run above.
- `examples/trace_op_histogram_618.py` — separate-process backward op histogram (the clean gate
  off/on comparison; one gate per process to avoid make_jaxpr trace-cache contamination).
- `profile_d3chi12_batch_618.json` — step-1 artifact (existing-gate warm/compile, prior session).

## Risks / caveats recorded

- **Probe in-process gate comparison is contaminated** (make_jaxpr trace reuse) → reports 1.00×.
  Always compare `TENAX_BATCH_BLOCKSPARSE` off/on in **separate processes**. (New caveat; generalizes
  the #610 cold-trace warning to the gate dimension.)
- Verdict scope: NO-GO is for batched *dispatch* as a **warm-step** lever. The batched **decomp**
  compile win (−21%) is real and orthogonal (a #566 compile-face knob).
