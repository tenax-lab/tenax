# #570 — batching (`TENAX_BATCH_BLOCKSPARSE`) IS a ~15–22% compile lever (HLO-measured), correcting the op-count dismissal; #570 conclusion

**Date:** 2026-06-08 (A100) · **Builds on / corrects:** PR #589 (`2026-06-08-570-relocalized-not-decomposition.md`) · **Issue:** #570

## TL;DR

PR #589 nailed the **mechanism**: the compile wall is **#566 per-sector *structural* emission**
(block pack/unpack + gauge-fixing) lexically inside the SVD/projector wrapper — the decomposition
math (SVD-gradient F-matrix) is **~0%**. (This corrects my earlier #570 docs, which wrongly
attributed the wall to the "SVD-gradient Lorentzian F-matrix algebra" — see the correction notes
on `2026-06-07-570-svd-vjp-compile-finding.md` etc.)

PR #589 also tested **batching** (`TENAX_BATCH_BLOCKSPARSE`) and dismissed it — but judged it by
**jaxpr op-count** (svd_vjp −0.7%, TOTAL +2.8%) and concluded "not a compile lever." That measure
is the wrong one: **jaxpr op-count ≠ lowered-HLO size**. Measured at the **HLO/compile level**, the
gate **does** help:

- **~15–22% smaller compiled backward** (deterministic HLO instruction count), compile time tracks.
- It is the **only lever that measurably moves the compile wall** — because it *is* the partial,
  already-available form of #589's recommended fix (stack the per-sector structural ops across
  sectors → fewer lowered kernels: #589 noted svd kernels 48→24).
- **Bounded** (~20%, doesn't scale with χ) and carries a **runtime penalty** (#569: warm-step loss).

Reconciliation: #589's "+2.8% jaxpr ops" and this "−20% HLO instrs" are **both correct and
consistent** — the stacked form adds a few trace-level ops but XLA lowers it to fewer kernels.

## Results (A100, fermionic, x64, sweep-VJP unit, `--full` = env+params)

### D=4 (production-scale; SVD operands (96,96))

| χ | OFF env HLO instr | ON env HLO instr | instr ratio | OFF tot compile | ON tot compile | compile ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 8  | 326,547 | 261,328 | **0.80×** | 320.3 s | 249.8 s | 0.78× |
| 12 | 400,940 | 326,199 | **0.81×** | 544.6 s | 474.9 s | 0.87× |
| 16 | 471,073 | 395,980 | **0.84×** | 852.6 s | 753.1 s | 0.88× |

### D=2 (corner sectors (6,6); SVD operands (24,24))

| χ | OFF env HLO instr | ON env HLO instr | instr ratio | OFF tot compile | ON tot compile |
|---:|---:|---:|---:|---:|---:|
| 12 | 111,934 | 102,223 | 0.91× | 104.1 s | 100.1 s |
| 24 | 307,580 | 240,354 | 0.78× | 296.8 s | 232.1 s |

HLO instruction count is **deterministic** (independent of run/CPU contention), so the 0.78–0.84×
ratios are the rigorous core; compile-time ratios track them. (D=4 OFF/ON ran concurrently, so their
*absolute* timings are equally CPU-contended; the ratios still hold against the deterministic instr
ratios.) Raw: `570_results/profile_570_d4_batch{off,on}.json`, `…/profile_570_batch_on_a100.json`
(D=2 ON) vs `…/profile_570_sweepvjp_a100.json` (D=2 OFF).

## Why ~20% and bounded (mechanism, per #589 + this)

The wall bucket is **~60% per-sector block pack/unpack + ~25% per-sector gauge-fix, ~0%
decomposition** (#589's drill-down: `broadcast_in_dim`, `scatter_mul`, `squeeze`/`reshape`/`slice`,
`_fix_svd_signs` sign logic). `TENAX_BATCH_BLOCKSPARSE` is an umbrella gate that **stacks the
per-sector block/contraction ops across sectors**, so XLA lowers them to fewer kernels → ~20% fewer
HLO instructions. But the gate does **not** vectorize the per-sector **gauge-fix / sign logic**
(`_gauge_fix_symmetric_svd`, `_fix_svd_signs` — #589's ~25%), and the SVD primitives stay per-call.
So the win is capped at the block-pack share and slightly **diminishes with χ** (0.80→0.84 across
χ=8→16 at D=4) as the un-stacked gauge-fix/sign part grows with surviving sectors.

(Direct corroboration: an XLA `slow_operation_alarm` fired constant-folding a `scatter-add` of
shape `f64[16,6,6,8,8,8]` during the D=4 backward — block-pack scatter emission is a real,
trimmable slice of compile.)

## Method lesson (recorded twice now, both directions)

`jaxpr` primitive op-count is **not** a proxy for lowered-HLO / compile size:
- I earlier predicted batching = "no-op" from an identical jaxpr `svd`-count → wrong (HLO −20%).
- #589 dismissed batching from jaxpr op-count +2.8% → also missed the HLO −20%.

For any **compile-cost** question, measure the **HLO/compiled** artifact, not the jaxpr.

## #570 conclusion — levers exhausted

| Lever | Verdict | Why |
|---|---|---|
| #200 cuTensorNet contraction backend | NO-GO | backend-invariant; wall isn't the contraction executor |
| Lever-1: QR projector | NO-GO | no-op as config flip (2×2 path is SVD-only; QR only in unused 1×1) |
| Lever-2: truncated backprop | NO-GO (compile) | per-sweep structural emission irreducible; explicit unroll ≈/worse than implicit. (Runtime/robustness lever, not compile — #589 §5 agrees.) |
| Lever-3: "implement block-sparse SVD VJP" | MOOT | already implemented; and the cost isn't the decomposition anyway (#589) |
| **Batching (`TENAX_BATCH_BLOCKSPARSE`)** | **PARTIAL** | **~15–22% compile (HLO), bounded; + runtime penalty (#569)** |

**The wall is #566 per-sector structural emission** (block pack/unpack + gauge-fix), super-linear in
χ via surviving-sector count, summed over projectors × the fixed-point backward. No available lever
reaches the original #200 target (~40× toward a ~13 s dense floor).

### The one lever that could go further

Both threads converge on it: **extend the stacked-block representation (PR #586, contraction-only)
to the SVD/projector wrapper** — stack the per-sector pack/unpack **and** gauge-fix/sign-fix across
sectors instead of looping. `TENAX_BATCH_BLOCKSPARSE` already does the block/contraction part (this
doc's ~20%); the remaining prize is vectorizing `_gauge_fix_symmetric_svd` + `_fix_svd_signs` (#589's
~25% bucket) across sectors. High blast radius (#589's lever 2/3); the larger win but the harder
build, and a scope decision since compile is one-time.

### Recommendation

- **Default `TENAX_BATCH_BLOCKSPARSE` stays OFF** — the ~20% compile saving doesn't justify the
  warm-step regression for normal optimization runs (many steps, one compile).
- **Flip it ON for compile-bound niches** (rapid recompiles, short runs, CI) — document as a knob.
- **To beat the wall**, pursue the stacked-block SVD/projector wrapper (above), not a cheaper
  decomposition (the decomposition is ~0% — #589). #570 is **characterized and concluded**.

## Artifacts

- `examples/profile_570_sweepvjp_compile.py` — the rig (honors `TENAX_BATCH_BLOCKSPARSE`)
- `570_results/profile_570_d4_batchoff.json`, `…_d4_batchon.json` — D=4 OFF/ON sweep
- `570_results/profile_570_batch_on_a100.json` — D=2 ON (vs `profile_570_sweepvjp_a100.json` OFF)
- See also (PR #589): `2026-06-08-570-relocalized-not-decomposition.md`,
  `examples/probe_bwd_subop_attribution_570.py`, `probe_decomp_vjp_cost_570.py`
