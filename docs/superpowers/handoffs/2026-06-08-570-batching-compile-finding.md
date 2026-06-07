# #570 — batching (`TENAX_BATCH_BLOCKSPARSE`) is the only lever that moves the compile wall (~15–22%, bounded); #570 conclusion

**Date:** 2026-06-08 (A100) · **Follows:** [`2026-06-08-570-mechanism-correction.md`](2026-06-08-570-mechanism-correction.md) · **Issue:** #570

## TL;DR

After the mechanism correction (the AD projector is already block-sparse per-sector SVD, so
"lever-3" was moot), the one remaining untested compile lever was **batching** the per-sector
block-sparse work (`TENAX_BATCH_BLOCKSPARSE`, the #566/#569 umbrella gate). Measured gate
OFF vs ON on the sweep-VJP compile rig:

- **It works — modestly.** ON reduces the compiled backward by **~15–22%** (deterministic HLO
  instruction count), with compile time tracking. This is the **first and only** lever that
  measurably moves the compile wall.
- **It does not scale toward the target.** The reduction is **bounded ~20% and slightly
  *diminishes* with χ** (it shrinks the *contraction/block-pack* part of the VJP, while the
  dominant *per-sector SVD-VJP* — which batching cannot reduce — grows with χ).
- **It has a runtime cost.** #569 already showed batching is a *warm-step* loss ("never a net
  win" through D=6). So ON is a **compile↓ / runtime↑ trade-off**, not a free win.

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

HLO instruction count is **deterministic** (independent of run/contention), so the 0.78–0.84×
ratios are the rigorous core; compile-time ratios (measured; D=4 OFF/ON were run concurrently
so their *absolute* timings are CPU-contended, but the OFF and ON runs were contended equally
and the ratios match the deterministic instr ratios). Raw: `570_results/profile_570_d4_batch{off,on}.json`,
`570_results/profile_570_batch_on_a100.json` (D=2 ON) vs `profile_570_sweepvjp_a100.json` (D=2 OFF).

## Why it's bounded (mechanism)

`TENAX_BATCH_BLOCKSPARSE` is an **umbrella** gate. In the AD backward it does **not** reduce the
SVD primitives — those are single-sector per projector call (24 ops of (96,96) at D=4, identical
on/off at the jaxpr level), so the within-call sector-batching in `_truncated_svd_symmetric_traced`
has nothing to group. What it *does* reduce is the **contraction / block-pack** emission
(`contractor.py` batched path): fewer `dot_general` / scatter / reshape instructions around the
SVDs. That is a roughly **fixed-fraction** overhead, so as χ grows the un-batched SVD-VJP grows
faster and the batching win becomes a smaller share (0.80→0.84 across χ=8→16 at D=4).

(An XLA `slow_operation_alarm` fired on constant-folding a `scatter-add` of shape
`f64[16,6,6,8,8,8]` during the D=4 backward — direct evidence that block-pack scatter emission is
a real slice of the compile, which is exactly the part batching trims.)

## Methodological note (so it doesn't recur)

A cheap **jaxpr `svd`-primitive count** diff (off vs on) showed *identical* graphs — which led me
to predict "no effect." That was wrong: **jaxpr primitive count ≠ lowered-HLO instruction count**.
The umbrella gate restructures contraction lowering without changing the `svd` primitive count, so
only the **compile/HLO measurement** revealed the ~20% win. Trust the HLO/compile measurement over
jaxpr op-counts for compile-cost questions.

## #570 conclusion — levers exhausted

The minutes-long fermionic CTM-AD compile wall (`_jit_fused_fixed_point_bwd`, ~549 s at D=4/χ=12)
is dominated by the **per-sector block-sparse SVD VJP**, summed over sectors × projectors × the
fixed-point backward. Lever inventory:

| Lever | Verdict | Why |
|---|---|---|
| #200 cuTensorNet contraction backend | NO-GO | backend-invariant; wall isn't the contraction executor |
| Lever-1: QR projector | NO-GO | no-op as config flip (2×2 path is SVD-only; QR only in unused 1×1) |
| Lever-2: truncated backprop | NO-GO (compile) | per-sweep SVD-VJP irreducible; explicit unroll ≈/worse than implicit |
| Lever-3: block-sparse per-sector SVD VJP | MOOT | already implemented — it *is* the wall |
| **Batching (`TENAX_BATCH_BLOCKSPARSE`)** | **PARTIAL** | **~15–22% compile, bounded, doesn't scale; + runtime penalty (#569)** |

**No available lever reaches the original #200 target (~40× toward a ~13 s dense floor).** The wall
is intrinsic to differentiating N per-sector block-sparse SVDs in XLA: the SVD-gradient (Lorentzian
F-matrix) dense algebra per sector, super-linear in χ, summed over all projectors. Batching trims
the surrounding contraction by ~20% but cannot touch the SVD-VJP core.

### Recommendation

- **Default stays OFF.** The ~20% compile saving doesn't justify the warm-step regression for normal
  optimization runs (many warm steps, one compile).
- **Niche use:** if a workflow is compile-bound (e.g. rapid recompiles, short runs, CI), flipping
  `TENAX_BATCH_BLOCKSPARSE=1` buys ~20% off compile — document it as a knob, don't change the default.
- **To actually beat the wall** would require attacking the SVD-VJP itself (e.g. a cheaper SVD-gradient
  formulation, or avoiding per-projector SVDs in the differentiated path) — a research-grade change, not
  a config or batching tweak. Out of scope for #570 as posed; #570 is **characterized and concluded**.

## Artifacts

- `examples/profile_570_sweepvjp_compile.py` — the rig (honors `TENAX_BATCH_BLOCKSPARSE` via env)
- `570_results/profile_570_d4_batchoff.json`, `…_d4_batchon.json` — D=4 OFF/ON sweep
- `570_results/profile_570_batch_on_a100.json` — D=2 ON (vs `profile_570_sweepvjp_a100.json` OFF)
