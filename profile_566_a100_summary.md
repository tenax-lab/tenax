# #566 CTM-AD Compile-Wall — Phase-0 Profiler Summary (A100)

**Platform:** NVIDIA A100-SXM4-80GB · x64=True · path=implicit (`adjoint_method="fixed_point"`, `_jit_fused_fixed_point_bwd`) · depth=8 · reps=5
**Artifacts:** `profile_566_a100_Dsweep.json`, `profile_566_a100_chi.json`

## (A) Dense-vs-fermionic D-scaling (χ = 4·D)

| sym | D | χ | blk | fwd_cmp | **vg_cmp** | bwd_cmp | ferm/dense vg |
|-----|---|----|-----|---------|-----------|---------|---------------|
| fermionic | 2 | 8 | 16 | 63.8s | **206.4s** | 142.6s | 3.6× |
| fermionic | 3 | 12 | 16 | 527.8s | **2111.3s** (~35 min) | 1583.5s | 52× |
| fermionic | 4 | 16 | 16 | 899.0s | **2379.2s** (~40 min) | 1480.2s | 61× |
| dense | 2 | 8 | 1 | 19.3s | 57.7s | 38.4s | — |
| dense | 3 | 12 | 1 | 27.8s | 40.6s | 12.8s | — |
| dense | 4 | 16 | 1 | 26.3s | 38.9s | 12.6s | — |

## (B) χ-scaling at D=3

| sym | χ | fwd_cmp | **vg_cmp** | bwd_cmp |
|-----|----|---------|-----------|---------|
| fermionic | 8 | 280.7s | 575.7s | 295.1s |
| fermionic | 12 | 552.8s | 2135.5s | 1582.8s |
| fermionic | 16 | 727.3s | 1365.2s | 637.9s |
| fermionic | 24 | 1156.2s | 3568.7s | 2412.5s |
| dense | 8 | 20.2s | 27.1s | 6.9s |
| dense | 12 | 18.3s | 27.7s | 9.4s |
| dense | 16 | 18.0s | 27.6s | 9.6s |
| dense | 24 | 19.9s | 29.6s | 9.7s |

## Conclusions (Phase-0 gate)

1. **Wall is entirely fermionic per-block emission (finding b).** Dense `vg_cmp` is flat at ~27–58s across all D and χ — no symmetry-independent scaling problem. Fermionic/dense ratio is 20×–120× at a constant 16 blocks, so it is *not* a flat "16× dense" multiplier; per-block emission itself grows with D.

2. **Backward dominates (finding d).** `bwd_cmp` is the majority of `vg_cmp` for fermionic (1583/2135s at χ=12; 2413/3569s at χ=24). The slow-op alarm fires on `jit__jit_fused_fixed_point_bwd`. The minutes-long wall is the implicit-diff fixed-point backward.

3. **χ is NOT a compile lever → does not pivot to #570.** Dense compile is flat in χ (27→30s). The large-χ SVD/eigh VJP (finding c) is a *runtime* axis, not compile. #570 does not address the wall.

4. **Scan-fusion is the wrong lever for compile.** The implicit path's compile is already depth-flat by construction (forward step reused across iters); scan-fusion buys eager-dispatch *runtime*, not compile. Redirect target = **reduce per-block op emission in the fermionic fixed-point backward**.

### Caveats
- **Reproducibility:** shared point D=3,χ=12 fermionic agrees across the two runs (2111s vs 2135s, ~1%) — magnitudes are solid.
- **Non-monotonic backward in χ:** fermionic `vg_cmp` is erratic (χ=16 *lower* than χ=12). Forward compile is clean/monotonic (281→553→727→1156s); the variance is isolated to the backward — likely XLA autotuning/caching or a χ-dependent shift in dominant blocks. Single-point fermionic backward timings carry real variance.

## (C) Stacked vs per-block backend — fermionic, depth 8 (default χ-factor 3)

**Seam gate:** `tests/stacked/` = **51/51 passed on GPU** (CUDA, 65s) — stacked VJP/contract/decomp/dtype/view seams are correct, so the comparison is valid.
**Artifacts:** `stacked.json`, `perblock.json`

| D | χ | metric | stacked | perblock | stacked win |
|---|----|--------|---------|----------|-------------|
| 2 | 6 | fwd_cmp | 47.5s | 44.7s | −6% |
| 2 | 6 | **vg_cmp** | 146.4s | 164.2s | **11%** |
| 2 | 6 | bwd_cmp | 98.9s | 119.5s | 17% |
| 4 | 12 | fwd_cmp | 514.9s | 537.4s | 4% |
| 4 | 12 | **vg_cmp** | 1061.2s | 1417.6s | **25%** |
| 4 | 12 | bwd_cmp | 546.2s | 880.3s | **38%** |
| 2 | 6 | warm_step | 4163.6ms | 4432.9ms | 6.1% |
| 4 | 12 | warm_step | 5573.6ms | 5833.7ms | 4.5% |

**Findings:**
0. **Runtime is positive too (not just compile).** Warm-step is 4.5–6.1% faster under stacked — small but on the *right* side, and notably the new contiguous-`_data` `StackedJaxBackend` does NOT reproduce the runtime-NEGATIVE behavior of the abandoned `TENAX_BATCH_BLOCKSPARSE` (#571/2/3) on A100. So the stacked backend is a clean partial win on BOTH axes (compile + runtime), which removes the last objection to a possible GPU default-flip of `TENAX_BLOCKSPARSE_BACKEND=stacked`.
1. **Win is entirely in the backward.** Forward compile is backend-independent (515 vs 537s at D=4); stacked's advantage is isolated to `bwd_cmp` (546 vs 880s, −38%). Consistent with the wall being the per-block fixed-point backward — stacking blocks into one fused op cuts op-emission count there.
2. **Mitigates but does not collapse.** D=4 `vg_cmp` is still ~18 min under stacked. The 25–38% cut is worth banking but is not the order-of-magnitude needed; per-block emission is softened, not solved, by the stacked backend alone.
3. **Win grows with D** (11%→25% on vg, 17%→38% on bwd from D=2 to D=4) — helps most where it hurts most, but won't reach a >10× collapse on its own.

**Bottom line:** stacked backend is GPU-correct and a solid partial lever (bank ~38% backward-compile reduction), but the #566 redirect still needs a deeper attack on per-block op emission to take the wall from minutes to seconds. Necessary-but-not-sufficient.
