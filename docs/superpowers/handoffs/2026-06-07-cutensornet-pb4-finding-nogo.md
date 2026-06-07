# P-B4 measurement — cuTensorNet contraction backend is a NO-GO for the #200 compile wall

**Date:** 2026-06-07
**Hardware:** A100-SXM4-80GB, CUDA13, jax 0.10.1, x64, `JAX_PLATFORMS=cuda,cpu`.
**Raw data:** `pb4_results/pb4_{d2,d4}_{perblock,stacked,cutensornet}.json` (next to this doc).
**Verdict:** the cuTensorNet block-sparse **contraction** backend (#200, Phase B) is
**correct** but does **not** move the CTM-AD compile wall. The wall is the
implicit-differentiation backward (`_jit_fused_fixed_point_bwd`), which is invariant
to the contraction backend. P-B5 (the FFI handler) targets the wrong cost — **do not
build it.** Pivot to #570 (block-sparse SVD/eigh VJP).

---

## What was measured

`examples/profile_ctm_ad_wall_566.py`, fermionic, implicit/fixed-point, depth 8,
reps 3, χ = 3·D (default chi-factor, **not** overridden — apples-to-apples with the
handoff §C baselines):

```bash
for be in perblock stacked cutensornet; do
  JAX_PLATFORMS=cuda,cpu TENAX_BLOCKSPARSE_BACKEND=$be \
  uv run python examples/profile_ctm_ad_wall_566.py \
    --D <2|4> --sym fermionic --depth 8 --reps 3 --json pb4_d<D>_$be.json
done
```

(D=4 ran `stacked` + `cutensornet` same-session; `perblock` D=4 reuses the handoff
baseline. D=2 ran all three.)

## Results

### D=2, χ=6 (16 blocks)

| metric | perblock | stacked | cutensornet |
|---|---:|---:|---:|
| vg_cmp (s) | 170.7 | 149.7 | 149.2 |
| bwd_cmp (s) | 123.3 | 99.8 | 104.9 |
| warm_step (s) | 27.5 | 26.7 | **37.8** |
| n_compiles | 230 | 325 | 328 |
| top_compile `_jit_fused_fixed_point_bwd` (s) | 113.9 | 91.2 | 94.0 |
| energy / grad_finite | −0.203 / ✓ | −0.203 / ✓ | −0.203 / ✓ |

### D=4, χ=12 (16 blocks) — the #200 bar

| metric | stacked | cutensornet | vs stacked |
|---|---:|---:|---:|
| fwd_cmp (s) | 516.1 | 485.2 | 0.94× |
| **vg_cmp (s)** | 1066.1 | 1046.7 | **0.98×** |
| **bwd_cmp (s)** | 550.1 | 561.5 | **1.02×** |
| **warm_step (s)** | 31.3 | 40.4 | **1.29× (worse)** |
| n_compiles | 324 | 327 | — |
| **top_compile `_jit_fused_fixed_point_bwd` (s)** | **548.9** | **548.7** | **1.00×** |
| energy / grad_finite | −0.028 / ✓ | −0.028 / ✓ | identical |

Same-session `stacked` (1066 s / 550 s) reproduces the handoff cross-run baseline
(1061 s / 546 s), so the comparison is sound.

## The finding (three facts)

1. **No compile collapse.** The #200 target was `bwd_cmp` 546 s → toward the ~13 s
   dense floor (~42×). cuTensorNet delivered **1.02×** at D=4 (and ≈1.00× at D=2).
   Nothing.

2. **The wall is backend-invariant and it is NOT the contraction.** The dominant
   compile is `_jit_fused_fixed_point_bwd` — the implicit-diff backward — at **548.9 s
   (stacked) vs 548.7 s (cutensornet)**, identical to 0.2 s, ≈100 % of `bwd_cmp`.
   Swapping the block-contraction kernel (perblock → stacked → cuTensorNet) does not
   touch it. The forward contraction is a small slice of the fermionic compile; the
   real cost is the block-sparse VJP graph (SVD/eigh/projectors) inside the
   fixed-point backward.

3. **Runtime regression.** The callback bridge's device↔host round-trip per
   contraction makes warm-step **worse** (1.29–1.42×). And `stacked` warm-step is
   already 31 s/step (far from dense), so the contraction transport is not the runtime
   lever either — #195's "tiny-kernel launches" hypothesis does not point here.

## Why cuTensorNet can't win this (mechanistic)

`stacked` already collapsed the per-block forward contraction into one batched
op (that banked the per-block→stacked win: 880→546 s `bwd_cmp` at D=4, handoff §A/§C).
cuTensorNet over `stacked` only swaps that one batched op's executor — but the
fermionic compile cost lives in the *rest* of the differentiated CTM step (block-sparse
SVD/eigh/projector VJPs in `_jit_fused_fixed_point_bwd`), which neither stacked nor
cuTensorNet changes. The dense floor is ~39 s because dense has no block machinery at
all — not because dense contracts faster.

## Decision

- **#200 contraction backend: STOP at P-B4.** P-B1–P-B3 (correct cuTensorNet forward +
  hand-written VJP, validated real + c128 at D=2/D=4, committed `7158a55`) stay as a
  proven, default-OFF backend — but they do not move the wall.
- **P-B5 FFI handler: NO-GO.** It removes the callback round-trip (a runtime fix for a
  cost that isn't dominant) and leaves the `_jit_fused_fixed_point_bwd` compile wall
  untouched. The P-B5 spec
  (`docs/superpowers/specs/2026-06-07-cutensornet-ffi-handler-pb5-design.md`) carries
  the gate that fired NO-GO; keep it as the record of the road not taken.
- **Pivot: #570** — the large-χ block-sparse **SVD/eigh VJP** compile inside the
  implicit-diff backward. That is what `_jit_fused_fixed_point_bwd` is compiling, and
  it is the actual lever for the fermionic compile wall (and likely the 31 s warm-step
  too). Confirm by profiling *which* sub-ops dominate `_jit_fused_fixed_point_bwd`.

## What would change this conclusion

Only if profiling `_jit_fused_fixed_point_bwd` showed the **contraction** VJP (not the
decomposition VJPs) dominating its 549 s — which contradicts the backend-invariance
measured here. Absent that, the contraction is settled as not the bottleneck.
