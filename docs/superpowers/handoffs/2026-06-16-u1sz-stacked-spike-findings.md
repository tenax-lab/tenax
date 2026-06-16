# U(1)-Sz Stacked Block-Sparse Feasibility Spike — Findings (#566)

**Date:** 2026-06-16
**Branch:** `spike/566-u1sz-stacked-feasibility` (based off `fix/605-u1sz-unfused-ctm` / PR #608)
**Spec:** `docs/superpowers/specs/2026-06-15-u1sz-stacked-feasibility-spike-566-design.md`
**Plan:** `docs/superpowers/plans/2026-06-16-u1sz-stacked-feasibility-spike-566.md`
**Verdict:** **NO-GO** for the stacked-block-sparse lever on the U(1)-Sz CTM-AD path.
**Representation profiled:** post-#605 *unfused* 2-plaquette projector path (this branch).

## Question

Can the existing stacked block-sparse machinery (`StackedView` / `StackedSymmetricTensor`
/ `TENAX_STACK_BLOCKSPARSE`) be made to improve **end-to-end** (compile + warm-step)
**U(1)-Sz iPEPS-AD** at D=3, via a single targeted lever (default: stacked-aware `fuse`)?

## Answer: No. Three independent cheap measurements converge on NO-GO.

The spike was designed measure-first with kill-switches. All three measurement axes agree;
the 30-min D=3 A100 confirmation was deliberately **not** spent because the D=2 + structural
evidence is conclusive (decision recorded at the Task 4 gate).

### Task 0 — prerequisites (both clean)
- **D=3 U(1)-Sz CTM forward runs on this branch** (#605/#608 unfused-projector fix confirmed):
  `test_u1sz_ctm_forward_runs[3-8]` PASSES, no xfail needed.
- **No stack-flag energy drift.** D=2 χ=8: `e_off == e_on == -0.16538453`, |drift| = **0.0**.
  The ~4.6e-4 drift flagged in the prior even-D fermionic work is **not present** here, so the
  GO gate's "energy unchanged" criterion was fully clean (and never the binding constraint).

### S1 — static block-shape fragmentation census (`examples/census_u1sz_block_shapes_566.py`)

Grouping collapse ceiling = `n_blocks / n_distinct_shapes`. Even-D FermionParity ≈ 16
(all blocks one shape → big batching win). U(1)-Sz:

| tensor | D=2 (χ=8) collapse | D=3 (χ=12) collapse |
|---|---|---|
| site | 8.0× (1 shape) | 32.0× (1 shape) |
| C1–C4 (corners) | 1.5× | 1.67× |
| T1–T4 (edges) | 1.75× | 1.9× |
| **median (all)** | **1.75×** | **1.90×** |

The **site** stacks beautifully but the cost is not there. The **CTM environment tensors
(corners/edges) fragment** — collapse 1.5–1.9× vs the ~16× that made even-D work. The plan's
interpretation gate ("≥~3 on hot tensors → headroom; ≈1 → NO-GO-leaning") fails: the hot
tensors sit at 1.5–1.9. Structural reason: U(1)-Sz unequal sectors ⇒ unequal block shapes
(prior spec, "general U(1) ⇒ no collapse from grouping alone").

### S2 — off/on compile+runtime grid (D=2 χ=8 depth=8, CPU; `examples/profile_ctm_ad_wall_566.py` u1sz arm)

| sym | stack | blk | fwd_cmp | vg_cmp | bwd_cmp | warm_ms |
|---|---|---|---|---|---|---|
| u1sz | off | 8 | 14.20s | **31.51s** | 17.31s | 1252.7 |
| u1sz | **on** | 8 | 14.39s | **31.20s** | 16.81s | 1242.3 |
| dense | off | 1 | 10.82s | 54.76s | 43.94s | 657.6 |
| dense | on | 1 | 10.79s | 54.19s | 43.40s | 652.4 |

- **Stacking ≈ 0% benefit for u1sz:** vg_cmp 31.51→31.20s (**−1%**, noise); warm −0.8% (noise).
- **Activation confirmed** (resolves the Task-0 drift caveat): `_backend_opt_in()` = True under
  the flag; `_STACK_PERSIST` after a u1sz CTM run = `{calls:1, persisted_inputs:0,
  fully_persisted:0, gathered_inputs:2}` — the stacked contractor **runs but finds nothing to
  amortize**. Stacking is active and inert, which is exactly why the D=2 drift was 0.0.
- Side note (D=2-specific, not the target): u1sz `vg_cmp` (31.5s) is actually *lower* than dense
  (54.8s) here — the dense implicit-fixed-point `jit(while)` backward is heavier at this small
  size. This does **not** generalize to D=3, where per-block dispatch dominates (prior study:
  symmetry loses 0.32× at D=3). The warm step already shows u1sz ≈ 2× dense — the #566 runtime
  dispatch overhead is real even at D=2.

### S3 — backward op-histogram (trace-only; `examples/probe_backward_jaxpr_566.py --sym u1sz`)

u1sz D=2 χ=8, 8 blocks, **19,319 total backward ops** (`TENAX_STACK/BATCH_BLOCKSPARSE` = 1.00×
on every bucket — no effect):

| cluster | share | primitives |
|---|---|---|
| block buffer-pack | ~37% | reshape 19.8%, slice 7.3%, gather 2.8%, concatenate 2.6%, scatter/​pad/​dyn_slice … |
| charge-mask / index arith | ~35% | broadcast_in_dim 14.0%, select_n 7.6%, lt 3.3%, convert_element_type 4.9%, eq … |
| math | ~30% | add 7.2%, transpose 6.6%, dot_general 5.3%, mul, add_any, div, abs |
| decomp (svd/eigh/qr) | <0.2% | negligible |

- **No `_fuse_indices_symmetric` primitive exists** — `fuse` is not a custom JAX op; it
  decomposes into reshape/slice/gather + charge-mask arithmetic. There is **no single fuse
  chain** for a stacked-aware-fuse rewrite to eliminate.
- The cost is **diffuse**: top op (`reshape`) is only 19.8%; three ~1/3 clusters.
- Block-count scaling D=2→D=3: 19,319 → 63,612 ops (3.3× for 4× blocks) — ~proportional to
  block count (≈1.1 ops/block, same as dense's 1 op/block); the wall is *quantity* of per-block
  emission, not a superlinear hotspot.

## Why NO-GO, and why the fallback was rejected too

- **Default lever (stacked-aware `fuse`):** structurally cannot help — there is no fuse-chain
  bottleneck (S3), the stacked contractor already runs and amortizes nothing (S2), because the
  env tensors don't share shapes (S1).
- **Pad-to-max fallback:** ceiling is the buffer-pack cluster only (~37% of ops), leaving
  charge-mask (35%) + math (30%) untouched; and S1 shows up to 10 distinct shapes per edge
  tensor at D=3 → heavy padding waste. Best case ≈ a fraction of a 37% op-count cut — below any
  meaningful end-to-end gate. (Even the favorable even-D path only achieved a 38% *backward
  compile* reduction, which itself missed the prior spec's 10× bar.)

## Recommendation / next lever

The U(1)-Sz block-sparse cost is **diffuse per-block XLA emission over fragmented charge
sectors**, not a stackable chain. Stacking — which wins only when many blocks share one shape —
is the wrong tool for U(1)-Sz. Levers that *could* matter, all outside this spike's scope:

1. **#566 option 1 (broad sweep-restructure):** carry stacked tensors through the *entire* CTM
   sweep so the cross-call collapse applies. Largest effort, broad blast radius; and for
   fragmented U(1)-Sz the per-call collapse ceiling (S1: 1.5–1.9×) caps the upside well below
   the even-D case. Low expected payoff for the cost.
2. **Reduce sector fragmentation / pad the virtual-leg charge structure** so env blocks share
   shapes — changes the representation, not just the execution; a research question.
3. **Accept the dispatch overhead** and use symmetry only where it already wins (D=2, where
   symmetry is 1.37–1.78× faster than dense per the #607 bench). For D≥3 U(1)-Sz, the dense
   path remains the pragmatic choice until a representation-level change lands.

**This spike conclusively closes the "stacking improves U(1)-Sz runtime" hypothesis.** No `src/`
changes were made; deliverables are a U(1)-Sz census tool, a profiler u1sz arm, a u1sz backward
probe, and prereq tests — all reusable for any future representation-level work.

## Artifacts (all on the branch, no `src/` touched)

- `tests/test_profiler_u1sz_arm.py` — D=3 forward smoke + drift characterization + arm test
- `examples/census_u1sz_block_shapes_566.py` + `census_u1sz.json` — S1
- `examples/profile_ctm_ad_wall_566.py` (u1sz arm) + `profile_u1sz_d2_stack{off,on}.json` — S2
- `examples/probe_backward_jaxpr_566.py` (u1sz arm) — S3

## Not done (deliberately)

- D=3 A100 off/on compile grid — deferred at the Task 4 gate; D=2 + structural evidence is
  conclusive and the predicted benefit is ~0.
- The stacked-aware-fuse prototype (plan Task 5) — not built, per NO-GO.
