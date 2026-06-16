# U(1)-Sz block-sparse runtime — feasibility spike (#566 for the U(1)-Sz CTM-AD path)

**Date:** 2026-06-15
**Issues:** #566 (block-sparse AD compile + runtime cost), depends on #605/#608 (U(1)-Sz CTM at D≥3)
**Status:** design, pending implementation plan
**Appetite:** feasibility spike — measure first, prototype ONE chain, GO/NO-GO before any broad rollout.

## Goal

Decide, with measurements, whether the existing stacked block-sparse machinery can be made to
improve **end-to-end** (compile + warm-step runtime) **U(1)-Sz iPEPS-AD** — concretely the
D=3 Heisenberg ground-state-AD path — and if so, by how much and via which single lever.
This is a GO/NO-GO spike, not a committed rollout.

## Why this is not already answered

The stacked rep (`StackedView` P1a, persisting `StackedSymmetricTensor` P1d slice 1, stacked
contractor + `blocksparse_vjp`, `TENAX_STACK_BLOCKSPARSE`) was **designed and gated for even-D
`FermionParity`** (see `2026-06-04-symmetric-ctm-ad-stacked-blocks-design.md`). Two facts from
that work make the U(1)-Sz case genuinely open:

1. **Fragmentation.** Even-D FermionParity ⇒ all blocks of a tensor share one shape (up to 16×
   grouping collapse). **General U(1) ⇒ unequal sectors ⇒ blocks fragment into distinct shapes**
   (the prior spec measured D3 site A: 16 blocks → 16 shapes → *no* collapse from grouping
   alone). U(1)-Sz therefore starts from a strictly worse position than the proven even-D case.
2. **The even-D result missed its own gate, and the chain breaks at `fuse`.** Measured on A100:
   the stacked backend cut fermionic **backward** compile only **38%** at D=4 (546s vs 880s
   bwd_cmp; 1061s vs 1418s vg_cmp), well short of the spec's ≥10× hard gate — it *mitigates,
   does not collapse*. The persisting rep collapses `dot_general` 9.3× but **backward total
   op-count moves 0.995×** because `.fuse()` reads `_data` via `_get_block` and interrupts most
   persist chains (hit-rate 27%, 8/97 fully-persisted calls). The landed commit's own verdict:
   *"fuse/bar must be made stacked-aware (Task 5b)."* That lever was never built.

This branch (`fix/605-u1sz-unfused-ctm`, PR #608, OPEN) makes D≥3 U(1)-Sz CTM run by applying
2-plaquette projectors **unfused** — which means *more, smaller per-block ops*, i.e. it may
*worsen* #566 emission. So profiling the post-#605 path is exactly what we want to measure.

## The measurement spine (cheap, decides everything; runs before any prototype)

Three steps, increasing cost. Each can independently kill the spike.

- **S1 — Static shape-fragmentation census (no compile).** For the U(1)-Sz CTM tensors
  (corners C1–C4, edges T1–T4, site, the bond matrices entering projector SVD) at D=2 and D=3,
  count `n_blocks` vs `n_distinct_shapes` per tensor → the **grouping collapse ceiling**. This
  is pure static metadata math (block keys/shapes), no XLA. *If U(1)-Sz collapse ≈ 1× across
  the hot tensors, stacking-as-built cannot help and the spike pivots to the ragged/pad-to-max
  question (P1e) or NO-GO — before spending a single compile.*
- **S2 — off/on grid + persist hit-rate.** Add a `u1sz` arm to `examples/profile_ctm_ad_wall_566.py`
  (a U(1)-Sz Heisenberg site + `heisenberg_gate`, matched D/χ to the `dense` arm). Measure the
  2×2 grid {dense, u1sz} × {`TENAX_STACK_BLOCKSPARSE` off, on} for `fwd_cmp / vg_cmp / bwd_cmp`
  + warm `step_s`, at D=2 (must run) and D=3 (if #608 holds; else D=3 χ=8 fallback). Reuse the
  `_STACK_PERSIST` instrumentation to report the U(1)-Sz persist hit-rate (the even-D baseline
  was 27%).
- **S3 — Backward localization.** Run `examples/probe_backward_jaxpr_566.py` (trace-only op
  histogram, no XLA) on the U(1)-Sz backward to identify the single dominant per-block-emitting
  op/chain and confirm where `_data` materialization (`fuse`/`bar`/`_get_block`) breaks the
  chain. This names the exact chain the prototype targets.

## Default prototype (Approach A, concretized by S1–S3)

Given the prior evidence, "penetrate the hottest chain" almost certainly resolves to
**make `fuse` (and `bar` if S3 flags it) stacked-aware** so the persist chain survives through
the U(1)-Sz CTM sweep (Task 5b, applied to the U(1)-Sz path), behind the existing
`TENAX_STACK_BLOCKSPARSE` flag. Scope strictly to the one chain S3 identifies — not a full
sweep rollout. If S3 instead shows a single dominant *contraction* chain that the contractor
already stacks but a pytree round-trip de-stacks, the prototype is closing that round-trip gap
instead. The prototype is chosen *after* S1–S3, from this menu; default = stacked-aware fuse.

Fallback if S1 shows U(1)-Sz fragments badly: a **pad-to-max-shape** micro-experiment on the
hottest tensor group (turn distinct shapes into one padded group) to test whether ragged
collapse is recoverable at acceptable padding waste — a scoped probe of P1e, not its delivery.

## GO/NO-GO gate

**GO** if the single-chain prototype yields **≥1.5× reduction in U(1)-Sz `vg_cmp` at the
largest D that runs (target D=3), with energy unchanged** (within the characterized drift,
see below) and warm `step_s` not regressed. Rationale for 1.5×: the even-D contractions-only
result was ~1.34× vg_cmp (1418→1061s); a U(1)-Sz result beating that is the signal that
fuse-aware persistence actually works on the harder, fragmented case and justifies funding the
broad rollout. **NO-GO** → document the wall (fragmentation and/or fuse coupling), do not
restructure the sweep.

## Correctness contract

- Reuse the existing tiered comparator (bit-identical round-trip; bounded-fp energy/grad;
  gauge-invariant-only for SVD factors) from the prior spec.
- **Characterize the flagged ~4.6e-4 `TENAX_STACK_BLOCKSPARSE` energy drift first** (reproduce
  on HEAD, bound it, decide if it taints the U(1)-Sz off/on comparison). The spike's energy
  equality is asserted against the stack-OFF U(1)-Sz golden, not the dense one (charge-conserving
  ≠ unconstrained truncation — expected to differ, per #602 resolution).
- All `tests/stacked/` and U(1)-Sz CTM tests stay green at each step.

## Risks / dependencies

- **#605/#608 must give a working U(1)-Sz D=3 forward+grad path.** Verify a single D=3 grad
  step runs on this branch before S2's D=3 arm; else cap the spike at D=2 + D=3 χ=8 and say so.
- **Fragmentation may be fatal** (S1 is the early kill switch).
- **Drift** could mask/contaminate the energy gate (characterized first, before S2's off/on
  comparison — see Correctness contract).
- **Unfused #605 projectors** change the emission profile vs the fused path — results are
  specific to the post-#605 representation and must be labelled as such.

## Deliverables

- A `u1sz` arm in the #566 profiler + a one-page measurement summary (S1 census table, S2 2×2
  grid, S3 op histogram) committed under `examples/`/`docs/superpowers/handoffs/`.
- Either a GO with the single-chain prototype landed behind the flag + measured win, or a
  NO-GO handoff documenting the specific wall.

## Non-goals

- No sweep-wide stacked rollout, no `_data`/single-leaf storage change (#87 substrate), no
  cuTensorNet kernel (P2). Those are gated on this spike's GO.
- No attempt to fix #605 itself here — we consume its result.
