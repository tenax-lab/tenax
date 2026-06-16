# U(1)-Sz CTM env de-fragmentation — research kickoff brief

**Date:** 2026-06-16
**Tracking issue:** #610 · **Parent:** #566 · **Predecessor spike:** PR #609
**Status:** kickoff brief for a FRESH session. **Start that session with the `superpowers:brainstorming` skill against this document** — it is a problem statement, not a design.

---

## One-paragraph problem

U(1)-Sz symmetric iPEPS-AD is the lever we want for D≥3 Heisenberg (the dense path is
runtime-bound; symmetry should help). It doesn't, because the **CTM environment tensors
fragment**: their charge blocks have *distinct shapes* and there are *many distinct sectors*, so
every block-sparse op emits per-block XLA primitives that don't amortize. The #566 spike (PR #609)
proved the *execution-level* fix (stacked block-sparse) is structurally dead for this case. The
remaining lever is **representation-level**: change how the iPEPS virtual legs / chi bonds are
charged so the env tensors stop fragmenting. That is an open research question — brainstorm it.

## Fixed facts the spike established (do NOT re-derive these)

From PR #609 / `docs/superpowers/handoffs/2026-06-16-u1sz-stacked-spike-findings.md`:

1. **Stacking is dead for fragmented U(1)-Sz.** Three independent measurements agree:
   - **S1 census** (`examples/census_u1sz_block_shapes_566.py`): block-shape collapse ceiling
     (`n_blocks/n_distinct_shapes`) on the hot env tensors is only **1.5–1.9×** (corners 1.5–1.67×,
     edges 1.75–1.9×) at D=2 and D=3 — vs ~16× for even-D FermionParity where stacking works.
     The **site** tensor collapses 8–32× (one shape), but the cost is **not** in the site.
   - **S2 off/on grid** (`examples/profile_ctm_ad_wall_566.py --sym u1sz`): `TENAX_STACK_BLOCKSPARSE`
     on vs off changes `vg_cmp` by −1% and warm step by −0.8% (both noise). The stacked path
     *activates* (`_backend_opt_in()`=True) but `_STACK_PERSIST` shows `persisted_inputs=0` — it
     runs and amortizes nothing.
   - **S3 backward op-histogram** (`examples/probe_backward_jaxpr_566.py --sym u1sz`): the
     per-block emission cost is **diffuse** — ~37% block buffer-packing (reshape/slice/gather/…),
     **~35% charge-mask / index arithmetic** (broadcast_in_dim/select_n/lt/convert_element_type/eq),
     ~30% math (dot_general/add/transpose/…). No single fusable chain. Op count scales ~linearly
     with block count (≈1.1 ops/block).
2. **Why representation-level, not execution-level (the load-bearing insight).** Execution tricks
   (stacking, pad-to-max *at the contractor*) can only touch the **buffer-pack third**. The
   **charge-mask third (~35%)** is index arithmetic emitted *per distinct sector*; it shrinks only
   if the **number of distinct charge sectors / blocks** shrinks — which is a property of the
   *representation* (how virtual legs are charged + how chi bonds are truncated), not of the
   execution backend. This is the precise reason stacking failed and why the next lever must change
   the representation.
3. **Where the fragmentation comes from.** U(1)-Sz unequal Sz-sector multiplicities on the virtual
   legs ⇒ after the CTM builds corners/edges (fuse of env `chi` and `D²` legs, then projector SVD
   truncation), the chi bonds carry an **asymmetric charge set with unequal per-sector dims** ⇒ C/T
   blocks have distinct shapes (and there are many of them). Even D / Z₂ are self-dual so this is
   masked (and is why FermionParity even-D stacks). See the even-D-vs-ragged section of
   `docs/superpowers/specs/2026-06-04-symmetric-ctm-ad-stacked-blocks-design.md`.
4. **Baseline behavior** (memory `u1sz-perf-study-d3-findings`, bench PR #607): symmetry **wins at
   D=2** (1.37×→1.78× as χ grows) but **loses at D=3** (0.32×, the #566 dispatch overhead over ~32
   small blocks). D=3 U(1)-Sz CTM **runs** on this branch (the #605 fix, PR #608).

## The direction to brainstorm

Make the virtual-leg / chi-bond **charge structure more uniform** so env tensors carry fewer,
equal-shaped blocks. Candidate angles (for the brainstorm to explore/cut — not committed):

- **A. Pad sectors to uniform multiplicity** on virtual legs / chi bonds → all blocks of a tensor
  share one shape *and* fewer effective distinct sectors. Core tension: zero-padding waste +
  whether the charge-mask cost actually drops (must re-measure), vs the batching/sector-reduction
  win. (Distinct from the spike's contractor-level pad-to-max, which capped at ~37% because it left
  the sector *count* unchanged.)
- **B. Charge-set design**: pick the virtual-leg Sz sector set + multiplicities to minimize distinct
  block shapes while keeping the variational manifold expressive enough for the physics.
- **C. Symmetric chi-bond truncation**: constrain the projector SVD truncation to retain *equal
  per-sector dims* (uniform per-sector χ). Quantify the accuracy cost vs unconstrained
  charge-conserving truncation.

## Open questions the brainstorm must answer
- At D=3, edge (T) tensors have ~10 distinct shapes — what is the padding overhead to collapse to
  1, and does the batched + sector-reduced end-to-end cost actually beat the fragmented baseline?
- How much accuracy is lost by uniform/padded sectors vs unconstrained truncation? (Energy gate.)
- **Re-profile under a uniform representation**: does the charge-mask ~35% cluster actually shrink
  when sector count drops, or is it dominated by something a uniform representation won't fix? This
  is the make-or-break measurement — design it early (reuse the S3 probe).
- Does any of this interact with the #605 unfused-projector path (this branch changed the env
  construction)? Profile against the post-#605 representation.

## Suggested success criteria (draft — refine in brainstorm)
Meaningful reduction in U(1)-Sz `vg_cmp` **and/or** warm-step at D=3 χ=12 vs the current fragmented
representation, with energy within a bounded tolerance of the unconstrained-truncation result.
Mirror the spike's measure-first / kill-switch discipline: cheapest decisive measurement first
(e.g. a static "distinct-sector count under candidate representation X" census before any compile).

## Risks
- **Accuracy:** uniform/padded sectors constrain the variational state — the energy cost may exceed
  the runtime benefit.
- **Charge-mask may be partly irreducible:** fewer sectors might not cut the index arithmetic as
  much as hoped (re-profile early — don't assume).
- **Padding-waste trap:** the same mechanism that made within-call batching net-negative in #566.
- **Blast radius:** touches iPEPS init (virtual-leg charges), the projector SVD truncation, and CTM
  env construction — the most-used symmetric paths.

## Reusable assets from the spike (already on this branch / PR #609)
- `examples/census_u1sz_block_shapes_566.py` — block-shape/sector census (extend for distinct-sector
  counts under candidate representations).
- `examples/profile_ctm_ad_wall_566.py` (`u1sz` arm) — compile/runtime grid.
- `examples/probe_backward_jaxpr_566.py` (`u1sz` arm) — backward op-histogram (the make-or-break
  re-profile tool).
- `tests/test_profiler_u1sz_arm.py` — U(1)-Sz CTM smoke + drift harness.

## Pointers
- #610 (this work), #566 (parent), PR #609 + its handoff (the NO-GO that motivates this).
- `docs/superpowers/specs/2026-06-04-symmetric-ctm-ad-stacked-blocks-design.md` (fragmentation
  rationale), memories `566-u1sz-stacking-nogo`, `u1sz-perf-study-d3-findings`,
  `570-u1sz-blocked-core-bug`, `570-dense-largeD-study`.
