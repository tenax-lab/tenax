# U(1)-Sz CTM env de-fragmentation — measure-first GO/NO-GO spike (#610)

**Date:** 2026-06-16
**Tracking issue:** #610 · **Parent:** #566 · **Predecessor spike:** PR #609 (NO-GO)
**Branch:** `spike/610-u1sz-env-defrag` (off `origin/main` @ 3b51689, which carries the #609 assets)
**Kickoff brief:** `docs/superpowers/handoffs/2026-06-16-u1sz-env-defrag-research-kickoff.md`
**Status:** design, pending implementation plan.
**Scope ceiling:** a measure-first spike that ends in a documented **GO/NO-GO finding**. No
committed `src/` change this round — the prototype representation stays throwaway on the branch,
mirroring #609. If GO, a *separate* follow-up issue implements the representation change.

---

## 1. Objective & question

Determine whether a **representation-level** change that reduces the *distinct charge-sector
count* on the U(1)-Sz CTM environment tensors materially shrinks the backward **charge-mask op
cluster** (#609's ~35% of backward ops), and whether that buys end-to-end D=3 χ=12 iPEPS-AD
runtime at least to **parity with dense**, within a **1% energy tolerance**.

Output: a documented GO/NO-GO finding plus reusable measurement tooling — *not* a shipped
representation change.

## 2. Fixed facts (established by #609 — do NOT re-derive)

From PR #609 / `docs/superpowers/handoffs/2026-06-16-u1sz-stacked-spike-findings.md`:

1. **Stacking is dead for fragmented U(1)-Sz.** Block-shape collapse ceiling on the hot env
   tensors is only 1.5–1.9× (vs ~16× for even-D FermionParity); the stacked contractor runs and
   amortizes nothing (`persisted_inputs=0`); the backward op cost is diffuse (~37% buffer-pack,
   **~35% charge-mask / index arithmetic**, ~30% math) with no single fusable chain.
2. **The charge-mask third shrinks only if the distinct-sector count shrinks** — it is index
   arithmetic emitted *per distinct sector*. This is a property of the **representation** (how
   virtual legs are charged + how chi bonds are truncated), not of the execution backend. That is
   the precise reason stacking failed and the load-bearing premise of this spike.
3. **Where the fragmentation comes from.** U(1)-Sz unequal Sz-sector multiplicities on the
   virtual legs, plus the `D²`-fuse producing charges `{−2..+2}` on the chi bonds after projector
   SVD truncation ⇒ env C/T blocks carry many distinct sectors with unequal shapes. Even D / Z₂
   are self-dual so this is masked.
4. **Baseline behavior:** symmetry **wins at D=2** (1.37×→1.78× as χ grows) but **loses at D=3**
   (≈0.32× vs dense — the #566 dispatch overhead over fragmented sectors). D=3 U(1)-Sz CTM-AD
   **runs** on this branch (post-#605/#608 unfused-projector fix).

## 3. Measurement staircase (the spike) — three kill-gates, cheapest decisive first

Each rung can kill the spike before the next is funded. Mirrors #609's measure-first / kill-switch
discipline and realizes the two-tier gate (cheap mechanism gate, then end-to-end worth-it gate).

| Stage | What | Cost | Gate |
|---|---|---|---|
| **0 — baseline lock** | Confirm D=3 χ=12 U(1)-Sz CTM-AD runs on this branch. Record the *fragmented* baseline: env distinct-sector **and** distinct-shape census, plus the unconstrained-truncation energy `E_frag` at D=3 χ=12. | CPU, cheap | — (prerequisite) |
| **1 — static sector census** | Extend the census to count distinct *charges/sectors* on the env C/T tensors (not just shapes) under each candidate representation. | No compile | **Gate A:** at least one candidate cuts the env distinct-sector count **≥2×** (toward the even-D ~1 regime) at tolerable expressivity cost. Else **NO-GO** before any compile. |
| **2 — make-or-break re-profile** | Build a throwaway prototype of the census-winning candidate; re-run the S3 backward op-histogram under it. | Trace-only, CPU | **Gate B (Tier 1):** the charge-mask cluster op count drops **materially — target ≥25% relative**, well beyond #609's ~1% noise — and the total backward op count drops commensurate with the sector reduction. Else **NO-GO**; do not spend A100. |
| **3 — end-to-end + energy** | Off/on compile+runtime grid under the prototype vs the fragmented baseline; compute `E_uniform`. | A100, **only if Gate B passes** | **Gate C (Tier 2):** `vg_cmp` **and/or** warm-step reaches **≥ parity with dense** (≈3× faster than the fragmented sym baseline) **AND** `\|E_uniform − E_frag\| / \|E_frag\| ≤ 1%`. Else documented **partial / NO-GO**. |

**Why this ordering.** Gate A is free (static metadata, no compile) and kills the whole idea if no
candidate even reduces the sector count on paper. Gate B is the make-or-break the brief names — it
directly tests the load-bearing premise (charge-mask ∝ sector count) for the cost of one trace,
before any A100 time. Only a representation that passes both cheap gates earns the A100
end-to-end + energy measurement.

## 4. Candidate representations (the census evaluates; C leads at D=3)

- **C — sector-dropping symmetric truncation (lead).** Constrain the projector SVD truncation to
  keep only `|Sz| ≤ 1` on the chi bond and/or enforce **uniform per-sector χ**. The chi bonds
  literally set the env block shapes, so this is the most direct control of env fragmentation, and
  the accuracy cost (dropping the `|Sz|=2` sectors) directly exercises the energy gate.
  **Injection site:** `_fishman_truncate_S` and the `chi`-keep step in `build_2x2_projectors`,
  both in `src/tenax/algorithms/_ctm_tensor_projector_2x2.py`.
- **B — alternative virtual-leg charge set / multiplicities.** Attack sectors at the source.
  **Cramped at D=3:** virtual legs need at least `{0,+1,−1}` (pure ±1 hits the parity obstruction
  documented in `heisenberg_u1sz_init_pair`), so B likely loses the census at D=3; kept for
  completeness and to record D≥4 headroom. **Injection site:** `heisenberg_u1sz_init_pair`,
  `src/tenax/algorithms/ipeps.py:96`.
- **A — pad-to-uniform-multiplicity (deprioritized, not prototyped).** Keeps the same charge set
  and only equalizes per-sector dims → fewer distinct *shapes* (helps the buffer-pack third +
  stacking, both already shown weak by #609) but unchanged sector *count* → will not move the
  charge-mask third. Recorded as the contrast that motivates the sector-count framing.

## 5. Prototype mechanism (throwaway, branch-local)

Inject the census-winning lever behind a **branch-local flag / monkeypatch** at the named site —
**no committed `src/` change.** The prototype exists only to produce a uniform-representation env
to profile.

**Faithfulness guard (mandatory before profiling):** verify the prototype CTM **converges** and
`E_uniform` is **finite and physically sane** at D=3 χ=12. A truncation hack that yields an invalid
charge-conserving env makes the re-profile and energy numbers meaningless; the guard runs before
Gate B and before Gate C.

## 6. Reused assets (all on the branch from #609)

- `examples/census_u1sz_block_shapes_566.py` — extend to a distinct *sector* count and to evaluate
  candidate representations (Stage 1).
- `examples/probe_backward_jaxpr_566.py` (u1sz arm) — the S3 backward op-histogram = Gate B (the
  make-or-break re-profile tool).
- `examples/profile_ctm_ad_wall_566.py` (u1sz arm) — off/on compile+runtime grid = Gate C.
- `tests/test_profiler_u1sz_arm.py` — D=3 forward smoke + drift harness (faithfulness guard).

## 7. Deliverables

- Extended census tool (distinct-sector count + candidate evaluation).
- Throwaway prototype of the winning candidate (branch only — never merged to `src/`).
- Re-profile op-histogram numbers (Gate B) and, if reached, the A100 end-to-end + energy numbers
  (Gate C).
- A findings handoff doc (`docs/superpowers/handoffs/`) recording the GO/NO-GO finding with the
  decision recorded at the binding gate, mirroring #609's writeup.
- A memory update.
- **If GO:** open a follow-up implementation issue for the representation change. Do **not**
  implement it in this spike.

## 8. Success criteria (the GO definition, restated)

**GO** = Gate A passes (a candidate cuts env distinct-sector count ≥2×) **and** Gate B passes
(charge-mask cluster ops drop ≥25%, beyond noise) **and** Gate C passes (≥ parity-with-dense on
`vg_cmp` and/or warm-step at D=3 χ=12 **and** `|E_uniform − E_frag| / |E_frag| ≤ 1%`).
**NO-GO** = any gate fails; the spike stops at that gate and documents why (cheapest-possible
kill). A pass on A+B but fail on C is a documented **partial** (mechanism real, not yet worth it).

## 9. Risks / kill-switches

- **Charge-mask partly irreducible** — Gate B catches it cheaply (trace-only) before A100 spend.
- **B cramped at D=3** — the census surfaces it; C is the lead lever by design.
- **Sector-dropping accuracy cost** — may blow the 1% energy gate even if runtime improves; that is
  exactly Gate C's job to expose (documented partial, not a silent pass).
- **Prototype unfaithfulness** — the faithfulness guard (§5) runs before any profiling.
- **Blast radius (deferred):** a *real* fix would touch iPEPS init + projector truncation + env
  construction — but this spike makes **no committed `src/` change**, so the blast radius is
  confined to the throwaway branch prototype. The blast-radius cost is a concern for the follow-up
  implementation issue, not for this measurement spike.

## 10. Non-goals

- No committed `src/` representation change this round (GO/NO-GO finding only).
- No stacking revisit (conclusively dead per #609).
- Not chasing D=2 (symmetry already wins there) or D≥4 (the census may *note* headroom, but the
  binding gate is D=3 χ=12, the case where symmetry currently loses).
- No accuracy-spine / golden infrastructure build-out (that belongs to the follow-up
  implementation if GO; the spike's accuracy check is the §5 faithfulness guard + the Gate C
  energy comparison).
