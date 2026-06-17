# U(1)-Sz CTM uniform-sector env — flag-gated Gate-B measurement (#615)

**Date:** 2026-06-17
**Tracking issue:** #615 · **Parent:** #566 · **Predecessor spike:** #610 (PR #614) — NO-GO-by-obstruction
**Branch:** `feat/615-u1sz-uniform-sector-env` (off `origin/main`; merge `origin/main` after #614 lands to pick up the #610 artifacts this spec references)
**Spec lineage:** `docs/superpowers/specs/2026-06-16-u1sz-env-defrag-design.md` (the #610 measure-first spike), `docs/superpowers/handoffs/2026-06-17-u1sz-env-defrag-findings.md` (the obstruction finding).
**Status:** design, pending implementation plan.

---

## 1. Objective & the one question

#610 proved the U(1)-Sz CTM env sector-drop ("C-lever": keep `|Sz| ≤ 1` on the chi bonds) is
**constructible and exact in the forward** path (env block counts 96 → 48, exactly 2×, at D=3 χ=12;
energy finite/sane) but could **not measure** the make-or-break backward charge-mask benefit (Gate
B), because the drop cannot be injected into the 2×2 multisite backward via a runtime monkeypatch.

This work builds the **minimal, opt-in, default-off** structural change that makes the 2×2 multisite
CTM carry a **uniform chi-bond sector set**, so the AD backward becomes **cold-traceable under the
drop**. With the backward finally traceable, run **Gate B** to convert #610's *inferred* ~40–50%
charge-mask reduction into a *measured* number, then decide GO/NO-GO on funding the full feature.

**Output:** a measured Gate-B charge-mask reduction at D=3 χ=12 and a documented GO/NO-GO finding —
**not** a shipped feature. The flag-gated `src/` change exists to make the measurement *possible*;
whether it merges to `main` is itself gated on Gate B (see §8).

## 2. Fixed facts (established by #610 — do NOT re-derive)

From `docs/superpowers/handoffs/2026-06-17-u1sz-env-defrag-findings.md` and
`examples/u1sz_defrag_prototype_610.py`:

1. **The forward drop is real.** Under the prototype, the converged forward `ctm_tensor` env has
   block counts `{C: 3, T: 9}` (from `{C: 5, T: 19}`), energy finite/sane. The representation change
   is faithful in the forward paired-moves path.
2. **The truncation problem is already solved.** The prototype's traced-SVD copy
   (`_make_keep_filtered_traced_svd`) drops non-`keep` sectors in the Phase-1 allocation **and
   suppresses the Phase-2 greedy χ-backfill** that otherwise re-adds `±2` to saturate χ. It produces
   an honest smaller bond (~10 SVs at D=3 χ=12). This is the reference for the truncation constraint;
   we promote it into a flag-gated branch — we do **not** need to re-invent it.
3. **The obstruction is a `5 → 3` sector *transition* mid-sweep, not the truncation.** Even with the
   truncation fixed, the cold backward VJP raises
   `ValueError: Size of label 'd' (4) != (5)` in `_build_enlarged_corner -> contract(C_r, T_v)`
   (`_ctm_tensor_projector_2x2.py`). Root cause: `initialize_ctm_tensor_env` seeds **every** chi bond
   at the full 5-sector `{−2..+2}`, while `_ctm_tensor_sweep_multisite` refreshes **only** the legs
   along the *current* absorption direction per move. A sweep transitioning 5→3 sectors then contracts
   a freshly-dropped 3-sector leg against an un-refreshed 5-sector perpendicular corner leg.
4. **A monkeypatch cannot reach it.** Making the backward honor the drop requires env-init to seed a
   uniform sector set *and* the chi-bond sector set to stay uniform across the sweep — a structural
   change to real `src/` functions. This is the entire reason #610 ended NO-GO-by-obstruction.
5. **Baseline backward op counts (D=3 χ=12, `probe_u1sz_baseline_610.txt`):** **63,612** total ops,
   **7,398** charge-mask / index-arith ops. #609 measured backward op count ≈ linear in block count
   (~1.1 ops/block). These are the Gate-B comparison anchors.

## 3. The flag (API surface)

A single optional knob threaded top-down, **off by default**:

```python
keep_sectors: frozenset[int] | None = None
```

- `keep_sectors=None` (default) → **byte-identical to today.** This is the blast-radius firewall:
  every touched function takes the new path only inside `if keep_sectors is not None:`.
- `keep_sectors=frozenset({-1, 0, 1})` (the documented C-lever, |Sz| ≤ 1) → uniform-sector path
  active end-to-end.

Threaded as a dedicated parameter (not folded into the existing `recipe` string) so default-off is
trivial to assert and the Gate-B probe can pass it explicitly. Flow:
`ctm_tensor(...)` → `initialize_ctm_tensor_env(...)` → corner/edge init helpers; and the AD entry
(`_ctm_energy_ad` / `_make_jit_ctm_step`) → `_ctm_tensor_sweep_multisite` → projector truncation.

## 4. Mechanism — uniform-by-construction (tactic A), with a scoped fallback

**Hypothesis (tactic A):** if every chi bond is **born `keep`** and **every truncation outputs
`keep`**, the sector set is a *uniform invariant* — there is never a 5→3 transition, so the
per-direction sweep refresh never produces a mixed-generation contraction. This avoids rewriting the
hot `_ctm_tensor_sweep_multisite` move loop.

Three injection points, each guarded by `keep_sectors is not None` (file:line from the current
tree — re-confirm after merging #614):

1. **Env-init seed** — `_init_symmetric_standard_corner` (`_ctm_tensor_init.py:~378`) and
   `_init_symmetric_standard_edge` (`~322–325`): filter the fused chi-bond charges to `keep` **before**
   `_grouped_chi_perm`. Chi bonds are born with only `keep` sectors.
2. **Truncation constraint** — `_truncated_svd_symmetric_traced` (`linalg.py:~732–747`, the traced /
   backward path): flag-gated branch that drops non-`keep` sectors in the Phase-1 `base_charges`
   allocation and **never backfills the leftover χ from a non-`keep` sector**. Mirror the same
   constraint in the eager `_retruncate_by_base_charges` (`_ctm_tensor_projector_2x2.py:~709`). This is
   the prototype's patch (4) promoted to a real branch.
3. **base_charges filter** — both `_get_base_charges` (in `_ctm_tensor_convergence.py:~239` and
   `_ctm_tensor_paired_moves.py:~38`): filter the extracted charges to `keep` so the allocation intent
   is consistent across forward and backward.

**Make-or-break check for the mechanism:** the **cold** backward VJP traces **without `ValueError`**
(`jax.clear_caches()` first — see §6 cold-trace caveat). This is the crisp, cheap acceptance gate for
tactic A.

**Risk — per-sector *multiplicities*, not just the sector set.** The `_build_enlarged_corner`
contraction matches legs on sector set **and** per-sector dim. Tactic A only guarantees a uniform
sector *set*; if the env-init seed allocation and the truncation allocation distribute `keep` over
different per-sector multiplicities, the cold trace can still mismatch on dims. Two mitigations, in
order:
- **A-consistency:** make env-init seed and truncation use the *same* per-`keep`-sector allocation
  rule (so multiplicities coincide by construction). Preferred — stays within tactic A.
- **Tactic C (scoped fallback):** if the cold trace still mismatches, conform the perpendicular legs
  to the uniform allocation **only at the `_build_enlarged_corner` boundary** — the smallest delta
  that closes the specific gap, short of a sweep rewrite.

**Tactic B (full all-legs-per-sweep refresh) is explicitly deferred** to the full-feature follow-up
(if Gate B passes); it is the heaviest change to the hottest path and is not needed to *measure*.

## 5. Faithfulness guard (mandatory before Gate B)

Reuse the #610 guard, repointed from the monkeypatch to the real flag: under
`keep_sectors={-1,0,1}`, the CTM **converges**, all env tensors are finite, and `E_uniform` is finite
and in the sane Heisenberg window at D=3 χ=12. An invalid charge-conserving env makes the Gate-B
op-histogram meaningless, so the guard runs **before** the re-profile.

## 6. Measurement (Gate B — the make-or-break, Tier 1)

Repoint `examples/probe_backward_jaxpr_566.py --defrag` from the prototype monkeypatch to the **real
flag** (`keep_sectors={-1,0,1}`). Trace-only, CPU, no XLA compile:

1. Baseline (flag-off) backward op-histogram at D=3 χ=12 → charge-mask cluster + total (anchor: 7,398
   / 63,612 from #610).
2. Flag-on backward op-histogram at D=3 χ=12 → charge-mask cluster + total.
3. Reduction = `1 − (charge_mask_on / charge_mask_off)` and the total-op reduction.

**Cold-trace caveat (load-bearing).** The flag-on backward must be traced **cold**
(`jax.clear_caches()` before the flag-on trace). A prior flag-off trace of the same jit unit with
identical avals seeds the `jax.jit` cache; a later flag-on call silently reuses the flag-off trace —
neither erroring nor dropping. Any "it traced" or measured-drop claim must verify cold tracing, or it
is re-measuring the baseline. The probe must clear caches between the off and on traces.

**Gate B pass criterion:** charge-mask cluster op count drops **≥ 25%** (well beyond #609's ~1%
noise), with total backward ops dropping commensurately.

## 7. Verification & blast radius

- **Default-off regression (the firewall):** with `keep_sectors=None`, existing CTM behavior is
  byte-identical. Run `uv run pytest -m core` and the U(1)-Sz arm
  (`tests/test_profiler_u1sz_arm.py`) flag-off; assert no diff in env block structure / energy vs
  `origin/main` on a D=3 χ=12 smoke.
- **Flag-on faithfulness guard (§5):** new test — CTM converges, env finite, energy sane under the
  flag.
- **Cold-trace-builds unit test:** the #610 obstruction test
  (`test_backward_trace_does_not_survive_surgical_drop`) **inverted to PASS** — under the real flag,
  the cold backward VJP of the 2×2 multisite step **builds** (no `ValueError`). This is the
  regression lock for the structural fix.
- **Blast radius:** flag-on touches env-init + traced/eager projector truncation + the multisite
  sweep's `base_charges` threading — the most-used symmetric CTM paths. Confined entirely behind
  `keep_sectors is not None`; default-off leaves them untouched.

## 8. Decision & merge discipline (gated on Gate B)

| Gate-B result | Action |
|---|---|
| **≥ 25% charge-mask drop** | **GO.** Open PR to `main` with the default-off flag. Open a follow-up issue for the full feature: Gate C (A100 end-to-end + `\|E_uniform − E_frag\|/\|E_frag\| ≤ 1%`, on an **optimized** state — not the random init), accuracy spine, and tactic B (all-legs sweep refresh). |
| **< 25% charge-mask drop** | **NO-GO.** The structural change finally *measured* the lever (converting #610's inference). Write findings + memory recording the measured number; **do not merge** the `src/` change to `main`. The branch preserves it for the record. |

The flag-gated `src/` change lives on the branch to make the measurement possible; **it merges to
`main` only if Gate B passes** — honoring the project's measure-first / don't-fund-on-inference
discipline (mirrors #609/#610).

## 9. Deliverables

- Flag-gated `src/` change (branch-local until Gate B; §3–§4).
- Faithfulness guard test + cold-trace-builds unit test + default-off regression assertion (§7).
- Gate-B op-histogram numbers (flag-off vs flag-on) at D=3 χ=12.
- A findings handoff (`docs/superpowers/handoffs/2026-06-17-u1sz-uniform-sector-env-findings.md`)
  recording the measured charge-mask drop and the GO/NO-GO decision.
- A memory update; link `[[610-u1sz-env-defrag]]`, `[[566-u1sz-stacking-nogo]]`,
  `[[u1sz-perf-study-d3-findings]]`.
- **If GO:** the follow-up full-feature issue (Gate C / accuracy spine / tactic B).

## 10. Risks / kill-switches

- **Multiplicity mismatch defeats tactic A** — caught cheaply by the cold-trace-builds check; mitigated
  by A-consistency then tactic C (§4). If tactic C also fails to build cheaply, the finding is
  "uniform-by-construction insufficient; needs tactic B sweep rewrite" — a documented escalation, not a
  silent expansion of scope.
- **Charge-mask partly irreducible** — Gate B catches it (trace-only) before any A100 spend; a <25%
  measured drop is a clean NO-GO with a real number.
- **Cold-trace cache trap** — §6; the probe clears caches between off/on traces.
- **Blast radius on the hot paths** — confined behind the default-off flag; the default-off regression
  test is the guard.

## 11. Non-goals

- No Gate C / A100 / end-to-end runtime work this round (gated on Gate B passing).
- No accuracy spine / golden infrastructure (belongs to the full-feature follow-up if GO).
- No tactic B (all-legs sweep refresh) this round — deferred to the full feature.
- Not chasing D=2 (symmetry already wins) or D≥4 (note headroom only); the binding case is D=3 χ=12.
- The flag is **not** advertised as public API in this round (no README/`__init__` export) — it is a
  measurement-enabling internal knob until the full feature ships.
