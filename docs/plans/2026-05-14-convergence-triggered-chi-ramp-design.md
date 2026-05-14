# Convergence-Triggered Adaptive χ Ramping — Design (#455)

**Date:** 2026-05-14
**Status:** brainstormed, ready for implementation plan
**Issue:** [#455](https://github.com/tenax-lab/tenax/issues/455)
**Builds on:** [#453 design (Section 3 — deferred)](2026-05-13-ipeps-stall-runaway-and-chi-ramp-design.md)

## Motivation

The v5 production benchmark (2026-05-13, `project_chi_schedule_v5_validated.md`) ran
`chi_schedule=[(8,30), (16,30), (24,20)]` on Heisenberg D=3 2-site C4v and produced
`E/site = -0.66422749` in 4h 19m, but **chi=24 never ran** — the optimizer saturated at
chi=16 by step 50 (dE machine-zero), exhausted its stall-recovery budget at step 55, and
exited before the step-60 scheduled boundary. Compared to variPEPS at chi=24
(`-0.6681927`), Tenax leaves 0.004 on the table for a stage it never reached.

After #459, env tensors are padded to `chi_max` from the first iteration, so steps at
small logical χ pay the same per-step cost as steps at `chi_max`. The only remaining
reason to spend optimizer steps at small χ is solver-trajectory shaping. Once converged
at the current logical χ, additional steps there are wasted resolution at full
`chi_max` cost.

## Goal

Make chi-schedule stage advancement happen on optimizer-state signals (convergence or
stall-cap exhaustion), not just step-count boundaries. The user-visible effect: chi=24
actually runs in the v5 schedule because chi=16 hands off as soon as it's not making
meaningful progress, not when an arbitrary step boundary is reached.

## Decisions locked during brainstorming

| # | Question | Decision |
|---|---|---|
| Q1 | Bump trigger signals | **Convergence (`_converged_outer`) OR stall-cap exhausted** |
| Q2 | Tolerance API | **Reuse global `gs_grad_norm_tol`** — no per-stage tol |
| Q3 | Unused-step semantics | **Per-stage `max_steps` budget; discard unused on early signal** |
| Q4 | Bump-trigger criterion | **Reuse `_converged_outer`** — user's `gs_conv_criterion` governs both bump and final exit |
| Q5 | PR sequencing | **Two PRs in sequence** — refactor first, behavior change second |
| Q6 | `chi_auto_bump` fate | **Keep + deprecation note** — orthogonal CTM-side signal; revisit after empirical data |

## Architecture

**Internal representation change:** `gs_chi_schedule_steps` becomes a list of per-stage
records `[(target_chi, max_steps_in_stage)]` — the raw stage list, *not* cumulative
boundaries.

**Two new optimizer-loop locals:**

- `current_stage_idx: int` — which stage we're in.
- `stage_start_step: int` — global step at which the current stage began.

**Derived signal:** `steps_in_stage = step - stage_start_step + 1`.

**Single helper** `_advance_chi_stage_if_due(...)` replaces both `_maybe_scheduled_bump`
and the inline convergence/stall break logic. Called once per optimizer step from each
of the four call sites (1-site, 2-site fused, 2-site C4v, multisite).

### Trigger pseudo-code

```python
converged = _converged_outer(config, delta_energy, grad_norm_val)
stall_exhausted = (
    config.gs_stall_recovery == "reset"
    and stall_count >= config.gs_stall_recovery_retries
)
budget_exhausted = (steps_in_stage >= stage_max_steps)
should_advance = converged or stall_exhausted or budget_exhausted

has_next = (current_stage_idx + 1) < len(chi_schedule)

if should_advance:
    if has_next:
        # bump chi to next stage's target; reset stall_count, lbfgs, opt_state
        # advance current_stage_idx, set stage_start_step = step + 1
        bump_fired = True
    else:
        # final-stage exit — same break behaviour as today
        should_break = True
```

At the **final** stage all three triggers exit (identical to today). At **non-final**
stages all three advance.

### Why this representation

Cumulative boundaries (the pre-#455 form) can't express "stage may end early": once
chi=8 ends at step 20 instead of 30, the `(30, 16)` boundary for the next stage is
anchored to a completion event that never happens. Per-stage budgets are user-facing —
they match the public `chi_schedule=[(chi, num_steps)]` literally, so the shim becomes
a pass-through.

## PR sequencing

Two PRs in sequence off `main`, per
[`feedback_separate_branches_per_concern`](../../.claude/projects/-home-yjkao-tenax/memory/feedback_separate_branches_per_concern.md).

### PR 1 — refactor to per-stage state (no behavior change)

**Branch:** `refactor/ipeps-chi-schedule-per-stage-state`

**Touches:**

- `src/tenax/algorithms/ipeps_config.py` — update `gs_chi_schedule_steps` docstring;
  new semantics `[(target_chi, max_steps)]`, no field-name change.
- `src/tenax/algorithms/ipeps_optimize.py:101` — delete `_maybe_scheduled_bump`.
- Introduce `_advance_chi_stage_if_due(...)` helper covering the existing budget path
  only (no new signals yet). Returns
  `(ctm_cfg, env_cache, current_stage_idx, stage_start_step, bump_fired, should_break)`.
- Four call sites (lines 1376, 1673, 2559, 3253) — replace `_maybe_scheduled_bump`
  with the helper. Initialise `current_stage_idx = 0` and `stage_start_step = 0`
  alongside the existing `stall_count` etc.
- `optimize_gs_ad_chi_schedule:526` — drop cumulative-boundary construction; pass
  `chi_schedule` straight through to `gs_chi_schedule_steps`.

**Equivalence:** the current cumulative-boundary search at step N reduces to "advance
when step crosses the next boundary", which under per-stage representation is "advance
when steps_in_stage ≥ max_steps". Identical behaviour; different bookkeeping.

**Risk:** the four call sites have subtly different local-variable names
(`_env_cache`, `_env_cache_2s`, `ctm_cfg_2s`, etc.). Helper signature must accept these
explicitly.

### PR 2 — add convergence + stall-cap triggers

**Branch:** `feat/ipeps-chi-adaptive-bump` (off `main`, after PR 1 merges).

**Touches:**

- `_advance_chi_stage_if_due` — extend the signal set to the three-way OR.
- Four call sites — route the existing convergence-check and stall-cap break paths
  through the helper. At non-final stages, advance instead of breaking.
- `chi_auto_bump` docstring — add the steering note (per Q6): prefer `chi_schedule` +
  convergence-triggered ramping for new code; `chi_auto_bump` retained for the
  CTM-truncation-bottleneck case that optimizer signals can't see. No deprecation
  warning yet — needs empirical post-#455 data first.
- `optimize_gs_ad_chi_schedule` docstring — document the new bump semantics and the
  rollback-then-advance behaviour at stall-cap (the next stage starts from
  `best_params`, not from the failed iterate; matches PR #464's "fresh landscape"
  intent).

**Stall-cap gating:** the stall-cap signal is restricted to
`gs_stall_recovery == "reset"`. The `"noise"` path has its own retry budget
(`gs_noise_recovery_retries`) and different break semantics; entangling them is a
separate analysis. v5 benchmark uses `"reset"`, so this is the production-relevant
path.

## Testing

Per [`feedback_test_mechanism_not_convergence`](../../.claude/projects/-home-yjkao-tenax/memory/feedback_test_mechanism_not_convergence.md),
tests assert state-transition mechanics, not optimization convergence. Convergence is
the production benchmark's job.

### Unit tests on `_advance_chi_stage_if_due` (PR 2)

All `core`, milliseconds each. No JAX. Inject
`(stall_count, grad_norm, delta_energy, steps_in_stage, current_stage_idx,
chi_schedule, gs_conv_criterion, gs_stall_recovery_retries, gs_stall_recovery)` as
inputs.

| # | Signal injected | Stage position | Expected |
|---|---|---|---|
| 1 | `steps_in_stage ≥ max_steps` | non-final | bump, advance, `should_break=False` |
| 2 | `grad_norm < tol` (criterion=`grad_norm`) | non-final | same as #1 |
| 3 | `\|dE\| < tol` (criterion=`dE`) | non-final | same as #1 |
| 4 | `stall_count ≥ retries` (recovery=`reset`) | non-final | same as #1 |
| 5 | `stall_count ≥ retries` (recovery=`noise`) | non-final | no bump (gated) |
| 6 | any signal | **final** | no bump, `should_break=True` |
| 7 | no signal | non-final | no bump, no break |
| 8 | grad-norm AND stall-cap simultaneously | non-final | one bump, advance by 1 stage |

### Wiring smoke test (PR 1 + PR 2)

`chi_schedule=[(2,2),(3,2)]` D=2 Heisenberg, default config. Assert chi=2→chi=3 at step
2 (the budget path — works pre-#455). Parametrise over the four unit-cell paths in PR 2
to confirm the helper got wired into all four call sites. ~few seconds total, `core`.

### Reactive + scheduled compose test (PR 2)

Specific to Risk #1 (Section "Risk"): set `chi_auto_bump=True` with a CTMConfig that
forces high ε_T at chi=4, and a `chi_schedule=[(4,5),(8,5)]`. Assert the reactive bump
fires *first* and the scheduled signal becomes a no-op for that step (idempotent).
Confirms compose-ordering preservation.

### Production benchmark (post-merge, not in CI)

Re-run `examples/heisenberg_ipeps_ad_2x2.py` with the v5 schedule
`chi_schedule=[(8,30),(16,30),(24,20)]` on GPU. Captured in a new memory file. Two
assertions:

- **Mechanism**: chi=24 actually executes (the v5 bug fix — direct evidence the loop
  doesn't cap-exit before the final stage).
- **Energy**: final energy improves on v5's `-0.66422749`; ideally closes some of the
  0.004 gap to variPEPS bipartite `-0.6681927`. Loose bound, not a hard CI gate.

Wall-clock target: within ~5h (vs v5's 4h 19m). #460 not landed yet, so per-bump
recompile cost still bites.

## Acceptance

### PR 1

- [ ] `gs_chi_schedule_steps` repurposed to `[(target_chi, max_steps)]`.
- [ ] `_maybe_scheduled_bump` deleted, replaced by `_advance_chi_stage_if_due`.
- [ ] All four optimizer paths updated.
- [ ] `optimize_gs_ad_chi_schedule` shim passes `chi_schedule` through directly.
- [ ] Wiring smoke test passes.
- [ ] `grep -nE "schedule_targets.*cum"` returns no hits.

### PR 2

- [ ] 8 truth-table unit tests pass.
- [ ] Smoke test parametrised over 4 unit-cell paths passes.
- [ ] Reactive + scheduled compose test passes.
- [ ] No behaviour change vs PR 1 when only the legacy budget path is hit.
- [ ] `chi_auto_bump` docstring carries the steering note.
- [ ] Production benchmark on v5 schedule: **chi=24 executes**; final energy and
  wall-clock captured in a memory file.

## Risks

1. **Compose-ordering with `_maybe_bump_chi` (Q6)**. Both can fire at the same step.
   Existing ordering is reactive-first, scheduled-second
   (`ipeps_optimize.py:1367-1382`). The new helper must preserve this — reactive
   pre-bump makes scheduled trigger a no-op (idempotent). Mitigation: explicit
   compose test in PR 2.
2. **Final-stage detection off-by-one**. `current_stage_idx + 1 >= len(chi_schedule)`
   must be checked *before* incrementing. Mitigation: truth-table test #6 probes
   directly.
3. **C4v 2-site path has a `ctm_cfg_2s` shadow**. Easy to miss in the four-site
   refactor. Mitigation: pattern-grep audit during PR 1 review.
4. **Stall-cap → advance semantics change**. Today: rollback to `best_params`, exit.
   New (non-final): rollback to `best_params`, advance to next chi. The advanced
   stage starts from a previously-tracked best, not the failed iterate. Matches PR
   #464's "fresh landscape" intent, but must be documented in the shim's docstring
   so users aren't surprised.
5. **#460 (chi static_argname recompile) still bites**. Each bump triggers a ~30s
   JIT recompile. With convergence-driven bumps the timing is unpredictable. Doesn't
   affect correctness; bounds the wall-clock benefit of #455 until #460 lands.

## Out of scope (deferred to follow-ups)

- **#458** — `ChiStageRecord` + per-χ recording for finite-χ scaling output. Issue
  body bundles this with #455 but they're orthogonal: recording is "log the bump
  event"; #455 is "decide when to bump". Additive change later.
- **#460** — chi as JIT `static_argname` recompile. Composes with #455 but separate
  concern.
- **Per-stage `grad_norm_tol`** (issue body's three-tuple form). Per Q2 decided
  against; if empirically needed, additive change later.
- **Noise-path adaptive bumping** (`gs_stall_recovery="noise"`). Per Q1, gated to
  `"reset"` only in PR 2.
- **`chi_auto_bump` deprecation/removal**. Per Q6, retained for now; revisit after
  empirical post-#455 data.

## Files of record

- `src/tenax/algorithms/ipeps_optimize.py` — helper + call-site changes.
- `src/tenax/algorithms/ipeps_config.py` — docstring updates for
  `gs_chi_schedule_steps` (PR 1) and `chi_auto_bump` (PR 2).
- `tests/test_ipeps_chi_adaptive_bump_unit.py` — new in PR 2 (8 truth-table tests).
- `tests/test_ipeps_chi_schedule_wiring.py` — new in PR 1, extended in PR 2.
- `examples/heisenberg_ipeps_ad_2x2.py` — production benchmark runner (already
  untracked locally).

## References

- Issue #455 — feature request.
- Issue #449 — grad-norm convergence criterion (`gs_grad_norm_tol`) reused.
- Issue #453 — unified χ-schedule mechanism (the substrate this builds on).
- PRs #464, #465 — stall/lbfgs/opt_state reset on bump; the "fresh landscape"
  policy this work preserves.
- `2026-05-13-ipeps-stall-runaway-and-chi-ramp-design.md` Section 3 — original
  deferred proposal.
- `project_chi_schedule_v5_validated.md` — v5 trace, the empirical motivation.
