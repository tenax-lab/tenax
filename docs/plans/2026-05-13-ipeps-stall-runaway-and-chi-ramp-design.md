# iPEPS-AD Stall Runaway and χ-Ramp Recompile Design

**Date:** 2026-05-13
**Issues:** #454 (stall runaway), #453 (χ-ramp JIT retrace)
**Scope:** Two independent PRs off `main`. A third item — adaptive convergence-triggered ramping — is split out to a new issue and deferred.

## Background

The 2026-05-13 v2 benchmark (D=3 2-site C4v Heisenberg, `chi_schedule=[(8,30), (16,30), (24,20)]`, GPU CUDA) was killed at χ=16 step 20/30 after 13h 09m wall-clock. Two pathologies surfaced:

1. **Stall runaway (#454).** L-BFGS Wolfe failure triggered `gs_stall_recovery="reset"`, which cleared optimizer history but did *not* roll back the iterate. The next iteration is steepest descent from the same `params` with the same gradient that just failed Wolfe — a mathematical fixed point. 18+ consecutive resets accumulated; energy frozen at -0.6708517722 (dE ≈ 1e-13) across steps 5/10/15/20 of χ=16. No retry cap exists for the `reset` path (the cap on `gs_noise_recovery_retries` gates only the `"noise"` path).
2. **χ-ramp recompile (#453).** `optimize_gs_ad_chi_schedule` re-instantiates CTM env tensors at each new χ in the schedule. JIT-compiled CTM / energy / backward kernels cache-miss on shape, triggering full XLA retraces. One ~57 min retrace cluster was observed mid-χ=16. variPEPS at fixed χ=24 pays one ~2.66 h compile and then runs at ~520 s/step steady state with zero retraces.

Both pathologies showed up because PR #449's grad-norm convergence criterion now refuses to false-exit on `dE ≈ 0`, exposing latent behaviors the old `dE`-based criterion had masked.

## Section 1 — #454 Stall Runaway Fix

**Branch:** `fix/ipeps-stall-runaway`.

### Design

Add two changes to the three reset sites in `src/tenax/algorithms/ipeps_optimize.py`:
- 1-site C4v: `ipeps_optimize.py:1435–1460`
- 2-site:     `ipeps_optimize.py:2264–2290`
- multisite:  `ipeps_optimize.py:2848–2865`

**Change A — retry cap.** Add `gs_stall_recovery_retries: int = 5` to `iPEPSConfig` (next to `gs_noise_recovery_retries`; same default as variPEPS's `optimizer_random_noise_max_retries`). When `stall_count > config.gs_stall_recovery_retries`, log `[iPEPS-AD] stall budget exhausted after N resets, returning best E=…` and break out of the outer optimizer loop via the same exit machinery the noise path already uses on its cap. The final `_eval_fresh` then runs on `best_params`.

**Change B — rollback to best on reset.** Inside the reset branch, before clearing L-BFGS / CG state, set `params = best_params`. The next iteration then starts steepest descent from the best iterate rather than from the iterate at which Wolfe just failed. This breaks the mathematical fixed point.

**Change C — log line + comment.** Update `(no rollback)` → `(rollback to best, retry k/N)`. Replace the `# Do NOT roll back params — issue #298's trajectory study…` comment block with a cross-reference: `# Rollback to best on reset (#454). #298's anti-rollback evidence was on a pre-trifecta CTM stack (pre-PR #406 2x2 projector, pre-multisite-CTM rewrite, pre-PR #447 AD stop_gradient) and no longer applies.`

### Tests

- **Unit test, `core` marker.** `tests/test_ipeps_stall_recovery_cap.py`. Use a synthetic Wolfe-failing driver (patch line-search to always return failure) at the smallest feasible problem size, drive the 2-site optimizer, assert (a) the loop exits after exactly `retries + 1` resets, (b) the returned energy equals `best_E` from when stalls started, (c) the log mentions `stall budget exhausted`.
- **Production canary, `slow` marker.** Re-run a 20-step version of `examples/heisenberg_ipeps_ad_2x2.py` at D=2 χ=8 and assert observed stall count ≤ 3 (today: unbounded).

### Out of scope

- Wolfe-condition diagnostic logging (#454 step 3). Useful but cheap to add later.
- Forced steepest-descent-step variant (b-flex). New algorithmic choice deserving its own benchmark.

### Risk

Rollback contradicts the closed issue #298's empirical finding. The contradiction is justified by the CTM-stack rewrite between 2026-04-11 (#298 closed) and 2026-05-13 (#454 surfaced). If the new tests pass and the production canary shows stall count ≤ 3, we accept the contradiction with a code comment. If stall count regresses on the canary, we reconsider before merging.

## Section 2 — #453 χ-Ramp Recompile Fix

**Branch:** `perf/chi-ramp-pad-env`. Independent of Section 1; can run in parallel.

### Design

Unify the `_maybe_bump_chi` reactive bump (§2.8.2 auto-χ_E path) and the schedule-driven χ ramp into one mechanism. Eliminate per-stage `optimize_gs_ad` calls; the optimizer runs once with env tensors sized to `chi_max` from the very first JIT trace.

**Step 1 — extract the bump mechanism.**

Split `_maybe_bump_chi` in `src/tenax/algorithms/ipeps_optimize.py:35–82` into:

- `_apply_chi_bump(ctm_cfg, env_cache, chi_new, *, base_charges) -> tuple[CTMConfig, dict]` — pure mechanism: `dataclasses.replace(ctm_cfg, chi=chi_new)` and in-place `env_cache["envs"]` padding via `pad_dense_env_chi`. No policy.
- `_maybe_bump_chi(...)` — keeps the ε_T-reactive trigger, delegates to `_apply_chi_bump`. API unchanged.
- `_maybe_scheduled_bump(ctm_cfg, env_cache, step, schedule_targets, *, base_charges)` — new. `schedule_targets` is a sorted list of `(cumulative_step, target_chi)` boundaries. Fires when `step` crosses a boundary, delegates to `_apply_chi_bump`.

**Step 2 — refactor `optimize_gs_ad_chi_schedule` to a shim.**

Public signature unchanged. Internals replaced:

```
def optimize_gs_ad_chi_schedule(hamiltonian_gate, A_init, config, chi_schedule):
    chi_max = max(chi for chi, _ in chi_schedule)
    total_steps = sum(n for _, n in chi_schedule)
    cum = 0
    schedule_targets = []
    for chi, n in chi_schedule:
        cum += n
        schedule_targets.append((cum, chi))  # bump-at-end-of-stage; first stage = initial chi

    # First stage's chi is the initial logical chi; chi_max caps padding.
    ctm_cfg = replace(config.ctm, chi=chi_schedule[0][0], chi_max=chi_max)
    step_cfg = replace(
        config,
        ctm=ctm_cfg,
        gs_num_steps=total_steps,
        gs_chi_schedule_steps=schedule_targets,   # NEW: stash schedule on config
    )
    return optimize_gs_ad(hamiltonian_gate, A_init, step_cfg)
```

The per-stage `for chi, num_steps in chi_schedule: optimize_gs_ad(...)` loop is deleted, along with the env-handoff complexity (today: `current_init = A_opt` only — env is dropped).

**Step 3 — inner-loop wiring.**

In each of `_optimize_gs_ad_*_*` (1-site C4v, 2-site, multisite), at the same step-end block where `_maybe_bump_chi` fires today (e.g. `ipeps_optimize.py:1476` for 1-site C4v), also invoke:

```
if config.gs_chi_schedule_steps is not None:
    ctm_cfg, _env_cache = _maybe_scheduled_bump(
        ctm_cfg, _env_cache, step, config.gs_chi_schedule_steps,
        base_charges=_bump_base_charges,
    )
```

Both bump paths compose: reactive (ε_T-driven) and scheduled. `chi_max` caps both.

**Step 4 — per-χ recording (additive output).**

Reuse the same step-end site. After any bump (reactive or scheduled) actually fires (detected by `ctm_cfg.chi` change), append a `ChiStageRecord(chi_pre, chi_post, step, E, grad_norm)` to a list threaded through the optimizer loop. The list is returned in the result.

API: add optional `return_stages: bool = False` parameter to `optimize_gs_ad_chi_schedule`. When `True`, return `(*existing_result, stages)`. Default `False` preserves existing tuple-unpacking callers. The optimizer loop always builds the list internally; the shim discards it when `return_stages=False`.

A new dataclass in `src/tenax/algorithms/ipeps_optimize.py` or a sibling module:

```
@dataclass(frozen=True)
class ChiStageRecord:
    chi_pre: int     # logical chi before bump
    chi_post: int    # logical chi after bump
    step: int        # optimizer step at which the bump fired
    E: float         # energy at that step (pre-bump CTM)
    grad_norm: float # gradient norm at that step
```

For finite-χ scaling analysis, users get one record per (auto + scheduled) bump.

### Numerical-equivalence audit

Before merging: grep `src/tenax/algorithms/_ctm_*.py` for any use of env tensor shapes as logical χ:

```
grep -nE "env\.(C[1-4]|T[1-4])\.shape" src/tenax/algorithms/_ctm_*.py
grep -nE "\.shape\[0\].*chi|chi.*\.shape" src/tenax/algorithms/_ctm_*.py
```

Any hit that uses `env.{...}.shape` as the truncation cap must be replaced with `ctm_cfg.chi` (logical χ). Padded `chi_max`-shaped envs with logical χ < `chi_max` must truncate at `ctm_cfg.chi` regardless of physical env shape. This invariant is already proven by the §2.8.2 reactive bump path; we're extending it to the schedule-driven case.

### Tests

- **Padding invariance, `core` marker.** `tests/test_ctm_env_pad_chi_schedule.py`. Converge CTM at χ=4 on a fixed `A`, pad envs to χ=8 via `pad_dense_env_chi`, evaluate energy + gradient. Compare to the unpadded χ=4 evaluation on the same `A`. Assert agreement to 1e-12 (energy) and 1e-10 (per-element gradient).
- **End-to-end short run, `slow` marker.** `tests/test_ipeps_chi_schedule_unified.py`. Run `chi_schedule=[(4, 5), (8, 5)]` on Heisenberg D=2 from a fixed seed under the unified mechanism vs a reference run that uses today's per-stage driver (pre-refactor) — pin via subprocess or a snapshot, depending on what's simpler. Final energy must agree within 1e-6 (some optimizer-state-carryover divergence is expected and physical; this checks ballpark equivalence).
- **Recording smoke test, `core` marker.** Run a 2-stage schedule with `return_stages=True`, assert the returned `stages` list has the expected number of records (= number of bump events) and that each `(chi_pre, chi_post)` matches the schedule.

### Trade-off (documented in docstring)

All stages now contract `chi_max`-shaped envs. Stages running at logical χ < `chi_max` pay extra FLOPs in CTM moves vs running at logical-χ-shaped envs. For the production [(8,30), (16,30), (24,20)] schedule, ~80% of steps pay chi=24-sized CTM cost. The recompile evidence in #453 suggests this is a net win; the post-fix benchmark is the verification.

### Out of scope

- Option 2 from #453 issue body (lift χ to `static_argnames`). Only revisited if the padded approach doesn't close the wall-clock gap to variPEPS.
- Adaptive convergence-triggered ramping. Split to a separate follow-up issue (below).
- Production GPU benchmark vs variPEPS. Re-run as a `run` once both fixes land; not a test artifact.

### Risk

The numerical-equivalence audit is the load-bearing step. If any CTM kernel silently uses env shape as truncation χ, padded-to-`chi_max` envs would over-resolve at the small-χ stages — physically a different state than the user asked for. We'll gate the merge on (a) the audit returning no unmasked hits, and (b) the padding-invariance test passing at 1e-12 / 1e-10.

## Section 3 — Deferred: Convergence-Triggered Adaptive Ramping

**Not in scope for this work.** Captured as a follow-up issue to be filed after the design doc lands.

The proposal: replace the schedule's `(chi, num_steps)` "max step budget per stage" with `(chi, max_steps, grad_norm_tol)`, and have `_maybe_scheduled_bump` fire when either the budget is exhausted *or* `grad_norm < grad_norm_tol` at the current logical χ. Since chi_max-shaped envs make small-χ steps as expensive as max-χ steps, the only benefit of staying at small χ is solver-trajectory shaping (warm starting); once converged at small χ there's no reason to keep grinding.

Combined with the per-χ recording from Section 2, this gives a converged data point per χ — the canonical input to a finite-χ extrapolation `E(χ) = E_∞ + a / χ^β`.

Deferred because it changes optimizer trajectory and needs its own benchmark to confirm no regression vs the fixed-step schedule.

## PR Sequencing

Per `feedback_separate_branches_per_concern` — two branches off `main`, parallel-ok:

1. **Branch `fix/ipeps-stall-runaway`** → PR fixing #454. Files touched: `src/tenax/algorithms/ipeps_optimize.py`, `src/tenax/algorithms/ipeps_config.py`, `tests/test_ipeps_stall_recovery_cap.py`.
2. **Branch `perf/chi-ramp-pad-env`** → PR fixing #453. Files touched: `src/tenax/algorithms/ipeps_optimize.py`, `src/tenax/algorithms/ipeps_config.py`, `tests/test_ctm_env_pad_chi_schedule.py`, `tests/test_ipeps_chi_schedule_unified.py`, and a possibly-new sibling module if `ChiStageRecord` doesn't sit cleanly in `ipeps_optimize.py`.

Both touch `ipeps_optimize.py` and `ipeps_config.py`; whichever lands second runs `git merge origin/main` to pick up the other's changes per the CLAUDE.md branch-protection note. Conflict surface is small — the reset sites and the bump sites are disjoint blocks.

## Acceptance

For #454:
- [ ] `gs_stall_recovery_retries: int = 5` lands in config.
- [ ] All three reset sites cap and rollback as specified.
- [ ] Unit test for the cap exits cleanly.
- [ ] Production canary shows stall count ≤ 3 on the 20-step D=2 χ=8 run.

For #453:
- [ ] `optimize_gs_ad_chi_schedule` runs a single inner optimization at `chi_max`-padded envs.
- [ ] Numerical-equivalence audit returns no env-shape-as-χ hits.
- [ ] Padding-invariance test passes to 1e-12 / 1e-10.
- [ ] `return_stages=True` returns a `ChiStageRecord` per bump.
- [ ] Post-merge GPU benchmark re-run shows χ-ramp wall-clock within ~10% of fixed-χ=`chi_max` variPEPS at the same step budget (issue acceptance criterion).
