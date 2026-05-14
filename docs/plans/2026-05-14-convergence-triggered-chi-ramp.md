# Convergence-Triggered χ Ramping Implementation Plan (#455)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make iPEPS chi-schedule stages advance on optimizer-state signals (convergence or stall-cap exhaustion), not just step-count boundaries. The user-visible payoff: the v5 production schedule `[(8,30),(16,30),(24,20)]` reaches chi=24 instead of cap-exiting at chi=16.

**Architecture:** Two-PR plan. PR 1 replaces the cumulative-boundary `schedule_targets` representation with per-stage state tracking (`current_stage_idx`, `stage_start_step`) — pure refactor, no behavior change. PR 2 layers convergence + stall-cap triggers onto the helper, reusing the user's `gs_conv_criterion` so policy is consistent between stage-advance and final-exit.

**Tech Stack:** Python 3.11/3.12, JAX, optax, pytest. Tenax iPEPS AD path. The design doc this implements is `docs/plans/2026-05-14-convergence-triggered-chi-ramp-design.md` (commit `44e65a4` on branch `design/chi-adaptive-bump-455`).

**Code-pointers context (verified at design time, may have drifted — verify before editing):**

- `src/tenax/algorithms/ipeps_optimize.py:68-98` — `_maybe_bump_chi` (reactive ε_T bump, keep).
- `src/tenax/algorithms/ipeps_optimize.py:101-142` — `_maybe_scheduled_bump` (delete in PR 1).
- `src/tenax/algorithms/ipeps_optimize.py:526-593` — `optimize_gs_ad_chi_schedule` shim.
- Call sites for `_maybe_scheduled_bump`: lines 1376 (1-site, convergence block), 1673 (1-site, end-of-step), 2559 (2-site, end-of-step), 3253 (multisite, end-of-step). C4v 2-site shares the 2-site call site.
- `src/tenax/algorithms/ipeps_config.py:317-329` — `gs_chi_schedule_steps` field + docstring.
- `src/tenax/algorithms/ipeps_config.py:43-58` — `chi_auto_bump` docstring (PR 2 steering note).
- `_converged_outer` helper in `ipeps_optimize.py` (already exists) — exit predicate using `gs_conv_criterion ∈ {"dE","grad_norm","both"}`.
- Stall-cap state: `stall_count` local + `config.gs_stall_recovery_retries` + `config.gs_stall_recovery ∈ {"reset","noise",None}`.

**CLAUDE.md house rules** (all PRs must comply):

- Open a PR via `gh pr create`; merge via `gh pr merge <#> --squash --delete-branch --auto`.
- Required CI checks: `Tests (Python 3.11)`, `Tests (Python 3.12)`, `Tests (macOS, Python 3.12)` — these run `pytest -m core` only. Plan tests target `core` marker where feasible.
- If branch falls behind main, `git merge origin/main` (not rebase).
- Pre-commit hooks installed; do NOT pass `--no-verify`.

---

## PR 1 — Refactor to per-stage state (no behavior change)

**Branch:** `refactor/ipeps-chi-schedule-per-stage-state` off `main`.

**Net change:** purely structural. After this PR, `_maybe_scheduled_bump` is gone; a new `_advance_chi_stage_if_due` helper drives the existing step-budget-triggered advance. Identical user-visible behavior.

### Task 1.0: Branch setup + verify clean tree

**Files:** none (git plumbing only)

**Step 1: Verify you're on a clean tree from main.**

Run: `git status && git log --oneline -1 main`
Expected: working tree clean (or only untracked files unrelated to this work); `main` head at or after `dd2c450` (the most recent stall-count reset fix).

**Step 2: Create the branch.**

Run: `git checkout main && git pull --ff-only && git checkout -b refactor/ipeps-chi-schedule-per-stage-state`
Expected: switched to new branch, no errors.

**Step 3: Confirm pre-commit installed.**

Run: `ls .git/hooks/pre-commit && pre-commit --version`
Expected: file exists; pre-commit ≥ 4.x.

---

### Task 1.1: Add the wiring smoke test (RED)

This is the TDD anchor for the whole refactor. It must pass on `main` BEFORE refactor (proves we're testing the existing behavior) and continue to pass after.

**Files:**

- Create: `tests/test_ipeps_chi_schedule_wiring.py`

**Step 1: Write the smoke test.**

```python
"""Wiring smoke test for ``optimize_gs_ad_chi_schedule`` (#455 PR 1).

Asserts that a 2-stage schedule actually bumps chi between stages,
exercising the helper that PR 1 introduces (and the legacy
_maybe_scheduled_bump function pre-refactor). Marker `core`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import iPEPSConfig, optimize_gs_ad_chi_schedule
from tenax.algorithms._ctm_config import CTMConfig


@pytest.mark.core
def test_chi_schedule_bumps_between_stages():
    """A 2-stage schedule [(chi=2, n=2), (chi=3, n=2)] must advance to chi=3."""
    # Tiny 2-site Heisenberg, D=2. Goal: prove the bump mechanism wires
    # correctly. Not asserting energy or convergence — that's the
    # production benchmark's job.
    d = 2
    sx = jnp.array([[0.0, 0.5], [0.5, 0.0]])
    sz = jnp.array([[0.5, 0.0], [0.0, -0.5]])
    sy = jnp.array([[0.0, -0.5j], [0.5j, 0.0]])
    H2 = (jnp.einsum("ab,cd->acbd", sx, sx)
          + jnp.einsum("ab,cd->acbd", sy, sy)
          + jnp.einsum("ab,cd->acbd", sz, sz)).real.astype(jnp.float64)
    H2 = H2.reshape(d, d, d, d)

    rng = np.random.default_rng(0)
    D = 2
    A = jnp.asarray(rng.standard_normal((D, D, D, D, d)))

    cfg = iPEPSConfig(
        unit_cell="2site",
        max_bond_dim=D,
        ctm=CTMConfig(chi=2, chi_max=3, max_iter=10, conv_tol=1e-4),
        gs_optimizer="lbfgs",
        gs_num_steps=4,  # overridden by shim
        gs_verbose=False,
        gs_conv_tol=1e-30,        # don't converge
        gs_grad_norm_tol=1e-30,   # don't converge
        gs_stall_recovery_retries=99,  # don't trip stall cap
    )

    # Chi-ramp shim; final stage is chi=3.
    result = optimize_gs_ad_chi_schedule(
        H2, (A, A), cfg, chi_schedule=[(2, 2), (3, 2)]
    )

    # Inspect the returned ctm_cfg or env shape to confirm chi advanced.
    # The exact field name depends on what optimize_gs_ad returns; this
    # test will fail-fast if the wiring is broken.
    # Convention: optimize_gs_ad returns a result dict with 'final_chi'
    # OR the tail tuple's ctm_cfg has chi=3. Verify against current API.
    # If the API doesn't expose final chi, add an assertion via the
    # info dict or env shapes.
    final_chi = _extract_final_chi(result)
    assert final_chi == 3, (
        f"Expected final chi=3 after 2-stage schedule [(2,2),(3,2)], "
        f"got chi={final_chi}. The chi-schedule bump did not fire."
    )


def _extract_final_chi(result):
    """Pull final chi out of optimize_gs_ad_chi_schedule's return value.

    Inspect ``optimize_gs_ad``'s actual return signature in
    ``src/tenax/algorithms/ipeps_optimize.py`` (search for ``return``
    statements at the end of ``optimize_gs_ad``). Update this helper to
    match. Typical shape is ``(A_final, info_dict)`` or similar with an
    ``info["ctm_cfg"].chi`` or env shape from which chi can be recovered.
    """
    # PLACEHOLDER — fill in based on optimize_gs_ad's actual return.
    raise NotImplementedError("Inspect optimize_gs_ad return shape and update this.")
```

**Step 2: Inspect `optimize_gs_ad`'s return signature.**

Run: `grep -nE "^    return |^        return " src/tenax/algorithms/ipeps_optimize.py | head -20`
Expected: see all return statements. Pick the one for `optimize_gs_ad` (the function the shim calls). Use that to fill in `_extract_final_chi`.

**Step 3: Run the test on `main` (pre-refactor).**

Switch to main temporarily — `git stash --include-untracked`, `git checkout main`, copy the test file via `cp`, run `pytest tests/test_ipeps_chi_schedule_wiring.py -v`. Expected: PASS (proves the legacy `_maybe_scheduled_bump` already produces this behavior).

Then return: `git checkout refactor/ipeps-chi-schedule-per-stage-state && git stash pop`. Keep the test file on the branch.

**Step 4: Commit the test on the branch.**

```bash
git add tests/test_ipeps_chi_schedule_wiring.py
git commit -m "test(ipeps): wiring smoke test for chi-schedule bumps (#455 PR1 setup)"
```

---

### Task 1.2: Update `gs_chi_schedule_steps` docstring + semantics

**Files:**

- Modify: `src/tenax/algorithms/ipeps_config.py:260-266` (the dataclass docstring entry)
- Modify: `src/tenax/algorithms/ipeps_config.py:317-329` (the field + inline comment)

**Step 1: Edit the docstring entry (lines 260-266).**

Replace the entry text with:

```
        gs_chi_schedule_steps: Outer-loop χ schedule for
                               ``optimize_gs_ad_chi_schedule`` (#453 / #455).
                               List of ``(target_chi, max_steps_in_stage)``
                               pairs; ``None`` disables the schedule.
                               Each pair is one stage: "run up to
                               max_steps_in_stage optimizer iterations
                               at logical chi = target_chi, then advance".
                               Normally set by
                               ``optimize_gs_ad_chi_schedule`` internally;
                               users should not set it directly.
```

**Step 2: Edit the inline field comment (around lines 317-329).**

Replace the long comment block above `gs_chi_schedule_steps` with:

```python
    # Outer-loop χ schedule for ``optimize_gs_ad_chi_schedule``
    # (#453 / #455).  List of ``(target_chi, max_steps_in_stage)``
    # pairs.  Each pair specifies one stage: at most
    # ``max_steps_in_stage`` optimizer iterations at logical
    # chi = ``target_chi``, then advance to the next stage.  Envs
    # are padded to ``ctm.chi_max`` from step 1, so the JIT-compiled
    # kernels never see a shape change.
    #
    # ``None`` (default) means the schedule mechanism is disabled and
    # the inner optimizer uses ``ctm.chi`` throughout.
    # ``optimize_gs_ad_chi_schedule`` sets this field internally
    # (#455 PR 1: passes ``chi_schedule`` through directly without
    # cumulative conversion).
    gs_chi_schedule_steps: list[tuple[int, int]] | None = None
```

**Step 3: Verify no other code depends on the cumulative semantics.**

Run: `grep -rnE "gs_chi_schedule_steps|schedule_targets" src/tenax/ tests/ | grep -v __pycache__`
Expected: every hit is one we'll touch in this PR. Note any unfamiliar callers — there should be none outside `ipeps_optimize.py` and the config.

**Step 4: Commit (semantic change is staged but not yet wired — tests will break next).**

```bash
git add src/tenax/algorithms/ipeps_config.py
git commit -m "refactor(ipeps): redefine gs_chi_schedule_steps as per-stage list (#455 PR1)"
```

---

### Task 1.3: Add `_advance_chi_stage_if_due` helper (budget-only)

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — add helper near `_maybe_scheduled_bump` (around line 101).

**Step 1: Write the helper.**

Insert immediately after `_maybe_bump_chi` (before the now-doomed `_maybe_scheduled_bump`):

```python
def _advance_chi_stage_if_due(
    ctm_cfg: CTMConfig,
    env_cache: dict,
    *,
    chi_schedule: list[tuple[int, int]] | None,
    current_stage_idx: int,
    steps_in_stage: int,
    base_charges: np.ndarray | None = None,
) -> tuple[CTMConfig, dict, int, bool, bool]:
    """Decide whether to advance to the next χ stage and apply it (#455).

    Inputs:
        ctm_cfg, env_cache: current CTM state.
        chi_schedule: per-stage list ``[(target_chi, max_steps), ...]``,
            or ``None`` (no schedule).
        current_stage_idx: which stage is currently active (0-based).
        steps_in_stage: number of completed optimizer steps in the
            current stage (1-based at end-of-step).
        base_charges: SymmetricTensor base charges (ignored on dense).

    Returns:
        (new_ctm_cfg, new_env_cache, new_stage_idx, bump_fired, should_break)

    PR 1 trigger: budget-exhausted only
    (``steps_in_stage >= chi_schedule[current_stage_idx][1]``).
    PR 2 will extend to convergence + stall-cap.

    Behavior:
        - No schedule, or not yet at budget: no-op, returns
          ``(ctm_cfg, env_cache, current_stage_idx, False, False)``.
        - Budget hit at non-final stage: bump chi to next stage's
          target via ``_apply_chi_bump``, advance ``current_stage_idx``,
          return ``bump_fired=True``.
        - Budget hit at final stage: return
          ``should_break=True`` (no bump; caller exits the loop).
    """
    if not chi_schedule:
        return ctm_cfg, env_cache, current_stage_idx, False, False

    _, stage_max_steps = chi_schedule[current_stage_idx]
    budget_exhausted = steps_in_stage >= stage_max_steps

    if not budget_exhausted:
        return ctm_cfg, env_cache, current_stage_idx, False, False

    has_next = (current_stage_idx + 1) < len(chi_schedule)
    if not has_next:
        # Final stage budget exhausted → caller should break out.
        return ctm_cfg, env_cache, current_stage_idx, False, True

    next_chi, _ = chi_schedule[current_stage_idx + 1]
    if ctm_cfg.chi_max is not None:
        next_chi = min(next_chi, ctm_cfg.chi_max)

    if next_chi <= ctm_cfg.chi:
        # Already at or above the next stage's target — advance index
        # without re-applying a bump (matches old idempotent semantics).
        return ctm_cfg, env_cache, current_stage_idx + 1, False, False

    new_ctm_cfg, new_env_cache = _apply_chi_bump(
        ctm_cfg, env_cache, next_chi, base_charges=base_charges
    )
    return new_ctm_cfg, new_env_cache, current_stage_idx + 1, True, False
```

**Step 2: Run a quick sanity check on the helper's signature with a unit-test stub.**

Run: `python -c "from tenax.algorithms.ipeps_optimize import _advance_chi_stage_if_due; print('importable')"`
Expected: `importable`.

**Step 3: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "refactor(ipeps): add _advance_chi_stage_if_due helper (#455 PR1)"
```

---

### Task 1.4: Wire helper into 1-site path (call site at line 1673)

The 1-site path has TWO existing call sites: one inside the `_converged_outer` block (around line 1376) and one at end-of-step (line 1673). PR 1 only rewires the end-of-step site. The convergence-block site stays as-is for PR 1 (it's a special case where the schedule bump fires *because* the inner optimizer wants to exit; this is what PR 2 will replace entirely). For PR 1, leave the line-1376 site calling `_maybe_scheduled_bump` for now.

WAIT — `_maybe_scheduled_bump` is going to be deleted in this PR. So PR 1 must either keep `_maybe_scheduled_bump` until PR 2, OR convert line 1376 too. Cleanest: convert all four `_maybe_scheduled_bump` calls in PR 1 to use the new helper (with `steps_in_stage = step + 1 - stage_start_step` derived locally), and leave PR 2 to add the new signals. Plan reflects this: convert all four sites, then delete the dead helper.

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — call site at line 1673, plus pre-loop initialization.

**Step 1: Locate the 1-site optimizer pre-loop initialization.**

Run: `grep -nE "stall_count = 0|stall_count: int = 0" src/tenax/algorithms/ipeps_optimize.py | head -5`
Expected: line numbers where `stall_count = 0` is initialized. Pick the one in the 1-site path (the for-loop nearest line 1673 going up).

**Step 2: Add per-stage locals beside `stall_count = 0`.**

Right after the existing `stall_count = 0` in the 1-site path, add:

```python
        current_stage_idx = 0
        stage_start_step = 0
```

**Step 3: Replace the call at line 1673.**

Find the block:

```python
        # Scheduled outer-loop χ bump (#453).  Composes with the reactive
        # bump above; ctm_cfg.chi_max caps both.
        if config.gs_chi_schedule_steps is not None:
            ctm_cfg, _env_cache = _maybe_scheduled_bump(
                ctm_cfg,
                _env_cache,
                step + 1,
                config.gs_chi_schedule_steps,
                base_charges=_bump_base_charges,
            )
```

Replace with:

```python
        # Scheduled outer-loop χ bump (#453 / #455).  Composes with the
        # reactive bump above; ctm_cfg.chi_max caps both.  Per-stage
        # state (current_stage_idx, stage_start_step) drives the new
        # helper; #455 PR2 will add convergence/stall-cap triggers.
        if config.gs_chi_schedule_steps is not None:
            steps_in_stage = (step + 1) - stage_start_step
            ctm_cfg, _env_cache, new_stage_idx, _bump_fired, _should_break = (
                _advance_chi_stage_if_due(
                    ctm_cfg,
                    _env_cache,
                    chi_schedule=config.gs_chi_schedule_steps,
                    current_stage_idx=current_stage_idx,
                    steps_in_stage=steps_in_stage,
                    base_charges=_bump_base_charges,
                )
            )
            if new_stage_idx != current_stage_idx:
                current_stage_idx = new_stage_idx
                stage_start_step = step + 1
```

Note: PR 1 ignores `_should_break` (final-stage budget exit) since the existing optimizer for-loop ends naturally at `gs_num_steps`. PR 2 will honor it.

**Step 4: Run the smoke test from Task 1.1.**

Run: `uv run pytest tests/test_ipeps_chi_schedule_wiring.py -v`
Expected: PASS — the 1-site path is wired correctly.

**Step 5: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "refactor(ipeps): wire 1-site path to _advance_chi_stage_if_due (#455 PR1)"
```

---

### Task 1.5: Wire helper into 1-site convergence-block call site (line 1376)

The convergence-block call is special: it fires when `_converged_outer` returns True, to ensure the final env matches `ctm_cfg.chi` at exit. In PR 1 the behavior stays — bump on convergence — but it routes through the new helper.

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — call site at line 1376.

**Step 1: Replace the call.**

Find:

```python
            # Scheduled outer-loop χ bump (#453).  Composes with the
            # reactive bump above; ctm_cfg.chi_max caps both.
            if config.gs_chi_schedule_steps is not None:
                ctm_cfg, _env_cache = _maybe_scheduled_bump(
                    ctm_cfg,
                    _env_cache,
                    step + 1,
                    config.gs_chi_schedule_steps,
                    base_charges=_bump_base_charges,
                )
```

Replace with:

```python
            # Scheduled outer-loop χ bump (#453 / #455).  In PR 1
            # this still only fires on the budget-exhausted path —
            # PR 2 layers convergence/stall signals on top.
            if config.gs_chi_schedule_steps is not None:
                steps_in_stage = (step + 1) - stage_start_step
                ctm_cfg, _env_cache, new_stage_idx, _bump_fired, _ = (
                    _advance_chi_stage_if_due(
                        ctm_cfg,
                        _env_cache,
                        chi_schedule=config.gs_chi_schedule_steps,
                        current_stage_idx=current_stage_idx,
                        steps_in_stage=steps_in_stage,
                        base_charges=_bump_base_charges,
                    )
                )
                if new_stage_idx != current_stage_idx:
                    current_stage_idx = new_stage_idx
                    stage_start_step = step + 1
```

**Step 2: Run smoke test.**

Run: `uv run pytest tests/test_ipeps_chi_schedule_wiring.py -v`
Expected: PASS.

**Step 3: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "refactor(ipeps): wire 1-site convergence-block path (#455 PR1)"
```

---

### Task 1.6: Wire helper into 2-site path (call site at line 2559)

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — call site at line 2559 + pre-loop init.

**Step 1: Add per-stage locals to the 2-site pre-loop.**

Find the 2-site path's `stall_count = 0` initialization (search backward from line 2559). Insert immediately after:

```python
        current_stage_idx = 0
        stage_start_step = 0
```

**Step 2: Replace the call at line 2559.**

Find the block `if config.gs_chi_schedule_steps is not None:` near line 2557 and replace with the analogous structure (using `ctm_cfg_2s`, `_env_cache_2s`, `_bump_base_charges_2s` — match the existing variable names).

```python
        # Scheduled outer-loop χ bump (#453 / #455).  Same invariant
        # as 1-site path: ramp at end-of-step so next iteration sees
        # the bumped χ across value_and_grad, line search and metric
        # precond evaluations.
        if config.gs_chi_schedule_steps is not None:
            steps_in_stage = (step + 1) - stage_start_step
            ctm_cfg_2s, _env_cache_2s, new_stage_idx, bump_fired, _ = (
                _advance_chi_stage_if_due(
                    ctm_cfg_2s,
                    _env_cache_2s,
                    chi_schedule=config.gs_chi_schedule_steps,
                    current_stage_idx=current_stage_idx,
                    steps_in_stage=steps_in_stage,
                    base_charges=_bump_base_charges_2s,
                )
            )
            if new_stage_idx != current_stage_idx:
                # χ bump fired: fresh landscape, fresh stall budget.
                # Clear L-BFGS curvature history so the first step at
                # the new χ is plain steepest descent.
                current_stage_idx = new_stage_idx
                stage_start_step = step + 1
                stall_count = 0
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_params_flat = None
                    prev_grad_flat = None
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                    opt_state = optimizer.init(params)
```

Note: the post-bump reset block (stall_count, lbfgs_history, opt_state) is preserved verbatim — it's just moved inside the `if new_stage_idx != current_stage_idx:` guard.

**Step 3: Run smoke test against the 2-site path.**

The smoke test in Task 1.1 already uses `unit_cell="2site"`, so the existing test exercises this path. Run:

Run: `uv run pytest tests/test_ipeps_chi_schedule_wiring.py -v`
Expected: PASS.

**Step 4: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "refactor(ipeps): wire 2-site path to _advance_chi_stage_if_due (#455 PR1)"
```

---

### Task 1.7: Wire helper into multisite path (call site at line 3253)

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — call site at line 3253 + pre-loop init.

**Step 1: Add per-stage locals to the multisite pre-loop.**

Find the `stall_count = 0` initialization in the multisite path (backward from line 3253). Insert immediately after:

```python
        current_stage_idx = 0
        stage_start_step = 0
```

**Step 2: Replace the call at line 3253.**

Same pattern as Task 1.6 but using the multisite vars: `ctm_cfg`, `_env_cache`, `_bump_base_charges_multi`. Match the existing variable names exactly.

**Step 3: Add multisite smoke-test parametrization.**

Extend `tests/test_ipeps_chi_schedule_wiring.py` to add a second test using a `Lattice(...)` (multisite path). Use a minimal 2x2 lattice. If setting up the lattice in a test is awkward, defer this to PR 2 (where the truth-table tests are anyway). Note in commit.

**Step 4: Run smoke test.**

Run: `uv run pytest tests/test_ipeps_chi_schedule_wiring.py -v`
Expected: PASS (both 2-site and multisite if parametrized).

**Step 5: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps_chi_schedule_wiring.py
git commit -m "refactor(ipeps): wire multisite path to _advance_chi_stage_if_due (#455 PR1)"
```

---

### Task 1.8: Update `optimize_gs_ad_chi_schedule` shim

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py:526-593` — the shim function.

**Step 1: Replace the cumulative-boundary construction.**

Find the body block:

```python
    cum = 0
    schedule_targets: list[tuple[int, int]] = []
    for i in range(1, len(chi_schedule)):
        cum += chi_schedule[i - 1][1]
        schedule_targets.append((cum, chi_schedule[i][0]))

    ctm_cfg = replace(config.ctm, chi=chi_schedule[0][0], chi_max=chi_max)
    step_cfg = replace(
        config,
        ctm=ctm_cfg,
        gs_num_steps=total_steps,
        gs_chi_schedule_steps=schedule_targets,
    )
```

Replace with:

```python
    # #455 PR 1: pass the per-stage schedule straight through. Each
    # stage's max_steps is now a per-stage budget (was cumulative).
    # The optimizer loop tracks current_stage_idx + stage_start_step
    # and advances via _advance_chi_stage_if_due.
    ctm_cfg = replace(config.ctm, chi=chi_schedule[0][0], chi_max=chi_max)
    step_cfg = replace(
        config,
        ctm=ctm_cfg,
        gs_num_steps=total_steps,
        gs_chi_schedule_steps=list(chi_schedule),
    )
```

**Step 2: Update the verbose-print block (around line 587).**

Change `boundaries={schedule_targets}` to `stages={list(chi_schedule)}`.

**Step 3: Run smoke test.**

Run: `uv run pytest tests/test_ipeps_chi_schedule_wiring.py -v`
Expected: PASS.

**Step 4: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "refactor(ipeps): pass chi_schedule through shim directly (#455 PR1)"
```

---

### Task 1.9: Delete the now-dead `_maybe_scheduled_bump`

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py:101-142` — delete the function.

**Step 1: Confirm no remaining callers.**

Run: `grep -nE "_maybe_scheduled_bump\b" src/tenax/ tests/ | grep -v __pycache__`
Expected: zero hits (we wired all four sites).

**Step 2: Delete the function and its docstring.**

Use the Read tool to find the function range (line ~101 to wherever it ends — find the next `def`), then delete those lines.

**Step 3: Update any module-level docstring or imports that referenced it.**

Run: `grep -nE "_maybe_scheduled_bump" src/tenax/algorithms/ipeps_optimize.py docs/`
Expected: any remaining hits are docstring/comment references — update or remove.

**Step 4: Run smoke test + a broader sanity sweep.**

Run: `uv run pytest tests/test_ipeps_chi_schedule_wiring.py tests/test_chi_auto_bump.py tests/test_ipeps_chi_bump_integration.py -v -m "not slow"`
Expected: all pass. These are the schedule/bump-adjacent tests; they catch the case where the cumulative→per-stage change broke the integration with `chi_auto_bump`.

**Step 5: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "refactor(ipeps): drop dead _maybe_scheduled_bump (#455 PR1)"
```

---

### Task 1.10: Run the full core test suite

**Files:** none.

**Step 1: Run core tests.**

Run: `uv run pytest -m core -x`
Expected: all pass. If any test fails, that's our regression — investigate before merging.

**Step 2: Run non-slow tests too.**

Run: `uv run pytest -m "not slow" --ignore=tests/test_chi_auto_bump.py --ignore=tests/test_ipeps_chi_bump_integration.py`

Actually drop the ignores — those tests SHOULD pass:

Run: `uv run pytest -m "not slow"`
Expected: all pass.

**Step 3: Run a grep audit for any lingering cumulative-boundary references.**

Run: `grep -rnE "cum_boundary|schedule_targets|cumulative.*boundar" src/tenax/ tests/ | grep -v __pycache__`
Expected: no hits except in docstrings explicitly noting the change.

---

### Task 1.11: Open PR 1

**Files:** none (git/gh plumbing).

**Step 1: Push branch.**

Run: `git push -u origin refactor/ipeps-chi-schedule-per-stage-state`

**Step 2: Open PR.**

```bash
gh pr create --title "refactor(ipeps): per-stage state for chi schedule (#455 PR1)" --body "$(cat <<'EOF'
## Summary

- Replace cumulative-boundary `schedule_targets` with per-stage state (`current_stage_idx`, `stage_start_step`) tracked in the optimizer loop.
- New `_advance_chi_stage_if_due` helper; delete `_maybe_scheduled_bump`.
- `optimize_gs_ad_chi_schedule` shim passes `chi_schedule` through directly.
- No behavior change; sets up #455 PR 2 (convergence + stall-cap triggers).

Design doc: `docs/plans/2026-05-14-convergence-triggered-chi-ramp-design.md`.

## Test plan

- [ ] `pytest tests/test_ipeps_chi_schedule_wiring.py -v` passes (2-site path; multisite if parametrized).
- [ ] `pytest -m core` passes locally.
- [ ] `pytest tests/test_chi_auto_bump.py tests/test_ipeps_chi_bump_integration.py -v` passes (no regression in adjacent bump paths).
- [ ] No `grep` hits for `_maybe_scheduled_bump` or `schedule_targets` outside expected docstring references.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 3: Enable auto-merge.**

Run: `gh pr merge <PR-num> --squash --delete-branch --auto`
Expected: queued; will merge after required CI passes.

**Step 4: Wait for CI; address any failures.**

Once PR 1 merges, proceed to PR 2.

---

## PR 2 — Add convergence + stall-cap triggers

**Branch:** `feat/ipeps-chi-adaptive-bump` off `main` (after PR 1 merges).

**Net change:** extends `_advance_chi_stage_if_due` with convergence and stall-cap signals, routes the existing `break` paths through the helper.

### Task 2.0: Branch setup

**Files:** none.

**Step 1: Update local main.**

Run: `git checkout main && git pull --ff-only`
Expected: main now contains PR 1 (verify with `git log --oneline -3`).

**Step 2: Create branch.**

Run: `git checkout -b feat/ipeps-chi-adaptive-bump`

---

### Task 2.1: Write the 8 truth-table unit tests (RED)

Per the design doc (and `feedback_test_mechanism_not_convergence`), tests assert state-transition mechanics on the helper directly.

**Files:**

- Create: `tests/test_ipeps_chi_adaptive_bump_unit.py`

**Step 1: Write the test file.**

```python
"""Truth-table unit tests for ``_advance_chi_stage_if_due`` (#455 PR 2).

These tests inject the signal inputs directly and assert state
transitions on the helper — no optimizer run, no JAX, milliseconds
each. Per `feedback_test_mechanism_not_convergence`, convergence on
a real physics problem is the production benchmark's job, not a
unit test.

Marker: ``core``.
"""

import numpy as np
import pytest

from tenax.algorithms._ctm_config import CTMConfig
from tenax.algorithms.ipeps_optimize import _advance_chi_stage_if_due
from tenax.algorithms.ipeps_config import iPEPSConfig


def _make_config(
    *,
    conv_criterion="grad_norm",
    grad_norm_tol=1e-5,
    conv_tol=1e-8,
    stall_recovery="reset",
    stall_retries=3,
):
    """Build an iPEPSConfig pinned to test-relevant fields."""
    return iPEPSConfig(
        ctm=CTMConfig(chi=2, chi_max=8),
        gs_conv_criterion=conv_criterion,
        gs_grad_norm_tol=grad_norm_tol,
        gs_conv_tol=conv_tol,
        gs_stall_recovery=stall_recovery,
        gs_stall_recovery_retries=stall_retries,
    )


@pytest.mark.core
def test_budget_exhausted_non_final_advances():
    """Test #1: budget hit at non-final stage → bump, advance, no break."""
    ctm = CTMConfig(chi=2, chi_max=8)
    env = {}
    schedule = [(2, 3), (4, 3)]

    new_ctm, _new_env, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm, env,
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=3,
        config=_make_config(),
        grad_norm=1e3,
        delta_energy=1e3,
        stall_count=0,
    )
    assert bump_fired is True
    assert should_break is False
    assert new_idx == 1
    assert new_ctm.chi == 4


@pytest.mark.core
def test_grad_norm_signal_non_final_advances():
    """Test #2: grad_norm < tol with criterion=grad_norm → bump."""
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    _, _, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm, {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,  # well within budget
        config=_make_config(conv_criterion="grad_norm", grad_norm_tol=1e-3),
        grad_norm=1e-6,
        delta_energy=1.0,
        stall_count=0,
    )
    assert bump_fired is True
    assert should_break is False
    assert new_idx == 1


@pytest.mark.core
def test_dE_signal_non_final_advances():
    """Test #3: |dE| < tol with criterion=dE → bump."""
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    _, _, new_idx, bump_fired, _ = _advance_chi_stage_if_due(
        ctm, {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,
        config=_make_config(conv_criterion="dE", conv_tol=1e-3),
        grad_norm=1.0,
        delta_energy=1e-6,
        stall_count=0,
    )
    assert bump_fired is True
    assert new_idx == 1


@pytest.mark.core
def test_stall_cap_reset_advances():
    """Test #4: stall_count ≥ retries with recovery=reset → bump."""
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    _, _, new_idx, bump_fired, _ = _advance_chi_stage_if_due(
        ctm, {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,
        config=_make_config(stall_recovery="reset", stall_retries=3),
        grad_norm=1.0,
        delta_energy=1.0,
        stall_count=3,
    )
    assert bump_fired is True
    assert new_idx == 1


@pytest.mark.core
def test_stall_cap_noise_does_not_advance():
    """Test #5: stall_count ≥ retries but recovery=noise → no bump.

    Noise path has its own retry budget; PR 2 explicitly gates the
    stall-cap bump signal to recovery=reset.
    """
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    _, _, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm, {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,
        config=_make_config(stall_recovery="noise", stall_retries=3),
        grad_norm=1.0,
        delta_energy=1.0,
        stall_count=99,
    )
    assert bump_fired is False
    assert should_break is False
    assert new_idx == 0


@pytest.mark.core
def test_final_stage_any_signal_breaks():
    """Test #6: at final stage, any signal returns should_break=True, no bump."""
    ctm = CTMConfig(chi=4, chi_max=8)
    schedule = [(2, 3), (4, 3)]

    # Budget signal at final stage.
    _, _, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm, {},
        chi_schedule=schedule,
        current_stage_idx=1,
        steps_in_stage=3,
        config=_make_config(),
        grad_norm=1.0,
        delta_energy=1.0,
        stall_count=0,
    )
    assert bump_fired is False
    assert should_break is True
    assert new_idx == 1


@pytest.mark.core
def test_no_signal_no_action():
    """Test #7: no signal tripped → no-op."""
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    new_ctm, _, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm, {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,
        config=_make_config(),
        grad_norm=1.0,
        delta_energy=1.0,
        stall_count=0,
    )
    assert bump_fired is False
    assert should_break is False
    assert new_idx == 0
    assert new_ctm.chi == 2


@pytest.mark.core
def test_simultaneous_signals_advance_once():
    """Test #8: grad_norm AND stall-cap together → single bump (idempotent)."""
    ctm = CTMConfig(chi=2, chi_max=8)
    schedule = [(2, 30), (4, 30)]

    _, _, new_idx, bump_fired, should_break = _advance_chi_stage_if_due(
        ctm, {},
        chi_schedule=schedule,
        current_stage_idx=0,
        steps_in_stage=5,
        config=_make_config(grad_norm_tol=1e-3, stall_retries=3),
        grad_norm=1e-6,
        delta_energy=1.0,
        stall_count=99,
    )
    assert bump_fired is True
    assert new_idx == 1  # advanced exactly once, not twice
    assert should_break is False
```

**Step 2: Run the tests; expect failure (signals not yet implemented).**

Run: `uv run pytest tests/test_ipeps_chi_adaptive_bump_unit.py -v`
Expected: all tests fail (TypeError on signature mismatch — helper doesn't accept `config`, `grad_norm`, etc. yet).

**Step 3: Commit.**

```bash
git add tests/test_ipeps_chi_adaptive_bump_unit.py
git commit -m "test(ipeps): truth-table unit tests for adaptive chi bump (#455 PR2 RED)"
```

---

### Task 2.2: Extend `_advance_chi_stage_if_due` with new signals (GREEN)

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — the helper added in PR 1.

**Step 1: Update the helper signature + body.**

Replace the PR-1 body with:

```python
def _advance_chi_stage_if_due(
    ctm_cfg: CTMConfig,
    env_cache: dict,
    *,
    chi_schedule: list[tuple[int, int]] | None,
    current_stage_idx: int,
    steps_in_stage: int,
    config: "iPEPSConfig",
    grad_norm: float,
    delta_energy: float,
    stall_count: int,
    base_charges: np.ndarray | None = None,
) -> tuple[CTMConfig, dict, int, bool, bool]:
    """Decide whether to advance to the next χ stage and apply it (#455).

    Three signals trigger an advance at non-final stages:
        - ``steps_in_stage >= max_steps`` (budget; existing).
        - ``_converged_outer(config, delta_energy, grad_norm)``
          (NEW PR 2 — reuses user's gs_conv_criterion).
        - ``stall_count >= config.gs_stall_recovery_retries`` AND
          ``config.gs_stall_recovery == "reset"`` (NEW PR 2 —
          gated to reset path; noise path has its own retries).

    At the final stage all three trigger ``should_break=True`` with
    no bump (matches existing exit semantics).
    """
    if not chi_schedule:
        return ctm_cfg, env_cache, current_stage_idx, False, False

    _, stage_max_steps = chi_schedule[current_stage_idx]
    budget_exhausted = steps_in_stage >= stage_max_steps

    converged = _converged_outer(config, delta_energy, grad_norm)

    stall_exhausted = (
        config.gs_stall_recovery == "reset"
        and stall_count >= config.gs_stall_recovery_retries
    )

    should_advance = budget_exhausted or converged or stall_exhausted
    if not should_advance:
        return ctm_cfg, env_cache, current_stage_idx, False, False

    has_next = (current_stage_idx + 1) < len(chi_schedule)
    if not has_next:
        return ctm_cfg, env_cache, current_stage_idx, False, True

    next_chi, _ = chi_schedule[current_stage_idx + 1]
    if ctm_cfg.chi_max is not None:
        next_chi = min(next_chi, ctm_cfg.chi_max)

    if next_chi <= ctm_cfg.chi:
        return ctm_cfg, env_cache, current_stage_idx + 1, False, False

    new_ctm_cfg, new_env_cache = _apply_chi_bump(
        ctm_cfg, env_cache, next_chi, base_charges=base_charges
    )
    return new_ctm_cfg, new_env_cache, current_stage_idx + 1, True, False
```

**Step 2: Run the truth-table tests; expect all green.**

Run: `uv run pytest tests/test_ipeps_chi_adaptive_bump_unit.py -v`
Expected: 8/8 PASS.

If any fail, fix the helper before proceeding. Common failure modes:

- Test #5 fails → check the `gs_stall_recovery == "reset"` gate.
- Test #6 fails → check `has_next` is evaluated *before* increment.
- Test #8 fails → check OR-short-circuit doesn't increment twice.

**Step 3: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "feat(ipeps): convergence + stall-cap triggers in chi-stage helper (#455 PR2)"
```

---

### Task 2.3: Wire new helper signature into 1-site path (line 1673)

The four call sites need to pass the new helper args: `config`, `grad_norm`, `delta_energy`, `stall_count`.

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — call site at line 1673.

**Step 1: Update the call site.**

Find the PR-1 call (around line 1673 — slightly shifted due to PR 1 changes). Update the kwargs:

```python
        if config.gs_chi_schedule_steps is not None:
            steps_in_stage = (step + 1) - stage_start_step
            # Compute grad_norm if not already on this branch.
            # (1-site path already computes grad_norm_val above for
            # the convergence check; reuse if available, else recompute.)
            ctm_cfg, _env_cache, new_stage_idx, bump_fired, _should_break = (
                _advance_chi_stage_if_due(
                    ctm_cfg,
                    _env_cache,
                    chi_schedule=config.gs_chi_schedule_steps,
                    current_stage_idx=current_stage_idx,
                    steps_in_stage=steps_in_stage,
                    config=config,
                    grad_norm=grad_norm_val if grad_norm_val is not None else _grad_l2_norm(grads),
                    delta_energy=delta_energy,
                    stall_count=stall_count,
                    base_charges=_bump_base_charges,
                )
            )
            if new_stage_idx != current_stage_idx:
                current_stage_idx = new_stage_idx
                stage_start_step = step + 1
            if _should_break:
                break  # final-stage exit
```

**Step 2: Run smoke test.**

Run: `uv run pytest tests/test_ipeps_chi_schedule_wiring.py -v`
Expected: PASS.

**Step 3: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "feat(ipeps): pass signals into chi helper at 1-site end-of-step (#455 PR2)"
```

---

### Task 2.4: Replace 1-site convergence-block break with helper-driven advance

This is the **load-bearing behavior change** for #455: when `_converged_outer` returns True at a non-final stage, advance instead of breaking.

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — the `if _converged_outer(...)` block around line 1352.

**Step 1: Find the block.**

Look for:

```python
        if _converged_outer(config, delta_energy, grad_norm_val):
            # ... (existing bump + reset + log + break)
            _converged = True
            break
```

**Step 2: Rewrite the block.**

Replace with:

```python
        if _converged_outer(config, delta_energy, grad_norm_val):
            # #455 PR 2: at non-final stages, advance chi instead of
            # breaking. The helper routes both convergence and final-
            # stage signals through one path.
            chi_before_bump = ctm_cfg.chi
            last_eps_t = float(_env_cache.get("max_truncation_error", 0.0))
            ctm_cfg, _env_cache = _maybe_bump_chi(
                ctm_cfg,
                _env_cache,
                last_eps_t,
                base_charges=_bump_base_charges,
            )
            if config.gs_chi_schedule_steps is not None:
                steps_in_stage = (step + 1) - stage_start_step
                ctm_cfg, _env_cache, new_stage_idx, bump_fired, should_break = (
                    _advance_chi_stage_if_due(
                        ctm_cfg,
                        _env_cache,
                        chi_schedule=config.gs_chi_schedule_steps,
                        current_stage_idx=current_stage_idx,
                        steps_in_stage=steps_in_stage,
                        config=config,
                        grad_norm=grad_norm_val,
                        delta_energy=delta_energy,
                        stall_count=stall_count,
                        base_charges=_bump_base_charges,
                    )
                )
                if new_stage_idx != current_stage_idx:
                    # Bump fired — fresh landscape, fresh stall budget.
                    current_stage_idx = new_stage_idx
                    stage_start_step = step + 1
                    stall_count = 0
                    if is_metric_lbfgs:
                        lbfgs_history.clear()
                        prev_A_flat = None
                        prev_grad_flat = None
                    if is_cg:
                        cg_direction = None
                        prev_grad = None
                        prev_precond_grad = None
                    if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                        opt_state = optimizer.init(params)
                    if _accepted_best_this_iter:
                        best_env_cache = dict(_env_cache)
                    # Don't break — continue at the new chi.
                    continue
            elif ctm_cfg.chi != chi_before_bump:
                # Reactive bump fired without a schedule — same reset.
                stall_count = 0
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_A_flat = None
                    prev_grad_flat = None
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                    opt_state = optimizer.init(params)
                if _accepted_best_this_iter:
                    best_env_cache = dict(_env_cache)
            if config.gs_verbose:
                if not logged:
                    _log_ad_step(
                        "1site-tensor",
                        step,
                        config.gs_num_steps,
                        energy_float,
                        delta_energy,
                        best_energy,
                    )
                _log_ad_converged(
                    "1site-tensor",
                    step,
                    delta_energy,
                    config.gs_conv_tol,
                    grad_norm=grad_norm_val,
                    grad_norm_tol=config.gs_grad_norm_tol,
                    criterion=config.gs_conv_criterion,
                )
            _converged = True
            break
```

Note: only break on the convergence path if we did NOT advance to a new stage. If we did advance, the loop continues at the new chi.

**Step 3: Run smoke test + truth-table tests.**

Run: `uv run pytest tests/test_ipeps_chi_adaptive_bump_unit.py tests/test_ipeps_chi_schedule_wiring.py -v`
Expected: all PASS.

**Step 4: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "feat(ipeps): convergence at non-final stage advances chi (#455 PR2)"
```

---

### Task 2.5: Replace 1-site stall-cap break with helper-driven advance

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — the stall-cap break in the 1-site path.

**Step 1: Find the stall-cap-exit block.**

Run: `grep -nE "stall_count >= config.gs_stall_recovery_retries|stall_count > config.gs_stall_recovery_retries" src/tenax/algorithms/ipeps_optimize.py`
Expected: a handful of hits — find the one in the 1-site path that `break`s after a `best_params` rollback (likely around the stall-recovery handler near line 2540 for 2-site, similar block in 1-site).

**Step 2: Wrap the break with a stage-advance check.**

Before the `break` that fires on stall-cap exhaustion in the 1-site path, insert:

```python
                # #455 PR 2: at non-final stages, advance chi instead
                # of exiting the optimizer.
                if config.gs_chi_schedule_steps is not None:
                    steps_in_stage = (step + 1) - stage_start_step
                    ctm_cfg, _env_cache, new_stage_idx, bump_fired, _ = (
                        _advance_chi_stage_if_due(
                            ctm_cfg,
                            _env_cache,
                            chi_schedule=config.gs_chi_schedule_steps,
                            current_stage_idx=current_stage_idx,
                            steps_in_stage=steps_in_stage,
                            config=config,
                            grad_norm=_grad_l2_norm(grads),
                            delta_energy=delta_energy,
                            stall_count=stall_count,
                            base_charges=_bump_base_charges,
                        )
                    )
                    if bump_fired:
                        current_stage_idx = new_stage_idx
                        stage_start_step = step + 1
                        stall_count = 0
                        # rolled-back-to-best params remain; fresh chi run
                        # starts from best (PR #464 intent preserved).
                        if is_metric_lbfgs:
                            lbfgs_history.clear()
                            prev_A_flat = None
                            prev_grad_flat = None
                        if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
                            opt_state = optimizer.init(params)
                        continue  # skip the break
                # Existing break path:
                break
```

The exact local-variable names depend on the 1-site path; match what's already there.

**Step 3: Run smoke test + truth-table tests.**

Run: `uv run pytest tests/test_ipeps_chi_adaptive_bump_unit.py tests/test_ipeps_chi_schedule_wiring.py -v`
Expected: all PASS.

**Step 4: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "feat(ipeps): stall-cap at non-final stage advances chi (1-site) (#455 PR2)"
```

---

### Task 2.6: Mirror Tasks 2.3, 2.4, 2.5 for 2-site path

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — 2-site call sites and break paths.

For each of the three changes (end-of-step call site, convergence-block intercept, stall-cap-break intercept) on the 2-site path:

1. Locate the equivalent block using grep (the variable names use `ctm_cfg_2s`, `_env_cache_2s`, `_bump_base_charges_2s`).
2. Apply the analogous edit.
3. Run truth-table + smoke tests.
4. Commit each block separately.

Suggested commit titles:

- `feat(ipeps): pass signals into chi helper at 2-site end-of-step (#455 PR2)`
- `feat(ipeps): 2-site convergence at non-final stage advances chi (#455 PR2)`
- `feat(ipeps): 2-site stall-cap at non-final stage advances chi (#455 PR2)`

After each commit: `uv run pytest tests/test_ipeps_chi_adaptive_bump_unit.py tests/test_ipeps_chi_schedule_wiring.py -v` must pass.

---

### Task 2.7: Mirror Tasks 2.3, 2.4, 2.5 for multisite path

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — multisite call sites and break paths.

Same as Task 2.6 but for the multisite path (vars `_bump_base_charges_multi`). Suggested commits:

- `feat(ipeps): pass signals into chi helper at multisite end-of-step (#455 PR2)`
- `feat(ipeps): multisite convergence at non-final stage advances chi (#455 PR2)`
- `feat(ipeps): multisite stall-cap at non-final stage advances chi (#455 PR2)`

---

### Task 2.8: Add reactive-+-scheduled compose test

This pins Risk #1 from the design doc: reactive `_maybe_bump_chi` and scheduled `_advance_chi_stage_if_due` must compose. Reactive fires first; scheduled becomes a no-op for that step if the target chi was already reached.

**Files:**

- Modify: `tests/test_ipeps_chi_schedule_wiring.py` — add a new test.

**Step 1: Write the test.**

```python
@pytest.mark.core
def test_reactive_and_scheduled_compose():
    """Reactive ε_T bump pre-empts scheduled bump in the same step.

    When ``chi_auto_bump=True`` triggers a chi bump to value X, and the
    scheduled bump for the same step would target the same X, the
    scheduled call must be idempotent — advance the stage index
    without re-applying the bump.
    """
    # Build a config with chi_auto_bump=True at a low threshold so
    # the reactive bump fires; chi_schedule asks for the same final chi.
    # ...
    # Run optimize_gs_ad_chi_schedule with both signals primed.
    # Assert final chi reached the target once, not twice; stage_idx
    # advanced correctly.
```

The exact setup depends on the path used; pick the 2-site path for consistency with the existing smoke test.

**Step 2: Run and confirm green.**

Run: `uv run pytest tests/test_ipeps_chi_schedule_wiring.py::test_reactive_and_scheduled_compose -v`
Expected: PASS.

**Step 3: Commit.**

```bash
git add tests/test_ipeps_chi_schedule_wiring.py
git commit -m "test(ipeps): reactive + scheduled compose test (#455 PR2)"
```

---

### Task 2.9: Add `chi_auto_bump` docstring steering note

**Files:**

- Modify: `src/tenax/algorithms/ipeps_config.py:43-58` — `chi_auto_bump` field docstring.

**Step 1: Update the docstring.**

Append (don't replace):

```
                            For new code, prefer ``chi_schedule`` +
                            ``optimize_gs_ad_chi_schedule`` with
                            convergence-triggered ramping (#455);
                            ``chi_auto_bump`` is retained as an
                            orthogonal CTM-truncation sentinel for the
                            case where the optimizer is making progress
                            but ε_T indicates CTM under-resolution.
                            These two mechanisms compose
                            (reactive fires first, scheduled second).
```

**Step 2: Commit.**

```bash
git add src/tenax/algorithms/ipeps_config.py
git commit -m "docs(ipeps): steering note on chi_auto_bump vs adaptive chi_schedule (#455 PR2)"
```

---

### Task 2.10: Update `optimize_gs_ad_chi_schedule` docstring

**Files:**

- Modify: `src/tenax/algorithms/ipeps_optimize.py` — the shim's docstring.

**Step 1: Add the new behavior + the rollback-then-advance note.**

Replace the `chi_schedule` argument's docstring with:

```
        chi_schedule:     List of ``(chi, max_steps)`` pairs, e.g.
                          ``[(8, 100), (16, 50), (32, 30)]``.  Each pair
                          says "run up to max_steps optimizer iterations
                          at logical chi = chi, then advance to the next
                          stage".

                          Three signals advance a stage at non-final
                          stages (#455):
                              - the per-stage ``max_steps`` budget is
                                exhausted;
                              - the user's ``gs_conv_criterion`` (dE,
                                grad_norm, or both) is met;
                              - the L-BFGS reset-recovery stall cap
                                ``gs_stall_recovery_retries`` is hit.
                          Unused steps from an early-exiting stage are
                          discarded (each stage's max_steps is an
                          upper bound, not a fixed quota).

                          Note: when stall-cap triggers a non-final
                          advance, the next stage starts from the
                          rolled-back ``best_params`` — fresh landscape,
                          fresh retry budget (PR #464's intent
                          preserved).
```

**Step 2: Commit.**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "docs(ipeps): document #455 advance semantics in chi_schedule shim"
```

---

### Task 2.11: Run the full test suite

**Files:** none.

**Step 1: Core tests.**

Run: `uv run pytest -m core -x`
Expected: all pass.

**Step 2: Adjacent bump tests.**

Run: `uv run pytest tests/test_chi_auto_bump.py tests/test_ipeps_chi_bump_integration.py tests/test_ipeps_stall_recovery_cap.py -v`
Expected: all pass.

**Step 3: Non-slow full sweep.**

Run: `uv run pytest -m "not slow"`
Expected: all pass.

---

### Task 2.12: Open PR 2

**Files:** none.

**Step 1: Push.**

Run: `git push -u origin feat/ipeps-chi-adaptive-bump`

**Step 2: Create PR.**

```bash
gh pr create --title "feat(ipeps): convergence-triggered adaptive chi ramping (#455)" --body "$(cat <<'EOF'
## Summary

- Stage advance now fires on three signals at non-final stages: budget exhausted (existing), `_converged_outer` (convergence), or `stall_count >= retries` with `recovery=reset` (stall cap).
- Reuses user's `gs_conv_criterion` and `gs_grad_norm_tol` — no new config knobs.
- Each `max_steps` is a per-stage upper bound; unused steps from an early-exiting stage are discarded.
- `chi_auto_bump` retains its CTM-truncation-sentinel role; docstring updated with steering toward the new path.

Closes #455.

Design doc: `docs/plans/2026-05-14-convergence-triggered-chi-ramp-design.md`.

## Test plan

- [ ] 8 truth-table unit tests on `_advance_chi_stage_if_due` pass.
- [ ] Wiring smoke test passes for 2-site and multisite paths.
- [ ] Reactive + scheduled compose test passes.
- [ ] `pytest -m core` passes locally.
- [ ] No regression in `test_chi_auto_bump.py` or `test_ipeps_chi_bump_integration.py`.
- [ ] Production benchmark: re-run `examples/heisenberg_ipeps_ad_2x2.py` with v5 schedule `[(8,30),(16,30),(24,20)]`; **chi=24 actually executes** (recorded in a memory file).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 3: Enable auto-merge.**

```bash
gh pr merge <PR-num> --squash --delete-branch --auto
```

---

### Task 2.13: Production benchmark validation (post-merge)

**Files:**

- Run: `examples/heisenberg_ipeps_ad_2x2.py` (still locally untracked).
- Create: `~/.claude/projects/-home-yjkao-tenax/memory/project_455_chi24_benchmark.md` capturing the result.

**Step 1: Run the v5 schedule.**

Run: `cd ~/tenax && uv run python examples/heisenberg_ipeps_ad_2x2.py 2>&1 | tee /tmp/heisenberg_ipeps_455.log`
Expected: full run completes; **chi=24 stage actually executes** (this is the #455 bug fix). Wall-clock target ~5h.

**Step 2: Audit the log for the chi=24 transition.**

Run: `grep -nE "chi.*16.*24|stage 3|chi=24" /tmp/heisenberg_ipeps_455.log`
Expected: at least one line showing the chi=16 → chi=24 advance, and step counts at chi=24.

**Step 3: Save a memory file.**

Create the memory file with: final energy, wall-clock, the chi=24 transition step, comparison to v5 baseline (`-0.66422749` in 4h 19m) and variPEPS bipartite (`-0.6681927` in 14h). Note any unexpected behavior.

**Step 4: Add memory pointer to MEMORY.md index.**

Append the one-liner to `~/.claude/projects/-home-yjkao-tenax/memory/MEMORY.md`.

---

## Out of scope (deferred to follow-ups)

Per the design doc, these are NOT in this plan and should not creep in:

- **#458** — `ChiStageRecord` + per-χ recording for finite-χ scaling output.
- **#460** — chi as JIT `static_argname` recompile elimination.
- **Per-stage `grad_norm_tol`** in the chi_schedule tuple.
- **Noise-path adaptive bumping** (`gs_stall_recovery="noise"`).
- **`chi_auto_bump` removal** — kept with steering note; revisit post-empirical-data.

---

## File map (summary)

- `src/tenax/algorithms/ipeps_optimize.py` — primary surgery for both PRs.
- `src/tenax/algorithms/ipeps_config.py` — docstring updates for `gs_chi_schedule_steps` (PR 1) and `chi_auto_bump` (PR 2).
- `tests/test_ipeps_chi_schedule_wiring.py` — new in PR 1; extended in PR 2 (compose test).
- `tests/test_ipeps_chi_adaptive_bump_unit.py` — new in PR 2 (8 truth-table tests).
- `examples/heisenberg_ipeps_ad_2x2.py` — production benchmark runner (untracked; for PR 2 post-merge).
- `docs/plans/2026-05-14-convergence-triggered-chi-ramp-design.md` — the design doc (already committed on `design/chi-adaptive-bump-455`).
