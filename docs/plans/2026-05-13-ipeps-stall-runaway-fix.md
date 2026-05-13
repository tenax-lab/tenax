# iPEPS-AD Stall Runaway Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stop the L-BFGS Wolfe-failure → reset → Wolfe-failure runaway in iPEPS-AD optimization (issue #454) by adding a retry cap and rolling back the iterate to `best_params` on reset.

**Architecture:** Two changes at three reset sites in `src/tenax/algorithms/ipeps_optimize.py` (1-site C4v ~L1435, 2-site ~L2264, multisite ~L2848). Add `gs_stall_recovery_retries: int = 5` to `iPEPSConfig`. When the cap is exceeded, log a reason and `break` out of the optimizer loop so the final `_eval_fresh` runs on `best_params`. When within the cap, set `params = best_params` (and refresh `_env_cache` from `best_env_cache`) *before* clearing optimizer history — breaking the mathematical fixed point where reset alone leaves the next steepest-descent step identical to the one that just failed Wolfe.

**Tech Stack:** Python, JAX, pytest, optax. Design doc: `docs/plans/2026-05-13-ipeps-stall-runaway-and-chi-ramp-design.md` (Section 1).

**Branch:** `fix/ipeps-stall-runaway` → PR closes #454.

**Pre-requisites:**
- Worktree on a feature branch off latest `main`.
- `pre-commit` installed (`pre-commit install`) — see `feedback_precommit.md`.
- Read these before starting:
  - `docs/plans/2026-05-13-ipeps-stall-runaway-and-chi-ramp-design.md` Section 1.
  - Issue #454 (gh issue view 454).
  - Issue #298 (gh issue view 298) — for the historical anti-rollback finding being superseded.
  - `src/tenax/algorithms/ipeps_optimize.py:1207-1216` — existing reset-with-rollback in the CTM-error handler; the pattern to mirror.
  - `src/tenax/algorithms/ipeps_optimize.py:1435-1460` — buggy reset block (no rollback).

---

## Task 1: Create the branch and verify baseline tests pass

**Files:** none (git only)

**Step 1: Create branch**

```bash
git checkout -b fix/ipeps-stall-runaway origin/main
```

**Step 2: Verify pre-commit + baseline**

```bash
pre-commit install
uv run pytest -m core -x -q
```

Expected: pre-commit reports already-installed or installs; pytest passes (count varies, but no failures or errors).

**Step 3: Commit nothing — branch only**

No commit yet.

---

## Task 2: Add `gs_stall_recovery_retries` field to `iPEPSConfig`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py:307` (insert after `gs_noise_recovery_retries`)
- Modify: `src/tenax/algorithms/ipeps_config.py:412` (insert validation after `gs_grad_norm_tol` check)
- Test: `tests/test_ipeps_config_stall_recovery_retries.py` (new)

**Step 1: Write the failing test**

Create `tests/test_ipeps_config_stall_recovery_retries.py`:

```python
"""gs_stall_recovery_retries field tests (issue #454)."""
import pytest

from tenax.algorithms.ipeps_config import iPEPSConfig


def test_gs_stall_recovery_retries_defaults_to_5():
    cfg = iPEPSConfig()
    assert cfg.gs_stall_recovery_retries == 5


def test_gs_stall_recovery_retries_must_be_non_negative():
    with pytest.raises(ValueError, match="gs_stall_recovery_retries"):
        iPEPSConfig(gs_stall_recovery_retries=-1)


def test_gs_stall_recovery_retries_zero_is_allowed():
    # 0 means: no resets allowed; first stall exits immediately.
    cfg = iPEPSConfig(gs_stall_recovery_retries=0)
    assert cfg.gs_stall_recovery_retries == 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_ipeps_config_stall_recovery_retries.py -v
```

Expected: FAIL — `AttributeError: ... has no attribute 'gs_stall_recovery_retries'`.

**Step 3: Implement minimal code**

In `src/tenax/algorithms/ipeps_config.py`, locate the `gs_noise_recovery_retries: int = 3` line (~line 307) and add immediately after:

```python
    gs_stall_recovery_retries: int = 5  # max consecutive resets before giving up (#454)
```

Then in the `__post_init__` validation block (after the `gs_grad_norm_tol` check ~L412–415), add:

```python
        if self.gs_stall_recovery_retries < 0:
            raise ValueError(
                f"gs_stall_recovery_retries must be non-negative, "
                f"got {self.gs_stall_recovery_retries}"
            )
```

Also update the class docstring around `gs_noise_recovery_retries` (currently at ~L260 area; find `gs_stall_recovery:` in the docstring) to add a sentence about `gs_stall_recovery_retries`. Example:

```
        gs_stall_recovery_retries: Maximum consecutive resets allowed on
                                   the ``"reset"`` recovery path before the
                                   optimizer exits with ``best_params``.
                                   Analogous to ``gs_noise_recovery_retries``
                                   for the ``"noise"`` path. (#454)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_ipeps_config_stall_recovery_retries.py -v
```

Expected: 3 passed.

**Step 5: Commit**

```bash
git add tests/test_ipeps_config_stall_recovery_retries.py src/tenax/algorithms/ipeps_config.py
git commit -m "feat(ipeps): add gs_stall_recovery_retries cap to iPEPSConfig (#454)"
```

---

## Task 3: Write the failing unit test for the cap + rollback behavior

**Files:**
- Test: `tests/test_ipeps_stall_recovery_cap.py` (new)

**Step 1: Write the test**

Create `tests/test_ipeps_stall_recovery_cap.py`. The test forces Wolfe failure on every iteration by patching the line search to return failure. The 2-site path is the production-relevant one — use it.

```python
"""Cap + rollback for gs_stall_recovery='reset' (issue #454).

The optimizer must exit cleanly after gs_stall_recovery_retries
consecutive resets and return best_params.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


@pytest.mark.core
def test_reset_loop_exits_after_retry_cap(monkeypatch, capsys):
    """Force every line search to fail; assert the loop exits at retries+1 resets."""
    # Patch hager_zhang_line_search to always return (alpha=0, f_alpha=energy, converged=False)
    import tenax.algorithms.ipeps_optimize as opt_mod

    def _always_fail(_phi, _dphi, phi0, _slope, **_kwargs):
        # f_alpha == phi0 means "no improvement" — triggers the stall_count += 1 branch.
        return 0.0, phi0, False

    monkeypatch.setattr(opt_mod, "hager_zhang_line_search", _always_fail)

    # Smallest meaningful 2-site iPEPS run: D=2, chi=4, Heisenberg, 50-step budget.
    d = 2
    gate = _heisenberg_gate(d)  # helper below
    cfg = iPEPSConfig(
        unit_cell="2site",
        ctm=CTMConfig(chi=4),
        gs_num_steps=50,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=3,
        gs_verbose=True,
        su_init=False,
    )
    A_init = _random_2site_init(d, D=2, seed=0)

    result = _opt.optimize_gs_ad(gate, A_init, cfg)

    out = capsys.readouterr().out
    # The loop should print "stall budget exhausted" once it caps out.
    assert "stall budget exhausted" in out, f"missing exhaustion log in: {out[-2000:]}"
    # Count "stall #" lines; should be exactly retries+1 (the +1 is the one that fails the cap).
    stall_lines = [ln for ln in out.splitlines() if "stall #" in ln and "reset L-BFGS" in ln]
    assert len(stall_lines) == cfg.gs_stall_recovery_retries, (
        f"expected {cfg.gs_stall_recovery_retries} reset events, got {len(stall_lines)}: {stall_lines}"
    )


def _heisenberg_gate(d):
    """S=1/2 two-site Heisenberg gate (d,d,d,d)."""
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    h = (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real
    return h


def _random_2site_init(d, D, seed):
    rng = np.random.default_rng(seed)
    A = jnp.asarray(rng.standard_normal((D, D, D, D, d)))
    B = jnp.asarray(rng.standard_normal((D, D, D, D, d)))
    return (A / jnp.linalg.norm(A), B / jnp.linalg.norm(B))
```

**Step 2: Run test — expect failure**

```bash
uv run pytest tests/test_ipeps_stall_recovery_cap.py -v
```

Expected: FAIL — either "stall budget exhausted" never printed (cap not implemented), or count mismatch, or the optimizer runs the full 50 steps without exiting.

**Step 3: Commit (test only, before implementing)**

```bash
git add tests/test_ipeps_stall_recovery_cap.py
git commit -m "test(ipeps): failing test for stall-recovery cap (#454)"
```

---

## Task 4: Fix the 2-site reset site (ipeps_optimize.py:2264) — cap + rollback

The 2-site path is the production path that surfaced the bug. Fix this first so Task 3's test passes, then mirror to the other two sites.

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:2264-2290`

**Step 1: Re-read the block in context**

```bash
sed -n '2255,2295p' src/tenax/algorithms/ipeps_optimize.py
```

Find the `elif config.gs_stall_recovery == "reset" and stall_count > 0:` block. It currently has:

```python
elif config.gs_stall_recovery == "reset" and stall_count > 0:
    # variPEPS-style reset ... Do NOT roll back params ... #298 ...
    if is_cg:
        cg_direction = None
        prev_grad = None
        prev_precond_grad = None
    if is_metric_lbfgs:
        lbfgs_history.clear()
        ...
    if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
        opt_state = optimizer.init(params)
    if config.gs_verbose:
        print(
            f"[iPEPS-AD] stall #{stall_count}, "
            f"reset L-BFGS history (no rollback)",
            flush=True,
        )
```

**Step 2: Apply the cap + rollback change**

Replace the block with:

```python
elif config.gs_stall_recovery == "reset" and stall_count > 0:
    # Rollback to best on reset (#454). #298's anti-rollback evidence
    # was on a pre-trifecta CTM stack (pre-PR #406 2x2 projector,
    # pre-multisite-CTM rewrite, pre-PR #447 AD stop_gradient) and no
    # longer applies. The CTM-error reset path already rolls back
    # (see the equivalent block above at the CTMRGGradientError catch).
    if stall_count > config.gs_stall_recovery_retries:
        if config.gs_verbose:
            print(
                f"[iPEPS-AD] stall budget exhausted after "
                f"{stall_count - 1} resets, returning best E={best_energy:.10f}",
                flush=True,
            )
        break
    params = best_params
    _env_cache.update(best_env_cache)
    if is_cg:
        cg_direction = None
        prev_grad = None
        prev_precond_grad = None
    if is_metric_lbfgs:
        lbfgs_history.clear()
        prev_A_flat = None
        prev_grad_flat = None
    if optimizer is not None and config.gs_optimizer.lower() == "lbfgs":
        opt_state = optimizer.init(params)
    if config.gs_verbose:
        print(
            f"[iPEPS-AD] stall #{stall_count}, reset L-BFGS history "
            f"(rollback to best, retry "
            f"{stall_count}/{config.gs_stall_recovery_retries})",
            flush=True,
        )
```

**Caveats / verification points while editing:**

- Confirm that `best_params`, `best_env_cache`, and `best_energy` are all in scope at this site (they are — search backward from L2264 in the 2-site function; the `best_*` bookkeeping is set up before the `for step in range(...)` loop).
- Confirm exact set of optimizer-state variables being cleared matches the surrounding block (i.e. `prev_A_flat`, `prev_grad_flat` only inside `is_metric_lbfgs`). Mirror exactly what was there.

**Step 3: Run Task 3's test**

```bash
uv run pytest tests/test_ipeps_stall_recovery_cap.py -v
```

Expected: PASS.

**Step 4: Run the broader iPEPS-AD test slice as a regression check**

```bash
uv run pytest -m core -k "ipeps_ad or ipeps_optimize" -v
```

Expected: all pass (no pre-existing tests should regress from rollback being added).

**Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "fix(ipeps): cap + rollback on reset for 2-site optimizer (#454)"
```

---

## Task 5: Mirror the fix to the 1-site C4v reset site (ipeps_optimize.py:1435)

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:1435-1460`

**Step 1: Apply the same diff pattern**

Find the block (`grep -n "no rollback" src/tenax/algorithms/ipeps_optimize.py` finds it). Apply the *same structural change* as Task 4 step 2:
- Add `if stall_count > config.gs_stall_recovery_retries: ... break` at the top.
- Add `params = best_params` and `_env_cache.update(best_env_cache)` immediately after.
- Replace the `(no rollback)` log line with the `(rollback to best, retry k/N)` form.
- Replace the `# Do NOT roll back ... #298 ...` comment block with the cross-reference comment from Task 4 step 2.

The 1-site site has fewer optimizer state vars (no `prev_A_flat` etc. on the C4v path). Mirror the exact set already being cleared in the existing block — do not invent new state to clear.

**Step 2: Write a 1-site canary test**

Add to `tests/test_ipeps_stall_recovery_cap.py`:

```python
@pytest.mark.core
def test_reset_loop_exits_after_retry_cap_c4v(monkeypatch, capsys):
    """Same as 2-site test, but 1-site C4v path."""
    import tenax.algorithms.ipeps_optimize as opt_mod

    def _always_fail(_phi, _dphi, phi0, _slope, **_kwargs):
        return 0.0, phi0, False

    monkeypatch.setattr(opt_mod, "hager_zhang_line_search", _always_fail)

    d = 2
    gate = _heisenberg_gate(d)
    cfg = iPEPSConfig(
        unit_cell="1x1",
        gs_c4v=True,
        ctm=CTMConfig(chi=4),
        gs_num_steps=50,
        gs_stall_recovery="reset",  # explicit override; default for 1x1 is "noise"
        gs_stall_recovery_retries=3,
        gs_verbose=True,
        su_init=False,
    )
    A_init = _random_1site_init(d, D=2, seed=0)

    _opt.optimize_gs_ad(gate, A_init, cfg)
    out = capsys.readouterr().out
    assert "stall budget exhausted" in out
    stall_lines = [ln for ln in out.splitlines() if "stall #" in ln and "reset L-BFGS" in ln]
    assert len(stall_lines) == cfg.gs_stall_recovery_retries


def _random_1site_init(d, D, seed):
    rng = np.random.default_rng(seed)
    A = jnp.asarray(rng.standard_normal((D, D, D, D, d)))
    return A / jnp.linalg.norm(A)
```

**Step 3: Run the tests**

```bash
uv run pytest tests/test_ipeps_stall_recovery_cap.py -v
```

Expected: 2 passed.

**Step 4: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps_stall_recovery_cap.py
git commit -m "fix(ipeps): cap + rollback on reset for 1-site C4v optimizer (#454)"
```

---

## Task 6: Mirror the fix to the multisite reset site (ipeps_optimize.py:2848)

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:2848-2865`

**Step 1: Apply the same structural change**

Same as Task 5, applied to the multisite block. `grep -n "no rollback" src/tenax/algorithms/ipeps_optimize.py` should now show only one remaining occurrence; fix it.

**Caveats:**
- The multisite optimizer uses `params` as a `dict[str, Tensor]` (per the Lattice unit cell), so `best_params` is also a dict. Mirror this in the assignment — it's just `params = best_params`, no special handling.
- `_env_cache` is the same closure-captured dict pattern; `best_env_cache` exists.

**Step 2: Add a multisite canary test (optional, marked slow)**

If a 2x2 multisite test is fast enough to be in `core`, add it analogous to Task 5's test. If not, mark `slow`:

```python
@pytest.mark.slow
def test_reset_loop_exits_after_retry_cap_multisite(monkeypatch, capsys):
    """Multisite (2x2 Lattice) path — same cap behavior."""
    # ... mirror previous tests with Lattice unit cell ...
```

Skip this test if the multisite path is heavy enough to need real CTM convergence to evaluate — the unit cap behavior is the same code structure; the 1-site and 2-site tests give sufficient coverage.

**Step 3: Verify all three remaining `(no rollback)` occurrences are gone**

```bash
grep -n "no rollback" src/tenax/algorithms/ipeps_optimize.py
```

Expected: no output (no matches).

**Step 4: Run regression slice**

```bash
uv run pytest -m core -k "ipeps" -v
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps_stall_recovery_cap.py
git commit -m "fix(ipeps): cap + rollback on reset for multisite optimizer (#454)"
```

---

## Task 7: Production canary — 20-step Heisenberg D=2 χ=8 stall count

**Files:**
- Test: `tests/test_ipeps_stall_runaway_canary.py` (new, `slow` marker)

**Step 1: Write the canary**

```python
"""Production canary for #454: a real 20-step Heisenberg D=2 χ=8 2-site run
should now see ≤ 3 stalls (was unbounded before the fix)."""
import re

import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


@pytest.mark.slow
def test_heisenberg_d2_chi8_stall_count_under_cap(capsys):
    # Mirror examples/heisenberg_ipeps_ad_2x2.py at small scale.
    import jax.numpy as jnp
    import numpy as np

    d = 2
    sx = 0.5 * jnp.array([[0, 1], [1, 0]])
    sy = 0.5 * jnp.array([[0, -1j], [1j, 0]])
    sz = 0.5 * jnp.array([[1, 0], [0, -1]])
    gate = (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real

    rng = np.random.default_rng(42)
    A = jnp.asarray(rng.standard_normal((2, 2, 2, 2, d)))
    B = jnp.asarray(rng.standard_normal((2, 2, 2, 2, d)))

    cfg = iPEPSConfig(
        unit_cell="2site",
        ctm=CTMConfig(chi=8),
        gs_num_steps=20,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=5,
        gs_verbose=True,
        su_init=False,
    )

    _opt.optimize_gs_ad(gate, (A, B), cfg)
    out = capsys.readouterr().out
    stalls = re.findall(r"stall #(\d+)", out)
    n_stalls = max((int(s) for s in stalls), default=0)
    assert n_stalls <= 3, (
        f"expected ≤ 3 stalls on D=2 χ=8 canary, got {n_stalls}; "
        "regression in stall recovery"
    )
```

**Step 2: Run**

```bash
uv run pytest tests/test_ipeps_stall_runaway_canary.py -v
```

Expected: PASS. If FAIL, the cap is firing on a real run — investigate whether the rollback is structurally correct or whether the production landscape needs the b-flex (forced steepest-descent step) variant. Don't merge without a passing canary.

**Step 3: Commit**

```bash
git add tests/test_ipeps_stall_runaway_canary.py
git commit -m "test(ipeps): production canary for #454 stall runaway"
```

---

## Task 8: Open the PR

**Step 1: Push and create PR**

```bash
git push -u origin fix/ipeps-stall-runaway
gh pr create --title "fix(ipeps): cap + rollback L-BFGS reset to stop stall runaway (#454)" --body "$(cat <<'EOF'
## Summary

- Add `gs_stall_recovery_retries: int = 5` to `iPEPSConfig`; cap the
  `gs_stall_recovery="reset"` path at all three optimizer sites.
- On reset, roll `params` back to `best_params` and refresh `_env_cache`
  from `best_env_cache` before clearing L-BFGS / CG state. Breaks the
  mathematical fixed point that produced 18+ consecutive resets in the
  2026-05-13 benchmark.
- When the cap is exceeded, log `stall budget exhausted` and break out
  of the optimizer loop; final `_eval_fresh` returns `best_params`.

Closes #454.

## Why rollback now (re: #298)

#298's anti-rollback evidence was on a pre-trifecta CTM stack
(pre-PR #406 2x2 projector, pre-multisite-CTM rewrite, pre-PR #447 AD
stop_gradient). The CTM-error reset path at `ipeps_optimize.py:1207`
already does this rollback today; this PR extends the same pattern to
the Wolfe-failure reset path that was missed.

## Test plan

- [x] Unit: synthetic Wolfe-failing line search exits cleanly at
  `retries + 1` resets on both 2-site and 1-site C4v paths.
- [x] Production canary (slow): D=2 χ=8 Heisenberg 20-step 2-site run
  records ≤ 3 stalls.
- [ ] Full suite (`run-full-tests` label) — no regressions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 2: Enable auto-merge per CLAUDE.md**

```bash
gh pr merge --squash --delete-branch --auto
```

Pre-conditions: CI must pass; required checks per CLAUDE.md are `Tests (Python 3.11)`, `Tests (Python 3.12)`, `Tests (macOS, Python 3.12)`. The merge auto-fires when they go green.

**Step 3: Done**

Return the PR URL to the user.

---

## Notes for the executing engineer

- **Don't bundle the #453 fix here.** That's a separate branch (`perf/chi-ramp-pad-env`). See `feedback_separate_branches_per_concern`.
- **Don't add Wolfe-condition diagnostic logging or forced-steepest-descent-step variants.** Those are out of scope per the design doc; deferred.
- **If the production canary fails**, the rollback may not be sufficient on the current optimizer trajectory at D=2 χ=8. Investigate with `gs_verbose=True` + dumping `||grad||`, `phi(0)`, `dphi(0)`, `phi(alpha)`, `dphi(alpha)` on the first stall before deciding whether to escalate to the b-flex variant (forced fixed-α steepest-descent step after reset). Open an issue, do not merge with a failing canary.
