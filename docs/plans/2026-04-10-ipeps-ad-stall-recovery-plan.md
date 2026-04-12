# iPEPS AD Stall-Recovery Fix (Issue #298) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace noise-injection stall recovery in the 2-site iPEPS AD path with a variPEPS-style L-BFGS reset + rollback so 2-site AD converges without the pathological interaction with non-variational CTM energy regions, while leaving the 1-site C4v production path bit-identical by default.

**Architecture:** Add two new `iPEPSConfig` fields (`gs_stall_recovery`, `gs_energy_floor`). Split the stall-recovery body in `_optimize_gs_ad_tensor` and `_optimize_gs_ad_tensor_2site` into two branches: the existing noise-injection body (now gated on `"noise"`) and a new reset body (`"reset"`) that clears L-BFGS history, rolls back `params ← best_params`, and forces steepest descent on the next step. Add an opt-in `gs_energy_floor` check on the in-loop best-state update. Auto-default the new field to `"noise"` for 1-site and `"reset"` for 2-site at dispatch so both bug-fix and zero-regression happen without user action.

**Tech Stack:** JAX, Python 3.11+, dataclasses, pytest. Targets `src/tenax/algorithms/ipeps_optimize.py` and `src/tenax/algorithms/ipeps_config.py`.

**Design doc:** `docs/plans/2026-04-10-ipeps-ad-stall-recovery-design.md`

**Issue:** https://github.com/ydkao/tenax/issues/298 (Closes)

---

## Preflight

Ensure pre-commit hooks installed and a clean workspace:

```bash
cd /home/yjkao/tenax
pre-commit install
git status  # expect main at or near 5f95ee0, design doc at 21f29f7 already committed
uv run pytest -m core -x -q --timeout=300 2>&1 | tail -20
```

Create a feature branch:

```bash
git checkout -b fix/298-ipeps-ad-stall-recovery
```

---

## Task 1: Add `gs_stall_recovery` and `gs_energy_floor` fields to `iPEPSConfig`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py` (around line 126, near existing `gs_noise_recovery_retries`)

**Step 1: Write the failing test**

Create `tests/test_ipeps_config.py` if it doesn't exist, else add to it.

```python
# tests/test_ipeps_config.py
def test_stall_recovery_default_is_none():
    from tenax.algorithms.ipeps_config import iPEPSConfig
    cfg = iPEPSConfig()
    assert cfg.gs_stall_recovery is None
    assert cfg.gs_energy_floor is None


def test_stall_recovery_accepts_noise_and_reset():
    from tenax.algorithms.ipeps_config import iPEPSConfig
    cfg_n = iPEPSConfig(gs_stall_recovery="noise")
    cfg_r = iPEPSConfig(gs_stall_recovery="reset")
    assert cfg_n.gs_stall_recovery == "noise"
    assert cfg_r.gs_stall_recovery == "reset"


def test_energy_floor_stores_float():
    from tenax.algorithms.ipeps_config import iPEPSConfig
    cfg = iPEPSConfig(gs_energy_floor=-1.5)
    assert cfg.gs_energy_floor == -1.5
```

**Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_ipeps_config.py -v
```

Expected: FAIL with `AttributeError: 'iPEPSConfig' object has no attribute 'gs_stall_recovery'` (or `TypeError` on the constructor).

**Step 3: Add the fields**

In `src/tenax/algorithms/ipeps_config.py`, add the import at the top if not present:

```python
from typing import Literal, NamedTuple
```

In the `iPEPSConfig` dataclass, after the existing `gs_noise_amplitude: float = 0.1` line, add:

```python
    # Stall recovery mode for L-BFGS / CG line search failures.
    #   "noise"  -> inject gs_noise_amplitude Frobenius perturbation (legacy,
    #               required for 1-site C4v production path to break out of the
    #               SU-init plateau at step 0).
    #   "reset"  -> clear L-BFGS (s, y) history, roll back params to best_params,
    #               force steepest descent on next step.  Matches variPEPS.
    #   None     -> auto-default per dispatcher: "noise" for 1-site, "reset" for
    #               2-site.  Set by optimize_gs_ad at entry.
    gs_stall_recovery: Literal["noise", "reset"] | None = None
    # Optional variational sanity floor on in-loop best-state tracking.  Any
    # candidate energy strictly below this value is rejected as a non-
    # variational CTM artifact (see issue #298).  None disables the check.
    gs_energy_floor: float | None = None
```

**Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_ipeps_config.py -v
```

Expected: 3 passed.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_config.py tests/test_ipeps_config.py
git commit -m "feat(ipeps): add gs_stall_recovery and gs_energy_floor config fields (#298)"
```

---

## Task 2: Auto-default `gs_stall_recovery` at the 1-site and 2-site dispatchers

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` — top of `_optimize_gs_ad_tensor` (~line 400 area, right after the function signature) and top of `_optimize_gs_ad_tensor_2site` (~line 900 area)
- Test: `tests/test_ipeps.py`

**Step 1: Write the failing test**

Append to `tests/test_ipeps.py`:

```python
def test_stall_recovery_auto_defaults():
    """1-site -> 'noise', 2-site -> 'reset' when user leaves gs_stall_recovery=None."""
    import jax.numpy as jnp
    from dataclasses import replace
    from tenax.algorithms.ipeps_config import iPEPSConfig, CTMConfig
    from tenax.algorithms import ipeps_optimize as io

    seen = {}

    original_1s = io._optimize_gs_ad_tensor_impl  # we will monkeypatch below
    # See _dispatch_capture_config in Task 2 notes below.
    # For the test, just check the normalization helper:
    cfg_in_1s = iPEPSConfig(unit_cell="1x1")
    cfg_out_1s = io._normalize_stall_recovery(cfg_in_1s, unit_cell="1x1")
    assert cfg_out_1s.gs_stall_recovery == "noise"

    cfg_in_2s = iPEPSConfig(unit_cell="2site")
    cfg_out_2s = io._normalize_stall_recovery(cfg_in_2s, unit_cell="2site")
    assert cfg_out_2s.gs_stall_recovery == "reset"

    cfg_explicit = iPEPSConfig(unit_cell="2site", gs_stall_recovery="noise")
    cfg_out_exp = io._normalize_stall_recovery(cfg_explicit, unit_cell="2site")
    assert cfg_out_exp.gs_stall_recovery == "noise", "explicit user setting must win"
```

(Delete the `original_1s` line and the monkeypatch preamble before saving — only the helper calls matter. The final test body is just the helper assertions.)

Actually, replace the test body with just:

```python
def test_stall_recovery_auto_defaults():
    """1-site -> 'noise', 2-site -> 'reset' when user leaves gs_stall_recovery=None."""
    from tenax.algorithms.ipeps_config import iPEPSConfig
    from tenax.algorithms.ipeps_optimize import _normalize_stall_recovery

    cfg_1s = _normalize_stall_recovery(iPEPSConfig(unit_cell="1x1"), unit_cell="1x1")
    assert cfg_1s.gs_stall_recovery == "noise"

    cfg_2s = _normalize_stall_recovery(iPEPSConfig(unit_cell="2site"), unit_cell="2site")
    assert cfg_2s.gs_stall_recovery == "reset"

    cfg_user = _normalize_stall_recovery(
        iPEPSConfig(unit_cell="2site", gs_stall_recovery="noise"), unit_cell="2site"
    )
    assert cfg_user.gs_stall_recovery == "noise", "explicit user setting must win"
```

**Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_ipeps.py::test_stall_recovery_auto_defaults -v
```

Expected: FAIL — `ImportError: cannot import name '_normalize_stall_recovery'`.

**Step 3: Add the helper and call it from both dispatchers**

In `src/tenax/algorithms/ipeps_optimize.py`, near the top of the file (after imports, before the first private helper), add:

```python
def _normalize_stall_recovery(config, *, unit_cell: str):
    """Auto-default gs_stall_recovery based on unit cell when unset.

    1-site C4v production path requires the noise kick to break out of the
    SU-init plateau (gradient ~1e-10 trips gs_conv_tol).  The 2-site path's
    larger parameter space interacts pathologically with non-variational CTM
    regions under noise; see issue #298.
    """
    from dataclasses import replace

    if config.gs_stall_recovery is not None:
        return config
    default = "noise" if unit_cell == "1x1" else "reset"
    return replace(config, gs_stall_recovery=default)
```

Then at the top of `_optimize_gs_ad_tensor` (after the opening docstring, before any use of `config`), add:

```python
    config = _normalize_stall_recovery(config, unit_cell="1x1")
```

And at the top of `_optimize_gs_ad_tensor_2site`, add:

```python
    config = _normalize_stall_recovery(config, unit_cell="2site")
```

Use Grep to find exact insertion lines:
```
Grep for "def _optimize_gs_ad_tensor(" and "def _optimize_gs_ad_tensor_2site(" in ipeps_optimize.py.
```

**Step 4: Run the test**

```bash
uv run pytest tests/test_ipeps.py::test_stall_recovery_auto_defaults -v
```

Expected: PASS.

**Step 5: Run the broader iPEPS smoke tests**

```bash
uv run pytest tests/test_ipeps.py -m core -x -q 2>&1 | tail -20
```

Expected: no regressions from this patch (only the new test plus all prior tests).

**Step 6: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps.py
git commit -m "feat(ipeps): auto-default gs_stall_recovery per unit cell (#298)"
```

---

## Task 3: Gate the existing noise-injection code on `gs_stall_recovery == "noise"`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` lines ~691–719 (1-site) and ~1225–1248 (2-site)

**Step 1: Read the current noise-injection blocks**

```
Read ipeps_optimize.py lines 685-720 and 1220-1250 to confirm current structure.
```

**Step 2: Gate each block**

**1-site block (~line 692):** Change

```python
            # Noise recovery on persistent stall
            if stall_count > 0 and stall_count <= config.gs_noise_recovery_retries:
```

to

```python
            # Noise recovery on persistent stall (legacy; see issue #298).
            if (
                config.gs_stall_recovery == "noise"
                and stall_count > 0
                and stall_count <= config.gs_noise_recovery_retries
            ):
```

**2-site block (~line 1226):** same textual change.

**Step 3: Run the 1-site C4v production test (should still pass — default is "noise")**

```bash
uv run pytest tests/test_ipeps.py -k "c4v or C4v or 1site" -x -q 2>&1 | tail -20
```

Expected: the same tests that passed before still pass (default for 1-site is `"noise"`, so behavior is unchanged).

**Step 4: Verify the 2-site path now has no stall-recovery effect at all (temporarily broken)**

```bash
uv run pytest tests/test_ipeps.py -k "2site and ad" -x -q 2>&1 | tail -20
```

Expected: 2-site AD tests may still pass because they don't trigger the stall branch, but if any 2-site test expected the noise kick, it will regress. Note: we are about to add the reset branch in Task 4, so do not commit partial state until Task 4 lands. The gate itself is the setup for Task 4.

**Step 5: Commit (gate only, without reset branch yet — acceptable because tests still pass)**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "refactor(ipeps): gate noise injection on gs_stall_recovery=='noise' (#298)"
```

---

## Task 4: Add the `"reset"` stall-recovery branch in both dispatchers

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` — add a new block immediately after each `if config.gs_stall_recovery == "noise" ...` block (1-site ~line 720, 2-site ~line 1249)

**Step 1: Write the failing regression test**

Append to `tests/test_ipeps.py`:

```python
def test_2site_ad_stall_reset_converges_heisenberg_d2():
    """Issue #298 acceptance test: 2-site AD, D=2, chi=8, SU init, L-BFGS +
    Hager-Zhang + metric precond, 20 steps.  With the reset stall-recovery
    default, the optimizer should reach within 0.01 of the literature
    Heisenberg AFM energy (-0.6548) and never record a spurious below-floor
    energy in best_params."""
    import jax.numpy as jnp
    from tenax.algorithms.ipeps_config import iPEPSConfig, CTMConfig
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad
    from tenax.models.spin_models import heisenberg_gate

    d = 2
    gate = heisenberg_gate(J=1.0)
    cfg = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=50,
        dt=0.05,
        unit_cell="2site",
        su_init=True,
        gs_num_steps=20,
        gs_optimizer="lbfgs",
        gs_line_search=True,
        gs_line_search_method="hager_zhang",
        gs_metric_precond=True,
        gs_energy_floor=-2.0,  # well below literature; catches non-variational noise
        gs_verbose=False,
        ctm=CTMConfig(chi=8, max_iter=100, conv_tol=1e-8),
    )
    result = optimize_gs_ad(gate, d, cfg)
    E_final = float(result.energy)
    assert E_final < -0.60, f"2-site AD failed to converge: E={E_final}"
    assert E_final > -1.0, f"2-site AD dipped into non-variational region: E={E_final}"
```

Mark it as a slow test by putting it in a `test_ipeps_*` filename or adding a `@pytest.mark.slow` marker (check the file's existing conventions — auto-marking by filename via `conftest.py` means an existing `test_ipeps.py` is marked `core`; if this test is too slow for CI `-m core`, move it to `tests/test_ipeps_slow.py` or add an explicit `@pytest.mark.slow` decorator).

Inspect the existing marker convention:

```
Grep "pytest.mark" in tests/test_ipeps.py and the test filenames under tests/ for _slow suffix.
```

If there is a `tests/test_ipeps_slow.py`, add the test there; otherwise add `@pytest.mark.slow` and import pytest at the top of `tests/test_ipeps.py` if missing.

**Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_ipeps.py::test_2site_ad_stall_reset_converges_heisenberg_d2 -v --timeout=600
```

Expected: FAIL. It should fail in one of two ways:
- The test function can't find `optimize_gs_ad` or `heisenberg_gate`. Fix the imports to match the actual module paths (grep for `def optimize_gs_ad` and `heisenberg` in `src/tenax/`).
- The test runs but fails the `E_final < -0.60` assertion because, without the reset branch, the 2-site path currently has no stall recovery at all after Task 3's gate.

Record the actual final E and failure mode before continuing.

**Step 3: Add the reset branch (1-site)**

After the 1-site noise block (after line ~719), add:

```python
            elif config.gs_stall_recovery == "reset" and stall_count > 0:
                # variPEPS-style reset: clear L-BFGS history, roll back to best,
                # force steepest descent next iteration.  See issue #298.
                params = best_params
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_A_flat = None
                    prev_grad_flat = None
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD] stall #{stall_count}, "
                        f"reset L-BFGS -> steepest descent from best",
                        flush=True,
                    )
```

**Step 4: Add the reset branch (2-site)**

After the 2-site noise block (after line ~1248), add:

```python
            elif config.gs_stall_recovery == "reset" and stall_count > 0:
                # variPEPS-style reset; see issue #298.
                params = best_params
                if is_cg:
                    cg_direction = None
                    prev_grad = None
                    prev_precond_grad = None
                if is_metric_lbfgs:
                    lbfgs_history.clear()
                    prev_params_flat = None
                    prev_grad_flat = None
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD] stall #{stall_count}, "
                        f"reset L-BFGS -> steepest descent from best",
                        flush=True,
                    )
```

Note: the 2-site path uses `prev_params_flat` (not `prev_A_flat`). Confirm by grepping for `prev_params_flat` in the 2-site function before writing. If the variable name differs, adjust.

**Step 5: Run the acceptance test**

```bash
uv run pytest tests/test_ipeps.py::test_2site_ad_stall_reset_converges_heisenberg_d2 -v --timeout=600
```

Expected: PASS, with `E_final` in roughly the -0.64 to -0.66 range.

If the test still fails with `E_final > -0.60`:
1. Print the full per-step trajectory by setting `gs_verbose=True` temporarily.
2. Check whether the reset branch is actually firing — look for `"reset L-BFGS"` log lines.
3. If the branch fires but L-BFGS still stalls, it means `best_params` itself is in a bad region. Verify Task 5 (the energy floor) is landing next and re-run.

**Step 6: Run the full iPEPS test suite**

```bash
uv run pytest tests/test_ipeps.py -x -q --timeout=600 2>&1 | tail -30
```

Expected: no regressions (the 1-site C4v default is `"noise"`, unchanged).

**Step 7: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps.py
git commit -m "feat(ipeps): variPEPS-style reset stall recovery for 2-site AD (#298)"
```

---

## Task 5: Apply the `gs_energy_floor` check to in-loop best-state tracking

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` — 1-site best update at ~line 508, 2-site best update at ~line 1020 (confirm with Grep for `best_energy = energy_float`)
- Test: `tests/test_ipeps.py`

**Step 1: Write the failing unit test**

Append to `tests/test_ipeps.py`:

```python
def test_energy_floor_rejects_spurious_best_state(monkeypatch):
    """When gs_energy_floor is set, an energy below the floor must not
    overwrite best_params/best_energy."""
    import jax.numpy as jnp
    from tenax.algorithms.ipeps_optimize import _should_accept_best

    # Pure helper unit test — no simulation needed.
    assert _should_accept_best(current_best=0.0, candidate=-0.5, floor=None) is True
    assert _should_accept_best(current_best=0.0, candidate=-0.5, floor=-1.0) is True
    assert _should_accept_best(current_best=0.0, candidate=-5.0, floor=-1.0) is False
    assert _should_accept_best(current_best=-0.5, candidate=-0.6, floor=-1.0) is True
    # Candidate strictly below floor -> reject, even if it beats current.
    assert _should_accept_best(current_best=-0.5, candidate=-2.0, floor=-1.0) is False
    # Candidate equal to floor -> reject (strictly above required).
    assert _should_accept_best(current_best=0.0, candidate=-1.0, floor=-1.0) is False
```

**Step 2: Run it to verify failure**

```bash
uv run pytest tests/test_ipeps.py::test_energy_floor_rejects_spurious_best_state -v
```

Expected: FAIL with `ImportError: cannot import name '_should_accept_best'`.

**Step 3: Add the helper**

In `src/tenax/algorithms/ipeps_optimize.py`, next to `_normalize_stall_recovery`:

```python
def _should_accept_best(*, current_best: float, candidate: float, floor: float | None) -> bool:
    """Return True iff ``candidate`` should overwrite ``best_energy``.

    Rejects candidates whose energy is strictly below ``floor`` (treated as a
    non-variational CTM artifact per issue #298).  A ``None`` floor disables
    the check.
    """
    if candidate >= current_best:
        return False
    if floor is not None and candidate <= floor:
        return False
    return True
```

**Step 4: Wire it into both best-state updates**

**1-site (~line 508):**

Change:

```python
        if energy_float < best_energy:
            best_energy = energy_float
            best_params = params
```

to:

```python
        if _should_accept_best(
            current_best=best_energy,
            candidate=energy_float,
            floor=config.gs_energy_floor,
        ):
            best_energy = energy_float
            best_params = params
```

**2-site (~line 1020):** same replacement.

**Step 5: Run the helper test**

```bash
uv run pytest tests/test_ipeps.py::test_energy_floor_rejects_spurious_best_state -v
```

Expected: PASS.

**Step 6: Re-run the full iPEPS suite**

```bash
uv run pytest tests/test_ipeps.py -x -q --timeout=600 2>&1 | tail -30
```

Expected: green.

**Step 7: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps.py
git commit -m "feat(ipeps): reject non-variational candidates via gs_energy_floor (#298)"
```

---

## Task 6: Verify descent-direction sanity check already exists (or add it)

**Context:** The design calls for a `<d, g> > 0 → reset to -g` guard before the line search. Inspection of the code shows this guard already exists inside the Hager-Zhang branch (1-site ~line 637, 2-site analogous) but **does not clear L-BFGS history** when it fires. The backtracking branch has no guard at all.

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` — 1-site Hager-Zhang branch ~line 636 and 2-site analogue; 1-site backtracking branch ~line 678 and 2-site analogue.

**Step 1: Grep for existing guards**

```
Grep for "slope >= 0" in src/tenax/algorithms/ipeps_optimize.py and note every line + its function context.
```

**Step 2: Augment each existing guard to clear L-BFGS history**

For each `if slope >= 0:` block, add history-clearing alongside the `direction = -grads` line:

```python
                if slope >= 0:
                    direction = jax.tree.map(lambda g: -g, grads)
                    slope = -_tree_dot(grads, grads)
                    if is_metric_lbfgs:
                        lbfgs_history.clear()
```

**Step 3: Add a guard to the backtracking branch**

In the backtracking else-branch (~line 677, 1-site; ~line 1211, 2-site), immediately before the call to `_backtracking_line_search`, add:

```python
                slope_bt = _tree_dot(grads, direction)
                if slope_bt >= 0:
                    direction = jax.tree.map(lambda g: -g, grads)
                    if is_metric_lbfgs:
                        lbfgs_history.clear()
```

**Step 4: Run the full iPEPS suite**

```bash
uv run pytest tests/test_ipeps.py -x -q --timeout=600 2>&1 | tail -30
```

Expected: green.  The sanity check is benign — it only fires when L-BFGS was about to produce an ascent direction.

**Step 5: Re-run the Task 4 acceptance test to confirm 2-site convergence still holds**

```bash
uv run pytest tests/test_ipeps.py::test_2site_ad_stall_reset_converges_heisenberg_d2 -v --timeout=600
```

Expected: PASS, ideally with a slightly better final energy than Task 4 alone.

**Step 6: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "fix(ipeps): clear L-BFGS history when ascent direction detected (#298)"
```

---

## Task 7: Update existing tests to assert the new default is used on the 2-site path

**Files:**
- Modify: any existing 2-site AD tests in `tests/test_ipeps.py` that may depend on noise injection for convergence. Expected count: 0–2. Use Grep to find them.

**Step 1: Locate 2-site AD tests**

```
Grep for "2site" and "unit_cell" under tests/ and list any that exercise optimize_gs_ad.
```

**Step 2: Run each suspect test in isolation**

```bash
uv run pytest tests/test_ipeps.py -k "2site" -x -v --timeout=600 2>&1 | tail -40
```

If all pass unchanged, proceed to Task 8. If any fail:

1. Check whether the failure is from the new default (`"reset"` instead of `"noise"`).
2. If the test was relying on noise injection for convergence, update its config to `gs_stall_recovery="noise"` explicitly and leave a `# pre-issue-298 behavior` comment, **or** accept that the new behavior is better and update the expected energy bound.

**Step 3: Commit any test updates**

```bash
git add tests/
git commit -m "test(ipeps): adjust 2-site AD tests for new reset default (#298)"
```

(Skip if there was nothing to update.)

---

## Task 8: Full test suite + core smoke

**Step 1: Core suite**

```bash
uv run pytest -m core -x -q --timeout=600 2>&1 | tail -30
```

Expected: all green.

**Step 2: Non-slow suite**

```bash
uv run pytest -m "not slow" -x -q --timeout=900 2>&1 | tail -30
```

Expected: all green.

**Step 3: Commit any incidental fixes**

Only if the run exposed something; do not amend prior commits.

---

## Task 9: Docs update

**Files:**
- Modify: `docs/guide/algorithms/ipeps_ad_paths.md` (if it exists; check first)
- Modify: `src/tenax/__init__.py` only if the new config fields need to be re-exported (they don't — `iPEPSConfig` is already exported)

**Step 1: Check for the relevant doc**

```
Glob docs/guide/algorithms/ipeps_ad_paths.md and related docs under docs/guide/algorithms/.
```

**Step 2: Append a short section**

In `docs/guide/algorithms/ipeps_ad_paths.md` (or the nearest existing iPEPS AD doc), add a subsection:

```markdown
## Stall recovery (`gs_stall_recovery`)

When the L-BFGS / CG line search fails to find a descent step, the
optimizer runs a stall-recovery routine.  Two modes are supported:

- ``"noise"`` — inject a ``gs_noise_amplitude`` (default 10 %) Frobenius
  perturbation on the current params and reset the L-BFGS history.
  **Required for the 1-site C4v production path**, which sits on an
  SU-init plateau with gradient norms around ``1e-10`` that would
  otherwise trip ``gs_conv_tol`` before the first real step.
- ``"reset"`` — clear the L-BFGS ``(s, y)`` history, roll back the
  iterate to the best seen so far, and take the next step along
  steepest descent.  **Default for the 2-site path** because the
  10 % noise kick in the 32-dim D=2 parameter space lands in
  non-variational CTM regions (see issue #298).

Leaving ``gs_stall_recovery=None`` (the default) auto-selects the right
mode for the unit cell.

For extra safety on 2-site runs, set ``gs_energy_floor`` to a value a
bit below the expected variational minimum (e.g. ``2 * E_literature``)
to reject spurious below-floor "best" energies that can arise from the
``_rdm2x1_tensor_2site`` trace-normalization at near-zero trace.
```

**Step 3: Commit**

```bash
git add docs/
git commit -m "docs(ipeps): document gs_stall_recovery and gs_energy_floor (#298)"
```

---

## Task 10: Push, open PR, attach to issue #298

**Step 1: Push**

```bash
git push -u origin fix/298-ipeps-ad-stall-recovery
```

**Step 2: Open the PR**

```bash
gh pr create --title "fix(ipeps): variPEPS-style stall recovery for 2-site AD (#298)" --body "$(cat <<'EOF'
## Summary

Resolves #298 (scope: items 1–4 of the proposed principled fix; item 5
deferred as a separate follow-up).

- Add ``gs_stall_recovery: Literal["noise","reset"]|None`` and
  ``gs_energy_floor: float|None`` to ``iPEPSConfig``.
- Auto-default ``gs_stall_recovery`` to ``"noise"`` for 1-site
  (unchanged production behavior) and ``"reset"`` for 2-site
  (fixes the bug).
- New ``"reset"`` branch: clear L-BFGS history, roll back to
  ``best_params``, force steepest descent next step.  Matches
  variPEPS.
- Optional ``gs_energy_floor`` rejects below-floor candidates in
  in-loop best-state tracking (catches the non-variational CTM
  artifact from ``_rdm2x1_tensor_2site`` at near-zero RDM trace).
- Descent-direction sanity check now also clears L-BFGS history
  when it fires.

## Why

Trajectory study in #298 showed that the default 10 % Frobenius noise
kick interacts pathologically with non-variational CTM regions in the
32-dim D=2 parameter space.  Disabling noise globally would break the
1-site C4v production path, which needs it to escape the SU-init
plateau.  This PR keeps both paths working with their respective
recovery strategies.

## Test plan

- [x] ``test_stall_recovery_auto_defaults`` (new) — helper dispatch
      behavior
- [x] ``test_energy_floor_rejects_spurious_best_state`` (new) — helper
      unit test
- [x] ``test_2site_ad_stall_reset_converges_heisenberg_d2`` (new) —
      end-to-end convergence assertion at D=2 χ=8
- [x] 1-site C4v production tests unchanged
- [x] ``uv run pytest -m core``
- [x] ``uv run pytest -m "not slow"``

## Out of scope (follow-ups)

- Item 5 of #298: migrate 1-site C4v off noise injection.
- Hardening ``_rdm2x1_tensor_2site`` / ``_rdm1x2_tensor_2site``
  against near-zero-trace RDMs.
- Optimizer shell polymorphism (#297).

Design doc: ``docs/plans/2026-04-10-ipeps-ad-stall-recovery-design.md``
EOF
)"
```

**Step 3: Enable auto-merge with squash**

```bash
# Replace <num> with the PR number printed by the previous command.
gh pr merge <num> --squash --delete-branch --auto
```

**Step 4: Report PR URL to the user and stop.**

Do not mark #298 closed manually — the `Closes #298` in the PR body will handle it when the PR merges.

---

## Rollback instructions (if something goes badly wrong)

If the 2-site acceptance test cannot be made to pass even after Tasks 4–6:

1. Record the failure mode (final E, whether reset branch fires, whether the energy floor traps anything) in a comment on #298.
2. Revert Tasks 2–6 with `git revert <commit-sha>` in reverse chronological order — **do not** force-push.
3. Keep Task 1 (the config fields) on a separate commit/PR as dead code plus docstring noting it's reserved.
4. Re-open brainstorming with the trajectory data.
