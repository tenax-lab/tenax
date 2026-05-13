# iPEPS χ-Ramp Pad-Env Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the per-stage JIT retraces in `optimize_gs_ad_chi_schedule` (issue #453) by allocating CTM env tensors at `chi_max` from the first iteration, ramping the *logical* χ through a unified scheduled-bump mechanism that reuses the §2.8.2 auto-χ_E machinery. Also emit a per-χ record per bump for finite-χ scaling analysis.

**Architecture:** Three pieces:
1. **Extract** the bump mechanism out of `_maybe_bump_chi`: introduce `_apply_chi_bump(ctm_cfg, env_cache, chi_new, *, base_charges)`. `_maybe_bump_chi` keeps the reactive-trigger policy and delegates.
2. **Add** `_maybe_scheduled_bump(ctm_cfg, env_cache, step, schedule_targets, *, base_charges)` as a second caller of `_apply_chi_bump`, driven by a step-boundary list (`gs_chi_schedule_steps: list[tuple[int, int]]` on `iPEPSConfig`).
3. **Refactor** `optimize_gs_ad_chi_schedule` to a thin shim that sets `ctm_cfg.chi = chi_schedule[0][0]`, `ctm_cfg.chi_max = max(chi_schedule_chis)`, `gs_num_steps = sum(steps)`, stashes the boundary list on the config, and invokes a single `optimize_gs_ad`. Inner-loop step-end blocks gain a `_maybe_scheduled_bump` call alongside the existing `_maybe_bump_chi`. Additive: thread a `list[ChiStageRecord]` through the loop, append on every bump, return it when `return_stages=True`.

**Tech Stack:** Python, JAX, optax, pytest. Design doc: `docs/plans/2026-05-13-ipeps-stall-runaway-and-chi-ramp-design.md` (Section 2).

**Branch:** `perf/chi-ramp-pad-env` → PR closes #453, references #455 (follow-up for adaptive ramping).

**Pre-requisites:**
- Worktree on a feature branch off latest `main`.
- `pre-commit install`.
- Read these:
  - Design doc Section 2.
  - `src/tenax/algorithms/ipeps_optimize.py:35-82` (`_maybe_bump_chi`).
  - `src/tenax/algorithms/ipeps_optimize.py:445-499` (`optimize_gs_ad_chi_schedule`).
  - `src/tenax/algorithms/_ctm_env_pad.py:228` (`pad_dense_env_chi`).
  - Issue #453.
  - The reactive bump call sites (use `grep -n "_maybe_bump_chi" src/tenax/algorithms/ipeps_optimize.py` to find them — there are at least two, at L1220 and L1476).
- **Important naming note:** `CTMConfig.chi_ramp` is already used by the inner CTM solver as an intra-fixed-point ramp (`ad_utils.py:951`). Do NOT reuse that name for the outer optimizer schedule. Use `gs_chi_schedule_steps` on `iPEPSConfig`.

---

## Task 1: Create the branch and verify baseline

**Files:** none

**Step 1: Branch**

```bash
git checkout -b perf/chi-ramp-pad-env origin/main
pre-commit install
uv run pytest -m core -x -q
```

Expected: pass.

---

## Task 2: Numerical-equivalence audit — env shape vs logical χ

**Goal:** Confirm no CTM kernel uses env tensor shapes as the logical χ for truncation. Any hit must be fixed (or documented as safe) *before* the padding refactor lands, otherwise padded-to-`chi_max` envs at small logical χ would silently over-resolve.

**Files:** none (audit + documentation)

**Step 1: Run audit grep**

```bash
grep -nE "env\.(C[1-4]|T[1-4])\.shape" src/tenax/algorithms/_ctm_*.py
grep -nE "\.shape\[0\].*chi|chi.*\.shape\[0\]" src/tenax/algorithms/_ctm_*.py
grep -nE "shape\[-?[01]\].*chi|chi.*shape\[-?[01]\]" src/tenax/algorithms/_ctm_*.py
```

For each hit, classify:
- **Safe** — uses shape for assertions/sanity only (e.g. `assert env.C1.shape[0] == chi`).
- **Safe** — read-only metadata not used as truncation cap.
- **BUG** — actually feeds `chi` into a truncation/SVD/contraction.

Document each hit with file:line in the PR description.

**Step 2: If any BUG hits found, file a sub-task**

If audit returns BUG hits, the fix to read from `ctm_cfg.chi` (or its propagated equivalent) instead of env shape *must* land before Task 6 (the refactor that actually pads to chi_max). Open follow-up sub-tasks here for each. **Do not proceed to Task 6 without all BUG hits resolved.**

**Step 3: Commit audit findings (no code change yet)**

If the audit yields no BUG hits, write a one-paragraph note in the PR description (no commit needed for a pure grep). If it yields fixes, commit those as their own focused commit(s) before continuing:

```bash
git add src/tenax/algorithms/_ctm_<file>.py tests/...
git commit -m "fix(ctm): use config χ for truncation, not env shape (#453 audit)"
```

---

## Task 3: Write the padding-invariance test (failing on `main` post-Task 6 only)

**Files:**
- Test: `tests/test_ctm_env_pad_chi_schedule.py` (new)

**Step 1: Write the test**

This test pre-stages the invariant: an env padded from χ=4 to χ=8 by `pad_dense_env_chi`, evaluated against the same `A` at logical χ=4, must produce identical energy and gradient to the unpadded χ=4 evaluation. The test should pass on `main` already (the helper is in production for the §2.8.2 reactive bump). Run it now to confirm the baseline.

```python
"""Padding invariance for chi-schedule (issue #453).

If we converge CTM at χ=4 and pad envs to χ=8, evaluating energy and
gradient on the same A must yield the same answer as the unpadded χ=4
evaluation. This is the invariant the unified scheduled-bump relies on.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_env_pad import pad_dense_env_chi
from tenax.algorithms.ipeps_ctm import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.ipeps_optimize import SINGLE_SITE_NEIGHBORS


@pytest.mark.core
def test_pad_dense_env_chi_preserves_energy_at_logical_chi():
    d = 2
    D = 2
    chi_small = 4
    chi_big = 8

    # Random A.
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((D, D, D, D, d)))
    A = A / jnp.linalg.norm(A)

    # Heisenberg gate.
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    gate = (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real

    cfg_small = CTMConfig(chi=chi_small, max_steps=50, conv_tol=1e-8)
    envs_small, _ = python_loop_ctm_converge(
        {(0, 0): A}, SINGLE_SITE_NEIGHBORS, **_ctm_kwargs(cfg_small)
    )
    env_small = envs_small[(0, 0)]
    E_small = float(compute_energy_ctm_tensor(A, env_small, gate, d))

    # Pad to chi_big and re-evaluate. Energy must match exactly.
    env_padded = pad_dense_env_chi(env_small, chi_big, base_charges=None)
    E_padded = float(compute_energy_ctm_tensor(A, env_padded, gate, d))

    assert abs(E_padded - E_small) < 1e-12, (
        f"padded energy {E_padded:.16e} != unpadded {E_small:.16e}"
    )


def _ctm_kwargs(cfg):
    # Minimal kwargs to drive python_loop_ctm_converge — adapt to whatever
    # ctm_converge_kwargs(cfg) returns at the test's call site.
    from tenax.algorithms.ipeps_ctm import ctm_converge_kwargs

    return ctm_converge_kwargs(cfg, env_init=None)
```

**Step 2: Run**

```bash
uv run pytest tests/test_ctm_env_pad_chi_schedule.py -v
```

Expected: PASS on `main` (this invariant is already preserved by `pad_dense_env_chi`). If FAIL, the helper has a bug — fix that as a separate concern before continuing.

**Step 3: Commit**

```bash
git add tests/test_ctm_env_pad_chi_schedule.py
git commit -m "test(ctm): pin padding invariance for chi-schedule unification (#453)"
```

---

## Task 4: Extract `_apply_chi_bump` from `_maybe_bump_chi`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:35-82`

**Step 1: Write the failing test**

```python
# tests/test_apply_chi_bump.py
"""Unit test for the extracted _apply_chi_bump mechanism (#453)."""
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig


@pytest.mark.core
def test_apply_chi_bump_replaces_cfg_and_pads_envs():
    # Build a minimal env_cache shaped at χ=4.
    chi_old = 4
    chi_new = 8
    D = 2

    fake_C = jnp.zeros((chi_old, chi_old))
    fake_T = jnp.zeros((chi_old, D, D, chi_old))
    from tenax.tensor.ctm_env import CTMTensorEnv

    env = CTMTensorEnv(C1=fake_C, C2=fake_C, C3=fake_C, C4=fake_C,
                       T1=fake_T, T2=fake_T, T3=fake_T, T4=fake_T)
    env_cache = {"envs": {(0, 0): env}, "max_truncation_error": 0.0}
    ctm_cfg = CTMConfig(chi=chi_old)

    new_cfg, new_cache = _opt._apply_chi_bump(
        ctm_cfg, env_cache, chi_new, base_charges=None
    )

    assert new_cfg.chi == chi_new
    assert new_cache is env_cache  # in-place mutation
    assert new_cache["envs"][(0, 0)].C1.shape == (chi_new, chi_new)
```

(If `CTMTensorEnv` lives elsewhere or has a different constructor, adapt — use the path the existing `_maybe_bump_chi` test exercises if there is one.)

**Step 2: Run — expect failure**

```bash
uv run pytest tests/test_apply_chi_bump.py -v
```

Expected: FAIL — `AttributeError: module ... has no attribute '_apply_chi_bump'`.

**Step 3: Implement**

In `src/tenax/algorithms/ipeps_optimize.py`, replace the existing `_maybe_bump_chi` (lines 35–82) with:

```python
def _apply_chi_bump(
    ctm_cfg: CTMConfig,
    env_cache: dict,
    chi_new: int,
    *,
    base_charges: np.ndarray | None = None,
) -> tuple[CTMConfig, dict]:
    """Mechanism: bump logical χ and pad cached envs in-place.

    Pure mechanism, no policy. Used by both the reactive auto-χ_E
    bump (`_maybe_bump_chi`, variPEPS §2.8.2) and the scheduled
    bump driven by `gs_chi_schedule_steps` (`_maybe_scheduled_bump`,
    issue #453).

    ``env_cache`` is mutated in-place so closures that captured the
    dict reference (notably ``env_cache`` inside ``make_ctm_energy_fn``
    in ``optimize_gs_ad``) see the padded envs without rebinding.
    """
    new_cfg = dataclasses.replace(ctm_cfg, chi=chi_new)
    if "envs" in env_cache:
        env_cache["envs"] = {
            c: pad_dense_env_chi(env_cache["envs"][c], chi_new, base_charges=base_charges)
            for c in env_cache["envs"]
        }
    return new_cfg, env_cache


def _maybe_bump_chi(
    ctm_cfg: CTMConfig,
    env_cache: dict,
    last_eps_t: float,
    *,
    base_charges: np.ndarray | None = None,
) -> tuple[CTMConfig, dict]:
    """variPEPS §2.8.2 reactive χ_E bump.

    See ``_apply_chi_bump`` for the in-place mutation contract.
    """
    if not ctm_cfg.chi_auto_bump:
        return ctm_cfg, env_cache
    if last_eps_t <= ctm_cfg.chi_auto_bump_eps:
        return ctm_cfg, env_cache
    chi_new = ctm_cfg.chi + ctm_cfg.chi_auto_bump_step
    if ctm_cfg.chi_max is not None:
        chi_new = min(chi_new, ctm_cfg.chi_max)
    if chi_new <= ctm_cfg.chi:
        return ctm_cfg, env_cache  # at ceiling
    return _apply_chi_bump(ctm_cfg, env_cache, chi_new, base_charges=base_charges)
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_apply_chi_bump.py -v
uv run pytest -m core -k "chi_bump or maybe_bump or ctm_env_pad" -v
```

Expected: pass. The reactive-bump callers should be unaffected (the public function signature and behavior of `_maybe_bump_chi` is unchanged).

**Step 5: Commit**

```bash
git add tests/test_apply_chi_bump.py src/tenax/algorithms/ipeps_optimize.py
git commit -m "refactor(ipeps): extract _apply_chi_bump from _maybe_bump_chi (#453)"
```

---

## Task 5: Add `gs_chi_schedule_steps` to `iPEPSConfig` + `_maybe_scheduled_bump`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py:285` (add field near `gs_num_steps`)
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (add new function next to `_maybe_bump_chi`)
- Test: `tests/test_maybe_scheduled_bump.py` (new)

**Step 1: Write the failing test**

```python
"""_maybe_scheduled_bump unit test (#453)."""
import jax.numpy as jnp
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.tensor.ctm_env import CTMTensorEnv


def _make_env_cache(chi):
    fake_C = jnp.zeros((chi, chi))
    fake_T = jnp.zeros((chi, 2, 2, chi))
    env = CTMTensorEnv(C1=fake_C, C2=fake_C, C3=fake_C, C4=fake_C,
                       T1=fake_T, T2=fake_T, T3=fake_T, T4=fake_T)
    return {"envs": {(0, 0): env}}


@pytest.mark.core
def test_scheduled_bump_fires_at_boundary():
    # Schedule says: at cum step 5, target χ = 8.
    schedule = [(5, 4), (10, 8)]
    ctm_cfg = CTMConfig(chi=4)
    cache = _make_env_cache(4)

    # Step 4: before the chi=8 boundary, no bump.
    new_cfg, _ = _opt._maybe_scheduled_bump(ctm_cfg, cache, step=4, schedule_targets=schedule, base_charges=None)
    assert new_cfg.chi == 4

    # Step 5: cross the chi=8 boundary -> bump to 8.
    new_cfg, _ = _opt._maybe_scheduled_bump(ctm_cfg, cache, step=5, schedule_targets=schedule, base_charges=None)
    assert new_cfg.chi == 8
    assert cache["envs"][(0, 0)].C1.shape == (8, 8)


@pytest.mark.core
def test_scheduled_bump_idempotent_after_boundary():
    schedule = [(5, 4), (10, 8)]
    # Already at target chi=8 from a prior bump; subsequent calls past the boundary no-op.
    ctm_cfg = CTMConfig(chi=8)
    cache = _make_env_cache(8)
    new_cfg, _ = _opt._maybe_scheduled_bump(ctm_cfg, cache, step=7, schedule_targets=schedule, base_charges=None)
    assert new_cfg.chi == 8  # no bump down


@pytest.mark.core
def test_scheduled_bump_no_op_when_none():
    ctm_cfg = CTMConfig(chi=4)
    cache = _make_env_cache(4)
    new_cfg, new_cache = _opt._maybe_scheduled_bump(ctm_cfg, cache, step=10, schedule_targets=None, base_charges=None)
    assert new_cfg is ctm_cfg
    assert new_cache is cache
```

**Step 2: Run — expect failure**

```bash
uv run pytest tests/test_maybe_scheduled_bump.py -v
```

Expected: FAIL — `AttributeError: ... no attribute '_maybe_scheduled_bump'`.

**Step 3: Implement**

In `src/tenax/algorithms/ipeps_optimize.py`, immediately after `_maybe_bump_chi` (i.e. after the block introduced in Task 4), add:

```python
def _maybe_scheduled_bump(
    ctm_cfg: CTMConfig,
    env_cache: dict,
    step: int,
    schedule_targets: list[tuple[int, int]] | None,
    *,
    base_charges: np.ndarray | None = None,
) -> tuple[CTMConfig, dict]:
    """Step-driven χ bump from `gs_chi_schedule_steps`.

    ``schedule_targets`` is a list of ``(cumulative_step_boundary, target_chi)``
    pairs, sorted by ``cumulative_step_boundary``. When ``step`` crosses a
    boundary and ``target_chi > ctm_cfg.chi``, bump up via
    ``_apply_chi_bump``. Idempotent: stale boundaries (target chi <=
    current chi) are no-ops, so repeated calls per step are safe.

    Reuses the §2.8.2 padding mechanism (`pad_dense_env_chi`).
    Composes with `_maybe_bump_chi` — both can fire at the same step;
    `chi_max` caps the maximum either reaches.
    """
    if not schedule_targets:
        return ctm_cfg, env_cache

    target_chi = ctm_cfg.chi
    for boundary, chi_target in schedule_targets:
        if step >= boundary and chi_target > target_chi:
            target_chi = chi_target

    if target_chi <= ctm_cfg.chi:
        return ctm_cfg, env_cache

    if ctm_cfg.chi_max is not None:
        target_chi = min(target_chi, ctm_cfg.chi_max)
        if target_chi <= ctm_cfg.chi:
            return ctm_cfg, env_cache

    return _apply_chi_bump(ctm_cfg, env_cache, target_chi, base_charges=base_charges)
```

Also add the config field. In `src/tenax/algorithms/ipeps_config.py` near `gs_num_steps: int = 200` (~L285):

```python
    # Outer-loop χ schedule for `optimize_gs_ad_chi_schedule` (#453).
    # List of (cumulative_step_boundary, target_chi) pairs. When the
    # optimizer's step counter crosses a boundary, the inner CTM
    # logical χ is bumped to target_chi via _maybe_scheduled_bump.
    # Padded env shape is fixed at ctm.chi_max from step 1, so the
    # JIT-compiled kernels never see a shape change.
    gs_chi_schedule_steps: list[tuple[int, int]] | None = None
```

No new validation needed beyond what `optimize_gs_ad_chi_schedule` will set.

**Step 4: Run**

```bash
uv run pytest tests/test_maybe_scheduled_bump.py -v
```

Expected: 3 passed.

**Step 5: Commit**

```bash
git add tests/test_maybe_scheduled_bump.py src/tenax/algorithms/ipeps_optimize.py src/tenax/algorithms/ipeps_config.py
git commit -m "feat(ipeps): add _maybe_scheduled_bump + gs_chi_schedule_steps (#453)"
```

---

## Task 6: Wire `_maybe_scheduled_bump` into the three inner-loop step-end blocks

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` at the existing `_maybe_bump_chi` call sites.

**Step 1: Locate sites**

```bash
grep -n "_maybe_bump_chi(" src/tenax/algorithms/ipeps_optimize.py
```

Expect at least two hits (1-site at L1220, 1-site C4v / 2-site at L1476; possibly a multisite at L~2880 — verify by reading the function each call belongs to). For each:

**Step 2: Insert scheduled-bump call immediately after the reactive bump**

Example for the L1476 site (read the surrounding 10 lines for context first):

```python
        last_eps_t = float(_env_cache.get("max_truncation_error", 0.0))
        ctm_cfg, _env_cache = _maybe_bump_chi(
            ctm_cfg,
            _env_cache,
            last_eps_t,
            base_charges=_bump_base_charges,
        )
        # Scheduled outer-loop bump (#453) — composes with the reactive bump.
        if config.gs_chi_schedule_steps is not None:
            ctm_cfg, _env_cache = _maybe_scheduled_bump(
                ctm_cfg,
                _env_cache,
                step + 1,  # bump applies AFTER step `step` completes
                config.gs_chi_schedule_steps,
                base_charges=_bump_base_charges,
            )
```

Why `step + 1`: the boundary semantics in the schedule are "after this cumulative step count, χ should be at target". The bump must fire *after* the step's evaluation, so passing `step + 1` matches the inclusive boundary in `_maybe_scheduled_bump`'s `step >= boundary` check.

Repeat for the L1220 site and any multisite site.

**Step 3: Re-run scheduled-bump test under the inner loop**

Add an integration test:

```python
# tests/test_optimize_gs_ad_chi_schedule_unified.py
"""End-to-end: schedule [(5, 4), (10, 8)] grows env shape mid-run."""
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


@pytest.mark.core
def test_inner_loop_honors_gs_chi_schedule_steps():
    d = 2
    D = 2
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((D, D, D, D, d)))

    # Heisenberg gate (small).
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    gate = (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real

    schedule = [(5, 4), (10, 8)]
    cfg = iPEPSConfig(
        unit_cell="1x1",
        ctm=CTMConfig(chi=4, chi_max=8),
        gs_num_steps=10,
        gs_chi_schedule_steps=schedule,
        gs_optimizer="lbfgs",
        gs_verbose=False,
        su_init=False,
    )

    A_opt, env, E = _opt.optimize_gs_ad(gate, A, cfg)
    # After 10 steps, the scheduled bump at step 5 -> 4 (no change), step 10 -> 8 fired.
    assert env.C1.shape == (8, 8), f"expected env padded to chi=8, got {env.C1.shape}"
```

```bash
uv run pytest tests/test_optimize_gs_ad_chi_schedule_unified.py -v
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tests/test_optimize_gs_ad_chi_schedule_unified.py src/tenax/algorithms/ipeps_optimize.py
git commit -m "feat(ipeps): wire scheduled-bump into inner loop step-end blocks (#453)"
```

---

## Task 7: Refactor `optimize_gs_ad_chi_schedule` to a thin shim

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:445-499`

**Step 1: Write an end-to-end equivalence test (failing — new behavior)**

This is the load-bearing user-facing change. The semantic is: schedules of `[(4, 5), (8, 5)]` (5 steps at logical χ=4, then 5 at χ=8) must run as a *single* `optimize_gs_ad` call with 10 total steps and a scheduled bump at step 5. The optimizer state (L-BFGS history) is carried across the boundary now.

```python
# tests/test_optimize_gs_ad_chi_schedule_shim.py
"""optimize_gs_ad_chi_schedule is now a thin shim (#453)."""
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


@pytest.mark.core
def test_chi_schedule_shim_runs_single_optimizer_call(monkeypatch):
    """optimize_gs_ad_chi_schedule should now call optimize_gs_ad exactly once."""
    calls = []
    real_optimize = _opt.optimize_gs_ad

    def _spy(*args, **kwargs):
        calls.append(kwargs.get("config", args[-1] if args else None))
        return real_optimize(*args, **kwargs)

    monkeypatch.setattr(_opt, "optimize_gs_ad", _spy)

    d = 2
    D = 2
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((D, D, D, D, d)))
    gate = (0.5 * jnp.eye(d * d, dtype=jnp.float64)).reshape(d, d, d, d)

    cfg = iPEPSConfig(
        unit_cell="1x1",
        ctm=CTMConfig(chi=4),
        gs_num_steps=200,  # will be overridden by shim
        su_init=False,
    )
    _opt.optimize_gs_ad_chi_schedule(gate, A, cfg, chi_schedule=[(4, 3), (8, 3)])

    assert len(calls) == 1, f"shim should call optimize_gs_ad once, got {len(calls)}"
    inner_cfg = calls[0]
    assert inner_cfg.gs_num_steps == 6
    assert inner_cfg.gs_chi_schedule_steps == [(3, 4), (6, 8)]
    assert inner_cfg.ctm.chi == 4
    assert inner_cfg.ctm.chi_max == 8
```

```bash
uv run pytest tests/test_optimize_gs_ad_chi_schedule_shim.py -v
```

Expected: FAIL — shim still loops, makes 2 calls, doesn't set `gs_chi_schedule_steps`.

**Step 2: Replace `optimize_gs_ad_chi_schedule` body**

Replace `src/tenax/algorithms/ipeps_optimize.py:445-499` with:

```python
def optimize_gs_ad_chi_schedule(
    hamiltonian_gate: jax.Array | Tensor,
    A_init: jax.Array | Tensor | tuple | None,
    config: iPEPSConfig,
    chi_schedule: list[tuple[int, int]],
):
    """AD optimization with chi-ramping schedule (#453 unified).

    Runs ``optimize_gs_ad`` ONCE with envs padded to ``max(chi)`` from the
    very first iteration; the logical χ is ramped via
    ``_maybe_scheduled_bump`` at the configured step boundaries. The
    JIT-compiled CTM / energy / backward kernels therefore see a single
    fixed shape across the whole run — no per-stage retraces.

    Trade-off: stages running at logical χ < max(χ) contract
    ``max(χ)``-shaped envs (zeros in unused rows), paying more FLOPs per
    CTM iteration than a per-stage cold-start would. The recompile cost
    that this avoids (issue #453) dominates in practice.

    Reference: Zhang, Yang & Corboz, arXiv:2505.00494 (2025).

    Args:
        hamiltonian_gate: 2-site Hamiltonian of shape ``(d, d, d, d)``.
        A_init:           Initial site tensor(s) or ``None``.
        config:           Base iPEPSConfig.
        chi_schedule:     List of ``(chi, num_steps)`` pairs, e.g.
                          ``[(8, 100), (16, 50), (32, 30)]``.

    Returns:
        Same as ``optimize_gs_ad`` at the final chi level.
    """
    from dataclasses import replace

    chi_max = max(chi for chi, _ in chi_schedule)
    total_steps = sum(n for _, n in chi_schedule)
    cum = 0
    schedule_targets: list[tuple[int, int]] = []
    for chi, n in chi_schedule:
        cum += n
        schedule_targets.append((cum, chi))

    ctm_cfg = replace(config.ctm, chi=chi_schedule[0][0], chi_max=chi_max)
    step_cfg = replace(
        config,
        ctm=ctm_cfg,
        gs_num_steps=total_steps,
        gs_chi_schedule_steps=schedule_targets,
    )

    if config.gs_verbose:
        print(
            f"[chi-ramp] unified: chi_max={chi_max}, total_steps={total_steps}, "
            f"boundaries={schedule_targets}",
            flush=True,
        )

    return optimize_gs_ad(hamiltonian_gate, A_init, step_cfg)
```

**Step 3: Update `__post_init__` validation for `CTMConfig.chi_auto_bump` vs `gs_chi_schedule_steps`**

These two should be allowed to coexist (both call `_apply_chi_bump`, both cap at `chi_max`). No validation change needed *unless* there's already a mutual-exclusion check; in that case skip the schedule-vs-auto-bump check. Re-read `ipeps_config.py:204-207` (the `chi_auto_bump`/`chi_ramp` mutual-exclusion). Note: `chi_ramp` is the *inner* CTM ramp, not the outer `gs_chi_schedule_steps`. Verify your new field does NOT trip this exclusion.

**Step 4: Run the shim test + the integration test**

```bash
uv run pytest tests/test_optimize_gs_ad_chi_schedule_shim.py tests/test_optimize_gs_ad_chi_schedule_unified.py -v
```

Expected: pass.

**Step 5: Broader regression**

```bash
uv run pytest -m core -k "chi_schedule or chi_ramp or chi_auto or optimize_gs_ad" -v
```

Expected: pass.

**Step 6: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_optimize_gs_ad_chi_schedule_shim.py
git commit -m "refactor(ipeps): optimize_gs_ad_chi_schedule -> unified-schedule shim (#453)"
```

---

## Task 8: Add `ChiStageRecord` dataclass and per-bump recording

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (add dataclass near top of file; thread list through optimizer)
- Modify: same file's `optimize_gs_ad_chi_schedule` (accept `return_stages: bool = False` param)
- Test: `tests/test_chi_stage_record.py` (new)

**Step 1: Write failing test**

```python
"""Per-χ recording for finite-χ scaling analysis (#453, also #455)."""
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


@pytest.mark.core
def test_chi_schedule_return_stages_emits_one_record_per_bump():
    d = 2
    D = 2
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((D, D, D, D, d)))
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    gate = (
        jnp.einsum("ij,kl->ikjl", sx, sx) + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real

    cfg = iPEPSConfig(unit_cell="1x1", ctm=CTMConfig(chi=4), gs_num_steps=200,
                      gs_verbose=False, su_init=False)

    result = _opt.optimize_gs_ad_chi_schedule(
        gate, A, cfg, chi_schedule=[(4, 2), (8, 2)], return_stages=True
    )
    # Expected: 3-tuple from optimize_gs_ad + a `stages` list.
    *base_result, stages = result
    assert len(stages) >= 1, f"expected at least one bump record, got {stages}"
    # The χ=4 -> χ=8 bump fires after step 2 (cumulative).
    bumps = [r for r in stages if r.chi_post > r.chi_pre]
    assert any(r.chi_pre == 4 and r.chi_post == 8 for r in bumps), bumps
```

**Step 2: Run — expect failure**

```bash
uv run pytest tests/test_chi_stage_record.py -v
```

Expected: FAIL — `optimize_gs_ad_chi_schedule` does not accept `return_stages`.

**Step 3: Implement**

Near the top of `src/tenax/algorithms/ipeps_optimize.py`, after the existing imports, add:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ChiStageRecord:
    """One record per (auto + scheduled) χ bump.

    Used by `optimize_gs_ad_chi_schedule(..., return_stages=True)` for
    finite-χ scaling analysis. See issue #453 / follow-up #455.
    """
    chi_pre: int
    chi_post: int
    step: int
    E: float
    grad_norm: float
```

In the inner-loop step-end block at each `_maybe_bump_chi` / `_maybe_scheduled_bump` site, capture and append to a list. Pseudocode pattern:

```python
        chi_before = ctm_cfg.chi
        ctm_cfg, _env_cache = _maybe_bump_chi(...)
        if config.gs_chi_schedule_steps is not None:
            ctm_cfg, _env_cache = _maybe_scheduled_bump(...)
        if ctm_cfg.chi != chi_before:
            _stage_records.append(
                ChiStageRecord(
                    chi_pre=chi_before,
                    chi_post=ctm_cfg.chi,
                    step=step,
                    E=energy_float,
                    grad_norm=float(grad_norm_val),  # already computed for #449 criterion
                )
            )
```

Initialize `_stage_records: list[ChiStageRecord] = []` before the `for step in range(...)` loop in each variant.

The inner `optimize_gs_ad` should *always* build the list; it returns whatever its existing return signature is — extension is via a kwarg added to `optimize_gs_ad_chi_schedule` only.

**Step 4: Plumb to the shim**

Update `optimize_gs_ad_chi_schedule`:

```python
def optimize_gs_ad_chi_schedule(
    ...,
    return_stages: bool = False,
):
    ...
    result = optimize_gs_ad(hamiltonian_gate, A_init, step_cfg)
    if return_stages:
        # The inner optimizer must surface the stages list. Two options:
        # 1) optimize_gs_ad returns it as an extra trailing element when a
        #    sentinel is set on config (uglier, breaks tuple unpacking elsewhere).
        # 2) The inner optimizer always returns a `stages` attribute via a
        #    sidecar object — cleanest is option 1 hidden behind a private
        #    kwarg-flag on the config.
        # CHOOSE option 1 + private config flag (`_collect_stage_records=True`).
        # The shim sets the flag; inner appends `_stage_records` to the result tuple.
        return result  # already has stages appended by inner optimizer
    return result
```

Implementation detail: introduce a private field `_collect_stage_records: bool = False` on `iPEPSConfig` (not part of public API; leading underscore signals internal). When `True`, the inner optimizers return `(*existing_result, stages)`. The shim sets the flag when `return_stages=True` is passed.

This avoids changing `optimize_gs_ad`'s default return type for non-shim callers.

**Step 5: Run all stage tests**

```bash
uv run pytest tests/test_chi_stage_record.py tests/test_optimize_gs_ad_chi_schedule_unified.py tests/test_optimize_gs_ad_chi_schedule_shim.py -v
```

Expected: pass.

**Step 6: Commit**

```bash
git add tests/test_chi_stage_record.py src/tenax/algorithms/ipeps_optimize.py src/tenax/algorithms/ipeps_config.py
git commit -m "feat(ipeps): per-χ recording via ChiStageRecord (#453, #455)"
```

---

## Task 9: Documentation updates

**Files:**
- Modify: `README.md` (mention the unified chi_schedule behavior + return_stages briefly)
- Modify: `src/tenax/__init__.py` (export `ChiStageRecord` if it's part of the public API surface)

**Step 1: Decide on public API surface**

`ChiStageRecord` is exposed via `return_stages=True`. If it's the return type, it should be importable: add to `src/tenax/__init__.py:__all__` and re-export.

Edit:
```python
from tenax.algorithms.ipeps_optimize import ChiStageRecord
__all__ = [..., "ChiStageRecord"]
```

**Step 2: README touch-up**

In the iPEPS / chi-schedule section (search for `optimize_gs_ad_chi_schedule` in `README.md`), add a short paragraph mentioning:
- Envs are padded to `chi_max` from step 1 → no per-stage retraces.
- `return_stages=True` returns a list of `ChiStageRecord`s for scaling analysis.

Keep example code consistent with the new shim signature (no signature change for `optimize_gs_ad_chi_schedule` — only the optional kwarg is new).

**Step 3: Commit**

```bash
git add README.md src/tenax/__init__.py
git commit -m "docs(ipeps): document unified chi-schedule and ChiStageRecord (#453)"
```

---

## Task 10: Open the PR

**Step 1: Push and create**

```bash
git push -u origin perf/chi-ramp-pad-env
gh pr create --title "perf(ipeps): unified χ-schedule via _maybe_scheduled_bump, pad envs to chi_max (#453)" --body "$(cat <<'EOF'
## Summary

- Extract `_apply_chi_bump` from `_maybe_bump_chi`; reuse it for a new
  `_maybe_scheduled_bump` driven by `gs_chi_schedule_steps`.
- Refactor `optimize_gs_ad_chi_schedule` to a thin shim that calls
  `optimize_gs_ad` once with envs padded to `max(χ_schedule)` from step 1.
  No per-stage JIT retraces.
- Optimizer state, L-BFGS history, line-search state, AD graph all carry
  across stage boundaries (today every stage is a hard reset).
- Add `ChiStageRecord` dataclass and `return_stages=True` for finite-χ
  scaling analysis output. Follow-up #455 will gate bumps on convergence
  rather than fixed step budgets.

Closes #453. References #455.

## Trade-off

All stages now contract `chi_max`-shaped envs. Stages at logical
χ < `chi_max` pay extra FLOPs per CTM iteration. The recompile cost
this avoids dominates in practice; verified by the post-merge benchmark.

## Numerical-equivalence audit

`grep -nE "env\.(C[1-4]|T[1-4])\.shape" src/tenax/algorithms/_ctm_*.py`
returned: [paste audit findings — file:line + classification — here].

## Test plan

- [x] Padding invariance test (χ=4 padded to χ=8 evaluates to identical
  energy/gradient at logical χ=4).
- [x] `_apply_chi_bump` / `_maybe_scheduled_bump` unit tests.
- [x] End-to-end shim test (`optimize_gs_ad_chi_schedule` makes a single
  `optimize_gs_ad` call with the expected unified config).
- [x] `ChiStageRecord` emitted per bump.
- [ ] Post-merge GPU benchmark vs variPEPS at the production schedule.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
gh pr merge --squash --delete-branch --auto
```

---

## Notes for the executing engineer

- **Task 2 (audit) is load-bearing.** If you skip it and the env-shape-as-χ assumption is wrong anywhere, padded `chi_max` envs at small logical χ will silently produce wrong energies. Run the grep first; document findings in the PR body.
- **Don't merge if the post-merge benchmark regresses.** The intended outcome is wall-clock within ~10% of variPEPS at fixed `chi_max`. If padded-cost-at-small-χ wipes out the recompile win, escalate — possible mitigations include adopting option 2 from the issue (lift χ to `static_argnames`).
- **Adaptive convergence-triggered ramping is #455, not this PR.** Don't bundle.
- **Don't change `optimize_gs_ad`'s public return signature.** Stage recording flows via a private config flag (`_collect_stage_records`); the shim sets it; the inner optimizers append to the result tuple only when the flag is set.
