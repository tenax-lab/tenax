# iPEPS AD Stall-Recovery Fix (Issue #298) — Design

**Date:** 2026-04-10
**Issue:** #298 — 2-site (and 1-site non-C4v) AD instability from stall-recovery noise injection interacting with non-variational CTM regions.
**Scope:** Minimal fix — items 1–4 of the issue's proposed principled fix. Item 5 (migrating 1-site C4v off noise injection) is deferred to a follow-up.

## Problem Summary

Trajectory study in #298 showed two interacting pathologies in
`_optimize_gs_ad_tensor_2site` (and the 1-site non-C4v path):

1. `_rdm2x1_tensor_2site` / `_rdm1x2_tensor_2site` compute
   `rdm / (trace(rdm) + EPS)` (`_ctm_tensor_energy.py:303`). At small χ
   or underconverged environments the trace is near zero, yielding
   arbitrarily negative "energies" (observed: -12.27, +10.88) that are
   pure numerical noise.
2. The 10% Frobenius noise kick fired on L-BFGS stall
   (`gs_noise_recovery_retries=3` default) teleports the state into
   the 32-dim non-variational regime, and the optimizer then
   "descends" along the noise.

The 1-site C4v path **needs** the noise kick (the SU-init fixed point
has gradient ~1e-10 which trips `gs_conv_tol=1e-8` before L-BFGS can
move). So the fix cannot be "remove noise injection globally."

## Goals

- 2-site AD converges monotonically within ~0.005 of the literature
  Heisenberg D=2 χ=8 value in ≤ 20 steps with SU init.
- 1-site C4v production path still reaches ≤ -0.6602 at D=2 χ=8 with
  no change to user-facing config defaults.
- Resolution matches the variPEPS-style discipline: on stall, reset
  the L-BFGS state and roll back to the best known point, rather than
  perturbing forward blindly.

## Non-Goals

- Migrating the 1-site C4v path off noise injection (item 5).
- Making `_rdm2x1_tensor_2site` robust to near-zero-trace RDMs.
- Optimizer shell polymorphism over `SymmetricTensor` (issue #297).
- Multi-site unit cells larger than 2×1.

## Design

### 1. Config additions

`src/tenax/algorithms/ipeps_config.py`:

```python
gs_stall_recovery: Literal["noise", "reset"] | None = None
gs_energy_floor: float | None = None
```

Semantics:
- `gs_stall_recovery=None` → auto-defaulted at dispatch:
  - 1-site → `"noise"` (preserves current C4v production behavior).
  - 2-site → `"reset"` (new default, fixes the reported bug).
- `gs_stall_recovery="noise"` → existing 10 % Frobenius kick code path.
- `gs_stall_recovery="reset"` → variPEPS-style L-BFGS reset + rollback
  (see §2).
- `gs_energy_floor=None` → in-loop best-state tracking unchanged
  (backward compatible).
- `gs_energy_floor=<float>` → any candidate `best_E` below the floor
  is rejected as non-variational noise.

### 2. Reset-on-stall recovery branch

In both `_optimize_gs_ad_tensor` (~line 691) and
`_optimize_gs_ad_tensor_2site` (~line 1225), when
`stall_count > 0 and config.gs_stall_recovery == "reset"`:

1. Clear the L-BFGS curvature history: `sk_history.clear();
   yk_history.clear()`.
2. Roll back: `params = best_params`.
3. Force the next iteration's direction to steepest descent by
   invalidating cached `prev_grad` / `prev_direction` sentinels so the
   L-BFGS two-loop recursion produces `-g` on the next step.
4. Emit log line `[iPEPS-AD] stall #N, resetting L-BFGS → steepest
   descent from best`.
5. **Do not** call `jax.random.*` and do not perturb.

The existing noise-injection body remains, gated behind
`config.gs_stall_recovery == "noise"`.

### 3. Descent-direction sanity check

Before each line search (both 1-site and 2-site), compute
`<d, g>`. If positive:

```python
if jnp.vdot(direction, grads).real > 0:
    direction = -grads
    sk_history.clear()
    yk_history.clear()
```

This catches corrupted L-BFGS state before it produces an ascent
step. It is cheap, universally safe, and matches variPEPS's
pre-line-search guard.

### 4. Sanity floor on best-state tracking

In the `E < best_E` branch (~lines 510, 1022):

```python
floor = config.gs_energy_floor
if floor is not None and float(E) < floor:
    # Reject: likely a non-variational CTM region.
    pass
else:
    best_E = E
    best_params = params
```

Default `None` preserves existing behavior; users set the floor
explicitly (e.g. `2 * E_literature`).

### 5. 2-site dispatcher auto-override

In `_optimize_gs_ad_tensor_2site` (top of function):

```python
if config.gs_stall_recovery is None:
    config = replace(config, gs_stall_recovery="reset")
```

In `_optimize_gs_ad_tensor`:

```python
if config.gs_stall_recovery is None:
    config = replace(config, gs_stall_recovery="noise")
```

This makes the bug fix zero-config for 2-site callers while leaving
1-site C4v production scripts bit-identical.

## Testing

### New tests

- `tests/test_ipeps.py::test_2site_ad_stall_reset_converges` —
  Heisenberg D=2 χ=8 SU init, 20 L-BFGS steps, assert final E within
  0.01 of -0.6548. Expected to fail on `main`, pass after the fix.
- `tests/test_ipeps.py::test_energy_floor_rejects_spurious_best` —
  construct or monkeypatch a run that reports a synthetic E=-5.0 at
  some step; assert `best_E` stays above the user-supplied floor.
- `tests/test_ipeps.py::test_stall_recovery_defaults` — assert that
  the 2-site dispatcher auto-sets `gs_stall_recovery="reset"` and the
  1-site dispatcher auto-sets `"noise"` when the user leaves the
  field as `None`.

### Regression

- Existing 1-site C4v tests must pass unchanged with default config.
- Existing dense 1-site / 2-site tests must pass unchanged.
- Run `uv run pytest -m core` end-to-end before PR.

## Risks and Mitigations

- **Pre-line-search sanity check fires on 1-site C4v runs that were
  previously working** — the reset is benign: a positive `<d, g>`
  means L-BFGS was about to produce an ascent step anyway, so
  falling back to `-g` is strictly better.
- **Rollback on stall hides real progress** — `best_params` is
  already updated on every strictly-improving step, so rollback
  lands on the most recent known-good point, not an ancient one.
- **Auto-default divergence (1-site vs 2-site)** — documented in the
  config field's docstring and exercised in
  `test_stall_recovery_defaults`.

## Follow-ups (out of scope for this PR)

- Item 5 of the issue: migrate 1-site C4v off noise injection via a
  tighter `gs_conv_tol` or a directed first-step perturbation.
- Harden `_rdm2x1_tensor_2site` / `_rdm1x2_tensor_2site` against
  near-zero-trace RDMs (or detect and reject such steps at the
  optimizer level).
- Issue #297: polymorphic optimizer shell for SymmetricTensor AD.
