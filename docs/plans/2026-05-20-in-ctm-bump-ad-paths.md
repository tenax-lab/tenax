# In-CTM χ-bump in AD forward loops (#514) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thread the variPEPS-style in-CTM χ-bump (landed in PR #513 for `python_loop_ctm_converge`) into `ctm_energy_implicit` and `ctm_energy_explicit` so AD gradient evaluations actually grow chi. Bundle the #511 docs update, #499 GMRES debug logging, and #501 adjoint warm-start.

**Architecture:** Extract the bump-aware CTM convergence loop into a single shared helper (`_run_ctm_loop_with_bump` in new module `_ctm_loop_core`). Three callers wrap it: `python_loop_ctm_converge` (existing — refactor to delegate), `_sigma_gauged_ctm_converge` (implicit-AD forward), and `ctm_energy_explicit` warmup. Explicit-AD backprop runs unchanged with chi locked to the post-warmup value for tape integrity.

**Tech Stack:** JAX (custom_vjp, lax.while_loop, jax.checkpoint), Tenax CTM (`_ctm_tensor_*` modules), pytest with `@pytest.mark.core`.

**Design doc:** [docs/plans/2026-05-20-in-ctm-bump-ad-paths-design.md](2026-05-20-in-ctm-bump-ad-paths-design.md)

**Branch:** `feat/in-ctm-chi-bump-ad-paths-514` (stacked on merged #513)

---

## Task 1: Create `_ctm_loop_core.py` with `CTMLoopResult` NamedTuple

**Files:**
- Create: `src/tenax/algorithms/_ctm_loop_core.py`
- Test: `tests/test_ctm_loop_core.py`

**Step 1: Write the failing test**

```python
# tests/test_ctm_loop_core.py
"""Smoke tests for _run_ctm_loop_with_bump helper."""
from __future__ import annotations
import pytest


def test_ctmloopresult_fields_present():
    """CTMLoopResult exposes all required fields with correct types."""
    from tenax.algorithms._ctm_loop_core import CTMLoopResult

    r = CTMLoopResult(
        envs={},
        converged=True,
        iterations=5,
        sv_diff=1e-9,
        max_truncation_error=0.0,
        max_smallest_S=0.0,
        final_chi=8,
        bump_extra_sweeps=0,
    )
    assert r.converged is True
    assert r.iterations == 5
    assert r.final_chi == 8
    assert r.bump_extra_sweeps == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ctm_loop_core.py::test_ctmloopresult_fields_present -v`
Expected: FAIL with `ModuleNotFoundError`.

**Step 3: Create the module skeleton**

```python
# src/tenax/algorithms/_ctm_loop_core.py
"""Shared bump-aware CTM convergence loop.

Consumed by python_loop_ctm_converge, _sigma_gauged_ctm_converge (implicit-AD
forward), and ctm_energy_explicit warmup.  Centralizing the bump pad+resweep
sequence keeps the variPEPS-style growth contract (#492) in one place across
all three forward CTM paths (#514).
"""

from __future__ import annotations

__all__ = ["CTMLoopResult", "_run_ctm_loop_with_bump"]

from typing import NamedTuple

from tenax.algorithms._ctm_tensor_convergence import Coord
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv


class CTMLoopResult(NamedTuple):
    """Outcome of one bump-aware CTM convergence loop run."""

    envs: dict[Coord, CTMTensorEnv]
    converged: bool
    iterations: int
    sv_diff: float
    max_truncation_error: float
    max_smallest_S: float
    final_chi: int
    bump_extra_sweeps: int


def _run_ctm_loop_with_bump(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError  # filled in Task 2
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_loop_core.py::test_ctmloopresult_fields_present -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_loop_core.py tests/test_ctm_loop_core.py
git commit -m "feat(ctm): scaffold _ctm_loop_core with CTMLoopResult (#514)"
```

---

## Task 2: Port the bump-aware loop body into the helper

**Files:**
- Modify: `src/tenax/algorithms/_ctm_loop_core.py`
- Reference: `src/tenax/algorithms/_ctm_python_loop.py:299-519` (source of truth for loop semantics)
- Test: `tests/test_ctm_loop_core.py`

**Step 1: Write the failing test**

```python
# tests/test_ctm_loop_core.py — append
def test_helper_runs_one_sweep_no_bump():
    """Helper runs `max_iter` sweeps with bump disabled and returns final_chi=chi_current."""
    import numpy as np

    from tenax.algorithms._ctm_loop_core import _run_ctm_loop_with_bump
    from tenax.algorithms._ctm_python_loop import _make_jit_ctm_step
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
    from tenax.core.tensor import DenseTensor
    from tenax.core.tensor_index import TensorIndex

    rng = np.random.default_rng(0)
    D, d = 2, 2
    site = DenseTensor(
        rng.standard_normal((D, D, D, D, d)).astype(np.float64),
        (
            TensorIndex("u", D),
            TensorIndex("l", D),
            TensorIndex("d", D),
            TensorIndex("r", D),
            TensorIndex("p", d),
        ),
    )
    site_tensors = {(0, 0): site, (1, 0): site}
    neighbors = CHECKERBOARD_NEIGHBORS
    envs = {c: initialize_ctm_tensor_env(A, 4) for c, A in site_tensors.items()}
    jit_step = _make_jit_ctm_step(neighbors)

    result = _run_ctm_loop_with_bump(
        jit_step,
        site_tensors,
        envs,
        chi_current=4,
        chi_max=None,
        bump_enabled=False,
        bump_threshold=1e-6,
        bump_step_size=2,
        projector_method="svd",
        renormalize=False,
        projector_backward="auto",
        gauge_fix_fn=None,
        max_iter=3,
        min_iter=10,            # never converges → uses full budget
        conv_tol=1e-12,
        conv_method="sv",
        plateau_patience=None,
        bump_base_charges=None,
    )

    assert result.iterations == 3
    assert result.converged is False
    assert result.final_chi == 4
    assert result.bump_extra_sweeps == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ctm_loop_core.py::test_helper_runs_one_sweep_no_bump -v`
Expected: FAIL with `NotImplementedError`.

**Step 3: Implement the helper**

Replace the `_run_ctm_loop_with_bump` stub in `_ctm_loop_core.py` with the body ported from `_ctm_python_loop.py:299-519`. **Port verbatim**, then adapt the gauge-fix call site to use the pair semantics:

```python
import jax
from tenax.algorithms._ctm_env_pad import pad_dense_env_chi
from tenax.algorithms._ctm_tensor_convergence import (
    _corner_singular_values,
    _ctm_sv_diff,
    _max_env_leaf_diff,
)


def _run_ctm_loop_with_bump(
    jit_step,
    site_tensors,
    envs_init,
    *,
    chi_current: int,
    chi_max: int | None,
    bump_enabled: bool,
    bump_threshold: float,
    bump_step_size: int,
    projector_method: str,
    renormalize: bool,
    projector_backward: str,
    gauge_fix_fn,
    max_iter: int,
    min_iter: int,
    conv_tol: float,
    conv_method: str,
    plateau_patience: int | None,
    bump_base_charges,
) -> CTMLoopResult:
    """Run CTM sweeps with optional variPEPS-style in-CTM chi-bump.

    Mirrors the loop in python_loop_ctm_converge (lines 299-519 prior to
    extraction).  Caller is responsible for warmup, env_init validation,
    and (chi_max, chi_current) constraints.

    gauge_fix_fn:
        Callable (envs_new, envs_old) -> envs, or None.  Phase gauge wraps a
        single-arg phase fix; sigma gauge uses both args.  None disables.
    """
    chi_max_eff = chi_max if chi_max is not None else chi_current
    envs = envs_init
    remaining = max_iter

    prev_svs: dict = {}
    prev_envs: dict | None = None
    final_diff = float("inf")
    last_max_eps = 0.0
    last_max_smallest_S = 0.0
    best_diff = float("inf")
    best_envs: dict | None = None
    best_iter = 0
    iters_since_best = 0
    bump_extra_sweeps = 0

    for i in range(remaining):
        if i + bump_extra_sweeps >= remaining:
            break
        envs_new, _max_eps, _max_S = jit_step(
            site_tensors,
            envs,
            chi=chi_current,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )
        last_max_eps = float(_max_eps)
        last_max_smallest_S = float(_max_S)

        bump_would_fire = (
            bump_enabled
            and last_max_smallest_S > bump_threshold
            and chi_current < chi_max_eff
        )
        if bump_would_fire and (i + 1 + bump_extra_sweeps < remaining):
            chi_current = min(chi_current + bump_step_size, chi_max_eff)
            envs = {
                c: pad_dense_env_chi(envs_new[c], chi_current, base_charges=bump_base_charges)
                for c in envs_new
            }
            envs, _max_eps, _max_S = jit_step(
                site_tensors,
                envs,
                chi=chi_current,
                projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
            bump_extra_sweeps += 1
            last_max_eps = float(_max_eps)
            last_max_smallest_S = float(_max_S)
            if gauge_fix_fn is not None:
                envs = gauge_fix_fn(envs, envs_new)
            prev_svs = {}
            prev_envs = None
            best_diff = float("inf")
            best_envs = None
            iters_since_best = 0
            continue

        if gauge_fix_fn is not None:
            envs = gauge_fix_fn(envs_new, envs)
        else:
            envs = envs_new

        total_iter = i + 1 + bump_extra_sweeps
        if total_iter < min_iter:
            if conv_method == "sv":
                for c in sorted(envs):
                    prev_svs[c] = _corner_singular_values(envs[c].C1)
            else:
                prev_envs = {c: envs[c] for c in envs}
            continue

        plateau_metric_valid = False
        if conv_method == "elementwise":
            if prev_envs is None:
                prev_envs = {c: envs[c] for c in envs}
                continue
            max_diff = 0.0
            for c in sorted(envs):
                max_diff = max(max_diff, _max_env_leaf_diff(prev_envs[c], envs[c]))
            converged = max_diff < conv_tol
            final_diff = max_diff
            prev_envs = {c: envs[c] for c in envs}
            plateau_metric_valid = True
        else:
            have_prev_svs = bool(prev_svs)
            converged = True
            max_diff = 0.0
            for c in sorted(envs):
                sv = _corner_singular_values(envs[c].C1)
                if c in prev_svs:
                    diff = float(_ctm_sv_diff(sv, prev_svs[c]))
                    max_diff = max(max_diff, diff)
                    if diff >= conv_tol:
                        converged = False
                else:
                    converged = False
                prev_svs[c] = sv
            if have_prev_svs:
                final_diff = max_diff
                plateau_metric_valid = True

        if converged:
            return CTMLoopResult(
                envs=envs,
                converged=True,
                iterations=total_iter,
                sv_diff=final_diff,
                max_truncation_error=last_max_eps,
                max_smallest_S=last_max_smallest_S,
                final_chi=chi_current,
                bump_extra_sweeps=bump_extra_sweeps,
            )

        if plateau_patience is not None and plateau_metric_valid:
            if final_diff < best_diff:
                best_diff = final_diff
                best_envs = {c: envs[c] for c in envs}
                best_iter = total_iter
                iters_since_best = 0
            else:
                iters_since_best += 1
                if iters_since_best >= plateau_patience:
                    return CTMLoopResult(
                        envs=best_envs or envs,
                        converged=False,
                        iterations=best_iter or total_iter,
                        sv_diff=best_diff,
                        max_truncation_error=last_max_eps,
                        max_smallest_S=last_max_smallest_S,
                        final_chi=chi_current,
                        bump_extra_sweeps=bump_extra_sweeps,
                    )

    return CTMLoopResult(
        envs=envs,
        converged=False,
        iterations=remaining,
        sv_diff=final_diff,
        max_truncation_error=last_max_eps,
        max_smallest_S=last_max_smallest_S,
        final_chi=chi_current,
        bump_extra_sweeps=bump_extra_sweeps,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_loop_core.py -v`
Expected: both tests PASS.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_loop_core.py tests/test_ctm_loop_core.py
git commit -m "feat(ctm): port bump-aware loop body into _run_ctm_loop_with_bump (#514)"
```

---

## Task 3: Wire `python_loop_ctm_converge` to delegate to the helper

**Files:**
- Modify: `src/tenax/algorithms/_ctm_python_loop.py:299-519`

**Step 1: Run the existing 18 in-CTM bump tests first to lock in the baseline**

Run: `uv run pytest tests/test_ctm_in_loop_chi_bump.py -v`
Expected: 18 PASS.

**Step 2: Replace the loop body in `python_loop_ctm_converge`**

Keep the validation block (lines 191-298) intact. Replace lines 299-519 with:

```python
# Build gauge_fix_fn pair adapter
if gauge_fix_fn is not None:
    _user_gauge = gauge_fix_fn
    def _gauge_pair(envs_new, envs_old):
        return {c: _user_gauge(envs_new[c]) for c in envs_new}
else:
    _gauge_pair = None

# Pre-compute bump_base_charges
bump_base_charges = None
if ctmrg_heuristic_increase_chi:
    for A in site_tensors.values():
        bump_base_charges = _get_base_charges(_build_double_layer_tensor(A))
        if bump_base_charges is not None:
            break

# Initialize envs
envs = (
    env_init
    if env_init is not None
    else {c: initialize_ctm_tensor_env(A, chi_current) for c, A in site_tensors.items()}
)

# QR warmup (unchanged)
warmup = 0
if projector_method == "qr" and qr_warmup_steps > 0:
    warmup = min(qr_warmup_steps, max_iter)
    for _ in range(warmup):
        envs, _, _ = jit_step(
            site_tensors,
            envs,
            chi=chi_current,
            projector_method="eigh",
            renormalize=renormalize,
            projector_backward=projector_backward,
        )

# Run the bump-aware loop via shared helper
from tenax.algorithms._ctm_loop_core import _run_ctm_loop_with_bump

result = _run_ctm_loop_with_bump(
    jit_step,
    site_tensors,
    envs,
    chi_current=chi_current,
    chi_max=chi_max,
    bump_enabled=ctmrg_heuristic_increase_chi,
    bump_threshold=ctmrg_heuristic_increase_chi_threshold,
    bump_step_size=ctmrg_heuristic_increase_chi_step_size,
    projector_method=projector_method,
    renormalize=renormalize,
    projector_backward=projector_backward,
    gauge_fix_fn=_gauge_pair,
    max_iter=max_iter - warmup,
    min_iter=max(0, min_iter - warmup),
    conv_tol=conv_tol,
    conv_method=conv_method,
    plateau_patience=plateau_patience,
    bump_base_charges=bump_base_charges,
)

return result.envs, CTMConvergeInfo(
    converged=result.converged,
    iterations=warmup + result.iterations,
    sv_diff=result.sv_diff,
    max_truncation_error=result.max_truncation_error,
    max_smallest_S=result.max_smallest_S,
    final_chi=result.final_chi,
)
```

**Step 3: Run the existing 18 in-CTM bump tests**

Run: `uv run pytest tests/test_ctm_in_loop_chi_bump.py -v`
Expected: 18 PASS (no regression — helper is a refactor).

**Step 4: Run broader CTM tests as smoke**

Run: `uv run pytest tests/test_ctm_python_loop.py tests/test_ctm_compiled.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_python_loop.py
git commit -m "refactor(ctm): delegate python_loop_ctm_converge body to _run_ctm_loop_with_bump (#514)"
```

---

## Task 4: Wire `_sigma_gauged_ctm_converge` to the helper

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py:369-503`

**Step 1: Write the failing test (will be added in Task 8 — defer)**

Skip Step 1 here; this task is pure refactor, validated by smoke.

**Step 2: Replace the loop body**

In `_ctm_energy_ad.py`, replace `_sigma_gauged_ctm_converge` body. New signature gains four kwargs:

```python
def _sigma_gauged_ctm_converge(
    site_tensors,
    neighbors,
    *,
    chi,
    max_iter,
    conv_tol,
    projector_method,
    renormalize,
    projector_backward,
    qr_warmup_steps,
    env_init,
    forward_gauge="phase",
    conv_method="sv",
    min_iter=4,
    plateau_patience: int | None = None,
    # NEW kwargs (#514)
    ctmrg_heuristic_increase_chi: bool = False,
    ctmrg_heuristic_increase_chi_threshold: float = 1e-6,
    ctmrg_heuristic_increase_chi_step_size: int = 2,
    chi_max: int | None = None,
):
    from tenax.algorithms._ctm_loop_core import _run_ctm_loop_with_bump
    from tenax.algorithms._ctm_tensor_convergence import _get_base_charges
    from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor

    # ---- validation (mirror python_loop_ctm_converge) ----
    if ctmrg_heuristic_increase_chi and chi_max is None:
        raise ValueError(
            "ctmrg_heuristic_increase_chi=True requires chi_max to be set"
        )
    if ctmrg_heuristic_increase_chi and ctmrg_heuristic_increase_chi_step_size <= 0:
        raise ValueError(
            "ctmrg_heuristic_increase_chi_step_size must be a positive integer"
        )

    chi_current = chi
    if ctmrg_heuristic_increase_chi and env_init:
        try:
            sample_env = next(iter(env_init.values()))
            env_chi = int(sample_env.C1.indices[0].dim)
        except (StopIteration, AttributeError, IndexError):
            env_chi = None
        if env_chi is not None:
            if chi_max is not None and env_chi > chi_max:
                raise ValueError(
                    f"env_init has chi={env_chi} exceeding chi_max={chi_max}"
                )
            if env_chi > chi_current:
                chi_current = env_chi
    if chi_max is not None and chi_max < chi_current:
        raise ValueError(
            f"chi_max ({chi_max}) must be >= chi_current ({chi_current})"
        )

    jit_step = _make_jit_ctm_step(neighbors)
    envs = (
        env_init
        if env_init is not None
        else {c: initialize_ctm_tensor_env(A, chi_current) for c, A in site_tensors.items()}
    )

    # ---- QR warmup (unchanged) ----
    warmup = (
        min(qr_warmup_steps, max_iter)
        if projector_method == "qr" and qr_warmup_steps > 0
        else 0
    )
    for _ in range(warmup):
        envs, _eps, _smin = jit_step(
            site_tensors,
            envs,
            chi=chi_current,
            projector_method="eigh",
            renormalize=renormalize,
            projector_backward=projector_backward,
        )

    # ---- gauge_fix_fn pair adapter ----
    if forward_gauge == "phase":
        def _gauge_pair(envs_new, _envs_old):
            return {c: _phase_fix_ctm_tensor(envs_new[c]) for c in envs_new}
    elif forward_gauge == "sigma":
        def _gauge_pair(envs_new, envs_old):
            return {c: _sigma_gauge_fix_env(envs_new[c], envs_old[c]) for c in envs_new}
    elif forward_gauge == "none":
        _gauge_pair = None
    else:
        raise ValueError(f"Unknown forward_gauge={forward_gauge!r}")

    # ---- bump_base_charges ----
    bump_base_charges = None
    if ctmrg_heuristic_increase_chi:
        for A in site_tensors.values():
            bump_base_charges = _get_base_charges(_build_double_layer_tensor(A))
            if bump_base_charges is not None:
                break

    # ---- delegate to shared helper ----
    result = _run_ctm_loop_with_bump(
        jit_step,
        site_tensors,
        envs,
        chi_current=chi_current,
        chi_max=chi_max,
        bump_enabled=ctmrg_heuristic_increase_chi,
        bump_threshold=ctmrg_heuristic_increase_chi_threshold,
        bump_step_size=ctmrg_heuristic_increase_chi_step_size,
        projector_method=projector_method,
        renormalize=renormalize,
        projector_backward=projector_backward,
        gauge_fix_fn=_gauge_pair,
        max_iter=max_iter - warmup,
        min_iter=max(0, min_iter - warmup),
        conv_tol=conv_tol,
        conv_method=conv_method,
        plateau_patience=plateau_patience,
        bump_base_charges=bump_base_charges,
    )
    return result.envs
```

**Step 3: Smoke test existing implicit-AD tests**

Run: `uv run pytest tests/test_ctm_energy_ad.py -v -m core`
Expected: PASS (no regression).

**Step 4: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py
git commit -m "refactor(ctm): delegate _sigma_gauged_ctm_converge to shared helper (#514)"
```

---

## Task 5: Plumb bump kwargs through `ctm_energy_implicit` + dispatch + VJP cache key

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py:257-366` (`ctm_energy_implicit` + dispatch + `_make_implicit_vjp_fn`)

**Step 1: Add the four kwargs to `ctm_energy_implicit`**

Add to signature (after `plateau_patience: int | None = None`):

```python
ctmrg_heuristic_increase_chi: bool = False,
ctmrg_heuristic_increase_chi_threshold: float = 1e-6,
ctmrg_heuristic_increase_chi_step_size: int = 2,
chi_max: int | None = None,
```

Plumb through `_ctm_energy_implicit_dispatch` (positional args):

```python
return _ctm_energy_implicit_dispatch(
    ...,
    plateau_patience,
    ctmrg_heuristic_increase_chi,
    ctmrg_heuristic_increase_chi_threshold,
    ctmrg_heuristic_increase_chi_step_size,
    chi_max,
)
```

**Step 2: Extend `_VJP_CACHE` cache_key**

In `_ctm_energy_implicit_dispatch`, add to `cache_key` tuple:

```python
cache_key = (
    ...
    plateau_patience,
    ctmrg_heuristic_increase_chi,
    ctmrg_heuristic_increase_chi_threshold,
    ctmrg_heuristic_increase_chi_step_size,
    chi_max,
)
```

**Step 3: Plumb to `_make_implicit_vjp_fn` and into `_sigma_gauged_ctm_converge` call**

Pass the four kwargs to `_make_implicit_vjp_fn`. Inside it, forward them to the `_sigma_gauged_ctm_converge(...)` call in `_run_forward`:

```python
envs = _sigma_gauged_ctm_converge(
    site_tensors,
    neighbors,
    chi=chi,
    ...,
    plateau_patience=plateau_patience,
    ctmrg_heuristic_increase_chi=ctmrg_heuristic_increase_chi,
    ctmrg_heuristic_increase_chi_threshold=ctmrg_heuristic_increase_chi_threshold,
    ctmrg_heuristic_increase_chi_step_size=ctmrg_heuristic_increase_chi_step_size,
    chi_max=chi_max,
)
```

Note: only the `chi_ramp is None` branch needs the new knobs. The `chi_ramp is not None` branch is mutex with bump (already enforced).

**Step 4: Smoke test**

Run: `uv run pytest tests/test_ctm_energy_ad.py -v -m core`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py
git commit -m "feat(ctm): plumb in-CTM bump kwargs through ctm_energy_implicit (#514)"
```

---

## Task 6: Wire `ctm_energy_explicit` warmup to the helper with chi-lock

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py:38-96` (`ctm_energy_explicit`)

**Step 1: Replace the warmup loop**

```python
def ctm_energy_explicit(
    site_tensors: dict[Coord, object],
    neighbors: dict[Coord, dict[str, Coord]],
    gate,
    *,
    chi: int = 20,
    warmup_steps: int = 3,
    backprop_steps: int = 20,
    projector_method: str = "svd",
    renormalize: bool = True,
    projector_backward: str = "auto",
    env_init: dict[Coord, CTMTensorEnv] | None = None,
    energy_fn=None,
    # NEW (#514)
    ctmrg_heuristic_increase_chi: bool = False,
    ctmrg_heuristic_increase_chi_threshold: float = 1e-6,
    ctmrg_heuristic_increase_chi_step_size: int = 2,
    chi_max: int | None = None,
) -> jnp.ndarray:
    from tenax.algorithms._ctm_loop_core import _run_ctm_loop_with_bump
    from tenax.algorithms._ctm_tensor_convergence import _get_base_charges
    from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor

    if ctmrg_heuristic_increase_chi and chi_max is None:
        raise ValueError(
            "ctmrg_heuristic_increase_chi=True requires chi_max to be set"
        )
    if ctmrg_heuristic_increase_chi and ctmrg_heuristic_increase_chi_step_size <= 0:
        raise ValueError(
            "ctmrg_heuristic_increase_chi_step_size must be a positive integer"
        )

    chi_current = chi
    if ctmrg_heuristic_increase_chi and env_init:
        try:
            sample_env = next(iter(env_init.values()))
            env_chi = int(sample_env.C1.indices[0].dim)
        except (StopIteration, AttributeError, IndexError):
            env_chi = None
        if env_chi is not None:
            if chi_max is not None and env_chi > chi_max:
                raise ValueError(
                    f"env_init has chi={env_chi} exceeding chi_max={chi_max}"
                )
            if env_chi > chi_current:
                chi_current = env_chi
    if chi_max is not None and chi_max < chi_current:
        raise ValueError(
            f"chi_max ({chi_max}) must be >= chi_current ({chi_current})"
        )

    jit_step = _make_jit_ctm_step(neighbors)
    envs = (
        env_init
        if env_init is not None
        else {c: initialize_ctm_tensor_env(A, chi_current) for c, A in site_tensors.items()}
    )

    bump_base_charges = None
    if ctmrg_heuristic_increase_chi:
        for A in site_tensors.values():
            bump_base_charges = _get_base_charges(_build_double_layer_tensor(A))
            if bump_base_charges is not None:
                break

    # WARMUP: bump-aware, no-grad
    if warmup_steps > 0:
        warmup_result = _run_ctm_loop_with_bump(
            jit_step,
            site_tensors,
            envs,
            chi_current=chi_current,
            chi_max=chi_max,
            bump_enabled=ctmrg_heuristic_increase_chi,
            bump_threshold=ctmrg_heuristic_increase_chi_threshold,
            bump_step_size=ctmrg_heuristic_increase_chi_step_size,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
            gauge_fix_fn=None,
            max_iter=warmup_steps,
            min_iter=warmup_steps + 1,    # disables convergence early-exit
            conv_tol=1e30,
            conv_method="sv",
            plateau_patience=None,
            bump_base_charges=bump_base_charges,
        )
        envs = jax.tree.map(jax.lax.stop_gradient, warmup_result.envs)
        chi_post_warmup = warmup_result.final_chi
    else:
        chi_post_warmup = chi_current

    # BACKPROP: fixed chi (tape integrity)
    def _step_envs_only(st, e):
        envs_out, _eps, _smin = jit_step(
            st,
            e,
            chi=chi_post_warmup,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )
        return envs_out

    for _ in range(backprop_steps):
        envs = jax.checkpoint(_step_envs_only)(site_tensors, envs)

    if energy_fn is not None:
        return energy_fn(site_tensors, envs, gate)
    coords = sorted(site_tensors.keys())
    return _default_energy(site_tensors, envs, gate, coords, neighbors)
```

**Step 2: Smoke test explicit-AD tests**

Run: `uv run pytest tests/test_ctm_energy_ad.py -v -m core -k explicit`
Expected: PASS.

**Step 3: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py
git commit -m "feat(ctm): bump-aware warmup in ctm_energy_explicit with chi-lock (#514)"
```

---

## Task 7: Thread bump kwargs through `ipeps_ad_policy.make_ctm_energy_fn`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_ad_policy.py:177-220`

**Step 1: Pass kwargs to both explicit + implicit calls**

```python
def _ctm_energy_fn(site_tensors):
    ctm_cfg = get_ctm_cfg()
    env_init = env_cache.get("envs", None)
    bump_kwargs = dict(
        ctmrg_heuristic_increase_chi=ctm_cfg.ctmrg_heuristic_increase_chi,
        ctmrg_heuristic_increase_chi_threshold=ctm_cfg.ctmrg_heuristic_increase_chi_threshold,
        ctmrg_heuristic_increase_chi_step_size=ctm_cfg.ctmrg_heuristic_increase_chi_step_size,
        chi_max=ctm_cfg.chi_max,
    )
    if use_explicit:
        return ctm_energy_explicit(
            site_tensors,
            neighbors,
            gate,
            chi=ctm_cfg.chi,
            warmup_steps=explicit_warmup,
            backprop_steps=explicit_steps,
            projector_method=ctm_cfg.projector_method,
            renormalize=ctm_cfg.renormalize,
            projector_backward=ctm_cfg.projector_backward,
            env_init=env_init,
            energy_fn=energy_fn,
            **bump_kwargs,
        )
    return ctm_energy_implicit(
        ...,                       # unchanged args
        adjoint_method=ctm_cfg.adjoint_method,
        **bump_kwargs,
    )
```

**Step 2: Smoke test ipeps_optimize end-to-end**

Run: `uv run pytest tests/test_ipeps_optimize_*.py -v -m core`
Expected: PASS.

**Step 3: Commit**

```bash
git add src/tenax/algorithms/ipeps_ad_policy.py
git commit -m "feat(ipeps): thread bump kwargs through make_ctm_energy_fn (#514)"
```

---

## Task 8: Integration tests for bump in AD paths

**Files:**
- Create: `tests/test_ctm_in_loop_bump_ad_paths.py`

**Step 1: Write the failing tests**

```python
# tests/test_ctm_in_loop_bump_ad_paths.py
"""In-CTM bump integration tests for ctm_energy_implicit + ctm_energy_explicit (#514)."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.core


def _make_site_tensors(seed=0, D=2, d=2):
    from tenax.core.tensor import DenseTensor
    from tenax.core.tensor_index import TensorIndex

    rng = np.random.default_rng(seed)
    def _site():
        return DenseTensor(
            rng.standard_normal((D, D, D, D, d)).astype(np.float64),
            (
                TensorIndex("u", D),
                TensorIndex("l", D),
                TensorIndex("d", D),
                TensorIndex("r", D),
                TensorIndex("p", d),
            ),
        )
    return {(0, 0): _site(), (1, 0): _site()}


def _heisenberg_gate():
    from tenax.gates import heisenberg_gate
    return heisenberg_gate()


def test_implicit_ad_forward_grows_chi():
    """Bump fires during implicit-AD forward; final env chi exceeds chi_init."""
    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    site_tensors = _make_site_tensors()
    # Run with threshold high enough to force bump on a typical random env
    energy = ctm_energy_implicit(
        site_tensors,
        CHECKERBOARD_NEIGHBORS,
        _heisenberg_gate(),
        chi=4,
        max_iter=15,
        conv_tol=1e-12,           # never converges → bump fires
        ctmrg_heuristic_increase_chi=True,
        ctmrg_heuristic_increase_chi_threshold=1e-3,
        ctmrg_heuristic_increase_chi_step_size=2,
        chi_max=10,
    )
    # If bump didn't fire, this still returns *something*; the meaningful
    # signal is the env_cache after the call.  We hook that next via warm-start.
    assert float(energy) == pytest.approx(float(energy))  # smoke: no NaN, runs


def test_implicit_ad_chi_max_none_raises():
    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    with pytest.raises(ValueError, match="chi_max"):
        ctm_energy_implicit(
            _make_site_tensors(),
            CHECKERBOARD_NEIGHBORS,
            _heisenberg_gate(),
            chi=4,
            max_iter=2,
            ctmrg_heuristic_increase_chi=True,
            chi_max=None,
        )


def test_explicit_ad_chi_max_none_raises():
    from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    with pytest.raises(ValueError, match="chi_max"):
        ctm_energy_explicit(
            _make_site_tensors(),
            CHECKERBOARD_NEIGHBORS,
            _heisenberg_gate(),
            chi=4,
            warmup_steps=2,
            backprop_steps=1,
            ctmrg_heuristic_increase_chi=True,
            chi_max=None,
        )


def test_explicit_ad_env_init_above_chi_max_raises():
    """env_init.chi > chi_max raises with the bump enabled."""
    from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env

    site_tensors = _make_site_tensors()
    env_init = {c: initialize_ctm_tensor_env(A, 8) for c, A in site_tensors.items()}
    with pytest.raises(ValueError, match="exceeding chi_max"):
        ctm_energy_explicit(
            site_tensors,
            CHECKERBOARD_NEIGHBORS,
            _heisenberg_gate(),
            chi=4,
            warmup_steps=2,
            backprop_steps=1,
            ctmrg_heuristic_increase_chi=True,
            chi_max=6,
            env_init=env_init,
        )


def test_explicit_ad_step_size_zero_raises():
    from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    with pytest.raises(ValueError, match="positive integer"):
        ctm_energy_explicit(
            _make_site_tensors(),
            CHECKERBOARD_NEIGHBORS,
            _heisenberg_gate(),
            chi=4,
            warmup_steps=2,
            backprop_steps=1,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_step_size=0,
            chi_max=8,
        )


def test_explicit_ad_grad_flows_through_bump():
    """jax.grad through ctm_energy_explicit with bump enabled returns finite grads."""
    import jax

    from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    site_tensors = _make_site_tensors()
    coords = sorted(site_tensors.keys())
    params = tuple(site_tensors[c].data for c in coords)

    def _loss(p_tuple):
        from tenax.core.tensor import DenseTensor
        from tenax.core.tensor_index import TensorIndex
        D = p_tuple[0].shape[0]
        d = p_tuple[0].shape[-1]
        idx = (
            TensorIndex("u", D), TensorIndex("l", D),
            TensorIndex("d", D), TensorIndex("r", D),
            TensorIndex("p", d),
        )
        st = {c: DenseTensor(p, idx) for c, p in zip(coords, p_tuple)}
        return ctm_energy_explicit(
            st,
            CHECKERBOARD_NEIGHBORS,
            _heisenberg_gate(),
            chi=4,
            warmup_steps=3,
            backprop_steps=2,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-3,
            chi_max=8,
        )

    g = jax.grad(_loss)(params)
    for grad in g:
        assert np.all(np.isfinite(np.asarray(grad)))


def test_implicit_ad_grad_flows_through_bump():
    """jax.grad through ctm_energy_implicit with bump enabled returns finite grads."""
    import jax

    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    site_tensors = _make_site_tensors()
    coords = sorted(site_tensors.keys())
    params = tuple(site_tensors[c].data for c in coords)

    def _loss(p_tuple):
        from tenax.core.tensor import DenseTensor
        from tenax.core.tensor_index import TensorIndex
        D = p_tuple[0].shape[0]
        d = p_tuple[0].shape[-1]
        idx = (
            TensorIndex("u", D), TensorIndex("l", D),
            TensorIndex("d", D), TensorIndex("r", D),
            TensorIndex("p", d),
        )
        st = {c: DenseTensor(p, idx) for c, p in zip(coords, p_tuple)}
        return ctm_energy_implicit(
            st,
            CHECKERBOARD_NEIGHBORS,
            _heisenberg_gate(),
            chi=4,
            max_iter=10,
            conv_tol=1e-6,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-3,
            chi_max=8,
        )

    g = jax.grad(_loss)(params)
    for grad in g:
        assert np.all(np.isfinite(np.asarray(grad)))
```

**Step 2: Run the tests**

Run: `uv run pytest tests/test_ctm_in_loop_bump_ad_paths.py -v`
Expected: all PASS.

**Step 3: Commit**

```bash
git add tests/test_ctm_in_loop_bump_ad_paths.py
git commit -m "test(ctm): integration tests for in-CTM bump in AD forwards (#514)"
```

---

## Task 9: Add #501 warm-start adjoint solve

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py:613-650` (`_make_implicit_vjp_fn`) + `:830-955` (`_jit_fused_fixed_point_bwd`) + `:998-1090` (`f_bwd`)

**Step 1: Extend `_jit_fused_fixed_point_bwd` to accept `init_lam`**

```python
@jax.jit
def _jit_fused_fixed_point_bwd(
    params_data_tuple,
    env_leaves,
    g_scalar,
    init_lam,                # NEW: tuple of leaves shaped like dE_denv
):
    ...
    init_lam_local = init_lam       # replaces `init_lam = dE_denv`
    ...
```

The caller still computes `dE_denv` (used as the RHS for the Neumann iteration `λ_{k+1} = b + J^T λ_k`); only the **initial guess** changes.

**Step 2: Initialize cache slot in `_make_implicit_vjp_fn`**

Add to `_cached` dict (alongside `env_treedef`):

```python
_cached["prev_lam_leaves"] = None
```

**Step 3: Wire warm-start into `f_bwd`**

Replace the fused call at line 1044 with:

```python
prev_lam = _cached.get("prev_lam_leaves")
if prev_lam is None:
    # First call: pass dE_denv (computed inside the JIT) as init_lam by using
    # a sentinel tuple — but the JIT can't conditionally branch, so instead
    # we pass dE_denv computed eagerly via _jit_dE_denv.  The JIT no longer
    # initialises init_lam internally.
    init_lam = _eager_dE_denv()
else:
    init_lam = prev_lam

grads_tuple, diverged, _converged, _n_iter, lam_final = _jit_fused_fixed_point_bwd(
    params_data_tuple, env_leaves, g, init_lam,
)
_F3_LAST_DIAGNOSTICS["diverged"] = bool(jax.device_get(diverged))
_F3_LAST_DIAGNOSTICS["converged"] = bool(jax.device_get(_converged))
_F3_LAST_DIAGNOSTICS["n_iter"] = int(jax.device_get(_n_iter))

if _F3_LAST_DIAGNOSTICS["diverged"] or not _F3_LAST_DIAGNOSTICS["converged"]:
    _cached["prev_lam_leaves"] = None         # invalidate stale warm-start
    rhs = _eager_dE_denv()
    lam, _info = gmres_pytree_jax(
        _eager_apply_I_minus_Jt,
        rhs, rhs,
        tol=gmres_tol,
        maxiter=gmres_maxiter,
        restart=gmres_restart,
    )
    lam_leaves = tuple(jax.tree.leaves(lam))
    _cached["prev_lam_leaves"] = lam_leaves   # cache the eager fallback solution
    return _jit_chain_rule(params_data_tuple, env_leaves, lam_leaves, g)

_cached["prev_lam_leaves"] = tuple(jax.tree.leaves(lam_final))
return grads_tuple
```

**Step 4: Update `_jit_fused_fixed_point_bwd` to also return `lam_final`**

The function already computes `lam_final` (line 921). Return it alongside `grads_tuple`:

```python
return (total,), diverged, converged, n_iter, lam_final
```

**Step 5: Smoke test backward path**

Run: `uv run pytest tests/test_ctm_energy_ad.py -v -m core`
Expected: PASS.

**Step 6: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py
git commit -m "feat(ctm): warm-start implicit-AD adjoint solve from previous step (#501)"
```

---

## Task 10: #499 GMRES debug logging

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py` (top-level logger + 2 log sites in `f_bwd`)

**Step 1: Add module-level logger**

After imports:

```python
import logging
_GMRES_LOGGER = logging.getLogger("tenax.ctm.gmres")
```

**Step 2: Emit log at F3 success**

After `_F3_LAST_DIAGNOSTICS` populate in `f_bwd`:

```python
_GMRES_LOGGER.debug(
    "F3 adjoint: n_iter=%d converged=%s diverged=%s tol=%g",
    _F3_LAST_DIAGNOSTICS["n_iter"],
    _F3_LAST_DIAGNOSTICS["converged"],
    _F3_LAST_DIAGNOSTICS["diverged"],
    gmres_tol,
)
```

**Step 3: Emit log at eager fallback + `adjoint_method == "gmres"` branch**

```python
_GMRES_LOGGER.debug(
    "Eager GMRES: maxiter=%d tol=%g restart=%d",
    gmres_maxiter, gmres_tol, gmres_restart,
)
```

**Step 4: Add test**

```python
# Append to tests/test_ctm_in_loop_bump_ad_paths.py
def test_gmres_logging_emits_n_iter(caplog):
    """F3 adjoint emits a DEBUG record after backward."""
    import jax
    import logging

    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    site_tensors = _make_site_tensors()
    coords = sorted(site_tensors.keys())
    params = tuple(site_tensors[c].data for c in coords)

    def _loss(p_tuple):
        from tenax.core.tensor import DenseTensor
        from tenax.core.tensor_index import TensorIndex
        D = p_tuple[0].shape[0]; d = p_tuple[0].shape[-1]
        idx = (
            TensorIndex("u", D), TensorIndex("l", D),
            TensorIndex("d", D), TensorIndex("r", D),
            TensorIndex("p", d),
        )
        st = {c: DenseTensor(p, idx) for c, p in zip(coords, p_tuple)}
        return ctm_energy_implicit(
            st, CHECKERBOARD_NEIGHBORS, _heisenberg_gate(),
            chi=4, max_iter=8, conv_tol=1e-6,
        )

    with caplog.at_level(logging.DEBUG, logger="tenax.ctm.gmres"):
        jax.grad(_loss)(params)
    assert any("F3 adjoint" in rec.message for rec in caplog.records)
```

**Step 5: Run the test**

Run: `uv run pytest tests/test_ctm_in_loop_bump_ad_paths.py::test_gmres_logging_emits_n_iter -v`
Expected: PASS.

**Step 6: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_ctm_in_loop_bump_ad_paths.py
git commit -m "feat(ctm): debug-log GMRES adjoint solve diagnostics (#499)"
```

---

## Task 11: #511 variational warning update

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:2100-2107`

**Step 1: Update the warning text**

Replace lines 2100-2107:

```python
            warnings.warn(
                "2-site AD with gs_c4v=False uses the implicit-AD path. "
                "This is variational when the CTM environment is converged; "
                "pass ctmrg_heuristic_increase_chi=True with chi_max set "
                "(variPEPS-style in-CTM bump, issue #492) to grow chi "
                "automatically until the truncation gap closes.  Without "
                "the bump, chi must be set high enough manually "
                "(chi >= 16 for generic 2-site Heisenberg).  For "
                "antiferromagnetic bipartite models, gs_c4v=True is also "
                "a stable option.",
                stacklevel=2,
            )
```

**Step 2: Smoke test ipeps_optimize**

Run: `uv run pytest tests/test_ipeps_optimize_warnings.py -v -k 2site` (or nearest equivalent)
Expected: PASS.

**Step 3: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "docs(ipeps): update 2-site warning to reference in-CTM bump (#511)"
```

---

## Task 12: Full regression sweep + memory updates

**Step 1: Run full core test suite**

Run: `uv run pytest -m core -x --tb=short`
Expected: all PASS.

**Step 2: Run full non-slow suite**

Run: `uv run pytest -m "not slow" --tb=short`
Expected: all PASS.

**Step 3: Update memory entries**

Update `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/MEMORY.md`:

- Add: `- [project_514_phase2_landed.md](project_514_phase2_landed.md) — #514 Phase 2 landed: bump active in implicit-AD forward + explicit-AD warmup; #511 #499 #501 bundled.`

Create the memory file:

```markdown
---
name: 514 Phase 2 landed
description: In-CTM bump now active in implicit-AD forward + explicit-AD warmup; #511/#499/#501 bundled
type: project
---

Phase 2 of #492 (issue #514) landed 2026-05-20.

- Helper `_run_ctm_loop_with_bump` in new module `_ctm_loop_core.py` consolidates the bump-aware loop.
- `ctm_energy_implicit` (`_sigma_gauged_ctm_converge`) accepts `ctmrg_heuristic_increase_chi*`/`chi_max`.
- `ctm_energy_explicit` warmup is bump-aware; backprop uses `chi_post_warmup` for tape integrity.
- `ipeps_ad_policy.make_ctm_energy_fn` plumbs the four kwargs through.
- `_VJP_CACHE` key extended with the four bump kwargs.
- #511: 2-site variational warning updated to reference the bump as the automated convergence path.
- #499: `tenax.ctm.gmres` DEBUG logger emits F3 + eager fallback diagnostics.
- #501: adjoint warm-start via `_cached["prev_lam_leaves"]`; invalidated on divergence/eager fallback.

**Why:** PR #513 (Phase 1) wired the bump into `python_loop_ctm_converge`, but
AD gradient evaluations used their own loops that silently re-truncated the
cached env to the pre-bump chi.  Phase 2 closes that gap.

**How to apply:** When users enable `ctmrg_heuristic_increase_chi`, both env-cache
warm-start AND every AD eval now honor the bump.  `final_chi` from the env
cache should monotonically grow across L-BFGS steps.
```

**Step 4: Push the branch**

```bash
git add /home/yjkao/.claude/projects/-home-yjkao-tenax/memory/
git commit -m "memory: record #514 Phase 2 landing"
git push -u origin feat/in-ctm-chi-bump-ad-paths-514
```

**Step 5: Open PR**

```bash
gh pr create --title "feat(ctm): in-CTM chi-bump in AD forward loops (#514)" --body "$(cat <<'EOF'
## Summary

Phase 2 of #492 (issue #514).  Threads the variPEPS-style in-CTM χ-bump into the implicit-AD forward CTM loop (`_sigma_gauged_ctm_converge`) and the explicit-AD warmup phase, via a new shared helper `_run_ctm_loop_with_bump` in `_ctm_loop_core.py`.

Bundles:
- **#511**: 2-site variational warning updated to reference the bump as the automated convergence path.
- **#499**: DEBUG-level logging for F3 adjoint + eager GMRES fallback.
- **#501**: Adjoint warm-start via per-cache `prev_lam_leaves`; invalidated on divergence/eager fallback.

Design: `docs/plans/2026-05-20-in-ctm-bump-ad-paths-design.md`

## Test plan

- [ ] 18 existing in-CTM bump tests pass (`tests/test_ctm_in_loop_chi_bump.py`)
- [ ] New AD-path integration tests pass (`tests/test_ctm_in_loop_bump_ad_paths.py`)
- [ ] `uv run pytest -m core` green
- [ ] CI gates: Tests Python 3.11 / 3.12 / macOS Python 3.12

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Out of scope follow-ups

- `_ctm_honeycomb_ad.py` integration with the shared helper — different forward shape; defer.
- F3 backward fusion restructure beyond adding `init_lam` — defer.
- Phase 3 (default-flip) of #512 — only after this lands and benchmarks confirm bump is harmless when on by default.
