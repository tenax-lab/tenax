# Auto-χ_E Bump on CTM Truncation Error — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement variPEPS SciPost Lect. Notes 86 §2.8.2 truncation-error gating: when ϵ_T = ‖discarded SVs‖ exceeds a threshold, automatically increase the CTM environment bond dimension `χ_E` between L-BFGS optimizer steps. Prevents AD from exploiting CTM inaccuracies as artificially low energy (paper §2.8.2: "leading to false ground states with artificially low energy").

**Architecture:** (a) Compute ϵ_T inside `_compute_projector_tensor` SVD path; (b) thread up through `_ctm_tensor_sweep` → `ctm_tensor` as max-over-bonds; (c) add four `CTMConfig` fields with `chi_ramp` mutual-exclusion validation; (d) zero-pad C/T tensors from old `χ` to new `χ` and re-converge from padded warm-start; (e) trigger between L-BFGS steps in `optimize_gs_ad` (single-site dense path). Multisite and symmetric-tensor support are follow-up issues.

**Tech Stack:** JAX (`jnp.linalg.svd`, `jnp.zeros`, `jnp.pad`), `dataclasses`, `pytest`. No new external deps.

**Reference paper:** Naumann, Weerda, Rizzi, Eisert, Schmoll, *SciPost Phys. Lect. Notes* **86** (2024), arXiv:2308.12358 §2.8.2. Audit verdict: every load-bearing Sec. 2 axis already implemented in Tenax (`memory/project_varipeps_ad_audit_2026_05_08.md`); §2.8.1 (GKL iterative SVD) and §2.8.2 (this plan) are the two open ergonomic items.

**Out of scope (follow-up issues):**
- Symmetric-tensor (block-sparse) padding — needs per-charge-sector distribution of `chi_step`; SymmetricTensor SVD path also needs ϵ_T extraction.
- 3-site PESS multisite + generic multisite path — adds complexity in aggregating ϵ_T across multiple absorption coords and padding per-site envs.
- `eigh` and `qr` projector methods — only `svd` projector path is in scope; the other paths' truncation-error definition needs separate design.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `src/tenax/algorithms/_ctm_truncation_error.py` | **create** | Pure helper: ϵ_T from a singular-value vector. |
| `src/tenax/algorithms/_ctm_env_pad.py` | **create** | Pure helper: zero-pad dense `CTMTensorEnv` C/T tensors from `χ_old` → `χ_new`. |
| `src/tenax/algorithms/_ctm_projector.py` | **modify** | `_compute_projector_tensor` (and the `_svd_projector_dense` helper it calls) returns `(P_1, P_2, eps_T)` instead of `(P_1, P_2)`. |
| `src/tenax/algorithms/_ctm_tensor_convergence.py` | **modify** | `_ctm_tensor_sweep` and `ctm_tensor` thread max ϵ_T through and expose it on the result. |
| `src/tenax/algorithms/ipeps_config.py` | **modify** | Add `chi_auto_bump`, `chi_auto_bump_eps`, `chi_auto_bump_step`, `chi_max` fields to `CTMConfig`; validate mutual exclusion with `chi_ramp` in `__post_init__`. |
| `src/tenax/algorithms/ipeps_optimize.py` | **modify** | `optimize_gs_ad` (single-site dense path): inspect ϵ_T after each L-BFGS step; if above threshold, pad env + bump `chi` in a copy of the config + re-warm-start; do not bump mid-step. |
| `tests/test_ctm_truncation_error.py` | **create** | Unit tests for ϵ_T helper. |
| `tests/test_ctm_env_pad.py` | **create** | Unit tests for env-padding helper. |
| `tests/test_chi_auto_bump.py` | **create** | Integration test on Heisenberg square at deliberately too-low χ; assert χ rises and final E falls. |
| `docs/ipeps-code-paths.md` | **modify** | Note new auto-bump feature and link to §2.8.2 of the paper. |
| `README.md` | **modify** | One-line bullet under iPEPS features; example config snippet. |

Each file has one responsibility; no file's diff exceeds ~80 lines.

---

## Task 1: Truncation-error helper

**Files:**
- Create: `src/tenax/algorithms/_ctm_truncation_error.py`
- Test: `tests/test_ctm_truncation_error.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ctm_truncation_error.py
"""Unit tests for the CTM truncation-error helper.

ϵ_T is the variPEPS §2.8.2 quantity: the L2 norm of the *normalized*
discarded singular values, i.e. ‖S[χ:]‖ / ‖S‖. We test against analytic
spectra so behavior is reproducible without a full CTM run.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_truncation_error import compute_truncation_error


def test_zero_when_chi_covers_full_spectrum():
    s = jnp.array([1.0, 0.5, 0.25, 0.125])
    assert float(compute_truncation_error(s, chi=4)) == pytest.approx(0.0)
    assert float(compute_truncation_error(s, chi=10)) == pytest.approx(0.0)


def test_matches_normalized_l2_of_discarded_tail():
    s = jnp.array([1.0, 0.5, 0.25, 0.125])
    expected = float(jnp.sqrt(jnp.sum(s[2:] ** 2) / jnp.sum(s ** 2)))
    assert float(compute_truncation_error(s, chi=2)) == pytest.approx(expected)


def test_one_when_chi_zero():
    s = jnp.array([1.0, 0.5])
    assert float(compute_truncation_error(s, chi=0)) == pytest.approx(1.0)


def test_handles_zero_spectrum_safely():
    """A zero S vector (degenerate edge case) returns 0, not NaN."""
    s = jnp.zeros(4)
    assert float(compute_truncation_error(s, chi=2)) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_ctm_truncation_error.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'tenax.algorithms._ctm_truncation_error'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tenax/algorithms/_ctm_truncation_error.py
"""Truncation-error metric for CTM projectors.

Implements the variPEPS §2.8.2 indicator that gates auto-χ_E bumps.
"""
from __future__ import annotations

import jax.numpy as jnp


def compute_truncation_error(s: jnp.ndarray, chi: int) -> jnp.ndarray:
    """Normalized L2 norm of discarded singular values.

    ε_T = ‖s[χ:]‖_2 / ‖s‖_2, where s is the full SV vector returned by the
    SVD inside CTM projector construction. variPEPS §2.8.2 (SciPost Lect.
    Notes 86) recommends bumping χ_E whenever this exceeds ~1e-5.

    Returns a JAX scalar so it composes inside `jit`-compiled CTM sweeps.
    """
    s_full_norm_sq = jnp.sum(s ** 2)
    discarded = s[chi:]
    discarded_norm_sq = jnp.sum(discarded ** 2)
    safe_total = jnp.where(s_full_norm_sq > 0.0, s_full_norm_sq, 1.0)
    eps = jnp.sqrt(discarded_norm_sq / safe_total)
    return jnp.where(s_full_norm_sq > 0.0, eps, jnp.array(0.0, dtype=eps.dtype))
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_ctm_truncation_error.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_truncation_error.py tests/test_ctm_truncation_error.py
git commit -m "feat(ctm): add truncation-error helper for variPEPS §2.8.2 gating"
```

---

## Task 2: Surface ϵ_T from `_compute_projector_tensor`

**Files:**
- Modify: `src/tenax/algorithms/_ctm_projector.py:758-900` (SVD branch only)
- Test: `tests/test_ctm_truncation_error.py` (extend)

- [ ] **Step 1: Read the existing SVD branch**

```
Read src/tenax/algorithms/_ctm_projector.py (lines 758–900)
```
Find the line that returns `(P_1, P_2)` from the `projector_method == "svd"` branch. Identify the local `s` (singular value) variable that is currently truncated to `chi`. The change preserves the truncation but additionally captures `eps_T` from the *full* spectrum before truncation.

- [ ] **Step 2: Write the failing test (uses smallest realistic CTM input)**

Append to `tests/test_ctm_truncation_error.py`:

```python
import jax
from tenax.algorithms._ctm_projector import _compute_projector_tensor


def test_compute_projector_tensor_returns_eps_t():
    """Smoke test: SVD projector path returns (P_1, P_2, eps_T)."""
    key = jax.random.PRNGKey(0)
    chi_in, chi_target = 8, 4
    # Construct two random "corner" tensors whose product has a nontrivial
    # singular spectrum. Shapes mirror C1g, C4g in the half-system M = C1g†·C4g.
    C1 = jax.random.normal(key, (chi_in, chi_in))
    C4 = jax.random.normal(jax.random.fold_in(key, 1), (chi_in, chi_in))
    out = _compute_projector_tensor(
        C1, C4, chi=chi_target, projector_method="svd"
    )
    assert len(out) == 3, "expected (P_1, P_2, eps_T) tuple"
    P_1, P_2, eps_T = out
    assert eps_T.shape == ()
    assert float(eps_T) > 0.0  # truncation by half should give nonzero ε_T
```

- [ ] **Step 3: Run test to verify it fails**

```
uv run pytest tests/test_ctm_truncation_error.py::test_compute_projector_tensor_returns_eps_t -v
```
Expected: FAIL — current return is a 2-tuple, `len(out) == 3` fails.

- [ ] **Step 4: Modify `_compute_projector_tensor` SVD branch**

In `src/tenax/algorithms/_ctm_projector.py`, locate the SVD branch (around line 818 per the audit). Where the code currently does the SVD and truncates to `chi`, capture the full singular values *before* truncation and compute `eps_T`. Update the return to include it. Update the `eigh` and `qr` branches to return `eps_T = jnp.array(0.0)` (correct truncation-error definition for those paths is out of scope for this plan).

```python
# Inside _compute_projector_tensor, SVD branch (illustrative; preserve existing code):
from tenax.algorithms._ctm_truncation_error import compute_truncation_error

# Existing SVD on the half-system matrix M:
U, S, Vh = jnp.linalg.svd(M, full_matrices=False)
eps_T = compute_truncation_error(S, chi)        # <-- NEW
S_trunc = S[:chi]
# ... existing P_1 / P_2 construction using S_trunc, U[:, :chi], Vh[:chi] ...
return P_1, P_2, eps_T
```

For the `eigh` and `qr` branches at the same level:

```python
# After existing P_1, P_2 construction:
eps_T = jnp.array(0.0, dtype=jnp.result_type(P_1))   # placeholder — see plan scope
return P_1, P_2, eps_T
```

- [ ] **Step 5: Update every caller of `_compute_projector_tensor` to unpack three values**

```
grep -rn "_compute_projector_tensor" src/tenax tests
```
Each call site that does `P_1, P_2 = _compute_projector_tensor(...)` becomes `P_1, P_2, _eps_t = _compute_projector_tensor(...)`. Save the discarded `_eps_t` only at the call sites that aggregate (Task 3 wires the live ones).

- [ ] **Step 6: Run all CTM tests**

```
uv run pytest -m core tests/ -k "ctm or projector" -v
```
Expected: all green (the existing assertions don't depend on the return tuple length once unpacked).

- [ ] **Step 7: Commit**

```bash
git add src/tenax/algorithms/_ctm_projector.py tests/test_ctm_truncation_error.py
git commit -m "feat(ctm): surface ε_T from SVD projector path"
```

---

## Task 3: Aggregate ϵ_T across the CTM sweep

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py:75-101` (`_ctm_tensor_sweep`)
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py:300-389` (`ctm_tensor`)
- Test: `tests/test_chi_auto_bump.py` (create with one tiny test)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chi_auto_bump.py
"""Integration tests for variPEPS §2.8.2 auto-χ_E bump.

This file currently only exercises the ε_T plumbing on `ctm_tensor`. The
end-to-end optimizer test lives at the bottom and is added by Task 7.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor


def _random_site_tensor(D: int, d: int, key) -> jnp.ndarray:
    return jax.random.normal(key, (D, D, D, D, d))


def test_ctm_tensor_returns_eps_t_field():
    """`ctm_tensor` must expose max ε_T from the last sweep alongside the env."""
    key = jax.random.PRNGKey(0)
    A = _random_site_tensor(D=2, d=2, key=key)
    config = CTMConfig(chi=4, max_iter=10, min_iter=2, conv_tol=1e-6)
    result = ctm_tensor(A, config)
    # Three layouts are acceptable depending on existing API; pick one and
    # commit to it in the implementation step. We assume an attribute on
    # the result NamedTuple/dataclass.
    assert hasattr(result, "max_truncation_error"), (
        "ctm_tensor must expose `max_truncation_error` for §2.8.2 auto-bump"
    )
    eps = float(result.max_truncation_error)
    assert eps >= 0.0
    assert eps <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_chi_auto_bump.py::test_ctm_tensor_returns_eps_t_field -v
```
Expected: FAIL — `max_truncation_error` attribute does not exist.

- [ ] **Step 3: Read `ctm_tensor` to confirm its return type**

```
Read src/tenax/algorithms/_ctm_tensor_convergence.py:300-389
```
Identify whether it returns a `NamedTuple`, a dataclass, or a plain tuple. Add a `max_truncation_error: float` field of the same kind. (If it's a `NamedTuple`, define a new namedtuple with the extra field; if a dataclass, add a default-valued field.)

- [ ] **Step 4: Modify `_ctm_tensor_sweep` to thread max ε_T**

In `_ctm_tensor_sweep`, every call to `_compute_projector_tensor` now returns a third element. Track `max_eps = jnp.maximum(max_eps, eps_T_this_call)` across the sweep. Initialize `max_eps = jnp.array(0.0)` at sweep entry. Return `(env_new, max_eps)` instead of `env_new`.

- [ ] **Step 5: Modify `ctm_tensor` to expose the final-sweep value**

After the convergence loop, store the `max_eps` from the *last* sweep on the returned env / result struct. If the existing return is `CTMResult(env=..., n_iter=...)`, extend it to `CTMResult(env=..., n_iter=..., max_truncation_error=...)`. Update every caller.

- [ ] **Step 6: Run tests**

```
uv run pytest -m core tests/ -k "ctm" -v
```
Expected: all green, plus the new test passes.

- [ ] **Step 7: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_convergence.py tests/test_chi_auto_bump.py
git commit -m "feat(ctm): expose max ε_T on ctm_tensor result"
```

---

## Task 4: `CTMConfig` fields with mutual-exclusion validation

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py:14-100` (`CTMConfig` dataclass + `__post_init__`)
- Test: `tests/test_chi_auto_bump.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chi_auto_bump.py`:

```python
import pytest
from dataclasses import FrozenInstanceError  # noqa: F401  (kept in case CTMConfig becomes frozen)


def test_ctm_config_auto_bump_defaults_off():
    """Auto-bump must be opt-in to preserve existing behavior."""
    config = CTMConfig(chi=4)
    assert config.chi_auto_bump is False
    assert config.chi_auto_bump_eps == 1e-5
    assert config.chi_auto_bump_step == 2
    assert config.chi_max is None


def test_ctm_config_auto_bump_rejects_chi_ramp_combo():
    """`chi_ramp` is a deterministic schedule; reactive auto-bump conflicts."""
    with pytest.raises(ValueError, match="chi_ramp"):
        CTMConfig(
            chi=4,
            chi_auto_bump=True,
            chi_ramp=[(4, 10), (8, None)],
        )


def test_ctm_config_auto_bump_validates_step_positive():
    with pytest.raises(ValueError, match="chi_auto_bump_step"):
        CTMConfig(chi=4, chi_auto_bump=True, chi_auto_bump_step=0)


def test_ctm_config_auto_bump_validates_chi_max_above_chi():
    with pytest.raises(ValueError, match="chi_max"):
        CTMConfig(chi=4, chi_auto_bump=True, chi_max=2)
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_chi_auto_bump.py -v -k "ctm_config_auto_bump"
```
Expected: 4 failures (`unexpected keyword argument`).

- [ ] **Step 3: Add fields and validation**

Edit `src/tenax/algorithms/ipeps_config.py` `CTMConfig`:

```python
@dataclass
class CTMConfig:
    # ... existing fields up to chi_ramp ...
    chi_ramp: list[tuple[int, int | None]] | None = None

    # NEW: variPEPS §2.8.2 reactive auto-bump.
    chi_auto_bump: bool = False
    chi_auto_bump_eps: float = 1e-5
    chi_auto_bump_step: int = 2
    chi_max: int | None = None

    # ... existing fields after ...

    def __post_init__(self):
        # ... preserve existing validation ...
        if self.chi_auto_bump and self.chi_ramp is not None:
            raise ValueError(
                "chi_auto_bump and chi_ramp are mutually exclusive: "
                "chi_ramp is a deterministic schedule, chi_auto_bump is reactive"
            )
        if self.chi_auto_bump and self.chi_auto_bump_step <= 0:
            raise ValueError("chi_auto_bump_step must be a positive integer")
        if self.chi_max is not None and self.chi_max < self.chi:
            raise ValueError(
                f"chi_max ({self.chi_max}) must be >= chi ({self.chi})"
            )
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_chi_auto_bump.py -v -k "ctm_config_auto_bump"
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_config.py tests/test_chi_auto_bump.py
git commit -m "feat(config): add CTMConfig.chi_auto_bump fields with validation"
```

---

## Task 5: Dense env-padding helper

**Files:**
- Create: `src/tenax/algorithms/_ctm_env_pad.py`
- Test: `tests/test_ctm_env_pad.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ctm_env_pad.py
"""Unit tests for zero-padding `CTMTensorEnv` from χ_old → χ_new."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_env_pad import pad_dense_env_chi
from tenax.algorithms.ipeps_config import CTMTensorEnv


def _make_dummy_env(chi: int, D: int, d: int = 2, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 8)
    C = lambda k: jax.random.normal(k, (chi, chi))                      # corner
    T_h = lambda k: jax.random.normal(k, (chi, D, D, chi))              # horizontal edge
    T_v = lambda k: jax.random.normal(k, (chi, D, D, chi))              # vertical edge
    return CTMTensorEnv(
        C1=C(keys[0]), C2=C(keys[1]), C3=C(keys[2]), C4=C(keys[3]),
        T1=T_h(keys[4]), T2=T_v(keys[5]), T3=T_h(keys[6]), T4=T_v(keys[7]),
    )


def test_pad_extends_corner_axes_to_new_chi():
    env_old = _make_dummy_env(chi=4, D=2)
    env_new = pad_dense_env_chi(env_old, chi_new=6)
    assert env_new.C1.shape == (6, 6)
    assert env_new.C2.shape == (6, 6)
    assert env_new.C3.shape == (6, 6)
    assert env_new.C4.shape == (6, 6)


def test_pad_extends_edge_chi_axes_only():
    env_old = _make_dummy_env(chi=4, D=2)
    env_new = pad_dense_env_chi(env_old, chi_new=6)
    # Edges have shape (chi, D, D, chi) — D-axes unchanged.
    assert env_new.T1.shape == (6, 2, 2, 6)
    assert env_new.T4.shape == (6, 2, 2, 6)


def test_pad_preserves_existing_block():
    env_old = _make_dummy_env(chi=4, D=2)
    env_new = pad_dense_env_chi(env_old, chi_new=6)
    assert jnp.allclose(env_new.C1[:4, :4], env_old.C1)
    assert jnp.allclose(env_new.T1[:4, :, :, :4], env_old.T1)


def test_pad_fills_new_block_with_zero():
    env_old = _make_dummy_env(chi=4, D=2)
    env_new = pad_dense_env_chi(env_old, chi_new=6)
    assert float(jnp.max(jnp.abs(env_new.C1[4:, :]))) == 0.0
    assert float(jnp.max(jnp.abs(env_new.C1[:, 4:]))) == 0.0


def test_pad_noop_when_chi_unchanged():
    env_old = _make_dummy_env(chi=4, D=2)
    env_new = pad_dense_env_chi(env_old, chi_new=4)
    assert env_new.C1 is env_old.C1  # cheap identity check
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_ctm_env_pad.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the helper**

```python
# src/tenax/algorithms/_ctm_env_pad.py
"""Zero-pad a dense CTMTensorEnv from χ_old to χ_new.

Used by the variPEPS §2.8.2 auto-bump: between L-BFGS steps we increase χ_E
and re-converge from the padded warm-start (cheaper than a cold restart).
"""
from __future__ import annotations

import jax.numpy as jnp

from tenax.algorithms.ipeps_config import CTMTensorEnv


def _pad_chi_axes(arr: jnp.ndarray, chi_new: int, chi_axes: tuple[int, ...]) -> jnp.ndarray:
    pad_width = [(0, 0)] * arr.ndim
    for ax in chi_axes:
        pad_width[ax] = (0, chi_new - arr.shape[ax])
    return jnp.pad(arr, pad_width)


def pad_dense_env_chi(env: CTMTensorEnv, chi_new: int) -> CTMTensorEnv:
    """Return a new env with corner χ axes zero-padded to ``chi_new``.

    Corners (C1..C4) have shape (χ, χ) — both axes pad.
    Edges (T1, T3) and (T2, T4) have shape (χ, D, D, χ) — only axes 0 and 3 pad.
    No-op if ``chi_new`` matches the existing χ (returns the same env).
    """
    chi_old = env.C1.shape[0]
    if chi_new == chi_old:
        return env
    if chi_new < chi_old:
        raise ValueError(f"chi_new ({chi_new}) must be >= chi_old ({chi_old})")
    return CTMTensorEnv(
        C1=_pad_chi_axes(env.C1, chi_new, (0, 1)),
        C2=_pad_chi_axes(env.C2, chi_new, (0, 1)),
        C3=_pad_chi_axes(env.C3, chi_new, (0, 1)),
        C4=_pad_chi_axes(env.C4, chi_new, (0, 1)),
        T1=_pad_chi_axes(env.T1, chi_new, (0, 3)),
        T2=_pad_chi_axes(env.T2, chi_new, (0, 3)),
        T3=_pad_chi_axes(env.T3, chi_new, (0, 3)),
        T4=_pad_chi_axes(env.T4, chi_new, (0, 3)),
    )
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_ctm_env_pad.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_env_pad.py tests/test_ctm_env_pad.py
git commit -m "feat(ctm): add zero-pad warm-start helper for auto-χ_E bump"
```

---

## Task 6: Wire auto-bump into `optimize_gs_ad` outer loop

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (the single-site dense `optimize_gs_ad` driver, NOT the multisite path)

- [ ] **Step 1: Read the existing single-site driver**

```
Read src/tenax/algorithms/ipeps_optimize.py
```
Find the L-BFGS outer loop in the single-site dense path. Identify (a) where the env cache is read/written between optimizer steps, (b) where `loss_fn(params)` is called, (c) where the `CTMConfig` is held as a local. Confirm whether ϵ_T from `ctm_tensor` is reachable via the env cache or whether `loss_fn` needs to surface it.

- [ ] **Step 2: Add a helper that decides the new χ given the last sweep's ε_T**

In `ipeps_optimize.py`, add a private helper near the optimizer entry:

```python
import dataclasses

from tenax.algorithms._ctm_env_pad import pad_dense_env_chi


def _maybe_bump_chi(
    config: CTMConfig,
    env_cache: dict,
    last_eps_t: float,
) -> tuple[CTMConfig, dict]:
    """variPEPS §2.8.2 reactive χ_E bump.

    Returns a new (config, env_cache) pair if the threshold was tripped,
    otherwise the same pair unchanged. Padding only happens on bump.
    """
    if not config.chi_auto_bump:
        return config, env_cache
    if last_eps_t <= config.chi_auto_bump_eps:
        return config, env_cache
    chi_new = config.chi + config.chi_auto_bump_step
    if config.chi_max is not None:
        chi_new = min(chi_new, config.chi_max)
    if chi_new <= config.chi:
        return config, env_cache  # already at ceiling
    new_config = dataclasses.replace(config, chi=chi_new)
    if "envs" in env_cache:
        env_cache = {**env_cache, "envs": pad_dense_env_chi(env_cache["envs"], chi_new)}
    return new_config, env_cache
```

- [ ] **Step 3: Call the helper after every accepted L-BFGS step**

In the outer L-BFGS loop, after `loss_fn(params)` returns and the env cache has been refreshed (look for the existing `env_cache["envs"] = ...` write in `ipeps_optimize.py`), call the helper before the next iteration:

```python
# After the existing env_cache update for this step:
last_eps_t = float(env_cache.get("max_truncation_error", 0.0))
config, env_cache = _maybe_bump_chi(config, env_cache, last_eps_t)
```

The `loss_fn` closure must surface `max_truncation_error` into `env_cache`. If it doesn't yet, modify the closure (in the same file) to set `env_cache["max_truncation_error"] = float(ctm_result.max_truncation_error)` after the CTM call.

- [ ] **Step 4: Run the existing iPEPS optimizer tests**

```
uv run pytest -m core tests/ -k "optimize_gs_ad and not multisite" -v
```
Expected: all green; defaults preserved (auto-bump off).

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "feat(ipeps): wire auto-χ_E bump into optimize_gs_ad outer loop"
```

---

## Task 7: Integration test — Heisenberg square at deliberately too-low χ

**Files:**
- Modify: `tests/test_chi_auto_bump.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chi_auto_bump.py`:

```python
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_optimize import optimize_gs_ad
from tenax.algorithms.ipeps_config import IPEPSConfig


@pytest.mark.algorithm
def test_auto_bump_raises_chi_under_pressure():
    """At deliberately-too-low χ, auto-bump must raise it during optimization.

    Heisenberg square D=2 with χ=2 has truncation error well above 1e-5; the
    optimizer should detect this and bump χ at least once. We assert on the
    final config's χ rather than energy because the energy convergence path
    depends on how aggressively the bump fires.
    """
    gate = heisenberg_gate(d=2)
    A_init = jax.random.normal(jax.random.PRNGKey(0), (2, 2, 2, 2, 2))
    config = IPEPSConfig(
        ctm=CTMConfig(
            chi=2,
            chi_auto_bump=True,
            chi_auto_bump_eps=1e-5,
            chi_auto_bump_step=2,
            chi_max=8,
            max_iter=20,
            min_iter=2,
            conv_tol=1e-5,
        ),
        max_iter=3,  # only need a few outer steps to trip the bump
    )
    A_opt, e_opt, info = optimize_gs_ad(gate, A_init, config)
    assert info["final_chi"] > 2, (
        f"auto-bump never fired (final_chi={info['final_chi']})"
    )
    assert info["final_chi"] <= 8
```

If `optimize_gs_ad` does not currently return `info["final_chi"]`, the implementation step adds it.

- [ ] **Step 2: Run test to verify it fails (or pass-by-luck check)**

```
uv run pytest tests/test_chi_auto_bump.py::test_auto_bump_raises_chi_under_pressure -v
```
Expected: FAIL — either missing `info["final_chi"]` key or χ stays at 2.

- [ ] **Step 3: Surface `final_chi` from the optimizer**

In `ipeps_optimize.py`, ensure `optimize_gs_ad` returns an `info` dict containing `final_chi` (the live `config.chi` after all auto-bumps). If the function currently returns only `(A_opt, e_opt)`, extend it to `(A_opt, e_opt, info)`. **Search for every existing caller** and update tuple unpacking. (This is a public-API change — note it in the commit message.)

- [ ] **Step 4: Run test**

```
uv run pytest tests/test_chi_auto_bump.py::test_auto_bump_raises_chi_under_pressure -v
```
Expected: passes; `final_chi` ≥ 4.

- [ ] **Step 5: Run full core suite to catch broken callers**

```
uv run pytest -m core
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add tests/test_chi_auto_bump.py src/tenax/algorithms/ipeps_optimize.py
git commit -m "feat(ipeps): return final_chi info; integration test for auto-bump"
```

---

## Task 8: Docs + README + `__all__`

**Files:**
- Modify: `docs/ipeps-code-paths.md`
- Modify: `README.md`
- Modify: `src/tenax/__init__.py` (only if `pad_dense_env_chi` becomes public — confirm scope first)

- [ ] **Step 1: Add a short section to `docs/ipeps-code-paths.md`**

Append a new subsection under the iPEPS optimization section. Show the config snippet and link to variPEPS §2.8.2.

```markdown
### Auto-χ_E bump (variPEPS §2.8.2)

When the CTM truncation error `ε_T = ‖discarded SVs‖ / ‖SVs‖` exceeds
`chi_auto_bump_eps`, `optimize_gs_ad` increases `chi` by `chi_auto_bump_step`
between L-BFGS steps and zero-pads the cached environment. Disabled by
default (opt-in). Mutually exclusive with `chi_ramp`.

```python
config = IPEPSConfig(
    ctm=CTMConfig(
        chi=10,
        chi_auto_bump=True,
        chi_auto_bump_eps=1e-5,   # variPEPS Sec. 2.8.2 default
        chi_auto_bump_step=2,
        chi_max=40,
    ),
    ...
)
```

Reference: Naumann et al., SciPost Phys. Lect. Notes 86 (2024), §2.8.2.
```

- [ ] **Step 2: Add a one-line bullet to `README.md`**

Find the iPEPS feature list in `README.md` and add:

```markdown
- **Auto-χ_E bump (variPEPS §2.8.2):** opt-in reactive increase of CTM bond dimension when truncation error exceeds threshold (`CTMConfig.chi_auto_bump`).
```

- [ ] **Step 3: Run docs build**

```
cd docs && make html && cd ..
```
Expected: build completes without warnings about new section.

- [ ] **Step 4: Commit**

```bash
git add docs/ipeps-code-paths.md README.md
git commit -m "docs: document auto-χ_E bump (variPEPS §2.8.2)"
```

---

## Task 9: Open follow-up issues for deferred scope

- [ ] **Step 1: File two issues**

Use `gh` CLI:

```bash
gh issue create \
  --title "auto-χ_E bump: extend to symmetric-tensor (block-sparse) path" \
  --body "Follow-up to PR for variPEPS §2.8.2 auto-bump (single-site dense). \
The block-sparse path needs (a) ε_T extraction from SymmetricTensor SVD, \
(b) per-charge-sector distribution of chi_step in the padding helper. \
Plan: docs/superpowers/plans/2026-05-08-auto-chi-bump.md"

gh issue create \
  --title "auto-χ_E bump: extend to multisite + 3-site PESS paths" \
  --body "Follow-up to PR for variPEPS §2.8.2 auto-bump (single-site dense). \
The multisite path aggregates ε_T across multiple absorption coords and \
pads per-site envs. Plan: docs/superpowers/plans/2026-05-08-auto-chi-bump.md"
```

- [ ] **Step 2: Note issue numbers in plan footer**

Append the resulting issue numbers as a footer at the bottom of this plan file.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-05-08-auto-chi-bump.md
git commit -m "docs(plan): record follow-up issue numbers"
```

---

## Self-review checklist (run before opening PR)

- [ ] Every task lists exact files with line ranges where modifications happen.
- [ ] Every code step contains the actual code (no "TBD", no "similar to above").
- [ ] No method signature changes silently — Task 7 explicitly notes the public-API change to `optimize_gs_ad` and updates callers.
- [ ] All four config-field names are spelled identically across config, helper, optimizer, test, and docs.
- [ ] `chi_auto_bump=False` default preserves all existing behavior; full core test suite passes without code changes outside the explicitly-modified files.
- [ ] Symmetric-tensor and multisite paths are covered by follow-up issues, not silently broken.
- [ ] `chi_ramp + chi_auto_bump` combination raises `ValueError` (Task 4 test).
- [ ] Plan filed at `docs/superpowers/plans/2026-05-08-auto-chi-bump.md` and referenced from the follow-up issues.

---

## Open design questions (decide before kicking off implementation)

These were called out by the planner; defaults are sensible but worth a sanity check before code is written:

1. **`chi_auto_bump_eps` default = `1e-5`** matches variPEPS §2.8.2 verbatim. Some users may want `1e-4` for cheaper runs — leave at `1e-5` and document.
2. **`chi_auto_bump_step` default = `2`** is conservative (matches typical `chi_ramp` increments seen in `feedback_default_lbfgs`). A multiplicative bump (e.g. `×1.25`) is plausible but harder to reason about at small χ — stick with additive.
3. **`chi_max = None`** means unbounded growth, capped only by available memory. Users running on 7 GB Linux runners (`project_jax_cache_threshold`) should set an explicit cap.
4. **No mid-CTM bump.** The bump fires *between* L-BFGS steps so the implicit-AD GMRES linearization (`_ctm_energy_ad.py:255-353`) sees a fixed χ within each gradient evaluation.

---

## Follow-up Issues

- **#410** — auto-χ_E bump: extend to symmetric-tensor (block-sparse) path
- **#411** — auto-χ_E bump: extend to multisite + 3-site PESS paths
