# Reduced-corner QR-CTMRG — Phase 2 (dense multisite + implicit-diff AD) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `projector_method="qr"` (the Phase 1 reduced-corner QR isometry) work end-to-end under the production **implicit-diff AD** (`optimize_gs_ad`) on **dense multisite** unit cells.

**Architecture:** Add a `regularized_qr` custom-VJP (stable QR backward near rank-deficiency), wire it into the `"qr"` AD-tracer projector path, add a `recipe` knob so the implicit-diff forward *and* fixed-point-adjoint sweeps run `recipe="1x1"` (which already honors `projector_method`), then validate multisite energy, gradient parity, and optimization. Dense only; block-sparse is Phase 3.

**Tech Stack:** JAX (`jax.custom_vjp`, `jax.test_util.check_grads`, `jnp.linalg.qr`), Tenax implicit-diff CTM-AD (`ctm_energy_implicit`, `_make_jit_ctm_step`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-reduced-corner-qr-ctmrg-phase2-multisite-implicit-ad-570.md`

---

## Key existing facts (verified)

- The `"qr"` **AD-tracer** sub-branch of `_compute_projector_tensor` (`_ctm_projector.py`, ~line 1069–1100) differentiates a **raw** `Q, R = jnp.linalg.qr(M)` (line ~1075) where `M = concat(C1g, C4g)` is `(fused, 2χ)`, then `regularized_svd(R)` (line ~1094), `P = Q @ U_R[:, :k]`. The raw big-QR backward is the instability Phase 1's spike found (`examples/probe_qr_vjp_stability_570.py`: near-rank-deficient FAILS). `regularized_svd` on the small `R` is already stable.
- `regularized_svd` (`_ad_primitives.py:331`) is the custom-VJP template: `@partial(jax.custom_vjp)`, `_fwd` returns `(out, residuals)`, `_bwd(residuals, g) -> (dM,)`, `.defvjp(...)`.
- Implicit-diff path: `ctm_energy_implicit` (`_ctm_energy_ad.py:337`) already threads `projector_method` through forward (`jit_step`, line 558) and backward (`jit_step_bwd`, line 974); both call `_make_jit_ctm_step` (`_ctm_python_loop.py:59`) → `_ctm_tensor_sweep_multisite(...)` (line 107) **without `recipe`** → defaults to `"2x2"` (Fishman, ignores `projector_method`). QR warm-up already present (`_ctm_energy_ad.py:571`). GMRES adjoint fallback exists (`:1323`).
- `iPEPSConfig` (`ipeps_config.py`) has `gs_projector_method` (line 571) and `gs_explicit_ad_*` fields to mirror for a new `gs_recipe`.
- 2-site fixtures: `ctm_tensor_2site`, `compute_energy_ctm_tensor_2site`, `tests/test_ctm_tensor.py` (`test_2site_*`), `small_peps_pair_dense` + `heisenberg_gate()` in `tests/conftest.py`.

Run: `JAX_PLATFORMS=cpu uv run pytest -m core` (fast); targeted files as noted.

---

## File structure

- **Modify** `src/tenax/algorithms/_ad_primitives.py` — add `regularized_qr` custom-VJP.
- **Modify** `src/tenax/algorithms/_ctm_projector.py` — route the `"qr"` AD-tracer `jnp.linalg.qr(M)` through `regularized_qr`.
- **Modify** `src/tenax/algorithms/_ctm_python_loop.py` — `recipe` param on `_make_jit_ctm_step`.
- **Modify** `src/tenax/algorithms/_ctm_energy_ad.py` — `recipe` param on `ctm_energy_implicit`, threaded to both step builders.
- **Modify** `src/tenax/algorithms/ipeps_config.py` + `ipeps_ad_policy.py` — `gs_recipe` field + wiring.
- **Create** `examples/probe_regularized_qr_vjp_570.py` — Task 1 spike.
- **Create** `tests/test_regularized_qr.py` — regularized_qr unit tests.
- **Extend** `tests/test_reduced_corner_qr.py` — multisite forward + AD/gradient tests.

---

## Task 1: SPIKE — `regularized_qr` backward (gates all AD)

Establish a QR custom-VJP whose backward is stable near rank-deficiency, validated by `check_grads`.

**Files:** Create `examples/probe_regularized_qr_vjp_570.py`

- [ ] **Step 1: Write the spike**

The standard thin-QR (`M = QR`, `M` is `m×n`, `m≥n`) reverse-mode VJP is
`M̄ = [Q̄ + Q · copyltu(Qᴴ Q̄ − R̄ Rᴴ)] R⁻ᴴ` with `copyltu(X) = tril(X) + tril(X,-1)ᴴ`. The
instability is the `R⁻ᴴ` solve when `diag(R)→0`. Regularize by flooring `diag(R)` in the solve.

```python
"""SPIKE (#570 Phase 2): a QR custom-VJP stable near rank-deficiency.

Run: JAX_PLATFORMS=cpu uv run python examples/probe_regularized_qr_vjp_570.py
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

_R_FLOOR = 1e-12


def _copyltu(X):
    L = jnp.tril(X)
    return L + jnp.tril(X, -1).conj().T


@partial(jax.custom_vjp)
def regularized_qr(M):
    return jnp.linalg.qr(M)


def _fwd(M):
    Q, R = jnp.linalg.qr(M)
    return (Q, R), (Q, R)


def _bwd(residuals, g):
    Q, R = residuals
    dQ, dR = g
    # copyltu(Q^H dQ - dR R^H); solve with regularized R^{-H}.
    Mbar = Q.conj().T @ dQ - dR @ R.conj().T
    X = dQ + Q @ _copyltu(Mbar)
    # Regularized triangular solve X @ R^{-H}: floor tiny diag(R).
    d = jnp.diag(R)
    safe = jnp.where(jnp.abs(d) > _R_FLOOR, d, _R_FLOOR)
    R_reg = R - jnp.diag(d) + jnp.diag(safe)
    dM = jax.scipy.linalg.solve_triangular(
        R_reg.conj().T, X.conj().T, lower=True
    ).conj().T
    return (dM,)


regularized_qr.defvjp(_fwd, _bwd)


def _rank_deficient(key, n, drop):
    A = jax.random.normal(key, (n, n))
    U, s, Vh = jnp.linalg.svd(A)
    s = s.at[n - drop:].set(0.0)
    return (U * s) @ Vh


def _scalar(fn, M):
    Q, R = fn(M)
    return jnp.real(jnp.sum(Q) + jnp.sum(R))


def main():
    key = jax.random.PRNGKey(0)
    for label, M in [
        ("well-conditioned 12x12", jax.random.normal(key, (12, 12))),
        ("tall 16x8", jax.random.normal(key, (16, 8))),
        ("near-rank-deficient 12x12", _rank_deficient(key, 12, 4)),
    ]:
        try:
            check_grads(lambda m: _scalar(regularized_qr, m), (M,),
                        order=1, modes=["rev"], atol=1e-4, rtol=1e-4)
            print(f"PASS  {label}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {label}: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `JAX_PLATFORMS=cpu uv run python examples/probe_regularized_qr_vjp_570.py`
Expected: PASS on all three (well-conditioned + tall match the unregularized result; near-rank-deficient now passes where Phase 1's raw QR failed). If near-rank-deficient still fails, iterate the backward (the `copyltu` sign/conjugation and the floor) until it passes — this is the research step.

- [ ] **Step 3: Record the validated backward in the spec**

Append a `## Phase 2 Task 1 result` block to the spec with the probe output and the exact backward formula that passed (so Task 2 productionizes it verbatim).

- [ ] **Step 4: Commit**

```bash
git add examples/probe_regularized_qr_vjp_570.py docs/superpowers/specs/2026-06-10-reduced-corner-qr-ctmrg-phase2-multisite-implicit-ad-570.md
git commit -m "spike(#570): regularized_qr backward — stable QR VJP near rank-deficiency"
```

---

## Task 2: `regularized_qr` in `_ad_primitives.py` + unit test

**Files:** Modify `src/tenax/algorithms/_ad_primitives.py`; Create `tests/test_regularized_qr.py`

- [ ] **Step 1: Write the failing unit test**

```python
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ad_primitives import regularized_qr


def _scalar(M):
    Q, R = regularized_qr(M)
    return jnp.real(jnp.sum(Q) + jnp.sum(R))


def test_regularized_qr_grads_well_conditioned():
    M = jax.random.normal(jax.random.PRNGKey(0), (12, 8))
    check_grads(_scalar, (M,), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)


def test_regularized_qr_matches_plain_qr_forward():
    M = jax.random.normal(jax.random.PRNGKey(1), (10, 6))
    Q, R = regularized_qr(M)
    Q0, R0 = jnp.linalg.qr(M)
    np.testing.assert_allclose(Q, Q0, atol=1e-12)
    np.testing.assert_allclose(R, R0, atol=1e-12)


def test_regularized_qr_grads_rank_deficient_finite():
    A = jax.random.normal(jax.random.PRNGKey(2), (12, 12))
    U, s, Vh = jnp.linalg.svd(A)
    s = s.at[8:].set(0.0)
    M = (U * s) @ Vh
    g = jax.grad(_scalar)(M)
    assert jnp.all(jnp.isfinite(g))  # no NaN/Inf through near-rank-deficient QR
```

- [ ] **Step 2: Run to verify it fails** — `JAX_PLATFORMS=cpu uv run pytest tests/test_regularized_qr.py -v` → `ImportError`.

- [ ] **Step 3: Implement `regularized_qr`** in `_ad_primitives.py` (after `regularized_svd`, ~line 358) using the Task-1-validated backward. Mirror the `regularized_svd` structure (`@partial(jax.custom_vjp)`, `_fwd`/`_bwd`/`.defvjp`). Use the exact `_copyltu` + regularized triangular-solve from Task 1's verdict. Add `regularized_qr` to the module's exports/`__all__` if one exists.

- [ ] **Step 4: Run to verify it passes** — `... pytest tests/test_regularized_qr.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ad_primitives.py tests/test_regularized_qr.py
git commit -m "feat(#570): regularized_qr custom-VJP (stable QR backward) + tests"
```

---

## Task 3: Route the `"qr"` AD-tracer path through `regularized_qr`

**Files:** Modify `src/tenax/algorithms/_ctm_projector.py` (the `"qr"` AD-tracer sub-branch, ~line 1069–1100); Test `tests/test_reduced_corner_qr.py`

- [ ] **Step 1: Write the failing differentiability test**

```python
def test_qr_projector_is_differentiable_dense():
    """jax.grad through the dense 'qr' projector dispatch is finite (uses
    regularized_qr under tracing)."""
    import jax
    import jax.numpy as jnp
    from tenax.algorithms._ctm_projector import _compute_projector_tensor
    C1g, C4g = _build_dense_enlarged_corners(chi=6)

    def loss(scale):
        c1 = C1g.__class__(C1g._data * scale, C1g.indices)
        P1, _P2, _eps = _compute_projector_tensor(c1, C4g, 6, "qr", None, "auto")
        return jnp.real(jnp.sum(jnp.abs(P1._data) ** 2))

    g = jax.grad(loss)(1.3)
    assert jnp.isfinite(g)
```

(Adjust the perturbation to whatever cleanly produces tracers through `_compute_projector_tensor`'s qr branch; the point is to exercise the AD-tracer sub-branch.)

- [ ] **Step 2: Run to verify it fails or is unstable** — `... pytest tests/test_reduced_corner_qr.py -k differentiable -v`. (May already pass via the raw-QR path on a well-conditioned input; the substantive change is replacing the unstable backward. If it passes pre-change on this input, keep it as a regression guard and proceed.)

- [ ] **Step 3: Replace the raw QR in the AD-tracer path**

In `_ctm_projector.py`, in the `"qr"` branch's tracer sub-branch, change `Q, R = jnp.linalg.qr(M)` (line ~1075) to:

```python
from tenax.algorithms._ad_primitives import regularized_qr
Q, R = regularized_qr(M)
```

Leave the rest (`_gauge_fix_qr_dense`, `regularized_svd(R)`, `Q @ U_R[:, :k]`) unchanged. Do **not** touch the non-tracer `_reduced_qr_projector` path (eager forward stays plain `jnp.linalg.qr`).

- [ ] **Step 4: Run to verify it passes** — `... pytest tests/test_reduced_corner_qr.py -v` (all prior + new pass).

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_projector.py tests/test_reduced_corner_qr.py
git commit -m "feat(#570): route dense 'qr' AD-tracer projector through regularized_qr"
```

---

## Task 4: Multisite forward energy validation (physics gate)

**Files:** Test `tests/test_reduced_corner_qr.py`

- [ ] **Step 1: Write the 2-site energy test**

```python
@pytest.mark.algorithm
@pytest.mark.parametrize("chi", [6, 10])
def test_reduced_qr_energy_matches_eigh_2site_heisenberg_D2(chi):
    """recipe='1x1'+qr converged energy matches eigh on a 2-site (A!=B) dense
    Heisenberg cell; gap shrinks with chi."""
    e_eigh = _heisenberg_D2_2site_energy(chi=chi, projector_method="eigh")
    e_qr = _heisenberg_D2_2site_energy(chi=chi, projector_method="qr")
    assert abs(e_qr - e_eigh) < 1e-3
```

Implement `_heisenberg_D2_2site_energy` reusing `ctm_tensor_2site` + `compute_energy_ctm_tensor_2site` and the `test_ctm_tensor.py` 2-site (A≠B) construction (`small_peps_pair_dense` / the sublattice-rotated Heisenberg pair). Add the gap-shrinks assertion (`gap(10) <= gap(6) + 1e-12`).

- [ ] **Step 2: Run** — `... pytest tests/test_reduced_corner_qr.py -k 2site_heisenberg -v` → pass. (If the gap is large, the single-isometry multisite assumption needs revisiting — STOP and report, don't loosen tolerance.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_reduced_corner_qr.py
git commit -m "test(#570): reduced-corner QR vs eigh energy on 2-site Heisenberg (multisite forward)"
```

---

## Task 5: `recipe` knob into the implicit-diff path

**Files:** Modify `_ctm_python_loop.py`, `_ctm_energy_ad.py`, `ipeps_config.py`, `ipeps_ad_policy.py`; Test `tests/test_reduced_corner_qr.py`

- [ ] **Step 1: Write the failing wiring test**

```python
def test_implicit_ad_recipe_threads_to_sweep(monkeypatch):
    """ctm_energy_implicit(recipe='1x1') makes the CTM sweep use recipe='1x1'
    (so projector_method='qr' is honored), not the hardcoded '2x2'."""
    import tenax.algorithms._ctm_python_loop as loop
    seen = {}
    orig = loop._ctm_tensor_sweep_multisite
    def spy(*a, **k):
        seen.setdefault("recipe", k.get("recipe"))
        return orig(*a, **k)
    monkeypatch.setattr(loop, "_ctm_tensor_sweep_multisite", spy)
    # drive a minimal ctm_energy_implicit(..., recipe="1x1", projector_method="qr")
    ...  # build a tiny dense 1-site state + neighbors; call ctm_energy_implicit
    assert seen["recipe"] == "1x1"
```

- [ ] **Step 2: Run to verify it fails** — recipe is `None`/`"2x2"`, not `"1x1"`.

- [ ] **Step 3: Implement the knob**

- `_ctm_python_loop.py`: add `recipe: str = "2x2"` to `_make_jit_ctm_step` and pass `recipe=recipe` into the `_ctm_tensor_sweep_multisite(...)` call (line ~107).
- `_ctm_energy_ad.py`: add `recipe: str = "2x2"` to `ctm_energy_implicit` (line 337); pass it to **both** `_make_jit_ctm_step(neighbors, recipe=recipe)` calls (forward line 558, backward line 974) and any intermediate helpers that build steps (grep `_make_jit_ctm_step` in the file).
- `ipeps_config.py`: add `gs_recipe: str = "2x2"` near `gs_projector_method` (line 571).
- `ipeps_ad_policy.py`: thread `gs_recipe` into the implicit-AD call the same way `gs_projector_method` is (grep `gs_projector_method` there).
- Default `"2x2"` everywhere preserves current behavior.

- [ ] **Step 4: Run to verify it passes** — wiring test green.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_python_loop.py src/tenax/algorithms/_ctm_energy_ad.py src/tenax/algorithms/ipeps_config.py src/tenax/algorithms/ipeps_ad_policy.py tests/test_reduced_corner_qr.py
git commit -m "feat(#570): recipe knob (gs_recipe) into implicit-diff CTM-AD path"
```

---

## Task 6: Gradient parity (the AD gate)

**Files:** Test `tests/test_reduced_corner_qr.py`

- [ ] **Step 1: Write the gradient-parity test**

```python
@pytest.mark.algorithm
def test_implicit_qr_gradient_matches_fd_and_eigh():
    # Energy as a function of one site-tensor parameter via implicit-diff AD.
    g_qr_ad = _implicit_energy_grad(recipe="1x1", projector_method="qr")
    g_qr_fd = _implicit_energy_grad_fd(recipe="1x1", projector_method="qr", eps=1e-5)
    np.testing.assert_allclose(g_qr_ad, g_qr_fd, atol=1e-4, rtol=1e-3)
    g_eigh_ad = _implicit_energy_grad(recipe="1x1", projector_method="eigh")
    np.testing.assert_allclose(g_qr_ad, g_eigh_ad, atol=1e-3, rtol=1e-2)
```

Implement `_implicit_energy_grad` via `jax.grad` of `ctm_energy_implicit` (recipe/projector_method threaded) on small dense Heisenberg D=2; `_implicit_energy_grad_fd` via central differences. Reuse the smallest dense iPEPS fixture available.

- [ ] **Step 2: Run** — `... -k gradient_matches_fd_and_eigh -v` → pass. (If QR-AD diverges from FD, the regularized_qr backward or the recipe threading is wrong — investigate, don't loosen.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_reduced_corner_qr.py
git commit -m "test(#570): implicit-AD QR gradient parity (FD + eigh) on Heisenberg D2"
```

---

## Task 7: Optimization + adjoint convergence

**Files:** Test `tests/test_reduced_corner_qr.py`

- [ ] **Step 1: Write the optimization test**

```python
@pytest.mark.algorithm
def test_optimize_gs_ad_qr_1x1_converges():
    """A short optimize_gs_ad run with gs_recipe='1x1' + gs_projector_method='qr'
    decreases the energy, stays finite, and tracks the eigh result."""
    e0, e_final_qr = _short_optimize(gs_recipe="1x1", gs_projector_method="qr", steps=5)
    assert np.isfinite(e_final_qr) and e_final_qr <= e0 + 1e-9     # energy decreases
    _e0e, e_final_eigh = _short_optimize(gs_recipe="1x1", gs_projector_method="eigh", steps=5)
    assert abs(e_final_qr - e_final_eigh) < 5e-3                   # tracks eigh
```

Implement `_short_optimize` via `optimize_gs_ad` (or the project's GS-optimization entry) on small dense Heisenberg D=2 with `iPEPSConfig(gs_recipe=..., gs_projector_method=..., gs_implicit_ad=True)`, a few steps. The implicit adjoint converging-or-GMRES-falling-back-cleanly is exercised here (assert no NaN; if the run logs an adjoint-divergence-then-GMRES, that is acceptable).

- [ ] **Step 2: Run** — `... -k optimize_gs_ad_qr -v` → pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_reduced_corner_qr.py
git commit -m "test(#570): optimize_gs_ad with QR + 1x1 recipe converges (implicit AD)"
```

---

## Task 8: Regression + docs + PR

**Files:** Docs (`ipeps_config.py`, `docs/guide/algorithms/ctm.md`, `README.md`); Test run

- [ ] **Step 1: Regression** — run the existing implicit-AD suite and core gate:
  `JAX_PLATFORMS=cpu uv run pytest -m core` and `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_tensor.py tests/test_reduced_corner_qr.py tests/test_regularized_qr.py -v`. Existing `"svd"`/`"eigh"` + `recipe="2x2"` tests must stay green (the new `gs_recipe`/`recipe` defaults are `"2x2"`).

- [ ] **Step 2: Mark fast new tests `core`** — give the fast new tests (`tests/test_regularized_qr.py` unit tests; the differentiability test) `@pytest.mark.core` so they run in the required gate (per the #596 lesson — don't leave the new production-AD guards out of `-m core`). Keep the slow energy/optimization tests `@pytest.mark.algorithm`. Verify `pytest -m core tests/test_regularized_qr.py tests/test_reduced_corner_qr.py` selects the fast ones.

- [ ] **Step 3: Docs** — document `gs_recipe` (and that `gs_recipe="1x1"` + `gs_projector_method="qr"` enables QR-CTMRG under implicit AD, dense; Phase 2) in `ipeps_config.py`, `docs/guide/algorithms/ctm.md`, `README.md`. Default stays `"2x2"`/`"svd"`.

- [ ] **Step 4: Commit + PR**

```bash
git add -A && git commit -m "docs(#570): document gs_recipe + mark fast QR-AD tests core (Phase 2)"
git push -u origin feat/qr-ctmrg-phase2-570
gh pr create --title "feat(#570): reduced-corner QR-CTMRG Phase 2 — dense multisite + implicit-diff AD" --body "$(cat <<'EOF'
Phase 2: projector_method="qr" works end-to-end under the production implicit-diff
AD (optimize_gs_ad) on dense multisite cells. Adds regularized_qr (stable QR backward
near rank-deficiency), routes the qr AD-tracer projector through it, and adds a
gs_recipe/recipe knob so the implicit forward+adjoint sweeps run recipe="1x1" (which
honors projector_method). Validated: 2-site energy vs eigh, FD/eigh gradient parity,
optimize_gs_ad convergence. Defaults unchanged (recipe="2x2", "svd"). Block-sparse is Phase 3.

Spec: docs/superpowers/specs/2026-06-10-reduced-corner-qr-ctmrg-phase2-multisite-implicit-ad-570.md
Plan: docs/superpowers/plans/2026-06-10-reduced-corner-qr-ctmrg-phase2-multisite-implicit-ad.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Auto-merge** — `gh pr merge --auto` (repo uses a merge queue; it sets the strategy and deletes the branch).

---

## Self-review notes

- **Spec coverage:** C1→Task 4; C2→Tasks 1,2; C2-wiring→Task 3; C3 (recipe knob)→Task 5; C4 (defaults)→Task 5 defaults. Tests T1→Task 1, T2→Task 2, T3→Task 4, T4→Task 6, T5→Task 7, T6→Task 8. Covered.
- **The research-risk step (Task 1, `regularized_qr` backward) is gated** by `check_grads` incl. near-rank-deficient before any wiring; the exact formula is validated, not assumed.
- **#596 lesson applied:** Task 8 Step 2 explicitly marks the fast new AD-guard tests `core`.
- **Consistency:** `regularized_qr(M) -> (Q, R)` and the `recipe`/`gs_recipe` names are used identically across Tasks 1–8. `_implicit_energy_grad`/`_short_optimize`/`_heisenberg_D2_2site_energy` helper names are placeholders to bind to real fixtures during execution (flagged).
- **Adjoint-convergence risk** is exercised by Task 7 (assert converge-or-GMRES-fallback, no NaN), per the spec's main AD risk.
