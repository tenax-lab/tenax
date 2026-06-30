# Split-CTM 1-site Validation + CG Guard Lock-in — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add characterization/regression tests that (a) prove the 1-site split-CTM path is variationally correct with C4v, and (b) lock in the intended CG-on-split rejection at both guard layers — a pre-flip prerequisite for #463, with **no production-code changes**.

**Architecture:** All three tests exercise *existing* behavior (the split path already ships; the guards already exist), so they pass on first run — they are regression locks, not TDD-driven new code. The Part-1 test runs one `optimize_gs_ad` with the split path + `gs_c4v=True` on the sublattice-rotated Heisenberg gate and asserts the converged energy lands in the physical variational window `[−0.6694, −0.60]` per site. The Part-2 tests assert the two CG guards raise `NotImplementedError`.

**Tech Stack:** Python, JAX, pytest. Tenax `optimize_gs_ad` / `CTMConfig` / `iPEPSConfig`. Tests reuse helpers from `tests/test_split_ctm_fuse_flag.py`.

**Spec:** `docs/superpowers/specs/2026-06-29-463-split-ctm-1site-validation-cg-guard-design.md`

**Why these tests / measured anchors (D=2, χ=10, gs_c4v=True, grad_norm |g|<1e-3):**
- split+c4v converged to **−0.6505/site** (variational, +0.019 above QMC −0.6694).
- fused+c4v converged to **−0.6601/site**. Gap 0.0096 persists at tight |g| ⇒ genuine bounded #425, both physical.
- WITHOUT C4v split breaches to **−0.714** (below floor) — so the window test is meaningful only with `gs_c4v=True`.

---

## File Structure

- **Create** `tests/test_split_ctm_production_correctness.py` — the one `slow` Part-1 variational-window test (+ optional companion). Isolated so the slow optimizer run stays out of the `core`-marked `test_split_ctm_fuse_flag.py`.
- **Modify** `tests/test_split_ctm_fuse_flag.py` — append the two fast CG-guard regression tests (Part 2). This file is already `pytestmark = pytest.mark.core`; the guard tests are fast and belong there.
- **Delete** `_split_floor_one.py`, `_split_floor_confirm.py` — scratch probe scripts in the repo root (untracked); remove so the tree is clean.
- No `src/` changes.

---

## Task 1: CG guard layer-2 regression test (split AD `energy_fn` reject)

**Files:**
- Modify: `tests/test_split_ctm_fuse_flag.py` (append at end)

- [ ] **Step 1: Add the test**

Append to `tests/test_split_ctm_fuse_flag.py`:

```python
def test_split_energy_fn_guard_rejects_custom_energy_fn():
    """Layer-2 CG guard: the split AD entry points reject a custom energy_fn.

    CG runs by passing a coarse-grain energy_fn; the split path does not
    support it yet (compute_energy_cg_split exists but is not wired through
    the split AD). Lock the rejection so a refactor can't silently drop it.
    """
    from tenax.algorithms._split_ctm_energy_ad import (
        ctm_energy_split_explicit,
        ctm_energy_split_implicit,
    )

    A = _make_site(2, 2, seed=0)
    gate = _heisenberg_gate()
    for fn in (ctm_energy_split_implicit, ctm_energy_split_explicit):
        with pytest.raises(NotImplementedError, match="custom energy_fn"):
            fn(
                {(0, 0): A},
                SINGLE_SITE_NEIGHBORS,
                gate,
                chi=4,
                energy_fn=lambda *a: 0.0,
            )
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py::test_split_energy_fn_guard_rejects_custom_energy_fn -v`
Expected: PASS (the guard already exists; this locks it).

- [ ] **Step 3: Commit**

```bash
git add tests/test_split_ctm_fuse_flag.py
git commit -m "test(#463): lock split-AD energy_fn (CG layer-2) rejection"
```

---

## Task 2: CG guard layer-1 regression test (`cg_gates` optimizer reject)

**Files:**
- Modify: `tests/test_split_ctm_fuse_flag.py` (append at end)

- [ ] **Step 1: Add the test**

Append to `tests/test_split_ctm_fuse_flag.py`:

```python
def test_optimize_gs_ad_split_rejects_cg_gates():
    """Layer-1 CG guard: the split optimizer path rejects cg_gates up front.

    cg_gates couples to the fused CTMTensorEnv (compute_energy_cg uses the
    fused diagonal-RDM env); the split path raises rather than silently
    feeding a SplitCTMTensorEnv to fused-only machinery (ipeps_optimize.py).
    """
    import jax.numpy as jnp

    from tenax.algorithms.coarse_grain import honeycomb_cg_gates
    from tenax.algorithms.ipeps_config import CTMConfig as _CTMConfig
    from tenax.algorithms.ipeps_config import iPEPSConfig
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    cfg = iPEPSConfig(
        max_bond_dim=2,
        ctm=_CTMConfig(chi=8, fuse_virtual_legs=False, max_iter=10, min_iter=2),
        gs_num_steps=1,
        gs_recipe="1x1",
        unit_cell="1x1",
        su_init=False,
        gs_c4v=True,
        gs_explicit_ad=True,
        gs_explicit_ad_steps=5,
        gs_explicit_ad_warmup=2,
        gs_metric_precond=False,
        cg_gates=honeycomb_cg_gates(),
    )
    dummy_gate = jnp.zeros((4, 4, 4, 4))  # d_eff placeholder for the CG supersite
    with pytest.raises(NotImplementedError, match="cg_gates"):
        optimize_gs_ad(dummy_gate, None, cfg)
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py::test_optimize_gs_ad_split_rejects_cg_gates -v`
Expected: PASS — raises `NotImplementedError` with message `"fuse_virtual_legs=False (split CTM) does not support cg_gates; ..."` (the guard fires before any CTM iteration, so the test is fast).

- [ ] **Step 3: Commit**

```bash
git add tests/test_split_ctm_fuse_flag.py
git commit -m "test(#463): lock cg_gates (CG layer-1) rejection on split optimizer"
```

---

## Task 3: 1-site split production-correctness window test (Part 1, `slow`)

**Files:**
- Create: `tests/test_split_ctm_production_correctness.py`

- [ ] **Step 1: Create the test file**

Create `tests/test_split_ctm_production_correctness.py`:

```python
"""Production-correctness of the 1-site split-CTM path (#463 pre-flip check).

Validates that the split (``fuse_virtual_legs=False``) single-site path, run
WITH C4v on the sublattice-rotated square Heisenberg gate (under which a 1-site
iPEPS represents Neel order), converges to a *variational* energy: above the
QMC ground-state energy and below the disordered energy.

Empirical anchors (D=2, chi=10, gs_c4v=True, grad_norm |g|<1e-3):
  split+c4v = -0.6505/site (variational, +0.019 above QMC -0.6694)
  fused+c4v = -0.6601/site
WITHOUT C4v the unconstrained 1-site CTM is non-variational for BOTH paths
(split breaches to -0.714) -- so gs_c4v=True is mandatory here. The split/fused
~0.01/site gap is the bounded #425 fixed-point difference (both physical).
"""
import jax.numpy as jnp
import pytest

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad
from tests.test_split_ctm_fuse_flag import _make_site

# Sandvik QMC square-lattice spin-1/2 Heisenberg, energy per site.
QMC_FLOOR = -0.6694
ORDERED_CEIL = -0.60  # below the disordered/product energy => genuine order


def _rotated_heisenberg():
    """H_rot = -SzSz - 0.5(S+S+ + S-S-): sublattice-rotated so a 1-site iPEPS
    represents Neel order (unitary image of the AFM Heisenberg gate)."""
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = -jnp.kron(Sz, Sz) - 0.5 * (jnp.kron(Sp, Sp) + jnp.kron(Sm, Sm))
    return H.reshape(2, 2, 2, 2)


def _config(fuse):
    return iPEPSConfig(
        ctm=CTMConfig(
            chi=10, chi_I=10, fuse_virtual_legs=fuse,
            max_iter=80, conv_tol=1e-10, min_iter=4,
        ),
        unit_cell="1x1", gs_recipe="1x1", gs_implicit_ad=True,
        gs_c4v=True, gs_metric_precond=False,
        gs_conv_criterion="grad_norm", gs_grad_norm_tol=1e-3,
        gs_num_steps=100, gs_log_interval=10, su_init=False,
    )


@pytest.mark.slow
def test_split_1site_is_variational_with_c4v():
    """Split + C4v on rotated Heisenberg must land in [-0.6694, -0.60]/site.

    The lower bound is THE assertion: a correct variational 1-site path stays
    above the QMC ground state; the #425-spurious sub-QMC fixed point (which
    appears WITHOUT C4v, E=-0.714) would breach it.
    """
    A = _make_site(2, 2, seed=3)
    _, _, E = optimize_gs_ad(_rotated_heisenberg(), A, _config(fuse=False))
    E = float(E)  # per site = E_h + E_v
    assert E >= QMC_FLOOR - 1e-3, (
        f"split breaches QMC variational floor: E/site={E:.6f} < {QMC_FLOOR}"
    )
    assert E <= ORDERED_CEIL, (
        f"split did not order: E/site={E:.6f} > {ORDERED_CEIL}"
    )
```

- [ ] **Step 2: Run the test (slow — minutes on GPU, longer on CPU)**

Run: `uv run pytest tests/test_split_ctm_production_correctness.py::test_split_1site_is_variational_with_c4v -v`
Expected: PASS. The split optimization converges around step ~69 to E/site ≈ −0.6505, which satisfies −0.6694 ≤ −0.6505 ≤ −0.60.

- [ ] **Step 3: Commit**

```bash
git add tests/test_split_ctm_production_correctness.py
git commit -m "test(#463): split 1-site variational-window production-correctness (slow, C4v)"
```

---

## Task 4 (OPTIONAL): split-tracks-fused companion test

Include only if the extra full optimization (~doubles the slow runtime) is acceptable. The window test (Task 3) is the primary deliverable; this documents the bounded #425 gap.

**Files:**
- Modify: `tests/test_split_ctm_production_correctness.py`

- [ ] **Step 1: Append the companion test**

```python
@pytest.mark.slow
def test_split_tracks_fused_with_c4v():
    """Split and fused (both + C4v) converge within the bounded #425 gap.

    Measured gap 0.0096/site at D=2 chi=10; assert <= 0.03 (3x margin). Both
    paths are variational; this documents that split is a faithful (not
    bit-identical) drop-in for fused on the 1-site path.
    """
    A = _make_site(2, 2, seed=3)
    gate = _rotated_heisenberg()
    _, _, E_split = optimize_gs_ad(gate, A, _config(fuse=False))
    _, _, E_fused = optimize_gs_ad(gate, A, _config(fuse=True))
    gap = abs(float(E_split) - float(E_fused))
    assert gap <= 0.03, f"split-vs-fused gap too large: {gap:.4f} (expected ~0.0096)"
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_split_ctm_production_correctness.py::test_split_tracks_fused_with_c4v -v`
Expected: PASS (gap ≈ 0.0096 ≤ 0.03).

- [ ] **Step 3: Commit**

```bash
git add tests/test_split_ctm_production_correctness.py
git commit -m "test(#463): document bounded split-vs-fused #425 gap (slow companion)"
```

---

## Task 5: Clean up scratch probe scripts

**Files:**
- Delete: `_split_floor_one.py`, `_split_floor_confirm.py`

- [ ] **Step 1: Remove the scratch scripts**

```bash
git rm -f --ignore-unmatch _split_floor_one.py _split_floor_confirm.py
rm -f _split_floor_one.py _split_floor_confirm.py
```

- [ ] **Step 2: Confirm the tree is clean of scratch files**

Run: `git status --porcelain | grep -E "_split_floor" || echo "clean"`
Expected: `clean`

- [ ] **Step 3: Commit (if anything was tracked)**

```bash
git commit -m "chore(#463): remove scratch split-CTM floor probe scripts" || true
```

---

## Task 6: Full regression + suite green

- [ ] **Step 1: Run the fast split-CTM suite (core)**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -v`
Expected: ALL PASS — including the two new guard tests and the pre-existing 17.

- [ ] **Step 2: Run the broader split-CTM suite (no slow)**

Run: `uv run pytest tests/test_split_ctm_doublelayer_projector.py tests/test_split_ctm_tensor.py -m "not slow" -q`
Expected: ALL PASS — no behavior change to any shipped path.

- [ ] **Step 3: Run the new slow test once to confirm green**

Run: `uv run pytest tests/test_split_ctm_production_correctness.py -v`
Expected: PASS (Task 3, and Task 4 if included).

---

## Self-Review Notes

- **Spec coverage:** Part 1 (1-site production-correctness) → Task 3 (+ optional Task 4). Part 2 (CG guard lock-in, both layers) → Tasks 1 + 2. CG enablement doc → lives in the spec (`CG enablement (future)` section); no separate task. Suite-green acceptance → Task 6. Scratch cleanup → Task 5.
- **No production code changes** — matches the "validation only, no new code" scope. All tests characterize/lock existing behavior, so each passes on first run (these are regression locks; to confirm a lock is real, temporarily delete the guard and re-run — the test should then fail).
- **All test code is verified** against the current tree (both guards confirmed raising; split+c4v window confirmed at −0.6505 on GPU).
- **Markers:** guard tests inherit `core` from `test_split_ctm_fuse_flag.py`'s `pytestmark`; window/companion tests use explicit `@pytest.mark.slow`.
