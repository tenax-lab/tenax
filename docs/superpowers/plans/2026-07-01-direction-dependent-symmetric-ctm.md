# Direction-Dependent Symmetric Multisite CTM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.
>
> **Read this first — this is a hybrid design+plan.** The *core* of the problem
> (keeping the two sublattices' renormalised environment bonds structurally
> identical across sweeps) is an **open design question**, not a known
> implementation. Phases 0–1 land the two pieces that ARE understood and
> correct; Phase 2 is a **decision phase** (spike + measure + choose) that must
> complete before the Phase 3 implementation can be written with real code.
> Do not skip Phase 2 — writing Phase 3 code before the mechanism is chosen
> would be guessing.

**Goal:** Make `ctm_tensor_2site` run to convergence on a valid unit-cell-consistent checkerboard iPEPS with `A.l != A.r` (direction-dependent virtual bonds) and match the densified result, unblocking `tests/test_ctm_direction_dependent_bonds.py`.

**Architecture:** The symmetric (block-sparse) multisite CTM builds each site's environment from that site's own tensor and truncates each sublattice's projector independently. For a *direction-uniform* cell this is self-consistent; for `A.l != A.r` the two sublattices' bonds acquire incompatible per-sector block structure and the cross-sublattice contractions in the sweep fail with block-size mismatches. The fix is layered: (0) a general tiling-canonicalisation bug fix, (1) recipe-correct env-init charge seeding, (2+3) a mechanism that forces both sublattices' renormalised bonds onto a shared per-bond charge template that persists across sweeps.

**Tech Stack:** JAX (float64), Tenax `SymmetricTensor` block-sparse tensors, U(1)-Sz symmetry, `opt_einsum` block contraction.

---

## Background: why this is hard (evidence from the 2026-07-01 investigation)

The RED test `tests/test_ctm_direction_dependent_bonds.py::test_symmetric_2site_ctm_matches_dense_on_direction_dependent_bonds` drives a `base_charges`-free U(1)-Sz simple update to produce a pair with:

```
A.u=[0,1,-1]  A.d=[1,3,1]  A.l=[0,1,-1]  A.r=[-2,0,-2]
B.u=[1,3,1]   B.d=[0,1,-1] B.l=[-2,0,-2] B.r=[0,1,-1]
```

Cell-consistent (`A.r==B.l`, `A.l==B.r`, `A.u==B.d`, `A.d==B.u`) but **direction-dependent** (`A.l != A.r`). The symmetric CTM crashes; the dense CTM (trivial charges) runs and gives the reference **E ≈ -0.542** for both `recipe="1x1"` and `recipe="2x2"`.

Four independent failure layers were found and characterised:

1. **Corner chi-leg seeding.** `_init_symmetric_standard_corner` seeds *both* corner chi legs from a single virtual axis. Fine when `A.l==A.r`; for direction-dependent bonds the two legs must seed from the axis each geometrically extends.
2. **Edge D² leg seeding.** A parallel edge's D² leg contracts the *neighbour's* double-layer (`move_left`: `T4.l2 · a.l2`, and `a.l2 == B.l == self.r`), so it must seed from the *opposite* virtual axis.
3. **Tiling canonicalisation (GENERAL BUG, recipe-independent).** `_compute_fused_charges` for two legs of the same bond that carry *opposite* flow (same virtual axis) produces sign-flipped enumerations of the *same* multiset. Padding a chi leg beyond D² in *enumeration* order appends a one-sided prefix and breaks multiset equality; padding in *sorted* order preserves it. This is a real bug independent of the direction-dependent feature.
4. **Cross-sublattice renormalised-bond consistency (THE HARD, UNSOLVED CORE).** The block-sparse SVD projector keeps `n_keep = min(chi, rank)` per sublattice. For `A != B` the two sublattices' renormalised bonds share a charge *support* but end up with different per-sector *counts* (e.g. after one `move_left`, `A.C1.c1_d` has charge-0 dim 3 while `B.T4.t4_d` has 6). The cross-sublattice contraction then fails. This persists across sweeps and appears in BOTH recipes.

**Recipe asymmetry (critical).** The corner↔edge contraction is:
- **1×1**: CROSS-site (`env_self.C1 · env_neighbor.T1`) → the two legs need *cell-partner* axes (`self.r == nb.l`). Fix (1) above is correct here.
- **2×2**: SAME-site (`env_src.C1 · env_src.T1` inside `_build_enlarged_corner` / `_ctm_tensor_absorb_*_2plaq`) → the two legs need the *same* axis (the original scheme). Fix (1) is WRONG here.

**1×1 also has a fundamental extra wall:** single-site absorption alternates the renormalised edge D² leg's orientation (after absorbing `B`, `T4.l2` becomes `B.r`-fused, but the next `move_left` contracts it against `B.l`-fused; `B.r != B.l`). This is irreconcilable by charge labels — it is real data. Only full-2-site-cell absorption (the 2×2 recipe) resolves it. **Therefore the target recipe is `2x2`.**

Dense "works" on this fixture only because trivial charges hide every mismatch above — so also sanity-check whether the fixture is physically meaningful (per issue #667, the `base_charges`-free SU is itself a degenerate near-classical attractor; see `examples/su_symmetric_ctm_e2e.py`).

Reproduction: the investigation used a cached SU pair and standalone debug scripts (regenerate the pair with the fixture `_su_direction_dependent_pair` from the test; ~90s).

---

## File Structure

- `src/tenax/algorithms/_ctm_tensor_init.py` — env init: corner/edge charge seeding, the `_tile_fused_to_chi` helper (Phase 0 + Phase 1 + Phase 3 templates).
- `src/tenax/algorithms/_ctm_projector.py` — `_svd_projector_symmetric` / `_compute_projector_tensor`: where per-sector truncation happens; template padding hooks (Phase 3).
- `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` — `_build_enlarged_corner`, `_compute_2x2_projector`: the 2×2 plaquette projector (Phase 2 spike + Phase 3).
- `src/tenax/algorithms/_ctm_tensor_moves.py` — `_ctm_tensor_absorb_*_2plaq`, `_compute_plaquette_projector_pair`: the 2×2 absorption (Phase 3).
- `src/tenax/algorithms/_ctm_tensor_convergence.py` — `_ctm_tensor_multisite`, `_ctm_tensor_sweep_multisite`: sweep orchestration, template computation & threading (Phase 3).
- `tests/test_ctm_tensor_tiling.py` (new) — unit test for the tiling fix (Phase 0).
- `tests/test_ctm_direction_dependent_bonds.py` (exists, RED) — the acceptance test; will be retargeted to `recipe="2x2"` (Phase 3).

---

## Phase 0 — Canonical tiling fix (understood, self-contained, land first)

This is a genuine bug independent of the direction-dependent feature: two opposite-flow legs of the same bond must produce equal charge multisets after tiling to `chi`.

### Task 0.1: Add `_tile_fused_to_chi` and unit-test it

**Files:**
- Create test: `tests/test_ctm_tensor_tiling.py`
- Modify: `src/tenax/algorithms/_ctm_tensor_init.py` (add helper after `_grouped_chi_perm`, ~line 285)

- [x] **Step 1: Write the failing test**

```python
# tests/test_ctm_tensor_tiling.py
"""Opposite-flow legs of one bond must tile to equal charge multisets (#667)."""
from __future__ import annotations
import numpy as np
from collections import Counter
from tenax.algorithms._ctm_tensor_init import _tile_fused_to_chi


def test_tile_preserves_multiset_under_sign_flip():
    # A U(1) fusion enumeration and its sign-flip (what opposite-flow legs of
    # the same virtual axis produce) must give equal multisets after tiling
    # past D**2 to chi.
    fused = np.array([0, -2, 0, 2, 0, 2, 0, -2, 0], dtype=np.int32)  # D**2 = 9
    flipped = -fused
    chi = 12
    a = Counter(_tile_fused_to_chi(fused, chi).tolist())
    b = Counter(_tile_fused_to_chi(flipped, chi).tolist())
    assert a == b, f"tiled multisets differ: {a} vs {b}"


def test_tile_keeps_charge_zero_at_index_0():
    # The rank-1 seed lives at pre-perm index 0 and must stay charge 0.
    fused = np.array([0, -1, 1, 1, 0, 2, -1, -2, 0], dtype=np.int32)
    out = _tile_fused_to_chi(fused, 12)
    assert int(out[0]) == 0


def test_tile_no_pad_when_chi_le_d2():
    fused = np.array([0, -2, 0, 2, 0, 2, 0, -2, 0], dtype=np.int32)
    out = _tile_fused_to_chi(fused, 5)
    assert np.array_equal(out, fused[:5])
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ctm_tensor_tiling.py -v`
Expected: FAIL with `ImportError: cannot import name '_tile_fused_to_chi'`

- [x] **Step 3: Implement the helper**

In `src/tenax/algorithms/_ctm_tensor_init.py`, add after `_grouped_chi_perm`:

```python
def _tile_fused_to_chi(fused: np.ndarray, chi: int) -> np.ndarray:
    """Tile size-D**2 ``fused`` charges up to length ``chi`` canonically.

    The leading D**2 block keeps the raw enumeration order (so index 0 stays
    the charge-0 diagonal that anchors the rank-1 seed at the vacuum slot); any
    padding beyond D**2 is appended in **sorted** order.  This makes two legs
    of one bond that carry opposite flow (sign-flipped enumerations of the same
    multiset) agree after tiling.  A no-op for ``chi <= D**2`` and for
    direction-uniform iPEPS.  #667.
    """
    fused = np.asarray(fused, dtype=np.int32)
    if chi <= len(fused):
        return fused[:chi]
    srt = np.sort(fused)
    reps = (chi - len(fused)) // len(srt) + 1
    tail = np.tile(srt, reps)[: chi - len(fused)]
    return np.concatenate([fused, tail]).astype(np.int32)
```

Also add `"_tile_fused_to_chi"` to `__all__` at the top of the file.

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_tensor_tiling.py -v`
Expected: PASS (3 tests)

- [x] **Step 5: Route existing tiling through the helper**

In `_init_symmetric_standard_edge`, replace the body of the inner `_fused_chi_charges` (the `if chi <= len(fused): ... else: tile ...` block) with `return _tile_fused_to_chi(fused, chi)`. Leave the corner as-is for now (it is rewritten in Phase 1).

- [x] **Step 6: Run the init regression + full core suite**

Run: `uv run pytest tests/test_ctm_tensor_init_rank1.py tests/test_ipeps_u1sz.py -q`
Expected: PASS (no regression — for uniform/trivial charges the helper is a no-op).

- [x] **Step 7: Commit**

```bash
git add tests/test_ctm_tensor_tiling.py src/tenax/algorithms/_ctm_tensor_init.py
git commit -m "fix(#667): canonical (sorted) tiling for opposite-flow env chi legs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1 — 2×2-correct env init (understood) — DONE (commit c9a77ed)

For the **2×2** recipe, all enlarged-corner and absorption corner↔edge contractions are SAME-site, so the *original* same-axis seeding is correct. The only init change 2×2 needs is Phase 0's tiling (already landed). This phase is therefore a **verification phase**: confirm the enlarged-corner and 2-plaquette absorption contractions all pass on the fixture with the current (original) init + Phase 0 tiling, and lock that with a test. No init-axis changes — the 1×1-oriented axis reseeding from the 2026-07-01 WIP is explicitly NOT applied here.

**Phase 1 outcome (2026-07-01):** the verification test initially FAILED at the
first corner (`top_left`) on the same-site contraction `C1.c1_r <-> T1.t1_l`
(7-vs-6 per-sector block mismatch). Root cause was the residual same-site gap the
plan anticipated: Phase 0 routed only the *edge* chi legs through the canonical
sorted `_tile_fused_to_chi`, but `_init_symmetric_standard_corner` still tiled its
chi legs in **enumeration** order. Both legs derive from the same virtual axis
(`ref_axis=d`) and share the same D²-fused multiset, but their padding past D²
diverged. Fix: route the corner through `_tile_fused_to_chi` too (no axis
reseeding — original same-axis 2×2 seeding kept). All four enlarged corners on
both sublattices now build; regression green (tiling + init-rank1 + u1sz, 30
passed).

### Task 1.1: Lock enlarged-corner charge-consistency on the fixture

**Files:**
- Test: `tests/test_ctm_direction_dependent_bonds.py` (add a focused sub-test)

- [x] **Step 1: Write a test that builds all four enlarged corners without error**

```python
def test_2x2_enlarged_corners_build_on_direction_dependent_init():
    """First-sweep 2x2 enlarged corners must build (charge-consistent) on the
    direction-dependent init env."""
    import numpy as np
    from tenax.algorithms._ctm_tensor_init import (
        initialize_ctm_tensor_env, _build_double_layer_tensor,
    )
    from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner

    A, B = _su_direction_dependent_pair()
    chi = 12
    envA = initialize_ctm_tensor_env(A, chi)
    envB = initialize_ctm_tensor_env(B, chi)
    aA = _build_double_layer_tensor(A)
    aB = _build_double_layer_tensor(B)
    # Same-site enlarged corners (each uses one site's env + that site's DL).
    for env, a in ((envA, aA), (envB, aB)):
        for pos, (C, Th, Tv) in {
            "top_left": (env.C1, env.T1, env.T4),
            "top_right": (env.C2, env.T1, env.T2),
            "bottom_left": (env.C4, env.T3, env.T4),
            "bottom_right": (env.C3, env.T3, env.T2),
        }.items():
            Q = _build_enlarged_corner(C, Th, Tv, a, position=pos)
            assert Q is not None
```

- [x] **Step 2: Run it**

Run: `uv run pytest tests/test_ctm_direction_dependent_bonds.py::test_2x2_enlarged_corners_build_on_direction_dependent_init -v`
Expected: PASS (with Phase 0 tiling, the same-site enlarged corners are charge-consistent). If it FAILS, the failing contraction identifies a residual same-site seeding/tiling gap — fix in `_ctm_tensor_init.py` and re-run before proceeding. Record the exact failing contraction in the commit message.

- [x] **Step 3: Commit**

```bash
git add tests/test_ctm_direction_dependent_bonds.py
git commit -m "test(#667): lock 2x2 enlarged-corner charge consistency on direction-dependent init

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — DECISION PHASE: choose the cross-sublattice bond-consistency mechanism

**Do not write Phase 3 until this phase produces a chosen mechanism.** The core
unknown: after a full 2×2 absorption sweep, the two sublattices' renormalised
bonds (e.g. `c4_u` and `t4_u`, produced by different projectors) drift to
different per-sector block structures and the next sweep's same-site enlarged
corner fails. The mechanism must force every renormalised env chi bond onto a
**shared, per-bond, persistent** charge template.

Candidate mechanisms (from the 2026-07-01 investigation — each has an open risk):

- **A. Fixed per-bond templates + zero-pad in the projector.** Compute a fixed
  charge template per env bond once (from the init env, per-sector-max over the
  sublattices), and make the 2×2 projector emit `chi_new` at exactly the
  template. Because each per-sector SVD `M_q` is at most `chi×chi`, total rank
  `<= chi`, so padding to `chi` is lossless (identical to the dense path keeping
  zero singular values). Risk: whether a single fixed template's per-sector
  budget always covers the renormalised counts (under-provisioning truncates
  real data → energy drift).
- **B. Adaptive shared template recomputed each sweep.** Per sweep, gather both
  sublattices' bonds, take per-sector max, reembed both. Risk: per-sector-max
  summed can exceed `chi`; needs a fit-to-`chi` policy.
- **C. Joint truncation.** Decide the kept per-sector counts once per seam
  (shared base_charges) and apply to both sublattices' projectors. Risk: still
  need to pad when a sublattice's rank in a sector is below the shared count.

### Task 2.1: Spike — measure renormalised-bond drift and template coverage

**Files:**
- Create (throwaway, do not commit): `scripts/spike_667_bond_drift.py`

- [ ] **Step 1: Write a spike that runs N=1..5 2×2 sweeps and records, per env bond, the per-sector counts on each sublattice after each sweep**

Use `initialize_ctm_tensor_env`, `_build_double_layer_tensor`, and
`_ctm_tensor_sweep_multisite(..., recipe="2x2")`; wrap the sweep in
try/except and, on the first failure, print the two mismatched legs' per-sector
counts (the failing contraction is reported by the `ValueError`). Regenerate the
SU pair with `_su_direction_dependent_pair` (from the test) or cache it to
`/tmp` to iterate fast.

- [ ] **Step 2: Run and record**

Run: `uv run python scripts/spike_667_bond_drift.py`
Record: (a) which sweep and which contraction first fails; (b) for the mismatching bond, the per-sector counts on A vs B across sweeps; (c) whether a fixed init-derived template's per-sector budget ever gets *exceeded* by a renormalised count (this decides A vs B/C).

- [ ] **Step 3: Decide the mechanism**

Write the decision (and the measured evidence) into
`docs/superpowers/plans/2026-07-01-direction-dependent-symmetric-ctm.md` under a
new "## Phase 2 outcome" section: chosen mechanism (A/B/C), the template
construction rule, and where it is applied (projector emit vs post-sweep
reembed). If the init-derived fixed template (A) is never exceeded in the spike,
choose A (simplest). Delete the spike script.

- [ ] **Step 4: Also record the fixture-validity check**

Run `examples/su_symmetric_ctm_e2e.py` and note whether the `base_charges`-free
SU state is degenerate/near-classical (issue #667). If it is, note in the Phase 2
outcome that the acceptance test's `E_sym < -0.3` bound may need revisiting and
the "matches dense" goal is a numerical-consistency check, not a physics check.

---

## Phase 3 — Implement the chosen mechanism for the 2×2 recipe (code depends on Phase 2 outcome)

**This phase's concrete code cannot be written until Phase 2 chooses the
mechanism.** The task skeleton below is fixed; the implementation bodies are
filled from the Phase 2 outcome. Each task keeps the TDD shape (write failing
assertion against the fixture → implement → green → commit).

### Task 3.1: Template computation + threading (skeleton)

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py` (`_ctm_tensor_multisite`, `_ctm_tensor_sweep_multisite` 2×2 branch)
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` and/or `_ctm_projector.py` per the chosen mechanism

- [ ] **Step 1:** Add a failing assertion: after two `recipe="2x2"` sweeps on the fixture, `envA.C4.c4_u` and `envA.T4.t4_u` (and each contracted pair) have identical per-sector charge structure. (Run to confirm it currently fails/raises.)
- [ ] **Step 2:** Implement template computation from the init env (per the Phase 2 rule) in `_ctm_tensor_multisite`; thread it into the 2×2 sweep and projector per the chosen mechanism.
- [ ] **Step 3:** Run the step-1 assertion → PASS.
- [ ] **Step 4:** Commit.

### Task 3.2: Full 2×2 convergence matches dense on the fixture (acceptance)

**Files:**
- Modify: `tests/test_ctm_direction_dependent_bonds.py` — retarget the acceptance test to `recipe="2x2"` for BOTH the dense reference and the symmetric run.

- [ ] **Step 1:** Change both `ctm_tensor_2site(...)` calls in `test_symmetric_2site_ctm_matches_dense_on_direction_dependent_bonds` from `recipe="1x1"` to `recipe="2x2"` (dense reference E ≈ -0.542 is unchanged between recipes, verified 2026-07-01). Add a comment citing this plan and the 1×1 fundamental-orientation limitation.
- [ ] **Step 2:** Run: `uv run pytest tests/test_ctm_direction_dependent_bonds.py -v`
  Expected: the symmetric run completes and `abs(E_sym - E_dense) < 1e-6`, `C1 norm > 1e-8`. If `E_sym < -0.3` fails because the fixture is a degenerate SU state (Phase 2 Step 4), relax/adjust that bound per the Phase 2 outcome and document why.
- [ ] **Step 3:** Commit.

### Task 3.3: Regression sweep

- [ ] **Step 1:** Run the CTM/iPEPS core suite: `uv run pytest -m core -q` plus `tests/test_ipeps_u1sz.py tests/test_ctm_tensor.py tests/test_ctm_tensor_projector_2x2.py -q`.
- [ ] **Step 2:** Confirm no regressions (the template mechanism must be a no-op for direction-uniform cells — assert this explicitly by checking a uniform C4v/D3 run's energy is unchanged to 1e-10).
- [ ] **Step 3:** Update `MEMORY.md` note `project_667_direction_dependent_ctm.md` to CLOSED with the final mechanism, and open a follow-up issue for `FermionParity`/`FermionicU1` direction-dependent support (out of scope here).
- [ ] **Step 4:** Commit; open PR per `CLAUDE.md` (`gh pr create`, squash-merge with CI).

---

## Self-review notes

- **Spec coverage:** the RED acceptance test is Task 3.2; the tiling bug is Phase 0; the recipe-asymmetry root cause is encoded in the Phase 1 "no axis reseeding" note and the Phase 3 `recipe="2x2"` retarget; the unsolved core is Phase 2 (explicitly a decision phase, not fabricated code).
- **Placeholders:** Phase 0/1 contain complete code. Phase 3 intentionally defers bodies to the Phase 2 outcome — this is honest (the mechanism is undecided), not a "TODO"; Phase 2 is a mandatory gate that produces the missing design.
- **Names:** `_tile_fused_to_chi`, `_build_enlarged_corner`, `_ctm_tensor_absorb_*_2plaq`, `_compute_plaquette_projector_pair`, `_svd_projector_symmetric` are the real current symbols (verified 2026-07-01).
- **Risk to core paths:** the AD backward path uses the dense/tracer projector, not `_svd_projector_symmetric`'s non-tracer branch, so a projector-emit template (mechanism A) confined to that branch does not touch AD; verify with Task 3.3.
