# #670 Symmetric 2×2 CTM carried-bond threading fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the symmetric (block-sparse U(1)) `ctm_tensor_2site(..., recipe="2x2")` sweep run to convergence on a genuine multi-charge U(1)-Sz environment (currently it raises a per-sector block mismatch), match the dense result, and un-xfail the #667 acceptance test.

**Architecture:** Approach A from `docs/superpowers/specs/2026-07-01-670-symmetric-2x2-carried-bond-design.md`. The 2×2 absorption already has the correct one-carried/one-compressed corner structure (confirmed against variPEPS). The bug is that the *carried* corner leg and the edge leg the enlarged corner glues it to are not the same bond. **Task 1 is a diagnostic gate** that pins the exact divergent bond using a variPEPS numerical oracle; **Task 3's fix content is chosen from Task 1's finding** (the leading candidate is given, with a decision rule). TDD throughout; the dense path must stay a bit-for-bit no-op.

**Tech Stack:** JAX (float64), Tenax `SymmetricTensor` (U(1)-Sz block-sparse), `opt_einsum`; variPEPS 1.4.2 (GPL, importable in `.venv`) as a **read-only test/diagnostic oracle only — never imported by `src/`**.

---

## Background the engineer needs (read before starting)

The failure: `ctm_tensor_2site(recipe="2x2")` on any multi-charge U(1) env raises
`ValueError: Size of label ... does not match previous terms` inside
`_build_enlarged_corner` (e.g. `bottom_left`'s `C4.c4_u ↔ T4.t4_u`). Dense
(trivial-charge) does not crash (single block of size χ).

Three facts already established (do NOT re-derive; cited in the spec):

1. **Env-consumption ground truth** (energy/RDM path, production-validated) —
   which env legs share a bond:
   ```
   C1.c1_r↔T1.t1_l   C1.c1_d↔T4.t4_d
   C2.c2_l↔T1.t1_r   C2.c2_d↔T2.t2_u
   C3.c3_l↔T3.t3_l   C3.c3_u↔T2.t2_d
   C4.c4_u↔T3.t3_r   C4.c4_r↔T4.t4_u
   ```
2. **Enlarged-corner pairing** (`_build_enlarged_corner`) agrees with (1) for
   C1/C2/C3 but **disagrees for C4**: `bottom_left` uses `C4.c4_u↔T4.t4_u`,
   `C4.c4_r↔T3.t3_r` (C4's two legs swapped vs the ground truth).
3. **Left absorption** (`_ctm_tensor_absorb_left_2plaq`) produces
   `new C4.c4_r` = projector-compressed (`P_top_curr`), `new C4.c4_u` = carried
   relabel of the old edge leg `T3.t3_l` (`{"chi_new": "c4_r", "t3_l": "c4_u"}`,
   `src/tenax/algorithms/_ctm_tensor_moves.py:506`).

A localized swap of *only* the `bottom_left` pairing was already tried and
**refuted** — it relocates the crash to the other C4 leg. So the diagnostic (Task
1) must pin whether the divergence is the carried leg's source cell, its
`t3_l`/`t3_r` end, the enlarged-corner pairing, or a combination, before editing.

variPEPS oracle finding (the anchor): variPEPS's `new C4` after a left move is
also `(carried_old_T3_chi, compressed_new_chi)`; it works because the carried leg
is a **verbatim copy of the unmodified edge tensor the consumer later glues to**.
The Tenax fix must make the carried leg equal that same unmodified edge leg.

---

## File Structure

- `tests/oracle_varipeps.py` (new) — test-only helper that runs variPEPS 2-site
  CTMRG and extracts env / post-absorption bond structure. Guarded by
  `pytest.importorskip("varipeps")`. **Never imported by `src/`.**
- `scripts/diag_670_bond_divergence.py` (new, throwaway — deleted in Task 6) —
  the Task 1 diagnostic that diffs Tenax vs variPEPS post-`left`-move bonds.
- `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` — `_build_enlarged_corner`
  (the C4 pairing in `bottom_left`, and by symmetry check the other positions).
- `src/tenax/algorithms/_ctm_tensor_moves.py` — the four
  `_ctm_tensor_absorb_*_2plaq` (carried-leg relabels).
- `tests/test_ctm_670_symmetric_2x2.py` (new) — the red→green unit test for
  post-absorption enlarged-corner consistency on a multi-charge env.
- `tests/test_ctm_direction_dependent_bonds.py` — un-xfail + retarget the
  acceptance test (Task 5).

---

## Task 1: variPEPS oracle + pin the exact divergent bond (DIAGNOSTIC GATE)

**This task writes no `src/` code. Its deliverable is (a) a reusable oracle
helper and (b) a recorded, exact statement of the divergent bond that Task 3
implements against.** Do not start Task 3 until this is done.

**Files:**
- Create: `tests/oracle_varipeps.py`
- Create: `scripts/diag_670_bond_divergence.py` (throwaway)

- [ ] **Step 1: Write the variPEPS oracle helper**

Create `tests/oracle_varipeps.py`. It must import variPEPS lazily and expose one
function that runs variPEPS dense 2-site CTMRG and returns the converged env
corner/edge tensor shapes + axis meaning. Use the working recipe already proven
in the investigation: build a 2-site checkerboard unit cell
(`structure=[[0,1],[1,0]]`, D=2, d=2) via `varipeps.PEPS_Unit_Cell.random(...)`
and converge with `varipeps.ctmrg.routine.calc_ctmrg_env(peps_arrays, uc)`.

```python
"""variPEPS 2-site CTMRG oracle (test/diagnostic only). GPL reference — NEVER
import this from src/. Skips cleanly if variPEPS is not installed."""
from __future__ import annotations
import numpy as np

def varipeps_available() -> bool:
    try:
        import varipeps  # noqa: F401
        return True
    except Exception:
        return False

def run_varipeps_2site_ctmrg(D: int = 2, d: int = 2, chi: int = 8, seed: int = 0):
    """Run dense 2-site checkerboard CTMRG; return {'C1':shape,...,'T4':shape}
    for unit-cell site 0 plus a short axis-convention note. Raises if variPEPS
    is unavailable (callers guard with varipeps_available())."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import varipeps
    from varipeps import PEPS_Unit_Cell
    from varipeps.ctmrg.routine import calc_ctmrg_env
    uc = PEPS_Unit_Cell.random(
        structure=[[0, 1], [1, 0]], d=d, D=D, chi=chi, chi_max=chi,
        seed=seed, dtype=np.complex128,
    )
    peps_arrays = [t.tensor for t in uc.get_unique_tensors()]
    conv_uc = calc_ctmrg_env(peps_arrays, uc)
    site0 = conv_uc[0, 0][0][0]
    return {
        name: tuple(getattr(site0, name).shape)
        for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")
    }
```

Note: the exact variPEPS constructor/env accessor names were verified working in
the 2026-07-01 investigation; if a name differs in the installed version, consult
`.venv/lib/python3.11/site-packages/varipeps/` examples and adjust — the goal is
"a converged env whose tensor shapes/axis order you can print", nothing more.

- [ ] **Step 2: Smoke-run the oracle**

Run: `uv run python -c "from tests.oracle_varipeps import run_varipeps_2site_ctmrg as r; print(r())"`
Expected: prints a dict of 8 shapes (corners `(chi,chi)`, edges rank-4). If
variPEPS errors, fix the constructor/accessor names per the installed package
before proceeding.

- [ ] **Step 3: Write the Tenax-vs-variPEPS divergence diagnostic**

Create `scripts/diag_670_bond_divergence.py`. It must:
1. Build a *direction-uniform multi-charge* U(1)-Sz pair (normal SU, `base_charges`
   kept) at D=3 (reuse `/tmp/su667_uniform.pkl` if present, else regenerate via
   the `_su_direction_dependent_pair`-style helper with `base_charges` kept — see
   `tests/test_ctm_direction_dependent_bonds.py` for the SU driver).
2. Run ONE Tenax `left` absorption on the init env (replicate the `left` branch
   of `_ctm_tensor_sweep_multisite`: `_compute_plaquette_projector_pair` for each
   anchor, then `_ctm_tensor_absorb_left_2plaq`, store at `s_dst`).
3. For the resulting env at each `s_dst`, print the per-sector charge Counter of
   `C4.c4_u`, `C4.c4_r`, `T4.t4_u`, `T4.t4_d`, `T3.t3_l`, `T3.t3_r`, and the
   `env_src` (neighbor) `T3.t3_l`/`t3_r`.
4. Explicitly test each candidate identity and print HOLDS/BROKEN:
   - `C4.c4_r (s_dst) == T4.t4_u (s_dst)`  (compressed↔compressed; expected HOLD)
   - `C4.c4_u (s_dst) == s_dst.T3.t3_r`     (carried↔own-T3 ground-truth glue)
   - `C4.c4_u (s_dst) == env_src.T3.t3_l`   (what is currently carried)
   - `C4.c4_u (s_dst) == env_src.T3.t3_r`
5. Print which single identity, if enforced, would make the `bottom_left`
   enlarged corner (energy/RDM pairing `c4_u↔t3_r`, `c4_r↔t4_u`) consistent.

- [ ] **Step 4: Run the diagnostic and RECORD the finding**

Run: `uv run python scripts/diag_670_bond_divergence.py`
Record, in a new "## Task 1 outcome" section appended to
`docs/superpowers/specs/2026-07-01-670-symmetric-2x2-carried-bond-design.md`:
- the exact divergent bond (which `C4` leg, which cell's `T3`, which end),
- which identity HOLDS vs BROKEN,
- the concrete correction it implies (carried-leg source/end and/or
  enlarged-corner pairing). This is the target Task 3 implements.

- [ ] **Step 5: Commit (oracle + finding; diagnostic script stays uncommitted)**

```bash
git add tests/oracle_varipeps.py docs/superpowers/specs/2026-07-01-670-symmetric-2x2-carried-bond-design.md
git commit -m "diag(#670): variPEPS oracle + pinned exact carried-bond divergence

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Red test — post-absorption enlarged corners on a multi-charge env

**Files:**
- Create: `tests/test_ctm_670_symmetric_2x2.py`

- [ ] **Step 1: Write the failing test**

This asserts that after ONE `left` sweep-direction on a multi-charge U(1) env, the
four enlarged corners for the NEXT direction all build (i.e. the env is internally
bond-consistent). It currently raises the `ValueError` block mismatch.

```python
"""Symmetric 2x2 must stay bond-consistent after absorption (#670)."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import tenax.algorithms.ipeps_simple_update as SU
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS as NB,
    _sort_coords_for_direction,
)
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_moves import (
    _compute_plaquette_projector_pair,
    _ctm_tensor_absorb_left_2plaq,
)
from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair

CHI = 12


def _uniform_multicharge_pair(D=3, steps=40, dt=0.1):
    """Normal U(1)-Sz SU (base_charges kept) -> direction-uniform multi-charge."""
    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    H = heisenberg_gate_u1sz()
    gate = SU._make_trotter_gate_tensor(H, dt, site_tensor=A)
    lh, lv = jnp.ones(D), jnp.ones(D)
    for s in range(steps):
        if s % 2 == 0:
            A, B, lh = SU._simple_update_2site_horizontal_tensor(A, B, gate, lh, lv, D)
        else:
            A, B, lv = SU._simple_update_2site_vertical_tensor(A, B, gate, lh, lv, D)
    return A, B


def _one_left(A, B):
    site = {(0, 0): A, (1, 0): B}
    dl = {c: _build_double_layer_tensor(t) for c, t in site.items()}
    envs = {c: initialize_ctm_tensor_env(t, CHI) for c, t in site.items()}
    projectors = {}
    for s_anchor in envs:
        s_TR = NB[s_anchor]["right"]; s_BL = NB[s_anchor]["bottom"]; s_BR = NB[s_TR]["bottom"]
        Pt, Pb, _, _ = _compute_plaquette_projector_pair(
            envs[s_anchor], envs[s_TR], envs[s_BL], envs[s_BR],
            dl[s_anchor], dl[s_TR], dl[s_BL], dl[s_BR], CHI, "left")
        projectors[s_anchor] = (Pt, Pb)
    new = {}
    for s_dst in _sort_coords_for_direction(list(envs), "left"):
        s_src = NB[s_dst]["left"]; sa = NB[s_src]["top"]
        Pta, Pba = projectors[sa]; Ptc, Pbc = projectors[s_src]
        C1, T4, C4 = _ctm_tensor_absorb_left_2plaq(envs[s_src], dl[s_src], Pta, Pba, Ptc, Pbc)
        new[s_dst] = envs[s_dst]._replace(C1=C1, T4=T4, C4=C4)
    return new, dl


def test_enlarged_corners_build_after_left_absorption_multicharge():
    A, B = _uniform_multicharge_pair()
    new, dl = _one_left(A, B)
    for s_dst, env in new.items():
        for pos, (C, Th, Tv) in {
            "top_left": (env.C1, env.T1, env.T4),
            "top_right": (env.C2, env.T1, env.T2),
            "bottom_left": (env.C4, env.T3, env.T4),
            "bottom_right": (env.C3, env.T3, env.T2),
        }.items():
            Q = _build_enlarged_corner(C, Th, Tv, dl[s_dst], position=pos)
            assert Q is not None, f"{pos} failed to build at {s_dst}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ctm_670_symmetric_2x2.py -o addopts="" -q`
Expected: FAIL with `ValueError: Size of label ... does not match previous terms`
(the block mismatch, most likely in `bottom_left`).

- [ ] **Step 3: Commit the red test**

```bash
git add tests/test_ctm_670_symmetric_2x2.py
git commit -m "test(#670): red — enlarged corners must build after 2x2 left absorption on multi-charge env

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Apply the carried-bond correction (content from Task 1)

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_moves.py` and/or
  `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` — per the Task 1 finding.

**Decision rule (from Task 1 outcome):**
- If Task 1 shows `C4.c4_r == T4.t4_u` HOLDS and the break is
  `C4.c4_u` vs `s_dst.T3.t3_r`, the fix has two coupled parts: (a) make
  `_build_enlarged_corner(bottom_left)` use the energy/RDM pairing
  (`c4_r↔t4_u`, `c4_u↔t3_r`), and (b) make the absorption carry onto `c4_u` the
  edge leg that equals `s_dst.T3.t3_r` (correct cell/end). Apply BOTH; a
  half-fix only relocates the crash (already observed).
- If Task 1 shows a different divergence (e.g. wrong source cell only), apply the
  minimal relabel/source correction it names instead.

**Leading-candidate edit (apply only if Task 1 confirms the pairing half).** The
current `bottom_left` in `_build_enlarged_corner` (~lines 235-244) is:

```python
    if position == "bottom_left":
        # C4.c4_u <-> T4.t4_u
        C_r = C.relabel("c4_u", "t4_u")
        CT_v = contract(C_r, T_v)  # -> (c4_r, t4_d, l2)
        # C4.c4_r <-> T3.t3_r
        T_h_r = T_h.relabel("t3_r", "c4_r")
        CTT = contract(CT_v, T_h_r)  # -> (t4_d, l2, d2, t3_l)
        Q = contract(CTT, a)  # -> (t4_d, t3_l, u2, r2) free legs
        return Q.relabels({"t4_d": "chi_T", "t3_l": "chi_R"})
```

Correct it to the energy/RDM pairing (matches C1/C2/C3 and the RDM path):

```python
    if position == "bottom_left":
        # C4.c4_r <-> T4.t4_u  (energy/RDM convention; #670)
        C_r = C.relabel("c4_r", "t4_u")
        CT_v = contract(C_r, T_v)  # -> (c4_u, t4_d, l2)
        # C4.c4_u <-> T3.t3_r
        T_h_r = T_h.relabel("t3_r", "c4_u")
        CTT = contract(CT_v, T_h_r)  # -> (t4_d, l2, d2, t3_l)
        Q = contract(CTT, a)  # -> (t4_d, t3_l, u2, r2) free legs
        return Q.relabels({"t4_d": "chi_T", "t3_l": "chi_R"})
```

Plus the absorption half named by Task 1 (the exact `_ctm_tensor_absorb_*_2plaq`
carried-leg source/end correction). Mirror the same correction across all four
directions where the analogous asymmetry exists (Task 1 confirms which).

- [ ] **Step 1: Apply the correction(s) named by the Task 1 outcome** (the
  leading-candidate `bottom_left` edit above + the absorption carried-leg fix).

- [ ] **Step 2: Run the Task 2 red test → green**

Run: `uv run pytest tests/test_ctm_670_symmetric_2x2.py -o addopts="" -q`
Expected: PASS (all enlarged corners build after the left absorption).

- [ ] **Step 3: Run a full symmetric 2×2 sweep smoke check**

Run:
```bash
uv run python -c "
import jax; jax.config.update('jax_enable_x64', True)
from tests.test_ctm_670_symmetric_2x2 import _uniform_multicharge_pair
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
A,B=_uniform_multicharge_pair()
eA,eB=ctm_tensor_2site(A,B,chi=12,max_iter=6,recipe='2x2')
import jax.numpy as jnp; print('C1 norm', float(jnp.linalg.norm(eA.C1.todense())))
"
```
Expected: completes without a block mismatch; prints a finite C1 norm.

- [ ] **Step 4: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py src/tenax/algorithms/_ctm_tensor_moves.py
git commit -m "fix(#670): correct 2x2 carried-bond threading so symmetric multi-charge env stays consistent

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Dense no-op guard (fix must not change the dense path)

**Files:**
- Add to: `tests/test_ctm_670_symmetric_2x2.py`

- [ ] **Step 1: Write the dense-parity test**

The direction-dependent *dense* 2×2 energy was −0.542116 before the fix and must
be unchanged (the fix is a relabel that is a no-op for trivial charges).

```python
def test_dense_2x2_energy_unchanged_by_fix():
    from tenax import compute_energy_ctm_tensor_2site, ctm_tensor_2site
    from tenax.algorithms.ipeps import heisenberg_gate
    from tenax.core.tensor import DenseTensor
    # direction-dependent pair (base_charges-free), same as the #667 fixture
    from tests.test_ctm_direction_dependent_bonds import _su_direction_dependent_pair
    A, B = _su_direction_dependent_pair()
    Ad = DenseTensor(np.array(A.todense()), A.indices)
    Bd = DenseTensor(np.array(B.todense()), B.indices)
    eA, eB = ctm_tensor_2site(Ad, Bd, chi=12, max_iter=60, conv_tol=1e-9, recipe="2x2")
    E = float(compute_energy_ctm_tensor_2site(Ad, Bd, eA, eB, heisenberg_gate()))
    assert abs(E - (-0.542116)) < 1e-5, f"dense 2x2 energy drifted: {E}"
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_ctm_670_symmetric_2x2.py::test_dense_2x2_energy_unchanged_by_fix -o addopts="" -q`
Expected: PASS (dense energy still −0.542116).

- [ ] **Step 3: Commit**

```bash
git add tests/test_ctm_670_symmetric_2x2.py
git commit -m "test(#670): guard dense 2x2 energy unchanged by carried-bond fix

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Un-xfail + retarget the #667 acceptance test to recipe="2x2"

**Files:**
- Modify: `tests/test_ctm_direction_dependent_bonds.py`

- [ ] **Step 1: Remove the `@pytest.mark.xfail(...)` decorator** from
  `test_symmetric_2site_ctm_matches_dense_on_direction_dependent_bonds` and change
  BOTH `ctm_tensor_2site(...)` calls in it from `recipe="1x1"` to `recipe="2x2"`.
  Update the docstring comment to cite #670 as fixed and this plan.

The two calls become:
```python
    eAd, eBd = ctm_tensor_2site(Ad, Bd, chi=12, max_iter=60, conv_tol=1e-9, recipe="2x2")
    ...
    eA, eB = ctm_tensor_2site(A, B, chi=12, max_iter=60, conv_tol=1e-9, recipe="2x2")
```

- [ ] **Step 2: Run the acceptance test**

Run: `uv run pytest tests/test_ctm_direction_dependent_bonds.py -o addopts="" -q`
Expected: all pass (no xfail). `abs(E_sym - E_dense) < 1e-6`, `C1 norm > 1e-8`,
`E_sym < -0.3` (dense reference ≈ −0.542 for recipe="2x2").
If `E_sym < -0.3` fails because the fixture SU state is degenerate (per the Phase
2 note), relax that one bound to match the dense reference and document why — the
load-bearing assertion is `abs(E_sym - E_dense) < 1e-6`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ctm_direction_dependent_bonds.py
git commit -m "test(#667): un-xfail direction-dependent 2-site CTM (fixed via #670), retarget to recipe=2x2

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Regression sweep, cleanup, PR

**Files:**
- Delete: `scripts/diag_670_bond_divergence.py`

- [ ] **Step 1: Delete the throwaway diagnostic**

```bash
git rm -f scripts/diag_670_bond_divergence.py 2>/dev/null || rm -f scripts/diag_670_bond_divergence.py
```

- [ ] **Step 2: Run the CTM/iPEPS regression suite**

Run:
```bash
uv run pytest -m core -q
uv run pytest tests/test_ctm_tensor.py tests/test_ctm_tensor_projector_2x2.py tests/test_ipeps_u1sz.py tests/test_ctm_670_symmetric_2x2.py tests/test_ctm_direction_dependent_bonds.py -o addopts="" -q
```
Expected: all green. In particular the 2×2 projector closure test
(`P_bot · P_top = I`) and the u1sz suite must be unchanged.

- [ ] **Step 3: Confirm no fermionic regression / scope note**

Run: `uv run pytest tests/ -k "fermion" -o addopts="" -q`
Expected: unchanged (the fix targets the non-fused path). If the same
carried-bond bug is visible on the fused path, open a follow-up issue rather than
expanding scope here.

- [ ] **Step 4: Update memory + close-out docs**

Update `MEMORY.md` note `project_667_direction_dependent_ctm.md` to record #670
FIXED (mechanism: carried-bond threading correction) and the #667 test un-xfailed.
Append the final mechanism to the design spec's "Task 1 outcome" section.

- [ ] **Step 5: Commit, push, PR, close #670**

```bash
git add -A
git commit -m "chore(#670): cleanup + memory/docs update; direction-dependent 2x2 CTM works

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push -u origin fix/670-symmetric-2x2-carried-bond
gh pr create --title "fix(#670): symmetric 2x2 CTM carried-bond threading (unblocks #667)" \
  --body "$(printf '> 🤖 **AI-generated PR** — written by Claude Code, posted by @yjkao.\n\nFixes #670. Corrects the 2x2 absorption carried-bond threading so the symmetric block-sparse CTM stays bond-consistent on genuine multi-charge U(1) envs. Anchored on a variPEPS numerical oracle. Un-xfails the #667 direction-dependent acceptance test (retargeted to recipe=2x2). Dense path is a bit-for-bit no-op (dense 2x2 energy unchanged at -0.542116).\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)')"
```
Merge per CLAUDE.md (auto-merge; CI must pass).

---

## Self-review notes

- **Spec coverage:** oracle harness = Task 1 Step 1; diagnostic/pin = Task 1
  Steps 3-4; fix = Task 3; acceptance (`|E_sym-E_dense|<1e-6`, un-xfail #667) =
  Task 5; dense no-op = Task 4; regression/closure/fermionic-scope = Task 6.
- **Diagnosis-gated honesty:** Task 3's exact content is chosen by Task 1's
  measured outcome (leading candidate + decision rule given). This is deliberate,
  not a placeholder — the localized-swap refutation proved guessing the fix blind
  is wrong; Task 1 removes the guess.
- **Names verified:** `_build_enlarged_corner`, `_ctm_tensor_absorb_left_2plaq`,
  `_compute_plaquette_projector_pair`, `_sort_coords_for_direction`,
  `initialize_ctm_tensor_env`, `_build_double_layer_tensor`,
  `CHECKERBOARD_NEIGHBORS`, `ctm_tensor_2site`,
  `compute_energy_ctm_tensor_2site` are current symbols.
- **No-op invariant:** Task 4 locks the dense path; Task 6 locks core/closure/
  fermionic — the fix must not regress the production (dense/trivial-charge) path.

---

## Outcome (2026-07-01) — DONE

The fix was **bounded** (Approach A held; not structural), but the scope grew from
the planned single site to **three**, because Task 1's diagnostic only ran one
`left` absorption and never exercised the full sweep. The scope-mapping (fix-forward
diagnostic) found every crash mapped cleanly to the production energy/RDM
convention:

1. `_build_enlarged_corner` `bottom_left` — C4 legs were swapped; now
   `c4_u↔t3_r`, `c4_r↔t4_u` (commit `805757e`).
2. `_ctm_tensor_absorb_bottom_2plaq` — C3·T2 paired `c3_l↔t2_d`; now `c3_u↔t2_d`.
3. `_ctm_sv_diff` — zero-pads SV vectors so warmup block-structure growth on
   asymmetric states reports not-converged instead of crashing (eager/non-AD
   call sites, so gradient-safe).

Result: direction-dependent symmetric 2×2 CTM matches dense to **4e-14**
(E = −0.5421160718). Dense path is a true no-op (the swap is a benign gauge
choice on a single block; a fatal charge-sector mismatch only for block-sparse
`A.l != A.r`), so the plan's dense-no-op assumption (Task 4) held after all.

Stale unit test `test_build_enlarged_corner_bottom_left_numerical` (its einsum
encoded the old swapped C4 pairing) updated. #667 acceptance test un-xfailed and
retargeted to `recipe="2x2"` (commit `07fa9ac`); dense guard added (commit
`0523818`).

**Out-of-scope follow-ups filed:** #674 (fermionic/fused twin
`_ctm_tensor_absorb_bottom_2plaq_fused` has the same latent C3 bug), #675
(compiled `_ctm_compiled_moves.py` uses the same pre-fix `c3_l↔t2_d` convention).
Both are no-ops on uniform states so production is unaffected.
