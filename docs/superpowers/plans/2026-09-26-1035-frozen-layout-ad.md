# Frozen SU Layout → Traced Graded CTM + AD — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimise a 2-site fermionic iPEPS by AD on a *fixed* quantum-number layout that an eager simple update discovered: the eager SU grows the bond sectors, the layout is frozen, an eager CTM finds the environment's χ-sector layout on that state, and the traced (jitted, implicit-AD) graded CTM then runs on exactly those layouts.

**Design principle (agreed 2026-09-26):** once the eager SU has stabilised the quantum-number sectors, the sector sizes are frozen and nothing downstream re-decides them -- the traced CTM keeps its per-sector counts fixed and truncates only *within* a sector, never across sectors. Every earlier #566 lever that needed a particular block-shape structure (the even-D padded-`vmap` port, dropped in PR #1043) predates graded fermions and is superseded by this: under a frozen layout every block shape is static, which is all `jit` needs.

**Architecture:** Three stages, one hand-off each. (1) Eager SU, *unpinned* so truncation can move weight between sectors, grown one slot per stage from D=2 to D -- the slot a growth truncation adds is the only layout decision the SU makes (Task 2). (2) Eager CTM on the frozen state; its global top-χ truncation picks the environment layout. (3) `optimize_gs_ad` 2-site, seeded with that environment. Under tracing nothing can change a layout: site-tensor blocks are static pytree aux data, and the traced projector SVD allocates χ from the incoming environment's own inventory (`_incoming_chi_charges`, `_ctm_tensor_projector_2x2.py:1196`) — so the seed *is* the frozen layout, and the AD step's compile cache (`_make_jit_ctm_step`) is hit on every optimizer step and line search.

**Tech Stack:** JAX (`jit`, `custom_vjp` implicit AD), tenax `SymmetricTensor` (FermionParity), optax L-BFGS, pytest.

**Spec:** the design stated in session `d29e6afe` ("we use eager SU to grow the quantum-number sectors"; "rewrite the plan around frozen SU layout and AD"), on top of #1035 Phase 3 (graded fused CTM + graded SU, branch `feat/1035-graded-ctm-phase3`).

## What exists already (verified 2026-09-26 on `feat/1035-graded-ctm-phase3`)

- **The AD path is graded.** `optimize_gs_ad(unit_cell="2site")` → `_optimize_gs_ad_tensor_2site` uses the fused CTM by default (`CTMConfig.fuse_virtual_legs=True`), whose sweep and 2-site RDM readouts go through `_ctm_graded`. `compute_energy_ctm_tensor_multisite` also reads through the graded 2-site readouts. The split path refuses fermions.
- **The AD forward is already a jitted loop:** `python_loop_ctm_converge` (`_ctm_python_loop.py:155`) runs `_make_jit_ctm_step` per sweep and accepts `env_init`. No new forward loop is needed; a χ scan can call it directly.
- **The seed is not wired.** The first 2-site AD forward starts from `initialize_ctm_tensor_env` (tiled double-layer charges), because `_env_cache_2s` starts empty (`ipeps_optimize.py:~2800`). There is no way to pass an environment in.
- **The SU pins sectors for fermions.** `_truncation_base_charges` (`ipeps_simple_update.py:56`) returns the old bond's charges for any fermionic leg, so the SU *cannot* grow sectors today. The pin exists for the 1-site path (`A.l`/`A.r` are one bond there); on the 2-site checkerboard they are different bonds, and the pin is what drives the #878 collapse (pin ON D=3 3/5 survive, OFF 5/5).
- **`su_init` is bosonic-only.** It calls `ipeps()` with a dense gate, so fermionic callers pass `AB_init` themselves.

## Prior art this plan stands on (read before Task 3 and Task 6)

- **#435 / PR #440 (`2026-05-12-issue-435-tracer-safe-2x2-projector.md`, merged `3d964b4`)** — the eager/traced split of the symmetric 2x2 projector: eager = global top-χ SV sort (re-truncated to `base_charges`); traced = static per-sector keep counts, **no global re-sort**, bond in sector-block order. Its eager-vs-traced equivalence test covers trivial charges only (`P P†` on an all-zero-charge fixture). Nothing there pins a non-trivial-charge fixed point across the two paths.
- **#929 (`tests/test_ctm_traced_chi_inventory_929.py`)** — the traced projector inherits the *incoming environment leg's* χ inventory instead of the tiled double-layer guess; measured on U(1)-Sz D=3 χ=16 it reproduces the eager environment exactly. This is the mechanism the whole plan relies on: the environment you seed AD with is the layout AD keeps. Task 3 is the FermionParity + graded instance of that test; the U(1) one already exists.
- **#566 (`examples/profile_566_a100_summary.md`, the two spike summaries)** — the compile wall of the jitted symmetric AD: on an A100, fermionic `value_and_grad` compile is 206 s at D=2 χ=8 and **~35 min at D=3 χ=12** (dense: 40 s); both #566 spikes (padded-vmap, C-adjoint) are NO-GO, and the standing recommendation is that JAX symmetric AD is a small-D tool (D ≤ 3–4). Two consequences for this plan: (a) a *changing* layout under AD would pay that compile again on every change, so freezing the layout is what makes fermionic AD affordable at all — that is the plan's justification, not just its mechanism; (b) Task 6 must budget ~35 min for the first D=3 step on a GPU (longer on CPU) and must not read that as a hang (#565).

- **YASTN (PyTorch, eager; `yastn/tn/fpeps/envs/fixed_pt.py`, checked 2026-09-26)** — the same requirement, enforced after the fact. Its forward CTMRG truncates globally across sectors (`truncation_mask`: per-block `D_block`, then one global sort and a `D_total` cut), so sector sizes move during convergence. Its fixed-point AD then runs one more CTM step and raises `NoFixedPointError("T tensors' symmetry sectors change after a CTM step!")` if any T leg's per-sector dimensions changed (`fixed_pt.py:369`); the per-sector gauge matrices only exist when they match. The PEPS parameters enter AD as raw data with fixed metadata, so the state's sectors are frozen too. Its simple update moves sectors (global `D_total`); its full update "assumes fixed sectorial bond dimensions" via a `D_block` dict -- `pin_sectors` by another name. Being eager, a moving layout costs YASTN nothing during forward; for us it costs a recompile per change, which is why we freeze *before* AD instead of only checking after. Task 3 borrows the check. **Measured in YASTN (numpy backend, NTU 'NN', global `D_total`, spinless fermions Z2, t=1 V=0 dt=0.05, 2-site checkerboard, `$SCRATCH/yastn_su_sectors.py`):** 0 layout changes in 300 steps at D=3 from a (2,1) or a (1,2) start (5 seeds each), 0 at D=2 (1,1); growing D=2 -> `D_total=3`, exactly 1 change, at step 1, and which sector gets the slot is seed- and bond-dependent (seed 0: three (2,1) bonds, one (1,2)). At D=4 and D=5 the same: 0 changes from (2,2), (3,1), (3,2), (2,3) or (4,1) starts (3 seeds each); growing to `D_total=4` from (1,1) or (2,1), exactly 1 change at step 1, and the split differs bond to bond within one run ((2,2) with a (1,3) or (3,1) bond). Sector sizes lock at the first truncation; the layout is decided when D grows, hence Task 2's staged growth -- one slot per stage, because a two-slot jump (D=2 -> 4) assigns both slots from the same random spectrum.
  **And CTM AD moves the chi layout when allowed** (`yastn_ad_sectors2.py`, fp_ctmrg, D=3 chi=12, 3 seeds): with a global `D_total` cut the environment split changed between gradient steps (6/6 <-> 5/7 <-> 7/5 per leg) and every free run then failed to converge within 2-7 steps, right after a move; with a hard per-sector pin (`D_block` alone) the layout never moved, one seed ran 20/20 steps, and the energies agree with the free run to 3e-4/site where both exist. The failures are layout limit cycles: at the failing step the split flips every sweep (seed 1 period 2, corner-spectrum change stuck at 9.5e-2 for 20 sweeps; seed 3 period 3; seed 2 an irregular wobble). So freezing the environment layout (Task 3/5) is not only what tracing forces -- an eager CTM that re-decides the split under AD has no fixed point to differentiate at.

## Global Constraints

- Start from `main` **after** the #1035 Phase 3 PR merges (stacked PRs get no CI). Branch: `feat/1035-frozen-layout-ad`.
- One PR against `main`; never push to `main`; don't arm `--auto` unasked; read both review endpoints with `--paginate` before merging.
- `uv run` for everything; `JAX_PLATFORMS=cpu`; `python -u` for runs over a minute; never pipe a long run to `tail`.
- `jax.config.update("jax_enable_x64", True)` in every new test file.
- Every new test gets a named mutant it kills; commit before mutating; assert anchors are unique.
- Fixtures assert their regime (the layout actually moves; truncation actually happens).
- 2-site checkerboard only. 1-site fermionic AD (`optimize_fpeps_ad`) is out of scope: it cannot hold the CDW, and on it the pin is structurally required.
- `chi_auto_bump` and χ schedules are **refused** with a seeded environment: a bump changes the layout the seed froze.
- The default behaviour of every existing entry point is unchanged (pin stays on unless asked; `envs_init=None` behaves as today).
- New public keyword arguments are documented in the docstring and in `README.md`.

## Review Focus

1. **The traced CTM drifts off the seeded layout.** If `_incoming_chi_charges` returns `None` for some direction it falls back to the double-layer `base_charges`, which is *not* the seed. Expected: after N AD steps every environment's pytree structure equals the seed's. Pinned in Tasks 3 and 5.
2. **Eager and traced fixed points differ on the same layout.** Eager keeps global top-χ each sweep; traced keeps the seed's per-sector counts. Seeded from the eager fixed point they must agree. Expected: |ΔE| ≤ 1e-8. Pinned in Task 3; never loosen — a failure is a design finding.
3. **A sign lost only in the backward pass.** A graded sign applied in a way AD does not see (e.g. through `stop_gradient` or a Python-side branch) leaves every forward test green while the gradient is the hard-core-boson one. Expected: AD directional derivative equals a finite difference within its h-scan uncertainty. Pinned in Task 4 with a named mutant.
4. **An unpinned SU collapses a sector to zero.** Growing sectors can also empty one (seed 0 collapses both pinned and unpinned). Expected: the freeze step reports a collapsed bond instead of freezing it. Pinned in Task 2.
5. **`envs_init` that does not match the tensors** (wrong χ, or built for different site tensors' layouts). Expected: a `ValueError` naming the mismatch, not a trace error deep in a contraction. Pinned in Task 5.

---

## File Structure

- Modify `src/tenax/algorithms/ipeps_simple_update.py` — `pin_sectors` on `_simple_update_checkerboard_sweep` and the two 2-site bond updates.
- Modify `src/tenax/algorithms/fermionic_ipeps.py` — `bond_layout`, `su_grow_layout` (staged growth).
- Modify `src/tenax/algorithms/ipeps_optimize.py` — `envs_init` on `optimize_gs_ad` → `_optimize_gs_ad_tensor_2site`.
- Modify `src/tenax/__init__.py`, `README.md` — export and document `su_grow_layout`; the SU → freeze → AD recipe.
- Create `tests/test_frozen_layout_ad.py` — all tests for this plan.
- Modify `tests/test_graded_simple_update.py` — its `exact` fixture uses `pin_sectors=False` instead of monkeypatching.

---

### Task 1: An unpinned 2-site SU (`pin_sectors`)

**Files:**
- Modify: `src/tenax/algorithms/ipeps_simple_update.py:56-80` (`_truncation_base_charges`), `:238` (`_simple_update_checkerboard_sweep`), the horizontal and vertical 2-site updates (`_truncation_base_charges(A, "r")` at `:524`/`:561`, `(A, "d")` at `:668`/`:699`)
- Modify: `tests/test_graded_simple_update.py:92-96`
- Test: `tests/test_frozen_layout_ad.py`

**Interfaces:**
- Produces: `_truncation_base_charges(A, leg, pin_sectors: bool = True)`; keyword `pin_sectors: bool = True` on `_simple_update_checkerboard_sweep`, `_simple_update_2site_horizontal_tensor`, `_simple_update_2site_vertical_tensor`; `bond_layout(A, B) -> tuple[tuple[int, int], ...]` in `fermionic_ipeps.py` — for each of A's bond legs `(u, d, l, r)` in that order, the pair `(n_even, n_odd)`.

- [ ] **Step 1: Write the failing tests**

```python
"""Frozen SU layout -> traced graded CTM + AD (#1035 follow-up)."""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _initialize_fpeps,
    _trotter_gate,
    bond_layout,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps_simple_update import _simple_update_checkerboard_sweep

jax.config.update("jax_enable_x64", True)


def _su(pin, D=3, seed=2, steps=4 * 40):
    cfg = FPEPSConfig(D=D, t=1.0, V=0.0, dt=0.05)
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(seed))
    gate = _trotter_gate(spinless_fermion_gate(cfg), cfg.dt)
    return A0, _simple_update_checkerboard_sweep(
        A0, A0, gate, D, steps, pin_sectors=pin
    )


def test_bond_layout_counts_even_and_odd_per_bond():
    A0, _ = _su(True, steps=0)
    lay = bond_layout(A0, A0)
    assert len(lay) == 8 and all(sum(p) == 3 for p in lay)


def test_the_pinned_su_never_moves_the_layout():
    A0, (A, B, _) = _su(True)
    assert bond_layout(A, B) == bond_layout(A0, A0)


def test_the_unpinned_su_grows_the_layout():
    A0, (A, B, _) = _su(False)
    lay = bond_layout(A, B)
    # Regime: the top-D truncation wants a different sector split than the
    # initial one -- otherwise this test cannot tell pinned from unpinned.
    assert lay != bond_layout(A0, A0)
    # Both ends of each bond agree: A.r/B.l and A.d/B.u (legs u, d, l, r).
    assert lay[3] == lay[4 + 2] and lay[1] == lay[4 + 0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest -q --no-cov -m "core or not core" tests/test_frozen_layout_ad.py`
Expected: FAIL at import (`cannot import name 'bond_layout'`).

- [ ] **Step 3: Implement**

`_truncation_base_charges` gains the switch (docstring: "``pin_sectors=False`` restores the global top-D truncation on a fermionic 2-site bond, which lets the SU move weight between sectors; the pin is required only on the 1-site path, where ``A.l``/``A.r`` are one bond"):

```python
def _truncation_base_charges(
    A: Tensor, leg: str, pin_sectors: bool = True
) -> np.ndarray | None:
    if not pin_sectors:
        return None
    if not A.indices[A.labels().index(leg)].symmetry.is_fermionic:
        return None
    return np.asarray(A.indices[A.labels().index(leg)].charges)
```

Add `pin_sectors: bool = True` (keyword-only) to `_simple_update_2site_horizontal_tensor` and `_simple_update_2site_vertical_tensor`, and pass it at all four `_truncation_base_charges(A, ...)` call sites: `_truncation_base_charges(A, "r", pin_sectors)`, `_truncation_base_charges(A, "d", pin_sectors)`. Add `pin_sectors: bool = True` to `_simple_update_checkerboard_sweep`'s keyword-only arguments (after `phase0`), document it, and pass it to every bond-update call inside.

In `fermionic_ipeps.py`:

```python
def bond_layout(A: SymmetricTensor, B: SymmetricTensor) -> tuple[tuple[int, int], ...]:
    """``(n_even, n_odd)`` per bond leg of ``A`` then ``B``, legs ``u, d, l, r``.

    The quantity the simple update grows and AD must hold fixed.  ``B``'s legs
    are included because on the checkerboard they are the far ends of ``A``'s
    bonds, and a mismatch between them is a broken state, not a layout.
    """
    out = []
    for T in (A, B):
        for leg in ("u", "d", "l", "r"):
            ix = T.indices[T.labels().index(leg)]
            par = np.asarray(ix.symmetry.parity(np.asarray(ix.charges)))
            out.append((int((par == 0).sum()), int((par == 1).sum())))
    return tuple(out)
```

In `tests/test_graded_simple_update.py`, replace the `exact` fixture's monkeypatch by passing `pin_sectors=False` to `update(...)` and delete the fixture.

- [ ] **Step 4: Run to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest -q --no-cov -m "core or not core" tests/test_frozen_layout_ad.py tests/test_graded_simple_update.py tests/test_fermionic_ipeps.py tests/test_ipeps_su.py`
Expected: PASS. If `test_the_unpinned_su_grows_the_layout` fails on its regime assert, scan `seed ∈ {1,2,3}` and `D ∈ {3,4}` and pick a fixture where the layout moves; record the choice in the test's comment. Do not delete the assert.

- [ ] **Step 5: Commit, then mutate**

```bash
git add src/tenax/algorithms/ipeps_simple_update.py src/tenax/algorithms/fermionic_ipeps.py tests/test_frozen_layout_ad.py tests/test_graded_simple_update.py
git commit -m "feat(#1035): pin_sectors switch on the 2-site SU; bond_layout"
```

Mutant: in `_truncation_base_charges`, delete the `if not pin_sectors: return None` lines (the switch is ignored, everything stays pinned). Expected: `test_the_unpinned_su_grows_the_layout` FAILS on `lay != ...`; the pinned test still PASSES. Restore with `git checkout`.

---

### Task 2: Grow the layout in stages with the eager SU (`su_grow_layout`)

**Why staged growth, not patience (measured in YASTN, 2026-09-26, `$SCRATCH/yastn_su_sectors.py`):** with a global top-D truncation the sector sizes lock at the *first* truncation and never move again -- 0 changes in 300 steps at D=3 from either a (2,1) or a (1,2) start, and exactly 1 change (at step 1) when growing from D=2 to `D_total=3`. The mechanism is generic: each step the enlarged bond carries the old singular values plus new ones of order dt, so a top-D cut keeps the old sectors unless one has decayed below dt times the largest. The only moment the SU reads the physics into the layout is therefore the truncation that *adds* a slot. So the layout is built one slot at a time: relax at D, raise `max_D` by one (the new slot goes to the sector with the largest discarded weight), relax, repeat. A patience counter would stop after the first cycle and call a random split "frozen".

Start at D=2 with the (1,1) split `_initialize_fpeps` gives, not D=1: a D=1 bond is a single even sector, which with a parity-even site tensor forces the physical leg even -- the vacuum, on which hopping does nothing (the YASTN (3,0)/(0,3) runs showed exactly that).

**Files:**
- Modify: `src/tenax/algorithms/fermionic_ipeps.py`, `src/tenax/__init__.py`
- Test: `tests/test_frozen_layout_ad.py`

**Interfaces:**
- Consumes: `bond_layout`, `_simple_update_checkerboard_sweep(A, B, gate, max_D, steps, lambdas=, pin_sectors=)`, `_to_physical_pair`, `_initialize_fpeps(config, key)` (called with `config.D` replaced by `D_start` via `dataclasses.replace`).
- Produces:

```python
class FrozenSU(NamedTuple):
    A: SymmetricTensor          # physical (lambda-absorbed) tensors, ready for CTM/AD
    B: SymmetricTensor
    layout: tuple[tuple[int, int], ...]        # bond_layout at the end
    stages: tuple[tuple[tuple[int, int], ...], ...]  # layout at the end of each stage, D_start..D
    frozen: bool                # False: the layout moved after the first cycle of the last stage
```

`su_grow_layout(gate: SymmetricTensor, config: FPEPSConfig, *, D_start: int = 2, cycles_per_stage: int = 25, final_cycles: int = 50, key=None) -> FrozenSU`. One cycle is the four checkerboard phases (`steps=4`).

- [ ] **Step 1: Write the failing tests** (mechanism: the SU is replaced by a scripted layout sequence)

```python
import dataclasses

import tenax.algorithms.fermionic_ipeps as fi


def _scripted(monkeypatch, layouts):
    """Replace one SU cycle and the layout probe by a script; record max_D."""
    it = iter(layouts)
    calls = []

    def fake_sweep(A, B, gate, max_D, steps, lambdas=None, **kw):
        calls.append((max_D, steps, kw.get("pin_sectors")))
        return A, B, lambdas

    monkeypatch.setattr(fi, "_simple_update_checkerboard_sweep", fake_sweep)
    monkeypatch.setattr(fi, "bond_layout", lambda A, B: next(it))
    monkeypatch.setattr(fi, "_to_physical_pair", lambda A, B, lam: (A, B))
    return calls


def _lay(n_even, n_odd):
    return ((n_even, n_odd),) * 8


def test_it_raises_max_d_one_slot_per_stage(monkeypatch):
    # stage 2: 3 cycles at (1,1); stage 3: 4 cycles, slot goes odd at cycle 1
    layouts = [_lay(1, 1)] * 3 + [_lay(1, 2)] * 4
    calls = _scripted(monkeypatch, layouts)
    cfg = FPEPSConfig(D=3)
    out = fi.su_grow_layout(
        spinless_fermion_gate(cfg), cfg, D_start=2, cycles_per_stage=3, final_cycles=4
    )
    assert [c[0] for c in calls] == [2] * 3 + [3] * 4
    assert all(c[1] == 4 and c[2] is False for c in calls)  # 4 phases, unpinned
    assert out.stages == (_lay(1, 1), _lay(1, 2))
    assert out.layout == _lay(1, 2) and out.frozen


def test_a_layout_that_moves_after_the_first_cycle_is_not_frozen(monkeypatch):
    layouts = [_lay(1, 1)] * 3 + [_lay(1, 2), _lay(2, 1), _lay(1, 2), _lay(1, 2)]
    _scripted(monkeypatch, layouts)
    cfg = FPEPSConfig(D=3)
    out = fi.su_grow_layout(
        spinless_fermion_gate(cfg), cfg, D_start=2, cycles_per_stage=3, final_cycles=4
    )
    assert not out.frozen


def test_it_refuses_to_freeze_a_collapsed_sector(monkeypatch):
    layouts = [_lay(1, 1)] * 3 + [_lay(3, 0)] * 4  # the odd sector died at growth
    _scripted(monkeypatch, layouts)
    cfg = FPEPSConfig(D=3)
    with pytest.raises(ValueError, match="collapsed"):
        fi.su_grow_layout(
            spinless_fermion_gate(cfg), cfg, D_start=2, cycles_per_stage=3, final_cycles=4
        )


def test_d_start_below_two_is_refused():
    cfg = FPEPSConfig(D=3)
    with pytest.raises(ValueError, match="D_start"):
        fi.su_grow_layout(spinless_fermion_gate(cfg), cfg, D_start=1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest -q --no-cov -m "core or not core" tests/test_frozen_layout_ad.py -k "stage or frozen or collapsed or D_start"`
Expected: FAIL (`AttributeError: ... su_grow_layout`).

- [ ] **Step 3: Implement** in `fermionic_ipeps.py`

```python
class FrozenSU(NamedTuple):
    """The output of :func:`su_grow_layout` (fields in the plan)."""

    A: SymmetricTensor
    B: SymmetricTensor
    layout: tuple[tuple[int, int], ...]
    stages: tuple[tuple[tuple[int, int], ...], ...]
    frozen: bool


def su_grow_layout(
    gate: SymmetricTensor,
    config: FPEPSConfig,
    *,
    D_start: int = 2,
    cycles_per_stage: int = 25,
    final_cycles: int = 50,
    key: jax.Array | None = None,
) -> FrozenSU:
    """Build the bond-sector layout one slot at a time with an eager, unpinned
    simple update, then stop.

    A global top-D truncation locks the sector sizes at the first truncation
    and never moves them again (measured in tenax and in YASTN), so the only
    step that reads the physics into the layout is the one that adds a slot:
    the new slot goes to the sector with the largest discarded weight.  This
    relaxes at ``D_start``, raises ``max_D`` by one, relaxes again, up to
    ``config.D``.  The last stage runs ``final_cycles`` cycles; ``frozen`` is
    whether the layout stayed put after that stage's first cycle.

    ``D_start`` must be at least 2: a D=1 bond is one even sector, which
    forces a parity-even site into the vacuum, on which hopping does nothing.
    A bond with an empty parity sector is refused rather than frozen -- AD
    could never refill it (#878).
    """
    if D_start < 2:
        raise ValueError("su_grow_layout: D_start must be >= 2 (D=1 is the vacuum)")
    key = jax.random.PRNGKey(0) if key is None else key
    A = B = _initialize_fpeps(dataclasses.replace(config, D=D_start), key)
    trotter = _trotter_gate(gate, config.dt)
    lam = None
    stages = []
    frozen = True
    for D in range(D_start, config.D + 1):
        n_cycles = final_cycles if D == config.D else cycles_per_stage
        first = None
        for cycle in range(n_cycles):
            A, B, lam = _simple_update_checkerboard_sweep(
                A, B, trotter, D, 4, lambdas=lam, pin_sectors=False
            )
            lay = bond_layout(A, B)
            if D == config.D:
                if cycle == 0:
                    first = lay
                elif lay != first:
                    frozen = False
        stages.append(lay)
    if any(0 in pair for pair in lay):
        raise ValueError(
            f"su_grow_layout: a bond sector collapsed to zero {lay}; AD cannot "
            "refill it. Try another key (see #878)."
        )
    A_phys, B_phys = _to_physical_pair(A, B, lam)
    return FrozenSU(A_phys, B_phys, lay, tuple(stages), frozen)
```

(`import dataclasses` at the top of the module.) Export `su_grow_layout` and `FrozenSU` in `src/tenax/__init__.py` (`__all__`).

- [ ] **Step 4: Run to verify they pass**, then the real thing as a slow test

Append:

```python
@pytest.mark.slow
def test_growth_puts_the_new_slot_where_the_discarded_weight_is():
    """The one moment the SU reads the physics: growing D=2 -> 3, the sector
    that gains the slot is the sector of the 3rd-largest singular value of the
    untruncated bond.  Then the layout stays put (YASTN: 0 moves in 300 steps)."""
    from tenax.algorithms.ipeps_simple_update import _graded_bond_update, scale_bond_axis

    cfg = FPEPSConfig(D=3, t=1.0, V=0.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    out = su_grow_layout(H, cfg, D_start=2, cycles_per_stage=10, final_cycles=20,
                         key=jax.random.PRNGKey(2))
    assert out.frozen
    assert all(sum(p) == 3 for p in out.layout)
    assert out.layout[3] == out.layout[4 + 2] and out.layout[1] == out.layout[4 + 0]
    # Regime + mechanism: redo the first growth truncation by hand on the
    # stage-2 state and read where the 3rd singular value lives.
    cfg2 = dataclasses.replace(cfg, D=2)
    A2 = B2 = fi._initialize_fpeps(cfg2, jax.random.PRNGKey(2))
    lam = None
    for _ in range(10):
        A2, B2, lam = _simple_update_checkerboard_sweep(
            A2, B2, _trotter_gate(H, cfg.dt), 2, 4, lambdas=lam, pin_sectors=False
        )
    # first phase of the growth cycle: horizontal A.r -- B.l, lambdas absorbed
    A_abs = scale_bond_axis(scale_bond_axis(scale_bond_axis(A2, "u", lam.v_BA), "d", lam.v_AB), "l", lam.h_BA)
    A_abs = scale_bond_axis(A_abs, "r", lam.h_AB)
    B_abs = scale_bond_axis(scale_bond_axis(scale_bond_axis(B2, "u", lam.v_AB), "d", lam.v_BA), "r", lam.h_BA)
    U, sigma, _ = _graded_bond_update(A_abs, B_abs, _trotter_gate(H, cfg.dt), "r", "l", None, None)
    order = np.argsort(-np.asarray(sigma))
    charges = np.asarray(U.indices[U.labels().index("r")].charges)[order[:3]]
    par = np.asarray(U.indices[0].symmetry.parity(charges))
    assert abs(np.asarray(sigma)[order[2]] - np.asarray(sigma)[order[3]]) > 1e-6  # not degenerate
    assert out.stages[1][3] == (int((par == 0).sum()), int((par == 1).sum()))
```

The absorption above is phase 0 of `_simple_update_checkerboard_sweep` (`ipeps_simple_update.py:295`, then `:506-517`): A gets `u<-v_BA, d<-v_AB, l<-h_BA, r<-h_AB`, B gets `u<-v_AB, d<-v_BA, r<-h_BA` (verified 2026-09-26; re-check the line numbers, not the mapping). If `_graded_bond_update` rejects `max_D=None`, pass `max_D=6` (the full rank of a D=2, d=2 bond).

Run: `JAX_PLATFORMS=cpu uv run python -u -m pytest -q --no-cov -m "core or not core" tests/test_frozen_layout_ad.py --durations=0`
Expected: PASS. If the mechanism assertion fails, print `sigma` with its charges and the stage layouts: either the phase-0 absorption above is wrong (fix the test) or the first growth truncation is not a global top-3 (then `pin_sectors=False` is not reaching the SVD -- a Task 1 defect).

- [ ] **Step 5: Commit, then mutate**

```bash
git add src/tenax/algorithms/fermionic_ipeps.py src/tenax/__init__.py tests/test_frozen_layout_ad.py
git commit -m "feat(#1035): su_grow_layout -- eager unpinned SU that grows the sector layout one slot per stage"
```

Mutants (one at a time, `git checkout` after each): M1 start every stage at `max_D=config.D` (no growth: `range(config.D, config.D + 1)`) → `raises_max_d_one_slot_per_stage` FAILS on the `max_D` sequence. M2 never set `frozen = False` → `moves_after_the_first_cycle` FAILS. M3 drop the collapse check → `collapsed` FAILS. M4 `pin_sectors=True` in the sweep call → the slow test FAILS (the stage-3 layout stays (1,1)+1 wherever the pin's tiling puts it, and the mechanism assertion breaks; record which).

---

### Task 3: The seeded traced CTM holds the frozen environment layout

**Files:**
- Test: `tests/test_frozen_layout_ad.py`

**Interfaces:**
- Consumes: `su_grow_layout` (Task 2), `ctm_tensor_2site` (eager), `python_loop_ctm_converge(site_tensors, neighbors, *, chi, max_iter, conv_tol, env_init, ...) -> (envs, info)`, `compute_energy_ctm_tensor_2site`.
- Produces: fixture `frozen_state` → `(A, B, H, d)`; fixture `eager_envs` → `{(0,0): env_A, (1,0): env_B}`.

- [ ] **Step 1: Write the tests**

```python
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    _ctm_tensor_sweep_multisite,
    ctm_tensor_2site,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
from tenax.algorithms.fermionic_ipeps import su_grow_layout

CHI = 12


@pytest.fixture(scope="module")
def frozen_state():
    cfg = FPEPSConfig(D=3, t=1.0, V=0.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    out = su_grow_layout(H, cfg, D_start=2, key=jax.random.PRNGKey(2))
    assert out.frozen
    return out.A, out.B, H, 2


@pytest.fixture(scope="module")
def eager_envs(frozen_state):
    A, B, _, _ = frozen_state
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        eA, eB = ctm_tensor_2site(A, B, CHI, max_iter=150, conv_tol=1e-10)
    assert eA.C1.indices[0].dim == CHI  # regime: the corner is truncated
    envs = {(0, 0): eA, (1, 0): eB}
    # YASTN's fixed-point check (fixed_pt.py:369): one more eager sweep must
    # not move any sector size, or there is no fixed point to hand to AD.
    dls = {c: _build_double_layer_tensor(t) for c, t in ((0, 0), A), ((1, 0), B)}
    again, _, _ = _ctm_tensor_sweep_multisite(envs, dls, CHECKERBOARD_NEIGHBORS, CHI, True)
    assert jax.tree_util.tree_structure(again) == jax.tree_util.tree_structure(envs)
    return envs


def _E(A, B, envs, H, d):
    return float(compute_energy_ctm_tensor_2site(A, B, envs[(0, 0)], envs[(1, 0)], H, d))


@pytest.mark.slow
def test_the_seeded_traced_ctm_keeps_the_layout_and_the_energy(
    frozen_state, eager_envs
):
    A, B, H, d = frozen_state
    envs, _ = python_loop_ctm_converge(
        {(0, 0): A, (1, 0): B},
        CHECKERBOARD_NEIGHBORS,
        chi=CHI,
        max_iter=50,
        conv_tol=1e-10,
        env_init=eager_envs,
    )
    tree = jax.tree_util.tree_structure
    assert tree(envs) == tree(eager_envs)
    assert abs(_E(A, B, envs, H, d) - _E(A, B, eager_envs, H, d)) <= 1e-8


@pytest.mark.slow
def test_the_cold_traced_ctm_is_why_the_seed_exists(frozen_state, eager_envs):
    """Measurement, kept as a regime guard: from the tiled initial
    environment the traced CTM lands on a different layout.  If this ever
    fails, the seed is a no-op for this fixture and Task 3's first test
    certifies nothing -- pick a fixture where it matters."""
    A, B, _, _ = frozen_state
    envs, _ = python_loop_ctm_converge(
        {(0, 0): A, (1, 0): B},
        CHECKERBOARD_NEIGHBORS,
        chi=CHI,
        max_iter=50,
        conv_tol=1e-10,
    )
    tree = jax.tree_util.tree_structure
    assert tree(envs) != tree(eager_envs)
```

- [ ] **Step 2: Run**

Run: `JAX_PLATFORMS=cpu uv run python -u -m pytest -q --no-cov -m "core or not core" tests/test_frozen_layout_ad.py -k "seeded or cold" --durations=0`
Expected: both PASS. **If the first fails, do not loosen it** (Review Focus 1–2): print, for each env tensor, `np.unique(ix.charges, return_counts=True)` of its χ legs for seed vs output, and the two energies; stop and report — this is the design's core assumption failing. If the second fails, the seed doesn't matter on this fixture: scan χ ∈ {9, 12, 16} for one where it does; if none does, report it (it would mean the layout freeze is automatic and Task 5's plumbing is optional).

- [ ] **Step 3: Commit**

```bash
git add tests/test_frozen_layout_ad.py
git commit -m "test(#1035): a seeded traced CTM holds the eager environment layout and energy"
```

- [ ] **Step 4: Mutation**

In `_ctm_tensor_projector_2x2.py`, assert the anchor `traced_base = _incoming_chi_charges(Q_TL, Q_TR, Q_BL, Q_BR, direction, chi)` is unique, then replace it with `traced_base = None`, which makes the traced SVD fall back to the double-layer charges. Expected: `test_the_seeded_traced_ctm_keeps...` FAILS. Restore.

---

### Task 4: The graded AD gradient is right (FD check with a backward-only mutant)

**Files:**
- Test: `tests/test_frozen_layout_ad.py`

**Interfaces:**
- Consumes: `frozen_state`, `eager_envs` (Task 3); `make_ctm_energy_fn` (`ipeps_ad_policy.py:227`), built exactly as `_optimize_gs_ad_tensor_2site` builds `_ctm_energy_fn_2s`, on the implicit path.

- [ ] **Step 1: Write the test**

```python
import jax.numpy as jnp

from tenax import CTMConfig
from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn


def _ad_energy_fn(H, d, env_init):
    """The 2-site optimizer's loss, implicit AD, warm-started from
    ``env_init`` (the cache is updated in place between calls, as in the
    optimizer; every call converges to 1e-10 so the start does not matter)."""
    cfg = CTMConfig(chi=CHI, max_iter=50, conv_tol=1e-10)
    energy = make_ctm_energy_fn(
        neighbors=CHECKERBOARD_NEIGHBORS,
        gate=H,
        get_ctm_cfg=lambda: cfg,
        env_cache={"envs": env_init},
        use_explicit=False,
        explicit_warmup=0,
        explicit_steps=0,
        energy_fn=lambda st, envs, g: compute_energy_ctm_tensor_2site(
            st[(0, 0)], st[(1, 0)], envs[(0, 0)], envs[(1, 0)], g, d
        ),
    )

    def f(A, B):
        A = A * (1.0 / (A.norm() + 1e-10))
        B = B * (1.0 / (B.norm() + 1e-10))
        return energy({(0, 0): A, (1, 0): B})

    return f


@pytest.mark.slow
def test_the_graded_ad_gradient_matches_finite_differences(frozen_state, eager_envs):
    A, B, H, d = frozen_state
    f = _ad_energy_fn(H, d, env_init=eager_envs)
    rng = np.random.default_rng(0)
    V = jax.tree_util.tree_map(lambda x: jnp.asarray(rng.standard_normal(x.shape)), A)
    g = jax.grad(lambda A_: f(A_, B))(A)
    slope = float(sum(jnp.vdot(a, b).real for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(V))))
    fd = {}
    for h in (1e-3, 3e-4, 1e-4, 3e-5):
        Ap = jax.tree_util.tree_map(lambda x, v: x + h * v, A, V)
        Am = jax.tree_util.tree_map(lambda x, v: x - h * v, A, V)
        fd[h] = (f(Ap, B) - f(Am, B)) / (2 * h)
    vals = np.array(list(fd.values()))
    fd_unc = vals.max() - vals.min()  # h-scan spread (memory: FD needs an h scan)
    assert abs(slope) > 10 * fd_unc  # regime: the slope is measurable
    assert abs(slope - np.median(vals)) <= 3 * fd_unc + 1e-7, (slope, fd)
```

- [ ] **Step 2: Run**

Run: `JAX_PLATFORMS=cpu uv run python -u -m pytest -q --no-cov -m "core or not core" tests/test_frozen_layout_ad.py -k finite_differences --durations=0`
Expected: PASS. Report `slope`, the four FD values and `fd_unc`. If the regime assert fails, choose another `V` seed; if the comparison fails, check first whether the forward is converged (`conv_tol` 1e-10; memory: the adjoint needs a converged forward) and whether corner spectra are degenerate (the ~5e-4 degenerate-SV floor), before suspecting the graded backward.

- [ ] **Step 3: Commit, then the backward-only mutant**

```bash
git add tests/test_frozen_layout_ad.py
git commit -m "test(#1035): the graded AD gradient matches an h-scanned finite difference"
```

Mutant — a graded sign the forward applies but the backward cannot see. In `src/tenax/algorithms/_ctm_graded.py`, `dense_rdm`, wrap the sign so it is invisible to AD:

```python
        signed = _scale_blocks(rdm_t, ...)          # the existing call, unchanged
        rdm_t = jax.tree_util.tree_map(
            lambda s, u: u + jax.lax.stop_gradient(s - u), signed, rdm_t
        )
```

The forward value is identical (every forward test stays green — check `tests/test_graded_ctm.py`); the gradient is the unsigned one. Expected: this test FAILS. Restore.

---

### Task 5: `envs_init` on `optimize_gs_ad` (2-site), and the end-to-end run

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (`optimize_gs_ad` signature and docstring; `_optimize_gs_ad_tensor_2site` signature; `_env_cache_2s` initialisation)
- Modify: `README.md` (the fPEPS section near `README.md:673`)
- Test: `tests/test_frozen_layout_ad.py`

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces: `optimize_gs_ad(hamiltonian_gate, init, config, *, envs_init: dict[Coord, CTMTensorEnv] | None = None)`; valid only with `unit_cell="2site"`, fused CTM, no `chi_auto_bump`, no χ schedule.

- [ ] **Step 1: Write the failing tests**

```python
from tenax import CTMConfig, iPEPSConfig
from tenax.algorithms import _ctm_python_loop as pl
from tenax.algorithms.ipeps_optimize import optimize_gs_ad


def _ad_config(steps, **ctm):
    return iPEPSConfig(
        max_bond_dim=3,
        unit_cell="2site",
        su_init=False,
        gs_implicit_ad=True,
        gs_num_steps=steps,
        gs_verbose=False,
        ctm=CTMConfig(chi=CHI, max_iter=50, conv_tol=1e-9, **ctm),
    )


def test_envs_init_is_the_first_forward_seed(frozen_state, eager_envs, monkeypatch):
    A, B, H, _ = frozen_state
    seeds = []
    real = pl.python_loop_ctm_converge
    monkeypatch.setattr(
        pl,
        "python_loop_ctm_converge",
        lambda *a, **k: seeds.append(k.get("env_init")) or real(*a, **k),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        optimize_gs_ad(H, (A, B), _ad_config(1), envs_init=eager_envs)
    assert seeds and seeds[0] is eager_envs


def test_envs_init_refuses_a_chi_bump(frozen_state, eager_envs):
    A, B, H, _ = frozen_state
    with pytest.raises(ValueError, match="envs_init.*chi_auto_bump"):
        optimize_gs_ad(
            H, (A, B), _ad_config(1, chi_auto_bump=True), envs_init=eager_envs
        )


def test_envs_init_refuses_a_mismatched_chi(frozen_state, eager_envs):
    A, B, H, _ = frozen_state
    cfg = _ad_config(1)
    cfg = iPEPSConfig(**{**cfg.__dict__, "ctm": CTMConfig(chi=CHI + 2)})
    with pytest.raises(ValueError, match="envs_init.*chi"):
        optimize_gs_ad(H, (A, B), cfg, envs_init=eager_envs)


@pytest.mark.slow
def test_frozen_layout_ad_lowers_the_energy_and_keeps_the_layouts(
    frozen_state, eager_envs
):
    A, B, H, d = frozen_state
    E0 = _E(A, B, eager_envs, H, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        (A1, B1), envs1, E1 = optimize_gs_ad(
            H, (A, B), _ad_config(10), envs_init=eager_envs
        )
    tree = jax.tree_util.tree_structure
    assert tree((A1, B1)) == tree((A, B))  # site layout frozen
    assert tree(envs1) == tree(eager_envs)  # environment layout frozen
    assert E1 <= E0 + 1e-6, (E0, E1)
    assert E1 >= -8 / np.pi**2 - 1e-3  # V=0 free-fermion bound, one-sided only
```

(If `iPEPSConfig`/`CTMConfig` are frozen dataclasses, build the mismatched config with `dataclasses.replace`.)

- [ ] **Step 2: Run to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest -q --no-cov -m "core or not core" tests/test_frozen_layout_ad.py -k envs_init`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'envs_init'`).

- [ ] **Step 3: Implement**

In `optimize_gs_ad`, add the keyword-only `envs_init=None`, document it ("A converged environment to seed the first forward CTM with. Under tracing the CTM keeps the χ-sector layout it is given, so this fixes the environment layout for the whole optimisation — pass the output of an eager `ctm_tensor_2site` on the initial tensors. 2-site fused only; refused with `chi_auto_bump` or a χ schedule."), and forward it to `_optimize_gs_ad_tensor_2site` only; for any other unit cell with `envs_init is not None`, raise `ValueError("envs_init is supported on unit_cell='2site' only")`.

In `_optimize_gs_ad_tensor_2site(hamiltonian_gate, AB_init, config, envs_init=None)`, right after `_env_cache_2s: dict[str, dict] = {}`:

```python
    if envs_init is not None:
        if use_split_2s:
            raise ValueError("envs_init: the split CTM is not supported; use fuse_virtual_legs=True")
        if ctm_cfg_2s.chi_auto_bump or config.gs_chi_schedule_steps is not None:
            raise ValueError(
                "envs_init fixes the environment layout; chi_auto_bump and a chi "
                "schedule change it. Turn them off."
            )
        chi_seen = {e.C1.indices[0].dim for e in envs_init.values()}
        if chi_seen != {ctm_cfg_2s.chi}:
            raise ValueError(
                f"envs_init has chi {sorted(chi_seen)}, the CTM config chi={ctm_cfg_2s.chi}"
            )
        _env_cache_2s["envs"] = envs_init
```

Then confirm with the spy test that the **first** `python_loop_ctm_converge` call receives it. If it does not, the first loss evaluation builds its own initial environment inside `make_ctm_energy_fn` (`ctm_converge_kwargs(..., env_init=...)`): pass `_env_cache_2s.get("envs")` there as well, and rerun the spy test.

In `README.md`, after the existing `fpeps` example, add the recipe:

````markdown
```python
from tenax import FPEPSConfig, spinless_fermion_gate, su_grow_layout
from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad, ctm_tensor_2site

cfg = FPEPSConfig(D=3, t=1.0, V=0.0, dt=0.05)
H = spinless_fermion_gate(cfg)
su = su_grow_layout(H, cfg)            # eager SU grows the sectors one slot per stage
eA, eB = ctm_tensor_2site(su.A, su.B, 12)      # eager CTM picks the chi layout
(A, B), envs, E = optimize_gs_ad(              # traced AD keeps both layouts
    H, (su.A, su.B),
    iPEPSConfig(max_bond_dim=3, unit_cell="2site", su_init=False,
                gs_implicit_ad=True, ctm=CTMConfig(chi=12)),
    envs_init={(0, 0): eA, (1, 0): eB},
)
```
````

(Check each import against `tenax.__all__`; import from the defining module if one is not exported.)

- [ ] **Step 4: Run to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run python -u -m pytest -q --no-cov -m "core or not core" tests/test_frozen_layout_ad.py tests/test_fpeps_ad.py --durations=0`
Expected: PASS. Record E0, E1 from the end-to-end test.

- [ ] **Step 5: Commit, then mutate**

```bash
git add src/tenax/algorithms/ipeps_optimize.py README.md tests/test_frozen_layout_ad.py
git commit -m "feat(#1035): envs_init seeds 2-site AD with a frozen environment layout"
```

Mutant: delete `_env_cache_2s["envs"] = envs_init`. Expected: `test_envs_init_is_the_first_forward_seed` FAILS, and (slow) `...keeps_the_layouts` FAILS on the env tree if Task 3's cold-start test showed the layouts differ. Restore.

---

### Task 6: Measure the AD step on the frozen layout

No code. #566 found block-sparse VJP compiles slow; the frozen layout should make that a one-time cost per run. This task checks.

- [ ] **Step 1: Time it**

For D=2 (χ=8) and D=3 (χ=12, 18) on `su_grow_layout` states: time the first `optimize_gs_ad(..., gs_num_steps=1, envs_init=...)` call (compile + one step) and then a second call with `gs_num_steps=5` in the same process (cache hits). Use `python -u`, record `uptime` load next to the numbers.

- [ ] **Step 2: Check the cache is actually hit**

Count traces of `_ctm_python_loop._ctm_tensor_sweep_multisite` (monkeypatched counter, as a scratch script) over the 5-step call. Expected: 0 new traces for the forward step after the first call. More than 0 means some layout is still moving under AD — go back to Task 3's layout dump.

- [ ] **Step 3: Record**

Post the table (compile s, s/step, traces) as a `🤖`-marked PR comment, next to the #566 baseline (D=2 χ=8: 206 s compile; D=3 χ=12: ~35 min, A100). The number that matters is the **second** call's s/step with zero new traces: that is the cost the frozen layout amortises the compile over. If the second call still compiles, some layout moved — go back to Task 3's layout dump. Do not optimise the compile inside this PR; #566 owns it.

---

## Out of scope, noted

- **The `_JIT_STEP_CACHE` id-reuse hazard.** `_make_jit_ctm_step` keys on `id(neighbors)`; `ctm_multisite` builds a fresh dict per call, so after GC a different topology can reuse the id and get the wrong compiled step. Real, independent of this plan — file it as its own issue.
- **A jitted forward-only loop for χ scans.** Not needed: `python_loop_ctm_converge` already is one; call it directly (with an eager seed, per Task 3) for the #1035 χ-convergence scan.
- **1-site fermionic AD** (`optimize_fpeps_ad`) and FermionicU1 layouts.
