# CTM Corner Leg Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The multisite 2x2 CTM sweep must return every corner in the leg order of the cold init. Then a cold start compiles each jitted piece only once.

**Architecture:**
- The four `_ctm_tensor_absorb_*_2plaq` moves are the only sweep code that writes corners in the swapped order. Every other sweep path keeps the documented order:
  - `C1[c1_d, c1_r]`
  - `C2[c2_l, c2_d]`
  - `C3[c3_u, c3_l]`
  - `C4[c4_r, c4_u]`
- A single table and a single helper in `_ctm_tensor_init.py` set that order. The four moves pass each corner through the helper.
- The helper is a static, label-based `transpose`. It is a no-op when the order is already right. For fermionic symmetries it carries the Koszul sign.

**Tech Stack:**
- Python 3.11/3.12
- JAX 0.10.2, x64
- pytest
- Tenax `DenseTensor` / `SymmetricTensor`

**Spec:** No separate spec. This plan argues from the evidence below, which was measured on 2026-10-24 on branch `perf/eager-layout-sweep-jit` (PR #1142, head 3755c05). The probe scripts are in `/tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/legorder/`. Rulings made without a spec are provisional.

### Problem record

| Probe | Result |
|---|---|
| `probe.py`: U(1) D=2 chi=6 `python_loop_ctm_converge` | The cold init has `C1 (c1_d IN, c1_r OUT)`. After eager sweep 1, all four corners are swapped: `C1 (c1_r, c1_d)`, `C2 (c2_d, c2_l)`, `C3 (c3_l, c3_u)`, `C4 (c4_u, c4_r)`. T1–T4 are unchanged. |
| `probe_paths.py`: one sweep per path, dense and U(1) | `_ctm_tensor_sweep` keeps the order (SAME). `_ctm_tensor_sweep_paired` keeps it (SAME). `_ctm_tensor_sweep_multisite(recipe="1x1")` keeps it (SAME). `_ctm_tensor_sweep_multisite(recipe="2x2")` swaps it (FLIP). |
| `probe_step.py`: `_make_jit_ctm_step(CHECKERBOARD_NEIGHBORS, "2x2")` called 3 times from the cold init | dense: 1, 1, 0 compiles (0.2 s each). U(1): 1, 1, 0 compiles (7.4 s, 7.6 s). `_cache_size()` is 2 after the first two calls, for each input type. |
| #1142 Task 5, D=3 U(1) chi=8 GPU | Sweep 2 does 58 compiles (about 21 s). The sector layout, leaf shapes and block keys into sweep 2 equal those of sweep 1. Only the corner leg order differs. |

So every cold start pays for one extra full jitted-step compile, for dense and symmetric input alike. It also pays for one extra compile of each `_ctm_eager_jit` piece. `_chi_layout` is keyed by label, so the #1098 layout logic does not see the swap (`_ctm_python_loop.py:200-222`).

## Global Constraints

- **Precondition.** Do not start Task 1 until PR #1142 has merged into `main`. Then run `git -C ~/tenax-legorder merge origin/main`. Use merge, not rebase. This plan depends on `src/tenax/algorithms/_ctm_eager_jit.py`, `benchmarks/ctm_eager_layout_compile.py` and `tests/test_ctm_eager_sweep_jit.py` from #1142.
- **Python.** Use the worktree's own venv: `~/tenax-legorder/.venv/bin/python`. Create it with:
  ```bash
  cd ~/tenax-legorder && uv venv .venv && uv pip install -p .venv/bin/python -e '.[dev,cuda13]'
  ```
  Use cuda13, never cuda12, because cuda12 hangs on this host.
- **CPU test runs:** `JAX_PLATFORMS=cpu taskset -c 0-15`. Run one suite at a time, and only when the 1-minute load in `uptime` is below 40.
- **GPU runs:** use only `CUDA_VISIBLE_DEVICES=1`. GPU 0 runs a long user job (pid 869113). Never touch it.
- **Long runs:** redirect output to a log file and wait on the pid. Never pipe a long run to `tail` or `head`.
- **Values must not change.** After you align legs by label, every env tensor of a sweep must equal the old result to `atol=1e-12`. This holds for dense, U(1) and FermionicU1 input.
- **Order is the only change.** Corners from every 2x2 multisite sweep must have exactly the leg order in `CORNER_LEG_ORDER`. The T1–T4 order must not change.
- **Scope.** Do not touch the other sweep paths (`_ctm_tensor_sweep`, `_ctm_tensor_sweep_paired`, the 1x1 recipe), the cold init, or any code that reads envs.
- **Mutation tests.** Every new test gets a mutation run:
  1. Commit first.
  2. Use a unique anchor, and assert that it occurs exactly once.
  3. Set `PYTHONDONTWRITEBYTECODE=1` and delete `__pycache__`.
  4. Restore with `git checkout`.
- **Commit trailers:**
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
  ```
- **No push and no PR inside a task.** The finishing step does that, and the user decides.

## Review Focus

1. **Fermionic corners.** `SymmetricTensor.transpose` applies a Koszul sign to odd blocks (`core/tensor.py:1281-1292`). The contractor no longer tracks Koszul signs automatically (#555). A reasonable user expects the FermionicU1 energy to be unchanged. Test: `test_sweep_values_match_the_old_order[fermionic_u1]` in Task 1. If it fails, stop with BLOCKED. Do not search for a sign rule yourself.
2. **Warm start from an env in the old order** (a checkpoint, a harness env, or an env from an older Tenax). The user expects it to work and to come back in the canonical order. Test: `test_old_order_input_comes_back_canonical` in Task 1.
3. **Dense input.** The dense 2x2 multisite path also swaps today. The user expects the same values and one compile. Every Task 1 test is run with dense input as well.
4. **The eager-jit cache key.** The #1142 `_jitted` cache keys on the function object. A monkeypatched helper does not reach pieces that were traced before the patch. Every test that patches `_to_corner_order` must also set `_ctm_eager_jit._ENABLED=False` and call the eager sweep directly.
5. **AD and fixed-point consumers.** They are label-based. Any of them that kept a swapped-order env (a frozen layout, the #1135 recheck) now gets a canonical env. Gate: the core suite in Task 2, plus `tests/test_ctm_eager_sweep_jit.py` and `tests/test_enlarged_corner_flow_invariant_834.py`.

---

## File Structure

- **Modify** `src/tenax/algorithms/_ctm_tensor_init.py`. Add `CORNER_LEG_ORDER` and `_to_corner_order` next to `CTMTensorEnv`. This file owns the env layout.
- **Modify** `src/tenax/algorithms/_ctm_tensor_moves.py`. In each of `_ctm_tensor_absorb_left_2plaq`, `_right_2plaq`, `_top_2plaq` and `_bottom_2plaq`, pass each returned corner through `_to_corner_order`.
- **Modify** `src/tenax/algorithms/_ctm_python_loop.py`. This is a docstring only. `_adapt_symmetric_layout` and `_chi_layout` mention the cold-init swap; make the text true after the fix.
- **Create** `tests/test_ctm_corner_leg_order.py`, mapped to `"core"` in `tests/conftest.py` `_FILE_MARKERS`.

---

### Task 1: Canonical corner order out of the 2x2 multisite moves

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_init.py`, after `class CTMTensorEnv` (line 39 on c082d7a; find it with `command grep -n "class CTMTensorEnv"`).
- Modify: `src/tenax/algorithms/_ctm_tensor_moves.py`, at the return line of each `_ctm_tensor_absorb_*_2plaq`, near lines 638, 721, 793 and 846 on c082d7a.
- Modify: `src/tenax/algorithms/_ctm_python_loop.py`, the docstrings of `_chi_layout` and `_adapt_symmetric_layout`.
- Modify: `tests/conftest.py`, `_FILE_MARKERS`.
- Test: `tests/test_ctm_corner_leg_order.py`.

**Interfaces:**
- Consumes:
  - `initialize_ctm_tensor_env(A, chi) -> CTMTensorEnv`
  - `_ctm_tensor_sweep_multisite(envs, double_layers, neighbors, chi, renormalize, projector_method="svd", projector_backward="auto", recipe="2x2", device_mesh=None, chunk_size=None) -> (envs, eps, ...)`
  - `_make_jit_ctm_step(neighbors, recipe="2x2")` returns a jitted function `step(site_tensors, envs, *, chi, ...)`. Its output `[0]` is the new envs. It has `_cache_size()`. It is memoised by `id(neighbors)`, so pass a fresh `copy.deepcopy(CHECKERBOARD_NEIGHBORS)` to get a fresh cache.
  - `_build_double_layer_tensor(A)` from `tenax.algorithms._ctm_python_loop`
  - `tenax.algorithms._ctm_eager_jit._ENABLED`
- Produces:
  - `CORNER_LEG_ORDER: dict[str, tuple[str, str]]`
  - `_to_corner_order(t: Tensor, field: str) -> Tensor`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ctm_corner_leg_order.py`:

```python
"""Corners leave the 2x2 multisite CTM sweep in the cold-init leg order.

The ``_ctm_tensor_absorb_*_2plaq`` moves used to return the corners with their
two legs swapped.  ``jax.jit`` keys its cache on the pytree treedef, which
holds the leg order, so every cold start compiled the jitted step and each
#1142 eager-jit piece twice.  The values were never affected: only the order.
"""

from __future__ import annotations

import copy

import jax
import numpy as np
import pytest

import tenax.algorithms._ctm_eager_jit as ej
import tenax.algorithms._ctm_python_loop as cpl
import tenax.algorithms._ctm_tensor_moves as moves
from tenax import DenseTensor
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    _ctm_tensor_sweep_multisite,
)
from tenax.algorithms._ctm_tensor_init import (
    CORNER_LEG_ORDER,
    _to_corner_order,
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionicU1
from tenax.core.tensor import SymmetricTensor

jax.config.update("jax_enable_x64", True)

CHI = 6
_EDGE_ORDER = {
    "T1": ("t1_l", "u2", "t1_r"),
    "T2": ("t2_u", "r2", "t2_d"),
    "T3": ("t3_r", "d2", "t3_l"),
    "T4": ("t4_d", "l2", "t4_u"),
}


def _u1_pair():
    A, B = heisenberg_u1sz_init_pair(2, jax.random.PRNGKey(0))
    return {(0, 0): A, (1, 0): B}


def _dense_pair():
    return {c: DenseTensor(t.todense(), t.indices) for c, t in _u1_pair().items()}


def _fermionic_pair():
    """A random FermionicU1 pair, normalised.

    Copied from ``_fermionic_pair`` in ``tests/test_ctm_eager_sweep_jit.py``
    (itself from ``tests/test_ctm_674_fermionic_fused.py``).
    """
    sym = FermionicU1()

    def idx(charges, flow, label):
        return TensorIndex.from_charges(
            sym, np.array(charges, dtype=np.int32), flow, label=label
        )

    lA = [0, 1, -1]
    rA = [0, 0, 1]
    ud = [0, 1, -1]
    phys = [0, 1]

    def make(l_charges, r_charges, key):
        return SymmetricTensor.random_normal(
            (
                idx(ud, FlowDirection.OUT, "u"),
                idx(ud, FlowDirection.IN, "d"),
                idx(l_charges, FlowDirection.OUT, "l"),
                idx(r_charges, FlowDirection.IN, "r"),
                idx(phys, FlowDirection.IN, "phys"),
            ),
            jax.random.PRNGKey(key),
        )

    A = make(lA, rA, 0)
    B = make(rA, lA, 1)
    return {
        (0, 0): A * (1.0 / (A.norm() + 1e-10)),
        (1, 0): B * (1.0 / (B.norm() + 1e-10)),
    }


PAIRS = {"dense": _dense_pair, "u1": _u1_pair, "fermionic_u1": _fermionic_pair}


def _order(t):
    return tuple(ix.label for ix in t.indices)


def _cold(sites):
    return {c: initialize_ctm_tensor_env(t, CHI) for c, t in sites.items()}


def _one_sweep(sites, envs):
    dls = {c: cpl._build_double_layer_tensor(t) for c, t in sites.items()}
    return _ctm_tensor_sweep_multisite(
        envs, dls, CHECKERBOARD_NEIGHBORS, CHI, True, recipe="2x2"
    )[0]


def _dense_by_label(t, labels):
    perm = tuple(_order(t).index(lab) for lab in labels)
    arr = t.transpose(perm).todense()
    return np.asarray(arr)


def test_the_table_is_the_cold_init_order():
    env = initialize_ctm_tensor_env(_u1_pair()[(0, 0)], CHI)
    assert {f: _order(getattr(env, f)) for f in CORNER_LEG_ORDER} == CORNER_LEG_ORDER


def test_to_corner_order_is_a_noop_on_the_canonical_order():
    env = initialize_ctm_tensor_env(_u1_pair()[(0, 0)], CHI)
    assert _to_corner_order(env.C1, "C1") is env.C1


@pytest.mark.parametrize("kind", ["dense", "u1"])
def test_2x2_sweep_returns_the_cold_init_order(kind, monkeypatch):
    monkeypatch.setattr(ej, "_ENABLED", False)
    sites = PAIRS[kind]()
    out = _one_sweep(sites, _cold(sites))
    for c, env in out.items():
        for f, labels in CORNER_LEG_ORDER.items():
            assert _order(getattr(env, f)) == labels, (kind, c, f)
        for f, labels in _EDGE_ORDER.items():
            assert _order(getattr(env, f)) == labels, (kind, c, f)


@pytest.mark.parametrize("kind", ["dense", "u1", "fermionic_u1"])
def test_sweep_values_match_the_old_order(kind, monkeypatch):
    """Same values as the swapped-order sweeps, once legs are aligned by label.

    Two sweeps, so every corner that a move transposed is consumed by a later
    move: the compared tensors include ones built from transposed corners,
    which is where a wrong Koszul sign would show.
    """
    monkeypatch.setattr(ej, "_ENABLED", False)
    sites = PAIRS[kind]()
    new = _one_sweep(sites, _one_sweep(sites, _cold(sites)))
    monkeypatch.setattr(moves, "_to_corner_order", lambda t, field: t)
    old = _one_sweep(sites, _one_sweep(sites, _cold(sites)))
    swapped = 0
    for c in new:
        for f in new[c]._fields:
            a, b = getattr(new[c], f), getattr(old[c], f)
            labels = _order(a)
            swapped += labels != _order(b)
            np.testing.assert_allclose(
                _dense_by_label(a, labels), _dense_by_label(b, labels), atol=1e-12
            )
    assert swapped > 0, "the identity patch must reproduce the old swapped order"



def test_old_order_input_comes_back_canonical(monkeypatch):
    monkeypatch.setattr(ej, "_ENABLED", False)
    sites = _u1_pair()
    envs = _cold(sites)
    flipped = {
        c: env._replace(**{f: getattr(env, f).transpose((1, 0)) for f in CORNER_LEG_ORDER})
        for c, env in envs.items()
    }
    out = _one_sweep(sites, flipped)
    for env in out.values():
        for f, labels in CORNER_LEG_ORDER.items():
            assert _order(getattr(env, f)) == labels


@pytest.mark.parametrize("kind", ["dense", "u1"])
def test_jitted_step_compiles_once_from_a_cold_start(kind):
    sites = PAIRS[kind]()
    envs = _cold(sites)
    step = cpl._make_jit_ctm_step(copy.deepcopy(CHECKERBOARD_NEIGHBORS), "2x2")
    for _ in range(3):
        envs = step(sites, envs, chi=CHI)[0]
    assert step._cache_size() == 1
```

Add this entry to `_FILE_MARKERS` in `tests/conftest.py`:

```python
    "test_ctm_corner_leg_order.py": "core",
```

- [ ] **Step 2: Run the tests and check that they fail for the right reason**

Run:
```bash
cd ~/tenax-legorder && JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_ctm_corner_leg_order.py -q > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/legorder/t1_red.log 2>&1
```

Expected: an ImportError on `CORNER_LEG_ORDER` and `_to_corner_order` (collection fails). Record the log line.

- [ ] **Step 3: Add the table and the helper**

In `src/tenax/algorithms/_ctm_tensor_init.py`, directly after `class CTMTensorEnv(...)`:

```python
# Leg order of each corner, as ``initialize_ctm_tensor_env`` builds it and as
# every CTM sweep must return it.  ``jax.jit`` keys its cache on the pytree
# treedef, which includes the leg order, so a sweep that returned a corner
# with its legs swapped made every cold start compile the jitted step twice.
CORNER_LEG_ORDER: dict[str, tuple[str, str]] = {
    "C1": ("c1_d", "c1_r"),
    "C2": ("c2_l", "c2_d"),
    "C3": ("c3_u", "c3_l"),
    "C4": ("c4_r", "c4_u"),
}


def _to_corner_order(t: Tensor, field: str) -> Tensor:
    """``t`` with its legs in ``CORNER_LEG_ORDER[field]``; ``t`` itself if already so.

    The check runs on static labels, so under ``jax.jit`` it adds no op when the
    order is right.  ``transpose`` carries the Koszul sign for fermionic
    symmetries, so the result is the same tensor in the canonical leg order.
    """
    labels = tuple(t.labels())
    want = CORNER_LEG_ORDER[field]
    if labels == want:
        return t
    return t.transpose(tuple(labels.index(lab) for lab in want))
```

If `Tensor` is not imported in that module, import it the way the module already imports tensor types.

- [ ] **Step 4: Route each corner through the helper in the four 2plaq moves**

In `src/tenax/algorithms/_ctm_tensor_moves.py`, add `_to_corner_order` to the existing import from `tenax.algorithms._ctm_tensor_init`, or add a new import line if there is none. Then change the return statement of each move.

`_ctm_tensor_absorb_left_2plaq`:
```python
    return _to_corner_order(C1_new, "C1"), T4_new, _to_corner_order(C4_new, "C4")
```

`_ctm_tensor_absorb_right_2plaq`: read the current return statement. It returns `C2_new`, `T2_new` and `C3_new` in some order. Keep that order, and wrap the two corners as `_to_corner_order(C2_new, "C2")` and `_to_corner_order(C3_new, "C3")`.

`_ctm_tensor_absorb_top_2plaq`:
```python
    return _to_corner_order(C1_new, "C1"), T1_new, _to_corner_order(C2_new, "C2")
```

`_ctm_tensor_absorb_bottom_2plaq`: read the current return statement. Keep its order, and wrap the two corners as `_to_corner_order(C4_new, "C4")` and `_to_corner_order(C3_new, "C3")`.

Put the wrap after `_phase_fix_normalize_tensor`, at the return, so it is the last op on each corner. Wrap all eight corner returns, including the ones already in canonical order. The helper is a no-op there, and this keeps the rule visible and safe against future edits.

- [ ] **Step 5: Run the tests and check that they pass**

Run the same command as Step 2, with the log `t1_green.log`.

Expected: all pass, including `test_sweep_values_match_the_old_order[fermionic_u1]`.

If the fermionic case fails, check that `_fermionic_pair` is the verbatim copy. If it still fails, stop with status BLOCKED. Report the max difference per field (Review Focus 1).

- [ ] **Step 6: Make the `_ctm_python_loop.py` docstrings true**

- In `_chi_layout`, the parenthesis "(the cold-init corners come back with their two legs swapped)" is no longer true for the 2x2 sweep. Replace it with: "(a sweep or a warm-start env may hold a tensor's legs in another order)".
- In `_adapt_symmetric_layout`, check that the sentence "so equal layouts give equal pytree structures" is now true from a cold start. Leave it as it is if so. If any other sentence claims the swap, fix it.

- [ ] **Step 7: Commit**

```bash
cd ~/tenax-legorder && .venv/bin/pre-commit run --files src/tenax/algorithms/_ctm_tensor_init.py src/tenax/algorithms/_ctm_tensor_moves.py src/tenax/algorithms/_ctm_python_loop.py tests/test_ctm_corner_leg_order.py tests/conftest.py
git add src/tenax/algorithms/_ctm_tensor_init.py src/tenax/algorithms/_ctm_tensor_moves.py src/tenax/algorithms/_ctm_python_loop.py tests/test_ctm_corner_leg_order.py tests/conftest.py
git commit -m "perf(ctm): return 2x2 multisite corners in the cold-init leg order

The _ctm_tensor_absorb_*_2plaq moves returned the corners with their two
legs swapped, so a cold start compiled the jitted CTM step (and every
#1142 eager-jit piece) twice.  One table and one label-based transpose
now fix the order; the values do not change.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

- [ ] **Step 8: Mutation-test the new tests**

Do this after the commit. For each mutant, apply it with a unique anchor (assert that the anchor occurs exactly once), set `PYTHONDONTWRITEBYTECODE=1`, delete `__pycache__`, run the test file, record the result, and restore with `git checkout -- <file>`.

| Mutant | Change | Must be killed by |
|---|---|---|
| L1 | In `_to_corner_order`, change `if labels == want:` to `if True:` | `test_2x2_sweep_returns_the_cold_init_order`, `test_jitted_step_compiles_once_from_a_cold_start`, `test_old_order_input_comes_back_canonical` |
| L2 | In `_ctm_tensor_absorb_top_2plaq`, change the return to `return C1_new, T1_new, _to_corner_order(C2_new, "C2")` | `test_2x2_sweep_returns_the_cold_init_order` |
| L3 | In `_to_corner_order`, change `return t.transpose(...)` to `return t.transpose(...) * 2.0` | `test_sweep_values_match_the_old_order` |
| L4 | In `CORNER_LEG_ORDER`, change `"C1": ("c1_d", "c1_r")` to `"C1": ("c1_r", "c1_d")` | `test_the_table_is_the_cold_init_order` |

Record each result in the report. After the last mutant, `git status` must be clean.

- [ ] **Step 9: Regression set**

Run these one at a time. For each, first record the pass count on the pre-task commit in this worktree (the commit before Step 7), then on HEAD. They must match.

```bash
JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_ctm_eager_sweep_jit.py -q
JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_enlarged_corner_flow_invariant_834.py -q
JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_ctm_eager_jit.py -q
```

Send each command's output to its own log file in the scratchpad `legorder/` directory.

---

### Task 2: Measure, and run the core gate

**Files:**
- Modify: none in `src/`. The numbers go into the report and the PR body.

**Interfaces:**
- Consumes: `benchmarks/ctm_eager_layout_compile.py` from #1142. It prints per-sweep compile counts and walls, the layout per sweep, `converge:` with wall and compiles, `energy per site`, and `device peak_bytes_in_use`.

- [ ] **Step 1: chi=8 on GPU 1, main against the branch**

Create the baseline worktree on the main commit this branch merged (Global Constraints):

```bash
git -C ~/tenax worktree add ~/tenax-legorder-base $(git -C ~/tenax-legorder merge-base HEAD origin/main)
cd ~/tenax-legorder-base && uv venv .venv && uv pip install -p .venv/bin/python -e '.[cuda13]'
```

Run each command below with `/usr/bin/time -v`, in the background, one after the other:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/ctm_eager_layout_compile.py --D 3 --chi 8 > <scratchpad>/legorder/bench_base_chi8.log 2>&1   # in ~/tenax-legorder-base
CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/ctm_eager_layout_compile.py --D 3 --chi 8 > <scratchpad>/legorder/bench_head_chi8.log 2>&1   # in ~/tenax-legorder
```

Acceptance:
- Sweep 2 compiles on HEAD are at most 5. The base had 58 in #1142 Task 5.
- The `energy per site` lines are equal to all printed digits.
- The layout line of every sweep is equal.

- [ ] **Step 2: chi=16 on GPU 1, main against the branch**

Run the same two commands as Step 1 with `--chi 16 --max-iter 40`, in the background, one after the other, never in parallel.

Record:
- the compiles and wall of each sweep;
- the converge wall and compiles;
- the energy;
- the device peak;
- the max RSS.

Acceptance:
- The energies are equal.
- The layouts are equal.
- The converge compiles on HEAD are below the base.

- [ ] **Step 3: Core gate**

```bash
cd ~/tenax-legorder && JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest -m core -q > <scratchpad>/legorder/core.log 2>&1
```

Expected: all pass. Report the summary line.

- [ ] **Step 4: Report**

Write a before/after table: sweep 1 and sweep 2 compiles, converge wall, compiles, energy, device peak, max RSS, and the log path for each number.

Do not push. The finishing step asks the user about push and the PR.

---

## Self-review notes

- **Coverage:**
  - The swap is fixed in Task 1.
  - The double jit compile is tested by `test_jitted_step_compiles_once_from_a_cold_start`.
  - The 58 eager recompiles are measured in Task 2 Steps 1–2.
  - Values are tested by `test_sweep_values_match_the_old_order` for three symmetry kinds.
  - Warm starts in the old order are tested by `test_old_order_input_comes_back_canonical`.
- **Not in scope:**
  - the other sweep paths, which were measured to keep the order;
  - the cold init, which is already canonical;
  - checkpoint migration, which is not needed because loading is label-based and costs at most one extra compile;
  - `device_mesh` sharding, which is not tested either way.
