# CTM Bug 3a — chi_init=1 in fixed-shape container — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Tenax's standard-CTM rank-D padded-identity init with variPEPS-style rank-1 init inside the existing chi-target-shaped container, so generic complex iPEPS converges to the physical fixed point instead of the paired-degenerate one.

**Architecture:** Four init functions get a one-line behavioural change (rank-D init → rank-1 init). Sweep loop, projector, absorption, AD path are untouched. The fixed-chi-target shape contract is preserved, so JIT signatures don't change. Zero modes introduced in early sweeps are pruned by the rank-aware SVD truncation that already ships (PR #400).

**Tech Stack:** JAX, NumPy, pytest, ruff/pre-commit.

**Design doc:** `docs/plans/2026-05-11-ctm-bug-3a-design.md`

---

## Pre-flight

Branch already exists: `fix/ctm-bug-3a-chi-init` (off `origin/main` at `5372f26`).

Run from repo root: `cd /home/yjkao/tenax`.

---

## Task 1: Init invariant test for dense standard corner

**Files:**
- Create: `tests/test_ctm_tensor_init_rank1.py`
- (No source change yet — TDD red.)

**Step 1: Write the failing test**

Create `tests/test_ctm_tensor_init_rank1.py` with:

```python
"""Init invariants for bug 3a (chi_init=1, rank-1 padded init)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_init import (
    _init_symmetric_standard_corner,
    _init_symmetric_standard_edge,
    _make_dense_standard_edge,
    initialize_ctm_tensor_env,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


def _peps_dense(D: int = 2, d: int = 2):
    sym = U1Symmetry()
    rng = np.random.RandomState(0)
    data = jnp.array(rng.standard_normal((D, D, D, D, d)))
    indices = (
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, indices)


@pytest.mark.parametrize("D, chi", [(2, 4), (2, 16), (3, 9)])
def test_initialize_dense_corner_rank1(D, chi):
    """Standard-CTM dense corner is rank-1: only entry (0, 0) is non-zero."""
    A = _peps_dense(D=D)
    env = initialize_ctm_tensor_env(A, chi)
    for C in (env.C1, env.C2, env.C3, env.C4):
        arr = np.asarray(C._data)
        assert arr.shape == (chi, chi)
        assert arr[0, 0] == pytest.approx(1.0)
        # Everything else exactly zero.
        mask = np.ones_like(arr, dtype=bool)
        mask[0, 0] = False
        assert np.all(arr[mask] == 0), f"corner has non-zero entries outside (0, 0)"
```

**Step 2: Run test to verify it fails (RED)**

Run: `uv run pytest tests/test_ctm_tensor_init_rank1.py::test_initialize_dense_corner_rank1 -v --no-cov`

Expected: FAIL — current code makes `eye(min(chi, D²))` zero-padded, so `arr[1, 1]`, `arr[2, 2]`, … are 1.0. The first failing assertion is `np.all(arr[mask] == 0)`.

**Step 3: Write minimal implementation**

Edit `src/tenax/algorithms/_ctm_tensor_init.py`. Add a new private helper near `_make_dense_standard_edge` (above it):

```python
def _make_rank1_dense_corner(
    chi: int,
    label_a: Label,
    label_b: Label,
    flow_a: FlowDirection,
    flow_b: FlowDirection,
    dtype,
) -> DenseTensor:
    """Rank-1 identity-like corner for the standard CTM chi_init=1 init.

    Writes only entry ``(0, 0) = 1`` inside the chi-target-shaped buffer.
    The rest of the (chi, chi) corner stays zero until subsequent CTM
    absorptions grow chi via SVD truncation.  Mirrors variPEPS's
    ``chi_init=1`` semantics (rank-1 corner) without breaking the
    fixed-shape JIT contract.
    """
    from tenax.core.symmetry import U1Symmetry

    sym = U1Symmetry()
    C = jnp.zeros((chi, chi), dtype=dtype).at[0, 0].set(1.0)
    return DenseTensor(
        C,
        (
            TensorIndex.from_charges(
                sym, np.zeros(chi, dtype=np.int32), flow_a, label=label_a
            ),
            TensorIndex.from_charges(
                sym, np.zeros(chi, dtype=np.int32), flow_b, label=label_b
            ),
        ),
    )
```

Then in `initialize_ctm_tensor_env`, swap the dense corner builder. Find:

```python
    else:
        corners = {}
        for name, (la, lb, fa, fb, _ref) in _CORNER_SPECS.items():
            corners[name] = _make_dense_corner(chi, D2, la, lb, fa, fb, dtype)
```

Replace with:

```python
    else:
        corners = {}
        for name, (la, lb, fa, fb, _ref) in _CORNER_SPECS.items():
            corners[name] = _make_rank1_dense_corner(chi, la, lb, fa, fb, dtype)
```

Add `_make_rank1_dense_corner` to the module's `__all__` (line 5).

The shared `_make_dense_corner` in `_ctm_utils.py` is intentionally untouched — split CTM still uses it. (Confirmed by `grep -rn "_make_dense_corner" src/` — only standard CTM and split CTM call it.)

**Step 4: Run test to verify it passes (GREEN)**

Run: `uv run pytest tests/test_ctm_tensor_init_rank1.py::test_initialize_dense_corner_rank1 -v --no-cov`

Expected: PASS for all 3 parametrizations.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_init.py tests/test_ctm_tensor_init_rank1.py
git commit -m "fix(ctm): rank-1 dense corner init (bug 3a, part 1)"
```

---

## Task 2: Init invariant test for dense standard edge

**Files:**
- Modify: `tests/test_ctm_tensor_init_rank1.py` (add a new test)
- Modify: `src/tenax/algorithms/_ctm_tensor_init.py:175-215` (`_make_dense_standard_edge`)

**Step 1: Write the failing test**

Append to `tests/test_ctm_tensor_init_rank1.py`:

```python
@pytest.mark.parametrize("D, chi", [(2, 4), (2, 16), (3, 9)])
def test_initialize_dense_edge_rank1(D, chi):
    """Standard-CTM dense edge has only D non-zero entries at (0, j*(D+1), 0)."""
    A = _peps_dense(D=D)
    env = initialize_ctm_tensor_env(A, chi)
    diag_idx = np.arange(D) * (D + 1)
    for T in (env.T1, env.T2, env.T3, env.T4):
        arr = np.asarray(T._data)
        assert arr.shape == (chi, D * D, chi)
        # Expected: T[0, diag_idx, 0] = 1, everything else 0.
        for j in range(D):
            assert arr[0, j * (D + 1), 0] == pytest.approx(1.0)
        mask = np.ones_like(arr, dtype=bool)
        for j in range(D):
            mask[0, j * (D + 1), 0] = False
        assert np.all(arr[mask] == 0), "edge has non-zero entries outside the rank-1 slot"
```

**Step 2: Run test to verify it fails (RED)**

Run: `uv run pytest tests/test_ctm_tensor_init_rank1.py::test_initialize_dense_edge_rank1 -v --no-cov`

Expected: FAIL — current edge writes `T[i, diag_idx, i] = 1` for `i in range(min(chi, D))`, so for D=2 chi=4 we get `T[1, 0, 1]` and `T[1, 3, 1]` both 1.0; the mask check fails.

**Step 3: Write minimal implementation**

Edit `src/tenax/algorithms/_ctm_tensor_init.py`. Find `_make_dense_standard_edge`:

```python
    D = int(np.round(np.sqrt(D2)))
    assert D * D == D2, f"D² leg dim {D2} is not a perfect square"
    diag_idx = np.arange(D, dtype=np.int32) * (D + 1)
    T = jnp.zeros((chi, D2, chi), dtype=dtype)
    T_chi = min(chi, D)
    for i in range(T_chi):
        T = T.at[i, diag_idx, i].set(jnp.ones(D, dtype=dtype))
```

Replace with:

```python
    # variPEPS chi_init=1: write the δ_{ket=bra} pattern only on the
    # leading (i=0) chi slot; subsequent absorptions grow chi via SVD
    # truncation.  See docs/plans/2026-05-11-ctm-bug-3a-design.md.
    D = int(np.round(np.sqrt(D2)))
    assert D * D == D2, f"D² leg dim {D2} is not a perfect square"
    diag_idx = np.arange(D, dtype=np.int32) * (D + 1)
    T = jnp.zeros((chi, D2, chi), dtype=dtype)
    T = T.at[0, diag_idx, 0].set(jnp.ones(D, dtype=dtype))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_tensor_init_rank1.py -v --no-cov`

Expected: 6 passes (3 corner parametrizations + 3 edge parametrizations).

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_init.py tests/test_ctm_tensor_init_rank1.py
git commit -m "fix(ctm): rank-1 dense edge init (bug 3a, part 2)"
```

---

## Task 3: Init invariant test for SymmetricTensor standard corner

**Files:**
- Modify: `tests/test_ctm_tensor_init_rank1.py` (add a new test)
- Modify: `src/tenax/algorithms/_ctm_tensor_init.py:265-301` (`_init_symmetric_standard_corner`)

**Step 1: Write the failing test**

Append:

```python
def _peps_symmetric(D: int = 2, d: int = 2):
    sym = U1Symmetry()
    rng = np.random.RandomState(99)
    data = jnp.array(rng.standard_normal((D, D, D, D, d)))
    indices = (
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label="phys"),
    )
    return SymmetricTensor.from_dense(data, indices)


@pytest.mark.parametrize("D, chi", [(2, 4), (2, 16), (3, 9)])
def test_initialize_symmetric_corner_rank1(D, chi):
    """Symmetric standard corner is rank-1: dense view has only (0, 0) = 1."""
    A = _peps_symmetric(D=D)
    env = initialize_ctm_tensor_env(A, chi)
    for C in (env.C1, env.C2, env.C3, env.C4):
        assert isinstance(C, SymmetricTensor)
        arr = np.asarray(C.todense())
        assert arr.shape == (chi, chi)
        assert arr[0, 0] == pytest.approx(1.0)
        mask = np.ones_like(arr, dtype=bool)
        mask[0, 0] = False
        assert np.all(arr[mask] == 0)
```

**Step 2: Run test to verify it fails (RED)**

Run: `uv run pytest tests/test_ctm_tensor_init_rank1.py::test_initialize_symmetric_corner_rank1 -v --no-cov`

Expected: FAIL — current code uses `jnp.eye(chi)`, so `arr[1, 1] = arr[2, 2] = … = 1`.

**Step 3: Write minimal implementation**

In `_init_symmetric_standard_corner`, find:

```python
    return SymmetricTensor.from_dense(
        jnp.eye(chi, dtype=A.dtype),
        (idx_a, idx_b),
    )
```

Replace with:

```python
    # variPEPS chi_init=1: rank-1 corner — only the leading (0, 0) entry
    # is non-zero. Subsequent absorptions grow chi via SVD truncation.
    C_dense = jnp.zeros((chi, chi), dtype=A.dtype).at[0, 0].set(1.0)
    return SymmetricTensor.from_dense(
        C_dense,
        (idx_a, idx_b),
        tol=float("inf"),
    )
```

(`tol=float("inf")` matches the dense-edge symmetric path that already ships and avoids spurious "non-zero outside sectors" rejections from `from_dense`'s validator on the trivial-charge case.)

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_tensor_init_rank1.py -v --no-cov`

Expected: 9 passes.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_init.py tests/test_ctm_tensor_init_rank1.py
git commit -m "fix(ctm): rank-1 symmetric corner init (bug 3a, part 3)"
```

---

## Task 4: Init invariant test for SymmetricTensor standard edge

**Files:**
- Modify: `tests/test_ctm_tensor_init_rank1.py` (add a new test)
- Modify: `src/tenax/algorithms/_ctm_tensor_init.py:218-262` (`_init_symmetric_standard_edge`)

**Step 1: Write the failing test**

Append:

```python
@pytest.mark.parametrize("D, chi", [(2, 4), (2, 16), (3, 9)])
def test_initialize_symmetric_edge_rank1(D, chi):
    """Symmetric standard edge: dense view has only D non-zero entries at (0, j*(D+1), 0)."""
    A = _peps_symmetric(D=D)
    env = initialize_ctm_tensor_env(A, chi)
    for T in (env.T1, env.T2, env.T3, env.T4):
        assert isinstance(T, SymmetricTensor)
        arr = np.asarray(T.todense())
        assert arr.shape == (chi, D * D, chi)
        for j in range(D):
            assert arr[0, j * (D + 1), 0] == pytest.approx(1.0)
        mask = np.ones_like(arr, dtype=bool)
        for j in range(D):
            mask[0, j * (D + 1), 0] = False
        assert np.all(arr[mask] == 0)
```

**Step 2: Run test to verify it fails (RED)**

Run: `uv run pytest tests/test_ctm_tensor_init_rank1.py::test_initialize_symmetric_edge_rank1 -v --no-cov`

Expected: FAIL — current edge loop writes the δ pattern across `i ∈ 0..min(chi, D)-1`.

**Step 3: Write minimal implementation**

In `_init_symmetric_standard_edge`, find:

```python
    T = jnp.zeros((chi, D2, chi), dtype=A.dtype)
    T_chi = min(chi, D2)
    for i in range(min(T_chi, chi)):
        T = T.at[i, :, i].add(jnp.ones(D2, dtype=A.dtype))
```

Wait — verify this against the actual file before editing. Current file may already use the post-PR-#422 `diag_idx` form. If so, the pattern to replace is the equivalent diag-form loop. **Read the function first** with `Read tool`, then write the precise replacement.

Replacement (variPEPS chi_init=1, δ_{ket=bra} on (0, ·, 0) only):

```python
    diag_idx = np.arange(D, dtype=np.int32) * (D + 1)
    T = jnp.zeros((chi, D2, chi), dtype=A.dtype)
    T = T.at[0, diag_idx, 0].set(jnp.ones(D, dtype=A.dtype))
```

Verify the function's `from_dense` call already uses `tol=float("inf")` (it should, per the post-PR-#422 code). If not, add `tol=float("inf")`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_tensor_init_rank1.py -v --no-cov`

Expected: 12 passes (3 D/chi × 4 init categories).

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_init.py tests/test_ctm_tensor_init_rank1.py
git commit -m "fix(ctm): rank-1 symmetric edge init (bug 3a, part 4)"
```

---

## Task 5: Convergence regression test (the smoking gun)

This is the test that fails on `main` even after PR #422 — and is the actual reason for this whole branch.

**Files:**
- Create: `tests/test_ctm_convergence_random_iPEPS.py`

**Step 1: Write the failing test**

```python
"""Convergence regression for bug 3a (random complex iPEPS, cold init).

Pre-bug-3a-fix this test fails: Tenax converges to the paired-degenerate
fixed point (corner SVs ≈ [0.68, 0.68, 0.20, 0.20]) where variPEPS
converges to the physical [0.95, 0.22, 0.19, 0.13] in ~10 iters.

See `project_ctm_two_init_bugs_found.md` and
`docs/plans/2026-05-11-ctm-bug-3a-design.md`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


@pytest.mark.algorithm
def test_random_complex_ipeps_converges_to_physical_fp():
    """D=2 chi=4 random complex iPEPS converges to non-degenerate physical fp."""
    D, d, chi = 2, 2, 4
    sym = U1Symmetry()
    rng = np.random.RandomState(0)
    raw = rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    raw = raw / np.linalg.norm(raw)
    indices = (
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label="phys"),
    )
    A = DenseTensor(jnp.array(raw), indices)
    env = initialize_ctm_tensor_env(A, chi)

    env_final, info = python_loop_ctm_converge(
        {"A": A},
        {"A": env},
        max_iter=50,
        tol=1e-5,
        projector_method="svd",
    )

    # Convergence reached.
    assert info["converged"], f"CTM did not converge in 50 iters; final_diff={info['final_diff']}"

    # Inspect leading C1 corner SVs.
    C1 = env_final["A"].C1
    sv = jnp.linalg.svd(C1.todense() if hasattr(C1, "todense") else C1._data, compute_uv=False)
    sv = np.asarray(sv)
    sv_sorted = np.sort(sv)[::-1]

    # Physical fixed point: leading SV ~0.95, decaying.  The pre-fix
    # paired-degenerate fp has SVs [0.68, 0.68, 0.20, 0.20].
    assert sv_sorted[0] > 0.85, f"leading SV {sv_sorted[0]:.4f} is too small (paired-degenerate fp?)"
    # Reject paired degeneracy.
    assert abs(sv_sorted[0] - sv_sorted[1]) > 0.05, f"top two SVs nearly degenerate: {sv_sorted[:2]}"
```

**Step 2: Run test to verify it would have failed pre-fix (sanity check)**

Already verified per the diagnostic in `project_ctm_two_init_bugs_found.md`. After Task 1–4 the test should pass; let's run it now.

Run: `uv run pytest tests/test_ctm_convergence_random_iPEPS.py -v --no-cov`

Expected: PASS. If it FAILS, capture `info["final_diff"]` and the actual `sv_sorted`, and check whether bugs 3b (already fixed in PR #422), the `_phase_fix_normalize_tensor` path, or projector rank-aware truncation is interfering. If the fp is still `[0.68, 0.68, 0.20, 0.20]`, one of the four init functions is still the old logic — re-check Task 1–4 changes.

**Step 3: Cross-check vs variPEPS reference (manual, optional)**

If variPEPS is available locally, verify `sv_sorted` matches `[0.948, 0.232, 0.194, 0.131]` (variPEPS chi=4 cold reference per the diagnostic memo) within ~1e-2 absolute. Skip if variPEPS isn't installed in the env.

**Step 4: Commit**

```bash
git add tests/test_ctm_convergence_random_iPEPS.py
git commit -m "test(ctm): random-iPEPS convergence regression for bug 3a"
```

---

## Task 6: Run broader CTM suite, confirm no new regressions

**Files:**
- (No source/test changes; verification only.)

**Step 1: Run the full CTM test suite**

Run: `uv run pytest tests/test_ctm_tensor.py tests/test_ctm_tensor_projector_2x2.py tests/test_ctm_tensor_flow_flip.py tests/test_ctm_tensor_init_rank1.py tests/test_ctm_convergence_random_iPEPS.py --no-cov -q`

Expected:
- New tests from this branch all pass (`test_ctm_tensor_init_rank1` → 12 passes, `test_ctm_convergence_random_iPEPS` → 1 pass).
- Pre-existing failures from `test_ctm_tensor.py` reproduce: `TestSweep::test_one_sweep_dense_finite`, `test_one_sweep_symmetric_finite`, `test_one_sweep_fpeps_finite`, `TestProjectorSymmetric::test_qr_projector_symmetric_matches_eigh`. These predate PR #422 and predate this branch. **Do not attempt to fix them in this PR.**
- No new failures.

If any new failure surfaces, stop and inspect — typical suspects are:
- `_ctm_tensor_c4v` tests if I accidentally touched `_make_dense_corner` (I didn't, but verify `git diff origin/main -- src/tenax/algorithms/_ctm_utils.py` is empty).
- `_split_ctm_tensor*` tests if the shared `_make_dense_corner` was accidentally edited.

**Step 2: Smoke-test forward CTM on a real workload**

Run a quick sanity check that the iPEPS-AD harness still completes one step:

```bash
uv run python -c "
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env

D, d, chi = 2, 2, 8
sym = U1Symmetry()
rng = np.random.RandomState(7)
raw = rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
raw /= np.linalg.norm(raw)
idx = lambda f, l: TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), f, label=l)
phys = TensorIndex.from_charges(sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label='phys')
A = DenseTensor(jnp.array(raw), (idx(FlowDirection.OUT, 'u'), idx(FlowDirection.IN, 'd'), idx(FlowDirection.OUT, 'l'), idx(FlowDirection.IN, 'r'), phys))
env = initialize_ctm_tensor_env(A, chi)
_, info = python_loop_ctm_converge({'A': A}, {'A': env}, max_iter=80, tol=1e-6, projector_method='svd')
print('converged:', info['converged'], 'final_diff:', info['final_diff'], 'iters:', info['iters'])
"
```

Expected: `converged: True final_diff: < 1e-6 iters: < 80`. If this fails, inspect `info['final_diff']` history.

**Step 3: Commit (verification — no code change but worth recording)**

This step has no commit; the previous commits already capture the changes.

---

## Task 7: Push branch and open PR

**Files:**
- (None; PR creation.)

**Step 1: Push**

```bash
git push -u origin fix/ctm-bug-3a-chi-init
```

**Step 2: Open PR**

```bash
gh pr create --title "fix(ctm): rank-1 chi_init for standard CTM (bug 3a)" --body "$(cat <<'EOF'
## Summary

Bug 3a follow-up to #422. Tenax's `initialize_ctm_tensor_env` was packing
the chi-target shape with rank-D padded-identity init, which traps CTM at
a paired-degenerate fixed point on generic complex iPEPS.  variPEPS
reproduces the same trap when forced to start at `chi_init=D`; its default
`chi_init=1` (rank-1 init) breaks the Z₂ from the start.

This PR replaces the four standard-CTM init builders (dense corner, dense
edge, symmetric corner, symmetric edge) with rank-1 init inside the
existing chi-target-shaped container — variPEPS chi_init=1 semantics
without touching the fixed JIT-shape contract.  The rank-aware SVD
truncation (PR #400) prunes the zero modes early sweeps see, so chi grows
organically rank-1 → D → D² → … → chi_target across roughly
⌈log_D(chi_target)⌉ growth sweeps before normal fixed-point convergence.

Sweep loop, projector, absorption, AD path: all unchanged.

## Verification

- New `tests/test_ctm_tensor_init_rank1.py` (12 invariants pass) confirms
  rank-1 init for D=2 chi=4, D=2 chi=16, D=3 chi=9 across both dense and
  symmetric paths.
- New `tests/test_ctm_convergence_random_iPEPS.py` (the smoking gun)
  passes: random complex iPEPS at D=2 chi=4 seed 0 converges in <50 iters
  to leading C1 SV ≈ 0.948 (variPEPS reference), non-degenerate.  Pre-fix
  this test plateaus at `[0.68, 0.68, 0.20, 0.20]`.
- Pre-existing `test_ctm_tensor.py` failures (`test_one_sweep_*_finite`,
  `test_qr_projector_symmetric_matches_eigh`) reproduce on `origin/main`
  without this patch — not regressions.
- No JIT recompile shapes change.

## Out of scope

- Split CTM (`_split_ctm_tensor_init.py`) — different chi-grow handling,
  separate diagnostic.  Shared helper `_make_dense_corner` in `_ctm_utils.py`
  is intentionally untouched.
- C4v reference path (`_ctm_tensor_c4v.py`).

## Test plan
- [x] Init invariants 12/12 pass
- [x] Convergence regression passes
- [x] Pre-existing failures unchanged
- [ ] CI `Tests (Python 3.11)` / `Tests (Python 3.12)` / `Tests (macOS, Python 3.12)`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 3: Enable auto-merge**

```bash
gh pr merge $(gh pr view --json number --jq .number) --squash --delete-branch --auto
```

**Step 4: Done**

Memory update: edit `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/project_ctm_two_init_bugs_found.md` and add a status note that bug 3a was shipped in PR #<num>. Add a new memory entry summarising the fix if the diagnostic file becomes too long.

---

## Quick reference

- **Pre-fix symptom**: random complex D=2 chi=4 iPEPS → corner SVs `[0.68, 0.68, 0.20, 0.20]` (paired-degenerate).
- **Post-fix expectation**: SVs `[0.948, 0.232, 0.194, 0.131]` matching variPEPS chi_init=1 cold reference.
- **Files touched**: `src/tenax/algorithms/_ctm_tensor_init.py` only.  All 4 init functions modified; one new private helper added.
- **Files NOT touched**: `_ctm_utils.py`, any `_ctm_tensor_*` sweep/projector/absorption file, AD path, split CTM, C4v.
- **Net commits**: 5 (one per task 1–4, one per task 5). Task 6 is verification, task 7 is PR.
