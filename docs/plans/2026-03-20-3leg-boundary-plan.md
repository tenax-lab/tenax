# 3-Leg Boundary Tensor Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make all MPS boundary tensors uniformly 3-leg, eliminating ~23 `ndim==2` special cases across 5 files.

**Architecture:** Change tensor construction (mps.py, dmrg.py builders) to produce `(1, d, chi)` / `(chi, d, 1)` boundaries, then delete all padding/unpadding code that currently promotes 2-leg → 3-leg at point of use. Add `target_charge` field to `FiniteMPS`.

**Tech Stack:** JAX, Tenax (core tensors: DenseTensor, SymmetricTensor, TensorIndex)

**Branch:** Work in existing worktree at `.claude/worktrees/tdvp/`

---

### Task 1: Add `target_charge` field to FiniteMPS

**Files:**
- Modify: `src/tenax/core/mps.py:44-47` (dataclass fields)
- Modify: `src/tenax/core/mps.py:55-84` (`from_tensors`)
- Modify: `src/tenax/core/mps.py:87-109` (`random`)
- Test: `tests/test_mps.py`

**Step 1: Write the failing test**

In `tests/test_mps.py`, add:

```python
def test_finite_mps_target_charge_field():
    """FiniteMPS stores target_charge."""
    key = jax.random.PRNGKey(0)
    mps = FiniteMPS.random(L=6, d=2, chi=4, key=key)
    assert mps.target_charge is None  # dense MPS has no charge

    mps2 = FiniteMPS.from_tensors(mps.tensors, target_charge=0)
    assert mps2.target_charge == 0
```

**Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_mps.py::test_finite_mps_target_charge_field -v`
Expected: FAIL — `target_charge` not a recognized field

**Step 3: Add `target_charge` field**

In `src/tenax/core/mps.py`, add field to the dataclass (after `log_norm`):

```python
target_charge: int | None = None
```

Update `from_tensors()` to accept and pass through `target_charge`:

```python
@staticmethod
def from_tensors(
    tensors: list[Tensor],
    orth_center: int | None = None,
    singular_values: list[jnp.ndarray | None] | None = None,
    log_norm: float = 0.0,
    target_charge: int | None = None,
    verify: bool = False,
) -> FiniteMPS:
```

And in its return statement, pass `target_charge=target_charge`.

Update `random()` to pass `target_charge` through to the returned instance.

Ensure `canonicalize()`, `compute_singular_values()`, and any other method that returns a new `FiniteMPS` propagates `target_charge`.

**Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_mps.py::test_finite_mps_target_charge_field -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/core/mps.py tests/test_mps.py
git commit -m "feat(mps): add target_charge field to FiniteMPS"
```

---

### Task 2: Make `_build_random_dense_tensors()` produce 3-leg boundaries

**Files:**
- Modify: `src/tenax/core/mps.py:607-651` (`_build_random_dense_tensors`)
- Test: `tests/test_mps.py`

**Step 1: Write the failing test**

```python
def test_dense_mps_3leg_boundaries():
    """All sites including boundaries are 3-leg."""
    key = jax.random.PRNGKey(0)
    mps = FiniteMPS.random(L=6, d=2, chi=4, key=key)
    for i, t in enumerate(mps.tensors):
        assert t.ndim == 3, f"Site {i} has ndim={t.ndim}, expected 3"
    # Left boundary: (1, d, chi)
    assert mps.tensors[0].shape[0] == 1
    # Right boundary: (chi, d, 1)
    assert mps.tensors[-1].shape[2] == 1
    # Labels follow v{i-1}_{i} pattern at boundaries
    assert mps.tensors[0].labels()[0] == "v_-1_0"
    assert mps.tensors[-1].labels()[2] == "v5_6"
```

**Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_mps.py::test_dense_mps_3leg_boundaries -v`
Expected: FAIL — boundary sites are 2-leg

**Step 3: Update `_build_random_dense_tensors`**

Currently (lines 607-651) this builds:
- Site 0: `(d, chi)` with labels `(p0, v0_1)`
- Site L-1: `(chi, d)` with labels `(v{L-2}_{L-1}, p{L-1})`

Change to:
- Site 0: `(1, d, chi)` with labels `(v_-1_0, p0, v0_1)`
- Site L-1: `(chi, d, 1)` with labels `(v{L-2}_{L-1}, p{L-1}, v{L-1}_{L})`

The single-site special case (L==1) becomes `(1, d, 1)` with labels `(v_-1_0, p0, v0_1)`.

**Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_mps.py::test_dense_mps_3leg_boundaries -v`
Expected: PASS

**Step 5: Run full test suite to see what breaks**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_mps.py -v`
Expected: Some existing tests may fail if they assert 2-leg shapes. Fix those assertions.

**Step 6: Commit**

```bash
git add src/tenax/core/mps.py tests/test_mps.py
git commit -m "feat(mps): 3-leg boundary tensors for dense MPS"
```

---

### Task 3: Make `_build_random_symmetric_tensors()` produce 3-leg boundaries

**Files:**
- Modify: `src/tenax/core/mps.py:654-723` (`_build_random_symmetric_tensors`)
- Test: `tests/test_mps.py`

**Step 1: Write the failing test**

```python
def test_symmetric_mps_3leg_boundaries():
    """Symmetric MPS boundaries are 3-leg with trivial charge bond."""
    from tenax.core.symmetry import U1Symmetry
    key = jax.random.PRNGKey(0)
    mps = FiniteMPS.random(L=6, d=2, chi=8, key=key,
                           symmetric=True, symmetry=U1Symmetry(),
                           target_charge=0)
    for i, t in enumerate(mps.tensors):
        assert t.ndim == 3, f"Site {i} has ndim={t.ndim}, expected 3"
    # Left boundary: trivial dim-1 bond with charge 0
    assert mps.tensors[0].indices[0].dim == 1
    assert mps.tensors[0].indices[0].label == "v_-1_0"
    # Right boundary: trivial dim-1 bond with target charge
    assert mps.tensors[-1].indices[2].dim == 1
    assert mps.tensors[-1].indices[2].label == "v5_6"
    assert mps.target_charge == 0
```

**Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_mps.py::test_symmetric_mps_3leg_boundaries -v`
Expected: FAIL — symmetric boundaries are 2-leg

**Step 3: Update `_build_random_symmetric_tensors`**

Currently (lines 654-723):
- Site 0: 2D `(p, v)` with only physical + right virtual indices
- Site L-1: 2D `(v, p)` with left virtual + physical indices, `site_target = target_charge`

Change to:
- Site 0: 3D `(v_trivial, p, v)` — prepend a trivial `TensorIndex` with charge 0, `FlowDirection.IN`, label `v_-1_0`
- Site L-1: 3D `(v, p, v_trivial)` — append a trivial `TensorIndex` with charge = `target_charge`, `FlowDirection.OUT`, label `v{L-1}_{L}`

The trivial index has a single charge sector of dimension 1.

**Step 4: Run tests**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_mps.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/core/mps.py tests/test_mps.py
git commit -m "feat(mps): 3-leg boundary tensors for symmetric MPS"
```

---

### Task 4: Update legacy builders in dmrg.py

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py:1642-1706` (`build_random_mps`)
- Modify: `src/tenax/algorithms/dmrg.py:1476-1584` (`build_random_symmetric_mps`)
- Test: `tests/test_dmrg.py`

**Step 1: Update `build_random_mps()` to produce 3-leg boundaries**

Mirror the same changes from Task 2: site 0 becomes `(1, d, chi)`, site L-1 becomes `(chi, d, 1)`, single-site becomes `(1, d, 1)`. Use labels `v_-1_0` and `v{L-1}_{L}`.

**Step 2: Update `build_random_symmetric_mps()` to produce 3-leg boundaries**

Mirror the same changes from Task 3.

**Step 3: Run DMRG tests**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_dmrg.py -v`
Expected: Likely many failures — the DMRG code still has padding logic that assumes 2-leg. These failures are expected and will be fixed in Tasks 5-6.

**Step 4: Commit the builder changes**

```bash
git add src/tenax/algorithms/dmrg.py
git commit -m "feat(dmrg): 3-leg boundary tensors in legacy MPS builders"
```

---

### Task 5: Remove boundary padding from DMRG dense path

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py`
  - Lines 507-514: `_update_left_env` — remove `ndim == 2` padding
  - Lines 553-560: `_update_right_env` — remove `ndim == 2` padding
  - Lines 670-687: `_two_site_update` — remove `ndim == 3` theta padding
  - Lines 710-722: `_one_site_update` — remove `ndim == 2` site padding
  - Lines 750-761: `_one_site_update` — remove unpadding after solve
- Test: `tests/test_dmrg.py`

**Step 1: Remove padding in `_update_left_env`**

Delete the `if A_dense.ndim == 2:` block (lines 507-514). The tensor is now always 3D.

**Step 2: Remove padding in `_update_right_env`**

Delete the `if B_dense.ndim == 2:` block (lines 553-560).

**Step 3: Remove padding/unpadding in `_two_site_update`**

Delete the `if theta_dense.ndim == 3:` padding block (lines 670-687) and the corresponding unpadding (lines 710-722 area). Theta is now always 4D.

**Step 4: Remove padding/unpadding in `_one_site_update`**

Delete the `ndim == 2` / `ndim == 1` padding block and the corresponding unpadding. Site is now always 3D.

**Step 5: Run DMRG dense tests**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_dmrg.py -k "not symmetric" -v`
Expected: PASS — dense DMRG should work with uniform 3-leg tensors

**Step 6: Commit**

```bash
git add src/tenax/algorithms/dmrg.py
git commit -m "refactor(dmrg): remove boundary padding from dense path"
```

---

### Task 6: Remove boundary padding from DMRG symmetric path

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py`
  - Lines 985-1011: Delete `_pad_boundary_symmetric()`
  - Lines 1014-1034: Delete `_unpad_boundary_symmetric()`
  - Lines 1239-1244: `_update_left_env_symmetric` — remove padding
  - Lines 1277-1281: `_update_right_env_symmetric` — remove padding
  - Lines 1326-1338: `_two_site_update_symmetric` — remove padding/unpadding
  - Lines 1381-1390: `_one_site_update_symmetric` — remove padding/unpadding
- Test: `tests/test_dmrg.py`

**Step 1: Delete `_pad_boundary_symmetric` and `_unpad_boundary_symmetric`**

Remove both functions entirely.

**Step 2: Remove all calls to them**

In `_update_left_env_symmetric`, `_update_right_env_symmetric`, `_two_site_update_symmetric`, `_one_site_update_symmetric` — remove the `if ndim == 2` / `if ndim == 3` blocks and their corresponding unpad calls. The tensors are now always 3D/4D.

**Step 3: Run symmetric DMRG tests**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_dmrg.py -k "symmetric" -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/tenax/algorithms/dmrg.py
git commit -m "refactor(dmrg): delete _pad/_unpad_boundary_symmetric and all symmetric padding"
```

---

### Task 7: Update `compute_mps_sector` and `validate_mps_sector`

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py:1587-1639`
- Test: `tests/test_dmrg.py`

**Step 1: Simplify `compute_mps_sector`**

Currently reads charge from the right boundary's blocks assuming 2-leg tensor. With 3-leg, the right boundary's last index (axis 2) carries the target charge. Update to read from axis 2 of the 3-leg tensor.

If we have `FiniteMPS`, prefer reading `mps.target_charge` directly.

**Step 2: Run tests**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_dmrg.py -k "sector" -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/tenax/algorithms/dmrg.py
git commit -m "refactor(dmrg): simplify compute_mps_sector for 3-leg boundaries"
```

---

### Task 8: Simplify TDVP boundary handling

**Files:**
- Modify: `src/tenax/algorithms/tdvp.py`
  - Lines 135-153: Delete or simplify `_site_to_3d` — just return `site.todense()` (always 3D now)
  - Lines 156-216: Simplify `_make_site_tensor` — always construct 3-leg DenseTensor
  - Lines 302-343: Simplify `_identity_mpo_site` — `d` is always `shape[1]`
- Test: `tests/test_tdvp.py`

**Step 1: Simplify `_site_to_3d`**

Replace with:
```python
def _site_to_3d(site: Tensor) -> jax.Array:
    return site.todense()  # always 3D now
```

Update all callers to remove the `is_left, is_right` return values.

**Step 2: Simplify `_make_site_tensor`**

Remove the left/right boundary branches. Always construct a 3-leg DenseTensor with labels `(v{i-1}_{i}, p{i}, v{i}_{i+1})`. Use `v_-1_0` for site 0 left bond, `v{L-1}_{L}` for site L-1 right bond.

**Step 3: Simplify `_identity_mpo_site`**

Physical dimension is always `shape[1]` for a 3-leg MPS tensor. Remove the `ndim == 2` / `ndim == 1` branches.

**Step 4: Run TDVP tests**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_tdvp.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/tdvp.py
git commit -m "refactor(tdvp): remove boundary special cases for 3-leg MPS"
```

---

### Task 9: Simplify observables boundary handling

**Files:**
- Modify: `src/tenax/algorithms/observables.py:112-200` (`_contract_sandwich`)
- Test: `tests/test_observables.py`

**Step 1: Remove boundary branch in `_contract_sandwich`**

Delete the `if A.ndim == 2:` block. All sites are 3D `(a, p, r)`, so use the middle-site einsum path uniformly:

```python
if tm is None:
    contracted = jnp.einsum("apr,aps->rs", A, A_conj)
else:
    contracted = jnp.einsum("ab,apr,bps->rs", tm, A, A_conj)
```

At the final site (right boundary), the result will be `(1, 1)` which collapses to a scalar naturally.

**Step 2: Run observables tests**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_observables.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/tenax/algorithms/observables.py
git commit -m "refactor(observables): uniform 3-leg contraction in _contract_sandwich"
```

---

### Task 10: Simplify CBE boundary handling

**Files:**
- Modify: `src/tenax/algorithms/cbe.py:198-211`
- Test: `tests/test_cbe.py`

**Step 1: Remove `ndim == 2` padding in `expand_bond_symmetric`**

Delete the boundary padding blocks for `site_data` and `right_tensor`. They are already 3D.

**Step 2: Run CBE tests**

Run: `cd .claude/worktrees/tdvp && uv run pytest tests/test_cbe.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/tenax/algorithms/cbe.py
git commit -m "refactor(cbe): remove boundary padding for 3-leg MPS"
```

---

### Task 11: Update FiniteMPS docstring and exports

**Files:**
- Modify: `src/tenax/core/mps.py:22-42` (docstring)
- Modify: `src/tenax/__init__.py`

**Step 1: Update FiniteMPS docstring**

Change line 28 from:
> Boundary sites are 2-leg (site 0: physical x right-bond, site L-1: left-bond x physical)

To:
> All sites are 3-leg (left-bond x physical x right-bond). Boundary sites have a trivial dimension-1 bond: site 0 has shape (1, d, chi), site L-1 has shape (chi, d, 1).

Add `target_charge` to the attributes docstring.

**Step 2: Verify exports**

Check that `FiniteMPS` is exported from `__init__.py`. No new exports needed.

**Step 3: Commit**

```bash
git add src/tenax/core/mps.py src/tenax/__init__.py
git commit -m "docs(mps): update FiniteMPS docstring for 3-leg boundaries"
```

---

### Task 12: Final integration test pass

**Step 1: Run all core tests**

Run: `cd .claude/worktrees/tdvp && uv run pytest -m core -v`
Expected: ALL PASS

**Step 2: Run full test suite (excluding slow)**

Run: `cd .claude/worktrees/tdvp && uv run pytest -m "not slow" -v`
Expected: ALL PASS

**Step 3: Commit any remaining fixes**

```bash
git add -u
git commit -m "fix: address remaining 3-leg boundary test failures"
```
