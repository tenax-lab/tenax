# Algorithm MPS Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update dmrg(), tdvp(), idmrg() to accept/return FiniteMPS/InfiniteMPS, replace all todense() calls in the 1-site DMRG sweep code with block-sparse tenax.linalg operations.

**Architecture:** Interface changes at algorithm boundaries (accept FiniteMPS, return FiniteMPS), plus rewrite of 1-site DMRG sweep helpers to use tenax.linalg.qr instead of todense()-based QR. The 2-site path, environment functions, and Lanczos solver internals are left unchanged — their todense() calls only affect DenseTensor (where it's a no-op).

**Tech Stack:** tenax.linalg (qr, svd), tenax.contraction.contractor (contract), tenax.core.mps (FiniteMPS, InfiniteMPS)

---

### Task 1: dmrg() interface — accept FiniteMPS, return FiniteMPS

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py`
- Modify: `tests/test_dmrg.py`

**What to change:**

1. `DMRGResult.mps` field: change type from `TensorNetwork` to `FiniteMPS`
2. `dmrg()` signature: change `initial_mps` from `TensorNetwork` to `FiniteMPS`
3. At function entry: extract `mps_tensors = list(initial_mps.tensors)` (replacing `[initial_mps.get_tensor(i) for ...]`)
4. At function exit: wrap result in `FiniteMPS.from_tensors(mps_tensors, orth_center=...)` instead of building TensorNetwork
5. Replace the defensive input canonicalization (lines 175-213) with `initial_mps = initial_mps.right_canonicalize()` then extract tensors

**Keep backward compat:** Also accept TensorNetwork by checking type and converting:
```python
if isinstance(initial_mps, TensorNetwork):
    mps_tensors = [initial_mps.get_tensor(i) for i in range(L)]
    initial_mps = FiniteMPS.from_tensors(mps_tensors)
```

**Tests:** Update existing dmrg tests to pass FiniteMPS. Verify DMRGResult.mps is FiniteMPS with orth_center set.

---

### Task 2: Replace 1-site DMRG sweep helpers with tenax.linalg.qr

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py`

**What to change:**

The current helpers `_qr_left_canonical()`, `_rq_right_canonical()`, `_absorb_r_into_next()`, `_absorb_l_into_prev()` all call `.todense()`. Replace the 1-site sweep code that uses them.

**In L→R sweep** (currently around line 297-308):
```python
# OLD: q_site, r_mat = _qr_left_canonical(new_site) ...
# NEW: use tenax.linalg.qr + contract (same pattern as FiniteMPS.canonicalize)
right_bond = f"v{i}_{i+1}"
left_labels = [lb for lb in new_site.labels() if lb != right_bond]
tmp_bond = f"_qr_{right_bond}"
Q, R = qr(new_site, left_labels, [right_bond], new_bond_label=tmp_bond)
mps_tensors[i] = Q.relabel(tmp_bond, right_bond)
absorbed = contract(R, mps_tensors[i + 1])
mps_tensors[i + 1] = absorbed.relabel(tmp_bond, right_bond)
```

**In R→L sweep** (currently around line 359-365):
```python
# Same pattern but reversed — see FiniteMPS.canonicalize() R→L sweep
left_bond = f"v{i-1}_{i}"
other_labels = [lb for lb in new_site.labels() if lb != left_bond]
tmp_bond = f"_qr_{left_bond}"
Q, R = qr(new_site, other_labels, [left_bond], new_bond_label=tmp_bond)
Q = Q.relabel(tmp_bond, left_bond)
# Reorder so left_bond is first (MPS convention)
labels = Q.labels()
bond_pos = labels.index(left_bond)
if bond_pos != 0:
    axes = (bond_pos,) + tuple(j for j in range(len(labels)) if j != bond_pos)
    Q = Q.transpose(axes)
mps_tensors[i] = Q
absorbed = contract(mps_tensors[i - 1], R)
mps_tensors[i - 1] = absorbed.relabel(tmp_bond, left_bond)
```

**Normalization between sweeps** (lines 252-257, 312-316): Replace todense()-based normalization with norm computation via `inner()` or from the center site Frobenius norm.

**Tests:** Run existing DMRG tests (both 1-site and 2-site, dense and symmetric) to verify no regressions.

---

### Task 3: Remove dead helper functions

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py`

After Task 2, the following functions are no longer called:
- `_qr_left_canonical()`
- `_rq_right_canonical()`
- `_absorb_r_into_next()`
- `_absorb_l_into_prev()`
- `_is_left_boundary()`
- `_wrap_tensor()`

Remove them. Run tests to confirm nothing breaks.

---

### Task 4: tdvp() interface — accept/return FiniteMPS

**Files:**
- Modify: `src/tenax/algorithms/tdvp.py`
- Modify: `tests/test_tdvp.py`

**What to change:**

1. `TDVPResult.mps`: change type to `FiniteMPS`
2. `tdvp()` and `tdvp_step()`: accept `FiniteMPS` (with TensorNetwork backward compat)
3. Internally, TDVP already converts to 3D arrays for sweeps. At entry: extract from FiniteMPS. At exit: wrap back into FiniteMPS.
4. Replace `_right_canonicalize_dense()` call with `mps.right_canonicalize()` then extract 3D arrays.

**Tests:** Run existing TDVP tests to verify no regressions.

---

### Task 5: idmrg() — return InfiniteMPS in result

**Files:**
- Modify: `src/tenax/algorithms/idmrg.py`
- Modify: `tests/test_idmrg.py`

**What to change:**

1. `iDMRGResult.mps_tensors` field: rename to `mps` with type `InfiniteMPS`
2. At both return sites (dense path ~line 818, symmetric path ~line 640): wrap [A_L, A_R] and singular_values into `InfiniteMPS.from_tensors()`
3. Remove the separate `singular_values` field from iDMRGResult since InfiniteMPS carries them

**Tests:** Update iDMRG tests to check result.mps is InfiniteMPS.

---

### Task 6: Update exports and run full regression

**Files:**
- Modify: `src/tenax/__init__.py` — ensure DMRGResult, TDVPResult, iDMRGResult re-exports are consistent
- Run: `uv run pytest -m core` and `uv run pytest -m "not slow"`

---
