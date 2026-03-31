# Cython DMRG Pipeline Optimization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cythonize the three remaining bottlenecks in the numpy blockwise DMRG path — eliminating opt_einsum plan building (3.6s), moving the matvec combo loop into pure C (9.2s), and fusing the Lanczos reorthogonalization (4.5s) — targeting a 2-3x total speedup for L=30, chi=128 (21s -> 7-10s).

**Architecture:** Three independent Cython optimizations layered on the existing `_cython_blas.pyx`. (1) Hard-coded DMRG contraction plans replace `opt_einsum.contract_path` for the two fixed einsum patterns. (2) A pre-transpose + raw-BLAS combo kernel replaces the current `cython_matvec_combos` that still calls numpy per step. (3) A fused Lanczos reorthogonalization function batches `ba_inner` + `ba_sub_scaled` into one C loop over blocks.

**Tech Stack:** Cython 3, scipy.linalg.cython_blas (dgemm/daxpy/ddot), numpy C API

**Profiling baseline** (L=30, chi=128, 4 sweeps, Cython compiled):

| Component | Time | % |
|-----------|------|---|
| Matvec combo loop (`_execute_matvec_combos`) | 9.2s | 44% |
| opt_einsum path finding (`build_blas_plan`) | 3.6s | 17% |
| `ba_inner` (Lanczos reorth) | 2.6s | 12% |
| `numpy.transpose` (inside matvec) | 2.2s | 10% |
| `_lanczos_solve_np` loop overhead | 1.9s | 9% |
| Other (env updates, SVD, precompute) | 1.5s | 8% |
| **Total** | **21.0s** | |

---

## File Structure

| File | Role | Change |
|------|------|--------|
| `src/tenax/contraction/_dmrg_plans.py` | Create | Hard-coded BLAS plans for DMRG einsum patterns |
| `src/tenax/contraction/_cython_blas.pyx` | Modify | New `cython_matvec_pretransposed`, `cython_lanczos_reorth` |
| `src/tenax/algorithms/dmrg.py` | Modify | Use new plans, pre-transpose env blocks, fused Lanczos |
| `tests/test_dmrg_cython.py` | Create | Unit tests for each Cython optimization |

---

## Task 1: Hard-coded DMRG contraction plans

Replace `opt_einsum.contract_path` + `build_blas_plan` for the two fixed DMRG einsum patterns with direct TTGT parameter computation. Eliminates 3.6s (17%) of overhead from 7159 calls to `opt_einsum.contract_path`.

**Files:**
- Create: `src/tenax/contraction/_dmrg_plans.py`
- Modify: `src/tenax/algorithms/dmrg.py:1960-2010` (`_precompute_matvec_combos`)
- Test: `tests/test_dmrg_cython.py`

### Background

The 2-site matvec pattern `"abc,apqd,bpse,eqtf,dfg->cstg"` has a natural left-to-right contraction order:
1. L(abc) @ theta(apqd) -> I1(bcpqd) [contract `a`]
2. I1(bcpqd) @ W1(bpse) -> I2(cqdse) [contract `b,p`]
3. I2(cqdse) @ W2(eqtf) -> I3(cdstf) [contract `e,q`]
4. I3(cdstf) @ R(dfg) -> out(cstg) [contract `d,f`]

The 1-site matvec pattern `"abc,apd,bpxe,def->cxf"`:
1. L(abc) @ site(apd) -> I1(bcpd) [contract `a`]
2. I1(bcpd) @ W(bpxe) -> I2(cdxe) [contract `b,p`]
3. I2(cdxe) @ R(def) -> out(cxf) [contract `d,e`]

For each step, given the block dimensions, we compute (M, N, K, left_perm, right_perm, out_shape) directly — no opt_einsum needed.

- [ ] **Step 1: Write test for hard-coded 2-site plan**

```python
# tests/test_dmrg_cython.py
import numpy as np
import pytest
from tenax.contraction._blas_plan import get_cached_blas_plan


class TestDmrgPlans:
    """Test hard-coded DMRG BLAS plans match opt_einsum plans."""

    def test_two_site_plan_matches_opt_einsum(self):
        """Hard-coded 2-site plan produces same result as opt_einsum plan."""
        from tenax.contraction._dmrg_plans import build_two_site_plan

        subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        # Typical DMRG shapes: L(chi,W,chi), theta(chi,d,d,chi),
        # W1(W,d,d,W), W2(W,d,d,W), R(chi,W,chi)
        shapes = ((16, 3, 16), (16, 2, 2, 21), (3, 2, 2, 3), (3, 2, 2, 3), (21, 3, 21))

        plan_new = build_two_site_plan(shapes)
        plan_ref = get_cached_blas_plan(subs, shapes)

        # Execute both on random data and compare results
        rng = np.random.default_rng(42)
        arrays = [rng.standard_normal(s) for s in shapes]

        result_new = plan_new.execute_numpy(arrays)
        result_ref = plan_ref.execute_numpy(arrays)
        np.testing.assert_allclose(result_new, result_ref, atol=1e-12)

    def test_one_site_plan_matches_opt_einsum(self):
        """Hard-coded 1-site plan produces same result as opt_einsum plan."""
        from tenax.contraction._dmrg_plans import build_one_site_plan

        subs = "abc,apd,bpxe,def->cxf"
        shapes = ((16, 3, 16), (16, 2, 21), (3, 2, 2, 3), (21, 3, 21))

        plan_new = build_one_site_plan(shapes)
        plan_ref = get_cached_blas_plan(subs, shapes)

        rng = np.random.default_rng(42)
        arrays = [rng.standard_normal(s) for s in shapes]

        result_new = plan_new.execute_numpy(arrays)
        result_ref = plan_ref.execute_numpy(arrays)
        np.testing.assert_allclose(result_new, result_ref, atol=1e-12)

    @pytest.mark.parametrize("chi_l,chi_r,W,d", [
        (4, 4, 3, 2), (16, 21, 5, 2), (32, 32, 3, 3), (1, 8, 3, 2),
    ])
    def test_two_site_plan_various_shapes(self, chi_l, chi_r, W, d):
        """Hard-coded 2-site plan works across typical DMRG block shapes."""
        from tenax.contraction._dmrg_plans import build_two_site_plan

        shapes = ((chi_l, W, chi_l), (chi_l, d, d, chi_r),
                  (W, d, d, W), (W, d, d, W), (chi_r, W, chi_r))
        plan = build_two_site_plan(shapes)

        rng = np.random.default_rng(123)
        arrays = [rng.standard_normal(s) for s in shapes]
        result = plan.execute_numpy(arrays)

        # Verify via np.einsum
        expected = np.einsum("abc,apqd,bpse,eqtf,dfg->cstg", *arrays)
        np.testing.assert_allclose(result, expected, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_dmrg_cython.py::TestDmrgPlans::test_two_site_plan_matches_opt_einsum -xvs`
Expected: `ModuleNotFoundError: No module named 'tenax.contraction._dmrg_plans'`

- [ ] **Step 3: Implement hard-coded DMRG plans**

```python
# src/tenax/contraction/_dmrg_plans.py
"""Hard-coded BLAS execution plans for DMRG einsum patterns.

Replaces opt_einsum.contract_path for the two fixed DMRG matvec
subscripts, eliminating ~3.6s of path-finding overhead per DMRG run.
"""

from __future__ import annotations

import functools

from tenax.contraction._blas_plan import BlasExecPlan, GemmStep


def _identity_perm(ndim: int) -> tuple[int, ...]:
    return tuple(range(ndim))


def _make_step(
    left_idx: int,
    right_idx: int,
    out_idx: int,
    left_sub: str,
    right_sub: str,
    result_sub: str,
    char_to_dim: dict[str, int],
) -> GemmStep:
    """Build a GemmStep from symbolic subscripts and dimension map."""
    contracted = set(left_sub) & set(right_sub)
    free_l = [c for c in left_sub if c not in contracted]
    free_r = [c for c in right_sub if c not in contracted]
    contracted_ordered = [c for c in left_sub if c in contracted]

    m = 1
    for c in free_l:
        m *= char_to_dim[c]
    n = 1
    for c in free_r:
        n *= char_to_dim[c]
    k = 1
    for c in contracted_ordered:
        k *= char_to_dim[c]

    target_left = free_l + contracted_ordered
    left_perm = tuple(left_sub.index(c) for c in target_left)
    if left_perm == _identity_perm(len(left_sub)):
        left_perm = ()

    target_right = contracted_ordered + free_r
    right_perm = tuple(right_sub.index(c) for c in target_right)
    if right_perm == _identity_perm(len(right_sub)):
        right_perm = ()

    out_chars = free_l + free_r
    out_shape = tuple(char_to_dim[c] for c in out_chars)

    return GemmStep(
        left_idx=left_idx,
        right_idx=right_idx,
        out_idx=out_idx,
        trans_a=False,
        trans_b=False,
        m=m,
        n=n,
        k=k,
        left_perm=left_perm,
        right_perm=right_perm,
        out_shape=out_shape,
    ), "".join(out_chars)


def build_two_site_plan(
    shapes: tuple[tuple[int, ...], ...],
) -> BlasExecPlan:
    """Build BLAS plan for 2-site DMRG matvec: abc,apqd,bpse,eqtf,dfg->cstg.

    Fixed contraction order (left-to-right along MPS chain):
      Step 0: L(abc) @ theta(apqd)  -> I0 [contract a]
      Step 1: I0 @ W1(bpse)         -> I1 [contract b,p]
      Step 2: I1 @ W2(eqtf)         -> I2 [contract e,q]
      Step 3: I2 @ R(dfg)           -> out [contract d,f]
    """
    s_L, s_theta, s_W1, s_W2, s_R = shapes
    char_to_dim = {}
    for chars, shape in zip(["abc", "apqd", "bpse", "eqtf", "dfg"], shapes):
        for c, d in zip(chars, shape):
            char_to_dim[c] = d

    # buf 0=L, 1=theta, 2=W1, 3=W2, 4=R, 5=I0, 6=I1, 7=I2
    step0, sub0 = _make_step(0, 1, 5, "abc", "apqd", "", char_to_dim)
    step1, sub1 = _make_step(5, 2, 6, sub0, "bpse", "", char_to_dim)
    step2, sub2 = _make_step(6, 3, 7, sub1, "eqtf", "", char_to_dim)
    step3, sub3 = _make_step(7, 4, 8, sub2, "dfg", "", char_to_dim)

    output_sub = sub3
    if output_sub == "cstg":
        output_perm = ()
    else:
        output_perm = tuple(output_sub.index(c) for c in "cstg")

    return BlasExecPlan(
        steps=(step0, step1, step2, step3),
        n_buffers=9,
        n_inputs=5,
        output_perm=output_perm,
    )


def build_one_site_plan(
    shapes: tuple[tuple[int, ...], ...],
) -> BlasExecPlan:
    """Build BLAS plan for 1-site DMRG matvec: abc,apd,bpxe,def->cxf.

    Fixed contraction order:
      Step 0: L(abc) @ site(apd) -> I0 [contract a]
      Step 1: I0 @ W(bpxe)      -> I1 [contract b,p]
      Step 2: I1 @ R(def)       -> out [contract d,e]
    """
    char_to_dim = {}
    for chars, shape in zip(["abc", "apd", "bpxe", "def"], shapes):
        for c, d in zip(chars, shape):
            char_to_dim[c] = d

    # buf 0=L, 1=site, 2=W, 3=R, 4=I0, 5=I1
    step0, sub0 = _make_step(0, 1, 4, "abc", "apd", "", char_to_dim)
    step1, sub1 = _make_step(4, 2, 5, sub0, "bpxe", "", char_to_dim)
    step2, sub2 = _make_step(5, 3, 6, sub1, "def", "", char_to_dim)

    output_sub = sub2
    if output_sub == "cxf":
        output_perm = ()
    else:
        output_perm = tuple(output_sub.index(c) for c in "cxf")

    return BlasExecPlan(
        steps=(step0, step1, step2),
        n_buffers=7,
        n_inputs=4,
        output_perm=output_perm,
    )


@functools.lru_cache(maxsize=4096)
def get_dmrg_plan(
    subscripts: str,
    shapes: tuple[tuple[int, ...], ...],
) -> BlasExecPlan:
    """Cached DMRG-specific plan, falling back to opt_einsum for unknown patterns."""
    if subscripts == "abc,apqd,bpse,eqtf,dfg->cstg":
        return build_two_site_plan(shapes)
    elif subscripts == "abc,apd,bpxe,def->cxf":
        return build_one_site_plan(shapes)
    else:
        from tenax.contraction._blas_plan import get_cached_blas_plan
        return get_cached_blas_plan(subscripts, shapes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_dmrg_cython.py::TestDmrgPlans -xvs`
Expected: All 6 tests PASS

- [ ] **Step 5: Wire into `_precompute_matvec_combos`**

In `src/tenax/algorithms/dmrg.py`, replace the `get_cached_blas_plan` import and call inside `_precompute_matvec_combos` (around line 1978):

Replace:
```python
from tenax.contraction._blas_plan import get_cached_blas_plan
```
with:
```python
from tenax.contraction._dmrg_plans import get_dmrg_plan
```

And in the loop body, replace:
```python
expr_cache[block_shapes] = get_cached_blas_plan(subscripts, block_shapes)
```
with:
```python
plan_cache[block_shapes] = ... # use get_dmrg_plan(subscripts, block_shapes)
```

Specifically, change the plan lookup inside `_precompute_matvec_combos`:
```python
if block_shapes not in plan_cache:
    plan = get_dmrg_plan(subscripts, block_shapes)
    sp = [
        (s.left_idx, s.right_idx, s.out_idx,
         s.m, s.n, s.k,
         s.left_perm, s.right_perm, s.out_shape)
        for s in plan.steps
    ]
    plan_cache[block_shapes] = (sp, plan.n_inputs, plan.n_buffers, plan.output_perm)
```

- [ ] **Step 6: Run full DMRG test suite**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_dmrg.py -x --no-header -q`
Expected: 52 passed

- [ ] **Step 7: Commit**

```bash
git add src/tenax/contraction/_dmrg_plans.py tests/test_dmrg_cython.py src/tenax/algorithms/dmrg.py
git commit -m "perf: hard-coded DMRG BLAS plans, eliminate opt_einsum overhead"
```

---

## Task 2: Pre-transpose env blocks + pure-C matvec combo loop

Replace the current `cython_matvec_combos` (which still calls `np.transpose`, `np.ascontiguousarray`, `np.empty` per GEMM step per combo) with a two-phase approach: (1) pre-transpose all blocks to GEMM-ready 2D C-contiguous arrays at setup time, (2) a tight C loop that only calls raw `dgemm` and `daxpy` on pre-prepared pointers. Targets 9.2s + 2.2s = 11.4s of overhead.

**Files:**
- Modify: `src/tenax/contraction/_cython_blas.pyx` (new `cython_matvec_pretransposed`)
- Modify: `src/tenax/algorithms/dmrg.py` (`_precompute_matvec_combos`, `_execute_matvec_combos`)
- Test: `tests/test_dmrg_cython.py`

### Design

**Phase 1 (Python, once per site update):** For each combo, pre-transpose and reshape ALL input blocks (both env and theta) to their GEMM-ready 2D layout. For env blocks, this is done once and cached. For theta blocks, this is done once per matvec call (outside the combo loop).

Store per-combo: a flat list of pre-transposed 2D numpy arrays (one per GEMM operand usage, not per input tensor), plus GEMM parameters (M, N, K) and output accumulation slot.

**Phase 2 (Cython, per matvec call):** A tight loop that:
- Receives pre-transposed 2D arrays for env blocks (cached) and theta blocks (fresh each call)
- Assembles per-combo operand lists (just pointer swaps, no transpose)
- Calls `_dgemm_row_major` directly on C pointers
- Accumulates results via `_daxpy` into pre-allocated output buffers

- [ ] **Step 1: Write test for pre-transposed matvec**

Add to `tests/test_dmrg_cython.py`:

```python
class TestPretransposedMatvec:
    """Test that pre-transposed matvec produces identical results."""

    def test_two_site_matvec_matches_einsum(self):
        """Pre-transposed combo execution matches np.einsum."""
        rng = np.random.default_rng(42)
        # Simulate a 2-site matvec: L @ theta @ W1 @ W2 @ R
        L = rng.standard_normal((8, 3, 8))
        theta = rng.standard_normal((8, 2, 2, 10))
        W1 = rng.standard_normal((3, 2, 2, 3))
        W2 = rng.standard_normal((3, 2, 2, 3))
        R = rng.standard_normal((10, 3, 10))

        expected = np.einsum("abc,apqd,bpse,eqtf,dfg->cstg", L, theta, W1, W2, R)

        from tenax.contraction._dmrg_plans import build_two_site_plan
        shapes = tuple(a.shape for a in [L, theta, W1, W2, R])
        plan = build_two_site_plan(shapes)
        result = plan.execute_numpy([L, theta, W1, W2, R])

        np.testing.assert_allclose(result, expected, atol=1e-12)
```

- [ ] **Step 2: Implement pre-transpose helper in `_precompute_matvec_combos`**

Modify `_precompute_matvec_combos` in `src/tenax/algorithms/dmrg.py` to pre-transpose ALL blocks (env + initial theta) at precompute time. Store the transposed 2D arrays directly in the combo descriptors.

For each combo, compute and store:
```python
# Per combo:
pretransposed_env = []  # list of 2D arrays for env tensor operands
theta_key = combo_keys[theta_buf_idx]
# For each GEMM step, determine which input buffer is used and pre-transpose it
buf_to_step_role = {}
for step in plan.steps:
    if step.left_idx not in buf_to_step_role:
        buf_to_step_role[step.left_idx] = (step, "left")
    if step.right_idx not in buf_to_step_role:
        buf_to_step_role[step.right_idx] = (step, "right")

for inp_idx in range(n_inputs):
    if inp_idx == theta_buf_idx:
        continue  # theta handled per matvec call
    arr = all_blocks[inp_idx]
    if inp_idx in buf_to_step_role:
        step, role = buf_to_step_role[inp_idx]
        perm = step.left_perm if role == "left" else step.right_perm
        shape_2d = (step.m, step.k) if role == "left" else (step.k, step.n)
        if perm:
            arr = np.transpose(arr, perm)
        arr = np.ascontiguousarray(arr.reshape(shape_2d))
    pretransposed_env.append(arr)
```

Also store the theta pre-transpose info (perm + shape_2d) so it can be applied per matvec call.

The combo descriptor becomes:
```python
(gemm_params,          # list of (M, N, K) per step
 pretransposed_env,    # list of 2D arrays for env inputs (pre-transposed)
 theta_key,            # block key for theta
 theta_perm,           # transpose perm for theta (or None)
 theta_shape_2d,       # (M, K) or (K, N) for theta
 theta_step_idx,       # which GEMM step uses theta
 theta_role,           # "left" or "right"
 n_steps,              # number of GEMM steps
 intermediate_shapes,  # list of out_shape per step (for buffer alloc)
 output_perm,          # final output permutation
 output_slot,          # index into output_buffers
)
```

- [ ] **Step 3: Write `cython_matvec_pretransposed` in `_cython_blas.pyx`**

New Cython function that takes pre-transposed 2D arrays and only does raw GEMM + accumulate:

```cython
def cython_matvec_pretransposed(
    list combo_data,        # list of combo tuples (see above)
    dict theta_blocks,      # raw theta blocks (N-d, need transpose)
    list output_buffers,    # pre-allocated output arrays (or None)
):
    """Execute all matvec combos with pre-transposed env blocks.
    
    Only theta blocks are transposed per call. All env blocks are
    pre-transposed 2D C-contiguous arrays. GEMM via raw dgemm/zgemm.
    """
    cdef int n_combos = len(combo_data)
    cdef int c, s, n_steps
    cdef int M, N, K
    cdef double alpha = 1.0, beta = 0.0
    cdef double* lp
    cdef double* rp
    cdef double* op
    cdef int inc = 1
    cdef cnp.ndarray left_arr, right_arr, out_arr
    
    for c in range(n_combos):
        combo = combo_data[c]
        gemm_params = combo[0]
        env_arrays = combo[1]
        theta_key = combo[2]
        theta_perm = combo[3]
        theta_shape_2d = combo[4]
        theta_step_idx = combo[5]
        theta_role = combo[6]
        n_steps = combo[7]
        intermediate_shapes = combo[8]
        output_perm_combo = combo[9]
        output_slot = combo[10]
        
        # Pre-transpose theta block
        theta_nd = theta_blocks[theta_key]
        if theta_perm:
            theta_2d = np.ascontiguousarray(
                np.transpose(theta_nd, theta_perm).reshape(theta_shape_2d))
        else:
            theta_2d = np.ascontiguousarray(theta_nd.reshape(theta_shape_2d))
        
        # Execute GEMM chain using pre-transposed env blocks
        # ... (raw dgemm calls, accumulate into output_buffers)
```

The key difference from the current `cython_matvec_combos`: env blocks arrive as 2D C-contiguous arrays — no `np.transpose` or `np.ascontiguousarray` calls for them.

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_dmrg_cython.py tests/test_dmrg.py -x --no-header -q`
Expected: All pass

- [ ] **Step 5: Rebuild Cython and benchmark**

```bash
uv run python setup.py build_ext --inplace
```

Run the benchmark script comparing old vs new matvec path.

- [ ] **Step 6: Commit**

```bash
git add src/tenax/contraction/_cython_blas.pyx src/tenax/algorithms/dmrg.py tests/test_dmrg_cython.py
git commit -m "perf: pre-transposed env blocks + raw BLAS matvec combo loop"
```

---

## Task 3: Fused Lanczos reorthogonalization

Fuse the full reorthogonalization loop — which currently makes separate Python calls to `ba_inner` + `ba_sub_scaled_inplace` per basis vector — into a single Cython function that iterates over all basis vectors and all blocks in C. Targets 2.6s (`ba_inner`) + 1.9s (loop overhead) = 4.5s.

**Files:**
- Modify: `src/tenax/contraction/_cython_blas.pyx` (new `cython_lanczos_reorth`)
- Modify: `src/tenax/algorithms/dmrg.py:1305-1393` (`_lanczos_solve_np`)
- Test: `tests/test_dmrg_cython.py`

### Design

Currently `_lanczos_solve_np` (lines 1305-1393) does per iteration:
```python
for q in basis:
    coeff = ba_inner(q, w)        # Python call → Cython ba_inner
    ba_sub_scaled_inplace(w, q, coeff)  # Python call → Cython daxpy
```

Each call iterates over all shared block keys separately. For k basis vectors and B blocks, this is 2*k*B Python→Cython transitions per iteration.

**Fused function:** `cython_lanczos_reorth(basis_blocks_list, w_blocks)` that:
1. For each basis vector q in basis_blocks_list:
   a. Compute `coeff = sum_k vdot(q[k], w[k])` (all blocks in one pass)
   b. For each block k: `w[k] -= coeff * q[k]` (daxpy, in-place)
2. Return the modified w_blocks (in-place)

This replaces 2*k Python→Cython calls with 1 call.

- [ ] **Step 1: Write test for fused reorthogonalization**

Add to `tests/test_dmrg_cython.py`:

```python
class TestFusedLanczosReorth:
    """Test fused Lanczos reorthogonalization."""

    def test_reorth_matches_sequential(self):
        """Fused reorth produces same result as sequential ba_inner + sub_scaled."""
        from tenax.algorithms._block_array import (
            BlockArray, ba_inner, ba_sub_scaled,
        )

        rng = np.random.default_rng(42)
        idx = None  # placeholder indices (not used by ba ops)

        # Create basis of 5 vectors, each with 3 blocks
        keys = [(0,), (1,), (-1,)]
        basis = []
        for _ in range(5):
            blocks = {k: rng.standard_normal((8, 10)) for k in keys}
            basis.append(BlockArray(blocks=blocks, indices=idx))

        w_blocks = {k: rng.standard_normal((8, 10)) for k in keys}
        w = BlockArray(blocks=w_blocks, indices=idx)

        # Sequential (reference)
        w_ref = BlockArray(blocks={k: v.copy() for k, v in w.blocks.items()}, indices=idx)
        for q in basis:
            coeff = ba_inner(q, w_ref)
            w_ref = ba_sub_scaled(w_ref, q, coeff)

        # Fused (Cython)
        from tenax.contraction._cython_blas import cython_lanczos_reorth
        w_fused_blocks = {k: v.copy() for k, v in w.blocks.items()}
        basis_blocks_list = [q.blocks for q in basis]
        cython_lanczos_reorth(basis_blocks_list, w_fused_blocks)

        for k in keys:
            np.testing.assert_allclose(
                w_fused_blocks[k], w_ref.blocks[k], atol=1e-14
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_dmrg_cython.py::TestFusedLanczosReorth -xvs`
Expected: `ImportError: cannot import name 'cython_lanczos_reorth'`

- [ ] **Step 3: Implement `cython_lanczos_reorth`**

Add to `src/tenax/contraction/_cython_blas.pyx`:

```cython
def cython_lanczos_reorth(list basis_blocks_list, dict w_blocks):
    """Fused full reorthogonalization: for each q in basis, w -= <q|w> * q.

    Replaces k separate (ba_inner + ba_sub_scaled_inplace) Python calls
    with a single C loop. In-place modification of w_blocks.

    Parameters
    ----------
    basis_blocks_list : list of dict
        Each dict maps charge key -> numpy array (block of basis vector).
    w_blocks : dict
        Blocks of the vector to orthogonalize. Modified in-place.
    """
    cdef int n_basis = len(basis_blocks_list)
    cdef int i, n_elem, inc
    cdef double coeff, alpha_neg
    cdef double* wp
    cdef double* qp
    cdef cnp.ndarray w_arr, q_arr

    inc = 1

    for i in range(n_basis):
        q_blocks = basis_blocks_list[i]

        # Phase 1: compute coeff = <q|w> = sum_k vdot(q[k], w[k])
        coeff = 0.0
        for k in q_blocks:
            wk = w_blocks.get(k)
            if wk is None:
                continue
            qk = q_blocks[k]
            w_arr = np.asarray(wk)
            q_arr = np.asarray(qk)
            n_elem = w_arr.size
            if w_arr.dtype == np.float64:
                wp = <double*>cnp.PyArray_DATA(w_arr)
                qp = <double*>cnp.PyArray_DATA(q_arr)
                with nogil:
                    coeff += _ddot(&n_elem, qp, &inc, wp, &inc)
            else:
                # Fallback for non-f64
                coeff += np.vdot(qk, wk).real

        # Phase 2: w -= coeff * q (daxpy with -coeff)
        if coeff == 0.0:
            continue
        alpha_neg = -coeff
        for k in q_blocks:
            wk = w_blocks.get(k)
            if wk is None:
                continue
            qk = q_blocks[k]
            w_arr = <cnp.ndarray>np.asarray(wk)
            q_arr = <cnp.ndarray>np.asarray(qk)
            n_elem = w_arr.size
            if w_arr.dtype == np.float64:
                wp = <double*>cnp.PyArray_DATA(w_arr)
                qp = <double*>cnp.PyArray_DATA(q_arr)
                with nogil:
                    _daxpy(&n_elem, &alpha_neg, qp, &inc, wp, &inc)
            else:
                w_blocks[k] = wk - coeff * qk
```

- [ ] **Step 4: Wire into `_lanczos_solve_np`**

In `src/tenax/algorithms/dmrg.py`, modify the reorthogonalization loop in `_lanczos_solve_np` (around line 1356):

Replace:
```python
# Full reorthogonalization (fused sub+scale)
for q in basis:
    coeff = ba_inner(q, w)
    if _USE_CYTHON_SUB:
        _cython_ba_sub_scaled_inplace(w.blocks, q.blocks, coeff)
    else:
        w = ba_sub_scaled(w, q, coeff)
```

With:
```python
# Full reorthogonalization
if _USE_CYTHON_REORTH:
    _cython_lanczos_reorth([q.blocks for q in basis], w.blocks)
else:
    for q in basis:
        coeff = ba_inner(q, w)
        if _USE_CYTHON_SUB:
            _cython_ba_sub_scaled_inplace(w.blocks, q.blocks, coeff)
        else:
            w = ba_sub_scaled(w, q, coeff)
```

Add import at the top of the file (near other Cython imports):
```python
try:
    from tenax.contraction._cython_blas import (
        cython_lanczos_reorth as _cython_lanczos_reorth,
    )
    _USE_CYTHON_REORTH = True
except ImportError:
    _USE_CYTHON_REORTH = False
```

- [ ] **Step 5: Run tests**

```bash
uv run python setup.py build_ext --inplace
JAX_PLATFORMS=cpu uv run pytest tests/test_dmrg_cython.py tests/test_dmrg.py -x --no-header -q
```
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/tenax/contraction/_cython_blas.pyx src/tenax/algorithms/dmrg.py tests/test_dmrg_cython.py
git commit -m "perf: fused Lanczos reorthogonalization in Cython"
```

---

## Task 4: Final benchmark and cleanup

Run the full benchmark suite comparing against TeNPy and the pre-optimization baseline.

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py` (cleanup dead code if any)

- [ ] **Step 1: Rebuild Cython**

```bash
uv run python setup.py build_ext --inplace
```

- [ ] **Step 2: Run full test suite**

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_dmrg.py tests/test_dmrg_cython.py -x --no-header -q
```
Expected: All pass

- [ ] **Step 3: Benchmark vs baseline**

Run benchmark comparing:
- Original code (git stash)
- New code (all three optimizations)
- TeNPy

For L=30, chi in [32, 64, 128, 256, 512], 4 sweeps.

- [ ] **Step 4: Commit and PR**

```bash
git add -A
git commit -m "perf: Cython DMRG pipeline — hard-coded plans, pre-transposed matvec, fused Lanczos"
```

Create PR with benchmark results in description.
