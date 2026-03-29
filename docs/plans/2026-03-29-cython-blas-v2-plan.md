# Cython BLAS v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the Cython BLAS kernel so the entire block-combo loop runs in C with zero Python re-entry, targeting < 3x gap to TeNPy.

**Architecture:** Three-phase approach: (1) Python setup pre-transposes all blocks and packs flat arrays, (2) Cython kernel walks GEMM steps with raw `scipy.linalg.cython_blas.dgemm`/`zgemm` pointers, (3) Python wraps results. The kernel uses typed memoryviews and `nogil` sections. Blocks are pre-transposed at setup time so the inner loop is pure GEMM + accumulate.

**Tech Stack:** Cython 3.x, `scipy.linalg.cython_blas` (dgemm/zgemm), NumPy typed memoryviews, existing `BlasExecPlan`/`GemmStep` from `_blas_plan.py`.

---

### Task 1: Add `prepare_kernel_data()` to `_blas_plan.py`

**Files:**
- Modify: `src/tenax/contraction/_blas_plan.py`
- Test: `tests/test_blas_plan.py`

**Step 1: Write failing test**

Add to `tests/test_blas_plan.py`:

```python
class TestPrepareKernelData:
    """Tests for prepare_kernel_data — pre-transposes blocks for C kernel."""

    def test_returns_correct_structure(self):
        from tenax.contraction._blas_plan import (
            build_blas_plan,
            prepare_kernel_data,
        )

        rng = np.random.default_rng(42)
        subs = "abc,apd,bpxe,def->cxf"
        shapes = [(2, 3, 4), (2, 5, 7), (3, 5, 6, 9), (7, 9, 4)]
        plan = build_blas_plan(subs, shapes)

        np_blocks = [{(0,): rng.standard_normal(s)} for s in shapes]
        combos = [
            ([(0,)] * len(shapes), (0,)),
            ([(0,)] * len(shapes), (0,)),
        ]

        kdata = prepare_kernel_data(plan, combos, np_blocks)

        # combo_blocks_2d: list of list of 2D arrays (n_combos x n_steps_needing_input)
        assert hasattr(kdata, "combo_input_blocks")
        # output_idx: int array mapping combo -> output slot
        assert hasattr(kdata, "combo_output_idx")
        assert len(kdata.combo_output_idx) == 2
        # output_buffers: list of zeroed 2D arrays
        assert hasattr(kdata, "output_buffers")
        # output_keys: list mapping slot -> output key tuple
        assert hasattr(kdata, "output_keys")

    def test_pre_transposed_shapes_match_gemm(self):
        from tenax.contraction._blas_plan import (
            build_blas_plan,
            prepare_kernel_data,
        )

        rng = np.random.default_rng(42)
        subs = "ij,jk->ik"
        shapes = [(3, 4), (4, 5)]
        plan = build_blas_plan(subs, shapes)
        step = plan.steps[0]

        np_blocks = [{(0,): rng.standard_normal(s)} for s in shapes]
        combos = [([(0,)] * 2, (0,))]

        kdata = prepare_kernel_data(plan, combos, np_blocks)

        # Left block should be (M, K) = (3, 4)
        left_2d = kdata.combo_input_blocks[0][step.left_idx]
        assert left_2d.shape == (step.m, step.k)
        assert left_2d.flags["C_CONTIGUOUS"]

        # Right block should be (K, N) = (4, 5)
        right_2d = kdata.combo_input_blocks[0][step.right_idx]
        assert right_2d.shape == (step.k, step.n)
        assert right_2d.flags["C_CONTIGUOUS"]

    def test_numerical_correctness(self):
        """Pre-transposed blocks, when multiplied, give same result as np.einsum."""
        from tenax.contraction._blas_plan import (
            build_blas_plan,
            prepare_kernel_data,
        )

        rng = np.random.default_rng(42)
        subs = "abc,apd,bpxe,def->cxf"
        shapes = [(2, 3, 4), (2, 5, 7), (3, 5, 6, 9), (7, 9, 4)]
        arrays = [rng.standard_normal(s) for s in shapes]

        expected = np.einsum(subs, *arrays)

        plan = build_blas_plan(subs, shapes)
        np_blocks = [{(0,): a} for a in arrays]
        combos = [([(0,)] * len(shapes), (0,))]
        kdata = prepare_kernel_data(plan, combos, np_blocks)

        # Manual execute: walk steps using pre-transposed blocks
        n_bufs = plan.n_buffers
        buffers = [None] * n_bufs
        # Load pre-transposed inputs
        for idx in range(plan.n_inputs):
            buffers[idx] = kdata.combo_input_blocks[0][idx]

        for step in plan.steps:
            left_2d = buffers[step.left_idx].reshape(step.m, step.k)
            right_2d = buffers[step.right_idx].reshape(step.k, step.n)
            buffers[step.out_idx] = (left_2d @ right_2d).reshape(step.out_shape)

        result = buffers[plan.steps[-1].out_idx]
        if plan.output_perm:
            result = np.transpose(result, plan.output_perm)

        np.testing.assert_allclose(result, expected, rtol=1e-10)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_blas_plan.py::TestPrepareKernelData -v`
Expected: FAIL — `prepare_kernel_data` does not exist

**Step 3: Implement `prepare_kernel_data`**

Add to `src/tenax/contraction/_blas_plan.py`:

```python
@dataclass
class KernelData:
    """Pre-processed data for the Cython BLAS kernel.

    All block arrays are pre-transposed and reshaped to their
    GEMM-ready 2D C-contiguous layout.
    """

    plan: BlasExecPlan
    combo_input_blocks: list[list[np.ndarray]]  # [combo_idx][buf_idx] -> 2D array
    combo_output_idx: np.ndarray  # int32 array, length n_combos
    output_buffers: list[np.ndarray]  # [slot_idx] -> zeroed 2D array
    output_keys: list[tuple[int, ...]]  # [slot_idx] -> charge key
    work_buffers: list[np.ndarray]  # intermediate buffers, reused per combo
    dtype: np.dtype


def prepare_kernel_data(
    plan: BlasExecPlan,
    combos: list[tuple[list[tuple[int, ...]], tuple[int, ...]]],
    np_blocks: list[dict[tuple[int, ...], np.ndarray]],
) -> KernelData:
    """Pre-transpose all blocks and pack into flat structures for the C kernel.

    Args:
        plan:      BlasExecPlan from build_blas_plan().
        combos:    List of (combo_keys, output_key) tuples.
        np_blocks: List of dicts {block_key: ndarray}, one per input tensor.

    Returns:
        KernelData ready for execute_blas_kernel().
    """
    n_combos = len(combos)
    n_inputs = plan.n_inputs
    steps = plan.steps

    # Detect dtype from first block
    first_block = next(iter(np_blocks[0].values()))
    dtype = first_block.dtype

    # Map each input buffer to the step that first uses it as left or right.
    # For input buffers (idx < n_inputs), find which step uses them and
    # what perm/reshape to apply.
    buf_to_step_role: dict[int, tuple[GemmStep, str]] = {}
    for step in steps:
        if step.left_idx not in buf_to_step_role:
            buf_to_step_role[step.left_idx] = (step, "left")
        if step.right_idx not in buf_to_step_role:
            buf_to_step_role[step.right_idx] = (step, "right")

    # Pre-transpose each input block for each combo
    combo_input_blocks: list[list[np.ndarray]] = []
    for combo_keys, _ in combos:
        blocks_for_combo: list[np.ndarray] = [None] * plan.n_buffers  # type: ignore
        for inp_idx in range(n_inputs):
            arr = np_blocks[inp_idx][combo_keys[inp_idx]]
            if inp_idx in buf_to_step_role:
                step, role = buf_to_step_role[inp_idx]
                if role == "left":
                    perm = step.left_perm
                    shape_2d = (step.m, step.k)
                else:
                    perm = step.right_perm
                    shape_2d = (step.k, step.n)
                if perm:
                    arr = np.transpose(arr, perm)
                arr = np.ascontiguousarray(arr.reshape(shape_2d))
            else:
                arr = np.ascontiguousarray(arr)
            blocks_for_combo[inp_idx] = arr
        combo_input_blocks.append(blocks_for_combo)

    # Assign integer IDs to unique output keys
    output_key_to_idx: dict[tuple[int, ...], int] = {}
    combo_output_idx = np.empty(n_combos, dtype=np.int32)
    for i, (_, out_key) in enumerate(combos):
        if out_key not in output_key_to_idx:
            output_key_to_idx[out_key] = len(output_key_to_idx)
        combo_output_idx[i] = output_key_to_idx[out_key]

    output_keys = [None] * len(output_key_to_idx)  # type: ignore
    for key, idx in output_key_to_idx.items():
        output_keys[idx] = key

    # Pre-allocate output buffers (zeroed)
    last_step = steps[-1]
    out_shape_2d = (last_step.m * (1 if not plan.output_perm else 1), last_step.n)
    # Actually compute the final output shape from the last step
    final_size = 1
    for d in last_step.out_shape:
        final_size *= d
    # Output is stored as flat 1D, reshaped later
    output_buffers = [
        np.zeros(last_step.out_shape, dtype=dtype)
        for _ in range(len(output_key_to_idx))
    ]

    # Pre-allocate work buffers for intermediate results
    work_buffers = []
    for step in steps[:-1]:
        work_buffers.append(np.empty(step.out_shape, dtype=dtype))

    return KernelData(
        plan=plan,
        combo_input_blocks=combo_input_blocks,
        combo_output_idx=combo_output_idx,
        output_buffers=output_buffers,
        output_keys=output_keys,
        work_buffers=work_buffers,
        dtype=dtype,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_blas_plan.py::TestPrepareKernelData -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/contraction/_blas_plan.py tests/test_blas_plan.py
git commit -m "feat: add prepare_kernel_data for pre-transposed BLAS blocks"
```

---

### Task 2: Rewrite `_cython_blas.pyx` with zero-Python-reentry kernel

**Files:**
- Modify: `src/tenax/contraction/_cython_blas.pyx`
- Test: `tests/test_blas_plan.py` (add kernel correctness test)

**Step 1: Write failing test**

Add to `tests/test_blas_plan.py`:

```python
class TestCythonKernelV2:
    """Tests for the zero-Python-reentry BLAS kernel."""

    @pytest.mark.skipif(
        not pytest.importorskip("tenax.contraction._cython_blas", reason="Cython not compiled"),
        reason="Cython BLAS not compiled",
    )
    def test_kernel_matches_numpy(self):
        """C kernel produces same result as np.einsum for 1-site matvec."""
        from tenax.contraction._blas_plan import (
            build_blas_plan,
            prepare_kernel_data,
        )
        from tenax.contraction._cython_blas import execute_blas_kernel_v2

        rng = np.random.default_rng(42)
        subs = "abc,apd,bpxe,def->cxf"
        shapes = [(2, 3, 4), (2, 5, 7), (3, 5, 6, 9), (7, 9, 4)]
        arrays = [rng.standard_normal(s) for s in shapes]

        expected = np.einsum(subs, *arrays)

        plan = build_blas_plan(subs, shapes)
        np_blocks = [{(0,): a} for a in arrays]
        combos = [([(0,)] * len(shapes), (0,))]
        kdata = prepare_kernel_data(plan, combos, np_blocks)

        execute_blas_kernel_v2(kdata)

        result = kdata.output_buffers[0]
        if plan.output_perm:
            result = np.transpose(result, plan.output_perm)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.skipif(
        not pytest.importorskip("tenax.contraction._cython_blas", reason="Cython not compiled"),
        reason="Cython BLAS not compiled",
    )
    def test_kernel_accumulates_multiple_combos(self):
        """Multiple combos with same output key are summed correctly."""
        from tenax.contraction._blas_plan import (
            build_blas_plan,
            prepare_kernel_data,
        )
        from tenax.contraction._cython_blas import execute_blas_kernel_v2

        rng = np.random.default_rng(42)
        subs = "ij,jk->ik"
        shapes = [(3, 4), (4, 5)]

        a1, b1 = rng.standard_normal((3, 4)), rng.standard_normal((4, 5))
        a2, b2 = rng.standard_normal((3, 4)), rng.standard_normal((4, 5))

        expected = a1 @ b1 + a2 @ b2

        plan = build_blas_plan(subs, shapes)
        np_blocks = [
            {(0,): a1, (1,): a2},
            {(0,): b1, (1,): b2},
        ]
        combos = [
            ([(0,), (0,)], (0,)),  # a1 @ b1 -> output slot 0
            ([(1,), (1,)], (0,)),  # a2 @ b2 -> output slot 0 (accumulate)
        ]
        kdata = prepare_kernel_data(plan, combos, np_blocks)

        execute_blas_kernel_v2(kdata)

        np.testing.assert_allclose(kdata.output_buffers[0], expected, rtol=1e-10)

    @pytest.mark.skipif(
        not pytest.importorskip("tenax.contraction._cython_blas", reason="Cython not compiled"),
        reason="Cython BLAS not compiled",
    )
    def test_kernel_complex128(self):
        """Kernel works with complex128 dtype."""
        from tenax.contraction._blas_plan import (
            build_blas_plan,
            prepare_kernel_data,
        )
        from tenax.contraction._cython_blas import execute_blas_kernel_v2

        rng = np.random.default_rng(42)
        subs = "ij,jk->ik"
        shapes = [(3, 4), (4, 5)]
        arrays = [rng.standard_normal(s) + 1j * rng.standard_normal(s) for s in shapes]

        expected = arrays[0] @ arrays[1]

        plan = build_blas_plan(subs, shapes)
        np_blocks = [{(0,): a} for a in arrays]
        combos = [([(0,)] * 2, (0,))]
        kdata = prepare_kernel_data(plan, combos, np_blocks)

        execute_blas_kernel_v2(kdata)

        np.testing.assert_allclose(kdata.output_buffers[0], expected, rtol=1e-10)
```

**Step 2: Rewrite `_cython_blas.pyx`**

Replace the entire file content. The key design: use `scipy.linalg.cython_blas.dgemm` / `zgemm` with raw double pointers. For C-contiguous arrays, use the row-major BLAS trick: to compute `C = A @ B` where A is (M,K) and B is (K,N) row-major, call `dgemm("N","N", N, M, K, alpha, B_ptr, N, A_ptr, K, beta, C_ptr, N)` — BLAS sees Fortran-order views which are the transposes, and computes B^T @ A^T = (A@B)^T, but since C is also row-major, Fortran's view of C is C^T = the correct result.

```cython
# cython: language_level=3, boundscheck=False, wraparound=False
"""Zero-Python-reentry BLAS kernel for block-sparse tensor contractions.

The inner loop calls scipy.linalg.cython_blas.dgemm/zgemm directly
via C function pointers — no Python re-entry per block combo.
"""
cimport numpy as cnp
import numpy as np

from scipy.linalg.cython_blas cimport dgemm as _dgemm
from scipy.linalg.cython_blas cimport zgemm as _zgemm
from scipy.linalg.cython_blas cimport sgemm as _sgemm


# Keep the v1 function for backward compatibility
def execute_block_plan(plan, list block_combos, list np_blocks):
    """V1 kernel (kept for backward compat). See execute_blas_kernel_v2."""
    # ... (keep existing v1 implementation unchanged)
    cdef int n_combos = len(block_combos)
    cdef int n_steps = len(plan.steps)
    cdef int n_buffers = plan.n_buffers
    cdef int n_inputs = plan.n_inputs
    cdef int i, s, j

    steps = plan.steps
    output_perm = plan.output_perm

    output_accum = {}
    cdef list buffers = [None] * n_buffers

    for i in range(n_combos):
        combo_keys = block_combos[i][0]
        output_key = block_combos[i][1]
        for j in range(n_inputs):
            buffers[j] = np_blocks[j][combo_keys[j]]
        for s in range(n_steps):
            step = steps[s]
            left = buffers[step.left_idx]
            right = buffers[step.right_idx]
            if step.left_perm:
                left = np.transpose(left, step.left_perm)
            if step.right_perm:
                right = np.transpose(right, step.right_perm)
            left_2d = np.ascontiguousarray(left.reshape(step.m, step.k))
            right_2d = np.ascontiguousarray(right.reshape(step.k, step.n))
            if left_2d.dtype == np.float64:
                from scipy.linalg import blas as scipy_blas
                out_2d = scipy_blas.dgemm(1.0, left_2d, right_2d)
            else:
                out_2d = left_2d @ right_2d
            buffers[step.out_idx] = out_2d.reshape(step.out_shape)
        result = buffers[steps[n_steps - 1].out_idx]
        if output_perm:
            result = np.ascontiguousarray(np.transpose(result, output_perm))
        if output_key in output_accum:
            output_accum[output_key] = output_accum[output_key] + result
        else:
            output_accum[output_key] = result.copy()
    return output_accum


def execute_blas_kernel_v2(kdata):
    """Execute the BLAS plan with zero Python re-entry per combo.

    Args:
        kdata: KernelData from prepare_kernel_data().
               Modifies kdata.output_buffers in-place (accumulates results).
    """
    cdef int n_combos = len(kdata.combo_input_blocks)
    cdef int n_steps = len(kdata.plan.steps)
    cdef int n_inputs = kdata.plan.n_inputs
    cdef int n_buffers = kdata.plan.n_buffers

    # Extract step parameters into C arrays for fast access
    cdef int[:] step_left_idx = np.array(
        [s.left_idx for s in kdata.plan.steps], dtype=np.int32
    )
    cdef int[:] step_right_idx = np.array(
        [s.right_idx for s in kdata.plan.steps], dtype=np.int32
    )
    cdef int[:] step_out_idx = np.array(
        [s.out_idx for s in kdata.plan.steps], dtype=np.int32
    )
    cdef int[:] step_m = np.array(
        [s.m for s in kdata.plan.steps], dtype=np.int32
    )
    cdef int[:] step_n = np.array(
        [s.n for s in kdata.plan.steps], dtype=np.int32
    )
    cdef int[:] step_k = np.array(
        [s.k for s in kdata.plan.steps], dtype=np.int32
    )
    cdef int[:] out_idx_arr = kdata.combo_output_idx

    # Check dtype
    is_complex = np.iscomplexobj(kdata.output_buffers[0])
    is_f32 = (not is_complex and kdata.dtype == np.float32)

    cdef int i, s
    cdef int M, N, K
    cdef double alpha_d = 1.0, beta_zero_d = 0.0, beta_one_d = 1.0
    cdef float alpha_f = 1.0, beta_zero_f = 0.0, beta_one_f = 1.0
    cdef double complex alpha_z = 1.0, beta_zero_z = 0.0, beta_one_z = 1.0

    # Buffer array: holds pointers to pre-transposed input blocks
    # and intermediate results
    cdef list buffers = [None] * n_buffers

    # Pre-place work buffers for intermediate steps
    cdef int n_work = len(kdata.work_buffers)
    for s in range(n_work):
        buffers[kdata.plan.steps[s].out_idx] = kdata.work_buffers[s]

    # Main loop — minimize Python calls per combo
    for i in range(n_combos):
        combo_blocks = kdata.combo_input_blocks[i]

        # Load pre-transposed input blocks
        for s in range(n_inputs):
            buffers[s] = combo_blocks[s]

        # Execute GEMM chain
        for s in range(n_steps):
            M = step_m[s]
            N = step_n[s]
            K = step_k[s]

            left_arr = buffers[step_left_idx[s]]
            right_arr = buffers[step_right_idx[s]]

            if s == n_steps - 1:
                # Last step: accumulate directly into output buffer
                out_arr = kdata.output_buffers[out_idx_arr[i]]

                if is_complex:
                    _zgemm_row_major(
                        <double complex*>cnp.PyArray_DATA(right_arr),
                        <double complex*>cnp.PyArray_DATA(left_arr),
                        <double complex*>cnp.PyArray_DATA(out_arr),
                        N, M, K, alpha_z, beta_one_z,
                    )
                elif is_f32:
                    _sgemm_row_major(
                        <float*>cnp.PyArray_DATA(right_arr),
                        <float*>cnp.PyArray_DATA(left_arr),
                        <float*>cnp.PyArray_DATA(out_arr),
                        N, M, K, alpha_f, beta_one_f,
                    )
                else:
                    _dgemm_row_major(
                        <double*>cnp.PyArray_DATA(right_arr),
                        <double*>cnp.PyArray_DATA(left_arr),
                        <double*>cnp.PyArray_DATA(out_arr),
                        N, M, K, alpha_d, beta_one_d,
                    )
            else:
                # Intermediate step: write to work buffer
                out_arr = buffers[step_out_idx[s]]

                if is_complex:
                    _zgemm_row_major(
                        <double complex*>cnp.PyArray_DATA(right_arr),
                        <double complex*>cnp.PyArray_DATA(left_arr),
                        <double complex*>cnp.PyArray_DATA(out_arr),
                        N, M, K, alpha_z, beta_zero_z,
                    )
                elif is_f32:
                    _sgemm_row_major(
                        <float*>cnp.PyArray_DATA(right_arr),
                        <float*>cnp.PyArray_DATA(left_arr),
                        <float*>cnp.PyArray_DATA(out_arr),
                        N, M, K, alpha_f, beta_zero_f,
                    )
                else:
                    _dgemm_row_major(
                        <double*>cnp.PyArray_DATA(right_arr),
                        <double*>cnp.PyArray_DATA(left_arr),
                        <double*>cnp.PyArray_DATA(out_arr),
                        N, M, K, alpha_d, beta_zero_d,
                    )

    # Apply output permutation to each buffer if needed
    if kdata.plan.output_perm:
        perm = kdata.plan.output_perm
        for s in range(len(kdata.output_buffers)):
            kdata.output_buffers[s] = np.ascontiguousarray(
                np.transpose(kdata.output_buffers[s], perm)
            )


# --- Raw BLAS wrappers for row-major C arrays ---
# Trick: to compute C = A @ B in row-major,
# call dgemm("N","N", N, M, K, alpha, B, N, A, K, beta, C, N)
# because Fortran sees our row-major arrays as their transposes.

cdef void _dgemm_row_major(
    double* B, double* A, double* C,
    int N, int M, int K,
    double alpha, double beta,
) noexcept nogil:
    cdef char transa = b'N'
    cdef char transb = b'N'
    _dgemm(&transa, &transb, &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N)


cdef void _sgemm_row_major(
    float* B, float* A, float* C,
    int N, int M, int K,
    float alpha, float beta,
) noexcept nogil:
    cdef char transa = b'N'
    cdef char transb = b'N'
    _sgemm(&transa, &transb, &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N)


cdef void _zgemm_row_major(
    double complex* B, double complex* A, double complex* C,
    int N, int M, int K,
    double complex alpha, double complex beta,
) noexcept nogil:
    cdef char transa = b'N'
    cdef char transb = b'N'
    _zgemm(&transa, &transb, &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N)
```

**Important notes for the implementer:**

1. The row-major BLAS trick swaps A and B and uses (N,M,K) instead of (M,N,K). This is critical for correctness. The `cdef` wrappers encode this.

2. For the **last step** (accumulation), `beta=1.0` adds to the output buffer. For **intermediate steps**, `beta=0.0` overwrites the work buffer.

3. The output buffers must have shape matching the last step's `out_shape`. The pre-transposition in `prepare_kernel_data` ensures all input blocks are 2D and C-contiguous.

4. The `execute_block_plan` (v1) function is kept for backward compatibility — existing tests use it.

5. After the kernel, the output permutation is applied once per output buffer (not per combo).

**Step 3: Build the Cython extension**

Run: `uv run python -c "from tenax.contraction._cython_blas import execute_blas_kernel_v2; print('OK')"`
If import fails, rebuild: `uv pip install -e .`

**Step 4: Run tests**

Run: `uv run pytest tests/test_blas_plan.py::TestCythonKernelV2 -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/tenax/contraction/_cython_blas.pyx tests/test_blas_plan.py
git commit -m "feat: zero-Python-reentry Cython BLAS kernel v2"
```

---

### Task 3: Wire v2 kernel into `_blockwise_contract` in `dmrg.py`

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py:1321-1347`
- Modify: `src/tenax/contraction/__init__.py`

**Step 1: Update `__init__.py` to export v2 kernel**

Add to `src/tenax/contraction/__init__.py`:

```python
CYTHON_BLAS_V2_AVAILABLE = False
if os.environ.get("TENAX_DISABLE_CYTHON_BLAS", "0") != "1":
    try:
        from tenax.contraction._cython_blas import execute_blas_kernel_v2  # noqa: F401
        CYTHON_BLAS_V2_AVAILABLE = True
    except ImportError:
        pass
```

**Step 2: Replace the Cython branch in `_blockwise_contract`**

In `src/tenax/algorithms/dmrg.py`, replace lines 1321-1347 (the `if block_plan is not None:` branch):

```python
    if block_plan is not None:
        from tenax.contraction import CYTHON_BLAS_V2_AVAILABLE

        if CYTHON_BLAS_V2_AVAILABLE:
            import numpy as np

            from tenax.contraction._blas_plan import (
                get_cached_blas_plan,
                prepare_kernel_data,
            )
            from tenax.contraction._cython_blas import execute_blas_kernel_v2

            np_blocks_list = [
                {k: np.asarray(v) for k, v in t.blocks.items()} for t in tensors
            ]

            # Group by shape signature (M,N,K differ across charge sectors)
            shape_groups: dict[tuple, list] = {}
            for combo_keys, output_key in block_plan:
                shapes = tuple(
                    np_blocks_list[i][k].shape for i, k in enumerate(combo_keys)
                )
                shape_groups.setdefault(shapes, []).append((combo_keys, output_key))

            for shapes_key, combos in shape_groups.items():
                blas_plan = get_cached_blas_plan(subscripts, shapes_key)
                kdata = prepare_kernel_data(blas_plan, combos, np_blocks_list)
                execute_blas_kernel_v2(kdata)

                # Collect results
                for slot_idx, key in enumerate(kdata.output_keys):
                    arr = kdata.output_buffers[slot_idx]
                    output_accum.setdefault(key, []).append(arr)
        else:
            # Fallback: per-block opt_einsum with JAX arrays
            for combo_keys, output_key in block_plan:
                combo_arrays = [tensors[i].blocks[k] for i, k in enumerate(combo_keys)]
                block_shapes = tuple(a.shape for a in combo_arrays)
                if block_shapes in expr_cache:
                    expr = expr_cache[block_shapes]
                else:
                    expr = opt_einsum.contract_expression(
                        subscripts, *block_shapes, optimize="auto"
                    )
                    expr_cache[block_shapes] = expr
                result_array = expr(*combo_arrays, backend="jax")
                output_accum.setdefault(output_key, []).append(result_array)
```

**Step 3: Run DMRG correctness tests**

Run: `uv run pytest tests/test_dmrg.py -k "symmetric" -v --tb=short`
Expected: PASS — energies converge to same values as before

**Step 4: Commit**

```bash
git add src/tenax/algorithms/dmrg.py src/tenax/contraction/__init__.py
git commit -m "feat: wire v2 Cython BLAS kernel into DMRG blockwise_contract"
```

---

### Task 4: Update benchmark test and verify speedup

**Files:**
- Modify: `tests/test_blas_benchmark.py`

**Step 1: Update benchmark to test v2 kernel**

Replace `tests/test_blas_benchmark.py` content — test the v2 kernel at chi=16 and verify >= 10x speedup over the opt_einsum fallback:

```python
"""Benchmark: Cython BLAS v2 vs opt_einsum fallback for DMRG matvec."""

import time

import numpy as np
import opt_einsum
import pytest

from tenax.contraction import CYTHON_BLAS_V2_AVAILABLE


@pytest.mark.slow
@pytest.mark.skipif(not CYTHON_BLAS_V2_AVAILABLE, reason="Cython BLAS v2 not compiled")
def test_v2_kernel_faster_than_fallback():
    """V2 Cython BLAS should be at least 5x faster than opt_einsum fallback."""
    from tenax.contraction._blas_plan import build_blas_plan, prepare_kernel_data
    from tenax.contraction._cython_blas import execute_blas_kernel_v2

    rng = np.random.default_rng(42)

    # 1-site DMRG matvec at chi=16
    subs = "abc,apd,bpxe,def->cxf"
    chi, d, dw = 16, 2, 5  # realistic block sizes
    shapes = [(chi, chi, chi), (chi, d, chi), (chi, d, dw, chi), (chi, chi, chi)]
    n_combos = 200  # realistic number of block combos

    plan = build_blas_plan(subs, shapes)

    block_keys = [(i,) for i in range(n_combos)]
    np_blocks = [{k: rng.standard_normal(s) for k in block_keys} for s in shapes]
    combos = [([block_keys[i]] * len(shapes), (i % 10,)) for i in range(n_combos)]

    # Warmup
    kdata = prepare_kernel_data(plan, combos[:1], np_blocks)
    execute_blas_kernel_v2(kdata)

    # Time v2 kernel (including prepare_kernel_data)
    t0 = time.perf_counter()
    for _ in range(5):
        kdata = prepare_kernel_data(plan, combos, np_blocks)
        execute_blas_kernel_v2(kdata)
    t_v2 = (time.perf_counter() - t0) / 5

    # Time opt_einsum fallback
    expr_cache: dict = {}

    def fallback():
        accum: dict = {}
        for combo_keys, out_key in combos:
            arrays = [np_blocks[j][combo_keys[j]] for j in range(len(shapes))]
            bshapes = tuple(a.shape for a in arrays)
            if bshapes not in expr_cache:
                expr_cache[bshapes] = opt_einsum.contract_expression(
                    subs, *bshapes, optimize="auto"
                )
            result = expr_cache[bshapes](*arrays)
            if out_key in accum:
                accum[out_key] = accum[out_key] + result
            else:
                accum[out_key] = result
        return accum

    fallback()  # warmup
    t0 = time.perf_counter()
    for _ in range(5):
        fallback()
    t_fallback = (time.perf_counter() - t0) / 5

    speedup = t_fallback / t_v2
    print(
        f"\nV2: {t_v2 * 1000:.1f}ms, "
        f"Fallback: {t_fallback * 1000:.1f}ms, "
        f"Speedup: {speedup:.1f}x"
    )

    assert speedup > 5.0, (
        f"V2 kernel should be at least 5x faster, got {speedup:.1f}x "
        f"(v2={t_v2 * 1000:.1f}ms, fallback={t_fallback * 1000:.1f}ms)"
    )
```

**Step 2: Run benchmark**

Run: `uv run pytest tests/test_blas_benchmark.py -v -s`
Expected: PASS with >= 5x speedup

**Step 3: Run full DMRG test suite**

Run: `uv run pytest tests/test_dmrg.py -v --tb=short`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/test_blas_benchmark.py
git commit -m "test: update BLAS benchmark for v2 kernel, target 5x speedup"
```

---

### Task 5: End-to-end benchmark — Tenax vs TeNPy

**Step 1: Run the benchmark script**

Run: `uv run python bench_tenax_vs_tenpy.py`

Compare the new Tenax times against TeNPy. Target: < 3x gap at chi=128.

**Step 2: If gap is still > 3x, profile to find remaining bottleneck**

Likely candidates:
- `np.asarray(v)` JAX→NumPy conversion in `_blockwise_contract` (called per matvec)
- Shape grouping Python loop
- `prepare_kernel_data` called too often (should cache env blocks)

**Step 3: Commit results**

Document benchmark results in a comment or update the design doc.
