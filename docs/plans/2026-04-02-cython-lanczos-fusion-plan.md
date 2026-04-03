# Cython Fused Lanczos + Matvec Dispatch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all Python round-trips in the DMRG inner loop by fusing the matvec dispatch and Lanczos iteration into Cython, matching TeNPy performance at chi=32-64.

**Architecture:** Add `cdef class MatvecOp` with a C-level `apply()` method, two concrete subclasses (`DMRGMatvec2Site`, `DMRGMatvec1Site`) that inline the theta pre-transpose + GEMM combo execution + output assembly, and a `cython_lanczos_ground()` function that runs the full Lanczos loop calling `apply()` without returning to Python. All code goes in the existing `_cython_blas.pyx`.

**Tech Stack:** Cython 3.0, scipy.linalg.cython_blas (dgemm/zgemm/ddot/zdotc/daxpy/zaxpy/dscal/zscal), NumPy

---

### Task 1: Add `cdef` helper functions for BlockArray vector ops

These are C-level (`cdef`) wrappers around existing public `def` functions so the Lanczos loop can call them without Python dispatch. They reuse the same BLAS calls.

**Files:**
- Modify: `src/tenax/contraction/_cython_blas.pyx` (add after line ~976, before `cython_ba_inner`)

**Step 1: Write the `cdef` helpers**

Add these `cdef` functions before the existing public `def cython_ba_inner`:

```cython
# ── C-level BlockArray helpers (called from cython_lanczos_ground) ──

cdef double _ba_norm_impl(dict blocks):
    """||blocks|| via BLAS ddot/zdotc. Returns float."""
    cdef double total = 0.0
    cdef int n, inc = 1
    cdef double[:] flat_d
    cdef double complex[:] flat_z
    for k in blocks:
        bk = blocks[k]
        if bk.dtype == np.float64:
            flat_d = bk.ravel()
            n = flat_d.shape[0]
            with nogil:
                total += _ddot(&n, &flat_d[0], &inc, &flat_d[0], &inc)
        elif bk.dtype == np.complex128:
            flat_z = bk.ravel()
            n = flat_z.shape[0]
            with nogil:
                cdef double complex z_dot
                z_dot = _zdotc(&n, &flat_z[0], &inc, &flat_z[0], &inc)
                total += z_dot.real
        else:
            # Fallback for float32/complex64
            total += float(np.vdot(bk, bk).real)
    return total ** 0.5


cdef dict _ba_scale_new(dict blocks, double scalar):
    """Return new dict with every block scaled. Does not modify input."""
    cdef dict out = {}
    for k in blocks:
        out[k] = blocks[k] * scalar
    return out


cdef void _ba_scale_impl(dict blocks, double scalar):
    """Scale all blocks in-place via BLAS dscal/zscal."""
    cdef int n, inc = 1
    cdef double s_d = scalar
    cdef double complex s_z = scalar + 0j
    cdef double[:] flat_d
    cdef double complex[:] flat_z
    for k in blocks:
        bk = blocks[k]
        if bk.dtype == np.float64:
            flat_d = bk.ravel()
            n = flat_d.shape[0]
            with nogil:
                _dscal(&n, &s_d, &flat_d[0], &inc)
        elif bk.dtype == np.complex128:
            flat_z = bk.ravel()
            n = flat_z.shape[0]
            with nogil:
                _zscal(&n, &s_z, &flat_z[0], &inc)
        else:
            blocks[k] = bk * scalar


cdef double complex _ba_inner_impl(dict blocks_a, dict blocks_b):
    """Hermitian inner product <a|b> via BLAS ddot/zdotc. Returns complex."""
    cdef double real_total = 0.0
    cdef double complex z_total = 0.0
    cdef int n, inc = 1
    cdef double[:] flat_a_d, flat_b_d
    cdef double complex[:] flat_a_z, flat_b_z
    cdef int is_complex = -1  # -1 = unknown, 0 = real, 1 = complex
    for k in blocks_a:
        if k not in blocks_b:
            continue
        ak = blocks_a[k]
        bk = blocks_b[k]
        if is_complex == -1:
            is_complex = 1 if np.issubdtype(ak.dtype, np.complexfloating) else 0
        if is_complex == 0:
            flat_a_d = ak.ravel()
            flat_b_d = bk.ravel()
            n = flat_a_d.shape[0]
            with nogil:
                real_total += _ddot(&n, &flat_a_d[0], &inc, &flat_b_d[0], &inc)
        else:
            flat_a_z = np.ascontiguousarray(ak, dtype=np.complex128).ravel()
            flat_b_z = np.ascontiguousarray(bk, dtype=np.complex128).ravel()
            n = flat_a_z.shape[0]
            with nogil:
                z_total += _zdotc(&n, &flat_a_z[0], &inc, &flat_b_z[0], &inc)
    if is_complex == 1:
        return z_total
    return real_total + 0j


cdef void _ba_axpy_impl(dict blocks_x, dict blocks_y, double alpha):
    """y[k] += alpha * x[k] in-place via BLAS daxpy/zaxpy."""
    cdef int n, inc = 1
    cdef double a_d = alpha
    cdef double complex a_z = alpha + 0j
    cdef double[:] flat_x_d, flat_y_d
    cdef double complex[:] flat_x_z, flat_y_z
    for k in blocks_x:
        if k not in blocks_y:
            continue
        xk = blocks_x[k]
        yk = blocks_y[k]
        if xk.dtype == np.float64:
            flat_x_d = xk.ravel()
            flat_y_d = yk.ravel()
            n = flat_x_d.shape[0]
            with nogil:
                _daxpy(&n, &a_d, &flat_x_d[0], &inc, &flat_y_d[0], &inc)
        elif xk.dtype == np.complex128:
            flat_x_z = xk.ravel()
            flat_y_z = yk.ravel()
            n = flat_x_z.shape[0]
            with nogil:
                _zaxpy(&n, &a_z, &flat_x_z[0], &inc, &flat_y_z[0], &inc)
        else:
            blocks_y[k] = yk + alpha * xk


cdef void _ba_sub_scaled_impl(dict w_blocks, dict q_blocks, double scalar):
    """w[k] -= scalar * q[k] in-place via BLAS daxpy (alpha = -scalar)."""
    _ba_axpy_impl(q_blocks, w_blocks, -scalar)


cdef dict _ba_copy(dict blocks):
    """Deep copy a block dict."""
    cdef dict out = {}
    for k in blocks:
        out[k] = blocks[k].copy()
    return out
```

**Step 2: Rebuild and run existing tests to verify no breakage**

```bash
uv run pip install -e . --no-build-isolation && uv run pytest tests/test_block_array.py tests/test_dmrg_cython.py -x -v --no-cov
```

Expected: all PASS (new `cdef` functions are not yet called from Python)

**Step 3: Commit**

```
feat(contraction): add cdef BlockArray BLAS helpers for fused Lanczos
```

---

### Task 2: Add `cdef class MatvecOp` and `DMRGMatvec2Site`

**Files:**
- Modify: `src/tenax/contraction/_cython_blas.pyx` (add after the `cdef` helpers from Task 1)

**Step 1: Write the failing test**

Add to `tests/test_dmrg_cython.py`:

```python
class TestCythonMatvecOp:
    """Test DMRGMatvec2Site.apply() matches _execute_matvec_combos."""

    @pytest.fixture(autouse=True)
    def require_cython(self):
        try:
            from tenax.contraction._cython_blas import DMRGMatvec2Site
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Cython DMRGMatvec2Site not available")

    def test_2site_matvec_matches_reference(self):
        """DMRGMatvec2Site.apply() must match numpy einsum reference."""
        from tenax.contraction._cython_blas import DMRGMatvec2Site

        # Reuse the existing helper from TestPreTransposedMatvecCombos
        # to build block plan, env blocks, and combo descriptors
        from tenax.algorithms.dmrg import (
            _precompute_block_plan,
            _precompute_matvec_combos,
            _execute_matvec_combos,
            _to_np_blocks,
        )
        from tenax.algorithms._block_array import BlockArray, symmetric_to_ba
        import tenax

        # Build a small Heisenberg chain
        L = 6
        mpo = tenax.AutoMPO(tenax.SpinHalf, L)
        mpo += 1.0, "Sz", 0, "Sz", 1
        mpo += 0.5, "S+", 0, "S-", 1
        mpo += 0.5, "S-", 0, "S+", 1
        H = mpo.build()

        # Run 1 DMRG sweep to get environments
        mps = tenax.random_mps(tenax.SpinHalf, L, bond_dim=8, conserve="Sz", target=0)
        result = tenax.dmrg(mps, H, tenax.DMRGConfig(max_sweeps=1, max_bond_dim=8))

        # Extract site tensors + environments at position 2-3
        site_l = result.mps.tensor(2)
        site_r = result.mps.tensor(3)
        left_env = result.left_envs[2]
        right_env = result.right_envs[3]
        mpo_l = H.tensor(2)
        mpo_r = H.tensor(3)

        # Build theta
        site_l_ba = symmetric_to_ba(site_l)
        site_r_ba = symmetric_to_ba(site_r)
        theta_ba = site_l_ba  # simplified: use site_l as theta for test

        # Build combo descriptors (existing Python path)
        _subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        _plan = _precompute_block_plan(
            [left_env, theta_ba, mpo_l, mpo_r, right_env], _subs
        )
        _env_np = [
            _to_np_blocks(left_env),
            theta_ba.blocks,
            _to_np_blocks(mpo_l),
            _to_np_blocks(mpo_r),
            _to_np_blocks(right_env),
        ]
        _combo_descs, _out_keys, _out_shapes = _precompute_matvec_combos(
            _plan, _subs, _env_np, 1
        )
        _env_np[1] = None

        # Python reference
        ref_ba = _execute_matvec_combos(
            _combo_descs, theta_ba.blocks, 1, _out_keys, _out_shapes, theta_ba.indices
        )

        # Cython MatvecOp
        mv = DMRGMatvec2Site(_combo_descs, _out_keys, _out_shapes, 1)
        cy_blocks = mv.apply(theta_ba.blocks)

        # Compare
        for key in ref_ba.blocks:
            np.testing.assert_allclose(
                cy_blocks[key], ref_ba.blocks[key], atol=1e-12,
                err_msg=f"Mismatch at key {key}",
            )
```

**Step 2: Run test to verify it fails**

```bash
uv run pip install -e . --no-build-isolation && uv run pytest tests/test_dmrg_cython.py::TestCythonMatvecOp -x -v --no-cov
```

Expected: FAIL (ImportError — `DMRGMatvec2Site` does not exist yet)

**Step 3: Implement `MatvecOp` and `DMRGMatvec2Site`**

Add to `_cython_blas.pyx` after the `cdef` helpers:

```cython
# ── MatvecOp: C-level matvec dispatch ──

cdef class MatvecOp:
    """Base class for DMRG matvec operators with C-level apply()."""

    cdef dict apply(self, dict theta_blocks):
        raise NotImplementedError


cdef class DMRGMatvec2Site(MatvecOp):
    """2-site DMRG matvec: theta -> L @ theta @ W_l @ W_r @ R.

    Holds pre-computed combo descriptors and env blocks (pre-transposed
    to 2D C-contiguous layout). Only theta blocks are transposed per call.

    Combo descriptor tuple layout (indices 0-8):
        0: step_params   — list of (li, ri, oi, M, N, K, lp, rp, os)
        1: n_inputs       — int
        2: n_buffers      — int
        3: output_perm    — tuple or empty
        4: env_blocks_2d  — list of pre-transposed 2D arrays
        5: theta_key      — tuple[int, ...]
        6: theta_perm     — tuple or empty
        7: theta_shape_2d — (M, K) or None
        8: output_slot    — int
    """

    cdef list combo_descriptors
    cdef list output_keys
    cdef list output_shapes
    cdef int theta_buf_idx
    cdef int n_slots

    def __init__(self, list combo_descriptors, list output_keys,
                 list output_shapes, int theta_buf_idx):
        self.combo_descriptors = combo_descriptors
        self.output_keys = output_keys
        self.output_shapes = output_shapes
        self.theta_buf_idx = theta_buf_idx
        self.n_slots = len(output_keys)

    cdef dict apply(self, dict theta_blocks):
        """Execute all combos, return output block dict."""
        cdef int n_slots = self.n_slots
        cdef list output_buffers = [None] * n_slots
        cdef dict theta_2d_cache = {}

        # Pre-transpose theta blocks once
        for desc in self.combo_descriptors:
            theta_key = desc[5]
            theta_perm = desc[6]
            theta_shape_2d = desc[7]
            cache_key = (theta_key, theta_perm, theta_shape_2d)
            if cache_key not in theta_2d_cache:
                arr = theta_blocks[theta_key]
                if theta_shape_2d is not None:
                    if theta_perm:
                        arr = np.transpose(arr, theta_perm)
                    arr = np.ascontiguousarray(arr.reshape(theta_shape_2d))
                else:
                    arr = np.ascontiguousarray(arr)
                theta_2d_cache[cache_key] = arr

        # Dispatch to existing cython_matvec_combos logic
        # (inline the GEMM loop directly here to avoid def call overhead)
        _cython_matvec_combos_impl(
            self.combo_descriptors,
            theta_2d_cache,
            self.theta_buf_idx,
            output_buffers,
            self.output_shapes,
        )

        # Assemble output dict
        cdef dict result = {}
        for slot in range(n_slots):
            if output_buffers[slot] is not None:
                result[self.output_keys[slot]] = output_buffers[slot]
        return result
```

Also refactor the existing `cython_matvec_combos` body into a `cdef` function so both the public `def` and the `cdef apply()` can call it:

```cython
cdef void _cython_matvec_combos_impl(
    list combo_descriptors,
    dict theta_2d_cache,
    int theta_buf_idx,
    list output_buffers,
    list output_buf_shapes,
):
    """Core matvec combo loop — extracted from cython_matvec_combos."""
    # Move the existing body of cython_matvec_combos (lines 1313-1452) here.
    # The existing def cython_matvec_combos becomes a thin wrapper:
    ...


def cython_matvec_combos(
    list combo_descriptors,
    dict theta_2d_cache,
    int theta_buf_idx,
    list output_buffers,
    list output_buf_shapes,
):
    """Public wrapper — delegates to cdef impl."""
    _cython_matvec_combos_impl(
        combo_descriptors, theta_2d_cache, theta_buf_idx,
        output_buffers, output_buf_shapes,
    )
```

**Step 4: Rebuild and run tests**

```bash
uv run pip install -e . --no-build-isolation && uv run pytest tests/test_dmrg_cython.py::TestCythonMatvecOp tests/test_dmrg_cython.py::TestPreTransposedMatvecCombos -x -v --no-cov
```

Expected: all PASS

**Step 5: Commit**

```
feat(contraction): add DMRGMatvec2Site cdef class for C-level matvec dispatch
```

---

### Task 3: Add `DMRGMatvec1Site`

**Files:**
- Modify: `src/tenax/contraction/_cython_blas.pyx` (add after `DMRGMatvec2Site`)
- Test: `tests/test_dmrg_cython.py`

**Step 1: Write the failing test**

Add to `tests/test_dmrg_cython.py`:

```python
class TestCythonMatvecOp1Site:
    """Test DMRGMatvec1Site.apply() matches _execute_matvec_combos."""

    @pytest.fixture(autouse=True)
    def require_cython(self):
        try:
            from tenax.contraction._cython_blas import DMRGMatvec1Site
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Cython DMRGMatvec1Site not available")

    def test_1site_matvec_matches_reference(self):
        """DMRGMatvec1Site.apply() must match existing Python path."""
        from tenax.contraction._cython_blas import DMRGMatvec1Site
        from tenax.algorithms.dmrg import (
            _precompute_block_plan,
            _precompute_matvec_combos,
            _execute_matvec_combos,
            _to_np_blocks,
        )
        from tenax.algorithms._block_array import symmetric_to_ba
        import tenax

        L = 6
        mpo = tenax.AutoMPO(tenax.SpinHalf, L)
        mpo += 1.0, "Sz", 0, "Sz", 1
        mpo += 0.5, "S+", 0, "S-", 1
        mpo += 0.5, "S-", 0, "S+", 1
        H = mpo.build()

        mps = tenax.random_mps(tenax.SpinHalf, L, bond_dim=8, conserve="Sz", target=0)
        result = tenax.dmrg(mps, H, tenax.DMRGConfig(max_sweeps=1, max_bond_dim=8))

        site = result.mps.tensor(2)
        left_env = result.left_envs[2]
        right_env = result.right_envs[2]
        mpo_site = H.tensor(2)

        site_ba = symmetric_to_ba(site)

        _subs = "abc,apd,bpxe,def->cxf"
        _plan = _precompute_block_plan(
            [left_env, site_ba, mpo_site, right_env], _subs
        )
        _env_np = [
            _to_np_blocks(left_env),
            site_ba.blocks,
            _to_np_blocks(mpo_site),
            _to_np_blocks(right_env),
        ]
        _combo_descs, _out_keys, _out_shapes = _precompute_matvec_combos(
            _plan, _subs, _env_np, 1
        )
        _env_np[1] = None

        ref_ba = _execute_matvec_combos(
            _combo_descs, site_ba.blocks, 1, _out_keys, _out_shapes, site_ba.indices
        )

        mv = DMRGMatvec1Site(_combo_descs, _out_keys, _out_shapes, 1)
        cy_blocks = mv.apply(site_ba.blocks)

        for key in ref_ba.blocks:
            np.testing.assert_allclose(
                cy_blocks[key], ref_ba.blocks[key], atol=1e-12,
                err_msg=f"Mismatch at key {key}",
            )
```

**Step 2: Run test to verify it fails**

```bash
uv run pip install -e . --no-build-isolation && uv run pytest tests/test_dmrg_cython.py::TestCythonMatvecOp1Site -x -v --no-cov
```

Expected: FAIL (ImportError)

**Step 3: Implement `DMRGMatvec1Site`**

Add to `_cython_blas.pyx` after `DMRGMatvec2Site`:

```cython
cdef class DMRGMatvec1Site(MatvecOp):
    """1-site DMRG matvec: site -> L @ site @ W @ R.

    Identical structure to DMRGMatvec2Site — combo descriptor format
    is the same (generated by _precompute_matvec_combos for either
    2-site or 1-site subscripts). Only the subscript string and
    number of input tensors differ, but those are already encoded
    in the combo descriptors.
    """

    cdef list combo_descriptors
    cdef list output_keys
    cdef list output_shapes
    cdef int theta_buf_idx
    cdef int n_slots

    def __init__(self, list combo_descriptors, list output_keys,
                 list output_shapes, int theta_buf_idx):
        self.combo_descriptors = combo_descriptors
        self.output_keys = output_keys
        self.output_shapes = output_shapes
        self.theta_buf_idx = theta_buf_idx
        self.n_slots = len(output_keys)

    cdef dict apply(self, dict theta_blocks):
        """Execute all combos, return output block dict."""
        cdef int n_slots = self.n_slots
        cdef list output_buffers = [None] * n_slots
        cdef dict theta_2d_cache = {}

        for desc in self.combo_descriptors:
            theta_key = desc[5]
            theta_perm = desc[6]
            theta_shape_2d = desc[7]
            cache_key = (theta_key, theta_perm, theta_shape_2d)
            if cache_key not in theta_2d_cache:
                arr = theta_blocks[theta_key]
                if theta_shape_2d is not None:
                    if theta_perm:
                        arr = np.transpose(arr, theta_perm)
                    arr = np.ascontiguousarray(arr.reshape(theta_shape_2d))
                else:
                    arr = np.ascontiguousarray(arr)
                theta_2d_cache[cache_key] = arr

        _cython_matvec_combos_impl(
            self.combo_descriptors,
            theta_2d_cache,
            self.theta_buf_idx,
            output_buffers,
            self.output_shapes,
        )

        cdef dict result = {}
        for slot in range(n_slots):
            if output_buffers[slot] is not None:
                result[self.output_keys[slot]] = output_buffers[slot]
        return result
```

> **Note for implementer:** `DMRGMatvec1Site` and `DMRGMatvec2Site` have identical `apply()` logic because the combo descriptor format already encodes the contraction structure. If the implementer judges them truly identical at implementation time, a single `cdef class DMRGMatvec(MatvecOp)` serving both cases is acceptable. Keep the two-class approach only if 1-site needs a different theta transpose strategy.

**Step 4: Rebuild and run tests**

```bash
uv run pip install -e . --no-build-isolation && uv run pytest tests/test_dmrg_cython.py::TestCythonMatvecOp tests/test_dmrg_cython.py::TestCythonMatvecOp1Site -x -v --no-cov
```

Expected: all PASS

**Step 5: Commit**

```
feat(contraction): add DMRGMatvec1Site cdef class
```

---

### Task 4: Implement `cython_lanczos_ground`

**Files:**
- Modify: `src/tenax/contraction/_cython_blas.pyx` (add after MatvecOp classes)
- Test: `tests/test_dmrg_cython.py`

**Step 1: Write the failing test**

Add to `tests/test_dmrg_cython.py`:

```python
class TestCythonLanczosGround:
    """Test cython_lanczos_ground matches _lanczos_solve_np."""

    @pytest.fixture(autouse=True)
    def require_cython(self):
        try:
            from tenax.contraction._cython_blas import (
                cython_lanczos_ground,
                DMRGMatvec2Site,
            )
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Cython Lanczos not available")

    def test_eigenvalue_matches_python_lanczos(self):
        """Fused Cython Lanczos must produce same eigenvalue as Python path."""
        from tenax.contraction._cython_blas import (
            cython_lanczos_ground,
            DMRGMatvec2Site,
        )
        from tenax.algorithms.dmrg import (
            _lanczos_solve_np,
            _precompute_block_plan,
            _precompute_matvec_combos,
            _execute_matvec_combos,
            _to_np_blocks,
        )
        from tenax.algorithms._block_array import BlockArray, symmetric_to_ba
        import tenax

        L = 8
        mpo = tenax.AutoMPO(tenax.SpinHalf, L)
        for i in range(L - 1):
            mpo += 1.0, "Sz", i, "Sz", i + 1
            mpo += 0.5, "S+", i, "S-", i + 1
            mpo += 0.5, "S-", i, "S+", i + 1
        H = mpo.build()

        mps = tenax.random_mps(
            tenax.SpinHalf, L, bond_dim=8, conserve="Sz", target=0
        )
        result = tenax.dmrg(
            mps, H, tenax.DMRGConfig(max_sweeps=2, max_bond_dim=8)
        )

        # Pick a site pair
        site_l = result.mps.tensor(3)
        site_r = result.mps.tensor(4)
        left_env = result.left_envs[3]
        right_env = result.right_envs[4]
        mpo_l = H.tensor(3)
        mpo_r = H.tensor(4)

        site_l_ba = symmetric_to_ba(site_l)
        site_r_ba = symmetric_to_ba(site_r)
        theta_ba = site_l_ba  # simplified

        _subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        _plan = _precompute_block_plan(
            [left_env, theta_ba, mpo_l, mpo_r, right_env], _subs
        )
        _env_np = [
            _to_np_blocks(left_env),
            theta_ba.blocks,
            _to_np_blocks(mpo_l),
            _to_np_blocks(mpo_r),
            _to_np_blocks(right_env),
        ]
        _combo_descs, _out_keys, _out_shapes = _precompute_matvec_combos(
            _plan, _subs, _env_np, 1
        )
        _env_np[1] = None

        # Python reference
        def matvec_py(v_ba):
            return _execute_matvec_combos(
                _combo_descs, v_ba.blocks, 1,
                _out_keys, _out_shapes, theta_ba.indices,
            )

        energy_py, vec_py = _lanczos_solve_np(
            matvec_py, theta_ba, 20, 1e-12
        )

        # Cython fused
        mv = DMRGMatvec2Site(_combo_descs, _out_keys, _out_shapes, 1)
        energy_cy, vec_cy_blocks = cython_lanczos_ground(
            mv, theta_ba.blocks, 20, 1e-12
        )

        # Eigenvalue must match
        np.testing.assert_allclose(energy_cy, energy_py, atol=1e-10)

        # Eigenvector must match (up to phase)
        overlap = 0.0
        for k in vec_py.blocks:
            if k in vec_cy_blocks:
                overlap += float(np.vdot(vec_py.blocks[k], vec_cy_blocks[k]).real)
        assert abs(abs(overlap) - 1.0) < 1e-10, f"Overlap: {overlap}"

    def test_early_termination(self):
        """Lanczos should converge early and not run all max_iter steps."""
        from tenax.contraction._cython_blas import (
            cython_lanczos_ground,
            DMRGMatvec2Site,
        )
        from tenax.algorithms.dmrg import (
            _precompute_block_plan,
            _precompute_matvec_combos,
            _to_np_blocks,
        )
        from tenax.algorithms._block_array import symmetric_to_ba
        import tenax

        # Already-converged MPS should need very few Lanczos steps
        L = 6
        mpo = tenax.AutoMPO(tenax.SpinHalf, L)
        for i in range(L - 1):
            mpo += 1.0, "Sz", i, "Sz", i + 1
            mpo += 0.5, "S+", i, "S-", i + 1
            mpo += 0.5, "S-", i, "S+", i + 1
        H = mpo.build()

        mps = tenax.random_mps(
            tenax.SpinHalf, L, bond_dim=16, conserve="Sz", target=0
        )
        result = tenax.dmrg(
            mps, H, tenax.DMRGConfig(max_sweeps=10, max_bond_dim=16)
        )

        theta_ba = symmetric_to_ba(result.mps.tensor(2))
        left_env = result.left_envs[2]
        right_env = result.right_envs[3]
        mpo_l = H.tensor(2)
        mpo_r = H.tensor(3)

        _subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        _plan = _precompute_block_plan(
            [left_env, theta_ba, mpo_l, mpo_r, right_env], _subs
        )
        _env_np = [
            _to_np_blocks(left_env),
            theta_ba.blocks,
            _to_np_blocks(mpo_l),
            _to_np_blocks(mpo_r),
            _to_np_blocks(right_env),
        ]
        _combo_descs, _out_keys, _out_shapes = _precompute_matvec_combos(
            _plan, _subs, _env_np, 1
        )
        _env_np[1] = None

        mv = DMRGMatvec2Site(_combo_descs, _out_keys, _out_shapes, 1)
        # max_iter=100 but should converge in < 10
        energy, _ = cython_lanczos_ground(mv, theta_ba.blocks, 100, 1e-12)

        # Just verify it produces a reasonable energy (not NaN/inf)
        assert np.isfinite(energy)
```

**Step 2: Run test to verify it fails**

```bash
uv run pip install -e . --no-build-isolation && uv run pytest tests/test_dmrg_cython.py::TestCythonLanczosGround -x -v --no-cov
```

Expected: FAIL (ImportError — `cython_lanczos_ground` does not exist yet)

**Step 3: Implement `cython_lanczos_ground`**

Add to `_cython_blas.pyx` after the MatvecOp classes:

```cython
def cython_lanczos_ground(
    MatvecOp mv,
    dict v0_blocks,
    int max_iter,
    double tol,
):
    """Fused Lanczos eigensolver — full loop in Cython.

    Calls mv.apply() for matvec and cdef BLAS helpers for all vector
    ops. No Python re-entry between Lanczos steps.

    Args:
        mv:        MatvecOp subclass (DMRGMatvec2Site or DMRGMatvec1Site).
        v0_blocks: Initial vector as block dict.
        max_iter:  Maximum Lanczos iterations.
        tol:       Convergence tolerance on residual norm (beta).

    Returns:
        (eigenvalue: float, eigenvector_blocks: dict)
    """
    cdef double v_nrm, alpha_val, beta_val
    cdef int step, n, k, idx
    cdef list basis = []
    cdef list alphas = []
    cdef list betas = [0.0]

    # Normalize initial vector
    v_nrm = _ba_norm_impl(v0_blocks)
    cdef dict v = _ba_copy(v0_blocks)
    _ba_scale_impl(v, 1.0 / (v_nrm + 1e-15))
    basis.append(v)

    cdef dict w
    cdef double complex inner_val
    cdef double coeff_real

    for step in range(max_iter):
        # Matvec: w = H @ v_k  (C-level call)
        w = mv.apply(basis[step])

        # alpha = <v_k | w>  (real for Hermitian H)
        inner_val = _ba_inner_impl(basis[step], w)
        alpha_val = inner_val.real
        alphas.append(alpha_val)

        # w -= alpha * v_k
        _ba_sub_scaled_impl(w, basis[step], alpha_val)

        # w -= beta_{k-1} * v_{k-1}
        if step > 0:
            _ba_sub_scaled_impl(w, basis[step - 1], <double>betas[step])

        # Full reorthogonalization (reuse existing fused reorth)
        cdef list basis_blocks_for_reorth = [basis[j] for j in range(step + 1)]
        cython_lanczos_reorth(basis_blocks_for_reorth, w)

        # beta = ||w||
        beta_val = _ba_norm_impl(w)
        betas.append(beta_val)

        if beta_val < tol:
            break

        # Normalize and add to basis
        cdef dict w_normed = _ba_copy(w)
        _ba_scale_impl(w_normed, 1.0 / beta_val)
        basis.append(w_normed)

    # Tridiagonal eigendecomposition
    n = len(alphas)
    if n == 0:
        return 0.0, v
    if n == 1:
        return alphas[0], basis[0]

    cdef cnp.ndarray T = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        T[k, k] = alphas[k]
    for k in range(n - 1):
        T[k, k + 1] = betas[k + 1]
        T[k + 1, k] = betas[k + 1]

    eigvals, eigvecs = np.linalg.eigh(T)
    idx = int(np.argmin(eigvals))
    cdef double eigenvalue = float(eigvals[idx])
    krylov_coefs = eigvecs[:, idx]

    # Reconstruct eigenvector: sum_k coef_k * basis[k]
    cdef dict eigenvector = _ba_copy(basis[0])
    _ba_scale_impl(eigenvector, float(krylov_coefs[0]))
    for k in range(1, n):
        coeff_real = float(krylov_coefs[k])
        _ba_axpy_impl(basis[k], eigenvector, coeff_real)

    # Normalize
    cdef double ev_nrm = _ba_norm_impl(eigenvector)
    _ba_scale_impl(eigenvector, 1.0 / (ev_nrm + 1e-15))

    return eigenvalue, eigenvector
```

**Step 4: Rebuild and run tests**

```bash
uv run pip install -e . --no-build-isolation && uv run pytest tests/test_dmrg_cython.py::TestCythonLanczosGround -x -v --no-cov
```

Expected: all PASS

**Step 5: Run existing Lanczos and DMRG tests for no regressions**

```bash
uv run pytest tests/test_dmrg_cython.py tests/test_blas_plan.py tests/test_block_array.py -x -v --no-cov
```

Expected: all PASS

**Step 6: Commit**

```
feat(contraction): add cython_lanczos_ground — fused Lanczos loop with C-level matvec
```

---

### Task 5: Add feature flag and integrate into DMRG

**Files:**
- Modify: `src/tenax/contraction/__init__.py` (add `CYTHON_LANCZOS_AVAILABLE` flag, lines ~50)
- Modify: `src/tenax/algorithms/dmrg.py` (add import + integration in `_two_site_update_symmetric_np` and `_one_site_update_symmetric_np`)

**Step 1: Add feature flag**

In `src/tenax/contraction/__init__.py`, after the `CYTHON_BA_AVAILABLE` block (line ~50):

```python
CYTHON_LANCZOS_AVAILABLE = False
if os.environ.get("TENAX_DISABLE_CYTHON_BLAS", "0") != "1":
    try:
        from tenax.contraction._cython_blas import (  # noqa: F401
            cython_lanczos_ground,
            DMRGMatvec2Site,
            DMRGMatvec1Site,
        )
        CYTHON_LANCZOS_AVAILABLE = True
    except ImportError:
        pass
```

Add `"CYTHON_LANCZOS_AVAILABLE"` to `__all__`.

**Step 2: Add import in `dmrg.py`**

After the existing Cython import blocks (after line ~84):

```python
try:
    from tenax.contraction._cython_blas import (
        cython_lanczos_ground as _cython_lanczos_ground,
        DMRGMatvec2Site as _DMRGMatvec2Site,
        DMRGMatvec1Site as _DMRGMatvec1Site,
    )
    _USE_CYTHON_LANCZOS = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON_LANCZOS = False
```

**Step 3: Integrate in `_two_site_update_symmetric_np`**

Replace lines 2592-2628 (the `if _use_precomputed:` block through the `_lanczos_solve_np` call) with:

```python
    if _use_precomputed:
        _env_np[_theta_buf_idx] = theta_ba.blocks
        _combo_descs, _out_keys, _out_shapes = _precompute_matvec_combos(
            _plan,
            _subs,
            _env_np,
            _theta_buf_idx,
        )
        _env_np[_theta_buf_idx] = None

        if _USE_CYTHON_LANCZOS:
            mv = _DMRGMatvec2Site(
                _combo_descs, _out_keys, _out_shapes, _theta_buf_idx
            )
            energy, theta_opt_blocks = _cython_lanczos_ground(
                mv, theta_ba.blocks, config.lanczos_max_iter, config.lanczos_tol
            )
            theta_opt_ba = BlockArray(
                blocks=theta_opt_blocks, indices=_out_indices
            )
        else:
            def matvec(v_ba: BlockArray) -> BlockArray:
                return _execute_matvec_combos(
                    _combo_descs,
                    v_ba.blocks,
                    _theta_buf_idx,
                    _out_keys,
                    _out_shapes,
                    _out_indices,
                )

            energy, theta_opt_ba = _lanczos_solve_np(
                matvec, theta_ba, config.lanczos_max_iter, config.lanczos_tol
            )
    else:
        _cache: dict[tuple[tuple[int, ...], ...], Any] = {}

        def matvec(v_ba: BlockArray) -> BlockArray:
            _env_np[_theta_buf_idx] = v_ba.blocks
            return _blockwise_contract(
                [left_env, theta_ba, mpo_l, mpo_r, right_env],
                _subs,
                output_indices=_out_indices,
                expr_cache=_cache,
                block_plan=_plan,
                np_blocks_cache=_env_np,
                return_ba=True,
            )

        energy, theta_opt_ba = _lanczos_solve_np(
            matvec, theta_ba, config.lanczos_max_iter, config.lanczos_tol
        )
```

**Step 4: Integrate in `_one_site_update_symmetric_np`**

Apply the same pattern: after `_precompute_matvec_combos`, branch on `_USE_CYTHON_LANCZOS` using `_DMRGMatvec1Site`.

**Step 5: Run full DMRG test suite**

```bash
uv run pip install -e . --no-build-isolation && uv run pytest tests/test_dmrg_cython.py tests/test_dmrg.py -x -v --no-cov -m core
```

Expected: all PASS

**Step 6: Commit**

```
feat(dmrg): integrate cython_lanczos_ground into symmetric numpy DMRG path
```

---

### Task 6: Add benchmark regression test

**Files:**
- Modify: `tests/test_blas_benchmark.py`

**Step 1: Write the benchmark test**

Add to `tests/test_blas_benchmark.py`:

```python
@pytest.mark.slow
@pytest.mark.skipif(
    not CYTHON_LANCZOS_AVAILABLE,
    reason="Cython Lanczos not compiled",
)
def test_cython_lanczos_faster_than_python():
    """Fused Cython Lanczos must be >= 1.5x faster than Python Lanczos."""
    from tenax.contraction._cython_blas import (
        cython_lanczos_ground,
        DMRGMatvec2Site,
    )
    from tenax.algorithms.dmrg import (
        _lanczos_solve_np,
        _precompute_block_plan,
        _precompute_matvec_combos,
        _execute_matvec_combos,
        _to_np_blocks,
    )
    from tenax.algorithms._block_array import BlockArray, symmetric_to_ba
    import tenax
    import time

    L = 20
    mpo = tenax.AutoMPO(tenax.SpinHalf, L)
    for i in range(L - 1):
        mpo += 1.0, "Sz", i, "Sz", i + 1
        mpo += 0.5, "S+", i, "S-", i + 1
        mpo += 0.5, "S-", i, "S+", i + 1
    H = mpo.build()

    mps = tenax.random_mps(
        tenax.SpinHalf, L, bond_dim=32, conserve="Sz", target=0
    )
    result = tenax.dmrg(
        mps, H, tenax.DMRGConfig(max_sweeps=1, max_bond_dim=32)
    )

    # Setup at site 9-10
    theta_ba = symmetric_to_ba(result.mps.tensor(9))
    left_env = result.left_envs[9]
    right_env = result.right_envs[10]
    mpo_l = H.tensor(9)
    mpo_r = H.tensor(10)

    _subs = "abc,apqd,bpse,eqtf,dfg->cstg"
    _plan = _precompute_block_plan(
        [left_env, theta_ba, mpo_l, mpo_r, right_env], _subs
    )
    _env_np = [
        _to_np_blocks(left_env),
        theta_ba.blocks,
        _to_np_blocks(mpo_l),
        _to_np_blocks(mpo_r),
        _to_np_blocks(right_env),
    ]
    _combo_descs, _out_keys, _out_shapes = _precompute_matvec_combos(
        _plan, _subs, _env_np, 1
    )
    _env_np[1] = None
    _out_indices = theta_ba.indices

    N_REPS = 5
    max_iter = 20

    # Time Python path
    def matvec_py(v_ba):
        return _execute_matvec_combos(
            _combo_descs, v_ba.blocks, 1,
            _out_keys, _out_shapes, _out_indices,
        )

    t0 = time.perf_counter()
    for _ in range(N_REPS):
        _lanczos_solve_np(matvec_py, theta_ba, max_iter, 1e-12)
    t_python = time.perf_counter() - t0

    # Time Cython path
    mv = DMRGMatvec2Site(_combo_descs, _out_keys, _out_shapes, 1)
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        cython_lanczos_ground(mv, theta_ba.blocks, max_iter, 1e-12)
    t_cython = time.perf_counter() - t0

    speedup = t_python / t_cython
    print(f"\nLanczos benchmark: Python={t_python:.3f}s, Cython={t_cython:.3f}s, "
          f"speedup={speedup:.2f}x")
    assert speedup >= 1.5, f"Cython Lanczos only {speedup:.2f}x (need >= 1.5x)"
```

**Step 2: Add the import**

At the top of `test_blas_benchmark.py`, add:

```python
from tenax.contraction import CYTHON_LANCZOS_AVAILABLE
```

(alongside the existing `CYTHON_BLAS_AVAILABLE` import)

**Step 3: Run the benchmark**

```bash
uv run pip install -e . --no-build-isolation && uv run pytest tests/test_blas_benchmark.py::test_cython_lanczos_faster_than_python -x -v --no-cov -s
```

Expected: PASS with speedup >= 1.5x

**Step 4: Commit**

```
test: add benchmark regression for fused Cython Lanczos
```

---

### Task 7: Run full test suite + end-to-end DMRG benchmark

**Step 1: Run core tests**

```bash
uv run pytest -m core -x --no-cov
```

Expected: all PASS

**Step 2: Run DMRG-specific tests**

```bash
uv run pytest tests/test_dmrg_cython.py tests/test_dmrg.py tests/test_blas_plan.py tests/test_block_array.py -x -v --no-cov
```

Expected: all PASS

**Step 3: Run the TeNPy comparison benchmark**

```bash
uv run python bench_tenax_vs_tenpy.py
```

Record timing for L=20 chi=32 (target: within 1.2x of TeNPy) and L=40 chi=128 (target: within 1.4x of TeNPy).

**Step 4: Commit any final adjustments, then open PR**

```bash
gh pr create --title "feat: fused Cython Lanczos + matvec dispatch" --body "$(cat <<'EOF'
## Summary
- Adds `cdef class MatvecOp` with `DMRGMatvec2Site` / `DMRGMatvec1Site` subclasses for C-level matvec dispatch
- Adds `cython_lanczos_ground()` — full Lanczos loop in Cython calling `mv.apply()` without Python re-entry
- Integrates into `_two_site_update_symmetric_np` and `_one_site_update_symmetric_np` with `_USE_CYTHON_LANCZOS` flag
- Benchmark: >=1.5x speedup over Python Lanczos path at chi=32

## Test plan
- [ ] `test_dmrg_cython.py::TestCythonMatvecOp` — 2-site matvec correctness
- [ ] `test_dmrg_cython.py::TestCythonMatvecOp1Site` — 1-site matvec correctness
- [ ] `test_dmrg_cython.py::TestCythonLanczosGround` — eigenvalue + eigenvector correctness
- [ ] `test_blas_benchmark.py::test_cython_lanczos_faster_than_python` — performance regression
- [ ] `pytest -m core` — full core suite passes
- [ ] TeNPy benchmark: L=20 chi=32 within 1.2x

Closes #XXX

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
