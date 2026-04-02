# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython BLAS kernel for block-sparse tensor contractions."""

from libc.string cimport memcpy, memset

import numpy as np
from scipy.linalg import blas as scipy_blas

cimport numpy as cnp
from scipy.linalg.cython_blas cimport daxpy as _daxpy
from scipy.linalg.cython_blas cimport ddot as _ddot
from scipy.linalg.cython_blas cimport dgemm as _dgemm
from scipy.linalg.cython_blas cimport dscal as _dscal
from scipy.linalg.cython_blas cimport sgemm as _sgemm
from scipy.linalg.cython_blas cimport zgemm as _zgemm
from scipy.linalg.cython_blas cimport cgemm as _cgemm
from scipy.linalg.cython_blas cimport zdotc as _zdotc
from scipy.linalg.cython_blas cimport cdotc as _cdotc
from scipy.linalg.cython_blas cimport zaxpy as _zaxpy
from scipy.linalg.cython_blas cimport caxpy as _caxpy
from scipy.linalg.cython_blas cimport zscal as _zscal
from scipy.linalg.cython_blas cimport cscal as _cscal

DEF MAX_NDIM = 8


def execute_block_plan(plan, list block_combos, list np_blocks):
    """Execute a BLAS plan over block combinations.

    Args:
        plan: BlasExecPlan instance
        block_combos: list of (combo_keys, output_key) tuples
            - combo_keys: list of block keys, one per input tensor
            - output_key: the charge sector key for the output block
        np_blocks: list of dicts {block_key: numpy_array}, one per input tensor

    Returns:
        dict {output_key: numpy_array} with accumulated results
        (multiple combos with same output_key are summed)
    """
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

        # Load input blocks
        for j in range(n_inputs):
            buffers[j] = np_blocks[j][combo_keys[j]]

        # Execute GEMM chain
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

            # Dispatch GEMM by dtype
            if left_2d.dtype == np.float64:
                out_2d = scipy_blas.dgemm(1.0, left_2d, right_2d)
            elif left_2d.dtype == np.float32:
                out_2d = scipy_blas.sgemm(1.0, left_2d, right_2d)
            elif left_2d.dtype == np.complex128:
                out_2d = scipy_blas.zgemm(1.0, left_2d, right_2d)
            elif left_2d.dtype == np.complex64:
                out_2d = scipy_blas.cgemm(1.0, left_2d, right_2d)
            else:
                out_2d = left_2d @ right_2d
            buffers[step.out_idx] = out_2d.reshape(step.out_shape)

        result = buffers[steps[n_steps - 1].out_idx]
        if output_perm:
            result = np.ascontiguousarray(np.transpose(result, output_perm))

        # Accumulate
        if output_key in output_accum:
            output_accum[output_key] = output_accum[output_key] + result
        else:
            output_accum[output_key] = result.copy()

    return output_accum


# ------------------------------------------------------------------ #
# V2 kernel: raw BLAS calls via scipy.linalg.cython_blas              #
# ------------------------------------------------------------------ #

cdef inline void _dgemm_row_major(
    int M, int N, int K,
    double alpha,
    double* A,  # (M, K) row-major
    double* B,  # (K, N) row-major
    double beta,
    double* C,  # (M, N) row-major
) noexcept nogil:
    """C = alpha * A @ B + beta * C for row-major arrays.

    Row-major BLAS trick: BLAS expects column-major (Fortran order).
    Our row-major (M,K) looks like (K,M) to Fortran, and (K,N) looks like (N,K).
    So we call dgemm("N","N", N, M, K, alpha, B, N, A, K, beta, C, N)
    which computes (in Fortran view) B^T @ A^T = (A @ B)^T, and since C is also
    row-major, Fortran's transposed view gives the correct result.
    """
    cdef char transa = b'N'
    cdef char transb = b'N'
    _dgemm(&transa, &transb, &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N)


cdef inline void _sgemm_row_major(
    int M, int N, int K,
    float alpha,
    float* A,
    float* B,
    float beta,
    float* C,
) noexcept nogil:
    """Same row-major trick for single precision."""
    cdef char transa = b'N'
    cdef char transb = b'N'
    _sgemm(&transa, &transb, &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N)


cdef inline void _zgemm_row_major(
    int M, int N, int K,
    double complex alpha,
    double complex* A,
    double complex* B,
    double complex beta,
    double complex* C,
) noexcept nogil:
    """Same row-major trick for complex128."""
    cdef char transa = b'N'
    cdef char transb = b'N'
    _zgemm(&transa, &transb, &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N)


def execute_blas_kernel_v2(kdata):
    """Execute BLAS plan with zero Python re-entry per combo.

    Uses raw scipy.linalg.cython_blas dgemm/sgemm/zgemm calls with typed
    memoryviews. Input blocks are pre-transposed by prepare_kernel_data().
    Intermediate results that need transpose for the next step are handled
    inline.

    Args:
        kdata: KernelData from prepare_kernel_data().
               Modifies kdata.output_buffers in-place.
    """
    plan = kdata.plan
    steps = plan.steps
    cdef int n_steps = len(steps)
    cdef int n_inputs = plan.n_inputs
    cdef int n_combos = len(kdata.combo_input_blocks)
    cdef int last_step = n_steps - 1
    cdef int i, s, j

    combo_input_blocks = kdata.combo_input_blocks
    cdef int[:] combo_output_idx = kdata.combo_output_idx
    output_buffers = kdata.output_buffers
    work_buffers = kdata.work_buffers
    output_perm = plan.output_perm

    # Pre-extract step parameters into Python lists for fast access.
    step_left_idx = []
    step_right_idx = []
    step_out_idx = []
    step_m = []
    step_n = []
    step_k = []
    step_out_shape = []
    # For intermediates: track which step produced each buffer and its perm info
    # when used as left or right in the NEXT step.
    step_left_perm = []
    step_right_perm = []

    for s in range(n_steps):
        st = steps[s]
        step_left_idx.append(st.left_idx)
        step_right_idx.append(st.right_idx)
        step_out_idx.append(st.out_idx)
        step_m.append(st.m)
        step_n.append(st.n)
        step_k.append(st.k)
        step_out_shape.append(st.out_shape)
        step_left_perm.append(st.left_perm)
        step_right_perm.append(st.right_perm)

    # Map from buffer index -> step index that produced it (for intermediates)
    # Work buffers are indexed by step index (0..n_steps-2)
    # step s (for s < last_step) produces work_buffers[s]

    # Determine dtype and dispatch
    dtype = kdata.dtype
    cdef int dtype_code  # 0=f64, 1=f32, 2=c128
    if dtype == np.float64:
        dtype_code = 0
    elif dtype == np.float32:
        dtype_code = 1
    elif dtype == np.complex128:
        dtype_code = 2
    else:
        raise ValueError(f"Unsupported dtype {dtype} for v2 kernel")

    # Buffer array: holds references to numpy arrays for current combo
    cdef list buffers = [None] * plan.n_buffers

    # Main loop over combos
    for i in range(n_combos):
        combo_blocks = combo_input_blocks[i]

        # Load pre-transposed input blocks
        for j in range(n_inputs):
            buffers[j] = combo_blocks[j]

        # Execute GEMM chain
        for s in range(n_steps):
            li = step_left_idx[s]
            ri = step_right_idx[s]
            oi = step_out_idx[s]
            M = step_m[s]
            N = step_n[s]
            K = step_k[s]

            # Get left operand: input blocks are already 2D C-contiguous.
            # Intermediate buffers need perm + reshape.
            left = buffers[li]
            if li >= n_inputs and step_left_perm[s]:
                left = np.ascontiguousarray(
                    np.transpose(left, step_left_perm[s]).reshape(M, K)
                )
            elif li >= n_inputs:
                left = np.ascontiguousarray(left.reshape(M, K))

            right = buffers[ri]
            if ri >= n_inputs and step_right_perm[s]:
                right = np.ascontiguousarray(
                    np.transpose(right, step_right_perm[s]).reshape(K, N)
                )
            elif ri >= n_inputs:
                right = np.ascontiguousarray(right.reshape(K, N))

            # Determine output target and beta
            if s == last_step:
                out = output_buffers[combo_output_idx[i]]
                # Flatten to 2D for GEMM, accumulate with beta=1.0
                out_flat = out.reshape(M, N)
                _dispatch_gemm(dtype_code, M, N, K, left, right, out_flat, 1.0)
                # output_buffers already updated in-place (shared memory)
            else:
                out = work_buffers[s]
                out_flat = out.reshape(M, N)
                _dispatch_gemm(dtype_code, M, N, K, left, right, out_flat, 0.0)
                # Store N-d view for potential transpose by next step
                buffers[oi] = out.reshape(step_out_shape[s])

    # Apply output_perm once per output buffer (outside combo loop)
    if output_perm:
        for idx in range(len(output_buffers)):
            output_buffers[idx] = np.ascontiguousarray(
                np.transpose(output_buffers[idx], output_perm)
            )
            kdata.output_buffers[idx] = output_buffers[idx]


cdef void _dispatch_gemm(
    int dtype_code,
    int M, int N, int K,
    cnp.ndarray left,
    cnp.ndarray right,
    cnp.ndarray out,
    double beta_val,
):
    """Dispatch raw BLAS GEMM based on dtype code."""
    cdef double alpha_d = 1.0
    cdef double beta_d = beta_val
    cdef float alpha_s = 1.0
    cdef float beta_s = <float>beta_val
    cdef double complex alpha_z = 1.0 + 0j
    cdef double complex beta_z = beta_val + 0j

    cdef double[:, ::1] left_d, right_d, out_d
    cdef float[:, ::1] left_s, right_s, out_s
    cdef double complex[:, ::1] left_z, right_z, out_z

    if dtype_code == 0:  # float64
        left_d = left
        right_d = right
        out_d = out
        with nogil:
            _dgemm_row_major(M, N, K, alpha_d, &left_d[0, 0], &right_d[0, 0], beta_d, &out_d[0, 0])
    elif dtype_code == 1:  # float32
        left_s = left
        right_s = right
        out_s = out
        with nogil:
            _sgemm_row_major(M, N, K, alpha_s, &left_s[0, 0], &right_s[0, 0], beta_s, &out_s[0, 0])
    elif dtype_code == 2:  # complex128
        left_z = left
        right_z = right
        out_z = out
        with nogil:
            _zgemm_row_major(M, N, K, alpha_z, &left_z[0, 0], &right_z[0, 0], beta_z, &out_z[0, 0])


# ------------------------------------------------------------------ #
# V3 kernel: per-plan execution with C-level transpose + raw BLAS     #
# ------------------------------------------------------------------ #

cdef void _transpose_nd_f64(
    double* src,
    double* dst,
    int ndim,
    int* shape,      # source shape
    int* perm,       # permutation
    int total_size,
) noexcept nogil:
    """Transpose an N-d array via strided copy.  Small arrays only (<64K elts)."""
    cdef int src_strides[MAX_NDIM]
    cdef int dst_shape[MAX_NDIM]
    cdef int dst_strides[MAX_NDIM]
    cdef int i, d, src_idx, dst_idx
    cdef int coords[MAX_NDIM]

    # Compute source strides (row-major)
    src_strides[ndim - 1] = 1
    for d in range(ndim - 2, -1, -1):
        src_strides[d] = src_strides[d + 1] * shape[d + 1]

    # Compute destination shape and strides
    for d in range(ndim):
        dst_shape[d] = shape[perm[d]]
    dst_strides[ndim - 1] = 1
    for d in range(ndim - 2, -1, -1):
        dst_strides[d] = dst_strides[d + 1] * dst_shape[d + 1]

    # Walk through destination in linear order, compute source index
    for d in range(ndim):
        coords[d] = 0

    for dst_idx in range(total_size):
        # Compute source index: coords[d] maps to src dimension perm[d]
        src_idx = 0
        for d in range(ndim):
            src_idx += coords[d] * src_strides[perm[d]]
        dst[dst_idx] = src[src_idx]

        # Increment coords (least-significant first)
        for d in range(ndim - 1, -1, -1):
            coords[d] += 1
            if coords[d] < dst_shape[d]:
                break
            coords[d] = 0


cdef void _transpose_nd_z128(
    double complex* src,
    double complex* dst,
    int ndim,
    int* shape,
    int* perm,
    int total_size,
) noexcept nogil:
    """Transpose an N-d complex128 array via strided copy. Small arrays only."""
    cdef int src_strides[MAX_NDIM]
    cdef int dst_shape[MAX_NDIM]
    cdef int dst_strides[MAX_NDIM]
    cdef int i, d, src_idx, dst_idx
    cdef int coords[MAX_NDIM]

    src_strides[ndim - 1] = 1
    for d in range(ndim - 2, -1, -1):
        src_strides[d] = src_strides[d + 1] * shape[d + 1]
    for d in range(ndim):
        dst_shape[d] = shape[perm[d]]
    dst_strides[ndim - 1] = 1
    for d in range(ndim - 2, -1, -1):
        dst_strides[d] = dst_strides[d + 1] * dst_shape[d + 1]
    for d in range(ndim):
        coords[d] = 0
    for dst_idx in range(total_size):
        src_idx = 0
        for d in range(ndim):
            src_idx += coords[d] * src_strides[perm[d]]
        dst[dst_idx] = src[src_idx]
        for d in range(ndim - 1, -1, -1):
            coords[d] += 1
            if coords[d] < dst_shape[d]:
                break
            coords[d] = 0


def execute_all_combos_v3(
    str subscripts,
    list block_plan,
    list np_blocks_list,
    dict plan_cache,
):
    """Execute all block combos with pre-transposed blocks and C-level BLAS.

    For each shape group:
    1. Pre-transpose unique input blocks to GEMM-ready 2D layout (Python).
    2. Build combo pointer table.
    3. Run combo loop with raw BLAS (C-level, no Python per combo).
    4. Scatter-add results to output dict.

    Parameters
    ----------
    subscripts : str
        Einsum subscript string.
    block_plan : list of (combo_keys, output_key)
        Precomputed charge-compatible block combinations.
    np_blocks_list : list of dict
        One dict per input tensor, mapping block keys to numpy arrays.
    plan_cache : dict
        Shared cache for BlasExecPlan objects (keyed by shape tuple).

    Returns
    -------
    dict mapping output_key -> numpy array (accumulated result).
    """
    from tenax.contraction._blas_plan import get_cached_blas_plan

    cdef int n_inputs = len(np_blocks_list)
    cdef int i, j

    # Group combos by shape signature
    cdef dict shape_groups = {}
    for i in range(len(block_plan)):
        combo_keys = block_plan[i][0]
        output_key = block_plan[i][1]
        shapes = tuple(
            np_blocks_list[j][combo_keys[j]].shape for j in range(n_inputs)
        )
        if shapes not in shape_groups:
            shape_groups[shapes] = []
        (<list>shape_groups[shapes]).append((combo_keys, output_key))

    cdef dict output_accum = {}

    for shapes, combos in shape_groups.items():
        if shapes not in plan_cache:
            plan_cache[shapes] = get_cached_blas_plan(subscripts, shapes)
        plan = plan_cache[shapes]

        _execute_group_pretransposed(
            plan, <list>combos, np_blocks_list, n_inputs, output_accum,
        )

    return output_accum


cdef void _execute_group_pretransposed(
    plan,
    list combos,
    list np_blocks_list,
    int n_inputs,
    dict output_accum,
):
    """Execute a shape group: pre-transpose inputs, then C-level combo loop."""
    steps = plan.steps
    cdef int n_steps = len(steps)
    output_perm = plan.output_perm
    cdef int n_combos = len(combos)
    cdef int i, s, j
    cdef int M, N, K

    # Detect dtype
    first_arr = np_blocks_list[0][combos[0][0][0]]
    dtype = first_arr.dtype
    cdef int dtype_code
    if dtype == np.float64:
        dtype_code = 0
    elif dtype == np.float32:
        dtype_code = 1
    elif dtype == np.complex128:
        dtype_code = 2
    elif dtype == np.complex64:
        dtype_code = 3
    else:
        dtype_code = -1  # fallback

    # --- Phase 1: Pre-transpose unique input blocks ---
    # Determine which perm+reshape each input buffer needs.
    # Input buffer i is first used in a specific step as left or right.
    cdef list input_perm = [None] * n_inputs  # perm to apply to input i
    cdef list input_2d_shape = [None] * n_inputs  # 2D shape after perm+reshape
    for s in range(n_steps):
        step = steps[s]
        li = step.left_idx
        ri = step.right_idx
        if li < n_inputs and input_perm[li] is None:
            input_perm[li] = step.left_perm
            input_2d_shape[li] = (step.m, step.k)
        if ri < n_inputs and input_perm[ri] is None:
            input_perm[ri] = step.right_perm
            input_2d_shape[ri] = (step.k, step.n)

    # Pre-transpose unique blocks into a pool
    # pool_key = (tensor_idx, block_key) -> pool_list_idx
    cdef dict block_pool = {}  # maps pool_key -> index in pool_list
    cdef list pool_list = []   # flat list of pre-transposed 2D arrays
    cdef int pool_idx

    for i in range(n_combos):
        combo_keys = combos[i][0]
        for j in range(n_inputs):
            block_key = combo_keys[j]
            pool_key = (j, block_key)
            if pool_key not in block_pool:
                arr = np_blocks_list[j][block_key]
                perm = input_perm[j]
                if perm:
                    arr = np.ascontiguousarray(np.transpose(arr, perm))
                arr = np.ascontiguousarray(arr.reshape(input_2d_shape[j]))
                block_pool[pool_key] = len(pool_list)
                pool_list.append(arr)

    # Build combo table: combo_idx -> [pool_idx_0, ..., pool_idx_{n_inputs-1}]
    cdef cnp.ndarray combo_table = np.empty((n_combos, n_inputs), dtype=np.intc)
    cdef int[:, ::1] combo_table_v = combo_table
    for i in range(n_combos):
        combo_keys = combos[i][0]
        for j in range(n_inputs):
            combo_table_v[i, j] = block_pool[(j, combo_keys[j])]

    # Build output mapping: assign integer IDs to unique output keys
    cdef dict out_key_to_idx = {}
    cdef cnp.ndarray combo_out = np.empty(n_combos, dtype=np.intc)
    cdef int[:] combo_out_v = combo_out
    for i in range(n_combos):
        output_key = combos[i][1]
        if output_key not in out_key_to_idx:
            out_key_to_idx[output_key] = len(out_key_to_idx)
        combo_out_v[i] = out_key_to_idx[output_key]

    cdef int n_outputs = len(out_key_to_idx)
    cdef list out_keys = [None] * n_outputs
    for k, idx in out_key_to_idx.items():
        out_keys[idx] = k

    # Pre-allocate output buffers (zeroed) — last step's out_shape
    last_step = steps[n_steps - 1]
    cdef list out_bufs = [
        np.zeros((last_step.m, last_step.n), dtype=dtype)
        for _ in range(n_outputs)
    ]

    # Pre-allocate work buffers for intermediate transposes
    # For each non-final step, we need a buffer for the intermediate result
    # AND a buffer for the transposed intermediate.
    cdef list work_gemm = []    # GEMM output buffer per step (2D)
    cdef list work_trans = []   # transpose output buffer per step (2D)
    cdef list int_perm = []     # perm to apply after GEMM (for next step)
    cdef list int_2d = []       # 2D shape for next step after perm

    for s in range(n_steps - 1):
        step = steps[s]
        work_gemm.append(np.empty((step.m, step.n), dtype=dtype))

        # Find the step that actually consumes this intermediate
        oi = step.out_idx
        consumer = None
        for ss in range(s + 1, n_steps):
            if steps[ss].left_idx == oi or steps[ss].right_idx == oi:
                consumer = steps[ss]
                break

        if consumer.left_idx == oi:
            p = consumer.left_perm
            sh = (consumer.m, consumer.k)
        else:
            p = consumer.right_perm
            sh = (consumer.k, consumer.n)
        int_perm.append(p)
        int_2d.append(sh)
        if p:
            work_trans.append(np.empty(sh, dtype=dtype))
        else:
            work_trans.append(None)  # no transpose needed

    # Add a dummy for the last step (output goes to out_bufs)
    work_gemm.append(None)

    # --- Phase 2: C-level combo loop ---
    if dtype_code == 0:
        _combo_loop_f64(
            steps, n_steps, n_inputs, n_combos,
            pool_list, combo_table_v, combo_out_v,
            out_bufs, work_gemm, work_trans,
            int_perm, int_2d,
        )
    elif dtype_code == 2:
        _combo_loop_z128(
            steps, n_steps, n_inputs, n_combos,
            pool_list, combo_table_v, combo_out_v,
            out_bufs, work_gemm, work_trans,
            int_perm, int_2d,
        )
    else:
        # Fallback to Python for float32, complex64, etc.
        _combo_loop_fallback(
            steps, n_steps, n_inputs, n_combos,
            pool_list, combo_table_v, combo_out_v,
            out_bufs, work_gemm, work_trans,
            int_perm, int_2d,
        )

    # --- Phase 3: Apply output perm and scatter to output_accum ---
    for idx in range(n_outputs):
        result = out_bufs[idx].reshape(last_step.out_shape)
        if output_perm:
            result = np.ascontiguousarray(np.transpose(result, output_perm))
        key = out_keys[idx]
        if key in output_accum:
            output_accum[key] = output_accum[key] + result
        else:
            output_accum[key] = result


cdef void _combo_loop_f64(
    tuple steps, int n_steps, int n_inputs, int n_combos,
    list pool_list, int[:, ::1] combo_table, int[:] combo_out,
    list out_bufs, list work_gemm, list work_trans,
    list int_perm, list int_2d,
):
    """Tight combo loop with raw BLAS for float64."""
    cdef int i, s, d, producer
    cdef int M, N, K
    cdef double alpha = 1.0
    cdef double beta_0 = 0.0
    cdef double beta_1 = 1.0
    cdef double[:, ::1] left_v, right_v, out_v, trans_v
    cdef double* src_p
    cdef double* dst_p

    cdef int shape_arr[MAX_NDIM]
    cdef int perm_arr[MAX_NDIM]
    cdef int ndim, total_size

    # Build buffer_idx -> step_idx mapping for intermediate recovery
    cdef dict buf_to_step = {}
    for s in range(n_steps):
        step = steps[s]
        if step.out_idx >= n_inputs:
            buf_to_step[step.out_idx] = s

    for i in range(n_combos):
        # Execute GEMM chain
        for s in range(n_steps - 1):
            step = steps[s]
            M = step.m; N = step.n; K = step.k

            # Get left operand
            if step.left_idx < n_inputs:
                left_v = pool_list[combo_table[i, step.left_idx]]
            else:
                # Intermediate from a previous step — already transposed to 2D
                producer = buf_to_step[step.left_idx]
                if int_perm[producer]:
                    left_v = work_trans[producer]
                else:
                    left_v = (<cnp.ndarray>work_gemm[producer]).reshape(int_2d[producer])

            # Get right operand
            if step.right_idx < n_inputs:
                right_v = pool_list[combo_table[i, step.right_idx]]
            else:
                producer = buf_to_step[step.right_idx]
                if int_perm[producer]:
                    right_v = work_trans[producer]
                else:
                    right_v = (<cnp.ndarray>work_gemm[producer]).reshape(int_2d[producer])

            out_v = work_gemm[s]

            with nogil:
                _dgemm_row_major(M, N, K, alpha,
                                 &left_v[0, 0], &right_v[0, 0],
                                 beta_0, &out_v[0, 0])

            # Transpose intermediate for next step if needed
            perm = int_perm[s]
            if perm:
                ndim = len(step.out_shape)
                for d in range(ndim):
                    shape_arr[d] = step.out_shape[d]
                    perm_arr[d] = perm[d]
                total_size = M * N

                trans_v = work_trans[s]
                src_p = &out_v[0, 0]
                dst_p = &trans_v[0, 0]
                with nogil:
                    _transpose_nd_f64(src_p, dst_p, ndim, shape_arr,
                                      perm_arr, total_size)

        # Last step: accumulate into output buffer
        step = steps[n_steps - 1]
        M = step.m; N = step.n; K = step.k

        if step.left_idx < n_inputs:
            left_v = pool_list[combo_table[i, step.left_idx]]
        else:
            producer = buf_to_step[step.left_idx]
            if int_perm[producer]:
                left_v = work_trans[producer]
            else:
                left_v = (<cnp.ndarray>work_gemm[producer]).reshape(int_2d[producer])

        if step.right_idx < n_inputs:
            right_v = pool_list[combo_table[i, step.right_idx]]
        else:
            producer = buf_to_step[step.right_idx]
            if int_perm[producer]:
                right_v = work_trans[producer]
            else:
                right_v = (<cnp.ndarray>work_gemm[producer]).reshape(int_2d[producer])

        out_v = out_bufs[combo_out[i]]
        with nogil:
            _dgemm_row_major(M, N, K, alpha,
                             &left_v[0, 0], &right_v[0, 0],
                             beta_1, &out_v[0, 0])


cdef void _combo_loop_z128(
    tuple steps, int n_steps, int n_inputs, int n_combos,
    list pool_list, int[:, ::1] combo_table, int[:] combo_out,
    list out_bufs, list work_gemm, list work_trans,
    list int_perm, list int_2d,
):
    """Tight combo loop with raw BLAS for complex128."""
    cdef int i, s, d, producer
    cdef int M, N, K
    cdef double complex alpha = 1.0 + 0j
    cdef double complex beta_0 = 0.0 + 0j
    cdef double complex beta_1 = 1.0 + 0j
    cdef double complex[:, ::1] left_v, right_v, out_v, trans_v
    cdef double complex* src_p
    cdef double complex* dst_p

    cdef int shape_arr[MAX_NDIM]
    cdef int perm_arr[MAX_NDIM]
    cdef int ndim, total_size

    # Build buffer_idx -> step_idx mapping for intermediate recovery
    cdef dict buf_to_step = {}
    for s in range(n_steps):
        step = steps[s]
        if step.out_idx >= n_inputs:
            buf_to_step[step.out_idx] = s

    for i in range(n_combos):
        # Execute GEMM chain
        for s in range(n_steps - 1):
            step = steps[s]
            M = step.m; N = step.n; K = step.k

            # Get left operand
            if step.left_idx < n_inputs:
                left_v = pool_list[combo_table[i, step.left_idx]]
            else:
                # Intermediate from a previous step — already transposed to 2D
                producer = buf_to_step[step.left_idx]
                if int_perm[producer]:
                    left_v = work_trans[producer]
                else:
                    left_v = (<cnp.ndarray>work_gemm[producer]).reshape(int_2d[producer])

            # Get right operand
            if step.right_idx < n_inputs:
                right_v = pool_list[combo_table[i, step.right_idx]]
            else:
                producer = buf_to_step[step.right_idx]
                if int_perm[producer]:
                    right_v = work_trans[producer]
                else:
                    right_v = (<cnp.ndarray>work_gemm[producer]).reshape(int_2d[producer])

            out_v = work_gemm[s]

            with nogil:
                _zgemm_row_major(M, N, K, alpha,
                                 &left_v[0, 0], &right_v[0, 0],
                                 beta_0, &out_v[0, 0])

            # Transpose intermediate for next step if needed
            perm = int_perm[s]
            if perm:
                ndim = len(step.out_shape)
                for d in range(ndim):
                    shape_arr[d] = step.out_shape[d]
                    perm_arr[d] = perm[d]
                total_size = M * N

                trans_v = work_trans[s]
                src_p = &out_v[0, 0]
                dst_p = &trans_v[0, 0]
                with nogil:
                    _transpose_nd_z128(src_p, dst_p, ndim, shape_arr,
                                       perm_arr, total_size)

        # Last step: accumulate into output buffer
        step = steps[n_steps - 1]
        M = step.m; N = step.n; K = step.k

        if step.left_idx < n_inputs:
            left_v = pool_list[combo_table[i, step.left_idx]]
        else:
            producer = buf_to_step[step.left_idx]
            if int_perm[producer]:
                left_v = work_trans[producer]
            else:
                left_v = (<cnp.ndarray>work_gemm[producer]).reshape(int_2d[producer])

        if step.right_idx < n_inputs:
            right_v = pool_list[combo_table[i, step.right_idx]]
        else:
            producer = buf_to_step[step.right_idx]
            if int_perm[producer]:
                right_v = work_trans[producer]
            else:
                right_v = (<cnp.ndarray>work_gemm[producer]).reshape(int_2d[producer])

        out_v = out_bufs[combo_out[i]]
        with nogil:
            _zgemm_row_major(M, N, K, alpha,
                             &left_v[0, 0], &right_v[0, 0],
                             beta_1, &out_v[0, 0])


cdef void _combo_loop_fallback(
    tuple steps, int n_steps, int n_inputs, int n_combos,
    list pool_list, int[:, ::1] combo_table, int[:] combo_out,
    list out_bufs, list work_gemm, list work_trans,
    list int_perm, list int_2d,
):
    """Fallback combo loop using scipy BLAS for non-float64."""
    cdef int i, s

    # Build buffer_idx -> step_idx mapping for intermediate recovery
    cdef dict buf_to_step = {}
    for s in range(n_steps):
        step = steps[s]
        if step.out_idx >= n_inputs:
            buf_to_step[step.out_idx] = s

    for i in range(n_combos):
        for s in range(n_steps - 1):
            step = steps[s]
            if step.left_idx < n_inputs:
                left = pool_list[combo_table[i, step.left_idx]]
            else:
                producer = buf_to_step[step.left_idx]
                if int_perm[producer]:
                    left = work_trans[producer]
                else:
                    left = work_gemm[producer].reshape(int_2d[producer])
            if step.right_idx < n_inputs:
                right = pool_list[combo_table[i, step.right_idx]]
            else:
                producer = buf_to_step[step.right_idx]
                if int_perm[producer]:
                    right = work_trans[producer]
                else:
                    right = work_gemm[producer].reshape(int_2d[producer])

            out = work_gemm[s]
            out[:] = left @ right

            perm = int_perm[s]
            if perm:
                nd_out = out.reshape(step.out_shape)
                trans = work_trans[s]
                trans[:] = np.ascontiguousarray(
                    np.transpose(nd_out, perm)
                ).reshape(int_2d[s])

        # Last step — accumulate
        step = steps[n_steps - 1]
        if step.left_idx < n_inputs:
            left = pool_list[combo_table[i, step.left_idx]]
        else:
            producer = buf_to_step[step.left_idx]
            left = work_trans[producer] if int_perm[producer] else work_gemm[producer].reshape(int_2d[producer])
        if step.right_idx < n_inputs:
            right = pool_list[combo_table[i, step.right_idx]]
        else:
            producer = buf_to_step[step.right_idx]
            right = work_trans[producer] if int_perm[producer] else work_gemm[producer].reshape(int_2d[producer])
        out_bufs[combo_out[i]] += left @ right


cdef cnp.ndarray _c_transpose(cnp.ndarray arr, tuple perm, int dtype_code):
    """Transpose: C-level for small float64/complex128, numpy fallback otherwise."""
    cdef int ndim = arr.ndim
    cdef int total = arr.size
    cdef int shape_arr[MAX_NDIM]
    cdef int perm_arr[MAX_NDIM]
    cdef int d
    cdef cnp.ndarray out, src
    cdef double* src_p_d
    cdef double* dst_p_d
    cdef double complex* src_p_z
    cdef double complex* dst_p_z

    if total <= 32768 and ndim <= MAX_NDIM:
        if dtype_code == 0:
            for d in range(ndim):
                shape_arr[d] = arr.shape[d]
                perm_arr[d] = perm[d]

            dst_shape_tuple = tuple(arr.shape[perm[d]] for d in range(ndim))
            out = np.empty(dst_shape_tuple, dtype=np.float64)
            src = np.ascontiguousarray(arr)
            src_p_d = <double*>cnp.PyArray_DATA(src)
            dst_p_d = <double*>cnp.PyArray_DATA(out)

            with nogil:
                _transpose_nd_f64(src_p_d, dst_p_d, ndim, shape_arr, perm_arr, total)
            return out

        elif dtype_code == 2:
            for d in range(ndim):
                shape_arr[d] = arr.shape[d]
                perm_arr[d] = perm[d]

            dst_shape_tuple = tuple(arr.shape[perm[d]] for d in range(ndim))
            out = np.empty(dst_shape_tuple, dtype=np.complex128)
            src = np.ascontiguousarray(arr)
            src_p_z = <double complex*>cnp.PyArray_DATA(src)
            dst_p_z = <double complex*>cnp.PyArray_DATA(out)

            with nogil:
                _transpose_nd_z128(src_p_z, dst_p_z, ndim, shape_arr, perm_arr, total)
            return out

    return np.ascontiguousarray(np.transpose(arr, perm))


# ------------------------------------------------------------------ #
# BlockArray arithmetic: BLAS-accelerated inner, axpy, scale          #
# ------------------------------------------------------------------ #

def cython_ba_inner(dict blocks_a, dict blocks_b):
    """Fast Hermitian inner product for block dicts using BLAS dot/dotc.

    Computes sum_k vdot(a[k], b[k]) over shared keys.
    Returns float for real inputs, complex for complex inputs.
    For real: ddot.  For complex128: zdotc.  For complex64: cdotc.
    Handles read-only arrays (e.g. from JAX) by copying when necessary.
    """
    cdef double total = 0.0
    cdef double complex z_total = 0.0
    cdef int n, inc = 1
    cdef double[::1] a_flat_d, b_flat_d
    cdef double complex[::1] a_flat_z, b_flat_z
    cdef float complex[::1] a_flat_c, b_flat_c
    cdef double complex z_result
    cdef float complex c_result
    cdef int dtype_code = -1  # 0=f64, 1=z128, 2=c64, -1=unknown

    # Detect dtype from first shared block
    for k in blocks_a:
        bk = blocks_b.get(k)
        if bk is not None:
            dt = blocks_a[k].dtype
            if dt == np.float64:
                dtype_code = 0
            elif dt == np.complex128:
                dtype_code = 1
            elif dt == np.complex64:
                dtype_code = 2
            break

    if dtype_code == 0:
        # float64 path: ddot
        for k in blocks_a:
            bk = blocks_b.get(k)
            if bk is not None:
                ak = blocks_a[k]
                a_arr = np.asarray(ak, dtype=np.float64)
                if not a_arr.flags.writeable:
                    a_arr = a_arr.copy()
                a_flat_d = np.ascontiguousarray(a_arr).ravel()
                b_arr = np.asarray(bk, dtype=np.float64)
                if not b_arr.flags.writeable:
                    b_arr = b_arr.copy()
                b_flat_d = np.ascontiguousarray(b_arr).ravel()
                n = a_flat_d.shape[0]
                with nogil:
                    total += _ddot(&n, &a_flat_d[0], &inc, &b_flat_d[0], &inc)
    elif dtype_code == 1:
        # complex128 path: zdotc (conjugates first arg)
        for k in blocks_a:
            bk = blocks_b.get(k)
            if bk is not None:
                ak = blocks_a[k]
                a_arr = np.asarray(ak, dtype=np.complex128)
                if not a_arr.flags.writeable:
                    a_arr = a_arr.copy()
                a_flat_z = np.ascontiguousarray(a_arr).ravel()
                b_arr = np.asarray(bk, dtype=np.complex128)
                if not b_arr.flags.writeable:
                    b_arr = b_arr.copy()
                b_flat_z = np.ascontiguousarray(b_arr).ravel()
                n = a_flat_z.shape[0]
                with nogil:
                    z_result = _zdotc(&n, &a_flat_z[0], &inc, &b_flat_z[0], &inc)
                z_total += z_result
    elif dtype_code == 2:
        # complex64 path: cdotc
        for k in blocks_a:
            bk = blocks_b.get(k)
            if bk is not None:
                ak = blocks_a[k]
                a_arr = np.asarray(ak, dtype=np.complex64)
                if not a_arr.flags.writeable:
                    a_arr = a_arr.copy()
                a_flat_c = np.ascontiguousarray(a_arr).ravel()
                b_arr = np.asarray(bk, dtype=np.complex64)
                if not b_arr.flags.writeable:
                    b_arr = b_arr.copy()
                b_flat_c = np.ascontiguousarray(b_arr).ravel()
                n = a_flat_c.shape[0]
                with nogil:
                    c_result = _cdotc(&n, &a_flat_c[0], &inc, &b_flat_c[0], &inc)
                z_total += <double complex>c_result
    else:
        # fallback: numpy vdot
        for k in blocks_a:
            bk = blocks_b.get(k)
            if bk is not None:
                z_total += np.vdot(blocks_a[k], bk)
        return complex(z_total.real, z_total.imag)

    if dtype_code == 0:
        return total
    return complex(z_total.real, z_total.imag)


def cython_ba_axpy(dict blocks_x, dict blocks_y, double alpha):
    """BLAS axpy: y[k] += alpha * x[k] for all shared keys (in-place).

    Used in Lanczos for orthogonalization: w -= q * inner(q, w).
    Modifies blocks_y in-place.  blocks_y arrays must be writable.
    Dispatches to daxpy/zaxpy/caxpy based on dtype.
    """
    cdef int n, inc = 1
    cdef double a_d = alpha
    cdef double complex a_z = alpha
    cdef float complex a_c = <float complex>alpha
    cdef double[::1] x_flat_d, y_flat_d
    cdef double complex[::1] x_flat_z, y_flat_z
    cdef float complex[::1] x_flat_c, y_flat_c
    cdef int dtype_code = -1

    # Detect dtype from first shared block
    for k in blocks_x:
        yk = blocks_y.get(k)
        if yk is not None:
            dt = blocks_x[k].dtype
            if dt == np.float64:
                dtype_code = 0
            elif dt == np.complex128:
                dtype_code = 1
            elif dt == np.complex64:
                dtype_code = 2
            break

    if dtype_code == 0:
        for k in blocks_x:
            yk = blocks_y.get(k)
            if yk is not None:
                xk = blocks_x[k]
                x_arr = np.asarray(xk, dtype=np.float64)
                if not x_arr.flags.writeable:
                    x_arr = x_arr.copy()
                x_flat_d = np.ascontiguousarray(x_arr).ravel()
                y_flat_d = yk.ravel()
                n = x_flat_d.shape[0]
                with nogil:
                    _daxpy(&n, &a_d, &x_flat_d[0], &inc, &y_flat_d[0], &inc)
    elif dtype_code == 1:
        for k in blocks_x:
            yk = blocks_y.get(k)
            if yk is not None:
                xk = blocks_x[k]
                x_arr = np.asarray(xk, dtype=np.complex128)
                if not x_arr.flags.writeable:
                    x_arr = x_arr.copy()
                x_flat_z = np.ascontiguousarray(x_arr).ravel()
                y_flat_z = yk.ravel()
                n = x_flat_z.shape[0]
                with nogil:
                    _zaxpy(&n, &a_z, &x_flat_z[0], &inc, &y_flat_z[0], &inc)
    elif dtype_code == 2:
        for k in blocks_x:
            yk = blocks_y.get(k)
            if yk is not None:
                xk = blocks_x[k]
                x_arr = np.asarray(xk, dtype=np.complex64)
                if not x_arr.flags.writeable:
                    x_arr = x_arr.copy()
                x_flat_c = np.ascontiguousarray(x_arr).ravel()
                y_flat_c = yk.ravel()
                n = x_flat_c.shape[0]
                with nogil:
                    _caxpy(&n, &a_c, &x_flat_c[0], &inc, &y_flat_c[0], &inc)
    else:
        for k in blocks_x:
            yk = blocks_y.get(k)
            if yk is not None:
                yk += alpha * blocks_x[k]


def cython_ba_scale_inplace(dict blocks, double scalar):
    """Scale all blocks in-place by scalar using BLAS dscal/zscal/cscal."""
    cdef int n, inc = 1
    cdef double s_d = scalar
    cdef double complex s_z = scalar
    cdef float complex s_c = <float complex>scalar
    cdef double[::1] flat_d
    cdef double complex[::1] flat_z
    cdef float complex[::1] flat_c
    cdef int dtype_code = -1

    # Detect dtype from first block
    for k in blocks:
        dt = blocks[k].dtype
        if dt == np.float64:
            dtype_code = 0
        elif dt == np.complex128:
            dtype_code = 1
        elif dt == np.complex64:
            dtype_code = 2
        break

    if dtype_code == 0:
        for k in blocks:
            flat_d = blocks[k].ravel()
            n = flat_d.shape[0]
            with nogil:
                _dscal(&n, &s_d, &flat_d[0], &inc)
    elif dtype_code == 1:
        for k in blocks:
            flat_z = blocks[k].ravel()
            n = flat_z.shape[0]
            with nogil:
                _zscal(&n, &s_z, &flat_z[0], &inc)
    elif dtype_code == 2:
        for k in blocks:
            flat_c = blocks[k].ravel()
            n = flat_c.shape[0]
            with nogil:
                _cscal(&n, &s_c, &flat_c[0], &inc)
    else:
        for k in blocks:
            blocks[k] *= scalar


# ------------------------------------------------------------------ #
# cython_execute_plan: fast per-combo BLAS plan execution              #
# ------------------------------------------------------------------ #

def cython_execute_plan(
    list step_params,
    int n_inputs,
    int n_buffers,
    tuple output_perm,
    list arrays,
):
    """Execute BLAS plan with minimal Python dispatch.

    Replaces BlasExecPlan.execute_numpy for the hot path in
    _blockwise_contract. Uses raw dgemm instead of numpy's ``@``
    operator, avoiding ~2us of numpy matmul dispatch per call.

    Parameters
    ----------
    step_params : list of tuples
        Pre-extracted (left_idx, right_idx, out_idx, m, n, k,
        left_perm, right_perm, out_shape) from plan.steps.
    n_inputs : int
        Number of input arrays.
    n_buffers : int
        Total buffer slots (inputs + intermediates).
    output_perm : tuple
        Final output transpose permutation; () for identity.
    arrays : list
        Input numpy arrays, length == n_inputs.
    """
    cdef int n_steps = len(step_params)
    cdef int s, li, ri, oi, M, N, K
    cdef double alpha_d = 1.0, beta_d = 0.0
    cdef double complex alpha_z = 1.0 + 0j, beta_z = 0.0 + 0j
    cdef double* left_p
    cdef double* right_p
    cdef double* out_p
    cdef double complex* left_p_z
    cdef double complex* right_p_z
    cdef double complex* out_p_z
    cdef cnp.ndarray left_arr, right_arr, out_arr

    cdef list buffers = list(arrays)
    if n_buffers > n_inputs:
        buffers.extend([None] * (n_buffers - n_inputs))

    for s in range(n_steps):
        li, ri, oi, M, N, K, lp, rp, os = step_params[s]

        left = buffers[li]
        right = buffers[ri]

        if lp:
            left = np.ascontiguousarray(np.transpose(left, lp).reshape(M, K))
        else:
            left = np.ascontiguousarray(left.reshape(M, K))

        if rp:
            right = np.ascontiguousarray(np.transpose(right, rp).reshape(K, N))
        else:
            right = np.ascontiguousarray(right.reshape(K, N))

        if left.dtype == np.float64:
            out_arr = np.empty((M, N), dtype=np.float64)
            # Use PyArray_DATA to avoid writable requirement of typed memoryviews
            left_arr = np.asarray(left)
            right_arr = np.asarray(right)
            left_p = <double*>cnp.PyArray_DATA(left_arr)
            right_p = <double*>cnp.PyArray_DATA(right_arr)
            out_p = <double*>cnp.PyArray_DATA(out_arr)
            with nogil:
                _dgemm_row_major(M, N, K, alpha_d, left_p, right_p, beta_d, out_p)
            buffers[oi] = out_arr.reshape(os)
        elif left.dtype == np.complex128:
            out_arr = np.empty((M, N), dtype=np.complex128)
            left_arr = np.asarray(left)
            right_arr = np.asarray(right)
            left_p_z = <double complex*>cnp.PyArray_DATA(left_arr)
            right_p_z = <double complex*>cnp.PyArray_DATA(right_arr)
            out_p_z = <double complex*>cnp.PyArray_DATA(out_arr)
            with nogil:
                _zgemm_row_major(M, N, K, alpha_z, left_p_z, right_p_z, beta_z, out_p_z)
            buffers[oi] = out_arr.reshape(os)
        else:
            buffers[oi] = (left @ right).reshape(os)

    result = buffers[step_params[n_steps - 1][2]]
    if output_perm:
        result = np.ascontiguousarray(np.transpose(result, output_perm))
    return result


# ------------------------------------------------------------------ #
# cython_matvec_combos: batched combo execution for DMRG matvec        #
# ------------------------------------------------------------------ #


def cython_matvec_combos(
    list combo_descriptors,
    dict theta_2d_cache,
    int theta_buf_idx,
    list output_buffers,
    list output_buf_shapes,
):
    """Execute ALL block combos for one matvec call in a single C loop.

    All input blocks (env and theta) are **pre-transposed** to their
    GEMM-ready 2D C-contiguous layout before this function is called.
    Only intermediate buffers (from multi-step plans) need transposing
    inside the loop.

    Each combo descriptor is a tuple:
        (step_params, n_inputs, n_buffers, output_perm,
         env_blocks_2d,       # list of pre-transposed 2D numpy arrays
         theta_key,           # block key into theta_blocks dict
         theta_perm,          # transpose perm for theta (used as cache key)
         theta_shape_2d,      # (M, K) or (K, N) for theta (used as cache key)
         output_slot,         # int index into output_buffers
        )

    ``theta_2d_cache`` maps ``(theta_key, theta_perm, theta_shape_2d)``
    to the pre-transposed 2D theta block.

    ``output_buffers`` is a list of pre-allocated numpy arrays (zeroed),
    one per unique output key. Results are accumulated in-place via +=.
    """
    cdef int n_combos = len(combo_descriptors)
    cdef int c, s, n_steps, n_elem, inc
    cdef int li, ri, oi, M, N, K, n_inputs
    cdef double alpha_d = 1.0, beta_d = 0.0
    cdef double complex alpha_z = 1.0 + 0j, beta_z = 0.0 + 0j
    cdef double* left_p
    cdef double* right_p
    cdef double* out_p
    cdef double complex* left_p_z
    cdef double complex* right_p_z
    cdef double complex* out_p_z
    cdef cnp.ndarray left_arr, right_arr, out_arr

    inc = 1

    for c in range(n_combos):
        desc = combo_descriptors[c]
        step_params = desc[0]
        n_inputs = desc[1]
        n_buffers = desc[2]
        output_perm = desc[3]
        env_blocks_2d = desc[4]
        theta_key = desc[5]
        theta_perm = desc[6]
        theta_shape_2d = desc[7]
        output_slot = desc[8]

        n_steps = len(step_params)

        # Build buffer list: all inputs are pre-transposed 2D
        theta_2d = theta_2d_cache[(theta_key, theta_perm, theta_shape_2d)]
        buffers = list(env_blocks_2d)
        buffers.insert(theta_buf_idx, theta_2d)
        if n_buffers > n_inputs:
            buffers.extend([None] * (n_buffers - n_inputs))

        # Execute GEMM chain
        for s in range(n_steps):
            li, ri, oi, M, N, K, lp, rp, os = step_params[s]

            left = buffers[li]
            right = buffers[ri]

            # Input buffers (idx < n_inputs) are already 2D C-contiguous.
            # Intermediate buffers need transpose + reshape.
            if li >= n_inputs:
                if lp:
                    left = np.ascontiguousarray(np.transpose(left, lp).reshape(M, K))
                else:
                    left = np.ascontiguousarray(left.reshape(M, K))

            if ri >= n_inputs:
                if rp:
                    right = np.ascontiguousarray(np.transpose(right, rp).reshape(K, N))
                else:
                    right = np.ascontiguousarray(right.reshape(K, N))

            if left.dtype == np.float64:
                out_arr = np.empty((M, N), dtype=np.float64)
                left_arr = np.asarray(left)
                right_arr = np.asarray(right)
                left_p = <double*>cnp.PyArray_DATA(left_arr)
                right_p = <double*>cnp.PyArray_DATA(right_arr)
                out_p = <double*>cnp.PyArray_DATA(out_arr)
                with nogil:
                    _dgemm_row_major(M, N, K, alpha_d, left_p, right_p, beta_d, out_p)
                buffers[oi] = out_arr.reshape(os)
            elif left.dtype == np.complex128:
                out_arr = np.empty((M, N), dtype=np.complex128)
                left_arr = np.asarray(left)
                right_arr = np.asarray(right)
                left_p_z = <double complex*>cnp.PyArray_DATA(left_arr)
                right_p_z = <double complex*>cnp.PyArray_DATA(right_arr)
                out_p_z = <double complex*>cnp.PyArray_DATA(out_arr)
                with nogil:
                    _zgemm_row_major(M, N, K, alpha_z, left_p_z, right_p_z, beta_z, out_p_z)
                buffers[oi] = out_arr.reshape(os)
            else:
                buffers[oi] = (left @ right).reshape(os)

        result = buffers[step_params[n_steps - 1][2]]
        if output_perm:
            result = np.ascontiguousarray(np.transpose(result, output_perm))

        # Accumulate into output buffer
        ob = output_buffers[output_slot]
        if ob is None:
            output_buffers[output_slot] = result.copy()
        else:
            # In-place add via daxpy (avoid allocation)
            n_elem = result.size
            result_c = np.ascontiguousarray(result.reshape(-1))
            ob_flat = ob.reshape(-1)
            if ob.dtype == np.float64:
                left_p = <double*>cnp.PyArray_DATA(<cnp.ndarray>result_c)
                out_p = <double*>cnp.PyArray_DATA(<cnp.ndarray>ob_flat)
                with nogil:
                    _daxpy(&n_elem, &alpha_d, left_p, &inc, out_p, &inc)
            elif ob.dtype == np.complex128:
                left_p_z = <double complex*>cnp.PyArray_DATA(<cnp.ndarray>result_c)
                out_p_z = <double complex*>cnp.PyArray_DATA(<cnp.ndarray>ob_flat)
                with nogil:
                    _zaxpy(&n_elem, &alpha_z, left_p_z, &inc, out_p_z, &inc)
            else:
                output_buffers[output_slot] = ob + result


# ------------------------------------------------------------------ #
# cython_lanczos_reorth: fused full reorthogonalization                #
# ------------------------------------------------------------------ #


def cython_lanczos_reorth(list basis_blocks_list, dict w_blocks):
    """Fused full reorthogonalization: for each q in basis, w -= <q|w> * q.

    In-place modification of w_blocks.
    basis_blocks_list: list of dicts, each mapping charge key -> numpy array
    w_blocks: dict mapping charge key -> numpy array (modified in-place)

    Fuses what was 2*k Python->Cython calls (one ba_inner + one
    ba_sub_scaled_inplace per basis vector) into a single call.
    """
    cdef int n_basis = len(basis_blocks_list)
    cdef int i, n, inc
    cdef double coeff, alpha_neg
    cdef double* wp
    cdef double* qp
    cdef cnp.ndarray w_arr, q_arr
    cdef double complex z_coeff, z_alpha_neg, z_coeff_accum
    cdef double complex* wp_z
    cdef double complex* qp_z
    cdef float complex c_coeff_val, c_alpha_neg, c_coeff_accum
    cdef float complex* wp_c
    cdef float complex* qp_c
    cdef int dtype_code = -1  # 0=f64, 1=z128, 2=c64

    inc = 1

    # Detect dtype from first shared block
    if n_basis > 0:
        q0_blocks = basis_blocks_list[0]
        for k in q0_blocks:
            wk = w_blocks.get(k)
            if wk is not None:
                dt = wk.dtype
                if dt == np.float64:
                    dtype_code = 0
                elif dt == np.complex128:
                    dtype_code = 1
                elif dt == np.complex64:
                    dtype_code = 2
                break

    for i in range(n_basis):
        q_blocks = basis_blocks_list[i]

        if dtype_code == 0:
            # Phase 1: coeff = <q|w> via ddot
            coeff = 0.0
            for k in q_blocks:
                wk = w_blocks.get(k)
                if wk is None:
                    continue
                qk = q_blocks[k]
                w_arr = np.ascontiguousarray(np.asarray(wk, dtype=np.float64).ravel())
                q_arr = np.ascontiguousarray(np.asarray(qk, dtype=np.float64).ravel())
                n = w_arr.size
                wp = <double*>cnp.PyArray_DATA(w_arr)
                qp = <double*>cnp.PyArray_DATA(q_arr)
                with nogil:
                    coeff += _ddot(&n, qp, &inc, wp, &inc)

            # Phase 2: w -= coeff * q via daxpy
            if coeff == 0.0:
                continue
            alpha_neg = -coeff
            for k in q_blocks:
                wk = w_blocks.get(k)
                if wk is None:
                    continue
                qk = q_blocks[k]
                if not wk.flags.writeable:
                    wk = wk.copy()
                    w_blocks[k] = wk
                w_arr = <cnp.ndarray>np.ascontiguousarray(np.asarray(wk).ravel())
                q_arr = <cnp.ndarray>np.ascontiguousarray(np.asarray(qk).ravel())
                n = w_arr.size
                wp = <double*>cnp.PyArray_DATA(w_arr)
                qp = <double*>cnp.PyArray_DATA(q_arr)
                with nogil:
                    _daxpy(&n, &alpha_neg, qp, &inc, wp, &inc)

        elif dtype_code == 1:
            # Phase 1: coeff = <q|w> via zdotc (full complex)
            z_coeff_accum = 0.0
            for k in q_blocks:
                wk = w_blocks.get(k)
                if wk is None:
                    continue
                qk = q_blocks[k]
                w_arr_z = np.asarray(wk, dtype=np.complex128)
                if not w_arr_z.flags.writeable:
                    w_arr_z = w_arr_z.copy()
                w_flat_z = np.ascontiguousarray(w_arr_z).ravel()
                q_arr_z = np.asarray(qk, dtype=np.complex128)
                if not q_arr_z.flags.writeable:
                    q_arr_z = q_arr_z.copy()
                q_flat_z = np.ascontiguousarray(q_arr_z).ravel()
                n = w_flat_z.shape[0]
                wp_z = <double complex*>cnp.PyArray_DATA(<cnp.ndarray>w_flat_z)
                qp_z = <double complex*>cnp.PyArray_DATA(<cnp.ndarray>q_flat_z)
                with nogil:
                    z_coeff = _zdotc(&n, qp_z, &inc, wp_z, &inc)
                z_coeff_accum += z_coeff

            # Phase 2: w -= coeff * q via zaxpy
            if z_coeff_accum == 0.0:
                continue
            z_alpha_neg = -z_coeff_accum
            for k in q_blocks:
                wk = w_blocks.get(k)
                if wk is None:
                    continue
                qk = q_blocks[k]
                if not wk.flags.writeable:
                    wk = wk.copy()
                    w_blocks[k] = wk
                w_flat_z2 = np.ascontiguousarray(np.asarray(wk, dtype=np.complex128)).ravel()
                q_flat_z2 = np.ascontiguousarray(np.asarray(qk, dtype=np.complex128)).ravel()
                n = w_flat_z2.shape[0]
                wp_z = <double complex*>cnp.PyArray_DATA(<cnp.ndarray>w_flat_z2)
                qp_z = <double complex*>cnp.PyArray_DATA(<cnp.ndarray>q_flat_z2)
                with nogil:
                    _zaxpy(&n, &z_alpha_neg, qp_z, &inc, wp_z, &inc)

        elif dtype_code == 2:
            # Phase 1: coeff = <q|w> via cdotc (full complex)
            c_coeff_accum = 0.0
            for k in q_blocks:
                wk = w_blocks.get(k)
                if wk is None:
                    continue
                qk = q_blocks[k]
                w_arr_c = np.asarray(wk, dtype=np.complex64)
                if not w_arr_c.flags.writeable:
                    w_arr_c = w_arr_c.copy()
                w_flat_c = np.ascontiguousarray(w_arr_c).ravel()
                q_arr_c = np.asarray(qk, dtype=np.complex64)
                if not q_arr_c.flags.writeable:
                    q_arr_c = q_arr_c.copy()
                q_flat_c = np.ascontiguousarray(q_arr_c).ravel()
                n = w_flat_c.shape[0]
                wp_c = <float complex*>cnp.PyArray_DATA(<cnp.ndarray>w_flat_c)
                qp_c = <float complex*>cnp.PyArray_DATA(<cnp.ndarray>q_flat_c)
                with nogil:
                    c_coeff_val = _cdotc(&n, qp_c, &inc, wp_c, &inc)
                c_coeff_accum += c_coeff_val

            # Phase 2: w -= coeff * q via caxpy
            if c_coeff_accum == 0.0:
                continue
            c_alpha_neg = -c_coeff_accum
            for k in q_blocks:
                wk = w_blocks.get(k)
                if wk is None:
                    continue
                qk = q_blocks[k]
                if not wk.flags.writeable:
                    wk = wk.copy()
                    w_blocks[k] = wk
                w_flat_c2 = np.ascontiguousarray(np.asarray(wk, dtype=np.complex64)).ravel()
                q_flat_c2 = np.ascontiguousarray(np.asarray(qk, dtype=np.complex64)).ravel()
                n = w_flat_c2.shape[0]
                wp_c = <float complex*>cnp.PyArray_DATA(<cnp.ndarray>w_flat_c2)
                qp_c = <float complex*>cnp.PyArray_DATA(<cnp.ndarray>q_flat_c2)
                with nogil:
                    _caxpy(&n, &c_alpha_neg, qp_c, &inc, wp_c, &inc)

        else:
            # Fallback: numpy (full complex overlap)
            fb_coeff = 0.0 + 0.0j
            for k in q_blocks:
                wk = w_blocks.get(k)
                if wk is None:
                    continue
                fb_coeff += np.vdot(q_blocks[k], wk)
            if fb_coeff == 0.0:
                continue
            for k in q_blocks:
                wk = w_blocks.get(k)
                if wk is not None:
                    w_blocks[k] = wk - fb_coeff * q_blocks[k]


# ------------------------------------------------------------------ #
# cython_ba_sub_scaled_inplace: in-place w -= scalar * q via daxpy     #
# ------------------------------------------------------------------ #

def cython_ba_sub_scaled_inplace(dict w_blocks, dict q_blocks, double scalar):
    """w[k] -= scalar * q[k] for all shared keys. In-place, no allocation.

    Replaces ``ba_sub_scaled`` in the Lanczos hot loop, avoiding dict and
    array creation on every call (~87K calls per DMRG run).
    Dispatches to daxpy/zaxpy/caxpy based on dtype.
    """
    cdef double neg_d = -scalar
    cdef double complex neg_z = -scalar
    cdef float complex neg_c = <float complex>(-scalar)
    cdef int n, inc = 1
    cdef double[::1] w_flat_d, q_flat_d
    cdef double complex[::1] w_flat_z, q_flat_z
    cdef float complex[::1] w_flat_c, q_flat_c
    cdef int dtype_code = -1

    # Detect dtype from first shared block
    for k in q_blocks:
        wk = w_blocks.get(k)
        if wk is not None:
            dt = wk.dtype
            if dt == np.float64:
                dtype_code = 0
            elif dt == np.complex128:
                dtype_code = 1
            elif dt == np.complex64:
                dtype_code = 2
            break

    if dtype_code == 0:
        for k in q_blocks:
            wk = w_blocks.get(k)
            if wk is not None:
                qk = q_blocks[k]
                if not wk.flags.writeable:
                    wk = wk.copy()
                    w_blocks[k] = wk
                w_flat_d = wk.ravel()
                q_flat_d = np.ascontiguousarray(qk).ravel()
                n = w_flat_d.shape[0]
                with nogil:
                    _daxpy(&n, &neg_d, &q_flat_d[0], &inc, &w_flat_d[0], &inc)
    elif dtype_code == 1:
        for k in q_blocks:
            wk = w_blocks.get(k)
            if wk is not None:
                qk = q_blocks[k]
                if not wk.flags.writeable:
                    wk = wk.copy()
                    w_blocks[k] = wk
                w_flat_z = wk.ravel()
                q_arr = np.asarray(qk, dtype=np.complex128)
                if not q_arr.flags.writeable:
                    q_arr = q_arr.copy()
                q_flat_z = np.ascontiguousarray(q_arr).ravel()
                n = w_flat_z.shape[0]
                with nogil:
                    _zaxpy(&n, &neg_z, &q_flat_z[0], &inc, &w_flat_z[0], &inc)
    elif dtype_code == 2:
        for k in q_blocks:
            wk = w_blocks.get(k)
            if wk is not None:
                qk = q_blocks[k]
                if not wk.flags.writeable:
                    wk = wk.copy()
                    w_blocks[k] = wk
                w_flat_c = wk.ravel()
                q_arr = np.asarray(qk, dtype=np.complex64)
                if not q_arr.flags.writeable:
                    q_arr = q_arr.copy()
                q_flat_c = np.ascontiguousarray(q_arr).ravel()
                n = w_flat_c.shape[0]
                with nogil:
                    _caxpy(&n, &neg_c, &q_flat_c[0], &inc, &w_flat_c[0], &inc)
    else:
        for k in q_blocks:
            wk = w_blocks.get(k)
            if wk is not None:
                if not wk.flags.writeable:
                    wk = wk.copy()
                    w_blocks[k] = wk
                wk -= scalar * q_blocks[k]
