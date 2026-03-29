# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython BLAS kernel for block-sparse tensor contractions."""

import numpy as np
from scipy.linalg import blas as scipy_blas

cimport numpy as cnp
from scipy.linalg.cython_blas cimport dgemm as _dgemm
from scipy.linalg.cython_blas cimport sgemm as _sgemm
from scipy.linalg.cython_blas cimport zgemm as _zgemm


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
