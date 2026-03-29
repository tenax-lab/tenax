# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython BLAS kernel for block-sparse tensor contractions."""

import numpy as np
from scipy.linalg import blas as scipy_blas


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
