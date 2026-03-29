"""BLAS execution plan builder for multi-tensor contractions.

Converts a multi-tensor einsum subscript + shapes into a list of GEMM steps
following the TTGT pattern (Transpose-Transpose-GEMM-Transpose).

The plan is built once per (subscripts, shapes) key and cached. Execution
uses ``numpy.matmul`` (or a future Cython BLAS kernel) per step.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import opt_einsum

# ------------------------------------------------------------------ #
# Data types                                                           #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class GemmStep:
    """One pairwise matrix multiply in a multi-tensor contraction plan.

    The caller is expected to:
    1. Transpose left_buf by ``left_perm`` (skip if identity ``()``).
    2. Transpose right_buf by ``right_perm`` (skip if identity ``()``).
    3. Reshape left to ``(m, k)`` and right to ``(k, n)``.
    4. Compute ``out = left @ right`` via BLAS GEMM (or use trans_a/trans_b).
    5. Reshape ``out`` to ``out_shape``.
    """

    left_idx: int  # buffer index to read from
    right_idx: int  # buffer index to read from
    out_idx: int  # buffer index to write to
    trans_a: bool  # BLAS transpose flag
    trans_b: bool  # BLAS transpose flag
    m: int  # rows of output
    n: int  # cols of output
    k: int  # contracted dimension
    left_perm: tuple[int, ...]  # transpose permutation before reshape; () = identity
    right_perm: tuple[int, ...]  # transpose permutation before reshape; () = identity
    out_shape: tuple[int, ...]  # reshape GEMM result back to N-d


@dataclass(frozen=True)
class BlasExecPlan:
    """Complete execution plan for a multi-tensor einsum via pairwise GEMMs."""

    steps: tuple[GemmStep, ...]
    n_buffers: int  # total buffer slots (inputs + intermediates)
    n_inputs: int  # first n_inputs buffers are inputs
    output_perm: tuple[int, ...]  # final transpose; () = identity

    def execute_numpy(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Execute the plan using NumPy, returning the final result.

        This is a pure-Python reference implementation and non-Cython fallback.
        For each step: transpose operands, reshape to 2D, matmul, reshape back.
        """
        assert len(arrays) == self.n_inputs
        buffers: list[np.ndarray | None] = list(arrays) + [None] * (
            self.n_buffers - self.n_inputs
        )

        for step in self.steps:
            left = buffers[step.left_idx]
            right = buffers[step.right_idx]
            assert left is not None and right is not None

            # 1. Transpose if needed
            if step.left_perm:
                left = np.transpose(left, step.left_perm)
            if step.right_perm:
                right = np.transpose(right, step.right_perm)

            # 2. Reshape to 2D matrices
            left_2d = left.reshape(step.m, step.k)
            right_2d = right.reshape(step.k, step.n)

            # 3. GEMM
            out_2d = left_2d @ right_2d

            # 4. Reshape back to N-d
            buffers[step.out_idx] = out_2d.reshape(step.out_shape)

        # The last step's output buffer holds the result
        result = buffers[self.steps[-1].out_idx]
        assert result is not None

        # 5. Final output permutation if needed
        if self.output_perm:
            result = np.transpose(result, self.output_perm)

        return result


@dataclass
class KernelData:
    """Pre-processed data for the Cython BLAS kernel.

    All block arrays are pre-transposed and reshaped to their
    GEMM-ready 2D C-contiguous layout.
    """

    plan: BlasExecPlan
    combo_input_blocks: list[list[np.ndarray]]  # [combo_idx][buf_idx] -> 2D array
    combo_output_idx: np.ndarray  # int32 array, length n_combos
    output_buffers: list[np.ndarray]  # [slot_idx] -> zeroed array with out_shape
    output_keys: list[tuple[int, ...]]  # [slot_idx] -> charge key
    work_buffers: list[np.ndarray]  # intermediate buffers, reused per combo
    dtype: np.dtype


def prepare_kernel_data(
    plan: BlasExecPlan,
    combos: list[tuple[list[tuple[int, ...]], tuple[int, ...]]],
    np_blocks: list[dict[tuple[int, ...], np.ndarray]],
) -> KernelData:
    """Pre-transpose all blocks and pack into flat structures for the C kernel.

    Parameters
    ----------
    plan : BlasExecPlan
        Execution plan from :func:`build_blas_plan`.
    combos : list of (combo_keys, output_key)
        Each element is ``(combo_keys, output_key)`` where *combo_keys* is a
        list of block keys (one per input tensor) and *output_key* is the
        charge-sector key for the output.
    np_blocks : list of dict
        One dict per input tensor, mapping block keys to N-d numpy arrays.

    Returns
    -------
    KernelData
        Pre-transposed data ready for the Cython kernel.
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

    output_keys: list[tuple[int, ...]] = [None] * len(output_key_to_idx)  # type: ignore
    for key, idx in output_key_to_idx.items():
        output_keys[idx] = key

    # Pre-allocate output buffers (zeroed) with the last step's out_shape
    last_step = steps[-1]
    output_buffers = [
        np.zeros(last_step.out_shape, dtype=dtype)
        for _ in range(len(output_key_to_idx))
    ]

    # Pre-allocate work buffers for intermediate steps
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


# ------------------------------------------------------------------ #
# Plan builder                                                         #
# ------------------------------------------------------------------ #


def _parse_subscripts(subscripts: str) -> tuple[list[str], str]:
    """Parse 'ab,cd,ef->gh' into (['ab', 'cd', 'ef'], 'gh')."""
    lhs, rhs = subscripts.split("->")
    inputs = lhs.split(",")
    return inputs, rhs


def _identity_perm(ndim: int) -> tuple[int, ...]:
    """Return the identity permutation tuple for *ndim* dimensions."""
    return tuple(range(ndim))


def build_blas_plan(subscripts: str, shapes: list[tuple[int, ...]]) -> BlasExecPlan:
    """Build a BLAS execution plan for the given einsum subscripts and shapes.

    Uses ``opt_einsum.contract_path`` to determine the pairwise contraction
    order, then for each pair computes the TTGT parameters.

    Parameters
    ----------
    subscripts : str
        Einsum subscripts, e.g. ``"abc,apd,bpxe,def->cxf"``.
    shapes : list of tuple[int, ...]
        Shapes of the input tensors, one per input operand.

    Returns
    -------
    BlasExecPlan
        A frozen execution plan.
    """
    input_subs, output_sub = _parse_subscripts(subscripts)
    n_inputs = len(input_subs)
    assert len(shapes) == n_inputs

    # Get optimal pairwise contraction path from opt_einsum.
    # Use broadcast_to to create zero-copy views — avoids allocating
    # large temporaries for big bond dimensions.
    _zero = np.float64(0)
    dummy_arrays = [np.broadcast_to(_zero, s) for s in shapes]
    path, _ = opt_einsum.contract_path(subscripts, *dummy_arrays)

    # Track "active operands" -- list of (subscript_chars, shape, buffer_idx).
    # Initially these are the inputs.
    active: list[tuple[str, tuple[int, ...], int]] = []
    for i, (sub, shape) in enumerate(zip(input_subs, shapes)):
        active.append((sub, shape, i))

    next_buffer = n_inputs  # next available buffer slot
    steps: list[GemmStep] = []

    for pair in path:
        # opt_einsum gives pairs of indices into the *current* active list.
        # We must pop in descending order to keep indices valid.
        idx_hi, idx_lo = sorted(pair, reverse=True)
        sub_hi, shape_hi, buf_hi = active.pop(idx_hi)
        sub_lo, shape_lo, buf_lo = active.pop(idx_lo)

        # We designate the operand popped second (lower index) as "left"
        # and the one popped first (higher index) as "right".
        sub_a, shape_a, buf_a = sub_lo, shape_lo, buf_lo
        sub_b, shape_b, buf_b = sub_hi, shape_hi, buf_hi

        # Determine which indices are contracted vs free.
        # Contracted indices appear in both operands.
        set_a = set(sub_a)
        set_b = set(sub_b)
        contracted_chars = set_a & set_b

        # Free indices: those in only one operand.
        free_a_chars = [c for c in sub_a if c not in contracted_chars]
        free_b_chars = [c for c in sub_b if c not in contracted_chars]

        # Build the result subscript: free_a + free_b
        result_chars = free_a_chars + free_b_chars
        result_sub = "".join(result_chars)

        # Build shape maps: char -> dimension size
        char_to_dim: dict[str, int] = {}
        for c, d in zip(sub_a, shape_a):
            char_to_dim[c] = d
        for c, d in zip(sub_b, shape_b):
            char_to_dim[c] = d

        # Compute M, N, K
        m = 1
        for c in free_a_chars:
            m *= char_to_dim[c]
        n = 1
        for c in free_b_chars:
            n *= char_to_dim[c]
        k = 1
        for c in contracted_chars:
            k *= char_to_dim[c]

        # Determine permutations to arrange operands for GEMM.
        # Left operand needs shape (free_a..., contracted...) -> reshape to (M, K)
        # Right operand needs shape (contracted..., free_b...) -> reshape to (K, N)

        # Order contracted indices consistently between left and right.
        # Use the order they appear in sub_a (arbitrary but consistent).
        contracted_ordered = [c for c in sub_a if c in contracted_chars]

        # Target order for left: free_a indices, then contracted indices
        target_left = free_a_chars + contracted_ordered
        left_perm = tuple(sub_a.index(c) for c in target_left)
        if left_perm == _identity_perm(len(sub_a)):
            left_perm = ()

        # Target order for right: contracted indices (same order as left), then free_b
        target_right = contracted_ordered + free_b_chars
        right_perm = tuple(sub_b.index(c) for c in target_right)
        if right_perm == _identity_perm(len(sub_b)):
            right_perm = ()

        # Output shape (in result_chars order: free_a + free_b)
        out_shape = tuple(char_to_dim[c] for c in result_chars)

        out_buf = next_buffer
        next_buffer += 1

        steps.append(
            GemmStep(
                left_idx=buf_a,
                right_idx=buf_b,
                out_idx=out_buf,
                trans_a=False,
                trans_b=False,
                m=m,
                n=n,
                k=k,
                left_perm=left_perm,
                right_perm=right_perm,
                out_shape=out_shape,
            )
        )

        # Add the result as a new active operand.
        active.append((result_sub, out_shape, out_buf))

    # After all steps, the single remaining active operand is the result.
    assert len(active) == 1
    final_sub, _final_shape, _final_buf = active[0]

    # Determine if a final permutation is needed to match the requested output.
    if final_sub == output_sub:
        output_perm: tuple[int, ...] = ()
    else:
        output_perm = tuple(final_sub.index(c) for c in output_sub)

    return BlasExecPlan(
        steps=tuple(steps),
        n_buffers=next_buffer,
        n_inputs=n_inputs,
        output_perm=output_perm,
    )


# ------------------------------------------------------------------ #
# Caching                                                              #
# ------------------------------------------------------------------ #


@functools.lru_cache(maxsize=256)
def get_cached_blas_plan(
    subscripts: str,
    shapes: tuple[tuple[int, ...], ...],
) -> BlasExecPlan:
    """Return a cached BLAS plan for the given subscripts and shapes.

    Parameters
    ----------
    subscripts : str
        Einsum subscripts string.
    shapes : tuple of tuple[int, ...]
        Shapes of input tensors. Must be a tuple (not list) for hashability.
    """
    return build_blas_plan(subscripts, list(shapes))
