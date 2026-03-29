"""cuTENSOR block-sparse backend for GPU-accelerated symmetric tensor contractions.

Maps Tenax's SymmetricTensor block structure to cuTENSOR's block-sparse
tensor descriptors for native GPU block-sparse contractions.

Requires: ``pip install cuquantum-cu13`` (includes nvmath with cutensor bindings).

Usage::

    # Automatic: set environment variable
    export TENAX_USE_CUTENSOR_BLOCKSPARSE=1

    # Then all SymmetricTensor contractions on GPU use cuTENSOR block-sparse.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# Lazy state
_available: bool | None = None
_handle: int | None = None


def is_available() -> bool:
    """Check if cuTENSOR block-sparse is available."""
    global _available

    if _available is not None:
        return _available

    try:
        from nvmath.bindings import cutensor as ct

        # Check for CUDA device
        devices = jax.devices()
        has_gpu = any(d.platform == "gpu" for d in devices)
        if not has_gpu:
            _available = False
            return False

        # Verify block-sparse API exists
        _ = ct.create_block_sparse_tensor_descriptor
        _ = ct.block_sparse_contract
        _available = True
        logger.info("cuTENSOR block-sparse backend available")
        return True
    except (ImportError, AttributeError, RuntimeError):
        _available = False
        return False


def _get_handle() -> int:
    """Get or create the cuTENSOR library handle."""
    global _handle
    if _handle is None:
        from nvmath.bindings import cutensor as ct

        _handle = ct.create()
    return _handle


def _build_section_map(
    charges: np.ndarray,
) -> tuple[list[int], list[int], dict[int, int]]:
    """Map charges to cuTENSOR sections.

    Args:
        charges: Array of charge values for one tensor leg.

    Returns:
        (unique_charges_sorted, section_sizes, charge_to_section_idx)
    """
    unique = sorted(set(int(c) for c in charges))
    sizes = [int(np.sum(charges == q)) for q in unique]
    charge_to_idx = {q: i for i, q in enumerate(unique)}
    return unique, sizes, charge_to_idx


def contract_blocksparse(
    tensor_a,  # SymmetricTensor
    tensor_b,  # SymmetricTensor
    subscripts: str,
    output_indices: tuple,
    plan_cache: dict | None = None,
):
    """Contract two SymmetricTensors using cuTENSOR block-sparse.

    Args:
        tensor_a:       First SymmetricTensor.
        tensor_b:       Second SymmetricTensor.
        subscripts:     Einsum subscript (e.g., "ipj,jqk->ipqk").
        output_indices: TensorIndex metadata for output legs.
        plan_cache:     Optional dict for caching cuTENSOR plans.

    Returns:
        SymmetricTensor with the contraction result.
    """
    import cupy as cp
    from nvmath import CudaDataType
    from nvmath.bindings import cutensor as ct
    from nvmath.bindings.cutensor import Operator

    from tenax.core.tensor import SymmetricTensor

    handle = _get_handle()

    # Ensure compute descriptors are loaded
    ct._load_cutensor_compute_descriptors()
    OP_ID = int(Operator.OP_IDENTITY)
    WS_LIMIT = 256 * 1024 * 1024  # 256 MB

    dtype_map = {
        jnp.float64: int(CudaDataType.CUDA_R_64F),
        jnp.float32: int(CudaDataType.CUDA_R_32F),
        jnp.complex128: int(CudaDataType.CUDA_C_64F),
        jnp.complex64: int(CudaDataType.CUDA_C_32F),
    }
    ct_dtype = dtype_map.get(tensor_a.dtype, int(CudaDataType.CUDA_R_64F))
    # Element size in bytes for pointer arithmetic
    _dtype_sizes = {
        jnp.float32: 4,
        jnp.float64: 8,
        jnp.complex64: 8,
        jnp.complex128: 16,
    }
    element_size = _dtype_sizes.get(tensor_a.dtype, np.dtype(tensor_a.dtype).itemsize)

    # Parse subscripts
    input_part, output_part = subscripts.split("->")
    input_subs = input_part.split(",")
    sub_a, sub_b = input_subs[0], input_subs[1]

    def _make_descriptor(tensor, subs):
        """Build cuTENSOR block-sparse descriptor from SymmetricTensor."""
        num_modes = len(tensor.indices)

        # Build section maps per mode
        section_maps = []
        all_extents = []
        num_sections = []
        for idx in tensor.indices:
            charges = np.asarray(idx.charges, dtype=np.int32)
            unique, sizes, c2i = _build_section_map(charges)
            section_maps.append((unique, sizes, c2i))
            num_sections.append(len(unique))
            all_extents.extend(sizes)

        # Map block keys to section coordinates (column-major for cuTENSOR)
        n_blocks = tensor.n_blocks
        coords = []
        for key in tensor._block_keys:
            for mode_idx, q in enumerate(key):
                _, _, c2i = section_maps[mode_idx]
                coords.append(c2i[int(q)])

        # Compute row-major strides per block.
        # cuTENSOR default (stride=0) is column-major, but JAX uses row-major.
        # Stride array: num_modes × num_non_zero_blocks (block-first).
        strides = []
        for i, key in enumerate(tensor._block_keys):
            shape = tensor._block_shapes[i]
            # Row-major strides: last dimension has stride 1
            block_strides = []
            stride = 1
            for d in reversed(shape):
                block_strides.append(stride)
                stride *= d
            block_strides.reverse()
            strides.extend(block_strides)

        # Create descriptor
        desc = ct.create_block_sparse_tensor_descriptor(
            handle,
            np.uint32(num_modes),
            np.uint64(n_blocks),
            np.array(num_sections, dtype=np.uint32),
            np.array(all_extents, dtype=np.int64),
            np.array(coords, dtype=np.int32),
            np.array(strides, dtype=np.int64),
            ct_dtype,
        )

        # Build pointer array: one GPU pointer per block
        cp_data = cp.from_dlpack(tensor._data)
        base_ptr = cp_data.data.ptr
        ptrs = []
        for i in range(n_blocks):
            offset = tensor._block_offsets[i]
            ptrs.append(base_ptr + offset * element_size)

        # Mode labels as ints
        modes = [ord(c) for c in subs]

        return desc, ptrs, modes, section_maps

    desc_a, ptrs_a, modes_a, maps_a = _make_descriptor(tensor_a, sub_a)
    desc_b, ptrs_b, modes_b, maps_b = _make_descriptor(tensor_b, sub_b)

    # Build output descriptor
    # Determine output block structure from input block structures
    out_section_maps = []
    out_num_sections = []
    out_extents = []
    for i, c in enumerate(output_part):
        # Find which input tensor/mode this output char came from
        found = False
        for tensor, subs, maps in [
            (tensor_a, sub_a, maps_a),
            (tensor_b, sub_b, maps_b),
        ]:
            if c in subs:
                mode_idx = subs.index(c)
                out_section_maps.append(maps[mode_idx])
                unique, sizes, _ = maps[mode_idx]
                out_num_sections.append(len(unique))
                out_extents.extend(sizes)
                found = True
                break
        if not found:
            raise ValueError(f"Output char '{c}' not found in inputs")

    # Determine output blocks: iterate all input block combinations
    # and collect valid output keys
    output_block_keys = set()
    contracted_chars = set(sub_a) & set(sub_b)

    for key_a in tensor_a._block_keys:
        charge_map_a = {c: int(q) for c, q in zip(sub_a, key_a)}
        for key_b in tensor_b._block_keys:
            charge_map_b = {c: int(q) for c, q in zip(sub_b, key_b)}
            # Check contracted indices match
            compatible = True
            for cc in contracted_chars:
                if charge_map_a.get(cc) != charge_map_b.get(cc):
                    compatible = False
                    break
            if compatible:
                merged = {**charge_map_a, **charge_map_b}
                out_key = tuple(merged[c] for c in output_part)
                output_block_keys.add(out_key)

    output_block_keys = sorted(output_block_keys)
    n_out_blocks = len(output_block_keys)

    if n_out_blocks == 0:
        # No compatible blocks — return empty tensor
        obj = object.__new__(SymmetricTensor)
        obj._indices = output_indices
        obj._init_flat_buffer({})
        return obj

    # Output coordinates (column-major)
    out_coords = []
    for key in output_block_keys:
        for mode_idx, q in enumerate(key):
            _, _, c2i = out_section_maps[mode_idx]
            out_coords.append(c2i[int(q)])

    # Output strides (row-major)
    out_strides = []
    for key in output_block_keys:
        shape = []
        for mode_idx, q in enumerate(key):
            _, sizes, c2i = out_section_maps[mode_idx]
            shape.append(sizes[c2i[int(q)]])
        stride = 1
        block_strides = []
        for d in reversed(shape):
            block_strides.append(stride)
            stride *= d
        block_strides.reverse()
        out_strides.extend(block_strides)

    desc_d = ct.create_block_sparse_tensor_descriptor(
        handle,
        np.uint32(len(output_part)),
        np.uint64(n_out_blocks),
        np.array(out_num_sections, dtype=np.uint32),
        np.array(out_extents, dtype=np.int64),
        np.array(out_coords, dtype=np.int32),
        np.array(out_strides, dtype=np.int64),
        ct_dtype,
    )
    modes_d = [ord(c) for c in output_part]

    # Allocate output blocks on GPU
    out_blocks_cp = []
    out_block_shapes = []
    for key in output_block_keys:
        shape = []
        for mode_idx, q in enumerate(key):
            _, sizes, c2i = out_section_maps[mode_idx]
            shape.append(sizes[c2i[int(q)]])
        shape = tuple(shape)
        out_block_shapes.append(shape)
        block = cp.zeros(shape, dtype=tensor_a._data.dtype)
        out_blocks_cp.append(block)

    ptrs_d = [b.data.ptr for b in out_blocks_cp]

    # Create contraction: D = alpha * A @ B + beta * C
    compute_desc = ct._COMPUTE_DESC_64F if element_size == 8 else ct._COMPUTE_DESC_32F

    op_desc = ct.create_block_sparse_contraction(
        handle,
        desc_a,
        modes_a,
        OP_ID,
        desc_b,
        modes_b,
        OP_ID,
        desc_d,
        modes_d,
        OP_ID,  # C = D (beta=0)
        desc_d,
        modes_d,
        compute_desc,
    )

    # Create plan
    pref = ct.create_plan_preference(handle, -1, 0)
    ws_size = ct.estimate_workspace_size(handle, op_desc, pref, 0)
    plan = ct.create_plan(handle, op_desc, pref, WS_LIMIT)

    # Workspace
    workspace = cp.zeros(max(ws_size, 256), dtype=np.uint8)

    # Execute
    alpha = np.array(1.0, dtype=np.float64 if element_size == 8 else np.float32)
    beta = np.array(0.0, dtype=alpha.dtype)

    ct.block_sparse_contract(
        handle,
        plan,
        alpha.ctypes.data,
        ptrs_a,
        ptrs_b,
        beta.ctypes.data,
        ptrs_d,  # C = D for beta=0
        ptrs_d,
        workspace.data.ptr,
        ws_size,
        0,  # default stream
    )
    cp.cuda.Stream.null.synchronize()

    # Build output SymmetricTensor
    output_blocks = {}
    for key, block_cp in zip(output_block_keys, out_blocks_cp):
        output_blocks[key] = jnp.from_dlpack(block_cp)

    obj = object.__new__(SymmetricTensor)
    obj._indices = output_indices
    obj._init_flat_buffer(output_blocks)

    # Cleanup cuTENSOR objects
    ct.destroy_block_sparse_tensor_descriptor(desc_a)
    ct.destroy_block_sparse_tensor_descriptor(desc_b)
    ct.destroy_block_sparse_tensor_descriptor(desc_d)

    return obj
