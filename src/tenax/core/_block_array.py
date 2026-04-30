"""Lightweight numpy-backed block-sparse array for DMRG hot loops.

Provides a minimal BlockArray container and free functions for arithmetic,
avoiding JAX overhead in inner loops. All operations use plain numpy.

When the Cython BLAS extension is available, ba_inner / ba_norm use BLAS
ddot (no Python-layer numpy dispatch), and ba_axpy / ba_scale_inplace
provide allocation-free in-place BLAS operations for the Lanczos loop.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from tenax.core.index import TensorIndex

# Cython BLAS handles are loaded lazily on first use to avoid a static
# import edge from ``tenax.core`` into ``tenax.contraction`` (which would
# invert the package layering).
_HAS_CYTHON_BA: bool | None = None
_c_axpy: Any = None
_c_inner: Any = None
_c_scale_inplace: Any = None


def _ensure_cython_ba_loaded() -> bool:
    """Lazily resolve the Cython BlockArray helpers; returns True if available."""
    global _HAS_CYTHON_BA, _c_axpy, _c_inner, _c_scale_inplace
    if _HAS_CYTHON_BA is not None:
        return _HAS_CYTHON_BA
    if os.environ.get("TENAX_DISABLE_CYTHON_BLAS", "0") == "1":
        _HAS_CYTHON_BA = False
        return False
    try:
        from tenax.contraction._cython_blas import (
            cython_ba_axpy,
            cython_ba_inner,
            cython_ba_scale_inplace,
        )
    except ImportError:
        _HAS_CYTHON_BA = False
        return False
    _c_axpy = cython_ba_axpy
    _c_inner = cython_ba_inner
    _c_scale_inplace = cython_ba_scale_inplace
    _HAS_CYTHON_BA = True
    return True


@dataclass
class BlockArray:
    """Block-sparse array backed by numpy ndarrays.

    Attributes:
        blocks:  Dict mapping block key (tuple of charges) to numpy array.
        indices: Tuple of TensorIndex objects, one per leg.
    """

    blocks: dict[tuple[int, ...], np.ndarray]
    indices: tuple[TensorIndex, ...]

    def labels(self) -> tuple[str, ...]:
        """Return label strings for each leg (compatible with Tensor API)."""
        return tuple(idx.label for idx in self.indices)


def ba_scale(ba: BlockArray, scalar: float) -> BlockArray:
    """Multiply all blocks by a scalar."""
    return BlockArray(
        blocks={k: v * scalar for k, v in ba.blocks.items()},
        indices=ba.indices,
    )


def ba_add(a: BlockArray, b: BlockArray) -> BlockArray:
    """Add two BlockArrays (union of keys)."""
    result: dict[tuple[int, ...], np.ndarray] = {}
    all_keys = set(a.blocks) | set(b.blocks)
    for k in all_keys:
        ak = a.blocks.get(k)
        bk = b.blocks.get(k)
        if ak is not None and bk is not None:
            result[k] = ak + bk
        elif ak is not None:
            result[k] = ak.copy()
        else:
            assert bk is not None
            result[k] = bk.copy()
    return BlockArray(blocks=result, indices=a.indices)


def ba_sub(a: BlockArray, b: BlockArray) -> BlockArray:
    """Subtract BlockArray b from a."""
    result: dict[tuple[int, ...], np.ndarray] = {}
    all_keys = set(a.blocks) | set(b.blocks)
    for k in all_keys:
        ak = a.blocks.get(k)
        bk = b.blocks.get(k)
        if ak is not None and bk is not None:
            result[k] = ak - bk
        elif ak is not None:
            result[k] = ak.copy()
        else:
            assert bk is not None
            result[k] = -bk
    return BlockArray(blocks=result, indices=a.indices)


def ba_sub_scaled(a: BlockArray, b: BlockArray, scalar: float) -> BlockArray:
    """Compute a - scalar * b without creating an intermediate dict.

    Fuses ba_sub + ba_scale into one pass: ``result[k] = a[k] - scalar * b[k]``.
    """
    result: dict[tuple[int, ...], np.ndarray] = {}
    for k, ak in a.blocks.items():
        bk = b.blocks.get(k)
        if bk is not None:
            result[k] = ak - scalar * bk
        else:
            result[k] = ak.copy()
    for k in b.blocks:
        if k not in a.blocks:
            result[k] = -(scalar * b.blocks[k])
    return BlockArray(blocks=result, indices=a.indices)


def ba_inner(a: BlockArray, b: BlockArray) -> float | complex:
    """Hermitian inner product: sum conj(a)*b over shared blocks.

    Uses ``np.vdot`` which conjugates the first argument, giving the correct
    Hermitian inner product for complex data and equivalent behavior for real.

    Returns float for real inputs, complex for complex inputs.
    """
    if a.blocks and _ensure_cython_ba_loaded():
        return _c_inner(a.blocks, b.blocks)
    total: float | complex = 0.0
    is_complex = False
    for k in a.blocks:
        bk = b.blocks.get(k)
        if bk is not None:
            if not is_complex and np.iscomplexobj(a.blocks[k]):
                is_complex = True
                total = complex(total)
            total += np.vdot(a.blocks[k], bk)
    return total


def ba_axpy(x: BlockArray, y: BlockArray, alpha: float) -> None:
    """y += alpha * x (in-place).  Uses BLAS daxpy when available.

    For shared keys only; keys present in x but not y are ignored.
    This is the pattern needed by Lanczos reorthogonalization.
    """
    if _ensure_cython_ba_loaded():
        _c_axpy(x.blocks, y.blocks, alpha)
    else:
        for k in x.blocks:
            yk = y.blocks.get(k)
            if yk is not None:
                yk += alpha * x.blocks[k]


def ba_scale_inplace(ba: BlockArray, scalar: float) -> None:
    """Scale all blocks in-place.  Uses BLAS dscal when available."""
    if _ensure_cython_ba_loaded():
        _c_scale_inplace(ba.blocks, scalar)
    else:
        for v in ba.blocks.values():
            v *= scalar


def ba_norm(ba: BlockArray) -> float:
    """Frobenius norm: sqrt(inner(ba, ba))."""
    # <v|v> is always real; extract .real for complex return values.
    return math.sqrt(float(ba_inner(ba, ba).real))


def ba_conj(ba: BlockArray) -> BlockArray:
    """Conjugate all blocks."""
    return BlockArray(
        blocks={k: np.conj(v) for k, v in ba.blocks.items()},
        indices=ba.indices,
    )


def ba_bar(ba: BlockArray) -> BlockArray:
    """Conjugate blocks and flip flow directions (bra version).

    Mirrors ``SymmetricTensor.bar()`` for the numpy fast path.
    """
    new_indices = tuple(idx.flip_flow() for idx in ba.indices)
    return BlockArray(
        blocks={k: np.conj(v) for k, v in ba.blocks.items()},
        indices=new_indices,
    )


def symmetric_to_ba(t) -> BlockArray:
    """Extract blocks and indices from a SymmetricTensor into a BlockArray.

    Converts JAX arrays to numpy arrays for zero-JAX-overhead arithmetic.
    """
    blocks = {k: np.asarray(v) for k, v in t.blocks.items()}
    return BlockArray(blocks=blocks, indices=t.indices)


def ba_to_symmetric(ba: BlockArray):
    """Reconstruct a SymmetricTensor from a BlockArray.

    Uses numpy concatenation -- no JAX overhead.  The SymmetricTensor
    stores numpy ``_data`` internally, which works for the DMRG numpy path
    where blocks are accessed via the ``.blocks`` property.
    """
    from tenax.core.tensor import SymmetricTensor

    sorted_keys = sorted(ba.blocks.keys())
    obj = object.__new__(SymmetricTensor)
    obj._indices = ba.indices
    if sorted_keys:
        flat_parts = [ba.blocks[k].ravel() for k in sorted_keys]
        obj._data = np.concatenate(flat_parts)
    else:
        obj._data = np.zeros(0, dtype=np.float64)
    obj._block_keys = tuple(sorted_keys)
    obj._block_shapes = tuple(ba.blocks[k].shape for k in sorted_keys)
    offsets = []
    offset = 0
    for k in sorted_keys:
        offsets.append(offset)
        offset += ba.blocks[k].size
    obj._block_offsets = tuple(offsets)
    return obj
