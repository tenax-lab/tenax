"""Lightweight numpy-backed block-sparse array for DMRG hot loops.

Provides a minimal BlockArray container and free functions for arithmetic,
avoiding JAX overhead in inner loops. All operations use plain numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tenax.core.index import TensorIndex


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


def ba_inner(a: BlockArray, b: BlockArray) -> float:
    """Frobenius inner product: sum of element-wise products over shared blocks."""
    total = 0.0
    for k in a.blocks:
        bk = b.blocks.get(k)
        if bk is not None:
            total += float(np.sum(a.blocks[k] * bk))
    return total


def ba_norm(ba: BlockArray) -> float:
    """Frobenius norm: sqrt(inner(ba, ba))."""
    return math.sqrt(ba_inner(ba, ba))


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
