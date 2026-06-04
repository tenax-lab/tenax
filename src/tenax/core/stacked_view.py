"""Stacked working representation over the flat _data buffer (#566 P1a).

Blocks are grouped by identical shape; each group is one array
(n_blocks_in_group, *block_shape) gathered from _data and reshaped. Grouping
and gather indices are STATIC (from block metadata), so only the gather/reshape
touches the tracer. For even-D FermionParity (one shape, contiguous) the gather
degenerates to a contiguous slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class StackGroup:
    keys: tuple  # block keys in this group, in sorted-key order
    array: jax.Array  # (n_blocks, *block_shape)


@dataclass(frozen=True)
class StackedView:
    groups: dict  # block_shape (tuple[int,...]) -> StackGroup
    indices: tuple  # source tensor's TensorIndex tuple


def _group_layout(block_keys, block_shapes, block_offsets):
    """STATIC: shape -> (keys_tuple, flat_gather_index_array of shape (n, size))."""
    by_shape: dict = {}
    for i, shape in enumerate(block_shapes):
        size = int(np.prod(shape)) if shape else 1
        off = block_offsets[i]
        flat_idx = np.arange(off, off + size, dtype=np.int64)
        rec = by_shape.setdefault(shape, ([], []))
        rec[0].append(block_keys[i])
        rec[1].append(flat_idx)
    layout = {}
    for shape, (keys, idx_lists) in by_shape.items():
        layout[shape] = (tuple(keys), np.stack(idx_lists, axis=0))  # (n, size)
    return layout


def build_stacked(
    data, block_keys, block_shapes, block_offsets, indices
) -> StackedView:
    layout = _group_layout(block_keys, block_shapes, block_offsets)
    groups = {}
    for shape, (keys, gather) in layout.items():
        n = gather.shape[0]
        # raw numpy index -> XLA compile-time-constant gather (one op per shape group)
        flat = data[gather.reshape(-1)]  # one gather per group
        groups[shape] = StackGroup(keys=keys, array=flat.reshape((n, *shape)))
    return StackedView(groups=groups, indices=indices)


def scatter_stacked(view, block_keys, block_shapes, block_offsets, total_size, dtype):
    """Inverse: write groups back into one flat buffer in canonical sorted-key layout."""
    layout = _group_layout(block_keys, block_shapes, block_offsets)
    data = jnp.zeros(total_size, dtype=dtype)
    for shape, (keys, gather) in layout.items():
        grp = view.groups[shape]
        # identity permutation in the round-trip path; O(n^2) only matters for cross-tensor scatter at large block counts
        order = [
            grp.keys.index(k) for k in keys
        ]  # reorder grp rows to canonical key order
        rows = grp.array[np.asarray(order)]
        data = data.at[gather.reshape(-1)].set(rows.reshape(-1))
    return data
