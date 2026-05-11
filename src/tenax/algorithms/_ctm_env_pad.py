"""Zero-pad the χ axes of a CTMTensorEnv for the variPEPS §2.8.2 auto-bump.

Used by the auto-χ_E bump protocol to grow an already-converged environment
from ``chi_old`` to ``chi_new`` before re-running CTM.  Both the dense path
(``DenseTensor`` env) and the block-sparse path (``SymmetricTensor`` env)
are supported; the symmetric path tiles existing χ-leg charges via
``_derive_charges`` (the same allocator used by the symmetric projectors,
keeping the χ-bond charge pattern consistent across CTM sweeps).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.algorithms._ctm_utils import _derive_charges
from tenax.core.index import TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor


def _pad_chi_index(idx: TensorIndex, chi_new: int) -> TensorIndex:
    """Return a new TensorIndex with dim grown to ``chi_new``.

    The original charges are preserved in the first ``chi_old`` slots;
    the new slots are filled by ``_derive_charges`` (a tile of the
    existing pattern), so each charge sector grows by zero or more slots
    while preserving every existing sector's position.  For DenseTensor
    indices (single trivial charge 0), this is equivalent to padding with
    zeros.
    """
    old_charges = np.asarray(idx.charges, dtype=np.int32)
    new_charges = _derive_charges(old_charges, chi_new)
    return TensorIndex.from_charges(
        idx.symmetry, new_charges, idx.flow, label=idx.label
    )


def _pad_corner(t: DenseTensor, chi_new: int) -> DenseTensor:
    """Zero-pad both χ axes of a corner tensor from chi_old to chi_new."""
    chi_old = t._data.shape[0]
    pad = chi_new - chi_old
    arr = jnp.pad(t._data, [(0, pad), (0, pad)])
    idx0 = _pad_chi_index(t._indices[0], chi_new)
    idx1 = _pad_chi_index(t._indices[1], chi_new)
    return DenseTensor(arr, (idx0, idx1))


def _pad_edge(t: DenseTensor, chi_new: int) -> DenseTensor:
    """Zero-pad the two χ axes (0 and 2) of an edge tensor; axis 1 (D²) unchanged."""
    chi_old = t._data.shape[0]
    pad = chi_new - chi_old
    arr = jnp.pad(t._data, [(0, pad), (0, 0), (0, pad)])
    idx0 = _pad_chi_index(t._indices[0], chi_new)
    idx1 = t._indices[1]  # D² leg — untouched
    idx2 = _pad_chi_index(t._indices[2], chi_new)
    return DenseTensor(arr, (idx0, idx1, idx2))


def _sector_dim(idx: TensorIndex, charge: int) -> int:
    """Number of slots in ``idx`` with the given charge."""
    return int(np.sum(np.asarray(idx.charges) == charge))


def _pad_symmetric_block_along_axes(
    block: jax.Array,
    old_shape: tuple[int, ...],
    new_shape: tuple[int, ...],
    chi_axes: tuple[int, ...],
) -> jax.Array:
    """Zero-pad ``block`` from ``old_shape`` to ``new_shape`` along ``chi_axes``."""
    pad_widths = [(0, 0)] * block.ndim
    for ax in chi_axes:
        pad_widths[ax] = (0, new_shape[ax] - old_shape[ax])
    return jnp.pad(block, pad_widths)


def _pad_symmetric_tensor_along_chi(
    t: SymmetricTensor,
    new_indices: tuple[TensorIndex, ...],
    chi_axes: tuple[int, ...],
) -> SymmetricTensor:
    """Rebuild ``t`` with new indices, zero-padding existing blocks along ``chi_axes``.

    Block keys present in the old tensor are preserved (their data is
    zero-padded along the chi axes).  Block keys valid under the new
    indices but missing from the old tensor are not materialised — the
    underlying flat-buffer representation only stores allocated blocks,
    so absent sectors stay implicitly zero.

    Because ``_derive_charges(c_old, chi_new)`` reproduces ``c_old`` as a
    strict prefix when ``chi_new >= chi_old``, every old block key remains
    valid under the new indices.
    """
    new_blocks: dict[tuple[int, ...], jax.Array] = {}
    for key, block in t.blocks.items():
        new_shape = tuple(
            _sector_dim(new_indices[ax], int(key[ax])) for ax in range(len(new_indices))
        )
        new_blocks[key] = _pad_symmetric_block_along_axes(
            block, block.shape, new_shape, chi_axes
        )
    return SymmetricTensor._from_blocks_unchecked(new_blocks, new_indices)


def _pad_symmetric_corner(t: SymmetricTensor, chi_new: int) -> SymmetricTensor:
    """Block-sparse pad of a corner SymmetricTensor along both χ axes."""
    idx0 = _pad_chi_index(t.indices[0], chi_new)
    idx1 = _pad_chi_index(t.indices[1], chi_new)
    return _pad_symmetric_tensor_along_chi(t, (idx0, idx1), chi_axes=(0, 1))


def _pad_symmetric_edge(t: SymmetricTensor, chi_new: int) -> SymmetricTensor:
    """Block-sparse pad of an edge SymmetricTensor along axes 0 and 2 only."""
    idx0 = _pad_chi_index(t.indices[0], chi_new)
    idx2 = _pad_chi_index(t.indices[2], chi_new)
    return _pad_symmetric_tensor_along_chi(
        t, (idx0, t.indices[1], idx2), chi_axes=(0, 2)
    )


def _current_chi(t: Tensor) -> int:
    """Return the current χ (first leg dim) of a corner or edge tensor."""
    return int(t.indices[0].dim)


def pad_dense_env_chi(env: CTMTensorEnv, chi_new: int) -> CTMTensorEnv:
    """Zero-pad the χ axes of a CTMTensorEnv from the current χ to ``chi_new``.

    Used by the variPEPS §2.8.2 auto-bump warm-start. Corners' both axes
    grow to ``chi_new``; edges' axes 0 and 2 grow (axis 1, the D² fused
    leg, is untouched).  For SymmetricTensor envs, new χ-leg charges are
    derived from the existing pattern via ``_derive_charges`` (the same
    allocator used by the symmetric projectors), and each block is padded
    along its χ axes; sectors absent from the old blocks stay implicitly
    zero in the flat-buffer representation.

    Returns the same env if ``chi_new`` matches the current χ. Raises
    ``ValueError`` if ``chi_new < chi_old``.

    Args:
        env:     CTMTensorEnv whose corners and edges are uniformly either
                 ``DenseTensor`` or ``SymmetricTensor``.
        chi_new: Target bond dimension. Must be >= the current χ.

    Returns:
        New CTMTensorEnv with all χ legs padded to ``chi_new`` with zeros.
    """
    chi_old = _current_chi(env.C1)

    if chi_new < chi_old:
        raise ValueError(
            f"chi_new={chi_new} must be >= chi_old={chi_old} (shrinking is not supported)"
        )

    if chi_new == chi_old:
        return env

    if isinstance(env.C1, SymmetricTensor):
        return CTMTensorEnv(
            C1=_pad_symmetric_corner(env.C1, chi_new),
            C2=_pad_symmetric_corner(env.C2, chi_new),
            C3=_pad_symmetric_corner(env.C3, chi_new),
            C4=_pad_symmetric_corner(env.C4, chi_new),
            T1=_pad_symmetric_edge(env.T1, chi_new),
            T2=_pad_symmetric_edge(env.T2, chi_new),
            T3=_pad_symmetric_edge(env.T3, chi_new),
            T4=_pad_symmetric_edge(env.T4, chi_new),
        )

    return CTMTensorEnv(
        C1=_pad_corner(env.C1, chi_new),
        C2=_pad_corner(env.C2, chi_new),
        C3=_pad_corner(env.C3, chi_new),
        C4=_pad_corner(env.C4, chi_new),
        T1=_pad_edge(env.T1, chi_new),
        T2=_pad_edge(env.T2, chi_new),
        T3=_pad_edge(env.T3, chi_new),
        T4=_pad_edge(env.T4, chi_new),
    )
