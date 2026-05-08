"""Zero-pad the χ axes of a dense CTMTensorEnv for auto-χ warm-start.

Used by the variPEPS §2.8.2 auto-bump protocol (Task 6) to grow an already-
converged environment from ``chi_old`` to ``chi_new`` before re-running CTM.
Only the dense path is supported in v1; SymmetricTensor envs raise
NotImplementedError (v2 follow-up, Task 9).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.core.index import TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor


def _pad_chi_index(idx: TensorIndex, chi_new: int) -> TensorIndex:
    """Return a new TensorIndex with dim grown to ``chi_new``.

    The original charges are preserved in the first ``chi_old`` slots;
    the new slots receive charge 0 (trivial sector), matching the
    zero-padded array entries.
    """
    chi_old = idx.dim
    old_charges = np.asarray(idx.charges, dtype=np.int32)
    pad_charges = np.zeros(chi_new - chi_old, dtype=np.int32)
    new_charges = np.concatenate([old_charges, pad_charges])
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


def pad_dense_env_chi(env: CTMTensorEnv, chi_new: int) -> CTMTensorEnv:
    """Zero-pad the χ axes of a dense CTMTensorEnv from current χ to ``chi_new``.

    Used by the variPEPS §2.8.2 auto-bump warm-start. Corners' both axes
    grow to ``chi_new``; edges' axes 0 and 2 grow (axis 1, the D² fused
    leg, is untouched).

    Returns the same env if ``chi_new`` matches the current χ. Raises
    ``ValueError`` if ``chi_new < chi_old`` and ``NotImplementedError``
    if any tensor is a ``SymmetricTensor`` (v2 follow-up).

    Args:
        env:     CTMTensorEnv whose corners and edges are all DenseTensor.
        chi_new: Target bond dimension. Must be >= the current χ.

    Returns:
        New CTMTensorEnv with all χ legs padded to ``chi_new`` with zeros.
    """
    # Detect SymmetricTensor before anything else.
    for field_name in CTMTensorEnv._fields:
        t = getattr(env, field_name)
        if isinstance(t, SymmetricTensor):
            raise NotImplementedError("padding SymmetricTensor envs is a v2 follow-up")

    chi_old = env.C1._data.shape[0]

    if chi_new < chi_old:
        raise ValueError(
            f"chi_new={chi_new} must be >= chi_old={chi_old} (shrinking is not supported)"
        )

    if chi_new == chi_old:
        return env

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
