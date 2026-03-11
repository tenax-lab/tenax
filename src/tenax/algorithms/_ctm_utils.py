"""Shared CTM utilities used by both standard and split CTM modules.

Contains corner/edge initialization helpers and charge derivation
that are needed by ``_ctm_tensor`` and ``_split_ctm_tensor``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tenax.core.index import FlowDirection, Label, TensorIndex
from tenax.core.tensor import DenseTensor


def _trivial_symmetry():
    from tenax.core.symmetry import U1Symmetry

    return U1Symmetry()


def _make_dense_corner(
    chi: int,
    D: int,
    label_a: Label,
    label_b: Label,
    flow_a: FlowDirection,
    flow_b: FlowDirection,
    dtype,
) -> DenseTensor:
    """Create an identity-like DenseTensor corner (chi x chi)."""
    C = jnp.eye(min(chi, D), dtype=dtype)
    C_pad = jnp.zeros((chi, chi), dtype=dtype)
    C_pad = C_pad.at[: C.shape[0], : C.shape[1]].set(C)
    sym = _trivial_symmetry()
    idx_a = TensorIndex(sym, np.zeros(chi, dtype=np.int32), flow_a, label=label_a)
    idx_b = TensorIndex(sym, np.zeros(chi, dtype=np.int32), flow_b, label=label_b)
    return DenseTensor(C_pad, (idx_a, idx_b))


def _derive_charges(base_charges: np.ndarray, target_dim: int) -> np.ndarray:
    """Derive charges of size target_dim from base charges by tiling."""
    n = len(base_charges)
    if target_dim <= n:
        return np.asarray(base_charges[:target_dim], dtype=np.int32)
    reps = target_dim // n + 1
    return np.asarray(np.tile(base_charges, reps)[:target_dim], dtype=np.int32)


# A labels: (u, d, l, r, phys), flows: (OUT, IN, OUT, IN, IN)
# Env label/flow conventions per the plan
_CORNER_SPECS = {
    "C1": ("c1_d", "c1_r", FlowDirection.IN, FlowDirection.OUT, 1),  # ref_axis=d(1)
    "C2": ("c2_l", "c2_d", FlowDirection.IN, FlowDirection.OUT, 0),  # ref_axis=u(0)
    "C3": ("c3_u", "c3_l", FlowDirection.OUT, FlowDirection.IN, 1),  # ref_axis=d(1)
    "C4": ("c4_r", "c4_u", FlowDirection.OUT, FlowDirection.IN, 0),  # ref_axis=u(0)
}
