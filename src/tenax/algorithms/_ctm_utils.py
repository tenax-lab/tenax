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
    D: int,  # noqa: ARG001 — kept for call-site compatibility (no longer used)
    label_a: Label,
    label_b: Label,
    flow_a: FlowDirection,
    flow_b: FlowDirection,
    dtype,
) -> DenseTensor:
    """Create a rank-1 (variPEPS ``chi_init=1``) DenseTensor corner (chi x chi).

    Writes only entry ``(0, 0) = 1``.  A rank-``min(chi, D)`` identity seed
    (the previous ``eye(min(chi, D))``) drove the split CTM onto an
    *artificially* degenerate corner fixed point (e.g. ``[0.5, 0.5, 0, 0]``
    for D=2) whose degenerate subspace rotates every sweep — the env then
    never converges element-wise, which blocks implicit-AD fixed-point
    differentiation (#463).  Rank-1 mirrors the fused/standard
    ``_make_rank1_dense_corner`` and converges element-wise to the
    non-degenerate ``[1, 0, …]`` corner.  (Used only by the split env init.)
    """
    C_pad = jnp.zeros((chi, chi), dtype=dtype).at[0, 0].set(1.0)
    sym = _trivial_symmetry()
    idx_a = TensorIndex.from_charges(
        sym, np.zeros(chi, dtype=np.int32), flow_a, label=label_a
    )
    idx_b = TensorIndex.from_charges(
        sym, np.zeros(chi, dtype=np.int32), flow_b, label=label_b
    )
    return DenseTensor(C_pad, (idx_a, idx_b))


def _derive_charges(base_charges: np.ndarray, target_dim: int) -> np.ndarray:
    """Derive charges of size target_dim from base charges by tiling."""
    n = len(base_charges)
    if target_dim <= n:
        return np.asarray(base_charges[:target_dim], dtype=np.int32)
    reps = target_dim // n + 1
    return np.asarray(np.tile(base_charges, reps)[:target_dim], dtype=np.int32)


def _sector_floor(base_charges: np.ndarray | None, *, floor: int = 1) -> dict[int, int]:
    """Minimum number of chi-bond slots each charge sector must keep.

    ``base_charges`` names the charges that carry the physical bond's own
    sector structure.  The floor exists so that a sector whose weight dips
    below the chi cut for one sweep still keeps a slot on the bond: a charge
    absent from the bond index cannot be produced by ``_contract_symmetric``,
    which pairs blocks by charge *value*, so dropping it is not a truncation
    but a deletion.  It is deliberately a *floor* and not a quota -- see
    :func:`_select_chi_slots`.
    """
    if base_charges is None or floor <= 0:
        return {}
    return {int(q): floor for q in np.asarray(base_charges)}


def _select_chi_slots(
    values: np.ndarray,
    slot_charges: np.ndarray,
    *,
    base_charges: np.ndarray | None,
    chi: int,
    floor: int = 1,
) -> list[int]:
    """Choose which slots of a full decomposition survive the chi cut.

    Global top-``chi`` by ``values``, after reserving ``floor`` slots for each
    charge named in ``base_charges`` (taking that sector's own largest values).

    ``base_charges`` used to *pin* the per-sector counts, via
    ``_derive_charges(base_charges, chi)``.  That capped every sector at its
    tiled share and gave the charges absent from ``base_charges`` a share of
    zero, so the CTM chi bond could never allocate a slot to them however much
    weight they carried -- the environment saturated below the dense reference
    and the gap *grew* with chi (#922).  It now only raises a floor.

    Two limits, both deliberate:

    * The floor can only ration slots that exist.  A charge named by
      ``base_charges`` that contributes no entry to ``values`` gets nothing
      here, because there is no vector to keep; callers needing such a sector
      on the bond have to supply one first, as
      :func:`~tenax.algorithms._ctm_projector._eigh_projector_symmetric` does.
    * With more distinct base charges than ``chi``, the floor spends the whole
      budget and the largest values can lose.  The floor costs at most one slot
      per distinct charge, bounded by D^2 and normally far below chi.

    This is the *eager* rule.  Under ``jax.jit`` the per-sector block shapes are
    baked at trace time, so the cut cannot read ``values`` at all and
    ``linalg._truncated_svd_symmetric_traced`` keeps the old quota -- see #929.

    Args:
        values:        Singular/eigen values, one per slot.  Need not be sorted.
        slot_charges:  Charge of each slot, in the same order as ``values``.
        base_charges:  Charges whose sectors get the floor; ``None`` disables it.
        chi:           Number of slots to keep (clamped to ``len(values)``).
        floor:         Slots reserved per charge in ``base_charges``.

    Returns:
        The kept slot indices, ascending.  Ascending order preserves the
        caller's slot ordering, which for a descending-sorted decomposition
        means the result is still ordered by decreasing value.
    """
    values = np.asarray(values, dtype=float)
    slot_charges = np.asarray(slot_charges, dtype=np.int32)
    n = len(values)
    chi = min(int(chi), n)
    if chi <= 0:
        return []

    order = [int(j) for j in np.argsort(-values, kind="stable")]
    by_sector: dict[int, list[int]] = {}
    for j in order:  # descending value within each sector
        by_sector.setdefault(int(slot_charges[j]), []).append(j)

    keep: list[int] = []
    taken: set[int] = set()
    for q, want in sorted(_sector_floor(base_charges, floor=floor).items()):
        for j in by_sector.get(q, [])[:want]:
            if j not in taken and len(keep) < chi:
                keep.append(j)
                taken.add(j)
    for j in order:
        if len(keep) >= chi:
            break
        if j not in taken:
            keep.append(j)
            taken.add(j)
    return sorted(keep)


# A labels: (u, d, l, r, phys), flows: (OUT, IN, OUT, IN, IN)
# Env label/flow conventions per the plan
_CORNER_SPECS = {
    "C1": ("c1_d", "c1_r", FlowDirection.IN, FlowDirection.OUT, 1),  # ref_axis=d(1)
    "C2": ("c2_l", "c2_d", FlowDirection.IN, FlowDirection.OUT, 0),  # ref_axis=u(0)
    "C3": ("c3_u", "c3_l", FlowDirection.OUT, FlowDirection.IN, 1),  # ref_axis=d(1)
    "C4": ("c4_r", "c4_u", FlowDirection.OUT, FlowDirection.IN, 0),  # ref_axis=u(0)
}
