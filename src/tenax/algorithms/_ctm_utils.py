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

    The per-sector counts come from :func:`_allocate_chi_counts`; ``values``
    then decides *which* slots inside each sector, largest first.  Splitting it
    that way is what keeps the eager and traced paths computing the same
    function: under ``jax.jit`` the block shapes are static, so the counts have
    to be fixed before any singular value exists, and a cut that chose them
    from ``values`` would make ``jax.grad`` differentiate a differently-shaped
    projector than the forward pass built (a 9.9% AD-vs-finite-difference gap,
    caught by
    ``test_compute_2x2_projector_grad_matches_finite_difference``).

    ``base_charges`` used to *pin* the counts, via
    ``_derive_charges(base_charges, chi)``.  That capped every sector at its
    tiled share and gave the sectors absent from ``base_charges`` a share of
    zero, so the CTM chi bond could never allocate a slot to them however much
    weight they carried -- the environment saturated below the dense reference
    and the gap *grew* with chi (#922).  It now only raises a floor.

    With ``base_charges`` ``None`` there is no sector structure to preserve and
    this is the plain global top-``chi``.

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
