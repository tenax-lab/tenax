"""Sector layer for the symmetric root-implicit CTMRG gradient (#715 Phase 3).

Nothing here knows about CTM.  It knows about charge sectors: how a chi bond
is split across them, how to decompose a block-diagonal matrix sector by
sector, and how to get back to a :class:`SymmetricTensor`.  The CTM layer in
``_ctm_root_implicit_symmetric`` calls into this at the cut and nowhere else.

The split exists because the two halves fail differently.  A bug here is a
wrong *shape* or a wrong charge and shows up as an exception; a bug in the CTM
layer is a mis-glued network and shows up as a wrong number.  Keeping them
apart keeps the tests that catch them apart too.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from tenax import FlowDirection, Label, TensorIndex
from tenax.core.symmetry import BaseSymmetry


class BondLayout(NamedTuple):
    """How a truncated chi bond is distributed over charge sectors.

    This is the symmetric analogue of "chi is an int".  Every downstream shape
    — ``u``, ``v``, ``S``, ``U*``, ``U_perp`` — is read off it, and it is
    *frozen* at the converged point: if it moved under AD the adjoint would be
    solving a different-sized system than the forward built.
    """

    dims: dict[int, int]

    @property
    def total(self) -> int:
        return int(sum(self.dims.values()))

    @property
    def charges(self) -> list[int]:
        """Retained charges, sorted, excluding sectors that kept nothing."""
        return sorted(q for q, d in self.dims.items() if d > 0)


def bond_index_from_layout(
    layout: BondLayout,
    symmetry: BaseSymmetry,
    flow: FlowDirection,
    label: Label,
) -> TensorIndex:
    """Build the truncated chi bond described by ``layout``.

    Sectors that retained nothing are dropped rather than kept at width zero.
    A zero multiplicity is a legal but poisonous index: it survives every
    charge check and then produces zero-size blocks that contract to zero.
    """
    charges = layout.charges
    if not charges:
        raise ValueError("BondLayout retained no charges; the cut is empty.")
    return TensorIndex(
        symmetry=symmetry,
        sectors=np.asarray(charges, dtype=np.int32),
        multiplicities=np.asarray([layout.dims[q] for q in charges], dtype=np.int32),
        flow=flow,
        label=label,
    )
