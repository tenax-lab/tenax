"""Sector layer for the symmetric root-implicit CTMRG gradient (#715 Phase 3).

Nothing here knows about CTM.  It knows about charge sectors: how a chi bond
is split across them, how to decompose a block-diagonal matrix sector by
sector, and how to get back to a :class:`SymmetricTensor`.  A future CTM
layer (``_ctm_root_implicit_symmetric``, not yet written) will call into this
at the cut and nowhere else; the decomposition/reassembly helpers implied by
that split are forthcoming and not part of this module yet.

The split exists because the two halves fail differently.  A bug here is a
wrong *shape* or a wrong charge and shows up as an exception; a bug in the CTM
layer is a mis-glued network and shows up as a wrong number.  Keeping them
apart keeps the tests that catch them apart too.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from tenax.core.index import FlowDirection, Label, TensorIndex
from tenax.core.symmetry import BaseSymmetry


@dataclass(frozen=True, slots=True)
class BondLayout:
    """How a truncated chi bond is distributed over charge sectors.

    This is the symmetric analogue of "chi is an int".  Every downstream shape
    — ``u``, ``v``, ``S``, ``U*``, ``U_perp`` — is read off it, and it is
    *frozen* at the converged point: if it moved under AD the adjoint would be
    solving a different-sized system than the forward built.

    Deliberately **not** a JAX pytree: a ``NamedTuple`` with a ``dict`` field
    flattens its dimensions to pytree leaves, which makes them tracers under
    ``jax.jit``/``jax.grad`` and makes the layout itself unhashable (dicts
    aren't hashable), so it cannot be closed over as static metadata either
    — exactly where this needs to live, since ``custom_vjp``'s
    ``nondiff_argnums`` requires hashability. A frozen dataclass over a tuple
    of ``(charge, dim)`` pairs is opaque to ``jax.tree_util`` and hashable,
    matching the repo's existing pattern for static shape metadata
    (:class:`~tenax.core.index.TensorIndex`).

    Attributes:
        dims: Sorted ``(charge, dim)`` pairs, zero-dim entries excluded.
              Construct via :meth:`from_dims` rather than directly, so the
              sort/filter invariant is guaranteed rather than assumed.
    """

    dims: tuple[tuple[int, int], ...]

    @classmethod
    def from_dims(cls, dims: Mapping[int, int]) -> BondLayout:
        """Build a :class:`BondLayout` from a ``{charge: dim}`` mapping.

        Normalises once: drops zero-dim entries, sorts by charge, and
        rejects negative dims. Downstream reads (``total``, ``sectors``,
        ``dim_of``) then trust the invariant instead of re-deriving it.
        """
        bad = {q: d for q, d in dims.items() if d < 0}
        if bad:
            raise ValueError(
                f"BondLayout dims must be non-negative, got negative entries {bad} "
                f"(full input: {dict(dims)})"
            )
        pairs = tuple(sorted((q, d) for q, d in dims.items() if d > 0))
        return cls(dims=pairs)

    @property
    def total(self) -> int:
        return sum(d for _, d in self.dims)

    @property
    def sectors(self) -> list[int]:
        """Retained charges, sorted, excluding sectors that kept nothing."""
        return [q for q, _ in self.dims]

    def dim_of(self, charge: int) -> int:
        """Return the retained dimension for ``charge`` (0 if not retained)."""
        for q, d in self.dims:
            if q == charge:
                return d
        return 0


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
    sectors = layout.sectors
    if not sectors:
        raise ValueError(
            f"BondLayout retained no charges; the cut is empty (dims={layout.dims})."
        )
    return TensorIndex(
        symmetry=symmetry,
        sectors=np.asarray(sectors, dtype=np.int32),
        multiplicities=np.asarray([layout.dim_of(q) for q in sectors], dtype=np.int32),
        flow=flow,
        label=label,
    )
