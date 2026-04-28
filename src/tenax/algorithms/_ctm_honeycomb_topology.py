"""Neighbor maps and direction labels for the honeycomb lattice."""

from __future__ import annotations

__all__ = ["Coord", "HONEYCOMB_DIRECTIONS", "HONEYCOMB_NEIGHBORS"]

from tenax.algorithms._ctm_tensor_convergence import Coord

#: Edge-direction labels for the honeycomb unit cell, in canonical order
#: ``α ∈ {0, 1, 2}``.
HONEYCOMB_DIRECTIONS: tuple[str, str, str] = ("e0", "e1", "e2")

#: Bipartite neighbor map: ``{coord: {direction: neighbor_coord}}``. Both
#: sublattices (``(0, 0)`` and ``(1, 0)``) point to each other across all
#: three edge directions, encoding the honeycomb's bipartite identification.
HONEYCOMB_NEIGHBORS: dict[Coord, dict[str, Coord]] = {
    (0, 0): {"e0": (1, 0), "e1": (1, 0), "e2": (1, 0)},
    (1, 0): {"e0": (0, 0), "e1": (0, 0), "e2": (0, 0)},
}
