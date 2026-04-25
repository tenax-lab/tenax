"""Neighbor maps and direction labels for the honeycomb lattice."""

from __future__ import annotations

__all__ = ["Coord", "HONEYCOMB_DIRECTIONS", "HONEYCOMB_NEIGHBORS"]

Coord = tuple[int, int]

HONEYCOMB_DIRECTIONS: tuple[str, str, str] = ("e0", "e1", "e2")

HONEYCOMB_NEIGHBORS: dict[Coord, dict[str, Coord]] = {
    (0, 0): {"e0": (1, 0), "e1": (1, 0), "e2": (1, 0)},
    (1, 0): {"e0": (0, 0), "e1": (0, 0), "e2": (0, 0)},
}
