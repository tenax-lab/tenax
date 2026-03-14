"""Lattice abstraction for iPEPS unit cells.

Provides frozen dataclasses for describing lattice geometry and factory
functions for common lattice types used in tensor-network simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "Bond",
    "Lattice",
    "square",
    "checkerboard",
    "honeycomb",
    "triangular",
    "kagome",
]


@dataclass(frozen=True)
class Bond:
    """An oriented bond between two sites in the unit cell."""

    site_i: str
    site_j: str
    direction: str


@dataclass(frozen=True)
class Lattice:
    """Immutable description of a lattice unit cell.

    Parameters
    ----------
    sites : tuple[str, ...]
        Labels for the inequivalent sites in the unit cell.
    bonds : tuple[Bond, ...]
        Bonds connecting sites (may include self-bonds for single-site cells).
    neighbor_map : dict[str, dict[str, str]]
        For each site, a mapping from direction ("left", "right", "top",
        "bottom") to the neighboring site label.  Frozen on construction
        via ``MappingProxyType`` to enforce immutability.
    """

    sites: tuple[str, ...]
    bonds: tuple[Bond, ...]
    neighbor_map: dict[str, dict[str, str]]

    def __post_init__(self) -> None:
        # Deep-freeze neighbor_map so callers cannot mutate topology.
        frozen: Any = MappingProxyType(
            {k: MappingProxyType(v) for k, v in self.neighbor_map.items()}
        )
        object.__setattr__(self, "neighbor_map", frozen)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def square() -> Lattice:
    """Single-site square lattice (all neighbors are self)."""
    return Lattice(
        sites=("a",),
        bonds=(),
        neighbor_map={"a": {"left": "a", "right": "a", "top": "a", "bottom": "a"}},
    )


def checkerboard() -> Lattice:
    """Two-site checkerboard lattice with alternating neighbors."""
    return Lattice(
        sites=("a", "b"),
        bonds=(Bond("a", "b", "horizontal"), Bond("a", "b", "vertical")),
        neighbor_map={
            "a": {"left": "b", "right": "b", "top": "b", "bottom": "b"},
            "b": {"left": "a", "right": "a", "top": "a", "bottom": "a"},
        },
    )


def honeycomb() -> Lattice:
    """Two-site honeycomb lattice mapped onto a brick-wall square grid."""
    return Lattice(
        sites=("a", "b"),
        bonds=(
            Bond("a", "b", "horizontal"),
            Bond("a", "b", "vertical_top"),
            Bond("a", "b", "vertical_bottom"),
        ),
        neighbor_map={
            "a": {"left": "b", "right": "b", "top": "b", "bottom": "b"},
            "b": {"left": "a", "right": "a", "top": "a", "bottom": "a"},
        },
    )


def triangular() -> Lattice:
    """Single-site triangular lattice (includes a diagonal bond)."""
    return Lattice(
        sites=("a",),
        bonds=(
            Bond("a", "a", "horizontal"),
            Bond("a", "a", "vertical"),
            Bond("a", "a", "diagonal"),
        ),
        neighbor_map={"a": {"left": "a", "right": "a", "top": "a", "bottom": "a"}},
    )


def kagome() -> Lattice:
    """Three-site kagome lattice."""
    return Lattice(
        sites=("u", "v", "w"),
        bonds=(
            Bond("u", "v", "horizontal"),
            Bond("u", "w", "vertical"),
            Bond("v", "w", "diagonal"),
        ),
        neighbor_map={
            "u": {"left": "w", "right": "v", "top": "w", "bottom": "v"},
            "v": {"left": "u", "right": "w", "top": "u", "bottom": "w"},
            "w": {"left": "v", "right": "u", "top": "v", "bottom": "u"},
        },
    )
