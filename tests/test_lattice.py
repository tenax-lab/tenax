"""Tests for tenax.core.lattice dataclasses and factory functions."""

from __future__ import annotations

import pytest

from tenax.core.lattice import (
    Bond,
    Lattice,
    checkerboard,
    honeycomb,
    kagome,
    square,
    triangular,
)

DIRECTIONS = {"left", "right", "top", "bottom"}


# ---------------------------------------------------------------------------
# Dataclass basics
# ---------------------------------------------------------------------------


class TestBondDataclass:
    def test_creation(self):
        b = Bond("a", "b", "horizontal")
        assert b.site_i == "a"
        assert b.site_j == "b"
        assert b.direction == "horizontal"

    def test_frozen(self):
        b = Bond("a", "b", "horizontal")
        with pytest.raises(AttributeError):
            b.site_i = "c"  # type: ignore[misc]


class TestLatticeDataclass:
    def test_creation(self):
        lat = Lattice(
            sites=("x",),
            bonds=(),
            neighbor_map={"x": {"left": "x", "right": "x", "top": "x", "bottom": "x"}},
        )
        assert lat.sites == ("x",)
        assert lat.bonds == ()

    def test_frozen(self):
        lat = square()
        with pytest.raises(AttributeError):
            lat.sites = ("z",)  # type: ignore[misc]

    def test_neighbor_map_has_four_directions(self):
        lat = square()
        for site in lat.sites:
            assert set(lat.neighbor_map[site].keys()) == DIRECTIONS


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestSquareFactory:
    def test_single_site(self):
        lat = square()
        assert lat.sites == ("a",)

    def test_self_neighbors(self):
        lat = square()
        for direction in DIRECTIONS:
            assert lat.neighbor_map["a"][direction] == "a"

    def test_no_bonds(self):
        lat = square()
        assert lat.bonds == ()


class TestCheckerboardFactory:
    def test_two_sites(self):
        lat = checkerboard()
        assert set(lat.sites) == {"a", "b"}

    def test_alternating_neighbors(self):
        lat = checkerboard()
        for direction in DIRECTIONS:
            assert lat.neighbor_map["a"][direction] == "b"
            assert lat.neighbor_map["b"][direction] == "a"

    def test_bonds_count(self):
        lat = checkerboard()
        assert len(lat.bonds) == 2


class TestHoneycombFactory:
    def test_two_sites(self):
        lat = honeycomb()
        assert set(lat.sites) == {"a", "b"}

    def test_complete_neighbor_map(self):
        lat = honeycomb()
        for site in lat.sites:
            assert set(lat.neighbor_map[site].keys()) == DIRECTIONS

    def test_has_bonds(self):
        lat = honeycomb()
        assert len(lat.bonds) > 0


class TestTriangularFactory:
    def test_single_site(self):
        lat = triangular()
        assert lat.sites == ("a",)

    def test_self_neighbors(self):
        lat = triangular()
        for direction in DIRECTIONS:
            assert lat.neighbor_map["a"][direction] == "a"

    def test_has_diagonal_bond(self):
        lat = triangular()
        directions = {b.direction for b in lat.bonds}
        assert "diagonal" in directions


class TestKagomeFactory:
    def test_three_sites(self):
        lat = kagome()
        assert set(lat.sites) == {"u", "v", "w"}

    def test_complete_neighbor_map(self):
        lat = kagome()
        for site in lat.sites:
            assert set(lat.neighbor_map[site].keys()) == DIRECTIONS

    def test_all_neighbors_valid(self):
        lat = kagome()
        valid = set(lat.sites)
        for site in lat.sites:
            for neighbor in lat.neighbor_map[site].values():
                assert neighbor in valid

    def test_has_bonds(self):
        lat = kagome()
        assert len(lat.bonds) == 3
