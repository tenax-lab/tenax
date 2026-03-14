"""Tests for tenax.core.lattice dataclasses and factory functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
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

    def test_neighbor_map_is_immutable(self):
        """neighbor_map and its inner dicts must be read-only."""
        lat = square()
        with pytest.raises(TypeError):
            lat.neighbor_map["a"] = {}  # type: ignore[index]
        with pytest.raises(TypeError):
            lat.neighbor_map["a"]["left"] = "z"  # type: ignore[index]


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


class TestExports:
    def test_importable_from_core(self):
        from tenax.core import Bond, Lattice  # noqa: F401

    def test_importable_from_tenax(self):
        from tenax import Bond, Lattice, checkerboard, kagome, square  # noqa: F401

    def test_factories_from_tenax(self):
        from tenax import checkerboard, honeycomb, kagome, square, triangular
        from tenax.core.lattice import Lattice as L

        assert isinstance(square(), L)
        assert isinstance(checkerboard(), L)
        assert isinstance(honeycomb(), L)
        assert isinstance(triangular(), L)
        assert isinstance(kagome(), L)


class TestCtmMultisiteExports:
    def test_importable_from_convergence(self):
        from tenax.algorithms._ctm_tensor_convergence import ctm_multisite  # noqa: F401

    def test_importable_from_algorithms(self):
        from tenax.algorithms import ctm_multisite  # noqa: F401

    def test_importable_from_tenax(self):
        from tenax import ctm_multisite  # noqa: F401

    def test_importable_from_ctm_tensor_shim(self):
        from tenax.algorithms._ctm_tensor import ctm_multisite  # noqa: F401


# ---------------------------------------------------------------------------
# Integration tests for ctm_multisite
# ---------------------------------------------------------------------------


def _make_random_dense_site_tensor(key, D=2, d=2):
    """Build a random DenseTensor site tensor with 5 legs (u, d, l, r, phys)."""
    from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry

    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="u"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="d"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="l"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="r"),
        TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
    )
    data = jax.random.normal(key, shape=(D, D, D, D, d))
    return DenseTensor(data, indices)


class TestCtmMultisiteIntegration:
    def test_checkerboard_returns_dict(self):
        """ctm_multisite with checkerboard returns envs keyed by site name."""
        from tenax.algorithms._ctm_tensor_convergence import ctm_multisite

        lat = checkerboard()
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        A = _make_random_dense_site_tensor(k1)
        B = _make_random_dense_site_tensor(k2)
        envs = ctm_multisite({"a": A, "b": B}, lat, chi=4, max_iter=5, conv_tol=1e-6)
        assert set(envs.keys()) == {"a", "b"}

    def test_checkerboard_matches_ctm_tensor_2site(self):
        """ctm_multisite with checkerboard matches ctm_tensor_2site."""
        from tenax.algorithms._ctm_tensor_convergence import (
            ctm_multisite,
            ctm_tensor_2site,
        )

        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        A = _make_random_dense_site_tensor(k1)
        B = _make_random_dense_site_tensor(k2)

        lat = checkerboard()
        envs_multi = ctm_multisite(
            {"a": A, "b": B}, lat, chi=4, max_iter=30, conv_tol=1e-10
        )
        env_A, env_B = ctm_tensor_2site(A, B, chi=4, max_iter=30, conv_tol=1e-10)

        # Compare normalized corner singular values
        sv_multi_a = jnp.linalg.svd(envs_multi["a"].C1.todense(), compute_uv=False)
        sv_direct_a = jnp.linalg.svd(env_A.C1.todense(), compute_uv=False)
        sv_multi_a = sv_multi_a / jnp.sum(sv_multi_a)
        sv_direct_a = sv_direct_a / jnp.sum(sv_direct_a)
        assert jnp.allclose(sv_multi_a, sv_direct_a, atol=1e-6)

    def test_kagome_runs_to_completion(self):
        """ctm_multisite with kagome runs without error."""
        from tenax.algorithms._ctm_tensor_convergence import ctm_multisite

        lat = kagome()
        key = jax.random.PRNGKey(7)
        keys = jax.random.split(key, 3)
        tensors = {
            name: _make_random_dense_site_tensor(keys[i])
            for i, name in enumerate(lat.sites)
        }
        envs = ctm_multisite(tensors, lat, chi=4, max_iter=5, conv_tol=1e-6)
        assert set(envs.keys()) == set(lat.sites)

    def test_missing_site_raises(self):
        """ctm_multisite raises ValueError when a lattice site is missing."""
        from tenax.algorithms._ctm_tensor_convergence import ctm_multisite

        lat = checkerboard()
        A = _make_random_dense_site_tensor(jax.random.PRNGKey(0))
        with pytest.raises(ValueError, match="missing sites.*'b'"):
            ctm_multisite({"a": A}, lat, chi=4)

    def test_extra_site_raises(self):
        """ctm_multisite raises ValueError for sites not in lattice."""
        from tenax.algorithms._ctm_tensor_convergence import ctm_multisite

        lat = square()
        A = _make_random_dense_site_tensor(jax.random.PRNGKey(0))
        B = _make_random_dense_site_tensor(jax.random.PRNGKey(1))
        with pytest.raises(ValueError, match="not in lattice.*'z'"):
            ctm_multisite({"a": A, "z": B}, lat, chi=4)
