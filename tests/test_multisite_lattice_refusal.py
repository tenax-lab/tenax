"""``optimize_gs_ad`` refuses a ``Lattice`` whose bonds its energy does not sum.

The multisite energy (``compute_energy_ctm_tensor_multisite``) sums each site's
``right`` and ``bottom`` neighbour in ``neighbor_map`` -- the nearest-neighbour
model of a square lattice coloured by that map -- and has no term for any other
bond.  ``kagome()`` declares a v-w ``diagonal``, ``triangular()`` an a-a
``diagonal`` and ``honeycomb()`` a ``vertical_top``/``vertical_bottom`` pair, so
``optimize_gs_ad`` minimised the square-lattice model under those names: on
``kagome()`` a uniform D=2 state gave the square-lattice ``e_h + e_v`` to 6e-17,
and emptying ``bonds`` changed nothing (#1004).

Each test stops the run at the hand-off to ``_optimize_gs_ad_multisite``: a
refusal must come before it, and an accepted lattice must reach it.
"""

from __future__ import annotations

import re

import pytest

import tenax.algorithms.ipeps_optimize as ipeps_optimize
from tenax import heisenberg_gate, iPEPSConfig, optimize_gs_ad
from tenax.core.lattice import (
    Bond,
    Lattice,
    checkerboard,
    honeycomb,
    kagome,
    square,
    triangular,
)


def _with_bonds(lattice: Lattice, *bonds: Bond) -> Lattice:
    """``lattice``'s sites and neighbour map, declaring ``bonds`` instead."""
    return Lattice(
        sites=lattice.sites,
        bonds=bonds,
        neighbor_map={k: dict(v) for k, v in lattice.neighbor_map.items()},
    )


# kagome()'s neighbour map colours the square lattice (x + y) mod 3.  Declared
# as its six right/bottom bonds, it is that square lattice under its own name.
_THREE_COLOUR_SQUARE = _with_bonds(
    kagome(),
    *(
        Bond(a, b, direction)
        for direction in ("horizontal", "vertical")
        for a, b in (("u", "v"), ("v", "w"), ("w", "u"))
    ),
)

# name -> (lattice, a bond the refusal must name)
REFUSED = {
    # A declared bond the energy never sums.
    "kagome": (kagome(), "('v', 'w', 'diagonal')"),
    "triangular": (triangular(), "('a', 'a', 'diagonal')"),
    "honeycomb": (honeycomb(), "('a', 'b', 'vertical_top')"),
    # A summed bond the lattice never declares: decoupled chains.
    "chains": (
        _with_bonds(square(), Bond("a", "a", "horizontal")),
        "('a', 'a', 'vertical')",
    ),
    # The summed directions, on pairs the neighbour map does not join.
    "wrong_pairs": (
        _with_bonds(
            checkerboard(), Bond("a", "a", "horizontal"), Bond("b", "b", "vertical")
        ),
        "('a', 'a', 'horizontal')",
    ),
}

ACCEPTED = {
    "square": square(),  # bonds=() declares the neighbour map's bonds
    "checkerboard": checkerboard(),
    "three_colour_square": _THREE_COLOUR_SQUARE,
}


@pytest.fixture
def reached(monkeypatch):
    """Replace the multisite optimizer; record the lattices handed to it."""
    seen = []

    def _stop(hamiltonian_gate, A_init, config):
        seen.append(config.unit_cell)
        return "reached"

    monkeypatch.setattr(ipeps_optimize, "_optimize_gs_ad_multisite", _stop)
    return seen


@pytest.mark.parametrize("name", REFUSED)
def test_a_lattice_whose_bonds_are_not_summed_is_refused(name, reached):
    lattice, bond = REFUSED[name]
    with pytest.raises(NotImplementedError, match=re.escape(bond)):
        optimize_gs_ad(heisenberg_gate(), None, iPEPSConfig(unit_cell=lattice))


@pytest.mark.parametrize("name", ACCEPTED)
def test_a_square_lattice_colouring_reaches_the_optimizer(name, reached):
    lattice = ACCEPTED[name]
    result = optimize_gs_ad(heisenberg_gate(), None, iPEPSConfig(unit_cell=lattice))
    assert result == "reached"
    assert reached == [lattice]
