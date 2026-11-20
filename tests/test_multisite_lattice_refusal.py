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
import warnings

import pytest

import tenax.algorithms.ipeps_optimize as ipeps_optimize
from tenax import CTMConfig, heisenberg_gate, iPEPSConfig, optimize_gs_ad
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


def _columns(horizontal: str, vertical: str) -> Lattice:
    """Two-column stripes, declaring a-b as ``horizontal`` and a-a, b-b as ``vertical``.

    a's right neighbour is b and its bottom neighbour is a.  Every other
    fixture has the same right and bottom neighbour, so only this one tells a
    horizontal bond from a vertical one.
    """
    return Lattice(
        sites=("a", "b"),
        bonds=(
            Bond("a", "b", horizontal),
            Bond("a", "a", vertical),
            Bond("b", "b", vertical),
        ),
        neighbor_map={
            "a": {"left": "b", "right": "b", "top": "a", "bottom": "a"},
            "b": {"left": "a", "right": "a", "top": "b", "bottom": "b"},
        },
    )


NEVER_SUMMED, NEVER_DECLARED = "declared, never summed", "summed, never declared"

# name -> (lattice, the refusal's line, a bond that line must name)
REFUSED = {
    # A declared bond the energy never sums.
    "kagome": (kagome(), NEVER_SUMMED, "('v', 'w', 'diagonal')"),
    "triangular": (triangular(), NEVER_SUMMED, "('a', 'a', 'diagonal')"),
    "honeycomb": (honeycomb(), NEVER_SUMMED, "('a', 'b', 'vertical_top')"),
    # A summed bond the lattice never declares: decoupled chains.
    "chains": (
        _with_bonds(square(), Bond("a", "a", "horizontal")),
        NEVER_DECLARED,
        "('a', 'a', 'vertical')",
    ),
    # The summed directions, on pairs the neighbour map does not join.
    "wrong_pairs": (
        _with_bonds(
            checkerboard(), Bond("a", "a", "horizontal"), Bond("b", "b", "vertical")
        ),
        NEVER_SUMMED,
        "('a', 'a', 'horizontal')",
    ),
    # The right pairs, with horizontal and vertical exchanged.
    "transposed_columns": (
        _columns("vertical", "horizontal"),
        NEVER_SUMMED,
        "('a', 'b', 'vertical')",
    ),
}

ACCEPTED = {
    "square": square(),  # bonds=() declares the neighbour map's bonds
    "checkerboard": checkerboard(),
    "three_colour_square": _THREE_COLOUR_SQUARE,
    "columns": _columns("horizontal", "vertical"),
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
    lattice, line, bond = REFUSED[name]
    match = re.escape(line) + r": [^\n]*" + re.escape(bond)
    with pytest.raises(NotImplementedError, match=match):
        optimize_gs_ad(heisenberg_gate(), None, iPEPSConfig(unit_cell=lattice))


def test_the_refusal_comes_before_anything_else(reached):
    """kagome() hears its own reason first, with no warning before it.

    The root-implicit engine refuses every Lattice too, after a line-search
    warning, and its reason (no rectangular-grid objective) sends the user back
    to the default path, which refuses kagome() as well.
    """
    gate = heisenberg_gate()
    config = iPEPSConfig(unit_cell=kagome(), ctm=CTMConfig(ctm_ad_mode="root_implicit"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            NotImplementedError, match=re.escape("('v', 'w', 'diagonal')")
        ):
            optimize_gs_ad(gate, None, config)


@pytest.mark.parametrize("name", ACCEPTED)
def test_a_square_lattice_colouring_reaches_the_optimizer(name, reached):
    lattice = ACCEPTED[name]
    result = optimize_gs_ad(heisenberg_gate(), None, iPEPSConfig(unit_cell=lattice))
    assert result == "reached"
    assert reached == [lattice]
