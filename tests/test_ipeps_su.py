"""The simple-update engine that cannot hold a stale bond spectrum (#882 Phase 2).

``_SUState`` has two fields and no third.  That is the whole premise of the
rewrite -- the defect class behind #667, #851, #865 and #869 is a stored
spectrum invalidated by a non-unitary gate on *another* bond, and the design
deletes it by making it unrepresentable rather than by fixing it a fifth time.
A guard on the dataclass is cheap and it is the only thing that stays true as
the module grows.
"""

from __future__ import annotations

import dataclasses

import pytest
from _ipeps_gauge_helpers import (  # tests/ is on sys.path
    _PAIRS,
    _chain_pair,
    _chain_pair_as_peps,
)

from tenax.algorithms.ipeps_su import _SUState


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
def test_su_state_from_pair_round_trips_the_tensors(kind):
    """``from_pair`` stores exactly what it is handed, and validates labels."""
    A, B = _PAIRS[kind](D=3)
    state = _SUState.from_pair(A, B)

    assert state.A is A
    assert state.B is B

    with pytest.raises(ValueError, match="checkerboard site tensor"):
        _SUState.from_pair(A.relabel("phys", "p"), B)


def test_su_state_has_no_lambda_fields():
    """The design's premise, as an executable assertion.

    If a lambda field ever appears here, the bug class this rewrite deletes
    (#667/#851/#865/#869) becomes representable again -- and it does not come
    back as a wrong number, it comes back as a plausible one.  #869's D=3 run
    peaks at -0.654 near 800 steps and falls to -0.477 by 4000, from the same
    code: one step count reports either verdict.

    ``hasattr`` on the *class* covers the property and method namespace as well
    as the fields, so a ``lambdas`` cached on the instance by a later task is
    caught only if it is declared; a bare ``object.__setattr__`` on a frozen
    dataclass is not something this can see.  The field check is the load-
    bearing half.
    """
    names = {f.name for f in dataclasses.fields(_SUState)}
    assert names == {"A", "B"}, f"_SUState grew fields: {names - {'A', 'B'}}"
    for bad in ("lam", "lam_h", "lam_v", "lambdas", "weights", "singular_values"):
        assert not hasattr(_SUState, bad), f"_SUState.{bad} must not exist"


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
def test_su_state_max_D_reads_every_virtual_leg(kind):
    """``max_D`` is the largest *bond* dimension, and ``phys`` is not a bond.

    A pair whose four virtual legs all share ``D`` cannot discriminate: it
    passes on ``A.r.dim``, on ``min(...)``, on ``max(A.shape)``, and on
    anything else that happens to return ``D``.  Two anisotropic pairs are used
    instead, and between them every wrong reading this could plausibly be lands
    somewhere else:

    * the chain anchor -- ``D_v = 1``, ``D_h = 4`` -- kills ``min`` (1) and
      ``A.u.dim`` (1);
    * the same pair with the horizontal and vertical legs exchanged -- a pure
      relabel, and flow-preserving, since ``u``/``l`` are both OUT and
      ``d``/``r`` both IN -- kills ``A.r.dim`` (1).

    ``phys`` stays out of it on both: it is 2, so a reading that included it
    would agree here, which is why the assertion is on the leg *set* rather
    than only on the number.
    """
    A, B = _PAIRS[kind](D=3)
    assert _SUState.from_pair(A, B).max_D == 3

    a, b, _vl, _vr = _chain_pair()
    Ac, Bc = _chain_pair_as_peps(a, b)
    swap = {"u": "l", "l": "u", "d": "r", "r": "d"}
    As, Bs = Ac.relabels(swap), Bc.relabels(swap)

    for name, pair, expected in (
        ("chain", (Ac, Bc), {"u": 1, "d": 1, "l": 4, "r": 4}),
        ("swapped", (As, Bs), {"u": 4, "d": 4, "l": 1, "r": 1}),
    ):
        t = pair[0]
        dims = {
            lab: t.indices[t.labels().index(lab)].dim for lab in ("u", "d", "l", "r")
        }
        assert dims == expected, f"{name}: {dims}"
        assert t.indices[t.labels().index("phys")].dim == 2
        assert _SUState.from_pair(*pair).max_D == 4, name
