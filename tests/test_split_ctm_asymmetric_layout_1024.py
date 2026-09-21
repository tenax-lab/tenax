"""#1024: every chi leg of the split-CTM env must be seeded from ONE axis of A.

Each chi leg is seeded by tiling one of ``A``'s virtual legs
(``_derive_charges``), and the chi bonds form a **ring**: within a cell
``C1.c1_d`` meets ``T4.t4_d`` and ``C4.c4_u`` meets ``T4.t4_u``, and across
cells ``T4(above).t4_u`` meets ``T4(below).t4_d``.  Chaining those, every chi
leg -- both corner legs, and both ends of every edge -- is forced onto a single
layout.  Matching seams *pairwise* satisfies each constraint locally and still
admits a global inconsistency, which is how the first round of fixes left the
2x2 plaquette projector dying with a shape error.

Two traps make this easy to get wrong, and both are exercised below.

* ``_derive_charges`` tiles the charge **array**, so it is order-sensitive:
  ``[0,1,0]`` tiles to ``{0: 11, 1: 5}`` at chi=16 while ``[1,0,0]`` tiles to
  ``{0: 10, 1: 6}``.  "An axis with the same charges" is therefore not "the
  same axis".
* ``u`` and ``d`` are opposite ends of one lattice bond, so they *always* agree
  as multisets -- which makes substituting one for the other look safe -- but
  simple update leaves them in different orders.  A fixture that passes the
  same array for both cannot reach this case at all.

None of this is contrived.  It is what simple update produces as soon as the
truncation may discover the bond charges instead of being pinned to the initial
guess (#878).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
from tenax.algorithms._split_ctm_tensor_init import initialize_split_ctm_tensor_env
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor

jax.config.update("jax_enable_x64", True)

_SYM = FermionParity()
_PHYS = np.array([0, 1], dtype=np.int32)

#: ``{even:2, odd:1}`` -- the layout ``_build_initial_fpeps_tensor`` seeds at D=3.
_VERT = np.array([0, 1, 0], dtype=np.int32)
#: ``{even:1, odd:2}`` -- where the horizontal bonds actually land once the
#: truncation is free to choose (#878).
_HORIZ = np.array([0, 1, 1], dtype=np.int32)
#: Same MULTISET as ``_VERT``, different ORDER.  This is the case a
#: multiset-level fixture cannot reach, and it is the real one: ``u`` and ``d``
#: are opposite ends of a lattice bond, so they always agree as multisets, but
#: simple update leaves them in different orders.  ``_derive_charges`` tiles the
#: array, so ``[0,1,0]`` gives ``{0: 11, 1: 5}`` at chi=16 while ``[1,0,0]``
#: gives ``{0: 10, 1: 6}`` -- which is exactly the 10-vs-11 mismatch measured on
#: the D=3 fermionic sweep.
_VERT_REORDERED = np.array([1, 0, 0], dtype=np.int32)

_CHI = 16

#: Each chi bond of the split env, as ``(corner, corner_leg, edge, edge_leg)``.
#: The four horizontal seams are the ones #1024 breaks.
_CHI_BONDS = (
    ("C1", "c1_r", "T1_ket", "t1k_l"),
    ("C2", "c2_l", "T1_bra", "t1b_r"),
    ("C4", "c4_r", "T3_ket", "t3k_r"),
    ("C3", "c3_l", "T3_bra", "t3b_l"),
    ("C1", "c1_d", "T4_ket", "t4k_d"),
    ("C2", "c2_d", "T2_ket", "t2k_u"),
    ("C3", "c3_u", "T2_bra", "t2b_d"),
    ("C4", "c4_u", "T4_bra", "t4b_u"),
)


def _site(u, d, ll, r, seed):
    idx = (
        TensorIndex.from_charges(_SYM, u, FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(_SYM, d, FlowDirection.IN, label="d"),
        TensorIndex.from_charges(_SYM, ll, FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(_SYM, r, FlowDirection.IN, label="r"),
        TensorIndex.from_charges(_SYM, _PHYS, FlowDirection.IN, label="phys"),
    )
    return SymmetricTensor.random_normal(idx, jax.random.PRNGKey(seed))


#: ``(label, u, d, l, r)``.
#:
#: ``uniform`` is the control -- it worked before any of this and must keep
#: working.  ``direction-dependent`` is the multiset-level case (#1024 as
#: originally filed).  ``reordered-vertical`` is the case a multiset-level
#: fixture cannot reach: every leg carries ``{even:2, odd:1}``, so nothing
#: "looks" asymmetric, yet ``u`` and ``d`` tile differently because their charge
#: ORDER differs.  It is the one the real fermionic sweep produces.
_FIXTURES = (
    ("uniform", _VERT, _VERT, _VERT, _VERT),
    ("direction-dependent", _VERT, _VERT, _HORIZ, _HORIZ),
    ("reordered-vertical", _VERT, _VERT_REORDERED, _VERT, _VERT),
)


def _layout(tensor, label):
    """The leg's charge multiset, as ``{charge: multiplicity}``."""
    for idx in tensor.indices:
        if idx.label == label:
            charges = np.asarray(idx.charges)
            uniq, counts = np.unique(charges, return_counts=True)
            return dict(zip(uniq.tolist(), counts.tolist()))
    raise AssertionError(f"{label!r} not found on {tensor.labels()}")


def _multiset(arr):
    uniq, counts = np.unique(np.asarray(arr), return_counts=True)
    return dict(zip(uniq.tolist(), counts.tolist()))


def test_the_fixtures_reach_the_cases_they_claim_to():
    """Regime guard: without this the seam tests can pass vacuously.

    Two distinct traps, and the second is the one that let this bug survive a
    first round of fixes:

    * ``direction-dependent`` needs ``_VERT`` and ``_HORIZ`` to still differ
      *after tiling to chi* -- two layouts can differ at D=3 and tile to the
      same multiset, which would make every seam agree by accident.
    * ``reordered-vertical`` needs ``_VERT`` and ``_VERT_REORDERED`` to be the
      same multiset (so nothing looks asymmetric) yet tile *differently* (so
      the seam is genuinely exercised).  A fixture that passes the same array
      for ``u`` and ``d`` cannot reach this at all.
    """
    from tenax.algorithms._ctm_utils import _derive_charges

    assert _multiset(_derive_charges(_VERT, _CHI)) != _multiset(
        _derive_charges(_HORIZ, _CHI)
    ), f"_VERT and _HORIZ tile to the same multiset at chi={_CHI}"

    assert _multiset(_VERT) == _multiset(_VERT_REORDERED), (
        "_VERT_REORDERED must be a REORDERING of _VERT, else it is just "
        "another direction-dependent case and tests nothing new"
    )
    assert _multiset(_derive_charges(_VERT, _CHI)) != _multiset(
        _derive_charges(_VERT_REORDERED, _CHI)
    ), (
        f"_VERT and _VERT_REORDERED tile identically at chi={_CHI}, so the "
        f"order-sensitivity of _derive_charges is not exercised"
    )


@pytest.mark.parametrize(("label", "u", "d", "ll", "r"), _FIXTURES)
@pytest.mark.parametrize(("corner", "corner_leg", "edge", "edge_leg"), _CHI_BONDS)
def test_both_ends_of_every_chi_bond_carry_the_same_layout(
    corner, corner_leg, edge, edge_leg, label, u, d, ll, r
):
    """The invariant itself, checked at init -- before any sweep runs.

    Asserts the charge *multiset* of the leg, not its total dimension: both
    ends are ``chi`` wide either way, and it is the per-sector split that
    disagrees (measured ``{0: 11, 1: 5}`` against ``{0: 6, 1: 10}`` on the
    horizontal seams, and ``{0: 11, 1: 5}`` against ``{0: 10, 1: 6}`` on the
    vertical ones).  A dimension check passes on the broken env.
    """
    A = _site(u, d, ll, r, 0)
    env = initialize_split_ctm_tensor_env(A, _CHI, _CHI)

    got = _layout(getattr(env, corner), corner_leg)
    want = _layout(getattr(env, edge), edge_leg)
    assert got == want, (
        f"[{label}] chi seam {corner}.{corner_leg} <-> {edge}.{edge_leg} "
        f"disagrees: {got} vs {want}.  Both ends must tile the SAME AXIS of A "
        f"-- not merely an axis carrying the same charges, since "
        f"_derive_charges tiles the array and is order-sensitive (#1024)."
    )


@pytest.mark.slow
@pytest.mark.parametrize(("label", "u", "d", "ll", "r"), _FIXTURES)
def test_the_2x2_split_ctm_runs_on_every_layout(label, u, d, ll, r):
    """End to end: the sweep itself, which is where #1024 surfaced.

    The uniform arm is the control -- it passed before any of this, so a
    regression that breaks it is distinguishable from the bug being fixed here.

    ``slow`` so the required ``-m core`` gate skips these: they are nearly all
    of the file's runtime (28 tests, 689s) while the init-time invariant above
    catches the same defect in seconds.  ``conftest`` withholds the file's
    ``core`` marker from any item carrying an explicit ``slow`` marker, so
    these stay in the full suite without weighing down the gate.

    They are not redundant with the init check, though: ``reordered-vertical``
    passed every init seam *and still crashed here*, because the chi bonds
    form a ring and pairwise agreement does not imply global agreement.  That
    is the defect this arm exists to catch.
    """
    A, B = _site(u, d, ll, r, 0), _site(u, d, ll, r, 1)
    env_A, env_B = ctm_split_tensor_2site(A, B, _CHI, max_iter=4, conv_tol=1e-6)
    assert env_A is not None and env_B is not None
