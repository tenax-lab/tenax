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
from tenax.core.symmetry import ZnSymmetry
from tenax.core.tensor import SymmetricTensor

jax.config.update("jax_enable_x64", True)

#: Bosonic Z2, not ``FermionParity``: the seeding layout is symmetry-agnostic
#: (the same 0/1 charge arrays), and since #1035 step 4 the split CTM refuses
#: fermionic input -- its split <-> fused conversions are sign-free -- so a
#: fermionic fixture would only test the refusal.  The layouts below are still
#: the ones the fermionic simple update produces.
_SYM = ZnSymmetry(2)
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
#: chi for the two full 2x2 split sweeps.  16 made them the dearest tests in
#: the slow bucket (>40 min for the five on bosonic Z2).  Not 8 or 9:
#: ``_derive_charges`` tiles ``[0,1,0]`` and ``[1,0,0]`` to different multisets
#: only when chi % 3 == 1, so below 10 ``reordered-vertical`` goes vacuous --
#: ``test_the_fixtures_reach_the_cases_they_claim_to`` checks both chis.
_SWEEP_CHI = 10

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

    assert _multiset(_VERT) == _multiset(_VERT_REORDERED), (
        "_VERT_REORDERED must be a REORDERING of _VERT, else it is just "
        "another direction-dependent case and tests nothing new"
    )
    for chi in (_CHI, _SWEEP_CHI):
        assert _multiset(_derive_charges(_VERT, chi)) != _multiset(
            _derive_charges(_HORIZ, chi)
        ), f"_VERT and _HORIZ tile to the same multiset at chi={chi}"
        assert _multiset(_derive_charges(_VERT, chi)) != _multiset(
            _derive_charges(_VERT_REORDERED, chi)
        ), (
            f"_VERT and _VERT_REORDERED tile identically at chi={chi}, so the "
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


def test_every_cell_of_a_multisite_env_shares_one_chi_seed():
    """The invariant does not stop at a cell boundary.

    A 2x2 plaquette spans four cells, so ``Q_TL.chi_R`` (cell TL's ``T1.t1_r``)
    contracts against ``Q_TR.chi_L`` (cell TR's ``T1.t1_l``).  Seeding each
    cell's env from *its own* site tensor therefore lets two sublattices
    disagree even when each one is internally consistent -- measured on a D=3
    fermionic pair, sublattice A seeded ``{0: 11, 1: 5}`` throughout and
    sublattice B ``{0: 10, 1: 6}``, and the plaquette died on 11-vs-10.

    The two cells here differ ONLY in the order of their axis-0 charges, which
    is what makes this case unreachable by the fixtures above: they build both
    cells from the same arrays, so every cell agrees trivially.

    Sorting the seeds would not be enough either.  On a checkerboard ``A.u``
    pairs with ``B.d`` and ``B.u`` with ``A.d`` -- different bonds, which need
    not share a multiset -- so one designated array is the only thing that makes
    every cell agree.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _initialize_split_multisite_env,
    )

    cell_a = _site(_VERT, _VERT, _VERT, _VERT, 0)
    cell_b = _site(_VERT_REORDERED, _VERT_REORDERED, _VERT, _VERT, 1)

    # Regime guard: the two cells must actually seed differently by default,
    # else the test cannot observe the defect.
    from tenax.algorithms._ctm_utils import _derive_charges

    assert _multiset(_derive_charges(_VERT, _CHI)) != _multiset(
        _derive_charges(_VERT_REORDERED, _CHI)
    ), "the two cells tile identically; this test would pass vacuously"

    envs = _initialize_split_multisite_env({(0, 0): cell_a, (1, 0): cell_b}, _CHI, _CHI)
    seeds = {c: _layout(e.C1, "c1_d") for c, e in envs.items()}
    assert len(set(map(str, seeds.values()))) == 1, (
        f"cells seed different chi layouts: {seeds}.  Every chi leg in a "
        f"multisite env must tile ONE array -- the chi ring spans cells (#1024)."
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
    env_A, env_B = ctm_split_tensor_2site(A, B, _SWEEP_CHI, max_iter=4, conv_tol=1e-6)
    assert env_A is not None and env_B is not None


@pytest.mark.slow
def test_the_2x2_split_ctm_runs_when_the_two_cells_seed_differently():
    """End to end for the cross-cell case, which the arms above cannot reach.

    The sublattices differ only in the ORDER of their axis-0 charges.  This is
    what the real D=3 fermionic sweep produces, and it is what kept seed 1
    crashing after the within-cell ring fix had landed.
    """
    A = _site(_VERT, _VERT, _VERT, _VERT, 0)
    B = _site(_VERT_REORDERED, _VERT_REORDERED, _VERT, _VERT, 1)
    env_A, env_B = ctm_split_tensor_2site(A, B, _SWEEP_CHI, max_iter=4, conv_tol=1e-6)
    assert env_A is not None and env_B is not None


# --------------------------------------------------------------------- #
# Different MULTISET across the sublattices, on a fixture verified legal #
# first.  The cross-cell arms above differ only in charge ORDER, and     #
# their fixtures leave the vertical bonds unpaired (`A.u=[0,1,0]` against#
# `B.d=[1,0,0]`), so they exercise the seed invariant but not a state    #
# the lattice could actually produce.                                    #
# --------------------------------------------------------------------- #

#: Checkerboard pairing is ``A.d<->B.u`` and ``B.d<->A.u`` (README:363), so a
#: LEGAL state can give the two sublattices different ``u`` layouts: B's ``u``
#: is the partner of A's ``d``, not of A's ``u``.
_SUB_X = np.array([0, 1, 0], dtype=np.int32)
_SUB_Y = np.array([0, 1, 1], dtype=np.int32)


def _legal_checkerboard_pair():
    """A/B whose four bonds all pair, with the sublattices' ``u`` differing."""
    return (
        _site(_SUB_X, _SUB_Y, _HORIZ, _HORIZ, 0),
        _site(_SUB_Y, _SUB_X, _HORIZ, _HORIZ, 1),
    )


def test_the_legal_sublattice_fixture_reaches_the_case_it_claims_to():
    """Regime guard: the bonds must pair AND the seeds must disagree.

    The pairing half matters on its own.  If the two ends of a vertical bond
    carry different charges the state is not something the lattice can hold,
    and a downstream failure would be correct behaviour rather than a defect
    -- so a test built on an unpaired fixture cannot distinguish the two.
    """
    from tenax.algorithms._ctm_utils import _derive_charges

    A, B = _legal_checkerboard_pair()
    assert _layout(A, "u") == _layout(B, "d"), "A.u<->B.d unpaired: illegal state"
    assert _layout(A, "d") == _layout(B, "u"), "A.d<->B.u unpaired: illegal state"
    assert _multiset(_derive_charges(_SUB_X, _CHI)) != _multiset(
        _derive_charges(_SUB_Y, _CHI)
    ), "both sublattices tile alike -- the envs would agree by accident"


#: Both legs of every corner, from ``_CORNER_SPECS``.  One leg each is not
#: enough: each corner carries one horizontal and one vertical leg, so a
#: partial initializer that shared the seed across cells for the horizontal
#: legs while keeping a per-cell seed for the vertical ones would satisfy a
#: horizontal-only assertion *and* the same-cell seam checks above, and still
#: produce incompatible vertical cross-cell contractions.  The only thing left
#: to catch that would be the end-to-end arm below, which is ``slow`` and so
#: sits outside the required gate this file is registered for -- which makes
#: this the assertion that has to discriminate.
_CORNER_LEGS = (
    ("C1", "c1_d"),
    ("C1", "c1_r"),
    ("C2", "c2_l"),
    ("C2", "c2_d"),
    ("C3", "c3_u"),
    ("C3", "c3_l"),
    ("C4", "c4_r"),
    ("C4", "c4_u"),
)


#: The OUTER chi leg of each edge half -- the end that contracts across a cell
#: boundary.  Corners alone are not enough: wiring the shared seed into
#: ``_init_symmetric_corner`` but dropping it from the edge builders leaves all
#: eight corner comparisons passing while every one of these differs.  Verified
#: by simulating exactly that: the corner assertion PASSED and all eight of
#: these read ``{0: 11, 1: 5}`` against ``{0: 6, 1: 10}``.
#:
#: Deliberately NOT the ``*_ket`` / ``*_bra`` D legs (``u_ket``, ``d_ket``, ...).
#: Those are the site's own virtual bonds and are *supposed* to differ between
#: sublattices -- on this fixture ``u`` is ``{0:2, 1:1}`` on A and ``{0:1, 1:2}``
#: on B.  Asserting them equal would fail correct code.
#:
#: Also not the ``*_I`` interlayer bonds: they happen to agree, but they are
#: internal to one edge's ket/bra split rather than contracted across cells, so
#: requiring agreement there would assert more than the invariant.
_EDGE_CHI_LEGS = (
    ("T1_ket", "t1k_l"),
    ("T1_bra", "t1b_r"),
    ("T2_ket", "t2k_u"),
    ("T2_bra", "t2b_d"),
    ("T3_ket", "t3k_r"),
    ("T3_bra", "t3b_l"),
    ("T4_ket", "t4k_d"),
    ("T4_bra", "t4b_u"),
)


def test_every_edge_chi_leg_agrees_across_a_legal_sublattice_split():
    """The edges too, not only the corners.

    The same-cell seam test earlier cannot cover this: it builds each
    environment with no cross-cell seed at all, so that env's corner and edge
    legs both derive from its own site tensor and agree trivially.  Only a
    cross-cell comparison of the edges themselves detects a seed wired into
    the corners but dropped from the edge builders.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _initialize_split_multisite_env,
    )

    A, B = _legal_checkerboard_pair()
    envs = _initialize_split_multisite_env({(0, 0): A, (1, 0): B}, _CHI, _CHI)
    for edge, leg in _EDGE_CHI_LEGS:
        got = _layout(getattr(envs[(0, 0)], edge), leg)
        want = _layout(getattr(envs[(1, 0)], edge), leg)
        assert got == want, f"{edge}.{leg} differs across sublattices: {got} vs {want}"


def test_every_corner_leg_agrees_across_a_legal_sublattice_split():
    """Both legs of all four corners: a partial fix must not pass.

    Measured before the fix: ``{0: 11, 1: 5}`` on sublattice A against
    ``{0: 6, 1: 10}`` on B.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _initialize_split_multisite_env,
    )

    A, B = _legal_checkerboard_pair()
    envs = _initialize_split_multisite_env({(0, 0): A, (1, 0): B}, _CHI, _CHI)
    for corner, leg in _CORNER_LEGS:
        got = _layout(getattr(envs[(0, 0)], corner), leg)
        want = _layout(getattr(envs[(1, 0)], corner), leg)
        assert got == want, (
            f"{corner}.{leg} differs across sublattices: {got} vs {want}"
        )


@pytest.mark.slow
def test_the_2x2_split_ctm_runs_on_a_legal_sublattice_dependent_state():
    """End to end on a state the lattice can actually hold.

    ``chi=5`` deliberately: the crash this reproduces
    (``ValueError: Size of label 'c' for operand 1 (3) does not match previous
    terms (2)``) needs a chi where the two seeds tile to different sector
    counts, and small chi reaches it in one sweep.
    """
    A, B = _legal_checkerboard_pair()
    env_A, env_B = ctm_split_tensor_2site(A, B, 5, max_iter=1, conv_tol=1e-6)
    assert env_A is not None and env_B is not None


#: A chi seed that is NOT any of the site tensor's own axes, so supplying it
#: genuinely changes the environment.  Needed because the first version of the
#: test below seeded with `A.indices[0].charges` -- which is what the function
#: already defaults to -- and so passed with the alias silently dropped.
_ALT_SEED = np.array([1, 1, 0], dtype=np.int32)


def _env_fields(env):
    """Every tensor in the env, densified. The seed moves EDGES, not corners."""
    return {f: np.asarray(getattr(env, f).todense()) for f in env._fields}


def _identical(lhs, rhs):
    return lhs.shape == rhs.shape and bool((lhs == rhs).all())


def test_the_alt_seed_actually_changes_the_environment():
    """Regime guard for the alias tests: prove the seed is not inert here.

    Without this, an alias test can pass while the alias is ignored -- which
    is exactly what happened on the first attempt: seeding with the function's
    own default axis and comparing only ``C1..C4`` (which the seed does not
    touch at all) left the 'silently dropped' mutant alive.
    """
    A = _site(_VERT, _VERT, _HORIZ, _HORIZ, seed=0)
    default = _env_fields(initialize_split_ctm_tensor_env(A, _CHI, _CHI))
    seeded = _env_fields(
        initialize_split_ctm_tensor_env(A, _CHI, _CHI, chi_ref_charges=_ALT_SEED)
    )
    differing = [f for f in default if not _identical(default[f], seeded[f])]
    assert differing, (
        "_ALT_SEED leaves the environment unchanged, so the alias tests below "
        "cannot distinguish 'honoured' from 'ignored'"
    )
    # It is the edges that carry the chi seed; corners are seed-independent.
    assert all(f.startswith("T") for f in differing), differing


def test_chi_seed_still_works_as_a_deprecated_alias():
    """``chi_seed`` was renamed to ``chi_ref_charges``; the old name must live.

    ``CHANGELOG.md`` documents ``chi_seed`` by name, so callers outside this
    repository were told to use it.  A bare rename turns every one of those
    calls into ``TypeError`` -- the rename is internal, the keyword is not.

    Asserts *equivalence over every field*, not mere acceptance.  A test that
    only checked "does not raise" passes when the argument is silently
    dropped, which is the worse failure: the caller's seed vanishes and every
    chi leg falls back to the default axis -- the #1024 bug the seed exists to
    prevent.
    """
    A = _site(_VERT, _VERT, _HORIZ, _HORIZ, seed=0)

    with pytest.warns(DeprecationWarning, match="chi_seed is deprecated"):
        aliased = initialize_split_ctm_tensor_env(A, _CHI, _CHI, chi_seed=_ALT_SEED)
    current = initialize_split_ctm_tensor_env(A, _CHI, _CHI, chi_ref_charges=_ALT_SEED)

    lhs, rhs = _env_fields(aliased), _env_fields(current)
    mismatched = [f for f in lhs if not _identical(lhs[f], rhs[f])]
    assert not mismatched, f"alias diverges from the new name on: {mismatched}"


def test_passing_both_the_alias_and_the_new_name_is_refused():
    """Silently preferring one would hide a caller mid-migration passing both."""
    A = _site(_VERT, _VERT, _HORIZ, _HORIZ, seed=0)
    with pytest.raises(TypeError, match="not both"):
        initialize_split_ctm_tensor_env(
            A, _CHI, _CHI, chi_ref_charges=_ALT_SEED, chi_seed=_ALT_SEED
        )
