"""#1024: the split-CTM env must be consistent when u/d and l/r layouts differ.

Every chi bond of the environment has two ends -- a corner leg and an edge leg
-- and both are seeded by tiling one of ``A``'s virtual legs
(``_derive_charges``).  If the two ends tile *different* legs, the seam is only
contractible when those legs happen to carry the same charge layout.  A uniform
iPEPS hides the defect; a state whose horizontal and vertical bonds differ does
not, and the 2x2 plaquette projector dies with a shape error.

That is not a contrived state.  It is what simple update produces as soon as the
truncation is allowed to discover the bond charges rather than being pinned to
the initial guess (#878): measured on ``fpeps()`` at D=3, ``u,d`` stay
``{even:2, odd:1}`` while ``l,r`` move to ``{even:1, odd:2}``.

The fused CTM already states and satisfies this invariant -- ``_STD_EDGE_SPECS``
annotates each chi leg with the corner it meets and uses that corner's reference
axis.  These tests hold the split path to the same rule.
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


def _site(vert, horiz, seed):
    idx = (
        TensorIndex.from_charges(_SYM, vert, FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(_SYM, vert, FlowDirection.IN, label="d"),
        TensorIndex.from_charges(_SYM, horiz, FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(_SYM, horiz, FlowDirection.IN, label="r"),
        TensorIndex.from_charges(_SYM, _PHYS, FlowDirection.IN, label="phys"),
    )
    return SymmetricTensor.random_normal(idx, jax.random.PRNGKey(seed))


def _layout(tensor, label):
    """The leg's charge multiset, as ``{charge: multiplicity}``."""
    for idx in tensor.indices:
        if idx.label == label:
            charges = np.asarray(idx.charges)
            uniq, counts = np.unique(charges, return_counts=True)
            return dict(zip(uniq.tolist(), counts.tolist()))
    raise AssertionError(f"{label!r} not found on {tensor.labels()}")


def test_the_fixture_actually_has_direction_dependent_layouts():
    """Regime guard: without this the other two tests assert nothing.

    ``_VERT`` and ``_HORIZ`` must disagree, and must disagree *after* tiling to
    ``chi`` -- two layouts can differ at D=3 and still tile to the same
    multiset, which would make the seam agree by accident and leave the
    invariant untested.
    """
    A = _site(_VERT, _HORIZ, 0)
    assert _layout(A, "u") != _layout(A, "l"), (
        "fixture is uniform -- #1024 cannot be reached"
    )

    from tenax.algorithms._ctm_utils import _derive_charges

    tiled_vert = _derive_charges(_VERT, _CHI)
    tiled_horiz = _derive_charges(_HORIZ, _CHI)

    def multiset(arr):
        uniq, counts = np.unique(np.asarray(arr), return_counts=True)
        return dict(zip(uniq.tolist(), counts.tolist()))

    assert multiset(tiled_vert) != multiset(tiled_horiz), (
        f"the two layouts tile to the same multiset at chi={_CHI} "
        f"({multiset(tiled_vert)}), so every seam would agree by accident"
    )


@pytest.mark.parametrize(("corner", "corner_leg", "edge", "edge_leg"), _CHI_BONDS)
def test_both_ends_of_every_chi_bond_carry_the_same_layout(
    corner, corner_leg, edge, edge_leg
):
    """The invariant itself, checked at init -- before any sweep runs.

    Asserts the charge *multiset*, not the total dimension: both ends are
    ``chi`` wide either way, and it is the per-sector split that disagrees
    (measured ``{0: 11, 1: 5}`` on the corners against ``{0: 6, 1: 10}`` on the
    horizontal edges).  A dimension check passes on the broken env.
    """
    A = _site(_VERT, _HORIZ, 0)
    env = initialize_split_ctm_tensor_env(A, _CHI, _CHI)

    got = _layout(getattr(env, corner), corner_leg)
    want = _layout(getattr(env, edge), edge_leg)
    assert got == want, (
        f"chi seam {corner}.{corner_leg} <-> {edge}.{edge_leg} disagrees: "
        f"{got} vs {want}.  Both ends must tile the same leg of A (#1024)."
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("label", "horiz"),
    [("uniform", _VERT), ("direction-dependent", _HORIZ)],
)
def test_the_2x2_split_ctm_runs_on_both_layouts(label, horiz):
    """End to end: the sweep itself, which is where #1024 surfaced.

    The uniform arm is the control -- it passed before the fix, so a regression
    that breaks it is distinguishable from the bug being fixed here.

    ``slow`` so the required ``-m core`` gate skips these two: measured 352s
    and 230s against 590s for the whole file, while the init-time invariant
    above catches the same defect in ~3s.  ``conftest`` withholds the file's
    ``core`` marker from any item carrying an explicit ``slow`` marker, so
    this stays in the full suite without weighing down the gate.
    """
    A, B = _site(_VERT, horiz, 0), _site(_VERT, horiz, 1)
    env_A, env_B = ctm_split_tensor_2site(A, B, _CHI, max_iter=4, conv_tol=1e-6)
    assert env_A is not None and env_B is not None
