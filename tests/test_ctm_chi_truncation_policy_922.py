"""The CTM chi bond follows the spectrum; ``base_charges`` is only a floor (#922).

#905 stopped the symmetric CTM deleting its charged environment sectors.  With
that fixed the environment still saturated below the dense reference, and the
gap *grew* with chi instead of shrinking — 9.9e-4 at chi=8, 2.55e-3 at chi=16,
2.58e-3 at chi=24 on a D=2 U(1)-Sz pair, while the dense arm kept improving.
Flat-or-growing disagreement in chi is this project's defect signature (#898).

The cause was the truncation policy, not the contraction.  ``base_charges``
(the double-layer ``u2`` charge list) was tiled by ``_derive_charges`` into a
per-sector *quota*: each sector was capped at its tiled share, and any charge
absent from ``base_charges`` was allocated a share of zero.  On that D=2 pair
the full bond offered ``{-4: 4, -2: 16, 0: 24, 2: 16, 4: 4}`` and the quota kept
``{-2: 4, 0: 8, 2: 4}`` — the ``|q| = 4`` sectors could never be given a slot at
any chi, however much weight they carried.

The policy is now global top-chi with ``base_charges`` reserving one slot per
named charge, which closes the gap to 4.2e-14 at chi=16 and 2.2e-16 at chi=24.
The floor is retained because a charge missing from a bond index cannot be
*re*-created by ``contract()``, which pairs blocks by charge value — dropping a
sector is a deletion, not a truncation.  It is unit-tested here rather than
claimed as a physics improvement: on every fixture measured for #922 the floor
never bound, and the energies with and without it are bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from tenax.algorithms._ctm_utils import _select_chi_slots

BASE = np.array([-2, 0, 0, 2], dtype=np.int32)


# --------------------------------------------------------------------------- #
# The value-aware cut (eager path)                                             #
# --------------------------------------------------------------------------- #


def test_selection_is_global_top_chi_when_the_floor_is_inert():
    """With every base sector already in the top chi, the cut is the plain one."""
    values = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    charges = np.array([0, 2, -2, 0, 2, -2], dtype=np.int32)

    assert _select_chi_slots(values, charges, base_charges=BASE, chi=4) == [0, 1, 2, 3]
    assert _select_chi_slots(values, charges, base_charges=None, chi=4) == [0, 1, 2, 3]


def test_a_charge_outside_base_charges_wins_slots_it_earns():
    """The #922 defect, at the level of the allocator.

    ``q = 4`` is not in ``base_charges``, so the old quota gave it zero slots
    however large its singular values were.  Here it holds the two *largest*.
    """
    values = np.array([9.0, 8.0, 0.3, 0.2, 0.1])
    charges = np.array([4, 4, 0, -2, 2], dtype=np.int32)

    keep = _select_chi_slots(values, charges, base_charges=BASE, chi=4)

    assert 4 in charges[keep], f"q=4 was excluded again: kept {charges[keep].tolist()}"
    assert max(values[keep]) == 9.0


def test_the_floor_is_bounded_by_the_budget():
    """Three floors do not fit in a chi=3 bond alongside anything else.

    Stated so the crowding-out is specified rather than incidental: the floor
    costs at most one slot per *distinct* charge in ``base_charges``, which is
    bounded by D^2 and normally far below chi.  When chi is that small the
    floor wins and the spectrum loses — including, as here, the largest values
    on the bond.
    """
    values = np.array([9.0, 8.0, 0.3, 0.2, 0.1])
    charges = np.array([4, 4, 0, -2, 2], dtype=np.int32)

    keep = _select_chi_slots(values, charges, base_charges=BASE, chi=3)

    assert sorted(charges[keep].tolist()) == [-2, 0, 2]
    assert 9.0 not in values[keep]


def test_the_floor_keeps_a_base_sector_whose_weight_is_below_the_cut():
    """A named charge survives even when its values are numerical noise.

    This is the only behaviour the floor adds over a plain global cut, and the
    reason it exists: ``contract()`` pairs by charge value, so a sector that
    leaves the bond index cannot come back through it.
    """
    values = np.array([1.0, 0.9, 0.8, 0.7, 1e-14, 1e-15])
    charges = np.array([0, 0, 0, 0, 2, -2], dtype=np.int32)

    keep = _select_chi_slots(values, charges, base_charges=BASE, chi=4)

    assert set(charges[keep].tolist()) >= {-2, 0, 2}, (
        f"floor did not survive: kept {charges[keep].tolist()}"
    )
    # ... and without the floor the same spectrum keeps only q=0.
    assert _select_chi_slots(values, charges, base_charges=None, chi=4) == [0, 1, 2, 3]


def test_the_floor_cannot_conjure_a_sector_the_decomposition_omits():
    """A named charge with no entry at all gets no slot — there is none to give.

    The floor rations existing slots; it does not create vectors.  Callers that
    need an absent sector represented have to seed one first, which
    ``_eigh_projector_symmetric`` does and ``_svd_projector_symmetric`` does
    not (pre-existing on both sides of #922; tracked in #929).
    """
    values = np.array([1.0, 0.9, 0.8])
    charges = np.array([0, 0, 0], dtype=np.int32)  # BASE also names -2 and +2

    keep = _select_chi_slots(values, charges, base_charges=BASE, chi=3)

    assert keep == [0, 1, 2]
    assert set(charges[keep].tolist()) == {0}


def test_selection_returns_chi_slots_in_ascending_order():
    values = np.array([0.5, 1.0, 0.2, 0.9, 0.1])
    charges = np.array([0, 2, -2, 0, 2], dtype=np.int32)

    keep = _select_chi_slots(values, charges, base_charges=BASE, chi=3)

    assert len(keep) == 3
    assert keep == sorted(keep)
    assert len(set(keep)) == 3


def test_selection_clamps_to_the_available_slots():
    values = np.array([1.0, 0.5])
    charges = np.array([0, 2], dtype=np.int32)

    assert _select_chi_slots(values, charges, base_charges=BASE, chi=8) == [0, 1]
    assert _select_chi_slots(values, charges, base_charges=BASE, chi=0) == []


# --------------------------------------------------------------------------- #
# End to end: the environment must reach the dense reference                   #
# --------------------------------------------------------------------------- #


def _su_pair(D: int, steps=((0.05, 300), (0.01, 200))):
    """A simple-update-optimised U(1)-Sz pair, charged enough to have structure."""
    import jax

    from tenax.algorithms.ipeps import (
        _make_trotter_gate_tensor,
        heisenberg_gate_u1sz,
        heisenberg_u1sz_init_pair,
    )
    from tenax.algorithms.ipeps_simple_update import (
        _simple_update_checkerboard_sweep,
        _to_physical_pair,
    )

    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    Hs = heisenberg_gate_u1sz()
    lam = None
    for dt, n in steps:
        gate = _make_trotter_gate_tensor(Hs, dt, site_tensor=A)
        A, B, lam = _simple_update_checkerboard_sweep(A, B, gate, D, n, lambdas=lam)
    return _to_physical_pair(A, B, lam)


@pytest.mark.slow
def test_the_symmetric_environment_reaches_the_dense_reference():
    """``DenseTensor`` contraction ignores charge, so E_dense is the reference.

    Pre-#922 this pair converged to 1.8e-07 off the dense arm at chi=16 with
    the gap *growing* in chi; the same run now agrees to ~1e-15.  The chi bond
    also has to carry a charge that ``base_charges`` does not name — the tiled
    quota made that impossible at any chi, which is the whole defect.
    """
    from collections import Counter

    from tenax.algorithms._ctm_tensor import (
        compute_energy_ctm_tensor_2site,
        ctm_tensor_2site,
    )
    from tenax.algorithms.ipeps import heisenberg_gate
    from tenax.core.tensor import DenseTensor

    A, B = _su_pair(D=3)
    gate = heisenberg_gate().todense()
    chi = 16

    envA, envB = ctm_tensor_2site(
        A, B, chi=chi, recipe="2x2", max_iter=200, conv_tol=1e-10
    )
    E_sym = float(compute_energy_ctm_tensor_2site(A, B, envA, envB, gate, d=2))

    Ad = DenseTensor(A.todense(), A.indices)
    Bd = DenseTensor(B.todense(), B.indices)
    envAd, envBd = ctm_tensor_2site(
        Ad, Bd, chi=chi, recipe="2x2", max_iter=200, conv_tol=1e-10
    )
    E_dense = float(compute_energy_ctm_tensor_2site(Ad, Bd, envAd, envBd, gate, d=2))

    assert abs(E_sym - E_dense) < 1e-9, (
        f"E_sym={E_sym!r} vs E_dense={E_dense!r} "
        f"(gap {abs(E_sym - E_dense):.3e}); the chi bond is not following its "
        "own spectrum — see the module docstring."
    )

    # The structural half: base_charges names the double-layer u2 charges, and
    # the converged bond must be free to hold charges outside that set.
    from tenax.algorithms._ctm_tensor_convergence import _get_base_charges
    from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor

    named = {int(q) for q in _get_base_charges(_build_double_layer_tensor(A))}
    on_bond = Counter(int(q) for q in np.asarray(envA.C1.indices[0].charges))
    assert set(on_bond) - named, (
        f"chi bond charges {dict(on_bond)} are all inside base_charges {named}; "
        "the quota is still capping the sector structure"
    )
