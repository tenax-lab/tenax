"""The explicit ``twist`` primitive #555 deferred and never delivered.

#555 removed the contractor's automatic Koszul tracking, on the stated
grounds that planar networks need no signs -- "FermionParity's R-symbol
contributes only at physical line crossings, which planar diagrams have
none of" -- and deferred the non-planar case:

    For future non-planar applications an explicit ``twist`` primitive can
    be added.

It never was, and its absence is not cosmetic.  ``reference_energy_2x2_pbc``
in ``tests/test_fermionic_ed_reference.py`` contracts a **torus**, which is
not planar, and is used as fermionic "ground truth".  Measured on the #995
adjudication, the periodic and planar oracles disagree by up to 13x and
reverse which CTM convention they favour.  A non-planar reference is not
admissible without this operation.

``twist(T, axes)`` multiplies each block by ``(-1)^(sum of the parities of
that block's charges on ``axes``)`` -- the categorical twist, matching
TensorKit's ``twist(t, i)`` and the ``twist(F_west, 3)`` PEPSKit applies
when fusing a ket/bra sandwich.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import BraidingStyle, FermionParity, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

jax.config.update("jax_enable_x64", True)


def _fp_tensor(seed=0):
    sym = FermionParity()
    ch = np.array([0, 1], dtype=np.int32)
    idx = tuple(
        TensorIndex.from_charges(sym, ch.copy(), f, label=lbl)
        for f, lbl in (
            (FlowDirection.OUT, "a"),
            (FlowDirection.IN, "b"),
            (FlowDirection.OUT, "c"),
        )
    )
    return SymmetricTensor.random_normal(idx, jax.random.PRNGKey(seed))


def test_twist_flips_exactly_the_odd_parity_blocks():
    """The defining property: sign is (-1)^(parity on the twisted axes)."""
    T = _fp_tensor()
    Tw = T.twist((0,))
    sym = T.indices[0].symmetry
    assert set(Tw.blocks) == set(T.blocks), "twist must not change block structure"
    for key, blk in T.blocks.items():
        p = int(sym.parity(np.array([key[0]]))[0])
        want = (-1.0) ** p * np.asarray(blk)
        np.testing.assert_allclose(np.asarray(Tw.blocks[key]), want, atol=0, rtol=0)


def test_twist_is_an_involution():
    T = _fp_tensor(1)
    back = T.twist((1,)).twist((1,))
    for key, blk in T.blocks.items():
        np.testing.assert_allclose(np.asarray(back.blocks[key]), np.asarray(blk))


def test_twisting_axes_together_equals_twisting_them_in_sequence():
    """Composition, and therefore order-independence."""
    T = _fp_tensor(2)
    both = T.twist((0, 2))
    seq = T.twist((0,)).twist((2,))
    rev = T.twist((2,)).twist((0,))
    for key in T.blocks:
        np.testing.assert_allclose(
            np.asarray(both.blocks[key]), np.asarray(seq.blocks[key])
        )
        np.testing.assert_allclose(
            np.asarray(both.blocks[key]), np.asarray(rev.blocks[key])
        )


def test_twisting_every_leg_is_the_identity_on_a_parity_even_tensor():
    """Charge conservation makes the total parity even, so all-leg twist is +1.

    This is the invariant that makes the operation safe to apply blindly to
    a closed diagram: it can only act through an *imbalance* across a cut.
    """
    T = _fp_tensor(3)
    allT = T.twist(tuple(range(len(T.indices))))
    for key, blk in T.blocks.items():
        np.testing.assert_allclose(np.asarray(allT.blocks[key]), np.asarray(blk))


def test_twist_is_a_no_op_without_grading():
    """Bosonic symmetry and DenseTensor: nothing to twist."""
    sym = U1Symmetry()
    ch = np.array([0, 1], dtype=np.int32)
    idx = tuple(
        TensorIndex.from_charges(sym, ch.copy(), f, label=lbl)
        for f, lbl in ((FlowDirection.OUT, "a"), (FlowDirection.IN, "b"))
    )
    S = SymmetricTensor.random_normal(idx, jax.random.PRNGKey(4))
    for key, blk in S.blocks.items():
        np.testing.assert_allclose(
            np.asarray(S.twist((0,)).blocks[key]), np.asarray(blk)
        )

    D = DenseTensor(jax.random.normal(jax.random.PRNGKey(5), (2, 2)), idx)
    np.testing.assert_allclose(
        np.asarray(D.twist((0,)).todense()), np.asarray(D.todense())
    )


def test_twist_rejects_an_out_of_range_axis():
    T = _fp_tensor(6)
    with pytest.raises((IndexError, ValueError)):
        T.twist((len(T.indices),))


class _AnyonicStub(FermionParity):
    """Stand-in for a symmetry whose ribbon element is not a sign.

    ``BraidingStyle.ANYONIC`` is declared and "reserved for future use", so
    there is no concrete anyonic symmetry in-tree to test against.  This
    subclass supplies one: the charge arithmetic of ``FermionParity`` with
    the braiding style of the future case.
    """

    @property
    def braiding_style(self) -> BraidingStyle:
        return BraidingStyle.ANYONIC

    @property
    def is_fermionic(self) -> bool:
        return self.braiding_style == BraidingStyle.FERMIONIC


def test_twist_refuses_an_anyonic_symmetry_rather_than_silently_doing_nothing():
    """The ``is_fermionic`` gate must not read as "nothing to twist" here.

    A bosonic no-op is *correct* -- the twist really is the identity with no
    grading.  An anyonic no-op would be silently **wrong**: the symmetry
    declares a ribbon phase this implementation cannot apply.  Failing loudly
    is the difference between an unsupported case and a wrong answer.
    """
    sym = _AnyonicStub()
    ch = np.array([0, 1], dtype=np.int32)
    idx = tuple(
        TensorIndex.from_charges(sym, ch.copy(), f, label=lbl)
        for f, lbl in ((FlowDirection.OUT, "a"), (FlowDirection.IN, "b"))
    )
    T = SymmetricTensor.random_normal(idx, jax.random.PRNGKey(7))
    with pytest.raises(NotImplementedError, match="anyonic"):
        T.twist((0,))


def test_twist_of_no_axes_is_accepted_on_an_anyonic_symmetry():
    """Twisting nothing is the identity for *any* braiding style."""
    sym = _AnyonicStub()
    ch = np.array([0, 1], dtype=np.int32)
    idx = tuple(
        TensorIndex.from_charges(sym, ch.copy(), f, label=lbl)
        for f, lbl in ((FlowDirection.OUT, "a"), (FlowDirection.IN, "b"))
    )
    T = SymmetricTensor.random_normal(idx, jax.random.PRNGKey(8))
    for key, blk in T.twist(()).blocks.items():
        np.testing.assert_allclose(np.asarray(blk), np.asarray(T.blocks[key]))
