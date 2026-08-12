"""The symmetry core must not disagree with itself about charges (#799).

Two silent-wrong-answer defects, both confirmed by direct execution:

1. **The conservation law wraps at int32.**  U(1) charges are *unbounded by
   definition* -- that is what distinguishes them from ``Z_n`` -- but the
   accumulator was forced to ``int32``, so four legs of ``2**30`` fused to
   ``0`` and a nonconserving block was reported conserved.  The failure is
   silent and fails **open**, i.e. in the direction of accepting bad data.

2. **A non-canonical block key passes validation and then vanishes.**  ``#733``
   canonicalises the *index*, but a block keyed with the caller's original
   representative still validated -- fusion reduces modulo ``n``, so the key is
   genuinely conserved -- and then ``todense()`` found no matching sector and
   returned zero.  Same caller, same charge, two different answers depending on
   which representative they happened to write.

The fix keeps charge *storage* at int32, which is a pervasive convention
(``TensorIndex.__post_init__`` re-forces it and ``test_index.py`` asserts it),
and widens only the *accumulator* inside the conservation law.  Storage of one
charge was never the problem; summing several of them in int32 was.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex, _net_charges
from tenax.core.symmetry import (
    FermionicU1,
    FermionParity,
    ProductSymmetry,
    U1Symmetry,
    ZnSymmetry,
)
from tenax.core.tensor import SymmetricTensor

#: Four of these fuse to 2**32, which is 0 in int32 and non-zero in int64.
BIG = 2**30


# ---------------------------------------------------------------------------
# 1. The conservation law must not wrap.
# ---------------------------------------------------------------------------


def test_the_scalar_conservation_law_does_not_wrap_at_int32():
    """``is_conserved`` returned True for a sum of ``2**32``."""
    u1 = U1Symmetry()

    assert u1.net_charge([BIG] * 4, [1] * 4) == 4 * BIG
    assert not u1.is_conserved([BIG] * 4, [1] * 4)


def test_the_vectorised_conservation_law_does_not_wrap_either():
    """``_net_charges`` is the copy that ``_validate`` actually calls.

    The issue names only the scalar ``net_charge``.  Fixing that alone would
    leave the hot path -- the one deciding which blocks enter a tensor -- still
    wrapping, which is the more consequential of the two.
    """
    u1 = U1Symmetry()
    indices = tuple(
        TensorIndex.from_charges(u1, [BIG], FlowDirection.IN, label=f"l{i}")
        for i in range(4)
    )

    assert int(_net_charges(indices, [(BIG,) * 4])[0]) == 4 * BIG


def test_a_nonconserving_block_is_rejected():
    """The consequence, at the boundary a user actually crosses.

    Four IN legs of ``2**30`` have net charge ``2**32``; the tensor was built
    without complaint and every downstream contraction then treated it as a
    conserving object.
    """
    u1 = U1Symmetry()
    indices = tuple(
        TensorIndex.from_charges(u1, [BIG], FlowDirection.IN, label=f"l{i}")
        for i in range(4)
    )

    with pytest.raises(ValueError, match="violates charge conservation"):
        SymmetricTensor({(BIG,) * 4: jnp.ones((1, 1, 1, 1))}, indices)


def test_an_ordinary_conserving_block_still_builds():
    """Negative control: widening must not start rejecting valid tensors."""
    u1 = U1Symmetry()
    a = TensorIndex.from_charges(u1, [1], FlowDirection.IN, label="a")
    b = TensorIndex.from_charges(u1, [1], FlowDirection.OUT, label="b")

    t = SymmetricTensor({(1, 1): jnp.ones((1, 1))}, (a, b))

    assert t._block_keys == ((1, 1),)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_zn_is_unaffected_by_the_widening(n):
    """``Z_n`` reduces mod ``n``, so its charges are bounded either way."""
    zn = ZnSymmetry(n)

    assert zn.is_conserved([1, n - 1], [1, 1]) == ((1 + n - 1) % n == 0)
    assert zn.net_charge([BIG, BIG], [1, -1]) == 0


def test_product_symmetry_charges_stay_packed():
    """The sharp negative control on the widening.

    ``ProductSymmetry`` packs two int16 charges into one int32.  Widening its
    accumulator would corrupt that encoding, so it must keep int32 -- and its
    charges are bounded by construction, so it does not need widening.
    """
    prod = ProductSymmetry(U1Symmetry(), FermionParity())
    q = prod.encode(3, 1)

    assert prod.decode(q) == (3, 1)
    assert prod.is_conserved([q, prod.dual(np.array([q]))[0]], [1, 1])


def test_the_charge_storage_dtype_is_unchanged():
    """int32 storage is a codebase-wide convention; only the accumulator moved.

    ``TensorIndex.__post_init__`` re-forces int32 and ``test_index.py`` asserts
    it, so a fix that widened stored charges would fight the whole tree.
    """
    idx = TensorIndex.from_charges(U1Symmetry(), [-1, 0, 1], FlowDirection.IN, "a")

    assert idx.charges.dtype == np.int32
    assert idx.sectors.dtype == np.int32


def test_fermionic_u1_widens_too():
    """Same unbounded-charge argument as U(1); the issue names it explicitly."""
    f = FermionicU1()

    assert f.net_charge([BIG] * 4, [1] * 4) == 4 * BIG
    assert not f.is_conserved([BIG] * 4, [1] * 4)


# ---------------------------------------------------------------------------
# 2. A block key and its index must agree on the representative.
# ---------------------------------------------------------------------------


def _z2_pair():
    z2 = ZnSymmetry(2)
    return (
        TensorIndex.from_charges(z2, [-1, 1], FlowDirection.IN, label="a"),
        TensorIndex.from_charges(z2, [-1, 1], FlowDirection.OUT, label="b"),
    )


def test_the_index_canonicalises_its_own_sectors():
    """Precondition (#733): the index already reduces ``-1`` to ``1`` mod 2.

    That is what creates the mismatch -- the index moved and the block key did
    not.
    """
    i_in, _i_out = _z2_pair()

    assert set(i_in.charges.tolist()) == {1}


def test_a_non_canonical_block_key_keeps_its_data():
    """``(-1, 1)`` used to validate and then ``todense()`` to exactly zero."""
    i_in, i_out = _z2_pair()

    non_canonical = SymmetricTensor({(-1, 1): jnp.ones((1, 1))}, (i_in, i_out))

    assert float(non_canonical.todense().sum()) != 0.0


def test_both_representatives_of_one_sector_agree():
    """The defect stated as the property it violates: same charge, same answer."""
    i_in, i_out = _z2_pair()

    a = SymmetricTensor({(-1, 1): jnp.ones((1, 1))}, (i_in, i_out))
    b = SymmetricTensor({(1, 1): jnp.ones((1, 1))}, (i_in, i_out))

    assert a._block_keys == b._block_keys
    assert np.array_equal(np.asarray(a.todense()), np.asarray(b.todense()))


def test_two_representatives_of_the_same_sector_in_one_dict_are_rejected():
    """Silently merging them would be a different silent wrong answer.

    Once keys are canonicalised, ``(-1, 1)`` and ``(1, 1)`` name the same
    block.  Summing them, or letting one overwrite the other by dict order, is
    exactly the class of failure this issue is about -- so it raises.
    """
    i_in, i_out = _z2_pair()

    with pytest.raises(ValueError, match="same sector"):
        SymmetricTensor(
            {(-1, 1): jnp.ones((1, 1)), (1, 1): jnp.full((1, 1), 2.0)},
            (i_in, i_out),
        )


def test_u1_keys_are_untouched():
    """Negative control: U(1) has no reduction, so no key may move."""
    u1 = U1Symmetry()
    a = TensorIndex.from_charges(u1, [-1, 1], FlowDirection.IN, label="a")
    b = TensorIndex.from_charges(u1, [-1, 1], FlowDirection.OUT, label="b")

    t = SymmetricTensor({(-1, -1): jnp.ones((1, 1)), (1, 1): jnp.ones((1, 1))}, (a, b))

    assert set(t._block_keys) == {(-1, -1), (1, 1)}


# ---------------------------------------------------------------------------
# 3. The abstract contract for ``fuse``: both operands are arrays.
# ---------------------------------------------------------------------------


def test_canonicalize_charges_passes_array_operands_to_fuse():
    """The default passed a *scalar* identity, so a conforming subclass breaks.

    ``fuse``'s documented contract is that both operands are arrays of shape
    ``(D,)``.  A subclass that relies on it -- indexing, or checking ``.shape``
    -- fails the moment the no-override default is invoked.
    """
    seen: list[tuple] = []

    class Recording(U1Symmetry):
        def fuse(self, charges_a, charges_b):
            seen.append((np.ndim(charges_a), np.ndim(charges_b)))
            return super().fuse(charges_a, charges_b)

        # force the base implementation rather than U1's identity override
        canonicalize_charges = U1Symmetry.__mro__[1].canonicalize_charges

    Recording().canonicalize_charges(np.array([1, 2, 3], dtype=np.int32))

    assert seen, "the default implementation did not call fuse at all"
    assert all(a == 1 and b == 1 for a, b in seen), (
        f"fuse received a scalar operand: ndims {seen}"
    )
