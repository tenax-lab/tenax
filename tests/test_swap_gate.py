"""SymmetricTensor.swap_gate — the PR #986 Phase 1 primitive.

The swap gate multiplies each block by ``(-1)**(p_i * p_j)`` for the two
legs it crosses: the Corboz-style build-time encoding of fermionic
exchange statistics.  These tests pin, per the design's Phase 1 gate:

* the hand-computed 2-leg sign pattern (only odd⊗odd flips);
* involution and axis symmetry;
* the graded-transpose cross-check: for an *adjacent* leg exchange the
  Koszul sign of ``transpose`` is exactly ``(-1)**(p_i p_j)``, so
  ``swap_gate`` followed by a sign-free block permutation must equal the
  graded ``transpose`` — this is the mutation anchor (dropping the sign
  in ``swap_gate`` fails it);
* the explicit ``grading`` override: a tensor retyped onto bosonic
  ZnSymmetry(2) reports all-even parity by definition, so the default
  gate is the identity there, and the override (built from the original
  fermionic symmetry) must reproduce the graded default exactly — the
  §3.1/§3.2 retyping contract;
* argument validation (equal/out-of-range axes; incomplete grading map).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, ZnSymmetry
from tenax.core.tensor import SymmetricTensor

_FP = FermionParity()
_Z2 = ZnSymmetry(2)


def _idx(sym, charges, flow, label):
    return TensorIndex.from_charges(
        sym, np.asarray(charges, dtype=np.int32), flow, label=label
    )


def _random_fp_tensor(rank: int, seed: int = 0, dim_per_charge: int = 2):
    """Random FermionParity tensor with both parities on every leg."""
    charges = [0] * dim_per_charge + [1] * dim_per_charge
    indices = tuple(
        _idx(
            _FP,
            charges,
            FlowDirection.OUT if k % 2 == 0 else FlowDirection.IN,
            f"leg{k}",
        )
        for k in range(rank)
    )
    return SymmetricTensor.random_normal(indices=indices, key=jax.random.PRNGKey(seed))


def _retype_to_bosonic_z2(A: SymmetricTensor) -> SymmetricTensor:
    """Block-identical rebuild on bosonic ZnSymmetry(2) (charges {0,1} map 1:1)."""
    new_indices = tuple(
        TensorIndex.from_charges(_Z2, idx.charges, idx.flow, label=idx.label)
        for idx in A.indices
    )
    return SymmetricTensor(dict(A.blocks), new_indices)


def test_two_leg_hand_computed_sign_pattern():
    """Only the odd⊗odd block flips; the other three sectors are untouched."""
    T = _random_fp_tensor(2, seed=1)
    G = T.swap_gate((0, 1))
    assert set(T.blocks) == set(G.blocks)
    for key, block in T.blocks.items():
        expected = -block if (key[0] % 2 and key[1] % 2) else block
        np.testing.assert_array_equal(np.asarray(G.blocks[key]), np.asarray(expected))
    # the regime assertion (#884 rule): the fixture must actually contain
    # an odd⊗odd block, or this test passes with the sign dropped.
    assert any(k[0] % 2 and k[1] % 2 for k in T.blocks), (
        "fixture regime violated: no odd-odd block present"
    )


def test_involution_and_axis_symmetry():
    T = _random_fp_tensor(3, seed=2)
    twice = T.swap_gate((0, 2)).swap_gate((0, 2))
    np.testing.assert_allclose(
        np.asarray(twice._data), np.asarray(T._data), rtol=0, atol=0
    )
    ab = T.swap_gate((1, 2))
    ba = T.swap_gate((2, 1))
    np.testing.assert_array_equal(np.asarray(ab._data), np.asarray(ba._data))


@pytest.mark.parametrize("rank,pair", [(3, (0, 1)), (3, (1, 2)), (4, (2, 3))])
def test_adjacent_transpose_cross_check(rank, pair):
    """swap_gate + sign-free permutation == graded transpose (adjacent swap).

    ``SymmetricTensor.transpose`` applies the Koszul sign
    ``_koszul_sign(parities, perm)``, which for an adjacent exchange
    (i, i+1) is exactly ``(-1)**(p_i * p_{i+1})`` — the swap gate.  This
    is the design's mutation anchor: drop the sign in ``swap_gate`` and
    the odd⊗odd blocks disagree.
    """
    i, j = pair
    perm = list(range(rank))
    perm[i], perm[j] = perm[j], perm[i]
    perm = tuple(perm)

    T = _random_fp_tensor(rank, seed=3 + rank)
    graded = T.transpose(perm)
    gated = T.swap_gate((i, j))
    # regime assertion: at least one block carries odd parity on BOTH
    # swapped legs, else the cross-check cannot see the sign at all.
    assert any(k[i] % 2 and k[j] % 2 for k in T.blocks), "fixture regime violated"
    for key, block in gated.blocks.items():
        new_key = tuple(key[p] for p in perm)
        naive = jnp.transpose(block, perm)
        np.testing.assert_allclose(
            np.asarray(graded.blocks[new_key]), np.asarray(naive), rtol=0, atol=0
        )


def test_bosonic_default_is_identity_and_grading_override_restores_signs():
    """Retyped-bosonic parity() is all-even → default gate is a no-op; the
    explicit grading captured from the fermionic symmetry restores the
    graded result exactly (the §3.2 retyping contract)."""
    T = _random_fp_tensor(2, seed=5)
    B = _retype_to_bosonic_z2(T)

    # default on bosonic: identity (returns without any flip)
    same = B.swap_gate((0, 1))
    np.testing.assert_array_equal(np.asarray(same._data), np.asarray(B._data))

    # override: grading recorded from the fermionic symmetry, per leg
    def gmap(idx):
        return {
            int(q): int(_FP.parity(np.array([q]))[0]) for q in np.unique(idx.charges)
        }

    graded_ref = T.swap_gate((0, 1))
    overridden = B.swap_gate((0, 1), grading=(gmap(T.indices[0]), gmap(T.indices[1])))
    np.testing.assert_array_equal(
        np.asarray(overridden._data), np.asarray(graded_ref._data)
    )
    # regime assertion: the override must have flipped something.
    assert not np.array_equal(np.asarray(overridden._data), np.asarray(B._data)), (
        "fixture regime violated: override flipped nothing"
    )


def test_axes_validation_and_grading_coverage():
    T = _random_fp_tensor(2, seed=7)
    with pytest.raises(ValueError, match="distinct"):
        T.swap_gate((1, 1))
    with pytest.raises(ValueError, match="out of range"):
        T.swap_gate((0, 5))
    with pytest.raises(KeyError, match="no\\s+entry for charge"):
        T.swap_gate((0, 1), grading=({0: 0}, {0: 0, 1: 1}))


def test_input_tensor_unchanged():
    T = _random_fp_tensor(2, seed=9)
    before = np.asarray(T._data).copy()
    T.swap_gate((0, 1))
    np.testing.assert_array_equal(np.asarray(T._data), before)
