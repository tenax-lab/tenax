import jax.numpy as jnp
import pytest

from tenax.contraction.contractor import contract
from tests.stacked._harness import assert_tiered, canonical_tensors


def test_bit_identical_tier_passes_on_equal():
    a = jnp.array([1.0, 2.0, 3.0])
    assert_tiered(a, a, tier="bit")


def test_bit_identical_tier_fails_on_tiny_diff():
    a = jnp.array([1.0, 2.0, 3.0])
    b = a + 1e-15
    with pytest.raises(AssertionError):
        assert_tiered(a, b, tier="bit")


def test_bounded_fp_tier_passes_within_rtol():
    a = jnp.array([1.0, 2.0, 3.0])
    b = a * (1 + 1e-13)
    assert_tiered(a, b, tier="fp")


def test_bounded_fp_tier_fails_outside_rtol():
    a = jnp.array([1.0, 2.0, 3.0])
    b = a * (1 + 1e-6)
    with pytest.raises(AssertionError):
        assert_tiered(a, b, tier="fp")


def test_canonical_tensors_cover_required_cases():
    cases = dict(canonical_tensors())
    assert {
        "ferm_D2",
        "ferm_D4",
        "dense_D2",
        "degenerate_sv",
        "rank_deficient",
    } <= set(cases)


def _phys_double_layer(A):
    """Contract ket A with conjugate bra A.bar() sharing ONLY the physical leg.

    The virtual legs of the bra are relabelled (u/d/l/r -> U/D/L/R) so they
    survive as free output legs, keeping the result a rank-8 SymmetricTensor
    rather than collapsing to a scalar.  ``bar()`` flips the physical-leg flow
    from IN to OUT so the shared ``phys`` label is a valid contraction.
    """
    bra = A.bar().relabels({"u": "U", "d": "D", "l": "L", "r": "R"})
    return contract(A, bra)


def test_comparator_self_validates_against_per_block_vs_batched_paths(monkeypatch):
    A = dict(canonical_tensors())["ferm_D2"]

    # Per-combo path (flag off). Gate is read fresh per call, no cache to clear.
    monkeypatch.setenv("TENAX_BATCH_BLOCKSPARSE", "0")
    per_block = _phys_double_layer(A)
    # Batched path (flag on).
    monkeypatch.setenv("TENAX_BATCH_BLOCKSPARSE", "1")
    batched = _phys_double_layer(A)

    assert per_block._data.size > 1, "double-layer collapsed to a scalar"
    assert per_block.labels() == batched.labels()
    assert_tiered(per_block._data, batched._data, tier="fp")


def test_svd_invariants_smoke():
    from tests.stacked._harness import assert_svd_invariants, canonical_tensors

    M = dict(canonical_tensors())["degenerate_sv"]
    U, S, Vh = jnp.linalg.svd(M, full_matrices=False)
    # reconstruction matches; raw U/Vh never compared
    assert_svd_invariants(M, U, S, Vh, tier="fp")
