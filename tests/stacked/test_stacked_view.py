import jax.numpy as jnp
import pytest

from tests.stacked._harness import canonical_tensors


@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_round_trip_bit_exact(name):
    A = dict(canonical_tensors())[name]
    view = A.stacked_blocks()
    B = A.from_stacked_blocks(view)
    assert bool(jnp.array_equal(A._data, B._data)), "round-trip not bit-exact"
    assert A._block_keys == B._block_keys
    assert A._block_shapes == B._block_shapes
    assert A._block_offsets == B._block_offsets


def test_even_D_is_single_shape_group():
    A = dict(canonical_tensors())["ferm_D4"]
    view = A.stacked_blocks()
    assert len(view.groups) == 1
    ((shape, grp),) = view.groups.items()
    assert grp.array.shape[0] == len(A._block_keys)
    assert grp.array.shape[1:] == shape


def test_stacked_block_values_match_get_block():
    A = dict(canonical_tensors())["ferm_D2"]
    view = A.stacked_blocks()
    for shape, grp in view.groups.items():
        for j, key in enumerate(grp.keys):
            idx = A._block_keys.index(key)
            assert bool(jnp.array_equal(grp.array[j], A._get_block(idx)))
