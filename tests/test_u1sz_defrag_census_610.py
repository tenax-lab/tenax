"""Static candidate-evaluation predictor for #610 Gate A."""
from examples.census_u1sz_block_shapes_566 import predict_sectors_under_keep


def test_drop_high_sz_sectors_reduces_block_count():
    # block-keys: each key is a per-axis charge tuple; axes (chi_a, chi_b, d2).
    # Keep-set {-1,0,1} on the chi axes (axes 0 and 1) drops any block whose
    # chi charge has |q| > 1.
    block_keys = [
        (0, 0, 0), (1, -1, 0), (-1, 1, 0),   # all chi charges within {-1,0,1}
        (2, 0, -2), (-2, 0, 2), (0, 2, -2),  # contain a chi charge with |q|=2
    ]
    kept = predict_sectors_under_keep(block_keys, chi_axes=(0, 1), keep={-1, 0, 1})
    assert kept == 3  # only the first three survive


def test_keep_all_is_identity():
    block_keys = [(0, 0, 0), (2, 0, -2)]
    kept = predict_sectors_under_keep(block_keys, chi_axes=(0, 1), keep={-2, -1, 0, 1, 2})
    assert kept == 2
