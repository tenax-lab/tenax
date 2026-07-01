"""Opposite-flow legs of one bond must tile to equal charge multisets (#667)."""

from __future__ import annotations

from collections import Counter

import numpy as np

from tenax.algorithms._ctm_tensor_init import _tile_fused_to_chi


def test_tile_preserves_multiset_under_sign_flip():
    # A U(1) fusion enumeration and its sign-flip (what opposite-flow legs of
    # the same virtual axis produce) must give equal multisets after tiling
    # past D**2 to chi.
    fused = np.array([0, -2, 0, 2, 0, 2, 0, -2, 0], dtype=np.int32)  # D**2 = 9
    flipped = -fused
    chi = 12
    a = Counter(_tile_fused_to_chi(fused, chi).tolist())
    b = Counter(_tile_fused_to_chi(flipped, chi).tolist())
    assert a == b, f"tiled multisets differ: {a} vs {b}"


def test_tile_keeps_charge_zero_at_index_0():
    # The rank-1 seed lives at pre-perm index 0 and must stay charge 0.
    fused = np.array([0, -1, 1, 1, 0, 2, -1, -2, 0], dtype=np.int32)
    out = _tile_fused_to_chi(fused, 12)
    assert int(out[0]) == 0


def test_tile_no_pad_when_chi_le_d2():
    fused = np.array([0, -2, 0, 2, 0, 2, 0, -2, 0], dtype=np.int32)
    out = _tile_fused_to_chi(fused, 5)
    assert np.array_equal(out, fused[:5])
