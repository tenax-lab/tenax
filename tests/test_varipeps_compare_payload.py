"""Round-trip test for benchmarks.varipeps_compare.payload."""

import numpy as np
import pytest

from benchmarks.varipeps_compare.payload import load_payload, save_payload


@pytest.mark.core
def test_payload_roundtrip(tmp_path):
    init = (
        np.random.default_rng(0).standard_normal((2, 2, 2, 2, 2)).astype(np.complex128)
    )
    gate = np.random.default_rng(1).standard_normal((2, 2, 2, 2)).astype(np.complex128)
    meta = {"path": "single_site", "D": 2, "chi": 16, "seed": 0}

    out = tmp_path / "payload.npz"
    save_payload(out, init=init, gate=gate, meta=meta)
    assert out.exists()

    init2, gate2, meta2 = load_payload(out)
    np.testing.assert_array_equal(init, init2)
    np.testing.assert_array_equal(gate, gate2)
    assert meta2 == meta
