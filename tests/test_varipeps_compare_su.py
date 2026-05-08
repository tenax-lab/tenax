"""Init dispatcher smoke test."""

import numpy as np
import pytest

from benchmarks.varipeps_compare.su_init import (
    build_heisenberg_gate,
    build_sublattice_rotated_gate,
    make_init,
)


@pytest.mark.core
def test_heisenberg_gate_shape_and_hermiticity():
    gate = build_heisenberg_gate()
    assert gate.shape == (2, 2, 2, 2)
    assert gate.dtype == np.complex128
    M = gate.reshape(4, 4)
    np.testing.assert_allclose(M, M.conj().T, atol=1e-12)


@pytest.mark.core
def test_sublattice_rotated_gate_shape():
    g_rot = build_sublattice_rotated_gate()
    assert g_rot.shape == (2, 2, 2, 2)
    assert g_rot.dtype == np.complex128


@pytest.mark.core
def test_make_init_single_site_random_deterministic():
    """single_site path: random init, deterministic given seed."""
    a = make_init(path="single_site", D=2, seed=0)
    b = make_init(path="single_site", D=2, seed=0)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (2, 2, 2, 2, 2)  # (D,D,D,D,d)
    assert a.dtype == np.complex128
    # Different seed → different array
    c = make_init(path="single_site", D=2, seed=1)
    assert not np.array_equal(a, c)


@pytest.mark.algorithm
def test_make_init_bipartite_2site_su_d2():
    """bipartite_2site path: Tenax SU returns stacked (A, B) of shape (2, D, D, D, D, d)."""
    init = make_init(path="bipartite_2site", D=2, seed=0)
    assert isinstance(init, np.ndarray)
    assert init.shape == (2, 2, 2, 2, 2, 2)  # (2, D, D, D, D, d)
    assert init.dtype == np.complex128
