"""The benchmark's state parser and U(1) scan."""

import json

import numpy as np
import pytest

from benchmarks.ipeps_spectrum_hasik import load_state, u1_residuals


def test_load_state_reads_the_leg_order_s_u_l_d_r(tmp_path):
    path = tmp_path / "state.json"
    entries = ["1 0 1 0 0 3.0", "0 1 0 0 1 4.0"]
    site = {"siteId": "A", "physDim": 2, "auxDim": 2, "entries": entries}
    path.write_text(json.dumps({"sites": [site]}))
    A = load_state(path)
    assert A.shape == (2, 2, 2, 2, 2)
    # s=1, u=0, l=1, d=0, r=0 is A[u, d, l, r, s] = A[0, 0, 1, 0, 1]
    assert A[0, 0, 1, 0, 1] == pytest.approx(3.0 / 5.0)
    assert A[1, 0, 0, 1, 0] == pytest.approx(4.0 / 5.0)


def _u1_symmetric_tensor():
    rng = np.random.default_rng(0)
    charge = np.array([0, 1])
    physical = np.array([1, -1])
    A = np.zeros((2, 2, 2, 2, 2))
    for u, d, left, r, s in np.ndindex(2, 2, 2, 2, 2):
        if charge[u] + charge[d] + charge[left] + charge[r] - 2 == physical[s]:
            A[u, d, left, r, s] = rng.normal()
    return A / np.linalg.norm(A)


def test_u1_scan_finds_a_symmetric_tensor():
    residuals = u1_residuals(_u1_symmetric_tensor())
    assert residuals["best_xz"] < 1e-12
    assert residuals["best_angle"] == pytest.approx(0.0)


def test_u1_scan_finds_it_in_another_bond_gauge():
    A = _u1_symmetric_tensor()
    theta = 0.7
    V = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = np.einsum("au,bd,cl,er,udlrs->abces", V, V, V, V, A)
    assert u1_residuals(rotated)["best_xz"] < 1e-12


def test_u1_scan_rejects_a_random_tensor():
    A = np.random.default_rng(1).normal(size=(2, 2, 2, 2, 2))
    assert u1_residuals(A / np.linalg.norm(A))["best_xz"] > 0.1
