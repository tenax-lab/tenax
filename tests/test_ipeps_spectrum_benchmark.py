"""The benchmark's state parser and U(1) scan."""

import json
import sys

import numpy as np
import pytest

from benchmarks.ipeps_spectrum_hasik import (
    OPERATORS,
    load_charges,
    load_state,
    main,
    operator_basis,
    plaquette_energy,
    reflection_basis,
    rotated_basis,
    rotated_gate,
    transverse_basis,
    u1_residuals,
)


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


def test_operator_basis_holds_each_o_a_and_a():
    A = np.random.default_rng(2).normal(size=(2, 2, 2, 2, 2))
    A /= np.linalg.norm(A)
    basis = operator_basis(A, OPERATORS)
    assert basis.shape == (A.size, len(OPERATORS) + 1)
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-12)
    for vector in (
        *(np.tensordot(A, op, axes=([4], [1])) for op in OPERATORS.values()),
        A,
    ):
        v = vector.ravel()
        np.testing.assert_allclose(basis @ (basis.T @ v), v, atol=1e-12)


def _write_u1_state(path):
    elem = [
        {"physDim": 2, "auxDim": 2, "numEntries": 1, "entries": ["1 0 1 0 0 1.0"]},
        {"physDim": 2, "auxDim": 2, "numEntries": 1, "entries": ["0 1 0 0 1 2.0"]},
    ]
    coeffs = [{"siteId": "A0", "numEntries": 2, "entries": ["0 3.0", "1 2.0"]}]
    data = {
        "total_u1_charge": 1,
        "u1_charges": [1, -1, 0, 2],
        "coeffs": coeffs,
        "elem_tensors": elem,
    }
    path.write_text(json.dumps(data))


def test_load_state_sums_the_elementary_tensors_of_a_u1_state(tmp_path):
    path = tmp_path / "state.json"
    _write_u1_state(path)
    A = load_state(path)
    assert A.shape == (2, 2, 2, 2, 2)
    # 3 * 1 at s=1, u=0, l=1, d=0, r=0, and 2 * 2 at s=0, u=1, l=0, d=0, r=1: norm 5.
    assert A[0, 0, 1, 0, 1] == pytest.approx(3.0 / 5.0)
    assert A[1, 0, 0, 1, 0] == pytest.approx(4.0 / 5.0)
    assert np.count_nonzero(A) == 2


def test_load_charges_splits_physical_virtual_and_total(tmp_path):
    path = tmp_path / "state.json"
    _write_u1_state(path)
    physical, virtual, total = load_charges(path)
    np.testing.assert_array_equal(physical, [1, -1])
    np.testing.assert_array_equal(virtual, [0, 2])
    assert total == 1


def _c2v_u1_tensor(physical, virtual, total):
    """Random real tensor with the charge rule, even under u<->d and l<->r."""
    rng = np.random.default_rng(3)
    D = len(virtual)
    A = np.zeros((D, D, D, D, 2))
    for index in np.ndindex(*A.shape):
        u, d, left, r, s = index
        if physical[s] + virtual[u] + virtual[d] + virtual[left] + virtual[r] == total:
            A[index] = rng.normal()
    A = A + A.transpose(1, 0, 2, 3, 4)
    A = A + A.transpose(0, 1, 3, 2, 4)
    return A / np.linalg.norm(A)


PHYSICAL, VIRTUAL, TOTAL = np.array([1, -1]), np.array([0, 2, 0]), 1


def test_transverse_basis_is_orthonormal_and_even_under_the_little_group():
    basis = transverse_basis(PHYSICAL, VIRTUAL, TOTAL)
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-12)
    tensors = basis.T.reshape(-1, 3, 3, 3, 3, 2)
    np.testing.assert_allclose(tensors, tensors.transpose(0, 2, 1, 3, 4, 5), atol=1e-12)
    np.testing.assert_allclose(tensors, tensors.transpose(0, 1, 2, 4, 3, 5), atol=1e-12)


def test_transverse_basis_holds_only_charge_offsets_plus_and_minus_two():
    basis = transverse_basis(PHYSICAL, VIRTUAL, TOTAL)
    for index in np.ndindex(3, 3, 3, 3, 2):
        u, d, left, r, s = index
        offset = (
            PHYSICAL[s] + VIRTUAL[u] + VIRTUAL[d] + VIRTUAL[left] + VIRTUAL[r] - TOTAL
        )
        row = np.ravel_multi_index(index, (3, 3, 3, 3, 2))
        if abs(offset) != 2:
            assert np.abs(basis[row]).max() == 0.0


def test_transverse_basis_holds_sx_a_and_sy_a_but_not_sz_a():
    A = _c2v_u1_tensor(PHYSICAL, VIRTUAL, TOTAL)
    basis = transverse_basis(PHYSICAL, VIRTUAL, TOTAL)
    for name in ("Sx", "Sy"):
        v = np.tensordot(A, OPERATORS[name], axes=([4], [1])).ravel()
        np.testing.assert_allclose(basis @ (basis.T @ v), v, atol=1e-12)
    v = np.tensordot(A, OPERATORS["Sz"], axes=([4], [1])).ravel()
    assert np.linalg.norm(basis.T @ v) < 1e-12


def test_transverse_basis_spans_every_even_vector_in_the_two_sectors():
    rng = np.random.default_rng(4)
    v = np.zeros((3, 3, 3, 3, 2))
    for index in np.ndindex(*v.shape):
        u, d, left, r, s = index
        charge = PHYSICAL[s] + VIRTUAL[u] + VIRTUAL[d] + VIRTUAL[left] + VIRTUAL[r]
        if abs(charge - TOTAL) == 2:
            v[index] = rng.normal()
    v = v + v.transpose(1, 0, 2, 3, 4)
    v = (v + v.transpose(0, 1, 3, 2, 4)).ravel()
    basis = transverse_basis(PHYSICAL, VIRTUAL, TOTAL)
    np.testing.assert_allclose(basis @ (basis.T @ v), v, atol=1e-12)
    assert basis.shape[1] < 72 / 2


def test_transverse_basis_keeps_every_element_without_the_reflections():
    basis = transverse_basis(PHYSICAL, VIRTUAL, TOTAL, even=False)
    assert basis.shape == (3**4 * 2, 72)
    np.testing.assert_allclose(basis.T @ basis, np.eye(72), atol=1e-12)


def test_reflection_basis_counts_the_orbits_of_both_reflections():
    # D(D+1)/2 unordered pairs on each axis, times d.
    for D, columns in ((2, 18), (3, 72), (4, 200)):
        basis = reflection_basis(D, 2)
        assert basis.shape == (2 * D**4, columns)
        np.testing.assert_allclose(basis.T @ basis, np.eye(columns), atol=1e-12)


def test_reflection_basis_holds_every_o_a_of_a_reflection_even_tensor():
    A = np.random.default_rng(5).normal(size=(3, 3, 3, 3, 2))
    A = A + A.transpose(1, 0, 2, 3, 4)
    A = A + A.transpose(0, 1, 3, 2, 4)
    basis = reflection_basis(3, 2)
    for op in OPERATORS.values():
        v = np.tensordot(A, op, axes=([4], [1])).ravel()
        np.testing.assert_allclose(basis @ (basis.T @ v), v, atol=1e-12)


def test_reflection_basis_leaves_out_a_vector_odd_under_one_reflection():
    v = np.random.default_rng(6).normal(size=(3, 3, 3, 3, 2))
    v = (v - v.transpose(1, 0, 2, 3, 4)).ravel()
    assert np.linalg.norm(reflection_basis(3, 2).T @ v) < 1e-12


def test_rotated_basis_keeps_the_span_and_mixes_every_column():
    basis = transverse_basis(PHYSICAL, VIRTUAL, TOTAL)
    mixed = rotated_basis(basis)
    np.testing.assert_allclose(mixed.T @ mixed, np.eye(basis.shape[1]), atol=1e-12)
    np.testing.assert_allclose(mixed @ (mixed.T @ basis), basis, atol=1e-12)
    # Every column overlaps every orbit: no column is left along a special direction.
    assert np.abs(basis.T @ mixed).min() > 1e-6


def test_main_passes_the_fd_step_to_the_config(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    entries = ["1 0 1 0 0 3.0", "0 1 0 0 1 4.0"]
    site = {"siteId": "A", "physDim": 2, "auxDim": 2, "entries": entries}
    path.write_text(json.dumps({"sites": [site]}))
    configs = []

    def stop_at_the_call(A, gate, momentum, operators, config, **kwargs):
        configs.append(config)
        raise InterruptedError

    monkeypatch.setattr("tenax.compute_spectrum", stop_at_the_call)
    argv = ["bench", str(path), "spectrum", "--fd-step", "1e-3"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(InterruptedError):
        main()
    assert configs[0].fd_step == 1e-3


def _c4v_tensor(seed, noise):
    """A D=2 tensor with the lattice's full point group, close to a product state."""
    rng = np.random.default_rng(seed)
    A = np.zeros((2, 2, 2, 2, 2))
    A[0, 0, 0, 0, :] = rng.normal(size=2)
    A = A + noise * rng.normal(size=A.shape)
    # (u, d, l, r): the two reflections and the diagonal one generate C4v.
    A = A + A.transpose(1, 0, 2, 3, 4)
    A = A + A.transpose(0, 1, 3, 2, 4)
    A = A + A.transpose(2, 3, 0, 1, 4)
    return A / np.linalg.norm(A)


def test_plaquette_energy_matches_the_two_site_window_on_a_short_ranged_state():
    # At small correlation length both windows reach the same limit, by a wide
    # margin below the difference a wrong contraction would make.
    A = _c4v_tensor(seed=2, noise=0.05)
    energy, two_site = plaquette_energy(A, chi=8, j2=0.3)
    assert energy == pytest.approx(two_site, abs=1e-8)
    assert abs(energy) > 0.01


def test_plaquette_energy_of_a_product_state_is_the_closed_form():
    theta = 0.4
    v = np.array([np.cos(theta), np.sin(theta)])
    A = np.zeros((2, 2, 2, 2, 2))
    A[0, 0, 0, 0, :] = v
    energy, _ = plaquette_energy(A, chi=4, j2=0.0)
    pair = np.kron(v, v)
    expected = 2 * pair @ rotated_gate().reshape(4, 4) @ pair
    assert energy == pytest.approx(expected, abs=1e-10)
