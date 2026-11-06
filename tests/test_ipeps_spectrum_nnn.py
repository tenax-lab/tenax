"""The next-nearest-neighbour gate in compute_spectrum: the mixed diagonal RDM."""

import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_spectrum_fixtures import (
    SY_REAL,
    apply_op,
    hasik_d2,
    heisenberg_gate,
    regauge_cell,
    rotated_gate,
)
from _window_oracle import Window, closed_double_layer

from tenax import SpectrumConfig, compute_spectrum
from tenax.algorithms._ctm_tensor_convergence import _ctm_tensor_multisite
from tenax.algorithms._ctm_tensor_energy import (
    _rdm_diagonal_tensor,
    _rdm_diagonal_tensor_4site,
)
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.algorithms._ipeps_excitation_ctm import (
    CELL,
    NEIGHBORS,
    ExcitationCTM,
    _bond_value,
    cell_phases,
    mixed_rdm_diag,
    unpack_env,
)
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
from tenax.algorithms.ipeps_excitations import _spectrum_fingerprint

J2 = 0.3


@pytest.fixture(scope="module")
def ctm():
    return ExcitationCTM(
        hasik_d2(), rotated_gate(), 10, 30, gate_nnn=J2 * heisenberg_gate()
    )


@pytest.fixture(scope="module")
def nearest():
    return ExcitationCTM(hasik_d2(), rotated_gate(), 10, 30)


def _plaquette(mapping, c):
    cx, cy = NEIGHBORS[c]["right"], NEIGHBORS[c]["bottom"]
    return tuple(mapping[x] for x in (c, cx, cy, NEIGHBORS[cx]["bottom"]))


@pytest.mark.parametrize("main", [True, False], ids=["main", "anti"])
def test_diagonal_rdm_is_gauge_invariant_on_the_cell(ctm, main):
    rng = np.random.default_rng(0)
    E = {c: unpack_env(ctm.envs0[c], ctm.D) for c in CELL}
    G = regauge_cell(E, rng)
    kets = {c: jnp.asarray(rng.normal(size=ctm.A.shape)) for c in CELL}
    bras = {c: jnp.asarray(rng.normal(size=ctm.A.shape)) for c in CELL}
    for c in CELL:
        ks, bs = _plaquette(kets, c), _plaquette(bras, c)
        np.testing.assert_allclose(
            mixed_rdm_diag(_plaquette(G, c), ks, bs, main=main),
            mixed_rdm_diag(_plaquette(E, c), ks, bs, main=main),
            rtol=1e-8,
            atol=0,
        )


@pytest.mark.parametrize("main", [True, False], ids=["main", "anti"])
def test_diagonal_rdm_at_bra_equal_ket_matches_the_existing_one(ctm, main):
    E = unpack_env(ctm.envs0[(0, 0)], ctm.D)
    A = ctm.A
    rho = mixed_rdm_diag((E,) * 4, (A,) * 4, (A,) * 4, main=main)
    rho = rho / jnp.einsum("ijij->", rho)
    reference = _rdm_diagonal_tensor(_wrap_as_dense_tensor(A), ctm.envs0[(0, 0)])
    np.testing.assert_allclose(rho, reference, atol=1e-12)


@pytest.mark.parametrize("main", [True, False], ids=["main", "anti"])
def test_diagonal_rdm_matches_the_window_oracle_when_ket_is_not_bra(main):
    """Unrelated tensors on a random environment: no wrong leg can cancel.

    The other tests have bra = ket, or compare the function with itself, so none
    of them sees a ket and a bra exchanged on one site.
    """
    rng = np.random.default_rng(13)
    chi, D, d = 3, 2, 2
    E = {f"C{i}": rng.normal(size=(chi, chi)) for i in range(1, 5)}
    E |= {f"T{i}": rng.normal(size=(chi, D, D, chi)) for i in range(1, 5)}
    kets = [rng.normal(size=(D, D, D, D, d)) for _ in range(4)]
    bras = [rng.normal(size=(D, D, D, D, d)) for _ in range(4)]
    window = Window(
        [E[f"C{i}"] for i in range(1, 5)]
        + [E[f"T{i}"].reshape(chi, D * D, chi) for i in range(1, 5)]
    )
    first, second = (0, 3) if main else (1, 2)  # positions in (TL, TR, BL, BR)
    expected = np.zeros((d,) * 4)
    for s1, s2, t1, t2 in np.ndindex(*expected.shape):
        ks, bs = list(kets), list(bras)
        ks[first], bs[first] = kets[first][..., [s1]], bras[first][..., [t1]]
        ks[second], bs[second] = kets[second][..., [s2]], bras[second][..., [t2]]
        tl, tr, bl, br = (closed_double_layer(k, b) for k, b in zip(ks, bs))
        left_column = window.apply(window.left(2), [tl, bl])
        expected[s1, s2, t1, t2] = window.close(window.apply(left_column, [tr, br]), 2)
    found = mixed_rdm_diag(
        ({k: jnp.asarray(v) for k, v in E.items()},) * 4,
        tuple(jnp.asarray(k) for k in kets),
        tuple(jnp.asarray(b) for b in bras),
        main=main,
    )
    np.testing.assert_allclose(found, expected, atol=1e-10 * np.abs(expected).max())


def test_both_diagonals_close_to_the_same_number(ctm):
    """Open then trace: each site sees its own ket and bra at any position."""
    rng = np.random.default_rng(3)
    E = unpack_env(ctm.envs0[(0, 0)], ctm.D)
    kets = tuple(jnp.asarray(rng.normal(size=ctm.A.shape)) for _ in range(4))
    bras = tuple(jnp.asarray(rng.normal(size=ctm.A.shape)) for _ in range(4))
    main = mixed_rdm_diag((E,) * 4, kets, bras, main=True)
    anti = mixed_rdm_diag((E,) * 4, kets, bras, main=False)
    assert float(jnp.einsum("ijij->", main)) == pytest.approx(
        float(jnp.einsum("ijij->", anti)), rel=1e-10
    )


def test_energy_density_adds_two_diagonals_per_site(ctm, nearest):
    zero = jnp.zeros_like(ctm.A)
    phases = cell_phases(1, 0)
    with_nnn, _ = ctm.densities(phases, 0.0, zero, zero)
    without, _ = nearest.densities(phases, 0.0, zero, zero)
    rho = _rdm_diagonal_tensor(_wrap_as_dense_tensor(ctm.A), ctm.envs0[(0, 0)])
    diagonal = jnp.einsum("ijkl,klij->", rho, jnp.asarray(heisenberg_gate()))
    assert float(with_nnn) == pytest.approx(
        float(without) + 2 * J2 * float(diagonal), abs=1e-9
    )


def _mirror_env(e):
    """Left and right exchanged: the same network, with the two diagonals exchanged."""
    return CTMTensorEnv(
        C1=e.C2.relabels({"c2_l": "c1_r", "c2_d": "c1_d"}),
        C2=e.C1.relabels({"c1_r": "c2_l", "c1_d": "c2_d"}),
        C3=e.C4.relabels({"c4_r": "c3_u", "c4_u": "c3_l"}),
        C4=e.C3.relabels({"c3_u": "c4_r", "c3_l": "c4_u"}),
        T1=e.T1.relabels({"t1_l": "t1_r", "t1_r": "t1_l"}),
        T2=e.T4.relabels({"t4_d": "t2_u", "l2": "r2", "t4_u": "t2_d"}),
        T3=e.T3.relabels({"t3_l": "t3_r", "t3_r": "t3_l"}),
        T4=e.T2.relabels({"t2_u": "t4_d", "r2": "l2", "t2_d": "t4_u"}),
    )


def test_energy_density_matches_the_older_diagonal_on_four_different_sites(
    ctm, nearest
):
    """Bra = ket, no two sites alike: the plaquette placement against existing code.

    ``_rdm_diagonal_tensor_4site`` opens only the main diagonal, so the anti
    diagonal goes through the mirrored network.
    """
    rng = np.random.default_rng(4)
    sites = {}
    for c in CELL:
        Y = rng.normal(size=ctm.A.shape)
        sites[c] = ctm.A + 0.1 * Y / np.linalg.norm(Y)
    wrapped = {c: _wrap_as_dense_tensor(sites[c]) for c in CELL}
    envs = _ctm_tensor_multisite(
        wrapped, NEIGHBORS, 10, max_iter=200, conv_tol=1e-10, _warn_unconverged=False
    )
    flipped = {
        c: _wrap_as_dense_tensor(jnp.transpose(sites[c], (0, 1, 3, 2, 4))) for c in CELL
    }
    flipped_envs = {c: _mirror_env(envs[c]) for c in CELL}
    gate = jnp.asarray(heisenberg_gate())
    values = []
    for c in CELL:
        tl, tr, bl, br = _plaquette({x: x for x in CELL}, c)
        main = _rdm_diagonal_tensor_4site(*_plaquette(wrapped, c), *_plaquette(envs, c))
        order = (tr, tl, br, bl)
        anti = _rdm_diagonal_tensor_4site(
            *(flipped[x] for x in order), *(flipped_envs[x] for x in order)
        )
        values += [float(jnp.einsum("ijkl,klij->", rho, gate)) for rho in (main, anti)]
    assert max(values) - min(values) > 1e-3  # the diagonals do differ on this state
    E = {c: unpack_env(envs[c], ctm.D) for c in CELL}
    found = ctm.energy_density(E, sites, sites) - nearest.energy_density(
        E, sites, sites
    )
    assert float(found) == pytest.approx(J2 * sum(values) / len(CELL), abs=1e-10)


def test_energy_density_gives_each_corner_its_own_ket_and_bra(ctm, nearest):
    """No two tensors alike: a ket or a bra of the wrong corner changes the value."""
    rng = np.random.default_rng(5)
    E = {c: unpack_env(ctm.envs0[c], ctm.D) for c in CELL}
    kets = {c: jnp.asarray(rng.normal(size=ctm.A.shape)) for c in CELL}
    bras = {c: jnp.asarray(rng.normal(size=ctm.A.shape)) for c in CELL}
    expected = 0.0
    for c in CELL:
        for main in (True, False):
            rho = mixed_rdm_diag(
                _plaquette(E, c), _plaquette(kets, c), _plaquette(bras, c), main=main
            )
            expected += float(_bond_value(rho, ctm.gate_nnn))
    found = ctm.energy_density(E, kets, bras) - nearest.energy_density(E, kets, bras)
    assert float(found) == pytest.approx(expected / len(CELL), rel=1e-8)


def test_h_with_the_nnn_gate_matches_a_double_finite_difference(ctm):
    rng = np.random.default_rng(7)
    phases = cell_phases(1, 0)
    X = rng.normal(size=ctm.A.shape)
    Y = rng.normal(size=ctm.A.shape)
    X, Y = X / np.linalg.norm(X), Y / np.linalg.norm(Y)
    step = 1e-3

    def energy(eps, delta):
        return float(
            ctm.densities(phases, eps, jnp.asarray(X), delta * jnp.asarray(Y))[0]
        )

    reference = (
        energy(step, step)
        - energy(step, -step)
        - energy(-step, step)
        + energy(-step, -step)
    ) / (4 * step**2)
    g_h, _ = ctm.ket_derivative(phases, jnp.asarray(X), 1e-4)
    assert float(np.vdot(Y, np.asarray(g_h))) == pytest.approx(reference, abs=1e-4)


def test_compute_spectrum_keeps_the_sum_rule_with_nnn(ctm, tmp_path):
    A = hasik_d2()
    stack = [apply_op(A, SY_REAL).ravel(), A.ravel()]
    basis, _ = np.linalg.qr(np.stack(stack, axis=1))
    config = SpectrumConfig(chi=10, checkpoint_dir=tmp_path / "nnn")
    result = compute_spectrum(
        A,
        rotated_gate(),
        (np.pi, 0.0),
        {"Sy": SY_REAL},
        config,
        gate_nnn=J2 * heisenberg_gate(),
        basis=basis,
    )
    assert abs(result.sum_rule_deficit["Sy"]) < 1e-6
    zero = jnp.zeros_like(ctm.A)
    assert result.energy_density == pytest.approx(
        float(ctm.densities(cell_phases(1, 0), 0.0, zero, zero)[0]), abs=1e-9
    )
    with pytest.raises(ValueError, match="different run"):
        compute_spectrum(
            A,
            rotated_gate(),
            (np.pi, 0.0),
            {"Sy": SY_REAL},
            config,
            gate_nnn=0.5 * heisenberg_gate(),
            basis=basis,
        )


def test_a_run_without_gate_nnn_keeps_its_fingerprint():
    """Columns saved before gate_nnn existed still resume: the digest is 0acdc2a5's."""
    A = np.arange(32.0).reshape((2,) * 5)
    gate = np.arange(16.0).reshape((2,) * 4)
    basis = np.eye(32)[:, :2]
    config = SpectrumConfig(chi=10, ctm_sweeps=30, fd_step=1e-4)
    digest = _spectrum_fingerprint(A, gate, None, (1, 0), config, basis)
    assert digest == "449edce3ffb6d16679e220abc8e18342103750516a41ed6a1612a6ba858356e7"


def test_bad_nnn_gate_shape_raises():
    with pytest.raises(ValueError, match="gate_nnn"):
        compute_spectrum(
            hasik_d2(),
            rotated_gate(),
            (np.pi, 0.0),
            {},
            SpectrumConfig(chi=10),
            gate_nnn=np.eye(3),
        )
