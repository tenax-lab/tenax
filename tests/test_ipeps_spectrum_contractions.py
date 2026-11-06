"""Contractions and second derivatives behind compute_spectrum.

No test here relies only on a symmetric direction. A reversed T4/C4 ring in the
first version of these contractions matched the energy to 10 digits and S^y A to
7 digits on a C4v state, and failed only on random vectors.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_spectrum_fixtures import (
    E_D2,
    apply_op,
    hasik_d2,
    regauge_cell,
    rotated_gate,
)
from _window_oracle import (
    closed_double_layer,
    correlators,
    ising_peps,
    norm_form,
    onsager_c10,
)

from tenax.algorithms._ctm_tensor_convergence import (
    _ctm_tensor_multisite,
    _ctm_tensor_sweep_multisite,
    ctm_tensor,
)
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_multisite
from tenax.algorithms._ipeps_excitation_ctm import (
    CELL,
    NEIGHBORS,
    ExcitationCTM,
    cell_phases,
    mixed_rdm_h,
    mixed_rdm_v,
    unpack_env,
    window_1x1,
    z00,
    z01,
    z10,
)
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
from tenax.algorithms.ipeps_excitations import _as_dense_env


@pytest.fixture(scope="module")
def ctm():
    return ExcitationCTM(hasik_d2(), rotated_gate(), chi=10, sweeps=30)


def _random_sites(rng, shape):
    return {c: jnp.asarray(rng.normal(size=shape)) for c in CELL}


def _environments(ctm):
    return {c: unpack_env(ctm.envs0[c], ctm.D) for c in CELL}


def test_every_window_is_gauge_invariant_on_the_cell(ctm):
    rng = np.random.default_rng(0)
    E = _environments(ctm)
    G = regauge_cell(E, rng)
    kets, bras = _random_sites(rng, ctm.A.shape), _random_sites(rng, ctm.A.shape)
    for c in CELL:
        cx, cy = NEIGHBORS[c]["right"], NEIGHBORS[c]["bottom"]
        cxy = NEIGHBORS[cx]["bottom"]
        pairs = [
            (window_1x1(E[c], kets[c], bras[c]), window_1x1(G[c], kets[c], bras[c])),
            (
                z00(E[cxy]["C1"], E[cy]["C2"], E[c]["C3"], E[cx]["C4"]),
                z00(G[cxy]["C1"], G[cy]["C2"], G[c]["C3"], G[cx]["C4"]),
            ),
            (z10(E[cy], E[c]), z10(G[cy], G[c])),
            (z01(E[cx], E[c]), z01(G[cx], G[c])),
            (
                mixed_rdm_h(E[c], E[cx], kets[c], bras[c], kets[cx], bras[cx]),
                mixed_rdm_h(G[c], G[cx], kets[c], bras[c], kets[cx], bras[cx]),
            ),
            (
                mixed_rdm_v(E[c], E[cy], kets[c], bras[c], kets[cy], bras[cy]),
                mixed_rdm_v(G[c], G[cy], kets[c], bras[c], kets[cy], bras[cy]),
            ),
        ]
        for plain, gauged in pairs:
            np.testing.assert_allclose(gauged, plain, rtol=1e-8, atol=0)


def test_energy_density_matches_the_published_energy(ctm):
    zero = jnp.zeros_like(ctm.A)
    energy, _ = ctm.densities(cell_phases(1, 0), 0.0, zero, zero)
    assert float(energy) == pytest.approx(E_D2, abs=1e-9)


def test_energy_density_matches_the_multisite_energy_on_a_staggered_state(ctm):
    """Bra = ket, not uniform: the bond placement against the existing code path."""
    rng = np.random.default_rng(2)
    phases = cell_phases(1, 1)
    Y = rng.normal(size=ctm.A.shape)
    sites = {c: ctm.A + 0.02 * phases[c] * Y / np.linalg.norm(Y) for c in CELL}
    wrapped = {c: _wrap_as_dense_tensor(sites[c]) for c in CELL}
    envs = _ctm_tensor_multisite(
        wrapped, NEIGHBORS, 10, max_iter=200, conv_tol=1e-10, _warn_unconverged=False
    )
    E = {c: unpack_env(envs[c], ctm.D) for c in CELL}
    reference = compute_energy_ctm_tensor_multisite(
        wrapped, envs, NEIGHBORS, jnp.asarray(ctm.gate)
    )
    assert float(ctm.energy_density(E, sites, sites)) == pytest.approx(
        float(reference), abs=1e-9
    )


def _unit(rng, shape):
    B = rng.normal(size=shape)
    return B / np.linalg.norm(B)


def _double_finite_difference(ctm, phases, X, Y, step=1e-3):
    def energy(eps, delta):
        return float(
            ctm.densities(phases, eps, jnp.asarray(X), delta * jnp.asarray(Y))[0]
        )

    return (
        energy(step, step)
        - energy(step, -step)
        - energy(-step, step)
        + energy(-step, -step)
    ) / (4 * step**2)


@pytest.mark.parametrize("ab", [(1, 0), (1, 1)], ids=["X", "M"])
def test_h_matches_a_double_finite_difference_on_random_pairs(ctm, ab):
    rng = np.random.default_rng(7)
    phases = cell_phases(*ab)
    X, Y = _unit(rng, ctm.A.shape), _unit(rng, ctm.A.shape)
    g_h, _ = ctm.ket_derivative(phases, jnp.asarray(X), 1e-4)
    reference = _double_finite_difference(ctm, phases, X, Y)
    assert float(np.vdot(Y, np.asarray(g_h))) == pytest.approx(reference, abs=1e-4)


def test_h_at_small_chi_needs_the_projector_derivatives():
    """Frozen projectors miss part of H: by 6e-4 at chi=4, and only 1e-5 at chi=10."""
    chi = 4
    ctm = ExcitationCTM(hasik_d2(), rotated_gate(), chi=chi, sweeps=30)
    rng = np.random.default_rng(7)
    phases = cell_phases(1, 0)
    X, Y = _unit(rng, ctm.A.shape), _unit(rng, ctm.A.shape)
    reference = _double_finite_difference(ctm, phases, X, Y)
    flow, _ = ctm.ket_derivative(phases, jnp.asarray(X), 1e-4)
    assert float(np.vdot(Y, np.asarray(flow))) == pytest.approx(reference, abs=1e-4)

    # The regime: if this fails, chi=4 no longer tells flow from frozen.
    ctm._sweep = jax.jit(
        lambda e, dl: _ctm_tensor_sweep_multisite(
            e, dl, NEIGHBORS, chi, True, "svd", "frozen", "2x2"
        )[0]
    )
    frozen, _ = ctm.ket_derivative(phases, jnp.asarray(X), 1e-4)
    assert abs(float(np.vdot(Y, np.asarray(frozen))) - reference) > 3e-4


@pytest.mark.parametrize("ab", [(1, 0), (1, 1)], ids=["X", "M"])
def test_n_matches_the_window_oracle_on_random_directions(ctm, ab):
    rng = np.random.default_rng(1)
    A = np.asarray(ctm.A)
    site = _wrap_as_dense_tensor(ctm.A)
    env = _as_dense_env(
        ctm_tensor(site, 10, max_iter=800, conv_tol=1e-11, return_meta=True)[0]
    )
    E = unpack_env(ctm.envs0[(0, 0)], ctm.D)
    z = float(window_1x1(E, ctm.A, ctm.A))
    phases = cell_phases(*ab)
    B = _unit(rng, A.shape)
    B_j = jnp.asarray(B)
    disconnected = (
        float(window_1x1(E, B_j, ctm.A)) / z * float(window_1x1(E, ctm.A, B_j)) / z
    )
    reference = norm_form(env, A, B, disconnected, *ab, R=6)
    _, g_n = ctm.ket_derivative(phases, B_j, 1e-4)
    assert float(np.vdot(B, np.asarray(g_n))) == pytest.approx(reference, abs=1e-4)


def test_window_oracle_reproduces_onsager():
    beta = 0.3
    A = ising_peps(beta)
    B = apply_op(A, np.diag([1.0, -1.0]))
    env = _as_dense_env(
        ctm_tensor(
            _wrap_as_dense_tensor(jnp.asarray(A)),
            16,
            max_iter=500,
            conv_tol=1e-12,
            return_meta=True,
        )[0]
    )
    C = correlators(
        env,
        closed_double_layer(A, A),
        closed_double_layer(B, A),
        closed_double_layer(A, B),
        closed_double_layer(B, B),
        1,
    )
    assert C[(1, 0)].real == pytest.approx(onsager_c10(beta), abs=1e-10)
