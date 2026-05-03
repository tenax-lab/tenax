"""Tests for the Husimi-tree local energy probe on kagome 3-PESS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.pess import (
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    pess_local_energy,
    pess_simple_update,
)


@pytest.mark.algorithm
def test_pess_local_energy_returns_real_finite():
    """Energy of a random 3-PESS state must be a finite real scalar."""
    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    h_tri = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    e = pess_local_energy(state, h_tri)
    assert jnp.isfinite(e)
    assert jnp.abs(jnp.imag(e)) < 1e-10


@pytest.mark.algorithm
def test_pess_local_energy_d2_classical_band():
    """At D=2, SU-converged 3-PESS lies near the classical 120° energy.

    The classical Heisenberg energy per site on kagome with S=1/2 is
    -|S|^2 * J/2 = -0.125 (per nearest-neighbor pair), times the 4 NN per
    site / 2 = 2 pairs per site / kagome counting → roughly -0.25 / site
    in the J=1 convention. We assert a generous band that catches sign
    flips and gross factor errors but tolerates D=2 quantum corrections.
    """
    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    h_tri = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    state = pess_simple_update(
        state,
        h_tri,
        dt_schedule=[(0.1, 100), (0.01, 100)],
        D_max=2,
    )
    e = float(jnp.real(pess_local_energy(state, h_tri)))
    assert -0.40 < e < -0.20, f"D=2 SU energy out of band: {e}"
