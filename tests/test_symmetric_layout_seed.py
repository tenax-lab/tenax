"""symmetric_layout_seed: converged envs_init, forced-sweep stability, cut gap."""

import dataclasses

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, ipeps, iPEPSConfig
from tenax.algorithms.ctm_layout import layout_of, symmetric_layout_seed
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate, xxz_gate_u1sz
from tenax.core.tensor import DenseTensor

CHI = 8


@pytest.fixture(scope="module")
def su_pair():
    A0, B0 = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
    cfg = iPEPSConfig(
        max_bond_dim=2, num_imaginary_steps=200, dt=0.05, ctm=CTMConfig(chi=CHI)
    )
    _, (A, B), _ = ipeps(xxz_gate(0.3), (A0, B0), cfg, compute_energy=False)
    return A, B


def _cfg(**kw):
    return dataclasses.replace(
        CTMConfig(chi=CHI), conv_tol=1e-10, max_iter=300, plateau_patience=None, **kw
    )


def test_seed_is_converged_and_stable(su_pair):
    r = symmetric_layout_seed(*su_pair, xxz_gate_u1sz(0.3), _cfg(), forced_sweeps=100)
    assert r.cold_converged
    assert r.stable, (r.energy_cold, r.energy_forced)
    assert r.layout == layout_of(r.envs_init)
    assert any(len(v) > 1 for v in r.layout.values())  # charged sectors present
    # The chi=8 corner spectrum has rank 6 (exact zeros at the cut), so there
    # is no weight at the cut and the gap is reported as inf, not NaN.
    assert not np.isnan(r.cut_gap) and r.cut_gap > 0


def test_cut_gap_is_finite_when_the_cut_bites(su_pair):
    # chi=4 cuts inside the nonzero spectrum, so the gap is a real number.
    cfg = dataclasses.replace(_cfg(), chi=4)
    r = symmetric_layout_seed(*su_pair, xxz_gate_u1sz(0.3), cfg, forced_sweeps=20)
    assert np.isfinite(r.cut_gap) and r.cut_gap > 0


def test_planted_early_stop_is_not_stable(su_pair):
    early = _cfg(on_unconverged="warn")
    early = dataclasses.replace(early, max_iter=3, min_iter=1)
    r = symmetric_layout_seed(*su_pair, xxz_gate_u1sz(0.3), early, forced_sweeps=100)
    assert not r.stable


def test_seed_accepts_dense_pair(su_pair):
    A, B = (DenseTensor(t.todense(), t.indices) for t in su_pair)
    r = symmetric_layout_seed(A, B, xxz_gate_u1sz(0.3), _cfg(), forced_sweeps=20)
    assert np.isfinite(r.energy_cold)
