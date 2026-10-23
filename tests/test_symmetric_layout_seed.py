"""symmetric_layout_seed: converged envs_init, forced-sweep stability, cut gap."""

import dataclasses

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, ipeps, iPEPSConfig
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms.ctm_layout import (
    layout_of,
    normalized_pair,
    symmetric_layout_seed,
)
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate, xxz_gate_u1sz
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
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
    assert r.gap_converged
    assert r.stable, (r.energy_cold, r.energy_forced)
    assert r.layout == layout_of(r.envs_init)
    assert any(len(v) > 1 for v in r.layout.values())  # charged sectors present
    # Regime (measured): the chi+4 corner spectrum has a 3-fold degeneracy,
    # s/s0 = 9.36709e-3 in sectors -4, 0, +4, at sorted positions 6, 7, 8.
    # The chi=8 cut falls inside it, so the gap is roundoff (4e-16 or exactly
    # 0.0 between builds): a tie at the cut, which cut_gap must report, not
    # hide as NaN or inf.  test_cut_gap_is_finite_when_the_cut_bites owns
    # the nonzero-gap case.
    assert np.isfinite(r.cut_gap) and 0 <= r.cut_gap < 1e-12, r.cut_gap


def test_cut_gap_is_finite_when_the_cut_bites(su_pair):
    # chi=4 cuts inside the nonzero spectrum, so the gap is a real number.
    cfg = dataclasses.replace(_cfg(), chi=4)
    r = symmetric_layout_seed(*su_pair, xxz_gate_u1sz(0.3), cfg, forced_sweeps=20)
    assert np.isfinite(r.cut_gap) and 0 < r.cut_gap <= 1
    # Independent value: chi+4 solve, global sort of block singular values.
    A_n, B_n = normalized_pair(*su_pair)
    big_cfg = dataclasses.replace(cfg, chi=cfg.chi + 4, on_unconverged="warn")
    big, _ = python_loop_ctm_converge(
        {(0, 0): A_n, (1, 0): B_n},
        CHECKERBOARD_NEIGHBORS,
        **ctm_converge_kwargs(big_cfg, env_init=None),
    )
    gaps = []
    for env in big.values():
        for f in ("C1", "C2", "C3", "C4"):
            sv = [
                np.linalg.svd(np.asarray(b), compute_uv=False)
                for b in getattr(env, f).blocks.values()
            ]
            s = np.sort(np.concatenate(sv))[::-1]
            gaps.append((s[3] - s[4]) / s[3])
    assert abs(min(gaps) - r.cut_gap) < 1e-12


def test_planted_early_stop_is_not_stable(su_pair):
    early = _cfg(on_unconverged="warn")
    early = dataclasses.replace(early, max_iter=3, min_iter=1)
    r = symmetric_layout_seed(*su_pair, xxz_gate_u1sz(0.3), early, forced_sweeps=100)
    assert not r.stable


def test_an_unconverged_gap_solve_is_reported(su_pair):
    """The chi+4 solve that ``cut_gap`` reads can stop unconverged; the seed
    must say so, not return the gap as if it were valid (Codex on #1133)."""
    early = dataclasses.replace(_cfg(on_unconverged="warn"), max_iter=3, min_iter=1)
    r = symmetric_layout_seed(*su_pair, xxz_gate_u1sz(0.3), early, forced_sweeps=0)
    assert not r.cold_converged  # regime: 3 sweeps cannot converge this point
    assert not r.gap_converged


def test_seed_accepts_dense_pair(su_pair):
    A, B = (DenseTensor(t.todense(), t.indices) for t in su_pair)
    cfg = dataclasses.replace(_cfg(), chi=4)
    gate = xxz_gate_u1sz(0.3)
    r = symmetric_layout_seed(A, B, gate, cfg, forced_sweeps=20)
    assert np.isfinite(r.energy_cold)
    assert isinstance(r.stable, bool)
    assert all(isinstance(env.C1, DenseTensor) for env in r.envs_init.values())
    assert r.layout == layout_of(r.envs_init)
    # dense chi legs carry one sector (the symmetric seed has several)
    assert all(len(v) == 1 for k, v in r.layout.items() if k[1].startswith("C"))
    rs = symmetric_layout_seed(*su_pair, gate, cfg, forced_sweeps=20)
    assert abs(r.energy_cold - rs.energy_cold) < 1e-8
