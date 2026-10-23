"""Eager #1098 layout sweeps jit their static-shape pieces.

The result is unchanged away from degenerate cuts.  At a tie the result can
differ: this is a property of the code, not something measured here.  The
jitted prime stage routes the M1/M2 SVDs to ``_truncated_svd_symmetric_traced``,
so the chosen sectors can differ.  No test in this file exercises a tie.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

import tenax.algorithms._ctm_eager_jit as ej
import tenax.algorithms._ctm_python_loop as cpl
from tenax import CTMConfig, ipeps, iPEPSConfig
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionicU1, U1Symmetry
from tenax.core.tensor import SymmetricTensor

jax.config.update("jax_enable_x64", True)

D = 2
CHI = 6  # U(1) chi=6 cuts the eager C1 spectrum in a clean gap (#1098 test)


def _sweep_kwargs(chi):
    return dict(
        neighbors=CHECKERBOARD_NEIGHBORS,
        recipe="2x2",
        device_mesh=None,
        projector_method="svd",
        renormalize=False,
        projector_backward="auto",
        chunk_size=None,
        chi=chi,
    )


@pytest.fixture(scope="module")
def sites():
    A0, B0 = heisenberg_u1sz_init_pair(D, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=D, num_imaginary_steps=200, dt=0.05, ctm=CTMConfig(chi=8)
    )
    _, (A, B), _ = ipeps(xxz_gate(0.3), (A0, B0), su, compute_energy=False)
    return {
        (0, 0): A * (1.0 / (A.norm() + 1e-10)),
        (1, 0): B * (1.0 / (B.norm() + 1e-10)),
    }


def _sweeps(sites, chi, n, enabled, monkeypatch):
    monkeypatch.setattr(ej, "_ENABLED", enabled)
    envs = {c: initialize_ctm_tensor_env(A, chi) for c, A in sites.items()}
    for _ in range(n):
        envs, _, _ = cpl._eager_ctm_sweep(sites, envs, **_sweep_kwargs(chi))
        envs = cpl._charge_sorted_envs(envs)
    return envs


def _assert_same_envs(a, b, atol=1e-10):
    assert cpl._chi_layout(a) == cpl._chi_layout(b)
    for c in a:
        for f in a[c]._fields:
            x = np.asarray(getattr(a[c], f).todense())  # chi=6, D^2=4: small
            y = np.asarray(getattr(b[c], f).todense())
            np.testing.assert_allclose(x, y, atol=atol, err_msg=f"{c}.{f}")


def test_jitted_eager_sweeps_match_plain_eager_sweeps(sites, monkeypatch):
    plain = _sweeps(sites, CHI, 3, False, monkeypatch)
    jitted = _sweeps(sites, CHI, 3, True, monkeypatch)
    _assert_same_envs(plain, jitted)


def test_eager_sweep_runs_the_absorb_moves_under_jit(sites, monkeypatch):
    seen = []
    raw_jitted = ej._jitted

    def spy(fn, static_argnames, flags):
        seen.append(fn.__name__)
        return raw_jitted(fn, static_argnames, flags)

    monkeypatch.setattr(ej, "_jitted", spy)
    _sweeps(sites, CHI, 1, True, monkeypatch)
    for d in ("left", "right", "top", "bottom"):
        assert f"_ctm_tensor_absorb_{d}_2plaq" in seen
    assert "_build_double_layer_tensor" in seen


def test_eager_sweep_runs_the_projector_stages_under_jit(sites, monkeypatch):
    seen = []
    raw_jitted = ej._jitted

    def spy(fn, static_argnames, flags):
        seen.append(fn.__name__)
        return raw_jitted(fn, static_argnames, flags)

    monkeypatch.setattr(ej, "_jitted", spy)
    _sweeps(sites, CHI, 1, True, monkeypatch)
    for name in (
        "_build_enlarged_corner",
        "_symmetric_projector_prime",
        "_symmetric_projector_finish",
    ):
        assert name in seen, name


def test_jitted_eager_sweep_compiles_far_less(sites, monkeypatch):
    """Mechanism: count XLA backend compiles in one cold eager sweep.

    The persistent compilation cache is off for the count: a cache hit skips
    the backend compile and would undercount either side.  ``is_cache_used``
    memoises its answer, so ``reset_cache`` makes the flag take effect.
    """
    from jax._src import compilation_cache as cc

    if not hasattr(cc, "reset_cache"):
        pytest.skip(
            "this JAX has no compilation_cache.reset_cache to turn the cache off"
        )
    n = [0]

    def listener(name, dur, **_):
        if name == "/jax/core/compile/backend_compile_duration":
            n[0] += 1

    saved_enable = jax.config.jax_enable_compilation_cache
    jax.config.update("jax_enable_compilation_cache", False)
    cc.reset_cache()
    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        jax.clear_caches()
        ej._jitted.cache_clear()
        _sweeps(sites, CHI, 1, True, monkeypatch)
        on = n[0]
        jax.clear_caches()
        n[0] = 0
        _sweeps(sites, CHI, 1, False, monkeypatch)
        off = n[0]
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
        jax.config.update("jax_enable_compilation_cache", saved_enable)
        cc.reset_cache()
    assert on * 5 < off, (on, off)


def _fermionic_pair():
    """A random FermionicU1 pair, normalised.

    Copied from ``_fermionic_direction_dependent_pair`` in
    ``tests/test_ctm_674_fermionic_fused.py``.
    """
    sym = FermionicU1()

    def idx(charges, flow, label):
        return TensorIndex.from_charges(
            sym, np.array(charges, dtype=np.int32), flow, label=label
        )

    lA = [0, 1, -1]  # A.l charges
    rA = [0, 0, 1]  # A.r charges  (!= A.l  -> direction-dependent)
    ud = [0, 1, -1]
    phys = [0, 1]

    def make(l_charges, r_charges, key):
        return SymmetricTensor.random_normal(
            (
                idx(ud, FlowDirection.OUT, "u"),
                idx(ud, FlowDirection.IN, "d"),
                idx(l_charges, FlowDirection.OUT, "l"),
                idx(r_charges, FlowDirection.IN, "r"),
                idx(phys, FlowDirection.IN, "phys"),
            ),
            jax.random.PRNGKey(key),
        )

    A = make(lA, rA, 0)
    B = make(rA, lA, 1)  # cell-consistent: B.l == A.r, B.r == A.l
    return {
        (0, 0): A * (1.0 / (A.norm() + 1e-10)),
        (1, 0): B * (1.0 / (B.norm() + 1e-10)),
    }


def _confined_pair():
    """A U(1) pair with no allowed block, so the M1 SVD bond is rank 0.

    The parity obstruction that ``heisenberg_u1sz_init_pair`` documents: pure
    +-1 virtual charges with +-1 physical charges admit no Sz-conserving block.
    The index code is ``heisenberg_u1sz_init_pair``'s
    (``src/tenax/algorithms/ipeps.py``) with the virtual charges ``[1, -1]``.
    """
    sym = U1Symmetry()
    virt_charges = np.array([1, -1], dtype=np.int32)
    phys_charges = np.array([1, -1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(
            sym, virt_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, virt_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    kA, kB = jax.random.split(jax.random.PRNGKey(0))
    A = SymmetricTensor.random_normal(indices, kA)
    B = SymmetricTensor.random_normal(indices, kB)
    assert not A.blocks and not B.blocks  # regime: no allowed block at all
    return {(0, 0): A, (1, 0): B}


def test_chi8_energy_matches_with_jit_on_and_off(monkeypatch):
    """Energy equivalence at U(1) D=2 chi=8: a converged CTM with the eager
    helper on and off gives the same energy.

    The M' cut here is not degenerate: at the converged environment it sits in
    a 2.2 % gap (8.764e-5 kept, 8.576e-5 discarded), and both runs keep the
    same chi layout."""
    A0, B0 = heisenberg_u1sz_init_pair(D, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=D, num_imaginary_steps=200, dt=0.05, ctm=CTMConfig(chi=8)
    )
    _, (A, B), _ = ipeps(xxz_gate(0.3), (A0, B0), su, compute_energy=False)
    A = A * (1.0 / (A.norm() + 1e-10))
    B = B * (1.0 / (B.norm() + 1e-10))
    g = xxz_gate(0.3).todense()
    energies = []
    for enabled in (False, True):
        monkeypatch.setattr(ej, "_ENABLED", enabled)
        cfg = dataclasses.replace(
            CTMConfig(chi=8),
            conv_tol=1e-12,
            max_iter=200,
            min_iter=10,
            plateau_patience=None,
            on_unconverged="warn",
        )
        envs, _ = cpl.python_loop_ctm_converge(
            {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, **ctm_converge_kwargs(cfg)
        )
        energies.append(
            float(
                compute_energy_ctm_tensor_2site(A, B, envs[(0, 0)], envs[(1, 0)], g, 2)
            )
        )
    assert abs(energies[0] - energies[1]) < 1e-9, energies


def test_fermionic_eager_sweeps_match(monkeypatch):
    """Review Focus 4: a random FermionicU1 pair through the jitted pieces."""
    sites = _fermionic_pair()
    plain = _sweeps(sites, 8, 2, False, monkeypatch)
    jitted = _sweeps(sites, 8, 2, True, monkeypatch)
    _assert_same_envs(plain, jitted)


def test_rank0_projector_still_raises_the_907_diagnosis(monkeypatch):
    """Review Focus 5: the rank-0 projector diagnosis (``_require_svd_connected``)
    still raises from inside the jitted prime stage, on a parity-obstructed
    U(1) pair (no blocks at all)."""
    sites = _confined_pair()
    for enabled in (False, True):
        with pytest.raises(ValueError, match="rank-0 SVD bond"):
            _sweeps(sites, 4, 1, enabled, monkeypatch)
