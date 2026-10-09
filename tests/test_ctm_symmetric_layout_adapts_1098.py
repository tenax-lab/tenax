"""#1098: the symmetric ``python_loop_ctm_converge`` must adapt the chi layout.

Every sweep of ``python_loop_ctm_converge`` ran under ``jax.jit``.  With
``SymmetricTensor`` input the 2x2 projector then takes its traced branch, which
sizes each charge sector from the leg it overwrites (``_seam_chi_charges``,
#929).  So the chi-leg layout never left the cold-init tiling, and the
converged energy was the energy of that restricted layout.  The fix runs eager
sweeps first, until the layout stops changing, and then the jitted step.

Fixture: D=2 XXZ Delta=0.3, 2-site simple update (200 steps, dt=0.05,
``PRNGKey(0)``), float64, ``CHECKERBOARD_NEIGHBORS``; CTM chi=6 (U(1)-Sz) and
chi=8 (Z2).  Measured before the fix: symmetric minus dense energy 6.6e-6
(U(1)-Sz, chi=6) and 8.3e-5 (Z2, chi=8).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import tenax.algorithms._ctm_python_loop as cpl
from tenax import CTMConfig, ipeps, iPEPSConfig
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import ZnSymmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

jax.config.update("jax_enable_x64", True)

D = 2
DELTA = 0.3
# chi per symmetry.  U(1): chi=6 cuts the eager C1 spectrum in a clean gap
# (relative 0.62); chi=8 cuts a three-fold degenerate set (0.00937 x3, two
# kept) where which two survive moves E by ~6e-11.  Z2 chi=8 cuts cleanly.
CHI = {"u1": 6, "z2": 8}
SU_CHI = 8
GATE = xxz_gate(DELTA)


def _z2_init_pair(key):
    """Random Z2 pair: Sz up -> charge 0, down -> charge 1; virtual [0, 1]."""
    sym = ZnSymmetry(2)
    virt = np.array([i % 2 for i in range(D)], dtype=np.int32)
    phys = np.array([0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    kA, kB = jax.random.split(key)
    return (
        SymmetricTensor.random_normal(indices, kA),
        SymmetricTensor.random_normal(indices, kB),
    )


def _su_pair(case):
    init = {
        "u1": lambda k: heisenberg_u1sz_init_pair(D, k),
        "z2": _z2_init_pair,
    }[case]
    A0, B0 = init(jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=D, num_imaginary_steps=200, dt=0.05, ctm=CTMConfig(chi=SU_CHI)
    )
    _, (A, B), _ = ipeps(GATE, (A0, B0), su, compute_energy=False)
    A = A * (1.0 / (A.norm() + 1e-10))
    B = B * (1.0 / (B.norm() + 1e-10))
    assert isinstance(A, SymmetricTensor) and isinstance(B, SymmetricTensor)
    return A, B


def _kwargs(chi, max_iter=1000, min_iter=10, conv_tol=1e-12, env_init=None):
    cfg = dataclasses.replace(
        CTMConfig(chi=chi),
        conv_tol=conv_tol,
        max_iter=max_iter,
        min_iter=min_iter,
        plateau_patience=None,
        on_unconverged="warn",
    )
    return ctm_converge_kwargs(cfg, env_init=env_init)


def _run(A, B, chi, **kw):
    extra = {k: kw.pop(k) for k in ("eager_layout_sweeps",) if k in kw}
    return python_loop_ctm_converge(
        {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, **_kwargs(chi, **kw), **extra
    )


def _energy(A, B, envs):
    g = GATE.todense()
    return float(
        compute_energy_ctm_tensor_2site(A, B, envs[(0, 0)], envs[(1, 0)], g, 2)
    )


def _layout(t):
    return tuple(
        tuple(
            sorted(
                zip(
                    np.asarray(ix.sectors).tolist(),
                    np.asarray(ix.multiplicities).tolist(),
                )
            )
        )
        for ix in t.indices
    )


def _corner_spectrum(t):
    if isinstance(t, SymmetricTensor):
        s = np.concatenate(
            [np.linalg.svd(np.asarray(b), compute_uv=False) for b in t.blocks.values()]
        )
    else:
        s = np.linalg.svd(np.asarray(t.todense()), compute_uv=False)
    s = np.sort(s)[::-1]
    return s / s[0]


@pytest.fixture(scope="module", params=["u1", "z2"])
def case(request):
    """One symmetry's SU state, its dense reference, and the symmetric CTM
    under test (cold start, default ``eager_layout_sweeps``).  Built once."""
    A, B = _su_pair(request.param)
    chi = CHI[request.param]
    Ad = DenseTensor(A.todense(), A.indices)
    Bd = DenseTensor(B.todense(), B.indices)
    # 200 dense sweeps: the energy equals the 1000-sweep value to 1e-16 here
    # (measured; the dense criterion never certifies on this fixture).
    envs_d, _ = _run(Ad, Bd, chi, max_iter=200)
    envs, info = _run(A, B, chi)
    return SimpleNamespace(
        name=request.param,
        chi=chi,
        A=A,
        B=B,
        Ad=Ad,
        Bd=Bd,
        envs_d=envs_d,
        E_dense=_energy(Ad, Bd, envs_d),
        envs=envs,
        info=info,
    )


def test_the_jitted_symmetric_ctm_reaches_the_dense_energy(case):
    name, A, B, E_dense, envs, info = (
        case.name,
        case.A,
        case.B,
        case.E_dense,
        case.envs,
        case.info,
    )
    E_sym = _energy(A, B, envs)
    assert info.converged, (name, info)
    assert abs(E_sym - E_dense) < 1e-10, (name, E_sym, E_dense, E_sym - E_dense)

    # The layout moved off the cold-init tiling (D^2 fused charges + q=0 pad).
    cold = initialize_ctm_tensor_env(A, case.chi)
    assert _layout(envs[(0, 0)].C1) != _layout(cold.C1), name

    # Regime guard: the traced-only path (the pre-fix behaviour, reachable
    # through the opt-out) still sits on the frozen layout with a real gap.
    # If this ever fails, the fixture no longer exercises #1098.
    envs_frozen, _ = _run(A, B, case.chi, eager_layout_sweeps=0)
    gap_frozen = abs(_energy(A, B, envs_frozen) - E_dense)
    assert gap_frozen > 1e-7, (name, gap_frozen)
    assert _layout(envs_frozen[(0, 0)].C1) == _layout(cold.C1), name


def test_the_kept_corner_spectrum_is_the_dense_top_chi(case):
    """Guard the property, not the energy: no slot of the symmetric corner
    holds an empty (zero) singular value while the dense spectrum still has
    weight there, and the kept spectrum is the dense top-chi spectrum.

    Both cuts sit in a clean gap of the spectrum (see ``CHI``).  At the
    degenerate U(1) chi=8 cut the dense corner splits a +-2 pair by 4.8e-5
    that the symmetric corner keeps exactly degenerate, so this comparison
    would need ~5e-5 there.
    """
    name, envs, envs_d = case.name, case.envs, case.envs_d
    atol = 1e-8
    for c in envs:
        s_sym = _corner_spectrum(envs[c].C1)
        s_dense = _corner_spectrum(envs_d[c].C1)
        assert s_sym.shape == (case.chi,) and s_dense.shape == (case.chi,)
        assert s_dense[-1] > 1e-6, (name, c, s_dense)  # regime: no dense zero
        assert s_sym[-1] > 1e-12 * s_sym[0], (name, c, s_sym)
        np.testing.assert_allclose(s_sym, s_dense, atol=atol, err_msg=f"{name}{c}")


def _count_eager(monkeypatch):
    calls = []
    real = cpl._eager_ctm_sweep

    def _spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(cpl, "_eager_ctm_sweep", _spy)
    return calls


def test_a_settled_warm_start_costs_one_eager_sweep_and_no_recompile(case, monkeypatch):
    """Requirement 4, and the cost measured in a 2-site optimize_gs_ad: a warm
    start whose layout has settled confirms it with one eager sweep, and a
    second warm start at perturbed tensors (a line-search probe) compiles
    nothing new -- the eager output's chi legs come back in the charge order
    the jitted step was compiled for."""
    name, A, B, envs = case.name, case.A, case.B, case.envs
    calls = _count_eager(monkeypatch)
    envs2, _ = _run(A, B, case.chi, env_init=envs, max_iter=20)
    assert len(calls) == 1, (name, len(calls))
    assert abs(_energy(A, B, envs2) - _energy(A, B, envs)) < 1e-10

    step = cpl._make_jit_ctm_step(CHECKERBOARD_NEIGHBORS)
    n_compiled = step._cache_size()
    kA, kB = jax.random.split(jax.random.PRNGKey(7))
    Ap = A + SymmetricTensor.random_normal(A.indices, kA) * 1e-3
    Bp = B + SymmetricTensor.random_normal(B.indices, kB) * 1e-3
    _run(Ap, Bp, case.chi, env_init=envs2, max_iter=20)
    assert len(calls) == 2, (name, len(calls))
    assert step._cache_size() == n_compiled, (name, n_compiled, step._cache_size())


def test_a_warm_start_from_the_frozen_layout_still_adapts(case, monkeypatch):
    """Requirement 4: an env with the cold-init layout (here the fixed point
    of the traced-only path) is the bug, not a settled warm start; it gets
    the two-sweep rule and reaches the dense energy."""
    name, A, B, E_dense = case.name, case.A, case.B, case.E_dense
    frozen, _ = _run(A, B, case.chi, eager_layout_sweeps=0)
    calls = _count_eager(monkeypatch)
    envs, info = _run(A, B, case.chi, env_init=frozen)
    assert len(calls) >= 2, (name, len(calls))
    assert info.converged, (name, info)
    assert abs(_energy(A, B, envs) - E_dense) < 1e-10, name


def test_the_dense_path_never_enters_the_eager_phase(case, monkeypatch):
    """Requirement: dense input keeps its behaviour and its compiled code."""
    Ad, Bd, E_dense = case.Ad, case.Bd, case.E_dense

    def _boom(*a, **k):
        raise AssertionError("eager layout phase entered on dense input")

    monkeypatch.setattr(cpl, "_eager_ctm_sweep", _boom)
    envs, info = _run(Ad, Bd, case.chi, max_iter=200)
    envs_off, info_off = _run(Ad, Bd, case.chi, max_iter=200, eager_layout_sweeps=0)
    assert _energy(Ad, Bd, envs) == E_dense == _energy(Ad, Bd, envs_off)
    assert info.iterations == info_off.iterations == 200


def test_the_cap_warns_and_the_eager_sweeps_count_toward_max_iter(case, monkeypatch):
    """A cap of 1 cannot certify a layout (that needs two consecutive eager
    sweeps), so it warns; and the eager sweep is spent from ``max_iter``."""
    name, A, B = case.name, case.A, case.B
    loop_calls = []
    real_loop = cpl._run_ctm_loop_with_bump

    def _loop_spy(*a, **k):
        res = real_loop(*a, **k)
        loop_calls.append((k["max_iter"], res.iterations))
        return res

    monkeypatch.setattr(cpl, "_run_ctm_loop_with_bump", _loop_spy)
    with pytest.warns(UserWarning, match=r"eager_layout_sweeps=1 .*#1098"):
        _, info = _run(
            A, B, case.chi, max_iter=4, min_iter=0, conv_tol=0.0, eager_layout_sweeps=1
        )
    # 1 eager sweep + 3 jitted sweeps: the eager one is inside the budget and
    # inside the reported counts.
    assert loop_calls == [(3, 3)], (name, loop_calls)
    assert info.iterations == 4, (name, info)
    assert info.best_iteration == 4, (name, info)


def test_running_out_of_max_iter_inside_the_eager_phase_warns(case):
    """A cold start needs two eager sweeps to certify; ``max_iter=1`` ends
    the phase first, and says so (not only the ``eager_layout_sweeps`` cap)."""
    name, A, B = case.name, case.A, case.B
    with pytest.warns(UserWarning, match=r"max_iter=1\).*#1098"):
        _, info = _run(A, B, case.chi, max_iter=1, min_iter=0, conv_tol=0.0)
    assert info.iterations == 1, (name, info)
