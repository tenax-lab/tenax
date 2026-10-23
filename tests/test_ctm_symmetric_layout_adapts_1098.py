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
import warnings
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import tenax.algorithms._ctm_python_loop as cpl
from tenax import CTMConfig, ipeps, iPEPSConfig
from tenax.algorithms._ctm_convergence_policy import _plateau_bailed
from tenax.algorithms._ctm_loop_core import CTMLoopResult
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
    extra = {k: kw.pop(k) for k in ("eager_layout_sweeps", "recipe") if k in kw}
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


def _watch_probes(monkeypatch, drop_records=False):
    """Record every ``probe=True`` call of the jitted step; with
    ``drop_records`` the probe returns no records (as if it saw no cut)."""
    real_make = cpl._make_jit_ctm_step
    probes = []

    def _make(*a, **k):
        step = real_make(*a, **k)

        def _w(*aa, **kk):
            out = step(*aa, **kk)
            if kk.get("probe"):
                probes.append(1)
                if drop_records:
                    return (*out[:3], ())
            return out

        return _w

    monkeypatch.setattr(cpl, "_make_jit_ctm_step", _make)
    return probes


def _count_eager(monkeypatch):
    calls = []
    real = cpl._eager_ctm_sweep

    def _spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(cpl, "_eager_ctm_sweep", _spy)
    return calls


def test_a_settled_warm_start_runs_no_eager_sweep_and_no_recompile(case, monkeypatch):
    """A warm start whose layout is already the eager top-chi layout is
    certified by one jitted probe sweep: no eager sweep.  A second warm start
    at perturbed tensors (a line-search probe) adds no ``jit(_step)`` cache
    entry -- the probe has static shapes and compiles once per layout."""
    name, A, B, envs = case.name, case.A, case.B, case.envs
    calls = _count_eager(monkeypatch)
    envs2, _ = _run(A, B, case.chi, env_init=envs, max_iter=20)
    assert len(calls) == 0, (name, len(calls))
    assert abs(_energy(A, B, envs2) - _energy(A, B, envs)) < 1e-10

    step = cpl._make_jit_ctm_step(CHECKERBOARD_NEIGHBORS)
    n_compiled = step._cache_size()
    kA, kB = jax.random.split(jax.random.PRNGKey(7))
    Ap = A + SymmetricTensor.random_normal(A.indices, kA) * 1e-3
    Bp = B + SymmetricTensor.random_normal(B.indices, kB) * 1e-3
    _run(Ap, Bp, case.chi, env_init=envs2, max_iter=20)
    assert len(calls) == 0, (name, len(calls))
    assert step._cache_size() == n_compiled, (name, n_compiled, step._cache_size())


def test_a_warm_start_on_a_suboptimal_layout_is_caught_by_the_probe(case, monkeypatch):
    """The probe must say "different" when the handed-in layout is neither
    the cold tiling nor the eager top-chi one.  Built by freezing (traced-only
    CTM) the layout of one eager sweep from the cold init.  U(1) only: for
    Z2 that first sweep returns the cold tiling itself, which skips the probe
    (see the cold-init test below)."""
    name, A, B, E_dense = case.name, case.A, case.B, case.E_dense
    if name != "u1":
        pytest.skip("Z2's first eager sweep returns the cold tiling")
    st = {(0, 0): A, (1, 0): B}
    cold = {c: initialize_ctm_tensor_env(t, case.chi) for c, t in st.items()}
    mid, _, _ = cpl._eager_ctm_sweep(
        st,
        cold,
        neighbors=CHECKERBOARD_NEIGHBORS,
        recipe="2x2",
        device_mesh=None,
        chi=case.chi,
        projector_method="svd",
        renormalize=False,
        projector_backward="auto",
        chunk_size=None,
    )
    mid = cpl._charge_sorted_envs(mid)
    stuck, _ = _run(A, B, case.chi, env_init=mid, eager_layout_sweeps=0)
    # Regime: off the cold tiling, and a real energy gap.
    assert _layout(stuck[(0, 0)].C1) != _layout(cold[(0, 0)].C1), name
    assert abs(_energy(A, B, stuck) - E_dense) > 1e-7, name

    calls = _count_eager(monkeypatch)
    envs, info = _run(A, B, case.chi, env_init=stuck)
    assert len(calls) >= 1, (name, len(calls))
    assert info.converged, (name, info)
    assert abs(_energy(A, B, envs) - E_dense) < 1e-10, name


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


def test_a_warm_start_from_the_cold_init_env_still_adapts(case):
    """Requirement 4, the rank-1 case: a caller that passes the cold init
    itself as ``env_init`` (as the envs_init seeding tests do).  From it the
    first Z2 eager sweep returns the initial tiling unchanged, so a one-sweep
    "output equals input" rule would stop on the frozen layout."""
    name, A, B, E_dense = case.name, case.A, case.B, case.E_dense
    cold = {(0, 0): initialize_ctm_tensor_env(A, case.chi)}
    cold[(1, 0)] = initialize_ctm_tensor_env(B, case.chi)
    envs, info = _run(A, B, case.chi, env_init=cold)
    assert info.converged, (name, info)
    assert abs(_energy(A, B, envs) - E_dense) < 1e-10, name


def test_the_dense_path_never_enters_the_eager_phase(case, monkeypatch):
    """Requirement: dense input keeps its behaviour and its compiled code --
    no eager phase, no end-of-call check sweep, no new ``_step`` compile."""
    Ad, Bd, E_dense = case.Ad, case.Bd, case.E_dense

    def _boom(*a, **k):
        raise AssertionError("eager layout phase entered on dense input")

    def _boom_check(*a, **k):
        raise AssertionError("end-of-call layout check ran on dense input")

    monkeypatch.setattr(cpl, "_eager_ctm_sweep", _boom)
    monkeypatch.setattr(cpl, "_end_of_call_layout_check", _boom_check)
    step = cpl._make_jit_ctm_step(CHECKERBOARD_NEIGHBORS)
    probes = _watch_probes(monkeypatch)
    envs, info = _run(Ad, Bd, case.chi, max_iter=200)
    n_compiled = step._cache_size()
    envs_off, info_off = _run(Ad, Bd, case.chi, max_iter=200, eager_layout_sweeps=0)
    assert _energy(Ad, Bd, envs) == E_dense == _energy(Ad, Bd, envs_off)
    assert info.iterations == info_off.iterations == 200
    assert info.best_iteration == 200, info
    # The opt-out (pre-#1098) run reuses every executable the default run
    # compiled: no probe step, no new trace.
    assert step._cache_size() == n_compiled, (n_compiled, step._cache_size())
    assert probes == [], len(probes)  # no probe sweep, so no probe compile


def test_the_cap_warns_and_the_eager_sweeps_count_toward_max_iter(case, monkeypatch):
    """A cap of 1 cannot certify a layout (that needs two consecutive eager
    sweeps), so it warns; and the eager sweep is spent from ``max_iter``, as
    is the end-of-call check sweep, which the loop leaves room for."""
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
    # inside the reported counts.  The phase ended unstable (and warned), so
    # no end-of-call check runs and no sweep is held back for one.
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


def test_a_qr_warm_up_that_uses_the_whole_budget_warns(case, monkeypatch):
    """With ``projector_method="qr"`` the jitted eigh warm-up runs first.  If it
    uses all of ``max_iter``, no eager sweep is left, so the layout is not
    adapted and the end-of-call check does not run; that must not be silent
    (Codex on #1132)."""
    name, A, B = case.name, case.A, case.B
    eager = _count_eager(monkeypatch)
    with pytest.warns(UserWarning, match=r"QR warm-up.*max_iter=3.*#1098"):
        kw = {
            **_kwargs(case.chi, max_iter=3, min_iter=0, conv_tol=0.0),
            "projector_method": "qr",
            "qr_warmup_steps": 3,
        }
        python_loop_ctm_converge({(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, **kw)
    assert eager == [], (name, eager)


def test_a_1x1_warm_start_runs_no_probe_sweep(case, monkeypatch):
    """The probe records only the traced 2x2 cut, so under ``recipe="1x1"``
    it could never certify a layout; it must be skipped, not run in front of
    the eager sweeps on every warm call (#1098 review, R14)."""
    name, A, B, envs = case.name, case.A, case.B, case.envs
    st = {(0, 0): A, (1, 0): B}
    # Regime: off the cold tiling, so a probe *would* be attempted for 2x2.
    assert cpl._chi_layout(envs) != cpl._cold_init_layout(st, case.chi), name
    assert cpl._probe_can_certify(st, "2x2"), name
    decisions = []
    real = cpl._probe_says_layout_is_eager

    def _spy(*a, **k):
        decisions.append(1)
        return real(*a, **k)

    monkeypatch.setattr(cpl, "_probe_says_layout_is_eager", _spy)
    eager = _count_eager(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # max_iter=1: the budget ends inside the eager phase, so no jitted
        # 1x1 sweep runs (on the U(1) fixture the jitted 1x1 sweep raises a
        # label-size mismatch in _ctm_tensor_move_bottom, unrelated to #1098).
        _run(A, B, case.chi, env_init=envs, max_iter=1, min_iter=0, recipe="1x1")
    assert decisions == [], (name, len(decisions))
    assert len(eager) >= 1, (name, len(eager))  # round-1 eager phase instead


# ---------------------------------------------------------------------------
# End-of-call layout check (#1098 fix round 4).
# ---------------------------------------------------------------------------


def _spy(monkeypatch, name):
    """Record every return value of the module-level function ``name``."""
    out = []
    real = getattr(cpl, name)

    def _s(*a, **k):
        r = real(*a, **k)
        out.append(r)
        return r

    monkeypatch.setattr(cpl, name, _s)
    return out


def _spy_loop(monkeypatch):
    out = []
    real = cpl._run_ctm_loop_with_bump

    def _s(*a, **k):
        r = real(*a, **k)
        out.append(r)
        return r

    monkeypatch.setattr(cpl, "_run_ctm_loop_with_bump", _s)
    return out


def test_the_end_of_call_check_passes_on_a_converged_call(case, monkeypatch):
    """A cold and a settled warm call each end with exactly one check sweep
    (here the jitted probe), it certifies the layout, the energy is the dense
    one, and the check sweep is counted but not returned."""
    name, A, B, E_dense = case.name, case.A, case.B, case.E_dense
    checks = _spy(monkeypatch, "_end_of_call_layout_check")
    decisions = _spy(monkeypatch, "_probe_says_layout_is_eager")
    loops = _spy_loop(monkeypatch)
    eager = _count_eager(monkeypatch)

    envs, info = _run(A, B, case.chi)
    assert checks == [(True, 1)], (name, checks)
    assert decisions == [True], (name, decisions)  # the check's; cold: no pre-probe
    assert info.converged, (name, info)
    assert abs(_energy(A, B, envs) - E_dense) < 1e-10, name
    (loop,) = loops
    assert info.iterations == len(eager) + loop.iterations + 1, (name, info)
    assert info.best_iteration == info.iterations, (name, info)  # no plateau

    checks.clear(), decisions.clear(), loops.clear(), eager.clear()
    envs2, info2 = _run(A, B, case.chi, env_init=envs, max_iter=20)
    assert checks == [(True, 1)], (name, checks)
    assert decisions == [True, True], (name, decisions)  # pre-loop probe + check
    assert eager == [], (name, len(eager))
    (loop,) = loops
    assert info2.iterations == 1 + loop.iterations + 1, (name, info2)
    assert info2.best_iteration == info2.iterations, (name, info2)
    assert abs(_energy(A, B, envs2) - E_dense) < 1e-10, name


def _force_frozen_pre_loop_layout(monkeypatch, frozen):
    """Make the *pre-loop* layout phase return ``frozen`` as if two eager
    sweeps had certified it.  Later (re-entry) calls run the real phase.

    This forces the hazard the end-of-call check exists for -- the loop
    converging on a layout that is not a top-chi cut of its own fixed point
    -- without waiting for a spectrum that moves after the choice.  The
    frozen layout is the traced-only fixed point (the cold tiling), whose
    energy gap is asserted by ``test_the_jitted_symmetric_ctm_reaches...``.
    """
    real = cpl._adapt_symmetric_layout
    calls = []

    def _fake(site_tensors, envs, **kw):
        calls.append(kw["compare_input"])
        if len(calls) == 1:
            return frozen, 2, "stable"
        return real(site_tensors, envs, **kw)

    monkeypatch.setattr(cpl, "_adapt_symmetric_layout", _fake)
    return calls


def test_the_end_of_call_check_catches_a_layout_the_loop_left_behind(case, monkeypatch):
    name, A, B, E_dense = case.name, case.A, case.B, case.E_dense
    frozen, _ = _run(A, B, case.chi, eager_layout_sweeps=0)
    assert abs(_energy(A, B, frozen) - E_dense) > 1e-7, name  # regime

    adapt_calls = _force_frozen_pre_loop_layout(monkeypatch, frozen)
    checks = _spy(monkeypatch, "_end_of_call_layout_check")
    loops = _spy_loop(monkeypatch)
    eager = _count_eager(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        envs, info = _run(A, B, case.chi)
    assert not [w for w in caught if "end-of-call" in str(w.message)], name

    # The check rejected the frozen layout, the phase re-entered once (from
    # the returned env, one-sweep rule), and the second check passed.
    assert adapt_calls == [False, True], (name, adapt_calls)
    assert checks == [(False, 1), (True, 1)], (name, checks)
    assert info.converged, (name, info)
    assert abs(_energy(A, B, envs) - E_dense) < 1e-10, name
    # Counts: 2 (forced phase) + loop + check + eager re-entry + loop + check.
    l0, l1 = loops
    n_eager = len(eager)  # the checks are probes here, so all eager = re-entry
    assert n_eager >= 1, name
    assert info.iterations == 2 + l0.iterations + 1 + n_eager + l1.iterations + 1
    assert info.best_iteration == info.iterations, (name, info)


def test_a_failed_check_with_no_budget_left_warns(case, monkeypatch):
    """Same forced layout, but ``max_iter`` ends at the check: no re-entry,
    a warning, and the loop's env and counts are returned."""
    name, A, B = case.name, case.A, case.B
    frozen, _ = _run(A, B, case.chi, eager_layout_sweeps=0)
    adapt_calls = _force_frozen_pre_loop_layout(monkeypatch, frozen)
    checks = _spy(monkeypatch, "_end_of_call_layout_check")
    with pytest.warns(UserWarning, match=r"end-of-call layout check.*#1098"):
        envs, info = _run(A, B, case.chi, max_iter=10, min_iter=0, conv_tol=0.0)
    assert adapt_calls == [False], (name, adapt_calls)
    assert checks == [(False, 1)], (name, checks)
    # 2 (forced phase) + 7 jitted + 1 check.  The budget ended the run, so
    # it must not read as a plateau bail (#781): best == iterations.
    assert info.iterations == 10, (name, info)
    assert info.best_iteration == 10, (name, info)
    assert not _plateau_bailed(info), (name, info)
    assert not info.converged, (name, info)
    assert cpl._chi_layout(envs) == cpl._chi_layout(frozen), name


def test_the_re_entries_are_capped(case, monkeypatch):
    """A check that never certifies stops after the cap of re-entries, with
    budget left, and warns."""
    name, A, B = case.name, case.A, case.B
    cap = cpl._LAYOUT_RECHECK_REENTRIES
    assert cap == 2
    real = cpl._end_of_call_layout_check
    checks = []

    def _never(*a, **k):
        r = real(*a, **k)
        checks.append(r)
        return False, r[1]

    monkeypatch.setattr(cpl, "_end_of_call_layout_check", _never)
    adapt = _spy(monkeypatch, "_adapt_symmetric_layout")
    with pytest.warns(UserWarning, match=r"after 2 re-entries.*#1098"):
        _, info = _run(A, B, case.chi, env_init=case.envs, max_iter=500)
    assert len(checks) == cap + 1, (name, checks)
    assert len(adapt) == cap, (name, len(adapt))  # warm: the probe certified
    assert info.iterations < 500, (name, info)


def test_a_1x1_call_checks_with_one_eager_sweep_and_no_probe(case, monkeypatch):
    """Where the probe cannot certify (``recipe="1x1"``), the end-of-call
    check is one eager sweep.  The jitted 1x1 loop is replaced by a stand-in
    that returns its input after one counted sweep: on this fixture the
    jitted 1x1 sweep raises a label-size mismatch in
    ``_ctm_tensor_move_bottom`` (pre-existing, also on base 847480e)."""
    name, A, B, envs = case.name, case.A, case.B, case.envs

    def _fake_loop(jit_step, site_tensors, envs_init, **k):
        return CTMLoopResult(
            envs=envs_init,
            converged=True,
            iterations=1,
            sv_diff=0.0,
            max_truncation_error=0.0,
            max_smallest_S=0.0,
            final_chi=k["chi_current"],
            bump_extra_sweeps=0,
            best_iteration=1,
        )

    monkeypatch.setattr(cpl, "_run_ctm_loop_with_bump", _fake_loop)
    decisions = _spy(monkeypatch, "_probe_says_layout_is_eager")
    eager = _count_eager(monkeypatch)
    real = cpl._end_of_call_layout_check
    checks = []

    def _check(*a, **k):
        n0 = len(eager)
        r = real(*a, **k)
        checks.append((*r, len(eager) - n0))
        return r

    monkeypatch.setattr(cpl, "_end_of_call_layout_check", _check)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, info = _run(A, B, case.chi, env_init=envs, max_iter=100, recipe="1x1")
    assert decisions == [], (name, decisions)
    # One check, one eager sweep inside it, and it certifies the layout the
    # eager phase settled on (the stand-in loop returns that env unchanged).
    assert checks == [(True, 1, 1)], (name, checks)
    n_phase = len(eager) - 1
    assert n_phase >= 1, (name, len(eager))
    assert info.iterations == n_phase + 1 + 1, (name, info)
    assert info.best_iteration == info.iterations, (name, info)


def test_every_sweep_that_runs_is_counted(case, monkeypatch):
    """Review I2: a probe that records no cut (forced here) is a sweep that
    ran -- counted when the pre-loop phase rejects it, and counted with the
    eager sweep the end-of-call check then falls back to."""
    name, A, B = case.name, case.A, case.B
    probes = _watch_probes(monkeypatch, drop_records=True)
    eager = _count_eager(monkeypatch)
    loops = _spy_loop(monkeypatch)
    real = cpl._end_of_call_layout_check
    checks = []

    def _check(*a, **k):
        n0 = len(eager)
        r = real(*a, **k)
        checks.append((*r, len(eager) - n0))
        return r

    monkeypatch.setattr(cpl, "_end_of_call_layout_check", _check)
    _, info = _run(A, B, case.chi, env_init=case.envs, max_iter=100)
    assert len(probes) == 2, (name, len(probes))  # pre-loop + check
    # The check: one (empty) probe + one eager sweep, both counted.
    assert checks == [(True, 2, 1)], (name, checks)
    (loop,) = loops
    n_phase = len(eager) - 1
    assert n_phase >= 1, (name, len(eager))
    # rejected probe + eager phase + jitted loop + check (2 sweeps)
    assert info.iterations == 1 + n_phase + loop.iterations + 2, (name, info)


def test_a_two_cycle_in_the_eager_phase_stops_and_warns_once(case, monkeypatch):
    """Review I3: an eager phase whose cut alternates between two layouts
    stops at the first repeat (not at the cap), warns once, and runs no
    end-of-call check or re-entry.  The eager sweep is a stand-in that
    alternates two real envs with different layouts.

    Cold ``X, Y, X, Y`` with ``Y`` the cold-init tiling (regime asserted).
    The first cold output does not count (round 7), so sweep 3 (``X`` again)
    is no cycle; sweep 4 repeats ``o_2 = Y`` and is.  The earlier layout of
    the detected pair is ``Y`` = the tiling, and ``o_3 = X`` is not, so the
    M8 rule keeps ``X``."""
    name, A, B = case.name, case.A, case.B
    st = {(0, 0): A, (1, 0): B}
    frozen, _ = _run(A, B, case.chi, eager_layout_sweeps=0)
    X, Y = case.envs, frozen
    assert cpl._chi_layout(X) != cpl._chi_layout(Y), name  # regime
    assert cpl._chi_layout(Y) == cpl._cold_init_layout(st, case.chi), name
    seq = []

    def _alt(site_tensors, envs, **k):
        out = (X, Y)[len(seq) % 2]
        seq.append(1)
        return out, 0.0, 0.0

    monkeypatch.setattr(cpl, "_eager_ctm_sweep", _alt)
    checks = _spy(monkeypatch, "_end_of_call_layout_check")
    loops = _spy_loop(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        envs, info = _run(A, B, case.chi)  # cold: X, Y, X, Y
    msgs = [str(w.message) for w in caught if "#1098" in str(w.message)]
    assert len(msgs) == 1 and "alternated" in msgs[0], (name, msgs)
    assert len(seq) == 4, (name, len(seq))  # stopped at the repeat of o_2
    assert checks == [], (name, checks)
    assert cpl._chi_layout(envs) == cpl._chi_layout(X), name  # M8: not Y
    (loop,) = loops
    assert info.iterations == 4 + loop.iterations, (name, info)


def test_a_re_entry_that_settles_on_the_last_sweep_does_not_warn(case, monkeypatch):
    """Review M6: after a failed check, a re-entry whose eager phase certifies
    the layout but spends the last of ``max_iter`` leaves the loop no sweep.
    The returned env is the phase's certified output, so no "failed the
    end-of-call layout check" warning.  ``max_iter`` is set from an unlimited
    run of the same forced case: 2 (forced phase) + loop + 1 check + the
    re-entry's eager sweeps."""
    name, A, B = case.name, case.A, case.B
    frozen, _ = _run(A, B, case.chi, eager_layout_sweeps=0)

    with monkeypatch.context() as mp:
        _force_frozen_pre_loop_layout(mp, frozen)
        loops = _spy_loop(mp)
        eager = _count_eager(mp)
        _, info_big = _run(A, B, case.chi)
    k = loops[0].iterations
    n_eager = len(eager)
    assert n_eager >= 1, name  # regime: the re-entry ran eager sweeps
    max_iter = 2 + k + 1 + n_eager
    assert max_iter < info_big.iterations, (name, max_iter, info_big)

    adapt_calls = _force_frozen_pre_loop_layout(monkeypatch, frozen)
    checks = _spy(monkeypatch, "_end_of_call_layout_check")
    loops = _spy_loop(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        envs, info = _run(A, B, case.chi, max_iter=max_iter)
    msgs = [str(w.message) for w in caught if "end-of-call" in str(w.message)]
    assert msgs == [], (name, msgs)
    assert adapt_calls == [False, True], (name, adapt_calls)
    assert checks == [(False, 1)], (name, checks)
    assert [r.iterations for r in loops] == [k, 0], (name, loops)
    assert info.iterations == max_iter, (name, info)
    assert cpl._chi_layout(envs) != cpl._chi_layout(frozen), name


def test_iterations_never_exceed_max_iter(case, monkeypatch):
    """Review M7: with probes that record no cut (forced), the check's eager
    fallback needs a second sweep.  When the loop has used its budget only
    the reserved sweep is left, so the check stops after the probe and the
    call reports exactly ``max_iter`` (and warns: nothing certified it)."""
    name, A, B = case.name, case.A, case.B
    _watch_probes(monkeypatch, drop_records=True)
    checks = _spy(monkeypatch, "_end_of_call_layout_check")
    max_iter = 8
    with pytest.warns(UserWarning, match=r"end-of-call layout check.*#1098"):
        _, info = _run(
            A,
            B,
            case.chi,
            env_init=case.envs,
            max_iter=max_iter,
            min_iter=0,
            conv_tol=0.0,
        )
    assert checks == [(False, 1)], (name, checks)
    assert info.iterations == max_iter, (name, info)


def test_a_cold_two_cycle_through_the_cold_tiling_keeps_the_other_layout(
    case, monkeypatch
):
    """Review M8, under the round-7 rule: from the rank-1 cold init the first
    eager output can be the cold-init tiling T, and it does not take part in
    the two-cycle test.  So a cold ``T, X, T, X`` is no cycle at sweep 3; it
    is one at sweep 4 (``o_4 = o_2 = X``, ``o_3 = T``).  The earlier layout
    of the detected pair is ``X``, not the tiling, so the phase keeps the
    current output (``X``) by the plain rule; M8 does not fire."""
    name, A, B = case.name, case.A, case.B
    st = {(0, 0): A, (1, 0): B}
    frozen, _ = _run(A, B, case.chi, eager_layout_sweeps=0)
    T_env, X_env = frozen, case.envs
    cold = cpl._cold_init_layout(st, case.chi)
    assert cpl._chi_layout(T_env) == cold, name  # regime: T is the tiling
    assert cpl._chi_layout(X_env) != cold, name
    seq = []

    def _alt(site_tensors, envs, **k):
        out = (T_env, X_env)[len(seq) % 2]
        seq.append(1)
        return out, 0.0, 0.0

    monkeypatch.setattr(cpl, "_eager_ctm_sweep", _alt)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        envs, _ = _run(A, B, case.chi)  # cold: T, X, T, X
    msgs = [str(w.message) for w in caught if "#1098" in str(w.message)]
    assert len(msgs) == 1 and "alternated" in msgs[0], (name, msgs)
    assert len(seq) == 4, (name, len(seq))
    assert cpl._chi_layout(envs) == cpl._chi_layout(X_env), name


def test_a_cold_first_output_that_echoes_the_tiling_is_not_a_two_cycle(
    case, monkeypatch
):
    """Round 7 (measured, U(1) XXZ Delta=0.3 D=3 chi=9, every sweep eager:
    ``T, X, T, T, ...`` settling on the tiling T).  The first cold output
    does not take part in the two-cycle test, so a cold ``T, X, T, T, T``
    is stable at sweep 4 with ``T``: no "alternated" warning, and the
    end-of-call check runs (it is skipped after a cycle)."""
    name, A, B = case.name, case.A, case.B
    st = {(0, 0): A, (1, 0): B}
    frozen, _ = _run(A, B, case.chi, eager_layout_sweeps=0)
    T_env, X_env = frozen, case.envs
    cold = cpl._cold_init_layout(st, case.chi)
    assert cpl._chi_layout(T_env) == cold, name  # regime: T is the tiling
    assert cpl._chi_layout(X_env) != cold, name
    seq = []

    def _settle(site_tensors, envs, **k):
        out = (T_env, X_env, T_env)[len(seq)] if len(seq) < 3 else T_env
        seq.append(1)
        return out, 0.0, 0.0

    monkeypatch.setattr(cpl, "_eager_ctm_sweep", _settle)
    phases = _spy(monkeypatch, "_adapt_symmetric_layout")
    checks = _spy(monkeypatch, "_end_of_call_layout_check")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        envs, _ = _run(A, B, case.chi)  # cold: T, X, T, T, ...
    msgs = [str(w.message) for w in caught if "alternated" in str(w.message)]
    assert msgs == [], (name, msgs)
    _, n, outcome = phases[0]
    assert (n, outcome) == (4, "stable"), (name, n, outcome)
    assert checks, name  # the end-of-call check ran
    assert cpl._chi_layout(envs) == cold, name
