"""Mechanism tests of the charged C4v AD path (spec 2), at chi <= 6.

Design: ``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``.
The converged FD checks, the D=2 optimum and the dense parity are in
``test_c4v_charged_ad_physics.py`` (algorithm bucket).
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

from _c4v_staggered_fixtures import (
    T_EXACT,
    c4v_fixture,
    charged_d2,
    fd_plateau,
    frozen_projectors,
    leg,
    with_data,
)

from tenax import (
    CTMConfig,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
    sublattice_rotate_gate,
)
from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad
from tenax.algorithms import ipeps_optimize_c4v_charged as opt
from tenax.algorithms._ctm_convergence_policy import (
    CTMConvergeInfo,
    CTMNotConvergedError,
)
from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout
from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v
from tenax.algorithms._ctm_tensor_c4v_staggered import (
    _c4v_staggered_enlarged_corner,
    _c4v_staggered_layout,
    _c4v_staggered_loop,
    _c4v_staggered_sweep,
    _c4v_staggered_to_full_env,
)
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.algorithms.ipeps import symmetrize_c4v
from tenax.algorithms.ipeps_ad_policy import use_charged_c4v_path
from tenax.core.symmetry import U1Symmetry, ZnSymmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

_NAMES = ("d2_off", "762", "z4")


@functools.cache
def _site(name):
    """The site tensors of these tests, each built once."""
    if name == "d2_off":  # the D=2 exact U(1) state, moved off its optimum
        return charged_d2(1.15 * T_EXACT)
    if name == "762":  # the #762 fixture: charged U(1), 10 block entries
        return c4v_fixture(U1Symmetry(), [-1, 1], [-2, 0])
    if name == "z4":  # Z4: the pair (1, 3) and the self-partner charge 2
        return c4v_fixture(ZnSymmetry(4), [0, 1], [0, 1])
    raise KeyError(name)


@functools.cache
def _cold(name):
    """``(A, C, T, layout)``: the #1211 loop at chi=6 to sv_diff < 1e-10."""
    A = _site(name)
    C, T, _a, info = _c4v_staggered_loop(A, 6, 200, 1e-10)
    assert info.converged, name
    return A, C, T, ad._layout_of(C)


# ------------------------------------------------------------------ #
# The aligned sweep (spec 2, section 1)                               #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("name", "dims"),
    [
        ("d2_off", {-1: 2, 0: 2, 1: 2}),
        ("762", {-2: 1, 0: 4, 2: 1}),
        ("z4", {0: 2, 1: 2, 3: 2}),
    ],
)
def test_layout_of_reads_the_corner_charges(name, dims):
    assert _cold(name)[3] == BondLayout.from_dims(dims)


@pytest.mark.parametrize("name", _NAMES)
def test_align_recovers_a_pure_sign_gauge(name):
    _A, C, T, _layout = _cold(name)
    n = leg(T, "t_l").dim
    g = np.where(np.random.default_rng(0).random(n) < 0.5, -1.0, 1.0)
    g[0] = -g[-1]  # at least one relative sign
    g = jnp.asarray(g)
    C_g = ad._scale_chi(C, g, ("c_a", "c_b"))
    T_g = ad._scale_chi(T, g, ("t_l", "t_r"))
    assert float(ad._rel_change(T_g, T)) > 0.1, "vacuous: the gauge leaves T as it is"
    C_a, T_a = ad._align(C_g, T_g, T)
    np.testing.assert_array_equal(np.asarray(C_a._data), np.asarray(C._data))
    np.testing.assert_array_equal(np.asarray(T_a._data), np.asarray(T._data))


@pytest.mark.parametrize("name", _NAMES)
def test_sweep_at_is_the_eager_sweep_up_to_signs(name):
    A, C, T, layout = _cold(name)
    C_e, T_e = _c4v_staggered_sweep(C, T, _build_double_layer_tensor(A), 6)
    assert ad._layout_of(C_e) == layout
    C_t, T_t = ad._align(*ad._sweep_at(A, C, T, layout), T_e)
    np.testing.assert_allclose(np.asarray(C_t._data), np.asarray(C_e._data), atol=1e-12)
    np.testing.assert_allclose(np.asarray(T_t._data), np.asarray(T_e._data), atol=1e-12)


def test_step_compiles_once_and_matches_eager():
    A, C, T, layout = _cold("d2_off")
    jax.clear_caches()
    C1, T1, r1 = ad._step(A, C, T, layout)
    ad._step(A, C1, T1, layout)
    assert ad._step._cache_size() == 1
    C_e, T_e, r_e = ad._step_impl(A, C, T, layout)
    np.testing.assert_allclose(np.asarray(C1._data), np.asarray(C_e._data), atol=1e-13)
    np.testing.assert_allclose(np.asarray(T1._data), np.asarray(T_e._data), atol=1e-13)
    assert abs(float(r1) - float(r_e)) < 1e-13


# ------------------------------------------------------------------ #
# The forward phases and the custom_vjp (spec 2, sections 2-3)        #
# ------------------------------------------------------------------ #

CTM6 = CTMConfig(chi=6, max_iter=200, conv_tol=1e-10)


def _loss_through_converge(C, T, layout, record):
    """A_ -> <w, T*(A_)> through ``charged_c4v_converge``, w fixed and random."""
    w = jax.random.normal(jax.random.PRNGKey(3), T._data.shape)

    def loss(A_):
        _C1, T1 = ad.charged_c4v_converge(A_, (C, T), layout, CTM6, record)
        return jnp.sum(w * T1._data)

    return loss


def test_cold_start_freezes_the_last_layout():
    A, C, T, layout = _cold("d2_off")
    C1, T1, layout1 = ad.charged_c4v_cold_start(A, CTM6)
    assert layout1 == layout
    np.testing.assert_array_equal(np.asarray(C1._data), np.asarray(C._data))
    a = _build_double_layer_tensor(A)
    assert _c4v_staggered_layout(_c4v_staggered_enlarged_corner(C1, T1, a), 6) == layout


def test_cold_start_raises_when_the_sweep_does_not_converge():
    ctm = dataclasses.replace(CTM6, max_iter=3, conv_tol=1e-300)
    with pytest.raises(CTMNotConvergedError, match="cold start"):
        ad.charged_c4v_cold_start(_site("d2_off"), ctm)


def test_elementwise_phase_raises_at_max_iter():
    A, C, T, layout = _cold("d2_off")
    ctm = dataclasses.replace(CTM6, max_iter=2, conv_tol=1e-300)
    with pytest.raises(CTMNotConvergedError, match="element-wise"):
        ad._elementwise_phase(A, (C, T), layout, ctm, {})


def test_converge_returns_a_fixed_point():
    A, C, T, layout = _cold("d2_off")
    record = {}
    C1, T1 = ad.charged_c4v_converge(A, (C, T), layout, CTM6, record)
    _C2, _T2, res = ad._step(A, C1, T1, layout)
    assert float(res) < CTM6.conv_tol
    assert record["forward_sweeps"] >= 1
    assert record["forward_residual"] < CTM6.conv_tol


def test_elementwise_phase_needs_the_alignment(monkeypatch):
    A, C, T, layout = _cold("d2_off")
    ctm = dataclasses.replace(CTM6, conv_tol=1e-12, max_iter=60)
    flip = jnp.asarray(np.r_[-1.0, np.ones(leg(T, "t_l").dim - 1)])
    assert float(ad._rel_change(ad._scale_chi(T, flip, ("t_l", "t_r")), T)) > 0.1, (
        "vacuous: the flip leaves T as it is"
    )
    sweep, calls = ad._sweep_at, [0]

    def flipping_sweep(A_, C_, T_, layout_):
        C1, T1 = sweep(A_, C_, T_, layout_)
        calls[0] += 1
        if calls[0] % 2 == 0:  # a pure gauge: chi index 0 changes sign
            C1 = ad._scale_chi(C1, flip, ("c_a", "c_b"))
            T1 = ad._scale_chi(T1, flip, ("t_l", "t_r"))
        return C1, T1

    monkeypatch.setattr(ad, "_step", ad._step_impl)  # eager, so the patches act
    monkeypatch.setattr(ad, "_sweep_at", flipping_sweep)
    ad._elementwise_phase(A, (C, T), layout, ctm, {})
    assert calls[0] >= 2, "vacuous: no sweep was flipped"
    monkeypatch.setattr(ad, "_align", lambda C_new, T_new, T_ref: (C_new, T_new))
    with pytest.raises(CTMNotConvergedError, match="element-wise"):
        ad._elementwise_phase(A, (C, T), layout, ctm, {})


def test_backward_refuses_an_unsolved_adjoint(monkeypatch):
    A, C, T, layout = _cold("d2_off")
    monkeypatch.setattr(ad, "gmres_eager", lambda matvec, b, **kw: (b, 1.0))
    with pytest.raises(RuntimeError, match="relative residual"):
        jax.grad(_loss_through_converge(C, T, layout, {}))(A)


def test_backward_refuses_a_non_finite_adjoint(monkeypatch):
    A, C, T, layout = _cold("d2_off")
    monkeypatch.setattr(ad, "gmres_eager", lambda matvec, b, **kw: (b * jnp.nan, 0.0))
    with pytest.raises(RuntimeError, match="not finite"):
        jax.grad(_loss_through_converge(C, T, layout, {}))(A)


def test_programs_compile_once_per_layout():
    _A, C, T, layout = _cold("d2_off")
    jax.clear_caches()
    record = {}
    for t in (1.15 * T_EXACT, 1.1 * T_EXACT):
        A_t = charged_d2(t)
        g = jax.grad(_loss_through_converge(C, T, layout, record))(A_t)
        assert bool(jnp.all(jnp.isfinite(g._data)))
    assert ad._step._cache_size() == 1
    assert ad._adjoint_matvec._cache_size() == 1
    assert ad._site_vjp._cache_size() == 1
    assert set(record) == {
        "forward_sweeps",
        "forward_residual",
        "forward_rate",
        "adjoint_matvecs",
        "adjoint_residual",
    }
    assert record["adjoint_residual"] <= max(10 * CTM6.adjoint_tol, 1e-12)
    assert record["adjoint_matvecs"] >= 1


# ------------------------------------------------------------------ #
# One-sweep gradient checks (spec 2, section 6)                       #
# ------------------------------------------------------------------ #


def _one_sweep(C, T, layout):
    """A_ -> F~((C, T); A_, ref=T): the map whose VJP is (dF~/dA)^T."""
    return lambda A_: ad._align(*ad._sweep_at(A_, C, T, layout), T)


def _orthogonal_to(x, key):
    """A random vector with no component along x."""
    w = jax.random.normal(key, x.shape)
    return w - jnp.vdot(x, w) / jnp.vdot(x, x) * x


def test_one_sweep_vjp_matches_fd():
    """Catches a frozen projector; the cotangent has no radial part (ruling 5)."""
    A, C, T, layout = _cold("d2_off")
    f = jax.jit(_one_sweep(C, T, layout))
    C1, T1 = f(A)
    w = (
        with_data(C1, _orthogonal_to(C1._data, jax.random.PRNGKey(1))),
        with_data(T1, _orthogonal_to(T1._data, jax.random.PRNGKey(2))),
    )
    v = jax.random.normal(jax.random.PRNGKey(3), A._data.shape)
    v = v / jnp.linalg.norm(v)

    def phi(eps):
        Ce, Te = f(with_data(A, A._data + eps * v))
        return float(jnp.vdot(w[0]._data, Ce._data) + jnp.vdot(w[1]._data, Te._data))

    (dA,) = jax.vjp(f, A)[1](w)
    slope = float(jnp.vdot(dA._data, v))
    fd, fd_unc = fd_plateau(phi)
    with frozen_projectors():
        (dA_frozen,) = jax.vjp(f, A)[1](w)
    frozen = float(jnp.vdot(dA_frozen._data, v))
    # grad_err first: a mutant must fail here, not on a regime assert.
    assert abs(slope - fd) <= 3 * fd_unc + 1e-12, (slope, fd, fd_unc)
    assert abs(slope) > 100 * fd_unc, (
        f"vacuous: |g.v|={abs(slope):.1e}, fd_unc={fd_unc:.1e}"
    )
    assert abs(slope - frozen) > 10 * fd_unc, "vacuous: dP/dA is in the FD noise"


def test_one_sweep_adjoint_matches_fd():
    """Catches a frozen projector in J^T, the operator of the adjoint solve:
    FD of <w, F~(e0 + h v)> from a fixed start e0 (spec 2, section 6), against
    the solve's own ``_adjoint_matvec``; w has no radial part (ruling 5)."""
    A, C, T, layout = _cold("d2_off")
    f = jax.jit(lambda e: ad._align(*ad._sweep_at(A, *e, layout), T))
    C1, T1 = f((C, T))
    w = (
        with_data(C1, _orthogonal_to(C1._data, jax.random.PRNGKey(1))),
        with_data(T1, _orthogonal_to(T1._data, jax.random.PRNGKey(2))),
    )
    vC = jax.random.normal(jax.random.PRNGKey(4), C._data.shape)
    vT = jax.random.normal(jax.random.PRNGKey(5), T._data.shape)
    norm = jnp.sqrt(jnp.vdot(vC, vC) + jnp.vdot(vT, vT))
    vC, vT = vC / norm, vT / norm

    def phi(eps):
        e = (with_data(C, C._data + eps * vC), with_data(T, T._data + eps * vT))
        Ce, Te = f(e)
        return float(jnp.vdot(w[0]._data, Ce._data) + jnp.vdot(w[1]._data, Te._data))

    # The flat vectors in the (C, T) order of the engine's ravel_pytree.
    w_flat = ravel_pytree(w)[0]
    v_flat = ravel_pytree((with_data(C, vC), with_data(T, vT)))[0]

    def jt_slope():
        """<J^T w, v>, with J^T w read off the matvec of the solve at tau = 0."""
        jt_w = w_flat - ad._adjoint_matvec(w_flat, A, C, T, layout, 0.0)
        return float(jnp.vdot(jt_w, v_flat))

    slope = jt_slope()
    fd, fd_unc = fd_plateau(phi)
    with frozen_projectors():
        frozen = jt_slope()
    print(
        f"one-sweep e-direction: |g.v|={abs(slope):.3e} fd_unc={fd_unc:.1e} "
        f"grad_err={abs(slope - fd):.1e} flow-frozen={abs(slope - frozen):.1e}"
    )
    # grad_err first: a mutant must fail here, not on a regime assert.
    assert abs(slope - fd) <= 3 * fd_unc + 1e-12, (slope, fd, fd_unc)
    assert abs(slope) > 100 * fd_unc, "vacuous: |g.v| is at the FD floor"
    assert abs(slope - frozen) > 10 * fd_unc, "vacuous: dP/de is in the FD noise"


@pytest.mark.parametrize("which", ["C", "T"])
def test_a_radial_cotangent_has_zero_vjp(which):
    """Catches the plain max_abs() in F_L (ruling 4): only the smooth scale
    gives a cotangent along the output a zero VJP."""
    A, C, T, layout = _cold("d2_off")
    f = _one_sweep(C, T, layout)
    C1, T1 = f(A)
    X = C1 if which == "C" else T1
    r = jax.random.normal(jax.random.PRNGKey(4), X._data.shape)
    r = r * (jnp.linalg.norm(X._data) / jnp.linalg.norm(r))
    zero_C = jax.tree_util.tree_map(jnp.zeros_like, C1)
    zero_T = jax.tree_util.tree_map(jnp.zeros_like, T1)
    radial = (C1, zero_T) if which == "C" else (zero_C, T1)
    rnd = (with_data(C1, r), zero_T) if which == "C" else (zero_C, with_data(T1, r))
    vjp = jax.vjp(f, A)[1]
    (d_radial,) = vjp(radial)
    (d_random,) = vjp(rnd)
    n_random = float(jnp.linalg.norm(d_random._data))
    assert n_random > 1e-3, "vacuous: the random cotangent has no VJP"
    assert float(jnp.linalg.norm(d_radial._data)) <= 1e-10 * n_random


# ------------------------------------------------------------------ #
# The optimiser loop (spec 2, section 5)                              #
# ------------------------------------------------------------------ #

GATE = sublattice_rotate_gate(heisenberg_gate())
_MESH = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("x",))


def _config(ctm=None, **kw):
    """A charged C4v config at chi=6: three Armijo steps, then ``kw``."""
    ctm_fields = {"chi": 6, "max_iter": 200, "conv_tol": 1e-10, **(ctm or {})}
    fields = {
        "max_bond_dim": 2,
        "gs_c4v": True,
        "su_init": False,
        "gs_num_steps": 3,
        "gs_line_search_method": "armijo",
        "ctm": CTMConfig(**ctm_fields),
    }
    return iPEPSConfig(**{**fields, **kw})


def _c4v_error(A):
    """max |symmetrize_c4v(todense(A)) - todense(A)|."""
    dense = np.asarray(A.todense())
    return float(np.max(np.abs(np.asarray(symmetrize_c4v(jnp.asarray(dense))) - dense)))


_REFUSED = [
    ({"gs_implicit_ad": False}, None, "gs_implicit_ad"),
    ({}, {"projector_method": "qr"}, "projector_method"),
    ({}, {"forward_gauge": "phase"}, "forward_gauge"),
    ({}, {"ctm_conv_method": "sv"}, "ctm_conv_method"),
    ({}, {"projector_backward": "standard"}, "projector_backward"),
    ({}, {"projector_backward": "lorentzian"}, "projector_backward"),
    ({}, {"on_unconverged": "warn"}, "on_unconverged"),
    ({}, {"chi_ramp": [(4, 2)]}, "chi_ramp"),
    ({}, {"chi_auto_bump": True}, "chi_auto_bump"),
    (
        {},
        {"ctmrg_heuristic_increase_chi": True, "chi_max": 8},
        "ctmrg_heuristic_increase_chi",
    ),
    ({}, {"fuse_virtual_legs": False}, "fuse_virtual_legs"),
    ({}, {"ctm_mixing": 0.5}, "ctm_mixing"),
    ({}, {"device_mesh": _MESH}, "device_mesh"),
    ({"gs_stall_recovery": "noise"}, None, "gs_stall_recovery"),
    ({"cg_gates": object()}, None, "cg_gates"),
    ({"gs_optimizer": "adam"}, None, "gs_optimizer"),
    ({"gs_line_search": False}, None, "gs_line_search"),
    ({"gs_chi_schedule_steps": [(4, 2)]}, None, "gs_chi_schedule_steps"),
    ({"gs_ctm_conv_tol_schedule": [(0.5, 1e-8)]}, None, "gs_ctm_conv_tol_schedule"),
    ({"gs_ctm_max_iter_schedule": [(0.5, 50)]}, None, "gs_ctm_max_iter_schedule"),
    (
        {"gs_plateau_patience_schedule": [(0.5, 10)]},
        None,
        "gs_plateau_patience_schedule",
    ),
]


@pytest.mark.parametrize(
    ("kw", "ctm", "match"),
    [
        pytest.param(kw, ctm, match, id=f"{match}-{i}")
        for i, (kw, ctm, match) in enumerate(_REFUSED)
    ],
)
def test_refusals(kw, ctm, match):
    """Each refused setting of spec 2, section 5 and ruling 8 raises first."""
    with pytest.raises(ValueError, match=f"charged C4v AD: {match}"):
        opt.optimize_gs_ad_c4v_charged(GATE, _site("d2_off"), _config(ctm, **kw))


def test_complex_data_is_refused():
    A = _site("d2_off")
    A_complex = with_data(A, A._data.astype(jnp.complex128))
    with pytest.raises(ValueError, match="charged C4v AD: complex data"):
        opt.optimize_gs_ad_c4v_charged(GATE, A_complex, _config())


_GUIDE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "guide"
    / "algorithms"
    / "ipeps_ad_paths.md"
)


def _guide_section(heading):
    """The guide's lines from ``heading`` up to the next heading.  A line in a
    code fence is not a heading (the Path 4b example has ``# ...`` lines)."""
    lines = _GUIDE.read_text(encoding="utf-8").splitlines()
    assert lines.count(heading) == 1, f"{heading!r} is not one line of {_GUIDE}"
    start, fenced = lines.index(heading), False
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,6} ", lines[i]):
            # An unclosed fence would carry the section into the next one.
            inner = [
                ln for ln in lines[start + 1 : i] if ln.startswith(("## ", "### "))
            ]
            assert not inner, f"the section runs into {inner}"
            return "\n".join(lines[start:i])
    raise AssertionError(f"no heading after {heading!r}")


def test_every_ctm_setting_has_a_disposition_on_the_charged_path():
    """Each CTMConfig field is read, checked, ignored or implied on this path,
    and both docs name each ignored one: the charged paragraph of the
    docstring, and Path 4b of the guide up to the next heading."""
    sets = {
        "read": opt._CTM_READ,
        "checked": opt._CTM_CHECKED,
        "ignored": opt._CTM_IGNORED,
        "implied": opt._CTM_IMPLIED,
    }
    for (a, x), (b, y) in itertools.combinations(sets.items(), 2):
        assert not x & y, f"{a} and {b} share {sorted(x & y)}"
    names = {f.name for f in dataclasses.fields(CTMConfig)}
    union = set().union(*sets.values())
    assert union == names, f"no disposition, or not a field: {sorted(union ^ names)}"
    doc = optimize_gs_ad.__doc__
    start, end = "A charged C4v ``SymmetricTensor``", "See Path 4b"
    assert doc.count(start) == 1 and doc.count(end) == 1, "the charged paragraph moved"
    doc = doc[doc.index(start) : doc.index(end)]
    guide = _guide_section("### Path 4b: Charged C4v Symmetric Input")
    unnamed = {
        "docstring": sorted(n for n in opt._CTM_IGNORED if f"``{n}``" not in doc),
        "guide": sorted(n for n in opt._CTM_IGNORED if f"`{n}`" not in guide),
    }
    assert unnamed == {"docstring": [], "guide": []}, f"not named: {unnamed}"


def test_zero_steps_returns_the_converged_start():
    """gs_num_steps=0: the projected start, its converged env, its energy."""
    A = _site("d2_off")
    A_out, env, E, hist = opt.optimize_gs_ad_c4v_charged(
        GATE, A, _config(gs_num_steps=0, return_history=True)
    )
    np.testing.assert_allclose(np.asarray(A_out._data), np.asarray(A._data), atol=1e-12)
    assert abs(E - float(compute_energy_ctm_tensor(A_out, env, GATE, d=2))) < 1e-12
    _A, C, T, _layout = _cold("d2_off")
    env_cold = _c4v_staggered_to_full_env(C, T)
    assert abs(E - float(compute_energy_ctm_tensor(A, env_cold, GATE, d=2))) < 1e-9
    assert hist["num_steps"] == 0 and hist["energies"] == []
    assert hist["ctm_converged"] == []
    assert hist["layout"] == BondLayout.from_dims({-1: 2, 0: 2, 1: 2}).dims
    assert set(hist) == {
        "energies",
        "grad_norms",
        "step_times",
        "num_steps",
        "converged",
        "ctm_converged",
        "forward_sweeps",
        "adjoint_matvecs",
        "adjoint_residual",
        "layout",
        "layout_mismatch_steps",
    }


# The four tests below marked slow evaluate a gradient and share a ~22 s compile
# of the gradient programs.  The conftest keeps ``core`` off a test marked
# ``slow`` (#731), so they run in the full matrix, not in the required gate.
@pytest.mark.slow
def test_a_non_c4v_start_is_projected():
    """Review Focus 1: a charged start with raw (non-C4v) blocks is projected,
    and the run stays C4v after an accepted step."""
    raw = SymmetricTensor.random_normal(_site("d2_off").indices, jax.random.PRNGKey(5))
    assert _c4v_error(raw) > 0.1, "vacuous: the raw start is C4v"
    A, _env, E, hist = opt.optimize_gs_ad_c4v_charged(
        GATE, raw, _config(gs_num_steps=2, return_history=True)
    )
    assert hist["energies"][1] < hist["energies"][0] - 1e-12, "vacuous: no step"
    assert _c4v_error(A) < 1e-12
    assert np.isfinite(E)


def test_gs_projector_method_overrides_before_the_check():
    """Review Focus 2: gs_projector_method replaces ctm.projector_method first."""
    A = _site("d2_off")
    run = _config({"projector_method": "qr"}, gs_projector_method="svd", gs_num_steps=0)
    assert np.isfinite(opt.optimize_gs_ad_c4v_charged(GATE, A, run)[2])
    refused = _config(
        {"projector_method": "svd"}, gs_projector_method="qr", gs_num_steps=0
    )
    with pytest.raises(ValueError, match="charged C4v AD: projector_method='qr'"):
        opt.optimize_gs_ad_c4v_charged(GATE, A, refused)


def test_a_missing_block_is_canonicalised():
    """Review Focus 4: an A_init that stores fewer blocks than its charges allow."""
    A = _site("d2_off")
    blocks = dict(A.blocks)
    del blocks[next(iter(blocks))]
    A_missing = SymmetricTensor(blocks, A.indices)
    assert A_missing._data.size < A._data.size, "vacuous: no block is missing"
    A_out, _env, E = opt.optimize_gs_ad_c4v_charged(
        GATE, A_missing, _config(gs_num_steps=0)
    )
    assert A_out._data.size == A._data.size
    assert _c4v_error(A_out) < 1e-12 and np.isfinite(E)


def test_two_chi_in_one_process():
    """Review Focus 5: each chi gets its own programs; no stale one serves a run."""
    A = _site("d2_off")
    for chi in (4, 6, 4):
        _A, _env, E = opt.optimize_gs_ad_c4v_charged(
            GATE, A, _config({"chi": chi}, gs_num_steps=0)
        )
        env = ctm_tensor_c4v(A, chi, max_iter=200, conv_tol=1e-10)
        E_eager = float(compute_energy_ctm_tensor(A, env, GATE, d=2))
        assert abs(E - E_eager) < 1e-9, (chi, E, E_eager)


def test_trial_energy_is_inf_without_a_fixed_point():
    A, C, T, layout = _cold("d2_off")
    gate = GATE.todense()
    no_budget = dataclasses.replace(CTM6, max_iter=1, conv_tol=1e-300)
    assert opt._trial_energy(A, (C, T), layout, no_budget, gate, 2) == float("inf")
    assert np.isfinite(opt._trial_energy(A, (C, T), layout, CTM6, gate, 2))


@pytest.mark.slow
@pytest.mark.parametrize("method", ["armijo", "hager_zhang"])
def test_an_unconverged_trial_is_a_bad_point(monkeypatch, method):
    """Review Focus 3: a trial with no fixed point is +inf; the run spends its
    stall budget instead of raising."""
    monkeypatch.setattr(opt, "_trial_energy", lambda *args: float("inf"))
    config = _config(
        gs_line_search_method=method, gs_stall_recovery_retries=1, return_history=True
    )
    _A, _env, E, hist = opt.optimize_gs_ad_c4v_charged(GATE, _site("d2_off"), config)
    assert hist["num_steps"] == 2  # two failed searches spend one reset
    assert abs(hist["energies"][1] - hist["energies"][0]) < 1e-9
    assert E == min(hist["energies"])


@pytest.mark.slow
@pytest.mark.parametrize(("retries", "raises"), [(1, False), (0, True)])
def test_reset_on_an_unconverged_forward(monkeypatch, retries, raises):
    """A forward with no fixed point at step 2 resets to the best point; with
    no retry left it re-raises (the 1-site rule, #1059).  Armijo trials do not
    call charged_c4v_converge, so call 2 is the second step's evaluation."""
    real, calls = ad.charged_c4v_converge, [0]

    def flaky(*args):
        calls[0] += 1
        if calls[0] == 2:
            info = CTMConvergeInfo(
                converged=False,
                iterations=1,
                sv_diff=1.0,
                best_iteration=1,
                step_multiplier=float("nan"),
            )
            raise CTMNotConvergedError(
                info, site="gradient", conv_tol=1e-10, chi=6, detail="mock"
            )
        return real(*args)

    monkeypatch.setattr(ad, "charged_c4v_converge", flaky)
    config = _config(gs_stall_recovery_retries=retries, return_history=True)
    if raises:
        with pytest.raises(CTMNotConvergedError):
            opt.optimize_gs_ad_c4v_charged(GATE, _site("d2_off"), config)
        return
    _A, _env, _E, hist = opt.optimize_gs_ad_c4v_charged(GATE, _site("d2_off"), config)
    assert calls[0] >= 3
    # Step 3 evaluates the best point again: the step-1 point.
    assert abs(hist["energies"][1] - hist["energies"][0]) < 1e-9
    # The 1x1 shapes: one entry per recorded step; the reset iteration has none.
    assert hist["num_steps"] == len(hist["energies"])
    assert hist["ctm_converged"] == [True] * len(hist["energies"])


@pytest.mark.slow
def test_layout_mismatch_is_recorded(monkeypatch):
    """A free ranking that keeps another layout at step 2 is recorded once;
    the run keeps the frozen layout."""
    real, calls = opt._c4v_staggered_layout, [0]
    other = BondLayout.from_dims({0: 6})

    def ranking(Qf, chi):
        calls[0] += 1
        return other if calls[0] == 2 else real(Qf, chi)

    monkeypatch.setattr(opt, "_c4v_staggered_layout", ranking)
    _A, _env, _E, hist = opt.optimize_gs_ad_c4v_charged(
        GATE, _site("d2_off"), _config(return_history=True)
    )
    assert calls[0] >= 2, "vacuous: the second check did not run"
    assert hist["layout_mismatch_steps"] == [2]
    assert hist["layout"] == BondLayout.from_dims({-1: 2, 0: 2, 1: 2}).dims
    assert set(hist) == {
        "energies",
        "grad_norms",
        "step_times",
        "num_steps",
        "converged",
        "ctm_converged",
        "forward_sweeps",
        "adjoint_matvecs",
        "adjoint_residual",
        "layout",
        "layout_mismatch_steps",
    }


def _pair(s_norm, y_norm, cos):
    """Vectors s, y with these norms and this cosine."""
    s = jnp.array([s_norm, 0.0])
    y = y_norm * jnp.array([cos, np.sqrt(1.0 - cos**2)])
    return s, y


def test_lbfgs_pair_rule_is_relative():
    """A curvature pair enters the history when s.y > 1e-8 |s| |y|.

    The kept pair is the one measured at D=3, chi=36: |s| = 3.6e-5,
    |y| = 2.2e-5, cos(s, y) = 0.082, so s.y = 6.5e-11, below the absolute
    floor s.y > 1e-10 that dropped it.
    """
    s, y = _pair(3.6e-5, 2.2e-5, 0.082)
    assert float(jnp.vdot(s, y)) < 1e-10, "vacuous: the old floor keeps the pair"
    assert opt._keep_lbfgs_pair(s, y)
    assert opt._keep_lbfgs_pair(1e-6 * s, 1e-6 * y)  # the angle decides, not the size
    for cos in (-0.5, 0.0, 1e-9):
        assert not opt._keep_lbfgs_pair(*_pair(3.6e-5, 2.2e-5, cos)), cos
    assert not opt._keep_lbfgs_pair(jnp.zeros(2), jnp.zeros(2))


# The mock energy scale: the first step is ~4e-5 with s.y ~ 1e-12 < 1e-10.
_PAIR_EPS = 3e-4


@pytest.mark.parametrize("rule", ["relative", "drop_all"])
def test_the_loop_keeps_a_small_positive_pair(monkeypatch, rule):
    """The loop puts a pair in the L-BFGS history by ``_keep_lbfgs_pair``.

    The engine and the energy are mocks: E = _PAIR_EPS |a - a*|^2 on the
    site buffer a, with a* the D=2 exact state.  The first pair has
    positive curvature and s.y < 1e-10.  The relative rule keeps it; the
    absolute floor s.y > 1e-10 drops it.  With the rule stubbed to False,
    no pair enters.
    """
    target = charged_d2(T_EXACT)._data

    def energy(A, *args, **kw):
        return _PAIR_EPS * jnp.sum((A._data - target) ** 2)

    monkeypatch.setattr(ad, "charged_c4v_converge", lambda A, warm, *args: warm)
    monkeypatch.setattr(opt, "compute_energy_ctm_tensor", energy)
    monkeypatch.setattr(opt, "_trial_energy", lambda A, *args: float(energy(A)))
    if rule == "drop_all":
        monkeypatch.setattr(opt, "_keep_lbfgs_pair", lambda s, y: False)
    real_two_loop, seen = opt.lbfgs_two_loop, []

    def two_loop(g, history, h0):
        seen.append([float(1.0 / rho) for _s, _y, rho in history])  # the s.y kept
        return real_two_loop(g, history, h0)

    monkeypatch.setattr(opt, "lbfgs_two_loop", two_loop)
    _A, _env, _E, hist = opt.optimize_gs_ad_c4v_charged(
        GATE, _site("d2_off"), _config(gs_grad_norm_tol=1e-30, return_history=True)
    )
    assert len(seen) >= 2, f"vacuous: no second direction ({hist['energies']})"
    if rule == "drop_all":
        assert seen == [[]] * len(seen), seen
        return
    small = [sy for kept in seen for sy in kept if 0.0 < sy < 1e-10]
    assert small, f"no pair with 0 < s.y < 1e-10 was kept: {seen}"


# The three tests below pin, in the required gate, what the slow tests
# test_a_non_c4v_start_is_projected, test_reset_on_an_unconverged_forward and
# test_layout_mismatch_is_recorded pin with the real engine.  The engine and
# the energy are mocks, so no gradient program compiles.


def _mock_engine(monkeypatch, target, converge=None):
    """E = _PAIR_EPS |a - target|^2 on the site buffer a, and no CTM.

    ``charged_c4v_converge`` returns its warm env, or calls ``converge``.
    Returns the list of the line-search trial energies, in call order.
    """
    trials = []

    def energy(A, *args, **kw):
        return _PAIR_EPS * jnp.sum((A._data - target) ** 2)

    def trial_energy(A, *args):
        trials.append(float(energy(A)))
        return trials[-1]

    def warm_env(A, warm, *args):
        return warm

    monkeypatch.setattr(ad, "charged_c4v_converge", converge or warm_env)
    monkeypatch.setattr(opt, "compute_energy_ctm_tensor", energy)
    monkeypatch.setattr(opt, "_trial_energy", trial_energy)
    return trials


def test_site_projects_in_the_required_gate(monkeypatch):
    """Review Focus 1: ``site`` projects every iterate to C4v.

    The target is not C4v, so without P in ``site`` the iterates leave the
    C4v subspace at step 1.  With a C4v target the gradient at a C4v point
    is C4v, the iterates stay in the subspace, and the test cannot fail.
    """
    indices = _site("d2_off").indices
    raw = SymmetricTensor.random_normal(indices, jax.random.PRNGKey(5))
    target = SymmetricTensor.random_normal(indices, jax.random.PRNGKey(6))._data
    assert _c4v_error(raw) > 0.1, "vacuous: the raw start is C4v"
    assert _c4v_error(with_data(raw, target)) > 0.1, "vacuous: the target is C4v"
    _mock_engine(monkeypatch, target)
    A, _env, _E, hist = opt.optimize_gs_ad_c4v_charged(
        GATE, raw, _config(gs_grad_norm_tol=1e-30, return_history=True)
    )
    assert hist["energies"][-1] < hist["energies"][0] - 1e-12, "vacuous: no step"
    assert _c4v_error(A) < 1e-12


@pytest.mark.parametrize(("retries", "raises"), [(1, False), (0, True)])
def test_reset_in_the_required_gate(monkeypatch, retries, raises):
    """The 1-site rule (#1059): call 2 of ``charged_c4v_converge`` raises.
    Armijo trials do not call it, so call 2 is the evaluation of step 2.
    With a retry left, the loop resets to the best point, the step-1 point,
    and step 3 evaluates it again.  With no retry left, it re-raises."""
    calls = [0]

    def flaky(A, warm, *args):
        calls[0] += 1
        if calls[0] == 2:
            info = CTMConvergeInfo(
                converged=False,
                iterations=1,
                sv_diff=1.0,
                best_iteration=1,
                step_multiplier=float("nan"),
            )
            raise CTMNotConvergedError(
                info, site="gradient", conv_tol=1e-10, chi=6, detail="mock"
            )
        return warm

    trials = _mock_engine(monkeypatch, charged_d2(T_EXACT)._data, converge=flaky)
    config = _config(
        gs_stall_recovery_retries=retries, gs_grad_norm_tol=1e-30, return_history=True
    )
    if raises:
        with pytest.raises(CTMNotConvergedError, match="mock"):
            opt.optimize_gs_ad_c4v_charged(GATE, _site("d2_off"), config)
        assert calls[0] == 2
        return
    _A, _env, _E, hist = opt.optimize_gs_ad_c4v_charged(GATE, _site("d2_off"), config)
    assert calls[0] == 3, f"vacuous: {calls[0]} evaluations, not 3"
    assert min(trials) < hist["energies"][0] - 1e-12, "vacuous: step 1 did not move"
    # Step 3 evaluates the step-1 point again, not the point that step 1 found.
    assert hist["energies"][1] == hist["energies"][0]
    # The 1x1 shapes: one entry per recorded step; the reset step has none.
    assert hist["num_steps"] == len(hist["energies"]) == 2


def test_layout_record_in_the_required_gate(monkeypatch):
    """A free ranking that keeps another layout at step 2 is recorded once.
    The check calls ``_free_layout``, which calls the loop module's
    ``_c4v_staggered_layout``."""
    _mock_engine(monkeypatch, charged_d2(T_EXACT)._data)
    real, calls = opt._c4v_staggered_layout, [0]
    other = BondLayout.from_dims({0: 6})

    def ranking(Qf, chi):
        calls[0] += 1
        return other if calls[0] == 2 else real(Qf, chi)

    monkeypatch.setattr(opt, "_c4v_staggered_layout", ranking)
    _A, _env, _E, hist = opt.optimize_gs_ad_c4v_charged(
        GATE, _site("d2_off"), _config(gs_grad_norm_tol=1e-30, return_history=True)
    )
    assert calls[0] >= 2, "vacuous: the check of step 2 did not run"
    assert hist["layout_mismatch_steps"] == [2]
    assert hist["layout"] == BondLayout.from_dims({-1: 2, 0: 2, 1: 2}).dims


# ------------------------------------------------------------------ #
# Dispatch (spec 2, section 5)                                        #
# ------------------------------------------------------------------ #


def test_dispatch_reaches_the_charged_loop(monkeypatch):
    """Charged input reaches the new branch and is not densified."""
    seen = {}

    def fake(gate, A_init, config):
        seen["A"] = A_init
        return "charged"

    monkeypatch.setattr(opt, "optimize_gs_ad_c4v_charged", fake)
    A = _site("d2_off")
    assert optimize_gs_ad(GATE, A, _config()) == "charged"
    assert seen["A"] is A


@pytest.mark.parametrize(
    ("kw", "ctm", "expected"),
    [
        ({}, None, True),
        ({"unit_cell": "2site"}, None, False),
        ({"gs_c4v": False}, None, False),
        ({}, {"ctm_ad_mode": "c4v_reference"}, False),
        ({}, {"ctm_ad_mode": "root_implicit_symmetric"}, False),
    ],
)
def test_charged_gate_reads_the_config(kw, ctm, expected):
    assert use_charged_c4v_path(_config(ctm, **kw), _site("d2_off")) is expected


def test_charged_gate_reads_the_input():
    """Only a charged C4v SymmetricTensor takes the branch."""
    config, A = _config(), _site("d2_off")
    assert not use_charged_c4v_path(config, DenseTensor(A.todense(), A.indices))
    assert not use_charged_c4v_path(config, None)
    self_dual = c4v_fixture(ZnSymmetry(2), [0, 1], [0, 1])
    assert not use_charged_c4v_path(config, self_dual)


@pytest.mark.parametrize("pair", [False, True])
def test_two_site_charged_c4v_is_refused(pair):
    A = _site("d2_off")
    with pytest.raises(ValueError, match="charged C4v input with unit_cell='2site'"):
        optimize_gs_ad(GATE, (A, A) if pair else A, _config(unit_cell="2site"))


def test_checkpoint_is_refused_on_the_charged_path(tmp_path):
    config = _config(gs_checkpoint_path=str(tmp_path / "run.npz"))
    with pytest.raises(NotImplementedError, match="charged C4v path"):
        optimize_gs_ad(GATE, _site("d2_off"), config)


def test_recipe_1x1_is_refused_on_the_charged_path():
    """The #938 check runs ahead of the charged branch, whose loop reads no
    gs_recipe: past it, a '1x1' run would report a 2x2 energy."""
    with pytest.warns(DeprecationWarning, match="gs_recipe='1x1'"):  # #911
        config = _config(gs_recipe="1x1", gs_num_steps=0)
    with pytest.raises(ValueError, match="gs_recipe='1x1' is only supported"):
        optimize_gs_ad(GATE, _site("d2_off"), config)


def test_dispatch_skips_the_fused_projector_check():
    """Ruling 16: the fused-path check refuses "eigh"; the charged loop takes it."""
    config = _config({"projector_method": "eigh"}, gs_num_steps=0)
    assert np.isfinite(optimize_gs_ad(GATE, _site("d2_off"), config)[2])
