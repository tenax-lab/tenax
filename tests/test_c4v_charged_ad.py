"""Mechanism tests of the charged C4v AD path (spec 2), at chi <= 6.

Design: ``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``.
The converged FD checks, the D=2 optimum and the dense parity are in
``test_c4v_charged_ad_physics.py`` (algorithm bucket).
"""

from __future__ import annotations

import dataclasses
import functools

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

from tenax import CTMConfig
from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad
from tenax.algorithms._ctm_convergence_policy import CTMNotConvergedError
from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout
from tenax.algorithms._ctm_tensor_c4v_staggered import (
    _c4v_staggered_enlarged_corner,
    _c4v_staggered_layout,
    _c4v_staggered_loop,
    _c4v_staggered_sweep,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.core.symmetry import U1Symmetry, ZnSymmetry

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
    f = _one_sweep(C, T, layout)
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
