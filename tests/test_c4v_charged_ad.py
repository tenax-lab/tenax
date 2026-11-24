"""Mechanism tests of the charged C4v AD path (spec 2), at chi <= 6.

Design: ``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``.
The converged FD checks, the D=2 optimum and the dense parity are in
``test_c4v_charged_ad_physics.py`` (algorithm bucket).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from _c4v_staggered_fixtures import T_EXACT, c4v_fixture, charged_d2, leg

from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad
from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout
from tenax.algorithms._ctm_tensor_c4v_staggered import (
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
