"""Physics tests of the charged C4v AD path (spec 2): converged FD with an h
scan, the D=2 optimum and the dense parity.  ``algorithm`` bucket.

Design: ``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``,
section 6.  Mechanism tests are in ``test_c4v_charged_ad.py`` (core).
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from _c4v_staggered_fixtures import (
    T_EXACT,
    c4v_direction,
    c4v_fixture,
    charged_d2,
    fd_plateau,
    frozen_projectors,
    with_data,
)

from tenax import CTMConfig, heisenberg_gate, sublattice_rotate_gate
from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad
from tenax.algorithms._ctm_tensor_c4v_staggered import _c4v_staggered_to_full_env
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.core.symmetry import U1Symmetry, ZnSymmetry

GATE = sublattice_rotate_gate(heisenberg_gate())

# The FD rule: an exact adjoint (tau = 0) solved to 1e-12, and every forward
# re-converged element-wise to 1e-13 from the base fixed point (the floor is
# about 1e-15: probe 7).
CTM_FD = CTMConfig(
    chi=6,
    max_iter=500,
    conv_tol=1e-13,
    adjoint_tol=1e-12,
    adjoint_maxiter=100,
    adjoint_tikhonov=0.0,
)

_SITES = {
    "d2_off": lambda: charged_d2(1.15 * T_EXACT),
    "762": lambda: c4v_fixture(U1Symmetry(), [-1, 1], [-2, 0]),
    "z4": lambda: c4v_fixture(ZnSymmetry(4), [0, 1], [0, 1]),
}


def _energy_of(A, C, T):
    return compute_energy_ctm_tensor(A, _c4v_staggered_to_full_env(C, T), GATE, d=2)


@functools.cache
def _fixed_point(name):
    """``(A, (C*, T*), layout)`` at chi=6, element-wise to 1e-13."""
    A = _SITES[name]()
    C, T, layout = ad.charged_c4v_cold_start(
        A, dataclasses.replace(CTM_FD, conv_tol=1e-10)
    )
    return A, ad._elementwise_phase(A, (C, T), layout, CTM_FD, {}), layout


def _grad(A, warm, layout, record):
    def energy(A_):
        C, T = ad.charged_c4v_converge(A_, warm, layout, CTM_FD, record)
        return _energy_of(A_, C, T)

    return jax.grad(energy)(A)


@pytest.mark.parametrize(
    ("name", "key"), [("d2_off", 0), ("762", 0), ("762", 1), ("z4", 0)]
)
def test_converged_gradient_matches_fd(name, key):
    """Catches a frozen projector and lambda = g (spec 2, section 6)."""
    A, warm, layout = _fixed_point(name)
    v = c4v_direction(A, jax.random.PRNGKey(key))
    record = {}
    slope = float(jnp.vdot(_grad(A, warm, layout, record)._data, v))

    def phi(eps):
        A_e = with_data(A, A._data + eps * v)
        C, T = ad._elementwise_phase(A_e, warm, layout, CTM_FD, {})
        return float(_energy_of(A_e, C, T))

    fd, fd_unc = fd_plateau(phi)
    with frozen_projectors():
        frozen = float(jnp.vdot(_grad(A, warm, layout, {})._data, v))
    print(
        f"{name} dir {key}: |g.v|={abs(slope):.3e} fd_unc={fd_unc:.1e} "
        f"grad_err={abs(slope - fd):.1e} flow-frozen={abs(slope - frozen):.1e}"
    )
    # grad_err first: a mutant must fail here, not on a regime assert.
    assert abs(slope - fd) <= 3 * fd_unc + 1e-12, (slope, fd, fd_unc)
    assert abs(slope) > 100 * fd_unc, "vacuous: |g.v| is at the FD floor"
    assert abs(slope - frozen) > 10 * fd_unc, "vacuous: dP/dA is in the FD noise"
    assert record["adjoint_residual"] <= 1e-11
