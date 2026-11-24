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
import numpy as np
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
from _ipeps_spectrum_fixtures import E_D2

from tenax import (
    CTMConfig,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
    sublattice_rotate_gate,
)
from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad
from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v
from tenax.algorithms._ctm_tensor_c4v_staggered import _c4v_staggered_to_full_env
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import symmetrize_c4v
from tenax.core.symmetry import U1Symmetry, ZnSymmetry
from tenax.core.tensor import DenseTensor

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


# ------------------------------------------------------------------ #
# The D=2 optimum and the dense parity (spec 2, section 6)            #
# ------------------------------------------------------------------ #

CHI = 16


def _eager_energy(A, chi=CHI):
    """The #1211 eager energy of A: ctm_tensor_c4v to sv_diff < 1e-12."""
    env = ctm_tensor_c4v(A, chi, max_iter=500, conv_tol=1e-12)
    return float(compute_energy_ctm_tensor(A, env, GATE, d=2))


def _run_config(conv_tol=1e-10, **kw):
    fields = {
        "max_bond_dim": 2,
        "gs_c4v": True,
        "su_init": False,
        "gs_num_steps": 40,
        "return_history": True,
        "ctm": CTMConfig(chi=CHI, max_iter=500, conv_tol=conv_tol),
    }
    return iPEPSConfig(**{**fields, **kw})


_STARTS = {
    "d2_off": lambda: charged_d2(1.15 * T_EXACT),
    # In the same one-parameter family (t = 1.5491), but its cold start at
    # chi=16 lands on another fixed point than charged_d2(1.5491) does.
    "random": lambda: c4v_fixture(U1Symmetry(), [0, 1], [0, 1]),
}


# Plan ruling 21 (Task 8): the "random" start does not reach the optimum, and
# the energy assert is not loosened.  From t = 1.5491 the energy falls to a
# second local minimum of the family at t = 1.13071 (eager E(t), Brent, chi=16:
# E = -0.582762987484, E'' = 0.39); a maximum at t = 0.813 (E = -0.56887)
# separates it from the optimum at t = 0.42102.  The run ends there, on a C4v
# tensor with |g| < 1e-5: the first three asserts hold, the fourth fails.
_RANDOM_ENDS_ELSEWHERE = pytest.mark.xfail(
    strict=True,
    reason=(
        "Task 8: Ruling: the random start converges to E=-0.582762987478 "
        "— another fixed point — reported to the user"
    ),
)


@pytest.mark.parametrize(
    "start", ["d2_off", pytest.param("random", marks=_RANDOM_ENDS_ELSEWHERE)]
)
def test_d2_optimum(start):
    """Catches a loss that skips the C4v projection (ruling 3).  The C4v
    assert comes first, so that mutant fails there."""
    A, _env, E, hist = optimize_gs_ad(GATE, _STARTS[start](), _run_config())
    dense = np.asarray(A.todense())
    sym = np.asarray(symmetrize_c4v(jnp.asarray(dense)))
    c4v_error = float(np.max(np.abs(sym - dense)))
    print(
        f"{start}: E={E:.12f} E-E_D2={E - E_D2:+.2e} "
        f"|g|={hist['grad_norms'][-1]:.2e} steps={hist['num_steps']} "
        f"c4v_error={c4v_error:.1e} mismatch={hist['layout_mismatch_steps']}"
    )
    assert c4v_error < 1e-12
    assert hist["converged"], hist["grad_norms"][-3:]
    assert hist["grad_norms"][-1] < 1e-5
    assert E <= _eager_energy(charged_d2(T_EXACT)) + 1e-9
    assert abs(E - E_D2) <= 1e-8


def test_dense_parity():
    """The dense CTM energy of todense(A*), and the dense C4v optimiser, agree
    with the charged path at chi=16 (spec 2, section 6)."""
    A0 = charged_d2(1.15 * T_EXACT)
    A, _env, E, _hist = optimize_gs_ad(GATE, A0, _run_config(conv_tol=1e-12))
    A_dense = DenseTensor(A.todense(), A.indices)
    env_dense, _info = ctm_tensor(A_dense, chi=CHI, max_iter=500, conv_tol=1e-12)
    E_dense = float(compute_energy_ctm_tensor(A_dense, env_dense, GATE, d=2))
    print(f"dense parity: |E - E_dense(ctm_tensor)| = {abs(E - E_dense):.2e}")
    assert abs(E - E_dense) <= 1e-10
    dense_run = _run_config(gs_num_steps=60, return_history=False)
    # Spec 2, section 6: the dense 1x1 C4v optimiser on DenseTensor(todense(A_init)).
    A0_dense = DenseTensor(A0.todense(), A0.indices)
    _A, _env_d, E_opt = optimize_gs_ad(GATE, A0_dense, dense_run)
    print(
        f"dense parity: E_dense_opt={E_opt:.12f} |E - E_dense_opt| = {abs(E - E_opt):.2e}"
    )
    assert abs(E - E_opt) <= 1e-8
