"""Symmetric 2x2 must stay bond-consistent after absorption (#670)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import tenax.algorithms.ipeps_simple_update as SU
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS as NB,
)
from tenax.algorithms._ctm_tensor_convergence import (
    _sort_coords_for_direction,
)
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_moves import (
    _compute_plaquette_projector_pair,
    _ctm_tensor_absorb_left_2plaq,
)
from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair

CHI = 12


def _uniform_multicharge_pair(D=3, steps=40, dt=0.1):
    """Normal U(1)-Sz SU (base_charges kept) -> direction-uniform multi-charge."""
    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    H = heisenberg_gate_u1sz()
    gate = SU._make_trotter_gate_tensor(H, dt, site_tensor=A)
    lh, lv = jnp.ones(D), jnp.ones(D)
    for s in range(steps):
        if s % 2 == 0:
            A, B, lh = SU._simple_update_2site_horizontal_tensor(A, B, gate, lh, lv, D)
        else:
            A, B, lv = SU._simple_update_2site_vertical_tensor(A, B, gate, lh, lv, D)
    return A, B


def _direction_dependent_pair(D=3, steps=40, dt=0.1):
    """base_charges-free U(1)-Sz SU -> A.l != A.r (direction-dependent bonds)."""
    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    H = heisenberg_gate_u1sz()
    gate = SU._make_trotter_gate_tensor(H, dt, site_tensor=A)
    lh, lv = jnp.ones(D), jnp.ones(D)
    orig = SU.truncated_svd
    SU.truncated_svd = lambda *a, **k: orig(*a, **{**k, "base_charges": None})
    try:
        for s in range(steps):
            if s % 2 == 0:
                A, B, lh = SU._simple_update_2site_horizontal_tensor(
                    A, B, gate, lh, lv, D
                )
            else:
                A, B, lv = SU._simple_update_2site_vertical_tensor(
                    A, B, gate, lh, lv, D
                )
    finally:
        SU.truncated_svd = orig
    return A, B


def _one_left(A, B):
    site = {(0, 0): A, (1, 0): B}
    dl = {c: _build_double_layer_tensor(t) for c, t in site.items()}
    envs = {c: initialize_ctm_tensor_env(t, CHI) for c, t in site.items()}
    projectors = {}
    for s_anchor in envs:
        s_TR = NB[s_anchor]["right"]
        s_BL = NB[s_anchor]["bottom"]
        s_BR = NB[s_TR]["bottom"]
        Pt, Pb, _, _ = _compute_plaquette_projector_pair(
            envs[s_anchor],
            envs[s_TR],
            envs[s_BL],
            envs[s_BR],
            dl[s_anchor],
            dl[s_TR],
            dl[s_BL],
            dl[s_BR],
            CHI,
            "left",
        )
        projectors[s_anchor] = (Pt, Pb)
    new = {}
    for s_dst in _sort_coords_for_direction(list(envs), "left"):
        s_src = NB[s_dst]["left"]
        sa = NB[s_src]["top"]
        Pta, Pba = projectors[sa]
        Ptc, Pbc = projectors[s_src]
        C1, T4, C4 = _ctm_tensor_absorb_left_2plaq(
            envs[s_src], dl[s_src], Pta, Pba, Ptc, Pbc
        )
        new[s_dst] = envs[s_dst]._replace(C1=C1, T4=T4, C4=C4)
    return new, dl


def test_enlarged_corners_build_after_left_absorption_multicharge():
    A, B = _direction_dependent_pair()
    new, dl = _one_left(A, B)
    for s_dst, env in new.items():
        for pos, (C, Th, Tv) in {
            "top_left": (env.C1, env.T1, env.T4),
            "top_right": (env.C2, env.T1, env.T2),
            "bottom_left": (env.C4, env.T3, env.T4),
            "bottom_right": (env.C3, env.T3, env.T2),
        }.items():
            Q = _build_enlarged_corner(C, Th, Tv, dl[s_dst], position=pos)
            assert Q is not None, f"{pos} failed to build at {s_dst}"


#: The dense 2x2 energy of ``_su_direction_dependent_pair()``.
#:
#: **Regenerate with**::
#:
#:     A, B = _su_direction_dependent_pair()
#:     Ad, Bd = (DenseTensor(np.array(t.todense()), t.indices) for t in (A, B))
#:     eA, eB = ctm_tensor_2site(Ad, Bd, chi=12, max_iter=60, conv_tol=1e-9,
#:                               recipe="2x2")
#:     print(compute_energy_ctm_tensor_2site(Ad, Bd, eA, eB, heisenberg_gate()))
#:
#: Measured bit-identical at ``max_iter`` 60 / 120 / 240 and ``chi`` 8 / 12 /
#: 16 / 24.  (``max_iter=30`` gives -0.4332902670293, 1.0e-09 away, so the
#: 1e-6 tolerance below is ~3 orders above the only budget sensitivity there
#: is.)
#:
#: The previous value here was ``-0.5421160718``, frozen on 2026-07-02 by
#: ``7b8e5ad`` and never revisited.  It was **never recoverable**: it was
#: measured through the pre-#898 convergence criterion, which certified this
#: environment on sweep ~3, so it recorded a sweep index rather than an
#: energy.  Post-#898 the same call reaches the actual fixed point of the
#: loop.  This is what a literal frozen without checking the state it came
#: from costs -- see also #836, and the "scan the budget before freezing"
#: rule that came out of it.
_E_DENSE_2X2 = -0.4332902680574


def test_dense_2x2_energy_unchanged_by_fix():
    """#670's leg-pairing correction is a no-op on the single-block dense path.

    This is a **determinism guard, not a physics check**, and the distinction
    is load-bearing.  The fixture's corner is rank 1 -- the environment is
    mean-field and its energy will not respond to ``chi`` -- so ``_E_DENSE_2X2``
    is emphatically *not* an approximation to the Heisenberg ground state.  What
    it is, is exactly reproducible: the same number at every ``chi`` from 8 to
    24 and every ``max_iter`` from 60 to 240.  That is all this test needs,
    because the question it asks is "did the leg pairing change the dense
    result", and a bit-stable number answers that.

    The regime is asserted rather than assumed (the previous version assumed it
    and went stale for 8 weeks without anyone noticing):

    * the loop **says** it cannot certify this environment -- if that warning
      ever stops, the state is no longer the collapsed one this constant was
      measured on and the constant must be re-derived;
    * the value is budget-independent, which is what makes freezing it legal.

    The fixture is deliberately left collapsed.  ``test_ctm_criterion_rank_
    blind_898.py`` imports this same pair as ``_collapsing_pair()`` and needs
    the collapse; "repairing" it here would silently gut those tests.  Seed 2
    was evaluated as a replacement and is worse -- its simple update is still
    moving 1.7e-02 at 320 steps and its CTM never converges at any budget.
    """
    import warnings

    from tenax import compute_energy_ctm_tensor_2site, ctm_tensor_2site
    from tenax.algorithms._ctm_diagnostics import ctm_corner_rank
    from tenax.algorithms.ipeps import heisenberg_gate
    from tenax.core.tensor import DenseTensor
    from tests.test_ctm_direction_dependent_bonds import _su_direction_dependent_pair

    A, B = _su_direction_dependent_pair()
    Ad = DenseTensor(np.array(A.todense()), A.indices)
    Bd = DenseTensor(np.array(B.todense()), B.indices)

    def run(max_iter, chi=12):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            eA, eB = ctm_tensor_2site(
                Ad, Bd, chi=chi, max_iter=max_iter, conv_tol=1e-9, recipe="2x2"
            )
        E = float(compute_energy_ctm_tensor_2site(Ad, Bd, eA, eB, heisenberg_gate()))
        return E, eA, [str(c.message) for c in caught]

    E, eA, msgs = run(60)

    # Regime, part 1: still the collapsed environment the constant came from.
    assert ctm_corner_rank(eA) == 1, (
        f"corner rank is {ctm_corner_rank(eA)}, not 1: the fixture is no longer "
        f"collapsed, so _E_DENSE_2X2 describes a different state and must be "
        f"re-derived (and check test_ctm_criterion_rank_blind_898.py, which "
        f"needs this pair collapsed)."
    )
    assert any("could not be certified" in m for m in msgs), (
        "the loop no longer reports this environment as uncertifiable; the "
        "regime this constant was measured in has changed."
    )

    # Regime, part 2: budget-independent, which is what licenses a literal.
    E_long, _, _ = run(240)
    assert abs(E - E_long) < 1e-12, (
        f"energy moved with max_iter ({E} at 60 vs {E_long} at 240): it is a "
        f"sweep snapshot, not a reproducible number, and must not be frozen."
    )

    assert abs(E - _E_DENSE_2X2) < 1e-6, f"dense 2x2 energy drifted: {E}"
