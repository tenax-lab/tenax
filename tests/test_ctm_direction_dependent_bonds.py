"""2-site CTM must handle direction-dependent virtual bonds (A.l != A.r).

Regression for the env-init single-tensor-uniformity assumption (issue #667
follow-up): a valid *unit-cell*-translation-invariant checkerboard iPEPS can have
A.l != A.r as long as bonds match AROUND the cell (A.r==B.l, A.l==B.r).  Dropping
the SU `base_charges` produces exactly such a state; `ctm_tensor_2site` used to
die with a `7 vs 4` contraction mismatch because the symmetric env init derived
both corner chi-legs from a single reference axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import tenax.algorithms.ipeps_simple_update as SU
from tenax import compute_energy_ctm_tensor_2site, ctm_tensor_2site
from tenax.algorithms.ipeps import (
    heisenberg_gate,
    heisenberg_gate_u1sz,
    heisenberg_u1sz_init_pair,
)
from tenax.core.tensor import DenseTensor


def _su_direction_dependent_pair(D=3, steps=40, dt=0.1):
    """Run base_charges-free U(1)-Sz SU -> a pair with A.l != A.r (unit-cell OK)."""
    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    H = heisenberg_gate_u1sz()
    gate = SU._make_trotter_gate_tensor(H, dt, site_tensor=A)
    lam_h = jnp.ones(D)
    lam_v = jnp.ones(D)
    orig = SU.truncated_svd
    SU.truncated_svd = lambda *a, **k: orig(*a, **{**k, "base_charges": None})
    try:
        for s in range(steps):
            if s % 2 == 0:
                A, B, lam_h = SU._simple_update_2site_horizontal_tensor(
                    A, B, gate, lam_h, lam_v, D
                )
            else:
                A, B, lam_v = SU._simple_update_2site_vertical_tensor(
                    A, B, gate, lam_h, lam_v, D
                )
    finally:
        SU.truncated_svd = orig
    return A, B


def _charges(T, leg):
    lab = T.labels()
    return list(np.asarray(T.indices[lab.index(leg)].charges))


def test_su_output_is_direction_dependent_but_cell_consistent():
    """Sanity: the fixture really has A.l != A.r yet A.r==B.l and A.l==B.r."""
    A, B = _su_direction_dependent_pair()
    assert _charges(A, "l") != _charges(A, "r"), "fixture must be direction-dependent"
    assert _charges(A, "r") == _charges(B, "l"), "A.r must match B.l (cell-consistent)"
    assert _charges(A, "l") == _charges(B, "r"), "A.l must match B.r (cell-consistent)"


def test_symmetric_2site_ctm_matches_dense_on_direction_dependent_bonds():
    """ctm_tensor_2site on A.l!=A.r must not error and must match the dense result.

    Fixed by #670 (carried-bond leg-pairing in the 2x2 enlarged-corner projector).
    Uses recipe="2x2"; the 1x1 recipe has a separate fundamental edge-orientation
    wall and is not targeted here.  Both paths should converge to E ≈ -0.5421.
    """
    A, B = _su_direction_dependent_pair()
    Hd = heisenberg_gate()

    # Dense reference (densify then run the same CTM path).
    Ad = DenseTensor(np.array(A.todense()), A.indices)
    Bd = DenseTensor(np.array(B.todense()), B.indices)
    eAd, eBd = ctm_tensor_2site(
        Ad, Bd, chi=12, max_iter=60, conv_tol=1e-9, recipe="2x2"
    )
    E_dense = float(compute_energy_ctm_tensor_2site(Ad, Bd, eAd, eBd, Hd))

    # Symmetric block-sparse path (the one that used to crash with 7 vs 4).
    eA, eB = ctm_tensor_2site(A, B, chi=12, max_iter=60, conv_tol=1e-9, recipe="2x2")
    c1 = float(jnp.linalg.norm(eA.C1.todense()))
    E_sym = float(compute_energy_ctm_tensor_2site(A, B, eA, eB, Hd))

    assert c1 > 1e-8, f"symmetric CTM corner collapsed (C1 norm={c1})"
    assert abs(E_sym - E_dense) < 1e-6, f"sym {E_sym} != dense {E_dense}"
    assert E_sym < -0.3, f"energy {E_sym} unphysical (expected AFM, ~ -0.5 to -0.7)"


def test_2x2_enlarged_corners_build_on_direction_dependent_init():
    """First-sweep 2x2 enlarged corners must build (charge-consistent) on the
    direction-dependent init env.

    Phase 1 of docs/superpowers/plans/2026-07-01-direction-dependent-symmetric-ctm.md:
    for the 2x2 recipe all enlarged-corner corner<->edge contractions are
    SAME-site, so the original same-axis env-init seeding + Phase 0 canonical
    tiling should already be charge-consistent (no 1x1-oriented axis reseeding).
    """
    import numpy as np

    from tenax.algorithms._ctm_tensor_init import (
        _build_double_layer_tensor,
        initialize_ctm_tensor_env,
    )
    from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner

    A, B = _su_direction_dependent_pair()
    chi = 12
    envA = initialize_ctm_tensor_env(A, chi)
    envB = initialize_ctm_tensor_env(B, chi)
    aA = _build_double_layer_tensor(A)
    aB = _build_double_layer_tensor(B)
    # Same-site enlarged corners (each uses one site's env + that site's DL).
    for env, a in ((envA, aA), (envB, aB)):
        for pos, (C, Th, Tv) in {
            "top_left": (env.C1, env.T1, env.T4),
            "top_right": (env.C2, env.T1, env.T2),
            "bottom_left": (env.C4, env.T3, env.T4),
            "bottom_right": (env.C3, env.T3, env.T2),
        }.items():
            Q = _build_enlarged_corner(C, Th, Tv, a, position=pos)
            assert Q is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
