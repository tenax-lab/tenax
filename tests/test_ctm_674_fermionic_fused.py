"""Fermionic/fused 2x2 CTM bottom absorption must use the c3_u<->t2_d bond
pairing (#674).

Background
----------
#670 established the correct corner-leg pairing for the BOTTOM absorption via
the production energy/RDM path (``_ctm_tensor_energy.py``): the ``C3.T2``
contraction pairs ``C3.c3_u`` with ``T2.t2_d`` (see e.g. line 67 there,
``T2.relabels({"t2_d": "c3_u"})``).  #670 fixed the NON-fused (bosonic/dense)
``_ctm_tensor_absorb_bottom_2plaq`` accordingly.

The FUSED twin ``_ctm_tensor_absorb_bottom_2plaq_fused`` — the path taken for
FERMIONIC envs (dispatched from the non-fused function when
``_env_is_fermionic`` is true) — carried the OLD swapped pairing
(``c3_l<->t2_d``).  This is #674.

Why a unit-level (3b) test and not end-to-end
---------------------------------------------
The swap is a *latent* per-sector block-bookkeeping bug.  On a freshly
initialised env the two C3 chi legs (``c3_u`` / ``c3_l``) are seeded with
IDENTICAL charges (both derive from the same virtual reference axis; see
``_init_symmetric_standard_corner``), so a single absorption on a fresh env is
bit-identical whether the pairing is swapped or not — verified for both
``FermionParity`` and ``FermionicU1``.  The bug only bites once the C3 corner
has developed distinguishable ``c3_u`` vs ``c3_l`` structure through CTM sweeps.
A full multi-sweep fermionic ``ctm_tensor_2site`` DOES expose it, but the
block-sparse fermionic forward CTM is far too slow for CI (~minutes for a
handful of iterations).

This test therefore reproduces the bug's precondition directly: it builds a
direction-dependent ``FermionicU1`` env, replaces the rank-1 C3 (and its
partner T2) with FULL random ``SymmetricTensor``s so ``c3_u`` and ``c3_l``
carry genuinely different block data, and drives the fused bottom absorption.
It asserts the resulting C3 corner equals an INDEPENDENTLY computed reference
that uses the authority ``c3_u<->t2_d`` contraction.  The buggy (swapped)
pairing produces a materially different corner (rel. Frobenius diff ~1.9), so
this test FAILS before the fix and PASSES after.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# Fermionic prerequisites — skip cleanly if unavailable so CI stays green.
FermionicU1 = pytest.importorskip("tenax.core.symmetry").FermionicU1
_moves = pytest.importorskip("tenax.algorithms._ctm_tensor_moves")

from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS as NB
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import SymmetricTensor

_compute_plaquette_projector_pair = _moves._compute_plaquette_projector_pair
_ctm_tensor_absorb_bottom_2plaq = _moves._ctm_tensor_absorb_bottom_2plaq
_apply_proj_unfused = _moves._apply_proj_unfused
_phase_fix_normalize_tensor = _moves._phase_fix_normalize_tensor
_env_is_fermionic = _moves._env_is_fermionic

CHI = 12


def _fermionic_direction_dependent_pair():
    """A direction-dependent FermionicU1 pair (A.l != A.r, cell-consistent).

    FermionicU1 (non-self-dual U(1) charges) is used deliberately: a plain
    ``FermionParity`` (Z2, self-dual) env makes even the multi-sweep bug a
    no-op, whereas non-self-dual charges expose the c3_u/c3_l distinction.
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
    return A, B


def _bottom_projectors(envs, dl):
    """Compute the 2x2 'bottom'-direction projector pair per anchor cell."""
    projectors = {}
    for s_anchor in envs:
        s_TR = NB[s_anchor]["right"]
        s_BL = NB[s_anchor]["bottom"]
        s_BR = NB[s_TR]["bottom"]
        P_top, P_bot, _, _ = _compute_plaquette_projector_pair(
            envs[s_anchor],
            envs[s_TR],
            envs[s_BL],
            envs[s_BR],
            dl[s_anchor],
            dl[s_TR],
            dl[s_BL],
            dl[s_BR],
            CHI,
            "bottom",
        )
        projectors[s_anchor] = (P_top, P_bot)
    return projectors


def test_fermionic_env_dispatches_to_fused_bottom():
    """Sanity: a FermionicU1 env is recognised as fermionic (fused dispatch)."""
    A, _ = _fermionic_direction_dependent_pair()
    env = initialize_ctm_tensor_env(A, CHI)
    assert _env_is_fermionic(env), "FermionicU1 env must take the fused CTM path"


def test_fused_bottom_c3_uses_authority_c3u_t2d_pairing():
    """The fused bottom absorption C3 must match the ``c3_u<->t2_d`` authority.

    Reproduces #674.  Builds a direction-dependent FermionicU1 env, then makes
    the C3 corner and its T2 partner FULL (non-rank-1) so ``c3_u`` and ``c3_l``
    carry distinguishable block data — the precondition under which the swapped
    (``c3_l<->t2_d``) pairing diverges from the correct one.  Compares the fused
    absorption's C3 against an independent reference built with the authority
    pairing.

    Buggy code (c3_l<->t2_d): C3 differs from the reference (test FAILS).
    Fixed code (c3_u<->t2_d): C3 matches the reference exactly (test PASSES).
    """
    A, B = _fermionic_direction_dependent_pair()
    site = {(0, 0): A, (1, 0): B}
    dl = {c: _build_double_layer_tensor(t) for c, t in site.items()}
    envs = {c: initialize_ctm_tensor_env(t, CHI) for c, t in site.items()}
    assert _env_is_fermionic(envs[(0, 0)])

    # Doctor env (0,0): full random C3 + T2 so c3_u / c3_l are distinguishable.
    e = envs[(0, 0)]
    C3_full = SymmetricTensor.random_normal(e.C3.indices, jax.random.PRNGKey(42))
    T2_full = SymmetricTensor.random_normal(e.T2.indices, jax.random.PRNGKey(43))
    envs[(0, 0)] = e._replace(C3=C3_full, T2=T2_full)

    projectors = _bottom_projectors(envs, dl)

    # Absorb bottom for s_dst=(1,0): s_src=(0,0) is the doctored env, and the
    # 'left' anchor is also (0,0) on the 2-site checkerboard.
    s_dst = (1, 0)
    s_src = NB[s_dst]["bottom"]
    s_left = NB[s_dst]["left"]
    assert s_src == (0, 0)
    P_top_left, P_bot_left = projectors[s_left]
    P_top_curr, P_bot_curr = projectors[s_dst]

    _, _, C3_new = _ctm_tensor_absorb_bottom_2plaq(
        envs[s_src], dl[s_src], P_top_left, P_bot_left, P_top_curr, P_bot_curr
    )

    # Independent authority reference for JUST the C3 branch, using the
    # energy/RDM convention C3.c3_u <-> T2.t2_d (see _ctm_tensor_energy.py:67).
    env_src = envs[s_src]
    C3_u = env_src.C3.relabel("c3_u", "t2_d")
    C3g = contract(C3_u, env_src.T2)  # free legs: (c3_l, t2_u, r2)
    C3_ref = _apply_proj_unfused(P_top_curr, C3g, "c3_l", "r2")
    C3_ref = C3_ref.relabels({"chi_new": "c3_u", "t2_u": "c3_l"})
    C3_ref = _phase_fix_normalize_tensor(C3_ref)

    got = np.asarray(C3_new.todense())
    ref = np.asarray(C3_ref.todense())
    max_abs = float(np.abs(got - ref).max())
    assert max_abs < 1e-10, (
        "fused bottom C3 does not match the c3_u<->t2_d authority pairing "
        f"(max abs diff {max_abs:.3e}); the fused path likely still uses the "
        "swapped c3_l<->t2_d pairing (#674)."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
