"""The four-phase checkerboard sweep is written once, and behaves identically.

The sweep loop existed in **seven** places, byte-identical in the body and
differing only in what wrapped it::

    src/tenax/algorithms/ipeps.py            (the shipped ``ipeps()`` path)
    examples/su_symmetric_ctm_e2e.py
    tests/_split_ctm_oracle.py
    tests/test_split_ctm_2site.py
    tests/test_split_ctm_2site_ad.py
    tests/test_ipeps_bp_gauge.py
    tests/test_su_865_symmetric_collapse.py

That is the shape that has already bitten this repo three times (#828, #829,
#842: a fix landing on some of N copies), and it bit it again here -- #667 had
to be applied to each copy by hand.  Seven is also two more than the count in
#863's own description, which is the point: nobody can hold the list.

These tests pin the consolidation as a **pure refactor**.  The reference
implementation below is the pre-consolidation loop, transcribed from ``main``
at 55691da; the assertion is that the shared helper reproduces it exactly, on
the same inputs, for step counts stopping at every phase of the cycle.

Deliberately *not* asserted here: anything about whether the two-lambda scheme
is correct.  It is not (#851: two stored spectra for four inequivalent bonds,
so ``steps % 4`` selects which bond's gauge is stamped on the lattice).  This
file only guarantees that consolidating the copies changed nothing, so that the
real fix lands in one place instead of seven.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps import _wrap_as_dense_tensor
from tenax.algorithms.ipeps_simple_update import (
    _make_trotter_gate_tensor,
    _simple_update_2site_horizontal_tensor,
    _simple_update_2site_vertical_tensor,
    _simple_update_checkerboard_sweep,
    _to_physical_pair,
    _to_physical_tensor,
)

jax.config.update("jax_enable_x64", True)


def _heisenberg_gate():
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    H = np.kron(Sz, Sz) + 0.5 * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp))
    return jnp.asarray(H).reshape(2, 2, 2, 2)


def _pair(D, seed=0, d=2):
    kA, kB = jax.random.split(jax.random.PRNGKey(seed))
    A = _wrap_as_dense_tensor(jax.random.normal(kA, (D, D, D, D, d)))
    B = _wrap_as_dense_tensor(jax.random.normal(kB, (D, D, D, D, d)))
    return A * (1.0 / float(A.norm())), B * (1.0 / float(B.norm()))


def _reference_sweep(A, B, gate, max_D, steps):
    """The pre-consolidation loop, transcribed verbatim from ``main``.

    Kept as a literal copy on purpose: if the shared helper is ever changed,
    this is what says whether the change was a behaviour change.
    """
    lam_h = jnp.ones(max_D)
    lam_v = jnp.ones(max_D)
    for step in range(steps):
        phase = step % 4
        if phase == 0:
            A, B, lam_h = _simple_update_2site_horizontal_tensor(
                A, B, gate, lam_h, lam_v, max_D
            )
        elif phase == 1:
            A, B, lam_v = _simple_update_2site_vertical_tensor(
                A, B, gate, lam_h, lam_v, max_D
            )
        elif phase == 2:
            B, A, lam_h = _simple_update_2site_horizontal_tensor(
                B, A, gate, lam_h, lam_v, max_D
            )
        else:
            B, A, lam_v = _simple_update_2site_vertical_tensor(
                B, A, gate, lam_h, lam_v, max_D
            )
    return A, B, lam_h, lam_v


# Every residue mod 4, so a phase-indexing slip cannot hide: the four-phase
# cycle means an off-by-one in the loop only shows for some stopping points.
@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5, 6, 7, 8, 13])
@pytest.mark.parametrize("D", [2, 3])
def test_the_shared_sweep_reproduces_the_open_coded_loop(D, steps):
    A0, B0 = _pair(D)
    gate = _make_trotter_gate_tensor(_heisenberg_gate(), 0.05, site_tensor=A0)

    rA, rB, r_h, r_v = _reference_sweep(A0, B0, gate, D, steps)
    sA, sB, s_h, s_v = _simple_update_checkerboard_sweep(A0, B0, gate, D, steps)

    np.testing.assert_array_equal(np.asarray(s_h), np.asarray(r_h))
    np.testing.assert_array_equal(np.asarray(s_v), np.asarray(r_v))
    np.testing.assert_array_equal(np.asarray(sA.todense()), np.asarray(rA.todense()))
    np.testing.assert_array_equal(np.asarray(sB.todense()), np.asarray(rB.todense()))


def test_resuming_a_sweep_matches_running_it_in_one_go():
    """The sweep takes its lambdas back, so a caller can drive it in chunks.

    Only exact if the resumed call restarts at the phase the first one stopped
    on, which is what ``phase0`` is for; without it the second call would
    re-run phase 0 and evolve the same bond twice.
    """
    D, gate_steps = 3, 8
    A0, B0 = _pair(D)
    gate = _make_trotter_gate_tensor(_heisenberg_gate(), 0.05, site_tensor=A0)

    oA, oB, o_h, o_v = _simple_update_checkerboard_sweep(A0, B0, gate, D, gate_steps)

    pA, pB, p_h, p_v = _simple_update_checkerboard_sweep(A0, B0, gate, D, 3)
    pA, pB, p_h, p_v = _simple_update_checkerboard_sweep(
        pA, pB, gate, D, gate_steps - 3, lam_h=p_h, lam_v=p_v, phase0=3
    )

    np.testing.assert_allclose(np.asarray(p_h), np.asarray(o_h), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(pA.todense()), np.asarray(oA.todense()), rtol=0, atol=0
    )


def test_to_physical_pair_matches_calling_to_physical_tensor_twice():
    """The pair helper is the two existing calls, not a new convention."""
    D = 3
    A, B = _pair(D)
    lam_h = jnp.linspace(1.0, 0.2, D)
    lam_v = jnp.linspace(1.0, 0.5, D)

    pA, pB = _to_physical_pair(A, B, lam_h, lam_v)

    np.testing.assert_array_equal(
        np.asarray(pA.todense()),
        np.asarray(_to_physical_tensor(A, lam_h, lam_v).todense()),
    )
    np.testing.assert_array_equal(
        np.asarray(pB.todense()),
        np.asarray(_to_physical_tensor(B, lam_h, lam_v).todense()),
    )
