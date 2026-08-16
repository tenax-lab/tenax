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

Since #851 landed, that reference transcription does double duty.  The sweep now
carries four spectra, but ``su_independent_bond_lambdas`` is **off** by default
and the default mirrors each freshly computed spectrum onto its partner bond --
which is what one ``lam_h`` written by phases 0 and 2 did.  Asserting the
default against this literal copy of the pre-#851 loop is what turns that
sentence from a claim into a measurement, at every residue mod 4.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps import _wrap_as_dense_tensor
from tenax.algorithms.ipeps_simple_update import (
    BondWeights,
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
    sA, sB, lam = _simple_update_checkerboard_sweep(A0, B0, gate, D, steps)

    # The shared default writes each spectrum to its partner, so both
    # horizontal bonds carry what the single ``lam_h`` carried.  Asserted
    # rather than assumed -- it is the whole content of the default.
    np.testing.assert_array_equal(np.asarray(lam.h_BA), np.asarray(lam.h_AB))
    np.testing.assert_array_equal(np.asarray(lam.v_BA), np.asarray(lam.v_AB))

    np.testing.assert_array_equal(np.asarray(lam.h_AB), np.asarray(r_h))
    np.testing.assert_array_equal(np.asarray(lam.v_AB), np.asarray(r_v))
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

    oA, oB, o_lam = _simple_update_checkerboard_sweep(A0, B0, gate, D, gate_steps)

    pA, pB, p_lam = _simple_update_checkerboard_sweep(A0, B0, gate, D, 3)
    pA, pB, p_lam = _simple_update_checkerboard_sweep(
        pA, pB, gate, D, gate_steps - 3, lambdas=p_lam, phase0=3
    )

    np.testing.assert_allclose(
        np.asarray(p_lam.h_AB), np.asarray(o_lam.h_AB), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(pA.todense()), np.asarray(oA.todense()), rtol=0, atol=0
    )


def test_to_physical_pair_matches_calling_to_physical_tensor_twice():
    """The pair helper is the two existing calls, not a new convention.

    Four *distinct* spectra, because A and B are mirror images rather than
    copies: ``A.r`` and ``B.l`` are the same bond.  With one ``lam_h`` for both
    sites the two calls took identical arguments, so a mirrored mapping was
    invisible here; four distinct spectra make a swap fail.
    """
    D = 3
    lam = BondWeights(
        h_AB=jnp.linspace(1.0, 0.2, D),
        h_BA=jnp.linspace(1.0, 0.3, D),
        v_AB=jnp.linspace(1.0, 0.4, D),
        v_BA=jnp.linspace(1.0, 0.5, D),
    )
    A, B = _pair(D)

    pA, pB = _to_physical_pair(A, B, lam)

    np.testing.assert_array_equal(
        np.asarray(pA.todense()),
        np.asarray(
            _to_physical_tensor(
                A, lam_u=lam.v_BA, lam_d=lam.v_AB, lam_l=lam.h_BA, lam_r=lam.h_AB
            ).todense()
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(pB.todense()),
        np.asarray(
            _to_physical_tensor(
                B, lam_u=lam.v_AB, lam_d=lam.v_BA, lam_l=lam.h_AB, lam_r=lam.h_BA
            ).todense()
        ),
    )
