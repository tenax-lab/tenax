"""The BP gauge is a gauge: it must move the weights and not the state.

Every way this can go wrong leaves a plausible-looking Schmidt spectrum behind,
so a test that only inspects the returned weights proves nothing.  Two mistakes
made while writing it, both of which produced perfectly reasonable spectra:

* dropping ``lambda_old`` from the re-gauging SVD -- moved the energy by 1.2e-01;
* contracting the gauge matrix into ``Gamma`` with matching flows, which on a
  ``SymmetricTensor`` silently collapses charge sectors instead of raising --
  broke the gauge by 2.7e-01 while ``DenseTensor`` stayed exact at 5e-16.

Only the invariance check below catches either, so it is the centre of this
file, and it runs on both tensor types because the second failure is invisible
on dense input.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps import (
    _make_trotter_gate_tensor,
    _wrap_as_dense_tensor,
    heisenberg_gate,
    heisenberg_u1sz_init_pair,
    sublattice_rotate_gate,
)
from tenax.algorithms.ipeps_bp_gauge import (
    BondWeights,
    _gauge_bond,
    _message,
    bp_gauge_checkerboard,
)
from tenax.algorithms.ipeps_simple_update import (
    _simple_update_2site_horizontal_tensor,
    _simple_update_2site_vertical_tensor,
)
from tenax.contraction.contractor import contract
from tenax.core._tensor_utils import scale_bond_axis

D = 3
GAUGE_TOL = 1e-13


def _dense_pair(D: int = D, seed: int = 0):
    kA, kB = jax.random.split(jax.random.PRNGKey(seed))
    A = _wrap_as_dense_tensor(jax.random.normal(kA, (D, D, D, D, 2)))
    B = _wrap_as_dense_tensor(jax.random.normal(kB, (D, D, D, D, 2)))
    return A * (1.0 / float(A.norm())), B * (1.0 / float(B.norm()))


def _symmetric_pair(D: int = D, seed: int = 0):
    return heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(seed))


_PAIRS = {"dense": _dense_pair, "symmetric": _symmetric_pair}


def _udlrp(t):
    """``t`` as a plain array in ``(u, d, l, r, phys)`` order."""
    labels = t.labels()
    return np.asarray(t.todense()).transpose(
        [labels.index(lab) for lab in ("u", "d", "l", "r", "phys")]
    )


def _torus_2x2(A, B, weights):
    """The closed 2x2 checkerboard torus, physical legs open.

    Exactly gauge invariant, and the only probe here that covers all four bonds
    at once: every bond appears twice, each time with the gauge on one end and
    its inverse on the other, so a gauge that fails to cancel shows up.
    ``_two_site`` cannot do this -- it leaves three legs open per site, so a
    gauge on any of them survives uncancelled and the comparison is meaningless.

    Sites ``A(0,0) B(1,0) / B(0,1) A(1,1)``.  Each site carries the weight of
    the bond leaving its ``r`` and ``d`` legs, which places each of the four
    weights exactly twice and each edge's weight exactly once::

        i = A00.r-B10.l  h_AB     m = A00.d-B01.u  v_AB
        j = B10.r-A00.l  h_BA *   n = B01.d-A00.u  v_BA *
        k = B01.r-A11.l  h_BA     o = B10.d-A11.u  v_BA
        l = A11.r-B01.l  h_AB *   p = A11.d-B10.u  v_AB *

    (``*`` = wraps around the torus.)
    """
    a = _udlrp(A) * np.asarray(weights.v_AB)[None, :, None, None, None]
    a = a * np.asarray(weights.h_AB)[None, None, None, :, None]
    b = _udlrp(B) * np.asarray(weights.v_BA)[None, :, None, None, None]
    b = b * np.asarray(weights.h_BA)[None, None, None, :, None]
    #      u  d  l  r  phys
    # A00  n  m  j  i  w
    # B10  p  o  i  j  x
    # B01  m  n  l  k  y
    # A11  o  p  k  l  z
    return np.einsum("nmjiw,poijx,mnlky,opklz->wxyz", a, b, b, a, optimize=True)


def _two_site(gam_L, gam_R, leg_L, leg_R, lam):
    """``gam_L -- lam -- gam_R`` across one bond, every other leg left free.

    Gauge-sensitive by construction: a gauge that does not cancel between the
    two ends shows up here, and nothing else in the pair changes.
    """
    left = scale_bond_axis(gam_L, leg_L, lam).relabel(leg_L, "__shared")
    right = gam_R.relabels(
        {lab: f"{lab}_R" for lab in gam_R.labels() if lab != leg_R}
    ).relabel(leg_R, "__shared")
    return contract(left, right)


@pytest.mark.parametrize("kind", list(_PAIRS))
def test_the_bond_gauge_leaves_the_physical_state_untouched(kind):
    """The whole construction rests on this, so it is checked exactly."""
    A, B = _PAIRS[kind]()
    lam = jnp.array([1.0, 0.4, 0.1])
    weights = BondWeights(h_AB=lam, h_BA=lam, v_AB=lam, v_BA=lam)

    before = _two_site(A, B, "r", "l", lam)
    A2, B2, lam_new = _gauge_bond(
        A,
        B,
        "r",
        "l",
        _message(A, "A", "r", weights),
        _message(B, "B", "l", weights),
        lam,
    )
    after = _two_site(A2, B2, "r", "l", lam_new)

    # The returned weight is renormalised to max 1 (the simple-update
    # convention), which rescales what the pair represents by a scalar -- so the
    # state is preserved up to normalisation, and only the direction is
    # meaningful.  Comparing without this reports the scale factor (~3.1e+01
    # here) and hides whether the gauge itself is right.
    a = np.asarray(before.todense())
    b = np.asarray(after.todense())
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    rel = float(np.linalg.norm(b - a))
    assert rel < GAUGE_TOL, (
        f"{kind}: re-gauging changed the physical state by {rel:.3e}; a gauge "
        f"transformation must leave Gamma_L lambda Gamma_R exactly invariant"
    )
    # A gauge that silently collapsed sectors would also flatten the spectrum,
    # so pin that the weights are a real, non-degenerate spectrum.
    assert float(jnp.min(lam_new)) > 0.0
    assert float(jnp.max(lam_new)) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", list(_PAIRS))
def test_the_whole_solve_preserves_a_state_with_nontrivial_weights(kind):
    """The invariance guarantee, end to end and on all four bonds at once.

    ``_gauge_bond`` is checked above with a nontrivial ``lambda``, but only one
    bond in isolation; the solve itself was only checked for structure, on
    unweighted bonds.  That gap is what let ``weights`` default to one: with
    ``lambda = 1`` every way of mishandling the incoming weights preserves the
    state anyway, so nothing here could see it (#870).
    """
    A, B = _PAIRS[kind]()
    w = BondWeights(
        h_AB=jnp.array([1.0, 0.4, 0.1]),
        h_BA=jnp.array([1.0, 0.6, 0.2]),
        v_AB=jnp.array([1.0, 0.3, 0.05]),
        v_BA=jnp.array([1.0, 0.7, 0.3]),
    )

    before = _torus_2x2(A, B, w)
    A2, B2, w2, info = bp_gauge_checkerboard(A, B, w, max_iter=400, tol=1e-13)
    assert info.converged, f"{kind}: BP did not converge ({info.residual:.2e})"
    after = _torus_2x2(A2, B2, w2)

    # Gamma and lambda are both renormalised each sweep, so only the direction
    # of the torus tensor is meaningful -- as in the single-bond check above.
    before = before / np.linalg.norm(before)
    after = after / np.linalg.norm(after)
    rel = float(np.linalg.norm(after - before))
    assert rel < GAUGE_TOL, (
        f"{kind}: the solve changed the physical state by {rel:.3e}; every step "
        f"is meant to be a gauge transformation of Gamma_A lambda Gamma_B"
    )


@pytest.mark.parametrize("kind", list(_PAIRS))
def test_the_solve_converges_and_hands_back_the_same_tensor_structure(kind):
    """Labels, axis order and flows must survive, or callers break silently."""
    A, B = _PAIRS[kind]()
    A2, B2, weights, info = bp_gauge_checkerboard(
        A, B, BondWeights.ones(D, D), max_iter=400, tol=1e-13
    )

    assert info.converged, f"{kind}: BP did not converge (residual {info.residual:.2e})"
    for original, gauged, tag in ((A, A2, "A"), (B, B2, "B")):
        assert gauged.labels() == original.labels(), f"{kind}/{tag}: axis order changed"
        assert [int(i.flow) for i in gauged.indices] == [
            int(i.flow) for i in original.indices
        ], f"{kind}/{tag}: flows changed"
        assert type(gauged) is type(original)
        assert np.isfinite(float(gauged.norm()))
    for name in weights._fields:
        w = np.asarray(getattr(weights, name))
        assert np.all(np.isfinite(w)) and np.all(w >= 0.0)


def test_bp_resolves_the_two_horizontal_bonds_separately():
    """#851's premise, measured: away from the fixed point h_AB != h_BA.

    The shipped simple update stores one spectrum for both, so it cannot
    represent this at all.
    """
    A, B = _symmetric_pair()
    _, _, weights, info = bp_gauge_checkerboard(
        A, B, BondWeights.ones(D, D), max_iter=400, tol=1e-13
    )
    assert info.converged
    h_AB = np.asarray(weights.h_AB)
    h_BA = np.asarray(weights.h_BA)
    rel = float(np.linalg.norm(h_AB - h_BA) / np.linalg.norm(h_AB))
    assert rel > 1e-2, (
        f"the two horizontal bonds came back equal to {rel:.2e}; on this input "
        f"they are inequivalent and BP should resolve them (#851)"
    )


def test_the_weights_simple_update_stores_are_not_bp_self_consistent():
    """Why this module exists: the stored weights have drifted from the spectra.

    A non-unitary gate on a neighbouring bond changes this bond's Schmidt
    values, and simple update never recomputes them -- the defect TeNPy's
    ``update_bond_imag`` and YASTN's ``EnvBP.post_truncation_`` are built to
    avoid (#869).  If this assertion ever fails, the module has lost its
    motivation and should be reconsidered, not "fixed".
    """
    A, B = _dense_pair()
    gate = sublattice_rotate_gate(heisenberg_gate())
    gate_t = _make_trotter_gate_tensor(gate, 0.05, site_tensor=A)
    lam_h, lam_v = jnp.ones(D), jnp.ones(D)
    for step in range(400):
        phase = step % 4
        if phase == 0:
            A, B, lam_h = _simple_update_2site_horizontal_tensor(
                A, B, gate_t, lam_h, lam_v, D
            )
        elif phase == 1:
            A, B, lam_v = _simple_update_2site_vertical_tensor(
                A, B, gate_t, lam_h, lam_v, D
            )
        elif phase == 2:
            B, A, lam_h = _simple_update_2site_horizontal_tensor(
                B, A, gate_t, lam_h, lam_v, D
            )
        else:
            B, A, lam_v = _simple_update_2site_vertical_tensor(
                B, A, gate_t, lam_h, lam_v, D
            )

    stored = BondWeights(h_AB=lam_h, h_BA=lam_h, v_AB=lam_v, v_BA=lam_v)
    _, _, weights, info = bp_gauge_checkerboard(A, B, stored, max_iter=400, tol=1e-13)
    assert info.converged, f"BP did not converge (residual {info.residual:.2e})"

    drift = float(
        np.linalg.norm(np.asarray(weights.h_AB) - np.asarray(lam_h))
        / np.linalg.norm(np.asarray(lam_h))
    )
    assert drift > 1e-2, (
        f"the stored spectrum and the BP-consistent one agree to {drift:.2e}; "
        f"simple update's weights were expected to have drifted (#869)"
    )
