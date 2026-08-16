"""Shared probes for BP-gauge tests: a small random checkerboard pair and the
exactly gauge-invariant closed 2x2 torus contraction.

Split out of ``test_ipeps_bp_gauge.py`` (#882) so ``test_ipeps_gauge.py`` can
reuse them without one test module importing another -- ``_su_fixtures.py``
and ``_split_ctm_oracle.py`` are the existing precedent for a shared,
non-``test_``-prefixed helper module living in ``tests/``.
"""

from __future__ import annotations

import jax
import numpy as np

from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_u1sz_init_pair

D = 3


def _dense_pair(D: int = D, seed: int = 0):
    kA, kB = jax.random.split(jax.random.PRNGKey(seed))
    A = _wrap_as_dense_tensor(jax.random.normal(kA, (D, D, D, D, 2)))
    B = _wrap_as_dense_tensor(jax.random.normal(kB, (D, D, D, D, 2)))
    return A * (1.0 / float(A.norm())), B * (1.0 / float(B.norm()))


def _symmetric_pair(D: int = D, seed: int = 0):
    return heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(seed))


_PAIRS = {"dense": _dense_pair, "symmetric": _symmetric_pair}


def _fermionic_pair(D: int = 2, seed: int = 0):
    """A random ``FermionParity`` checkerboard pair, same legs and flows.

    **The default is D=2 while ``_dense_pair``/``_symmetric_pair`` default to
    D=3**, so pass ``D`` explicitly whenever two builders are iterated together
    -- both shipped call sites do.  Kept at 2 because every measured number in
    ``task-4-report.md`` was taken there and re-pointing it would invalidate
    them for no gain in this round.

    D=2 is not a general operating point for a *gauge-structure* question.
    ``_build_initial_fpeps_tensor`` builds every virtual leg from
    ``virt_charges = [i % 2 for i in range(D)]`` (``fermionic_ipeps.py:167``),
    so at D=2 each leg carries parity sectors ``{0: 1, 1: 1}``: every
    parity-preserving matrix on it is 1x1 per sector, hence diagonal, and a
    non-diagonal parity-preserving gauge cannot be expressed at all.  Anything
    asking whether a gauge is exact *in general* needs D >= 3 (this is why
    deferred Task 5 was re-pointed there; #882 §5.2).  The witness tests in
    ``test_ipeps_gauge.py`` use D=2 deliberately -- they gauge with an explicit
    diagonal matrix whose answer is known a priori, which is a claim about the
    *witness*, not about non-diagonal gauges.

    Deliberately **not** in ``_PAIRS``.  ``test_ipeps_bp_gauge.py``
    parametrises four tests over ``list(_PAIRS)``, all of which assert that
    ``bp_gauge_checkerboard`` is an exact gauge; adding a fermionic entry
    would silently extend those assertions to the graded case, which is
    exactly the open question (#882) that is deliberately left unanswered
    here.  Callers that want it must ask for it by name.
    """
    from tenax.algorithms.fermionic_ipeps import (
        FPEPSConfig,
        _build_initial_fpeps_tensor,
    )

    cfg = FPEPSConfig(D=D)
    kA, kB = jax.random.split(jax.random.PRNGKey(seed))
    A = _build_initial_fpeps_tensor(cfg, kA)
    B = _build_initial_fpeps_tensor(cfg, kB)
    return A * (1.0 / float(A.norm())), B * (1.0 / float(B.norm()))


#: Pair builders the CTM witness runs on: dense plus one *block-sparse*
#: fermionic path.  The U(1)-Sz ``"symmetric"`` pair is left out on cost
#: grounds -- its CTM is the slowest of the three and it adds no coverage the
#: other two lack (dense covers the dense contraction path, fermionic covers
#: the block-sparse one and the graded one).  It was measured in scratch and
#: behaves like the other two.
#:
#: An earlier version of this comment said the fermionic entry also covers "the
#: Koszul-carrying ``fuse`` in the double layer".  It does not: ``fuse`` carries
#: no Koszul sign (``_fuse_indices_symmetric`` permutes blocks with a bare
#: ``jnp.transpose``; grep ``_koszul_sign`` -- only ``SymmetricTensor.transpose``
#: and the ``linalg`` decompositions call it).
#:
#: The two builders here have **different default D** -- 3 for ``_dense_pair``,
#: 2 for ``_fermionic_pair`` -- so iterating this dict without an explicit ``D``
#: silently compares two bond dimensions.  Both call sites in
#: ``test_ipeps_gauge.py`` pass ``D=2`` explicitly; keep it that way.
_WITNESS_PAIRS = {"dense": _dense_pair, "fermionic": _fermionic_pair}


def _udlrp(t):
    """``t`` as a plain array in ``(u, d, l, r, phys)`` order."""
    labels = t.labels()
    return np.asarray(t.todense()).transpose(
        [labels.index(lab) for lab in ("u", "d", "l", "r", "phys")]
    )


_LEG_AXIS = {"u": 0, "d": 1, "l": 2, "r": 3}

# Which bond sits on each leg of each site, independently re-derived from
# BondWeights' own docstring (ipeps_simple_update.py) --
# ``h_AB: A.r<->B.l``, ``h_BA: B.r<->A.l``, ``v_AB: A.d<->B.u``,
# ``v_BA: B.d<->A.u`` -- rather than imported from
# ``ipeps_bp_gauge._BOND_OF``.  If the implementation's map were ever wrong,
# importing its own map back into the test would echo the same mistake into
# the expectation; writing it out again from the physical definition means
# the two have to agree independently.
_INDEPENDENT_BOND_OF: dict[tuple[str, str], str] = {
    ("A", "u"): "v_BA",
    ("A", "d"): "v_AB",
    ("A", "l"): "h_BA",
    ("A", "r"): "h_AB",
    ("B", "u"): "v_AB",
    ("B", "d"): "v_BA",
    ("B", "l"): "h_AB",
    ("B", "r"): "h_BA",
}


def assert_leg_split(site, before, after, scale_of_leg, tol, msg=""):
    """Assert ``after`` is ``before`` scaled independently along each leg.

    ``_torus_2x2`` is a closed loop, so it cannot see an asymmetric split: a
    diagonal weight factors arbitrarily between the two ends of a bond
    without changing the total the torus sums over (``sqrt(lam)*sqrt(lam)``
    and ``lam*1`` agree on the shared index), so a routine that dumps the
    *whole* weight onto one site's legs and leaves the other at 1 reproduces
    the identical torus value -- and, because BP-gauging is insensitive to
    which valid gauge of the same state it starts from, would likely still
    reach the same fixed point too.  This compares dense ``(u,d,l,r,phys)``
    arrays directly instead, leg by leg, on a single tensor with no torus and
    no BP involved, so that failure mode cannot hide.

    Args:
        site:          Label used only in the assertion message.
        before, after: The tensor before and after the claimed scaling.
        scale_of_leg:  ``{"u": ..., "d": ..., "l": ..., "r": ...}``, the
                       factor expected on each leg (already at the intended
                       power, e.g. already ``sqrt(lambda)``).
        tol:           Max allowed elementwise abs difference.
        msg:           Prefix for the assertion message.
    """
    expected = _udlrp(before)
    for leg, axis in _LEG_AXIS.items():
        scale = np.asarray(scale_of_leg[leg])
        shape = [1, 1, 1, 1, 1]
        shape[axis] = scale.shape[0]
        expected = expected * scale.reshape(shape)
    got = _udlrp(after)
    d = float(np.max(np.abs(got - expected)))
    assert d < tol, (
        f"{msg}site {site}: leg-wise split check failed (max abs diff {d:.3e})"
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
