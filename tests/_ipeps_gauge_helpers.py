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
