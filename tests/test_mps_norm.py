"""``FiniteMPS.norm()`` must agree with an explicit contraction (#819).

Three separate defects, one symptom:

1. ``_raw_overlap`` built the bra with ``conj()``, which conjugates the data but
   leaves the flows alone.  On a ``SymmetricTensor`` every physical pairing was
   then OUT-against-OUT, no block matched, and the overlap collapsed to exactly
   ``0j`` -- so ``norm()`` returned ``0.0`` for a perfectly good state.
2. ``norm()`` took ``jnp.abs()`` of a quantity that is a positive real by
   construction, laundering that ``0j`` into a plausible float instead of
   failing.
3. ``norm()`` always ran the full O(L*chi^3) contraction even with a known
   orthogonality centre, where the answer is local and free.

Neither backend had a test comparing ``norm()`` against an explicit
contraction, which is why an exactly-zero norm survived.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.dmrg import build_random_symmetric_mps
from tenax.core.mps import FiniteMPS

L = 6


def _explicit_tensor_norm(mps) -> float:
    """sqrt(<t|t>) from a plain dense contraction of the site tensors."""
    n = np.ones((1, 1), dtype=complex)
    for i in range(len(mps)):
        a = np.asarray(mps[i].todense())
        n = np.einsum("ab,apc,bpd->cd", n, a.conj(), a, optimize=True)
    return float(n.ravel()[0].real) ** 0.5


def _symmetric_mps() -> FiniteMPS:
    m = build_random_symmetric_mps(L=L, bond_dim=8, seed=1, target_charge=0)
    return FiniteMPS.from_tensors([m.get_tensor(i) for i in range(L)])


def test_symmetric_norm_is_not_zero():
    """The #819 bug: a normalizable symmetric MPS reported norm exactly 0.0."""
    f = _symmetric_mps()
    assert f.orth_center is None, "this test must exercise the contraction path"
    got = f.norm()
    assert got > 0.0, "symmetric MPS reported a zero norm"
    assert abs(got - _explicit_tensor_norm(f)) < 1e-9


def test_dense_norm_matches_explicit_contraction_times_log_norm():
    """Dense was never broken -- pin it so the fix does not regress it.

    ``norm() == exp(log_norm) * sqrt(<t|t>)`` is the documented contract; the
    tensors are normalized by ``right_canonicalize`` and the scale lives in
    ``log_norm``.
    """
    d = FiniteMPS.random(L=L, d=2, chi=8, key=jax.random.PRNGKey(0))
    expected = float(np.exp(d.log_norm)) * _explicit_tensor_norm(d)
    assert abs(d.norm() - expected) < 1e-12


@pytest.mark.parametrize("symmetric", [False, True])
def test_orthogonality_centre_fast_path_agrees_with_the_contraction(symmetric):
    """The fast path must be an optimization, never a different answer."""
    mps = (
        _symmetric_mps()
        if symmetric
        else FiniteMPS.random(L=L, d=2, chi=8, key=jax.random.PRNGKey(0))
    )
    if mps.orth_center is None:
        pytest.skip("no orthogonality centre to exercise the fast path")
    slow = dataclasses.replace(mps, orth_center=None).norm()
    assert abs(mps.norm() - slow) < 1e-9


def test_the_fast_path_does_not_touch_the_contraction(monkeypatch):
    """Prove the orthogonality-centre path is actually *taken*.

    Comparing fast against slow cannot show this -- delete the fast path and
    both sides simply run the contraction and still agree.  Making
    ``_raw_overlap`` explode is what pins it, and it is also the property that
    matters: with a centre, ``norm()`` must be immune to a contraction bug.
    """
    d = FiniteMPS.random(L=L, d=2, chi=8, key=jax.random.PRNGKey(0))
    assert d.orth_center is not None

    def _boom(self, other):
        raise AssertionError("_raw_overlap must not be called with a centre set")

    monkeypatch.setattr(FiniteMPS, "_raw_overlap", _boom)
    assert d.norm() > 0.0


def _zero_state_disjoint_support() -> FiniteMPS:
    """Nonzero site tensors whose contraction is exactly the zero state.

    Site 0 lives only in bond channel 0, site 1 only in channel 1: every
    term of the contraction crosses a zero, but both tensors have norm 1.
    """
    from tenax.algorithms.tdvp import _make_site_tensor

    a = np.zeros((1, 2, 2))
    a[0, 0, 0] = 1.0
    b = np.zeros((2, 2, 1))
    b[1, 0, 0] = 1.0
    return FiniteMPS.from_tensors(
        [
            _make_site_tensor(jnp.array(a), 0, 2),
            _make_site_tensor(jnp.array(b), 1, 2),
        ]
    )


def test_a_zero_state_from_disjoint_bond_support_has_zero_norm():
    """The premise the old guard encoded is false (#948).

    ``test_a_zero_overlap_from_nonzero_tensors_raises`` asserted that nonzero
    site tensors imply a nonzero state, and the guard it pinned raised a
    "this is a tenax bug" ValueError on this perfectly legitimate state.
    Zero states are reachable representations, not contraction failures, and
    no local certificate over site norms can tell the two apart -- the #819
    defect class is pinned by the exact-contraction oracle tests above.
    """
    mps = _zero_state_disjoint_support()
    assert all(float(t.norm()) > 0.0 for t in mps.tensors)
    assert _explicit_tensor_norm(mps) == 0.0
    assert mps.norm() == 0.0


def test_a_single_zero_site_among_nonzero_sites_has_zero_norm():
    """|0>*0*|0>: one zero tensor annihilates the state; the rest are fine."""
    from tenax.algorithms.tdvp import _make_site_tensor

    arrs = [np.zeros((1, 2, 1)) for _ in range(3)]
    arrs[0][0, 0, 0] = 1.0
    arrs[2][0, 0, 0] = 1.0  # arrs[1] stays identically zero
    mps = FiniteMPS.from_tensors(
        [_make_site_tensor(jnp.array(a), i, 3) for i, a in enumerate(arrs)]
    )
    assert mps.norm() == 0.0


def test_a_negative_overlap_raises(monkeypatch):
    """<psi|psi> < 0 is impossible; abs() previously hid it."""
    f = _symmetric_mps()
    monkeypatch.setattr(FiniteMPS, "_raw_overlap", lambda self, other: -4.0 + 0j)
    with pytest.raises(ValueError, match="negative"):
        f.norm()


def test_a_complex_overlap_raises(monkeypatch):
    """<psi|psi> with a nonzero imaginary part is impossible; abs() hid it."""
    f = _symmetric_mps()
    monkeypatch.setattr(FiniteMPS, "_raw_overlap", lambda self, other: 1.0 + 0.5j)
    with pytest.raises(ValueError, match="complex"):
        f.norm()
