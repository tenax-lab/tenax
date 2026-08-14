"""Simple update must not throw away the largest singular value (#865).

The U(1)-Sz state reached **exactly** zero at step 22 and stayed there: every
bond weight zero, ``|A| = 0``, the RDM's trace 0 and its normalisation ``NaN``.
The filed diagnosis blamed the additive epsilon in ``sigma / (max(sigma) +
EPS)`` and said the underlying cause was unexplained.  The epsilon is real but
it is the *amplifier*; the cause is that ``base_charges`` pinned the new bond's
charge layout to the old one, which fixes the per-sector keep counts and stops
the truncation from taking the globally largest singular values.

Measured at D=3, step 0, before the fix::

    kept        [4.611e+00 1.428e+00 1.594e-01]   25.6% of the weight
    true top-3  [6.378e+00 4.611e+00 4.183e+00]   87.0% of the weight

-- the largest singular value of ``theta`` discarded outright.  Repeated, that
drives ``||theta||`` down (9.6 at step 0, 5.2e-07 by step 15, 1.9e-15 at step
16), at which point ``max(sigma)`` is comparable to ``EPS`` and the additive
epsilon returns a spectrum whose maximum is 0.643 instead of 1.  Two steps
later the state is zero.

Dense was unaffected only because ``base_charges`` is a no-op on trivial
charges -- not because the representations disagree.  Run from the *same*
state, dense survived and symmetric died, which is what localised this.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms.ipeps_simple_update as su
from tenax.algorithms.ipeps import (
    _make_trotter_gate_tensor,
    _wrap_as_dense_tensor,
    heisenberg_gate,
    heisenberg_u1sz_init_pair,
)
from tenax.algorithms.ipeps_simple_update import (
    _normalise_lambda,
    _simple_update_2site_horizontal_tensor,
    _simple_update_checkerboard_sweep,
    _truncation_base_charges,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor

#: The converged D=3 Schmidt spectrum, measured independently on the dense
#: path (and again by the BP gauge in #869).  The symmetric path must reach the
#: same physics.
DENSE_REFERENCE_D3 = (1.0, 0.165865, 0.015641)


def _four_phase(A, B, D, steps, gate_t):
    return _simple_update_checkerboard_sweep(A, B, gate_t, D, steps)


def _symmetric_run(D=3, steps=60):
    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    # Unrotated: the sublattice rotation does not conserve Sz, so it cannot be
    # cast to a U(1)-Sz SymmetricTensor at all.
    gate_t = _make_trotter_gate_tensor(heisenberg_gate(), 0.05, site_tensor=A)
    return _four_phase(A, B, D, steps, gate_t)


@pytest.mark.parametrize(
    "D",
    [
        2,
        # D=3 and D=4 reproduce the same defect and cost 26s and 176s against
        # D=2's 6s, so they are withheld from the required gate.  D=2 is not a
        # weaker check: it kills the defect mutant on its own (the collapse
        # happens at every D, only the survival time grows with it).
        pytest.param(3, marks=pytest.mark.slow),
        pytest.param(4, marks=pytest.mark.slow),
    ],
)
def test_the_symmetric_state_survives_and_stays_normalised(D):
    """The filed symptom: exactly zero at step 22, on every D.

    ``> 0`` on the norm and ``== 1`` on every ``max(lambda)``, because the
    failure produced a finite, well-formed, entirely zero answer -- an
    ``isfinite`` assertion passes on it.
    """
    A, B, lam_h, lam_v = _symmetric_run(D=D, steps=60)

    assert float(A.norm()) > 0.0, f"D={D}: A collapsed to zero"
    assert float(B.norm()) > 0.0, f"D={D}: B collapsed to zero"
    for name, lam in (("lam_h", lam_h), ("lam_v", lam_v)):
        assert float(jnp.max(lam)) == pytest.approx(1.0), (
            f"D={D}: max({name}) = {float(jnp.max(lam)):.3e}, not 1 -- the "
            f"normalisation inverted, which is how the runaway starts"
        )
        assert float(jnp.min(lam)) > 0.0, f"D={D}: {name} has a dead direction"


@pytest.mark.slow
def test_the_symmetric_spectrum_matches_the_dense_reference():
    """Surviving is not enough -- it has to be the *right* state.

    A state that stays non-zero but converges somewhere else would pass every
    assertion above.
    """
    _, _, lam_h, _ = _symmetric_run(D=3, steps=400)
    got = np.sort(np.asarray(lam_h))[::-1]
    want = np.asarray(DENSE_REFERENCE_D3)
    assert got == pytest.approx(want, rel=1e-3), (
        f"symmetric SU converged to {got}, not the dense reference {want}"
    )


def test_the_truncation_keeps_the_globally_largest_singular_values():
    """The root cause, asserted directly on the mechanism.

    Without this, the fix could regress to "survives but truncates badly" and
    only the slow spectrum comparison above would notice.
    """
    A, B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    gate_t = _make_trotter_gate_tensor(heisenberg_gate(), 0.05, site_tensor=A)

    seen = []
    real = su.truncated_svd

    def spy(theta, **kw):
        out = real(theta, **kw)
        _, sigma, _, s_full = out
        seen.append((np.asarray(sigma), np.asarray(s_full)))
        return out

    su.truncated_svd = spy
    try:
        _four_phase(A, B, 3, 8, gate_t)
    finally:
        su.truncated_svd = real

    assert seen, "the SVD was never called"
    for i, (kept, full) in enumerate(seen):
        kept = np.sort(kept)[::-1]
        best = np.sort(full)[::-1][: len(kept)]
        assert kept == pytest.approx(best, rel=1e-10, abs=0.0), (
            f"step {i}: truncation kept {kept} but the largest available were "
            f"{best} -- the charge layout is constraining the truncation"
        )


def test_the_layout_is_still_pinned_on_the_fermionic_path():
    """The constraint is not wrong, only misapplied -- fPEPS still needs it.

    There ``A.l`` and ``A.r`` are the same physical bond and the next step
    crashes if the layout drifts (#558/#559/#563), so lifting it everywhere
    would trade this bug for that one.
    """
    bosonic, _ = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    assert not bosonic.indices[bosonic.labels().index("r")].symmetry.is_fermionic
    assert _truncation_base_charges(bosonic, "r") is None

    dense = _wrap_as_dense_tensor(
        jax.random.normal(jax.random.PRNGKey(1), (2, 2, 2, 2, 2))
    )
    assert _truncation_base_charges(dense, "r") is None, "dense is unconstrained"

    D = 3
    sym = FermionParity()
    virt = np.array([i % 2 for i in range(D)], dtype=np.int32)
    phys = np.array([0, 1], dtype=np.int32)
    fermionic = SymmetricTensor.random_normal(
        (
            TensorIndex.from_charges(sym, virt, FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, virt, FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, virt, FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, virt, FlowDirection.IN, label="r"),
            TensorIndex.from_charges(sym, phys, FlowDirection.IN, label="phys"),
        ),
        jax.random.PRNGKey(2),
    )
    got = _truncation_base_charges(fermionic, "r")
    assert got is not None, "the fermionic path lost its canonical layout"
    np.testing.assert_array_equal(got, virt)


def test_lambda_normalisation_is_exact_where_the_additive_epsilon_inverted():
    """The amplifier: an absolute epsilon in a denominator (#748's class).

    At ``max(sigma) ~ EPS`` the shipped expression returned a spectrum whose
    maximum was 0.643 rather than 1; the next sweep absorbed that shrunken
    lambda and sigma shrank again.  The measured trajectories were two steps
    wide -- 2.7e-04 then 1.5e-85 -- which is a runaway, not a decay.
    """
    for scale in (1.0, 1e-8, 1e-15, 1e-30, 1e-200):
        sigma = jnp.asarray([1.0, 0.4, 0.01]) * scale
        lam = _normalise_lambda(sigma)
        assert float(jnp.max(lam)) == pytest.approx(1.0, rel=1e-15), (
            f"max(sigma)={scale:.0e}: normalisation returned "
            f"max(lambda)={float(jnp.max(lam)):.6f}"
        )
        assert float(lam[1]) == pytest.approx(0.4, rel=1e-12)

    assert np.all(np.asarray(_normalise_lambda(jnp.zeros(3))) == 0.0)
