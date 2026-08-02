"""Environment-phase (gauge) invariance of the split-CTM energy path.

Regression guard for #725.  Every environment tensor carries an arbitrary
complex phase -- a gauge of the CTM, fixed by nothing in the sweep -- so the
raw RDM network is only ever defined up to an overall complex scalar.  The
energy ``tr(rho H) / tr(rho)`` is a ratio, so it cancels that scalar and every
phase is separately a null direction.

Symmetrising the RDM *before* trace-normalising does not survive it.  Writing
the raw RDM as ``H + K`` with ``H`` Hermitian and ``K`` anti-Hermitian,

    Herm(e^{i.phi} (H + K)) = cos(phi) H + sin(phi) iK,

so the phase rescales the physical part by ``cos(phi)`` and mixes ``K`` in with
weight ``sin(phi)``.  A converged environment has ``|K|/|H| ~ 1e-14``, which
hides this at generic angles -- the #725 measurement was 1e-14 of movement at
phi = 0.7 but 5.4e-3 at phi = pi/2, where the cosine annihilates ``H`` outright
and the energy is computed entirely from the noise in ``K``.  **pi/2 is the
angle that catches it; a sweep of generic angles passes either way.**

These paths are only observably broken on a complex state: for real data
``tr Herm(R) = tr R`` and the two orderings agree identically.  Hence the
complex site tensor below -- the same construction as
``tests/test_ctm_root_implicit_asym.py::_complex_site_tensor`` so the numbers
stay comparable with #721.

Companion to ``test_the_energy_is_invariant_under_every_environment_phase`` in
``tests/test_ctm_root_implicit_asym.py``, which covers the fused path fixed in
#721/#724.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
from tenax.algorithms._split_ctm_tensor_energy import compute_energy_split_ctm_tensor
from tenax.core.tensor import DenseTensor


def _oracle():
    try:
        from tests._split_ctm_oracle import heisenberg_gate, make_site
    except ModuleNotFoundError:
        from _split_ctm_oracle import heisenberg_gate, make_site
    return make_site, heisenberg_gate


def _complex_site(D=2, d=2, seed=7, imag_seed=7, scale=0.25):
    """The real split-CTM test state plus an imaginary part, renormalised."""
    make_site, _ = _oracle()
    A = make_site(D, d, seed=seed)
    rng = np.random.RandomState(imag_seed)
    base = np.asarray(A.todense())
    data = base + scale * 1j * rng.standard_normal(base.shape)
    data = jnp.asarray(data / np.linalg.norm(data))
    return DenseTensor(data, A.indices)


# The 12 fields of SplitCTMTensorEnv: 4 corners + ket/bra halves of 4 edges.
_ENV_FIELDS = (
    "C1",
    "C2",
    "C3",
    "C4",
    "T1_ket",
    "T1_bra",
    "T2_ket",
    "T2_bra",
    "T3_ket",
    "T3_bra",
    "T4_ket",
    "T4_bra",
)


@pytest.fixture(scope="module")
def _converged():
    A = _complex_site()
    _, heisenberg_gate = _oracle()
    gate = heisenberg_gate()
    env = ctm_split_tensor(A, chi=4, chi_I=8, max_iter=300, conv_tol=0.0)
    return A, env, gate


def _energy_with_phase(A, env, gate, theta, fields):
    """Energy with ``exp(i.theta)`` applied to each named environment tensor."""
    phase = jnp.exp(1j * theta)
    rotated = env._replace(
        **{name: phase * getattr(env, name) for name in fields},
    )
    return float(jnp.real(compute_energy_split_ctm_tensor(A, rotated, gate)))


@pytest.mark.parametrize("field", _ENV_FIELDS)
def test_split_energy_is_invariant_under_each_environment_phase(field, _converged):
    """A phase on any single split-env tensor cannot move the energy.

    Each of the 12 tensors is tested on its own: the RDM is one network, so a
    phase on any tensor in it is an overall complex scalar on ``rho``, and all
    12 phases are separately free rather than only their sum.
    """
    A, env, gate = _converged
    base = _energy_with_phase(A, env, gate, 0.0, ())
    assert abs(base) > 1e-3, f"degenerate baseline energy {base}"

    for theta in (0.2, 0.7, float(np.pi / 2)):
        moved = _energy_with_phase(A, env, gate, theta, (field,))
        assert abs(moved - base) < 1e-11, (field, theta, moved - base)


def test_split_energy_is_invariant_under_a_global_environment_phase(_converged):
    """The same, with the phase on all 12 tensors at once."""
    A, env, gate = _converged
    base = _energy_with_phase(A, env, gate, 0.0, ())
    for theta in (0.2, 0.7, float(np.pi / 2)):
        moved = _energy_with_phase(A, env, gate, theta, _ENV_FIELDS)
        assert abs(moved - base) < 1e-11, (theta, moved - base)


def test_pi_over_2_is_the_angle_that_catches_the_bug(_converged):
    """Guard the guard: pi/2 must be materially harder than a generic angle.

    Without it this suite would pass against the pre-#725 ordering.  Reverting
    ``_normalise_rdm`` to symmetrise-first moves the energy by ~1e-3 at pi/2 and
    by ~1e-14 at 0.7, so a test that only swept generic angles would have been
    green on the bug.  This asserts the *raw* RDM really is non-Hermitian enough
    for the ordering to matter -- i.e. that the test above has teeth.
    """
    A, env, gate = _converged
    base = _energy_with_phase(A, env, gate, 0.0, ())

    # Reproduce the old (buggy) ordering on the same network: symmetrise the
    # phased RDM first, then trace-normalise.
    from tenax.algorithms._split_ctm_tensor_energy import _rdm2x1_split_tensor

    rdm = jnp.asarray(_rdm2x1_split_tensor(A, env))
    d = rdm.shape[0]
    mat = rdm.reshape(d * d, d * d)

    def old_order(theta):
        m = jnp.exp(1j * theta) * mat
        m = 0.5 * (m + m.conj().T)
        m = m / (jnp.trace(m) + 1e-30)
        return jnp.real(jnp.einsum("ijkl,ijkl->", m.reshape(d, d, d, d), gate))

    generic = abs(float(old_order(0.7)) - float(old_order(0.0)))
    at_pi_2 = abs(float(old_order(float(np.pi / 2))) - float(old_order(0.0)))
    assert at_pi_2 > 1e3 * max(generic, 1e-16), (
        "pi/2 no longer distinguishes the two orderings, so the invariance "
        f"tests above have lost their teeth (generic={generic:.3e}, "
        f"pi/2={at_pi_2:.3e})"
    )
    assert abs(base) > 1e-3
