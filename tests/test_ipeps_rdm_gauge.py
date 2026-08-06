"""Environment-phase (gauge) invariance of the ``ipeps_rdm`` energy paths.

Regression guard for #748, the follow-up to #725/#742 (PR #745 fixed the
split-CTM and PESS paths and deliberately left these).

Every CTM environment tensor carries an arbitrary complex phase -- a gauge
fixed by nothing in the sweep -- so the raw RDM network is defined only up to
an overall complex scalar.  The energy ``tr(rho H) / tr(rho)`` is a ratio, so
that scalar cancels and every phase is separately a null direction.  Measured
directly while writing this: phasing ``C1`` multiplies the raw RDM by a *pure*
phase -- ``|ratio|`` constant to 1.00000000 with an argument spread of 2e-15 --
so the normalisation is the only step that can break the invariance.

Symmetrising the RDM *before* trace-normalising does not survive it.  Writing
the raw RDM as ``H + K`` with ``H`` Hermitian and ``K`` anti-Hermitian,

    Herm(e^{i.phi} (H + K)) = cos(phi) H + sin(phi) iK,

so the phase rescales the physical part by ``cos(phi)`` and mixes ``K`` in with
weight ``sin(phi)``.

**Why a real state is the right fixture here.**  The issue frames the bug as
"only observable on complex states", which is true when comparing two
*unphased* runs -- for real data ``tr Herm(R) = tr R`` exactly.  A gauge test
applies the phase itself, and a real state then exposes it maximally: for a
real symmetric raw RDM ``M``, ``Herm(iM) = i(M - M^T)/2`` vanishes outright at
``phi = pi/2``.

The obvious alternative -- a complex site tensor pushed through ``ctm()`` --
was tried and rejected.  It converges to a degenerate environment whose raw
RDM has ``|tr| ~ 1e-28``, far below the ``EPS = 1e-15`` floor in the
normalisation denominator, so *nothing* is invariant there and the test would
be measuring fixture pathology rather than this bug.  (Standing warning: CTM
on random or near-product input is not a valid oracle.)  The state below is a
physical simple-update state whose normalised RDM trace is 0.97.

Companion to ``tests/test_split_ctm_energy_gauge.py`` (#725) and to
``test_the_energy_is_invariant_under_every_environment_phase`` in
``tests/test_ctm_root_implicit_asym.py`` (#721/#724).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ipeps import heisenberg_gate, ipeps, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_ctm_convergence import ctm
from tenax.algorithms.ipeps_rdm import compute_energy_ctm

# The angle the bug lives at: cos(pi/2) = 0 annihilates the physical part.
CRITICAL_PHASE = 0.5 * np.pi
BENIGN_PHASE = 0.7

ENV_FIELDS = ["C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"]


@pytest.fixture(scope="module")
def _state():
    """A physical simple-update state and its converged environment."""
    gate_t = sublattice_rotate_gate(heisenberg_gate())
    cfg = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=40,
        dt=0.05,
        unit_cell="1x1",
        su_init=True,
        gs_num_steps=0,
        ctm=CTMConfig(chi=4, max_iter=50, conv_tol=1e-10),
    )
    _E, tensors, _envs = ipeps(gate_t, None, cfg)
    A_t = tensors[0] if isinstance(tensors, (list, tuple)) else tensors
    A = jnp.asarray(np.asarray(A_t.todense()))
    env = ctm(A, CTMConfig(chi=8, max_iter=200, conv_tol=1e-12))
    gate = jnp.asarray(gate_t.todense()).reshape(2, 2, 2, 2)
    return A, env, gate


def _phased(env, field, phi):
    return env._replace(**{field: getattr(env, field) * jnp.exp(1j * phi)})


def test_the_fixture_is_well_conditioned(_state):
    """Guard on the premise.

    If the environment were degenerate the normalised RDM trace would collapse
    and every assertion below would be measuring noise -- which is exactly what
    happened with a complex-tensor fixture (``|tr| ~ 1e-28``).  Pin that the
    unphased path is healthy before asserting anything about phased ones.
    """
    A, env, gate = _state
    e0 = complex(compute_energy_ctm(A, env, gate, 2))
    assert np.isfinite(e0.real)
    assert -1.0 < e0.real < 0.0, f"implausible Heisenberg energy: {e0}"


@pytest.mark.parametrize("field", ENV_FIELDS)
@pytest.mark.parametrize("phi,label", [(CRITICAL_PHASE, "pi/2"), (BENIGN_PHASE, "0.7")])
def test_energy_is_invariant_under_an_environment_phase(_state, field, phi, label):
    """A pure gauge must not move the energy, at any angle."""
    A, env, gate = _state
    e0 = complex(compute_energy_ctm(A, env, gate, 2))
    e1 = complex(compute_energy_ctm(A, _phased(env, field, phi), gate, 2))
    assert abs(e1 - e0) < 1e-10 * max(abs(e0), 1.0), (
        f"phase {label} on {field} moved the energy by {abs(e1 - e0):.3e} "
        f"({e0} -> {e1}); the RDM is being symmetrised before "
        f"trace-normalising (#748)"
    )


def test_all_eight_phases_at_once_is_still_a_gauge(_state):
    """Independent phases on every tensor simultaneously -- the general case,
    since nothing couples the eight gauges to one another."""
    A, env, gate = _state
    e0 = complex(compute_energy_ctm(A, env, gate, 2))
    rng = np.random.RandomState(3)
    phased = env
    for f in ENV_FIELDS:
        phased = _phased(phased, f, float(rng.uniform(0, 2 * np.pi)))
    e1 = complex(compute_energy_ctm(A, phased, gate, 2))
    assert abs(e1 - e0) < 1e-10 * max(abs(e0), 1.0)


def test_ctm_accepts_a_complex_site_tensor():
    """``ctm()`` must not raise on a complex state.

    Found while writing the fixture above.  ``prev_sv`` was seeded as
    ``jnp.zeros(..., dtype=env.C1.dtype)`` -- complex for a complex state --
    while the loop body assigns ``_dense_svd(...)``, which is always real.
    ``lax.while_loop`` requires an invariant carry type, so it raised:

        The input carry component carry[1] has type complex128[8] but the
        corresponding output carry component has type float64[8]

    Real states are unaffected (``C1.dtype`` is already real), which is why
    nothing caught it.  Note this only asserts the call *runs* -- the
    environment it converges to on a complex state is degenerate, which is a
    separate problem and why the fixture above is real.
    """
    rng = np.random.RandomState(7)
    base = rng.standard_normal((2, 2, 2, 2, 2))
    data = base + 0.25j * rng.standard_normal((2, 2, 2, 2, 2))
    A = jnp.asarray(data / np.linalg.norm(data))
    env = ctm(A, CTMConfig(chi=8, max_iter=60, conv_tol=1e-12))
    assert bool(jnp.all(jnp.isfinite(env.C1)))
    assert jnp.iscomplexobj(env.C1)
