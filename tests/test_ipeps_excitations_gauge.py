"""Environment-phase (gauge) invariance of the excitation RDMs.

Regression guard for #748 (follow-up to #725/#742), updated for #954/#955.
See ``tests/test_ipeps_rdm_gauge.py`` for the full derivation; in short, every
CTM environment tensor carries an arbitrary complex phase, so any raw RDM
contraction is defined only up to an overall complex scalar, and that gauge
must not survive into the excitation ``H_eff`` / ``N`` matrices — a leak here
moves the whole quasiparticle spectrum, not just an energy.

Since #954 the cancellation lives one level up from where #748 first pinned
it: the raw open-tensor helpers intentionally return *unnormalised* transition
RDMs (dividing a transition operator by its own trace cancels the excitation
amplitude), and ``_rdm2x1_mixed`` / ``_rdm1x2_mixed`` divide by the
B-independent pure-``AA`` contraction of the same geometry, which carries the
same environment phase and cancels it exactly.  So this file asserts two
things, each at the level where it holds:

* the *raw* helpers are exactly covariant — a phase ``phi`` on one environment
  field multiplies the RDM by ``exp(i k phi)`` where ``k`` is the number of
  times that field enters the geometry (the T tensor along the 2-site axis
  enters twice).  This localises a gauge failure to the contraction that
  caused it;
* the *normalised* mixed RDMs — the quantities that feed ``H_eff`` / ``N`` —
  are exactly invariant.
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
from tenax.algorithms.ipeps_excitations import (
    _rdm1x2_mixed,
    _rdm1x2_with_open_tensors,
    _rdm2x1_mixed,
    _rdm2x1_with_open_tensors,
)
from tenax.algorithms.ipeps_rdm import _build_double_layer_open

CRITICAL_PHASE = 0.5 * np.pi
BENIGN_PHASE = 0.7
ENV_FIELDS = ["C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"]

# How many times each environment field enters each raw contraction: the T
# tensors along the 2-site axis appear in both the left/top and right/bottom
# halves, so a phase on them enters squared.
_FIELD_MULTIPLICITY = {
    "_rdm2x1_with_open_tensors": {**dict.fromkeys(ENV_FIELDS, 1), "T1": 2, "T3": 2},
    "_rdm1x2_with_open_tensors": {**dict.fromkeys(ENV_FIELDS, 1), "T2": 2, "T4": 2},
}


@pytest.fixture(scope="module")
def _state():
    gate_t = sublattice_rotate_gate(heisenberg_gate())
    cfg = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=40,
        dt=0.05,
        unit_cell="1x1",
        su_init=True,
        gs_num_steps=0,
        # This CTM is dead weight: ``ipeps()`` runs simple update first and the
        # tensors are fixed before it starts, so ``config.ctm`` cannot affect
        # them -- and this fixture discards both the energy and the env.  It
        # was spending its whole budget without converging (the
        # "CTM did not converge in ipeps()" warning), then throwing the
        # result away.  chi is unchanged; only the sweep count is cut (#933).
        ctm=CTMConfig(chi=4, max_iter=2, conv_tol=1e-10),
    )
    _E, tensors, _envs = ipeps(gate_t, None, cfg, compute_energy=False)
    A_t = tensors[0] if isinstance(tensors, (list, tuple)) else tensors
    A = jnp.asarray(np.asarray(A_t.todense()))
    env = ctm(A, CTMConfig(chi=8, max_iter=200, conv_tol=1e-12))
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    B = jax.random.normal(k1, A.shape) + 1j * jax.random.normal(k2, A.shape)
    return A, B, env


def _phased(env, field, phi):
    return env._replace(**{field: getattr(env, field) * jnp.exp(1j * phi)})


@pytest.mark.parametrize(
    "builder", [_rdm2x1_with_open_tensors, _rdm1x2_with_open_tensors]
)
@pytest.mark.parametrize("field", ENV_FIELDS)
@pytest.mark.parametrize("phi,label", [(CRITICAL_PHASE, "pi/2"), (BENIGN_PHASE, "0.7")])
def test_raw_rdm_is_exactly_covariant_under_an_environment_phase(
    _state, builder, field, phi, label
):
    """The raw helper carries the gauge with the geometry's exact weight.

    This is the mechanism behind the cancellation in the mixed builders: the
    pure-``AA`` trace they divide by picks up the *same* ``exp(i k phi)``.  A
    stray normalisation inside the helper (the #954 bug class, in reverse)
    or a miscounted field would break the factor here, one contraction away
    from the culprit.
    """
    A, _B, env = _state
    ao = _build_double_layer_open(A)
    r0 = builder(ao, ao, env, 2)
    r1 = builder(ao, ao, _phased(env, field, phi), 2)
    k = _FIELD_MULTIPLICITY[builder.__name__][field]
    expected = r0 * jnp.exp(1j * k * phi)
    scale = float(jnp.max(jnp.abs(r0)))
    delta = float(jnp.max(jnp.abs(r1 - expected)))
    assert delta < 1e-10 * max(scale, 1.0), (
        f"{builder.__name__}: phase {label} on {field} should scale the raw "
        f"RDM by exp({k}*i*phi); off by {delta:.3e} (scale {scale:.3e})"
    )


@pytest.mark.parametrize("builder", [_rdm2x1_mixed, _rdm1x2_mixed])
@pytest.mark.parametrize("field", ENV_FIELDS)
@pytest.mark.parametrize("phi,label", [(CRITICAL_PHASE, "pi/2"), (BENIGN_PHASE, "0.7")])
def test_mixed_rdm_is_invariant_under_an_environment_phase(
    _state, builder, field, phi, label
):
    """The normalised transition RDMs — what ``H_eff``/``N`` integrate — are
    gauge-free: the pure-``AA`` trace divides out the environment phase."""
    A, B, env = _state
    subs = (("B", "A"), ("A", "A"))
    r0 = builder(A, B, env, 2, *subs)
    r1 = builder(A, B, _phased(env, field, phi), 2, *subs)
    scale = float(jnp.max(jnp.abs(r0)))
    delta = float(jnp.max(jnp.abs(r1 - r0)))
    assert delta < 1e-10 * max(scale, 1.0), (
        f"{builder.__name__}: phase {label} on {field} moved the normalised "
        f"RDM by {delta:.3e} (scale {scale:.3e}); the AA-trace normalisation "
        f"no longer cancels the environment gauge (#748/#954)"
    )


def test_the_fixture_rdm_is_nondegenerate(_state):
    """Guard on the premise: a collapsed RDM would make the above vacuous."""
    A, B, env = _state
    ao = _build_double_layer_open(A)
    r = _rdm2x1_with_open_tensors(ao, ao, env, 2)
    assert float(jnp.max(jnp.abs(r))) > 1e-6
    tr = complex(jnp.trace(r.reshape(4, 4)))
    assert abs(tr) > 1e-3, f"degenerate RDM, trace {tr}"
    # The transition RDM the invariance test watches must be nonzero too, and
    # the all-A mixed RDM must reduce to the trace-normalised standard one.
    rt = _rdm2x1_mixed(A, B, env, 2, ("B", "A"), ("A", "A"))
    assert float(jnp.max(jnp.abs(rt))) > 1e-6, "transition RDM collapsed"
    r_aa = _rdm2x1_mixed(A, B, env, 2, ("A", "A"), ("A", "A"))
    tr_aa = complex(jnp.trace(r_aa.reshape(4, 4)))
    assert abs(tr_aa - 1.0) < 1e-12, f"all-A mixed RDM trace {tr_aa}, not 1"
