"""Environment-phase (gauge) invariance of the excitation RDMs.

Regression guard for #748 (follow-up to #725/#742).  See
``tests/test_ipeps_rdm_gauge.py`` for the full derivation; in short, every CTM
environment tensor carries an arbitrary complex phase, so the raw RDM is
defined only up to an overall complex scalar, and symmetrising *before*
trace-normalising turns that gauge into a physical change:

    Herm(e^{i.phi} (H + K)) = cos(phi) H + sin(phi) iK

These two RDM builders feed the excitation ``H_eff`` / ``N`` matrices, so a
gauge leak here moves the whole quasiparticle spectrum, not just an energy.

Asserted on the *RDM* rather than on a derived energy: the RDM is where the
normalisation lives, so this localises a failure instead of reporting it three
layers downstream.
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
    _rdm1x2_with_open_tensors,
    _rdm2x1_with_open_tensors,
)
from tenax.algorithms.ipeps_rdm import _build_double_layer_open

CRITICAL_PHASE = 0.5 * np.pi
BENIGN_PHASE = 0.7
ENV_FIELDS = ["C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"]


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
    _E, tensors, _envs = ipeps(gate_t, None, cfg)
    A_t = tensors[0] if isinstance(tensors, (list, tuple)) else tensors
    A = jnp.asarray(np.asarray(A_t.todense()))
    env = ctm(A, CTMConfig(chi=8, max_iter=200, conv_tol=1e-12))
    ao = _build_double_layer_open(A)
    return ao, env


def _phased(env, field, phi):
    return env._replace(**{field: getattr(env, field) * jnp.exp(1j * phi)})


@pytest.mark.parametrize(
    "builder", [_rdm2x1_with_open_tensors, _rdm1x2_with_open_tensors]
)
@pytest.mark.parametrize("field", ENV_FIELDS)
@pytest.mark.parametrize("phi,label", [(CRITICAL_PHASE, "pi/2"), (BENIGN_PHASE, "0.7")])
def test_excitation_rdm_is_invariant_under_an_environment_phase(
    _state, builder, field, phi, label
):
    ao, env = _state
    r0 = builder(ao, ao, env, 2)
    r1 = builder(ao, ao, _phased(env, field, phi), 2)
    scale = float(jnp.max(jnp.abs(r0)))
    delta = float(jnp.max(jnp.abs(r1 - r0)))
    assert delta < 1e-10 * max(scale, 1.0), (
        f"{builder.__name__}: phase {label} on {field} moved the RDM by "
        f"{delta:.3e} (scale {scale:.3e}); symmetrised before "
        f"trace-normalising (#748)"
    )


def test_the_fixture_rdm_is_nondegenerate(_state):
    """Guard on the premise: a collapsed RDM would make the above vacuous."""
    ao, env = _state
    r = _rdm2x1_with_open_tensors(ao, ao, env, 2)
    assert float(jnp.max(jnp.abs(r))) > 1e-6
    tr = complex(jnp.trace(r.reshape(4, 4)))
    assert abs(tr) > 1e-3, f"degenerate RDM, trace {tr}"
