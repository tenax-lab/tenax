"""Environment-phase (gauge) invariance of the honeycomb CTM RDMs.

Regression guard for #748 (follow-up to #725/#742).  Full derivation in
``tests/test_ipeps_rdm_gauge.py``; in short, every CTM environment tensor
carries an arbitrary complex phase, so the raw RDM is defined only up to an
overall complex scalar, and symmetrising *before* trace-normalising turns that
gauge into a physical change:

    Herm(e^{i.phi} (H + K)) = cos(phi) H + sin(phi) iK

The honeycomb environment has nine tensors *per sublattice* (three corners and
two column families over three edge directions), each with its own independent
phase -- so this path has more gauge freedom than the square-lattice one, not
less.

Asserted on the RDM rather than a derived energy, so a failure localises to
the normalisation instead of surfacing layers downstream.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_honeycomb_energy import _rdm1, _rdm2_bond
from tenax.algorithms._ctm_honeycomb_forward import honeycomb_ctm_run
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

CRITICAL_PHASE = 0.5 * np.pi
BENIGN_PHASE = 0.7
ENV_FIELDS = ["C0", "C1", "C2", "L0", "L1", "L2", "R0", "R1", "R2"]


def _site(D=2, d=2, key=None):
    """Rank-4 honeycomb site tensor with labels (e0, e1, e2, phys)."""
    sym = U1Symmetry()
    z = np.zeros(D, dtype=np.int32)
    zp = np.zeros(d, dtype=np.int32)
    data = jax.random.normal(key, (D, D, D, d))
    data = data / (jnp.linalg.norm(data) + 1e-12)
    idx = (
        TensorIndex.from_charges(sym, z.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, z.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, z.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, zp.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, idx)


@pytest.fixture(scope="module")
def _state():
    A = _site(key=jax.random.PRNGKey(0))
    B = _site(key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    envs, _info = honeycomb_ctm_run(
        sites,
        chi=4,
        max_iter=30,
        conv_tol=0.0,
        projector_method="biorthogonal",
        forward_gauge="phase",
    )
    return sites, envs


def _phased(envs, coord, field, phi):
    """Phase one tensor of one sublattice's environment."""
    env = envs[coord]
    t = getattr(env, field)
    scaled = DenseTensor(t.todense() * jnp.exp(1j * phi), t.indices)
    out = dict(envs)
    out[coord] = env._replace(**{field: scaled})
    return out


def test_the_fixture_rdm_is_nondegenerate(_state):
    """Guard on the premise: a collapsed RDM makes every assertion vacuous."""
    sites, envs = _state
    r = _rdm2_bond(sites, envs, alpha=0)
    assert float(jnp.max(jnp.abs(r))) > 1e-8
    assert abs(complex(jnp.trace(r))) > 1e-3


@pytest.mark.parametrize("field", ENV_FIELDS)
@pytest.mark.parametrize("phi,label", [(CRITICAL_PHASE, "pi/2"), (BENIGN_PHASE, "0.7")])
def test_bond_rdm_is_invariant_under_an_environment_phase(_state, field, phi, label):
    sites, envs = _state
    r0 = _rdm2_bond(sites, envs, alpha=0)
    r1 = _rdm2_bond(sites, _phased(envs, (0, 0), field, phi), alpha=0)
    scale = float(jnp.max(jnp.abs(r0)))
    delta = float(jnp.max(jnp.abs(r1 - r0)))
    assert delta < 1e-10 * max(scale, 1.0), (
        f"_rdm2_bond: phase {label} on {field} moved the RDM by {delta:.3e} "
        f"(scale {scale:.3e}); symmetrised before trace-normalising (#748)"
    )


@pytest.mark.parametrize("field", ENV_FIELDS)
def test_one_site_rdm_is_invariant_under_a_pi_over_2_phase(_state, field):
    sites, envs = _state
    r0 = _rdm1(sites, envs, sublattice=(0, 0))
    r1 = _rdm1(sites, _phased(envs, (0, 0), field, CRITICAL_PHASE), sublattice=(0, 0))
    scale = float(jnp.max(jnp.abs(r0)))
    delta = float(jnp.max(jnp.abs(r1 - r0)))
    assert delta < 1e-10 * max(scale, 1.0), (
        f"_rdm1: phase pi/2 on {field} moved the RDM by {delta:.3e} "
        f"(scale {scale:.3e}); symmetrised before trace-normalising (#748)"
    )
