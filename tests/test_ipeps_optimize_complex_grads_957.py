"""Complex iPEPS gradients must descend and be global-phase gauge invariant.

#957: the optimizers fed ``jax.value_and_grad``'s complex cotangent straight
into Optax.  For a real objective of complex parameters JAX's cotangent pairs
unconjugated (``df = Re sum(g * dz)``), so the descent vector is ``-conj(g)``,
not ``-g``: the raw update *ascends* along the imaginary coordinates.  On the
exact product-state fixture here the explicit path returned its untouched
initial energy (0.6) because every trial step went uphill, and multiplying
the initial tensor by a physically meaningless global phase changed the final
energy (0.6 vs 0.47 on the reference path).

The fixture is the issue's own: D = chi = 1, a local Z-field gate distributed
over the square-lattice bonds, and the complex product state ``[1, 0.5j]``
with exact initial energy <Z> = 0.6.  Everything is a product state, so there
is no CTM-convergence caveat anywhere and the runs cost seconds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad

# <psi|Z|psi> for psi = [1, 0.5j]/sqrt(1.25): (1 - 0.25)/1.25.
_E_INITIAL = 0.6

# (path, optimizer): the two paths the issue measured, plus the two other
# optimizers of the explicit path.  The reference loop drives Optax directly,
# so it is exercised with adam only, as in the issue.
_CASES = [
    ("explicit", "adam"),
    ("explicit", "lbfgs"),
    ("explicit", "cg"),
    ("reference", "adam"),
]


def _gate():
    z = jnp.diag(jnp.array([1.0, -1.0]))
    gate = jnp.kron(z, jnp.eye(2)) + jnp.kron(jnp.eye(2), z)
    return gate.reshape(2, 2, 2, 2) / 4


def _initial_state():
    return jnp.array([1.0, 0.5j]).reshape(1, 1, 1, 1, 2)


def _config(path: str, optimizer: str) -> iPEPSConfig:
    reference = path == "reference"
    return iPEPSConfig(
        max_bond_dim=1,
        unit_cell="1x1",
        ctm=CTMConfig(
            chi=1,
            max_iter=5,
            min_iter=2,
            ctm_ad_mode="c4v_reference" if reference else None,
        ),
        gs_c4v=reference,
        gs_implicit_ad=reference,
        gs_explicit_ad_steps=2,
        gs_explicit_ad_warmup=1,
        gs_optimizer=optimizer,
        gs_learning_rate=0.05,
        gs_num_steps=4,
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e-12,
        su_init=False,
        gs_metric_precond=False,
    )


def test_jax_pairs_complex_cotangents_unconjugated():
    """The convention the fix is built on, pinned independently of any CTM.

    If JAX ever changes what ``grad`` returns for a real function of complex
    inputs, this fires first and points at ``_euclidean_grads``.
    """
    z = jnp.diag(jnp.array([1.0, -1.0]))
    v = jnp.array([1.0, 0.5j]) / jnp.sqrt(1.25)

    def f(v):
        return jnp.real(jnp.vdot(v, z @ v) / jnp.vdot(v, v))

    g = jax.grad(f)(v)
    raw = float(f(v - 0.05 * g))
    conjugated = float(f(v - 0.05 * jnp.conj(g)))
    base = float(f(v))
    np.testing.assert_allclose(base, _E_INITIAL, atol=1e-12)  # regime pin
    assert conjugated < base < raw, (
        f"expected conj-step < base < raw-step, got {conjugated} / {base} / "
        f"{raw} — JAX's complex-cotangent convention changed, revisit "
        f"_euclidean_grads (#957)."
    )


@pytest.mark.parametrize(("path", "optimizer"), _CASES)
def test_descends_from_a_complex_product_state(path, optimizer):
    """A valid downhill direction exists; the optimizer must find it.

    Pre-#957 the explicit path returned exactly its initial best (0.6): the
    unconjugated update ascended, so every step was rejected.
    """
    _, _, energy = optimize_gs_ad(_gate(), _initial_state(), _config(path, optimizer))
    assert float(energy) < _E_INITIAL - 0.1, (
        f"{path}/{optimizer} failed to descend from E={_E_INITIAL}: "
        f"E_final={float(energy)} (#957)"
    )


@pytest.mark.parametrize(("path", "optimizer"), _CASES)
def test_global_phase_of_the_initial_tensor_is_gauge(path, optimizer):
    """``A`` and ``i A`` are the same physical state; the result must match.

    All updates are phase-equivariant once the gradient convention is right,
    so the two trajectories are exact global-phase copies of each other and
    the energies agree to machine precision — pre-#957 they differed in the
    second decimal.
    """
    cfg = _config(path, optimizer)
    _, _, e1 = optimize_gs_ad(_gate(), _initial_state(), cfg)
    _, _, e2 = optimize_gs_ad(_gate(), 1j * _initial_state(), cfg)
    np.testing.assert_allclose(float(e1), float(e2), atol=1e-9)
