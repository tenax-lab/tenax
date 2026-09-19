"""PESS optimizers must be global-phase gauge invariant on complex parameters.

The #957 class, in a module the original #957 pass missed. Both PESS optimizers
(:func:`optimize_pess_ad`, :func:`optimize_pess_3site_multisite_ad`) feed
``jax.value_and_grad``'s complex cotangent straight into Optax + the Hermitian
line-search slope ``_tree_real_dot`` without the ``_euclidean_grads`` conversion
that ``ipeps_optimize`` applies at every gradient-production site.

For a real objective of complex parameters JAX's cotangent pairs *unconjugated*
(``df = Re sum(g * dz)``), so the descent vector is ``-conj(g)``, not ``-g``.
``IPESSState.random`` produces complex128 primitives, so the raw update ascends
along the imaginary coordinates and the result depends on the physically
meaningless global phase of the initial state. ``conj`` is the identity on real
leaves, so the real-valued kagome benchmarks are bit-for-bit unchanged.

Measured at D=2, d=3, chi=8, seed 2, a phase ``e^{i0.7}`` on ``T_u`` (a gauge of
both losses — asserted below):

============  ==========  ==========
path          pre-fix     post-fix
============  ==========  ==========
supersite     1.2e-1      < 1e-9
multisite*    2.0e-3      5.6e-5
============  ==========  ==========

**Mutation anchor:** the supersite line is killed at ``atol=1e-9`` (0.12 → <1e-9).
The multisite optimiser is a *separate* ``_euclidean_grads`` site; deleting only
its line reintroduces the ~2e-3 gap and fails the ``atol=5e-4`` multisite case.

\\* The multisite optimiser warm-starts its CTM env cache, which itself breaks
exact phase-equivariance (warm-on the gap floors at ~3e-3 regardless of the
gradient convention). The multisite case therefore disables warm-start to isolate
the gradient convention; the residual 5.6e-5 is the multisite CTM forward floor
(the supersite loss, being smaller, reaches 1e-9).
"""

from __future__ import annotations

import contextlib
import dataclasses
from unittest.mock import patch as _patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms import pess_optimize
from tenax.algorithms._pess_multisite_energy import kagome_3site_bond_gates
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import IPESSState, kagome_xxz_pess_cg_gates
from tenax.algorithms.pess_optimize import (
    build_pess_loss,
    build_pess_loss_3site_multisite,
    optimize_pess_3site_multisite_ad,
    optimize_pess_ad,
)

_PHASE = jnp.exp(1j * 0.7)  # a physically meaningless global phase


def _cfg(chi: int = 8) -> CTMConfig:
    return CTMConfig(
        chi=chi,
        max_iter=20,
        min_iter=4,
        conv_tol=1e-6,
        projector_method="svd",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
        gmres_tol=1e-4,
        gmres_maxiter=50,
        gmres_restart=20,
        chi_ramp=None,
    )


def _rotate(state: IPESSState) -> IPESSState:
    """Multiply the physical-index primitive by a global phase.

    A phase on ``T_u`` is a global phase on the wavefunction (the loss is
    invariant to ~1e-16, asserted as the regime pin below), so an optimizer
    with the correct gradient convention must reach the same energy.
    """
    return dataclasses.replace(state, T_u=_PHASE * state.T_u)


def _no_warm_start():
    """Disable the multisite optimiser's CTM env-cache warm-start.

    Warm-start reuses envs across steps, which itself breaks exact
    phase-equivariance (the cached env picks up the phase), flooring the gap at
    ~3e-3 no matter the gradient convention. No-op'ing ``python_loop_ctm_converge``
    (used *only* by ``_update_env_cache``; the energy path uses
    ``ctm_energy_implicit``) makes every step cold-start, isolating the gradient
    convention this test targets.
    """
    return _patch.object(
        pess_optimize, "python_loop_ctm_converge", lambda *a, **k: (None, {})
    )


# (path, atol, disable_warm_start): supersite has no warm-start and reaches
# machine precision; multisite disables warm-start and floors at the multisite
# CTM forward floor (~5.6e-5, well under 5e-4; pre-fix ~2e-3 fails it).
_CASES = [
    ("supersite", 1e-9, False),
    ("multisite", 5e-4, True),
]


@pytest.mark.parametrize(("which", "atol", "no_warm"), _CASES)
def test_pess_optimizer_is_global_phase_gauge_invariant(
    which: str, atol: float, no_warm: bool
) -> None:
    state0 = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(2))
    rot = _rotate(state0)
    cfg = _cfg(chi=8)

    if which == "supersite":
        gates = kagome_xxz_pess_cg_gates(delta=1.0, d=3)
        loss_fn = build_pess_loss(gates, cfg)

        def run(s):
            return optimize_pess_ad(s, gates, cfg, max_iter=5, verbose=False)
    else:
        gates = kagome_3site_bond_gates(delta=1.0, d=3)
        loss_fn = build_pess_loss_3site_multisite(gates, cfg)

        def run(s):
            return optimize_pess_3site_multisite_ad(
                s, gates, cfg, max_iter=5, verbose=False
            )

    # Regime pin: the phase must actually be a gauge of THIS path's loss, or the
    # invariance below is vacuous (a real-parameter or non-gauge rotation would
    # pass trivially and prove nothing about the gradient convention).
    e_plain0 = float(loss_fn(state0))
    e_rot0 = float(loss_fn(rot))
    np.testing.assert_allclose(
        e_rot0, e_plain0, atol=1e-10, err_msg=f"{which}: T_u phase is not a gauge"
    )
    # And the state must genuinely be complex, or conj is a no-op.
    assert (
        np.iscomplexobj(np.asarray(state0.T_u))
        and float(np.max(np.abs(np.asarray(state0.T_u).imag))) > 1e-3
    ), f"{which}: fixture is effectively real — conj would be a no-op"

    with _no_warm_start() if no_warm else contextlib.nullcontext():
        _, e_plain = run(state0)
        _, e_rot = run(rot)

    assert jnp.isfinite(e_plain) and jnp.isfinite(e_rot)
    np.testing.assert_allclose(
        float(e_plain),
        float(e_rot),
        atol=atol,
        err_msg=(
            f"{which}: global phase on the init changed the optimized energy "
            f"({float(e_plain):.10f} vs {float(e_rot):.10f}) — the PESS optimizer "
            f"feeds the raw JAX cotangent to Optax instead of _euclidean_grads "
            f"(#957)."
        ),
    )
