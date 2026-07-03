"""#463 Phase 2 — dense 2-site split-CTM AD (explicit + implicit).

Parity is validated on a PHYSICAL, convergent Heisenberg Néel checkerboard
(2-site simple update), never random tensors: the fused 2-site CTM oracle
oscillates on random input, making any split-vs-fused comparison meaningless.
The trusted AD gate is implicit==explicit (not implicit==finite-difference):
the split energy_fn carries a pre-existing Wirtinger gap that AD-vs-FD inherits.
"""

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tests.test_split_ctm_2site import _build_su_neel, _heisenberg_gate


@pytest.fixture(scope="module")
def su_state():
    """Convergent (A, B) Heisenberg Néel checkerboard via 2-site simple update."""
    A, B = _build_su_neel(D=2)
    return A, B


def test_converge_split_env_2site_matches_forward(su_state):
    """Forward-only multisite converge lands on the same fixed-point energy as
    ctm_split_tensor_2site (both are the Γ-gauge-fixed coupled fixed point)."""
    from tenax.algorithms._split_ctm_energy_ad import converge_split_env_2site
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 8

    envs_ref = ctm_split_tensor_2site(
        A, B, chi, max_iter=100, conv_tol=1e-12, chi_I=chi
    )
    E_ref = float(
        compute_energy_split_ctm_tensor_2site(A, B, envs_ref[0], envs_ref[1], gate, d=2)
    )

    envs = converge_split_env_2site(
        {(0, 0): A, (1, 0): B},
        CHECKERBOARD_NEIGHBORS,
        chi=chi,
        chi_I=chi,
        max_iter=100,
        conv_tol=1e-12,
        min_iter=2,
    )
    E = float(
        compute_energy_split_ctm_tensor_2site(
            A, B, envs[(0, 0)], envs[(1, 0)], gate, d=2
        )
    )
    assert abs(E - E_ref) < 1e-9, f"forward converge mismatch: {E} vs {E_ref}"


def test_explicit_multisite_converge_grad_finite(su_state):
    """Unrolled explicit multisite converge yields a finite, non-zero gradient
    w.r.t. A on the convergent state."""
    from tenax.algorithms._split_ctm_energy_ad import (
        _explicit_split_multisite_converge,
    )
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 4

    def loss(a):
        envs = _explicit_split_multisite_converge(
            {(0, 0): a, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            chi=chi,
            chi_I=chi,
            warmup_steps=10,
            backprop_steps=10,
        )
        return compute_energy_split_ctm_tensor_2site(
            a, B, envs[(0, 0)], envs[(1, 0)], gate, d=2
        ).real

    e, g = jax.value_and_grad(loss)(A)
    gs = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])
    assert jnp.isfinite(e)
    assert jnp.all(jnp.isfinite(gs)) and float(jnp.sum(jnp.abs(gs))) > 0
