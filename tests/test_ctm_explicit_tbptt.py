"""TBPTT truncation for explicit-AD CTM backward (#506).

Covers:

* default ``backward_steps=None`` produces a finite gradient (the pre-#506
  contract is unchanged).
* ``backward_steps == backprop_steps`` matches the ``None`` baseline
  numerically — verifies the ``truncated = 0`` short-circuit is a true
  no-op.
* ``backward_steps < backprop_steps`` still produces a finite gradient
  with a bounded relative deviation from full backward on a contractive
  CTM (geometric-decay assumption #506 relies on).
* validation: ``backward_steps`` outside ``[1, backprop_steps]`` raises
  ``ValueError`` at the ``ctm_energy_explicit`` boundary and in
  ``iPEPSConfig.__post_init__``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry

pytestmark = pytest.mark.core


def _trivial_site(D: int = 2, d: int = 2, seed: int = 0) -> DenseTensor:
    """Trivial-U(1) ``(D, D, D, D, d)`` DenseTensor for fast CTM smoke tests."""
    rng = np.random.default_rng(seed)
    sym = U1Symmetry()
    bond_charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    data = rng.standard_normal((D, D, D, D, d)).astype(np.float64)
    return DenseTensor(data, indices)


def _grad_energy(A: DenseTensor, *, backward_steps, backprop_steps: int = 8):
    """Run ``ctm_energy_explicit`` once and return (energy, |grad|)."""
    gate = heisenberg_gate()

    def _loss(data):
        site = DenseTensor(data, A.indices)
        return ctm_energy_explicit(
            {(0, 0): site},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            warmup_steps=2,
            backprop_steps=backprop_steps,
            backward_steps=backward_steps,
        )

    e, g = jax.value_and_grad(_loss)(jnp.asarray(A.todense()))
    return float(e), float(jnp.linalg.norm(g))


def test_default_none_runs_and_returns_finite_gradient():
    A = _trivial_site()
    e, gnorm = _grad_energy(A, backward_steps=None)
    assert np.isfinite(e) and np.isfinite(gnorm)
    assert gnorm > 0.0


def test_k_equals_backprop_steps_matches_full_backward():
    """``backward_steps == backprop_steps`` is the ``truncated=0`` path and
    must match the ``None`` baseline to floating-point precision."""
    A = _trivial_site()
    e_none, g_none = _grad_energy(A, backward_steps=None, backprop_steps=8)
    e_full, g_full = _grad_energy(A, backward_steps=8, backprop_steps=8)
    assert e_none == pytest.approx(e_full, rel=0.0, abs=1e-14)
    assert g_none == pytest.approx(g_full, rel=1e-12, abs=1e-14)


def test_truncated_backward_produces_finite_gradient():
    """K=4 on N=8 still produces a finite gradient; the value differs from
    full backward because the leading 4 sweeps are stop-gradient'd, but
    the optimizer-visible norm should be O(1) of the full gradient (the
    early-iter contribution is not the entire gradient)."""
    A = _trivial_site()
    e_full, g_full = _grad_energy(A, backward_steps=None, backprop_steps=8)
    e_trunc, g_trunc = _grad_energy(A, backward_steps=4, backprop_steps=8)
    assert np.isfinite(e_trunc) and np.isfinite(g_trunc)
    # Energy is identical — only the gradient changes under TBPTT.
    assert e_trunc == pytest.approx(e_full, rel=0.0, abs=1e-14)
    # Gradient norm should remain in the same order of magnitude.  Full
    # backward at chi=4 D=2 lands at gnorm ~1; we tolerate an order of
    # magnitude either side as a smoke bound (the issue's ρ^K bound is
    # tightest at converged ρ<1, but a fixed-iter 8-sweep CTM is not
    # converged, so the K=4 vs N=8 ratio is just a smoke proxy).
    assert 0.1 * g_full < g_trunc < 10.0 * g_full


def test_backward_steps_zero_raises():
    A = _trivial_site()
    with pytest.raises(ValueError, match="backward_steps must be >= 1"):
        _grad_energy(A, backward_steps=0, backprop_steps=8)


def test_backward_steps_above_backprop_steps_raises():
    A = _trivial_site()
    with pytest.raises(ValueError, match="cannot exceed"):
        _grad_energy(A, backward_steps=9, backprop_steps=8)


def test_ipeps_config_rejects_invalid_backward_steps():
    with pytest.raises(ValueError, match="must be >= 1"):
        iPEPSConfig(max_bond_dim=2, gs_explicit_ad_backward_steps=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        iPEPSConfig(
            max_bond_dim=2,
            gs_explicit_ad_steps=5,
            gs_explicit_ad_backward_steps=10,
        )


def test_ipeps_config_accepts_valid_backward_steps_and_default_none():
    c_default = iPEPSConfig(max_bond_dim=2)
    assert c_default.gs_explicit_ad_backward_steps is None
    c_set = iPEPSConfig(max_bond_dim=2, gs_explicit_ad_backward_steps=10)
    assert c_set.gs_explicit_ad_backward_steps == 10
