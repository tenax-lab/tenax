"""Belief-propagation gauge on an absorbed-form iPEPS state.

The state here is the site tensors alone: every bond weight is split
``sqrt(lambda)`` into each of its two ends, so a site tensor is the
wavefunction rather than one factor of it.  Vidal form exists only
transiently, inside this module and inside one simple-update step.

That convention is what lets the simple-update engine hold no lambdas at
all (#882 §3).  The weights this module returns are a **diagnostic** -- the
honest Schmidt spectrum at the BP fixed point -- not part of the state.
Dropping them loses a report, not physics.
"""

from __future__ import annotations

import jax.numpy as jnp

from tenax.algorithms.ipeps_bp_gauge import (
    _BOND_OF,
    _LEGS,
    BondWeights,
    BPGaugeInfo,
    bp_gauge_checkerboard,
)
from tenax.core._tensor_utils import scale_bond_axis
from tenax.core.tensor import Tensor

__all__ = ["absorb_weights", "gauge_fix"]


def absorb_weights(A: Tensor, B: Tensor, weights: BondWeights) -> tuple[Tensor, Tensor]:
    """Split every bond weight symmetrically into its two ends.

    Vidal ``Gamma_A lambda Gamma_B`` becomes absorbed
    ``(Gamma_A sqrt(lambda)) (sqrt(lambda) Gamma_B)`` -- the same physical
    object, since ``sqrt(lam) * sqrt(lam) == lam`` on every bond exactly once.
    """
    out = {"A": A, "B": B}
    for site in ("A", "B"):
        for leg in _LEGS:
            lam = getattr(weights, _BOND_OF[(site, leg)])
            out[site] = scale_bond_axis(out[site], leg, jnp.sqrt(lam))
    return out["A"], out["B"]


def gauge_fix(
    A: Tensor,
    B: Tensor,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> tuple[Tensor, Tensor, BondWeights, BPGaugeInfo]:
    """Re-derive the BP gauge of an **absorbed-form** pair.

    Takes no incoming weights, because there are none to hand over: the pair
    already carries them.  Internally this is ``bp_gauge_checkerboard`` with
    ``BondWeights.ones`` -- correct precisely because the absorbed tensors
    already *are* ``Gamma_A lambda Gamma_B``.

    Args:
        A, B:     Absorbed-form site tensors, labels ``(u,d,l,r,phys)``.
        tol:      Stop once the largest relative weight change falls below this.
        max_iter: Maximum BP sweeps.

    Returns:
        ``(A, B, weights, info)``.  ``weights`` is a diagnostic: the Schmidt
        spectrum of the returned state at the BP fixed point.
    """
    labels = A.labels()
    D_h = A.indices[labels.index("r")].dim
    D_v = A.indices[labels.index("d")].dim
    return bp_gauge_checkerboard(
        A, B, BondWeights.ones(D_h, D_v), tol=tol, max_iter=max_iter
    )
