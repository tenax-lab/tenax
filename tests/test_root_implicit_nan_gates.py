"""Root-implicit gates must fail CLOSED on a non-finite value.

Every one of these gates exists to stop an invalid root or an inconsistent
adjoint from reaching a gradient.  Written as ``value > tolerance`` they do the
opposite of their purpose: ``nan > x`` is ``False``, so a NaN takes the
*non-reporting* branch precisely when the quantity is most obviously broken.

This is a recurring defect in this subsystem, not a one-off — it has now been
found and fixed on the asymmetric path (#791), on the default iPEPS AD adjoint
gate (#801/#807), and here on the C4v, symmetric and multisite engines (#796)
plus the ``gauge_consistency`` diagnostic (#787).  These tests pin the
comparison itself so the spelling cannot come back.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

# --- #796: the root/covariant residual gates ------------------------------


def test_the_residual_gate_reports_a_nan_residual():
    """A NaN residual is the strongest possible evidence the root is invalid.

    ``residual > tolerance`` sends it to the silent branch, so the engine
    proceeds to solve an adjoint at a root it cannot even measure.
    """
    from tenax.algorithms._ad_primitives import _residual_exceeds

    assert _residual_exceeds(float("nan"), 1e-6) is True
    assert _residual_exceeds(float("inf"), 1e-6) is True


def test_the_residual_gate_still_accepts_a_good_residual():
    """Without this, the fix could be ``return True`` and still pass above."""
    from tenax.algorithms._ad_primitives import _residual_exceeds

    assert _residual_exceeds(1e-9, 1e-6) is False
    assert _residual_exceeds(1e-3, 1e-6) is True
    # Exactly at tolerance is inside the gate: the callers' messages all read
    # "exceeds", and the polish loops accept on ``residual <= tol``.
    assert _residual_exceeds(1e-6, 1e-6) is False


# --- #787: the gauge_consistency diagnostic -------------------------------


def _fake_pair(value):
    """One (bar, tensor) pair whose pairing ratio is controllable.

    The pairing is ``Re(sum(bar * 1j*tensor))``, i.e. the component of the
    cotangent along the tensor's *phase* direction, so ``bar`` must be
    imaginary to produce a non-zero ratio -- a real ``bar`` gives
    ``Re(x * 1j) == 0`` for every ``x`` and the fixture would measure nothing.
    """
    return jnp.asarray([[value * 1j]], dtype=jnp.complex128), jnp.asarray(
        [[1.0]], dtype=jnp.complex128
    )


def test_gauge_consistency_does_not_report_a_nan_as_perfect():
    """``max(0.0, nan)`` keeps ``0.0`` -- the accumulator swallows the NaN.

    This is the inverted failure: the diagnostic does not merely stay quiet on
    a NaN cotangent, it reports **0.0**, i.e. *perfect* gauge consistency, on
    the exact input (#772's NaN cotangent) it was added to catch.
    """
    from tenax.algorithms._ctm_root_implicit_asym import _gauge_consistency

    bars, tensors = zip(_fake_pair(float("nan")), _fake_pair(1e-12))
    out = _gauge_consistency(bars, tensors, 1.0)
    assert math.isnan(out), f"expected NaN to propagate, got {out!r}"


def test_gauge_consistency_is_nan_regardless_of_pair_order():
    """``max`` is order-dependent on NaN: ``max(nan, 0.0)`` keeps ``nan`` but
    ``max(0.0, nan)`` keeps ``0.0``.  A fix that happens to work for one
    ordering is not a fix -- the accumulator starts at 0.0, so the swallowing
    order is the one that actually occurs.
    """
    from tenax.algorithms._ctm_root_implicit_asym import _gauge_consistency

    bars, tensors = zip(_fake_pair(1e-12), _fake_pair(float("nan")))
    assert math.isnan(_gauge_consistency(bars, tensors, 1.0))


def test_gauge_consistency_still_returns_the_max_ratio_when_finite():
    """The diagnostic must keep measuring what it measured before.

    Without this, ``return nan`` unconditionally passes both tests above.
    """
    from tenax.algorithms._ctm_root_implicit_asym import _gauge_consistency

    bars, tensors = zip(_fake_pair(2.0), _fake_pair(0.5))
    out = _gauge_consistency(bars, tensors, 1.0)
    # pairing = Re(conj-free sum(bar * 1j*tensor)); the larger |bar| dominates.
    assert math.isfinite(out)
    assert out > 0.0
