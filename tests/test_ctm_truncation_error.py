"""Unit tests for the CTM truncation-error helper.

ϵ_T is the variPEPS §2.8.2 quantity: the L2 norm of the *normalized*
discarded singular values, i.e. ‖S[χ:]‖ / ‖S‖. We test against analytic
spectra so behavior is reproducible without a full CTM run.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_truncation_error import compute_truncation_error


def test_zero_when_chi_covers_full_spectrum():
    s = jnp.array([1.0, 0.5, 0.25, 0.125])
    assert float(compute_truncation_error(s, chi=4)) == pytest.approx(0.0)
    assert float(compute_truncation_error(s, chi=10)) == pytest.approx(0.0)


def test_matches_normalized_l2_of_discarded_tail():
    s = jnp.array([1.0, 0.5, 0.25, 0.125])
    expected = float(jnp.sqrt(jnp.sum(s[2:] ** 2) / jnp.sum(s**2)))
    assert float(compute_truncation_error(s, chi=2)) == pytest.approx(expected)


def test_one_when_chi_zero():
    s = jnp.array([1.0, 0.5])
    assert float(compute_truncation_error(s, chi=0)) == pytest.approx(1.0)


def test_handles_zero_spectrum_safely():
    """A zero S vector (degenerate edge case) returns 0, not NaN."""
    s = jnp.zeros(4)
    assert float(compute_truncation_error(s, chi=2)) == 0.0
