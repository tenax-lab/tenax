"""Phase-1 deprecation regression tests for ``chi_ramp`` and ``chi_auto_bump``
(issue #512).

Both legacy chi-growth knobs zero-pad the env between L-BFGS steps and hand
a non-physical partial env back to the optimizer, which can descend to a
ghost minimum below the variational floor.  PR #514 landed the variPEPS
§2.8.2 in-CTM bump (``ctmrg_heuristic_increase_chi``) as the cliff-edge-
free replacement; Phase 1 emits ``DeprecationWarning`` for the legacy
paths, Phase 2 flips the default, Phase 3 removes them.

These tests pin the Phase-1 contract: the warning fires *only* on the
legacy paths, includes a pointer to ``ctmrg_heuristic_increase_chi``, and
does not fire on the recommended path or on default-constructed configs.
"""

from __future__ import annotations

import warnings

import pytest

from tenax.algorithms.ipeps_config import CTMConfig

pytestmark = pytest.mark.core


def test_chi_ramp_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="chi_ramp is deprecated"):
        CTMConfig(chi=8, chi_ramp=[(8, 10), (16, 10)])


def test_chi_auto_bump_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="chi_auto_bump .* is deprecated"):
        CTMConfig(chi=8, chi_auto_bump=True)


def test_deprecation_message_points_to_in_ctm_bump():
    """Both warnings must reference the recommended migration path."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        CTMConfig(chi=8, chi_ramp=[(8, 10)])
    chi_ramp_msgs = [
        str(w.message) for w in caught if "chi_ramp is deprecated" in str(w.message)
    ]
    assert len(chi_ramp_msgs) == 1
    assert "ctmrg_heuristic_increase_chi" in chi_ramp_msgs[0]
    assert "#512" in chi_ramp_msgs[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        CTMConfig(chi=8, chi_auto_bump=True)
    chi_bump_msgs = [
        str(w.message)
        for w in caught
        if "chi_auto_bump" in str(w.message) and "deprecated" in str(w.message)
    ]
    assert len(chi_bump_msgs) == 1
    assert "ctmrg_heuristic_increase_chi" in chi_bump_msgs[0]
    assert "#512" in chi_bump_msgs[0]


def test_default_config_does_not_warn():
    """Default-constructed CTMConfig emits no chi-knob deprecation warnings.

    Confirms the warning is gated on the legacy knobs being actively set,
    not on the dataclass defaults.  Required so the rest of the test suite
    (which constructs ``iPEPSConfig()`` everywhere) doesn't drown in
    Phase-1 noise.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        CTMConfig(chi=8)
    chi_warns = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and ("chi_ramp" in str(w.message) or "chi_auto_bump" in str(w.message))
    ]
    assert chi_warns == []


def test_recommended_in_ctm_bump_does_not_warn():
    """``ctmrg_heuristic_increase_chi=True`` (the replacement) emits no
    Phase-1 deprecation warnings.  Regression for the docstring claim
    that this is the migration target."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        CTMConfig(
            chi=8,
            chi_max=16,
            ctmrg_heuristic_increase_chi=True,
        )
    chi_warns = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and ("chi_ramp" in str(w.message) or "chi_auto_bump" in str(w.message))
    ]
    assert chi_warns == []


def test_chi_ramp_plus_chi_auto_bump_still_raises_value_error():
    """The pre-existing mutex (``chi_ramp`` and ``chi_auto_bump`` cannot
    both be set) still raises ``ValueError`` — Phase-1 only adds the
    soft warning, it does not loosen the existing validation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="mutually exclusive"):
            CTMConfig(chi=8, chi_ramp=[(8, 10)], chi_auto_bump=True)
