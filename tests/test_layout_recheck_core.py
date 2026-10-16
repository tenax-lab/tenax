"""Layout re-check between optimizer steps: cheap (core) tests."""

from __future__ import annotations

import numpy as np
import pytest

import tenax.algorithms._ctm_python_loop as cpl


def _near_tie_record(gap):
    """Two sectors, s_max = 1.  Kept: 1.0, 0.5 (sector 0) and 0.9 (sector 1).
    Dropped: 0.1 (sector 0) and 0.5 + gap (sector 1).  So the largest dropped
    value exceeds the smallest kept one by exactly ``gap``."""
    return {
        "alloc": [2, 1],
        "floor": [0, 0],
        "s": [np.array([1.0, 0.5, 0.1]), np.array([0.9, 0.5 + gap])],
    }


def test_the_probe_tie_tolerance_is_the_hysteresis_margin():
    rec = [_near_tie_record(1e-9)]
    # Default (1e-10): a dropped value 1e-9 above a kept one is a wrong cut.
    assert not cpl._probe_says_layout_is_eager(rec)
    # Margin 1e-8: the same cut is a tie, so the layout is kept.
    assert cpl._probe_says_layout_is_eager(rec, rtol=1e-8)
    # A real gap above the margin is still rejected.
    assert not cpl._probe_says_layout_is_eager([_near_tie_record(1e-6)], rtol=1e-8)


@pytest.mark.parametrize("bad", [-1e-8, float("nan"), float("inf")])
def test_python_loop_refuses_a_bad_layout_tie_rtol(bad):
    with pytest.raises(ValueError, match="layout_tie_rtol"):
        cpl.python_loop_ctm_converge({}, {}, chi=4, layout_tie_rtol=bad)
