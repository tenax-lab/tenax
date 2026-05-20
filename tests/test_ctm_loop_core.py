"""Smoke tests for _run_ctm_loop_with_bump helper."""

from __future__ import annotations

import pytest


def test_ctmloopresult_fields_present():
    """CTMLoopResult exposes all required fields with correct types."""
    from tenax.algorithms._ctm_loop_core import CTMLoopResult

    r = CTMLoopResult(
        envs={},
        converged=True,
        iterations=5,
        sv_diff=1e-9,
        max_truncation_error=0.0,
        max_smallest_S=0.0,
        final_chi=8,
        bump_extra_sweeps=0,
    )
    assert r.converged is True
    assert r.iterations == 5
    assert r.final_chi == 8
    assert r.bump_extra_sweeps == 0
