"""Source-inspection test: the gs_stall_recovery_retries cap is applied at
the CTMRGGradientError-driven reset paths in all three optimizer functions
(#454 follow-up, codex review on PR #457).

Inspecting the source rather than driving a real run because the failure mode
(Arnoldi precheck rejecting CTM gradient repeatedly) requires a degenerate
spectrum that's expensive to construct; the structural assertion that the
cap exists before the rollback at each site catches regressions without
needing to JIT-compile + run an iPEPS-AD step.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SOURCE = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "tenax"
    / "algorithms"
    / "ipeps_optimize.py"
).read_text()


def _resets_from_ctm_error_handler() -> list[tuple[str, int]]:
    """Find each (line, body) for a reset branch inside an except-CTMRGGradientError
    handler — the path that surfaced this bug.

    Both ``elif config.gs_stall_recovery == "reset":`` (1-site, 2-site) and
    ``if config.gs_stall_recovery == "reset":`` (multisite) variants count.
    """
    pattern = re.compile(
        r"(?:elif|if) config\.gs_stall_recovery == \"reset\":(?!\s*and stall_count > 0)",
        re.MULTILINE,
    )
    matches = []
    for m in pattern.finditer(_SOURCE):
        line_no = _SOURCE[: m.start()].count("\n") + 1
        # Pull the next ~200 chars of body to inspect.
        body = _SOURCE[m.end() : m.end() + 1200]
        matches.append((line_no, body))
    return matches


@pytest.mark.core
def test_ctm_error_reset_paths_have_retry_cap():
    """Each CTMRGGradientError-driven reset block must check
    ``stall_count > config.gs_stall_recovery_retries`` and ``break`` before
    rolling params back."""
    sites = _resets_from_ctm_error_handler()
    assert len(sites) == 3, (
        f"expected 3 CTMRGGradientError reset sites (1-site, 2-site, multisite); "
        f"got {len(sites)} at lines {[ln for ln, _ in sites]}"
    )

    for line_no, body in sites:
        assert "stall_count > config.gs_stall_recovery_retries" in body, (
            f"reset path at L{line_no} is missing the retry-budget check; "
            f"the gs_stall_recovery_retries cap from PR #457 is bypassed at this site"
        )
        # The cap-check should occur BEFORE the rollback (params = best_params).
        cap_idx = body.find("stall_count > config.gs_stall_recovery_retries")
        rollback_idx = body.find("params = best_params")
        assert rollback_idx == -1 or cap_idx < rollback_idx, (
            f"at L{line_no}, the cap check must precede the rollback "
            f"(cap_idx={cap_idx}, rollback_idx={rollback_idx})"
        )
        # The exhaustion log line must be present.
        assert "CTM-error stall budget exhausted" in body, (
            f"reset path at L{line_no} is missing the exhaustion log line"
        )
        # And a break statement before the closing `continue`.
        assert (
            "break" in body[: body.find("continue") if "continue" in body else 1200]
        ), (
            f"reset path at L{line_no} must `break` out of the loop on cap exhaustion, "
            f"not just `continue` (which would loop forever)"
        )
