"""The iPEPS AD guide must describe the code it documents (#808).

Two claims in ``docs/guide/algorithms/ipeps_ad_paths.md`` had drifted into
being false, and neither could be caught by anything else in the suite:

* the Path 2 configuration block specified ``forward_gauge="sigma"``, which
  ``validate_ctm_for_implicit_ad`` refuses with a ``ValueError`` before the
  first CTM sweep — a user following the guide verbatim could not run it;
* the guide marked Path 1 *Recommended* and Path 2 *Experimental* while
  ``iPEPSConfig`` defaulted to Path 2, so it recommended against its own
  default.

Both are prose, so they drift silently. These tests read the shipped markdown
and check it against the shipped code. They are deliberately narrow: they
assert the two facts that were wrong, not the guide's wording.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tenax.algorithms.ipeps_ad_policy import validate_ctm_for_implicit_ad
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

_GUIDE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "guide"
    / "algorithms"
    / "ipeps_ad_paths.md"
)


def _section(title_prefix: str) -> str:
    """The ``### <title_prefix>...`` section, up to the next ``###``/``##``."""
    text = _GUIDE.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.startswith("### ") and title_prefix in ln
        ),
        None,
    )
    assert start is not None, (
        f"no '### ...{title_prefix}...' heading in {_GUIDE}; the section was "
        "renamed and this guard is now measuring nothing"
    )
    end = next(
        (
            j
            for j in range(start + 1, len(lines))
            if lines[j].startswith("### ") or lines[j].startswith("## ")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_the_guide_exists_where_this_test_looks_for_it():
    """Guard the guard: a moved file would make every check below vacuous."""
    assert _GUIDE.is_file(), _GUIDE


def test_the_documented_path2_forward_gauge_is_one_the_policy_accepts():
    """Transcribe the guide's own configuration bullet and run it.

    This is the part of #808 that was not cosmetic: the block said
    ``forward_gauge="sigma"`` -- "required for stable element-wise
    convergence" -- and the policy check has no ``sigma`` branch at all.
    """
    section = _section("Path 2:")
    documented = re.findall(
        r"^- `forward_gauge=\"([a-z]+)\"`", section, flags=re.MULTILINE
    )
    assert documented, (
        "no '- `forward_gauge=\"...\"`' bullet found in the Path 2 section, so "
        "this test is not checking the thing it was written for"
    )
    for gauge in documented:
        # Raises ValueError if the implicit path cannot honour it.
        validate_ctm_for_implicit_ad(CTMConfig(chi=6, forward_gauge=gauge))


def test_the_documented_path2_gauge_is_actually_the_config_default():
    """And it is the value a user gets without asking, which is the point."""
    section = _section("Path 2:")
    documented = re.findall(
        r"^- `forward_gauge=\"([a-z]+)\"`", section, flags=re.MULTILINE
    )
    assert set(documented) == {CTMConfig(chi=6).forward_gauge}, documented


@pytest.mark.parametrize("gauge", ["sigma", "qr", "none"])
def test_the_guide_would_have_failed_this_check_before(gauge):
    """The check is not vacuous: every other gauge really is refused.

    Without this, a policy that accepted everything would leave the test above
    passing on any documentation at all.
    """
    with pytest.raises(ValueError):
        validate_ctm_for_implicit_ad(CTMConfig(chi=6, forward_gauge=gauge))


def test_no_paragraph_anywhere_still_pairs_sigma_with_the_implicit_path():
    """The whole document, not just the Path 2 block.

    The first version of this guard checked only the configuration bullet, and
    Codex found **four** further passages that still told a reader to use sigma
    gauge on the implicit path -- a section heading, a "this is required for
    the implicit-diff backward", a benchmark row, and a "reserve sigma gauge
    for the implicit-diff path" in Known Limitations. Any one of them sends a
    reader to a `ValueError`, so a guard scoped to one block was worth very
    little.

    What it looks for is the **claim shape**, not mere co-occurrence: "sigma"
    and an implicit-path marker within a short span, together with a directive
    word. Plain co-occurrence produces false positives on every bullet list
    that happens to mention both paths, and a guard that needs an
    ever-growing allow-list is the "checks an adjacent property" trap this
    repo keeps hitting -- it gets deleted the first time it fails on an
    unrelated edit.

    Its limits, stated rather than implied: it cannot see a claim split across
    sentences ("Sigma gauge aligns ... . This is required for the
    implicit-diff backward"), so the heading check below covers that section
    instead. It is a tripwire for the phrasing that actually occurred, not a
    natural-language linter.
    """
    text = " ".join(_GUIDE.read_text(encoding="utf-8").split()).lower()

    span = 90  # characters; long enough for one clause, short enough to mean it
    directive = re.compile(r"require|needed|need\b|reserve|necessary|must use")
    offenders = []
    for match in re.finditer(r"sigma", text):
        window = text[max(0, match.start() - span) : match.end() + span]
        if not re.search(r"implicit|gmres backward", window):
            continue
        if not directive.search(window):
            continue
        if re.search(r"no longer|used to|refuse|not available|#808", window):
            continue
        offenders.append(window.strip())

    assert not offenders, (
        "these passages still direct a reader to sigma gauge on the "
        "implicit-AD path, which refuses it:\n  " + "\n  ".join(offenders)
    )


def test_the_sigma_gauge_section_heading_names_the_path_that_accepts_it():
    """The claim the scan above cannot see, because it spans two sentences.

    ``### 2. Sigma Gauge Fixing (implicit-diff path)`` was the heading over a
    paragraph reading "This is required for the implicit-diff backward" -- the
    subject sits in the previous sentence, so no single-clause scan reaches it.
    The heading is the durable anchor, so it is what gets asserted.
    """
    text = _GUIDE.read_text(encoding="utf-8")
    headings = [
        ln for ln in text.splitlines() if ln.startswith("### ") and "Sigma" in ln
    ]
    assert headings, "no sigma-gauge section heading found; this guard is inert"
    for heading in headings:
        assert "implicit" not in heading.lower(), heading


def test_the_opening_recommended_configuration_uses_the_recommended_path():
    """The first thing a reader copies must not contradict the guidance.

    The opening ``## Recommended Configuration`` block is executable, and it
    set ``gs_implicit_ad=False`` -- explicit AD -- while the section below
    declared implicit the recommended path. Whichever way the recommendation
    goes, the example has to follow it.
    """
    text = _GUIDE.read_text(encoding="utf-8")
    start = text.index("## Recommended Configuration")
    end = text.index("## Benchmark Results", start)
    block = text[start:end]

    assignments = re.findall(r"^\s*gs_implicit_ad=(True|False),", block, re.MULTILINE)
    assert len(assignments) == 1, (
        f"expected exactly one gs_implicit_ad assignment in the opening "
        f"example; found {assignments}"
    )
    documented = assignments[0] == "True"
    assert documented is iPEPSConfig(max_bond_dim=2).gs_implicit_ad, (
        f"the opening example sets gs_implicit_ad={assignments[0]} while the "
        "config default -- and therefore the recommended path -- is "
        f"{iPEPSConfig(max_bond_dim=2).gs_implicit_ad}"
    )


def test_the_recommended_path_is_the_one_the_code_defaults_to():
    """Whichever heading carries "Recommended" must match ``gs_implicit_ad``.

    Path 1 is explicit AD (``gs_implicit_ad=False``), Path 2 is implicit
    (``True``).  The guide is free to change its recommendation; it is not free
    to disagree with the default silently.
    """
    text = _GUIDE.read_text(encoding="utf-8")
    headings = [ln for ln in text.splitlines() if ln.startswith("### Path ")]
    recommended = [ln for ln in headings if "Recommended" in ln]
    assert len(recommended) == 1, (
        f"exactly one Path heading should claim to be recommended: {recommended}"
    )

    implicit_is_default = iPEPSConfig(max_bond_dim=2).gs_implicit_ad
    expected = "Path 2" if implicit_is_default else "Path 1"
    assert recommended[0].startswith(f"### {expected}:"), (
        f"gs_implicit_ad defaults to {implicit_is_default}, so {expected} is "
        f"what runs without asking, but the guide recommends {recommended[0]!r}"
    )
