"""Every test file must land in a CI bucket on purpose.

``tests/conftest.py`` assigns each test file a marker (``core`` / ``algorithm``
/ ``slow``) from ``_FILE_MARKERS``.  A file that is not in that table carries
**no** marker, and the consequences are not symmetric:

* ``pytest -m core`` — the *required* CI gate — deselects it outright;
* the full suite runs only on push to ``main`` or behind the ``run-full-tests``
  label.

So an unregistered file can run in **no job that gates a merge**, and can then
fail on ``main`` indefinitely without anyone noticing.  That is not
hypothetical: it is how #790 (`test_architecture_imports`, red since #773) and
#803 (`test_fixed_point_matches_gmres_gradient`) both went unseen, and how the
80 gauge tests added by #788 turned out to be running nowhere.

This file is the guard.  It is pure filesystem inspection — no JAX, no
fixtures, microseconds — and it is itself registered as ``core`` so it runs in
the gate it protects.

**On ``_UNBUCKETED_LEGACY``.**  95 of 187 files were unregistered when this
guard was written.  Assigning each a bucket requires knowing its runtime, which
is a per-file measurement this change does not attempt.  They are therefore
listed explicitly as frozen debt: the list may **shrink, never grow**, so the
backlog is visible and drains over time while no *new* file can slip in
unbucketed.  Tracked in #805.
"""

from __future__ import annotations

import pathlib
import re

TESTS_DIR = pathlib.Path(__file__).parent


def _registered_keys() -> set[str]:
    """Filenames appearing as keys of ``_FILE_MARKERS`` in conftest."""
    src = (TESTS_DIR / "conftest.py").read_text()
    start = src.index("_FILE_MARKERS = {")
    end = src.index("\n}\n", start)
    return set(re.findall(r'"(test_[A-Za-z0-9_]+\.py)"', src[start:end]))


def _legacy_keys() -> set[str]:
    src = (TESTS_DIR / "conftest.py").read_text()
    start = src.index("_UNBUCKETED_LEGACY = {")
    end = src.index("\n}\n", start)
    return set(re.findall(r'"(test_[A-Za-z0-9_]+\.py)"', src[start:end]))


def _test_files() -> set[str]:
    """Basenames of every collected test file, including subdirectories.

    ``rglob``, not ``glob``: ``pytest_collection_modifyitems`` keys on
    ``item.path.name``, so a file keeps its bucket when it moves into a
    subdirectory — as the ``tests/stacked/`` block-sparse files did. A
    top-level-only scan would report all eight as stale keys and delete
    markers that are live.
    """
    return {p.name for p in TESTS_DIR.rglob("test_*.py")}


def test_every_test_file_is_assigned_a_ci_bucket():
    """A new test file must pick a bucket, or CI fails here.

    This is the whole point of the guard: without it, forgetting to register a
    file is silent, and the file runs in no required job.
    """
    unassigned = sorted(_test_files() - _registered_keys() - _legacy_keys())
    assert not unassigned, (
        "these test files are in no CI bucket, so `pytest -m core` (the "
        "required gate) will not run them:\n  "
        + "\n  ".join(unassigned)
        + "\n\nAdd each to `_FILE_MARKERS` in tests/conftest.py with the "
        "bucket it belongs in ('core' for cheap tests that gate a merge, "
        "'algorithm' or 'slow' otherwise). Do NOT add to "
        "`_UNBUCKETED_LEGACY` — that list is frozen debt and may only shrink."
    )


def test_the_registry_has_no_keys_for_files_that_no_longer_exist():
    """A stale key silently protects nothing.

    It also hides a rename: the old name keeps its bucket, the new one has
    none, and the file quietly leaves the gate.
    """
    files = _test_files()
    stale = sorted((_registered_keys() | _legacy_keys()) - files)
    assert not stale, (
        "tests/conftest.py registers files that do not exist (renamed or "
        "deleted?):\n  " + "\n  ".join(stale)
    )


def test_the_legacy_list_is_a_ratchet():
    """The unbucketed backlog may shrink, never grow.

    Without this the legacy list becomes the path of least resistance and the
    guard above is decorative.
    """
    legacy = _legacy_keys()
    assert len(legacy) <= 95, (
        f"_UNBUCKETED_LEGACY grew to {len(legacy)} (ceiling 95). It is frozen "
        "debt from #805: move files OUT of it into `_FILE_MARKERS`, never in. "
        "If you are draining it, lower the ceiling in this test to match."
    )


def test_a_file_is_not_in_both_the_registry_and_the_legacy_list():
    """Overlap makes the ratchet unenforceable and the bucket ambiguous."""
    both = sorted(_registered_keys() & _legacy_keys())
    assert not both, (
        "these files are both bucketed and listed as legacy debt; remove them "
        "from `_UNBUCKETED_LEGACY`:\n  " + "\n  ".join(both)
    )
