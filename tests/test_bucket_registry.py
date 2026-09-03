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

import ast
import pathlib
import re

import pytest

TESTS_DIR = pathlib.Path(__file__).parent

_VALID_BUCKETS = {"core", "algorithm", "slow"}

# Frozen at the commit that introduced the guard (#805). Removals only:
# a file leaves this set by being bucketed in _FILE_MARKERS, never by being
# added here. Kept in the TEST, not in conftest, so the thing being checked
# and the thing checking it cannot be edited in one place.
_FROZEN_LEGACY = {
    "test_ad_primitives_rank_aware.py",
    "test_apply_chi_bump.py",
    "test_architecture_imports.py",
    "test_arnoldi.py",
    "test_block_sparse_ctm_ad.py",
    "test_c4v_reference_ad.py",
    "test_chi_ramp_chi_auto_bump_deprecation.py",
    "test_coarse_grain.py",
    "test_complex128_ad.py",
    "test_ctm_2x2_projector_symmetric.py",
    "test_ctm_670_symmetric_2x2.py",
    "test_ctm_674_fermionic_fused.py",
    "test_ctm_700_env_collapse.py",
    "test_ctm_chi_ramp.py",
    "test_ctm_compiled.py",
    "test_ctm_direction_dependent_bonds.py",
    "test_ctm_energy_implicit.py",
    "test_ctm_energy_implicit_chi_bump.py",
    "test_ctm_env_pad_chi_schedule.py",
    "test_ctm_explicit_tbptt.py",
    "test_ctm_honeycomb_ad.py",
    "test_ctm_honeycomb_convergence.py",
    "test_ctm_honeycomb_cross_path.py",
    "test_ctm_honeycomb_energy.py",
    "test_ctm_honeycomb_env.py",
    "test_ctm_honeycomb_forward.py",
    "test_ctm_honeycomb_init.py",
    "test_ctm_honeycomb_lukin_sotnikov.py",
    "test_ctm_honeycomb_moves.py",
    "test_ctm_honeycomb_projector.py",
    "test_ctm_honeycomb_safeguards.py",
    "test_ctm_implicit_warm_start_adjoint.py",
    "test_ctm_in_loop_bump_ad_paths.py",
    "test_ctm_in_loop_chi_bump.py",
    "test_ctm_loop_core.py",
    "test_ctm_multisite_2x2_contract.py",
    "test_ctm_projector.py",
    "test_ctm_recipe_2x2_production_correctness.py",
    "test_ctm_sharding_backward.py",
    "test_ctm_tensor_flow_flip.py",
    "test_ctm_tensor_init_rank1.py",
    "test_ctm_tensor_projector_2x2.py",
    "test_ctm_tensor_tiling.py",
    "test_gmres_lax.py",
    "test_hotrg_sharding.py",
    "test_ipeps_ad_adjoint_methods.py",
    "test_ipeps_ad_conv_criterion.py",
    "test_ipeps_ad_f3_fused_bwd.py",
    "test_ipeps_ad_history.py",
    "test_ipeps_ad_policy.py",
    "test_ipeps_checkpoint.py",
    "test_ipeps_checkpoint_resume.py",
    "test_ipeps_chi_adaptive_bump_unit.py",
    "test_ipeps_chi_bump_rollback_desync.py",
    "test_ipeps_chi_schedule_wiring.py",
    "test_ipeps_config_chi_ceiling_bailout.py",
    "test_ipeps_config_grad_spike.py",
    "test_ipeps_config_hz_max_iter.py",
    "test_ipeps_config_stall_recovery_retries.py",
    "test_ipeps_ctm_stall_recovery_cap.py",
    "test_ipeps_stall_recovery_cap.py",
    "test_ipeps_tree_dot.py",
    "test_ipeps_u1sz.py",
    "test_line_search.py",
    "test_line_search_auto.py",
    "test_lorentzian_eigh_kernel.py",
    "test_make_neighbors.py",
    "test_metric_precond.py",
    "test_optimize_gs_ad_chi_schedule_shim.py",
    "test_optimize_gs_ad_chi_schedule_unified.py",
    "test_pess_3site_multisite_encoding.py",
    "test_pess_3site_multisite_rdm_invariants.py",
    "test_pess_3site_multisite_wavefunction.py",
    "test_pess_ad_honeycomb.py",
    "test_pess_local_energy.py",
    "test_profiler_u1sz_arm.py",
    "test_projector_backward_dispatch.py",
    "test_reduced_corner_qr.py",
    "test_regularized_qr.py",
    "test_regularized_svd.py",
    "test_split_ctm_chi_frozen_726.py",
    "test_split_ctm_doublelayer_projector.py",
    "test_split_ctm_energy_gauge.py",
    "test_split_ctm_fuse_flag.py",
    "test_split_ctm_large_d_memory.py",
    "test_split_ctm_production_correctness.py",
    "test_sublattice_rotation.py",
    "test_svd_adjoint_fd_750.py",
    "test_symmetric_custom_vjp.py",
    "test_truncated_lowrank_svd.py",
    "test_tuning_registry.py",
    "test_u1sz_defrag_prototype_610.py",
    "test_varipeps_compare.py",
    "test_varipeps_compare_payload.py",
    "test_varipeps_compare_su.py",
}


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


def _registered_items() -> dict[str, str]:
    """``{filename: bucket}`` for every entry of ``_FILE_MARKERS``.

    Parsed rather than imported so a syntax-level typo in the *value* is
    visible; importing conftest would only surface the keys.
    """
    src = (TESTS_DIR / "conftest.py").read_text()
    start = src.index("_FILE_MARKERS = {")
    end = src.index("\n}\n", start)
    return dict(
        re.findall(r'"(test_[A-Za-z0-9_]+\.py)"\s*:\s*"([^"]*)"', src[start:end])
    )


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


def test_the_legacy_list_can_only_shrink():
    """Freeze *membership*, not size — a size ceiling is dodgeable.

    ``len(legacy) <= N`` leaves a reusable slot the moment one file is
    correctly bucketed: a later change can remove one entry and add a
    different one in the same diff, keep the count at N, and pass every
    guard while the new file still carries no marker and ``-m core`` skips
    it. That is precisely the hole this whole file exists to close.

    So the frozen set lives *here*, independent of conftest, and conftest's
    list is checked as a subset of it. Removals pass; additions cannot.
    """
    extra = sorted(_legacy_keys() - _FROZEN_LEGACY)
    assert not extra, (
        "these files were added to `_UNBUCKETED_LEGACY`, which is frozen "
        "debt (#805) and accepts removals only:\n  "
        + "\n  ".join(extra)
        + "\n\nBucket them in `_FILE_MARKERS` instead."
    )


def test_every_registered_bucket_is_a_real_bucket():
    """A typo'd bucket silently deselects the file.

    ``pytest_collection_modifyitems`` does ``getattr(pytest.mark, marker)``,
    which happily creates an unknown marker — pytest only *warns* about it,
    since this repo does not enable strict markers. So
    ``"test_new.py": "algoritm"`` registers cleanly, passes every other guard
    here, and is still deselected by ``pytest -m core``: the file looks
    bucketed and runs nowhere.
    """
    bad = {f: b for f, b in _registered_items().items() if b not in _VALID_BUCKETS}
    assert not bad, (
        "these entries in `_FILE_MARKERS` name a bucket that does not "
        f"exist (valid: {sorted(_VALID_BUCKETS)}):\n  "
        + "\n  ".join(f"{f}: {b!r}" for f, b in sorted(bad.items()))
    )


def test_a_file_is_not_in_both_the_registry_and_the_legacy_list():
    """Overlap makes the ratchet unenforceable and the bucket ambiguous."""
    both = sorted(_registered_keys() & _legacy_keys())
    assert not both, (
        "these files are both bucketed and listed as legacy debt; remove them "
        "from `_UNBUCKETED_LEGACY`:\n  " + "\n  ".join(both)
    )


# ------------------------------------------------------------------ #
# The third route in (#933)                                           #
# ------------------------------------------------------------------ #
#
# The guard above assumes ``_FILE_MARKERS`` and ``_UNBUCKETED_LEGACY`` between
# them describe every file's bucket. They do not. A file can also opt *itself*
# into a bucket with a module-level ``pytestmark = pytest.mark.core`` -- one
# line, no justification, and invisible here, because such files sit in
# ``_UNBUCKETED_LEGACY`` and the guard therefore considers them accounted for.
#
# That route was carrying **29% of the required gate's runtime** when this was
# written: test_split_ctm_doublelayer_projector.py (843s), fuse_flag (241s) and
# implicit_warm_start_adjoint (217s), none of them reviewed as a `core` entry,
# while adding a 1-second file to `_FILE_MARKERS` requires writing a paragraph
# defending it. The asymmetry is the defect.


def _module_level_pytestmarks(root: pathlib.Path | None = None) -> dict[str, set[str]]:
    """Files declaring a bucket via a module-level ``pytestmark``.

    Parsed with ``ast``, not a regex.  The first version of this matched
    ``^pytestmark\\s*=\\s*(.+)$``, which captures only to end-of-line -- so the
    standard multiline form

        pytestmark = [
            pytest.mark.core,
        ]

    captured ``"["`` and was not detected, leaving the bypass this guard exists
    to close wide open (#935 review).  The AST sees the assignment whatever its
    layout, and covers the list/tuple and ``pytest.mark.x(...)`` call forms too.

    ``rglob``, matching :func:`_test_files` above: eight test files live under
    ``tests/stacked/``, and a top-level-only scan would exempt every one of
    them.
    """

    def _buckets(node: ast.AST) -> set[str]:
        # Any ``pytest.mark.<bucket>`` anywhere in the assigned value, including
        # inside a list/tuple or as the callee of ``pytest.mark.x(...)``.
        return {
            n.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Attribute)
            and n.attr in _VALID_BUCKETS
            and isinstance(n.value, ast.Attribute)
            and n.value.attr == "mark"
        }

    found: dict[str, set[str]] = {}
    for path in sorted((root or TESTS_DIR).rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        for node in tree.body:  # module level only, not inside classes/functions
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            if not any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets
            ):
                continue
            if node.value is not None and (b := _buckets(node.value)):
                found.setdefault(path.name, set()).update(b)
    return found


def test_module_level_pytestmark_does_not_bypass_the_registry():
    """A file may not put itself in a bucket without a registry entry.

    Registering in ``_FILE_MARKERS`` is reviewed -- every entry there carries a
    written justification for the cost it adds to the gate. ``pytestmark`` is
    not. Requiring both means the expensive path and the cheap path get the
    same scrutiny, and ``--durations`` stops being the only way to discover
    what is actually in the required gate.
    """
    offenders = sorted(
        f"{name} (pytestmark={'/'.join(sorted(b))})"
        for name, b in _module_level_pytestmarks().items()
        if name not in _registered_keys()
    )
    assert not offenders, (
        "these files set a module-level `pytestmark` bucket but are absent "
        "from `_FILE_MARKERS`, so they enter the CI bucket without the "
        "justification every registered file has to carry:\n  "
        + "\n  ".join(offenders)
        + "\n\nAdd each to `_FILE_MARKERS` in tests/conftest.py (and drop it "
        "from `_UNBUCKETED_LEGACY`), keeping or removing the `pytestmark` as "
        "appropriate -- conftest will apply the registered bucket either way."
    )


def test_a_registered_file_does_not_contradict_its_own_pytestmark():
    """Both routes must agree, or the file lands in two buckets at once.

    ``conftest`` *adds* its marker rather than replacing, so a file registered
    as ``slow`` while declaring ``pytestmark = pytest.mark.core`` ends up with
    both -- and ``-m core``, a positive selector, still runs it. Silently
    moving such a file to ``slow`` would therefore change nothing, which is a
    trap worth failing on rather than discovering from a flat profile.
    """
    registered = _registered_items()
    conflicts = sorted(
        f"{name}: registered {registered[name]!r} but declares pytestmark={'/'.join(sorted(b))}"
        for name, b in _module_level_pytestmarks().items()
        if name in registered and registered[name] not in b
    )
    assert not conflicts, (
        "registry bucket and module-level pytestmark disagree; conftest adds "
        "both markers, so the file runs in both buckets:\n  " + "\n  ".join(conflicts)
    )


@pytest.mark.parametrize(
    "layout,src",
    [
        ("single line", "pytestmark = pytest.mark.core\n"),
        ("list, one line", "pytestmark = [pytest.mark.core]\n"),
        # The form the first version of the helper missed (#935 review): a
        # line-anchored regex captures "[" and finds no bucket in it.
        ("list, multiline", "pytestmark = [\n    pytest.mark.core,\n]\n"),
        (
            "tuple, multiline",
            "pytestmark = (\n    pytest.mark.core,\n    pytest.mark.slow,\n)\n",
        ),
        ("annotated", "pytestmark: object = pytest.mark.core\n"),
        (
            "mixed with a call",
            "pytestmark = [\n    pytest.mark.usefixtures('x'),\n    pytest.mark.core,\n]\n",
        ),
    ],
)
def test_the_pytestmark_scan_sees_every_declaration_layout(tmp_path, layout, src):
    """The bypass guard must not be defeated by formatting.

    A guard that only recognises one spelling of the thing it forbids is worse
    than none: it reports success while the route stays open, and the failure
    is invisible because nothing currently uses the other spelling.
    """
    f = tmp_path / "test_probe.py"
    f.write_text("import pytest\n" + src)
    tree = ast.parse(f.read_text())
    found = {
        n.attr
        for node in tree.body
        for n in ast.walk(node)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(n, ast.Attribute)
        and n.attr in _VALID_BUCKETS
        and isinstance(n.value, ast.Attribute)
        and n.value.attr == "mark"
    }
    assert "core" in found, f"{layout}: bucket not detected in {src!r}"


def test_the_scan_covers_subdirectories(tmp_path):
    """``rglob``, matching ``_test_files``.

    Exercised through the helper, not by re-running ``rglob`` beside it: the
    first version of this test asserted on ``TESTS_DIR.rglob`` directly, so
    reverting the helper to ``glob`` left it passing.  A nested probe file is
    the only thing that actually distinguishes the two.
    """
    nested = tmp_path / "stacked"
    nested.mkdir()
    (nested / "test_probe.py").write_text(
        "import pytest\npytestmark = pytest.mark.core\n"
    )
    found = _module_level_pytestmarks(root=tmp_path)
    assert found == {"test_probe.py": {"core"}}, (
        "a bucketed file in a subdirectory was not seen; the scan is "
        "top-level only and every file under tests/stacked/ is exempt"
    )


def test_real_subdirectory_files_are_still_bucketed():
    """The eight ``tests/stacked/`` files exist and are registered."""
    nested = {p.name for p in TESTS_DIR.rglob("test_*.py") if p.parent != TESTS_DIR}
    assert nested, "expected nested test files; fixture assumption changed"
    assert nested <= _test_files()
