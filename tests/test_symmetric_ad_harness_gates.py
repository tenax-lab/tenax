"""Pure gate logic and matrix bookkeeping of the symmetric-AD harness."""

import json

import jax

jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402

from benchmarks.symmetric_ad import matrix  # noqa: E402
from benchmarks.symmetric_ad.pipeline import (  # noqa: E402
    gate_s2,
    gate_s4_direction,
    gate_s5,
)

nan = float("nan")
GOOD5 = dict(
    settled=True,
    drift_stable=True,
    e_start=-0.50,
    e_ret=-0.56,
    e_cold=-0.56,
    e_warm=-0.56,
    warm_converged=True,
    e_dense=-0.56,
)


def test_s2_rejects_nan_gap_and_unconverged_cold():
    assert gate_s2(True, True, 0.3) and gate_s2(True, True, float("inf"))
    assert not gate_s2(True, True, nan)
    assert not gate_s2(True, False, 0.3)
    assert not gate_s2(False, True, 0.3)


def test_s4_direction_gate():
    assert gate_s4_direction(1.0, 1.0 + 1e-8, 1e-12)[3]
    assert not gate_s4_direction(1.0, 1.0 + 1e-5, 1e-12)[3]  # off by 1e-5
    assert not gate_s4_direction(1.0, 1.0, 0.1)[3]  # no plateau
    assert not gate_s4_direction(nan, 1.0, 1e-12)[3]
    # near-degenerate: tolerance not gated, NaN still fails
    assert gate_s4_direction(1.0, 2.0, 1e-12, True)[3]
    assert not gate_s4_direction(nan, 2.0, 1e-12, True)[3]
    # the sample that failed on the GPU point: 1.6e-6 relative
    assert not gate_s4_direction(0.0089045310, 0.0089045453, 2.4e-11)[3]


@pytest.mark.parametrize(
    "key,val",
    [
        ("settled", False),
        ("drift_stable", False),
        ("e_ret", nan),
        ("e_ret", -0.40),  # above the start energy
        ("e_warm", -0.56 + 1e-9),  # cold/warm disagree
        ("warm_converged", False),
        ("e_dense", -0.56 + 1e-7),
        ("e_dense", nan),
    ],
)
def test_s5_gate_fails_each_part(key, val):
    assert gate_s5(**GOOD5)[0]
    assert not gate_s5(**{**GOOD5, key: val})[0]


def _summary(d, **kw):
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(kw))
    for s in ("S1", "S2", "S3", "S4", "S5"):
        (d / f"{s}.json").write_text(json.dumps({"passed": True, "measured": {}}))


def test_matrix_complete_needs_current_hash(tmp_path):
    _summary(tmp_path, complete=True, code_hash="abc", stopped_at=None)
    assert matrix._complete(tmp_path, "abc")
    assert not matrix._complete(tmp_path, "other")
    _summary(tmp_path, complete=False, code_hash="abc")
    assert not matrix._complete(tmp_path, "abc")


def test_matrix_row_marks_crash_failure_and_vacuous(tmp_path):
    _summary(tmp_path, complete=True, stopped_at=None, vacuous=False)
    assert matrix._row("p", 0, tmp_path)[1] == "PASS"
    assert matrix._row("p", 1, tmp_path)[1].startswith("CRASHED")  # rc != 0
    _summary(tmp_path, complete=False)
    assert matrix._row("p", 0, tmp_path)[1].startswith("CRASHED")
    _summary(tmp_path, complete=True, stopped_at="S4", error_stage="S4")
    assert matrix._row("p", 0, tmp_path)[1] == "FAILED at S4 error"
    _summary(tmp_path, complete=True, stopped_at=None, vacuous=True)
    assert "vacuous" in matrix._row("p", 0, tmp_path)[1]


def test_code_hash_covers_commits_and_uncommitted_edits(tmp_path):
    import subprocess

    from benchmarks.symmetric_ad.codehash import code_hash

    def git(*a):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "benchmarks").mkdir()
    f = tmp_path / "src" / "lib.py"
    f.write_text("x = 1\n")
    git("add", ".")
    git("commit", "-qm", "a")
    h0 = code_hash(tmp_path)
    assert code_hash(tmp_path) == h0
    f.write_text("x = 2\n")  # uncommitted edit of a library file
    h1 = code_hash(tmp_path)
    assert h1 != h0
    (tmp_path / "benchmarks" / "new.py").write_text("y = 1\n")  # untracked
    h2 = code_hash(tmp_path)
    assert h2 != h1
    git("add", ".")
    git("commit", "-qm", "b")
    assert code_hash(tmp_path) != h2  # HEAD moved
