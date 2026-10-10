"""The harness runs one tiny point end to end and writes every stage record."""

import json

import jax

jax.config.update("jax_enable_x64", True)

from benchmarks.symmetric_ad import pipeline  # noqa: E402
from benchmarks.symmetric_ad.pipeline import run_point  # noqa: E402


def _rec(d, s):
    return json.loads((d / f"{s}.json").read_text())


def test_one_point_writes_all_stage_records(tmp_path):
    summary = run_point(
        "xxz03",
        "z2",
        D=2,
        chi=4,
        seed=0,
        out_dir=tmp_path,
        su_steps=20,
        opt_steps=1,
        fd_dirs=1,  # plumbing only; 3 directions cost ~200 s more
    )
    recs = {s: _rec(tmp_path, s) for s in ("S1", "S2", "S3", "S4", "S5")}
    for s, rec in recs.items():
        assert "passed" in rec and "measured" in rec
        assert "error" not in rec["measured"], (s, rec["measured"]["error"])
    assert summary["complete"] and summary["error_stage"] is None
    assert summary["stopped_at"] in (None, "S4", "S5")  # S1-S3 are plumbing
    for s in ("S1", "S2", "S3"):
        assert recs[s]["passed"], (s, recs[s]["measured"])
    assert "chi_binding" in recs["S2"]["measured"]
    assert "fd" in recs["S4"]["measured"] and "sym_diag" in recs["S4"]["measured"]
    assert len(recs["S4"]["measured"]["fd"]) == 1
    if summary["stopped_at"] is None:
        assert len(recs["S5"]["measured"]["energies"]) >= 1
        assert "E_dense_final" in recs["S5"]["measured"]


def test_a_failing_stage_stops_the_point(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("forced")

    monkeypatch.setattr(pipeline, "ipeps", boom)
    summary = run_point(
        "xxz03", "z2", D=2, chi=4, seed=0, out_dir=tmp_path, su_steps=5, opt_steps=1
    )
    assert summary["stopped_at"] == "S1" and summary["error_stage"] == "S1"
    s1 = _rec(tmp_path, "S1")
    assert s1["passed"] is False and "forced" in s1["measured"]["error"]
    s2 = _rec(tmp_path, "S2")
    assert s2["passed"] is False and "skipped" in s2["measured"]
