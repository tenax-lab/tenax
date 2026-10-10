"""The harness runs one tiny point end to end and writes every stage record."""

import json

import jax

jax.config.update("jax_enable_x64", True)

from benchmarks.symmetric_ad import pipeline  # noqa: E402
from benchmarks.symmetric_ad.pipeline import run_point  # noqa: E402


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
    for s in ("S1", "S2", "S3", "S4", "S5"):
        rec = json.loads((tmp_path / f"{s}.json").read_text())
        assert "passed" in rec and "measured" in rec
    assert summary["stopped_at"] in (None, "S1", "S2", "S3", "S4", "S5")


def test_a_failing_stage_stops_the_point(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("forced")

    monkeypatch.setattr(pipeline, "ipeps", boom)
    summary = run_point(
        "xxz03", "z2", D=2, chi=4, seed=0, out_dir=tmp_path, su_steps=5, opt_steps=1
    )
    assert summary["stopped_at"] == "S1"
    s1 = json.loads((tmp_path / "S1.json").read_text())
    assert s1["passed"] is False and "forced" in s1["measured"]["error"]
    s2 = json.loads((tmp_path / "S2.json").read_text())
    assert "skipped" in s2["measured"]
