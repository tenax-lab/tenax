"""Pure gate logic and matrix bookkeeping of the symmetric-AD harness."""

import json
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from benchmarks.symmetric_ad import matrix, pipeline  # noqa: E402
from benchmarks.symmetric_ad.pipeline import (  # noqa: E402
    gate_s1,
    gate_s2,
    gate_s4_direction,
    gate_s5,
)
from tenax.core.symmetry import U1Symmetry, ZnSymmetry  # noqa: E402

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
    viol=0.0,
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


def test_s1_gate_table():
    assert gate_s1(1e-12, 0.0, True)  # symmetry kept, energies agree
    assert not gate_s1(1e-6, 0.0, True)  # symmetry kept, energies differ
    assert gate_s1(-4e-3, 0.13, True)  # symmetry broken: reported, not gated
    assert not gate_s1(nan, 0.0, True)
    assert not gate_s1(nan, 0.13, True)
    assert not gate_s1(1e-12, nan, True)
    assert not gate_s1(1e-12, 0.0, False)  # empty blocks
    assert not gate_s1(-4e-3, 0.13, False)
    assert gate_s1(1e-12, 1e-7, True) and not gate_s1(1e-6, 1e-7, True)
    assert gate_s1(1e-12, 1e-5, True) and not gate_s1(1e-6, 1e-5, True)  # gated
    assert gate_s1(1e-6, 1e-3, True)  # above 1e-4: broken, not gated


def test_s5_dense_gate_follows_viol():
    big = {**GOOD5, "e_dense": -0.56 + 1e-3}
    assert not gate_s5(**{**big, "viol": 0.0})[0]  # kept symmetry: gated
    assert not gate_s5(**{**big, "viol": 1e-5})[0]
    ok, parts = gate_s5(**{**big, "viol": 0.6})  # broken: recorded, not gated
    assert ok and parts["dense_ok"] is False
    assert not gate_s5(**{**GOOD5, "viol": nan})[0]
    assert not gate_s5(**{**big, "viol": nan})[0]
    # the other parts still gate when the symmetry is broken
    assert not gate_s5(**{**big, "viol": 0.6, "settled": False})[0]


# ---- R28: S5 CTM budget, split legs, resume across hashes -------------------


def test_ctm_budget_and_policy():
    c = pipeline._ctm(8)
    assert c.max_iter == 1000
    assert c.on_unconverged == "warn"


def test_s5_config_raises_on_unconverged_others_warn():
    cfg = pipeline._s5_config(3, 8, 5)
    assert cfg.ctm.on_unconverged == "raise"
    assert cfg.ctm.max_iter == 1000 and cfg.ctm.chi == 8
    assert cfg.gs_num_steps == 5 and cfg.return_history
    assert cfg.gs_verbose is True
    assert cfg.gs_log_interval == 1


def test_s5_optimizer_exception_is_a_failed_record(monkeypatch):
    from tenax.algorithms._ctm_convergence_policy import (
        CTMConvergeInfo,
        CTMNotConvergedError,
    )

    info = CTMConvergeInfo(False, 1000, 1e-6)

    def boom(*a, **k):
        raise CTMNotConvergedError(info, "gradient", 3, conv_tol=1e-10, chi=8)

    monkeypatch.setattr(pipeline, "optimize_gs_ad", boom)
    out, failed = pipeline._optimize_or_fail(None, (1, 2), None, None, 2, [[0.1]])
    assert out is None
    passed, measured, extra = failed
    assert passed is False
    exc = measured["optimizer_exception"]
    assert exc["type"] == "CTMNotConvergedError" and exc["stage"] == 2
    assert "did not converge" in exc["message"]
    assert measured["error"].startswith("CTMNotConvergedError: ")
    assert "traceback" in extra


def test_s5_optimizer_success_passes_through(monkeypatch):
    seen = {}

    def ok(gate, ab, cfg, envs_init=None):
        seen["cfg"] = cfg
        return "result"

    monkeypatch.setattr(pipeline, "optimize_gs_ad", ok)
    out, failed = pipeline._optimize_or_fail(None, (1, 2), "CFG", None, 1, [])
    assert (out, failed) == ("result", None) and seen["cfg"] == "CFG"


def test_leg_is_split_u1_and_z2():
    u1, z2 = U1Symmetry(), ZnSymmetry(2)
    assert not pipeline._leg_is_split(((-1, 2), (0, 3), (1, 2)), u1)
    assert pipeline._leg_is_split(((-1, 2), (0, 3), (1, 3)), u1)
    assert pipeline._leg_is_split(((0, 3), (1, 2)), u1)  # 1 has no partner -1
    # Z2 is self-dual: every layout is unsplit
    assert not pipeline._leg_is_split(((0, 3), (1, 2)), z2)


def _fake_env(legs):
    ix = [
        SimpleNamespace(
            label=f"l{i}",
            sectors=np.array([q for q, _ in p]),
            multiplicities=np.array([m for _, m in p]),
        )
        for i, p in enumerate(legs)
    ]
    t = SimpleNamespace(indices=ix)
    return SimpleNamespace(
        **{f: t for f in ("C1", "C2", "C3", "C4")},
        **{f: t for f in ("T1", "T2", "T3", "T4")},
    )


def test_split_legs_counts_end_to_end():
    sym_leg, split_leg = ((-1, 2), (1, 2)), ((-1, 2), (1, 3))
    envs = {(0, 0): _fake_env([sym_leg, split_leg])}
    assert pipeline.split_legs(envs, U1Symmetry()) == (8, 16)
    z2_envs = {(0, 0): _fake_env([((0, 2), (1, 3)), ((0, 1), (1, 4))])}
    assert pipeline.split_legs(z2_envs, ZnSymmetry(2)) == (0, 16)


def test_accept_hash_decides_done_and_table_shows_hash(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    _summary(old, complete=True, code_hash="H_OLD", stopped_at=None)
    _summary(new, complete=True, code_hash="H_CUR", stopped_at=None)
    assert not matrix._complete(old, "H_CUR")
    assert matrix._complete(old, "H_CUR", ["H_OLD"])
    assert matrix._complete(new, "H_CUR", ["H_OLD"])
    assert not matrix._complete(old, "H_CUR", ["H_OTHER"])
    _summary(old, complete=False, code_hash="H_OLD")
    assert not matrix._complete(old, "H_CUR", ["H_OLD"])
    _summary(new, complete=True, code_hash="H_CUR", stopped_at=None)
    assert matrix._row("p", 0, new)[-1] == "H_CUR"


SYM_DIAG = dict(
    forward_converged=True,
    forward_stationarity_residual=1.3263834475196745e-12,
    projector_backward_used="flow",
)
ORACLE_DIAG = dict(
    forward_converged=False,
    forward_stationarity_residual=1.1928425420313273,
    projector_backward_used="frozen_fallback",
)


def test_oracle_valid_measured_diags():
    assert pipeline._oracle_valid(SYM_DIAG, 1e-10) is True
    assert pipeline._oracle_valid(ORACLE_DIAG, 1e-10) is False


@pytest.mark.parametrize(
    "patch",
    [
        dict(forward_converged=False),
        dict(forward_converged=None),
        dict(forward_stationarity_residual=nan),
        dict(forward_stationarity_residual=None),
        dict(forward_stationarity_residual=1.0001e-8),
        dict(projector_backward_used="frozen_fallback"),
    ],
)
def test_oracle_valid_rejects(patch):
    assert pipeline._oracle_valid({**SYM_DIAG, **patch}, 1e-10) is False


def test_oracle_valid_threshold_and_missing_key():
    assert pipeline._oracle_valid(
        {**SYM_DIAG, "forward_stationarity_residual": 0.9999e-8}, 1e-10
    )
    # conv_tol=1e-9 -> threshold 1e-7; residual equal to it is rejected.
    thr = max(100.0 * 1e-9, 1e-8)
    assert not pipeline._oracle_valid(
        {**SYM_DIAG, "forward_stationarity_residual": thr}, 1e-9
    )
    for k in SYM_DIAG:
        d = {kk: v for kk, v in SYM_DIAG.items() if kk != k}
        assert pipeline._oracle_valid(d, 1e-10) is False


class _FakeGrad:
    def todense(self):
        return np.ones(3)


def _record(monkeypatch, broken, diag, raises=False):
    calls = []

    def fake_oracle(A, B, gate, cfg, envs):
        calls.append(1)
        if raises:
            raise RuntimeError("boom")
        return 0.0, (np.ones(3) * 2, np.ones(3) * 2)

    monkeypatch.setattr(pipeline, "oracle_value_and_grad", fake_oracle)
    monkeypatch.setattr(pipeline, "_diag", lambda: diag)
    cfg = SimpleNamespace(conv_tol=1e-10)
    g = (_FakeGrad(), _FakeGrad())
    rec = pipeline._oracle_record(None, None, None, cfg, None, broken, g)
    return rec, calls


def test_oracle_skipped_when_s1_broken(monkeypatch):
    rec, calls = _record(monkeypatch, True, SYM_DIAG)
    assert calls == []
    assert rec == {"oracle_skipped": "s1_symmetry_broken", "oracle_valid": False}


def test_oracle_runs_and_marks_validity(monkeypatch):
    rec, calls = _record(monkeypatch, False, SYM_DIAG)
    assert calls == [1] and rec["oracle_valid"] is True
    assert rec["oracle_rel"]["A"] == pytest.approx(0.5)
    rec, _ = _record(monkeypatch, False, ORACLE_DIAG)
    assert rec["oracle_valid"] is False and "oracle_rel" in rec
    assert "oracle_skipped" not in rec


def test_oracle_error_marks_invalid(monkeypatch):
    rec, _ = _record(monkeypatch, False, SYM_DIAG, raises=True)
    assert rec["oracle_valid"] is False and "RuntimeError" in rec["oracle_error"]
