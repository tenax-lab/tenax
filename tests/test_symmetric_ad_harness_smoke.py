"""The harness runs one tiny point end to end and writes every stage record."""

import json

import jax
import pytest

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


def _sym_env(sym):
    """A few eager symmetric CTM sweeps at a random D=2 pair (the measure is
    tested as a mechanism, not at a fixed point)."""
    import dataclasses

    from benchmarks.symmetric_ad.pipeline import SYMS, _ctm
    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    from tenax.algorithms.ctm_layout import normalized_pair
    from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs

    A, B = SYMS[sym][0](2, jax.random.PRNGKey(0))
    A_n, B_n = normalized_pair(A, B)
    cfg = dataclasses.replace(_ctm(4), max_iter=6, conv_tol=1e-30)
    envs, _ = python_loop_ctm_converge(
        {(0, 0): A_n, (1, 0): B_n}, CHECKERBOARD_NEIGHBORS, **ctm_converge_kwargs(cfg)
    )
    return A_n, B_n, envs


@pytest.mark.parametrize("sym", ["z2", "u1"])
def test_charge_violation_measure_sees_only_violating_weight(sym):
    """A symmetric env has viol = 0; a charge-violating perturbation of one
    env tensor gives viol >> 0.  The regime is asserted: the perturbation
    adds weight outside the symmetric blocks."""
    import jax.numpy as jnp
    import numpy as np

    from benchmarks.symmetric_ad.pipeline import charge_violation
    from tenax.algorithms._symmetric_ad_validation import densify_envs, densify_pair
    from tenax.core.tensor import DenseTensor

    A_n, B_n, envs = _sym_env(sym)
    Ad, Bd = densify_pair(A_n, B_n)
    denv = densify_envs(envs)
    assert charge_violation(Ad, Bd, denv) < 1e-12

    # Every env tensor of both sites enters some bond RDM; perturbing each
    # must be seen (a measure that drops the B-side or the vertical bonds
    # would miss some).  Regime: the noise has weight where the symmetric
    # tensor is exactly 0.
    for site in ((0, 0), (1, 0)):
        for field in denv[site]._fields:
            T = getattr(denv[site], field)
            sym_data = np.asarray(T.todense())
            noise = np.asarray(jax.random.normal(jax.random.PRNGKey(7), sym_data.shape))
            off = sym_data == 0
            assert off.any() and np.linalg.norm(noise[off]) > 0
            bad = dict(denv)
            bad[site] = denv[site]._replace(
                **{field: DenseTensor(jnp.asarray(sym_data + 0.3 * noise), T.indices)}
            )
            v = charge_violation(Ad, Bd, bad)
            assert v > 1e-6, (site, field, v)  # symmetric: < 1e-12
