"""Provenance gate of examples/heisenberg_d3_chi_convergence.py (#938 round 2).

The example's resume workflow trusts three artifacts blindly (A_opt.pkl,
chi_convergence.json, and the optimizer checkpoint dir), so an outdir written
before the migration off the collapsed "1x1" recipe (#747/#911) would silently
publish or blend pre-migration numbers. ``check_provenance`` must stamp fresh
dirs, accept its own stamp, and refuse everything else.

The example is loaded by path (it is not an importable package); its module
level imports jax + tenax, so this stays a light integration point, not a run.
"""

import importlib.util
import json
import pathlib

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "examples"
    / "heisenberg_d3_chi_convergence.py"
)
_spec = importlib.util.spec_from_file_location("heisenberg_d3_chi_convergence", _PATH)
chi_study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chi_study)


def test_fresh_outdir_is_stamped_and_accepted(tmp_path):
    outdir = str(tmp_path)
    chi_study.check_provenance(outdir)
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["gs_recipe"] == chi_study.RECIPE
    # Idempotent: a second call (the resume case) accepts its own stamp.
    chi_study.check_provenance(outdir)


@pytest.mark.parametrize("artifact", ["A_opt.pkl", "chi_convergence.json", "ckpt_opt"])
def test_unstamped_artifacts_are_refused(tmp_path, artifact):
    """Any pre-#938 artifact without provenance.json must abort the run —
    each of the three is individually enough to poison a resumed study."""
    target = tmp_path / artifact
    if artifact == "ckpt_opt":
        target.mkdir()
    else:
        target.write_bytes(b"stale")
    with pytest.raises(SystemExit, match="predate the #938 migration"):
        chi_study.check_provenance(str(tmp_path))


def test_mismatched_stamp_is_refused(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"gs_recipe": "1x1"}))
    (tmp_path / "A_opt.pkl").write_bytes(b"stale")
    with pytest.raises(SystemExit, match="mix regimes"):
        chi_study.check_provenance(str(tmp_path))


def test_config_recipe_matches_the_stamp():
    """The stamp is only meaningful if it equals what the optimizer actually
    runs; pin the coupling so neither drifts alone."""
    cfg = chi_study.make_opt_config(
        chi=4, ckpt_path="/dev/null", num_steps=1, resume=False, probe_max_iter=None
    )
    assert cfg.gs_recipe == chi_study.RECIPE == "2x2"


def test_main_wires_the_gate_before_any_artifact_reuse():
    """Direct tests prove the gate works; this pins that main() actually calls
    it, and does so before optimize_state touches A_opt.pkl / the checkpoint.
    (Running main() itself would cost a full optimization, so assert on the
    source order instead — deleting the call must fail a test.)"""
    body = _PATH.read_text().split("def main(")[1]
    gate = body.index("check_provenance(outdir)")
    first_reuse = body.index("optimize_state(")
    assert gate < first_reuse
