"""End-to-end smoke for benchmarks.varipeps_compare on the cheapest possible point.

Exercises run_tenax.py via subprocess with VARIPEPS_COMPARE_TEST_FAST=1, which
swaps the production implicit-AD + L-BFGS path for explicit-AD + Adam.  This
keeps the JIT compile cost under a minute so the smoke runs in CI; the
accumulator code in the history hook is path-agnostic, so the JSON schema
contract is still validated.

variPEPS subprocess coverage is deferred to the manual benchmark run
(``python -m benchmarks.varipeps_compare.compare``) in Task 9 — variPEPS
itself only runs an implicit-AD path with no equivalent fast knob, so a
unit-test smoke would pay the full multi-minute JIT cost on every CI run.
"""

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.varipeps_compare.payload import save_payload
from benchmarks.varipeps_compare.su_init import (
    build_sublattice_rotated_gate,
    make_init,
)


@pytest.mark.algorithm
def test_smoke_run_tenax_single_site_d2_chi4_test_fast(tmp_path):
    """Tenax runner end-to-end: payload → run_tenax CLI → JSON with right schema.

    Uses VARIPEPS_COMPARE_TEST_FAST=1 to dodge implicit-AD JIT cost.
    """
    init = make_init(path="single_site", D=2, seed=0)
    gate = build_sublattice_rotated_gate()
    payload = tmp_path / "payload.npz"
    save_payload(
        payload,
        init=init,
        gate=gate,
        meta={"path": "single_site", "D": 2, "chi": 4},
    )

    out = tmp_path / "tenax.json"
    env = {**os.environ, "VARIPEPS_COMPARE_TEST_FAST": "1"}
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.varipeps_compare.run_tenax",
        "--payload",
        str(payload),
        "--path",
        "single_site",
        "--D",
        "2",
        "--chi",
        "4",
        "--tol",
        "1e-2",
        "--max-steps",
        "3",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, timeout=600, env=env)

    data = json.loads(out.read_text())
    for key in (
        "lib",
        "path",
        "D",
        "chi",
        "energy_history",
        "step_times",
        "jit_compile_time",
        "final_energy",
        "num_steps",
        "converged",
        "device",
        "lib_version",
    ):
        assert key in data, f"missing {key}"
    assert data["lib"] == "tenax"
    assert data["path"] == "single_site"
    assert data["D"] == 2 and data["chi"] == 4
    assert len(data["energy_history"]) == data["num_steps"]
    assert data["num_steps"] >= 1
    # Energy can be anything from random init + 3 explicit-AD steps;
    # we only validate the schema and that the run completed.
    assert isinstance(data["final_energy"], float)


_HAVE_VARIPEPS = importlib.util.find_spec("varipeps") is not None


@pytest.mark.skip(
    reason=(
        "variPEPS subprocess smoke deferred to Task 9 manual benchmark "
        "(`python -m benchmarks.varipeps_compare.compare`) — variPEPS only "
        "runs implicit-AD with no fast-test override; cold JIT compile "
        "exceeds reasonable unit-test timeouts."
    )
)
@pytest.mark.skipif(not _HAVE_VARIPEPS, reason="varipeps not installed")
def test_smoke_run_varipeps_single_site_d2_chi4(tmp_path):
    """Placeholder for variPEPS-side smoke; see decorator reason."""
