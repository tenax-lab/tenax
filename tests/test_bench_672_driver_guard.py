"""Guards for the #672 re-derivation driver's bookkeeping helpers.

Same shape and scope as ``test_frontier_bench_guard.py``: path-loads a driver
script and exercises its pure helpers, no JAX and no subprocess.  These are
bookkeeping guards, but the bookkeeping is what the benchmark's conclusions
rest on -- #672's headline was invalidated by a protocol difference
(``recipe=1x1``) that the recorded output could not distinguish, and every
check here is a way the re-derivation could reproduce that failure one level
down.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load("bench_672_rederivation")
analyze = _load("analyze_672_rederivation")


# --- the collapse gate must not confirm itself -----------------------------


@pytest.mark.parametrize(
    "out,timed_out,expected",
    [
        ("GATE D=10 chi=32 corner_rank=6/32", False, ("GATE_OK", 6)),
        ("GATE D=10 chi=32 corner_rank=2/32", False, ("GATE_OK", 2)),
        ("GATE D=10 chi=32 corner_rank=1/32", False, ("GATE_COLLAPSED", 1)),
        ("GATE D=10 chi=32 corner_rank=0/32", False, ("GATE_COLLAPSED", 0)),
    ],
)
def test_a_parsed_rank_classifies_on_the_rank(out, timed_out, expected):
    assert bench._classify_gate(out, timed_out) == expected


@pytest.mark.parametrize(
    "out,timed_out,expected_status",
    [
        ("path=split FAILED(RESOURCE_EXHAUSTED: out of memory)", False, "GATE_NO_RANK"),
        ("", True, "GATE_TIMEOUT"),
        ("", False, "GATE_NO_RANK"),
        ("Traceback (most recent call last):\nRuntimeError", False, "GATE_NO_RANK"),
    ],
)
def test_an_execution_failure_is_never_reported_as_a_collapse(
    out, timed_out, expected_status
):
    """The gate exists to test whether recipe=2x2 collapses to a rank-1 corner.

    Reporting a crashed, OOMed or timed-out cell as ``GATE_COLLAPSED`` would
    make every execution failure into evidence for the proposition under test,
    so the gate could only ever confirm itself.  Rank ``None`` means unmeasured.
    """
    status, rank = bench._classify_gate(out, timed_out)
    assert status == expected_status
    assert rank is None
    assert status != "GATE_COLLAPSED"


# --- cache identity --------------------------------------------------------


def test_the_cache_key_separates_runs_that_measured_different_things():
    """A knob that changes what is measured must change the cache identity.

    Otherwise re-running into the same ``--outdir`` under a different protocol
    silently reuses the old cell, which is #672's original failure mode.
    """
    base = dict(path="split", D=10, chi=32, n_dev=1, recipe="2x2", chunk=0)
    k30 = bench._cell_key(**base, max_iter=30)
    k60 = bench._cell_key(**base, max_iter=60)
    assert k30 != k60, (k30, k60)

    # ...and the knobs that already worked still do.
    assert bench._cell_key(**base, max_iter=30, autotune0=True) != k30
    assert bench._cell_key(**{**base, "recipe": "1x1"}, max_iter=30) != k30
    assert bench._cell_key(**{**base, "n_dev": 2}, max_iter=30) != k30


# --- the two scripts must agree on what a wall is --------------------------


def test_the_driver_and_analyzer_agree_on_the_wall_set():
    """They disagreed once, and the disagreement is silent.

    The analyzer counted TIMEOUT as a wall while the ladder climbed past it, so
    a timed-out cell could be followed by larger-chi cells and then reported as
    a ceiling by the very tool that considered it terminal.
    """
    assert set(bench.WALL) == set(analyze.WALL), (bench.WALL, analyze.WALL)


@pytest.mark.parametrize("status", ["TIMEOUT", "NO_OUTPUT"])
def test_non_oom_failures_stop_the_ladder(status):
    """chi is monotone, so a cell that could not run bounds every later one.

    ``NO_OUTPUT`` is the OOM-killer case -- the process dies before the driver
    prints its ``path=`` line, so it never reaches the ``OOM`` classification.
    """
    assert status in bench.WALL


# --- device labelling ------------------------------------------------------


def test_shard_on_the_split_path_is_rejected_rather_than_relabelled(monkeypatch):
    """``bench_ctm_frontier_grad`` ignores --shard on split and warns to stdout.

    The warning never reaches results.jsonl, and the analyzer keys arms on
    ``(D, path, n_devices)``, so accepting the flag would publish a split
    multi-GPU arm that was never run -- on exactly the axis #672's "split on 1
    GPU dominates dense multi-GPU" claim is measured.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench",
            "--path",
            "split",
            "--D",
            "10",
            "--chi",
            "16",
            "--shard",
            "--gpus",
            "0,1",
        ],
    )
    with pytest.raises(SystemExit) as ex:
        bench.main()
    assert ex.value.code != 0


def test_the_dense_path_still_accepts_shard(monkeypatch, tmp_path):
    """Guards against fixing the above by rejecting --shard everywhere."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench",
            "--path",
            "dense",
            "--D",
            "10",
            "--chi",
            "16",
            "--shard",
            "--gpus",
            "0,1",
            "--outdir",
            os.path.relpath(tmp_path, bench.REPO),
        ],
    )
    # Runs the ladder, but every cell shells out to the driver; stub it so the
    # test stays subprocess-free and only the argument handling is exercised.
    monkeypatch.setattr(
        bench, "run_cell", lambda **kw: {"status": "SKIP", "wall_s": 0.0}
    )
    assert bench.main() == 0
