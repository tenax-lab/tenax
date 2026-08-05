"""Unit tests for the pure orchestration helpers of the D=4 χ-scaling driver.
The example file is path-loaded (it is not an importable package) so these
tests stay jax-free and fast."""

import importlib.util
import os
import pathlib

import pytest


@pytest.fixture(autouse=True)
def _isolate_device_opt_out():
    """Keep TENAX_ALLOW_NON_A100 from leaking between tests.

    ``_apply_device_opt_out`` writes ``os.environ`` directly, and
    ``monkeypatch.delenv(..., raising=False)`` records *nothing* when the key is
    absent (pytest returns early), so it cannot undo that write. Without this
    the flag leaks into every later test in the session -- it broke the D=8
    driver's strict-default guard, which reads the same variable.
    """
    prev = os.environ.get("TENAX_ALLOW_NON_A100")
    os.environ.pop("TENAX_ALLOW_NON_A100", None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TENAX_ALLOW_NON_A100", None)
        else:
            os.environ["TENAX_ALLOW_NON_A100"] = prev


_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "examples"
    / "heisenberg_d4_chi_scaling.py"
)
_spec = importlib.util.spec_from_file_location("heisenberg_d4_chi_scaling", _PATH)
d4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d4)


def test_cuda_visible_for_maps_to_real_a100_indices():
    assert d4.cuda_visible_for(1) == "0"
    assert d4.cuda_visible_for(2) == "0,1"
    assert d4.cuda_visible_for(4) == "0,1,2,4"


def test_cuda_visible_for_never_emits_the_display_gpu():
    # Index 3 is the 4 GB DGX Display GPU — it must never appear.
    for n in (1, 2, 3, 4):
        assert "3" not in d4.cuda_visible_for(n).split(",")


def test_cuda_visible_for_rejects_out_of_range():
    with pytest.raises(ValueError):
        d4.cuda_visible_for(0)
    with pytest.raises(ValueError):
        d4.cuda_visible_for(5)


def test_build_grid_enumerates_cells_in_device_then_chi_order():
    cells = d4.build_grid(chi_ladder=[16, 32], device_counts=[1, 2, 4])
    assert len(cells) == 6
    # device-major, chi-minor ordering:
    assert (cells[0].n_devices, cells[0].chi) == (1, 16)
    assert (cells[1].n_devices, cells[1].chi) == (1, 32)
    assert (cells[2].n_devices, cells[2].chi) == (2, 16)
    assert all(c.D == 4 for c in cells)


def test_cell_result_path_is_unique_per_chi_and_device():
    a = d4.Cell(D=4, chi=32, n_devices=1)
    b = d4.Cell(D=4, chi=32, n_devices=4)
    pa = d4.cell_result_path("runs/x", a)
    pb = d4.cell_result_path("runs/x", b)
    assert pa != pb
    assert pa.endswith("D4_chi32_n1.json")
    assert pb.endswith("D4_chi32_n4.json")


def test_should_stop_row_on_oom_or_error():
    assert d4.should_stop_row({"oom": True}) is True
    assert d4.should_stop_row({"error": "timeout after 600s"}) is True
    assert d4.should_stop_row({"oom": False, "error": None}) is False


def _sample_results():
    # Two device rows over χ∈{16,32}. E identical across n (device-independent);
    # 1-GPU is the speedup baseline.
    return [
        {
            "D": 4,
            "chi": 16,
            "n_devices": 1,
            "E_site": -0.6601,
            "err_vs_qmc": 0.0093,
            "ms_per_sweep": 100.0,
            "n_sweeps": 30,
            "peak_gb": 1.0,
            "converged": True,
            "oom": False,
            "error": None,
        },
        {
            "D": 4,
            "chi": 32,
            "n_devices": 1,
            "E_site": -0.6640,
            "err_vs_qmc": 0.0054,
            "ms_per_sweep": 400.0,
            "n_sweeps": 40,
            "peak_gb": 4.0,
            "converged": True,
            "oom": False,
            "error": None,
        },
        {
            "D": 4,
            "chi": 16,
            "n_devices": 4,
            "E_site": -0.6601,
            "err_vs_qmc": 0.0093,
            "ms_per_sweep": 50.0,
            "n_sweeps": 30,
            "peak_gb": 0.3,
            "converged": True,
            "oom": False,
            "error": None,
        },
        {
            "D": 4,
            "chi": 32,
            "n_devices": 4,
            "E_site": -0.6640,
            "err_vs_qmc": 0.0054,
            "ms_per_sweep": 200.0,
            "n_sweeps": 40,
            "peak_gb": 1.1,
            "converged": True,
            "oom": False,
            "error": None,
        },
    ]


def test_convergence_table_dedups_by_chi_and_shows_qmc_error():
    md = d4.results_to_convergence_md(_sample_results())
    assert "E/site" in md and "err_vs_QMC" in md
    # one row per distinct χ (device-independent), not one per (χ, n):
    assert md.count("| 16 |") == 1
    assert md.count("| 32 |") == 1
    assert "-0.660100" in md


def test_performance_table_reports_speedup_vs_one_gpu():
    md = d4.results_to_performance_md(_sample_results())
    assert "1-GPU" in md and "4-GPU" in md
    # 4-GPU at χ=16 is 100/50 = 2.00× the 1-GPU baseline:
    assert "2.00" in md


def test_csv_rows_have_stable_keys():
    rows = d4.results_to_csv_rows(_sample_results())
    assert len(rows) == 4
    assert set(rows[0]) == {
        "D",
        "chi",
        "n_devices",
        "E_site",
        "err_vs_qmc",
        "ms_per_sweep",
        "n_sweeps",
        "peak_gb",
        "converged",
        # #780: `converged` alone is not readable — the default `elementwise`
        # criterion is gauge-dependent and cannot reach any usable tolerance,
        # so an `N` says nothing about the environment. Store the criterion
        # used and the metric it achieved alongside the flag.
        "conv_metric",
        "conv_method",
        # #747: rank(C1) per cell, so a completed sweep can be checked for the
        # rank-1 collapse after the fact rather than only at run time.
        "corner_rank",
        "oom",
        "error",
    }


def test_atomic_write_text_roundtrips_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "cell.json"
    d4._atomic_write_text(str(p), '{"a": 1}')
    assert p.read_text() == '{"a": 1}'
    assert not (tmp_path / "cell.json.tmp").exists()  # temp file was renamed away


def test_read_json_or_none_handles_missing_corrupt_and_valid(tmp_path):
    # missing -> None (treated as 'not done', re-run)
    assert d4._read_json_or_none(str(tmp_path / "nope.json")) is None
    # truncated / corrupt -> None (never crashes the sweep)
    bad = tmp_path / "bad.json"
    bad.write_text("{ truncated")
    assert d4._read_json_or_none(str(bad)) is None
    # valid -> parsed dict
    good = tmp_path / "good.json"
    good.write_text('{"k": 5}')
    assert d4._read_json_or_none(str(good)) == {"k": 5}


# --- device opt-out (#747): --allow-non-a100 must reach worker mode too ------
#
# The flag used to be published to the environment inside main(), which a
# worker started with --cell never reaches -- so `--allow-non-a100 --cell`
# was silently a no-op and _assert_only_a100s refused non-A100 hardware.
# This is the same defect fixed for the D=8 driver in #775.


class _FakeDevice:
    def __init__(self, kind, bytes_limit):
        self.device_kind = kind
        self._limit = bytes_limit

    def memory_stats(self):
        return {"bytes_limit": self._limit}

    def __str__(self):
        return f"fake({self.device_kind})"


def _install_fake_jax(monkeypatch, devices):
    """_assert_only_a100s does `import jax` inside the function, so a stub in
    sys.modules is enough to exercise it without a GPU or a real backend."""
    import sys
    import types

    fake = types.ModuleType("jax")
    fake.devices = lambda: devices
    monkeypatch.setitem(sys.modules, "jax", fake)


def test_apply_device_opt_out_publishes_the_flag(monkeypatch):
    monkeypatch.delenv("TENAX_ALLOW_NON_A100", raising=False)
    args = d4._build_argparser().parse_args(["--allow-non-a100"])
    assert d4._apply_device_opt_out(args) is True
    import os

    assert os.environ["TENAX_ALLOW_NON_A100"] == "1"


def test_apply_device_opt_out_absent_leaves_environment_untouched(monkeypatch):
    monkeypatch.delenv("TENAX_ALLOW_NON_A100", raising=False)
    args = d4._build_argparser().parse_args([])
    assert d4._apply_device_opt_out(args) is False
    import os

    assert "TENAX_ALLOW_NON_A100" not in os.environ


def test_guard_refuses_small_gpu_without_the_opt_out(monkeypatch):
    monkeypatch.delenv("TENAX_ALLOW_NON_A100", raising=False)
    _install_fake_jax(
        monkeypatch, [_FakeDevice("NVIDIA GeForce RTX 4070 Ti SUPER", 16 * 1000**3)]
    )
    with pytest.raises(RuntimeError, match="non-A100"):
        d4._assert_only_a100s()


def test_guard_passes_on_small_gpu_once_the_flag_is_applied(monkeypatch):
    """The end-to-end contract the worker path depends on: applying the parsed
    flag is sufficient to make the guard accept non-A100 hardware."""
    monkeypatch.delenv("TENAX_ALLOW_NON_A100", raising=False)
    _install_fake_jax(
        monkeypatch, [_FakeDevice("NVIDIA GeForce RTX 4070 Ti SUPER", 16 * 1000**3)]
    )
    args = d4._build_argparser().parse_args(
        ["--cell", "--phase", "scan", "--allow-non-a100"]
    )
    d4._apply_device_opt_out(args)
    d4._assert_only_a100s()  # must not raise


def test_guard_still_refuses_a_display_gpu_without_the_flag(monkeypatch):
    monkeypatch.delenv("TENAX_ALLOW_NON_A100", raising=False)
    _install_fake_jax(monkeypatch, [_FakeDevice("NVIDIA DGX Display", 4 * 1000**3)])
    with pytest.raises(RuntimeError):
        d4._assert_only_a100s()


def test_dispatch_applies_the_opt_out_before_running_the_worker(monkeypatch):
    """The regression guard: worker mode must see TENAX_ALLOW_NON_A100 already
    set. Asserting on the environment *at the moment _run_worker is called* is
    what makes deleting the _apply_device_opt_out call a test failure."""
    import os

    monkeypatch.delenv("TENAX_ALLOW_NON_A100", raising=False)
    seen = {}
    monkeypatch.setattr(
        d4,
        "_run_worker",
        lambda a: seen.update(env=os.environ.get("TENAX_ALLOW_NON_A100")),
    )
    monkeypatch.setattr(
        d4, "main", lambda a: pytest.fail("worker mode must not call main()")
    )
    args = d4._build_argparser().parse_args(
        ["--cell", "--phase", "scan", "--chi", "16", "--allow-non-a100"]
    )
    d4._dispatch(args)
    assert seen["env"] == "1"


def test_dispatch_routes_to_the_orchestrator_without_cell(monkeypatch):
    monkeypatch.delenv("TENAX_ALLOW_NON_A100", raising=False)
    called = {}
    monkeypatch.setattr(d4, "main", lambda a: called.update(main=True))
    monkeypatch.setattr(d4, "_run_worker", lambda a: pytest.fail("not worker mode"))
    d4._dispatch(d4._build_argparser().parse_args([]))
    assert called == {"main": True}


# --------------------------------------------------------------------- #
# #780: make `converged` interpretable                                    #
# --------------------------------------------------------------------- #


def test_convergence_table_reports_the_criterion_and_its_metric():
    """A bare `conv` column cannot be read without the criterion behind it.

    #780: the D=4 benchmark published `converged=false` in all seven cells
    while the environments were converged to ten digits — the default
    `elementwise` criterion is gauge-dependent and plateaus around 2.6e-01.
    Recording the achieved metric and which criterion produced it is what
    makes an `N` in that column diagnosable rather than alarming.
    """
    results = _sample_results()
    results[0].update(converged=False, conv_metric=2.55e-01, conv_method="elementwise")
    results[1].update(converged=True, conv_metric=6.46e-09, conv_method="sv")
    md = d4.results_to_convergence_md(results)

    assert "metric" in md and "crit" in md
    assert "2.55e-01" in md, "the achieved metric must be shown, not just Y/N"
    assert "6.46e-09" in md
    assert "elementwise" in md and "sv" in md


def test_convergence_table_tolerates_missing_criterion_fields():
    """Results recorded before #780 have neither key; the table must not
    crash on them (the driver is record-and-resume, so old per-cell JSONs
    are re-read verbatim)."""
    md = d4.results_to_convergence_md(_sample_results())
    assert "| 16 |" in md and "| 32 |" in md


def test_scan_ctm_config_uses_the_gauge_invariant_criterion():
    """The forward-only χ-scan must not inherit the `elementwise` default.

    #780: `CTMConfig.ctm_conv_method` defaults to `elementwise`, which is
    gauge-dependent and cannot reach any usable tolerance on a physical
    state, so `converged` is False regardless of the environment. That
    default exists for the implicit-AD path (#351, warm-start consistency);
    a forward-only scan has no such constraint and must opt out explicitly.

    Unlike the rest of this file, this test imports tenax (hence jax) --
    the point is precisely what the real CTMConfig ends up holding.
    """
    cfg = d4.scan_ctm_config(chi=32, mesh=None)
    assert cfg.ctm_conv_method == "sv"
    assert cfg.chi == 32
    # The knobs the scan does NOT deviate on, so the deviation stays minimal.
    assert cfg.projector_method == "svd"
    assert cfg.forward_gauge == "phase"
