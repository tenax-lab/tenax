"""Unit tests for the pure orchestration helpers of the D=4 χ-scaling driver.
The example file is path-loaded (it is not an importable package) so these
tests stay jax-free and fast."""

import importlib.util
import pathlib

import pytest

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
