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
