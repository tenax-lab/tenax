"""Unit tests for the pure orchestration helpers of the Heisenberg scaling
showcase. The example file is loaded by path (it is not an importable package)
so these tests stay jax-free and fast."""

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parent.parent / "examples" / "showcase_heisenberg_scaling.py"
_spec = importlib.util.spec_from_file_location("showcase_heisenberg_scaling", _PATH)
showcase = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(showcase)


def test_build_grid_enumerates_metrics_and_anchor_cells():
    cells = showcase.build_grid(
        D_list=[2, 3],
        chi_ramp=[16, 32],
        device_counts=[1, 4],
        anchors=[(2, 32)],
        metrics_steps=5,
        anchor_steps=80,
    )
    metrics = [c for c in cells if not c.is_anchor]
    anchors = [c for c in cells if c.is_anchor]
    # 2 device_counts * 2 D * 2 chi metrics cells:
    assert len(metrics) == 8
    # 2 device_counts * 1 anchor:
    assert len(anchors) == 2
    # spot-check a metrics cell carries the right fields:
    c = next(c for c in metrics if c.D == 3 and c.chi == 16 and c.n_devices == 4)
    assert c.gs_num_steps == 5 and c.is_anchor is False
    # anchor cells carry the anchor step budget:
    assert all(c.gs_num_steps == 80 for c in anchors)


def test_cell_result_path_is_unique_and_descriptive():
    metrics = showcase.Cell(D=3, chi=48, n_devices=4, gs_num_steps=5, is_anchor=False)
    anchor = showcase.Cell(D=3, chi=48, n_devices=4, gs_num_steps=80, is_anchor=True)
    pm = showcase.cell_result_path("results", metrics)
    pa = showcase.cell_result_path("results", anchor)
    assert pm.endswith("D3_chi48_n4_metrics.json")
    assert pa.endswith("D3_chi48_n4_anchor.json")
    assert pm != pa  # anchor and metrics at the same (D,chi,n) must not collide


def test_should_stop_row_on_oom_or_error_only():
    assert showcase.should_stop_row({"oom": True, "error": None}) is True
    assert showcase.should_stop_row({"oom": False, "error": "Boom"}) is True
    assert showcase.should_stop_row({"oom": False, "error": None}) is False


def test_cell_to_argv_env_sets_devices_and_flags():
    cell1 = showcase.Cell(D=4, chi=64, n_devices=1, gs_num_steps=5, is_anchor=False)
    cell4 = showcase.Cell(D=4, chi=64, n_devices=4, gs_num_steps=80, is_anchor=True)
    base_env = {"PATH": "/usr/bin", "HOME": "/home/u"}

    argv1, env1 = showcase.cell_to_argv_env(
        cell1, results_dir="results", python_exe="python",
        script_path="examples/showcase_heisenberg_scaling.py", base_env=base_env)
    argv4, env4 = showcase.cell_to_argv_env(
        cell4, results_dir="results", python_exe="python",
        script_path="examples/showcase_heisenberg_scaling.py", base_env=base_env)

    # single-GPU cell pins device 0; 4-GPU cell pins 0,1,2,3 (NOT the display GPU)
    assert env1["CUDA_VISIBLE_DEVICES"] == "0"
    assert env4["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    # preallocation must be off so peak_bytes_in_use is meaningful
    assert env1["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    # base env is preserved, not replaced
    assert env1["PATH"] == "/usr/bin"
    # argv carries the worker flag, the cell params, and the right out path
    assert "--cell" in argv1
    assert "--D" in argv1 and "4" in argv1
    assert argv1[-1].endswith("D4_chi64_n1_metrics.json")
    assert argv4[-1].endswith("D4_chi64_n4_anchor.json")
