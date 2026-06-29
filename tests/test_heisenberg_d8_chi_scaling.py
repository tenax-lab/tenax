"""Unit tests for the pure orchestration helpers of the D=8 χ-scaling driver.
Path-loaded (the example is not an importable package) so these tests stay
jax-free and fast."""

import importlib.util
import pathlib

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "examples"
    / "heisenberg_d8_chi_scaling.py"
)
_spec = importlib.util.spec_from_file_location("heisenberg_d8_chi_scaling", _PATH)
d8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d8)
d4 = d8.d4  # the d8 module path-loads the D=4 sibling for its reused helpers


# nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu
# --format=csv,noheader,nounits  (this box: A100s at 0,1,2,4; display at 3)
_SMI = (
    "0, NVIDIA A100-SXM4-80GB, 56, 0\n"
    "1, NVIDIA A100-SXM4-80GB, 20, 0\n"
    "2, NVIDIA A100-SXM4-80GB, 20, 0\n"
    "3, NVIDIA DGX Display, 7, 0\n"
    "4, NVIDIA A100-SXM4-80GB, 4066, 91\n"
)


def test_parse_nvidia_smi_reads_index_name_mem_util():
    rows = d8._parse_nvidia_smi(_SMI)
    assert rows[0] == (0, "NVIDIA A100-SXM4-80GB", 56, 0)
    assert rows[4] == (4, "NVIDIA A100-SXM4-80GB", 4066, 91)
    assert len(rows) == 5


def test_select_free_a100s_picks_most_idle_first():
    rows = d8._parse_nvidia_smi(_SMI)
    # idle A100s are 0 (56), 1 (20), 2 (20); sort by (mem, index) -> 1,2,0
    assert d8.select_free_a100s(rows, 1) == [1]
    assert d8.select_free_a100s(rows, 2) == [1, 2]
    assert d8.select_free_a100s(rows, 3) == [1, 2, 0]


def test_select_free_a100s_never_picks_the_display_gpu():
    rows = d8._parse_nvidia_smi(_SMI)
    assert 3 not in d8.select_free_a100s(rows, 3)


def test_select_free_a100s_excludes_busy_a100():
    rows = d8._parse_nvidia_smi(_SMI)
    # index 4 is busy (4066 MiB, 91%) -> never selected, and only 3 are free
    assert 4 not in d8.select_free_a100s(rows, 3)
    with pytest.raises(RuntimeError):
        d8.select_free_a100s(rows, 4)


def test_parse_nvidia_smi_skips_malformed_and_na_lines():
    text = (
        "0, NVIDIA A100-SXM4-80GB, 56, 0\n"
        "garbage line with too few fields\n"
        "1, NVIDIA A100-SXM4-80GB, [N/A], [N/A]\n"  # MIG/vGPU state
        "2, NVIDIA A100-SXM4-80GB, 20, 0\n"
    )
    rows = d8._parse_nvidia_smi(text)
    assert [r[0] for r in rows] == [0, 2]  # malformed + N/A rows skipped


def test_build_grid_is_device_major_chi_minor_at_D8():
    cells = d8.build_grid(chi_ladder=[64, 96], device_counts=[1, 2])
    assert [(c.D, c.chi, c.n_devices) for c in cells] == [
        (8, 64, 1), (8, 96, 1), (8, 64, 2), (8, 96, 2),
    ]


def test_build_grid_uses_D8():
    cells = d8.build_grid(chi_ladder=[128], device_counts=[1])
    assert cells[0].D == 8


def test_worker_env_pins_idle_a100s_and_disables_prealloc(monkeypatch):
    monkeypatch.setattr(d8, "cuda_visible_for", lambda n: "1,2")
    env = d8._worker_env(2, {"PATH": "/usr/bin"})
    assert env["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    # cuda_async avoids BFC fragmentation OOMs at large χ; autotuning off drops
    # the autotuner's transient workspace (see _worker_env docstring).
    assert env["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"
    assert "--xla_gpu_autotune_level=0" in env["XLA_FLAGS"]
    assert env["PATH"] == "/usr/bin"  # base env preserved


def test_worker_env_appends_to_existing_xla_flags(monkeypatch):
    monkeypatch.setattr(d8, "cuda_visible_for", lambda n: "0")
    env = d8._worker_env(1, {"XLA_FLAGS": "--foo=1"})
    assert "--foo=1" in env["XLA_FLAGS"]
    assert "--xla_gpu_autotune_level=0" in env["XLA_FLAGS"]


def test_argparser_defaults_target_the_wall():
    args = d8._build_argparser().parse_args([])
    assert args.chi_ladder == "64,96,128,160,192,224,256"
    assert args.device_counts == "1,2"
    assert args.outdir == "runs/d8_chi_scaling"


def test_smoke_args_shrink_the_run():
    args = d8._build_argparser().parse_args(["--smoke"])
    d8._apply_smoke(args)
    assert args.outdir.endswith("_smoke")
    assert args.chi_ladder == "8,12"
    assert args.device_counts == "1"
    assert args.imaginary_steps <= 20


def test_load_or_run_scan_returns_cached_cell(tmp_path):
    cell = d4.Cell(D=8, chi=64, n_devices=1)
    path = d4.cell_result_path(str(tmp_path), cell)
    d4._atomic_write_text(path, '{"D": 8, "chi": 64, "n_devices": 1, "oom": false}')
    # cached file present -> no subprocess launched, returns the parsed dict
    res = d8._load_or_run_scan(cell, str(tmp_path), timeout_s=1)
    assert res["chi"] == 64 and res["oom"] is False


def test_wait_for_free_a100s_gives_up_past_deadline(monkeypatch):
    # A shared box with no idle A100s: free_a100_indices keeps raising. With a
    # zero-length wait window the helper gives up immediately (returns False)
    # instead of blocking or propagating the RuntimeError.
    def _none_free(n):
        raise RuntimeError("no idle A100s")

    monkeypatch.setattr(d8, "free_a100_indices", _none_free)
    assert d8._wait_for_free_a100s(2, gpu_wait_s=0, poll_s=0) is False


def test_wait_for_free_a100s_true_when_available(monkeypatch):
    monkeypatch.setattr(d8, "free_a100_indices", lambda n: [0, 1][:n])
    assert d8._wait_for_free_a100s(2, gpu_wait_s=0, poll_s=0) is True


def test_launch_returns_none_when_no_idle_gpus(monkeypatch):
    # When no idle A100s free up in time, _launch must NOT raise; it returns None
    # so the orchestrator can stop gracefully (the bug that crashed a real run).
    monkeypatch.setattr(d8, "_wait_for_free_a100s", lambda n, w, poll_s=30: False)
    assert d8._launch(["x", "--cell", "--phase", "scan"], 2, timeout_s=1,
                      gpu_wait_s=0) is None


def test_load_or_run_scan_returns_none_and_writes_no_poison(monkeypatch, tmp_path):
    # _launch signalling 'no GPUs' (None) must propagate as None and leave NO
    # cell JSON behind, so a resume retries the cell rather than trusting a
    # transient-infra failure as a completed result.
    monkeypatch.setattr(d8, "_launch", lambda *a, **k: None)
    cell = d4.Cell(D=8, chi=128, n_devices=2)
    res = d8._load_or_run_scan(cell, str(tmp_path), timeout_s=1, gpu_wait_s=0)
    assert res is None
    assert not pathlib.Path(d4.cell_result_path(str(tmp_path), cell)).exists()
