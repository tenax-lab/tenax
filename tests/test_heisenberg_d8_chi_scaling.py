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


# A non-A100 box (the #747 re-runs move to other hardware): H100s plus the same
# kind of small display GPU the A100-only guard exists to avoid.
_SMI_H100 = (
    "0, NVIDIA H100 80GB HBM3, 12, 0\n"
    "1, NVIDIA H100 80GB HBM3, 40, 0\n"
    "2, NVIDIA DGX Display, 7, 0\n"
)


def test_select_free_a100s_refuses_non_a100_by_default(monkeypatch):
    """Default stays strict: the vendor-string filter is what stops a run from
    landing on the origin box's display GPU.

    Clears the opt-out explicitly rather than trusting the ambient environment,
    so a variable leaked by another test cannot make this pass or fail for the
    wrong reason -- which is exactly what happened once (see the isolation
    fixture in ``test_heisenberg_d4_chi_scaling.py``).
    """
    monkeypatch.delenv("TENAX_ALLOW_NON_A100", raising=False)
    rows = d8._parse_nvidia_smi(_SMI_H100)
    with pytest.raises(RuntimeError):
        d8.select_free_a100s(rows, 1)


def test_select_free_a100s_honours_the_opt_out_argument():
    rows = d8._parse_nvidia_smi(_SMI_H100)
    assert d8.select_free_a100s(rows, 2, allow_non_a100=True) == [0, 1]


def test_select_free_a100s_honours_the_opt_out_env_var(monkeypatch):
    """`--allow-non-a100` reaches the selector through the environment, since
    the orchestrator pins GPUs in a spawned worker that inherits env, not args."""
    rows = d8._parse_nvidia_smi(_SMI_H100)
    monkeypatch.setenv("TENAX_ALLOW_NON_A100", "1")
    assert d8.select_free_a100s(rows, 2) == [0, 1]


def test_opt_out_still_never_picks_the_display_gpu():
    """Relaxing the A100 requirement must not relax the display-GPU exclusion —
    that is the failure the guard was built for."""
    rows = d8._parse_nvidia_smi(_SMI_H100)
    assert 2 not in d8.select_free_a100s(rows, 2, allow_non_a100=True)
    with pytest.raises(RuntimeError):
        d8.select_free_a100s(rows, 3, allow_non_a100=True)


def test_opt_out_still_excludes_busy_devices():
    rows = d8._parse_nvidia_smi(
        "0, NVIDIA H100 80GB HBM3, 12, 0\n1, NVIDIA H100 80GB HBM3, 9000, 97\n"
    )
    assert d8.select_free_a100s(rows, 1, allow_non_a100=True) == [0]
    with pytest.raises(RuntimeError):
        d8.select_free_a100s(rows, 2, allow_non_a100=True)


def test_d8_parser_accepts_allow_non_a100():
    """The handover doc points operators at this flag; before this change the
    D=8 driver rejected it with argparse exit 2."""
    args = d8._build_argparser().parse_args(["--path", "split", "--allow-non-a100"])
    assert args.allow_non_a100 is True
    assert d8._build_argparser().parse_args(["--path", "split"]).allow_non_a100 is False


def test_allow_non_a100_flag_reaches_the_environment(monkeypatch):
    """The selector runs inside spawned workers, so the flag has to travel as
    an env var; a worker started directly with --cell never reaches main()."""
    monkeypatch.delenv("TENAX_ALLOW_NON_A100", raising=False)
    args = d8._build_argparser().parse_args(["--path", "split", "--allow-non-a100"])
    assert d8._apply_device_opt_out(args) is True
    import os

    assert os.environ["TENAX_ALLOW_NON_A100"] == "1"
    # and the selector, called with no explicit argument, now honours it
    rows = d8._parse_nvidia_smi(_SMI_H100)
    assert d8.select_free_a100s(rows, 1) == [0]


def test_allow_non_a100_absent_leaves_environment_untouched(monkeypatch):
    monkeypatch.delenv("TENAX_ALLOW_NON_A100", raising=False)
    args = d8._build_argparser().parse_args(["--path", "split"])
    assert d8._apply_device_opt_out(args) is False
    import os

    assert "TENAX_ALLOW_NON_A100" not in os.environ


def test_parse_nvidia_smi_skips_malformed_and_na_lines():
    text = (
        "0, NVIDIA A100-SXM4-80GB, 56, 0\n"
        "garbage line with too few fields\n"
        "1, NVIDIA A100-SXM4-80GB, [N/A], [N/A]\n"  # MIG/vGPU state
        "2, NVIDIA A100-SXM4-80GB, 20, 0\n"
    )
    rows = d8._parse_nvidia_smi(text)
    assert [r[0] for r in rows] == [0, 2]  # malformed + N/A rows skipped


def test_build_grid_both_paths_dense_rows_then_split_n1():
    cells = d8.build_grid(
        chi_ladder=[64, 96], device_counts=[1, 2], paths=["dense", "split"]
    )
    quads = [(c.D, c.chi, c.n_devices, c.path) for c in cells]
    # dense: device-major chi-minor over {1,2}; split: n=1 only, appended last
    assert quads == [
        (8, 64, 1, "dense"),
        (8, 96, 1, "dense"),
        (8, 64, 2, "dense"),
        (8, 96, 2, "dense"),
        (8, 64, 1, "split"),
        (8, 96, 1, "split"),
    ]


def test_build_grid_split_only_ignores_multi_device():
    cells = d8.build_grid(chi_ladder=[128], device_counts=[1, 2], paths=["split"])
    assert [(c.chi, c.n_devices, c.path) for c in cells] == [(128, 1, "split")]
    assert all(c.D == 8 for c in cells)


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
    assert args.chi_ladder == "64,96,112,128,192,256,320,384,448"
    assert args.device_counts == "1,2"
    assert args.path == "both"
    assert args.outdir == "runs/d8_chi_scaling"


def test_argparser_path_choices_validated():
    for p in ("dense", "split", "both"):
        assert d8._build_argparser().parse_args(["--path", p]).path == p
    with pytest.raises(SystemExit):
        d8._build_argparser().parse_args(["--path", "bogus"])


def test_smoke_args_shrink_the_run():
    args = d8._build_argparser().parse_args(["--smoke"])
    d8._apply_smoke(args)
    assert args.outdir.endswith("_smoke")
    assert args.chi_ladder == "8,12"
    assert args.device_counts == "1"
    assert args.path == "both"  # smoke still exercises both paths
    assert args.imaginary_steps <= 20


def test_load_or_run_scan_returns_cached_cell(tmp_path):
    cell = d8.Cell8(D=8, chi=64, n_devices=1, path="split")
    path = d8._cell_path(str(tmp_path), cell)
    d4._atomic_write_text(
        path, '{"D": 8, "chi": 64, "n_devices": 1, "path": "split", "oom": false}'
    )
    # cached file present -> no subprocess launched, returns the parsed dict
    res = d8._load_or_run_scan(cell, str(tmp_path), timeout_s=1)
    assert res["chi"] == 64 and res["path"] == "split" and res["oom"] is False


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
    assert (
        d8._launch(["x", "--cell", "--phase", "scan"], 2, timeout_s=1, gpu_wait_s=0)
        is None
    )


def test_load_or_run_scan_returns_none_and_writes_no_poison(monkeypatch, tmp_path):
    # _launch signalling 'no GPUs' (None) must propagate as None and leave NO
    # cell JSON behind, so a resume retries the cell rather than trusting a
    # transient-infra failure as a completed result.
    monkeypatch.setattr(d8, "_launch", lambda *a, **k: None)
    cell = d8.Cell8(D=8, chi=128, n_devices=2, path="dense")
    res = d8._load_or_run_scan(cell, str(tmp_path), timeout_s=1, gpu_wait_s=0)
    assert res is None
    assert not pathlib.Path(d8._cell_path(str(tmp_path), cell)).exists()


def test_cell8_has_path_and_distinct_paths_dont_collide(tmp_path):
    dense = d8.Cell8(D=8, chi=128, n_devices=1, path="dense")
    split = d8.Cell8(D=8, chi=128, n_devices=1, path="split")
    assert dense.path == "dense" and split.path == "split"
    pd = d8._cell_path(str(tmp_path), dense)
    ps = d8._cell_path(str(tmp_path), split)
    assert pd != ps
    assert pd.endswith("D8_chi128_n1_dense.json")
    assert ps.endswith("D8_chi128_n1_split.json")


def test_aggregate8_writes_both_path_sections(tmp_path):
    results = [
        {
            "D": 8,
            "chi": 96,
            "n_devices": 1,
            "path": "dense",
            "E_site": -0.605,
            "err_vs_qmc": 0.064,
            "total_s": 785.0,
            "n_sweeps": 14,
            "ms_per_sweep": 56000.0,
            "peak_gb": 59.7,
            "converged": False,
            "oom": False,
            "error": None,
        },
        {
            "D": 8,
            "chi": 128,
            "n_devices": 1,
            "path": "dense",
            "E_site": None,
            "err_vs_qmc": None,
            "total_s": None,
            "n_sweeps": None,
            "ms_per_sweep": None,
            "peak_gb": 72.8,
            "converged": False,
            "oom": True,
            "error": "RESOURCE_EXHAUSTED",
        },
        {
            "D": 8,
            "chi": 128,
            "n_devices": 1,
            "path": "split",
            "E_site": -0.6005,
            "err_vs_qmc": 0.0689,
            "total_s": 14.2,
            "n_sweeps": 8,
            "ms_per_sweep": 1775.0,
            "peak_gb": 6.59,
            "converged": True,
            "oom": False,
            "error": None,
        },
    ]
    d8._aggregate8(results, str(tmp_path))
    md = (tmp_path / "convergence.md").read_text()
    assert "Dense" in md and "Split" in md
    # comparison table lists the shared χ rows
    assert "128" in md
    csv_text = (tmp_path / "results.csv").read_text()
    assert "path" in csv_text.splitlines()[0]
