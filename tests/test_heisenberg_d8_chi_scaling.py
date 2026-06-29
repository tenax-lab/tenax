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
