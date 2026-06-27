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
