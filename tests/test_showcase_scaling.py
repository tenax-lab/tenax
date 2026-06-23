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
