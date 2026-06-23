"""iPEPS square-lattice Heisenberg scaling / perf showcase.

Orchestrator + per-cell worker in one file. Run modes:

    # full sweep (orchestrator): launches one subprocess per cell
    uv run python examples/showcase_heisenberg_scaling.py

    # single cell (worker; normally invoked by the orchestrator):
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/showcase_heisenberg_scaling.py --cell \
        --D 2 --chi 16 --n-devices 1 --gs-num-steps 5 --out /tmp/cell.json

Pure helpers (Cell/build_grid/...) import only stdlib; jax/tenax imports live
inside run_cell so CUDA_VISIBLE_DEVICES (set by the parent) takes effect before
the child initialises a JAX backend, and so the helper unit tests stay fast.
"""

# NB: deliberately NO ``from __future__ import annotations`` — it stringizes the
# dataclass field annotations, and the test path-loads this module without
# registering it in ``sys.modules``, so CPython's ``dataclasses._is_type`` lookup
# (``sys.modules[cls.__module__]``) returns None and ``@dataclass`` crashes. Do
# not re-add it on this module while the importlib path-loader test exists.
from dataclasses import dataclass
from pathlib import Path

REFERENCE_E = -0.669437  # Sandvik QMC, square-lattice spin-1/2 Heisenberg AFM


@dataclass(frozen=True)
class Cell:
    """One point of the scaling grid (one benchmark run)."""

    D: int
    chi: int
    n_devices: int        # 1 or 4
    gs_num_steps: int     # small => metrics-only; large => anchor (trusted energy)
    is_anchor: bool       # True => converged-energy anchor cell


def build_grid(D_list, chi_ramp, device_counts, anchors, metrics_steps, anchor_steps):
    """Enumerate all cells in deterministic order.

    ``D_list``, ``chi_ramp``, ``device_counts`` are flat lists; ``anchors`` is a
    list of ``(D, chi)`` tuples. Emits a metrics cell per ``(n_devices, D, chi)``,
    then an anchor cell per ``(n_devices, (D, chi))``."""
    cells = []
    for n in device_counts:
        for D in D_list:
            for chi in chi_ramp:
                cells.append(Cell(D, chi, n, metrics_steps, is_anchor=False))
    for n in device_counts:
        for (D, chi) in anchors:
            cells.append(Cell(D, chi, n, anchor_steps, is_anchor=True))
    return cells


def cell_result_path(results_dir, cell):
    """Per-cell JSON path. Anchor and metrics cells at the same (D,chi,n) get
    distinct files so resume never confuses them."""
    kind = "anchor" if cell.is_anchor else "metrics"
    return str(Path(results_dir) / f"D{cell.D}_chi{cell.chi}_n{cell.n_devices}_{kind}.json")


def should_stop_row(result):
    """Stop ramping chi for a (D, n_devices) row once a cell OOMs or errors."""
    return bool(result.get("oom") or result.get("error"))


def cell_to_argv_env(cell, results_dir, python_exe, script_path, base_env):
    """Map a Cell to (argv, env) for its subprocess. Pins CUDA_VISIBLE_DEVICES
    (device 0 for 1-GPU; 0..n-1 for n-GPU — never the display GPU at index 4)
    and disables XLA preallocation so peak memory is real."""
    out = cell_result_path(results_dir, cell)
    argv = [
        python_exe, script_path, "--cell",
        "--D", str(cell.D),
        "--chi", str(cell.chi),
        "--n-devices", str(cell.n_devices),
        "--gs-num-steps", str(cell.gs_num_steps),
        "--out", out,
    ]
    devices = "0" if cell.n_devices == 1 else ",".join(str(i) for i in range(cell.n_devices))
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return argv, env
