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
import argparse
import json
import statistics
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


def _status(r):
    if r.get("oom"):
        return "OOM"
    if r.get("error"):
        return "ERR"
    return "ok"


def _fmt(x, spec):
    return format(x, spec) if isinstance(x, (int, float)) else "-"


def results_to_csv_rows(results):
    """Flatten results to stable-keyed dicts for CSV export."""
    keys = ["D", "chi", "n_devices", "is_anchor", "gs_num_steps",
            "ms_per_step", "peak_gb", "E_site", "converged", "oom", "error"]
    return [{k: r.get(k) for k in keys} for r in results]


def results_to_markdown(results):
    """Render a scaling table grouped by device count, sorted by (D, chi)."""
    lines = []
    for n in sorted({r["n_devices"] for r in results}):
        lines.append(f"\n### {n}-GPU\n")
        lines.append("| D | χ | kind | status | ms/step | peak GB | E/site | dE_ref | conv |")
        lines.append("|---|---|------|--------|---------|---------|--------|--------|------|")
        rows = sorted([r for r in results if r["n_devices"] == n],
                      key=lambda r: (r["D"], r["chi"], r.get("is_anchor", False)))
        for r in rows:
            kind = "anchor" if r.get("is_anchor") else "metrics"
            e = r.get("E_site")
            d_ref = (e - REFERENCE_E) if isinstance(e, (int, float)) else None
            lines.append(
                f"| {r['D']} | {r['chi']} | {kind} | {_status(r)} | "
                f"{_fmt(r.get('ms_per_step'), '.1f')} | {_fmt(r.get('peak_gb'), '.2f')} | "
                f"{_fmt(e, '.6f')} | {_fmt(d_ref, '+.2e')} | "
                f"{'Y' if r.get('converged') else 'N'} |"
            )
    return "\n".join(lines)


def _peak_gb():
    # import inside the try: _peak_gb is called from run_cell's except handler,
    # and if the original failure was a JAX backend/init error, an unguarded
    # `import jax` here would re-raise and crash the worker (breaking the
    # record-and-resume contract).
    try:
        import jax

        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return None


def run_cell(D, chi, n_devices, gs_num_steps):
    """Run ONE cell and return a result dict. One faithful entry point:
    optimize_gs_ad(return_history=True). ms_per_step = median of warm
    step_times (compile excluded). Anchor cells (large gs_num_steps) yield a
    trusted converged E_site."""
    result = {
        "D": D, "chi": chi, "n_devices": n_devices, "gs_num_steps": gs_num_steps,
        # Standalone fallback only: the orchestrator overwrites is_anchor from the
        # Cell. 40 is the midpoint of the grid's metrics_steps=5 / anchor_steps=80;
        # keep those two far from 40 if you change them.
        "is_anchor": gs_num_steps >= 40,
        "ms_per_step": None, "peak_gb": None, "E_site": None,
        "converged": False, "jit_compile_time": None, "oom": False, "error": None,
    }
    try:
        import jax  # noqa: F401  (import after CUDA_VISIBLE_DEVICES is set)

        from tenax.algorithms.ipeps import heisenberg_gate, sublattice_rotate_gate
        from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
        from tenax.algorithms.ipeps_optimize import optimize_gs_ad

        mesh = None
        if n_devices > 1:
            from tenax.algorithms.ctm_sharding import build_ctm_mesh
            mesh = build_ctm_mesh()

        gate = sublattice_rotate_gate(heisenberg_gate())
        config = iPEPSConfig(
            max_bond_dim=D,
            ctm=CTMConfig(
                chi=chi, max_iter=100, conv_tol=1e-8,
                projector_method="svd", forward_gauge="sigma",
                device_mesh=mesh,
            ),
            unit_cell="1x1",
            gs_recipe="1x1",
            gs_optimizer="lbfgs",
            gs_implicit_ad=True,
            gs_num_steps=gs_num_steps,
            su_init=True,
            return_history=True,
            gs_verbose=False,
        )
        _, _, E_gs, history = optimize_gs_ad(gate, None, config)

        step_times = history.get("step_times") or []
        warm = step_times[1:] if len(step_times) > 1 else step_times
        if warm:
            result["ms_per_step"] = 1000.0 * statistics.median(warm)
        result["E_site"] = float(E_gs)
        result["converged"] = bool(history.get("converged"))
        result["jit_compile_time"] = (
            float(history["jit_compile_time"]) if history.get("jit_compile_time") is not None else None
        )
        result["peak_gb"] = _peak_gb()
    except Exception as e:  # noqa: BLE001 — record and resume, never crash the sweep
        msg = f"{type(e).__name__}: {e}"
        result["error"] = msg
        if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower():
            result["oom"] = True
        result["peak_gb"] = _peak_gb()
    return result


def _run_worker(args):
    res = run_cell(args.D, args.chi, args.n_devices, args.gs_num_steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


def _build_argparser():
    p = argparse.ArgumentParser(description="iPEPS Heisenberg scaling showcase")
    p.add_argument("--cell", action="store_true", help="run a single cell (worker mode)")
    p.add_argument("--D", type=int)
    p.add_argument("--chi", type=int)
    p.add_argument("--n-devices", dest="n_devices", type=int, default=1)
    p.add_argument("--gs-num-steps", dest="gs_num_steps", type=int, default=5)
    p.add_argument("--out", type=str)
    p.add_argument("--results-dir", dest="results_dir", type=str,
                   default="examples/showcase_results")
    return p


if __name__ == "__main__":
    _args = _build_argparser().parse_args()
    if _args.cell:
        _run_worker(_args)
    else:
        main(_args)  # noqa: F821 — main defined in the orchestrator section (Task 7)
