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
import csv
import json
import os
import subprocess
import sys
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
    ]
    if cell.is_anchor:
        argv.append("--is-anchor")
    argv += ["--out", out]
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
            "ms_per_step", "peak_gb", "E_site", "converged", "corner_rank",
            "oom", "error"]
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
            # #747: never publish an energy that is not entitled to be read as
            # one.  A cell qualifies only if the optimizer converged AND the CTM
            # environment did not collapse to a rank-1 corner.  The original
            # sweep printed six-decimal energies for cells that were neither --
            # 6 optimizer steps, converged=false, and (on gs_recipe="1x1") a
            # chi_eff=1 mean-field boundary -- and those numbers were then read
            # as a physics result.  Render them as "n/c" instead.
            collapsed = (r.get("corner_rank") or 2) <= 1
            if not r.get("converged") or collapsed:
                e = None
            d_ref = (e - REFERENCE_E) if isinstance(e, (int, float)) else None
            lines.append(
                f"| {r['D']} | {r['chi']} | {kind} | {_status(r)} | "
                f"{_fmt(r.get('ms_per_step'), '.1f')} | {_fmt(r.get('peak_gb'), '.2f')} | "
                f"{_fmt(e, '.6f')} | {_fmt(d_ref, '+.2e')} | "  # n/c when unusable
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


def run_cell(D, chi, n_devices, gs_num_steps, is_anchor):
    """Run ONE cell and return a result dict via optimize_gs_ad(return_history).

    Two profiles (the showcase has two distinct goals):
    - metrics (is_anchor=False): a CHEAP fixed-step optimizer (adam, no line
      search / no metric preconditioning) over a FIXED-work CTM
      (min_iter=max_iter=20, conv_tol=0 => exactly 20 sweeps/step). Each step is
      one deterministic forward-CTM + one implicit-AD backward, so the per-step
      cost is trajectory-independent. The worker records the raw ``step_times``
      and emits a rough live ``ms_per_step = min(step_times[1:])``; the robust
      number for the findings is ``median(step_times[1:])`` recomputed post-hoc
      (see showcase_analyze.py). Energy is NOT trusted here.
    - anchor (is_anchor=True): the ACCURATE optimizer (L-BFGS + line search) over
      a bounded CTM (max_iter=40, conv_tol=1e-6) for a best-effort converged
      E_site. Expensive (~25x the fixed-step cost), so only a few small cells.

    implicit AD requires forward_gauge="phase" (+ projector_method in {svd,qr},
    ctm_conv_method="elementwise"); "sigma" is rejected.
    """
    result = {
        "D": D, "chi": chi, "n_devices": n_devices, "gs_num_steps": gs_num_steps,
        "is_anchor": is_anchor,
        "ms_per_step": None, "step_times": None, "peak_gb": None, "E_site": None,
        "corner_rank": None,
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

        if is_anchor:
            # Accurate-but-BOUNDED: full L-BFGS + line search for a best-effort
            # converged energy, but a cheaper CTM (40 sweeps / 1e-6) than a
            # publication run — the L-BFGS x line-search x CTM-to-1e-8 x precond
            # stack costs ~60s/step even at D2 chi16 and blows a 30-min budget.
            ctm = CTMConfig(chi=chi, max_iter=40, conv_tol=1e-6,
                            projector_method="svd", forward_gauge="phase",
                            device_mesh=mesh)
            opt_kwargs = dict(gs_optimizer="lbfgs", gs_metric_precond=False)
        else:
            # FIXED CTM work (min_iter=max_iter, conv_tol=0 => exactly N sweeps
            # every step) so per-step cost is deterministic and trajectory-
            # independent — a clean ~chi^2 D^6 scaling signal, not optimization
            # noise. Energy is not trusted for metrics cells anyway.
            ctm = CTMConfig(chi=chi, max_iter=20, min_iter=20, conv_tol=0.0,
                            projector_method="svd", forward_gauge="phase",
                            device_mesh=mesh)
            opt_kwargs = dict(gs_optimizer="adam", gs_learning_rate=1e-2,
                              gs_line_search=False, gs_metric_precond=False)

        gate = sublattice_rotate_gate(heisenberg_gate())
        config = iPEPSConfig(
            max_bond_dim=D,
            ctm=ctm,
            unit_cell="1x1",
            gs_recipe="1x1",
            gs_implicit_ad=True,
            gs_num_steps=gs_num_steps,
            su_init=True,
            return_history=True,
            gs_verbose=False,
            **opt_kwargs,
        )
        _A_opt, envs, E_gs, history = optimize_gs_ad(gate, None, config)

        # #747: this driver runs gs_recipe="1x1", whose corner-pair projector
        # collapses the environment to rank-1 corners -- a chi_eff=1 mean-field
        # boundary whose energy does not respond to chi.  Every energy this
        # sweep has ever recorded was measured that way.  Record the rank so a
        # reader can tell, and warn loudly at run time.
        try:
            from tenax.algorithms._ctm_diagnostics import check_ctm_env

            env = envs[(0, 0)] if isinstance(envs, dict) else envs
            result["corner_rank"] = check_ctm_env(
                env, context=f"D={D} chi={chi} n={n_devices} "
                             f"gs_recipe={config.gs_recipe}"
            )
        except Exception:  # noqa: BLE001 — a diagnostic must never fail a cell
            pass

        step_times = history.get("step_times") or []
        result["step_times"] = [float(x) for x in step_times]
        # Steady-state per-step = MIN of the warm steps (drop step 0's initial
        # compile). At chi>=48 XLA re-autotunes on the first 1-2 warm steps too
        # (those spike to ~compile time); min robustly ignores all warmup
        # recompiles, leaving the pure-compute per-step cost. (median caught the
        # recompile spikes -> a 16.7s artifact at D2 chi48 vs ~3s true.)
        warm = step_times[1:] if len(step_times) > 1 else step_times
        if warm:
            result["ms_per_step"] = 1000.0 * min(warm)
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
    res = run_cell(args.D, args.chi, args.n_devices, args.gs_num_steps, args.is_anchor)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


def _build_argparser():
    p = argparse.ArgumentParser(description="iPEPS Heisenberg scaling showcase")
    p.add_argument("--cell", action="store_true", help="run a single cell (worker mode)")
    p.add_argument("--is-anchor", dest="is_anchor", action="store_true",
                   help="anchor cell: accurate optimizer for a trusted energy")
    p.add_argument("--D", type=int)
    p.add_argument("--chi", type=int)
    p.add_argument("--n-devices", dest="n_devices", type=int, default=1)
    p.add_argument("--gs-num-steps", dest="gs_num_steps", type=int, default=5)
    p.add_argument("--out", type=str)
    p.add_argument("--results-dir", dest="results_dir", type=str,
                   default="examples/showcase_results")
    return p


def make_plots(results, outdir):
    """Write the showcase plots. Returns the list of PNG paths written.
    Best-effort: cells without a metric are skipped, not errored."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if not r.get("oom") and not r.get("error")]
    written = []

    # Plot 1: ms/step vs chi per D (single-GPU scaling curves).
    fig, ax = plt.subplots()
    plotted = False
    for D in sorted({r["D"] for r in ok if r["n_devices"] == 1}):
        pts = sorted((r["chi"], r["ms_per_step"]) for r in ok
                     if r["D"] == D and r["n_devices"] == 1 and r.get("ms_per_step") is not None)
        if pts:
            ax.plot(*zip(*pts), marker="o", label=f"D={D}")
            plotted = True
    if plotted:
        ax.set_xlabel("χ")
        ax.set_ylabel("ms / optimizer step")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("Per-step cost vs χ (1 GPU)")
        p = outdir / "ms_per_step_vs_chi.png"
        fig.savefig(p, dpi=120)
        written.append(str(p))
    plt.close(fig)

    # Plot 2: peak GB vs chi, 1-GPU vs 4-GPU overlay across every (D, n_devices).
    fig, ax = plt.subplots()
    plotted = False
    for n in sorted({r["n_devices"] for r in ok}):
        for D in sorted({r["D"] for r in ok if r["n_devices"] == n}):
            pts = sorted((r["chi"], r["peak_gb"]) for r in ok
                         if r["D"] == D and r["n_devices"] == n and r.get("peak_gb") is not None)
            if pts:
                ax.plot(*zip(*pts), marker="s", label=f"D={D}, {n}-GPU")
                plotted = True
    if plotted:
        ax.set_xlabel("χ")
        ax.set_ylabel("per-device peak GB")
        ax.legend(fontsize="small")
        ax.set_title("Peak memory: 1-GPU vs 4-GPU")
        p = outdir / "peak_gb_vs_chi.png"
        fig.savefig(p, dpi=120)
        written.append(str(p))
    plt.close(fig)

    # Plot 3: anchor E/site vs chi with the QMC reference line.
    anchors = [r for r in ok if r.get("is_anchor") and r.get("E_site") is not None]
    if anchors:
        fig, ax = plt.subplots()
        for D in sorted({r["D"] for r in anchors}):
            pts = sorted((r["chi"], r["E_site"]) for r in anchors if r["D"] == D)
            ax.plot(*zip(*pts), marker="o", label=f"D={D}")
        ax.axhline(REFERENCE_E, ls="--", color="k", label=f"QMC {REFERENCE_E}")
        ax.set_xlabel("χ")
        ax.set_ylabel("E / site")
        ax.legend()
        ax.set_title("Anchor energies vs QMC reference")
        p = outdir / "energy_vs_chi.png"
        fig.savefig(p, dpi=120)
        written.append(str(p))
        plt.close(fig)

    return written


# Default sweep envelope. Metrics cells use the cheap fixed-step profile across
# the (D, chi) grid x {1,4} GPU; anchors use the accurate profile on a few small
# 1-GPU cells only (energy needs no device comparison).
#
# These 80 GB A100s have so much memory that the chi ramp would never OOM at
# tractable D (peak ~0.17 GB at D3 chi32), so the ramp is bounded by a per-cell
# WALL-CLOCK timeout instead: a cell that exceeds it is recorded as a timeout
# error, which stops that (D, n_devices) row (cost is monotone in chi). This
# makes "how far does each D get under a fixed per-cell time budget" the scaling
# story, and hard-bounds total runtime.
DEFAULT_D_LIST = [2, 3, 4]
DEFAULT_CHI_RAMP = [16, 24, 32, 48, 64, 96, 128]
DEFAULT_DEVICE_COUNTS = [1, 4]
DEFAULT_ANCHOR_DEVICE_COUNTS = [1]
DEFAULT_ANCHORS = [(2, 16), (2, 32)]
DEFAULT_METRICS_STEPS = 6
DEFAULT_ANCHOR_STEPS = 30
DEFAULT_CELL_TIMEOUT_S = 600
DEFAULT_ANCHOR_TIMEOUT_S = 1800


def _load_or_run_cell(cell, results_dir, timeout_s):
    """Resume: if a result JSON exists, load it; else launch the worker
    subprocess (bounded by timeout_s) and load what it wrote. Always returns a
    result dict with is_anchor annotated from the Cell (so the reporter groups
    correctly). A timeout is recorded as an error so the row stops."""
    path = Path(cell_result_path(results_dir, cell))
    if path.exists():
        res = json.loads(path.read_text())
    else:
        argv, env = cell_to_argv_env(
            cell, results_dir=results_dir, python_exe=sys.executable,
            script_path=str(Path(__file__).resolve()), base_env=dict(os.environ))
        print(f"[run] {argv[-1]}", flush=True)
        timed_out = False
        try:
            subprocess.run(argv, env=env, check=False, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        if path.exists():
            res = json.loads(path.read_text())
        else:
            err = f"timeout after {timeout_s}s" if timed_out else "worker produced no result file"
            res = {"D": cell.D, "chi": cell.chi, "n_devices": cell.n_devices,
                   "gs_num_steps": cell.gs_num_steps, "is_anchor": cell.is_anchor,
                   "oom": False, "error": err,
                   "ms_per_step": None, "peak_gb": None, "E_site": None,
                   "converged": False}
            path.write_text(json.dumps(res, indent=2))
    res["is_anchor"] = cell.is_anchor
    return res


def main(args):
    """Run the full sweep: per-cell subprocesses (resume + OOM-aware chi ramp),
    then write the table, CSV, and plots."""
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    anchor_cells = [c for c in build_grid(
        [], [], DEFAULT_ANCHOR_DEVICE_COUNTS, DEFAULT_ANCHORS,
        DEFAULT_METRICS_STEPS, DEFAULT_ANCHOR_STEPS) if c.is_anchor]

    results = []
    # Metrics: ramp chi ascending per (n_devices, D) row; stop the row on OOM/err.
    for n in DEFAULT_DEVICE_COUNTS:
        for D in DEFAULT_D_LIST:
            for chi in DEFAULT_CHI_RAMP:
                cell = Cell(D, chi, n, DEFAULT_METRICS_STEPS, is_anchor=False)
                res = _load_or_run_cell(cell, results_dir, DEFAULT_CELL_TIMEOUT_S)
                results.append(res)
                if should_stop_row(res):
                    print(f"[stop] row n={n} D={D} stopped at chi={chi} "
                          f"({_status(res)})", flush=True)
                    break
    # Anchors (specific cells; run regardless of the metrics ramp).
    for cell in anchor_cells:
        results.append(_load_or_run_cell(cell, results_dir, DEFAULT_ANCHOR_TIMEOUT_S))

    # Aggregate.
    md = results_to_markdown(results)
    (Path(results_dir) / "scaling_table.md").write_text(md)
    rows = results_to_csv_rows(results)
    if rows:
        with open(Path(results_dir) / "scaling_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    try:
        make_plots(results, results_dir)
    except Exception as e:  # noqa: BLE001 — plotting is best-effort
        print(f"[warn] plotting failed: {e}", flush=True)
    print(md)
    print(f"\n[done] wrote {results_dir}/scaling_table.md, scaling_results.csv, *.png")


if __name__ == "__main__":
    _args = _build_argparser().parse_args()
    if _args.cell:
        _run_worker(_args)
    else:
        main(_args)
