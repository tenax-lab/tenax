"""iPEPS D=4 square-lattice Heisenberg AFM: χ-convergence + multi-GPU performance.

Orchestrator + per-cell worker in one file.

    # full sweep (orchestrator): optimize once, then scan χ × {1,2,4} GPU
    uv run python examples/heisenberg_d4_chi_scaling.py --outdir runs/d4_chi_scaling

    # quick validation (tiny D=4 run end-to-end)
    uv run python examples/heisenberg_d4_chi_scaling.py --smoke

    # single cell (worker; normally invoked by the orchestrator):
    CUDA_VISIBLE_DEVICES=0 uv run python examples/heisenberg_d4_chi_scaling.py \
        --cell --phase scan --chi 32 --n-devices 1 --outdir runs/d4_chi_scaling \
        --out /tmp/cell.json

Pure helpers import only stdlib; jax/tenax imports live inside the worker so the
parent's CUDA_VISIBLE_DEVICES takes effect before the child initialises a JAX
backend, and so the helper unit tests stay fast and jax-free.
"""

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REFERENCE_E = -0.669437  # Sandvik QMC, square-lattice spin-1/2 Heisenberg AFM

# This box: A100-SXM4-80GB at CUDA/PCI indices 0,1,2,4. Index 3 is a 4 GB DGX
# Display GPU and must never be selected. Drivers set CUDA_DEVICE_ORDER=PCI_BUS_ID
# so these indices match nvidia-smi deterministically.
A100_INDICES = [0, 1, 2, 4]


def cuda_visible_for(n_devices):
    """CUDA_VISIBLE_DEVICES string for an n-GPU run: the first n A100 indices.

    Never emits the display GPU (index 3). Raises ValueError if n is out of the
    1..len(A100_INDICES) range."""
    if not 1 <= n_devices <= len(A100_INDICES):
        raise ValueError(
            f"n_devices must be 1..{len(A100_INDICES)}, got {n_devices}"
        )
    return ",".join(str(i) for i in A100_INDICES[:n_devices])


D = 4  # fixed bond dimension for this driver


@dataclass(frozen=True)
class Cell:
    """One scan cell: the fixed D=4 state contracted at (chi, n_devices)."""

    D: int
    chi: int
    n_devices: int


def build_grid(chi_ladder, device_counts):
    """Scan cells in device-major, chi-minor order (one row per n_devices)."""
    return [
        Cell(D=D, chi=chi, n_devices=n)
        for n in device_counts
        for chi in chi_ladder
    ]


def cell_result_path(results_dir, cell):
    """Per-cell JSON path, unique per (D, chi, n_devices)."""
    return str(
        Path(results_dir) / f"D{cell.D}_chi{cell.chi}_n{cell.n_devices}.json"
    )


def should_stop_row(result):
    """Stop ramping χ for a given n_devices row once a cell OOMs or errors
    (CTM cost is monotone in χ)."""
    return bool(result.get("oom") or result.get("error"))


def _atomic_write_bytes(path, data):
    """Write bytes via a temp file + os.replace so a kill mid-write never leaves
    a truncated file on disk (matches _checkpoint.save_checkpoint). Critical for
    resume: a half-written A_opt.pkl / cell JSON would otherwise be trusted by the
    existence-based short-circuits and corrupt every downstream load."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _atomic_write_text(path, text):
    """Atomic UTF-8 text write (see _atomic_write_bytes)."""
    _atomic_write_bytes(path, text.encode("utf-8"))


def _read_json_or_none(path):
    """Load a JSON file, or None if missing or truncated/corrupt — so a
    half-written resume artifact is treated as 'not done' (re-run) rather than
    crashing the whole sweep on json.loads."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _status(r):
    if r.get("oom"):
        return "OOM"
    if r.get("error"):
        return "ERR"
    return "ok"


def _fmt(x, spec):
    return format(x, spec) if isinstance(x, (int, float)) else "-"


def _e_by_chi(results):
    """First valid-energy result per χ (E is device-independent), χ-sorted."""
    by_chi = {}
    for r in results:
        if r.get("E_site") is None:
            continue
        by_chi.setdefault(r["chi"], r)
    return [by_chi[c] for c in sorted(by_chi)]


def results_to_convergence_md(results, d_label=4):
    """E/site vs χ on the fixed optimized state (device-independent).

    ``d_label`` only sets the title's bond-dimension tag (default 4), so the
    D=8 sibling driver can reuse this formatter with the correct label."""
    lines = [
        f"### Convergence: E/site vs χ (D={d_label} square-lattice Heisenberg AFM)",
        "",
        f"QMC reference E/site = {REFERENCE_E}",
        "",
        "| χ | E/site | err_vs_QMC | sweeps | conv |",
        "|---|--------|------------|--------|------|",
    ]
    for r in _e_by_chi(results):
        e = r["E_site"]
        lines.append(
            f"| {r['chi']} | {e:.6f} | {e - REFERENCE_E:+.2e} | "
            f"{_fmt(r.get('n_sweeps'), 'd')} | "
            f"{'Y' if r.get('converged') else 'N'} |"
        )
    return "\n".join(lines)


def _ms_baseline(results):
    """ms/sweep of the 1-GPU run, keyed by χ — the speedup denominator."""
    return {
        r["chi"]: r["ms_per_sweep"]
        for r in results
        if r["n_devices"] == 1 and r.get("ms_per_sweep") is not None
    }


def results_to_performance_md(results):
    """Per-sweep CTM cost + peak memory vs χ, grouped by device count, with
    speedup against the 1-GPU baseline at the same χ."""
    base = _ms_baseline(results)
    lines = ["### Performance: per-sweep CTM cost & memory vs χ × n_devices"]
    for n in sorted({r["n_devices"] for r in results}):
        lines += [
            "",
            f"#### {n}-GPU",
            "",
            "| χ | status | ms/sweep | sweeps | peak GB | speedup vs 1-GPU |",
            "|---|--------|----------|--------|---------|------------------|",
        ]
        for r in sorted(
            (r for r in results if r["n_devices"] == n), key=lambda r: r["chi"]
        ):
            ms = r.get("ms_per_sweep")
            b = base.get(r["chi"])
            sp = (b / ms) if (ms and b) else None
            lines.append(
                f"| {r['chi']} | {_status(r)} | {_fmt(ms, '.2f')} | "
                f"{_fmt(r.get('n_sweeps'), 'd')} | {_fmt(r.get('peak_gb'), '.3f')} | "
                f"{_fmt(sp, '.2f')} |"
            )
    return "\n".join(lines)


def results_to_csv_rows(results):
    keys = [
        "D", "chi", "n_devices", "E_site", "err_vs_qmc", "ms_per_sweep",
        "n_sweeps", "peak_gb", "converged", "oom", "error",
    ]
    return [{k: r.get(k) for k in keys} for r in results]


def _peak_gb():
    """Per-device peak GB, or None. Import inside try: this is also called from
    the worker's except handler, where an unguarded `import jax` could re-raise
    a backend-init failure and break the record-and-resume contract."""
    try:
        import jax

        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return None


def _assert_only_a100s():
    """Refuse to run if any visible device is not an 80 GB A100 (e.g. the 4 GB
    DGX Display GPU). Backstops a wrong CUDA_VISIBLE_DEVICES / index-order
    mismatch so a run can never silently land on the display GPU."""
    import jax

    bad = []
    for dev in jax.devices():
        try:
            limit = dev.memory_stats().get("bytes_limit", 0)
        except Exception:
            limit = 0
        kind = getattr(dev, "device_kind", "")
        ok = limit > 40e9 or ("A100" in kind and "Display" not in kind)
        if not ok:
            bad.append(f"{dev} kind={kind!r} bytes_limit={limit}")
    if bad:
        raise RuntimeError(
            "refusing to run: non-A100/display GPU visible: " + "; ".join(bad)
        )


def _build_mesh(n_devices):
    """An n-device GSPMD mesh, or None for single-device. Asserts the visible
    devices are A100s first."""
    _assert_only_a100s()
    if n_devices <= 1:
        return None
    from tenax.algorithms.ctm_sharding import build_ctm_mesh

    return build_ctm_mesh()  # over all visible devices (== the pinned A100s)


def optimize_once(outdir, chi_opt, opt_steps, n_devices, probe_max_iter=15):
    """Optimize the D=4 state once at χ_opt; cache the optimized tensor to
    `<outdir>/A_opt.pkl`. Resumes from its gs checkpoint; if A_opt.pkl already
    exists, returns immediately."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from tenax import (
        CTMConfig,
        heisenberg_gate,
        iPEPSConfig,
        optimize_gs_ad,
        sublattice_rotate_gate,
    )

    tensor_path = os.path.join(outdir, "A_opt.pkl")
    if os.path.exists(tensor_path):
        print(f"[opt] cached {tensor_path}; skipping optimization", flush=True)
        return tensor_path

    mesh = _build_mesh(n_devices)
    ckpt = os.path.join(outdir, "ckpt_opt", "ckpt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    resume = os.path.exists(os.path.join(ckpt, "ckpt.last.pkl"))
    probe = None if probe_max_iter <= 0 else probe_max_iter

    cfg = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=200,
        dt=0.05,
        ctm=CTMConfig(
            chi=chi_opt,
            max_iter=100,
            conv_tol=1e-8,
            projector_method="svd",   # implicit AD requires svd/qr (not eigh)
            forward_gauge="phase",    # implicit AD requires phase gauge
            probe_max_iter=probe,     # #503 cap on HZ line-search CTM probes
            device_mesh=mesh,         # #632 GSPMD sharding when n_devices > 1
        ),
        unit_cell="1x1",
        gs_c4v=True,                  # removes bond-gauge freedom -> stable backward
        gs_implicit_ad=True,          # variational (true expectation value)
        gs_recipe="1x1",
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_metric_precond=True,
        gs_num_steps=opt_steps,
        gs_conv_criterion="grad_norm",
        gs_energy_floor=REFERENCE_E,  # reject sub-GS CTM-artifact spikes (#298)
        gs_grad_spike_ratio=5.0,      # roll back >5x gradient blowups (#524)
        gs_verbose=True,
        gs_log_interval=1,
        su_init=True,
        gs_checkpoint_path=ckpt,
        gs_checkpoint_every=2,
        gs_resume=resume,
    )
    gate = sublattice_rotate_gate(heisenberg_gate())
    print(
        f"[opt] D=4 optimize at χ={chi_opt} (resume={resume}, {opt_steps} steps, "
        f"n_devices={n_devices})",
        flush=True,
    )
    t0 = time.perf_counter()
    A_opt, _env, E = optimize_gs_ad(gate, None, cfg)
    print(
        f"[opt] done in {time.perf_counter() - t0:.0f}s; in-loop E_best={float(E):.6f}",
        flush=True,
    )
    # Gather to host before pickling. Under a device_mesh A_opt is a sharded
    # jax.Array whose sharding references Device objects that pickle can't
    # serialise ("cannot pickle 'Device' object"). jax.device_get → numpy leaves
    # makes the saved tensor device-agnostic; each scan worker re-shards on load.
    A_opt_host = jax.device_get(A_opt)
    _atomic_write_bytes(
        tensor_path, pickle.dumps(A_opt_host, protocol=pickle.HIGHEST_PROTOCOL)
    )
    return tensor_path


def scan_cell(tensor_path, chi, n_devices):
    """Converge forward CTM at χ on the fixed optimized state; return E/site +
    per-sweep timing + peak memory. Record-and-resume safe."""
    result = {
        "D": D, "chi": chi, "n_devices": n_devices,
        "E_site": None, "err_vs_qmc": None, "total_s": None, "n_sweeps": None,
        "ms_per_sweep": None, "peak_gb": None, "converged": False,
        "oom": False, "error": None,
    }
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        from tenax import CTMConfig, compute_energy_ctm_tensor, heisenberg_gate, \
            sublattice_rotate_gate
        from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
        from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
        from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs

        mesh = _build_mesh(n_devices)  # also runs the A100-only guard
        with open(tensor_path, "rb") as fh:
            A_opt = pickle.load(fh)
        H = sublattice_rotate_gate(heisenberg_gate())

        cfg = CTMConfig(
            chi=chi, max_iter=200, conv_tol=1e-10,
            projector_method="svd", forward_gauge="phase", device_mesh=mesh,
        )
        # ctm_converge_kwargs forwards device_mesh but emits no `recipe`, so the
        # scan uses python_loop_ctm_converge's default CTM recipe — intentionally
        # independent of the optimizer's gs_recipe="1x1" (mirrors the validated
        # d3 backbone heisenberg_d3_chi_convergence.scan_chi; energy parity holds).
        kwargs = ctm_converge_kwargs(cfg)  # forwards device_mesh

        # Warm-up: compile the χ-specific @jit step (reused via the process
        # cache), so the timed converge measures pure per-sweep compute.
        warm_envs, _ = python_loop_ctm_converge(
            {(0, 0): A_opt}, SINGLE_SITE_NEIGHBORS, **kwargs
        )
        jax.block_until_ready(warm_envs[(0, 0)])

        t0 = time.perf_counter()
        envs, info = python_loop_ctm_converge(
            {(0, 0): A_opt}, SINGLE_SITE_NEIGHBORS, **kwargs
        )
        jax.block_until_ready(envs[(0, 0)])
        total_s = time.perf_counter() - t0

        env = envs[(0, 0)]
        if mesh is not None:  # gather the tiny env to device 0 for energy eval
            env = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, jax.devices()[0]), env
            )
        E = float(compute_energy_ctm_tensor(A_opt, env, H, 2))
        sweeps = int(info.iterations)

        result.update(
            E_site=E, err_vs_qmc=E - REFERENCE_E, total_s=float(total_s),
            n_sweeps=sweeps, ms_per_sweep=1000.0 * total_s / max(sweeps, 1),
            converged=bool(info.converged), peak_gb=_peak_gb(),
        )
    except Exception as e:  # noqa: BLE001 — record and resume, never crash the sweep
        msg = f"{type(e).__name__}: {e}"
        result["error"] = msg
        if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower():
            result["oom"] = True
        result["peak_gb"] = _peak_gb()
    return result


def _run_worker(args):
    """Worker entry: run one phase, write its result JSON, echo it."""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "optimize":
        try:
            optimize_once(
                args.outdir, args.chi_opt, args.opt_steps, args.n_devices,
                probe_max_iter=args.probe_max_iter,
            )
            res = {"phase": "optimize", "ok": True, "error": None}
        except Exception as e:  # noqa: BLE001
            res = {"phase": "optimize", "ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        tensor_path = os.path.join(args.outdir, "A_opt.pkl")
        res = scan_cell(tensor_path, args.chi, args.n_devices)
    _atomic_write_text(args.out, json.dumps(res, indent=2))
    print(json.dumps(res))


def _build_argparser():
    p = argparse.ArgumentParser(description="iPEPS D=4 Heisenberg χ-scaling benchmark")
    p.add_argument("--cell", action="store_true", help="worker mode: run one phase")
    p.add_argument("--phase", choices=["optimize", "scan"], default="scan")
    p.add_argument("--chi", type=int, help="scan χ (worker scan phase)")
    p.add_argument("--n-devices", dest="n_devices", type=int, default=1)
    p.add_argument("--out", type=str, help="worker result JSON path")
    # shared / orchestrator:
    p.add_argument("--outdir", default="runs/d4_chi_scaling")
    p.add_argument("--smoke", action="store_true",
                   help="quick validation: tiny χ_opt, few steps, short scan")
    p.add_argument("--chi-opt", dest="chi_opt", type=int, default=32)
    p.add_argument("--opt-steps", dest="opt_steps", type=int, default=100)
    p.add_argument("--probe-max-iter", dest="probe_max_iter", type=int, default=15)
    p.add_argument("--opt-devices", dest="opt_devices", type=int, default=1,
                   help="GPUs for the one-time optimization. Default 1: multi-GPU "
                        "optimize is blocked by sharded gs-checkpoint pickling "
                        "(_checkpoint.save_checkpoint can't pickle mesh-sharded "
                        "jax.Arrays); the single-GPU path keeps crash-resilient "
                        "checkpointing. The χ-scan carries the multi-GPU story.")
    p.add_argument("--chi-ladder", dest="chi_ladder", type=str,
                   default="16,24,32,48,64,96,128")
    p.add_argument("--device-counts", dest="device_counts", type=str, default="1,2,4")
    p.add_argument("--cell-timeout-s", dest="cell_timeout_s", type=int, default=1800)
    return p


def _worker_env(n_devices, base_env):
    """Subprocess env: pin the n A100s deterministically, no XLA preallocation."""
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_for(n_devices)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # CUDA indices == nvidia-smi indices
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # so peak_gb is real
    return env


def _launch(argv, n_devices, timeout_s):
    """Run a worker subprocess; return True if it exited within the timeout."""
    env = _worker_env(n_devices, dict(os.environ))
    print(f"[run] {' '.join(argv[argv.index('--cell'):])}", flush=True)
    try:
        subprocess.run(argv, env=env, check=False, timeout=timeout_s)
        return True
    except subprocess.TimeoutExpired:
        return False


def _optimize_phase(outdir, chi_opt, opt_steps, opt_devices, probe_max_iter):
    """Run the one-time optimization in a subprocess pinned to opt_devices."""
    if os.path.exists(os.path.join(outdir, "A_opt.pkl")):
        print("[opt] A_opt.pkl present; optimization skipped", flush=True)
        return
    out = os.path.join(outdir, "optimize_status.json")
    argv = [
        sys.executable, str(Path(__file__).resolve()), "--cell",
        "--phase", "optimize", "--outdir", outdir,
        "--chi-opt", str(chi_opt), "--opt-steps", str(opt_steps),
        "--probe-max-iter", str(probe_max_iter),
        "--n-devices", str(opt_devices), "--out", out,
    ]
    # Optimization can be long; allow generous wall-clock (resume-safe anyway).
    _launch(argv, opt_devices, timeout_s=None)


def _load_or_run_scan(cell, outdir, timeout_s):
    """Resume: load an existing cell JSON, else launch the scan worker and load
    what it wrote. A timeout/no-file is recorded as an error so the row stops."""
    path = Path(cell_result_path(outdir, cell))
    cached = _read_json_or_none(path) if path.exists() else None
    if cached is not None:
        return cached
    argv = [
        sys.executable, str(Path(__file__).resolve()), "--cell",
        "--phase", "scan", "--outdir", outdir, "--chi", str(cell.chi),
        "--n-devices", str(cell.n_devices), "--out", str(path),
    ]
    ok = _launch(argv, cell.n_devices, timeout_s)
    loaded = _read_json_or_none(path)
    if loaded is not None:
        return loaded
    res = {
        "D": cell.D, "chi": cell.chi, "n_devices": cell.n_devices,
        "E_site": None, "err_vs_qmc": None, "ms_per_sweep": None,
        "n_sweeps": None, "peak_gb": None, "converged": False, "oom": False,
        "error": ("timeout" if not ok else "worker produced no result file"),
    }
    _atomic_write_text(path, json.dumps(res, indent=2))
    return res


def make_plots(results, outdir, d_label=4):
    """Best-effort PNGs: E vs χ (with QMC line), ms/sweep vs χ per n, speedup vs
    χ per n, peak GB vs χ per n. Returns the list of paths written.

    ``d_label`` only sets the bond-dimension tag in plot titles (default 4)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    ok = [r for r in results if not r.get("oom") and not r.get("error")]
    written = []

    conv = _e_by_chi(ok)
    if conv:
        fig, ax = plt.subplots()
        ax.plot([r["chi"] for r in conv], [r["E_site"] for r in conv], marker="o")
        ax.axhline(REFERENCE_E, ls="--", color="k", label=f"QMC {REFERENCE_E}")
        ax.set_xlabel("χ"); ax.set_ylabel("E / site")
        ax.set_title(f"D={d_label} convergence: E/site vs χ"); ax.legend()
        p = outdir / "convergence_E_vs_chi.png"
        fig.savefig(p, dpi=120); written.append(str(p)); plt.close(fig)

    base = _ms_baseline(ok)
    for metric, ylabel, fname, logy in [
        ("ms_per_sweep", "ms / CTM sweep", "perf_ms_per_sweep_vs_chi.png", True),
        ("peak_gb", "per-device peak GB", "perf_peak_gb_vs_chi.png", False),
    ]:
        fig, ax = plt.subplots(); plotted = False
        for n in sorted({r["n_devices"] for r in ok}):
            pts = sorted((r["chi"], r[metric]) for r in ok
                         if r["n_devices"] == n and r.get(metric) is not None)
            if pts:
                ax.plot(*zip(*pts), marker="o", label=f"{n}-GPU"); plotted = True
        if plotted:
            ax.set_xlabel("χ"); ax.set_ylabel(ylabel)
            if logy:
                ax.set_yscale("log")
            ax.legend(); ax.set_title(f"D={d_label} {ylabel} vs χ")
            p = outdir / fname
            fig.savefig(p, dpi=120); written.append(str(p))
        plt.close(fig)

    fig, ax = plt.subplots(); plotted = False
    for n in sorted({r["n_devices"] for r in ok if r["n_devices"] > 1}):
        pts = sorted((r["chi"], base[r["chi"]] / r["ms_per_sweep"]) for r in ok
                     if r["n_devices"] == n and r.get("ms_per_sweep")
                     and base.get(r["chi"]))
        if pts:
            ax.plot(*zip(*pts), marker="o", label=f"{n}-GPU"); plotted = True
    if plotted:
        ax.axhline(1.0, ls=":", color="k")
        ax.set_xlabel("χ"); ax.set_ylabel("speedup vs 1-GPU")
        ax.legend(); ax.set_title(f"D={d_label} multi-GPU speedup vs χ")
        p = outdir / "perf_speedup_vs_chi.png"
        fig.savefig(p, dpi=120); written.append(str(p))
    plt.close(fig)
    return written


def _aggregate(results, outdir, d_label=4):
    conv_md = results_to_convergence_md(results, d_label=d_label)
    perf_md = results_to_performance_md(results)
    (Path(outdir) / "convergence.md").write_text(conv_md)
    (Path(outdir) / "performance.md").write_text(perf_md)
    rows = results_to_csv_rows(results)
    if rows:
        with open(Path(outdir) / "results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    try:
        make_plots(results, outdir, d_label=d_label)
    except Exception as e:  # noqa: BLE001 — plotting is best-effort
        print(f"[warn] plotting failed: {e}", flush=True)
    print(conv_md); print(); print(perf_md)
    print(f"\n[done] wrote {outdir}/convergence.md, performance.md, results.csv, *.png")


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    chi_ladder = [int(x) for x in args.chi_ladder.split(",")]
    device_counts = [int(x) for x in args.device_counts.split(",")]

    # Phase 1: optimize once (pinned to opt_devices GPUs).
    _optimize_phase(outdir, args.chi_opt, args.opt_steps, args.opt_devices,
                    args.probe_max_iter)
    if not os.path.exists(os.path.join(outdir, "A_opt.pkl")):
        print("[abort] optimization produced no A_opt.pkl; see "
              f"{outdir}/optimize_status.json", flush=True)
        return

    # Phase 2: scan χ per device row; stop a row on OOM/error/timeout.
    results = []
    for n in device_counts:
        for chi in chi_ladder:
            res = _load_or_run_scan(Cell(D=D, chi=chi, n_devices=n), outdir,
                                    args.cell_timeout_s)
            results.append(res)
            if should_stop_row(res):
                print(f"[stop] n={n} row stopped at χ={chi} ({_status(res)})",
                      flush=True)
                break

    _aggregate(results, outdir)


if __name__ == "__main__":
    _args = _build_argparser().parse_args()
    if _args.smoke:
        _args.outdir = _args.outdir + "_smoke"
        _args.chi_opt = 8
        _args.opt_steps = 6
        _args.opt_devices = 1
        _args.chi_ladder = "8,12"
        _args.device_counts = "1,2"
        _args.cell_timeout_s = 1200
    if _args.cell:
        _run_worker(_args)
    else:
        main(_args)
