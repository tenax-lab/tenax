"""iPEPS D=8 square-lattice Heisenberg AFM: simple-update seed + forward-CTM
χ-scan mapping the single-GPU memory wall (dense χ²·D⁶) and the split-CTM rescue
(``--path {dense,split,both}``; multi-GPU GSPMD gives no per-device relief).

    # full run (orchestrator): SU seed once, then scan χ × {1,2} GPU × paths
    uv run python examples/heisenberg_d8_chi_scaling.py --outdir runs/d8_chi_scaling

    # quick validation (tiny end-to-end, both paths)
    uv run python examples/heisenberg_d8_chi_scaling.py --smoke

    # single cell (worker; normally invoked by the orchestrator, which pins the
    # GPU for it). For a DIRECT run, pin an idle A100 yourself (never the display
    # GPU) -- a directly-invoked worker does NOT self-pin:
    CUDA_VISIBLE_DEVICES=1 uv run python examples/heisenberg_d8_chi_scaling.py \
        --cell --phase scan --chi 64 --n-devices 1 --path split \
        --outdir runs/d8_chi_scaling --out /tmp/cell.json

When the orchestrator launches a worker it pins idle A100s for it (``_worker_env``
sets ``CUDA_VISIBLE_DEVICES`` via ``cuda_visible_for``); a worker started directly
with ``--cell`` does NOT self-pin, so set ``CUDA_VISIBLE_DEVICES`` yourself (on the
DGX the display GPU is otherwise visible and ``_build_mesh`` will refuse to run).

Pure helpers import only stdlib; jax/tenax imports live inside the worker so the
parent's CUDA_VISIBLE_DEVICES takes effect before the child initialises a JAX
backend, and so the helper unit tests stay fast and jax-free.

The D-agnostic formatting/plot/IO/mesh helpers are reused from the sibling D=4
driver (path-loaded as ``d4``) so the merged file is not modified.
"""

import argparse
import csv
import importlib.util
import json
import os
import pathlib
import pickle
import subprocess
import sys
import time
from typing import NamedTuple

# Path-load the D=4 driver to reuse its D-agnostic pure helpers. Its top level
# imports only stdlib (jax/tenax live inside functions), so this stays jax-free.
_D4_PATH = pathlib.Path(__file__).resolve().parent / "heisenberg_d4_chi_scaling.py"
_d4_spec = importlib.util.spec_from_file_location("heisenberg_d4_chi_scaling", _D4_PATH)
d4 = importlib.util.module_from_spec(_d4_spec)
_d4_spec.loader.exec_module(d4)

REFERENCE_E = d4.REFERENCE_E  # Sandvik QMC, square-lattice spin-1/2 Heisenberg AFM
D = 8  # fixed bond dimension for this driver


class Cell8(NamedTuple):
    """One D=8 scan cell: the fixed SU seed contracted at (chi, n_devices) via
    the given forward-CTM ``path`` ('dense' or 'split')."""

    D: int
    chi: int
    n_devices: int
    path: str


def _cell_path(outdir, cell):
    """Per-cell JSON path, unique per (chi, n_devices, path) so dense and split
    cells at the same chi never collide (resume-safe)."""
    return os.path.join(
        outdir, f"D{cell.D}_chi{cell.chi}_n{cell.n_devices}_{cell.path}.json"
    )


def _parse_nvidia_smi(text):
    """Parse `nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu
    --format=csv,noheader,nounits` into (index, name, mem_used_mib, util_pct)
    tuples. Lines that don't have all four fields are skipped."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, name, mem, util = parts[0], parts[1], parts[2], parts[3]
        try:
            rows.append((int(idx), name, int(float(mem)), int(float(util))))
        except ValueError:
            continue  # header row or [N/A] field -> skip, per the skip contract
    return rows


def select_free_a100s(rows, n, mem_threshold_mib=2048, util_threshold=50):
    """The n most-idle 80 GB A100 indices from parsed nvidia-smi rows.

    Idle = an A100 (never the DGX Display GPU) with memory.used and
    utilization below the thresholds. Sorted by (memory.used, index) so the
    most-idle device comes first; deterministic tiebreak by index. Raises
    RuntimeError if fewer than n idle A100s are available (so a row stops
    rather than landing on a busy or display GPU)."""
    free = [
        r for r in rows
        if "A100" in r[1] and "Display" not in r[1]
        and r[2] <= mem_threshold_mib and r[3] <= util_threshold
    ]
    free.sort(key=lambda r: (r[2], r[0]))
    if len(free) < n:
        raise RuntimeError(
            f"need {n} idle A100s, found {len(free)}: "
            + ", ".join(f"gpu{r[0]}({r[2]}MiB,{r[3]}%)" for r in free)
        )
    return [r[0] for r in free[:n]]


def free_a100_indices(n, mem_threshold_mib=2048, util_threshold=50):
    """Query nvidia-smi and return the n most-idle A100 indices."""
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return select_free_a100s(_parse_nvidia_smi(out), n, mem_threshold_mib, util_threshold)


def cuda_visible_for(n_devices):
    """CUDA_VISIBLE_DEVICES string pinning the n most-idle A100s right now."""
    return ",".join(str(i) for i in free_a100_indices(n_devices))


def build_grid(chi_ladder, device_counts, paths):
    """Scan cells as Cell8. Dense rows cross device_counts × chi_ladder (one row
    per n_devices); the split path has no device_mesh, so it emits n=1 rows only
    and is appended after all dense rows. Dense rows come first so the shared χ
    ladder lets each row self-stop at its own wall."""
    cells = []
    if "dense" in paths:
        cells += [
            Cell8(D=D, chi=chi, n_devices=n, path="dense")
            for n in device_counts
            for chi in chi_ladder
        ]
    if "split" in paths:
        cells += [Cell8(D=D, chi=chi, n_devices=1, path="split") for chi in chi_ladder]
    return cells


def _worker_env(n_devices, base_env):
    """Subprocess env: pin the n most-idle A100s, deterministic index order, no
    XLA preallocation so peak_gb is the real high-water mark.

    Uses JAX's ``cuda_async`` (cudaMallocAsync) allocator instead of the default
    BFC allocator: at D=8/large χ the dominant CTM intermediate needs a single
    ~18 GB contiguous block that BFC cannot place in its fragmented free space,
    OOMing tens of GB below the device limit (a fragmentation artifact, not a
    real wall). ``cuda_async``'s pool places it, exposing the *true* memory wall.
    ``peak_bytes_in_use`` (the ``peak_gb`` probe) tracks real usage under it.
    Autotuning is disabled to drop the autotuner's transient profiling workspace
    (another fragmentation-prone large alloc) and to keep compile deterministic."""
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_for(n_devices)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"
    ).strip()
    return env


def _wait_for_free_a100s(n_devices, gpu_wait_s, poll_s=30):
    """Block until n idle A100s are available, up to ``gpu_wait_s`` seconds, then
    return True. On a shared box GPU contention is usually transient, so we wait
    rather than crash. Returns False if the deadline passes (probing every
    ``poll_s`` seconds). Never raises."""
    deadline = time.monotonic() + gpu_wait_s
    while True:
        try:
            free_a100_indices(n_devices)
            return True
        except RuntimeError as e:
            if time.monotonic() >= deadline:
                print(f"[gpu] {e}; gave up after {gpu_wait_s}s", flush=True)
                return False
            print(f"[gpu] {e}; retrying in {poll_s}s", flush=True)
            time.sleep(poll_s)


def _launch(argv, n_devices, timeout_s, gpu_wait_s=1800):
    """Run a worker subprocess once n idle A100s are free. Returns True if it
    exited within the timeout, False on timeout, or None if no n idle A100s
    became free within ``gpu_wait_s`` (so the caller can stop the run gracefully
    without poisoning the resume cache). Never raises on GPU unavailability."""
    if not _wait_for_free_a100s(n_devices, gpu_wait_s):
        return None
    try:
        env = _worker_env(n_devices, dict(os.environ))
    except RuntimeError:
        return None  # a GPU was grabbed in the race between the wait and the pin
    print(f"[run] {' '.join(argv[argv.index('--cell'):])}", flush=True)
    try:
        subprocess.run(argv, env=env, check=False, timeout=timeout_s)
        return True
    except subprocess.TimeoutExpired:
        return False


def _run_worker(args):
    """Worker entry: run one phase, write its result JSON, echo it."""
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "su":
        try:
            su_seed_once(args.outdir, args.chi_su, args.imaginary_steps, args.dt)
            res = {"phase": "su", "ok": True, "error": None}
        except Exception as e:  # noqa: BLE001
            res = {"phase": "su", "ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        tensor_path = os.path.join(args.outdir, "A_opt.pkl")
        res = scan_cell(tensor_path, args.chi, args.n_devices, args.path)
    d4._atomic_write_text(args.out, json.dumps(res, indent=2))
    print(json.dumps(res))


def _build_argparser():
    p = argparse.ArgumentParser(description="iPEPS D=8 Heisenberg χ-scaling wall+rescue")
    p.add_argument("--cell", action="store_true", help="worker mode: run one phase")
    p.add_argument("--phase", choices=["su", "scan"], default="scan")
    p.add_argument("--chi", type=int, help="scan χ (worker scan phase)")
    p.add_argument("--n-devices", dest="n_devices", type=int, default=1)
    p.add_argument("--out", type=str, help="worker result JSON path")
    # shared / orchestrator:
    p.add_argument("--outdir", default="runs/d8_chi_scaling")
    p.add_argument("--smoke", action="store_true",
                   help="quick validation: tiny SU seed, short χ ladder")
    p.add_argument("--chi-su", dest="chi_su", type=int, default=24,
                   help="CTM χ for the SU-phase energy eval (kept small/cheap)")
    p.add_argument("--imaginary-steps", dest="imaginary_steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--chi-ladder", dest="chi_ladder", type=str,
                   default="64,96,112,128,192,256,320,384,448")
    p.add_argument("--device-counts", dest="device_counts", type=str, default="1,2")
    p.add_argument("--path", choices=["dense", "split", "both"], default="both",
                   help="forward-CTM path: dense (chi^2*D^6, wall ~chi=112), "
                        "split (chi^2*D^4, wall ~chi=448), or both (comparison)")
    p.add_argument("--cell-timeout-s", dest="cell_timeout_s", type=int, default=2400)
    p.add_argument("--gpu-wait-s", dest="gpu_wait_s", type=int, default=1800,
                   help="max seconds to wait for n idle A100s before a cell "
                        "(shared box: transient contention); then stop "
                        "gracefully and aggregate completed cells")
    return p


def su_seed_once(outdir, chi_su, imaginary_steps, dt):
    """Produce the single-site C4v seed via 2-site simple update and cache it to
    `<outdir>/A_opt.pkl`. SU is intrinsically 2-site; we take the A-sublattice
    tensor as the single-site seed (the optimize_gs_ad su_init convention) and
    C4v-symmetrize. No AD. Existence-cached: a present A_opt.pkl returns at once."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from tenax import (
        CTMConfig,
        heisenberg_gate,
        iPEPSConfig,
        sublattice_rotate_gate,
        symmetrize_c4v,
    )
    from tenax.algorithms.ipeps import _wrap_as_dense_tensor, ipeps

    tensor_path = os.path.join(outdir, "A_opt.pkl")
    if os.path.exists(tensor_path):
        print(f"[su] cached {tensor_path}; skipping simple update", flush=True)
        return tensor_path

    os.makedirs(outdir, exist_ok=True)
    gate = sublattice_rotate_gate(heisenberg_gate())
    cfg = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=imaginary_steps,
        dt=dt,
        ctm=CTMConfig(
            chi=chi_su, max_iter=50, conv_tol=1e-8,
            projector_method="svd", forward_gauge="phase",
        ),
        unit_cell="2site",  # ipeps() always runs 2-site SU
        su_init=True,
    )
    print(f"[su] D={D} simple update ({imaginary_steps} steps, dt={dt}, "
          f"χ_su={chi_su})", flush=True)
    t0 = time.perf_counter()
    e_su, (A_su, _B_su), _ = ipeps(gate, None, cfg)
    A_seed = _wrap_as_dense_tensor(symmetrize_c4v(A_su.todense()))
    print(f"[su] done in {time.perf_counter() - t0:.0f}s; SU E/site≈{float(e_su):.6f}",
          flush=True)
    A_host = jax.device_get(A_seed)  # numpy leaves -> device-agnostic, picklable
    d4._atomic_write_bytes(
        tensor_path, pickle.dumps(A_host, protocol=pickle.HIGHEST_PROTOCOL)
    )
    return tensor_path


def scan_cell(tensor_path, chi, n_devices, path):
    """Converge the forward CTM at χ on the fixed SU seed via ``path`` ('dense'
    or 'split'); return E/site + per-sweep timing + per-device peak memory.

    dense: closed χ²·D⁶ CTM (python_loop_ctm_converge); the single-site path is
    the only one with #632 device_mesh sharding, so n_devices>1 shards the D²=64
    axis. split: ctm_split_tensor — never forms the χ²·D⁶ edge (peak χ²·D³·d,
    forward χ²·D⁴-bounded, #641); single-GPU only (no device_mesh). Record-and-
    resume safe: never raises (OOM/errors are recorded)."""
    result = {
        "D": D, "chi": chi, "n_devices": n_devices, "path": path,
        "E_site": None, "err_vs_qmc": None, "total_s": None, "n_sweeps": None,
        "ms_per_sweep": None, "peak_gb": None, "converged": False,
        "oom": False, "error": None,
    }
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        from tenax import heisenberg_gate, sublattice_rotate_gate

        with open(tensor_path, "rb") as fh:
            A_opt = pickle.load(fh)
        H = sublattice_rotate_gate(heisenberg_gate())

        if path == "split":
            E, total_s, sweeps, converged = _converge_split(A_opt, H, chi)
        else:
            E, total_s, sweeps, converged = _converge_dense(
                A_opt, H, chi, n_devices
            )

        result.update(
            E_site=E, err_vs_qmc=E - REFERENCE_E, total_s=float(total_s),
            n_sweeps=sweeps, ms_per_sweep=1000.0 * total_s / max(sweeps, 1),
            converged=bool(converged), peak_gb=d4._peak_gb(),
        )
    except Exception as e:  # noqa: BLE001 — record and resume, never crash the sweep
        msg = f"{type(e).__name__}: {e}"
        result["error"] = msg
        if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower():
            result["oom"] = True
        result["peak_gb"] = d4._peak_gb()
    return result


def _converge_dense(A_opt, H, chi, n_devices):
    """Dense closed-path forward CTM (χ²·D⁶); device_mesh-sharded for n>1."""
    import jax

    from tenax import CTMConfig, compute_energy_ctm_tensor
    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
    from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs

    mesh = d4._build_mesh(n_devices)  # A100-only guard + GSPMD mesh for n>1
    cfg = CTMConfig(
        chi=chi, max_iter=200, conv_tol=1e-10,
        projector_method="svd", forward_gauge="phase", device_mesh=mesh,
    )
    kwargs = ctm_converge_kwargs(cfg)

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
    return E, total_s, int(info.iterations), bool(info.converged)


def _converge_split(A_opt, H, chi):
    """Split forward CTM (χ²·D⁴-bounded); single-GPU only. chi_I=chi is already
    lossless at D=8 (the spike showed chi_I=2χ gives identical E and memory)."""
    import jax

    from tenax import compute_energy_split_ctm_tensor
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor

    # Warm-up compile (process-cached) so the timed converge is pure compute.
    warm = ctm_split_tensor(A_opt, chi=chi, max_iter=200, conv_tol=1e-10, chi_I=chi)
    jax.block_until_ready(list(warm))

    t0 = time.perf_counter()
    env, info = ctm_split_tensor(
        A_opt, chi=chi, max_iter=200, conv_tol=1e-10, chi_I=chi, return_info=True
    )
    jax.block_until_ready(list(env))
    total_s = time.perf_counter() - t0

    E = float(compute_energy_split_ctm_tensor(A_opt, env, H, 2))
    return E, total_s, int(info.iterations), bool(info.converged)


def _su_phase(outdir, chi_su, imaginary_steps, dt, gpu_wait_s=1800):
    """Run the one-time SU seed in a subprocess pinned to one idle A100."""
    if os.path.exists(os.path.join(outdir, "A_opt.pkl")):
        print("[su] A_opt.pkl present; simple update skipped", flush=True)
        return
    out = os.path.join(outdir, "su_status.json")
    argv = [
        sys.executable, str(pathlib.Path(__file__).resolve()), "--cell",
        "--phase", "su", "--outdir", outdir, "--chi-su", str(chi_su),
        "--imaginary-steps", str(imaginary_steps), "--dt", str(dt),
        "--n-devices", "1", "--out", out,
    ]
    # resume-safe; allow a long wall and wait out transient GPU contention
    _launch(argv, n_devices=1, timeout_s=None, gpu_wait_s=gpu_wait_s)


def _load_or_run_scan(cell, outdir, timeout_s, gpu_wait_s=1800):
    """Resume: load an existing cell JSON, else launch the scan worker and load
    what it wrote. A timeout/no-file is recorded as an error so the row stops.

    Returns None (writing no cell JSON) when no idle A100s freed up within
    ``gpu_wait_s`` -- a transient-infra condition, not a result -- so the caller
    stops gracefully and a later resume retries the cell."""
    path = pathlib.Path(_cell_path(outdir, cell))
    cached = d4._read_json_or_none(path) if path.exists() else None
    if cached is not None:
        return cached
    argv = [
        sys.executable, str(pathlib.Path(__file__).resolve()), "--cell",
        "--phase", "scan", "--outdir", outdir, "--chi", str(cell.chi),
        "--n-devices", str(cell.n_devices), "--path", cell.path, "--out", str(path),
    ]
    ok = _launch(argv, cell.n_devices, timeout_s, gpu_wait_s)
    if ok is None:  # no idle A100s -> don't poison the resume cache
        return None
    loaded = d4._read_json_or_none(path)
    if loaded is not None:
        return loaded
    res = {
        "D": cell.D, "chi": cell.chi, "n_devices": cell.n_devices, "path": cell.path,
        "E_site": None, "err_vs_qmc": None, "ms_per_sweep": None,
        "n_sweeps": None, "peak_gb": None, "converged": False, "oom": False,
        "error": ("timeout" if not ok else "worker produced no result file"),
    }
    d4._atomic_write_text(str(path), json.dumps(res, indent=2))
    return res


def _apply_smoke(args):
    """Shrink an args namespace to a fast end-to-end validation run."""
    args.outdir = args.outdir + "_smoke"
    args.chi_su = 8
    args.imaginary_steps = 20
    args.chi_ladder = "8,12"
    args.device_counts = "1"
    args.cell_timeout_s = 1200


def _comparison_table_md(dense, split):
    """One row per χ present in either path: dense vs split peak/E/status."""
    def by_chi(rs):
        return {r["chi"]: r for r in rs}

    d, s = by_chi(dense), by_chi(split)
    chis = sorted(set(d) | set(s))
    lines = [
        "### Wall comparison: dense (χ²·D⁶) vs split (χ²·D⁴)",
        "",
        "| χ | dense peak GB | dense E | dense | split peak GB | split E | split |",
        "|---|---------------|---------|-------|---------------|---------|-------|",
    ]
    for chi in chis:
        dr, sr = d.get(chi), s.get(chi)
        lines.append(
            f"| {chi} "
            f"| {d4._fmt(dr.get('peak_gb') if dr else None, '.2f')} "
            f"| {d4._fmt(dr.get('E_site') if dr else None, '.5f')} "
            f"| {d4._status(dr) if dr else '-'} "
            f"| {d4._fmt(sr.get('peak_gb') if sr else None, '.2f')} "
            f"| {d4._fmt(sr.get('E_site') if sr else None, '.5f')} "
            f"| {d4._status(sr) if sr else '-'} |"
        )
    return "\n".join(lines)


def _plot_wall_comparison(dense, split, outdir):
    """Two-panel PNG: per-device peak GB and converge wall-time vs χ, dense vs
    split, with the 80 GB device limit marked. Best-effort."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pts(rs, metric):
        return sorted((r["chi"], r[metric]) for r in rs
                      if not r.get("oom") and not r.get("error")
                      and r.get(metric) is not None)

    fig, (ax_m, ax_t) = plt.subplots(1, 2, figsize=(11, 4))
    for rs, label, color in [(dense, "dense", "C0"), (split, "split", "C1")]:
        m = pts(rs, "peak_gb")
        if m:
            ax_m.plot(*zip(*m), marker="o", label=label, color=color)
        t = pts(rs, "total_s")
        if t:
            ax_t.plot(*zip(*t), marker="o", label=label, color=color)
    ax_m.axhline(80.0, ls="--", color="k", label="80 GB A100")
    ax_m.set_xlabel("χ"); ax_m.set_ylabel("per-device peak GB")
    ax_m.set_title("D=8 single-GPU memory wall"); ax_m.legend()
    ax_t.set_xlabel("χ"); ax_t.set_ylabel("converge wall (s)")
    ax_t.set_yscale("log"); ax_t.set_title("D=8 converge time"); ax_t.legend()
    fig.tight_layout()
    p = os.path.join(outdir, "d8_wall_comparison.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def _aggregate8(results, outdir):
    """Write per-path convergence + performance markdown, a dense-vs-split wall
    comparison table, results.csv (with a path column), and the comparison PNG."""
    dense = [r for r in results if r.get("path") == "dense"]
    split = [r for r in results if r.get("path") == "split"]

    sections = [_comparison_table_md(dense, split), ""]
    for name, rs in [("Dense", dense), ("Split", split)]:
        if rs:
            sections += [f"## {name} path", "",
                         d4.results_to_convergence_md(rs, d_label=D), "",
                         d4.results_to_performance_md(rs), ""]
    conv_md = "\n".join(sections)
    d4._atomic_write_text(os.path.join(outdir, "convergence.md"), conv_md)

    keys = ["D", "chi", "n_devices", "path", "E_site", "err_vs_qmc",
            "ms_per_sweep", "n_sweeps", "peak_gb", "converged", "oom", "error"]
    rows = [{k: r.get(k) for k in keys} for r in results]
    if rows:
        with open(os.path.join(outdir, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    try:
        _plot_wall_comparison(dense, split, outdir)
    except Exception as e:  # noqa: BLE001 — plotting is best-effort
        print(f"[warn] plotting failed: {e}", flush=True)
    print(conv_md)
    print(f"\n[done] wrote {outdir}/convergence.md, results.csv, "
          "d8_wall_comparison.png")


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    chi_ladder = [int(x) for x in args.chi_ladder.split(",")]
    device_counts = [int(x) for x in args.device_counts.split(",")]
    paths = ["dense", "split"] if args.path == "both" else [args.path]

    # Phase 1: simple-update seed once (single idle A100).
    _su_phase(outdir, args.chi_su, args.imaginary_steps, args.dt, args.gpu_wait_s)
    if not os.path.exists(os.path.join(outdir, "A_opt.pkl")):
        print("[abort] SU produced no A_opt.pkl; see "
              f"{outdir}/su_status.json", flush=True)
        return

    # Phase 2: scan each (path, n_devices) row; stop a row at its wall. A None
    # result means no idle A100s freed up in time -> stop gracefully and
    # aggregate what completed (re-run to resume).
    cells = build_grid(chi_ladder, device_counts, paths)
    rows = {}
    for c in cells:
        rows.setdefault((c.path, c.n_devices), []).append(c)

    results = []
    aborted = False
    for key in rows:
        if aborted:
            break
        for cell in rows[key]:
            res = _load_or_run_scan(cell, outdir, args.cell_timeout_s, args.gpu_wait_s)
            if res is None:
                print(f"[abort] no {cell.n_devices} idle A100(s) for "
                      f"{cell.path} χ={cell.chi} within {args.gpu_wait_s}s; "
                      "aggregating completed cells (re-run to resume)", flush=True)
                aborted = True
                break
            results.append(res)
            if d4.should_stop_row(res):
                print(f"[stop] {cell.path} n={cell.n_devices} row stopped at "
                      f"χ={cell.chi} ({d4._status(res)})", flush=True)
                break

    _aggregate8(results, outdir)


if __name__ == "__main__":
    _args = _build_argparser().parse_args()
    if _args.smoke:
        _apply_smoke(_args)
    if _args.cell:
        _run_worker(_args)
    else:
        main(_args)
