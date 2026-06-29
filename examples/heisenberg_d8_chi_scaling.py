"""iPEPS D=8 square-lattice Heisenberg AFM: simple-update seed + forward-CTM
χ-scan showing the single-GPU memory wall and the #632 multi-GPU rescue.

    # full run (orchestrator): SU seed once, then scan χ × {1,2} GPU
    uv run python examples/heisenberg_d8_chi_scaling.py --outdir runs/d8_chi_scaling

    # quick validation (tiny end-to-end)
    uv run python examples/heisenberg_d8_chi_scaling.py --smoke

    # single cell (worker; normally invoked by the orchestrator):
    uv run python examples/heisenberg_d8_chi_scaling.py --cell --phase scan \
        --chi 64 --n-devices 1 --outdir runs/d8_chi_scaling --out /tmp/cell.json

Pure helpers import only stdlib; jax/tenax imports live inside the worker so the
parent's CUDA_VISIBLE_DEVICES takes effect before the child initialises a JAX
backend, and so the helper unit tests stay fast and jax-free.

The D-agnostic formatting/plot/IO/mesh helpers are reused from the sibling D=4
driver (path-loaded as ``d4``) so the merged file is not modified.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import pickle
import subprocess
import sys
import time

# Path-load the D=4 driver to reuse its D-agnostic pure helpers. Its top level
# imports only stdlib (jax/tenax live inside functions), so this stays jax-free.
_D4_PATH = pathlib.Path(__file__).resolve().parent / "heisenberg_d4_chi_scaling.py"
_d4_spec = importlib.util.spec_from_file_location("heisenberg_d4_chi_scaling", _D4_PATH)
d4 = importlib.util.module_from_spec(_d4_spec)
_d4_spec.loader.exec_module(d4)

REFERENCE_E = d4.REFERENCE_E  # Sandvik QMC, square-lattice spin-1/2 Heisenberg AFM
D = 8  # fixed bond dimension for this driver


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


def build_grid(chi_ladder, device_counts):
    """Scan cells in device-major, chi-minor order (one row per n_devices).
    Reuses the D=4 module's frozen ``Cell`` dataclass with D=8."""
    return [
        d4.Cell(D=D, chi=chi, n_devices=n)
        for n in device_counts
        for chi in chi_ladder
    ]


def _worker_env(n_devices, base_env):
    """Subprocess env: pin the n most-idle A100s, deterministic index order, no
    XLA preallocation so peak_gb is the real high-water mark."""
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_for(n_devices)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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
        res = scan_cell(tensor_path, args.chi, args.n_devices)
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
                   default="64,96,128,160,192,224,256")
    p.add_argument("--device-counts", dest="device_counts", type=str, default="1,2")
    p.add_argument("--cell-timeout-s", dest="cell_timeout_s", type=int, default=2400)
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


def scan_cell(tensor_path, chi, n_devices):
    """Converge forward CTM at χ on the fixed SU seed; return E/site + per-sweep
    timing + per-device peak memory. The single-site path is the only one with
    #632 device_mesh sharding, so n_devices>1 shards the D²=64 axis. Record-and-
    resume safe: never raises (OOM/errors are recorded)."""
    result = {
        "D": D, "chi": chi, "n_devices": n_devices,
        "E_site": None, "err_vs_qmc": None, "total_s": None, "n_sweeps": None,
        "ms_per_sweep": None, "peak_gb": None, "converged": False,
        "oom": False, "error": None,
    }
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        from tenax import (
            CTMConfig, compute_energy_ctm_tensor, heisenberg_gate,
            sublattice_rotate_gate,
        )
        from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
        from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
        from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs

        mesh = d4._build_mesh(n_devices)  # A100-only guard + GSPMD mesh for n>1
        with open(tensor_path, "rb") as fh:
            A_opt = pickle.load(fh)
        H = sublattice_rotate_gate(heisenberg_gate())

        cfg = CTMConfig(
            chi=chi, max_iter=200, conv_tol=1e-10,
            projector_method="svd", forward_gauge="phase", device_mesh=mesh,
        )
        kwargs = ctm_converge_kwargs(cfg)  # forwards device_mesh; default recipe

        # Warm-up: compile the χ-specific @jit step (process-cached) so the timed
        # converge measures pure per-sweep compute.
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
            converged=bool(info.converged), peak_gb=d4._peak_gb(),
        )
    except Exception as e:  # noqa: BLE001 — record and resume, never crash the sweep
        msg = f"{type(e).__name__}: {e}"
        result["error"] = msg
        if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower():
            result["oom"] = True
        result["peak_gb"] = d4._peak_gb()
    return result


def _su_phase(outdir, chi_su, imaginary_steps, dt):
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
    _launch(argv, n_devices=1, timeout_s=None)  # resume-safe; allow long wall


def _load_or_run_scan(cell, outdir, timeout_s):
    """Resume: load an existing cell JSON, else launch the scan worker and load
    what it wrote. A timeout/no-file is recorded as an error so the row stops."""
    path = pathlib.Path(d4.cell_result_path(outdir, cell))
    cached = d4._read_json_or_none(path) if path.exists() else None
    if cached is not None:
        return cached
    argv = [
        sys.executable, str(pathlib.Path(__file__).resolve()), "--cell",
        "--phase", "scan", "--outdir", outdir, "--chi", str(cell.chi),
        "--n-devices", str(cell.n_devices), "--out", str(path),
    ]
    ok = _launch(argv, cell.n_devices, timeout_s)
    loaded = d4._read_json_or_none(path)
    if loaded is not None:
        return loaded
    res = {
        "D": cell.D, "chi": cell.chi, "n_devices": cell.n_devices,
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


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    chi_ladder = [int(x) for x in args.chi_ladder.split(",")]
    device_counts = [int(x) for x in args.device_counts.split(",")]

    # Phase 1: simple-update seed once (single idle A100).
    _su_phase(outdir, args.chi_su, args.imaginary_steps, args.dt)
    if not os.path.exists(os.path.join(outdir, "A_opt.pkl")):
        print("[abort] SU produced no A_opt.pkl; see "
              f"{outdir}/su_status.json", flush=True)
        return

    # Phase 2: scan χ per device row; stop a row on OOM/error/timeout.
    results = []
    for n in device_counts:
        for chi in chi_ladder:
            res = _load_or_run_scan(d4.Cell(D=D, chi=chi, n_devices=n), outdir,
                                    args.cell_timeout_s)
            results.append(res)
            if d4.should_stop_row(res):
                print(f"[stop] n={n} row stopped at χ={chi} "
                      f"({d4._status(res)})", flush=True)
                break

    d4._aggregate(results, outdir, d_label=D)


if __name__ == "__main__":
    _args = _build_argparser().parse_args()
    if _args.smoke:
        _apply_smoke(_args)
    if _args.cell:
        _run_worker(_args)
    else:
        main(_args)
