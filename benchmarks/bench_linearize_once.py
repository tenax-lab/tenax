"""Compile and run cost of the implicit-AD backward, for an A/B against main.

U(1) 2-site XXZ (Delta=0.3) from a simple-update state, ``optimize_gs_ad``
for a few L-BFGS steps on the default backward.  Run it from the worktree
whose tenax you measure, with that worktree's interpreter, on GPU 1:

    CUDA_VISIBLE_DEVICES=1 JAX_LOG_COMPILES=1 JAX_ENABLE_COMPILATION_CACHE=false \
      /usr/bin/time -v .venv/bin/python benchmarks/bench_linearize_once.py \
      --D 3 --chi 16 --out OUT.json > OUT.log 2>&1

Per-program compile times come from the log (``JAX_LOG_COMPILES``); the
JSON holds energies, step times and device peak.  Spec:
docs/superpowers/specs/2026-10-25-linearize-once-adjoint-design.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, iPEPSConfig, ipeps, optimize_gs_ad  # noqa: E402
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate  # noqa: E402

_COMPILE_RE = re.compile(
    r"Finished XLA compilation of jit\((.+?)\) in ([0-9.eE+-]+) sec"
)


def parse_compiles(text: str, *, upto_optimize_done: bool = True) -> dict[str, float]:
    """Sum compile seconds per program; by default stop at ``MARK optimize_done``.

    The SU-state gradient taken after the run compiles more programs; they
    stay out of the A1 totals unless ``upto_optimize_done`` is False.
    """
    if upto_optimize_done:
        text = text.split("MARK optimize_done", 1)[0]
    out: dict[str, float] = {}
    for name, secs in _COMPILE_RE.findall(text):
        out[name] = out.get(name, 0.0) + float(secs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=3)
    ap.add_argument("--chi", type=int, default=16)
    ap.add_argument("--su-steps", type=int, default=200)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parse", help="print the compile table of a log and exit")
    ap.add_argument(
        "--parse-all",
        action="store_true",
        help="with --parse: count the whole log, not only before optimize_done",
    )
    a = ap.parse_args()
    if a.parse:
        table = parse_compiles(
            open(a.parse).read(), upto_optimize_done=not a.parse_all
        )
        for name, secs in sorted(table.items(), key=lambda kv: -kv[1])[:12]:
            print(f"{secs:10.1f}  {name}")
        print(f"{sum(table.values()):10.1f}  TOTAL")
        return
    print("devices", jax.devices(), flush=True)
    t0 = time.time()
    A0, B0 = heisenberg_u1sz_init_pair(a.D, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=a.D,
        num_imaginary_steps=a.su_steps,
        dt=0.05,
        ctm=CTMConfig(chi=a.chi),
    )
    _, (A, B), _ = ipeps(xxz_gate(0.3), (A0, B0), su, compute_energy=False)
    A, B = A * (1.0 / A.norm()), B * (1.0 / B.norm())
    print(f"MARK su_done {time.time() - t0:.1f} s", flush=True)
    ctm = dataclasses.replace(
        CTMConfig(chi=a.chi),
        conv_tol=1e-10,
        max_iter=1000,
        plateau_patience=None,
        on_unconverged="warn",
    )
    cfg = iPEPSConfig(
        max_bond_dim=a.D,
        unit_cell="2site",
        ctm=ctm,
        gs_num_steps=a.steps,
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e-12,
        return_history=True,
        gs_verbose=True,
        gs_log_interval=1,
    )
    t1 = time.time()
    print("MARK optimize_start", flush=True)
    _, _envs, E, hist = optimize_gs_ad(xxz_gate(0.3), (A, B), cfg)
    wall = time.time() - t1
    print(f"MARK optimize_done {wall:.1f} s E={float(E):.12f}", flush=True)
    stats = jax.devices()[0].memory_stats() or {}
    # Full-precision gradient at the SU input state, default backward: the
    # log prints |g| to 4 digits only, too coarse for the A3 gate.
    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    g = jax.grad(
        lambda p: jnp.real(
            ctm_energy_implicit(
                {(0, 0): p[0], (1, 0): p[1]},
                CHECKERBOARD_NEIGHBORS,
                xxz_gate(0.3),
                chi=a.chi,
                max_iter=1000,
                conv_tol=1e-10,
            )
        )
    )((A, B))
    gvec = np.concatenate(
        [np.asarray(x).ravel().astype(np.complex128) for x in jax.tree.leaves(g)]
    )
    if np.allclose(gvec.imag, 0.0):
        gvec = gvec.real.astype(np.float64)
    grad_path = (a.out[:-5] if a.out.endswith(".json") else a.out) + ".grad.npy"
    np.save(grad_path, gvec)
    grad_norm = float(np.linalg.norm(gvec))
    print(f"MARK grad_done |g|={grad_norm!r}", flush=True)
    out = {
        "D": a.D,
        "chi": a.chi,
        "su_steps": a.su_steps,
        "steps": a.steps,
        "argv": sys.argv,
        "devices": [str(d) for d in jax.devices()],
        "energy_final": float(E),
        "energies": [float(e) for e in hist.get("energies", [])],
        "step_times": [float(s) for s in hist.get("step_times", [])],
        "optimize_wall_s": wall,
        "device_peak_bytes": stats.get("peak_bytes_in_use"),
        "grad_norm_su": grad_norm,
        "grad_path": grad_path,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("MARK json_written", a.out, flush=True)


if __name__ == "__main__":
    main()
