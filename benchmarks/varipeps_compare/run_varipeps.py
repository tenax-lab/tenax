"""variPEPS 1.4.2 CLI runner for the compare benchmark.

Usage:
    python -m benchmarks.varipeps_compare.run_varipeps \\
        --payload payload.npz --path single_site --D 2 --chi 16 \\
        --tol 1e-6 --max-steps 100 --out varipeps_<key>.json
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Configure JAX BEFORE importing varipeps (variPEPS reads jax.config at import).
jax.config.update("jax_enable_x64", True)

import varipeps  # noqa: E402

from .payload import load_payload  # noqa: E402
from .protocol import CTM_MAX_ITER, CTM_TOL  # noqa: E402


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _make_peps_tensor(arr_lt_p_rb: np.ndarray, *, chi_start: int, chi_max: int):
    """Wrap a single (l, t, p, r, b) array as a varipeps PEPS_Tensor."""
    d = int(arr_lt_p_rb.shape[2])
    D_seq = (
        int(arr_lt_p_rb.shape[0]),
        int(arr_lt_p_rb.shape[1]),
        int(arr_lt_p_rb.shape[3]),
        int(arr_lt_p_rb.shape[4]),
    )
    return varipeps.peps.PEPS_Tensor.from_tensor(
        jnp.asarray(arr_lt_p_rb),
        d=d,
        D=D_seq,
        chi=chi_start,
        max_chi=chi_max,
    )


def _build_unitcell_bipartite_2site(
    init_AB: np.ndarray, D: int, chi_start: int, chi_max: int
):
    """Build a 2-tensor checkerboard PEPS_Unit_Cell from stacked (2, D, D, D, D, d) init.

    Tenax produces (l, t, r, b, p) layout; variPEPS expects (l, t, p, r, b)
    (per ``varipeps/peps/tensor.py`` ``PEPS_Tensor.from_tensor`` shape check).
    Transpose accordingly.
    """
    A = np.transpose(init_AB[0], (0, 1, 4, 2, 3))
    B = np.transpose(init_AB[1], (0, 1, 4, 2, 3))
    A_pt = _make_peps_tensor(A, chi_start=chi_start, chi_max=chi_max)
    B_pt = _make_peps_tensor(B, chi_start=chi_start, chi_max=chi_max)
    structure = [[0, 1], [1, 0]]
    return varipeps.peps.PEPS_Unit_Cell.from_tensor_list([A_pt, B_pt], structure)


def _build_unitcell_single_site(
    init_A: np.ndarray, D: int, chi_start: int, chi_max: int
):
    """1×1 unconstrained PEPS_Unit_Cell — same ansatz as Tenax's single_site path."""
    A = np.transpose(init_A, (0, 1, 4, 2, 3))
    A_pt = _make_peps_tensor(A, chi_start=chi_start, chi_max=chi_max)
    structure = [[0]]
    return varipeps.peps.PEPS_Unit_Cell.from_tensor_list([A_pt], structure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, type=Path)
    ap.add_argument("--path", required=True, choices=("single_site", "bipartite_2site"))
    ap.add_argument("--D", required=True, type=int)
    ap.add_argument("--chi", required=True, type=int)
    ap.add_argument("--tol", required=True, type=float)
    ap.add_argument("--max-steps", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    init_np, gate_np, _ = load_payload(args.payload)

    # Match variPEPS knobs to protocol.
    varipeps.config.optimizer_method = varipeps.config.Optimizing_Methods.L_BFGS
    varipeps.config.ctmrg_full_projector_method = (
        varipeps.config.Projector_Method.FISHMAN
    )
    varipeps.config.optimizer_max_steps = args.max_steps
    varipeps.config.ctmrg_max_steps = CTM_MAX_ITER
    varipeps.config.ctmrg_convergence_eps = CTM_TOL
    varipeps.config.ctmrg_print_steps = False
    varipeps.config.ad_custom_print_steps = False

    # variPEPS expects the two-site gate as a (d², d²) matrix.
    gate = jnp.asarray(gate_np.reshape(4, 4))
    exp_func = varipeps.expectation.Two_Sites_Expectation_Value(
        horizontal_gates=(gate,),
        vertical_gates=(gate,),
    )

    chi_start = min(args.D**2, args.chi)
    if args.path == "single_site":
        unitcell = _build_unitcell_single_site(init_np, args.D, chi_start, args.chi)
    else:
        unitcell = _build_unitcell_bipartite_2site(init_np, args.D, chi_start, args.chi)

    autosave = args.out.with_suffix(".hdf5")
    autosave.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = varipeps.optimization.optimize_peps_network(
        unitcell, exp_func, autosave_filename=str(autosave)
    )
    total = time.perf_counter() - t0

    # variPEPS returns step_energies/step_runtime as dicts keyed by run index;
    # we want the best-run trajectory (the one used to compute result.fun).
    best_run = int(result.best_run)
    energies = [float(e) for e in result.step_energies[best_run]]
    step_times_all = [float(t) for t in result.step_runtime[best_run]]

    # variPEPS step_runtime[0] includes JIT compile; split it out.
    if step_times_all:
        jit_time = step_times_all[0]
        step_times = step_times_all[1:]
    else:
        jit_time = 0.0
        step_times = []

    out = {
        "lib": "varipeps",
        "path": args.path,
        "D": args.D,
        "chi": args.chi,
        "dtype": "complex128",
        "seed": 0,
        "energy_history": energies,
        "step_times": step_times,
        "jit_compile_time": jit_time,
        "final_energy": float(result.fun),
        "num_steps": int(result.nit),
        "converged": bool(result.success),
        "total_wall_clock": total,
        "peak_memory_mb": _peak_rss_mb(),
        "device": str(jax.devices()[0]).lower(),
        "lib_version": varipeps.__version__,
    }
    args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
