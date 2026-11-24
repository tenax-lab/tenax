"""Charged C4v iPEPS AD at D=3 against Hasik's U(1)-C4v state (spec 2).

Outside CI (spec 2, section 6, "Benchmark").  ``--state`` is a local copy of
Hasik's D=3 state file; the ``.dat`` with the same stem holds his energies
by chi.

Modes:
    load    convert to a charged C4v SymmetricTensor and check it
    energy  the tenax energy at each --chi against the .dat value
    fd      the FD rule in three C4v directions at a perturbed state
    opt     optimise from a perturbed Hasik start and from a random start
    all     energy, fd and opt in turn

The exit status is 1 when the conversion loses weight or an FD check fails.
The optimisation targets are reported, not enforced.

    JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync python -u \\
        -m benchmarks.ipeps_c4v_charged_ad --state <path> --mode load
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.ipeps_spectrum_hasik import load_charges, load_state
from tenax import (
    CTMConfig,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
    sublattice_rotate_gate,
)
from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad
from tenax.algorithms._ctm_c4v_charges import _is_charged_c4v
from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v
from tenax.algorithms._ctm_tensor_c4v_staggered import _c4v_staggered_to_full_env
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor

# One FD rule for CI and this benchmark: the h grid, the plateau, the C4v
# directions and the "P frozen" gradient come from the test fixtures, which
# import only jax, numpy and tenax.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from _c4v_staggered_fixtures import (  # noqa: E402
    FD_STEPS,
    c4v_direction,
    fd_plateau,
    frozen_projectors,
    with_data,
)

GATE = sublattice_rotate_gate(heisenberg_gate())
PERTURBATION = 0.05  # relative size of the C4v kick away from Hasik's optimum


def charged_state(path):
    """Hasik's state as a charged C4v SymmetricTensor, and the weight it loses.

    Hasik's tensor is nonzero where phys + sum(virtual) = total.  The tenax
    legs u, l (OUT, vc), d, r (IN, dual(vc)) and phys (IN, pc) conserve
    sum(virtual) = pc, so pc = total - phys.
    """
    A = np.asarray(load_state(path))
    if np.iscomplexobj(A):
        assert float(np.max(np.abs(A.imag))) < 1e-14, "the state is complex"
        A = A.real
    phys, virt, total = load_charges(path)
    sym = U1Symmetry()
    vc = np.asarray(virt, dtype=np.int32)
    pc = np.asarray([total - p for p in phys], dtype=np.int32)
    vd = sym.dual(vc)
    indices = (
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, vd.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, vd.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pc, FlowDirection.IN, label="phys"),
    )
    A_sym = SymmetricTensor.from_dense(jnp.asarray(A), indices)
    lost = float(np.max(np.abs(np.asarray(A_sym.todense()) - A)))
    return A_sym, lost, vc.tolist(), pc.tolist()


def eager_energy(A, chi):
    """The #1211 eager energy: ctm_tensor_c4v to sv_diff < 1e-12."""
    env = ctm_tensor_c4v(A, chi, max_iter=2000, conv_tol=1e-12)
    return float(compute_energy_ctm_tensor(A, env, GATE, d=2))


def dat_energies(path):
    """Hasik's energy by chi_obs, from the .dat next to the state file."""
    rows = {}
    for line in pathlib.Path(path).with_suffix(".dat").read_text().splitlines():
        try:
            values = [float(v) for v in line.split()]
        except ValueError:
            continue  # a header or a comment
        if len(values) >= 4:
            rows[int(round(values[2]))] = values[3]
    return rows


def perturbed(A, key, size=PERTURBATION):
    x = A._data + size * c4v_direction(A, key)
    return with_data(A, x / jnp.linalg.norm(x))


def fd_check(A, chi, n_dirs=3):
    """The FD rule of spec 2, section 6, at A."""
    ctm = CTMConfig(
        chi=chi,
        max_iter=2000,
        conv_tol=1e-13,
        adjoint_tol=1e-12,
        adjoint_maxiter=200,
        adjoint_tikhonov=0.0,
    )
    C, T, layout = ad.charged_c4v_cold_start(
        A, dataclasses.replace(ctm, conv_tol=1e-10)
    )
    warm = ad._elementwise_phase(A, (C, T), layout, ctm, {})
    record = {}

    def energy(A_):
        C_, T_ = ad.charged_c4v_converge(A_, warm, layout, ctm, record)
        return compute_energy_ctm_tensor(
            A_, _c4v_staggered_to_full_env(C_, T_), GATE, d=2
        )

    def phi(v, eps):
        A_e = with_data(A, A._data + eps * v)
        C_, T_ = ad._elementwise_phase(A_e, warm, layout, ctm, {})
        return float(
            compute_energy_ctm_tensor(
                A_e, _c4v_staggered_to_full_env(C_, T_), GATE, d=2
            )
        )

    t = time.perf_counter()
    g = jax.grad(energy)(A)
    first_grad_s = time.perf_counter() - t
    adjoint = dict(record)
    print(
        f"fd: first gradient {first_grad_s:.1f} s, layout {layout.dims}, {adjoint}",
        flush=True,
    )
    with frozen_projectors():
        g_frozen = jax.grad(energy)(A)
    dirs = []
    for k in range(n_dirs):
        v = c4v_direction(A, jax.random.PRNGKey(k))
        slope = float(jnp.vdot(g._data, v))
        frozen = float(jnp.vdot(g_frozen._data, v))
        fd, fd_unc = fd_plateau(lambda eps: phi(v, eps))
        row = {
            "dir": k,
            "g_dot_v": slope,
            "fd": fd,
            "fd_unc": fd_unc,
            "grad_err": abs(slope - fd),
            "flow_minus_frozen": abs(slope - frozen),
            "ok": abs(slope - fd) <= 3.0 * fd_unc + 1e-12,
            "regime": abs(slope) > 100.0 * fd_unc
            and abs(slope - frozen) > 10.0 * fd_unc,
        }
        print(
            f"fd dir {k}: |g.v|={abs(slope):.3e} fd_unc={fd_unc:.1e} "
            f"grad_err={row['grad_err']:.1e} "
            f"flow-frozen={row['flow_minus_frozen']:.1e} "
            f"ok={row['ok']} regime={row['regime']}",
            flush=True,
        )
        dirs.append(row)
    return {
        "chi": chi,
        "layout": layout.dims,
        "fd_steps": list(FD_STEPS),
        "first_grad_s": first_grad_s,
        "adjoint": adjoint,
        "dirs": dirs,
    }


def optimise(A_start, chi, steps, label):
    config = iPEPSConfig(
        max_bond_dim=3,
        gs_c4v=True,
        su_init=False,
        gs_num_steps=steps,
        return_history=True,
        gs_verbose=True,
        ctm=CTMConfig(chi=chi, max_iter=2000, conv_tol=1e-10),
    )
    t = time.perf_counter()
    _A, _env, E, hist = optimize_gs_ad(GATE, A_start, config)
    wall_s = time.perf_counter() - t
    times = hist["step_times"]
    row = {
        "label": label,
        "E": E,
        "grad_norm": hist["grad_norms"][-1] if hist["grad_norms"] else None,
        "steps": hist["num_steps"],
        "converged": hist["converged"],
        "wall_s": wall_s,
        "first_step_s": times[0] if times else None,
        "median_later_step_s": float(np.median(times[1:])) if len(times) > 1 else None,
        "layout": hist["layout"],
        "layout_mismatch_steps": hist["layout_mismatch_steps"],
    }
    print(f"opt {label}: {row}", flush=True)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Charged C4v iPEPS AD at D=3")
    parser.add_argument("--state", required=True, type=pathlib.Path)
    parser.add_argument("--chi", type=int, nargs="+", default=[36, 54, 72])
    parser.add_argument(
        "--mode", choices=("load", "energy", "fd", "opt", "all"), default="all"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="opt mode: the step budget of each of the two runs (default: %(default)s)",
    )
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args(argv)

    A, lost, vc, pc = charged_state(args.state)
    report = {
        "state": str(args.state),
        "vc": vc,
        "pc": pc,
        "entries": int(A._data.size),
        "lost": lost,
        "charged": _is_charged_c4v(A),
    }
    print(
        f"load: vc={vc} pc={pc} charged={report['charged']} "
        f"entries={report['entries']} max|dense - A|={lost:.1e}",
        flush=True,
    )
    ok = report["charged"] and lost < 1e-12
    modes = ("energy", "fd", "opt") if args.mode == "all" else (args.mode,)
    if ok and "energy" in modes:
        dat = dat_energies(args.state)
        report["energy"] = []
        for chi in args.chi:
            E = eager_energy(A, chi)
            ref = dat.get(chi)
            gap = None if ref is None else E - ref
            report["energy"].append(
                {"chi": chi, "E_tenax": E, "E_hasik": ref, "gap": gap}
            )
            print(f"energy chi={chi}: tenax {E:.13f} hasik {ref} gap {gap}", flush=True)
    if ok and "fd" in modes:
        report["fd"] = fd_check(perturbed(A, jax.random.PRNGKey(100)), args.chi[0])
        ok = all(row["ok"] for row in report["fd"]["dirs"])
    if ok and "opt" in modes:
        chi = args.chi[0]
        E_ref = eager_energy(A, chi)
        start = perturbed(A, jax.random.PRNGKey(101))
        random_start = SymmetricTensor.random_normal(A.indices, jax.random.PRNGKey(7))
        runs = [
            optimise(start, chi, args.steps, "perturbed Hasik"),
            optimise(random_start, chi, args.steps, "random"),
        ]
        for row in runs:
            row["target_met"] = (
                row["E"] <= E_ref + 1e-9
                and row["grad_norm"] is not None
                and row["grad_norm"] < 1e-5
            )
        report["opt"] = {
            "chi": chi,
            "max_steps": args.steps,
            "E_ref": E_ref,
            "runs": runs,
        }
        print(
            f"opt targets (E <= {E_ref:.12f} + 1e-9, |g| < 1e-5): "
            f"{[row['target_met'] for row in runs]}",
            flush=True,
        )
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"peak RSS {report['peak_rss_mb']:.0f} MB; ok={ok}", flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
