"""Gated S1-S5 pipeline for one symmetric 2-site AD point (spec §5)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import traceback
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from tenax import CTMConfig, ipeps, iPEPSConfig, optimize_gs_ad  # noqa: E402
from tenax.algorithms import _ctm_energy_ad  # noqa: E402
from tenax.algorithms._symmetric_ad_validation import (  # noqa: E402
    converged_energy,
    densify_pair,
    fd_directional,
    grad_dot,
    make_2site_loss,
    oracle_value_and_grad,
)
from tenax.algorithms.ctm_layout import (  # noqa: E402
    layout_of,
    same_layout,
    symmetric_layout_seed,
)
from tenax.algorithms.ipeps import (  # noqa: E402
    heisenberg_u1sz_init_pair,
    xxz_gate,
    xxz_gate_u1sz,
    xxz_gate_z2,
    xxz_z2_init_pair,
)
from tenax.core.tensor import SymmetricTensor  # noqa: E402

MODELS = {"xxz03": 0.3, "heis": 1.0}
SYMS = {
    "u1": (heisenberg_u1sz_init_pair, xxz_gate_u1sz),
    "z2": (xxz_z2_init_pair, xxz_gate_z2),
}
STAGES = ("S1", "S2", "S3", "S4", "S5")
MAX_STAGES = 3
# Element-wise sv tolerance of the reference CTMs.  1e-12 is unreachable on
# some states (sv_diff floor ~3e-11 with the energy fixed to 2e-16).
REF_TOL = 1e-10
_DIAG_KEYS = (
    "forward_converged",
    "forward_stationarity_residual",
    "projector_backward_used",
    "forward_iterations",
    "adjoint_residual",
)


def _ctm(chi):
    return dataclasses.replace(
        CTMConfig(chi=chi),
        conv_tol=1e-10,
        max_iter=300,
        plateau_patience=None,
        on_unconverged="warn",
    )


def _diag():
    d = _ctm_energy_ad.get_last_implicit_ad_diagnostics()
    return {k: d.get(k) for k in _DIAG_KEYS}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write(out_dir, name, passed, measured, **extra):
    rec = {"stage": name, "passed": bool(passed), "measured": measured, **extra}
    (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


def _stage(out_dir, name, fn):
    t0 = time.time()
    _log(f"{name} start")
    try:
        passed, measured, extra = fn()
    except Exception as e:  # one point must not kill a sweep
        return _write(
            out_dir,
            name,
            False,
            {"error": f"{type(e).__name__}: {e}"},
            traceback=traceback.format_exc(),
            wall=time.time() - t0,
        )
    return _write(out_dir, name, passed, measured, wall=time.time() - t0, **extra)


def _unit_direction(A, B, key):
    kA, kB = jax.random.split(key)
    dA = SymmetricTensor.random_normal(A.indices, kA)
    dB = SymmetricTensor.random_normal(B.indices, kB)
    nrm = float(np.sqrt(dA.norm() ** 2 + dB.norm() ** 2))
    return dA * (1.0 / nrm), dB * (1.0 / nrm)


def run_point(
    model, sym, D, chi, seed, out_dir, *, su_steps=400, opt_steps=200, fd_dirs=3
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    delta = MODELS[model]
    init, gate_fn = SYMS[sym]
    gate_dense, gate = xxz_gate(delta), gate_fn(delta)
    st = {}
    summary = {
        "model": model,
        "sym": sym,
        "D": D,
        "chi": chi,
        "seed": seed,
        "stopped_at": None,
        "near_degenerate": False,
    }

    def s1():
        A0, B0 = init(D, jax.random.PRNGKey(seed))
        cfg = iPEPSConfig(
            max_bond_dim=D,
            num_imaginary_steps=su_steps,
            dt=0.05,
            ctm=CTMConfig(chi=chi),
        )
        _, (A, B), _ = ipeps(gate_dense, (A0, B0), cfg, compute_energy=False)
        _, (Ad, Bd), _ = ipeps(
            gate_dense, densify_pair(A0, B0), cfg, compute_energy=False
        )
        e_s, cs = converged_energy(A, B, gate, _ctm(chi), None, REF_TOL)
        e_d, cd = converged_energy(Ad, Bd, gate_dense, _ctm(chi), None, REF_TOL)
        nonempty = all(len(t.blocks) > 0 for t in (A, B))
        st.update(A=A, B=B, dense_start=(Ad, Bd))
        return (
            abs(e_s - e_d) < 1e-8
            and nonempty,  # conv flags reported, not gated (ruling S1)
            {
                "E_sym": e_s,
                "E_dense": e_d,
                "diff": e_s - e_d,
                "conv_sym": cs,
                "conv_dense": cd,
                "blocks_nonempty": nonempty,
            },
            {},
        )

    def s2():
        seed_r = symmetric_layout_seed(st["A"], st["B"], gate, _ctm(chi))
        st["seed"] = seed_r
        near = not seed_r.cut_gap > 1e-2
        summary["near_degenerate"] = near
        return (
            seed_r.stable,
            {
                "E_cold": seed_r.energy_cold,
                "E_forced": seed_r.energy_forced,
                "stable": seed_r.stable,
                "cut_gap": seed_r.cut_gap,
                "cold_converged": seed_r.cold_converged,
                "cold_iterations": seed_r.cold_iterations,
                "layout": {str(k): v for k, v in seed_r.layout.items()},
            },
            {"near_degenerate": near},
        )

    def s3():
        loss, cache = make_2site_loss(gate, _ctm(chi), st["seed"].envs_init)
        e = float(loss((st["A"], st["B"])))
        kept = same_layout(layout_of(cache["envs"]), st["seed"].layout)
        return (
            kept and abs(e - st["seed"].energy_cold) < 1e-10,
            {
                "E": e,
                "dE_vs_seed": e - st["seed"].energy_cold,
                "layout_kept": kept,
            },
            {},
        )

    def s4():
        A, B, sd = st["A"], st["B"], st["seed"]
        loss, _ = make_2site_loss(gate, _ctm(chi), sd.envs_init)
        t0 = time.time()
        _, (gA, gB) = jax.value_and_grad(loss)((A, B))
        jax.block_until_ready(jax.tree_util.tree_leaves((gA, gB)))
        t_grad = time.time() - t0
        sym_diag = _diag()
        measured = {"first_grad_wall": t_grad, "sym_diag": sym_diag}
        # Oracle: reported, never gated (valid only without cross-sector
        # degenerate singular values; see its docstring).
        try:
            _, (oA, oB) = oracle_value_and_grad(A, B, gate, _ctm(chi), sd.envs_init)
            measured["oracle_diag"] = _diag()
            gAd, gBd = np.asarray(gA.todense()), np.asarray(gB.todense())
            measured["oracle_rel"] = {
                "A": float(np.linalg.norm(gAd - oA) / np.linalg.norm(oA)),
                "B": float(np.linalg.norm(gBd - oB) / np.linalg.norm(oB)),
            }
        except Exception as e:
            measured["oracle_error"] = f"{type(e).__name__}: {e}"
        fds, ok = [], True
        for k in range(fd_dirs):
            dA, dB = _unit_direction(A, B, jax.random.PRNGKey(1000 * (seed + 1) + k))
            r = fd_directional(
                A, B, dA, dB, gate, _ctm(chi), sd.envs_init, conv_tol=REF_TOL
            )
            gd = grad_dot((gA, gB), (dA, dB))
            tol = max(1e-6 * abs(r["fd"]), r["fd_unc"])
            plateau = bool(r["fd_unc"] < 1e-3 * abs(r["fd"]))
            ok_k = bool(abs(gd - r["fd"]) < tol) and plateau
            fds.append(
                {
                    "gd": gd,
                    "fd": r["fd"],
                    "fd_unc": r["fd_unc"],
                    "h_best": r["h_best"],
                    "plateau": plateau,
                    "pass": ok_k,
                }
            )
            ok = ok and ok_k
        measured["fd"] = fds
        measured["gate_met"] = ok
        if summary["near_degenerate"]:
            ok = True  # reported, not gated (spec §5)
        return ok, measured, {}

    def s5():
        layouts, stages = [st["seed"].layout], 0
        A, B, envs_init = st["A"], st["B"], st["seed"].envs_init
        hist_all = []
        while True:
            stages += 1
            cfg = iPEPSConfig(
                max_bond_dim=D,
                unit_cell="2site",
                ctm=_ctm(chi),
                gs_num_steps=opt_steps,
                gs_conv_criterion="grad_norm",
                gs_grad_norm_tol=1e-5,
                return_history=True,
            )
            (A, B), (env_A, env_B), E_ret, hist = optimize_gs_ad(
                gate_dense, (A, B), cfg, envs_init=envs_init
            )
            hist_all.append(hist["energies"])
            _log(f"S5 stage {stages}: optimizer done, {len(hist['energies'])} steps")
            drift = symmetric_layout_seed(A, B, gate, _ctm(chi))
            if same_layout(drift.layout, layouts[-1]) or stages >= MAX_STAGES:
                break
            layouts.append(drift.layout)
            envs_init = drift.envs_init
        _log("S5 drift seed done")
        e_cold = drift.energy_forced
        e_warm, c_warm = converged_energy(
            A, B, gate, _ctm(chi), {(0, 0): env_A, (1, 0): env_B}, REF_TOL
        )
        _log("S5 warm converged energy done; dense arm start")
        Ad, Bd = st["dense_start"]  # the dense SU state of S1
        cfg_d = dataclasses.replace(cfg, ctm=_ctm(chi))
        (Adf, Bdf), _, _, _ = optimize_gs_ad(gate_dense, (Ad, Bd), cfg_d)
        e_dense, c_dense = converged_energy(
            Adf, Bdf, gate_dense, _ctm(chi), None, REF_TOL
        )
        settled = same_layout(drift.layout, layouts[-1])
        ok = settled and drift.stable
        return (
            ok,
            {
                "stages": stages,
                "layout_settled": settled,
                "drift_stable": drift.stable,
                "E_returned": E_ret,
                "E_cold": e_cold,
                "E_warm": e_warm,
                "warm_converged": c_warm,
                "E_dense": e_dense,
                "dense_converged": c_dense,
                "dE_sym_minus_dense": min(e_cold, e_warm) - e_dense,
                "energies": hist_all,
            },
            {},
        )

    for name, fn in zip(STAGES, (s1, s2, s3, s4, s5), strict=True):
        rec = _stage(out_dir, name, fn)
        _log(f"{name} passed={rec['passed']} wall={rec.get('wall', 0):.0f}s")
        if not rec["passed"]:
            summary["stopped_at"] = name
            for later in STAGES[STAGES.index(name) + 1 :]:
                _write(out_dir, later, False, {"skipped": f"stopped at {name}"})
            break
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--sym", choices=sorted(SYMS), required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--chi", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--su-steps", type=int, default=400)
    ap.add_argument("--opt-steps", type=int, default=200)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cache_dir is not None:
        a.cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(a.cache_dir))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    print(
        json.dumps(
            run_point(
                a.model,
                a.sym,
                a.D,
                a.chi,
                a.seed,
                a.out,
                su_steps=a.su_steps,
                opt_steps=a.opt_steps,
            )
        )
    )


if __name__ == "__main__":
    main()
