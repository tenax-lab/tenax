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

from benchmarks.symmetric_ad.codehash import code_hash  # noqa: E402
from tenax import CTMConfig, ipeps, iPEPSConfig, optimize_gs_ad  # noqa: E402
from tenax.algorithms import _ctm_energy_ad  # noqa: E402
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    CHECKERBOARD_NEIGHBORS,
)
from tenax.algorithms._ctm_tensor_energy import (  # noqa: E402
    _rdm1x2_tensor_2site,
    _rdm2x1_tensor_2site,
)
from tenax.algorithms._symmetric_ad_validation import (  # noqa: E402
    converged_energy,
    densify_pair,
    fd_directional,
    grad_dot,
    make_2site_loss,
    oracle_value_and_grad,
)
from tenax.algorithms.ctm_layout import (  # noqa: E402
    _converge,
    _corner_values,
    energy_2site,
    layout_of,
    normalized_pair,
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
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs  # noqa: E402
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
# S4 measures the gradient tighter than the optimizer needs: with the default
# gmres_tol=1e-6 the adjoint residual (4e-6 on the Z2 chi=8 point) alone put
# one direction 1.6e-6 (relative) off the converged FD.
GRAD_GMRES_TOL = 1e-9
_DIAG_KEYS = (
    "forward_converged",
    "forward_stationarity_residual",
    "projector_backward_used",
    "forward_iterations",
    "adjoint_residual",
)


def _ctm(chi, **kw):
    return dataclasses.replace(
        CTMConfig(chi=chi),
        conv_tol=1e-10,
        max_iter=300,
        plateau_patience=None,
        on_unconverged="warn",
        **kw,
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


def gate_s2(stable, cold_converged, cut_gap):
    """S2 passes: stable seed, converged cold CTM, a real (non-NaN) cut gap."""
    return bool(stable and cold_converged and not np.isnan(cut_gap))


def gate_s4_direction(gd, fd, fd_unc, near_degenerate=False):
    """One FD direction (R15).  Returns (finite, plateau, within, ok).

    ``within``: |gd - fd| < max(1e-6 |fd|, fd_unc).  ``plateau``:
    fd_unc < 1e-3 |fd|.  Non-finite values always fail; a near-degenerate
    point reports ``within``/``plateau`` without gating them."""
    finite = all(np.isfinite(x) for x in (gd, fd, fd_unc))
    if not finite:
        return False, False, False, False
    within = bool(abs(gd - fd) < max(1e-6 * abs(fd), fd_unc))
    plateau = bool(fd_unc < 1e-3 * abs(fd))
    return True, plateau, within, bool(near_degenerate or (within and plateau))


def gate_s5(
    *, settled, drift_stable, e_start, e_ret, e_cold, e_warm, warm_converged, e_dense
):
    """S5 (R18).  Returns (ok, parts)."""
    parts = {
        "settled": bool(settled),
        "drift_stable": bool(drift_stable),
        "finite": all(
            np.isfinite(x) for x in (e_start, e_ret, e_cold, e_warm, e_dense)
        ),
        "descended": bool(e_ret <= e_start + 1e-10),
        "agree": bool(abs(e_cold - e_warm) < 1e-10),
        "warm_converged": bool(warm_converged),
        "dense_ok": bool(abs(e_cold - e_dense) < 1e-8),
    }
    return all(parts.values()), parts


def gate_s1(diff, viol, nonempty, viol_tol=1e-6):
    """S1 (ruling R25).

    The dense reference CTM of the densified symmetric SU state is a fixed
    point of the dense CTM, which need not keep the symmetry.  Where its
    environment keeps it (``viol < viol_tol``) the energies must agree to
    1e-8.  Where it broke it, it is a different fixed point: the energy
    comparison is reported, not gated.  NaN ``diff`` or ``viol`` and empty
    blocks always fail."""
    if not nonempty or not np.isfinite(diff) or not np.isfinite(viol):
        return False
    if viol < viol_tol:
        return bool(abs(diff) < 1e-8)
    return True


def _dense_converged_env(A, B, gate, ctm_cfg, conv_tol):
    """Dense tight eager CTM at dense ``(A, B)`` (same config as
    ``converged_energy``).  Returns ``(energy, converged, A_n, B_n, envs)``."""
    tight = dataclasses.replace(
        ctm_cfg,
        conv_tol=conv_tol,
        max_iter=1000,
        plateau_patience=None,
        on_unconverged="warn",
    )
    A_n, B_n = normalized_pair(A, B)
    envs, info = python_loop_ctm_converge(
        {(0, 0): A_n, (1, 0): B_n},
        CHECKERBOARD_NEIGHBORS,
        **ctm_converge_kwargs(tight),
    )
    return energy_2site(A_n, B_n, envs, gate), bool(info.converged), A_n, B_n, envs


def charge_violation(A_n, B_n, envs):
    """Charge-violating weight ``||P(rho)||_F / ||rho||_F`` of the 2-site RDMs.

    ``rho[a, b, c, d]`` is the RDM of one bond (ket ``a, b``; bra ``c, d``).
    ``P`` keeps the elements whose charge is not conserved, i.e. whose net
    charge ``sym.net_charge([qa, qb, qc, qd], [f, f, -f, -f])`` (``f`` the flow
    of A's ``phys`` leg) is not the identity; the arithmetic is the
    symmetry's own (Z_n wraps).  All four bonds of the checkerboard cell
    (h AB, h BA, v AB, v BA) enter, as one Frobenius norm over their stack.
    They are used one by one, not through ``compute_energy_ctm_tensor_2site``
    (a sum over the four bonds, divided by two): in that sum an order
    parameter that is staggered between A-B and B-A bonds would cancel.
    An environment that keeps the symmetry gives exactly 0."""
    phys = next(i for i in A_n.indices if i.label == "phys")
    sym, f = phys.symmetry, int(phys.flow)
    q = [int(c) for c in np.asarray(phys.charges)]
    d = len(q)
    mask = np.zeros((d, d, d, d), dtype=bool)  # True: charge NOT conserved
    for a in range(d):
        for b in range(d):
            for c in range(d):
                for e in range(d):
                    mask[a, b, c, e] = not sym.is_conserved(
                        [q[a], q[b], q[c], q[e]], [f, f, -f, -f]
                    )
    ea, eb = envs[(0, 0)], envs[(1, 0)]
    rdms = [
        _rdm2x1_tensor_2site(A_n, B_n, ea, eb),
        _rdm2x1_tensor_2site(B_n, A_n, eb, ea),
        _rdm1x2_tensor_2site(A_n, B_n, ea, eb),
        _rdm1x2_tensor_2site(B_n, A_n, eb, ea),
    ]
    stack = np.stack([np.asarray(r) for r in rdms])
    num = np.linalg.norm(stack[:, mask])
    return float(num / np.linalg.norm(stack))


def _all_sectors_populated(t):
    """Every sector of every leg carries at least one block."""
    for i, idx in enumerate(t.indices):
        used = {k[i] for k in t.blocks}
        if not {int(c) for c in np.asarray(idx.charges)} <= used:
            return False
    return True


def _chi_binding(A, B, chi, tol=1e-12):
    """Is the truncation chi binding?  Per corner: more than chi singular
    values above ``tol * max`` at chi + 4.  ``any`` is the point-level flag."""
    A_n, B_n = normalized_pair(A, B)
    big = dataclasses.replace(_ctm(chi), chi=chi + 4)
    envs, _ = _converge({(0, 0): A_n, (1, 0): B_n}, big, None)
    per = {}
    for c, env in envs.items():
        for f in ("C1", "C2", "C3", "C4"):
            s = _corner_values(getattr(env, f))
            n = int(np.sum(s > tol * s[0])) if len(s) else 0
            per[f"{c}{f}"] = {"rank": n, "binding": n > chi}
    return {"corners": per, "any": any(v["binding"] for v in per.values())}


def _unit_direction(A, B, key):
    kA, kB = jax.random.split(key)
    dA = SymmetricTensor.random_normal(A.indices, kA)
    dB = SymmetricTensor.random_normal(B.indices, kB)
    nrm = float(np.sqrt(dA.norm() ** 2 + dB.norm() ** 2))
    return dA * (1.0 / nrm), dB * (1.0 / nrm)


def run_point(
    model,
    sym,
    D,
    chi,
    seed,
    out_dir,
    *,
    su_steps=200,
    opt_steps=200,
    fd_dirs=3,
):
    assert fd_dirs >= 1, "fd_dirs must be at least 1"
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
        "s1_symmetry_broken": False,
        "vacuous": False,
        "error_stage": None,
        "complete": False,
        "code_hash": code_hash(),
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
        e_s, cs = converged_energy(A, B, gate, _ctm(chi), None, REF_TOL)
        # Same-state dense reference: the dense CTM of the densified (A, B).
        Ad0, Bd0 = densify_pair(A, B)
        e_d, cd, Ad_n, Bd_n, envs_d = _dense_converged_env(
            Ad0, Bd0, gate_dense, _ctm(chi), REF_TOL
        )
        viol = charge_violation(Ad_n, Bd_n, envs_d)
        nonempty = all(_all_sectors_populated(t) for t in (A, B))
        # Dense SU from the same start: a different state at delta=0.3 (it
        # drifts to a U(1)-broken one).  Recorded only, never gated.
        _, (Asu, Bsu), _ = ipeps(
            gate_dense, densify_pair(A0, B0), cfg, compute_energy=False
        )
        e_dsu, cdsu = converged_energy(
            Asu, Bsu, gate_dense, _ctm(chi), None, REF_TOL
        )
        diff = e_s - e_d
        ok = gate_s1(diff, viol, nonempty)
        broken = bool(np.isfinite(viol) and viol >= 1e-6)
        summary["s1_symmetry_broken"] = broken
        st.update(A=A, B=B)
        return (
            ok,  # conv flags reported, not gated (ruling S1)
            {
                "E_sym": e_s,
                "E_dense": e_d,
                "diff": diff,
                "conv_sym": cs,
                "conv_dense": cd,
                "viol": viol,
                "symmetry_broken": broken,
                "E_dense_su": e_dsu,
                "diff_su": e_s - e_dsu,
                "conv_dense_su": cdsu,
                "blocks_nonempty": nonempty,
            },
            {},
        )

    def s2():
        seed_r = symmetric_layout_seed(st["A"], st["B"], gate, _ctm(chi))
        st["seed"] = seed_r
        nan_gap = bool(np.isnan(seed_r.cut_gap))
        near = (not nan_gap) and not seed_r.cut_gap > 1e-2
        binding = _chi_binding(st["A"], st["B"], chi)
        summary["near_degenerate"] = near
        summary["vacuous"] = not binding["any"]
        return (
            gate_s2(seed_r.stable, seed_r.cold_converged, seed_r.cut_gap),
            {
                "E_cold": seed_r.energy_cold,
                "E_forced": seed_r.energy_forced,
                "stable": seed_r.stable,
                "cut_gap": seed_r.cut_gap,
                "cold_converged": seed_r.cold_converged,
                "cold_iterations": seed_r.cold_iterations,
                "chi_binding": binding,
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
        loss, _ = make_2site_loss(
            gate, _ctm(chi, gmres_tol=GRAD_GMRES_TOL), sd.envs_init
        )
        t0 = time.time()
        _, (gA, gB) = jax.value_and_grad(loss)((A, B))
        jax.block_until_ready(jax.tree_util.tree_leaves((gA, gB)))
        t_grad = time.time() - t0
        sym_diag = _diag()
        flow = sym_diag.get("projector_backward_used") == "flow"
        measured = {
            "first_grad_wall": t_grad,
            "sym_diag": sym_diag,
            "flow_backward": flow,
            "gmres_tol": GRAD_GMRES_TOL,
        }
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
            finite, plateau, within, ok_k = gate_s4_direction(
                gd, r["fd"], r["fd_unc"], summary["near_degenerate"]
            )
            fds.append(
                {
                    "gd": gd,
                    "fd": r["fd"],
                    "fd_unc": r["fd_unc"],
                    "h_best": r["h_best"],
                    "plateau": plateau,
                    "within_tol": within,
                    "finite": finite,
                    "pass": ok_k,
                }
            )
            ok = ok and ok_k
        measured["fd"] = fds
        measured["gate_met"] = ok and flow
        return ok and flow, measured, {}

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
        t_d = time.time()
        Ad, Bd = densify_pair(A, B)
        e_dense, c_dense = converged_energy(
            Ad, Bd, gate_dense, _ctm(chi), None, REF_TOL
        )
        wall_dense = time.time() - t_d
        _log("S5 dense final energy done")
        e_start = hist_all[0][0]
        e_sym = e_cold
        settled = same_layout(drift.layout, layouts[-1])
        ok, parts = gate_s5(
            settled=settled,
            drift_stable=drift.stable,
            e_start=e_start,
            e_ret=E_ret,
            e_cold=e_cold,
            e_warm=e_warm,
            warm_converged=c_warm,
            e_dense=e_dense,
        )
        return (
            ok,
            {
                "stages": stages,
                "layout_settled": settled,
                "gate_parts": parts,
                "E_start": e_start,
                "E_returned": E_ret,
                "E_cold": e_cold,
                "E_warm": e_warm,
                "cold_warm_diff": e_cold - e_warm,
                "warm_converged": c_warm,
                "E_dense_final": e_dense,
                "dense_converged": c_dense,
                "dE_sym_minus_dense": e_sym - e_dense,
                "dense_final_wall": wall_dense,
                "energies": hist_all,
            },
            {},
        )

    for name, fn in zip(STAGES, (s1, s2, s3, s4, s5), strict=True):
        rec = _stage(out_dir, name, fn)
        _log(f"{name} passed={rec['passed']} wall={rec.get('wall', 0):.0f}s")
        if not rec["passed"]:
            summary["stopped_at"] = name
            if "error" in rec["measured"]:
                summary["error_stage"] = name
            for later in STAGES[STAGES.index(name) + 1 :]:
                _write(out_dir, later, False, {"skipped": f"stopped at {name}"})
            break
    summary["complete"] = True
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--sym", choices=sorted(SYMS), required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--chi", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--su-steps", type=int, default=200)
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
