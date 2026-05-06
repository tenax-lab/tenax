"""Phase C.3 HZ diagnostic: instrument the first L-BFGS step at D=4 χ=16.

The HZ-default C.3 run hung for 55+ min on the very first L-BFGS step
without producing output.  HZ's line search (`hager_zhang_line_search`)
runs Phase 1 bracket expansion (up to ``max_iter=40``) and Phase 2 zoom
via interval bisection (up to ``max_bisect=50`` per ``_bisect`` call) —
worst case ~90 ``phi``/``dphi`` evaluations.  Each ``dphi`` requires a
forward CTM + JIT GMRES backward, which at D=4, χ=16 takes ~25s, so
worst-case one HZ call is ~37 min.

This diagnostic wraps HZ with per-call timing + logging so we can see:

* how long ``phi`` (forward only) vs ``dphi`` (forward + backward) take,
* whether GMRES backward is converging in ``gmres_maxiter=80``,
* how many HZ iterations the curvature condition needs at this scale,
* whether HZ exits with ``converged=False`` (curvature condition never
  satisfied — the regression case for plateaued energy landscapes).

Output: a single L-BFGS step with full HZ trace, written to
``examples/kagome_pess_multisite_phase_c3_hz_diag.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from tenax.algorithms._line_search import hager_zhang_line_search
from tenax.algorithms._pess_multisite_energy import kagome_3site_bond_gates
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import (
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    pess_simple_update,
)
from tenax.algorithms.pess_optimize import (
    _tree_real_dot,
    build_pess_loss_3site_multisite,
)


def _ctm_config(chi: int) -> CTMConfig:
    return CTMConfig(
        chi=chi,
        max_iter=30,
        min_iter=4,
        conv_tol=1e-7,
        projector_method="svd",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
        gmres_tol=1e-5,
        gmres_maxiter=80,
        gmres_restart=30,
        chi_ramp=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    args = parser.parse_args()

    print(f"=== Phase C.3 HZ diagnostic: D={args.D}, χ={args.chi} ===", flush=True)

    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=2)
    config = _ctm_config(args.chi)
    su_steps = ((0.1, 200), (0.01, 200), (0.001, 100))

    print("\n[1] SU warmstart...", flush=True)
    t0 = time.perf_counter()
    state0 = IPESSState.random(D=args.D, d=2, key=jax.random.PRNGKey(args.seed))
    state0 = pess_simple_update(state0, H, dt_schedule=list(su_steps), D_max=args.D)
    print(f"    SU done: {time.perf_counter() - t0:.2f}s", flush=True)

    loss_fn_state = build_pess_loss_3site_multisite(bond_gates, config)
    params = {
        "R_a": state0.R_a,
        "R_b": state0.R_b,
        "R_c": state0.R_c,
        "T_u": state0.T_u,
        "T_d": state0.T_d,
        "lambdas": tuple(state0.lambdas),
    }

    def _params_to_state(p: dict) -> IPESSState:
        return IPESSState(
            R_a=p["R_a"],
            R_b=p["R_b"],
            R_c=p["R_c"],
            T_u=p["T_u"],
            T_d=p["T_d"],
            lambdas=tuple(p["lambdas"]),
        )

    def loss(p: dict) -> jnp.ndarray:
        return loss_fn_state(_params_to_state(p))

    grad_fn = jax.value_and_grad(loss)

    print("\n[2] Initial energy + gradient at SU state...", flush=True)
    t0 = time.perf_counter()
    e0_val, grads = grad_fn(params)
    e0 = float(e0_val)
    t_initial_grad = time.perf_counter() - t0
    grad_norm_sq = _tree_real_dot(grads, grads)
    grad_norm = math.sqrt(max(grad_norm_sq, 1e-30))
    print(f"    e0 = {e0:.10f}", flush=True)
    print(f"    ||grad|| = {grad_norm:.6e}", flush=True)
    print(f"    First grad call: {t_initial_grad:.2f}s (incl. JIT compile)", flush=True)

    print("\n[3] Compute L-BFGS direction (first step → -gradient)...", flush=True)
    optimizer = optax.chain(
        optax.scale_by_lbfgs(memory_size=10),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)
    direction, opt_state = optimizer.update(grads, opt_state, params)
    slope = _tree_real_dot(grads, direction)
    if slope >= 0.0:
        direction = jax.tree.map(lambda g: -g, grads)
        slope = -grad_norm_sq
    print(f"    slope at α=0 = {slope:.6e}", flush=True)
    p_norm = math.sqrt(max(_tree_real_dot(params, params), 1e-30))
    d_norm = math.sqrt(max(_tree_real_dot(direction, direction), 1e-30))
    alpha0 = min(1.0, 0.1 * p_norm / d_norm)
    print(f"    alpha0 = {alpha0:.6e} (= min(1, 0.1·|p|/|d|))", flush=True)

    print("\n[4] Hager-Zhang line search with per-call instrumentation...", flush=True)
    phi_calls: list[dict] = []
    dphi_calls: list[dict] = []

    def _phi(a: float) -> float:
        t0 = time.perf_counter()
        trial = jax.tree.map(lambda p, d: p + a * d, params, direction)
        val = float(loss(trial))
        dt = time.perf_counter() - t0
        phi_calls.append({"alpha": a, "phi": val, "dt": dt})
        print(f"      phi({a:.4e}) = {val:.10f}  [{dt:.2f}s]", flush=True)
        return val

    def _dphi(a: float) -> float:
        t0 = time.perf_counter()
        trial = jax.tree.map(lambda p, d: p + a * d, params, direction)
        _, g = grad_fn(trial)
        dval = _tree_real_dot(g, direction)
        dt = time.perf_counter() - t0
        dphi_calls.append({"alpha": a, "dphi": dval, "dt": dt})
        print(f"      dphi({a:.4e}) = {dval:.6e}  [{dt:.2f}s]", flush=True)
        return dval

    t_hz0 = time.perf_counter()
    alpha_final, phi_final, converged = hager_zhang_line_search(
        _phi,
        _dphi,
        e0,
        slope,
        alpha_init=alpha0,
        rho=1.5,
        max_step=2.0 * alpha0,
        energy_bound=max(2.0, 2.0 * abs(e0)),
    )
    t_hz = time.perf_counter() - t_hz0

    print("\n[5] HZ result:", flush=True)
    print(f"    alpha_final = {alpha_final:.6e}", flush=True)
    print(f"    phi_final   = {phi_final:.10f}", flush=True)
    print(f"    converged   = {converged}", flush=True)
    print(f"    Δ = phi_final - e0 = {phi_final - e0:+.6e}", flush=True)
    print(f"    HZ wall time = {t_hz:.1f}s", flush=True)
    print(
        f"    {len(phi_calls)} phi calls, {len(dphi_calls)} dphi calls",
        flush=True,
    )
    if phi_calls:
        avg_phi = sum(c["dt"] for c in phi_calls) / len(phi_calls)
        print(f"    avg phi call: {avg_phi:.2f}s", flush=True)
    if dphi_calls:
        avg_dphi = sum(c["dt"] for c in dphi_calls) / len(dphi_calls)
        print(f"    avg dphi call: {avg_dphi:.2f}s", flush=True)

    result = {
        "D": args.D,
        "chi": args.chi,
        "seed": args.seed,
        "e0": e0,
        "grad_norm": grad_norm,
        "slope": slope,
        "alpha0": alpha0,
        "first_grad_call_seconds": t_initial_grad,
        "hz_wall_seconds": t_hz,
        "alpha_final": alpha_final,
        "phi_final": phi_final,
        "converged": bool(converged),
        "delta_phi": phi_final - e0,
        "phi_calls": phi_calls,
        "dphi_calls": dphi_calls,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
