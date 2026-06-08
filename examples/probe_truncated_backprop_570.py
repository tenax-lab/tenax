#!/usr/bin/env python3
"""Truncated backprop (TBPTT) as a #570 lever — op-count + gradient parity (CPU).

PR #589 re-localized the CTM-AD compile wall to per-sector STRUCTURAL emission in
the symmetric SVD/projector wrapper (not the decomposition). Truncated backprop
(``ctm_energy_explicit(backward_steps=K)``, issue #506) attacks a different axis:
it differentiates only the last ``K`` CTM sweeps (the rest under
``stop_gradient``), so it cuts the backward *regardless of per-sweep cost*.

This probe answers, on CPU (the D=4/χ=12 compile-*time* itself is an A100 task):

  1. **Is TBPTT a compile lever?** Count the backward (``jax.grad``) jaxpr ops vs
     ``backward_steps`` K, and vs the implicit fixed-point backward. The explicit
     differentiated sweeps are an UNROLLED checkpoint loop, so the backward graph
     should scale ~linearly in K — meaning small K is a real op-count cut. Compare
     K=1 against the implicit fused backward to see if either is fundamentally
     smaller (the implicit compiles ONE sweep-VJP in a while_loop + adjoint
     machinery; explicit-K compiles K sweep-VJPs unrolled).
  2. **Is the truncated gradient accurate?** At D=2 (cheap), compute the actual
     gradient with full backprop vs TBPTT-K and report the relative error. CTM is
     contractive, so the error should decay geometrically in K.

Usage::

    JAX_PLATFORMS=cpu uv run python examples/probe_truncated_backprop_570.py \
        --D 2 --depth 12 --K 1 2 4 8
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_energy_ad import _default_energy  # noqa: E402
from tenax.algorithms._ctm_python_loop import _make_jit_ctm_step  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS  # noqa: E402
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env  # noqa: E402
from tenax.algorithms.fermionic_ipeps import (  # noqa: E402
    FPEPSConfig,
    _build_initial_fpeps_tensor,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn  # noqa: E402
from tenax.algorithms.ipeps_config import CTMConfig  # noqa: E402


def make_site_and_gate(D: int, seed: int = 42):
    cfg = FPEPSConfig(D=D, t=1.0, V=0.0)
    A = _build_initial_fpeps_tensor(cfg, jax.random.PRNGKey(seed))
    gate = spinless_fermion_gate(cfg).todense().reshape(2, 2, 2, 2)
    return A, gate


def _count_jaxpr(jx):
    """Total jaxpr ops, recursing into sub-jaxprs."""

    def rec(jaxpr):
        jpr = getattr(jaxpr, "jaxpr", jaxpr)
        n = 0
        for eqn in jpr.eqns:
            n += 1
            for v in eqn.params.values():
                for it in v if isinstance(v, (list, tuple)) else (v,):
                    sub = getattr(it, "jaxpr", it)
                    if hasattr(sub, "eqns"):
                        n += rec(sub)
        return n

    return rec(jx)


def k_sweep_backward_ops(A, gate, chi, K):
    """Op-count of the TBPTT-K backward: grad through K differentiated CTM sweeps.

    Mirrors ``ctm_energy_explicit``'s backprop loop (the last K sweeps are the
    differentiated unit; the leading sweeps are stop_gradient'd, contributing no
    backward). Traceable (no host convergence loop), so make_jaxpr(grad) works —
    unlike the full production energy_fn, whose forward uses a host Python loop.
    """
    A_norm = A * (1.0 / (A.norm() + 1e-10))
    env0 = jax.lax.stop_gradient(initialize_ctm_tensor_env(A_norm, chi))
    envs0 = {(0, 0): env0}
    jit_step = _make_jit_ctm_step(SINGLE_SITE_NEIGHBORS)

    def loss(Ap):
        An = Ap * (1.0 / (Ap.norm() + 1e-10))
        envs = envs0
        for _ in range(K):
            envs, _eps, _smin = jit_step(
                {(0, 0): An},
                envs,
                chi=chi,
                projector_method="svd",
                renormalize=True,
                projector_backward="auto",
            )
        return _default_energy(
            {(0, 0): An}, envs, gate, [(0, 0)], SINGLE_SITE_NEIGHBORS
        )

    return _count_jaxpr(jax.make_jaxpr(jax.grad(loss))(A))


def build_loss(gate, chi, depth, *, explicit, warmup=3, backward_steps=None):
    ctm_cfg = CTMConfig(chi=chi, max_iter=depth, conv_tol=1e-4)
    energy_fn = make_ctm_energy_fn(
        neighbors=SINGLE_SITE_NEIGHBORS,
        gate=gate,
        get_ctm_cfg=lambda: ctm_cfg,
        env_cache={},
        use_explicit=explicit,
        explicit_warmup=warmup,
        explicit_steps=depth,
        explicit_backward_steps=backward_steps,
    )

    def loss_fn(A_param):
        A_norm = A_param * (1.0 / (A_param.norm() + 1e-10))
        return energy_fn({(0, 0): A_norm})

    return loss_fn


def _gnorm(g):
    return jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(g)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--chi-factor", type=int, default=3)
    ap.add_argument("--depth", type=int, default=12, help="backprop_steps (explicit).")
    ap.add_argument("--K", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument(
        "--parity",
        action="store_true",
        help="Also compute actual eager gradients and report TBPTT-vs-full error.",
    )
    args = ap.parse_args()
    chi = args.chi_factor * args.D
    A, gate = make_site_and_gate(args.D)

    print(f"# #570 truncated backprop probe | D={args.D} chi={chi} depth={args.depth}")
    print(f"# x64={jax.config.read('jax_enable_x64')} | fermionic\n")

    # ---- (1) backward op-count vs K (compile proxy; the K-sweep differentiated
    #          unit, traced directly since the full energy_fn has a host loop) ----
    print("## TBPTT backward op-count (compile proxy, trace-only)")
    print("#   (implicit fixed-point sweep-VJP baseline at this D/chi from")
    print("#    probe_bwd_subop_attribution_570: D2/chi6=63543, D4/chi12=150663)")
    n_prev = None
    for K in args.K:
        nK = k_sweep_backward_ops(A, gate, chi, K)
        delta = (
            f"  (+{nK - n_prev} vs K={args.K[args.K.index(K) - 1]})" if n_prev else ""
        )
        print(f"  TBPTT K={K:<2} backward ops : {nK:>9}{delta}")
        n_prev = nK

    # ---- (2) gradient + energy parity vs K (eager grad on the REAL energy_fn) ----
    if args.parity:
        print("\n## gradient + energy parity (eager; full backprop = reference)")
        full = build_loss(gate, chi, args.depth, explicit=True, backward_steps=None)
        e_full = full(A)
        gfull = jax.grad(full)(A)
        gfull_norm = _gnorm(gfull)
        print(
            f"  full (K={args.depth}): E={float(e_full):.10f}  |g|={float(gfull_norm):.6e}"
        )
        # implicit reference too (production default path).
        imp = build_loss(gate, chi, args.depth, explicit=False)
        e_imp = imp(A)
        print(
            f"  implicit:    E={float(e_imp):.10f}  dE_vs_full={float(e_imp - e_full):+.2e}"
        )
        for K in args.K:
            if K > args.depth:
                continue
            lf = build_loss(gate, chi, args.depth, explicit=True, backward_steps=K)
            eK = lf(A)
            gK = jax.grad(lf)(A)
            diff = jax.tree.map(lambda a, b: a - b, gK, gfull)
            rel = float(_gnorm(diff) / (gfull_norm + 1e-30))
            print(
                f"  K={K:<2}: E={float(eK):.10f}  dE={float(eK - e_full):+.2e}"
                f"  rel|g-g_full|={rel:.3e}"
            )


if __name__ == "__main__":
    main()
