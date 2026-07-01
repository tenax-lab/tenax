#!/usr/bin/env python3
"""Gradient-free spinless-fermion (t-V) forward-CTM pipeline + collapse probe.

Goal: SU -> forward FermionParity CTM -> energy, with NO autodiff (sidesteps
the large-D symmetric-AD wall).  The CTM+energy *machinery* is single-site and
verified sound here; the blocker is the fermionic simple update, which collapses
the state.

What this script demonstrates (all block-sparse FermionParity SymmetricTensors):

  * ``steps=0`` (fresh random init, just normalized): ``fermionic_ctm`` +
    ``compute_energy_fermionic_ctm`` return a finite, NONZERO energy -> the
    forward CTM + energy path itself is correct.
  * ``steps>=2``: the SU-updated tensor still has unit norm but the CTM energy
    collapses to exactly 0.0 (environment/RDM structural zero).
  * ``steps>=~10``: the SU output tensor norm itself collapses to 0.

Caveat this exposes: ``fpeps(H, cfg)`` runs SU then CTM and returns the energy,
but the shipped test only asserts ``jnp.isfinite(energy)`` -- and 0.0 is finite,
so the collapse is not caught.  Treat this script as a reproducer, not a
converged t-V benchmark.

Usage::

    JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR=/tmp/jaxfresh \\
        uv run python examples/fpeps_forward_ctm_e2e.py --D 2 --V 2.0
"""

from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.fermionic_ipeps import (  # noqa: E402
    FPEPSConfig,
    _absorb_lambdas,
    _fpeps_simple_update,
    _initialize_fpeps,
    _normalize_tensor,
    compute_energy_fermionic_ctm,
    fermionic_ctm,
    spinless_fermion_gate,
)


def energy_after_su(A0, H, cfg, steps):
    """Run `steps` of fermionic SU, then forward CTM + energy.

    Returns (norm_after_su, energy) — energy is None if the tensor collapsed.
    """
    if steps == 0:
        A_abs = _normalize_tensor(A0)
    else:
        A_opt, lam_h, lam_v = _fpeps_simple_update(
            A0, H, max_D=cfg.D, dt=cfg.dt, steps=steps
        )
        A_abs = _normalize_tensor(_absorb_lambdas(A_opt, lam_h, lam_v))
    n = float(A_abs.norm())
    if n == 0.0:
        return n, None
    env = fermionic_ctm(A_abs, cfg)
    return n, float(compute_energy_fermionic_ctm(A_abs, env, H))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--D", type=int, default=2)
    p.add_argument("--V", type=float, default=2.0, help="n_i n_j interaction")
    p.add_argument("--t", type=float, default=1.0, help="hopping")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--chi", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--steps-sweep",
        nargs="+",
        type=int,
        default=[0, 2, 5, 10, 20, 40],
        help="SU step counts to probe",
    )
    args = p.parse_args()

    cfg = FPEPSConfig(
        D=args.D,
        t=args.t,
        V=args.V,
        dt=args.dt,
        ctm_chi=args.chi,
        ctm_max_iter=60,
        ctm_conv_tol=1e-8,
    )
    H = spinless_fermion_gate(cfg)
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(args.seed))

    print("\nSpinless t-V forward-CTM: fermionic SU-collapse probe")
    print(f"D={args.D}  t={args.t}  V={args.V}  dt={args.dt}  chi={args.chi}")
    print(f"fresh init |A0| = {float(A0.norm()):.4f}")
    print("-" * 56)
    print(f"{'SU steps':>8}  {'|A_abs|':>9}  {'E':>12}   note")
    print("-" * 56)
    for steps in args.steps_sweep:
        n, E = energy_after_su(A0, H, cfg, steps)
        if E is None:
            print(f"{steps:>8}  {n:>9.4f}  {'-':>12}   tensor collapsed to 0")
        else:
            note = "OK (nonzero)" if abs(E) > 1e-9 else "energy collapsed to 0"
            print(f"{steps:>8}  {n:>9.4f}  {E:>12.6f}   {note}")
    print("-" * 56)
    print(
        "Expected: steps=0 nonzero (machinery OK); steps>=2 energy->0; "
        "steps>=~10 norm->0."
    )


if __name__ == "__main__":
    main()
