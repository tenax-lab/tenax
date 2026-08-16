#!/usr/bin/env python3
"""Gradient-free spinless-fermion (t-V) forward-CTM pipeline.

Goal: 2-site checkerboard simple update -> forward FermionParity split-CTM ->
energy, with NO autodiff (sidesteps the large-D symmetric-AD wall).

This script started life as the reproducer for #878, where the fermionic simple
update drove the state to exactly 0.0 by step 10 and ``fpeps()`` reported a
perfectly finite energy of 0.0 on the corpse.  That is fixed, so it is now a
sweep over SU step counts that prints what the run actually produced:

  * ``E`` -- energy per site from the coupled ``(env_A, env_B)`` fixed point.
  * ``gap`` -- :func:`~tenax.sublattice_gap`, the trace distance between the two
    sublattices' one-site reduced density matrices.  For spinless fermions this
    is exactly ``|<n_A> - <n_B>|``, the charge-density-wave order parameter, so
    expect it to grow with ``V``.  A nonzero value means the returned pair is
    carrying real charge order; a ~0 does **not** prove a single tensor would
    do, because a one-body probe cannot see two-site (e.g. dimer) order.

Two things this will show you that are **not** bugs in the script:

  * The sweep is **seed-dependent**.  ``--seed 3`` (the default) survives at
    D=2; ``--seed 1`` does not -- its bond spectrum is already ``[1, 1.0e-02]``
    after two steps and ``[1, 3.5e-07]`` after five, and everything downstream
    reads 0 from then on.  Over seeds 0-4 at 600 steps the surviving fraction is
    4/5 at D=2, 2/5 at D=3, 4/5 at D=4, 4/5 at D=6.
  * The **energy is not certified** (#392).  ``H`` has no chemical potential, so
    the empty state and the fully polarised checkerboard are both exact ``E = 0``
    eigenstates -- fixed points of imaginary time that are not the ground state
    -- and the sweep settles on them.  Read ``gap`` to see which one you got;
    do not read ``E`` as a variational energy.

The collapse guards themselves live in ``tests/test_fpeps_878_su_collapse.py``
and assert on the bond *spectrum*, never the norm -- the update normalises last,
so ``|A|`` reads a healthy 1.0 right up to the step where it is exactly 0.

Usage::

    JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR=/tmp/jaxfresh \\
        uv run python examples/fpeps_forward_ctm_e2e.py --D 2 --V 2.0
"""

from __future__ import annotations

import argparse
import dataclasses

import jax

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.fermionic_ipeps import (  # noqa: E402
    FPEPSConfig,
    fpeps,
    spinless_fermion_gate,
    sublattice_gap,
)


def run(H, cfg, steps, key):
    """Run `steps` of 2-site fermionic SU, then forward split-CTM + energy.

    Returns ``(energy, gap)``.
    """
    cfg = dataclasses.replace(cfg, num_imaginary_steps=steps)
    energy, (A, B), (env_A, env_B) = fpeps(H, cfg, key=key)
    return energy, sublattice_gap(A, B, env_A, env_B)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--D", type=int, default=2)
    p.add_argument("--V", type=float, default=2.0, help="n_i n_j interaction")
    p.add_argument("--t", type=float, default=1.0, help="hopping")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--chi", type=int, default=8)
    # Seed 3 survives at D=2; seed 1 does not.  See the module docstring -- this
    # is a documented property of the sweep, not a choice that hides one.
    p.add_argument("--seed", type=int, default=3)
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
    key = jax.random.PRNGKey(args.seed)

    print("\nSpinless t-V forward split-CTM, 2-site checkerboard")
    print(f"D={args.D}  t={args.t}  V={args.V}  dt={args.dt}  chi={args.chi}")
    print("-" * 56)
    print(f"{'SU steps':>8}  {'E':>12}  {'CDW gap':>9}")
    print("-" * 56)
    for steps in args.steps_sweep:
        energy, gap = run(H, cfg, steps, key)
        print(f"{steps:>8}  {energy:>12.6f}  {gap:>9.4f}")
    print("-" * 56)
    print("gap > 0 => real charge order.  gap ~ 0 => no CHARGE order; it does")
    print("NOT mean one tensor would do -- the gap is a one-body probe, and a")
    print("dimerised state reads ~0 while still needing two sites.")


if __name__ == "__main__":
    main()
