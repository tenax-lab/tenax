#!/usr/bin/env python3
"""Kagome AFM Heisenberg benchmark via 3-site multisite iPEPS.

Uses the multisite CTM infrastructure with implicit AD (GMRES backward)
to optimize a 3-site unit cell iPEPS on the Kagome lattice.

The Kagome Heisenberg model has frustration and no simple Néel order:
  E/site ≈ -0.4386 (DMRG, Yan-Huse-White, Science 2011)
  E/site ≈ -0.4332 (iPEPS D=13, Liao et al., PRB 2017)

Usage::

    JAX_ENABLE_X64=1 uv run python examples/kagome_heisenberg_benchmark.py
"""

from __future__ import annotations

import time

import jax

jax.config.update("jax_enable_x64", True)

from tenax import (  # noqa: E402
    CTMConfig,
    heisenberg_gate,
    iPEPSConfig,
    kagome,
    optimize_gs_ad,
)


def run_benchmark(D: int, chi: int, num_steps: int, label: str):
    """Run 3-site Kagome iPEPS AD optimization and report results."""
    gate = heisenberg_gate()

    config = iPEPSConfig(
        max_bond_dim=D,
        ctm=CTMConfig(
            chi=chi,
            max_iter=100,
            min_iter=30,
            conv_tol=1e-8,
        ),
        unit_cell=kagome(),
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_metric_precond=True,
        gs_num_steps=num_steps,
        gs_verbose=True,
        gs_log_interval=5,
    )

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  D={D}, chi={chi}, steps={num_steps}")
    print("  optimizer=lbfgs, implicit_ad=True")
    print("  lattice=kagome (3 sites: u, v, w)")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    site_tensors, envs, E_gs = optimize_gs_ad(gate, None, config)
    dt = time.perf_counter() - t0

    print(f"\n  E/site  = {E_gs:.8f}")
    print(f"  Time    = {dt:.1f}s ({dt / num_steps:.1f}s/step)")
    print("  DMRG ref = -0.4386 (Yan-Huse-White, Science 2011)")
    print("  iPEPS ref = -0.4332 (D=13, Liao et al., PRB 2017)")
    return E_gs, dt


def main():
    print("Kagome AFM Heisenberg — 3-site iPEPS benchmark")
    print("H = sum_{<i,j>} S_i . S_j   (J=1, antiferromagnetic)")
    print("Lattice: Kagome (3 sites per unit cell)")

    results = []

    # D=2, chi=8
    E, dt = run_benchmark(D=2, chi=8, num_steps=30, label="D=2, chi=8")
    results.append({"D": 2, "chi": 8, "E": E, "time": dt})

    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  {'D':>3}  {'chi':>4}  {'E/site':>12}  {'time':>8}  {'s/step':>8}")
    print(f"  {'-' * 3}  {'-' * 4}  {'-' * 12}  {'-' * 8}  {'-' * 8}")
    for r in results:
        steps = 30
        print(
            f"  {r['D']:3d}  {r['chi']:4d}  {r['E']:12.8f}  "
            f"{r['time']:7.1f}s  {r['time'] / steps:7.1f}s"
        )
    print("\n  DMRG reference: -0.4386 (Yan-Huse-White)")
    print("  iPEPS reference (D=13): -0.4332 (Liao et al.)")


if __name__ == "__main__":
    main()
