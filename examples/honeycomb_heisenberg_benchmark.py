#!/usr/bin/env python3
"""Honeycomb AFM Heisenberg benchmark via 2-site iPEPS.

Tests the CG (coarse-grained) workflow: 2-site checkerboard iPEPS
on the honeycomb lattice mapped to a square grid.

The honeycomb Heisenberg model (z=3) has:
  E/site ≈ -0.5445 (QMC, Albuquerque et al.)

The 2-site checkerboard iPEPS with all 4 NN bonds gives the square
lattice energy (z=4):
  E/site ≈ -0.6694 (QMC)

This benchmark uses the 2-site C4v shared-tensor path.

Usage::

    JAX_ENABLE_X64=1 uv run python examples/honeycomb_heisenberg_benchmark.py
"""

from __future__ import annotations

import time

import jax

jax.config.update("jax_enable_x64", True)

from tenax import (  # noqa: E402
    CTMConfig,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
)


def run_benchmark(D: int, chi: int, num_steps: int, label: str):
    """Run 2-site iPEPS AD optimization and report results."""
    # Use unrotated gate: the 2-site C4v path rotates B from A internally
    gate = heisenberg_gate()

    config = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=200,
        dt=0.05,
        ctm=CTMConfig(
            chi=chi,
            max_iter=100,
            min_iter=30,
            conv_tol=1e-8,
        ),
        unit_cell="2site",
        gs_c4v=True,
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_metric_precond=True,
        gs_num_steps=num_steps,
        gs_verbose=True,
        gs_log_interval=5,
        su_init=True,
    )

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  D={D}, chi={chi}, steps={num_steps}")
    print(f"  optimizer={config.gs_optimizer}, implicit_ad={config.gs_implicit_ad}")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    result = optimize_gs_ad(gate, None, config)
    dt = time.perf_counter() - t0

    # Unpack — 2-site returns ((A, B), (env_A, env_B), E)
    (A_opt, B_opt), (env_A, env_B), E_gs = result

    print(f"\n  E/site  = {E_gs:.8f}")
    print(f"  Time    = {dt:.1f}s ({dt / num_steps:.1f}s/step)")
    print("  QMC ref = -0.6694 (square lattice)")
    return E_gs, dt


def main():
    print("Honeycomb AFM Heisenberg — 2-site iPEPS benchmark")
    print("H = sum_{<i,j>} S_i . S_j   (J=1, antiferromagnetic)")

    results = []

    # D=2, chi=8
    E, dt = run_benchmark(D=2, chi=8, num_steps=30, label="D=2, chi=8")
    results.append({"D": 2, "chi": 8, "E": E, "time": dt})

    # D=3, chi=16
    E, dt = run_benchmark(D=3, chi=16, num_steps=50, label="D=3, chi=16")
    results.append({"D": 3, "chi": 16, "E": E, "time": dt})

    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  {'D':>3}  {'chi':>4}  {'E/site':>12}  {'time':>8}  {'s/step':>8}")
    print(f"  {'-' * 3}  {'-' * 4}  {'-' * 12}  {'-' * 8}  {'-' * 8}")
    for r in results:
        steps = 30 if r["chi"] == 8 else 50
        print(
            f"  {r['D']:3d}  {r['chi']:4d}  {r['E']:12.8f}  "
            f"{r['time']:7.1f}s  {r['time'] / steps:7.1f}s"
        )
    print("\n  QMC reference (square): -0.66944 (Sandvik)")
    print("  QMC reference (honeycomb): -0.54454 (Albuquerque et al.)")


if __name__ == "__main__":
    main()
