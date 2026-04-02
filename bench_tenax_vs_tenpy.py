#!/usr/bin/env python3
"""Head-to-head benchmark: Tenax vs TeNPy for Heisenberg DMRG / iDMRG.

Both libraries use U(1) symmetry (Sz conservation).

Usage::

    uv run python bench_tenax_vs_tenpy.py
"""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ── Exact reference ──────────────────────────────────────────────────
E_BETHE = -np.log(2) + 0.25  # E/site = -0.4431…  (Bethe ansatz)


# =====================================================================
#  TeNPy helpers
# =====================================================================
def tenpy_finite_dmrg(L: int, chi: int, num_sweeps: int = 10) -> dict:
    """Run finite DMRG with TeNPy on spin-1/2 Heisenberg chain."""
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
    from tenpy.models.xxz_chain import XXZChain
    from tenpy.networks.mps import MPS

    model_params = {
        "L": L,
        "Jxx": 1.0,
        "Jz": 1.0,
        "hz": 0.0,
        "bc_MPS": "finite",
        "conserve": "Sz",
    }
    model = XXZChain(model_params)

    psi = MPS.from_lat_product_state(model.lat, [["up"], ["down"]])

    dmrg_params = {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-14},
        "max_sweeps": num_sweeps,
        "max_E_err": 1e-12,
        "N_sweeps_check": 1,
        "mixer": None,
    }
    eng = TwoSiteDMRGEngine(psi, model, dmrg_params)

    t0 = time.perf_counter()
    E, _ = eng.run()
    elapsed = time.perf_counter() - t0

    return {"energy": float(E), "time_s": elapsed, "sweeps": eng.sweeps}


def tenpy_idmrg(chi: int, max_iters: int = 100) -> dict:
    """Run iDMRG with TeNPy on infinite spin-1/2 Heisenberg chain."""
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
    from tenpy.models.xxz_chain import XXZChain
    from tenpy.networks.mps import MPS

    model_params = {
        "L": 2,
        "Jxx": 1.0,
        "Jz": 1.0,
        "hz": 0.0,
        "bc_MPS": "infinite",
        "conserve": "Sz",
    }
    model = XXZChain(model_params)

    psi = MPS.from_lat_product_state(model.lat, [["up"], ["down"]])

    dmrg_params = {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-14},
        "max_sweeps": max_iters,
        "max_E_err": 1e-10,
        "N_sweeps_check": 1,
        "mixer": None,
    }
    eng = TwoSiteDMRGEngine(psi, model, dmrg_params)

    t0 = time.perf_counter()
    E, _ = eng.run()
    elapsed = time.perf_counter() - t0

    return {
        "energy_per_site": float(E),
        "time_s": elapsed,
        "sweeps": eng.sweeps,
    }


# =====================================================================
#  Tenax helpers
# =====================================================================
def tenax_finite_dmrg(L: int, chi: int, num_sweeps: int = 10) -> dict:
    """Run finite DMRG with Tenax on spin-1/2 Heisenberg chain (U(1))."""
    from tenax import DMRGConfig, build_mpo_heisenberg, dmrg
    from tenax.core.mps import FiniteMPS
    from tenax.core.symmetry import U1Symmetry

    mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0, hz=0.0)
    mps = FiniteMPS.random(
        L,
        d=2,
        chi=min(chi, 4),
        key=jax.random.PRNGKey(0),
        symmetric=True,
        symmetry=U1Symmetry(),
        target_charge=0,
    )

    config = DMRGConfig(
        max_bond_dim=chi,
        num_sweeps=num_sweeps,
        convergence_tol=1e-12,
        lanczos_max_iter=30,
        two_site=True,
        target_charge=0,
        verbose=False,
    )

    t0 = time.perf_counter()
    result = dmrg(mpo, mps, config)
    elapsed = time.perf_counter() - t0

    return {
        "energy": float(result.energy),
        "time_s": elapsed,
        "converged": result.converged,
    }


def tenax_idmrg(chi: int, max_iters: int = 100) -> dict:
    """Run iDMRG with Tenax on infinite spin-1/2 Heisenberg chain (U(1))."""
    from tenax import build_bulk_mpo_heisenberg_symmetric, idmrg, iDMRGConfig

    W = build_bulk_mpo_heisenberg_symmetric(Jz=1.0, Jxy=1.0, hz=0.0)

    config = iDMRGConfig(
        max_bond_dim=chi,
        max_iterations=max_iters,
        convergence_tol=1e-10,
        lanczos_max_iter=30,
        two_site=True,
        verbose=False,
    )

    t0 = time.perf_counter()
    result = idmrg(W, config=config, dtype=jnp.float64)
    elapsed = time.perf_counter() - t0

    return {
        "energy_per_site": float(result.energy_per_site),
        "time_s": elapsed,
        "converged": result.converged,
    }


# =====================================================================
#  Runner
# =====================================================================
def run_finite_dmrg_benchmark():
    print("\n" + "=" * 70)
    print("  FINITE DMRG: Heisenberg chain, U(1) symmetry")
    print("=" * 70)

    cases = [
        {"L": 20, "chi": 32, "sweeps": 10},
        {"L": 40, "chi": 64, "sweeps": 10},
        {"L": 80, "chi": 128, "sweeps": 10},
    ]

    results = []
    for c in cases:
        L, chi, sweeps = c["L"], c["chi"], c["sweeps"]
        print(f"\n--- L={L}, chi={chi}, sweeps={sweeps} ---")

        print("  TeNPy ... ", end="", flush=True)
        tp = tenpy_finite_dmrg(L, chi, sweeps)
        print(f"E = {tp['energy']:.10f}  t = {tp['time_s']:.2f}s")

        print("  Tenax ... ", end="", flush=True)
        tx = tenax_finite_dmrg(L, chi, sweeps)
        print(f"E = {tx['energy']:.10f}  t = {tx['time_s']:.2f}s")

        ratio = tx["time_s"] / tp["time_s"] if tp["time_s"] > 0 else float("inf")
        print(f"  Ratio (Tenax/TeNPy): {ratio:.2f}x")

        results.append(
            {
                "L": L,
                "chi": chi,
                "sweeps": sweeps,
                "tenpy_energy": tp["energy"],
                "tenpy_time_s": tp["time_s"],
                "tenax_energy": tx["energy"],
                "tenax_time_s": tx["time_s"],
                "ratio": ratio,
            }
        )

    return results


def run_idmrg_benchmark():
    print("\n" + "=" * 70)
    print("  iDMRG: Infinite Heisenberg chain, U(1) symmetry")
    print(f"  Bethe ansatz reference: E/site = {E_BETHE:.10f}")
    print("=" * 70)

    chi_list = [32, 64, 128, 256]
    max_iters = 100

    results = []
    for chi in chi_list:
        print(f"\n--- chi={chi}, max_iters={max_iters} ---")

        print("  TeNPy ... ", end="", flush=True)
        tp = tenpy_idmrg(chi, max_iters)
        print(f"E/site = {tp['energy_per_site']:.10f}  t = {tp['time_s']:.2f}s")

        print("  Tenax ... ", end="", flush=True)
        tx = tenax_idmrg(chi, max_iters)
        print(f"E/site = {tx['energy_per_site']:.10f}  t = {tx['time_s']:.2f}s")

        ratio = tx["time_s"] / tp["time_s"] if tp["time_s"] > 0 else float("inf")
        print(f"  Ratio (Tenax/TeNPy): {ratio:.2f}x")

        results.append(
            {
                "chi": chi,
                "max_iters": max_iters,
                "tenpy_energy_per_site": tp["energy_per_site"],
                "tenpy_time_s": tp["time_s"],
                "tenax_energy_per_site": tx["energy_per_site"],
                "tenax_time_s": tx["time_s"],
                "ratio": ratio,
                "bethe_exact": E_BETHE,
            }
        )

    return results


def main():
    print("Tenax vs TeNPy benchmark: spin-1/2 Heisenberg AFM")
    print("Both use U(1) (Sz) symmetry, 2-site DMRG, float64")

    finite_results = run_finite_dmrg_benchmark()
    idmrg_results = run_idmrg_benchmark()

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print("\nFinite DMRG:")
    print(f"  {'L':>4} {'chi':>4} | {'TeNPy (s)':>10} {'Tenax (s)':>10} {'Ratio':>8}")
    print(f"  {'-' * 4} {'-' * 4} | {'-' * 10} {'-' * 10} {'-' * 8}")
    for r in finite_results:
        print(
            f"  {r['L']:>4} {r['chi']:>4} | "
            f"{r['tenpy_time_s']:>10.2f} {r['tenax_time_s']:>10.2f} "
            f"{r['ratio']:>7.2f}x"
        )

    print("\niDMRG:")
    print(f"  {'chi':>4} | {'TeNPy (s)':>10} {'Tenax (s)':>10} {'Ratio':>8}")
    print(f"  {'-' * 4} | {'-' * 10} {'-' * 10} {'-' * 8}")
    for r in idmrg_results:
        print(
            f"  {r['chi']:>4} | "
            f"{r['tenpy_time_s']:>10.2f} {r['tenax_time_s']:>10.2f} "
            f"{r['ratio']:>7.2f}x"
        )

    all_results = {
        "finite_dmrg": finite_results,
        "idmrg": idmrg_results,
    }
    with open("bench_tenax_vs_tenpy_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to bench_tenax_vs_tenpy_results.json")


if __name__ == "__main__":
    main()
