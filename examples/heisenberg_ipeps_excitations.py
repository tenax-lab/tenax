#!/usr/bin/env python3
"""iPEPS excitation spectrum and spectral function of the 2D Heisenberg model.

Computes the excitations of the spin-1/2 antiferromagnetic Heisenberg model on
the infinite square lattice at X = (pi, 0), (0, pi) and M = (pi, pi), with the
iPEPS excitation ansatz (Ponsioen, Assaad & Corboz, SciPost Phys. 12, 006,
2022) and its complete momentum sum (Ponsioen, Hasik & Corboz, PRB 108,
195111, 2023).

The workflow is:

1. Build the sublattice-rotated Heisenberg gate.  A 1x1 cell puts one tensor
   on both sublattices, so the Neel state needs the rotation.
2. Find the D=2 ground state with ``optimize_gs_ad_chi_schedule``, in the
   recommended configuration of the AD guide: L-BFGS from a simple-update
   start, at chi=8 and then at chi=16.
3. At each momentum, ``compute_spectrum`` gives the excitation energies, the
   spectral weights of S^y and the static structure factor s(k).
4. ``spectral_function`` broadens the weights to S(k, omega).

The sublattice rotation leaves S^y unchanged, so its momenta are the lattice
momenta.  The example passes i S^y, the real form of S^y, which has the same
weights.

``compute_excitations`` (deprecated, #1052) is not used: it truncates the
momentum sum, so its low energies are not reliable.

Measured on 12 CPU cores (2026-11-18): the ground state takes 17 s and
reaches E/site = -0.6602310935.  Each momentum takes about 280 s at chi=16.
At X and (0, pi), the lowest omega is 2.2930 and carries 93% of
s(k) = 0.20399; at M, it is 0.6487 and carries 96% of s(k) = 0.80858.
One chi cannot show convergence: run again with another ``SPECTRUM_CHI`` and
compare.

Usage::

    uv run python examples/heisenberg_ipeps_excitations.py
"""

from __future__ import annotations

import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax import (  # noqa: E402
    CTMConfig,
    SpectrumConfig,
    SpectrumResult,
    compute_spectrum,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad_chi_schedule,
    spectral_function,
    sublattice_rotate_gate,
)

# i S^y: real, with the spectral weights of S^y.
SY_REAL = np.array([[0.0, 0.5], [-0.5, 0.0]])
MOMENTA = {"X": (np.pi, 0.0), "(0,pi)": (0.0, np.pi), "M": (np.pi, np.pi)}
CHI_SCHEDULE = [(8, 30), (16, 30)]  # (chi, L-BFGS steps)
SPECTRUM_CHI = 16

# ---------------------------------------------------------------------------
# Ground state optimization
# ---------------------------------------------------------------------------


def find_ground_state(gate, D: int = 2):
    """Find the iPEPS ground state with the recommended AD configuration.

    Returns:
        ``(A_opt, E_gs)`` -- optimized site tensor and ground state energy
        per site.
    """
    config = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=200,  # the simple-update start
        dt=0.05,
        ctm=CTMConfig(chi=16, max_iter=80, conv_tol=1e-8, projector_method="qr"),
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_metric_precond=True,
        gs_c4v=True,
        su_init=True,
    )

    print(f"  D={D}, (chi, L-BFGS steps) = {CHI_SCHEDULE}")

    t0 = time.perf_counter()
    A_opt, _, E_gs = optimize_gs_ad_chi_schedule(gate, None, config, CHI_SCHEDULE)
    dt = time.perf_counter() - t0

    print(f"  E/site = {float(E_gs):.10f}  (time: {dt:.1f}s)")
    return A_opt, float(E_gs)


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------


def print_spectrum(name: str, result: SpectrumResult) -> None:
    """Print s(k), the modes with the largest weights, and S(k, omega)."""
    omega = np.asarray(result.omega)
    weights = np.asarray(result.weights["Sy"])
    s_k = result.structure_factor["Sy"]

    print(f"\n  k = {name}:  s(k) = {s_k:.5f},  lowest omega = {omega[0]:.4f}")
    print(
        f"    sum-rule deficit = {result.sum_rule_deficit['Sy']:.1e}, "
        f"negative modes = {result.n_negative_modes}"
    )
    print(f"    {'omega':>8s}  {'weight':>8s}  {'weight/s(k)':>11s}")
    for i in np.argsort(weights)[::-1][:3]:
        print(f"    {omega[i]:8.4f}  {weights[i]:8.4f}  {weights[i] / s_k:11.4f}")

    grid = np.linspace(0.0, 5.0, 501)
    S = spectral_function(result, "Sy", grid, eta=0.1)
    print(
        f"    S(k, omega), eta=0.1: maximum {S.max():.3f} "
        f"at omega = {grid[np.argmax(S)]:.2f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("iPEPS excitation spectrum: 2D Heisenberg model")
    print("H = sum_{<i,j>} S_i . S_j   (J=1, antiferromagnetic)")
    print("Method: Ponsioen, Hasik & Corboz, PRB 108, 195111 (2023)")

    # A 1x1 cell puts one tensor on both sublattices: the Neel state needs
    # the sublattice-rotated gate.
    gate = sublattice_rotate_gate(heisenberg_gate())

    # Step 1: Find the ground state
    print(f"\n{'=' * 60}")
    print("  Step 1: Ground state optimization")
    print(f"{'=' * 60}")
    A_opt, _ = find_ground_state(gate)

    # Step 2: Spectrum at each momentum
    print(f"\n{'=' * 60}")
    print(f"  Step 2: Spectrum at X, (0, pi) and M  (chi={SPECTRUM_CHI})")
    print(f"{'=' * 60}")
    config = SpectrumConfig(chi=SPECTRUM_CHI)
    for name, k in MOMENTA.items():
        t0 = time.perf_counter()
        result = compute_spectrum(A_opt, gate, k, {"Sy": SY_REAL}, config)
        print_spectrum(name, result)
        print(f"    time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
