#!/usr/bin/env python3
"""iPEPS excitation spectrum for the 2D Heisenberg model.

Computes the quasiparticle excitation dispersion of the spin-1/2
antiferromagnetic Heisenberg model on the infinite square lattice using
the iPEPS excitation ansatz (Ponsioen, Assaad & Corboz, SciPost Phys. 12,
006, 2022).

The workflow is:

1. Build the 2-site Heisenberg gate.
2. Find the ground state via AD optimization (``optimize_gs_ad``, D=2, chi=16).
3. Generate a momentum path along high-symmetry lines Gamma-X-M-Gamma.
4. Compute excitation energies at each momentum point.
5. Print the excitation spectrum.

Note: this example is slow (~minutes) due to the ground state optimization
followed by the excitation calculation at each k-point.

Usage::

    uv run python examples/heisenberg_ipeps_excitations.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax import (
    CTMConfig,
    ExcitationConfig,
    ExcitationResult,
    compute_excitations,
    heisenberg_gate,
    iPEPSConfig,
    make_momentum_path,
    optimize_gs_ad,
)

# ---------------------------------------------------------------------------
# Ground state optimization
# ---------------------------------------------------------------------------


def find_ground_state(
    gate: jnp.ndarray,
    D: int = 2,
    chi: int = 16,
    gs_steps: int = 100,
    lr: float = 1e-3,
) -> tuple[jnp.ndarray, object, float]:
    """Find the iPEPS ground state using AD optimization.

    Returns:
        ``(A_opt, env, E_gs)`` -- optimized tensor, CTM environment, and
        ground state energy per site.
    """
    config = iPEPSConfig(
        max_bond_dim=D,
        ctm=CTMConfig(chi=chi, max_iter=60),
        gs_optimizer="adam",
        gs_learning_rate=lr,
        gs_num_steps=gs_steps,
    )

    print(f"  D={D}, chi={chi}, lr={lr}, AD steps={gs_steps}")

    t0 = time.perf_counter()
    A_opt, env, E_gs = optimize_gs_ad(gate, None, config)
    dt = time.perf_counter() - t0

    print(f"  E/site = {E_gs:.6f}  (time: {dt:.1f}s)")
    return A_opt, env, E_gs


# ---------------------------------------------------------------------------
# Excitation spectrum
# ---------------------------------------------------------------------------


def compute_spectrum(
    A_opt: jnp.ndarray,
    env: object,
    gate: jnp.ndarray,
    E_gs: float,
    num_points: int = 12,
    num_excitations: int = 2,
) -> ExcitationResult:
    """Compute excitation energies along Gamma-X-M-Gamma.

    Args:
        A_opt:            Optimized ground state tensor.
        env:              Converged CTM environment.
        gate:             2-site Hamiltonian gate.
        E_gs:             Ground state energy per site.
        num_points:       Total number of momentum points on the path.
        num_excitations:  Number of lowest excitations per k-point.

    Returns:
        ExcitationResult with ``.energies`` and ``.momenta``.
    """
    momenta = make_momentum_path("brillouin", num_points=num_points)

    exc_config = ExcitationConfig(
        num_excitations=num_excitations,
        null_space_tol=1e-2,
    )

    print(f"  Momentum path: Gamma -> X -> M -> Gamma ({len(momenta)} points)")
    print(f"  num_excitations={num_excitations}")

    t0 = time.perf_counter()
    result = compute_excitations(A_opt, env, gate, E_gs, momenta, exc_config)
    dt = time.perf_counter() - t0

    print(f"  Excitation calculation time: {dt:.1f}s")
    return result


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------


def print_spectrum(result: ExcitationResult) -> None:
    """Print the excitation spectrum in a readable table."""
    print(f"\n{'=' * 60}")
    print("  Excitation Spectrum")
    print(f"  Ground state energy: E/site = {result.ground_state_energy:.6f}")
    print(f"{'=' * 60}")

    n_k, n_exc = result.energies.shape
    header = f"  {'kx/pi':>8s}  {'ky/pi':>8s}"
    for i in range(n_exc):
        header += f"  {'omega_' + str(i):>10s}"
    print(header)
    print(f"  {'-' * len(header.strip())}")

    for ik in range(n_k):
        kx, ky = result.momenta[ik]
        row = f"  {kx / np.pi:8.4f}  {ky / np.pi:8.4f}"
        for ie in range(n_exc):
            row += f"  {result.energies[ik, ie]:10.6f}"
        print(row)

    # Highlight high-symmetry points
    print("\n  High-symmetry excitation gaps:")
    # Find points closest to Gamma, X, M
    targets = {"Gamma": (0.0, 0.0), "X": (np.pi, 0.0), "M": (np.pi, np.pi)}
    for name, (kx_t, ky_t) in targets.items():
        dists = np.sqrt(
            (result.momenta[:, 0] - kx_t) ** 2 + (result.momenta[:, 1] - ky_t) ** 2
        )
        idx = int(np.argmin(dists))
        omega_0 = result.energies[idx, 0]
        kx, ky = result.momenta[idx]
        print(
            f"    {name:>6s} (kx/pi={kx / np.pi:.3f}, ky/pi={ky / np.pi:.3f})"
            f":  omega_0 = {omega_0:.6f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("iPEPS excitation spectrum: 2D Heisenberg model")
    print("H = sum_{<i,j>} S_i . S_j   (J=1, antiferromagnetic)")
    print("Method: Ponsioen, Assaad & Corboz, SciPost Phys. 12, 006 (2022)")

    gate = heisenberg_gate()

    # Step 1: Find the ground state
    print(f"\n{'=' * 60}")
    print("  Step 1: Ground state optimization")
    print(f"{'=' * 60}")
    A_opt, env, E_gs = find_ground_state(gate, D=2, chi=16, gs_steps=100, lr=1e-3)

    # Step 2: Compute excitation spectrum
    print(f"\n{'=' * 60}")
    print("  Step 2: Excitation spectrum")
    print(f"{'=' * 60}")
    result = compute_spectrum(
        A_opt,
        env,
        gate,
        E_gs,
        num_points=12,
        num_excitations=2,
    )

    # Step 3: Print results
    print_spectrum(result)


if __name__ == "__main__":
    main()
