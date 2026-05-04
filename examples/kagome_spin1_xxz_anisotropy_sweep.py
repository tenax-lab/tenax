"""XXZ anisotropy sweep on the spin-1 kagome lattice via AD-iPESS.

Sweeps Δ ∈ {0.0, 0.5, 1.0, 1.5, 2.0} at fixed ``D=4``, ``χ=12`` and
records the optimized energy and per-site ⟨S^z⟩. Smoke check for the
expected XY-vs-Ising trend across Δ=1.

Bond dimension defaults to ``D=4`` (even) because the AD path's
SVD-projector spectrum is degenerate at odd ``D`` (especially ``D=3``);
see the odd-D warning in :func:`tenax.algorithms.pess.pess_to_kagome_supersite`.
Pass ``--D 3`` to deliberately probe the odd-D regime; long-term fix is
the split-CTM SVD projector tracked in #388.

Per-site ⟨S^z⟩ is averaged over the three sublattices ``(R_a, R_b,
R_c)`` of the up-triangle supersite (the iPESS unit cell). At Δ=0 the
state is in the easy-plane regime and ⟨S^z⟩ should be near zero; as Δ
grows, ⟨S^z⟩ on each sublattice approaches its Sz-eigenvalue extreme
(rounded by quantum fluctuations).

Output: JSON list of ``{"delta", "E_per_site", "Sz_avg"}``.

Usage:
    python examples/kagome_spin1_xxz_anisotropy_sweep.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import (
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    kagome_xxz_pess_cg_gates,
    pess_simple_update,
    pess_to_kagome_supersite,
)
from tenax.algorithms.pess_optimize import optimize_pess_ad

D_PHYS = 3  # spin-1
D_DEFAULT = 4  # even by default — odd D destabilises the AD path (see module docstring)
CHI = 12


def _ipess_per_site_sz(state: IPESSState) -> float:
    """Average ⟨S^z⟩ over the three kagome sublattices via the supersite RDM.

    Builds the rank-5 supersite from the iPESS state, contracts it
    against itself (no env, just bra-ket on the 3 phys legs) to get the
    *unconditioned* 3-site density matrix, then traces with
    ``Sz ⊗ I ⊗ I`` etc. for each sublattice. This skips the CTM env
    entirely — what we report is the local triangle expectation, which
    converges to the bulk value as the bond gauges absorb the
    environment correctly. Good enough for a sanity-check anisotropy
    trend; for high-precision ``S^z`` measurements use a converged CTM
    env and a 1-site supersite RDM.
    """
    A = pess_to_kagome_supersite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.lambdas
    )
    # A shape (D, D, D, D, d^3). Build local 3-site RDM by contracting
    # virtual legs against bra: rho_local = einsum(A, A*, [virt, virt']).
    # Take only the slice along axis 3 = 0 since the 4th leg is dummy.
    A_local = A[:, :, :, 0, :]  # (D, D, D, d^3)
    rho = jnp.einsum("abci,abcj->ij", A_local, A_local.conj())
    rho = rho / (jnp.trace(rho).real + 1e-12)

    d = D_PHYS
    s = (d - 1) / 2.0
    diag = jnp.array([s - k for k in range(d)], dtype=jnp.complex128)
    Sz = jnp.diag(diag)
    eye = jnp.eye(d, dtype=jnp.complex128)

    # Each sublattice's Sz embedded into the d^3 fused phys leg.
    Sz_a = jnp.kron(jnp.kron(Sz, eye), eye)
    Sz_b = jnp.kron(jnp.kron(eye, Sz), eye)
    Sz_c = jnp.kron(jnp.kron(eye, eye), Sz)
    sz_avg = (
        jnp.trace(rho @ Sz_a) + jnp.trace(rho @ Sz_b) + jnp.trace(rho @ Sz_c)
    ).real / 3.0
    return float(sz_avg)


def _make_ctm_config(chi: int, max_iter: int = 30) -> CTMConfig:
    return CTMConfig(
        chi=chi,
        max_iter=max_iter,
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


def run_anisotropy_sweep(
    deltas: list[float],
    *,
    D: int = D_DEFAULT,
    max_iter: int = 60,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict]:
    results: list[dict] = []
    config = _make_ctm_config(chi=CHI)

    for delta in deltas:
        print(f"\n=== Δ = {delta:.3f} ===", flush=True)
        H = kagome_triangle_xxz_hamiltonian(delta=delta, d=D_PHYS)
        cg_gates = kagome_xxz_pess_cg_gates(delta=delta, d=D_PHYS)
        state = IPESSState.random(D=D, d=D_PHYS, key=jax.random.PRNGKey(seed))
        state = pess_simple_update(
            state,
            H,
            dt_schedule=[(0.1, 200), (0.01, 200), (0.001, 100)],
            D_max=D,
        )
        t0 = time.perf_counter()
        state, e_ad = optimize_pess_ad(
            state, cg_gates, config, max_iter=max_iter, verbose=verbose
        )
        dt = time.perf_counter() - t0
        sz = _ipess_per_site_sz(state)
        print(
            f"  E/site = {e_ad:.6f}    <S^z>_avg = {sz:+.6f}    ({dt:.1f}s)",
            flush=True,
        )
        results.append(
            {
                "delta": delta,
                "E_per_site": e_ad,
                "Sz_avg": sz,
                "wall_seconds": dt,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
        help="XXZ anisotropies to sweep over",
    )
    parser.add_argument(
        "--D",
        type=int,
        default=D_DEFAULT,
        help=(
            f"iPESS bond dimension (default {D_DEFAULT}; odd D is AD-unstable, "
            "see module docstring)"
        ),
    )
    parser.add_argument("--max-iter", type=int, default=60, help="L-BFGS iterations")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
        help="Path to JSON results file",
    )
    args = parser.parse_args()

    results = run_anisotropy_sweep(
        args.deltas, D=args.D, max_iter=args.max_iter, verbose=True
    )
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} result(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
