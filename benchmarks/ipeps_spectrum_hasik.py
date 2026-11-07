"""Benchmark compute_spectrum on Hasik's J1-J2 states.

States: github.com/jurajHasik/j1j2_ipeps_states (CC BY 4.0), the JSON files in
single-site_pg-C4v-A1/. They are data: this script only parses them, and never
runs code from that repository.

Example::

    uv run python benchmarks/ipeps_spectrum_hasik.py state.json energy --chi 36 --j2 0.3
    uv run python benchmarks/ipeps_spectrum_hasik.py state.json u1
    uv run python benchmarks/ipeps_spectrum_hasik.py state.json spectrum --chi 36 \
        --momentum X --checkpoint-dir runs/d3_chi36_x --out d3_chi36_x.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SY_REAL = np.array([[0.0, 0.5], [-0.5, 0.0]])
SX = np.array([[0.0, 0.5], [0.5, 0.0]])
SZ = np.diag([0.5, -0.5])
MOMENTA = {"X": (np.pi, 0.0), "Y": (0.0, np.pi), "M": (np.pi, np.pi)}


def load_state(path: Path) -> np.ndarray:
    """Parse entries ``"s u l d r value"`` into ``A[u, d, l, r, s]``, normalised."""
    site = json.loads(Path(path).read_text())["sites"][0]
    D = site["auxDim"]
    A = np.zeros((D, D, D, D, site["physDim"]))
    for entry in site["entries"]:
        s, u, left, d, r, value = entry.split()[:6]
        A[int(u), int(d), int(left), int(r), int(s)] = float(value)
    return A / np.linalg.norm(A)


def heisenberg_gate() -> np.ndarray:
    return np.array(
        [
            [0.25, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.5, 0.0],
            [0.0, 0.5, -0.25, 0.0],
            [0.0, 0.0, 0.0, 0.25],
        ]
    ).reshape(2, 2, 2, 2)


def rotated_gate() -> np.ndarray:
    W = np.kron(np.eye(2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    H = heisenberg_gate().reshape(4, 4)
    return (W @ H @ W.T).reshape(2, 2, 2, 2)


def u1_residuals(A: np.ndarray, angles: int = 721) -> dict[str, float]:
    """How far ``A`` is from invariance under a U(1) with any bond generators.

    The residual is ``|P G A| / |G A|``, with ``P`` the projector off the span of
    an independent ``Q`` on each of the four legs. Zero means a charge sector
    exists in some bond gauge. The physical axis ``G`` is scanned in the real
    xz plane, and ``i S^y`` is tested separately.
    """
    D = A.shape[0]
    columns = []
    for p in range(D):
        for q in range(D):
            E = np.zeros((D, D))
            E[p, q] = 1.0
            columns += [
                np.einsum("xu,udlrs->xdlrs", E, A).ravel(),
                np.einsum("xd,udlrs->uxlrs", E, A).ravel(),
                np.einsum("xl,udlrs->udxrs", E, A).ravel(),
                np.einsum("xr,udlrs->udlxs", E, A).ravel(),
            ]
    U, singular, _ = np.linalg.svd(np.stack(columns, axis=1), full_matrices=False)
    Q = U[:, singular > 1e-10 * singular[0]]

    def residual(generator):
        b = np.tensordot(A, generator, axes=([4], [1])).ravel()
        return float(np.linalg.norm(b - Q @ (Q.T @ b)) / np.linalg.norm(b))

    scan = [
        (residual(np.cos(t) * SZ + np.sin(t) * SX), float(t))
        for t in np.linspace(0.0, np.pi, angles)
    ]
    best, angle = min(scan)
    return {"best_xz": best, "best_angle": angle, "i_sy": residual(SY_REAL)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("task", choices=["u1", "energy", "spectrum"])
    parser.add_argument("--chi", type=int, default=36)
    parser.add_argument("--j2", type=float, default=0.0)
    parser.add_argument("--momentum", choices=sorted(MOMENTA), default="X")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    A = load_state(args.state)
    print(f"state {args.state}: D={A.shape[0]}, {A.size} columns")
    if args.task == "u1":
        print(u1_residuals(A))
        return

    from tenax import SpectrumConfig, compute_spectrum
    from tenax.algorithms._ipeps_excitation_ctm import ExcitationCTM, cell_phases

    gate_nnn = args.j2 * heisenberg_gate() if args.j2 else None
    if args.task == "energy":
        ctm = ExcitationCTM(A, rotated_gate(), args.chi, 1, gate_nnn=gate_nnn)
        zero = np.zeros_like(A)
        energy, _ = ctm.densities(cell_phases(1, 0), 0.0, zero, zero)
        print(f"chi={args.chi} J2={args.j2} energy per site {float(energy):.10f}")
        return

    start = time.time()
    result = compute_spectrum(
        A,
        rotated_gate(),
        MOMENTA[args.momentum],
        {"Sy": SY_REAL, "Sx": SX, "Sz": SZ},
        SpectrumConfig(chi=args.chi, checkpoint_dir=args.checkpoint_dir),
        gate_nnn=gate_nnn,
    )
    print(f"{time.time() - start:.0f} s, rank {result.rank}")
    print("lowest omega", np.round(result.omega[:6], 4))
    print("s(k)", result.structure_factor)
    print("sum rule deficit", result.sum_rule_deficit)
    print(
        f"energy {result.energy_density:.10f} negative modes "
        f"{result.n_negative_modes} n_min_eig {result.n_min_eig:.2e} "
        f"gauge residual {result.gauge_residual}"
    )
    if args.out is not None:
        np.savez(
            args.out,
            omega=result.omega,
            **{f"w_{name}": w for name, w in result.weights.items()},
            **{f"s_{name}": s for name, s in result.structure_factor.items()},
        )


if __name__ == "__main__":
    main()
