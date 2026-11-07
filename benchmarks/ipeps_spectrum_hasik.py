"""Benchmark compute_spectrum on Hasik's J1-J2 states.

States: github.com/jurajHasik/j1j2_ipeps_states (CC BY 4.0), the JSON files in
single-site_pg-C4v-A1/ and single-site_pg-C4v-A1_internal-U1/. They are data: this
script only parses them, and never runs code from that repository.

Example::

    uv run python benchmarks/ipeps_spectrum_hasik.py state.json energy --chi 36 --j2 0.3
    uv run python benchmarks/ipeps_spectrum_hasik.py state.json u1
    uv run python benchmarks/ipeps_spectrum_hasik.py state.json spectrum --chi 36 \
        --momentum X --checkpoint-dir runs/d3_chi36_x --out d3_chi36_x.npz
    uv run python benchmarks/ipeps_spectrum_hasik.py state.json spectrum --chi 36 \
        --momentum X --basis sma
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
OPERATORS = {"Sy": SY_REAL, "Sx": SX, "Sz": SZ}
MOMENTA = {"X": (np.pi, 0.0), "Y": (0.0, np.pi), "M": (np.pi, np.pi)}


def load_state(path: Path) -> np.ndarray:
    """Parse entries ``"s u l d r value"`` into ``A[u, d, l, r, s]``, normalised.

    A U(1) state lists them per elementary tensor, and ``A`` is the sum of those
    tensors weighted by ``coeffs``.
    """
    data = json.loads(Path(path).read_text())
    if "elem_tensors" in data:
        lines = (line.split() for line in data["coeffs"][0]["entries"])
        parts = [(float(c), data["elem_tensors"][int(i)]) for i, c in lines]
    else:
        parts = [(1.0, data["sites"][0])]
    D, d = parts[0][1]["auxDim"], parts[0][1]["physDim"]
    A = np.zeros((D, D, D, D, d))
    for weight, part in parts:
        for entry in part["entries"]:
            s, u, left, down, r, value = entry.split()[:6]
            A[int(u), int(down), int(left), int(r), int(s)] += weight * float(value)
    return A / np.linalg.norm(A)


def load_charges(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """``(physical charges, virtual charges, total charge)`` of a U(1) state."""
    data = json.loads(Path(path).read_text())
    charges = np.array(data["u1_charges"])
    return charges[:2], charges[2:], data["total_u1_charge"]


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


def operator_basis(A: np.ndarray, operators: dict[str, np.ndarray]) -> np.ndarray:
    """Orthonormal columns that hold ``A`` and each ``O A``.

    Four columns give the same ``s(k)`` and first moment as the full space, since
    both need only ``N`` and ``H`` on ``O A``. ``A`` is a null vector of ``N`` at
    ``k != 0``, which checks the run.
    """
    vectors = [
        np.tensordot(A, op, axes=([4], [1])).ravel() for op in operators.values()
    ]
    return np.linalg.qr(np.stack([A.ravel(), *vectors], axis=1))[0]


def rotated_basis(basis: np.ndarray, seed: int = 0) -> np.ndarray:
    """``basis`` times a random orthogonal matrix: the same span, no special column.

    On an exact U(1) state a column along ``A``, ``S^z A`` or one orbit can blow up
    (1e195 at D=3, chi=27, and 1e98 at chi=12), and in this basis the same runs stay
    clean. ``H`` and ``N`` are linear in the column, so the span sets the answer.
    """
    n = basis.shape[1]
    return basis @ np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))[0]


def transverse_basis(
    physical: np.ndarray, virtual: np.ndarray, total: int, even: bool = True
) -> np.ndarray:
    """Orthonormal tangent vectors that ``S^x A`` and ``S^y A`` need on a U(1) state.

    Both operators shift the charge of an element by +2 or -2. Both also stay even
    under u<->d and l<->r, which keep the momenta X, Y and M, so their spectral
    weight stays in the even block. With ``even`` each column is one orbit of the
    two reflections; without it, one element.
    """
    D = len(virtual)
    shape = (D, D, D, D, len(physical))
    orbits: dict[tuple, list[int]] = {}
    for index in np.ndindex(*shape):
        u, d, left, r, s = index
        charge = physical[s] + virtual[u] + virtual[d] + virtual[left] + virtual[r]
        if abs(charge - total) == 2:
            key = (
                (min(u, d), max(u, d), min(left, r), max(left, r), s) if even else index
            )
            orbits.setdefault(key, []).append(np.ravel_multi_index(index, shape))
    basis = np.zeros((int(np.prod(shape)), len(orbits)))
    for column, rows in enumerate(orbits.values()):
        basis[rows, column] = 1.0 / np.sqrt(len(rows))
    return basis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("task", choices=["u1", "energy", "spectrum"])
    parser.add_argument("--chi", type=int, default=36)
    parser.add_argument("--j2", type=float, default=0.0)
    parser.add_argument("--momentum", choices=sorted(MOMENTA), default="X")
    parser.add_argument(
        "--basis", choices=["full", "sma", "charge", "even"], default="full"
    )
    parser.add_argument("--fd-step", type=float, default=1e-4)
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
        print(
            f"chi={args.chi} J2={args.j2} energy per site {float(energy):.10f} "
            f"converged {ctm.ctm_converged}"
        )
        return

    transverse = {name: OPERATORS[name] for name in ("Sy", "Sx")}
    operators, basis = OPERATORS, None
    if args.basis == "sma":
        basis = rotated_basis(operator_basis(A, OPERATORS))
    elif args.basis in ("charge", "even"):
        operators = transverse
        basis = rotated_basis(
            transverse_basis(*load_charges(args.state), even=args.basis == "even")
        )
    if basis is not None:
        print(f"basis {args.basis}: {basis.shape[1]} columns")

    start = time.time()
    result = compute_spectrum(
        A,
        rotated_gate(),
        MOMENTA[args.momentum],
        operators,
        SpectrumConfig(
            chi=args.chi, fd_step=args.fd_step, checkpoint_dir=args.checkpoint_dir
        ),
        gate_nnn=gate_nnn,
        basis=basis,
    )
    print(f"{time.time() - start:.0f} s, rank {result.rank}")
    print("lowest omega", np.round(result.omega[:6], 4))
    print("s(k)", result.structure_factor)
    print("sum rule deficit", result.sum_rule_deficit)
    print(
        "first moment", {n: float(result.omega @ w) for n, w in result.weights.items()}
    )
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
