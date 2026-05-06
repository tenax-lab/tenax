"""Phase C.3 RDM brute-force vs multisite-CTM diagnostic.

Audit-only probe — writes a JSON of per-bond Frobenius deltas and per-bond
energies across a (D, χ, state-kind) ladder. Stop-and-ask checkpoint #5
in ``docs/plans/2026-05-05-multisite-kagome-pess.md`` is reached by
inspecting the resulting JSON / printed table; this script does NOT assert
the variational floor.

See the design at
``docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic-design.md`` and the
implementation plan at
``docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md``.

Usage::

    python examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.py
    python examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.py \\
        --D-ladder 1,2,3,4 --chi-ladder 8,16,32 --state-kind both

Note: the brute-force / multisite-CTM RDM helpers below are inline copies
of those in ``tests/test_pess_3site_multisite_rdm_invariants.py`` (the
test directory is not an installable package, and we do not want it to
become one). If those helpers change, this script must be updated in
lockstep.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_convergence import ctm_multisite
from tenax.algorithms._ctm_tensor_energy import (
    _rdm1x2_tensor_2site,
    _rdm2x1_tensor_2site,
)
from tenax.algorithms._pess_multisite_energy import (
    _rdm_3site_marginal_vw_col,
    _rdm_3site_marginal_vw_row,
    kagome_xxz_pair_hamiltonian,
)
from tenax.algorithms.pess import (
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    pess_simple_update,
    pess_to_kagome_3site_multisite,
)
from tenax.algorithms.pess_optimize import _make_multisite_indices
from tenax.core.lattice import kagome
from tenax.core.tensor import DenseTensor

DELTA = 1.0
D_PHYS = 2

# ---------------------------------------------------------------------------
# Inline helpers (copy of tests/test_pess_3site_multisite_rdm_invariants.py).
# ---------------------------------------------------------------------------

# 3×3 PBC torus position → sublattice.
_POS_TO_NAME: tuple[str, ...] = (
    "u",
    "v",
    "w",
    "v",
    "w",
    "u",
    "w",
    "u",
    "v",
)

# Position-index representatives for the 6 brute-force RDMs.
_BF_BOND_POSITIONS: dict[str, tuple[int, ...]] = {
    "uv_h": (0, 1),
    "uv_v": (0, 3),
    "wu_h": (2, 0),
    "wu_v": (2, 5),
    "vw_row": (1, 2),
    "vw_col": (3, 6),
}


def _contract_multisite_3x3_torus(sites: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Exact contraction of the 3-site multisite encoding on a 3×3 PBC torus."""
    strings = (
        "pjcaA",  # pos 0 (u)
        "qkabB",  # pos 1 (v)
        "rlbcC",  # pos 2 (w)
        "jmfdD",  # pos 3 (v)
        "kndeE",  # pos 4 (w)
        "loefF",  # pos 5 (u)
        "mpigG",  # pos 6 (w)
        "nqghH",  # pos 7 (u)
        "orhiI",  # pos 8 (v)
    )
    spec = ",".join(strings) + "->ABCDEFGHI"
    args = [sites[_POS_TO_NAME[p]] for p in range(9)]
    return jnp.einsum(spec, *args, optimize="optimal")


def _brute_force_rdm_from_torus_psi(
    psi: jnp.ndarray, sites_to_keep: tuple[int, ...]
) -> jnp.ndarray:
    """ρ = Tr_{rest}(|ψ⟩⟨ψ|) / ⟨ψ|ψ⟩ on `sites_to_keep`, axes ordered to match."""
    n = psi.ndim
    assert n == 9, f"expected rank-9 torus ψ, got rank-{n}"
    sites_to_trace = tuple(i for i in range(n) if i not in sites_to_keep)
    rho = jnp.tensordot(psi, jnp.conj(psi), axes=(sites_to_trace, sites_to_trace))
    norm_sq = jnp.tensordot(psi, jnp.conj(psi), axes=(tuple(range(n)), tuple(range(n))))
    rho = rho / norm_sq
    sorted_keep = sorted(sites_to_keep)
    perm_ket = [sorted_keep.index(p) for p in sites_to_keep]
    k = len(sites_to_keep)
    perm = tuple(perm_ket) + tuple(p + k for p in perm_ket)
    return jnp.transpose(rho, perm)


def _brute_force_rdms(state: IPESSState) -> dict[str, jnp.ndarray]:
    """Compute the 6 brute-force RDMs from the 3×3 torus wavefunction."""
    sites = pess_to_kagome_3site_multisite(
        state.R_a,
        state.R_b,
        state.R_c,
        state.T_u,
        state.T_d,
        state.lambdas,
    )
    psi = _contract_multisite_3x3_torus(sites)
    out: dict[str, jnp.ndarray] = {}
    for name, sites_to_keep in _BF_BOND_POSITIONS.items():
        out[name] = _brute_force_rdm_from_torus_psi(psi, sites_to_keep)
    return out


def _collect_ctm_rdms(
    state: IPESSState, chi: int, max_iter: int = 100, conv_tol: float = 1e-10
) -> dict[str, jnp.ndarray]:
    """Run multisite-CTM on the encoded state and extract the 6 energy-formula RDMs."""
    d = 2
    sites = pess_to_kagome_3site_multisite(
        state.R_a,
        state.R_b,
        state.R_c,
        state.T_u,
        state.T_d,
        state.lambdas,
    )
    D = sites["u"].shape[0]
    indices = _make_multisite_indices(D, d)
    site_tensors_by_name = {}
    for name, A in sites.items():
        A_norm = A / (jnp.linalg.norm(A) + 1e-12)
        site_tensors_by_name[name] = DenseTensor(A_norm, indices)

    envs_by_name = ctm_multisite(
        site_tensors_by_name,
        kagome(),
        chi=chi,
        max_iter=max_iter,
        conv_tol=conv_tol,
    )
    S_u = site_tensors_by_name["u"]
    S_v = site_tensors_by_name["v"]
    S_w = site_tensors_by_name["w"]
    env_u = envs_by_name["u"]
    env_v = envs_by_name["v"]
    env_w = envs_by_name["w"]

    return {
        "uv_h": _rdm2x1_tensor_2site(S_u, S_v, env_u, env_v),
        "uv_v": _rdm1x2_tensor_2site(S_u, S_v, env_u, env_v),
        "wu_h": _rdm2x1_tensor_2site(S_w, S_u, env_w, env_u),
        "wu_v": _rdm1x2_tensor_2site(S_w, S_u, env_w, env_u),
        "vw_row": _rdm_3site_marginal_vw_row(S_u, S_v, S_w, env_u, env_v, env_w),
        "vw_col": _rdm_3site_marginal_vw_col(S_u, S_v, S_w, env_u, env_v, env_w),
    }


# ---------------------------------------------------------------------------
# Probe driver.
# ---------------------------------------------------------------------------


def _su_warmstart(state: IPESSState, D: int) -> IPESSState:
    H = kagome_triangle_xxz_hamiltonian(delta=DELTA, d=D_PHYS)
    return pess_simple_update(
        state,
        H,
        dt_schedule=[(0.1, 200), (0.01, 200), (0.001, 100)],
        D_max=D,
    )


def _per_bond_metrics(rdms_bf, rdms_ctm, H_pair) -> dict[str, dict[str, float]]:
    out = {}
    for name in rdms_bf:
        bf = rdms_bf[name]
        ctm = rdms_ctm[name]
        diff = float(jnp.linalg.norm(bf - ctm))
        e_bf = float(complex(jnp.einsum("ijkl,ijkl->", bf, H_pair)).real)
        e_ctm = float(complex(jnp.einsum("ijkl,ijkl->", ctm, H_pair)).real)
        out[name] = {
            "frobenius_delta": diff,
            "energy_bf": e_bf,
            "energy_ctm": e_ctm,
            "energy_delta": e_ctm - e_bf,
        }
    return out


def _run_one(D: int, chi: int, state_kind: str, seed: int, H_pair) -> dict:
    t0 = time.perf_counter()
    state = IPESSState.random(D=D, d=D_PHYS, key=jax.random.PRNGKey(seed))
    if state_kind == "su":
        state = _su_warmstart(state, D)
    t_state = time.perf_counter() - t0

    t0 = time.perf_counter()
    rdms_bf = _brute_force_rdms(state)
    t_bf = time.perf_counter() - t0

    t0 = time.perf_counter()
    rdms_ctm = _collect_ctm_rdms(state, chi=chi, max_iter=100, conv_tol=1e-9)
    t_ctm = time.perf_counter() - t0

    metrics = _per_bond_metrics(rdms_bf, rdms_ctm, H_pair)
    return {
        "D": D,
        "chi": chi,
        "state_kind": state_kind,
        "seed": seed,
        "wall_seconds_state": t_state,
        "wall_seconds_brute_force": t_bf,
        "wall_seconds_ctm": t_ctm,
        "metrics": metrics,
    }


def _print_table(rows: list[dict]) -> None:
    bonds = list(_BF_BOND_POSITIONS.keys())
    header = f"{'D':>3} {'chi':>4} {'kind':>6}  " + "  ".join(f"{b:>13}" for b in bonds)
    print(header)
    print("-" * len(header))
    for r in rows:
        row = f"{r['D']:>3} {r['chi']:>4} {r['state_kind']:>6}  " + "  ".join(
            f"{r['metrics'][b]['frobenius_delta']:>13.3e}" for b in bonds
        )
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D-ladder", default="1,2,3,4")
    parser.add_argument("--chi-ladder", default="8,16,32")
    parser.add_argument("--state-kind", choices=["random", "su", "both"], default="su")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    args = parser.parse_args()

    Ds = [int(x) for x in args.D_ladder.split(",")]
    chis = [int(x) for x in args.chi_ladder.split(",")]
    kinds = ["random", "su"] if args.state_kind == "both" else [args.state_kind]

    H_pair = jnp.asarray(kagome_xxz_pair_hamiltonian(delta=DELTA, d=D_PHYS))
    rows: list[dict] = []
    for D in Ds:
        for chi in chis:
            for kind in kinds:
                print(f"=== D={D} χ={chi} state={kind} ===", flush=True)
                row = _run_one(D, chi, kind, args.seed, H_pair)
                rows.append(row)

    print("\n=== Frobenius ‖ρ_brute − ρ_CTM‖_F per bond ===\n")
    _print_table(rows)

    args.output.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
