"""Exact kagome supersite blocking (#991): T_d kept, no dummy leg.

Four gates, ordered from state-level to end-to-end:

* wavefunction fidelity on the 1-cell PBC torus against an independent
  contraction of the raw PESS state in the physical gauge (each bond
  carries its full lambda exactly once) — kills blocking mutations
  (lambda gauge, T_d omission, leg mis-wiring);
* energy-formula vs brute-force parity on a 2x2-cell supersite torus with
  a random (asymmetric) state — kills gate sub-site/direction mutations
  (h/v slot swaps, diagonal orientation, n_sites) at O(0.1) signal;
* D=1 product-state energy through ``build_pess_loss_exact`` == analytic;
* D=2 SU energy pin: the #991 control state's readout, cross-validated
  against variPEPS 1.4.2 (agreement 1e-9) and exact L x inf cylinder
  extrapolations (~2e-4) during the #991 investigation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import (
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    kagome_xxz_pess_cg_gates_exact,
    pess_simple_update,
    pess_to_kagome_supersite_exact,
)
from tenax.algorithms.pess_optimize import build_pess_loss_exact

D_PHYS = 2


def _nontrivial_state(D: int, seed: int = 0) -> IPESSState:
    """Random iPESS state with deliberately non-uniform lambdas (#884 rule:
    with unit lambdas every power of lambda is invisible to any gauge test)."""
    state = IPESSState.random(D=D, d=D_PHYS, key=jax.random.PRNGKey(seed))
    lam_vals = [0.4, 0.75, 1.3, 0.55, 0.9, 1.6]
    state = IPESSState(
        R_a=state.R_a,
        R_b=state.R_b,
        R_c=state.R_c,
        T_u=state.T_u,
        T_d=state.T_d,
        lambdas=tuple(v * jnp.linspace(1.0, 0.5, D) for v in jnp.asarray(lam_vals)),
    )
    assert (
        max(float(jnp.max(jnp.abs(lam - 1.0))) for lam in state.lambdas[3:6]) > 0.3
    ), "fixture regime violated: down lambdas too close to 1"
    return state


def _contract_ipess_one_cell_pbc(state: IPESSState) -> jnp.ndarray:
    """Raw PESS on the 1-cell PBC kagome torus, PHYSICAL gauge (each bond
    one full lambda; simplex tensors bare). Independent of the blocking."""
    lam = state.lambdas
    dtype = state.R_a.dtype

    def full_lam(x):
        return jnp.power(jnp.real(x) ** 2 + 1e-28, 0.5).astype(dtype)

    Ra = jnp.einsum("i,ijp,j->ijp", full_lam(lam[3]), state.R_a, lam[0].astype(dtype))
    Rb = jnp.einsum("i,ijp,j->ijp", full_lam(lam[4]), state.R_b, lam[1].astype(dtype))
    Rc = jnp.einsum("i,ijp,j->ijp", full_lam(lam[5]), state.R_c, lam[2].astype(dtype))
    return jnp.einsum(
        "ABu,CDv,EFw,BDF,ACE->uvw",
        Ra,
        Rb,
        Rc,
        state.T_u.astype(dtype),
        state.T_d.astype(dtype),
    )


@pytest.mark.core
@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_supersite_exact_wavefunction_matches_ipess_on_1cell_torus(D):
    """Fidelity == 1: the exact supersite IS the PESS state (no Convention-C
    T_d approximation, no dummy leg, full lambdas)."""
    state = _nontrivial_state(D)
    A = pess_to_kagome_supersite_exact(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas
    )
    # 1-cell torus closure: top<->bottom (R_c-down with T_d-c), left<->right
    # (T_d-b with R_b-down) — all 6 kagome bonds close inside the cell.
    psi_ss = jnp.einsum("zzBBp->p", A)
    psi_ref = _contract_ipess_one_cell_pbc(state).reshape(-1)

    n_ss = float(jnp.linalg.norm(psi_ss))
    n_ref = float(jnp.linalg.norm(psi_ref))
    assert n_ss > 0.0 and n_ref > 0.0
    fidelity = float(jnp.abs(jnp.vdot(psi_ref, psi_ss)) ** 2 / (n_ss**2 * n_ref**2))
    np.testing.assert_allclose(fidelity, 1.0, atol=1e-12)


def _torus_psi(A: jnp.ndarray, M: int, N: int) -> jnp.ndarray:
    """Contract M x N copies of the supersite on a PBC torus.

    Cell (m, n): m = row (down = +m), n = column (right = +n). Supersite
    axes (top, bottom, left, right, phys): vertical bond joins
    bottom(m, n) with top(m+1, n); horizontal joins right(m, n) with
    left(m, n+1). Returns psi with one fused d**3 leg per cell, cells in
    row-major (m, n) order.
    """
    nc = M * N

    def vid(m, n):  # vertical bond below cell (m, n)
        return (m % M) * N + (n % N)

    def hid(m, n):  # horizontal bond right of cell (m, n)
        return nc + (m % M) * N + (n % N)

    def pid(m, n):
        return 2 * nc + (m % M) * N + (n % N)

    ops, subs = [], []
    for m in range(M):
        for n in range(N):
            ops.append(A)
            # (top, bottom, left, right, phys)
            subs.append([vid(m - 1, n), vid(m, n), hid(m, n - 1), hid(m, n), pid(m, n)])
    out = [pid(m, n) for m in range(M) for n in range(N)]
    interleaved = []
    for op, sub in zip(ops, subs):
        interleaved += [op, sub]
    return jnp.einsum(*interleaved, out)


@pytest.mark.core
@pytest.mark.parametrize("M,N", [(2, 3), (3, 2)])
def test_supersite_exact_energy_formula_matches_torus_brute_force(M, N):
    """The CG gate assignment reproduces <H_kagome> exactly on M x N tori.

    Uses a RANDOM state with non-uniform lambdas: its bond energies differ
    at O(0.1) between sub-site pairs, so any gate slot swap or wrong
    diagonal orientation shifts the formula value far above the 1e-9 gate.

    Torus sizes are (2, 3) AND (3, 2) deliberately: on a period-2
    direction, translation exchanges the two cells of a pair, which
    aliases slot-ORDER swaps into invariance (verified: on a 2x2 torus the
    h/diag slot-swap mutants pass). Period 3 in the horizontal direction
    (2x3) makes h and diag swaps visible; period 3 vertically (3x2) makes
    v swaps visible.
    """
    d = D_PHYS
    state = _nontrivial_state(2, seed=3)
    A = pess_to_kagome_supersite_exact(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas
    )
    A = A / jnp.linalg.norm(A)
    psi = _torus_psi(A, M, N)
    psi = psi / jnp.linalg.norm(psi)
    gates = kagome_xxz_pess_cg_gates_exact(delta=1.0, d=d)
    nc = M * N
    de = d**3

    def cell_pos(m, n):
        return (m % M) * N + (n % N)

    def rho2(p1, p2):
        rest = [i for i in range(nc) if i not in (p1, p2)]
        pk = jnp.transpose(psi, [p1, p2] + rest).reshape(de, de, -1)
        return jnp.einsum("PQr,pqr->PQpq", pk, pk.conj())

    # -- formula side: sum the CG terms over every cell of the torus --
    e_formula = 0.0
    for m in range(M):
        for n in range(N):
            c = cell_pos(m, n)
            rest = [i for i in range(nc) if i != c]
            pk = jnp.transpose(psi, [c] + rest).reshape(de, -1)
            r1 = jnp.einsum("Pr,pr->Pp", pk, pk.conj())
            e_formula = e_formula + jnp.einsum("Pp,pP->", r1, gates.h_intra)
            e_formula = e_formula + jnp.einsum(
                "PQpq,pqPQ->",
                rho2(c, cell_pos(m, n + 1)),
                gates.h_inter["h"],
            )
            e_formula = e_formula + jnp.einsum(
                "PQpq,pqPQ->",
                rho2(c, cell_pos(m + 1, n)),
                gates.h_inter["v"],
            )
            e_formula = e_formula + jnp.einsum(
                "PQpq,pqPQ->",
                rho2(c, cell_pos(m + 1, n + 1)),
                gates.h_inter["diag"],
            )
    e_formula = float(e_formula.real) / (nc * gates.n_sites)

    # -- brute force: raw kagome H on the 3 * nc physical sites --
    psi_phys = psi.reshape((d,) * (3 * nc))
    Sz = jnp.asarray(np.diag([0.5, -0.5]).astype(complex))
    Sp = jnp.asarray(np.array([[0, 1], [0, 0]], dtype=complex))
    Sm = Sp.T.conj()
    terms = [(Sz, Sz), (0.5 * Sp, Sm), (0.5 * Sm, Sp)]

    def site_index(m, n, x):
        return cell_pos(m, n) * 3 + "abc".index(x)

    def bond_e(s1, s2):
        rest = [i for i in range(3 * nc) if i not in (s1, s2)]
        pk = jnp.transpose(psi_phys, [s1, s2] + rest).reshape(d, d, -1)
        val = 0.0
        for X, Y in terms:
            val = val + jnp.einsum("abr,ai,bj,ijr->", pk.conj(), X, Y, pk)
        return val.real

    e_brute = 0.0
    for m in range(M):
        for n in range(N):
            a = site_index(m, n, "a")
            b = site_index(m, n, "b")
            c = site_index(m, n, "c")
            # up triangle (intra-cell)
            e_brute += bond_e(a, b) + bond_e(b, c) + bond_e(a, c)
            # down triangle of cell (m, n): a(m,n), b at (m, n-1) [left],
            # c at (m+1, n) [below] — per the blocking's leg geometry.
            bl = site_index(m, n - 1, "b")
            cb = site_index(m + 1, n, "c")
            e_brute += bond_e(a, bl) + bond_e(a, cb) + bond_e(bl, cb)
    e_brute = float(e_brute) / (3 * nc)

    # Regime assertion: pair energies must be asymmetric between sub-site
    # choices, else slot swaps are invisible regardless of torus size.
    eh = float(
        jnp.einsum("PQpq,pqPQ->", rho2(0, cell_pos(0, 1)), gates.h_inter["h"]).real
    )
    ev = float(
        jnp.einsum("PQpq,pqPQ->", rho2(0, cell_pos(1, 0)), gates.h_inter["v"]).real
    )
    assert abs(eh - ev) > 1e-3, "fixture regime violated: h/v degenerate"

    np.testing.assert_allclose(e_formula, e_brute, atol=1e-9)


@pytest.mark.core
def test_supersite_exact_d1_product_energy_analytic():
    """D=1 product state end-to-end through build_pess_loss_exact."""
    rng = np.random.default_rng(7)
    vs = {}
    for k in "abc":
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        vs[k] = v / np.linalg.norm(v)
    state = IPESSState(
        R_a=jnp.asarray(vs["a"].reshape(1, 1, 2)),
        R_b=jnp.asarray(vs["b"].reshape(1, 1, 2)),
        R_c=jnp.asarray(vs["c"].reshape(1, 1, 2)),
        T_u=jnp.ones((1, 1, 1), dtype=jnp.complex128),
        T_d=jnp.ones((1, 1, 1), dtype=jnp.complex128),
        lambdas=tuple(jnp.ones(1) for _ in range(6)),
    )
    Sz = np.diag([0.5, -0.5]).astype(complex)
    Sp = np.array([[0, 1], [0, 0]], dtype=complex)
    Sm = Sp.T.conj()
    h = np.kron(Sz, Sz) + 0.5 * (np.kron(Sp, Sm) + np.kron(Sm, Sp))

    def pair(x, y):
        w = np.kron(vs[x], vs[y])
        return (w.conj() @ h @ w).real / np.vdot(w, w).real

    e_analytic = 2.0 * (pair("a", "b") + pair("b", "c") + pair("a", "c")) / 3.0

    config = CTMConfig(chi=8, max_iter=50, conv_tol=1e-10)
    loss = build_pess_loss_exact(kagome_xxz_pess_cg_gates_exact(1.0, 2), config)
    e = float(loss(state).real)
    np.testing.assert_allclose(e, e_analytic, atol=1e-9)


def test_supersite_exact_d2_su_energy_pin():
    """End-to-end D=2 SU readout pin — the #991 control measurement.

    Provenance of the pinned value (2026-09-14 investigation, issue #991):
    the same state and settings gave -0.386195270 through this path,
    -0.3861952703 through variPEPS 1.4.2's independent PESS3 blocking
    (agreement 1e-9), and -0.3862 +- 2e-4 from exact L x inf cylinder
    transfer-matrix extrapolations. Bit-stable under chi 16->32. The
    3-site multisite path reads -0.388527 on the same state — the #991
    rank-truncation bias; tolerance 1e-4 sits 20x below that signal while
    absorbing cross-platform SU drift.
    """
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    state = pess_simple_update(
        state, H, dt_schedule=[(0.1, 200), (0.01, 200), (0.001, 100)], D_max=2
    )
    # Regime assertions: SU actually ran (lambdas moved off 1) and the
    # state is in the SU-converged energy band.
    assert max(float(jnp.max(jnp.abs(lam - 1.0))) for lam in state.lambdas) > 0.05, (
        "fixture regime violated: SU left lambdas at 1"
    )

    config = CTMConfig(
        chi=16,
        max_iter=100,
        min_iter=4,
        conv_tol=1e-9,
        projector_method="svd",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
    )
    loss = build_pess_loss_exact(kagome_xxz_pess_cg_gates_exact(1.0, 2), config)
    e = float(loss(state).real)
    assert -0.42 < e < -0.35, f"outside the SU band: {e}"
    np.testing.assert_allclose(e, -0.386195270, atol=1e-4)
