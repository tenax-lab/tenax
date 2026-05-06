"""Structural-correctness gate for the 3-site multisite kagome encoding.

Task B.3a of plan ``docs/plans/2026-05-05-multisite-kagome-pess.md`` — replaces
the original B.3 (supersite-vs-multisite energy parity).  That parity test is
structurally invalid: ``pess_to_kagome_supersite`` does not take ``T_d`` as an
argument (the supersite/CG path freezes ``T_d`` bit-exact, see
``tests/test_pess_ad.py:116``).  The supersite and 3-site multisite encodings
live on different variational manifolds and cannot agree on an arbitrary
``IPESSState``.

This test instead asserts wavefunction-level equality between

  (a) the iPESS state contracted directly on a 1-unit-cell PBC kagome torus,
  (b) the 3-site multisite tensors contracted on a 1-cell 3-cycle PBC torus
      (closing the kagome neighbour map cyclically among the 3 sublattices).

Both produce a vector in the same ``(d, d, d)`` Hilbert space and should be
colinear (fidelity = 1) for any iPESS state.  CTM-free, AD-free, χ-free.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.pess import (
    IPESSState,
    pess_to_kagome_3site_multisite,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _contract_ipess_one_cell_pbc(state: IPESSState) -> jnp.ndarray:
    """Contract the iPESS state on a 1-unit-cell PBC kagome torus.

    All 6 bonds (3 R-T_u up-triangle + 3 R-T_d down-triangle) close inside
    the cell.  Gauge convention matches :func:`pess_to_kagome_3site_multisite`:
    sqrt(λ) on each R's T_d-side (axis 0) and full λ on each R's T_u-side
    (axis 1).  T_u and T_d carry no extra gauges.

    Returns:
        Rank-3 array ``(d, d, d)`` indexed by ``(p_u, p_v, p_w)`` (sublattice
        ``a/b/c`` mapped to multisite names ``u/v/w``).
    """
    R_a, R_b, R_c = state.R_a, state.R_b, state.R_c
    T_u, T_d = state.T_u, state.T_d
    lam_au, lam_bu, lam_cu = state.lambdas[0:3]
    lam_ad, lam_bd, lam_cd = state.lambdas[3:6]

    dtype = R_a.dtype

    def sqrt_lam(x):
        # Smooth ``sqrt`` mirrors :func:`pess_to_kagome_3site_multisite`.
        return jnp.power(jnp.real(x) ** 2 + 1e-28, 0.25).astype(dtype)

    sda = sqrt_lam(lam_ad)
    sdb = sqrt_lam(lam_bd)
    sdc = sqrt_lam(lam_cd)
    lau = lam_au.astype(dtype)
    lbu = lam_bu.astype(dtype)
    lcu = lam_cu.astype(dtype)

    # iPESS axis convention on R: (T_d-side, T_u-side, phys).
    Ra = jnp.einsum("i,ijp,j->ijp", sda, R_a, lau)
    Rb = jnp.einsum("i,ijp,j->ijp", sdb, R_b, lbu)
    Rc = jnp.einsum("i,ijp,j->ijp", sdc, R_c, lcu)

    # Closure: T_u[ja, jb, jc] couples the three T_u-sides.  T_d[ia, ib, ic]
    # couples the three T_d-sides.  Output is the rank-3 cell wavefunction
    # in ``(p_a, p_b, p_c)`` order = ``(p_u, p_v, p_w)``.
    return jnp.einsum(
        "ABu,CDv,EFw,BDF,ACE->uvw",
        Ra,
        Rb,
        Rc,
        T_u.astype(dtype),
        T_d.astype(dtype),
    )


def _contract_multisite_3cycle_pbc(sites: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Contract the 3-site multisite tensors on a 1-cell 3-cycle PBC torus.

    Tensor leg layout (matches :func:`pess_to_kagome_3site_multisite` output)::

        axes = (top, bottom, left, right, phys)

    Bond closures from :func:`tenax.core.lattice.kagome` 's neighbour map
    (each pair of sublattices is connected by 2 bonds — one up-triangle, one
    down-triangle)::

        u.top   ↔ w.bottom   (down-tri u-w bond)            label "T"
        u.bottom↔ v.top      (down-tri u-v bond)            label "B"
        u.left  ↔ w.right    (up-tri  u-w bond)             label "L"
        u.right ↔ v.left     (up-tri  u-v bond)             label "R"
        v.bottom↔ w.top      (v-w bond, both sides dim 1)   label "X"
        v.right ↔ w.left     (v-w bond, both sides dim 1)   label "Y"

    Returns:
        Rank-3 array ``(d, d, d)`` indexed by ``(p_u, p_v, p_w)``.
    """
    S_u = sites["u"]  # axes (top=T, bot=B, lft=L, rgt=R, phys=u)
    S_v = sites["v"]  # axes (top=B, bot=X, lft=R, rgt=Y, phys=v)
    S_w = sites["w"]  # axes (top=X, bot=T, lft=Y, rgt=L, phys=w)
    return jnp.einsum(
        "TBLRu,BXRYv,XTYLw->uvw",
        S_u,
        S_v,
        S_w,
    )


@pytest.mark.core
@pytest.mark.parametrize("D", [1, 2, 3, 4, 5, 6])
def test_3site_multisite_wavefunction_matches_ipess_on_1cell_torus(D):
    """Fidelity == 1 on the smallest valid kagome PBC torus.

    A failure at D≥2 with success at D=1 would localise a leg-axis bug to
    the encoding of the non-trivial bond legs (which are dim 1 at D=1).
    """
    d = 2
    state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(0))

    psi_ipess = _contract_ipess_one_cell_pbc(state)
    sites = pess_to_kagome_3site_multisite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas
    )
    psi_ms = _contract_multisite_3cycle_pbc(sites)

    assert psi_ipess.shape == (d, d, d)
    assert psi_ms.shape == (d, d, d)

    n_ipess = float(jnp.linalg.norm(psi_ipess))
    n_ms = float(jnp.linalg.norm(psi_ms))
    assert n_ipess > 0.0, "iPESS wavefunction is zero — random IPESSState gave 0"
    assert n_ms > 0.0, "multisite wavefunction is zero"
    overlap = jnp.vdot(psi_ipess.reshape(-1), psi_ms.reshape(-1))
    fidelity = float(jnp.abs(overlap) ** 2 / (n_ipess**2 * n_ms**2))

    np.testing.assert_allclose(
        fidelity,
        1.0,
        atol=1e-12,
        err_msg=(
            f"Encoding fidelity FAILED at D={D}: |<ψ_iPESS|ψ_ms>|² / "
            f"(||ψ_iPESS||²·||ψ_ms||²) = {fidelity:.12f} != 1. "
            f"Multisite encoding does not faithfully represent the iPESS "
            f"state (plan stop-and-ask checkpoint #2)."
        ),
    )


def _torus_brute_force_kagome_energy(psi: jnp.ndarray, H_pair: jnp.ndarray) -> float:
    """Brute-force kagome XXZ energy per microscopic site on the 1-cell torus.

    The 1-cell PBC kagome torus has 3 sites (``u, v, w``) and 6 bonds: 3
    up-triangle (``u-v, u-w, v-w``) plus 3 down-triangle bonds on the same
    pairs.  Each pair therefore appears twice, so

        ⟨H_kagome⟩_torus = 2 (⟨H_uv⟩ + ⟨H_uw⟩ + ⟨H_vw⟩) ,

    and per microscopic site = ⟨H_kagome⟩_torus / 3.

    Each ⟨H_xy⟩ is computed from the 2-site marginal of |ψ⟩⟨ψ|, contracted
    against ``H_pair`` in the ``(s1_ket, s2_ket, s1_bra, s2_bra)``
    convention used by :func:`tenax.algorithms._ctm_tensor_energy._rdm2x1_tensor_2site`.
    """
    psi_c = psi.conj()
    norm_sq = jnp.real(jnp.einsum("uvw,uvw->", psi_c, psi))
    rho_uv = jnp.einsum("ijw,klw->ijkl", psi, psi_c) / norm_sq
    rho_uw = jnp.einsum("ivj,kvl->ijkl", psi, psi_c) / norm_sq
    rho_vw = jnp.einsum("uij,ukl->ijkl", psi, psi_c) / norm_sq
    e_uv = jnp.einsum("ijkl,ijkl->", rho_uv, H_pair).real
    e_uw = jnp.einsum("ijkl,ijkl->", rho_uw, H_pair).real
    e_vw = jnp.einsum("ijkl,ijkl->", rho_vw, H_pair).real
    return float(2.0 * (e_uv + e_uw + e_vw) / 3.0)


def _torus_formula_energy(
    psi: jnp.ndarray,
    bond_gates: dict,
    neighbors: dict,
    d: int,
) -> float:
    """Wavefunction-direct version of :func:`compute_energy_pess_3site_multisite`.

    Mirrors the dispatcher logic of
    :func:`tenax.algorithms._pess_multisite_energy.compute_energy_pess_3site_multisite`
    — same NN visits ``("u","right"), ("u","bottom"), ("w","right"),
    ("w","bottom")`` plus the 2 marginalised b-c bonds keyed by
    ``frozenset({("v","right"), ("w","left")})`` (up-triangle, mediated by
    ``T_u``) and ``frozenset({("v","bottom"), ("w","top")})`` (down-triangle,
    mediated by ``T_d``) — but builds every RDM by tracing |ψ_torus⟩
    directly rather than from a CTM environment.

    A correct bond decomposition has

        Σ_{NN bonds} ⟨H_pair · ρ_pair⟩
        + Σ_{marginalised bonds} ⟨H_pair · ρ_(vw)⟩
        = ⟨H_kagome⟩_torus

    on the 1-cell PBC kagome torus, divided by ``len(neighbors)`` for the
    per-microscopic-site normalisation.
    """
    psi_c = psi.conj()
    norm_sq = jnp.real(jnp.einsum("uvw,uvw->", psi_c, psi))

    # 2-site marginals from |ψ_torus⟩ in the same (s_A_ket, s_B_ket, s_A_bra,
    # s_B_bra) convention as :func:`_rdm2x1_tensor_2site` /
    # :func:`_rdm1x2_tensor_2site`.  ψ has shape ``(p_u, p_v, p_w)``.
    rho_uv = jnp.einsum("ijw,klw->ijkl", psi, psi_c) / norm_sq
    rho_wu = jnp.einsum("jvi,lvk->ijkl", psi, psi_c) / norm_sq
    rho_vw = jnp.einsum("uij,ukl->ijkl", psi, psi_c) / norm_sq

    def gate(bond_id):
        H = bond_gates[bond_id]
        return jnp.asarray(H).reshape(d, d, d, d)

    e = jnp.array(0.0)
    nn_visits = (
        ("u", "right"),  # u.right ↔ v.left  (up-triangle a-b)
        ("u", "bottom"),  # u.bottom ↔ v.top  (down-triangle a-b)
        ("w", "right"),  # w.right ↔ u.left  (up-triangle a-c, iterated from w)
        ("w", "bottom"),  # w.bottom ↔ u.top  (down-triangle a-c, iterated from w)
    )
    for name, direction in nn_visits:
        nb = neighbors[name][direction]
        rev = "left" if direction == "right" else "top"
        bond_id = frozenset([(name, direction), (nb, rev)])
        rdm = rho_uv if name == "u" else rho_wu
        e = e + jnp.einsum("ijkl,ijkl->", rdm, gate(bond_id))

    # 2 marginalised-3-site b-c bonds (row+col share ρ_(vw) on the 1-cell
    # torus because the marginal over u doesn't depend on row/col direction).
    e = e + jnp.einsum(
        "ijkl,ijkl->",
        rho_vw,
        gate(frozenset({("v", "right"), ("w", "left")})),
    )
    e = e + jnp.einsum(
        "ijkl,ijkl->",
        rho_vw,
        gate(frozenset({("v", "bottom"), ("w", "top")})),
    )

    return float(e.real / len(neighbors))


@pytest.mark.core
@pytest.mark.parametrize("D", [1, 2, 3, 4, 5, 6])
def test_pess_3site_multisite_energy_formula_matches_torus_brute_force(D):
    """Energy formula bond-accounting structural-correctness gate (Task B.4).

    Replays :func:`compute_energy_pess_3site_multisite`'s "4 NN + 2
    marginalised-3-site" bond decomposition with wavefunction-direct RDMs
    on the 1-cell PBC kagome torus and asserts equality with the
    brute-force ``⟨H_kagome⟩``.  CTM-free, AD-free, χ-free.

    What this catches (relative to the buggy 6-NN sum at commit ``bd27d8c``,
    see ``project_kagome_3site_energy_bug.md``):

    - **Wrong bond inclusion**: a regression that visits the 2 dim-1 v-w
      NN bonds as physical kagome bonds (instead of routing them through
      the marginalised-3-site helpers) would land on a different RDM at
      those bonds and the wf-direct sum would no longer reproduce the
      brute-force torus energy.
    - Wrong site pairing in any bond.
    - Wrong gate orientation / Hermiticity convention.
    - Missing factor of 2 (up-triangle + down-triangle).
    - Wrong ``1/n_sites`` normalisation.

    What this deliberately does NOT exercise (covered elsewhere):

    - The CTM-based RDM helpers
      (:func:`_rdm2x1_tensor_2site`, :func:`_rdm1x2_tensor_2site`,
      :func:`_rdm_3site_marginal_vw_row`,
      :func:`_rdm_3site_marginal_vw_col`) — those share contraction
      patterns with already-tested kin in ``_ctm_tensor_energy.py``.
    - End-to-end CTM correctness — tracked via Phase C.3 stop-and-ask
      checkpoint #5: the AD-optimised energy must satisfy
      ``E_multisite ≥ −0.43752`` (Liao 2017 asymptotic ground state).
    """
    from tenax.algorithms._pess_multisite_energy import (
        kagome_3site_bond_gates,
        kagome_xxz_pair_hamiltonian,
    )
    from tenax.core.lattice import kagome

    d = 2
    delta = 1.0
    state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(0))
    sites = pess_to_kagome_3site_multisite(
        state.R_a,
        state.R_b,
        state.R_c,
        state.T_u,
        state.T_d,
        state.lambdas,
    )

    psi = _contract_multisite_3cycle_pbc(sites)
    H_pair = jnp.asarray(kagome_xxz_pair_hamiltonian(delta, d))
    e_brute = _torus_brute_force_kagome_energy(psi, H_pair)

    bond_gates = kagome_3site_bond_gates(delta=delta, d=d)
    e_formula = _torus_formula_energy(psi, bond_gates, kagome().neighbor_map, d=d)

    np.testing.assert_allclose(
        e_formula,
        e_brute,
        atol=1e-12,
        err_msg=(
            f"D={D}: formula's wf-direct E={e_formula:.12f} disagrees with "
            f"brute-force E={e_brute:.12f} "
            f"(diff={e_formula - e_brute:.3e}).  "
            f"compute_energy_pess_3site_multisite's bond decomposition "
            f"(4 NN + 2 marginalised-3-site) is structurally incorrect; "
            f"see project_kagome_3site_energy_bug.md and plan "
            f"stop-and-ask checkpoint #3."
        ),
    )
