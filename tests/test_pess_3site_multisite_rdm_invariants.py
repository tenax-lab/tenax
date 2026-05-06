"""Brute-force vs multisite-CTM RDM diagnostic + structural-invariants gates.

See docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md for design.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import ctm_multisite
from tenax.algorithms._ctm_tensor_energy import (
    _rdm1x2_tensor_2site,
    _rdm2x1_tensor_2site,
)
from tenax.algorithms._pess_multisite_energy import (
    _rdm_3site_marginal_vw_col,
    _rdm_3site_marginal_vw_row,
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

# 3×3 PBC torus position → sublattice. See the plan's "Lattice constants".
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


def _contract_multisite_3x3_torus(sites: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Exact contraction of the 3-site multisite encoding on a 3×3 PBC torus.

    Sublattice = (x + y) mod 3 ↦ {0:u, 1:v, 2:w}. 9 sites at row-major
    positions 0..8. Each site has 4 virtual legs (top, bot, lft, rgt) and
    1 physical leg. Output: rank-9 array indexed by physical legs in
    row-major position order.

    Bond labels (1 letter per bond):
      Horizontal (right→left): a..i for the 9 H_BONDS pairs.
      Vertical   (bot→top):    j..r for the 9 V_BONDS pairs.
      Physical:                A..I (for positions 0..8).

    See the plan for derivation.
    """
    # Per-site einsum strings: each is "top, bot, lft, rgt, phys".
    strings = (
        "pjcaA",  # pos 0 (u): top=p (V6 wrap), bot=j (V0), lft=c (H2 wrap), rgt=a (H0)
        "qkabB",  # pos 1 (v): top=q (V7 wrap), bot=k (V1), lft=a (H0),     rgt=b (H1)
        "rlbcC",  # pos 2 (w): top=r (V8 wrap), bot=l (V2), lft=b (H1),     rgt=c (H2)
        "jmfdD",  # pos 3 (v): top=j (V0),      bot=m (V3), lft=f (H5 wrap),rgt=d (H3)
        "kndeE",  # pos 4 (w): top=k (V1),      bot=n (V4), lft=d (H3),     rgt=e (H4)
        "loefF",  # pos 5 (u): top=l (V2),      bot=o (V5), lft=e (H4),     rgt=f (H5)
        "mpigG",  # pos 6 (w): top=m (V3),      bot=p (V6), lft=i (H8 wrap),rgt=g (H6)
        "nqghH",  # pos 7 (u): top=n (V4),      bot=q (V7), lft=g (H6),     rgt=h (H7)
        "orhiI",  # pos 8 (v): top=o (V5),      bot=r (V8), lft=h (H7),     rgt=i (H8)
    )
    spec = ",".join(strings) + "->ABCDEFGHI"
    args = [sites[_POS_TO_NAME[p]] for p in range(9)]
    return jnp.einsum(spec, *args, optimize="optimal")


@pytest.mark.core
@pytest.mark.parametrize("D", [1, 2, 3])
def test_multisite_3x3_torus_translation_invariant_diagonal(D):
    """ψ on the 3×3 PBC torus must be invariant under the (1,-1) diagonal
    shift: this preserves the (x+y) mod 3 sublattice assignment, so a
    correct contraction reproduces ψ exactly under the induced 9-position
    permutation.

    Permutation π: (x,y) → ((x+1) mod 3, (y-1) mod 3). For row-major
    pos = 3y+x, the new pos for old pos p with (x = p%3, y = p//3) is
    new_pos = 3*((y-1) % 3) + ((x+1) % 3).
    """
    state = IPESSState.random(D=D, d=2, key=jax.random.PRNGKey(0))
    sites = pess_to_kagome_3site_multisite(
        state.R_a,
        state.R_b,
        state.R_c,
        state.T_u,
        state.T_d,
        state.lambdas,
    )
    psi = _contract_multisite_3x3_torus(sites)

    perm = tuple(3 * ((p // 3 - 1) % 3) + ((p % 3 + 1) % 3) for p in range(9))
    # Permute the AXES of psi by perm (axis 0 should now hold what was at axis perm[0]).
    psi_shifted = jnp.transpose(psi, perm)

    np.testing.assert_allclose(
        np.asarray(psi),
        np.asarray(psi_shifted),
        rtol=1e-12,
        atol=1e-12,
        err_msg=(
            f"Multisite 3×3 torus wavefunction not invariant under (1,-1) "
            f"diagonal shift at D={D}. Bond labelling in "
            f"_contract_multisite_3x3_torus is wrong."
        ),
    )


def _brute_force_rdm_from_torus_psi(
    psi: jnp.ndarray, sites_to_keep: tuple[int, ...]
) -> jnp.ndarray:
    """Reduce ρ = Tr_{rest}(|ψ⟩⟨ψ|) / ⟨ψ|ψ⟩ on the kept sites.

    Args:
        psi: rank-9 wavefunction from `_contract_multisite_3x3_torus` or
             equivalent, indexed by physical legs at row-major positions 0..8.
        sites_to_keep: ordered tuple of position indices in {0..8}. The
            returned ρ has axes (s_keep[0], s_keep[1], ..., s_keep[0]', ...)
            — i.e. the kept-axis order matches `sites_to_keep`, not the
            sorted numerical order that `jnp.tensordot` returns natively.
            The bra axes mirror the ket order.

    Returns:
        rho with shape `(d,) * (2 * len(sites_to_keep))`. To get the
        matrix form, reshape to `(d**k, d**k)` where k = len(sites_to_keep).
    """
    n = psi.ndim
    assert n == 9, f"expected rank-9 torus ψ, got rank-{n}"
    sites_to_trace = tuple(i for i in range(n) if i not in sites_to_keep)
    # tensordot contracts the traced axes with their conjugates. The output
    # places the surviving ket axes first (in sorted numerical order),
    # followed by the surviving bra axes (also in sorted order).
    rho = jnp.tensordot(psi, jnp.conj(psi), axes=(sites_to_trace, sites_to_trace))
    # Normalise.
    norm_sq = jnp.tensordot(psi, jnp.conj(psi), axes=(tuple(range(n)), tuple(range(n))))
    rho = rho / norm_sq

    # tensordot returns kept axes in sorted numerical order. Permute so
    # axes match `sites_to_keep`, with bra axes mirroring ket order.
    sorted_keep = sorted(sites_to_keep)
    perm_ket = [sorted_keep.index(p) for p in sites_to_keep]
    k = len(sites_to_keep)
    perm = tuple(perm_ket) + tuple(p + k for p in perm_ket)
    return jnp.transpose(rho, perm)


@pytest.mark.core
@pytest.mark.parametrize("D", [1, 2, 3])
def test_brute_force_rdm_is_physical(D):
    """Brute-force RDMs from the 3×3 torus ψ must be Hermitian, trace 1, PSD."""
    state = IPESSState.random(D=D, d=2, key=jax.random.PRNGKey(0))
    sites = pess_to_kagome_3site_multisite(
        state.R_a,
        state.R_b,
        state.R_c,
        state.T_u,
        state.T_d,
        state.lambdas,
    )
    psi = _contract_multisite_3x3_torus(sites)
    d = 2

    failures: list[str] = []
    for sites_to_keep in [(0, 1), (0, 3), (2, 0), (2, 5), (1, 2), (3, 6)]:
        rho_t = _brute_force_rdm_from_torus_psi(psi, sites_to_keep)
        k = len(sites_to_keep)
        rho = jnp.reshape(rho_t, (d**k, d**k))

        err_h = float(jnp.linalg.norm(rho - jnp.conj(rho.T)))
        if err_h >= 1e-10:
            failures.append(f"sites={sites_to_keep}: not Hermitian, ‖ρ-ρ†‖={err_h:.3e}")
        tr = complex(jnp.trace(rho))
        if abs(tr - 1.0) >= 1e-10:
            failures.append(f"sites={sites_to_keep}: trace ≠ 1, tr={tr}")
        eig = np.linalg.eigvalsh(np.asarray(rho))
        if eig.min() <= -1e-10:
            failures.append(f"sites={sites_to_keep}: not PSD, λ_min={eig.min():.3e}")

    assert not failures, f"D={D} failures:\n  " + "\n  ".join(failures)


def _collect_ctm_rdms(
    state: IPESSState, chi: int, max_iter: int = 100, conv_tol: float = 1e-10
) -> dict[str, jnp.ndarray]:
    """Run multisite-CTM on the encoded state and extract the 6 RDMs the
    energy formula consumes.

    Returns dict with keys {"uv_h", "uv_v", "wu_h", "wu_v", "vw_row", "vw_col"}.
    Each value has shape (d, d, d, d) in the same axis convention used by
    `compute_energy_pess_3site_multisite` (i.e. `(s_A, s_B, s_A', s_B')` for
    2-site bonds; for the marginalised v-w bonds it's `(s_v, s_w, s_v', s_w')`).
    """
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
    # Per-site projective normalisation (matches `build_pess_loss_3site_multisite`).
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
        # 4 NN bonds (matching nn_visits order in compute_energy_pess_3site_multisite).
        "uv_h": _rdm2x1_tensor_2site(S_u, S_v, env_u, env_v),
        "uv_v": _rdm1x2_tensor_2site(S_u, S_v, env_u, env_v),
        "wu_h": _rdm2x1_tensor_2site(S_w, S_u, env_w, env_u),
        "wu_v": _rdm1x2_tensor_2site(S_w, S_u, env_w, env_u),
        # 2 marginalised-3-site v-w bonds.
        "vw_row": _rdm_3site_marginal_vw_row(S_u, S_v, S_w, env_u, env_v, env_w),
        "vw_col": _rdm_3site_marginal_vw_col(S_u, S_v, S_w, env_u, env_v, env_w),
    }


@pytest.mark.algorithm
def test_collect_ctm_rdms_returns_six_physical_rdms():
    """Smoke test: at D=2 χ=8 SU-warmstart, all 6 RDMs are 4-tensor (d,d,d,d),
    each Hermitian when reshaped to (d²,d²) and trace 1."""
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    state = pess_simple_update(state, H, dt_schedule=[(0.1, 50)], D_max=2)

    rdms = _collect_ctm_rdms(state, chi=8, max_iter=30, conv_tol=1e-7)
    assert set(rdms.keys()) == {"uv_h", "uv_v", "wu_h", "wu_v", "vw_row", "vw_col"}
    for name, rdm in rdms.items():
        assert rdm.shape == (2, 2, 2, 2), f"{name}: shape={rdm.shape}"
        m = jnp.reshape(rdm, (4, 4))
        # Hermitian (loose tol — χ=8 is unconverged at D=2 but should still be ~Herm).
        err_h = float(jnp.linalg.norm(m - jnp.conj(m.T)))
        assert err_h < 1e-6, f"{name}: ‖ρ-ρ†‖={err_h:.3e}"
        # Trace ~1.
        tr = complex(jnp.trace(m))
        assert abs(tr - 1.0) < 1e-6, f"{name}: tr={tr}"


# Position-index representatives for the 6 brute-force RDMs (see plan table).
_BF_BOND_POSITIONS: dict[str, tuple[int, ...]] = {
    "uv_h": (0, 1),  # (u^(0,0), v^(1,0))
    "uv_v": (0, 3),  # (u^(0,0), v^(0,1))
    "wu_h": (2, 0),  # (w^(2,0), u^(0,0))
    "wu_v": (2, 5),  # (w^(2,0), u^(2,1))
    "vw_row": (1, 2),  # marginalise (0,1,2) over u → keep (1,2) = (v,w)
    "vw_col": (3, 6),  # marginalise (0,3,6) over u → keep (3,6) = (v,w)
}


def _brute_force_rdms(state: IPESSState) -> dict[str, jnp.ndarray]:
    """Compute the 6 brute-force RDMs from the 3×3 torus wavefunction.

    Returns dict with the same keys + axis convention as `_collect_ctm_rdms`.
    """
    sites = pess_to_kagome_3site_multisite(
        state.R_a,
        state.R_b,
        state.R_c,
        state.T_u,
        state.T_d,
        state.lambdas,
    )
    psi = _contract_multisite_3x3_torus(sites)
    out = {}
    for name, sites_to_keep in _BF_BOND_POSITIONS.items():
        rho = _brute_force_rdm_from_torus_psi(psi, sites_to_keep)
        # rho axes: (s_keep[0], s_keep[1], s_keep[0]', s_keep[1]') — already
        # matches the 4-tensor convention of `_collect_ctm_rdms`.
        out[name] = rho
    return out


@pytest.mark.core
def test_d1_brute_force_equals_ctm_rdms():
    """Gate #6: at D=1 the encoded state is a product state, so finite-torus
    brute-force = infinite-lattice CTM exactly. All 6 RDMs must agree to 1e-10
    at any χ ≥ 4."""
    state = IPESSState.random(D=1, d=2, key=jax.random.PRNGKey(0))
    rdms_bf = _brute_force_rdms(state)
    rdms_ctm = _collect_ctm_rdms(state, chi=4, max_iter=20, conv_tol=1e-10)
    for name in rdms_bf:
        diff = float(jnp.linalg.norm(rdms_bf[name] - rdms_ctm[name]))
        assert diff < 1e-10, (
            f"D=1 brute-force ≠ CTM for {name}: ‖Δρ‖_F = {diff:.3e}. "
            f"Either the multisite encoding has a leg-mapping bug at the "
            f"product-state level, or the CTM RDM extraction does."
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase C.3 blocker: multisite-CTM RDMs at D=2 χ=16 are non-PSD "
        "with O(1) negative eigenvalues (and 5/6 violate trace=1 by "
        "1e-6 to 1e-3). All 6 RDMs fail equally across both helper "
        "types (_rdm{2x1,1x2}_tensor_2site and _rdm_3site_marginal_vw_*), "
        "consistent with a shared upstream multisite-CTM environment "
        "construction bug rather than a bug confined to one helper. "
        "To be fixed in a follow-up PR; this test is the regression gate."
    ),
)
@pytest.mark.algorithm
def test_ctm_rdms_hermitian_psd_trace1_at_d2_chi16():
    """Gates #1-3: the 6 multisite-CTM RDMs at D=2 χ=16 SU-warmstart must be
    Hermitian (1e-10), PSD (λ_min ≥ -1e-10), and trace 1 (1e-8 — looser to
    absorb χ-conv noise).

    All gates assert on the same SU state; failure of any gate fires
    immediately and points at the offending RDM.
    """
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    state = pess_simple_update(
        state,
        H,
        dt_schedule=[(0.1, 100), (0.01, 100)],
        D_max=2,
    )

    rdms_ctm = _collect_ctm_rdms(state, chi=16, max_iter=100, conv_tol=1e-9)
    failures: list[str] = []
    for name, rdm in rdms_ctm.items():
        m = jnp.reshape(rdm, (4, 4))
        # Gate #1: Hermitian.
        err_h = float(jnp.linalg.norm(m - jnp.conj(m.T)))
        if err_h > 1e-10:
            failures.append(f"{name}: ‖ρ-ρ†‖_F={err_h:.3e} > 1e-10")
        # Gate #2: PSD. Hermitise first (real symmetric eigenvalues are
        # what physics demands; small Hermiticity violation is OK at χ=16).
        m_h = 0.5 * (m + jnp.conj(m.T))
        eig = np.linalg.eigvalsh(np.asarray(m_h))
        if eig.min() < -1e-10:
            failures.append(f"{name}: λ_min={eig.min():.3e} < -1e-10 (not PSD)")
        # Gate #3: trace 1 (looser tol).
        tr = complex(jnp.trace(m))
        if abs(tr - 1.0) > 1e-8:
            failures.append(f"{name}: |tr-1|={abs(tr - 1.0):.3e} > 1e-8")
    assert not failures, "Structural-invariants violations:\n  " + "\n  ".join(failures)
