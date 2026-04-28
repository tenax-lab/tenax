"""Cross-path consistency: rank-4 honeycomb CTM ↔ square supersite CTM.

The 2-vertex bond reduced density matrix on the intra-cell (α=0) honeycomb
edge is a *gauge-invariant physical observable* — it must agree across any
two valid ways of computing it on the same iPEPS state. Two paths:

* **Rank-4 native** (PR #347): `_rdm2_bond({0: A, 1: B}, envs_rank4, alpha=0)`,
  where ``envs_rank4`` is the converged honeycomb-CTM environment.
* **Square supersite** (existing on main): contract ``A·B`` over the
  e0-bond and fuse ``(p_a, p_b)`` → ``p_ab``, run the standard rank-5
  square-lattice CTM (single-site unit cell), then take the 2-site square
  RDM and partial-trace one supersite. The remaining (d_eff, d_eff) matrix
  is *literally* ρ_AB on the intra-cell bond.

This is the M2b "kagome iPESS swap-out" smoke from the plan, in the form
that does not require pulling in the CG-iPEPS infrastructure (the CG path
on ``feat/coarse-grain-ipeps`` integrates the intra-cell bond into a
supersite via SU; here we build the same supersite directly from the
rank-4 (A, B) by contraction, no SU needed). The cross-path agreement
gates the rank-4 CTM topology / energy machinery against an independent
implementation.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_honeycomb_energy import _rdm2_bond
from tenax.algorithms._ctm_honeycomb_forward import honeycomb_ctm_run
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import _rdm2x1_tensor
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

# ------------------------------------------------------------------ #
# Fixtures                                                              #
# ------------------------------------------------------------------ #


def _make_near_product_honeycomb_site(
    d: int, eps: float, key: jax.Array
) -> DenseTensor:
    """Rank-4 honeycomb site close to a normalized product state.

    ``data[0, 0, 0, :]`` carries the random unit-norm product vector; every
    other entry is ``eps * complex_normal``. This is the same fixture used
    in the implicit-AD strict gate (``test_fd_vs_ad_d2_near_product_state``)
    and gives a state where biorthogonal CTM stably reaches an
    energy-converged fixed point in ~60 sweeps at chi=4.
    """
    sym = U1Symmetry()
    virt = np.zeros(2, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    data = np.zeros((2, 2, 2, d), dtype=np.complex128)
    re0 = np.array(jax.random.normal(key, (d,)))
    im0 = np.array(jax.random.normal(jax.random.fold_in(key, 1), (d,)))
    v0 = re0 + 1j * im0
    data[0, 0, 0, :] = v0 / np.linalg.norm(v0)

    pkey = jax.random.fold_in(key, 99)
    pre = np.array(jax.random.normal(pkey, (2, 2, 2, d)))
    pim = np.array(jax.random.normal(jax.random.fold_in(pkey, 1), (2, 2, 2, d)))
    pert = (pre + 1j * pim).astype(np.complex128)
    pert[0, 0, 0, :] = 0.0
    data = data + eps * pert
    data = data / np.linalg.norm(data)
    return DenseTensor(jnp.asarray(data), indices)


# ------------------------------------------------------------------ #
# Converter: rank-4 (A, B) → square supersite T_cg                      #
# ------------------------------------------------------------------ #


def _rank4_pair_to_supersite(A: DenseTensor, B: DenseTensor) -> DenseTensor:
    """Contract A·B over the intra-cell e0-bond, fuse phys → supersite.

    Honeycomb leg layout for A and B (both rank-4): ``(e0, e1, e2, phys)``.
    The intra-cell α=0 bond identifies ``A.e0`` with ``B.e0``; contracting
    over it produces a rank-6 tensor ``(e1_A, e2_A, p_a, e1_B, e2_B, p_b)``.
    Fusing the two physical legs gives a rank-5 *square* supersite tensor
    with labels ``(u, d, l, r, phys)`` (square-CTM convention) and physical
    dimension ``d_eff = d_a · d_b``.

    Leg mapping chosen for the test:

    ====== ============== =========================================
    Square Honeycomb leg  Notes
    ====== ============== =========================================
    ``l``  ``e1_A``       horizontal "left" virtual, owned by A
    ``r``  ``e2_B``       horizontal "right" virtual, owned by B
    ``u``  ``e1_B``       vertical "up" virtual, owned by B
    ``d``  ``e2_A``       vertical "down" virtual, owned by A
    ====== ============== =========================================

    Flow convention (square CTM): ``u=OUT, d=IN, l=OUT, r=IN, phys=IN``.
    The honeycomb e_i are all OUT; for the supersite we re-wrap with the
    flows demanded by the square convention. This is purely a re-labeling
    of the metadata — the underlying data tensor is identical.
    """
    A_data = A.todense()  # (D, D, D, d) for (e0, e1, e2, phys)
    B_data = B.todense()
    D = A_data.shape[0]
    d = A_data.shape[3]

    # Contract over A.e0 ↔ B.e0 (axis 0 of each)
    T = jnp.einsum("xLRa,xUDb->LRUDab", A_data, B_data)
    # T.shape == (D, D, D, D, d, d) for (e1_A, e2_A, e1_B, e2_B, p_a, p_b)

    # Reorder axes to square convention: (u=e1_B, d=e2_A, l=e1_A, r=e2_B, phys)
    # Source axis indices: e1_A=0, e2_A=1, e1_B=2, e2_B=3, p_a=4, p_b=5
    # Target order:        u(=e1_B=2), d(=e2_A=1), l(=e1_A=0), r(=e2_B=3), p_a, p_b
    T = jnp.transpose(T, (2, 1, 0, 3, 4, 5))
    T = T.reshape(D, D, D, D, d * d)

    sym = U1Symmetry()
    chi_charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d * d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, chi_charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, chi_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, chi_charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, chi_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    return DenseTensor(T, indices)


# ------------------------------------------------------------------ #
# Cross-path RDM agreement                                              #
# ------------------------------------------------------------------ #


@pytest.mark.slow
def test_rank4_rdm_matches_square_supersite_rdm():
    """ρ_AB(α=0) from rank-4 honeycomb CTM == ρ_supersite from square CTM.

    Builds a near-product D=2 uniform iPEPS (A=B), evaluates the intra-cell
    bond RDM through both paths, and compares element-wise. Diagonal
    entries agree to ``< 1e-3`` (verified during development; see the
    matrices in the assertion-failure printout). The looser ``< 1.5e-2``
    bound accommodates the *off-diagonal* "coherence" entries, which are
    sensitive to the residual U(1) gauge oscillation in the rank-4
    biorthogonal env on near-product states. Increasing ``chi`` does not
    tighten this; the limit is the gauge fluctuation, not truncation.

    Catches bugs that would change the RDM at the ``O(1)`` level
    (sublattice mix-ups, projector pair swap, wrong leg pairing, wrong
    α-bond identification) — those would fail with diff ~ 1, not 1e-2.
    """
    d = 2
    chi = 12
    A = _make_near_product_honeycomb_site(d=d, eps=0.05, key=jax.random.PRNGKey(7))
    sites_rank4 = {(0, 0): A, (1, 0): A}  # uniform A=B

    # --- Rank-4 path ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        envs_rank4, _ = honeycomb_ctm_run(
            sites_rank4,
            chi=chi,
            max_iter=80,
            conv_tol=1e-8,
            projector_method="biorthogonal",
            forward_gauge="phase",
        )
    rho_rank4 = _rdm2_bond(sites_rank4, envs_rank4, alpha=0)
    rho_rank4 = np.asarray(rho_rank4)
    assert rho_rank4.shape == (d * d, d * d)

    # --- Square supersite path ---
    T_cg = _rank4_pair_to_supersite(A, A)
    sites_sq = {(0, 0): T_cg}
    envs_sq, info = python_loop_ctm_converge(
        sites_sq,
        SINGLE_SITE_NEIGHBORS,
        chi=chi,
        max_iter=80,
        min_iter=4,
        conv_tol=1e-8,
        conv_method="sv",
        renormalize=True,
        projector_method="eigh",
    )
    # 2-site horizontal RDM on the supersite (4-supersite-leg array reshaped
    # to (d_eff, d_eff, d_eff, d_eff)). Partial-trace over the right
    # supersite to recover the single-supersite RDM == ρ_AB.
    rdm_2site = np.asarray(_rdm2x1_tensor(T_cg, envs_sq[(0, 0)]))
    # rdm_2site shape: (s1_ket, s2_ket, s1_bra, s2_bra) per the function docstring.
    # Trace over s2 (axes 1 and 3): rdm_left[s1_ket, s1_bra] = sum_j rdm[s1_ket, j, s1_bra, j]
    rho_supersite = np.einsum("ijkj->ik", rdm_2site)
    rho_supersite = 0.5 * (rho_supersite + rho_supersite.conj().T)
    rho_supersite = rho_supersite / (np.trace(rho_supersite) + 1e-30)

    # --- Compare ---
    diff = np.max(np.abs(rho_rank4 - rho_supersite))
    assert np.isfinite(rho_rank4).all() and np.isfinite(rho_supersite).all()
    assert diff < 1.5e-2, (
        f"max |ρ_rank4 - ρ_supersite| = {diff:.3e} > 1.5e-2\n"
        f"rho_rank4=\n{rho_rank4}\nrho_supersite=\n{rho_supersite}"
    )
    # Sanity: traces both = 1 exactly (normalization is path-invariant).
    assert abs(np.trace(rho_rank4) - 1.0) < 1e-10
    assert abs(np.trace(rho_supersite) - 1.0) < 1e-10


def test_supersite_converter_shape_and_flows():
    """Smoke: converter produces a rank-5 square site with correct shape + flows."""
    d, D = 2, 2
    A = _make_near_product_honeycomb_site(d=d, eps=0.05, key=jax.random.PRNGKey(0))
    B = _make_near_product_honeycomb_site(d=d, eps=0.05, key=jax.random.PRNGKey(1))
    T = _rank4_pair_to_supersite(A, B)

    # Rank 5, dims (D, D, D, D, d²)
    dims = tuple(idx.dim for idx in T.indices)
    assert dims == (D, D, D, D, d * d), dims

    # Square-CTM flow convention
    flows = {idx.label: idx.flow for idx in T.indices}
    assert flows["u"] == FlowDirection.OUT
    assert flows["d"] == FlowDirection.IN
    assert flows["l"] == FlowDirection.OUT
    assert flows["r"] == FlowDirection.IN
    assert flows["phys"] == FlowDirection.IN
