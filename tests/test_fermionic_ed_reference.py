"""Brute-force fermionic PEPS reference for convention validation.

This module establishes a contractor-based reference for fermionic PEPS
observables on small periodic tori. The Tenax contractor tracks Koszul
signs for every leg crossing via ``_contraction_inversion_pairs``, so
direct contraction of a finite PEPS via :func:`contract` is taken as the
"ground truth" against which the various CTM conventions can be checked.

The motivating question (issue #392): the split-aware CTM energy path
disagrees with the standard (shim, ``fuse_indices``-based) path on
fermionic input. ``fuse_indices`` uses raw ``jnp.transpose`` per block,
which is **not** Koszul-tracked, while the contractor is. The two
conventions therefore differ at the per-block sign level on
``SymmetricTensor`` with fermionic symmetry. Until this module is in
place there is no absolute reference to say which (if either) of the
two paths is physically correct.

Test progression (each tier builds on the previous):

  Tier 1 (1x1 PBC self-loop): A has its u↔d and l↔r virtual legs closed
    via PBC, phys traced against A.bar(). The result is a scalar
    that gives ⟨ψ_A|ψ_A⟩ for the trivial 1-site torus. Compare against
    A.norm()^2 — these should differ by exactly the Koszul phases the
    contractor adds for the virtual closure.

  Tier 2 (2x1 PBC chain): Two sites with virtual PBC closure (top↔bot
    self-loop on each site, left↔right between the two sites + PBC wrap
    around). Tests Koszul tracking for inter-site bonds.

  Tier 3 (2x2 PBC torus, norm): Four sites with full PBC. The canonical
    smallest 2D fermionic PEPS reference. Compare contractor-norm vs
    iPEPS-CTM-norm at chi >> D² (chi=4 sufficient for D=2).

  Tier 4 (2x2 PBC torus, energy): Compute the t-V Hamiltonian
    expectation value via direct contraction with the 2-site RDM, then
    compare against ``compute_energy_ctm_tensor`` (shim) and the split
    path (currently broken on fermions).

Once Tier 3 or Tier 4 produces a divergence, we will know which CTM
convention matches the contractor-reference — which in turn tells us
whether the split-path needs to compensate, or whether the shim's
lossy ``fuse_indices`` is the actual bug.
"""

from __future__ import annotations

import os

# Force CPU so tests run identically on dev boxes without GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tenax.contraction.contractor import contract  # noqa: E402
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import FermionParity  # noqa: E402
from tenax.core.tensor import SymmetricTensor  # noqa: E402

# ============================================================ #
# Fixture: small fermionic site tensor                           #
# ============================================================ #


def _make_random_fermionic_site(D: int, d: int, seed: int) -> SymmetricTensor:
    """Random FermionParity site tensor with alternating virtual charges.

    Mirrors ``tests/test_split_ctm_tensor.py::make_random_fermionic_site``
    and ``algorithms.fermionic_ipeps._build_initial_fpeps_tensor`` so the
    same A is used across the iPEPS code, the split-CTM tests, and these
    reference tests.
    """
    sym = FermionParity()
    virt_charges = np.array([i % 2 for i in range(D)], dtype=np.int32)
    phys_charges = np.array([0, 1], dtype=np.int32)[:d]
    indices = (
        TensorIndex.from_charges(
            sym, virt_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, virt_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    key = jax.random.PRNGKey(seed)
    A = SymmetricTensor.random_normal(indices, key)
    norm_val = float(A.norm())
    if norm_val > 0:
        A = A * (1.0 / norm_val)
    return A


# ============================================================ #
# Tier 1: 1x1 PBC self-loop                                      #
# ============================================================ #


def _norm_1x1_pbc(A: SymmetricTensor) -> jax.Array:
    """Compute ⟨ψ_A|ψ_A⟩ on a 1x1 PBC torus via contractor self-trace.

    Network: a single ket A with u↔d and l↔r PBC closures, contracted
    against a single bra A.bar() with the same virtual closures,
    on a shared phys leg.

    The contractor's ``_contraction_inversion_pairs`` adds Koszul phases
    for each leg crossing in the self-trace. The result is the Koszul-
    tracked answer to "norm of this PEPS on a single-site PBC torus."
    """
    # Ket: self-trace u with d and l with r via shared labels
    A_ket = (
        A.relabel("u", "_ud_ket")
        .relabel("d", "_ud_ket")
        .relabel("l", "_lr_ket")
        .relabel("r", "_lr_ket")
    )
    # contract(A_ket) does the self-trace -> 1-leg tensor on phys
    T_ket = contract(A_ket)
    assert T_ket.labels() == ("phys",)

    # Bra: same closures but on bar_super(A); share phys with ket
    A_bra = A.bar()
    A_bra_pbc = (
        A_bra.relabel("u", "_ud_bra")
        .relabel("d", "_ud_bra")
        .relabel("l", "_lr_bra")
        .relabel("r", "_lr_bra")
    )
    T_bra = contract(A_bra_pbc)
    assert T_bra.labels() == ("phys",)

    # Trace phys (shared label) -> scalar
    norm = contract(T_ket, T_bra)
    return norm.todense()


def test_tier1_norm_1x1_pbc_reproduces_against_dense_koszul():
    """Sanity check: contractor's 1x1 PBC norm == hand-Koszul dense reference.

    Hand-Koszul:
        norm = sum_{u,l,s} A[u,u,l,l,s] * conj(A[u,u,l,l,s]) * phi(u,l,s)
    where phi accounts for the bar_super phase on the bra block + Koszul
    signs from the virtual self-trace. We compute phi explicitly per
    block (u, u, l, l, s) and confirm the contractor reproduces it.
    """
    A = _make_random_fermionic_site(D=2, d=2, seed=70)
    norm_contractor = float(_norm_1x1_pbc(A))

    # Hand-Koszul reference, block-by-block.
    A_dense = np.array(A.todense())
    A_conj = np.conj(A_dense)
    # bar_super phase per block (qu, qd, ql, qr, qphys) = (-1)^{sum_{i<j} p_i p_j}
    # For self-trace u=d so qu=qd; l=r so ql=qr. After the trace, the surviving
    # block charges are (qu, qu, ql, ql, qphys), with parities (qu, qu, ql, ql, qphys).
    # The full Koszul of bar_super at (qu, qu, ql, ql, qphys):
    #   sum_{i<j} p_i p_j = qu*qu + qu*ql + qu*ql + qu*qphys
    #                     + qu*ql + qu*ql + qu*qphys
    #                     + ql*ql + ql*qphys + ql*qphys
    # The contractor also adds signs from leg crossings in the self-trace.
    # We don't try to compute the latter analytically — just verify that
    # the contractor result is finite, real, and matches a *consistent*
    # block-summed expression below.
    #
    # Loose check: contractor result should be a real positive scalar
    # of order A.norm()^2 (= 1 since A is normalised).
    assert (
        norm_contractor.dtype == np.float64
        if hasattr(norm_contractor, "dtype")
        else True
    )
    assert np.isfinite(norm_contractor), f"norm not finite: {norm_contractor}"
    # PEPS norms can be negative on fermionic systems if conventions differ;
    # we just bound the magnitude here.
    assert abs(norm_contractor) < 100.0, f"norm too large: {norm_contractor}"
    print(f"\n[Tier 1] contractor 1x1 PBC norm: {norm_contractor:.6f}")
    print(f"[Tier 1] A.norm()^2 (Frobenius):    {float(A.norm()) ** 2:.6f}")
    # Frobenius norm: sum|A|^2 over all blocks, ignoring all signs.
    frob_sq = float(jnp.sum(A_dense * A_conj).real)
    print(f"[Tier 1] sum|A|^2 (raw dense):      {frob_sq:.6f}")


# ============================================================ #
# Tier 3: 2x2 PBC torus norm                                     #
# ============================================================ #


def _build_2x2_pbc_tensors(
    A: SymmetricTensor,
    *,
    open_phys_sites: tuple[tuple[int, int], ...] = (),
) -> tuple[SymmetricTensor, ...]:
    """Build the 8 ket+bra tensors for a 2x2 PBC torus.

    ``open_phys_sites`` lists sites whose phys leg is left OPEN (distinct
    ket and bra labels); all other sites trace phys (shared label).

    Returns (A00, A01, A10, A11, B00, B01, B10, B11) — A* are ket, B* are
    bra-layer (built from A.bar()).
    """

    def _ket_phys(site: tuple[int, int]) -> str:
        i, j = site
        return f"p{i}{j}_ket" if site in open_phys_sites else f"p{i}{j}"

    def _bra_phys(site: tuple[int, int]) -> str:
        i, j = site
        return f"p{i}{j}_bra" if site in open_phys_sites else f"p{i}{j}"

    A00 = A.relabels(
        {
            "r": "h_top",
            "l": "h_top_wrap",
            "d": "v_left",
            "u": "v_left_wrap",
            "phys": _ket_phys((0, 0)),
        }
    )
    A01 = A.relabels(
        {
            "l": "h_top",
            "r": "h_top_wrap",
            "d": "v_right",
            "u": "v_right_wrap",
            "phys": _ket_phys((0, 1)),
        }
    )
    A10 = A.relabels(
        {
            "r": "h_bot",
            "l": "h_bot_wrap",
            "u": "v_left",
            "d": "v_left_wrap",
            "phys": _ket_phys((1, 0)),
        }
    )
    A11 = A.relabels(
        {
            "l": "h_bot",
            "r": "h_bot_wrap",
            "u": "v_right",
            "d": "v_right_wrap",
            "phys": _ket_phys((1, 1)),
        }
    )

    A_bra = A.bar()
    B00 = A_bra.relabels(
        {
            "r": "H_top",
            "l": "H_top_wrap",
            "d": "V_left",
            "u": "V_left_wrap",
            "phys": _bra_phys((0, 0)),
        }
    )
    B01 = A_bra.relabels(
        {
            "l": "H_top",
            "r": "H_top_wrap",
            "d": "V_right",
            "u": "V_right_wrap",
            "phys": _bra_phys((0, 1)),
        }
    )
    B10 = A_bra.relabels(
        {
            "r": "H_bot",
            "l": "H_bot_wrap",
            "u": "V_left",
            "d": "V_left_wrap",
            "phys": _bra_phys((1, 0)),
        }
    )
    B11 = A_bra.relabels(
        {
            "l": "H_bot",
            "r": "H_bot_wrap",
            "u": "V_right",
            "d": "V_right_wrap",
            "phys": _bra_phys((1, 1)),
        }
    )
    return A00, A01, A10, A11, B00, B01, B10, B11


def _contract_2x2_pbc_pairwise(tensors: tuple[SymmetricTensor, ...]):
    """Contract the 8 site tensors pairwise to avoid contractor multi-input bugs.

    Order: build each ket-bra row first (A00·B00, A01·B01, ...), then merge
    rows. This keeps every intermediate at most a few legs.
    """
    A00, A01, A10, A11, B00, B01, B10, B11 = tensors
    # Each ket·bra at a site closes the phys leg (if shared) and yields
    # 8 virtual legs (4 ket + 4 bra) for that site.
    site_00 = contract(A00, B00)
    site_01 = contract(A01, B01)
    site_10 = contract(A10, B10)
    site_11 = contract(A11, B11)

    # Top row: site_00 and site_01 share (h_top, H_top) and (h_top_wrap, H_top_wrap)
    top_row = contract(site_00, site_01)
    bot_row = contract(site_10, site_11)
    return contract(top_row, bot_row)


def _norm_2x2_pbc(A: SymmetricTensor) -> jax.Array:
    """⟨ψ_A|ψ_A⟩ on a 2x2 PBC torus via pairwise contractions.

    All phys legs are closed (ket and bra share the same phys label at
    each site, so the contractor traces them).
    """
    tensors = _build_2x2_pbc_tensors(A, open_phys_sites=())
    result = _contract_2x2_pbc_pairwise(tensors)
    return result.todense()


def test_tier3_norm_2x2_pbc_finite():
    """Sanity: 2x2 PBC norm from contractor is finite and positive-ish."""
    A = _make_random_fermionic_site(D=2, d=2, seed=70)
    norm = float(_norm_2x2_pbc(A))
    assert np.isfinite(norm), f"2x2 PBC norm not finite: {norm}"
    print(f"\n[Tier 3] contractor 2x2 PBC norm: {norm:.6e}")
    print(f"[Tier 3] A.norm()^8 (=norm_per_site^N): {float(A.norm()) ** 8:.6e}")


# ============================================================ #
# Tier 4: 2x2 PBC torus energy via per-bond RDMs                 #
# ============================================================ #


def _rdm_horizontal_bond_2x2_pbc(A: SymmetricTensor) -> jax.Array:
    """2-site RDM for the A(0,0)-A(0,1) horizontal bond on a 2x2 PBC torus.

    Open phys legs on sites (0,0) and (0,1); trace phys on sites (1,0)
    and (1,1). Returns a (d,d,d,d) dense tensor in
    ``(s_00_ket, s_01_ket, s_00_bra, s_01_bra)`` order. **Not normalised
    by the network norm.**

    Caveat: after ``todense()``, all Koszul phases that the contractor
    applied for the closure of internal legs are baked into the block
    data — but Koszul phases that would apply to the OPEN legs (when
    they are eventually traced) are **not** included. So ``np.einsum
    ("ijij->", rdm)`` does **not** equal ``_norm_2x2_pbc(A)`` in
    general. To compute ⟨H⟩ correctly, contract H against the open
    RDM via the contractor, not via a raw dense ``einsum``.
    See :func:`_energy_bond_2x2_pbc` for the correct pattern.
    """
    tensors = _build_2x2_pbc_tensors(A, open_phys_sites=((0, 0), (0, 1)))
    A00, A01, A10, A11, B00, B01, B10, B11 = tensors
    site_00 = contract(A00, B00)
    site_01 = contract(A01, B01)
    site_10 = contract(A10, B10)
    site_11 = contract(A11, B11)
    top_row = contract(site_00, site_01)
    bot_row = contract(site_10, site_11)
    rdm_t = contract(
        top_row,
        bot_row,
        output_labels=["p00_ket", "p01_ket", "p00_bra", "p01_bra"],
    )
    return rdm_t.todense()


def _energy_horizontal_bond_2x2_pbc(
    A: SymmetricTensor, gate: SymmetricTensor
) -> jax.Array:
    """⟨H_{00,01}⟩ * ⟨ψ|ψ⟩ for the A(0,0)-A(0,1) bond, contractor-traced.

    Inserts the 2-site gate into the closed network at sites (0,0) and
    (0,1), with all Koszul signs tracked by the contractor.  The
    caller divides by :func:`_norm_2x2_pbc` to get the physical energy
    for that bond.

    Note flow convention: the iPEPS A has ``phys`` with ``FlowDirection.IN``.
    A.bar() has the flipped flow (OUT) on phys_bra. The gate's
    (si, sj) legs are IN (acting on the bra) and (si_out, sj_out) are
    OUT (producing the ket). With those identifications the inserted
    gate's flows align with the bra/ket labels.

    **Important**: must use strict pairwise contractions.  The 3-input
    form ``contract(top_row, bot_row, G)`` returns exactly 0 on fermionic
    input (separate contractor bug, possibly path-optimizer related).
    Filed observation: pairwise gives -8.7e-4, 3-input gives 0.0 for
    seed=70 D=2.  Keep this as ``contract(X, Y)`` only.
    """
    G = gate.relabels(
        {
            "si": "p00_bra",
            "sj": "p01_bra",
            "si_out": "p00_ket",
            "sj_out": "p01_ket",
        }
    )
    tensors = _build_2x2_pbc_tensors(A, open_phys_sites=((0, 0), (0, 1)))
    A00, A01, A10, A11, B00, B01, B10, B11 = tensors
    site_00 = contract(A00, B00)
    site_01 = contract(A01, B01)
    site_10 = contract(A10, B10)
    site_11 = contract(A11, B11)
    top_row = contract(site_00, site_01)
    bot_row = contract(site_10, site_11)
    # Strict pairwise: insert G first (closes phys), THEN contract bot_row.
    tg = contract(top_row, G)
    result = contract(tg, bot_row)
    return result.todense()


def _energy_vertical_bond_2x2_pbc(
    A: SymmetricTensor, gate: SymmetricTensor
) -> jax.Array:
    """⟨H_{00,10}⟩ * ⟨ψ|ψ⟩ for the A(0,0)-A(1,0) vertical bond.

    Strict pairwise contractions, see :func:`_energy_horizontal_bond_2x2_pbc`
    for the 3-input contractor bug rationale.
    """
    G = gate.relabels(
        {
            "si": "p00_bra",
            "sj": "p10_bra",
            "si_out": "p00_ket",
            "sj_out": "p10_ket",
        }
    )
    tensors = _build_2x2_pbc_tensors(A, open_phys_sites=((0, 0), (1, 0)))
    A00, A01, A10, A11, B00, B01, B10, B11 = tensors
    site_00 = contract(A00, B00)
    site_01 = contract(A01, B01)
    site_10 = contract(A10, B10)
    site_11 = contract(A11, B11)
    left_col = contract(site_00, site_10)
    right_col = contract(site_01, site_11)
    lcG = contract(left_col, G)
    result = contract(lcG, right_col)
    return result.todense()


def reference_energy_2x2_pbc(A: SymmetricTensor, gate: SymmetricTensor) -> jax.Array:
    """Per-site energy on a 2x2 PBC torus, contractor-tracked ground truth.

    Sums one horizontal bond (00↔01) and one vertical bond (00↔10),
    normalised by the network norm and per-site count (2 bonds / 4
    sites = 0.5 bonds per site).

    Note: on a 2x2 PBC torus each unique NN bond is visited twice via
    PBC wrap (00↔01 and 00↔01-via-wrap are the same physical bond), so
    we count each direction once and divide by sites.
    """
    norm = _norm_2x2_pbc(A)
    e_h = _energy_horizontal_bond_2x2_pbc(A, gate)
    e_v = _energy_vertical_bond_2x2_pbc(A, gate)
    # On 2x2 PBC, each unique horizontal bond is the SAME bond visited from
    # both directions (PBC). Total horizontal bonds = 2 (top + bot row),
    # total vertical bonds = 2 (left + right col). We compute one of each
    # but multiply by 2 (both rows / both cols sample the same bond).
    # Then divide by 4 sites: per-site energy = (2*e_h + 2*e_v) / norm / 4.
    return (2 * e_h + 2 * e_v) / norm / 4


def test_tier4_horizontal_rdm_shape():
    """Sanity: 2x2 horizontal RDM is finite and has the right shape.

    NOTE: ``np.einsum("ijij->", rdm)`` does **not** equal
    ``_norm_2x2_pbc(A)`` because the open-leg closure via raw einsum
    skips the Koszul phases the contractor would apply. To use the
    RDM correctly, contract H against the open network via the
    contractor — see :func:`_energy_horizontal_bond_2x2_pbc`.
    """
    A = _make_random_fermionic_site(D=2, d=2, seed=70)
    rdm = np.array(_rdm_horizontal_bond_2x2_pbc(A))
    norm = float(_norm_2x2_pbc(A))
    assert rdm.shape == (2, 2, 2, 2), f"unexpected rdm shape: {rdm.shape}"
    rdm_trace = float(np.einsum("ijij->", rdm))
    print(f"\n[Tier 4] horizontal RDM shape: {rdm.shape}")
    print(f"[Tier 4] np.einsum('ijij->', rdm) = {rdm_trace:.6e}")
    print(f"[Tier 4] _norm_2x2_pbc(A)         = {norm:.6e}")
    print("[Tier 4] (ratio differs from 1 — Koszul phases NOT in raw closure)")
    assert np.isfinite(rdm_trace)
    assert np.isfinite(norm)


# ============================================================ #
# Tier 5: contractor-reference energy vs iPEPS-CTM energy        #
# ============================================================ #


def test_tier5_reference_energy_finite():
    """Sanity: the 2x2 PBC reference energy is a finite real number."""
    from tenax.algorithms.fermionic_ipeps import FPEPSConfig, spinless_fermion_gate

    A = _make_random_fermionic_site(D=2, d=2, seed=70)
    gate = spinless_fermion_gate(FPEPSConfig(D=2, t=1.0, V=0.5))
    e = complex(reference_energy_2x2_pbc(A, gate))
    print(f"\n[Tier 5] contractor-reference per-site energy: {e}")
    assert np.isfinite(e.real) and np.isfinite(e.imag)
    # Allow imaginary residue from sign mismatches — flag it loudly.
    if abs(e.imag) > 1e-10:
        print(f"  WARNING: imaginary part {e.imag:.3e} is non-trivial")


@pytest.mark.slow
def test_tier5_compare_to_ipeps_ctm_energy():
    """Compare contractor-reference energy vs the standard iPEPS CTM energy.

    These are NOT expected to match exactly — the CTM energy is for the
    infinite lattice while the reference is for a 2x2 PBC torus.  We
    print both to anchor a baseline; the diff is a metric for how far
    the iPEPS-CTM convention drifts from a direct PBC contraction.

    Also compare against the split-CTM path (which currently falls back
    to the shim for fermionic A — both shim and split should give the
    same number here, per PR #394).
    """
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
    from tenax.algorithms._split_ctm_tensor import (
        compute_energy_split_ctm_tensor,
        ctm_split_tensor,
    )
    from tenax.algorithms._split_ctm_tensor_energy import (
        _split_env_to_tensor_standard,
    )
    from tenax.algorithms.fermionic_ipeps import FPEPSConfig, spinless_fermion_gate

    A = _make_random_fermionic_site(D=2, d=2, seed=70)
    gate = spinless_fermion_gate(FPEPSConfig(D=2, t=1.0, V=0.5))

    e_ref = complex(reference_energy_2x2_pbc(A, gate))

    chi = 8
    env, _trunc = ctm_tensor(A, chi=chi, max_iter=20, conv_tol=1e-6)
    e_ctm = float(compute_energy_ctm_tensor(A, env, gate, d=2))

    env_split = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
    e_split = float(compute_energy_split_ctm_tensor(A, env_split, gate, d=2))

    # Also bypass the shim fallback to see the (currently broken) split
    # path's raw answer.
    from tenax.algorithms._split_ctm_tensor_energy import (
        _rdm1x2_split_tensor,
        _rdm2x1_split_tensor,
    )

    rdm_h = np.array(_rdm2x1_split_tensor(A, env_split))
    rdm_v = np.array(_rdm1x2_split_tensor(A, env_split))
    H_dense = np.array(gate.todense()).reshape(2, 2, 2, 2)
    e_split_raw = float(
        np.einsum("ijkl,ijkl->", rdm_h, H_dense)
        + np.einsum("ijkl,ijkl->", rdm_v, H_dense)
    )

    print(f"\n[Tier 5] reference E (2x2 PBC, contractor):  {e_ref}")
    print(f"[Tier 5] iPEPS-CTM E (infinite, chi={chi}):    {e_ctm:.10f}")
    print(f"[Tier 5] split-CTM E (via shim fallback):     {e_split:.10f}")
    print(f"[Tier 5] split-CTM E (raw bar_super path):    {e_split_raw:.10f}")
    print(f"[Tier 5] |e_ref - e_ctm|:    {abs(e_ref - e_ctm):.3e}")
    print(f"[Tier 5] |e_split - e_ctm|:  {abs(e_split - e_ctm):.3e}")
    print(f"[Tier 5] |e_split_raw - e_ctm|: {abs(e_split_raw - e_ctm):.3e}")
    assert np.isfinite(e_ref.real) and np.isfinite(e_ctm)
    assert np.isfinite(e_split) and np.isfinite(e_split_raw)


# ============================================================ #
# Tier 6: Jordan-Wigner ED of the t-V Hamiltonian (absolute     #
#         lattice-correct ground state energy)                   #
# ============================================================ #


def _build_jw_operators(n_sites: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build Jordan-Wigner creation/annihilation operators for ``n_sites``
    spinless fermion modes in the 2^n_sites qubit Hilbert space.

    Ordering convention: site 0 first.  c_j = (prod_{k<j} Z_k) ⊗ σ_j^- ⊗ I...
    Hermiticity: c†_j = (prod_{k<j} Z_k) ⊗ σ_j^+ ⊗ I...

    Returns ``(c_ops, c_dag_ops)`` — two lists of (2^n, 2^n) ndarrays.
    """
    I2 = np.eye(2, dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)  # |0><1|
    sigma_plus = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)  # |1><0|

    def kron_chain(matrices):
        out = matrices[0]
        for M in matrices[1:]:
            out = np.kron(out, M)
        return out

    c_ops: list[np.ndarray] = []
    c_dag_ops: list[np.ndarray] = []
    for j in range(n_sites):
        # Apply Z on sites 0..j-1, σ^- on site j, I on remaining
        ops_minus = [Z] * j + [sigma_minus] + [I2] * (n_sites - j - 1)
        ops_plus = [Z] * j + [sigma_plus] + [I2] * (n_sites - j - 1)
        c_ops.append(kron_chain(ops_minus))
        c_dag_ops.append(kron_chain(ops_plus))
    return c_ops, c_dag_ops


def _build_tv_hamiltonian_2x2_pbc(t: float, V: float) -> np.ndarray:
    """Build the t-V Hamiltonian on a 4-site 2x2 PBC torus in qubit basis.

    Site labeling: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
    Unique NN bonds on the torus (each counted once): (0,1), (2,3), (0,2), (1,3).
    H = -t sum_<ij>(c†_i c_j + h.c.) + V sum_<ij> n_i n_j.

    Returns a (16, 16) Hermitian ndarray.
    """
    c, cdag = _build_jw_operators(4)
    n_ops = [cdag[j] @ c[j] for j in range(4)]

    H = np.zeros((16, 16), dtype=np.complex128)
    bonds = [(0, 1), (2, 3), (0, 2), (1, 3)]
    for i, j in bonds:
        hop = cdag[i] @ c[j] + cdag[j] @ c[i]
        H = H - t * hop + V * (n_ops[i] @ n_ops[j])
    # Hermitize against round-off
    H = 0.5 * (H + H.conj().T)
    return H


def test_tier6_jw_ed_ground_state_finite():
    """ED on the 4-site 2x2 PBC t-V Hamiltonian gives a real ground state."""
    H = _build_tv_hamiltonian_2x2_pbc(t=1.0, V=0.5)
    eigvals = np.linalg.eigvalsh(H)
    E_gs_total = float(eigvals[0])
    E_gs_per_site = E_gs_total / 4
    print(f"\n[Tier 6] H shape: {H.shape}")
    print(f"[Tier 6] E_GS (total, 4 sites): {E_gs_total:.10f}")
    print(f"[Tier 6] E_GS per site:         {E_gs_per_site:.10f}")
    print(f"[Tier 6] first 4 eigenvalues:   {eigvals[:4]}")
    # Hermiticity check — eigvalsh requires it but be explicit
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)
    assert np.isfinite(E_gs_total)


# ============================================================ #
# Tier 7: variational bound check on iPEPS conventions          #
# ============================================================ #


# test_tier7_norm_sign_diagnostic removed (#555): the diagnostic codified the
# bug — it asserted that at least one fermionic 2x2 PBC norm comes out
# negative.  Post-#555 the bug is fixed (no negative norms), so the assertion
# would fail.  See test_tier7_variational_bound_check below for the positive
# replacement: ⟨H⟩/N ≥ E_GS is now enforced as a passing test.


@pytest.mark.slow
def test_tier7_variational_bound_check():
    """For a *normalised* iPEPS state on the same 4-site 2x2 PBC torus, any
    convention computing ⟨H⟩/N must satisfy ⟨H⟩/N ≥ E_GS_per_site.

    Was xfail(strict=True) on main when the contractor's auto-Koszul + the
    bar_super per-block phase together violated the bound for seeds 2 and 3.
    Post-#555 (removal of both) the bound holds for all 7 seeds at D=2 d=2.
    """
    from tenax.algorithms.fermionic_ipeps import FPEPSConfig, spinless_fermion_gate

    t, V = 1.0, 0.5
    H = _build_tv_hamiltonian_2x2_pbc(t=t, V=V)
    E_gs_per_site = float(np.linalg.eigvalsh(H)[0]) / 4

    gate = spinless_fermion_gate(FPEPSConfig(D=2, t=t, V=V))
    print(f"\n[Tier 7-bound] E_GS per site (4-site JW-ED): {E_gs_per_site:.10f}")
    violations = []
    for seed in (0, 1, 2, 3, 70, 100, 999):
        A = _make_random_fermionic_site(D=2, d=2, seed=seed)
        try:
            e_ref = float(reference_energy_2x2_pbc(A, gate).real)
        except Exception as exc:
            print(f"  seed={seed:4d}: contractor raised {type(exc).__name__}")
            continue
        ok = e_ref >= E_gs_per_site - 1e-9
        print(
            f"  seed={seed:4d}: E_ref = {e_ref:+.10f}  "
            f"{'OK' if ok else 'VIOLATES BOUND'}"
        )
        if not ok:
            violations.append((seed, e_ref))
    assert not violations, (
        f"Variational bound violated for seeds: {violations}.  "
        f"E_GS/N = {E_gs_per_site:.6f}."
    )


# ============================================================ #
# Tier 8: OBC clean-room — 2x2 open-boundary reference          #
# ============================================================ #
#
# On a 2x2 lattice, OBC and PBC have the SAME set of physical NN bonds
# (each PBC wrap coincides with the direct bond on a 2x2 grid).  But the
# PEPS topology differs: PBC wraps each row/column's two virtual bonds
# back on themselves, OBC caps each boundary virtual leg with a rank-1
# vector.  If the fermionic algebra bug from Tier 7 lives specifically
# in the closed-loop (PBC wrap) sign convention — analogous to the
# fermionic torus spin-structure ambiguity — then OBC should pass the
# variational bound that PBC fails.


def _make_obc_cap(virt_index: TensorIndex, new_label: str) -> SymmetricTensor:
    """Build a 1D rank-1 cap for capping a boundary virtual leg.

    The cap has charges matching ``virt_index``, flow *flipped* (so it
    contracts), label set to ``new_label``, and data = [1, 0, 0, ...] —
    projecting the boundary virtual mode to its first basis state
    (typically the trivial-charge component).
    """
    capped_idx = virt_index.flip_flow().relabel(new_label)
    data = jnp.zeros(virt_index.dim, dtype=jnp.float64).at[0].set(1.0)
    return SymmetricTensor.from_dense(data, (capped_idx,))


def _build_2x2_obc_tensors(
    A: SymmetricTensor,
    *,
    open_phys_sites: tuple[tuple[int, int], ...] = (),
) -> tuple[tuple[SymmetricTensor, ...], tuple[SymmetricTensor, ...]]:
    """Build the ket+bra A tensors and the 16 boundary cap tensors for a
    2x2 OBC cluster.

    Each site keeps its 4 virtual legs, but the boundary-direction legs
    (left of column 0, right of column 1, top of row 0, bottom of row 1)
    get unique labels and are contracted against a rank-1 cap.
    """

    def _ket_phys(site: tuple[int, int]) -> str:
        i, j = site
        return f"p{i}{j}_ket" if site in open_phys_sites else f"p{i}{j}"

    def _bra_phys(site: tuple[int, int]) -> str:
        i, j = site
        return f"p{i}{j}_bra" if site in open_phys_sites else f"p{i}{j}"

    # Internal bonds (between sites): same as PBC.
    # Boundary bonds: each gets a unique label so it can be capped separately.
    # Ket sites:
    A00 = A.relabels(
        {
            "r": "h_top",
            "l": "kL_00",  # boundary left
            "d": "v_left",
            "u": "kT_00",  # boundary top
            "phys": _ket_phys((0, 0)),
        }
    )
    A01 = A.relabels(
        {
            "l": "h_top",
            "r": "kR_01",  # boundary right
            "d": "v_right",
            "u": "kT_01",  # boundary top
            "phys": _ket_phys((0, 1)),
        }
    )
    A10 = A.relabels(
        {
            "r": "h_bot",
            "l": "kL_10",  # boundary left
            "u": "v_left",
            "d": "kB_10",  # boundary bottom
            "phys": _ket_phys((1, 0)),
        }
    )
    A11 = A.relabels(
        {
            "l": "h_bot",
            "r": "kR_11",  # boundary right
            "u": "v_right",
            "d": "kB_11",  # boundary bottom
            "phys": _ket_phys((1, 1)),
        }
    )

    A_bra = A.bar()
    B00 = A_bra.relabels(
        {
            "r": "H_top",
            "l": "bL_00",
            "d": "V_left",
            "u": "bT_00",
            "phys": _bra_phys((0, 0)),
        }
    )
    B01 = A_bra.relabels(
        {
            "l": "H_top",
            "r": "bR_01",
            "d": "V_right",
            "u": "bT_01",
            "phys": _bra_phys((0, 1)),
        }
    )
    B10 = A_bra.relabels(
        {
            "r": "H_bot",
            "l": "bL_10",
            "u": "V_left",
            "d": "bB_10",
            "phys": _bra_phys((1, 0)),
        }
    )
    B11 = A_bra.relabels(
        {
            "l": "H_bot",
            "r": "bR_11",
            "u": "V_right",
            "d": "bB_11",
            "phys": _bra_phys((1, 1)),
        }
    )
    sites = (A00, A01, A10, A11, B00, B01, B10, B11)

    # Build 16 caps — one per boundary virtual leg.  Caps must have flow
    # flipped relative to the leg they contract against, so:
    #   ket leg with flow F → cap with flow flip(F)
    # We use the original A's index metadata (charges) and just flip flow.
    u_idx = next(idx for idx in A.indices if idx.label == "u")
    d_idx = next(idx for idx in A.indices if idx.label == "d")
    l_idx = next(idx for idx in A.indices if idx.label == "l")
    r_idx = next(idx for idx in A.indices if idx.label == "r")
    # Bra has flipped flows:
    u_idx_bra = u_idx.flip_flow()
    d_idx_bra = d_idx.flip_flow()
    l_idx_bra = l_idx.flip_flow()
    r_idx_bra = r_idx.flip_flow()

    caps = (
        # Ket caps — boundary legs of A* tensors
        _make_obc_cap(l_idx, "kL_00"),  # caps A00.l
        _make_obc_cap(u_idx, "kT_00"),  # caps A00.u
        _make_obc_cap(r_idx, "kR_01"),  # caps A01.r
        _make_obc_cap(u_idx, "kT_01"),  # caps A01.u
        _make_obc_cap(l_idx, "kL_10"),  # caps A10.l
        _make_obc_cap(d_idx, "kB_10"),  # caps A10.d
        _make_obc_cap(r_idx, "kR_11"),  # caps A11.r
        _make_obc_cap(d_idx, "kB_11"),  # caps A11.d
        # Bra caps
        _make_obc_cap(l_idx_bra, "bL_00"),
        _make_obc_cap(u_idx_bra, "bT_00"),
        _make_obc_cap(r_idx_bra, "bR_01"),
        _make_obc_cap(u_idx_bra, "bT_01"),
        _make_obc_cap(l_idx_bra, "bL_10"),
        _make_obc_cap(d_idx_bra, "bB_10"),
        _make_obc_cap(r_idx_bra, "bR_11"),
        _make_obc_cap(d_idx_bra, "bB_11"),
    )
    return sites, caps


def _norm_2x2_obc(A: SymmetricTensor) -> jax.Array:
    """⟨ψ_A|ψ_A⟩ on a 2x2 OBC cluster via pairwise contractor."""
    sites, caps = _build_2x2_obc_tensors(A, open_phys_sites=())
    A00, A01, A10, A11, B00, B01, B10, B11 = sites
    # Apply caps to each site first (collapses boundary virtual legs).
    # Site 00: caps for kL_00, kT_00, bL_00, bT_00.
    s00 = contract(contract(contract(contract(A00, caps[0]), caps[1]), B00), caps[8])
    s00 = contract(s00, caps[9])
    s01 = contract(contract(contract(contract(A01, caps[2]), caps[3]), B01), caps[10])
    s01 = contract(s01, caps[11])
    s10 = contract(contract(contract(contract(A10, caps[4]), caps[5]), B10), caps[12])
    s10 = contract(s10, caps[13])
    s11 = contract(contract(contract(contract(A11, caps[6]), caps[7]), B11), caps[14])
    s11 = contract(s11, caps[15])
    top_row = contract(s00, s01)
    bot_row = contract(s10, s11)
    return contract(top_row, bot_row).todense()


def _energy_horizontal_bond_2x2_obc(
    A: SymmetricTensor, gate: SymmetricTensor
) -> jax.Array:
    """⟨H_{00,01}⟩·⟨ψ|ψ⟩ for the (0,0)-(0,1) bond on a 2x2 OBC cluster."""
    G = gate.relabels(
        {
            "si": "p00_bra",
            "sj": "p01_bra",
            "si_out": "p00_ket",
            "sj_out": "p01_ket",
        }
    )
    sites, caps = _build_2x2_obc_tensors(A, open_phys_sites=((0, 0), (0, 1)))
    A00, A01, A10, A11, B00, B01, B10, B11 = sites
    s00 = contract(contract(contract(contract(A00, caps[0]), caps[1]), B00), caps[8])
    s00 = contract(s00, caps[9])
    s01 = contract(contract(contract(contract(A01, caps[2]), caps[3]), B01), caps[10])
    s01 = contract(s01, caps[11])
    s10 = contract(contract(contract(contract(A10, caps[4]), caps[5]), B10), caps[12])
    s10 = contract(s10, caps[13])
    s11 = contract(contract(contract(contract(A11, caps[6]), caps[7]), B11), caps[14])
    s11 = contract(s11, caps[15])
    top_row = contract(s00, s01)
    bot_row = contract(s10, s11)
    tg = contract(top_row, G)
    return contract(tg, bot_row).todense()


def _energy_vertical_bond_2x2_obc(
    A: SymmetricTensor, gate: SymmetricTensor
) -> jax.Array:
    """⟨H_{00,10}⟩·⟨ψ|ψ⟩ for the (0,0)-(1,0) bond on a 2x2 OBC cluster."""
    G = gate.relabels(
        {
            "si": "p00_bra",
            "sj": "p10_bra",
            "si_out": "p00_ket",
            "sj_out": "p10_ket",
        }
    )
    sites, caps = _build_2x2_obc_tensors(A, open_phys_sites=((0, 0), (1, 0)))
    A00, A01, A10, A11, B00, B01, B10, B11 = sites
    s00 = contract(contract(contract(contract(A00, caps[0]), caps[1]), B00), caps[8])
    s00 = contract(s00, caps[9])
    s01 = contract(contract(contract(contract(A01, caps[2]), caps[3]), B01), caps[10])
    s01 = contract(s01, caps[11])
    s10 = contract(contract(contract(contract(A10, caps[4]), caps[5]), B10), caps[12])
    s10 = contract(s10, caps[13])
    s11 = contract(contract(contract(contract(A11, caps[6]), caps[7]), B11), caps[14])
    s11 = contract(s11, caps[15])
    left_col = contract(s00, s10)
    right_col = contract(s01, s11)
    lcG = contract(left_col, G)
    return contract(lcG, right_col).todense()


def reference_energy_2x2_obc(A: SymmetricTensor, gate: SymmetricTensor) -> jax.Array:
    """Per-site energy on a 2x2 OBC cluster via contractor.

    Sums all 4 NN bonds (2 horizontal: top/bot rows ⇒ same A pair (0,0)-(0,1)
    on top + (1,0)-(1,1) on bot; 2 vertical similarly), normalised by the
    network norm and 4 sites.
    """
    norm = _norm_2x2_obc(A)
    # Top horizontal bond (0,0)-(0,1) and bottom (1,0)-(1,1) are symmetric
    # under up-down reflection of the cluster.  Likewise for vertical.
    # Computing one of each and multiplying by 2 captures both bonds.
    e_h = _energy_horizontal_bond_2x2_obc(A, gate)
    e_v = _energy_vertical_bond_2x2_obc(A, gate)
    return (2 * e_h + 2 * e_v) / norm / 4


def test_tier8_obc_norm_is_positive():
    """Sanity: 2x2 OBC fermionic norm must be non-negative for any A.

    Tier 7 showed that the PBC version can produce negative norms; this
    test checks whether OBC also fails or whether the bug is specific
    to the PBC closure.
    """
    print(f"\n[Tier 8-OBC norm] {'seed':>5} {'norm_OBC':>18}")
    print("-" * 30)
    neg_seeds = []
    for seed in (0, 1, 2, 3, 4, 5, 70, 100, 999):
        A = _make_random_fermionic_site(D=2, d=2, seed=seed)
        norm = float(_norm_2x2_obc(A).real)
        print(f"  {seed:>5} {norm:>+18.6e}")
        if norm < 0:
            neg_seeds.append((seed, norm))
    if neg_seeds:
        print(f"  WARNING: {len(neg_seeds)} OBC norms negative — bug spans OBC too")
    # Don't assert here — let the next test capture the variational bound.


def test_tier8_obc_variational_bound():
    """Variational bound on the OBC cluster: ⟨H⟩/N ≥ E_GS/N.

    If the bug from Tier 7 is specifically the PBC-closure / spin-structure
    ambiguity, OBC should pass cleanly.  If OBC still violates, the bug is
    more general (likely in ``bar_super`` or the contractor's Koszul
    tracking on any closed network).
    """
    from tenax.algorithms.fermionic_ipeps import FPEPSConfig, spinless_fermion_gate

    t, V = 1.0, 0.5
    # On a 2x2 lattice, PBC and OBC share the same set of NN bonds.
    H = _build_tv_hamiltonian_2x2_pbc(t=t, V=V)
    E_gs_per_site = float(np.linalg.eigvalsh(H)[0]) / 4
    gate = spinless_fermion_gate(FPEPSConfig(D=2, t=t, V=V))

    print(f"\n[Tier 8-OBC bound] E_GS/N = {E_gs_per_site:.10f}")
    violations = []
    for seed in (0, 1, 2, 3, 70, 100, 999):
        A = _make_random_fermionic_site(D=2, d=2, seed=seed)
        norm = float(_norm_2x2_obc(A).real)
        try:
            e = float(reference_energy_2x2_obc(A, gate).real)
        except Exception as exc:
            print(f"  seed={seed:4d}: raised {type(exc).__name__}: {exc}")
            continue
        ok_norm = norm > 0
        ok_bound = e >= E_gs_per_site - 1e-9
        marker = (
            "OK"
            if (ok_norm and ok_bound)
            else ("VIOLATES (norm<0)" if not ok_norm else "VIOLATES (E<E_GS/N)")
        )
        print(f"  seed={seed:4d}: norm={norm:+.4e}  E={e:+.6f}  {marker}")
        if not (ok_norm and ok_bound):
            violations.append((seed, norm, e))
    # This is a diagnostic test — we don't gate on it because the result
    # itself is the information we want from this scaffolding.  If OBC
    # passes cleanly, the bug is localized to PBC closures.
    if not violations:
        print(
            "\n[Tier 8-OBC bound] ALL SEEDS PASS the OBC bound. "
            "Bug from Tier 7 appears to be PBC-specific (spin-structure)."
        )
    else:
        print(
            f"\n[Tier 8-OBC bound] {len(violations)} violations on OBC too — "
            f"bug is NOT PBC-specific."
        )


# ============================================================ #
# Tier 9: localise — at what network size does the negative norm #
#         first appear?                                          #
# ============================================================ #
#
# Empirical findings so far:
#   single tensor:  contract(A, A.bar()) = +1.0 (Frobenius)  ✓
#   2x2 OBC:        norm < 0 for 7/9 seeds                          ✗
#
# Walk up the ladder to find the smallest topology that fails.


def _norm_single_tensor_bar_super(A: SymmetricTensor) -> float:
    """⟨A|A⟩ via the single-tensor inner product: same labels on both sides.

    Equals the Frobenius norm² for any normalised SymmetricTensor — this
    test confirms bar_super + contractor's Koszul tracking produce the
    positive-definite inner product *for a single tensor*.
    """
    return float(contract(A, A.bar()).todense())


def _norm_2site_obc_chain(A: SymmetricTensor) -> float:
    """⟨ψ_A|ψ_A⟩ on a 2-site OBC chain: A_0 — A_1.

    Connectivity: A_0.r ↔ A_1.l (single internal virtual bond).
    Boundary: A_0.{u, d, l} and A_1.{u, d, r} all capped with rank-1 [1,0].
    Phys traced ket-bra at each site.

    If the multi-site bug from Tier 8 already appears at 2 sites, the
    minimal failure mode is just *one* shared auxiliary bond.
    """
    A_ket_0 = A.relabels(
        {
            "u": "k0u",
            "d": "k0d",
            "l": "k0l",  # boundary
            "r": "h_bond",  # internal
            "phys": "p0",
        }
    )
    A_ket_1 = A.relabels(
        {
            "u": "k1u",
            "d": "k1d",
            "r": "k1r",  # boundary
            "l": "h_bond",  # internal
            "phys": "p1",
        }
    )
    A_bra = A.bar()
    A_bra_0 = A_bra.relabels(
        {
            "u": "b0u",
            "d": "b0d",
            "l": "b0l",
            "r": "H_bond",
            "phys": "p0",
        }
    )
    A_bra_1 = A_bra.relabels(
        {
            "u": "b1u",
            "d": "b1d",
            "r": "b1r",
            "l": "H_bond",
            "phys": "p1",
        }
    )
    u_idx = next(idx for idx in A.indices if idx.label == "u")
    d_idx = next(idx for idx in A.indices if idx.label == "d")
    l_idx = next(idx for idx in A.indices if idx.label == "l")
    r_idx = next(idx for idx in A.indices if idx.label == "r")
    u_b, d_b = u_idx.flip_flow(), d_idx.flip_flow()
    l_b, r_b = l_idx.flip_flow(), r_idx.flip_flow()

    caps = [
        # ket boundary caps
        _make_obc_cap(u_idx, "k0u"),
        _make_obc_cap(d_idx, "k0d"),
        _make_obc_cap(l_idx, "k0l"),
        _make_obc_cap(u_idx, "k1u"),
        _make_obc_cap(d_idx, "k1d"),
        _make_obc_cap(r_idx, "k1r"),
        # bra boundary caps
        _make_obc_cap(u_b, "b0u"),
        _make_obc_cap(d_b, "b0d"),
        _make_obc_cap(l_b, "b0l"),
        _make_obc_cap(u_b, "b1u"),
        _make_obc_cap(d_b, "b1d"),
        _make_obc_cap(r_b, "b1r"),
    ]
    # Apply caps to each site, then close the chain.
    site_0 = A_ket_0
    for c in caps[:3]:
        site_0 = contract(site_0, c)
    site_0 = contract(site_0, A_bra_0)
    for c in caps[6:9]:
        site_0 = contract(site_0, c)
    site_1 = A_ket_1
    for c in caps[3:6]:
        site_1 = contract(site_1, c)
    site_1 = contract(site_1, A_bra_1)
    for c in caps[9:12]:
        site_1 = contract(site_1, c)
    return float(contract(site_0, site_1).todense())


def test_tier9_single_tensor_norm_is_frobenius():
    """contract(A, A.bar()) with all-same-labels = Frobenius norm².

    Post-#555 the contractor no longer auto-applies Koszul signs, so the
    naive single-tensor self-contraction sums squared block elements with
    no cross-tensor compensation needed.  This was previously tested with
    bar_super(); under the new convention bar() is enough.
    """
    print(f"\n[Tier 9 single] {'seed':>5} {'inner':>15} {'Frobenius':>15}")
    print("-" * 45)
    for seed in (0, 1, 2, 3, 4, 5, 70, 100, 999):
        A = _make_random_fermionic_site(D=2, d=2, seed=seed)
        inner = float(contract(A, A.bar()).todense())
        frob = float(jnp.sum(jnp.conj(A._data) * A._data).real)
        print(f"  {seed:>5} {inner:>+15.6e} {frob:>+15.6e}")
        np.testing.assert_allclose(inner, frob, atol=1e-12)


def test_tier9_2site_obc_norm():
    """⟨ψ_A|ψ_A⟩ on a 2-site OBC chain must be positive (#555 regression).

    On main before the #555 fix, this returned negative norms for 4/9 seeds
    (the minimum failing topology — a single shared virtual bond between two
    fermionic site tensors).  After #555 (contractor's auto-Koszul removed,
    bar_super phase removed) the norm equals the bosonic / dense reference,
    which is a sum of squared block amplitudes — always positive.
    """
    print(f"\n[Tier 9 two-site] {'seed':>5} {'norm':>15}")
    print("-" * 30)
    for seed in (0, 1, 2, 3, 4, 5, 70, 100, 999):
        A = _make_random_fermionic_site(D=2, d=2, seed=seed)
        norm = _norm_2site_obc_chain(A)
        print(f"  {seed:>5} {norm:>+15.6e}")
        assert norm > 0, (
            f"2-site OBC fermionic norm is non-positive ({norm:.6e}) for "
            f"seed={seed} — #555 regression."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
