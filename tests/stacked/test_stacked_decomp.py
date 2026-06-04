"""Gauge-invariant decomposition equivalence spine for the stacked migration (#566, P1c).

ASSESSMENT (Step A of the P1c task) — Outcome 1, tests-only:

The symmetric ``svd`` / ``qr`` / ``eigh`` paths in ``tenax.linalg`` do NOT branch
on ``TENAX_STACK_BLOCKSPARSE`` (the stacked-contraction gate).  They assemble
per-*charge-sector* dense matrices (``_group_blocks_by_bond_charge`` fuses several
distinct-shape blocks into one ``(rows, cols)`` matrix), which is a different
partition from ``stacked_blocks()`` (groups blocks by identical raw *shape*).
The sector-level decompositions are already batched via
``_grouped_decomp_by_shape`` (#572) under the SEPARATE ``TENAX_BATCH_BLOCKSPARSE``
gate, default-off.  A "stacked SVD over ``_data``" was therefore NOT implemented:
the partitions are orthogonal, and the localization probe shows decomposition is
~12 calls/sweep (noise) versus ``_get_block`` ~1428 — not a material structural-op
source.

These tests are the accuracy spine for P1d regardless of the path decision.  They
toggle both ``TENAX_STACK_BLOCKSPARSE`` *and* ``TENAX_BATCH_BLOCKSPARSE`` and assert
the gauge-INVARIANT quantities are unchanged: sorted singular values / eigenvalues
and the reconstruction ``A = U Σ Vh`` / ``A = Q R``.  Raw ``U``/``Vh``/``Q`` factors
are NEVER compared — vmap LAPACK flips signs/bases in degenerate subspaces (the #572
trap), so only invariants are contractual.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.linalg import eigh, qr, svd

from ._harness import assert_tiered, canonical_tensors

jax.config.update("jax_enable_x64", True)


# Left/right partition for the 5-leg fermionic iPEPS site tensors.
_LEFT = ["u", "d"]
_RIGHT = ["l", "r", "phys"]


def _sym_tensors():
    return {
        name: t for name, t in canonical_tensors() if name in ("ferm_D2", "ferm_D4")
    }


def _dense_tensor():
    """The U(1)-trivial DenseTensor iPEPS site exposed by the harness."""
    return dict(canonical_tensors())["dense_D2"]


def _run_svd(tensor):
    """Full (untruncated) block-sparse SVD; returns (U, s, Vh)."""
    U, s, Vh, _ = svd(
        tensor,
        left_labels=_LEFT,
        right_labels=_RIGHT,
        new_bond_label="bond",
    )
    return U, s, Vh


def _set_flags(monkeypatch, stack: str | None, batch: str | None):
    for var, val in (
        ("TENAX_STACK_BLOCKSPARSE", stack),
        ("TENAX_BATCH_BLOCKSPARSE", batch),
    ):
        if val is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, val)


# --------------------------------------------------------------------------- #
# 1. SymmetricTensor SVD: gauge-invariants equal across flag states.          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_svd_invariants_stack_flag_independent(name, monkeypatch):
    """svd does NOT branch on TENAX_STACK_BLOCKSPARSE (Outcome-1 finding):
    flag-off vs flag-on give identical sorted SVs and reconstruction.

    The svd path ignores the stacked-contraction gate entirely (it assembles
    per-charge-sector matrices, a different partition from stacked_blocks()).
    The assertion is therefore trivially identical — which is exactly the
    documented decision: there is no separate stacked-svd branch to diverge.
    """
    tensor = _sym_tensors()[name]

    _set_flags(monkeypatch, stack=None, batch=None)
    U_off, s_off, Vh_off = _run_svd(tensor)

    _set_flags(monkeypatch, stack="1", batch=None)
    U_on, s_on, Vh_on = _run_svd(tensor)

    # Gauge-invariant: sorted singular values (never raw U/Vh).
    assert_tiered(jnp.sort(s_off), jnp.sort(s_on), tier="fp")

    # Gauge-invariant: reconstruction in dense space must match.
    assert_tiered(
        _reconstruct(U_off, s_off, Vh_off),
        _reconstruct(U_on, s_on, Vh_on),
        tier="fp",
    )


@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_svd_invariants_batch_flag_independent(name, monkeypatch):
    """svd's per-sector batching gate (TENAX_BATCH_BLOCKSPARSE, #572) leaves
    sorted SVs and reconstruction invariant: vmap(f) == [f(x_i)]."""
    tensor = _sym_tensors()[name]

    _set_flags(monkeypatch, stack=None, batch=None)
    U_off, s_off, Vh_off = _run_svd(tensor)

    _set_flags(monkeypatch, stack=None, batch="1")
    U_on, s_on, Vh_on = _run_svd(tensor)

    assert_tiered(jnp.sort(s_off), jnp.sort(s_on), tier="fp")
    assert_tiered(
        _reconstruct(U_off, s_off, Vh_off),
        _reconstruct(U_on, s_on, Vh_on),
        tier="fp",
    )


@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_svd_reconstruction_recovers_input(name, monkeypatch):
    """Full block-sparse SVD reconstructs the original dense matricization."""
    tensor = _sym_tensors()[name]
    _set_flags(monkeypatch, stack=None, batch=None)
    U, s, Vh = _run_svd(tensor)

    A_ref = _matricize(tensor)
    recon = _reconstruct(U, s, Vh)
    assert_tiered(A_ref, recon, tier="fp")


def test_dense_svd_reconstruction_and_batch_flag_independent(monkeypatch):
    """DenseTensor (U(1)-trivial) SVD broadens the spine beyond fermionic.

    Mirrors the ferm SVD case: full SVD reconstructs the dense matricization,
    and sorted SVs + reconstruction are invariant across TENAX_BATCH_BLOCKSPARSE
    (the dense path shares the same _LEFT/_RIGHT partition and factor layout).
    """
    tensor = _dense_tensor()
    A_ref = _matricize(tensor)

    _set_flags(monkeypatch, stack=None, batch=None)
    U_off, s_off, Vh_off = _run_svd(tensor)

    _set_flags(monkeypatch, stack=None, batch="1")
    U_on, s_on, Vh_on = _run_svd(tensor)

    # Reconstruction recovers the original dense matricization.
    assert_tiered(A_ref, _reconstruct(U_off, s_off, Vh_off), tier="fp")

    # Gauge-invariant across the batch gate.
    assert_tiered(jnp.sort(s_off), jnp.sort(s_on), tier="fp")
    assert_tiered(
        _reconstruct(U_off, s_off, Vh_off),
        _reconstruct(U_on, s_on, Vh_on),
        tier="fp",
    )


# --------------------------------------------------------------------------- #
# 2. Plain-matrix degenerate-SV + rank-deficient guard.                       #
#    Relies on invariants (reconstruction), not raw factors.                  #
# --------------------------------------------------------------------------- #


def _plain_matrices():
    cases = dict(canonical_tensors())
    return {k: cases[k] for k in ("degenerate_sv", "rank_deficient")}


@pytest.mark.parametrize("name", ["degenerate_sv", "rank_deficient"])
def test_plain_svd_reconstruction_invariant(name):
    """Degenerate-SV / rank-deficient matrices: jnp SVD reconstruction is exact.

    Proves the accuracy contract relies on A = U Σ Vh, never on the (sign- and
    basis-ambiguous) raw U/Vh factors.
    """
    A = jnp.asarray(_plain_matrices()[name])
    U, s, Vh = jnp.linalg.svd(A, full_matrices=False)
    recon = (U * s[None, :]) @ Vh
    assert_tiered(A, recon, tier="fp")


@pytest.mark.parametrize("name", ["degenerate_sv", "rank_deficient"])
def test_plain_qr_reconstruction_invariant(name):
    """QR reconstruction A = Q R is exact on degenerate / rank-deficient input."""
    A = jnp.asarray(_plain_matrices()[name])
    Q, R = jnp.linalg.qr(A)
    assert_tiered(A, Q @ R, tier="fp")


# --------------------------------------------------------------------------- #
# 3. SymmetricTensor qr / eigh: gauge-invariants across the batch gate.       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_qr_reconstruction_batch_flag_independent(name, monkeypatch):
    """Symmetric QR: A = Q R reconstruction invariant under TENAX_BATCH_BLOCKSPARSE."""
    tensor = _sym_tensors()[name]
    A_ref = _matricize(tensor)

    _set_flags(monkeypatch, stack=None, batch=None)
    Q0, R0 = qr(tensor, left_labels=_LEFT, right_labels=_RIGHT, new_bond_label="bond")

    _set_flags(monkeypatch, stack=None, batch="1")
    Q1, R1 = qr(tensor, left_labels=_LEFT, right_labels=_RIGHT, new_bond_label="bond")

    recon0 = _matricize_factors(Q0, R0)
    recon1 = _matricize_factors(Q1, R1)
    assert_tiered(A_ref, recon0, tier="fp")
    assert_tiered(recon0, recon1, tier="fp")


def test_eigh_invariants_batch_flag_independent(monkeypatch):
    """Symmetric eigh on a multi-sector Hermitian H = M Mᴴ: sorted eigenvalues
    AND the gauge-invariant reconstruction H = V diag(ev) Vᴴ are invariant under
    the batch gate (never compare raw eigenvectors).

    The Hermitian is built with a genuinely multi-sector charge structure (the
    fused left index of ferm_D4 carries both FermionParity sectors) so the
    TENAX_BATCH_BLOCKSPARSE gate actually has multiple sectors to batch.
    """
    H = _hermitian_2leg(_sym_tensors()["ferm_D4"])
    H_ref = _matricize_2leg(H)

    _set_flags(monkeypatch, stack=None, batch=None)
    V0, ev0 = eigh(H, left_labels=["a"], right_labels=["b"], new_bond_label="bond")

    _set_flags(monkeypatch, stack=None, batch="1")
    V1, ev1 = eigh(H, left_labels=["a"], right_labels=["b"], new_bond_label="bond")

    # Gauge-invariant: sorted eigenvalues unchanged across the gate.
    assert_tiered(jnp.sort(ev0), jnp.sort(ev1), tier="fp")

    # Gauge-invariant: H = V diag(ev) Vᴴ recovers the input for BOTH gate states.
    assert_tiered(H_ref, _reconstruct_eigh(V0, ev0), tier="fp")
    assert_tiered(H_ref, _reconstruct_eigh(V1, ev1), tier="fp")


# --------------------------------------------------------------------------- #
# Dense-space helpers (gauge-invariant reconstruction only).                  #
# --------------------------------------------------------------------------- #


def _matricize(tensor):
    """Dense (rows=_LEFT, cols=_RIGHT) matricization of a SymmetricTensor."""
    dense = tensor.todense()
    labels = list(tensor.labels())
    perm = [labels.index(lbl) for lbl in _LEFT + _RIGHT]
    moved = jnp.transpose(dense, perm)
    left_dim = int(np.prod([moved.shape[i] for i in range(len(_LEFT))]))
    return moved.reshape(left_dim, -1)


def _reconstruct(U, s, Vh):
    """U Σ Vh in dense matricized space, using the SVD factor tensors."""
    Ud = _factor_to_matrix(U, bond_last=True)
    Vhd = _factor_to_matrix(Vh, bond_last=False)
    return (Ud * s[None, : Ud.shape[1]]) @ Vhd


def _matricize_factors(Q, R):
    Qd = _factor_to_matrix(Q, bond_last=True)
    Rd = _factor_to_matrix(R, bond_last=False)
    return Qd @ Rd


def _factor_to_matrix(factor, *, bond_last):
    """Reshape a U/Q (bond_last=True) or Vh/R (bond_last=False) factor tensor
    into a 2-D matrix with the bond axis as cols / rows respectively."""
    dense = factor.todense()
    labels = list(factor.labels())
    bond_axis = labels.index("bond")
    other = [i for i in range(len(labels)) if i != bond_axis]
    if bond_last:
        moved = jnp.transpose(dense, other + [bond_axis])
        rows = int(np.prod([dense.shape[i] for i in other])) if other else 1
        return moved.reshape(rows, dense.shape[bond_axis])
    else:
        moved = jnp.transpose(dense, [bond_axis] + other)
        cols = int(np.prod([dense.shape[i] for i in other])) if other else 1
        return moved.reshape(dense.shape[bond_axis], cols)


def _matricize_2leg(tensor):
    """Dense (rows="a", cols="b") matricization of a 2-leg SymmetricTensor."""
    dense = tensor.todense()
    labels = list(tensor.labels())
    perm = [labels.index("a"), labels.index("b")]
    return jnp.transpose(dense, perm)


def _reconstruct_eigh(V, ev):
    """H = V diag(ev) Vᴴ in dense space, from the eigh factor tensor V.

    V carries labels (left_labels..., bond); reuse the same bond-last layout as
    the SVD/QR U/Q factors so the bond axis becomes the eigenvector columns.
    """
    Vd = _factor_to_matrix(V, bond_last=True)
    return (Vd * jnp.asarray(ev)[None, : Vd.shape[1]]) @ Vd.conj().T


def _hermitian_2leg(tensor):
    """Build a multi-sector Hermitian 2-leg SymmetricTensor H = M Mᴴ from the
    fused left index of ``tensor`` so eigh has a real, well-defined spectrum.

    The fused (u, d) index of the fermionic site tensor carries both
    FermionParity sectors, so H has a genuine multi-block charge structure and
    the per-sector batch gate (TENAX_BATCH_BLOCKSPARSE) actually has several
    sectors to batch — not a single trivial-charge block.
    """
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.tensor import SymmetricTensor

    A = _matricize(tensor)  # rows = fused (u, d), row-major
    M = A @ A.conj().T  # (left_dim, left_dim), Hermitian PSD
    n = M.shape[0]

    # Charges of the fused left index, in the same row-major order as _matricize
    # (u outer, d inner) so they label the rows/cols of M correctly.
    labels = list(tensor.labels())
    sym = tensor.indices[0].symmetry
    cu = np.asarray(tensor.indices[labels.index("u")].charges)
    cd = np.asarray(tensor.indices[labels.index("d")].charges)
    fused = np.asarray(
        sym.fuse(np.repeat(cu, len(cd)), np.tile(cd, len(cu))), dtype=np.int32
    )
    assert fused.shape[0] == n and len(np.unique(fused)) > 1, (
        "expected a genuinely multi-sector fused index"
    )

    idx_a = TensorIndex.from_charges(sym, fused, FlowDirection.IN, label="a")
    idx_b = TensorIndex.from_charges(sym, fused, FlowDirection.OUT, label="b")
    return SymmetricTensor.from_dense(M, (idx_a, idx_b))
