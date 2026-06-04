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


def test_eigh_eigenvalues_batch_flag_independent(monkeypatch):
    """Symmetric eigh on a Hermitian tensor: sorted eigenvalues invariant under
    the batch gate (never compare raw eigenvectors)."""
    tensor = _sym_tensors()["ferm_D4"]
    # Build a Hermitian operator H = A^dag A in the matricized space so eigh has
    # a well-defined spectrum; do it via the dense matricization then wrap back
    # is overkill — instead form a 2-leg Hermitian SymmetricTensor directly.
    H = _hermitian_2leg(tensor)

    _set_flags(monkeypatch, stack=None, batch=None)
    _, ev0 = eigh(H, left_labels=["a"], right_labels=["b"], new_bond_label="bond")

    _set_flags(monkeypatch, stack=None, batch="1")
    _, ev1 = eigh(H, left_labels=["a"], right_labels=["b"], new_bond_label="bond")

    assert_tiered(jnp.sort(ev0), jnp.sort(ev1), tier="fp")


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


def _hermitian_2leg(tensor):
    """Build a Hermitian 2-leg SymmetricTensor H = M^dag M from the left-leg
    fused index of ``tensor`` so eigh has a real, well-defined spectrum."""
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.tensor import SymmetricTensor

    A = _matricize(tensor)
    M = A @ A.conj().T  # (left_dim, left_dim), Hermitian PSD
    n = M.shape[0]

    # A single trivial-charge sector keeps the SymmetricTensor construction
    # simple while still exercising the block-sparse eigh code path.
    sym = tensor.indices[0].symmetry
    charges = np.zeros(n, dtype=np.int32)
    idx_a = TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="a")
    idx_b = TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="b")
    return SymmetricTensor.from_dense(M, (idx_a, idx_b))
