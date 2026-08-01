"""Tests for tenax.linalg module (svd, qr, eigh)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor
from tenax.linalg import eigh

IN = FlowDirection.IN
OUT = FlowDirection.OUT


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def hermitian_dense():
    """Small Hermitian DenseTensor (4x4 reshaped as 2x2 x 2x2)."""
    sym = U1Symmetry()
    key = jax.random.PRNGKey(42)
    M = jax.random.normal(key, (4, 4))
    M = M @ M.T  # make positive semidefinite
    indices = (
        TensorIndex.from_charges(sym, np.zeros(2, dtype=np.int32), IN, label="i"),
        TensorIndex.from_charges(sym, np.zeros(2, dtype=np.int32), OUT, label="j"),
        TensorIndex.from_charges(sym, np.zeros(2, dtype=np.int32), OUT, label="k"),
        TensorIndex.from_charges(sym, np.zeros(2, dtype=np.int32), IN, label="l"),
    )
    return DenseTensor(M.reshape(2, 2, 2, 2), indices)


@pytest.fixture
def hermitian_symmetric():
    """Small Hermitian SymmetricTensor with nontrivial U(1) charges."""
    sym = U1Symmetry()
    charges = np.array([-1, 0, 1], dtype=np.int32)
    idx_in = TensorIndex.from_charges(sym, charges, IN, label="row")
    idx_out = TensorIndex.from_charges(sym, charges, OUT, label="col")

    # Build a symmetric-compatible random tensor, then form M M^T
    key = jax.random.PRNGKey(7)
    A = SymmetricTensor.random_normal((idx_in, idx_out), key)
    # Form Hermitian: A @ A^dagger via dense
    A_d = A.todense()
    M = A_d @ A_d.conj().T
    return SymmetricTensor.from_dense(M, (idx_in, idx_out))


# ------------------------------------------------------------------ #
# eigh tests                                                           #
# ------------------------------------------------------------------ #


class TestEigh:
    def test_eigh_dense_basic(self, hermitian_dense):
        """eigh of a known Hermitian DenseTensor gives correct eigenvalues."""
        T = hermitian_dense
        V, eigvals = eigh(T, ["i", "j"], ["k", "l"], new_bond_label="ev")

        # Verify V has correct labels
        assert V.labels() == ("i", "j", "ev")
        # Eigenvalues should be sorted descending
        assert jnp.all(eigvals[:-1] >= eigvals[1:])
        # Eigenvalues should be non-negative (positive semidefinite input)
        assert jnp.all(eigvals >= -1e-10)

    def test_eigh_dense_reconstruction(self, hermitian_dense):
        """V @ diag(eigenvalues) @ V^T reconstructs the original matrix."""
        T = hermitian_dense
        V, eigvals = eigh(T, ["i", "j"], ["k", "l"], new_bond_label="ev")

        # Reconstruct: V @ diag(eigvals) @ V^T
        V_mat = V.todense().reshape(4, 4)
        reconstructed = V_mat @ jnp.diag(eigvals) @ V_mat.T
        original = T.todense().reshape(4, 4)
        np.testing.assert_allclose(reconstructed, original, atol=1e-10)

    def test_eigh_symmetric_matches_dense(self, hermitian_symmetric):
        """eigh of SymmetricTensor matches dense path."""
        T_sym = hermitian_symmetric
        V_sym, eigvals_sym = eigh(T_sym, ["row"], ["col"], new_bond_label="ev")

        # Dense reference
        M = T_sym.todense()
        M = 0.5 * (M + M.conj().T)
        eigvals_ref, eigvecs_ref = jnp.linalg.eigh(M)
        eigvals_ref = eigvals_ref[::-1]

        # Compare eigenvalues (sorted descending)
        np.testing.assert_allclose(
            np.sort(np.array(eigvals_sym))[::-1],
            np.sort(np.array(eigvals_ref))[::-1],
            atol=1e-10,
        )

    def test_eigh_truncation(self, hermitian_dense):
        """max_eigenvalues truncation keeps only top-k."""
        T = hermitian_dense
        V, eigvals = eigh(
            T, ["i", "j"], ["k", "l"], new_bond_label="ev", max_eigenvalues=2
        )
        assert len(eigvals) == 2
        assert V.todense().shape[-1] == 2


# ------------------------------------------------------------------ #
# Import compatibility tests                                           #
# ------------------------------------------------------------------ #


class TestImportCompat:
    def test_svd_import_from_contractor(self):
        """truncated_svd is still importable from contractor."""
        from tenax.contraction.contractor import truncated_svd

        assert callable(truncated_svd)

    def test_qr_import_from_contractor(self):
        """qr_decompose is still importable from contractor."""
        from tenax.contraction.contractor import qr_decompose

        assert callable(qr_decompose)

    def test_svd_from_init(self):
        """svd is importable from tenax top-level."""
        from tenax import svd as svd_init  # noqa: F811

        assert callable(svd_init)

    def test_qr_from_init(self):
        """qr is importable from tenax top-level."""
        from tenax import qr as qr_init  # noqa: F811

        assert callable(qr_init)

    def test_eigh_from_init(self):
        """eigh is importable from tenax top-level."""
        from tenax import eigh as eigh_init  # noqa: F811

        assert callable(eigh_init)

    def test_svd_is_truncated_svd(self):
        """svd and truncated_svd are the same function."""
        from tenax import svd as svd_top  # noqa: F811
        from tenax import truncated_svd
        from tenax.linalg import svd as linalg_svd

        assert truncated_svd is linalg_svd
        assert svd_top is linalg_svd


# ------------------------------------------------------------------ #
# base_charges per-sector keep on the eager path (#558)               #
# ------------------------------------------------------------------ #


class TestSvdBaseChargesEagerPath:
    """Eager-path block-sparse SVD honors base_charges for per-sector keep.

    The eager path historically did global democratic truncation regardless
    of base_charges (only the traced/AD path consumed it). Callers that
    depend on a canonical bond layout being preserved across iterations
    (fpeps SU at D>2 — #558) need per-sector keep matching base_charges.
    Tests cover both fermionic and bosonic symmetries since the SVD code is
    shared.
    """

    def _build_2leg_sym(self, sym, left_charges, right_charges, seed=0):
        idx_l = TensorIndex.from_charges(sym, left_charges, IN, label="l")
        idx_r = TensorIndex.from_charges(sym, right_charges, OUT, label="r")
        return SymmetricTensor.random_normal((idx_l, idx_r), jax.random.PRNGKey(seed))

    def test_eager_base_charges_preserves_canonical_layout_u1(self):
        """U(1) eager SVD with base_charges keeps {-1:1, 0:1, 1:1} layout."""
        from tenax.linalg import svd

        sym = U1Symmetry()
        charges = np.array([-1, 0, 1, -1, 0, 1], dtype=np.int32)
        T = self._build_2leg_sym(sym, charges, charges, seed=11)
        # Canonical layout has 2 of each charge; ask for 3 total to force a
        # choice. Without base_charges, global truncation may keep 3 from one
        # sector. With base_charges=[-1,0,1], we expect exactly {-1:1,0:1,1:1}.
        U, s, Vh, _ = svd(
            T,
            left_labels=["l"],
            right_labels=["r"],
            new_bond_label="bond",
            max_singular_values=3,
            base_charges=np.array([-1, 0, 1], dtype=np.int32),
        )
        bond_idx = U.indices[U.labels().index("bond")]
        counts = {int(q): 0 for q in (-1, 0, 1)}
        for q in bond_idx.charges:
            counts[int(q)] += 1
        assert counts == {-1: 1, 0: 1, 1: 1}, (
            f"bond charges {bond_idx.charges.tolist()} not balanced — "
            f"got counts {counts}"
        )
        assert s.shape == (3,)

    def test_eager_base_charges_preserves_canonical_layout_fermionic(self):
        """FermionParity eager SVD with base_charges keeps {0:2, 1:2}."""
        from tenax.core.symmetry import FermionParity
        from tenax.linalg import svd

        sym = FermionParity()
        charges = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        T = self._build_2leg_sym(sym, charges, charges, seed=22)
        U, s, Vh, _ = svd(
            T,
            left_labels=["l"],
            right_labels=["r"],
            new_bond_label="bond",
            max_singular_values=4,
            base_charges=np.array([0, 1, 0, 1], dtype=np.int32),
        )
        bond_idx = U.indices[U.labels().index("bond")]
        counts = {0: 0, 1: 0}
        for q in bond_idx.charges:
            counts[int(q)] += 1
        assert counts == {0: 2, 1: 2}, (
            f"fermionic bond charges {bond_idx.charges.tolist()} not balanced — "
            f"got counts {counts}"
        )
        assert s.shape == (4,)

    def test_eager_no_base_charges_keeps_global_truncation(self):
        """Without base_charges, eager SVD still does global democratic keep."""
        from tenax.linalg import svd

        sym = U1Symmetry()
        charges = np.array([-1, 0, 1, -1, 0, 1], dtype=np.int32)
        T = self._build_2leg_sym(sym, charges, charges, seed=33)
        U, s, Vh, _ = svd(
            T,
            left_labels=["l"],
            right_labels=["r"],
            new_bond_label="bond",
            max_singular_values=3,
        )
        # No constraint asserted on bond charge distribution — just that we
        # got 3 singular values total (global truncation).
        assert s.shape == (3,)

    def test_eager_base_charges_emits_kept_in_canonical_position_order(self):
        """bond_charges + s_final follow base_charges position order, not SV magnitude.

        Codex review of PR #560 flagged: emitting kept in global SV-magnitude
        order silently corrupts downstream ``scale_bond_axis`` calls on the
        unchanged opposite bond axis (whose ``idx.charges`` is the canonical
        pattern). The scale vector slice ``scale[np.where(idx.charges == q)]``
        then picks values that are NOT actually for sector q.

        Test: build a SymmetricTensor whose theta SVD has clearly-ordered
        singular values per sector (charge=1 sector dominates charge=0 by
        construction), then verify that after the eager SVD with
        base_charges, ``s_final[i]`` is the within-sector i-th value for the
        sector ``base_charges[i]`` — i.e., the lambda vector lines up with
        canonical positions.
        """
        from tenax.core.symmetry import FermionParity
        from tenax.linalg import svd

        sym = FermionParity()
        # 4-dim virtual leg with 2 even (charge 0) and 2 odd (charge 1).
        charges = np.array([0, 1, 0, 1], dtype=np.int32)
        idx_l = TensorIndex.from_charges(sym, charges, IN, label="l")
        idx_r = TensorIndex.from_charges(sym, charges, OUT, label="r")
        # Build a tensor where sector 1 has clearly larger SVs than sector 0,
        # so global ordering would interleave [1, ..., 0, ...] differently
        # from canonical [0, 1, 0, 1].
        T_dense = np.zeros((4, 4), dtype=np.float64)
        # Sector 0 block (positions [0, 2] x [0, 2]):
        T_dense[0, 0] = 2.0  # smaller singular values for sector 0
        T_dense[2, 2] = 1.0
        # Sector 1 block (positions [1, 3] x [1, 3]):
        T_dense[1, 1] = 10.0  # larger singular values for sector 1
        T_dense[3, 3] = 5.0
        T = SymmetricTensor.from_dense(T_dense, (idx_l, idx_r))

        U, s, Vh, _ = svd(
            T,
            left_labels=["l"],
            right_labels=["r"],
            new_bond_label="bond",
            max_singular_values=4,
            base_charges=np.array([0, 1, 0, 1], dtype=np.int32),
        )
        bond_idx = U.indices[U.labels().index("bond")]
        bond_charges = bond_idx.charges.tolist()
        s_vals = np.array(s).tolist()

        # Canonical: positions 0, 2 are sector 0; positions 1, 3 are sector 1.
        # If kept is in canonical order, bond_charges == [0, 1, 0, 1].
        # If kept is in global SV order, bond_charges would be [1, 1, 0, 0]
        # (sectors interleaved by SV magnitude).
        assert bond_charges == [0, 1, 0, 1], (
            f"bond_charges {bond_charges} not in canonical [0,1,0,1] order — "
            f"would silently corrupt scale_bond_axis on opposite axis"
        )
        # s_final[0] = top SV of sector 0 = 2.0
        # s_final[1] = top SV of sector 1 = 10.0
        # s_final[2] = 2nd SV of sector 0 = 1.0
        # s_final[3] = 2nd SV of sector 1 = 5.0
        np.testing.assert_allclose(s_vals, [2.0, 10.0, 1.0, 5.0], atol=1e-10)

    def test_eager_base_charges_honors_max_truncation_err(self):
        """max_truncation_err must trim n_keep below max_singular_values.

        Codex P2 review of PR #560 flagged: with base_charges + max_singular_values
        + max_truncation_err, the fix originally derived target_count from the
        hard cap (max_singular_values) instead of the effective n_keep, so a
        tight err cutoff was silently ignored. The fix uses n_keep (post-err
        cutoff) to build target_charges.
        """
        from tenax.core.symmetry import FermionParity
        from tenax.linalg import svd

        sym = FermionParity()
        charges = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        idx_l = TensorIndex.from_charges(sym, charges, IN, label="l")
        idx_r = TensorIndex.from_charges(sym, charges, OUT, label="r")
        # Rapidly-decaying SVs: 1e-3 err cutoff keeps only the top two.
        T_dense = np.diag(np.array([10.0, 9.0, 0.001, 0.001, 0.0001, 0.0001]))
        T = SymmetricTensor.from_dense(T_dense, (idx_l, idx_r))

        U, s, _, _ = svd(
            T,
            left_labels=["l"],
            right_labels=["r"],
            new_bond_label="bond",
            max_singular_values=6,
            max_truncation_err=1e-3,
            base_charges=np.array([0, 1], dtype=np.int32),
        )
        # Expect 2 SVs (the two big ones); not 6 (the hard cap).
        assert s.shape == (2,), (
            f"max_truncation_err ignored when base_charges present — "
            f"got {s.shape[0]} SVs, expected 2"
        )
        bond_charges = U.indices[U.labels().index("bond")].charges.tolist()
        assert bond_charges == [0, 1], (
            f"bond_charges {bond_charges} should follow canonical order from "
            f"_derive_charges(base_charges, n_keep=2) = [0, 1]"
        )
        np.testing.assert_allclose(np.array(s).tolist(), [10.0, 9.0], atol=1e-10)

    def test_eager_base_charges_expands_n_keep_to_honor_err(self):
        """n_keep expands iteratively when the canonical prefix violates err.

        Codex P2 follow-up on #561: when ``base_charges`` forces the canonical
        prefix to keep weaker SVs from required sectors while discarding
        larger ones from over-represented sectors, the global-cumulative err
        estimate underestimates the actual err. The fix iteratively expands
        ``n_keep`` (up to ``max_singular_values``) until the canonical-prefix
        kept set actually meets the err budget.

        Setup: ``base_charges=[0,1,1]`` with sector SVs ``q0:[1]``,
        ``q1:[100, 90]``. Global top-2 = ``[100, 90]`` meets a tight err
        budget, but ``_derive_charges([0,1,1], 2) = [0,1]`` keeps
        ``[1, 100]`` and discards ``90`` → actual err = 0.67 ≫ 0.05.
        Expansion to ``n_keep=3`` keeps all three SVs and meets the budget.
        """
        from tenax.linalg import svd

        sym = U1Symmetry()
        charges = np.array([0, 1, 1], dtype=np.int32)
        idx_l = TensorIndex.from_charges(sym, charges, IN, label="l")
        idx_r = TensorIndex.from_charges(sym, charges, OUT, label="r")
        T_dense = np.diag(np.array([1.0, 100.0, 90.0]))
        T = SymmetricTensor.from_dense(T_dense, (idx_l, idx_r))

        U, s, _, _ = svd(
            T,
            left_labels=["l"],
            right_labels=["r"],
            new_bond_label="bond",
            max_singular_values=3,
            max_truncation_err=0.05,
            base_charges=np.array([0, 1, 1], dtype=np.int32),
        )
        assert s.shape == (3,), (
            f"n_keep should have expanded from 2 to 3 to honor err — got "
            f"{s.shape[0]} SVs"
        )
        bond_charges = U.indices[U.labels().index("bond")].charges.tolist()
        assert bond_charges == [0, 1, 1], (
            f"bond_charges {bond_charges} should match canonical [0, 1, 1]"
        )
        np.testing.assert_allclose(np.array(s).tolist(), [1.0, 100.0, 90.0], atol=1e-10)

    def test_eager_base_charges_caps_expansion_at_max_singular_values(self):
        """Expansion stops at ``max_singular_values`` even if err still violated.

        When the err budget cannot be met under the base_charges constraint
        within the hard cap, we return what we have at the cap rather than
        expand further. The actual err may exceed the budget — caller should
        check ``s.shape[0]`` against expectations if a strict budget matters.
        """
        from tenax.linalg import svd

        sym = U1Symmetry()
        # Same structure but cap at 2 — cannot expand past 2.
        charges = np.array([0, 1, 1], dtype=np.int32)
        idx_l = TensorIndex.from_charges(sym, charges, IN, label="l")
        idx_r = TensorIndex.from_charges(sym, charges, OUT, label="r")
        T_dense = np.diag(np.array([1.0, 100.0, 90.0]))
        T = SymmetricTensor.from_dense(T_dense, (idx_l, idx_r))

        U, s, _, _ = svd(
            T,
            left_labels=["l"],
            right_labels=["r"],
            new_bond_label="bond",
            max_singular_values=2,
            max_truncation_err=0.05,
            base_charges=np.array([0, 1, 1], dtype=np.int32),
        )
        # Cap is 2, so we return 2 SVs even though err budget can't be met.
        assert s.shape == (2,)
        bond_charges = U.indices[U.labels().index("bond")].charges.tolist()
        # Canonical prefix for n=2 with base=[0,1,1] is [0, 1].
        assert bond_charges == [0, 1]


# ------------------------------------------------------------------ #
# Batched block-sparse decompositions (#569, Milestone A increment 1) #
# ------------------------------------------------------------------ #

import os  # noqa: E402
from contextlib import contextmanager  # noqa: E402

from tenax.core.symmetry import FermionParity, ZnSymmetry  # noqa: E402
from tenax.linalg import qr, svd  # noqa: E402


@contextmanager
def _batch_gate(on: bool):
    """Toggle the TENAX_BATCH_BLOCKSPARSE umbrella gate, restoring prior value."""
    key = "TENAX_BATCH_BLOCKSPARSE"
    prev = os.environ.get(key)
    if on:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _two_leg(sym, charges, seed=0):
    """A 2-leg (l:IN, r:OUT) symmetric tensor.

    With repeated charges of equal multiplicity, several bond-charge sectors
    share the same assembled-matrix shape, so the batched ``vmap`` branch of
    ``_grouped_decomp_by_shape`` is genuinely exercised (groups of size > 1),
    not silently reduced to singletons.
    """
    ch = np.asarray(charges, dtype=np.int32)
    idx_l = TensorIndex.from_charges(sym, ch, IN, label="l")
    idx_r = TensorIndex.from_charges(sym, ch, OUT, label="r")
    return SymmetricTensor.random_normal(
        (idx_l, idx_r), jax.random.PRNGKey(seed), dtype=jnp.float64
    )


def _recon_us_vh(U, s, Vh):
    """Gauge-invariant dense reconstruction U·diag(s)·Vh.

    Compares decompositions without depending on the (sign/basis) gauge of the
    raw U/Vh singular vectors, which batched vs sequential LAPACK may pick
    differently on degenerate subspaces.
    """
    Ud = np.asarray(U.todense())  # (Dl, Dbond)
    Vhd = np.asarray(Vh.todense())  # (Dbond, Dr)
    return (Ud * np.asarray(s)[None, :]) @ Vhd


def _two_leg_prescribed_svals(sym, charges, sval_fn, seed=0):
    """A 2-leg (l:IN, r:OUT) tensor whose per-sector matrices have *prescribed*
    singular values, so we can force (near-)degenerate spectra.

    Motivation: the energy-level off-vs-on divergence in the #569 iPEPS CTM
    benchmark was driven by *near-degenerate singular values* in the CTM
    environment. There, batched-vmap and looped LAPACK pick different (equally
    valid) invariant subspaces, and the iterated CTM map amplifies that benign
    gauge choice into an O(0.1) energy gap — which looked like a batching bug
    but is not. This builder reproduces that regime at the op level, where the
    well-posed invariant (singular values + U·diag(s)·Vh) must still match to
    machine precision regardless of the subspace gauge.
    """
    ch = np.asarray(charges, dtype=np.int32)
    idx_l = TensorIndex.from_charges(sym, ch, IN, label="l")
    idx_r = TensorIndex.from_charges(sym, ch, OUT, label="r")
    template = SymmetricTensor.random_normal(
        (idx_l, idx_r), jax.random.PRNGKey(seed), dtype=jnp.float64
    )
    rng = np.random.RandomState(seed)
    new_blocks = {}
    for key, blk in template.blocks.items():
        n, m = blk.shape
        r = min(n, m)
        s = np.asarray(sval_fn(r, rng), dtype=np.float64)
        # Random orthogonal U (n×n), V (m×m): M = U[:, :r]·diag(s)·V[:r, :] has
        # exactly the singular values s, with a generic (random) left/right gauge.
        U, _ = np.linalg.qr(rng.standard_normal((n, n)))
        V, _ = np.linalg.qr(rng.standard_normal((m, m)))
        M = (U[:, :r] * s[None, :]) @ V[:r, :]
        new_blocks[key] = jnp.asarray(M, dtype=blk.dtype)
    return SymmetricTensor(new_blocks, (idx_l, idx_r))


def _two_leg_prescribed_evals(sym, charges, eval_fn, seed=0):
    """A 2-leg symmetric tensor with *prescribed*, possibly degenerate per-sector
    eigenvalues (Hermitian blocks Q·diag(w)·Qᵀ) — the eigh analogue of
    ``_two_leg_prescribed_svals``."""
    ch = np.asarray(charges, dtype=np.int32)
    idx_l = TensorIndex.from_charges(sym, ch, IN, label="l")
    idx_r = TensorIndex.from_charges(sym, ch, OUT, label="r")
    template = SymmetricTensor.random_normal(
        (idx_l, idx_r), jax.random.PRNGKey(seed), dtype=jnp.float64
    )
    rng = np.random.RandomState(seed)
    new_blocks = {}
    for key, blk in template.blocks.items():
        n, _ = blk.shape
        w = np.asarray(eval_fn(n, rng), dtype=np.float64)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        M = (Q * w[None, :]) @ Q.T  # symmetric, eigenvalues w
        new_blocks[key] = jnp.asarray(M, dtype=blk.dtype)
    return SymmetricTensor(new_blocks, (idx_l, idx_r))


# (name, symmetry, charges) — each yields >=2 sectors sharing a matrix shape.
_SYM_CASES = [
    ("u1", U1Symmetry(), [-1, -1, 0, 0, 1, 1]),
    ("z2", ZnSymmetry(2), [0, 0, 1, 1]),
    ("fp", FermionParity(), [0, 0, 1, 1]),
]


class TestBatchedDecompEquivalence:
    """Gate-on (batched vmap) must equal gate-off (per-sector loop).

    Equivalence is asserted on gauge-invariant quantities — singular/eigen
    values and the dense reconstruction — rather than raw U/Vh, since batched
    LAPACK may pick a different sign/basis gauge on (near-)degenerate subspaces.
    """

    @pytest.mark.parametrize("name,sym,charges", _SYM_CASES)
    def test_svd_full(self, name, sym, charges):
        T = _two_leg(sym, charges, seed=1)
        with _batch_gate(False):
            U0, s0, Vh0, _ = svd(T, ["l"], ["r"])
        with _batch_gate(True):
            U1, s1, Vh1, _ = svd(T, ["l"], ["r"])
        np.testing.assert_allclose(
            np.sort(np.asarray(s0)), np.sort(np.asarray(s1)), rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            _recon_us_vh(U0, s0, Vh0),
            _recon_us_vh(U1, s1, Vh1),
            rtol=1e-10,
            atol=1e-12,
        )
        # Full SVD must reconstruct the input (sanity that recon is meaningful).
        np.testing.assert_allclose(
            _recon_us_vh(U0, s0, Vh0), np.asarray(T.todense()), rtol=1e-8, atol=1e-9
        )

    @pytest.mark.parametrize("name,sym,charges", _SYM_CASES)
    def test_svd_truncated(self, name, sym, charges):
        T = _two_leg(sym, charges, seed=2)
        with _batch_gate(False):
            _, s0, _, _ = svd(T, ["l"], ["r"], max_singular_values=3)
        with _batch_gate(True):
            _, s1, _, _ = svd(T, ["l"], ["r"], max_singular_values=3)
        assert s0.shape == s1.shape
        np.testing.assert_allclose(
            np.sort(np.asarray(s0)), np.sort(np.asarray(s1)), rtol=1e-10, atol=1e-12
        )

    @pytest.mark.parametrize("name,sym,charges", _SYM_CASES)
    def test_qr(self, name, sym, charges):
        T = _two_leg(sym, charges, seed=3)
        with _batch_gate(False):
            Q0, R0 = qr(T, ["l"], ["r"])
        with _batch_gate(True):
            Q1, R1 = qr(T, ["l"], ["r"])
        recon0 = np.asarray(Q0.todense()) @ np.asarray(R0.todense())
        recon1 = np.asarray(Q1.todense()) @ np.asarray(R1.todense())
        np.testing.assert_allclose(recon0, recon1, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            recon0, np.asarray(T.todense()), rtol=1e-8, atol=1e-9
        )

    @pytest.mark.parametrize("name,sym,charges", _SYM_CASES)
    def test_eigh(self, name, sym, charges):
        T = _two_leg(sym, charges, seed=4)
        with _batch_gate(False):
            _, w0 = eigh(T, ["l"], ["r"])
        with _batch_gate(True):
            _, w1 = eigh(T, ["l"], ["r"])
        np.testing.assert_allclose(
            np.sort(np.asarray(w0)), np.sort(np.asarray(w1)), rtol=1e-10, atol=1e-12
        )

    def test_single_sector_sanity(self):
        # One charge -> one sector -> singleton group -> direct call, must
        # still equal the sequential path exactly.
        T = _two_leg(U1Symmetry(), [0, 0, 0], seed=5)
        with _batch_gate(False):
            _, s0, _, _ = svd(T, ["l"], ["r"])
        with _batch_gate(True):
            _, s1, _, _ = svd(T, ["l"], ["r"])
        np.testing.assert_allclose(
            np.sort(np.asarray(s0)), np.sort(np.asarray(s1)), rtol=1e-12, atol=1e-14
        )

    @pytest.mark.parametrize("name,sym,charges", _SYM_CASES)
    def test_svd_near_degenerate_spectrum(self, name, sym, charges):
        """The #569 regime: (near-)degenerate singular values.

        With a degenerate cluster the raw U/Vh subspace is *ambiguous* — batched
        vmap and looped LAPACK may pick different bases — which is exactly what
        the iterated CTM amplified into the benchmark's O(0.1) energy gap. But
        the gauge-invariant content (sorted singular values + U·diag(s)·Vh) must
        still be identical off vs on. This is the well-posed correctness guard
        the downstream energy comparison could never be.
        """

        def svals(r, _rng):
            s = np.linspace(1.0, 0.25, r)
            if r >= 2:
                s[1] = s[0] + 1e-9  # tight near-degenerate pair at the top
            return s

        T = _two_leg_prescribed_svals(sym, charges, svals, seed=21)
        with _batch_gate(False):
            U0, s0, Vh0, _ = svd(T, ["l"], ["r"])
        with _batch_gate(True):
            U1, s1, Vh1, _ = svd(T, ["l"], ["r"])
        np.testing.assert_allclose(
            np.sort(np.asarray(s0)), np.sort(np.asarray(s1)), rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            _recon_us_vh(U0, s0, Vh0),
            _recon_us_vh(U1, s1, Vh1),
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            _recon_us_vh(U0, s0, Vh0), np.asarray(T.todense()), rtol=1e-8, atol=1e-9
        )

    def test_svd_exactly_degenerate_large_block(self):
        """Larger blocks (4×4) with an *exactly* degenerate spectrum, two sectors
        sharing the shape so the vmap group is genuinely size>1. The
        gauge-invariant SVD must match off vs on to machine precision even when
        whole singular subspaces are degenerate."""

        def svals(r, _rng):
            # e.g. [1, 1, 0.5, 0.5] — two exactly-degenerate pairs.
            half = r // 2
            return np.repeat(np.linspace(1.0, 0.5, max(half, 1)), 2)[:r]

        T = _two_leg_prescribed_svals(
            U1Symmetry(), [-1, -1, -1, -1, 1, 1, 1, 1], svals, seed=22
        )
        with _batch_gate(False):
            U0, s0, Vh0, _ = svd(T, ["l"], ["r"])
        with _batch_gate(True):
            U1, s1, Vh1, _ = svd(T, ["l"], ["r"])
        np.testing.assert_allclose(
            np.sort(np.asarray(s0)), np.sort(np.asarray(s1)), rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            _recon_us_vh(U0, s0, Vh0),
            _recon_us_vh(U1, s1, Vh1),
            rtol=1e-10,
            atol=1e-12,
        )

    @pytest.mark.parametrize("name,sym,charges", _SYM_CASES)
    def test_eigh_near_degenerate_spectrum(self, name, sym, charges):
        """eigh analogue: (near-)degenerate eigenvalues. Eigenvectors are
        gauge-ambiguous on the degenerate subspace, but the eigenvalues — the
        invariant — must match off vs on to machine precision."""

        def evals(n, _rng):
            w = np.linspace(2.0, 0.5, n)
            if n >= 2:
                w[1] = w[0] + 1e-9
            return w

        T = _two_leg_prescribed_evals(sym, charges, evals, seed=23)
        with _batch_gate(False):
            _, w0 = eigh(T, ["l"], ["r"])
        with _batch_gate(True):
            _, w1 = eigh(T, ["l"], ["r"])
        np.testing.assert_allclose(
            np.sort(np.asarray(w0)), np.sort(np.asarray(w1)), rtol=1e-10, atol=1e-12
        )


class TestBatchedTracedSvdAD:
    """The AD-critical path: vmap over the ``truncated_svd_ad`` custom_vjp.

    Differentiating a scalar of the SVD outputs forces the block arrays to be
    tracers, routing through ``_truncated_svd_symmetric_traced``. The loss is
    ``sum(s**2)`` — gauge-invariant and smooth, so finite differences are
    reliable even when singular vectors are sign-ambiguous.
    """

    @staticmethod
    def _flat_loss(T):
        leaves, treedef = jax.tree_util.tree_flatten(T)
        x0 = leaves[0]

        def loss_x(x):
            Tx = jax.tree_util.tree_unflatten(treedef, [x])
            # max_singular_values=None -> traced path keeps k_q = available_q,
            # uniform across same-shape sectors -> genuine multi-member
            # (shape, k_q) vmap group.
            _, s, _, _ = svd(Tx, ["l"], ["r"], max_singular_values=None)
            return jnp.sum(s**2)

        return loss_x, x0

    def _check(self, T, do_fd=True):
        loss_x, x0 = self._flat_loss(T)
        with _batch_gate(False):
            g_off = jax.grad(loss_x)(x0)
        with _batch_gate(True):
            g_on = jax.grad(loss_x)(x0)
        # Batched gradient must equal the sequential gradient (vmap(f)==[f]).
        np.testing.assert_allclose(
            np.asarray(g_on), np.asarray(g_off), rtol=1e-8, atol=1e-10
        )
        if do_fd:
            # Central finite difference on every flat-buffer entry (gate on).
            with _batch_gate(True):
                eps = 1e-6
                x0n = np.asarray(x0)
                fd = np.zeros_like(x0n)
                for i in range(x0n.shape[0]):
                    xp = x0n.copy()
                    xp[i] += eps
                    xm = x0n.copy()
                    xm[i] -= eps
                    fd[i] = (
                        float(loss_x(jnp.asarray(xp))) - float(loss_x(jnp.asarray(xm)))
                    ) / (2 * eps)
            np.testing.assert_allclose(np.asarray(g_on), fd, rtol=1e-4, atol=1e-6)

    def test_grad_parity_u1(self):
        self._check(_two_leg(U1Symmetry(), [-1, -1, 0, 0, 1, 1], seed=11))

    def test_grad_parity_fermion(self):
        self._check(_two_leg(FermionParity(), [0, 0, 1, 1], seed=12))

    def test_grad_parity_rank_deficient(self):
        # Rank-1 sector blocks (one zero singular value per sector) exercise
        # _zero_subrank_singular_values + the rank-aware F-mask under vmap.
        # FD is unreliable at the rank boundary, so assert only batched==seq.
        sym = U1Symmetry()
        ch = np.array([0, 0, 1, 1], dtype=np.int32)
        idx_l = TensorIndex.from_charges(sym, ch, IN, label="l")
        idx_r = TensorIndex.from_charges(sym, ch, OUT, label="r")
        key = jax.random.PRNGKey(13)
        blocks = {}
        for i, q in enumerate((0, 1)):
            u = jax.random.normal(jax.random.fold_in(key, 2 * i), (2,))
            v = jax.random.normal(jax.random.fold_in(key, 2 * i + 1), (2,))
            blocks[(q, q)] = jnp.outer(u, v)  # rank-1 (2,2) block
        T = SymmetricTensor(blocks, (idx_l, idx_r))
        self._check(T, do_fd=False)

    def test_grad_parity_degenerate(self):
        # Scaled-identity sector blocks -> fully degenerate singular values,
        # exercising the Lorentzian regularization under vmap.
        sym = U1Symmetry()
        ch = np.array([0, 0, 1, 1], dtype=np.int32)
        idx_l = TensorIndex.from_charges(sym, ch, IN, label="l")
        idx_r = TensorIndex.from_charges(sym, ch, OUT, label="r")
        blocks = {
            (0, 0): 1.5 * jnp.eye(2),
            (1, 1): 2.5 * jnp.eye(2),
        }
        T = SymmetricTensor(blocks, (idx_l, idx_r))
        self._check(T, do_fd=False)


# ------------------------------------------------------------------ #
# Ill-conditioned SVD must not NaN (jaxlib>=0.10.2 GPU gesvdj guard)  #
# ------------------------------------------------------------------ #


def _ill_conditioned_dense(n: int = 128):
    """A finite, ill-conditioned (cond~1e16) symmetric DenseTensor.

    Singular values decay smoothly 1e0 -> 1e-16. This is the regime of HOTRG /
    CTM environment tensors that trips the cuSOLVER gesvdj (Jacobi) SVD.
    """
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = (Q * np.logspace(0, -16, n)) @ Q.T
    sym = U1Symmetry()
    ch = np.zeros(n, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, ch, IN, label="l"),
        TensorIndex.from_charges(sym, ch, OUT, label="r"),
    )
    return DenseTensor(jnp.asarray(A), idx), A


class TestSvdIllConditioned:
    """Regression guard for the jaxlib>=0.10.2 GPU cuSOLVER ``gesvdj`` (Jacobi)
    non-convergence NaN: on CUDA the default SVD returns all-NaN U/s/Vh for
    ill-conditioned f64 matrices. On CPU (LAPACK gesdd) this is a robustness
    baseline; on GPU it directly catches the NaN-filled decomposition.
    """

    def test_svd_no_nan_on_ill_conditioned(self):
        from tenax.linalg import svd

        T, A = _ill_conditioned_dense()
        U, s, Vh, _ = svd(T, left_labels=["l"], right_labels=["r"])
        assert not np.isnan(np.asarray(U.todense())).any(), "U has NaN"
        assert not np.isnan(np.asarray(Vh.todense())).any(), "Vh has NaN"
        assert not np.isnan(np.asarray(s)).any(), "s has NaN"

    def test_svd_singular_values_match_numpy(self):
        from tenax.linalg import svd

        T, A = _ill_conditioned_dense()
        _, s, _, _ = svd(T, left_labels=["l"], right_labels=["r"])
        s_ref = np.linalg.svd(A, compute_uv=False)
        k = 16  # top of the spectrum is well-determined
        got = np.sort(np.asarray(s))[::-1][:k]
        np.testing.assert_allclose(got, s_ref[:k], rtol=1e-6)


# ------------------------------------------------------------------ #
# Canonical bond labels out of svd (#733 / #734)                       #
# ------------------------------------------------------------------ #

_BOND_LABEL_CASES = [
    ("U1", U1Symmetry(), [0, 1]),
    ("Z2", ZnSymmetry(2), [0, 1]),
    ("Z3", ZnSymmetry(3), [0, 1, 2]),
    ("FermionParity", FermionParity(), [0, 1]),
]


@pytest.mark.parametrize(
    "name,sym,sectors", _BOND_LABEL_CASES, ids=[c[0] for c in _BOND_LABEL_CASES]
)
def test_svd_bond_labels_are_canonical_for_either_left_leg_flow(name, sym, sectors):
    """``svd`` must label the new bond so its keys name sectors the bond has.

    Mirroring the flows *dualises* the bond charge, and for U(1) that is real
    physics rather than a relabelling: the partner of ``1`` is ``-1``, so the
    OUT orientation legitimately gets bond sectors ``[-1, 0]``.  What must not
    happen is the library writing a **non-canonical representative** of that
    partner.  Before #734 a ``Z2`` left leg flowing OUT produced bond *keys*
    ``(1, -1)`` while the bond *index* carried sectors ``[0, 1]`` -- the key
    named a charge the leg did not have, because
    ``_group_blocks_by_bond_charge`` fused a single flow-weighted charge and
    ``fuse_many`` of one array skips the ``% n``.  Such a tensor still passes
    ``_validate`` (``fuse`` reduces mod ``n``), and then silently fails to pair
    with canonically-built tensors during contraction (#733).
    """

    def leg(flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.array(sectors, dtype=np.int32),
            multiplicities=np.array([2] * len(sectors), dtype=np.int32),
            flow=flow,
            label=lbl,
        )

    def canon(charges):
        return sorted(
            int(q)
            for q in sym.canonicalize_charges(np.asarray(charges, dtype=np.int32))
        )

    seen = {}
    for left_flow, right_flow in (
        (FlowDirection.IN, FlowDirection.OUT),
        (FlowDirection.OUT, FlowDirection.IN),
    ):
        t = SymmetricTensor.random_normal_np(
            (leg(left_flow, "a"), leg(right_flow, "b")), np.random.RandomState(0)
        )
        U, _, Vh, _ = svd(t, left_labels=["a"], right_labels=["b"])

        # Every block key must name a sector its own leg actually carries.
        # This is what a non-canonical representative breaks.
        for factor, fname in ((U, "U"), (Vh, "Vh")):
            for key in factor._block_keys:
                for idx, q in zip(factor.indices, key):
                    assert idx.has_sector(int(q)), (
                        f"{name} left={left_flow.name}: {fname} block key {key} "
                        f"names charge {int(q)} on leg {idx.label!r}, whose sectors "
                        f"are {list(idx.sectors)}"
                    )

        bond = [i for i in U.indices if i.label != "a"][0]
        bond_sectors = sorted(int(q) for q in bond.sectors)
        assert bond_sectors == canon(bond_sectors), (
            f"{name} left={left_flow.name}: bond labels are not canonical: "
            f"{bond_sectors}"
        )
        seen[left_flow.name] = bond_sectors

    # Flipping the left leg's flow dualises the bond, nothing more: no extra
    # sector appears or disappears, and the result stays canonical.
    assert seen["OUT"] == canon(sym.dual(np.array(seen["IN"], dtype=np.int32))), (
        f"{name}: mirrored bond labels are not the canonical dual: {seen}"
    )


# ------------------------------------------------------------------ #
# #689 — block-sparse decompositions must honour left/right label      #
# order rather than the tensor's native stored axis order.             #
# ------------------------------------------------------------------ #


@pytest.fixture
def u1_sym_tensor_3leg_mult2():
    """3-leg U(1) tensor whose every charge sector has multiplicity 2.

    Multiplicity matters here: with multiplicity 1 on every charge, each sector
    block is 1x1x1 and reshaping it in native versus ``left+right`` axis order
    gives the identical matrix — so a multiplicity-1 fixture cannot detect #689
    at all.  Repeating each charge twice makes the blocks 2x2x2, where the two
    orders genuinely differ.
    """
    sym = U1Symmetry()
    pl = np.array([0, 0, 1, 1], dtype=np.int32)  # phys / left: charges {0,1} x2
    r = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)  # right: {0,1,2} x2
    indices = (
        TensorIndex.from_charges(sym, pl, IN, label="phys"),
        TensorIndex.from_charges(sym, pl, IN, label="left"),
        TensorIndex.from_charges(sym, sym.dual(r), OUT, label="right"),
    )
    return SymmetricTensor.random_normal(indices, jax.random.PRNGKey(7))


def _reference_in_split_order(tensor, left_labels, right_labels):
    """``tensor`` densified and transposed to ``left_labels + right_labels``."""
    stored = list(tensor.labels())
    perm = tuple(stored.index(lbl) for lbl in list(left_labels) + list(right_labels))
    return np.asarray(tensor.todense()).transpose(perm)


def _contract_over_bond(left_factor, right_factor, scale=None):
    """Contract ``(left..., bond)`` against ``(bond, right...)``, optionally
    scaling the bond by ``scale`` (the singular values)."""
    lf = np.asarray(left_factor.todense())
    rf = np.asarray(right_factor.todense())
    if scale is not None:
        lf = lf * np.asarray(scale).reshape((1,) * (lf.ndim - 1) + (-1,))
    return np.tensordot(lf, rf, axes=([lf.ndim - 1], [0]))


class TestBlockSparseDecompLabelOrder:
    """``svd``/``qr``/``eigh`` accept ``left_labels``/``right_labels``, but the
    block-sparse path reshaped each sector block in its *native* stored axis
    order — silently decomposing a permuted matrix (#689).

    Each case requests a split whose ``left + right`` concatenation is NOT the
    identity permutation of the stored axes, on a fixture with multiplicity 2,
    which is exactly when the two orders disagree.  The dense path already
    transposes to ``left_axes + right_axes`` before reshaping, so it defines
    the contract.  Reconstruction is gauge-free, so it holds for any correct
    decomposition regardless of basis choice.
    """

    LEFT = ["left"]
    RIGHT = ["phys", "right"]  # stored order is (phys, left, right)

    def test_qr_reconstructs_under_permuted_label_split(self, u1_sym_tensor_3leg_mult2):
        """Q·R == T when left_labels+right_labels reorders the stored axes."""
        from tenax.linalg import qr

        T = u1_sym_tensor_3leg_mult2
        Q, R = qr(T, left_labels=self.LEFT, right_labels=self.RIGHT)
        np.testing.assert_allclose(
            _contract_over_bond(Q, R),
            _reference_in_split_order(T, self.LEFT, self.RIGHT),
            atol=1e-12,
        )

    def test_svd_reconstructs_under_permuted_label_split(
        self, u1_sym_tensor_3leg_mult2
    ):
        """U·diag(S)·Vh == T when the requested split reorders the stored axes."""
        from tenax.linalg import svd

        T = u1_sym_tensor_3leg_mult2
        U, S, Vh, _ = svd(T, left_labels=self.LEFT, right_labels=self.RIGHT)
        np.testing.assert_allclose(
            _contract_over_bond(U, Vh, scale=S),
            _reference_in_split_order(T, self.LEFT, self.RIGHT),
            atol=1e-12,
        )

    def test_qr_symmetric_matches_dense_under_permuted_split(
        self, u1_sym_tensor_3leg_mult2
    ):
        """Block-sparse QR reconstructs the same tensor as the dense path."""
        from tenax.linalg import qr

        T = u1_sym_tensor_3leg_mult2
        T_dense = DenseTensor(T.todense(), T.indices)
        np.testing.assert_allclose(
            _contract_over_bond(*qr(T, self.LEFT, self.RIGHT)),
            _contract_over_bond(*qr(T_dense, self.LEFT, self.RIGHT)),
            atol=1e-12,
        )

    def test_rsvd_reconstructs_under_permuted_label_split(
        self, u1_sym_tensor_3leg_mult2
    ):
        """Randomized SVD also assembles its blocks in the requested order.

        ``rank`` exceeds the true per-sector rank, so HMT randomized SVD is
        essentially exact here and reconstruction is still the right invariant.
        """
        from tenax.linalg import rsvd

        T = u1_sym_tensor_3leg_mult2
        U, S, Vh = rsvd(
            T, left_labels=self.LEFT, right_labels=self.RIGHT, rank=8, oversampling=6
        )
        np.testing.assert_allclose(
            _contract_over_bond(U, Vh, scale=S),
            _reference_in_split_order(T, self.LEFT, self.RIGHT),
            atol=1e-8,
        )


@pytest.fixture
def u1_hermitian_4leg_mult2():
    """4-leg U(1) tensor, Hermitian under the ``(a,b) | (c,d)`` bipartition.

    Multiplicity 2 on every charge so the sector blocks are 2x2x2x2, where
    native-order and ``left+right``-order flattening genuinely differ (#689).
    """
    sym = U1Symmetry()
    c = np.array([0, 0, 1, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, c, IN, label="a"),
        TensorIndex.from_charges(sym, c, IN, label="b"),
        TensorIndex.from_charges(sym, sym.dual(c), OUT, label="c"),
        TensorIndex.from_charges(sym, sym.dual(c), OUT, label="d"),
    )
    # Project onto the allowed sectors FIRST, then symmetrize.  Legs c/d are
    # the duals of a/b in the same index order, so the charge condition is
    # invariant under (a,b) <-> (c,d) and symmetrizing keeps the result inside
    # the allowed sectors.  Symmetrizing a dense random matrix instead would
    # leave most of its weight outside them.
    T0 = SymmetricTensor.random_normal(indices, jax.random.PRNGKey(11))
    M = np.asarray(T0.todense()).reshape(16, 16)
    M = M + M.T
    return SymmetricTensor.from_dense(jnp.asarray(M.reshape(4, 4, 4, 4)), indices)


def test_eigh_reconstructs_under_permuted_label_split(u1_hermitian_4leg_mult2):
    """``eigh``: V·diag(w)·V^dag == T when the split reorders the stored axes.

    Rows and columns are permuted consistently (``b,a | d,c``) so the matrix
    stays Hermitian; the permutation ``(1,0,3,2)`` is still non-identity, which
    is what #689 turns on.
    """
    T = u1_hermitian_4leg_mult2
    left, right = ["b", "a"], ["d", "c"]
    V, w = eigh(T, left_labels=left, right_labels=right)
    Vd = np.asarray(V.todense()).reshape(16, -1)
    recon = (Vd * np.asarray(w)[None, :]) @ Vd.conj().T
    ref = _reference_in_split_order(T, left, right).reshape(16, 16)
    np.testing.assert_allclose(recon, ref, atol=1e-10)


def test_traced_svd_honors_permuted_label_split(u1_sym_tensor_3leg_mult2):
    """The traced (jit/AD) SVD path assembles blocks in the requested order too.

    ``_truncated_svd_symmetric_traced`` is a separate implementation reached
    when the blocks are tracers, so it needs its own guard against #689.

    The probe is the nuclear norm ``sum(s)``, which depends on *how* the tensor
    is folded into a matrix.  (``sum(s**2)`` — what the other traced-SVD tests
    use — is the Frobenius norm and is invariant under any reshuffling of
    elements, so it cannot detect a mis-ordered block assembly.)

    The reference is computed densely from the correctly folded matrix, NOT
    from the eager block-sparse path: both block-sparse paths shared the same
    defect, so comparing them against each other agrees while both are wrong.
    Per-sector singular values are the singular values of the block-diagonal
    matrix, so the two nuclear norms are equal for a correct implementation.
    """
    T = u1_sym_tensor_3leg_mult2
    left, right = ["left"], ["phys", "right"]
    leaves, treedef = jax.tree_util.tree_flatten(T)
    x0 = leaves[0]

    def loss(x):
        Tx = jax.tree_util.tree_unflatten(treedef, [x])
        _, s, _, _ = svd(Tx, left, right, max_singular_values=None)
        return jnp.sum(s)

    folded = _reference_in_split_order(T, left, right).reshape(4, 24)
    expected = float(np.linalg.svd(folded, compute_uv=False).sum())

    # jit forces the blocks to be tracers -> _truncated_svd_symmetric_traced.
    np.testing.assert_allclose(float(jax.jit(loss)(x0)), expected, rtol=1e-10)
