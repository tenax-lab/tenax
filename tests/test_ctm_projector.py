"""Tests for block-sparse SVD (Fishman) projector in _ctm_projector.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_projector import _compute_projector_tensor
from tenax.algorithms._ctm_tensor import (
    _build_double_layer_tensor,
    _ctm_tensor_sweep_multisite,
    _fuse_pair_by_label,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_peps_symmetric():
    """Random SymmetricTensor iPEPS with trivial U(1) charges."""
    D, d = 2, 2
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    rng = np.random.RandomState(99)
    data = jnp.array(rng.standard_normal((D, D, D, D, d)))
    return SymmetricTensor.from_dense(data, indices)


@pytest.fixture
def fpeps_tensor():
    """Random SymmetricTensor iPEPS with FermionParity (nontrivial charges)."""
    key = jax.random.PRNGKey(7)
    sym = FermionParity()
    virt_charges = np.array([0, 1], dtype=np.int32)
    phys_charges = np.array([0, 1], dtype=np.int32)
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
    return SymmetricTensor.random_normal(indices, key)


# ------------------------------------------------------------------ #
# Helpers                                                               #
# ------------------------------------------------------------------ #


def _build_grown_corners(A, chi):
    """Build grown corners from a symmetric iPEPS tensor (left move)."""
    from tenax.contraction.contractor import contract

    env = initialize_ctm_tensor_env(A, chi)

    C1_r = env.C1.relabel("c1_r", "t1_l")
    C1g = contract(C1_r, env.T1)
    C1g = _fuse_pair_by_label(C1g, "c1_d", "u2", "fused", FlowDirection.IN)

    C4_u = env.C4.relabel("c4_u", "t3_r")
    C4g = contract(C4_u, env.T3)
    C4g = _fuse_pair_by_label(C4g, "c4_r", "d2", "fused", FlowDirection.IN)

    return C1g, C4g


def _build_grown_corners_converged(A, chi, n_sweeps=2):
    """Like :func:`_build_grown_corners` but with ``n_sweeps`` CTM sweeps first.

    PR #422 (c42c60d) + PR #424 (d752077) intentionally made CTM init rank-1
    (variPEPS-style ``chi_init=1`` + diagonal δ_{ket=bra} edge tensors) to fix
    random-init convergence. The grown corners ``C·T`` built directly from the
    cold init are therefore rank-1, so the Fishman SVD projector formula
    ``P = C·V·S^{-1/2}`` (with the safe ``S^{-1/2}`` mask) correctly produces
    rank-1 outputs and ``P_L @ P_R = diag(1,0,…,0)`` rather than identity.

    Running a couple of CTM sweeps through the multisite path lifts the env
    off the rank-1 cold init into the full-rank geometry that real CTM
    convergence (``python_loop_ctm_converge``, which uses the multisite
    sweep) sees, which is what biorthogonality tests need.  Note: the
    single-site ``_ctm_tensor_sweep`` is a fixed point on rank-1 input, so
    we use ``_ctm_tensor_sweep_multisite`` here even for a single coord —
    this matches the production CTM driver.
    """
    from tenax.contraction.contractor import contract

    envs = {(0, 0): initialize_ctm_tensor_env(A, chi)}
    double_layers = {(0, 0): _build_double_layer_tensor(A)}
    for _ in range(n_sweeps):
        envs, _ = _ctm_tensor_sweep_multisite(
            envs, double_layers, SINGLE_SITE_NEIGHBORS, chi, True, "svd"
        )
    env = envs[(0, 0)]

    C1_r = env.C1.relabel("c1_r", "t1_l")
    C1g = contract(C1_r, env.T1)
    C1g = _fuse_pair_by_label(C1g, "c1_d", "u2", "fused", FlowDirection.IN)

    C4_u = env.C4.relabel("c4_u", "t3_r")
    C4g = contract(C4_u, env.T3)
    C4g = _fuse_pair_by_label(C4g, "c4_r", "d2", "fused", FlowDirection.IN)

    return C1g, C4g


# ------------------------------------------------------------------ #
# Tests                                                                 #
# ------------------------------------------------------------------ #


class TestSVDProjectorSymmetric:
    """Tests for block-sparse Fishman SVD projector."""

    def test_svd_projector_returns_symmetric_tensors(self, small_peps_symmetric):
        """SVD projector should return SymmetricTensor pair."""
        A = small_peps_symmetric
        chi = 4
        C1g, C4g = _build_grown_corners(A, chi)

        P1, P2, _ = _compute_projector_tensor(C1g, C4g, chi, projector_method="svd")

        assert isinstance(P1, SymmetricTensor)
        assert isinstance(P2, SymmetricTensor)
        assert P1.labels() == ("fused", "chi_new")
        assert P2.labels() == ("fused", "chi_new")

    def test_svd_projector_biorthogonality(self, small_peps_symmetric):
        """P1^H @ P2 should be close to identity."""
        A = small_peps_symmetric
        chi = 4
        # Lift the rank-1 cold init (PR #422/#424) into a full-rank env before
        # testing biorthogonality — the projector formula correctly produces
        # rank-1 outputs on rank-1 inputs (mathematically), so the assertion
        # needs full-rank input geometry.
        C1g, C4g = _build_grown_corners_converged(A, chi)

        P1, P2, _ = _compute_projector_tensor(C1g, C4g, chi, projector_method="svd")

        P1d = P1.todense()
        P2d = P2.todense()
        gram = P1d.conj().T @ P2d
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-10)

    def test_svd_projector_matches_dense_subspace(self, small_peps_symmetric):
        """Block-sparse SVD projector truncated env matches dense SVD projector."""
        A = small_peps_symmetric
        chi = 4
        C1g, C4g = _build_grown_corners(A, chi)

        # Block-sparse path
        P1_sym, P2_sym, _ = _compute_projector_tensor(
            C1g, C4g, chi, projector_method="svd"
        )

        # Dense reference: convert to DenseTensor to force dense path
        fused_pos = C1g.labels().index("fused")
        fused_idx = C1g.indices[fused_pos]
        col1_idx = C1g.indices[1 - fused_pos]
        col2_idx = C4g.indices[1 - fused_pos]

        C1g_dense = DenseTensor(C1g.todense(), (fused_idx, col1_idx))
        C4g_dense = DenseTensor(C4g.todense(), (fused_idx, col2_idx))
        P1_ref, P2_ref, _ = _compute_projector_tensor(
            C1g_dense, C4g_dense, chi, projector_method="svd"
        )

        # The truncated environment P1^H @ C1g should be similar (up to gauge)
        # Compare truncated cross-product: P1^H @ C4g and P2^H @ C1g
        C4gd = C4g.todense()

        trunc_sym = P1_sym.todense().conj().T @ C4gd
        trunc_ref = P1_ref.todense().conj().T @ C4gd

        # The truncated matrices should span the same row space.
        # Compare via SVD singular values of the truncated matrices.
        sv_sym = np.sort(np.linalg.svd(np.array(trunc_sym), compute_uv=False))[::-1]
        sv_ref = np.sort(np.linalg.svd(np.array(trunc_ref), compute_uv=False))[::-1]
        np.testing.assert_allclose(sv_sym, sv_ref, atol=1e-10)

    def test_svd_projector_fpeps_returns_symmetric(self, fpeps_tensor):
        """SVD projector works with FermionParity (nontrivial charges)."""
        A = fpeps_tensor
        chi = 4
        # Lift the rank-1 cold init (PR #422/#424) into a full-rank env before
        # testing biorthogonality — the projector formula correctly produces
        # rank-1 outputs on rank-1 inputs (mathematically), so the assertion
        # needs full-rank input geometry.
        C1g, C4g = _build_grown_corners_converged(A, chi)

        P1, P2, _ = _compute_projector_tensor(C1g, C4g, chi, projector_method="svd")

        assert isinstance(P1, SymmetricTensor)
        assert isinstance(P2, SymmetricTensor)

        # Biorthogonality
        P1d = P1.todense()
        P2d = P2.todense()
        gram = P1d.conj().T @ P2d
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-10)

    def test_svd_projector_fpeps_matches_dense(self, fpeps_tensor):
        """Block-sparse SVD projector singular values match dense for FermionParity."""
        A = fpeps_tensor
        chi = 4
        C1g, C4g = _build_grown_corners(A, chi)

        # Block-sparse
        P1_sym, P2_sym, _ = _compute_projector_tensor(
            C1g, C4g, chi, projector_method="svd"
        )

        # Dense reference
        fused_pos = C1g.labels().index("fused")
        fused_idx = C1g.indices[fused_pos]
        col1_idx = C1g.indices[1 - fused_pos]
        col2_idx = C4g.indices[1 - fused_pos]
        C1g_dense = DenseTensor(C1g.todense(), (fused_idx, col1_idx))
        C4g_dense = DenseTensor(C4g.todense(), (fused_idx, col2_idx))
        P1_ref, P2_ref, _ = _compute_projector_tensor(
            C1g_dense, C4g_dense, chi, projector_method="svd"
        )

        # Compare truncated cross-product singular values
        C4gd = C4g.todense()
        trunc_sym = P1_sym.todense().conj().T @ C4gd
        trunc_ref = P1_ref.todense().conj().T @ C4gd
        sv_sym = np.sort(np.linalg.svd(np.array(trunc_sym), compute_uv=False))[::-1]
        sv_ref = np.sort(np.linalg.svd(np.array(trunc_ref), compute_uv=False))[::-1]
        np.testing.assert_allclose(sv_sym, sv_ref, atol=1e-10)

    def test_svd_no_todense_on_symmetric_path(self, small_peps_symmetric):
        """Block-sparse SVD path must not call todense()."""
        from unittest.mock import patch

        A = small_peps_symmetric
        chi = 4
        C1g, C4g = _build_grown_corners(A, chi)

        todense_calls = []
        orig_todense = SymmetricTensor.todense

        def tracking_todense(self):
            todense_calls.append(True)
            return orig_todense(self)

        with patch.object(SymmetricTensor, "todense", tracking_todense):
            P1, P2, _ = _compute_projector_tensor(C1g, C4g, chi, projector_method="svd")

        assert isinstance(P1, SymmetricTensor)
        assert len(todense_calls) == 0, (
            "todense() was called on the block-sparse SVD path"
        )

    def test_svd_projector_ad_dense_fallback_gradient(self, small_peps_symmetric):
        """AD-traced path falls back to dense SVD and produces finite gradients.

        During AD (backward pass), SymmetricTensor._data is a JAX tracer,
        so the block-sparse _svd_projector_symmetric is skipped and the
        dense Fishman SVD fallback is used instead.  This test verifies
        that the dense fallback produces finite gradients that agree with
        finite differences.

        This documents the Task 2.2 design decision: the AD-traced path
        intentionally uses the dense fallback rather than a block-sparse
        AD-traced SVD.
        """
        A = small_peps_symmetric
        chi = 4
        C1g, C4g = _build_grown_corners(A, chi)

        # To test the AD path, we define a scalar loss that goes through
        # _compute_projector_tensor with SVD.  We differentiate w.r.t.
        # the raw data array (C1g._data) to exercise the tracer path.
        fused_pos = C1g.labels().index("fused")
        fused_idx = C1g.indices[fused_pos]
        col1_idx = C1g.indices[1 - fused_pos]
        col2_idx = C4g.indices[1 - fused_pos]

        # Use DenseTensor wrapper so we can differentiate w.r.t. plain arrays
        # (SymmetricTensor under AD goes through todense -> tracer check).
        c1_data = C1g.todense()
        c4_data = C4g.todense()

        def loss_fn(c1_arr):
            C1g_dt = DenseTensor(c1_arr, (fused_idx, col1_idx))
            C4g_dt = DenseTensor(c4_data, (fused_idx, col2_idx))
            P1, P2, _ = _compute_projector_tensor(
                C1g_dt, C4g_dt, chi, projector_method="svd"
            )
            # Scalar loss: sum of squared elements of P1
            return jnp.sum(jnp.abs(P1.todense()) ** 2)

        grad = jax.grad(loss_fn)(c1_data)
        assert jnp.all(jnp.isfinite(grad)), "Gradient through SVD projector has NaN/Inf"

        # Finite-difference check
        eps = 1e-5
        c1_np = np.array(c1_data)
        f0 = float(loss_fn(c1_data))
        # Check a few random elements for FD agreement
        rng = np.random.RandomState(42)
        indices_to_check = [
            tuple(rng.randint(0, s) for s in c1_np.shape) for _ in range(5)
        ]
        for idx in indices_to_check:
            c1_plus = c1_np.copy()
            c1_plus[idx] += eps
            f_plus = float(loss_fn(jnp.array(c1_plus)))
            fd_val = (f_plus - f0) / eps
            ad_val = float(grad[idx])
            # Loose tolerance: projector gradients can be noisy
            if abs(fd_val) > 1e-8:
                np.testing.assert_allclose(
                    ad_val,
                    fd_val,
                    rtol=0.1,
                    atol=1e-6,
                    err_msg=f"AD vs FD mismatch at {idx}",
                )
