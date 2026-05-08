"""Tests for standard CTM with Tensor protocol (polymorphic dense/symmetric)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_projector import _compute_projector_tensor
from tenax.algorithms._ctm_tensor import (
    CHECKERBOARD_NEIGHBORS,
    CTMTensorEnv,
    _build_double_layer_open_tensor,
    _build_double_layer_tensor,
    _ctm_tensor_sweep,
    _ctm_tensor_sweep_multisite,
    _fuse_pair_by_label,
    compute_energy_ctm_tensor,
    compute_energy_ctm_tensor_2site,
    ctm_tensor,
    ctm_tensor_2site,
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ipeps_ctm import (
    _build_double_layer,
    _ctm_2site_sweep,
    _ctm_sweep,
    _initialize_ctm_env,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_peps_dense():
    """Random DenseTensor iPEPS site tensor, D=2, d=2."""
    D, d = 2, 2
    rng = np.random.RandomState(42)
    data = jnp.array(rng.standard_normal((D, D, D, D, d)))
    data = data / (jnp.linalg.norm(data) + 1e-10)
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
    return DenseTensor(data, indices)


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
    """Random SymmetricTensor iPEPS with FermionParity."""
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


@pytest.fixture
def heisenberg_gate():
    """Heisenberg 2-site Hamiltonian gate as dense array."""
    d = 2
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(d, d, d, d)


# ------------------------------------------------------------------ #
# Double-layer tests                                                   #
# ------------------------------------------------------------------ #


class TestDoubleLayer:
    def test_double_layer_tensor_matches_dense(self, small_peps_dense):
        """DenseTensor double-layer matches raw einsum."""
        A = small_peps_dense
        a_tensor = _build_double_layer_tensor(A)
        a_dense = a_tensor.todense()

        # Reference: raw dense path
        A_raw = A.todense()
        a_ref = _build_double_layer(A_raw)
        D = A_raw.shape[0]
        a_ref = a_ref.reshape(D**2, D**2, D**2, D**2)

        np.testing.assert_allclose(a_dense, a_ref, atol=1e-12)

    def test_double_layer_tensor_labels(self, small_peps_dense):
        a = _build_double_layer_tensor(small_peps_dense)
        assert a.labels() == ("u2", "d2", "l2", "r2")
        assert a.ndim == 4

    def test_double_layer_tensor_symmetric(self, small_peps_symmetric):
        """SymmetricTensor double-layer todense matches DenseTensor result."""
        A_sym = small_peps_symmetric
        a_sym = _build_double_layer_tensor(A_sym)

        A_dense = DenseTensor(A_sym.todense(), A_sym.indices)
        a_dense = _build_double_layer_tensor(A_dense)

        np.testing.assert_allclose(a_sym.todense(), a_dense.todense(), atol=1e-12)

    def test_double_layer_open_tensor(self, small_peps_dense):
        """Open double-layer has correct shape and labels."""
        a_open = _build_double_layer_open_tensor(small_peps_dense)
        assert a_open.ndim == 6
        labels = a_open.labels()
        assert "phys" in labels
        assert "phys_bra" in labels
        dense = a_open.todense()
        # Check: tracing phys,phys_bra gives double-layer
        a_closed = jnp.einsum("...ss->...", dense)
        a_ref = _build_double_layer_tensor(small_peps_dense).todense()
        np.testing.assert_allclose(a_closed, a_ref, atol=1e-12)


# ------------------------------------------------------------------ #
# Initialization tests                                                 #
# ------------------------------------------------------------------ #


class TestInitialization:
    def test_init_dense_shapes(self, small_peps_dense):
        chi = 4
        env = initialize_ctm_tensor_env(small_peps_dense, chi)
        assert isinstance(env, CTMTensorEnv)
        for C in [env.C1, env.C2, env.C3, env.C4]:
            assert C.todense().shape == (chi, chi)
        D2 = small_peps_dense.indices[0].dim ** 2
        for T in [env.T1, env.T2, env.T3, env.T4]:
            assert T.todense().shape == (chi, D2, chi)

    def test_init_dense_labels(self, small_peps_dense):
        env = initialize_ctm_tensor_env(small_peps_dense, chi=4)
        assert env.C1.labels() == ("c1_d", "c1_r")
        assert env.T1.labels() == ("t1_l", "u2", "t1_r")
        assert env.T4.labels() == ("t4_d", "l2", "t4_u")

    def test_init_dense_finite(self, small_peps_dense):
        env = initialize_ctm_tensor_env(small_peps_dense, chi=4)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_init_symmetric(self, small_peps_symmetric):
        chi = 4
        env = initialize_ctm_tensor_env(small_peps_symmetric, chi)
        for C in [env.C1, env.C2, env.C3, env.C4]:
            assert isinstance(C, SymmetricTensor)
            assert C.todense().shape == (chi, chi)
        for T in [env.T1, env.T2, env.T3, env.T4]:
            assert isinstance(T, SymmetricTensor)

    def test_init_fpeps(self, fpeps_tensor):
        """FermionParity iPEPS produces valid symmetric env."""
        chi = 4
        env = initialize_ctm_tensor_env(fpeps_tensor, chi)
        for field in env:
            assert isinstance(field, SymmetricTensor)
            assert jnp.all(jnp.isfinite(field.todense()))


# ------------------------------------------------------------------ #
# Sweep tests                                                          #
# ------------------------------------------------------------------ #


class TestSweep:
    def test_one_sweep_dense_finite(self, small_peps_dense):
        """One CTM sweep produces finite tensors."""
        chi = 4
        a = _build_double_layer_tensor(small_peps_dense)
        env = initialize_ctm_tensor_env(small_peps_dense, chi)
        env = _ctm_tensor_sweep(env, a, chi, renormalize=True)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense())), (
                f"Non-finite values in {field.labels()}"
            )

    def test_one_sweep_symmetric_finite(self, small_peps_symmetric):
        chi = 4
        a = _build_double_layer_tensor(small_peps_symmetric)
        env = initialize_ctm_tensor_env(small_peps_symmetric, chi)
        env = _ctm_tensor_sweep(env, a, chi, renormalize=True)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_one_sweep_fpeps_finite(self, fpeps_tensor):
        chi = 4
        a = _build_double_layer_tensor(fpeps_tensor)
        env = initialize_ctm_tensor_env(fpeps_tensor, chi)
        env = _ctm_tensor_sweep(env, a, chi, renormalize=True)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))


# ------------------------------------------------------------------ #
# Convergence + energy tests                                           #
# ------------------------------------------------------------------ #


class TestConvergence:
    def test_ctm_tensor_dense_converges(self, small_peps_dense):
        """DenseTensor CTM converges (finite env after max_iter)."""
        env = ctm_tensor(small_peps_dense, chi=4, max_iter=20, conv_tol=1e-6)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    @pytest.mark.skip(
        reason="Tensor CTM path diverged from legacy _ctm_sweep in PR #291 "
        "(differentiable Fishman SVD projector + per-absorption normalization). "
        "The legacy path is retained only for 2-site SU and is not expected "
        "to match the tensor path sweep-for-sweep."
    )
    def test_ctm_tensor_dense_matches_ctm(self, small_peps_dense):
        """Obsolete: legacy CTM sweep no longer matches tensor path."""

    def test_ctm_tensor_symmetric_converges(self, small_peps_symmetric):
        """SymmetricTensor CTM converges (trivial charges)."""
        env = ctm_tensor(small_peps_symmetric, chi=4, max_iter=20, conv_tol=1e-6)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_energy_symmetric_matches_dense(
        self, small_peps_symmetric, heisenberg_gate
    ):
        """SymmetricTensor CTM energy matches DenseTensor CTM energy."""
        chi = 6
        A_dense = DenseTensor(
            small_peps_symmetric.todense(), small_peps_symmetric.indices
        )

        env_dense = ctm_tensor(A_dense, chi=chi, max_iter=40, conv_tol=1e-8)
        E_dense = float(
            compute_energy_ctm_tensor(A_dense, env_dense, heisenberg_gate, d=2)
        )

        env_sym = ctm_tensor(small_peps_symmetric, chi=chi, max_iter=40, conv_tol=1e-8)
        E_sym = float(
            compute_energy_ctm_tensor(
                small_peps_symmetric, env_sym, heisenberg_gate, d=2
            )
        )

        np.testing.assert_allclose(E_sym, E_dense, atol=1e-4)


# ------------------------------------------------------------------ #
# Fermionic integration test                                           #
# ------------------------------------------------------------------ #


class TestFermionicIntegration:
    def test_fermionic_ctm_no_densify(self, fpeps_tensor):
        """Fermionic CTM works with SymmetricTensor (no todense)."""
        env = ctm_tensor(fpeps_tensor, chi=4, max_iter=15, conv_tol=1e-5)
        assert isinstance(env, CTMTensorEnv)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_fpeps_uses_ctm_tensor(self):
        """fpeps() with SymmetricTensor now returns CTMTensorEnv."""
        from tenax.algorithms.fermionic_ipeps import (
            FPEPSConfig,
            fpeps,
            spinless_fermion_gate,
        )

        config = FPEPSConfig(
            D=2,
            t=1.0,
            V=0.0,
            dt=0.05,
            num_imaginary_steps=5,
            ctm_chi=4,
            ctm_max_iter=10,
            ctm_conv_tol=1e-4,
        )
        H = spinless_fermion_gate(config)
        energy, A_opt, env = fpeps(H, config)
        assert isinstance(env, CTMTensorEnv)
        assert np.isfinite(energy)


# ------------------------------------------------------------------ #
# 2-site CTM tests                                                     #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_peps_pair_dense():
    """Two random DenseTensor iPEPS site tensors for 2-site unit cell."""
    D, d = 2, 2
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)

    def _make(seed):
        rng = np.random.RandomState(seed)
        data = jnp.array(rng.standard_normal((D, D, D, D, d)))
        data = data / (jnp.linalg.norm(data) + 1e-10)
        indices = (
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex.from_charges(
                sym, phys_charges.copy(), FlowDirection.IN, label="phys"
            ),
        )
        return DenseTensor(data, indices)

    return _make(42), _make(123)


@pytest.fixture
def small_peps_pair_symmetric():
    """Two random SymmetricTensor iPEPS site tensors (trivial U(1) charges)."""
    D, d = 2, 2
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)

    def _make(seed):
        rng = np.random.RandomState(seed)
        data = jnp.array(rng.standard_normal((D, D, D, D, d)))
        indices = (
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex.from_charges(
                sym, phys_charges.copy(), FlowDirection.IN, label="phys"
            ),
        )
        return SymmetricTensor.from_dense(data, indices)

    return _make(99), _make(200)


class TestTwoSiteCTM:
    def test_2site_dense_converges(self, small_peps_pair_dense):
        """2-site DenseTensor CTM converges (finite env after max_iter)."""
        A, B = small_peps_pair_dense
        env_A, env_B = ctm_tensor_2site(A, B, chi=4, max_iter=20, conv_tol=1e-6)
        for field in env_A:
            assert jnp.all(jnp.isfinite(field.todense()))
        for field in env_B:
            assert jnp.all(jnp.isfinite(field.todense()))

    @pytest.mark.skip(
        reason="Tensor CTM path diverged from legacy _ctm_2site_sweep in "
        "PR #291 (differentiable Fishman SVD projector + per-absorption "
        "normalization). Each path is exercised in its own tests; cross-path "
        "equivalence after a single sweep is no longer a design goal."
    )
    def test_2site_dense_matches_legacy(self, small_peps_pair_dense):
        """Obsolete: legacy 2-site CTM sweep no longer matches tensor path."""

    def test_2site_symmetric_converges(self, small_peps_pair_symmetric):
        """2-site SymmetricTensor CTM converges.

        Pinned to ``recipe="1x1"``: the 2x2 plaquette projector path does
        not yet support the mixed Dense/Symmetric contraction pattern that
        appears when ``_apply_projector_with_reembed`` re-embeds the
        Fishman SVD output (the projector is dense; the env is symmetric).
        Wiring SymmetricTensor through the 2x2 path is a follow-up task;
        until then the 1x1 recipe remains the symmetric-friendly default
        for ``ctm_tensor_2site``.
        """
        A, B = small_peps_pair_symmetric
        env_A, env_B = ctm_tensor_2site(
            A, B, chi=4, max_iter=20, conv_tol=1e-6, recipe="1x1"
        )
        for field in env_A:
            assert jnp.all(jnp.isfinite(field.todense()))
        for field in env_B:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_2site_symmetric_energy_matches_dense(
        self, small_peps_pair_symmetric, heisenberg_gate
    ):
        """2-site SymmetricTensor energy matches DenseTensor result.

        Pinned to ``recipe="1x1"`` for the same reason as
        ``test_2site_symmetric_converges`` — the 2x2 plaquette path does
        not yet support SymmetricTensor inputs.
        """
        A_sym, B_sym = small_peps_pair_symmetric
        chi = 6

        A_dense = DenseTensor(A_sym.todense(), A_sym.indices)
        B_dense = DenseTensor(B_sym.todense(), B_sym.indices)

        env_Ad, env_Bd = ctm_tensor_2site(
            A_dense, B_dense, chi=chi, max_iter=40, conv_tol=1e-8, recipe="1x1"
        )
        E_dense = float(
            compute_energy_ctm_tensor_2site(
                A_dense, B_dense, env_Ad, env_Bd, heisenberg_gate, d=2
            )
        )

        env_As, env_Bs = ctm_tensor_2site(
            A_sym, B_sym, chi=chi, max_iter=40, conv_tol=1e-8, recipe="1x1"
        )
        E_sym = float(
            compute_energy_ctm_tensor_2site(
                A_sym, B_sym, env_As, env_Bs, heisenberg_gate, d=2
            )
        )

        np.testing.assert_allclose(E_sym, E_dense, atol=1e-4)


# ------------------------------------------------------------------ #
# Projector tests                                                      #
# ------------------------------------------------------------------ #


class TestQRProjectorMethod:
    def test_ctm_tensor_qr_converges(self, small_peps_dense):
        """CTM with projector_method='qr' converges to finite env."""
        env = ctm_tensor(
            small_peps_dense,
            chi=4,
            max_iter=20,
            conv_tol=1e-6,
            projector_method="qr",
            qr_warmup_steps=3,
        )
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_ctm_tensor_qr_energy_finite(self, small_peps_dense, heisenberg_gate):
        """QR projector method produces a finite energy."""
        env_qr = ctm_tensor(
            small_peps_dense,
            chi=6,
            max_iter=40,
            conv_tol=1e-8,
            projector_method="qr",
            qr_warmup_steps=3,
        )
        E_qr = float(
            compute_energy_ctm_tensor(small_peps_dense, env_qr, heisenberg_gate, d=2)
        )
        assert np.isfinite(E_qr)

    def test_invalid_projector_method_raises(self, small_peps_dense):
        """Unknown projector_method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown projector_method"):
            ctm_tensor(
                small_peps_dense,
                chi=4,
                max_iter=1,
                projector_method="invalid",
            )


class TestProjectorSymmetric:
    @staticmethod
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

    def test_projector_symmetric_matches_dense(self, small_peps_symmetric):
        """Block-sparse eigh projector matches dense projector output."""
        A = small_peps_symmetric
        chi = 4

        _build_double_layer_tensor(A)  # ensure it doesn't error
        C1g, C4g = self._build_grown_corners(A, chi)

        # Compute projector via symmetric path
        P_sym, _ = _compute_projector_tensor(C1g, C4g, chi, projector_method="eigh")

        # Dense reference
        C1g_d = C1g.todense()
        C4g_d = C4g.todense()
        rho = C1g_d @ C1g_d.conj().T + C4g_d @ C4g_d.conj().T
        rho = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = jnp.linalg.eigh(rho)
        k = min(chi, len(eigvals))
        P_ref = eigvecs[:, -k:][:, ::-1]

        # The projector subspaces should match (up to sign/phase per column)
        P_sym_d = P_sym.todense()
        # Compare via P P^T (projection matrix)
        proj_sym = P_sym_d @ P_sym_d.T
        proj_ref = P_ref @ P_ref.T
        np.testing.assert_allclose(proj_sym, proj_ref, atol=1e-10)

    def test_qr_projector_symmetric_matches_eigh(self, small_peps_symmetric):
        """Block-sparse QR projector subspace matches eigh projector subspace."""
        A = small_peps_symmetric
        chi = 4

        C1g, C4g = self._build_grown_corners(A, chi)

        P_eigh, _ = _compute_projector_tensor(C1g, C4g, chi, projector_method="eigh")
        P_qr, _ = _compute_projector_tensor(C1g, C4g, chi, projector_method="qr")

        # Both should be SymmetricTensor
        assert isinstance(P_eigh, SymmetricTensor)
        assert isinstance(P_qr, SymmetricTensor)

        # Projection matrices should span the same subspace
        Pe = P_eigh.todense()
        Pq = P_qr.todense()
        proj_eigh = Pe @ Pe.T
        proj_qr = Pq @ Pq.T
        np.testing.assert_allclose(proj_qr, proj_eigh, atol=1e-10)

    def test_ctm_tensor_qr_symmetric_converges(self, small_peps_symmetric):
        """CTM with QR projector converges for symmetric tensors."""
        env = ctm_tensor(
            small_peps_symmetric,
            chi=4,
            max_iter=20,
            conv_tol=1e-6,
            projector_method="qr",
            qr_warmup_steps=3,
        )
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))


# ------------------------------------------------------------------ #
# Architecture guard tests                                             #
# ------------------------------------------------------------------ #


class TestStandardCTMArchitectureGuards:
    """Architecture guards for the standard CTM tensor path."""

    def test_symmetric_sweep_no_todense(self, small_peps_symmetric):
        """CTM sweeps on SymmetricTensor must not silently densify large tensors.

        Guards against regressions that would call todense()/from_dense()
        on the bulk of the CTM machinery — that would defeat block-sparsity
        on the dominant ``chi×D²``-cost contractions.

        ``_phase_fix_normalize_tensor`` is exempt: it densifies each C/T
        tensor (chi×chi or chi×D²×chi — small) once per sweep to compute
        the "first dense-order element above threshold" anchor, which is
        what ``test_energy_symmetric_matches_dense`` relies on for the
        symmetric and dense paths to agree element-by-element on the
        converged env. A buffer-order anchor would diverge for tensors
        with non-trivial charge sectors. Per CLAUDE.md: "Avoid todense()
        on the symmetric tensor path unless the resulting dense matrix is
        guaranteed to be small" — the phase-fix path satisfies that
        carve-out. Any *other* densification call should still fail the
        guard.
        """
        import traceback
        from unittest.mock import patch

        chi = 4
        a = _build_double_layer_tensor(small_peps_symmetric)
        env = initialize_ctm_tensor_env(small_peps_symmetric, chi)

        todense_callers: list[str] = []
        from_dense_callers: list[str] = []

        orig_todense = SymmetricTensor.todense
        orig_from_dense = SymmetricTensor.from_dense

        def _caller_function() -> str:
            """Name of the tenax-side function that invoked the patched op."""
            stack = traceback.extract_stack()
            for frame in reversed(stack):
                if (
                    "/tenax/" in frame.filename
                    and "tensor.py" not in frame.filename
                    and frame.name not in {"tracking_todense", "tracking_from_dense"}
                ):
                    return frame.name
            return "<unknown>"

        # Calls from these helpers are an intentional, narrowly-scoped
        # exception (see docstring).
        EXEMPT = {"_phase_fix_normalize_tensor"}

        def tracking_todense(self):
            caller = _caller_function()
            if caller not in EXEMPT:
                todense_callers.append(caller)
            return orig_todense(self)

        @classmethod
        def tracking_from_dense(cls, *args, **kwargs):
            caller = _caller_function()
            if caller not in EXEMPT:
                from_dense_callers.append(caller)
            return orig_from_dense(*args, **kwargs)

        with (
            patch.object(SymmetricTensor, "todense", tracking_todense),
            patch.object(SymmetricTensor, "from_dense", tracking_from_dense),
        ):
            _ctm_tensor_sweep(env, a, chi, renormalize=True)

        assert not todense_callers, (
            f"todense() called from non-exempt sites during symmetric sweep: "
            f"{todense_callers}"
        )
        assert not from_dense_callers, (
            f"from_dense() called from non-exempt sites during symmetric sweep: "
            f"{from_dense_callers}"
        )

    def test_standard_vs_split_energy_equivalence(self, heisenberg_gate):
        """Standard and split CTM produce the same energy for dense tensors.

        Uses a near-product-state iPEPS (dominant diagonal element with
        small perturbation) so that both algorithms converge to the same
        unique CTM fixed point, and lossless chi_I so the split path
        does not lose information.
        """
        from tenax.algorithms._split_ctm_tensor import (
            compute_energy_split_ctm_tensor,
            ctm_split_tensor,
        )

        D, d = 2, 2
        # Near-product-state tensor: unique CTM fixed point guaranteed
        rng = np.random.RandomState(42)
        data = 0.01 * jnp.array(rng.standard_normal((D, D, D, D, d)))
        data = data.at[0, 0, 0, 0, 0].set(1.0)
        data = data / (jnp.linalg.norm(data) + 1e-10)

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
        A = DenseTensor(data, indices)

        chi = 8
        chi_I = 16  # lossless

        # Standard CTM
        env_std = ctm_tensor(A, chi=chi, max_iter=50, conv_tol=1e-10)
        E_std = float(compute_energy_ctm_tensor(A, env_std, heisenberg_gate, d=d))

        # Split CTM with lossless chi_I
        env_split = ctm_split_tensor(A, chi=chi, max_iter=50, chi_I=chi_I)
        E_split = float(
            compute_energy_split_ctm_tensor(A, env_split, heisenberg_gate, d=d)
        )

        assert np.isfinite(E_std), f"Standard CTM energy is not finite: {E_std}"
        assert np.isfinite(E_split), f"Split CTM energy is not finite: {E_split}"
        np.testing.assert_allclose(
            E_split,
            E_std,
            atol=1e-6,
            err_msg=(f"Energy mismatch: standard={E_std}, split={E_split}"),
        )
