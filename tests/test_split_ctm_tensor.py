"""Tests for split CTM with Tensor protocol (polymorphic dense/symmetric)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._split_ctm_tensor import (
    SplitCTMTensorEnv,
    _split_ctm_tensor_sweep,
    compute_energy_split_ctm_tensor,
    ctm_split_tensor,
    initialize_split_ctm_tensor_env,
)
from tenax.algorithms.ipeps import CTMConfig, compute_energy_ctm, ctm, ctm_split
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_peps_dense():
    """Random DenseTensor iPEPS site tensor, D=2, d=2."""
    key = jax.random.PRNGKey(42)
    D, d = 2, 2
    data = jax.random.normal(key, (D, D, D, D, d))
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, indices)


@pytest.fixture
def small_peps_symmetric():
    """Random SymmetricTensor iPEPS site tensor with trivial U(1) charges."""
    key = jax.random.PRNGKey(99)
    D, d = 2, 2
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
    )
    data = jax.random.normal(key, (D, D, D, D, d))
    return SymmetricTensor.from_dense(data, indices)


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
# Phase 1: Initialization tests                                        #
# ------------------------------------------------------------------ #


class TestSplitCTMTensorInit:
    """Tests for SplitCTMTensorEnv initialization."""

    def test_dense_init_shapes(self, small_peps_dense):
        """All 12 tensors should have correct shapes and labels."""
        chi, chi_I, D = 8, 4, 2
        env = initialize_split_ctm_tensor_env(small_peps_dense, chi, chi_I)
        assert isinstance(env, SplitCTMTensorEnv)

        # Corners: (chi, chi)
        for C in [env.C1, env.C2, env.C3, env.C4]:
            assert C.todense().shape == (chi, chi)
            assert C.ndim == 2

        # Ket edges: (chi, D, chi_I)
        for T_ket in [env.T1_ket, env.T2_ket, env.T3_ket, env.T4_ket]:
            assert T_ket.todense().shape == (chi, D, chi_I)

        # Bra edges: (chi_I, D, chi)
        for T_bra in [env.T1_bra, env.T2_bra, env.T3_bra, env.T4_bra]:
            assert T_bra.todense().shape == (chi_I, D, chi)

    def test_dense_init_labels(self, small_peps_dense):
        """Check that labels are assigned correctly."""
        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(small_peps_dense, chi, chi_I)

        assert env.C1.labels() == ("c1_d", "c1_r")
        assert env.C2.labels() == ("c2_l", "c2_d")
        assert env.C3.labels() == ("c3_u", "c3_l")
        assert env.C4.labels() == ("c4_r", "c4_u")

        assert env.T1_ket.labels() == ("t1k_l", "u_ket", "t1k_I")
        assert env.T1_bra.labels() == ("t1b_I", "u_bra", "t1b_r")

    def test_dense_init_finite(self, small_peps_dense):
        """All initialized tensors should be finite."""
        chi, chi_I = 8, 4
        env = initialize_split_ctm_tensor_env(small_peps_dense, chi, chi_I)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense()))

    def test_symmetric_init_shapes(self, small_peps_symmetric):
        """SymmetricTensor initialization should produce correct shapes."""
        chi, chi_I = 4, 2
        D = 2
        env = initialize_split_ctm_tensor_env(small_peps_symmetric, chi, chi_I)
        assert isinstance(env, SplitCTMTensorEnv)

        for C in [env.C1, env.C2, env.C3, env.C4]:
            assert isinstance(C, SymmetricTensor)
            assert C.todense().shape == (chi, chi)

        for T_ket in [env.T1_ket, env.T2_ket, env.T3_ket, env.T4_ket]:
            assert isinstance(T_ket, SymmetricTensor)
            assert T_ket.todense().shape == (chi, D, chi_I)

        for T_bra in [env.T1_bra, env.T2_bra, env.T3_bra, env.T4_bra]:
            assert isinstance(T_bra, SymmetricTensor)
            assert T_bra.todense().shape == (chi_I, D, chi)


# ------------------------------------------------------------------ #
# Phase 2: Single-move tests                                           #
# ------------------------------------------------------------------ #


class TestSplitCTMMoves:
    """Tests for individual CTM moves."""

    def test_one_sweep_produces_finite(self, small_peps_dense):
        """One full sweep should produce finite tensors."""
        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(small_peps_dense, chi, chi_I)
        env = _split_ctm_tensor_sweep(env, small_peps_dense, chi, chi_I, True)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense())), (
                "Sweep produced non-finite tensors"
            )


# ------------------------------------------------------------------ #
# Phase 3: Convergence tests                                           #
# ------------------------------------------------------------------ #


class TestSplitCTMTensorConvergence:
    """Tests for full CTM convergence."""

    def test_converges(self, small_peps_dense):
        """Split-CTM should produce finite environment after convergence."""
        env = ctm_split_tensor(small_peps_dense, chi=8, max_iter=30, chi_I=4)
        assert isinstance(env, SplitCTMTensorEnv)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense()))

    def test_chi_I_equals_chi(self, small_peps_dense):
        """chi_I=chi should also work (no interlayer compression)."""
        env = ctm_split_tensor(small_peps_dense, chi=8, max_iter=30, chi_I=8)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense()))


# ------------------------------------------------------------------ #
# Correctness tests                                                    #
# ------------------------------------------------------------------ #


class TestSplitCTMTensorEnergy:
    """Energy correctness tests."""

    def test_energy_is_finite(self, small_peps_dense, heisenberg_gate):
        """Split-CTM should produce finite energy."""
        env = ctm_split_tensor(small_peps_dense, chi=8, max_iter=50)
        E = compute_energy_split_ctm_tensor(small_peps_dense, env, heisenberg_gate, d=2)
        assert jnp.isfinite(E)

    def test_energy_roundtrip_via_standard(self, small_peps_dense, heisenberg_gate):
        """Energy via split env should match energy via standard-converted env.

        This mirrors the existing ``test_split_ctm_energy_matches_standard``
        for the dense split-CTM: convert the split environment to a standard
        CTMEnvironment and verify the energy is identical.
        """
        from tenax.algorithms._split_ctm_tensor import _split_env_to_dense_standard
        from tenax.algorithms.ipeps import CTMEnvironment

        d = 2
        chi = 8
        chi_I = chi * 2  # lossless

        env = ctm_split_tensor(small_peps_dense, chi=chi, max_iter=50, chi_I=chi_I)
        E_split = compute_energy_split_ctm_tensor(
            small_peps_dense, env, heisenberg_gate, d
        )

        # Manually convert and compute
        C1, C2, C3, C4, T1, T2, T3, T4 = _split_env_to_dense_standard(env)
        std_env = CTMEnvironment(C1=C1, C2=C2, C3=C3, C4=C4, T1=T1, T2=T2, T3=T3, T4=T4)
        A_raw = small_peps_dense.todense()
        H = heisenberg_gate.reshape(d, d, d, d)
        E_from_std = compute_energy_ctm(A_raw, std_env, H, d)

        assert jnp.abs(E_split - E_from_std) < 1e-12, (
            f"Roundtrip mismatch: split={float(E_split)}, from_std={float(E_from_std)}"
        )

    def test_grow_edge_matches_double_layer(self, small_peps_dense):
        """_grow_edge_no_double_layer matches the old double-layer approach.

        Verifies the no-double-layer contraction produces the same
        grown T-edge tensor as the merge + double-layer + einsum path.
        """
        from tenax.algorithms._split_ctm_tensor import _grow_edge_no_double_layer
        from tenax.algorithms.ipeps import (
            _build_double_layer,
            _initialize_split_ctm_env,
        )

        A_raw = small_peps_dense.todense()
        D = A_raw.shape[0]
        chi, chi_I = 8, 4

        env_t = initialize_split_ctm_tensor_env(small_peps_dense, chi, chi_I)
        env_d = _initialize_split_ctm_env(A_raw, chi, chi_I)
        A_bar = small_peps_dense.bar()

        a = _build_double_layer(A_raw)
        if a.ndim == 8:
            a = a.reshape(D**2, D**2, D**2, D**2)

        # --- Left move: T4 growth ---
        T4_full = jnp.einsum("alc,cLg->alLg", env_d.T4_ket, env_d.T4_bra)
        T4_full = T4_full.reshape(chi, D * D, chi)
        T4g_old = jnp.einsum("alg,udlr->augdr", T4_full, a)
        T4g_old = T4g_old.transpose(0, 1, 4, 2, 3).reshape(chi * D**2, D**2, chi * D**2)

        T4g_new = _grow_edge_no_double_layer(
            env_t.T4_ket,
            env_t.T4_bra,
            small_peps_dense,
            A_bar,
            "l",
            "t4k_I",
            "t4b_I",
            ("t4k_d", "u", "U", "r", "R", "t4b_u", "d", "D"),
        )
        assert jnp.allclose(T4g_old, T4g_new, atol=1e-12), "T4 (left) growth mismatch"

        # --- Right move: T2 growth ---
        T2_full = jnp.einsum("alc,cLg->alLg", env_d.T2_ket, env_d.T2_bra)
        T2_full = T2_full.reshape(chi, D * D, chi)
        T2g_old = jnp.einsum("erm,udlr->eumdl", T2_full, a)
        T2g_old = T2g_old.transpose(0, 1, 4, 2, 3).reshape(chi * D**2, D**2, chi * D**2)

        T2g_new = _grow_edge_no_double_layer(
            env_t.T2_ket,
            env_t.T2_bra,
            small_peps_dense,
            A_bar,
            "r",
            "t2k_I",
            "t2b_I",
            ("t2k_u", "u", "U", "l", "L", "t2b_d", "d", "D"),
        )
        assert jnp.allclose(T2g_old, T2g_new, atol=1e-12), "T2 (right) growth mismatch"

        # --- Top move: T1 growth ---
        T1_full = jnp.einsum("alc,cLg->alLg", env_d.T1_ket, env_d.T1_bra)
        T1_full = T1_full.reshape(chi, D * D, chi)
        T1g_old = jnp.einsum("buc,udlr->bcdlr", T1_full, a)
        T1g_old = T1g_old.transpose(0, 3, 2, 1, 4).reshape(chi * D**2, D**2, chi * D**2)

        T1g_new = _grow_edge_no_double_layer(
            env_t.T1_ket,
            env_t.T1_bra,
            small_peps_dense,
            A_bar,
            "u",
            "t1k_I",
            "t1b_I",
            ("t1k_l", "l", "L", "d", "D", "t1b_r", "r", "R"),
        )
        assert jnp.allclose(T1g_old, T1g_new, atol=1e-12), "T1 (top) growth mismatch"

        # --- Bottom move: T3 growth ---
        T3_full = jnp.einsum("alc,cLg->alLg", env_d.T3_ket, env_d.T3_bra)
        T3_full = T3_full.reshape(chi, D * D, chi)
        T3g_old = jnp.einsum("hdi,udlr->hiulr", T3_full, a)
        T3g_old = T3g_old.transpose(0, 3, 2, 1, 4).reshape(chi * D**2, D**2, chi * D**2)

        T3g_new = _grow_edge_no_double_layer(
            env_t.T3_ket,
            env_t.T3_bra,
            small_peps_dense,
            A_bar,
            "d",
            "t3k_I",
            "t3b_I",
            ("t3k_r", "l", "L", "u", "U", "t3b_l", "r", "R"),
        )
        assert jnp.allclose(T3g_old, T3g_new, atol=1e-12), "T3 (bottom) growth mismatch"


# ------------------------------------------------------------------ #
# SymmetricTensor tests                                                #
# ------------------------------------------------------------------ #


class TestSplitCTMSymmetric:
    """Tests for SymmetricTensor iPEPS with trivial and nontrivial charges."""

    def test_symmetric_one_sweep_finite(self, small_peps_symmetric):
        """One CTM sweep with trivial-charge SymmetricTensor A produces finite tensors."""
        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(small_peps_symmetric, chi, chi_I)
        env = _split_ctm_tensor_sweep(env, small_peps_symmetric, chi, chi_I, True)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense())), (
                "SymmetricTensor sweep produced non-finite tensors"
            )

    def test_fermionic_u1_one_sweep_finite(self):
        """One CTM sweep with FermionicU1 charges produces finite tensors.

        This is the key regression test: with dagger(), the physical trace
        loses blocks because charge 1 is dualled to -1 and mismatches.
        With bar(), charges stay identical so all blocks are preserved.
        """
        from tenax.core.symmetry import FermionicU1

        key = jax.random.PRNGKey(77)
        sym = FermionicU1()
        virt_charges = np.array([0, 1], dtype=np.int32)
        phys_charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex(sym, virt_charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex(sym, virt_charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex(sym, virt_charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex(sym, virt_charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
        )
        A = SymmetricTensor.random_normal(indices, key)
        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(A, chi, chi_I)
        env = _split_ctm_tensor_sweep(env, A, chi, chi_I, True)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense())), (
                "FermionicU1 sweep produced non-finite tensors"
            )
