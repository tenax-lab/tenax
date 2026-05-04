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
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.ipeps_ctm import ctm, ctm_split
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


def make_random_dense_site(D: int, d: int, seed: int) -> DenseTensor:
    """Build a random U(1)-trivial DenseTensor iPEPS site for parity tests."""
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, d))
    data = data / jnp.linalg.norm(data)
    sym = U1Symmetry()
    z_D = np.zeros(D, dtype=np.int32)
    z_d = np.zeros(d, dtype=np.int32)
    return DenseTensor(
        data,
        (
            TensorIndex.from_charges(sym, z_D, FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, z_D, FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, z_D, FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, z_D, FlowDirection.IN, label="r"),
            TensorIndex.from_charges(sym, z_d, FlowDirection.IN, label="phys"),
        ),
    )


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
    """Random SymmetricTensor iPEPS site tensor with trivial U(1) charges."""
    key = jax.random.PRNGKey(99)
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
# Public API smoke test                                                #
# ------------------------------------------------------------------ #


def test_public_exports_resolve():
    """Top-level tenax package exposes the new split-CTM energy entry points."""
    import tenax

    assert hasattr(tenax, "compute_energy_split_ctm_tensor_2site")
    assert hasattr(tenax, "compute_energy_split_ctm_tensor_multisite")


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
        CTMTensorEnv and verify the energy is identical.
        """
        from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor
        from tenax.algorithms._split_ctm_tensor import _split_env_to_tensor_standard

        d = 2
        chi = 8
        chi_I = chi * 2  # lossless

        env = ctm_split_tensor(small_peps_dense, chi=chi, max_iter=50, chi_I=chi_I)
        E_split = compute_energy_split_ctm_tensor(
            small_peps_dense, env, heisenberg_gate, d
        )

        # Manually convert and compute
        std_env = _split_env_to_tensor_standard(env)
        E_from_std = compute_energy_ctm_tensor(
            small_peps_dense, std_env, heisenberg_gate, d
        )

        assert jnp.abs(E_split - E_from_std) < 1e-12, (
            f"Roundtrip mismatch: split={float(E_split)}, from_std={float(E_from_std)}"
        )

    def test_grow_edge_matches_double_layer(self, small_peps_dense):
        """_grow_edge_no_double_layer matches the old double-layer approach.

        Verifies the no-double-layer contraction produces the same
        grown T-edge tensor as the merge + double-layer + einsum path.
        """
        from tenax.algorithms._split_ctm_tensor import _grow_edge_no_double_layer
        from tenax.algorithms.ipeps_ctm import (
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

        T4g_new = (
            _grow_edge_no_double_layer(
                env_t.T4_ket,
                env_t.T4_bra,
                small_peps_dense,
                A_bar,
                "l",
                "t4k_I",
                "t4b_I",
                ("t4k_d", "u", "U", "r", "R", "t4b_u", "d", "D"),
            )
            .todense()
            .reshape(chi * D * D, D * D, chi * D * D)
        )
        assert jnp.allclose(T4g_old, T4g_new, atol=1e-12), "T4 (left) growth mismatch"

        # --- Right move: T2 growth ---
        T2_full = jnp.einsum("alc,cLg->alLg", env_d.T2_ket, env_d.T2_bra)
        T2_full = T2_full.reshape(chi, D * D, chi)
        T2g_old = jnp.einsum("erm,udlr->eumdl", T2_full, a)
        T2g_old = T2g_old.transpose(0, 1, 4, 2, 3).reshape(chi * D**2, D**2, chi * D**2)

        T2g_new = (
            _grow_edge_no_double_layer(
                env_t.T2_ket,
                env_t.T2_bra,
                small_peps_dense,
                A_bar,
                "r",
                "t2k_I",
                "t2b_I",
                ("t2k_u", "u", "U", "l", "L", "t2b_d", "d", "D"),
            )
            .todense()
            .reshape(chi * D * D, D * D, chi * D * D)
        )
        assert jnp.allclose(T2g_old, T2g_new, atol=1e-12), "T2 (right) growth mismatch"

        # --- Top move: T1 growth ---
        T1_full = jnp.einsum("alc,cLg->alLg", env_d.T1_ket, env_d.T1_bra)
        T1_full = T1_full.reshape(chi, D * D, chi)
        T1g_old = jnp.einsum("buc,udlr->bcdlr", T1_full, a)
        T1g_old = T1g_old.transpose(0, 3, 2, 1, 4).reshape(chi * D**2, D**2, chi * D**2)

        T1g_new = (
            _grow_edge_no_double_layer(
                env_t.T1_ket,
                env_t.T1_bra,
                small_peps_dense,
                A_bar,
                "u",
                "t1k_I",
                "t1b_I",
                ("t1k_l", "l", "L", "d", "D", "t1b_r", "r", "R"),
            )
            .todense()
            .reshape(chi * D * D, D * D, chi * D * D)
        )
        assert jnp.allclose(T1g_old, T1g_new, atol=1e-12), "T1 (top) growth mismatch"

        # --- Bottom move: T3 growth ---
        T3_full = jnp.einsum("alc,cLg->alLg", env_d.T3_ket, env_d.T3_bra)
        T3_full = T3_full.reshape(chi, D * D, chi)
        T3g_old = jnp.einsum("hdi,udlr->hiulr", T3_full, a)
        T3g_old = T3g_old.transpose(0, 3, 2, 1, 4).reshape(chi * D**2, D**2, chi * D**2)

        T3g_new = (
            _grow_edge_no_double_layer(
                env_t.T3_ket,
                env_t.T3_bra,
                small_peps_dense,
                A_bar,
                "d",
                "t3k_I",
                "t3b_I",
                ("t3k_r", "l", "L", "u", "U", "t3b_l", "r", "R"),
            )
            .todense()
            .reshape(chi * D * D, D * D, chi * D * D)
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
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.OUT, label="u"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.IN, label="d"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.OUT, label="l"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.IN, label="r"
            ),
            TensorIndex.from_charges(
                sym, phys_charges.copy(), FlowDirection.IN, label="phys"
            ),
        )
        A = SymmetricTensor.random_normal(indices, key)
        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(A, chi, chi_I)
        env = _split_ctm_tensor_sweep(env, A, chi, chi_I, True)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense())), (
                "FermionicU1 sweep produced non-finite tensors"
            )

    def test_symmetric_multi_sweep_converges(self, small_peps_symmetric):
        """Multiple CTM sweeps with SymmetricTensor A don't crash (regression for type mixing)."""
        env = ctm_split_tensor(small_peps_symmetric, chi=4, max_iter=5, chi_I=2)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense())), (
                "Multi-sweep symmetric CTM produced non-finite tensors"
            )

    def test_symmetric_energy_matches_dense(self, small_peps_dense, heisenberg_gate):
        """SymmetricTensor split-CTM energy should match DenseTensor result.

        Uses the same tensor data wrapped as both DenseTensor and SymmetricTensor
        (with trivial U(1) charges) to verify the symmetric path is correct.
        """
        # Build SymmetricTensor with same data as small_peps_dense
        A_sym = SymmetricTensor.from_dense(
            small_peps_dense.todense(), small_peps_dense.indices
        )

        chi, chi_I = 8, 4
        d = 2

        env_d = ctm_split_tensor(small_peps_dense, chi=chi, max_iter=50, chi_I=chi_I)
        E_dense = compute_energy_split_ctm_tensor(
            small_peps_dense, env_d, heisenberg_gate, d
        )

        env_s = ctm_split_tensor(A_sym, chi=chi, max_iter=50, chi_I=chi_I)
        E_sym = compute_energy_split_ctm_tensor(A_sym, env_s, heisenberg_gate, d)

        assert jnp.isfinite(E_sym), f"Symmetric energy is not finite: {float(E_sym)}"
        assert jnp.abs(E_dense - E_sym) < 1e-6, (
            f"Energy mismatch: dense={float(E_dense)}, sym={float(E_sym)}"
        )

    def test_fermionic_u1_charges_preserved_across_sweeps(self):
        """Nontrivial charge sectors survive multiple CTM sweeps (regression for charge collapse)."""
        from tenax.core.symmetry import FermionicU1

        key = jax.random.PRNGKey(77)
        sym = FermionicU1()
        virt_charges = np.array([0, 1], dtype=np.int32)
        phys_charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.OUT, label="u"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.IN, label="d"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.OUT, label="l"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.IN, label="r"
            ),
            TensorIndex.from_charges(
                sym, phys_charges.copy(), FlowDirection.IN, label="phys"
            ),
        )
        A = SymmetricTensor.random_normal(indices, key)
        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(A, chi, chi_I)
        # Run 3 sweeps
        for _ in range(3):
            env = _split_ctm_tensor_sweep(env, A, chi, chi_I, True)
        # All tensors must remain SymmetricTensors with at least 1 block.
        # Block count may decrease from initial because the block-sparse
        # projector keeps the dominant charge sectors (unlike the old
        # dense code which forced charge distribution via _derive_charges).
        for t in env:
            assert isinstance(t, SymmetricTensor), (
                f"Expected SymmetricTensor, got {type(t)}"
            )
            assert t.n_blocks >= 1, (
                "All blocks collapsed to 0 — environment degenerated"
            )

    def test_symmetric_sweep_no_todense(self, small_peps_symmetric):
        """CTM sweeps on SymmetricTensor must not call todense() or from_dense().

        This guards against regressions that silently densify during the
        symmetric path.  The energy measurement is excluded because it
        currently bridges to dense (known limitation).
        """
        from unittest.mock import patch

        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(small_peps_symmetric, chi, chi_I)

        todense_calls = []
        from_dense_calls = []

        orig_todense = SymmetricTensor.todense
        orig_from_dense = SymmetricTensor.from_dense

        def tracking_todense(self):
            todense_calls.append(True)
            return orig_todense(self)

        @classmethod
        def tracking_from_dense(cls, *args, **kwargs):
            from_dense_calls.append(True)
            return orig_from_dense(*args, **kwargs)

        with (
            patch.object(SymmetricTensor, "todense", tracking_todense),
            patch.object(SymmetricTensor, "from_dense", tracking_from_dense),
        ):
            _split_ctm_tensor_sweep(env, small_peps_symmetric, chi, chi_I, True)

        assert len(todense_calls) == 0, (
            f"todense() called {len(todense_calls)} times during symmetric sweep"
        )
        assert len(from_dense_calls) == 0, (
            f"from_dense() called {len(from_dense_calls)} times during symmetric sweep"
        )


class TestSplitEdgeHelper:
    """Tests for the new _make_split_edge / _make_split_edges helpers."""

    def test_make_split_edge_shape_and_labels(self, small_peps_dense):
        """_make_split_edge contracts T_ket·T_bra on _I, leaves D-legs unfused."""
        from tenax.algorithms._split_ctm_tensor_energy import _make_split_edge

        env = initialize_split_ctm_tensor_env(small_peps_dense, chi=4, chi_I=4)
        T1 = _make_split_edge(
            env.T1_ket,
            env.T1_bra,
            ket_I="t1k_I",
            bra_I="t1b_I",
            ket_chi="t1k_l",
            bra_chi="t1b_r",
            out_chi_l="t1_l",
            out_chi_r="t1_r",
        )
        labels = T1.labels()
        # Four legs: chi_l, u_ket, u_bra, chi_r — D-legs unchanged from inputs.
        assert set(labels) == {"t1_l", "u_ket", "u_bra", "t1_r"}
        # Dimensions: (chi, D, D, chi). With D=2, chi=4 → (4, 2, 2, 4).
        dim_by_label = {idx.label: idx.dim for idx in T1.indices}
        assert dim_by_label["t1_l"] == 4
        assert dim_by_label["t1_r"] == 4
        assert dim_by_label["u_ket"] == 2
        assert dim_by_label["u_bra"] == 2

    def test_make_split_edges_no_label_collisions(self, small_peps_dense):
        """The four T_split tensors share no D-leg labels; chi labels follow the
        standard CTM convention (t1_l/t1_r, t2_u/t2_d, t3_l/t3_r, t4_d/t4_u)."""
        from tenax.algorithms._split_ctm_tensor_energy import _make_split_edges

        env = initialize_split_ctm_tensor_env(small_peps_dense, chi=4, chi_I=4)
        splits = _make_split_edges(env)

        assert set(splits.keys()) == {"T1", "T2", "T3", "T4"}
        # D-leg label sets per edge are disjoint:
        d_labels = {
            "T1": {"u_ket", "u_bra"},
            "T2": {"r_ket", "r_bra"},
            "T3": {"d_ket", "d_bra"},
            "T4": {"l_ket", "l_bra"},
        }
        for name, want in d_labels.items():
            labs = set(splits[name].labels())
            assert want.issubset(labs), f"{name} missing {want - labs}"
        # No D-leg label collides across edges:
        all_d = set().union(*d_labels.values())
        assert len(all_d) == 8  # all distinct


class TestSplitRDMs:
    """Parity vs the shim for each split-aware RDM."""

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_rdm_1site_matches_shim(self, D, chi):
        from tenax.algorithms._ctm_tensor_energy import _rdm_1site_tensor
        from tenax.algorithms._split_ctm_tensor_energy import (
            _rdm_1site_split_tensor,
            _split_env_to_tensor_standard,
        )

        A = make_random_dense_site(D, d=2, seed=0)
        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

        rdm_split = _rdm_1site_split_tensor(A, env)
        rdm_shim = _rdm_1site_tensor(A, _split_env_to_tensor_standard(env))
        assert jnp.allclose(rdm_split, rdm_shim, atol=1e-10)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12), (4, 16)])
    def test_rdm1x2_matches_shim(self, D, chi):
        from tenax.algorithms._ctm_tensor_energy import _rdm1x2_tensor
        from tenax.algorithms._split_ctm_tensor_energy import (
            _rdm1x2_split_tensor,
            _split_env_to_tensor_standard,
        )

        A = make_random_dense_site(D, d=2, seed=1)
        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

        rdm_split = _rdm1x2_split_tensor(A, env)
        rdm_shim = _rdm1x2_tensor(A, _split_env_to_tensor_standard(env))
        assert jnp.allclose(rdm_split, rdm_shim, atol=1e-10)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12), (4, 16)])
    def test_rdm2x1_matches_shim(self, D, chi):
        from tenax.algorithms._ctm_tensor_energy import _rdm2x1_tensor
        from tenax.algorithms._split_ctm_tensor_energy import (
            _rdm2x1_split_tensor,
            _split_env_to_tensor_standard,
        )

        A = make_random_dense_site(D, d=2, seed=2)
        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

        rdm_split = _rdm2x1_split_tensor(A, env)
        rdm_shim = _rdm2x1_tensor(A, _split_env_to_tensor_standard(env))
        assert jnp.allclose(rdm_split, rdm_shim, atol=1e-10)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_rdm_diagonal_matches_shim(self, D, chi):
        from tenax.algorithms._ctm_tensor_energy import _rdm_diagonal_tensor
        from tenax.algorithms._split_ctm_tensor_energy import (
            _rdm_diagonal_split_tensor,
            _split_env_to_tensor_standard,
        )

        A = make_random_dense_site(D, d=2, seed=3)
        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

        rdm_split = _rdm_diagonal_split_tensor(A, env)
        rdm_shim = _rdm_diagonal_tensor(A, _split_env_to_tensor_standard(env))
        assert jnp.allclose(rdm_split, rdm_shim, atol=1e-10)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_rdm1x2_2site_matches_shim(self, D, chi):
        from tenax.algorithms._ctm_tensor_energy import _rdm1x2_tensor_2site
        from tenax.algorithms._split_ctm_tensor_energy import (
            _rdm1x2_split_tensor_2site,
            _split_env_to_tensor_standard,
        )

        A = make_random_dense_site(D, d=2, seed=10)
        B = make_random_dense_site(D, d=2, seed=11)
        env_A = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
        env_B = ctm_split_tensor(B, chi=chi, max_iter=20, chi_I=chi)

        rdm_split = _rdm1x2_split_tensor_2site(A, B, env_A, env_B)
        rdm_shim = _rdm1x2_tensor_2site(
            A,
            B,
            _split_env_to_tensor_standard(env_A),
            _split_env_to_tensor_standard(env_B),
        )
        assert jnp.allclose(rdm_split, rdm_shim, atol=1e-10)

        # Distinctness check: mixed-env result must differ from single-env result with A on both sites.
        from tenax.algorithms._split_ctm_tensor_energy import _rdm1x2_split_tensor

        rdm_single = _rdm1x2_split_tensor(A, env_A)
        assert not jnp.allclose(rdm_split, rdm_single, atol=1e-6)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_rdm2x1_2site_matches_shim(self, D, chi):
        from tenax.algorithms._ctm_tensor_energy import _rdm2x1_tensor_2site
        from tenax.algorithms._split_ctm_tensor_energy import (
            _rdm2x1_split_tensor_2site,
            _split_env_to_tensor_standard,
        )

        A = make_random_dense_site(D, d=2, seed=20)
        B = make_random_dense_site(D, d=2, seed=21)
        env_A = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
        env_B = ctm_split_tensor(B, chi=chi, max_iter=20, chi_I=chi)

        rdm_split = _rdm2x1_split_tensor_2site(A, B, env_A, env_B)
        rdm_shim = _rdm2x1_tensor_2site(
            A,
            B,
            _split_env_to_tensor_standard(env_A),
            _split_env_to_tensor_standard(env_B),
        )
        assert jnp.allclose(rdm_split, rdm_shim, atol=1e-10)

        # Distinctness check: mixed-env result must differ from single-env result with A on both sites.
        from tenax.algorithms._split_ctm_tensor_energy import _rdm2x1_split_tensor

        rdm_single = _rdm2x1_split_tensor(A, env_A)
        assert not jnp.allclose(rdm_split, rdm_single, atol=1e-6)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12), (4, 16)])
    def test_compute_energy_split_native_matches_shim(self, D, chi, heisenberg_gate):
        """Split-aware energy must match shim energy at small D."""
        from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
            compute_energy_split_ctm_tensor,
        )

        A = make_random_dense_site(D, d=2, seed=30)
        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

        E_split = compute_energy_split_ctm_tensor(A, env, heisenberg_gate, d=2)
        E_shim = compute_energy_ctm_tensor(
            A, _split_env_to_tensor_standard(env), heisenberg_gate, d=2
        )
        assert jnp.allclose(E_split, E_shim, atol=1e-10)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_compute_energy_split_2site_matches_shim(self, D, chi, heisenberg_gate):
        from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
            compute_energy_split_ctm_tensor_2site,
        )

        A = make_random_dense_site(D, d=2, seed=40)
        B = make_random_dense_site(D, d=2, seed=41)
        env_A = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
        env_B = ctm_split_tensor(B, chi=chi, max_iter=20, chi_I=chi)

        E_split = compute_energy_split_ctm_tensor_2site(
            A, B, env_A, env_B, heisenberg_gate, d=2
        )
        E_shim = compute_energy_ctm_tensor_2site(
            A,
            B,
            _split_env_to_tensor_standard(env_A),
            _split_env_to_tensor_standard(env_B),
            heisenberg_gate,
            d=2,
        )
        assert jnp.allclose(E_split, E_shim, atol=1e-10)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_compute_energy_split_multisite_matches_shim(self, D, chi, heisenberg_gate):
        """Split-aware multisite energy must match shim at small D on a Y-shaped 3-site cell."""
        from tenax.algorithms._ctm_tensor_energy import (
            compute_energy_ctm_tensor_multisite,
        )
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
            compute_energy_split_ctm_tensor_multisite,
        )

        coords = [(0, 0), (1, 0), (0, 1)]
        site_tensors = {
            (0, 0): make_random_dense_site(D, d=2, seed=50),
            (1, 0): make_random_dense_site(D, d=2, seed=51),
            (0, 1): make_random_dense_site(D, d=2, seed=52),
        }
        envs_split = {
            c: ctm_split_tensor(site_tensors[c], chi=chi, max_iter=20, chi_I=chi)
            for c in coords
        }
        # Y-shaped neighbors: (0,0) connects right→(1,0) and bottom→(0,1).
        # Other coords loop back to themselves on directions that don't exit the cell.
        neighbors = {
            (0, 0): {
                "right": (1, 0),
                "bottom": (0, 1),
                "left": (1, 0),
                "top": (0, 1),
            },
            (1, 0): {
                "left": (0, 0),
                "top": (0, 1),
                "right": (0, 0),
                "bottom": (0, 1),
            },
            (0, 1): {
                "top": (0, 0),
                "right": (1, 0),
                "bottom": (0, 0),
                "left": (1, 0),
            },
        }

        E_split = compute_energy_split_ctm_tensor_multisite(
            site_tensors, envs_split, neighbors, heisenberg_gate, d=2
        )
        envs_std = {c: _split_env_to_tensor_standard(envs_split[c]) for c in coords}
        E_shim = compute_energy_ctm_tensor_multisite(
            site_tensors, envs_std, neighbors, heisenberg_gate, d=2
        )
        assert jnp.allclose(E_split, E_shim, atol=1e-10)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_compute_energy_split_multisite_1site_cell(self, D, chi, heisenberg_gate):
        """Single-site cell with self-loops exercises coord == nb_coord branch."""
        from tenax.algorithms._ctm_tensor_energy import (
            compute_energy_ctm_tensor_multisite,
        )
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
            compute_energy_split_ctm_tensor_multisite,
        )

        coord = (0, 0)
        A = make_random_dense_site(D, d=2, seed=60)
        site_tensors = {coord: A}
        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
        envs_split = {coord: env}
        # All directions self-loop — every bond is single-env (coord == nb_coord).
        neighbors = {
            coord: {"right": coord, "bottom": coord, "left": coord, "top": coord}
        }

        E_split = compute_energy_split_ctm_tensor_multisite(
            site_tensors, envs_split, neighbors, heisenberg_gate, d=2
        )
        envs_std = {coord: _split_env_to_tensor_standard(env)}
        E_shim = compute_energy_ctm_tensor_multisite(
            site_tensors, envs_std, neighbors, heisenberg_gate, d=2
        )
        assert jnp.allclose(E_split, E_shim, atol=1e-10)
