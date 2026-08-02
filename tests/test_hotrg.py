"""Tests for the HOTRG (Higher-Order Tensor Renormalization Group) algorithm."""

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.hotrg import (
    HOTRGConfig,
    _hotrg_step_horizontal,
    _hotrg_step_vertical,
    hotrg,
)
from tenax.algorithms.trg import compute_ising_tensor, ising_free_energy_exact
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


def _make_dense_tensor(arr: np.ndarray) -> DenseTensor:
    """Wrap a raw (d,d,d,d) array as a DenseTensor with HOTRG labels."""
    sym = U1Symmetry()
    d = arr.shape[0]
    charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="up"),
        TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="down"),
        TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="right"),
    )
    return DenseTensor(jnp.array(arr), indices)


class TestHOTRGConfig:
    def test_default_values(self):
        cfg = HOTRGConfig()
        assert cfg.max_bond_dim == 16
        assert cfg.num_steps == 10
        assert cfg.direction_order == "alternating"

    def test_custom_values(self):
        cfg = HOTRGConfig(max_bond_dim=8, num_steps=5, direction_order="horizontal")
        assert cfg.max_bond_dim == 8
        assert cfg.num_steps == 5
        assert cfg.direction_order == "horizontal"


class TestHOTRGStepHorizontal:
    def test_output_is_tensor(self):
        """Horizontal HOTRG step: output should be a Tensor with 4 legs."""
        T = _make_dense_tensor(
            np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float64)
        )
        T_new, log_norm = _hotrg_step_horizontal(T, max_bond_dim=3)
        assert isinstance(T_new, DenseTensor)
        assert T_new.todense().ndim == 4

    def test_log_norm_finite(self):
        T = _make_dense_tensor(
            np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float64)
        )
        _, log_norm = _hotrg_step_horizontal(T, max_bond_dim=3)
        assert np.isfinite(float(log_norm))

    def test_bond_dim_truncation(self):
        """Up/down bond dims should be bounded by max_bond_dim after step."""
        T = _make_dense_tensor(
            np.random.default_rng(0).random((4, 4, 4, 4)).astype(np.float64)
        )
        T_new, _ = _hotrg_step_horizontal(T, max_bond_dim=3)
        arr = T_new.todense()
        # After horizontal step, up/down legs (axes 0,1) are compressed
        assert arr.shape[0] <= 3
        assert arr.shape[1] <= 3

    def test_output_normalized(self):
        """Output tensor should be normalized (max |entry| ≈ 1)."""
        T = _make_dense_tensor(
            np.random.default_rng(42).random((2, 2, 2, 2)).astype(np.float64)
        )
        T_new, _ = _hotrg_step_horizontal(T, max_bond_dim=4)
        arr = np.array(T_new.todense())
        max_val = np.max(np.abs(arr))
        assert np.isclose(max_val, 1.0, atol=0.05), f"Expected ~1, got {max_val}"


class TestHOTRGStepVertical:
    def test_output_is_tensor(self):
        T = _make_dense_tensor(
            np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float64)
        )
        T_new, log_norm = _hotrg_step_vertical(T, max_bond_dim=3)
        assert isinstance(T_new, DenseTensor)
        assert T_new.todense().ndim == 4

    def test_log_norm_finite(self):
        T = _make_dense_tensor(
            np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float64)
        )
        _, log_norm = _hotrg_step_vertical(T, max_bond_dim=3)
        assert np.isfinite(float(log_norm))

    def test_bond_dim_truncation(self):
        """Left/right bond dims should be bounded by max_bond_dim."""
        T = _make_dense_tensor(
            np.random.default_rng(0).random((4, 4, 4, 4)).astype(np.float64)
        )
        T_new, _ = _hotrg_step_vertical(T, max_bond_dim=3)
        arr = T_new.todense()
        assert arr.shape[2] <= 3
        assert arr.shape[3] <= 3

    def test_output_normalized(self):
        T = _make_dense_tensor(
            np.random.default_rng(42).random((2, 2, 2, 2)).astype(np.float64)
        )
        T_new, _ = _hotrg_step_vertical(T, max_bond_dim=4)
        arr = np.array(T_new.todense())
        max_val = np.max(np.abs(arr))
        assert np.isclose(max_val, 1.0, atol=0.05), f"Expected ~1, got {max_val}"


class TestHOTRGRun:
    @pytest.fixture
    def ising_tensor_high_temp(self):
        return compute_ising_tensor(beta=0.2)

    def test_high_temp_free_energy(self):
        """At high temperature (beta=0.2), HOTRG chi=16 should be within 0.5%."""
        beta = 0.2
        tensor = compute_ising_tensor(beta=beta)
        config = HOTRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = hotrg(tensor, config)
        hotrg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(hotrg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.005, (
            f"HOTRG free energy {hotrg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )

    def test_mid_temp_free_energy(self):
        """At beta=0.3, HOTRG chi=16 should be within 1%."""
        beta = 0.3
        tensor = compute_ising_tensor(beta=beta)
        config = HOTRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = hotrg(tensor, config)
        hotrg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(hotrg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.01, (
            f"HOTRG free energy {hotrg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )

    def test_near_critical_free_energy(self):
        """Near critical point (beta=0.44), HOTRG chi=16 should be within 2%."""
        beta = 0.44
        tensor = compute_ising_tensor(beta=beta)
        config = HOTRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = hotrg(tensor, config)
        hotrg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(hotrg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.02, (
            f"HOTRG free energy {hotrg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )

    def test_low_temp_free_energy(self):
        """At low temperature (beta=0.6), HOTRG chi=16 should be within 0.5%."""
        beta = 0.6
        tensor = compute_ising_tensor(beta=beta)
        config = HOTRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = hotrg(tensor, config)
        hotrg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(hotrg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.005, (
            f"HOTRG free energy {hotrg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )

    def test_hotrg_vs_trg_sign(self):
        """HOTRG and TRG log(Z)/N should have the same sign."""
        from tenax.algorithms.trg import TRGConfig, trg

        beta = 0.3
        tensor = compute_ising_tensor(beta=beta)

        result_hotrg = float(hotrg(tensor, HOTRGConfig(max_bond_dim=8, num_steps=6)))
        result_trg = float(trg(tensor, TRGConfig(max_bond_dim=8, num_steps=6)))

        assert (
            np.sign(result_hotrg) == np.sign(result_trg) or abs(result_hotrg) < 0.01
        ), f"HOTRG and TRG have different signs: {result_hotrg:.4f} vs {result_trg:.4f}"

    def test_hotrg_converges_with_chi(self):
        """HOTRG systematically improves with chi (unlike plain TRG's CDL
        plateau): near criticality the free-energy error at chi=24 is well below
        chi=8. Guards against a regression that stops using the larger bond dim.

        Measured (beta=0.44, 20 steps): err(chi=8) ~ 3.7e-3, err(chi=24) ~ 1.5e-3
        (ratio ~2.45).
        """
        beta = 0.44
        tensor = compute_ising_tensor(beta=beta)
        exact = ising_free_energy_exact(beta)
        err = {}
        for chi in (8, 24):
            f = float(
                -hotrg(tensor, HOTRGConfig(max_bond_dim=chi, num_steps=20)) / beta
            )
            err[chi] = abs(f - exact)
        assert err[24] < 0.8 * err[8], (
            f"HOTRG not converging in chi: err(chi=8)={err[8]:.3e} "
            f"vs err(chi=24)={err[24]:.3e}"
        )

    def test_hotrg_more_accurate_than_trg(self):
        """HOTRG should achieve better accuracy than TRG at the same chi."""
        from tenax.algorithms.trg import TRGConfig, trg

        beta = 0.3
        tensor = compute_ising_tensor(beta=beta)
        exact = ising_free_energy_exact(beta)

        f_trg = -float(trg(tensor, TRGConfig(max_bond_dim=8, num_steps=10))) / beta
        f_hotrg = (
            -float(hotrg(tensor, HOTRGConfig(max_bond_dim=8, num_steps=10))) / beta
        )

        err_trg = abs(f_trg - exact)
        err_hotrg = abs(f_hotrg - exact)
        assert err_hotrg <= err_trg + 1e-6, (
            f"HOTRG should be at least as accurate as TRG: "
            f"err_trg={err_trg:.6f}, err_hotrg={err_hotrg:.6f}"
        )

    def test_rejects_non_tensor(self):
        """hotrg() should reject raw arrays with TypeError."""
        raw_arr = np.random.default_rng(0).random((2, 2, 2, 2))
        config = HOTRGConfig(max_bond_dim=4, num_steps=3)
        with pytest.raises(TypeError, match="requires a Tensor"):
            hotrg(raw_arr, config)


class TestHOTRGSymmetric:
    """Tests for HOTRG with SymmetricTensor (Z₂-symmetric Ising tensor)."""

    def test_symmetric_hotrg_preserves_blocks(self):
        """HOTRG should not collapse Z₂ block sectors during coarse-graining."""
        tensor = compute_ising_tensor(beta=0.3, symmetric=True)
        initial_blocks = tensor.n_blocks
        assert initial_blocks == 8  # sanity check

        T, _ = _hotrg_step_horizontal(tensor, max_bond_dim=8)
        assert isinstance(T, SymmetricTensor)
        assert T.n_blocks >= initial_blocks, (
            f"Block count collapsed from {initial_blocks} to {T.n_blocks}"
        )

    def test_symmetric_hotrg_matches_exact(self):
        """Symmetric HOTRG at beta=0.3 should match exact free energy within 1%."""
        beta = 0.3
        tensor = compute_ising_tensor(beta=beta, symmetric=True)
        config = HOTRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = hotrg(tensor, config)
        hotrg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(hotrg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.01, (
            f"Symmetric HOTRG free energy {hotrg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )


def _compute_ising_tensor_fermionic(beta: float, J: float = 1.0) -> SymmetricTensor:
    """Build an Ising tensor with FermionParity symmetry for testing.

    Uses the same Hadamard-basis construction as compute_ising_tensor(symmetric=True)
    but wraps with FermionParity instead of ZnSymmetry(2).
    """
    spins = jnp.array([1.0, -1.0])
    Q = jnp.exp(beta * J * jnp.outer(spins, spins))
    evals, evecs = jnp.linalg.eigh(Q)
    sqrtQ = evecs @ jnp.diag(jnp.sqrt(evals)) @ evecs.T
    T = jnp.einsum("us,ds,ls,rs->udlr", sqrtQ, sqrtQ, sqrtQ, sqrtQ)

    H = jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2.0)
    T_z2 = jnp.einsum("ua,vb,wc,xd,uvwx->abcd", H, H, H, H, T)

    sym = FermionParity()
    charges = np.array([0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="up"),
        TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="down"),
        TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="right"),
    )
    return SymmetricTensor.from_dense(T_z2, indices)


class TestHOTRGFermionic:
    """Tests for HOTRG with FermionParity (fermionic Koszul signs)."""

    def test_fermionic_hotrg_preserves_blocks(self):
        """HOTRG should preserve FermionParity block sectors."""
        tensor = _compute_ising_tensor_fermionic(beta=0.3)
        initial_blocks = tensor.n_blocks
        assert initial_blocks == 8

        T, _ = _hotrg_step_horizontal(tensor, max_bond_dim=8)
        assert isinstance(T, SymmetricTensor)
        assert T.n_blocks >= initial_blocks, (
            f"Block count collapsed from {initial_blocks} to {T.n_blocks}"
        )

    def test_fermionic_hotrg_koszul_signs_active(self):
        """FermionParity HOTRG should give a finite result.

        With Koszul signs active, the coarse-grained tensor differs from the
        bosonic case but should still produce a valid free energy.
        """
        beta = 0.3
        tensor = _compute_ising_tensor_fermionic(beta=beta)
        config = HOTRGConfig(max_bond_dim=8, num_steps=10)
        result = float(hotrg(tensor, config))
        assert np.isfinite(result)
        assert abs(result) < 10, f"Fermionic HOTRG result {result} out of range"
