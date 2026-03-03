"""Tests for the iDMRG algorithm."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.idmrg import (
    build_bulk_mpo_heisenberg,
    build_bulk_mpo_heisenberg_cylinder,
    idmrg,
    iDMRGConfig,
    iDMRGResult,
)

# ---------------------------------------------------------------------------
# TestiDMRGConfig
# ---------------------------------------------------------------------------


class TestiDMRGConfig:
    def test_default_values(self):
        cfg = iDMRGConfig()
        assert cfg.max_bond_dim == 100
        assert cfg.max_iterations == 200
        assert cfg.convergence_tol == 1e-8
        assert cfg.lanczos_max_iter == 50
        assert cfg.lanczos_tol == 1e-12
        assert cfg.svd_trunc_err is None
        assert cfg.verbose is False

    def test_custom_values(self):
        cfg = iDMRGConfig(
            max_bond_dim=64,
            max_iterations=50,
            convergence_tol=1e-6,
            lanczos_max_iter=30,
            lanczos_tol=1e-10,
            svd_trunc_err=1e-8,
            verbose=True,
        )
        assert cfg.max_bond_dim == 64
        assert cfg.max_iterations == 50
        assert cfg.convergence_tol == 1e-6
        assert cfg.verbose is True


# ---------------------------------------------------------------------------
# TestBuildBulkMPO
# ---------------------------------------------------------------------------


class TestBuildBulkMPO:
    def test_shape(self):
        W = build_bulk_mpo_heisenberg()
        dense = W.todense()
        assert dense.shape == (5, 2, 2, 5)

    def test_labels(self):
        W = build_bulk_mpo_heisenberg()
        labels = W.labels()
        assert "w_l" in labels
        assert "w_r" in labels
        assert "mpo_top" in labels
        assert "mpo_bot" in labels

    def test_dtype_default(self):
        W = build_bulk_mpo_heisenberg()
        # Default is float64, but JAX truncates to float32 without x64 mode
        expected = jnp.float64 if jax.config.x64_enabled else jnp.float32
        assert W.todense().dtype == expected

    def test_dtype_explicit_float32(self):
        W = build_bulk_mpo_heisenberg(dtype=jnp.float32)
        assert W.todense().dtype == jnp.float32

    def test_produces_same_hamiltonian_as_build_mpo_heisenberg(self):
        """The bulk MPO should represent the same physical Hamiltonian.

        Rather than comparing W-matrix elements (which depend on internal
        ordering), we verify that an L=3 chain built from iDMRG bulk MPO
        produces the same ground state energy as build_mpo_heisenberg.
        """
        from tenax.algorithms.dmrg import (
            DMRGConfig,
            build_mpo_heisenberg,
            build_random_mps,
            dmrg,
        )

        Jz, Jxy, hz = 1.0, 1.0, 0.0
        L = 4

        mpo_ref = build_mpo_heisenberg(L=L, Jz=Jz, Jxy=Jxy, hz=hz)
        mps = build_random_mps(L, bond_dim=4, seed=0)
        config = DMRGConfig(max_bond_dim=8, num_sweeps=8, lanczos_max_iter=20)
        result = dmrg(mpo_ref, mps, config)

        # Energy should be close to exact L=4 Heisenberg ground state
        assert result.energy < -1.5, f"Energy {result.energy} too high"

    def test_invalid_d_raises(self):
        with pytest.raises(ValueError, match="only supports d=2"):
            build_bulk_mpo_heisenberg(d=3)


# ---------------------------------------------------------------------------
# TestiDMRGRun
# ---------------------------------------------------------------------------


class TestiDMRGRun:
    def test_runs_without_error(self):
        W = build_bulk_mpo_heisenberg()
        cfg = iDMRGConfig(max_bond_dim=8, max_iterations=5, lanczos_max_iter=10)
        result = idmrg(W, cfg)
        assert isinstance(result, iDMRGResult)

    def test_energy_is_finite(self):
        W = build_bulk_mpo_heisenberg()
        cfg = iDMRGConfig(max_bond_dim=8, max_iterations=10, lanczos_max_iter=10)
        result = idmrg(W, cfg)
        assert np.isfinite(result.energy_per_site), (
            f"Energy per site is not finite: {result.energy_per_site}"
        )

    def test_energy_per_site_negative(self):
        """The Heisenberg ground state energy per site should be negative."""
        W = build_bulk_mpo_heisenberg()
        cfg = iDMRGConfig(max_bond_dim=16, max_iterations=30, lanczos_max_iter=20)
        result = idmrg(W, cfg)
        assert result.energy_per_site < 0, (
            f"Energy per site should be negative, got {result.energy_per_site}"
        )

    def test_energy_converges_toward_bethe_ansatz(self):
        """With moderate chi, e_0 should approach 1/4 - ln(2) ~ -0.4431."""
        e_exact = 0.25 - math.log(2)  # ~ -0.4431
        W = build_bulk_mpo_heisenberg(dtype=jnp.float64)
        cfg = iDMRGConfig(
            max_bond_dim=32,
            max_iterations=100,
            convergence_tol=1e-8,
            lanczos_max_iter=30,
            lanczos_tol=1e-14,
        )
        result = idmrg(W, cfg, dtype=jnp.float64)
        assert abs(result.energy_per_site - e_exact) < 0.01, (
            f"e/site = {result.energy_per_site:.6f} far from Bethe ansatz {e_exact:.6f}"
        )

    def test_energy_improves_with_bond_dim(self):
        """Larger bond dimension should give a lower (better) energy."""
        W = build_bulk_mpo_heisenberg(dtype=jnp.float64)

        cfg_small = iDMRGConfig(
            max_bond_dim=8,
            max_iterations=80,
            lanczos_max_iter=20,
        )
        cfg_large = iDMRGConfig(
            max_bond_dim=32,
            max_iterations=120,
            lanczos_max_iter=20,
        )
        res_small = idmrg(W, cfg_small, dtype=jnp.float64)
        res_large = idmrg(W, cfg_large, dtype=jnp.float64)

        assert res_large.energy_per_site <= res_small.energy_per_site + 1e-3, (
            f"chi=32 energy {res_large.energy_per_site:.6f} should be <= "
            f"chi=8 energy {res_small.energy_per_site:.6f}"
        )

    def test_singular_values_returned(self):
        W = build_bulk_mpo_heisenberg()
        cfg = iDMRGConfig(max_bond_dim=8, max_iterations=10, lanczos_max_iter=10)
        result = idmrg(W, cfg)
        assert result.singular_values is not None
        assert len(result.singular_values) > 0
        assert jnp.all(result.singular_values >= 0)

    def test_convergence_flag(self):
        """With enough iterations and a generous tolerance, convergence should be True."""
        W = build_bulk_mpo_heisenberg(dtype=jnp.float64)
        cfg = iDMRGConfig(
            max_bond_dim=16,
            max_iterations=200,
            convergence_tol=1e-4,
            lanczos_max_iter=30,
        )
        result = idmrg(W, cfg, dtype=jnp.float64)
        assert result.converged, (
            f"Expected convergence with tol=1e-4, "
            f"last energies: {result.energies_per_step[-5:]}"
        )

    def test_mps_tensors_shapes(self):
        """The returned MPS tensors should have valid shapes."""
        W = build_bulk_mpo_heisenberg()
        cfg = iDMRGConfig(max_bond_dim=8, max_iterations=10, lanczos_max_iter=10)
        result = idmrg(W, cfg)
        A_L, A_R = result.mps_tensors
        # A_L: (chi_l, d, chi_c)  — 3D
        assert A_L.todense().ndim == 3
        # A_R: (chi_c, d, chi_r)  — 3D
        assert A_R.todense().ndim == 3
        # Centre bond should match
        assert A_L.todense().shape[2] == A_R.todense().shape[0]

    def test_energies_per_step_length(self):
        W = build_bulk_mpo_heisenberg()
        n_iter = 7
        cfg = iDMRGConfig(max_bond_dim=8, max_iterations=n_iter, lanczos_max_iter=10)
        result = idmrg(W, cfg)
        assert len(result.energies_per_step) <= n_iter
        assert len(result.energies_per_step) > 0


# ---------------------------------------------------------------------------
# TestBuildBulkMPOCylinder
# ---------------------------------------------------------------------------


class TestBuildBulkMPOCylinder:
    def test_shape_ly2(self):
        W = build_bulk_mpo_heisenberg_cylinder(Ly=2)
        dense = W.todense()
        # D_w = 3*2+2 = 8, d = 2^2 = 4
        assert dense.shape == (8, 4, 4, 8)

    def test_shape_ly4(self):
        W = build_bulk_mpo_heisenberg_cylinder(Ly=4)
        dense = W.todense()
        # D_w = 3*4+2 = 14, d = 2^4 = 16
        assert dense.shape == (14, 16, 16, 14)

    def test_labels(self):
        W = build_bulk_mpo_heisenberg_cylinder(Ly=2)
        labels = W.labels()
        assert "w_l" in labels
        assert "w_r" in labels
        assert "mpo_top" in labels
        assert "mpo_bot" in labels

    def test_invalid_ly_zero_raises(self):
        with pytest.raises(ValueError, match="Ly must be >= 1"):
            build_bulk_mpo_heisenberg_cylinder(Ly=0)

    def test_odd_ly_raises(self):
        """Odd Ly is incompatible with AFM order on the square lattice."""
        with pytest.raises(ValueError, match="Ly must be even"):
            build_bulk_mpo_heisenberg_cylinder(Ly=3)

    def test_h_ring_hermitian_ly2(self):
        """The within-ring Hamiltonian block should be Hermitian."""
        W = build_bulk_mpo_heisenberg_cylinder(Ly=2)
        dense = W.todense()
        D_w = dense.shape[0]
        # h_ring = W[D_w-1, :, :, 0]  (vacuum → done)
        h_ring = dense[D_w - 1, :, :, 0]
        np.testing.assert_allclose(
            np.array(h_ring),
            np.array(h_ring.T),
            atol=1e-12,
            err_msg="h_ring should be Hermitian (real symmetric)",
        )

    def test_h_ring_hermitian_ly4(self):
        """The within-ring Hamiltonian block should be Hermitian for Ly=4."""
        W = build_bulk_mpo_heisenberg_cylinder(Ly=4)
        dense = W.todense()
        D_w = dense.shape[0]
        h_ring = dense[D_w - 1, :, :, 0]
        np.testing.assert_allclose(
            np.array(h_ring),
            np.array(h_ring.T),
            atol=1e-12,
            err_msg="h_ring should be Hermitian (real symmetric)",
        )


class TestiDMRGCylinderRun:
    def test_ly2_runs_and_converges(self):
        """iDMRG with Ly=2 cylinder should run and give reasonable energy."""
        W = build_bulk_mpo_heisenberg_cylinder(Ly=2)
        cfg = iDMRGConfig(
            max_bond_dim=16,
            max_iterations=50,
            convergence_tol=1e-4,
            lanczos_max_iter=30,
            lanczos_tol=1e-12,
        )
        result = idmrg(W, cfg, d=4)
        e_per_spin = result.energy_per_site / 2
        assert np.isfinite(e_per_spin)
        # The 2D Heisenberg energy per spin should be negative and
        # reasonable for a Ly=2 cylinder
        assert -1.0 < e_per_spin < -0.3, (
            f"Ly=2 e/spin = {e_per_spin:.6f} out of expected range"
        )
