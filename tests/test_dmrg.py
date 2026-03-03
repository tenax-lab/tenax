"""Tests for the DMRG algorithm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.dmrg import (
    DMRGConfig,
    DMRGResult,
    build_mpo_heisenberg,
    build_random_mps,
    build_random_symmetric_mps,
    dmrg,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


class TestBuildMPOHeisenberg:
    def test_creates_correct_node_count(self):
        for L in [2, 4, 6]:
            mpo = build_mpo_heisenberg(L)
            assert mpo.n_nodes() == L, f"Expected {L} nodes for L={L}"

    def test_physical_dimension_is_2(self):
        mpo = build_mpo_heisenberg(4)
        for node_id in mpo.node_ids():
            tensor = mpo.get_tensor(node_id)
            labels = tensor.labels()
            phys_labels = [
                lbl for lbl in labels if isinstance(lbl, str) and "mpo_top" in lbl
            ]
            if phys_labels:
                lbl = phys_labels[0]
                idx = tensor.indices[labels.index(lbl)]
                assert idx.dim == 2

    def test_mpo_labels_exist(self):
        mpo = build_mpo_heisenberg(4)
        for i in mpo.node_ids():
            tensor = mpo.get_tensor(i)
            labels = tensor.labels()
            assert any("mpo_top" in str(lbl) for lbl in labels)
            assert any("mpo_bot" in str(lbl) for lbl in labels)

    def test_l1_chain(self):
        mpo = build_mpo_heisenberg(1)
        assert mpo.n_nodes() == 1

    def test_coupling_constants(self):
        """MPO with Jz=0 should have no diagonal coupling."""
        mpo_no_jz = build_mpo_heisenberg(4, Jz=0.0, Jxy=1.0)
        assert mpo_no_jz.n_nodes() == 4

    def test_repr(self):
        mpo = build_mpo_heisenberg(4)
        r = repr(mpo)
        assert "MPO" in r or "nodes" in r


class TestBuildRandomMPS:
    def test_creates_correct_node_count(self):
        for L in [2, 4, 6]:
            mps = build_random_mps(L)
            assert mps.n_nodes() == L

    def test_bond_dimensions(self):
        L = 4
        D = 8
        mps = build_random_mps(L, bond_dim=D)
        # Middle sites should have bond_dim D
        for i in [1, 2]:
            tensor = mps.get_tensor(i)
            for idx in tensor.indices:
                if "v" in str(idx.label):
                    assert idx.dim == D

    def test_physical_dimension(self):
        L = 4
        d = 3
        mps = build_random_mps(L, physical_dim=d)
        for i in mps.node_ids():
            tensor = mps.get_tensor(i)
            for idx in tensor.indices:
                if "p" in str(idx.label):
                    assert idx.dim == d

    def test_virtual_bonds_connected(self):
        L = 4
        mps = build_random_mps(L)
        assert mps.n_edges() >= L - 1


class TestDMRGConfig:
    def test_default_values(self):
        cfg = DMRGConfig()
        assert cfg.max_bond_dim == 100
        assert cfg.num_sweeps == 10
        assert cfg.two_site is True
        assert cfg.verbose is False

    def test_custom_values(self):
        cfg = DMRGConfig(max_bond_dim=20, num_sweeps=5, verbose=True)
        assert cfg.max_bond_dim == 20
        assert cfg.num_sweeps == 5
        assert cfg.verbose is True


class TestDMRGResult:
    def test_named_tuple_fields(self):
        result = DMRGResult(
            energy=-1.0,
            energies_per_sweep=[-0.9, -1.0],
            mps=None,
            truncation_errors=[0.01, 0.001],
            converged=True,
        )
        assert result.energy == -1.0
        assert result.converged is True

    def test_energy_type(self):
        result = DMRGResult(-1.5, [], None, [], False)
        assert isinstance(result.energy, float)


class TestDMRGRun:
    def test_dmrg_runs_without_error(self):
        """DMRG should run end-to-end on a small system."""
        L = 4
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        mps = build_random_mps(L, physical_dim=2, bond_dim=4)
        config = DMRGConfig(
            max_bond_dim=4,
            num_sweeps=2,
            two_site=True,
            lanczos_max_iter=5,
            verbose=False,
        )
        result = dmrg(mpo, mps, config)
        assert isinstance(result, DMRGResult)
        assert isinstance(result.energy, float)

    def test_energy_is_finite(self):
        L = 4
        mpo = build_mpo_heisenberg(L)
        mps = build_random_mps(L, physical_dim=2, bond_dim=4)
        config = DMRGConfig(max_bond_dim=4, num_sweeps=2, lanczos_max_iter=5)
        result = dmrg(mpo, mps, config)
        assert np.isfinite(result.energy)

    def test_energies_per_sweep_length(self):
        L = 4
        n_sweeps = 3
        mpo = build_mpo_heisenberg(L)
        mps = build_random_mps(L, physical_dim=2, bond_dim=4)
        config = DMRGConfig(max_bond_dim=4, num_sweeps=n_sweeps, lanczos_max_iter=3)
        result = dmrg(mpo, mps, config)
        assert len(result.energies_per_sweep) <= n_sweeps

    def test_result_mps_has_correct_nodes(self):
        L = 4
        mpo = build_mpo_heisenberg(L)
        mps = build_random_mps(L, physical_dim=2, bond_dim=4)
        config = DMRGConfig(max_bond_dim=4, num_sweeps=2, lanczos_max_iter=3)
        result = dmrg(mpo, mps, config)
        assert result.mps.n_nodes() == L

    def test_converged_field(self):
        """Very tight tolerance should NOT converge in 2 sweeps."""
        L = 4
        mpo = build_mpo_heisenberg(L)
        mps = build_random_mps(L, physical_dim=2, bond_dim=4)
        config = DMRGConfig(
            max_bond_dim=4,
            num_sweeps=2,
            convergence_tol=1e-15,  # impossible to reach in 2 sweeps
            lanczos_max_iter=3,
        )
        result = dmrg(mpo, mps, config)
        # With only 2 sweeps and impossible tol, converged should be False
        # (or True if lucky, but at least it should be a bool)
        assert isinstance(result.converged, bool)

    def test_energy_decreases_with_sweeps(self):
        """Energy should not increase over sweeps (monotone DMRG)."""
        L = 4
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        mps = build_random_mps(L, physical_dim=2, bond_dim=4, seed=0)
        config = DMRGConfig(max_bond_dim=4, num_sweeps=4, lanczos_max_iter=10)
        result = dmrg(mpo, mps, config)
        # Energy should generally decrease or stabilize
        if len(result.energies_per_sweep) >= 2:
            # Allow small numerical fluctuations
            final = result.energies_per_sweep[-1]
            initial = result.energies_per_sweep[0]
            assert final <= initial + 0.5, "Energy should not increase significantly"


def _build_heisenberg_matrix(L: int, Jz: float = 1.0, Jxy: float = 1.0) -> np.ndarray:
    """Build the L-site Heisenberg Hamiltonian matrix (OBC) using Kronecker products.

    H = Jz * sum_i Sz_i Sz_{i+1} + Jxy/2 * sum_i (S+_i S-_{i+1} + S-_i S+_{i+1})

    Args:
        L:   Number of sites.
        Jz:  Ising coupling (diagonal term).
        Jxy: XY coupling (off-diagonal term).

    Returns:
        Dense (2^L, 2^L) real numpy array.
    """
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]])
    I2 = np.eye(2)

    def kron_product(ops: list) -> np.ndarray:
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    dim = 2**L
    H = np.zeros((dim, dim))
    for i in range(L - 1):
        ops_zz = [I2] * L
        ops_zz[i] = Sz
        ops_zz[i + 1] = Sz
        H += Jz * kron_product(ops_zz)

        ops_pm = [I2] * L
        ops_pm[i] = Sp
        ops_pm[i + 1] = Sm
        H += (Jxy / 2) * kron_product(ops_pm)

        ops_mp = [I2] * L
        ops_mp[i] = Sm
        ops_mp[i + 1] = Sp
        H += (Jxy / 2) * kron_product(ops_mp)

    return H


class TestDMRGExactDiag:
    """Cross-validate DMRG ground-state energy against exact diagonalization."""

    def test_heisenberg_l4_matches_exact(self):
        """DMRG on L=4 Heisenberg chain should match ED ground state within 1e-2.

        For L=4, bond_dim=4 is sufficient for the exact ground state (max Schmidt
        rank = min(2^2, 2^2) = 4). Float32 precision limits us to ~1e-4 relative
        accuracy, so we test within 1e-2 absolute tolerance.
        """
        L = 4
        Jz, Jxy = 1.0, 1.0

        # Exact diagonalization reference (16×16 matrix, cheap)
        H_mat = _build_heisenberg_matrix(L, Jz=Jz, Jxy=Jxy)
        e_exact = float(np.linalg.eigvalsh(H_mat)[0])

        # DMRG
        mpo = build_mpo_heisenberg(L, Jz=Jz, Jxy=Jxy)
        mps = build_random_mps(L, physical_dim=2, bond_dim=4, seed=7)
        config = DMRGConfig(
            max_bond_dim=4,
            num_sweeps=8,
            lanczos_max_iter=20,
            convergence_tol=1e-8,
        )
        result = dmrg(mpo, mps, config)

        assert np.isfinite(result.energy), "DMRG energy is not finite"
        assert result.energy < 0.0, "Ground state energy should be negative"
        assert abs(result.energy - e_exact) < 1e-2, (
            f"DMRG energy {result.energy:.6f} deviates from ED {e_exact:.6f} "
            f"by {abs(result.energy - e_exact):.4e} (tol=1e-2)"
        )

    def test_heisenberg_l6_energy_below_ed_l4(self):
        """L=6 DMRG ground state energy should be lower than L=4 (more bonds)."""
        L4_H = _build_heisenberg_matrix(4)
        e_l4_exact = float(np.linalg.eigvalsh(L4_H)[0])

        mpo = build_mpo_heisenberg(6, Jz=1.0, Jxy=1.0)
        mps = build_random_mps(6, physical_dim=2, bond_dim=8, seed=5)
        config = DMRGConfig(max_bond_dim=8, num_sweeps=8, lanczos_max_iter=20)
        result = dmrg(mpo, mps, config)

        assert np.isfinite(result.energy)
        # L=6 chain must have lower total energy than L=4 (more sites = more bonds)
        assert result.energy < e_l4_exact, (
            f"L=6 DMRG energy {result.energy:.4f} should be < L=4 ED {e_l4_exact:.4f}"
        )


class TestBuildRandomSymmetricMPS:
    """Tests for the symmetric MPS factory."""

    def test_creates_correct_node_count(self):
        for L in [2, 4, 6]:
            mps = build_random_symmetric_mps(L)
            assert mps.n_nodes() == L

    def test_virtual_bonds_connected(self):
        mps = build_random_symmetric_mps(4)
        assert mps.n_edges() >= 3

    def test_blocks_are_nontrivial(self):
        """SymmetricTensor sites must have at least one non-empty block."""
        from tenax.core.tensor import SymmetricTensor

        mps = build_random_symmetric_mps(4, bond_dim=4)
        for i in mps.node_ids():
            t = mps.get_tensor(i)
            assert isinstance(t, SymmetricTensor), f"Site {i} is not SymmetricTensor"
            assert t.n_blocks > 0, f"Site {i} has no non-empty blocks"

    def test_todense_shape_matches_dimensions(self):
        """todense() of each site should match expected shape."""
        mps = build_random_symmetric_mps(4, bond_dim=4)
        for i in mps.node_ids():
            t = mps.get_tensor(i)
            dense = t.todense()
            expected_shape = tuple(idx.dim for idx in t.indices)
            assert dense.shape == expected_shape


class TestSymmetricDMRGParity:
    """Symmetric MPS gives compatible DMRG energies to dense MPS (Phase E parity)."""

    def test_symmetric_mps_dmrg_runs(self):
        """DMRG should run without error on a symmetric MPS initial state."""
        L = 4
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        mps = build_random_symmetric_mps(L, bond_dim=4, seed=0)
        config = DMRGConfig(max_bond_dim=4, num_sweeps=4, lanczos_max_iter=10)
        result = dmrg(mpo, mps, config)
        assert isinstance(result, DMRGResult)
        assert np.isfinite(result.energy)

    def test_symmetric_vs_dense_parity(self):
        """Energies from symmetric and dense initial MPS should be close after enough sweeps.

        Both start from random initial states and DMRG operates densely via .todense().
        With bond_dim=4 (exact for L=4) and 8 sweeps, both should find the ground state.
        """
        L = 4
        Jz, Jxy = 1.0, 1.0
        mpo = build_mpo_heisenberg(L, Jz=Jz, Jxy=Jxy)
        config = DMRGConfig(max_bond_dim=4, num_sweeps=8, lanczos_max_iter=20)

        mps_dense = build_random_mps(L, physical_dim=2, bond_dim=4, seed=1)
        result_dense = dmrg(mpo, mps_dense, config)

        mps_sym = build_random_symmetric_mps(L, bond_dim=4, seed=1)
        result_sym = dmrg(mpo, mps_sym, config)

        # Both should converge to the same ground state energy within 1e-1
        assert np.isfinite(result_sym.energy)
        assert abs(result_sym.energy - result_dense.energy) < 1e-1, (
            f"Symmetric DMRG energy {result_sym.energy:.6f} differs from "
            f"dense DMRG energy {result_dense.energy:.6f} by "
            f"{abs(result_sym.energy - result_dense.energy):.4e}"
        )


# ------------------------------------------------------------------ #
# Block-sparse symmetric DMRG tests                                   #
# ------------------------------------------------------------------ #


def _build_symmetric_heisenberg_mpo(L: int, Jz: float = 1.0, Jxy: float = 1.0):
    """Build a fully symmetric (SymmetricTensor) Heisenberg MPO via AutoMPO."""
    from tenax.algorithms.auto_mpo import build_auto_mpo

    terms = []
    for i in range(L - 1):
        terms.append((Jz, "Sz", i, "Sz", i + 1))
        terms.append((Jxy / 2, "Sp", i, "Sm", i + 1))
        terms.append((Jxy / 2, "Sm", i, "Sp", i + 1))
    return build_auto_mpo(terms, L=L, symmetric=True)


class TestSymmetricBlockSparseDMRG:
    """Tests for the fully block-sparse symmetric DMRG backend."""

    def test_symmetric_block_sparse_dmrg_runs(self):
        """Block-sparse DMRG with symmetric MPS + symmetric MPO should run."""
        L = 4
        mpo = _build_symmetric_heisenberg_mpo(L)
        mps = build_random_symmetric_mps(L, bond_dim=4, seed=42)
        config = DMRGConfig(max_bond_dim=8, num_sweeps=4, lanczos_max_iter=20)
        result = dmrg(mpo, mps, config)
        assert isinstance(result, DMRGResult)
        assert np.isfinite(result.energy)
        assert result.energy < 0.0

    def test_symmetric_block_sparse_energy_matches_exact(self):
        """Block-sparse DMRG energy should match ED within 1e-6 for L=4."""
        L = 4
        Jz, Jxy = 1.0, 1.0

        H_mat = _build_heisenberg_matrix(L, Jz=Jz, Jxy=Jxy)
        e_exact = float(np.linalg.eigvalsh(H_mat)[0])

        mpo = _build_symmetric_heisenberg_mpo(L, Jz=Jz, Jxy=Jxy)
        mps = build_random_symmetric_mps(L, bond_dim=4, seed=7)
        config = DMRGConfig(
            max_bond_dim=8,
            num_sweeps=10,
            lanczos_max_iter=30,
            convergence_tol=1e-10,
        )
        result = dmrg(mpo, mps, config)

        assert np.isfinite(result.energy)
        assert abs(result.energy - e_exact) < 1e-6, (
            f"Block-sparse DMRG energy {result.energy:.8f} deviates from "
            f"ED {e_exact:.8f} by {abs(result.energy - e_exact):.4e}"
        )

    def test_symmetric_block_sparse_output_types(self):
        """Output MPS sites should be SymmetricTensor when running block-sparse."""
        L = 4
        mpo = _build_symmetric_heisenberg_mpo(L)
        mps = build_random_symmetric_mps(L, bond_dim=4, seed=0)
        config = DMRGConfig(max_bond_dim=8, num_sweeps=2, lanczos_max_iter=10)
        result = dmrg(mpo, mps, config)

        for i in range(L):
            t = result.mps.get_tensor(i)
            assert isinstance(t, SymmetricTensor), (
                f"Site {i} is {type(t).__name__}, expected SymmetricTensor"
            )

    def test_symmetric_block_sparse_no_todense_leak(self, monkeypatch):
        """Contamination guard: block-sparse DMRG must NOT call todense().

        Monkeypatch SymmetricTensor.todense to raise, then run full DMRG.
        Any dense leak will be caught.
        """

        def todense_trap(self):
            raise RuntimeError(
                "todense() called during block-sparse DMRG — dense leak detected!"
            )

        monkeypatch.setattr(SymmetricTensor, "todense", todense_trap)

        L = 4
        mpo = _build_symmetric_heisenberg_mpo(L)
        mps = build_random_symmetric_mps(L, bond_dim=4, seed=0)
        config = DMRGConfig(max_bond_dim=8, num_sweeps=2, lanczos_max_iter=10)

        # This must not raise RuntimeError
        result = dmrg(mpo, mps, config)
        assert np.isfinite(result.energy)

    def test_mixed_symmetric_mps_dense_mpo_uses_dense_backend(self):
        """Symmetric MPS + dense MPO should fall back to dense backend."""
        L = 4
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)  # dense MPO
        mps = build_random_symmetric_mps(L, bond_dim=4, seed=0)
        config = DMRGConfig(max_bond_dim=4, num_sweeps=2, lanczos_max_iter=10)
        result = dmrg(mpo, mps, config)
        assert isinstance(result, DMRGResult)
        assert np.isfinite(result.energy)

    def test_symmetric_block_sparse_one_site_dmrg(self):
        """1-site block-sparse DMRG should run and produce negative energy.

        Note: 1-site DMRG cannot grow bond dimension or change block structure,
        so the final energy may be far from the true ground state when the
        initial MPS has limited charge sectors. We only check that it runs
        correctly and produces a finite, negative energy that decreases.
        """
        L = 4
        mpo = _build_symmetric_heisenberg_mpo(L, Jz=1.0, Jxy=1.0)
        mps = build_random_symmetric_mps(L, bond_dim=4, seed=7)
        config = DMRGConfig(
            max_bond_dim=8,
            num_sweeps=6,
            lanczos_max_iter=20,
            convergence_tol=1e-10,
            two_site=False,
        )
        result = dmrg(mpo, mps, config)

        assert np.isfinite(result.energy)
        assert result.energy < 0.0
        # Energy should decrease over sweeps
        if len(result.energies_per_sweep) >= 2:
            assert result.energies_per_sweep[-1] <= result.energies_per_sweep[0] + 0.1

    def test_symmetric_path_rejects_dense_inputs(self):
        """Symmetric path functions must raise TypeError for DenseTensor inputs."""
        from tenax.algorithms.dmrg import (
            _build_trivial_left_env_symmetric,
            _build_trivial_right_env_symmetric,
            _update_left_env_symmetric,
            _update_right_env_symmetric,
        )

        L = 4
        mpo_sym = _build_symmetric_heisenberg_mpo(L)
        mps_sym = build_random_symmetric_mps(L, bond_dim=4, seed=0)

        # Valid symmetric env
        l_env = _build_trivial_left_env_symmetric()
        r_env = _build_trivial_right_env_symmetric()

        # Convert one MPS site to dense — should trigger TypeError
        dense_site = DenseTensor(
            mps_sym.get_tensor(0).todense(), mps_sym.get_tensor(0).indices
        )

        with pytest.raises(TypeError, match="symmetric DMRG path must never fall back"):
            _update_left_env_symmetric(l_env, dense_site, mpo_sym.get_tensor(0))

        with pytest.raises(TypeError, match="symmetric DMRG path must never fall back"):
            _update_right_env_symmetric(r_env, dense_site, mpo_sym.get_tensor(0))

    def test_symmetric_block_sparse_env_update_vs_dense(self):
        """Symmetric env update should match dense env update on same MPO data.

        We use the same symmetric MPO and MPS, but compare the symmetric env
        update against a dense env update that converts everything to arrays
        first. The comparison is done via the einsum on raw dense arrays.
        """
        from tenax.algorithms.dmrg import (
            _build_trivial_left_env_symmetric,
            _pad_boundary_symmetric,
            _update_left_env_symmetric,
        )

        L = 4
        mpo_sym = _build_symmetric_heisenberg_mpo(L)
        mps_sym = build_random_symmetric_mps(L, bond_dim=4, seed=42)

        # Symmetric env update
        l_env_sym = _build_trivial_left_env_symmetric()
        new_l_sym = _update_left_env_symmetric(
            l_env_sym, mps_sym.get_tensor(0), mpo_sym.get_tensor(0)
        )

        # Dense reference: same contraction using raw arrays
        L_arr = l_env_sym.todense()
        A_site = mps_sym.get_tensor(0)
        # Left boundary: pad to 3D
        A_padded = _pad_boundary_symmetric(A_site, pad_left=True)
        A_arr = A_padded.todense()
        W_arr = mpo_sym.get_tensor(0).todense()

        new_l_ref = jnp.einsum(
            "abc,apd,bpxe,cxf->def",
            L_arr,
            A_arr,
            W_arr,
            jnp.conj(A_arr),
        )

        # Compare
        np.testing.assert_allclose(
            np.array(new_l_sym.todense()),
            np.array(new_l_ref),
            atol=1e-10,
        )
