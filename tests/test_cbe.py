"""Tests for Controlled Bond Expansion (CBE) algorithm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax import (
    DMRGConfig,
    build_mpo_heisenberg,
    dmrg,
)
from tenax.algorithms.cbe import expand_bond
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.mps import FiniteMPS
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


def _make_left_canonical_site(chi_l, d, chi_r, key, dtype=jnp.float64):
    """Create a left-canonical DenseTensor site with 3 legs."""
    sym = U1Symmetry()
    charges_l = np.zeros(chi_l, dtype=np.int32)
    charges_d = np.zeros(d, dtype=np.int32)
    charges_r = np.zeros(chi_r, dtype=np.int32)

    data = jax.random.normal(key, (chi_l, d, chi_r), dtype=dtype)
    mat = data.reshape(chi_l * d, chi_r)
    q, _ = jnp.linalg.qr(mat)
    q = q[:, :chi_r]
    data_lc = q.reshape(chi_l, d, chi_r)

    indices = (
        TensorIndex.from_charges(sym, charges_l, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(sym, charges_d, FlowDirection.IN, label="phys"),
        TensorIndex.from_charges(sym, charges_r, FlowDirection.OUT, label="right"),
    )
    return DenseTensor(data_lc, indices)


def _make_right_tensor(chi_r, d2, chi_rr, key, dtype=jnp.float64):
    """Create a DenseTensor for the right neighbor site."""
    sym = U1Symmetry()
    charges_r = np.zeros(chi_r, dtype=np.int32)
    charges_d2 = np.zeros(d2, dtype=np.int32)
    charges_rr = np.zeros(chi_rr, dtype=np.int32)

    data = jax.random.normal(key, (chi_r, d2, chi_rr), dtype=dtype)

    indices = (
        TensorIndex.from_charges(sym, charges_r, FlowDirection.IN, label="right"),
        TensorIndex.from_charges(sym, charges_d2, FlowDirection.IN, label="phys2"),
        TensorIndex.from_charges(sym, charges_rr, FlowDirection.OUT, label="right2"),
    )
    return DenseTensor(data, indices)


def _make_random_hermitian_matvec(theta_shape, key):
    """Create a random Hermitian matvec that mixes all directions.

    Returns a matvec function f(v_flat, shape) -> result_flat where the
    linear operator is a random symmetric matrix, ensuring that the CBE
    sketch produces vectors outside the original column space.
    """
    n = 1
    for s in theta_shape:
        n *= s
    # Random symmetric matrix
    M = jax.random.normal(key, (n, n))
    M = (M + M.T) / 2.0

    def matvec(v_flat, shape):
        return M @ v_flat

    return matvec


class TestExpandedShape:
    """Test that expand_bond produces the correct output shape."""

    def test_expanded_shape(self):
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        chi_l, d, chi_r = 4, 2, 3
        d2, chi_rr = 2, 5
        k = 4

        site = _make_left_canonical_site(chi_l, d, chi_r, k1)
        right = _make_right_tensor(chi_r, d2, chi_rr, k2)
        theta_shape = (chi_l, d, d2, chi_rr)
        matvec = _make_random_hermitian_matvec(theta_shape, k3)

        result = expand_bond(
            site, matvec, right, k=k, oversampling=3, key=jax.random.PRNGKey(42)
        )
        dense = result.todense()
        assert dense.shape == (chi_l, d, chi_r + k), (
            f"Expected shape {(chi_l, d, chi_r + k)}, got {dense.shape}"
        )

    def test_expanded_shape_small_k(self):
        key = jax.random.PRNGKey(1)
        k1, k2, k3 = jax.random.split(key, 3)
        chi_l, d, chi_r = 3, 2, 2
        d2, chi_rr = 2, 3
        k = 2

        site = _make_left_canonical_site(chi_l, d, chi_r, k1)
        right = _make_right_tensor(chi_r, d2, chi_rr, k2)
        theta_shape = (chi_l, d, d2, chi_rr)
        matvec = _make_random_hermitian_matvec(theta_shape, k3)

        result = expand_bond(
            site, matvec, right, k=k, oversampling=2, key=jax.random.PRNGKey(42)
        )
        dense = result.todense()
        assert dense.shape == (chi_l, d, chi_r + k)


class TestExpansionOrthogonal:
    """Test that expansion columns are orthogonal to original site columns."""

    def test_expansion_orthogonal_to_original(self):
        key = jax.random.PRNGKey(10)
        k1, k2, k3 = jax.random.split(key, 3)
        chi_l, d, chi_r = 6, 2, 4
        d2, chi_rr = 2, 5
        k = 3

        site = _make_left_canonical_site(chi_l, d, chi_r, k1)
        right = _make_right_tensor(chi_r, d2, chi_rr, k2)
        theta_shape = (chi_l, d, d2, chi_rr)
        matvec = _make_random_hermitian_matvec(theta_shape, k3)

        result = expand_bond(
            site, matvec, right, k=k, oversampling=3, key=jax.random.PRNGKey(42)
        )
        dense = result.todense()
        A_mat = site.todense().reshape(chi_l * d, chi_r)
        new_cols = dense.reshape(chi_l * d, chi_r + k)[:, chi_r:]

        # A_mat^dagger @ new_cols should be approximately zero
        overlap = A_mat.conj().T @ new_cols
        assert jnp.allclose(overlap, 0.0, atol=1e-10), (
            f"Overlap between original and new columns too large: "
            f"{float(jnp.max(jnp.abs(overlap)))}"
        )


class TestComplexTensors:
    """Test that CBE works with complex-valued tensors."""

    def test_complex_tensors(self):
        key = jax.random.PRNGKey(20)
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        chi_l, d, chi_r = 4, 2, 3
        d2, chi_rr = 2, 4
        k = 2

        # Build complex site tensor (left-canonical)
        sym = U1Symmetry()
        charges_l = np.zeros(chi_l, dtype=np.int32)
        charges_d = np.zeros(d, dtype=np.int32)
        charges_r = np.zeros(chi_r, dtype=np.int32)

        re = jax.random.normal(k1, (chi_l, d, chi_r))
        im = jax.random.normal(k2, (chi_l, d, chi_r))
        data = re + 1j * im
        mat = data.reshape(chi_l * d, chi_r)
        q, _ = jnp.linalg.qr(mat)
        q = q[:, :chi_r]
        data_lc = q.reshape(chi_l, d, chi_r)

        indices = (
            TensorIndex.from_charges(sym, charges_l, FlowDirection.IN, label="left"),
            TensorIndex.from_charges(sym, charges_d, FlowDirection.IN, label="phys"),
            TensorIndex.from_charges(sym, charges_r, FlowDirection.OUT, label="right"),
        )
        site = DenseTensor(data_lc, indices)
        right = _make_right_tensor(chi_r, d2, chi_rr, k3, dtype=jnp.complex128)

        # Complex Hermitian matvec
        n = chi_l * d * d2 * chi_rr
        M = jax.random.normal(k4, (n, n)) + 1j * jax.random.normal(k5, (n, n))
        M = (M + M.conj().T) / 2.0

        def matvec(v_flat, shape):
            return M @ v_flat

        result = expand_bond(
            site, matvec, right, k=k, oversampling=3, key=jax.random.PRNGKey(99)
        )
        dense = result.todense()

        # Check shape
        assert dense.shape == (chi_l, d, chi_r + k)

        # Check orthogonality
        A_mat = data_lc.reshape(chi_l * d, chi_r)
        new_cols = dense.reshape(chi_l * d, chi_r + k)[:, chi_r:]
        overlap = A_mat.conj().T @ new_cols
        assert jnp.allclose(overlap, 0.0, atol=1e-10), (
            f"Complex overlap too large: {float(jnp.max(jnp.abs(overlap)))}"
        )

        # Check dtype preserved
        assert jnp.iscomplexobj(dense)


class TestEnergyImprovement:
    """Integration test: 2-site DMRG gives lower energy than 1-site."""

    def test_energy_improves_with_expansion(self):
        L = 6
        mpo = build_mpo_heisenberg(L)

        # 1-site DMRG with small bond dim
        mps = FiniteMPS.random(
            L,
            d=2,
            chi=4,
            key=jax.random.PRNGKey(0),
            symmetric=True,
            symmetry=U1Symmetry(),
            target_charge=0,
        )
        config_1site = DMRGConfig(
            max_bond_dim=4,
            num_sweeps=5,
            two_site=False,
            verbose=False,
        )
        result_1site = dmrg(mpo, mps, config_1site)
        E_1site = result_1site.energy

        # 2-site DMRG with larger bond dim (more variational freedom)
        mps2 = FiniteMPS.random(
            L,
            d=2,
            chi=4,
            key=jax.random.PRNGKey(0),
            symmetric=True,
            symmetry=U1Symmetry(),
            target_charge=0,
        )
        config_2site = DMRGConfig(
            max_bond_dim=6,
            num_sweeps=10,
            two_site=True,
            verbose=False,
        )
        result_2site = dmrg(mpo, mps2, config_2site)
        E_2site = result_2site.energy

        # 2-site should achieve lower or equal energy
        assert E_2site <= E_1site + 1e-8, (
            f"2-site energy {E_2site} should be <= 1-site energy {E_1site}"
        )


class TestExpandBondSymmetric:
    """Tests for block-sparse expand_bond_symmetric."""

    def test_symmetric_cbe_runs(self):
        """expand_bond_symmetric runs without error on Heisenberg MPS."""
        from tenax import build_mpo_heisenberg
        from tenax.algorithms.cbe import expand_bond_symmetric
        from tenax.algorithms.dmrg import (
            _build_right_environments_list,
            _build_trivial_left_env,
            _one_site_update_symmetric,
            _symmetric_ops,
        )

        L = 4
        chi = 4
        k = 2

        mpo = build_mpo_heisenberg(L)
        mps = FiniteMPS.random(
            L,
            d=2,
            chi=chi,
            key=jax.random.PRNGKey(42),
            symmetric=True,
            symmetry=U1Symmetry(),
            target_charge=0,
        )
        config = DMRGConfig(max_bond_dim=chi, num_sweeps=3, two_site=False)
        result = dmrg(mpo, mps, config)

        mps_t = [result.mps.get_tensor(i) for i in range(L)]
        mpo_t = [mpo.get_tensor(i) for i in range(L)]

        from tenax.algorithms.dmrg import _build_left_environments_list

        ops = _symmetric_ops(config)
        # Don't re-canonicalize — _right_canonicalize has issues with
        # SymmetricTensor. Use the DMRG result directly (already in
        # mixed canonical form from the last sweep).
        left_envs = _build_left_environments_list(mps_t, mpo_t, L, ops)
        right_envs = _build_right_environments_list(mps_t, mpo_t, L, ops)

        # Expand bond at site 1 (middle site, guaranteed 3-leg)
        site_i = 1
        l_env = left_envs[site_i]
        r_env = right_envs[site_i + 2] or ops.build_trivial_right_env()

        expanded = expand_bond_symmetric(
            mps_t[site_i],
            l_env,
            mpo_t[site_i],
            mpo_t[site_i + 1],
            r_env,
            mps_t[site_i + 1],
            k=k,
            key=jax.random.PRNGKey(7),
        )

        assert isinstance(expanded, SymmetricTensor)
        # Bond dimension should increase
        orig_chi_r = mps_t[site_i].indices[-1].dim
        new_chi_r = expanded.indices[-1].dim
        assert new_chi_r == orig_chi_r + k, (
            f"Expected chi_r={orig_chi_r + k}, got {new_chi_r}"
        )

    def test_symmetric_cbe_charges_correct(self):
        """Expanded bond has correct per-sector charge assignment."""
        from tenax import build_mpo_heisenberg
        from tenax.algorithms.cbe import expand_bond_symmetric
        from tenax.algorithms.dmrg import (
            _build_right_environments_list,
            _symmetric_ops,
        )

        L = 4
        chi = 4
        k = 2

        mpo = build_mpo_heisenberg(L)
        mps = FiniteMPS.random(
            L,
            d=2,
            chi=chi,
            key=jax.random.PRNGKey(42),
            symmetric=True,
            symmetry=U1Symmetry(),
            target_charge=0,
        )
        config = DMRGConfig(max_bond_dim=chi, num_sweeps=3, two_site=False)
        result = dmrg(mpo, mps, config)

        mps_t = [result.mps.get_tensor(i) for i in range(L)]
        mpo_t = [mpo.get_tensor(i) for i in range(L)]

        from tenax.algorithms.dmrg import _build_left_environments_list

        ops = _symmetric_ops(config)
        left_envs = _build_left_environments_list(mps_t, mpo_t, L, ops)
        right_envs = _build_right_environments_list(mps_t, mpo_t, L, ops)

        site_i = 1
        l_env = left_envs[site_i]
        r_env = right_envs[site_i + 2] or ops.build_trivial_right_env()

        orig_charges = np.asarray(mps_t[site_i].indices[-1].charges)

        expanded = expand_bond_symmetric(
            mps_t[site_i],
            l_env,
            mpo_t[site_i],
            mpo_t[site_i + 1],
            r_env,
            mps_t[site_i + 1],
            k=k,
            key=jax.random.PRNGKey(42),
        )

        new_charges = np.asarray(expanded.indices[-1].charges)
        # Original charges should be preserved
        np.testing.assert_array_equal(new_charges[: len(orig_charges)], orig_charges)
        # New charges should be valid (from the original charge set)
        unique_orig = set(int(q) for q in orig_charges)
        for q in new_charges[len(orig_charges) :]:
            assert int(q) in unique_orig, (
                f"New charge {q} not in original set {unique_orig}"
            )
