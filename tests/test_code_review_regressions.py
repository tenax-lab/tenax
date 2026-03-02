"""Regression tests for code review findings (P1/P2)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.contraction.contractor import truncated_svd
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor
from tenax.network.network import TensorNetwork

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _make_tensor(shape, labels, flows=None, seed=0):
    """Create a DenseTensor with trivial U(1) charges."""
    u1 = U1Symmetry()
    if flows is None:
        flows = [FlowDirection.IN] * len(shape)
    charges = [np.zeros(s, dtype=np.int32) for s in shape]
    indices = tuple(
        TensorIndex(u1, charges[i], flows[i], label=labels[i])
        for i in range(len(shape))
    )
    data = jax.random.normal(jax.random.PRNGKey(seed), shape)
    return DenseTensor(data, indices)


def _build_heisenberg_matrix(L, Jz=1.0, Jxy=1.0):
    """Build exact Heisenberg Hamiltonian matrix for validation."""
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]])
    I2 = np.eye(2)

    def kron_product(ops):
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


# ------------------------------------------------------------------ #
# P1: one-site DMRG right environment index                           #
# ------------------------------------------------------------------ #


class TestOneSiteDMRG:
    """P1 regression: one-site DMRG must not crash and must produce
    finite energy close to the exact ground state."""

    def test_one_site_dmrg_runs_without_error(self):
        """1-site DMRG on L=6 Heisenberg chain should run end-to-end."""
        from tenax.algorithms.dmrg import (
            DMRGConfig,
            DMRGResult,
            build_mpo_heisenberg,
            build_random_mps,
            dmrg,
        )

        L = 6
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        mps = build_random_mps(L, physical_dim=2, bond_dim=8, seed=42)
        config = DMRGConfig(
            max_bond_dim=8,
            num_sweeps=4,
            two_site=False,
            lanczos_max_iter=20,
        )
        result = dmrg(mpo, mps, config)
        assert isinstance(result, DMRGResult)
        assert np.isfinite(result.energy)

    def test_one_site_dmrg_energy_reasonable(self):
        """1-site DMRG energy on L=4 should be in the right ballpark."""
        from tenax.algorithms.dmrg import (
            DMRGConfig,
            build_mpo_heisenberg,
            build_random_mps,
            dmrg,
        )

        L = 4
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        mps = build_random_mps(L, physical_dim=2, bond_dim=4, seed=7)
        config = DMRGConfig(
            max_bond_dim=4,
            num_sweeps=8,
            two_site=False,
            lanczos_max_iter=20,
        )
        result = dmrg(mpo, mps, config)

        e_exact = float(np.linalg.eigvalsh(_build_heisenberg_matrix(L))[0])
        assert result.energy < 0.0
        # 1-site DMRG conserves bond dim, so may not converge as well as 2-site,
        # but energy should still be in a reasonable range (within 1.0 of exact)
        assert abs(result.energy - e_exact) < 1.0, (
            f"1-site DMRG energy {result.energy:.4f} too far from exact {e_exact:.4f}"
        )


# ------------------------------------------------------------------ #
# P1: TensorNetwork.connect() relabel collision                        #
# ------------------------------------------------------------------ #


class TestConnectRelabelCollision:
    """P1 regression: connecting A['a'] to B['u'] when B already has
    its own 'a' leg must contract the intended pair, not B's 'a'."""

    def test_explicit_edge_overrides_label_matching(self):
        """connect(A, 'a', B, 'u') must contract A's 'a' with B's 'u',
        not with B's pre-existing 'a' leg."""
        u1 = U1Symmetry()
        charges = np.zeros(3, dtype=np.int32)

        # A is a 1D vector with leg 'a'
        A = DenseTensor(
            jnp.array([1.0, 2.0, 3.0]),
            (TensorIndex(u1, charges, FlowDirection.IN, label="a"),),
        )
        # B is a 2D matrix with legs ('a', 'u') — note 'a' already exists!
        # B has asymmetric data so contracting over 'a' vs 'u' gives
        # different results.
        B = DenseTensor(
            jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]),
            (
                TensorIndex(u1, charges, FlowDirection.IN, label="a"),
                TensorIndex(u1, charges, FlowDirection.OUT, label="u"),
            ),
        )

        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)
        # We want to contract A's 'a' with B's 'u' (not B's 'a')
        tn.connect("A", "a", "B", "u")

        result = tn.contract()

        # Intended: result[a_B] = sum_k A[k] * B[a_B, k]
        # = [1*10+2*20+3*30, 1*40+2*50+3*60, 1*70+2*80+3*90] = [140, 320, 500]
        A_data = np.array(A.todense())
        B_data = np.array(B.todense())
        expected = np.einsum("k,ak->a", A_data, B_data)

        # Wrong (if A['a'] contracts with B['a']): result[u] = sum_k A[k]*B[k,u]
        # = [300, 360, 420] — different from [140, 320, 500]
        wrong = np.einsum("k,ku->u", A_data, B_data)
        assert not np.allclose(expected, wrong), (
            "Test setup: correct and wrong must differ"
        )

        np.testing.assert_allclose(
            result.todense(),
            expected,
            rtol=1e-5,
            err_msg="connect() contracted the wrong pair of legs",
        )

    def test_both_labels_present_on_both_tensors(self):
        """When both tensors have labels 'a' and 'b', connecting
        A['a'] to B['b'] must pair them correctly."""
        u1 = U1Symmetry()
        charges = np.zeros(2, dtype=np.int32)

        A = DenseTensor(
            jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            (
                TensorIndex(u1, charges, FlowDirection.IN, label="a"),
                TensorIndex(u1, charges, FlowDirection.OUT, label="b"),
            ),
        )
        B = DenseTensor(
            jnp.array([[5.0, 6.0], [7.0, 8.0]]),
            (
                TensorIndex(u1, charges, FlowDirection.IN, label="a"),
                TensorIndex(u1, charges, FlowDirection.OUT, label="b"),
            ),
        )

        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)
        tn.connect("A", "a", "B", "b")

        result = tn.contract()
        # result_{b_A, a_B} = sum_k A[k, b_A] * B[a_B, k]
        A_data = np.array(A.todense())
        B_data = np.array(B.todense())
        expected = np.einsum("kb,ak->ba", A_data, B_data)

        np.testing.assert_allclose(
            result.todense(),
            expected,
            rtol=1e-5,
            err_msg="Contraction with overlapping label names gave wrong result",
        )


# ------------------------------------------------------------------ #
# P2: HOTRG invalid direction_order                                    #
# ------------------------------------------------------------------ #


class TestHOTRGDirectionValidation:
    """P2 regression: invalid direction_order must raise ValueError,
    not silently fall through to vertical mode."""

    def test_typo_in_direction_order_raises(self):
        from tenax.algorithms.hotrg import HOTRGConfig, hotrg
        from tenax.algorithms.trg import compute_ising_tensor

        tensor = compute_ising_tensor(beta=0.3)
        config = HOTRGConfig(max_bond_dim=4, num_steps=2, direction_order="alteranting")
        with pytest.raises(ValueError, match="Invalid direction_order"):
            hotrg(tensor, config)

    def test_empty_direction_order_raises(self):
        from tenax.algorithms.hotrg import HOTRGConfig, hotrg
        from tenax.algorithms.trg import compute_ising_tensor

        tensor = compute_ising_tensor(beta=0.3)
        config = HOTRGConfig(max_bond_dim=4, num_steps=2, direction_order="")
        with pytest.raises(ValueError, match="Invalid direction_order"):
            hotrg(tensor, config)


# ------------------------------------------------------------------ #
# P2: HOTRG svd_trunc_err now wired through                           #
# ------------------------------------------------------------------ #


class TestHOTRGSvdTruncErr:
    """P2 regression: svd_trunc_err should affect HOTRG output."""

    def test_svd_trunc_err_reduces_bond_dim(self):
        from tenax.algorithms.hotrg import _hotrg_step_horizontal

        T = np.random.default_rng(0).random((4, 4, 4, 4)).astype(np.float64)
        T_no_trunc, _ = _hotrg_step_horizontal(T, max_bond_dim=16)
        T_trunc, _ = _hotrg_step_horizontal(T, max_bond_dim=16, svd_trunc_err=0.01)
        # With truncation error, bond dim should be <= without
        assert T_trunc.shape[0] <= T_no_trunc.shape[0]


# ------------------------------------------------------------------ #
# P2: truncated_svd off-by-one                                        #
# ------------------------------------------------------------------ #


class TestTruncatedSvdOffByOne:
    """P2 regression: truncated_svd with max_truncation_err should not
    keep an extra singular value beyond what the error bound requires."""

    def test_spectrum_10_1_1_1_err_0p2(self):
        """Singular spectrum [10, 1, 1, 1]: keeping 1 value gives
        relative trunc error = sqrt(3)/sqrt(103) ≈ 0.17 < 0.2.
        So max_truncation_err=0.2 should keep exactly 1 value."""
        u1 = U1Symmetry()
        charges_4 = np.zeros(4, dtype=np.int32)

        # Build a 4x4 matrix with known singular values [10, 1, 1, 1]
        U_rand, _ = np.linalg.qr(np.random.default_rng(0).random((4, 4)))
        V_rand, _ = np.linalg.qr(np.random.default_rng(1).random((4, 4)))
        S = np.diag([10.0, 1.0, 1.0, 1.0])
        M = U_rand @ S @ V_rand

        tensor = DenseTensor(
            jnp.array(M),
            (
                TensorIndex(u1, charges_4, FlowDirection.IN, label="left"),
                TensorIndex(u1, charges_4, FlowDirection.OUT, label="right"),
            ),
        )

        U_t, s, Vh_t, s_full = truncated_svd(
            tensor,
            left_labels=["left"],
            right_labels=["right"],
            new_bond_label="bond",
            max_truncation_err=0.2,
        )

        # Relative truncation error with 1 kept = sqrt(3/103) ≈ 0.171 < 0.2
        # So keeping 1 already satisfies the bound.
        assert len(s) == 1, (
            f"Expected 1 singular value kept (trunc err 0.171 < 0.2), but got {len(s)}"
        )

    def test_strict_threshold_keeps_all(self):
        """With max_truncation_err=0 (no tolerance), all values should be kept."""
        u1 = U1Symmetry()
        charges_3 = np.zeros(3, dtype=np.int32)

        M = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.1]])
        tensor = DenseTensor(
            M,
            (
                TensorIndex(u1, charges_3, FlowDirection.IN, label="left"),
                TensorIndex(u1, charges_3, FlowDirection.OUT, label="right"),
            ),
        )

        _, s, _, _ = truncated_svd(
            tensor,
            left_labels=["left"],
            right_labels=["right"],
            new_bond_label="bond",
            max_truncation_err=0.0,
        )
        assert len(s) == 3


# ------------------------------------------------------------------ #
# P2: DMRG site >= 1000 label ceiling                                 #
# ------------------------------------------------------------------ #


class TestDMRGSiteLabelCeiling:
    """P2 regression: _svd_and_truncate_site must handle site >= 1000."""

    def test_svd_at_site_1000(self):
        """SVD label construction should work for site index 1000."""
        from tenax.algorithms.dmrg import DMRGConfig, _svd_and_truncate_site

        u1 = U1Symmetry()
        site = 1000
        # Build a small 2-site tensor theta with labels matching site=1000
        # left_virt = "v999_1000", left_phys = "p1000",
        # right_phys = "p1001", right_virt = "v1001_1002"
        charges_d = np.zeros(2, dtype=np.int32)
        charges_chi = np.zeros(3, dtype=np.int32)

        theta_data = jax.random.normal(jax.random.PRNGKey(0), (3, 2, 2, 3))
        theta = DenseTensor(
            theta_data,
            (
                TensorIndex(u1, charges_chi, FlowDirection.IN, label="v999_1000"),
                TensorIndex(u1, charges_d, FlowDirection.IN, label="p1000"),
                TensorIndex(u1, charges_d, FlowDirection.IN, label="p1001"),
                TensorIndex(u1, charges_chi, FlowDirection.OUT, label="v1001_1002"),
            ),
        )

        config = DMRGConfig(max_bond_dim=4)
        A, s, B, trunc_err = _svd_and_truncate_site(theta, site, config)
        assert np.isfinite(float(jnp.sum(A.todense())))
        assert np.isfinite(float(jnp.sum(B.todense())))
        assert len(s) > 0
