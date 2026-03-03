"""Tests for AutoMPO: automatic MPO construction."""

import numpy as np
import pytest

from tenax.algorithms.auto_mpo import (
    AutoMPO,
    HamiltonianTerm,
    _assign_bond_states,
    _build_w_matrices,
    _compute_bond_charges,
    _compute_operator_charge,
    build_auto_mpo,
    spin_half_ops,
    spin_one_ops,
)
from tenax.algorithms.dmrg import (
    DMRGConfig,
    build_mpo_heisenberg,
    build_random_mps,
    dmrg,
)
from tenax.core.tensor import DenseTensor, SymmetricTensor
from tenax.network.network import TensorNetwork

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _heisenberg_terms(L: int, Jz: float = 1.0, Jxy: float = 1.0) -> list[tuple]:
    """Return term spec list for NN Heisenberg (open BC)."""
    terms = []
    for i in range(L - 1):
        terms.append((Jz, "Sz", i, "Sz", i + 1))
        terms.append((Jxy / 2, "Sp", i, "Sm", i + 1))
        terms.append((Jxy / 2, "Sm", i, "Sp", i + 1))
    return terms


def _densify_tensor_network(tn: TensorNetwork) -> TensorNetwork:
    """Convert all tensors in a TensorNetwork to DenseTensor."""
    result = TensorNetwork()
    for nid in tn.node_ids():
        t = tn.get_tensor(nid)
        if isinstance(t, DenseTensor):
            result.add_node(nid, t)
        else:
            result.add_node(nid, DenseTensor(t.todense(), t.indices))
    return result


def _build_heisenberg_matrix(L: int, Jz: float = 1.0, Jxy: float = 1.0) -> np.ndarray:
    """Build dense (2^L, 2^L) Heisenberg Hamiltonian via Kronecker products."""
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]])
    I2 = np.eye(2)

    def kron(ops: list) -> np.ndarray:
        r = ops[0]
        for op in ops[1:]:
            r = np.kron(r, op)
        return r

    H = np.zeros((2**L, 2**L))
    for i in range(L - 1):
        zz = [I2] * L
        zz[i], zz[i + 1] = Sz, Sz
        H += Jz * kron(zz)

        pm = [I2] * L
        pm[i], pm[i + 1] = Sp, Sm
        H += (Jxy / 2) * kron(pm)

        mp = [I2] * L
        mp[i], mp[i + 1] = Sm, Sp
        H += (Jxy / 2) * kron(mp)
    return H


# ---------------------------------------------------------------------------
# TestSpinOps
# ---------------------------------------------------------------------------


class TestSpinOps:
    def test_spin_half_shapes(self):
        ops = spin_half_ops()
        for name, mat in ops.items():
            assert mat.shape == (2, 2), f"{name} should be 2x2"

    def test_spin_half_hermitian_sz(self):
        ops = spin_half_ops()
        np.testing.assert_allclose(ops["Sz"], ops["Sz"].T, atol=1e-14)

    def test_spin_half_sp_sm_adjoint(self):
        ops = spin_half_ops()
        np.testing.assert_allclose(ops["Sp"], ops["Sm"].T, atol=1e-14)

    def test_spin_half_commutation(self):
        """[Sp, Sm] = 2 Sz."""
        ops = spin_half_ops()
        comm = ops["Sp"] @ ops["Sm"] - ops["Sm"] @ ops["Sp"]
        np.testing.assert_allclose(comm, 2.0 * ops["Sz"], atol=1e-14)

    def test_spin_one_shapes(self):
        ops = spin_one_ops()
        for name, mat in ops.items():
            assert mat.shape == (3, 3), f"{name} should be 3x3"

    def test_spin_one_hermitian_sz(self):
        ops = spin_one_ops()
        np.testing.assert_allclose(ops["Sz"], ops["Sz"].T, atol=1e-14)

    def test_spin_one_sp_sm_adjoint(self):
        ops = spin_one_ops()
        np.testing.assert_allclose(ops["Sp"], ops["Sm"].T, atol=1e-14)

    def test_spin_one_commutation(self):
        """[Sp, Sm] = 2 Sz for spin-1."""
        ops = spin_one_ops()
        comm = ops["Sp"] @ ops["Sm"] - ops["Sm"] @ ops["Sp"]
        np.testing.assert_allclose(comm, 2.0 * ops["Sz"], atol=1e-12)


# ---------------------------------------------------------------------------
# TestAutoMPOStructure
# ---------------------------------------------------------------------------


class TestAutoMPOStructure:
    def test_node_count(self):
        for L in [2, 4, 6]:
            mpo = build_auto_mpo(_heisenberg_terms(L), L=L)
            assert mpo.n_nodes() == L

    def test_physical_dimension(self):
        mpo = build_auto_mpo(_heisenberg_terms(4), L=4)
        for node_id in mpo.node_ids():
            tensor = mpo.get_tensor(node_id)
            labels = tensor.labels()
            top = [lbl for lbl in labels if "mpo_top" in str(lbl)]
            if top:
                idx = tensor.indices[labels.index(top[0])]
                assert idx.dim == 2

    def test_mpo_leg_labels_exist(self):
        mpo = build_auto_mpo(_heisenberg_terms(4), L=4)
        for i in mpo.node_ids():
            tensor = mpo.get_tensor(i)
            labels = tensor.labels()
            assert any("mpo_top" in str(lbl) for lbl in labels)
            assert any("mpo_bot" in str(lbl) for lbl in labels)

    def test_l1_single_site(self):
        auto = AutoMPO(L=1)
        auto.add_term(0.5, "Sz", 0)
        mpo = auto.to_mpo()
        assert mpo.n_nodes() == 1
        # W should be shape (1, 2, 2, 1)
        tensor = mpo.get_tensor(0)
        assert tensor.todense().shape == (1, 2, 2, 1)

    def test_nn_heisenberg_bond_dim_5(self):
        """NN Heisenberg should yield bond dimension 5 (3 two-site terms + done + vac)."""
        auto = AutoMPO(L=4)
        for i in range(3):
            auto += (1.0, "Sz", i, "Sz", i + 1)
            auto += (0.5, "Sp", i, "Sm", i + 1)
            auto += (0.5, "Sm", i, "Sp", i + 1)
        dims = auto.bond_dims()
        assert all(d == 5 for d in dims), f"Expected bond dim 5 everywhere, got {dims}"

    def test_single_site_field_bond_dim_2(self):
        """Only on-site field: bond dim = 2 everywhere (done + vac only)."""
        auto = AutoMPO(L=4)
        for i in range(4):
            auto.add_term(0.5, "Sz", i)
        dims = auto.bond_dims()
        assert all(d == 2 for d in dims), f"Expected bond dim 2, got {dims}"

    def test_n_terms(self):
        auto = AutoMPO(L=4)
        for i in range(3):
            auto += (1.0, "Sz", i, "Sz", i + 1)
        assert auto.n_terms() == 3


# ---------------------------------------------------------------------------
# TestAutoMPOAlgorithm  (W-matrix correctness without full DMRG)
# ---------------------------------------------------------------------------


class TestAutoMPOAlgorithm:
    """Tests that verify the W-matrix content directly."""

    def test_single_site_term_w_content(self):
        """Single-site term h*Sz at every site: W[vac, :, :, done] = h*Sz."""
        ops = spin_half_ops()
        L = 3
        terms = [
            HamiltonianTerm(coefficient=0.5, ops=((i, ops["Sz"]),)) for i in range(L)
        ]
        bond_states = _assign_bond_states(terms, L)
        identity = np.eye(2)
        w_mats = _build_w_matrices(terms, bond_states, L, 2, identity)

        for i, W in enumerate(w_mats):
            D_l, d, _, D_r = W.shape
            vac_l = 0 if i == 0 else D_l - 1
            done_r = 0
            # Each single-site term contributes 0.5*Sz at (vac_l, done_r)
            # All 3 terms sum: total at site i = 0.5*Sz
            np.testing.assert_allclose(
                W[vac_l, :, :, done_r],
                0.5 * ops["Sz"],
                atol=1e-14,
                err_msg=f"Site {i}: single-site W entry incorrect",
            )

    def test_two_site_term_path(self):
        """Single NN term c*A_0*B_1: trace the correct path through W matrices."""
        ops = spin_half_ops()
        L = 2
        coeff = 0.7
        terms = [
            HamiltonianTerm(coefficient=coeff, ops=((0, ops["Sp"]), (1, ops["Sm"])))
        ]
        bond_states = _assign_bond_states(terms, L)
        identity = np.eye(2)
        w_mats = _build_w_matrices(terms, bond_states, L, 2, identity)

        W0, W1 = w_mats  # shapes (1,2,2,3) and (3,2,2,1)
        # State at bond 0: done=0, in-flight state 1=1, vac=2
        state = bond_states[0][0]  # = 1
        np.testing.assert_allclose(
            W0[0, :, :, state],
            coeff * ops["Sp"],
            atol=1e-14,
            err_msg="W0[vac, :, :, state] should equal coeff*Sp",
        )
        np.testing.assert_allclose(
            W1[state, :, :, 0],
            ops["Sm"],
            atol=1e-14,
            err_msg="W1[state, :, :, done] should equal Sm",
        )

    def test_identity_passthrough_bulk(self):
        """At bulk sites with no active terms, done→done and vac→vac are identity."""
        ops = spin_half_ops()
        L = 4
        # Only terms spanning bond 0 (sites 0 and 1)
        terms = [HamiltonianTerm(coefficient=1.0, ops=((0, ops["Sp"]), (1, ops["Sm"])))]
        bond_states = _assign_bond_states(terms, L)
        identity = np.eye(2)
        w_mats = _build_w_matrices(terms, bond_states, L, 2, identity)

        # Site 2: no term active; W should have done→done and vac→vac = I
        W2 = w_mats[2]
        D_l = W2.shape[0]  # = 2 (done + vac, no in-flight)
        D_r = W2.shape[3]
        np.testing.assert_allclose(W2[0, :, :, 0], identity, atol=1e-14)  # done
        np.testing.assert_allclose(
            W2[D_l - 1, :, :, D_r - 1], identity, atol=1e-14
        )  # vac

    def test_long_range_term_identity_propagation(self):
        """A term O_0 * O_3 on L=5: sites 1 and 2 should carry identity."""
        ops = spin_half_ops()
        L = 5
        coeff = 2.0
        terms = [
            HamiltonianTerm(coefficient=coeff, ops=((0, ops["Sz"]), (3, ops["Sz"])))
        ]
        bond_states = _assign_bond_states(terms, L)
        identity = np.eye(2)
        w_mats = _build_w_matrices(terms, bond_states, L, 2, identity)

        # Bond 0: term starts → bond_states[0][0] = some state
        state0 = bond_states[0][0]
        # Bond 1: same term in-flight → bond_states[1][0] = some state
        state1 = bond_states[1][0]
        state2 = bond_states[2][0]

        # W0 (left boundary): W[0, :, :, state0] = coeff * Sz
        np.testing.assert_allclose(
            w_mats[0][0, :, :, state0], coeff * ops["Sz"], atol=1e-14
        )
        # W1 (bulk): W[state0, :, :, state1] = I (identity propagation)
        np.testing.assert_allclose(
            w_mats[1][state0, :, :, state1], identity, atol=1e-14
        )
        # W2 (bulk): W[state1, :, :, state2] = I (identity propagation)
        np.testing.assert_allclose(
            w_mats[2][state1, :, :, state2], identity, atol=1e-14
        )
        # W3 (bulk): W[state2, :, :, done=0] = Sz (last op site)
        np.testing.assert_allclose(w_mats[3][state2, :, :, 0], ops["Sz"], atol=1e-14)


# ---------------------------------------------------------------------------
# TestAutoMPOReproducesHeisenberg
# ---------------------------------------------------------------------------


class TestAutoMPOReproducesHeisenberg:
    """AutoMPO Heisenberg energy should match the hand-coded build_mpo_heisenberg."""

    @pytest.mark.parametrize("L", [3, 4])
    def test_dmrg_energy_matches_hardcoded(self, L: int):
        auto_mpo = build_auto_mpo(_heisenberg_terms(L), L=L)
        ref_mpo_sym = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        # Densify ref_mpo to match DenseTensor MPS
        ref_mpo = TensorNetwork()
        for nid in ref_mpo_sym.node_ids():
            t = ref_mpo_sym.get_tensor(nid)
            if not isinstance(t, DenseTensor):
                ref_mpo.add_node(nid, DenseTensor(t.todense(), t.indices))
            else:
                ref_mpo.add_node(nid, t)

        config = DMRGConfig(
            max_bond_dim=16, num_sweeps=6, lanczos_max_iter=15, verbose=False
        )
        seed = 42
        mps_auto = build_random_mps(L, physical_dim=2, bond_dim=4, seed=seed)
        mps_ref = build_random_mps(L, physical_dim=2, bond_dim=4, seed=seed)

        res_auto = dmrg(auto_mpo, mps_auto, config)
        res_ref = dmrg(ref_mpo, mps_ref, config)

        assert np.isfinite(res_auto.energy), "AutoMPO DMRG returned non-finite energy"
        assert abs(res_auto.energy - res_ref.energy) < 0.05, (
            f"AutoMPO energy {res_auto.energy:.4f} differs from reference "
            f"{res_ref.energy:.4f} by more than 0.05"
        )

    def test_vs_exact_diag_l4(self):
        """AutoMPO DMRG energy should be close to exact ground state for L=4."""
        L = 4
        mpo = build_auto_mpo(_heisenberg_terms(L), L=L)
        mps = build_random_mps(L, physical_dim=2, bond_dim=8, seed=0)
        config = DMRGConfig(max_bond_dim=16, num_sweeps=8, lanczos_max_iter=20)
        result = dmrg(mpo, mps, config)

        H_exact = _build_heisenberg_matrix(L)
        e_exact = float(np.linalg.eigvalsh(H_exact)[0])

        assert abs(result.energy - e_exact) < 0.05, (
            f"AutoMPO DMRG energy {result.energy:.4f} far from exact {e_exact:.4f}"
        )


# ---------------------------------------------------------------------------
# TestAutoMPOSingleSite
# ---------------------------------------------------------------------------


class TestAutoMPOSingleSite:
    def test_field_only_bond_dim_2(self):
        L = 4
        auto = AutoMPO(L=L)
        for i in range(L):
            auto.add_term(0.5, "Sz", i)
        assert all(d == 2 for d in auto.bond_dims())

    def test_field_only_dmrg_runs(self):
        """DMRG with a pure magnetic-field MPO should run without error."""
        L = 4
        auto = AutoMPO(L=L)
        for i in range(L):
            auto.add_term(1.0, "Sz", i)
        mpo = auto.to_mpo()
        mps = build_random_mps(L, physical_dim=2, bond_dim=4)
        config = DMRGConfig(max_bond_dim=4, num_sweeps=2, lanczos_max_iter=5)
        result = dmrg(mpo, mps, config)
        assert np.isfinite(result.energy)

    def test_field_only_l1(self):
        auto = AutoMPO(L=1)
        auto.add_term(0.5, "Sz", 0)
        mpo = auto.to_mpo()
        assert mpo.n_nodes() == 1


# ---------------------------------------------------------------------------
# TestAutoMPONNN
# ---------------------------------------------------------------------------


class TestAutoMPONNN:
    """Next-nearest-neighbor Heisenberg: tests correctness via DMRG energy."""

    def test_nnn_mpo_bond_dim_larger_than_nn(self):
        """NNN model should have larger bond dim than NN (more in-flight terms)."""
        L = 6

        auto_nn = AutoMPO(L=L)
        auto_nnn = AutoMPO(L=L)
        for i in range(L - 1):
            auto_nn += (1.0, "Sz", i, "Sz", i + 1)
            auto_nn += (0.5, "Sp", i, "Sm", i + 1)
            auto_nn += (0.5, "Sm", i, "Sp", i + 1)
            auto_nnn += (1.0, "Sz", i, "Sz", i + 1)
            auto_nnn += (0.5, "Sp", i, "Sm", i + 1)
            auto_nnn += (0.5, "Sm", i, "Sp", i + 1)
        for i in range(L - 2):
            auto_nnn += (0.5, "Sz", i, "Sz", i + 2)
            auto_nnn += (0.25, "Sp", i, "Sm", i + 2)
            auto_nnn += (0.25, "Sm", i, "Sp", i + 2)

        nn_dims = auto_nn.bond_dims()
        nnn_dims = auto_nnn.bond_dims()
        assert max(nnn_dims) > max(nn_dims), (
            f"NNN bond dims {nnn_dims} should exceed NN bond dims {nn_dims}"
        )

    def test_nnn_dmrg_energy_below_nn(self):
        """Adding J2 > 0 frustration raises energy; adding J2 < 0 lowers it."""
        L = 6
        nn_terms = _heisenberg_terms(L, Jz=1.0, Jxy=1.0)
        nnn_terms = list(nn_terms)
        for i in range(L - 2):
            nnn_terms += [
                (-0.5, "Sz", i, "Sz", i + 2),
                (-0.25, "Sp", i, "Sm", i + 2),
                (-0.25, "Sm", i, "Sp", i + 2),
            ]

        mpo_nn = build_auto_mpo(nn_terms, L=L)
        mpo_nnn = build_auto_mpo(nnn_terms, L=L)

        config = DMRGConfig(max_bond_dim=16, num_sweeps=4, lanczos_max_iter=10)
        mps_nn = build_random_mps(L, physical_dim=2, bond_dim=4, seed=7)
        mps_nnn = build_random_mps(L, physical_dim=2, bond_dim=4, seed=7)

        res_nn = dmrg(mpo_nn, mps_nn, config)
        res_nnn = dmrg(mpo_nnn, mps_nnn, config)

        assert res_nnn.energy < res_nn.energy, (
            f"NNN (ferromagnetic J2) energy {res_nnn.energy:.4f} should be "
            f"below NN energy {res_nn.energy:.4f}"
        )


# ---------------------------------------------------------------------------
# TestAutoMPOCustomOps
# ---------------------------------------------------------------------------


class TestAutoMPOCustomOps:
    def test_custom_operators_accepted(self):
        """User can pass a custom site_ops dictionary."""
        custom_ops = {
            "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
            "Id": np.eye(2),
        }
        auto = AutoMPO(L=3, d=2, site_ops=custom_ops)
        for i in range(2):
            auto.add_term(1.0, "Z", i, "Z", i + 1)
            auto.add_term(0.5, "X", i)
        mpo = auto.to_mpo()
        assert mpo.n_nodes() == 3

    def test_spin_one_ops_default(self):
        """AutoMPO with d=3 should default to spin_one_ops."""
        auto = AutoMPO(L=3, d=3)
        auto.add_term(1.0, "Sz", 0, "Sz", 1)
        mpo = auto.to_mpo()
        assert mpo.n_nodes() == 3
        # Physical dimension should be 3
        for node_id in mpo.node_ids():
            t = mpo.get_tensor(node_id)
            labels = t.labels()
            top = [lbl for lbl in labels if "mpo_top" in str(lbl)]
            if top:
                idx = t.indices[labels.index(top[0])]
                assert idx.dim == 3

    def test_unknown_operator_raises(self):
        auto = AutoMPO(L=3)
        with pytest.raises(KeyError, match="not in site_ops"):
            auto.add_term(1.0, "BadOp", 0)

    def test_invalid_site_raises(self):
        auto = AutoMPO(L=3)
        with pytest.raises(ValueError, match="out of range"):
            auto.add_term(1.0, "Sz", 5)

    def test_no_terms_raises(self):
        auto = AutoMPO(L=3)
        with pytest.raises(ValueError, match="No terms"):
            auto.to_mpo()

    def test_duplicate_site_raises(self):
        auto = AutoMPO(L=3)
        with pytest.raises(ValueError, match="Duplicate sites"):
            auto.add_term(1.0, "Sz", 1, "Sp", 1)

    def test_iadd_interface(self):
        auto = AutoMPO(L=4)
        auto += (1.0, "Sz", 0, "Sz", 1)
        auto += (0.5, "Sp", 1, "Sm", 2)
        assert auto.n_terms() == 2


# ---------------------------------------------------------------------------
# TestAutoMPOCompression
# ---------------------------------------------------------------------------


class TestAutoMPOCompression:
    def test_compressed_bond_dim_le_uncompressed(self):
        """SVD compression should not increase bond dimensions."""
        L = 6
        terms = _heisenberg_terms(L)
        mpo_uncompressed = build_auto_mpo(terms, L=L, compress=False)
        mpo_compressed = build_auto_mpo(terms, L=L, compress=True)

        for i in range(1, L):
            t_unc = mpo_uncompressed.get_tensor(i)
            t_cmp = mpo_compressed.get_tensor(i)
            # Left bond dimension
            d_unc = t_unc.todense().shape[0]
            d_cmp = t_cmp.todense().shape[0]
            assert d_cmp <= d_unc + 1, (
                f"Site {i}: compressed left bond {d_cmp} > uncompressed {d_unc}"
            )

    def test_compressed_dmrg_energy_close(self):
        """DMRG with SVD-compressed MPO should give energy close to uncompressed."""
        L = 4
        terms = _heisenberg_terms(L)
        mpo_unc = build_auto_mpo(terms, L=L, compress=False)
        mpo_cmp = build_auto_mpo(terms, L=L, compress=True, compress_tol=1e-10)

        config = DMRGConfig(max_bond_dim=16, num_sweeps=4, lanczos_max_iter=10)
        mps_unc = build_random_mps(L, physical_dim=2, bond_dim=4, seed=3)
        mps_cmp = build_random_mps(L, physical_dim=2, bond_dim=4, seed=3)

        res_unc = dmrg(mpo_unc, mps_unc, config)
        res_cmp = dmrg(mpo_cmp, mps_cmp, config)

        assert abs(res_unc.energy - res_cmp.energy) < 0.1, (
            f"Compressed energy {res_cmp.energy:.4f} differs too much from "
            f"uncompressed {res_unc.energy:.4f}"
        )


# ---------------------------------------------------------------------------
# TestBuildAutoMPOFunctional
# ---------------------------------------------------------------------------


class TestBuildAutoMPOFunctional:
    def test_build_auto_mpo_returns_tensor_network(self):
        from tenax.network.network import TensorNetwork

        mpo = build_auto_mpo(_heisenberg_terms(4), L=4)
        assert isinstance(mpo, TensorNetwork)

    def test_build_auto_mpo_node_count(self):
        for L in [2, 3, 5]:
            mpo = build_auto_mpo(_heisenberg_terms(L), L=L)
            assert mpo.n_nodes() == L

    def test_hamiltonian_term_dataclass(self):
        ops = spin_half_ops()
        term = HamiltonianTerm(
            coefficient=0.5,
            ops=((0, ops["Sp"]), (1, ops["Sm"])),
        )
        assert term.coefficient == 0.5
        assert len(term.ops) == 2


# ---------------------------------------------------------------------------
# TestAutoMPOSymmetric
# ---------------------------------------------------------------------------


class TestAutoMPOSymmetric:
    """Tests for symmetric (SymmetricTensor-based) MPO construction."""

    def test_symmetric_mpo_todense_matches_dense(self):
        """The dense representation of a symmetric MPO should match a dense MPO."""
        L = 4
        terms = _heisenberg_terms(L)
        mpo_dense = build_auto_mpo(terms, L=L, dtype=np.float64)
        mpo_sym = build_auto_mpo(terms, L=L, dtype=np.float64, symmetric=True)

        assert mpo_sym.n_nodes() == L
        for i in range(L):
            d_arr = mpo_dense.get_tensor(i).todense()
            s_arr = mpo_sym.get_tensor(i).todense()
            np.testing.assert_allclose(
                np.array(s_arr),
                np.array(d_arr),
                atol=1e-12,
                err_msg=f"Site {i}: symmetric todense != dense",
            )

    def test_symmetric_mpo_blocks_are_nontrivial(self):
        """Symmetric MPO tensors should have fewer elements than the full dense size."""
        L = 4
        terms = _heisenberg_terms(L)
        mpo_sym = build_auto_mpo(terms, L=L, symmetric=True)

        for i in range(L):
            tensor = mpo_sym.get_tensor(i)
            assert isinstance(tensor, SymmetricTensor), (
                f"Site {i}: expected SymmetricTensor, got {type(tensor).__name__}"
            )
            assert tensor.n_blocks > 0, f"Site {i}: no blocks"
            # The total number of stored elements should be less than the
            # full dense tensor size for at least bulk sites.
            dense = tensor.todense()
            total_dense = int(np.prod(dense.shape))
            stored = sum(int(np.prod(b.shape)) for b in tensor.blocks.values())
            if i > 0 and i < L - 1:
                assert stored < total_dense, (
                    f"Site {i}: stored ({stored}) >= dense ({total_dense})"
                )

    def test_symmetric_heisenberg_bond_charges(self):
        """Bond charge arrays should be [0, 0, 2, -2, 0] for Heisenberg."""
        L = 4
        ops = spin_half_ops()
        terms_list = []
        for i in range(L - 1):
            terms_list.append(
                HamiltonianTerm(1.0, ((i, ops["Sz"]), (i + 1, ops["Sz"])))
            )
            terms_list.append(
                HamiltonianTerm(0.5, ((i, ops["Sp"]), (i + 1, ops["Sm"])))
            )
            terms_list.append(
                HamiltonianTerm(0.5, ((i, ops["Sm"]), (i + 1, ops["Sp"])))
            )
        bond_states = _assign_bond_states(terms_list, L)
        phys_charges = np.array([1, -1], dtype=np.int32)
        bond_charges = _compute_bond_charges(terms_list, bond_states, L, phys_charges)

        # Each bond should have D=5: [done=0, Sz=0, Sp=2, Sm=-2, vac=0]
        for j, bc in enumerate(bond_charges):
            assert len(bc) == 5, f"Bond {j}: expected 5 charges, got {len(bc)}"
            assert bc[0] == 0, f"Bond {j}: done state should be charge 0"
            assert bc[-1] == 0, f"Bond {j}: vacuum state should be charge 0"
            # In-flight charges should include 0 (Sz), 2 (Sp), -2 (Sm)
            inner = sorted(bc[1:-1])
            assert inner == [-2, 0, 2], (
                f"Bond {j}: expected inner charges [-2, 0, 2], got {inner}"
            )

    def test_symmetric_dmrg_energy_matches_dense(self):
        """DMRG with a symmetric MPO should give the same energy as dense."""
        L = 4
        terms = _heisenberg_terms(L)
        mpo_dense = build_auto_mpo(terms, L=L)
        mpo_sym = build_auto_mpo(terms, L=L, symmetric=True)
        # Densify the symmetric MPO so it can be used with dense MPS
        mpo_sym_dense = _densify_tensor_network(mpo_sym)

        config = DMRGConfig(
            max_bond_dim=16, num_sweeps=6, lanczos_max_iter=15, verbose=False
        )
        seed = 42
        mps1 = build_random_mps(L, physical_dim=2, bond_dim=4, seed=seed)
        mps2 = build_random_mps(L, physical_dim=2, bond_dim=4, seed=seed)

        res_dense = dmrg(mpo_dense, mps1, config)
        res_sym = dmrg(mpo_sym_dense, mps2, config)

        assert abs(res_sym.energy - res_dense.energy) < 0.05, (
            f"Symmetric DMRG energy {res_sym.energy:.4f} differs from "
            f"dense {res_dense.energy:.4f}"
        )

    def test_symmetric_spin_one(self):
        """Symmetric MPO should work for d=3 (spin-1)."""
        L = 3
        auto = AutoMPO(L=L, d=3)
        for i in range(L - 1):
            auto += (1.0, "Sz", i, "Sz", i + 1)
            auto += (0.5, "Sp", i, "Sm", i + 1)
            auto += (0.5, "Sm", i, "Sp", i + 1)

        mpo_dense = auto.to_mpo(dtype=np.float64)
        mpo_sym = auto.to_mpo(dtype=np.float64, symmetric=True)

        assert mpo_sym.n_nodes() == L
        for i in range(L):
            d_arr = mpo_dense.get_tensor(i).todense()
            s_arr = mpo_sym.get_tensor(i).todense()
            np.testing.assert_allclose(
                np.array(s_arr),
                np.array(d_arr),
                atol=1e-12,
                err_msg=f"Spin-1 site {i}: symmetric todense != dense",
            )

    def test_operator_charge_detection(self):
        """_compute_operator_charge should detect correct charges."""
        ops = spin_half_ops()
        phys = np.array([1, -1], dtype=np.int32)

        assert _compute_operator_charge(ops["Sz"], phys) == 0
        assert _compute_operator_charge(ops["Sp"], phys) == 2
        assert _compute_operator_charge(ops["Sm"], phys) == -2
        assert _compute_operator_charge(ops["Id"], phys) == 0

    def test_operator_charge_spin_one(self):
        """Charge detection for spin-1 operators."""
        ops = spin_one_ops()
        phys = np.array([2, 0, -2], dtype=np.int32)

        assert _compute_operator_charge(ops["Sz"], phys) == 0
        assert _compute_operator_charge(ops["Sp"], phys) == 2
        assert _compute_operator_charge(ops["Sm"], phys) == -2
        assert _compute_operator_charge(ops["Id"], phys) == 0
