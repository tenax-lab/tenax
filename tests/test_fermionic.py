"""Tests for fermionic tensor network support."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.contraction.contractor import contract, qr_decompose, truncated_svd
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import (
    BraidingStyle,
    FermionicU1,
    FermionParity,
    ProductSymmetry,
    U1Symmetry,
    ZnSymmetry,
)
from tenax.core.tensor import SymmetricTensor, _koszul_sign

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def fp():
    return FermionParity()


@pytest.fixture
def fu1():
    return FermionicU1()


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def rng2():
    return jax.random.PRNGKey(99)


# ------------------------------------------------------------------ #
# BraidingStyle and backward compatibility                             #
# ------------------------------------------------------------------ #


class TestBraidingStyleBackwardCompat:
    def test_u1_is_bosonic(self):
        sym = U1Symmetry()
        assert sym.braiding_style == BraidingStyle.BOSONIC
        assert not sym.is_fermionic

    def test_zn_is_bosonic(self):
        sym = ZnSymmetry(2)
        assert sym.braiding_style == BraidingStyle.BOSONIC
        assert not sym.is_fermionic

    def test_bosonic_parity_all_zero(self):
        sym = U1Symmetry()
        charges = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
        np.testing.assert_array_equal(sym.parity(charges), np.zeros(5, dtype=np.int32))

    def test_bosonic_exchange_sign_always_one(self):
        sym = U1Symmetry()
        assert sym.exchange_sign(1, 1) == 1
        assert sym.exchange_sign(0, 1) == 1

    def test_bosonic_twist_phase_always_one(self):
        sym = U1Symmetry()
        assert sym.twist_phase(1) == 1.0
        assert sym.twist_phase(0) == 1.0


# ------------------------------------------------------------------ #
# FermionParity                                                        #
# ------------------------------------------------------------------ #


class TestFermionParity:
    def test_braiding_style(self, fp):
        assert fp.braiding_style == BraidingStyle.FERMIONIC
        assert fp.is_fermionic

    def test_fuse(self, fp):
        a = np.array([0, 0, 1, 1], dtype=np.int32)
        b = np.array([0, 1, 0, 1], dtype=np.int32)
        expected = np.array([0, 1, 1, 0], dtype=np.int32)
        np.testing.assert_array_equal(fp.fuse(a, b), expected)

    def test_dual_self_dual(self, fp):
        a = np.array([0, 1], dtype=np.int32)
        np.testing.assert_array_equal(fp.dual(a), a)

    def test_identity(self, fp):
        assert fp.identity() == 0

    def test_n_values(self, fp):
        assert fp.n_values() == 2

    def test_parity(self, fp):
        charges = np.array([0, 1, 2, 3], dtype=np.int32)
        expected = np.array([0, 1, 0, 1], dtype=np.int32)
        np.testing.assert_array_equal(fp.parity(charges), expected)

    def test_exchange_sign_both_odd(self, fp):
        assert fp.exchange_sign(1, 1) == -1

    def test_exchange_sign_mixed(self, fp):
        assert fp.exchange_sign(0, 1) == 1
        assert fp.exchange_sign(1, 0) == 1

    def test_exchange_sign_both_even(self, fp):
        assert fp.exchange_sign(0, 0) == 1

    def test_twist_phase(self, fp):
        assert fp.twist_phase(0) == 1.0
        assert fp.twist_phase(1) == -1.0

    def test_equality_and_hash(self):
        a = FermionParity()
        b = FermionParity()
        assert a == b
        assert hash(a) == hash(b)
        assert a != U1Symmetry()

    def test_repr(self, fp):
        assert repr(fp) == "FermionParity()"


# ------------------------------------------------------------------ #
# FermionicU1                                                          #
# ------------------------------------------------------------------ #


class TestFermionicU1:
    def test_braiding_style(self, fu1):
        assert fu1.braiding_style == BraidingStyle.FERMIONIC
        assert fu1.is_fermionic

    def test_fuse_addition(self, fu1):
        a = np.array([0, 1, -1], dtype=np.int32)
        b = np.array([1, -1, 2], dtype=np.int32)
        np.testing.assert_array_equal(fu1.fuse(a, b), a + b)

    def test_dual_negation(self, fu1):
        a = np.array([0, 1, -2], dtype=np.int32)
        np.testing.assert_array_equal(fu1.dual(a), -a)

    def test_identity(self, fu1):
        assert fu1.identity() == 0

    def test_n_values_is_none(self, fu1):
        assert fu1.n_values() is None

    def test_parity_default_grading(self, fu1):
        # Default: |q| % 2
        charges = np.array([0, 1, -1, 2, -2, 3], dtype=np.int32)
        expected = np.array([0, 1, 1, 0, 0, 1], dtype=np.int32)
        np.testing.assert_array_equal(fu1.parity(charges), expected)

    def test_exchange_sign(self, fu1):
        # Both odd parity -> -1
        assert fu1.exchange_sign(1, 1) == -1
        assert fu1.exchange_sign(1, -1) == -1
        # Mixed parity -> +1
        assert fu1.exchange_sign(0, 1) == 1
        assert fu1.exchange_sign(2, 1) == 1
        # Both even -> +1
        assert fu1.exchange_sign(0, 2) == 1

    def test_twist_phase(self, fu1):
        assert fu1.twist_phase(0) == 1.0
        assert fu1.twist_phase(1) == -1.0
        assert fu1.twist_phase(2) == 1.0

    def test_custom_grading(self):
        # mod_2 grading: q % 2
        sym = FermionicU1(grading_key="mod_2")
        # For mod_2: charge -1 -> (-1) % 2 = 1 (odd)
        assert sym.exchange_sign(-1, -1) == -1
        assert sym.exchange_sign(0, -1) == 1

    def test_equality_same_grading(self):
        a = FermionicU1(grading_key="abs_mod_2")
        b = FermionicU1()  # default is abs_mod_2
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality_different_grading(self):
        a = FermionicU1(grading_key="abs_mod_2")
        b = FermionicU1(grading_key="mod_2")
        assert a != b

    def test_inequality_with_u1(self):
        assert FermionicU1() != U1Symmetry()

    def test_repr(self, fu1):
        assert "FermionicU1" in repr(fu1)
        assert "abs_mod_2" in repr(fu1)


# ------------------------------------------------------------------ #
# ProductSymmetry                                                      #
# ------------------------------------------------------------------ #


class TestProductSymmetry:
    def test_encode_decode_roundtrip(self):
        for q1, q2 in [(0, 0), (1, 2), (-1, 3), (100, -50), (-32768, 32767)]:
            encoded = ProductSymmetry.encode(q1, q2)
            d1, d2 = ProductSymmetry.decode(encoded)
            assert d1 == q1, f"q1 mismatch: {d1} != {q1}"
            assert d2 == q2, f"q2 mismatch: {d2} != {q2}"

    def test_encode_decode_vectorized(self):
        a1 = np.array([0, 1, -1, 5], dtype=np.int32)
        a2 = np.array([3, -2, 0, 7], dtype=np.int32)
        packed = ProductSymmetry.encode_charges(a1, a2)
        d1, d2 = ProductSymmetry.decode_charges(packed)
        np.testing.assert_array_equal(d1, a1)
        np.testing.assert_array_equal(d2, a2)

    def test_dual_negates_both_components(self):
        """Verify that dual() correctly negates both components."""
        sym = ProductSymmetry(U1Symmetry(), U1Symmetry())
        for q1, q2 in [(2, 1), (-1, 3), (0, 5)]:
            packed = ProductSymmetry.encode_charges(
                np.array([q1], dtype=np.int32),
                np.array([q2], dtype=np.int32),
            )
            dualled = sym.dual(packed)
            d1, d2 = ProductSymmetry.decode_charges(dualled)
            assert int(d1[0]) == -q1, f"-q1 mismatch: {d1[0]} != {-q1}"
            assert int(d2[0]) == -q2, f"-q2 mismatch: {d2[0]} != {-q2}"

    def test_fuse(self):
        sym = ProductSymmetry(FermionParity(), U1Symmetry())
        a = ProductSymmetry.encode_charges(
            np.array([0, 1], dtype=np.int32),
            np.array([1, -1], dtype=np.int32),
        )
        b = ProductSymmetry.encode_charges(
            np.array([1, 0], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
        )
        result = sym.fuse(a, b)
        r1, r2 = ProductSymmetry.decode_charges(result)
        np.testing.assert_array_equal(
            r1, np.array([1, 1], dtype=np.int32)
        )  # (0+1)%2, (1+0)%2
        np.testing.assert_array_equal(r2, np.array([3, 2], dtype=np.int32))  # 1+2, -1+3

    def test_dual(self):
        sym = ProductSymmetry(FermionParity(), U1Symmetry())
        charges = ProductSymmetry.encode_charges(
            np.array([0, 1], dtype=np.int32),
            np.array([3, -2], dtype=np.int32),
        )
        result = sym.dual(charges)
        r1, r2 = ProductSymmetry.decode_charges(result)
        np.testing.assert_array_equal(
            r1, np.array([0, 1], dtype=np.int32)
        )  # Z2 self-dual
        np.testing.assert_array_equal(
            r2, np.array([-3, 2], dtype=np.int32)
        )  # U1 negation

    def test_identity(self):
        sym = ProductSymmetry(FermionParity(), U1Symmetry())
        identity = sym.identity()
        q1, q2 = ProductSymmetry.decode(identity)
        assert q1 == 0
        assert q2 == 0

    def test_braiding_style_fermionic(self):
        sym = ProductSymmetry(FermionParity(), U1Symmetry())
        assert sym.braiding_style == BraidingStyle.FERMIONIC
        assert sym.is_fermionic

    def test_braiding_style_bosonic(self):
        sym = ProductSymmetry(U1Symmetry(), ZnSymmetry(3))
        assert sym.braiding_style == BraidingStyle.BOSONIC
        assert not sym.is_fermionic

    def test_parity(self):
        sym = ProductSymmetry(FermionParity(), U1Symmetry())
        # FermionParity parity: 0->0, 1->1
        # U1Symmetry parity: always 0
        # Combined: just FermionParity contribution
        charges = ProductSymmetry.encode_charges(
            np.array([0, 1, 0, 1], dtype=np.int32),
            np.array([0, 0, 5, 5], dtype=np.int32),
        )
        expected = np.array([0, 1, 0, 1], dtype=np.int32)
        np.testing.assert_array_equal(sym.parity(charges), expected)

    def test_exchange_sign(self):
        sym = ProductSymmetry(FermionParity(), U1Symmetry())
        # Both odd parity (FP=1, U1 doesn't matter)
        q_odd = ProductSymmetry.encode(1, 5)
        assert sym.exchange_sign(q_odd, q_odd) == -1
        # Mixed
        q_even = ProductSymmetry.encode(0, 5)
        assert sym.exchange_sign(q_even, q_odd) == 1

    def test_no_nesting(self):
        inner = ProductSymmetry(FermionParity(), U1Symmetry())
        with pytest.raises(TypeError, match="Nested"):
            ProductSymmetry(inner, U1Symmetry())

    def test_equality(self):
        a = ProductSymmetry(FermionParity(), U1Symmetry())
        b = ProductSymmetry(FermionParity(), U1Symmetry())
        assert a == b
        assert hash(a) == hash(b)
        c = ProductSymmetry(U1Symmetry(), U1Symmetry())
        assert a != c

    def test_repr(self):
        sym = ProductSymmetry(FermionParity(), U1Symmetry())
        assert "ProductSymmetry" in repr(sym)

    def test_n_values(self):
        sym1 = ProductSymmetry(ZnSymmetry(2), ZnSymmetry(3))
        assert sym1.n_values() == 6
        sym2 = ProductSymmetry(FermionParity(), U1Symmetry())
        assert sym2.n_values() is None


# ------------------------------------------------------------------ #
# Koszul sign                                                          #
# ------------------------------------------------------------------ #


class TestKoszulSign:
    def test_identity_perm(self):
        assert _koszul_sign([1, 1, 1], (0, 1, 2)) == 1

    def test_single_swap_both_odd(self):
        # Swap positions 0 and 1, both odd parity -> -1
        assert _koszul_sign([1, 1], (1, 0)) == -1

    def test_single_swap_one_even(self):
        # Swap positions 0 and 1, but position 0 is even -> +1
        assert _koszul_sign([0, 1], (1, 0)) == 1

    def test_single_swap_both_even(self):
        assert _koszul_sign([0, 0], (1, 0)) == 1

    def test_three_elements_two_swaps(self):
        # perm (2, 0, 1): inversions are (2,0) and (2,1)
        # parities [1, 0, 1]: only (2,0) has both odd -> one sign flip
        assert _koszul_sign([1, 0, 1], (2, 0, 1)) == -1

    def test_three_elements_all_odd(self):
        # perm (2, 1, 0): inversions are (2,1), (2,0), (1,0) = 3 inversions
        # All odd parity -> (-1)^3 = -1
        assert _koszul_sign([1, 1, 1], (2, 1, 0)) == -1

    def test_four_elements(self):
        # perm (1, 0, 3, 2): inversions are (1,0) and (3,2)
        # parities [1, 1, 1, 1]: both inversions have odd pairs -> (-1)^2 = 1
        assert _koszul_sign([1, 1, 1, 1], (1, 0, 3, 2)) == 1

    def test_empty(self):
        assert _koszul_sign([], ()) == 1


# ------------------------------------------------------------------ #
# Fermionic tensor: transpose roundtrip                                #
# ------------------------------------------------------------------ #


class TestFermionicTranspose:
    def test_transpose_roundtrip(self, fp, rng):
        """transpose(perm).transpose(inv_perm) recovers original."""
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex(fp, charges, FlowDirection.IN, label="a"),
            TensorIndex(fp, charges, FlowDirection.IN, label="b"),
            TensorIndex(fp, fp.dual(charges), FlowDirection.OUT, label="c"),
        )
        t = SymmetricTensor.random_normal(indices, rng)

        perm = (2, 0, 1)
        inv_perm = (1, 2, 0)
        t2 = t.transpose(perm).transpose(inv_perm)

        for key in t.blocks:
            np.testing.assert_allclose(
                np.array(t.blocks[key]),
                np.array(t2.blocks[key]),
                atol=1e-7,
            )

    def test_transpose_roundtrip_fu1(self, fu1, rng):
        """transpose roundtrip with FermionicU1."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(fu1, charges, FlowDirection.IN, label="a"),
            TensorIndex(fu1, charges, FlowDirection.IN, label="b"),
            TensorIndex(fu1, fu1.dual(charges), FlowDirection.OUT, label="c"),
        )
        t = SymmetricTensor.random_normal(indices, rng)

        perm = (1, 2, 0)
        inv_perm = (2, 0, 1)
        t2 = t.transpose(perm).transpose(inv_perm)

        for key in t.blocks:
            np.testing.assert_allclose(
                np.array(t.blocks[key]),
                np.array(t2.blocks[key]),
                atol=1e-7,
            )

    def test_bosonic_transpose_unchanged(self, rng):
        """Bosonic symmetry transpose should NOT add any signs."""
        u1 = U1Symmetry()
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(u1, charges, FlowDirection.IN, label="b"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="c"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        perm = (2, 0, 1)

        t_transposed = t.transpose(perm)
        # Check todense matches jnp.transpose
        dense_original = t.todense()
        dense_transposed = t_transposed.todense()
        np.testing.assert_allclose(
            np.array(dense_transposed),
            np.array(jnp.transpose(dense_original, perm)),
            atol=1e-7,
        )


# ------------------------------------------------------------------ #
# Fermionic contraction: order independence                            #
# ------------------------------------------------------------------ #


class TestFermionicContraction:
    def test_contraction_fermion_parity(self, fp, rng, rng2):
        """contract(A, B) should give consistent results for FermionParity."""
        charges = np.array([0, 1], dtype=np.int32)
        indices_A = (
            TensorIndex(fp, charges, FlowDirection.IN, label="p0"),
            TensorIndex(fp, charges, FlowDirection.OUT, label="bond"),
        )
        indices_B = (
            TensorIndex(fp, charges, FlowDirection.IN, label="bond"),
            TensorIndex(fp, charges, FlowDirection.IN, label="p1"),
        )
        A = SymmetricTensor.random_normal(indices_A, rng)
        B = SymmetricTensor.random_normal(indices_B, rng2)

        result = contract(A, B)
        assert isinstance(result, SymmetricTensor)
        # Verify conservation law
        for key in result.blocks:
            net = sum(int(idx.flow) * int(q) for idx, q in zip(result.indices, key))
            assert net % 2 == fp.identity()

    def test_contraction_fermionic_u1(self, fu1, rng, rng2):
        """contract(A, B) with FermionicU1 should produce valid result."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices_A = (
            TensorIndex(fu1, charges, FlowDirection.IN, label="p0"),
            TensorIndex(fu1, charges, FlowDirection.OUT, label="bond"),
        )
        indices_B = (
            TensorIndex(fu1, charges, FlowDirection.IN, label="bond"),
            TensorIndex(fu1, charges, FlowDirection.IN, label="p1"),
        )
        A = SymmetricTensor.random_normal(indices_A, rng)
        B = SymmetricTensor.random_normal(indices_B, rng2)

        result = contract(A, B)
        assert isinstance(result, SymmetricTensor)
        # Verify conservation law
        for key in result.blocks:
            net = sum(int(idx.flow) * int(q) for idx, q in zip(result.indices, key))
            assert net == fu1.identity()


# ------------------------------------------------------------------ #
# SVD roundtrip for fermionic tensors                                  #
# ------------------------------------------------------------------ #


class TestFermionicSVD:
    def test_svd_roundtrip_fermion_parity(self, fp, rng):
        """SVD and reconstruction should preserve fermionic tensor data."""
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex(fp, charges, FlowDirection.IN, label="row"),
            TensorIndex(fp, fp.dual(charges), FlowDirection.OUT, label="col"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        original_dense = t.todense()

        U, s, Vh, _ = truncated_svd(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        recon = U.todense() * s[None, :] @ Vh.todense()
        np.testing.assert_allclose(recon, original_dense, rtol=1e-4, atol=1e-6)

    def test_svd_roundtrip_fermionic_u1(self, fu1, rng):
        """SVD roundtrip with FermionicU1."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(fu1, charges, FlowDirection.IN, label="row"),
            TensorIndex(fu1, fu1.dual(charges), FlowDirection.OUT, label="col"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        original_dense = t.todense()

        U, s, Vh, _ = truncated_svd(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        recon = U.todense() * s[None, :] @ Vh.todense()
        np.testing.assert_allclose(recon, original_dense, rtol=1e-4, atol=1e-6)

    def test_svd_3leg_fermionic_u1(self, fu1, rng):
        """SVD of a 3-leg fermionic tensor should reconstruct."""
        phys_c = np.array([-1, 0, 1], dtype=np.int32)
        virt_c = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(fu1, phys_c, FlowDirection.IN, label="phys"),
            TensorIndex(fu1, virt_c, FlowDirection.IN, label="left"),
            TensorIndex(fu1, fu1.dual(virt_c), FlowDirection.OUT, label="right"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        original_dense = t.todense()

        U, s, Vh, _ = truncated_svd(
            t,
            left_labels=["phys", "left"],
            right_labels=["right"],
            new_bond_label="bond",
        )

        U_d = U.todense()
        Vh_d = Vh.todense()
        Us = U_d * s[None, None, :]
        recon = jnp.einsum("plb,br->plr", Us, Vh_d)
        np.testing.assert_allclose(recon, original_dense, rtol=1e-4, atol=1e-6)

    def test_svd_returns_symmetric(self, fu1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(fu1, charges, FlowDirection.IN, label="row"),
            TensorIndex(fu1, fu1.dual(charges), FlowDirection.OUT, label="col"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        U, s, Vh, _ = truncated_svd(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )
        assert isinstance(U, SymmetricTensor)
        assert isinstance(Vh, SymmetricTensor)


# ------------------------------------------------------------------ #
# QR roundtrip for fermionic tensors                                   #
# ------------------------------------------------------------------ #


class TestFermionicQR:
    def test_qr_roundtrip_fermionic_u1(self, fu1, rng):
        """QR roundtrip with FermionicU1."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(fu1, charges, FlowDirection.IN, label="row"),
            TensorIndex(fu1, fu1.dual(charges), FlowDirection.OUT, label="col"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        original_dense = t.todense()

        Q, R = qr_decompose(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        recon = Q.todense() @ R.todense()
        np.testing.assert_allclose(recon, original_dense, rtol=1e-4, atol=1e-6)

    def test_qr_returns_symmetric(self, fu1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(fu1, charges, FlowDirection.IN, label="row"),
            TensorIndex(fu1, fu1.dual(charges), FlowDirection.OUT, label="col"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        Q, R = qr_decompose(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )
        assert isinstance(Q, SymmetricTensor)
        assert isinstance(R, SymmetricTensor)


# ------------------------------------------------------------------ #
# Dagger                                                               #
# ------------------------------------------------------------------ #


class TestDagger:
    def test_dagger_bosonic_is_conj_dual(self, rng):
        """For bosonic tensors, dagger should equal conj with dual indices."""
        u1 = U1Symmetry()
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="a"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="b"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        td = t.dagger()

        # dagger should conjugate blocks
        t_conj = t.conj()
        for key in t_conj.blocks:
            # Find corresponding key in td (with dual charges)
            dual_key = tuple(int(u1.dual(np.array([q]))[0]) for q in key)
            assert dual_key in td.blocks
            np.testing.assert_allclose(
                np.array(td.blocks[dual_key]),
                np.array(t_conj.blocks[key]),
                atol=1e-7,
            )

    def test_dagger_fermionic_twist(self, fu1, rng):
        """For fermionic tensors, dagger should apply twist phases."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        indices = (
            TensorIndex(fu1, charges, FlowDirection.IN, label="a"),
            TensorIndex(fu1, fu1.dual(charges), FlowDirection.OUT, label="b"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        td = t.dagger()

        # Indices should be dual
        for i, idx in enumerate(td.indices):
            assert idx.flow == FlowDirection(-int(t.indices[i].flow))

    def test_dagger_indices_are_dual(self, fp, rng):
        charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex(fp, charges, FlowDirection.IN, label="a"),
            TensorIndex(fp, fp.dual(charges), FlowDirection.OUT, label="b"),
        )
        t = SymmetricTensor.random_normal(indices, rng)
        td = t.dagger()

        assert td.indices[0].flow == FlowDirection.OUT
        assert td.indices[1].flow == FlowDirection.IN


# ------------------------------------------------------------------ #
# Fermionic site operators                                             #
# ------------------------------------------------------------------ #


class TestFermionSiteOps:
    def test_fermion_site_ops_keys(self):
        from tenax.algorithms.auto_mpo import fermion_site_ops

        ops = fermion_site_ops()
        assert set(ops.keys()) == {"C", "Cd", "N", "F", "Id"}

    def test_anticommutation_C_Cd(self):
        """C Cd + Cd C = Id (canonical anticommutation)."""
        from tenax.algorithms.auto_mpo import fermion_site_ops

        ops = fermion_site_ops()
        C, Cd, Id = ops["C"], ops["Cd"], ops["Id"]
        np.testing.assert_allclose(C @ Cd + Cd @ C, Id)

    def test_N_equals_Cd_C(self):
        from tenax.algorithms.auto_mpo import fermion_site_ops

        ops = fermion_site_ops()
        np.testing.assert_allclose(ops["N"], ops["Cd"] @ ops["C"])

    def test_F_equals_minus_one_to_N(self):
        """F = (-1)^N = Id - 2*N."""
        from tenax.algorithms.auto_mpo import fermion_site_ops

        ops = fermion_site_ops()
        expected = ops["Id"] - 2 * ops["N"]
        np.testing.assert_allclose(ops["F"], expected)


class TestAutoMPOJordanWigner:
    def test_jw_insertion_nn_hopping(self):
        """Nearest-neighbor hopping: no F needed between adjacent sites."""
        from tenax.algorithms.auto_mpo import AutoMPO, fermion_site_ops

        auto = AutoMPO(L=4, d=2, site_ops=fermion_site_ops(),
                       fermionic_ops={"C", "Cd"})
        auto.add_term(1.0, "Cd", 0, "C", 1)
        # NN: no intermediate sites, so only 1 term with 2 ops
        assert auto.n_terms() == 1
        term = auto._terms[0]
        assert len(term.ops) == 2
        sites = [s for s, _ in term.ops]
        assert sites == [0, 1]

    def test_jw_insertion_nnn_hopping(self):
        """Next-nearest-neighbor: F inserted on intermediate site 1."""
        from tenax.algorithms.auto_mpo import AutoMPO, fermion_site_ops

        auto = AutoMPO(L=4, d=2, site_ops=fermion_site_ops(),
                       fermionic_ops={"C", "Cd"})
        auto.add_term(1.0, "Cd", 0, "C", 2)
        assert auto.n_terms() == 1
        term = auto._terms[0]
        # Should have 3 ops: Cd@0, F@1, C@2
        assert len(term.ops) == 3
        sites = [s for s, _ in term.ops]
        assert sites == [0, 1, 2]
        # Middle op should be F = diag(1, -1)
        np.testing.assert_allclose(term.ops[1][1], np.diag([1.0, -1.0]))

    def test_jw_long_range(self):
        """Long-range hopping Cd@0 C@4: F inserted on sites 1, 2, 3."""
        from tenax.algorithms.auto_mpo import AutoMPO, fermion_site_ops

        auto = AutoMPO(L=6, d=2, site_ops=fermion_site_ops(),
                       fermionic_ops={"C", "Cd"})
        auto.add_term(1.0, "Cd", 0, "C", 4)
        term = auto._terms[0]
        assert len(term.ops) == 5  # Cd@0, F@1, F@2, F@3, C@4
        for i in range(1, 4):
            np.testing.assert_allclose(term.ops[i][1], np.diag([1.0, -1.0]))

    def test_no_jw_for_bosonic_ops(self):
        """Density-density N@0 N@2: no F insertion (N is not fermionic)."""
        from tenax.algorithms.auto_mpo import AutoMPO, fermion_site_ops

        auto = AutoMPO(L=4, d=2, site_ops=fermion_site_ops(),
                       fermionic_ops={"C", "Cd"})
        auto.add_term(1.0, "N", 0, "N", 2)
        term = auto._terms[0]
        assert len(term.ops) == 2  # No F inserted

    def test_jw_mixed_term(self):
        """Term with one fermionic and one bosonic op: Cd@0 N@2 needs F@1."""
        from tenax.algorithms.auto_mpo import AutoMPO, fermion_site_ops

        auto = AutoMPO(L=4, d=2, site_ops=fermion_site_ops(),
                       fermionic_ops={"C", "Cd"})
        auto.add_term(1.0, "Cd", 0, "N", 2)
        term = auto._terms[0]
        # Odd number of fermionic ops in the range => F on intermediates
        assert len(term.ops) == 3  # Cd@0, F@1, N@2

    def test_no_jw_when_not_specified(self):
        """Without fermionic_ops, no JW insertion even for C/Cd."""
        from tenax.algorithms.auto_mpo import AutoMPO, fermion_site_ops

        auto = AutoMPO(L=4, d=2, site_ops=fermion_site_ops())
        auto.add_term(1.0, "Cd", 0, "C", 2)
        term = auto._terms[0]
        assert len(term.ops) == 2  # No auto-insertion


class TestBuildAutoMPOFermionic:
    def test_build_auto_mpo_with_fermionic_ops(self):
        """build_auto_mpo should accept fermionic_ops kwarg."""
        from tenax.algorithms.auto_mpo import build_auto_mpo, fermion_site_ops

        L = 4
        ops = fermion_site_ops()
        terms = [
            (-1.0, "Cd", i, "C", i + 1) for i in range(L - 1)
        ] + [
            (-1.0, "C", i, "Cd", i + 1) for i in range(L - 1)
        ]
        mpo = build_auto_mpo(
            terms, L=L, d=2, site_ops=ops,
            fermionic_ops={"C", "Cd"},
        )
        assert mpo is not None
        assert mpo.n_nodes() == L

    def test_fermion_site_ops_importable_from_tenax(self):
        from tenax import fermion_site_ops

        ops = fermion_site_ops()
        assert "C" in ops


class TestFermionicDMRG:
    def test_free_fermion_chain_energy(self):
        """DMRG on free fermion chain H = -t sum c†_i c_{i+1} + h.c.

        Exact ground state energy for L sites, N=L/2 particles:
        E = -2t * sum_{k=1}^{N} cos(pi*k/(L+1))
        For L=6, half-filling (N=3).
        """
        from tenax.algorithms.auto_mpo import AutoMPO, fermion_site_ops
        from tenax.algorithms.dmrg import DMRGConfig, build_random_mps, dmrg

        L = 6
        t_hop = 1.0
        ops = fermion_site_ops()
        auto = AutoMPO(L=L, d=2, site_ops=ops, fermionic_ops={"C", "Cd"})
        for i in range(L - 1):
            auto += (-t_hop, "Cd", i, "C", i + 1)
            auto += (-t_hop, "C", i, "Cd", i + 1)
        mpo = auto.to_mpo()

        mps = build_random_mps(L=L, physical_dim=2, bond_dim=16, seed=0)
        config = DMRGConfig(num_sweeps=20, max_bond_dim=16, convergence_tol=1e-8)
        result = dmrg(mpo, mps, config)

        # Exact GS energy for free fermions on L=6 open chain at half filling
        # E = -2t * sum_{k=1}^{N} cos(pi*k/(L+1))
        # N = 3 (half filling), L = 6
        exact_energy = -2 * t_hop * sum(
            np.cos(np.pi * k / (L + 1)) for k in range(1, L // 2 + 1)
        )
        np.testing.assert_allclose(result.energy, exact_energy, atol=1e-6)
