"""Tests for observable computation from MPS ground states."""

import jax
import numpy as np
import pytest

from tenax.algorithms.dmrg import (
    DMRGConfig,
    build_mpo_heisenberg,
    dmrg,
)
from tenax.algorithms.observables import (
    correlation,
    expectation_value,
    operator_charge,
)
from tenax.core.mps import FiniteMPS
from tenax.core.symmetry import U1Symmetry

# Standard spin-1/2 operators
Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
Sm = np.array([[0.0, 0.0], [1.0, 0.0]])
I2 = np.eye(2)


def _build_symmetric_heisenberg_mpo(L: int, Jz: float = 1.0, Jxy: float = 1.0):
    """Build a fully symmetric Heisenberg MPO."""
    from tenax.algorithms.auto_mpo import build_auto_mpo

    terms = []
    for i in range(L - 1):
        terms.append((Jz, "Sz", i, "Sz", i + 1))
        terms.append((Jxy / 2, "Sp", i, "Sm", i + 1))
        terms.append((Jxy / 2, "Sm", i, "Sp", i + 1))
    return build_auto_mpo(terms, L=L, symmetric=True)


def _run_dmrg_sz0(L: int, bond_dim: int = 8) -> object:
    """Run DMRG on Heisenberg chain in Sz=0 sector, return result."""
    mpo = _build_symmetric_heisenberg_mpo(L)
    mps = FiniteMPS.random(
        L,
        d=2,
        chi=bond_dim,
        key=jax.random.PRNGKey(7),
        symmetric=True,
        symmetry=U1Symmetry(),
        target_charge=0,
    )
    config = DMRGConfig(
        max_bond_dim=bond_dim,
        num_sweeps=10,
        lanczos_max_iter=30,
        convergence_tol=1e-10,
        target_charge=0,
    )
    return dmrg(mpo, mps, config)


def _ed_correlator(L: int, op_i, site_i: int, op_j, site_j: int) -> float:
    """Compute <gs|O_i O_j|gs> via exact diagonalization."""

    # Build Hamiltonian
    def kron_product(ops):
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result

    dim = 2**L
    H = np.zeros((dim, dim))
    for i in range(L - 1):
        ops_zz = [I2] * L
        ops_zz[i] = Sz
        ops_zz[i + 1] = Sz
        H += kron_product(ops_zz)
        ops_pm = [I2] * L
        ops_pm[i] = Sp
        ops_pm[i + 1] = Sm
        H += 0.5 * kron_product(ops_pm)
        ops_mp = [I2] * L
        ops_mp[i] = Sm
        ops_mp[i + 1] = Sp
        H += 0.5 * kron_product(ops_mp)

    eigvals, eigvecs = np.linalg.eigh(H)
    gs = eigvecs[:, 0]

    # Build O_i O_j operator
    if site_i == site_j:
        ops = [I2] * L
        ops[site_i] = op_i @ op_j
        full_op = kron_product(ops)
    else:
        ops_combined = [I2] * L
        ops_combined[site_i] = op_i
        ops_combined[site_j] = op_j
        full_op = kron_product(ops_combined)

    return float(gs @ full_op @ gs)


class TestExpectationValue:
    def test_sz_expectation_ground_state(self):
        """<Sz_i> should be approximately 0 for Heisenberg Sz=0 ground state."""
        L = 4
        result = _run_dmrg_sz0(L)

        for site in range(L):
            sz_val = expectation_value(result.mps, Sz, site)
            assert abs(sz_val) < 0.1, (
                f"<Sz_{site}> = {sz_val:.6f}, expected ~0 for Sz=0 state"
            )

    def test_sp_expectation_zero(self):
        """<S+_i> should be 0 for Sz=0 ground state (charge-changing operator)."""
        L = 4
        result = _run_dmrg_sz0(L)

        for site in range(L):
            sp_val = expectation_value(result.mps, Sp, site)
            assert abs(sp_val) < 1e-10, (
                f"<S+_{site}> = {sp_val:.6e}, expected 0 for Sz=0 state"
            )


class TestCorrelation:
    def test_szsz_correlation_vs_ed(self):
        """<Sz_i Sz_j> should match exact diagonalization for L=4."""
        L = 4
        result = _run_dmrg_sz0(L, bond_dim=8)

        for i in range(L):
            for j in range(i, L):
                corr_dmrg = correlation(result.mps, Sz, i, Sz, j)
                corr_ed = _ed_correlator(L, Sz, i, Sz, j)
                assert abs(corr_dmrg - corr_ed) < 1e-4, (
                    f"<Sz_{i} Sz_{j}>: DMRG={corr_dmrg:.6f} vs ED={corr_ed:.6f}, "
                    f"diff={abs(corr_dmrg - corr_ed):.4e}"
                )


class TestOperatorCharge:
    def test_sz_charge(self):
        """Sz has charge 0 (diagonal in charge basis)."""
        assert operator_charge(Sz) == 0

    def test_sp_charge(self):
        """S+ has charge +2 (raises Sz by 1, charges are ±1 → delta=2)."""
        assert operator_charge(Sp) == 2

    def test_sm_charge(self):
        """S- has charge -2 (lowers Sz by 1, charges are ±1 → delta=-2)."""
        assert operator_charge(Sm) == -2

    def test_mixed_operator_raises(self):
        """An operator mixing charges should raise ValueError."""
        mixed = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="mixed charges"):
            operator_charge(mixed)

    def test_identity_charge(self):
        """Identity has charge 0."""
        assert operator_charge(I2) == 0


class TestCorrelationAnticommute:
    def test_anticommute_sign_flip(self):
        """correlation() with anticommute=True should negate result when sites are swapped."""
        L = 4
        result = _run_dmrg_sz0(L, bond_dim=8)

        # Compute <Sp_0 Sm_2> in both orderings
        corr_normal = correlation(result.mps, Sp, 0, Sm, 2)
        # Call with sites reversed — should swap internally
        corr_reversed = correlation(result.mps, Sm, 2, Sp, 0)
        corr_reversed_anti = correlation(result.mps, Sm, 2, Sp, 0, anticommute=True)

        # Without anticommute: swap makes no sign difference (default behavior)
        np.testing.assert_allclose(corr_normal, corr_reversed, atol=1e-10)
        # With anticommute: swap introduces a sign flip
        np.testing.assert_allclose(corr_normal, -corr_reversed_anti, atol=1e-10)


class TestExpectationValueWarning:
    def test_hermitian_no_warning(self):
        """Hermitian operator should not trigger warning."""
        L = 4
        result = _run_dmrg_sz0(L, bond_dim=8)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Sz is Hermitian — should not warn
            expectation_value(result.mps, Sz, 0)

    def test_non_hermitian_warning(self):
        """Non-Hermitian operator with large imaginary expectation should warn."""
        L = 4
        result = _run_dmrg_sz0(L, bond_dim=8)
        # i * identity gives expectation value = i, which is purely imaginary
        non_herm_op = 1j * I2
        with pytest.warns(match="non-negligible imaginary part"):
            expectation_value(result.mps, non_herm_op, 0)
