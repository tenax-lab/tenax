"""Tests for coarse-grained iPEPS gate construction and 1-site RDM."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor import initialize_ctm_tensor_env
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import _rdm_1site_tensor
from tenax.algorithms.ad_utils import _config_to_tuple, ctm_tensor_converge
from tenax.algorithms.coarse_grain import CGGates, compute_energy_cg, honeycomb_cg_gates
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


@pytest.fixture(autouse=True)
def _enable_x64():
    """Enable float64 for this test module and restore afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_returns_cg_gates():
    """honeycomb_cg_gates returns a CGGates with n_sites == 2."""
    gates = honeycomb_cg_gates()
    assert isinstance(gates, CGGates)
    assert gates.n_sites == 2


def test_h_intra_shape_and_hermiticity():
    """h_intra is a (4,4) Hermitian matrix."""
    gates = honeycomb_cg_gates()
    assert gates.h_intra.shape == (4, 4)
    np.testing.assert_allclose(gates.h_intra, gates.h_intra.conj().T, atol=1e-14)


def test_h_intra_eigenvalues():
    """S*S eigenvalues are [-3/4, +1/4, +1/4, +1/4] (singlet-triplet)."""
    gates = honeycomb_cg_gates()
    eigvals = np.sort(np.linalg.eigvalsh(np.asarray(gates.h_intra)))
    expected = np.array([-3 / 4, 1 / 4, 1 / 4, 1 / 4])
    np.testing.assert_allclose(eigvals, expected, atol=1e-14)


def test_h_inter_keys_and_shapes():
    """h_inter has keys {'h', 'v'}, each with shape (4,4,4,4)."""
    gates = honeycomb_cg_gates()
    assert set(gates.h_inter.keys()) == {"h", "v"}
    for key in ("h", "v"):
        assert gates.h_inter[key].shape == (4, 4, 4, 4)


def test_h_inter_hermiticity():
    """Each h_inter reshaped to (16,16) is Hermitian."""
    gates = honeycomb_cg_gates()
    for key in ("h", "v"):
        mat = gates.h_inter[key].reshape(16, 16)
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-14)


# ---------------------------------------------------------------------------
# 1-site RDM tests
# ---------------------------------------------------------------------------


def _make_product_state_cg_tensor(state_a, state_b, D=1):
    """Build a D=1 CG tensor |state_a>x|state_b> as a DenseTensor.

    The effective physical dimension is len(state_a)*len(state_b).
    Virtual indices are trivial (D=1, charge=0).
    """
    d_eff = len(state_a) * len(state_b)
    phys = np.kron(state_a, state_b)
    data = np.zeros((D, D, D, D, d_eff))
    data[0, 0, 0, 0, :] = phys

    sym = U1Symmetry()
    charges_D = np.zeros(D, dtype=np.int32)
    charges_phys = np.zeros(d_eff, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges_D.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges_D.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges_D.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges_D.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, charges_phys.copy(), FlowDirection.IN, label="phys"
        ),
    )
    return DenseTensor(jnp.array(data), indices)


def _converge_ctm_env(A, chi=4, max_iter=20):
    """Run CTM to convergence and return (env_leaves, env_treedef)."""
    env_init = initialize_ctm_tensor_env(A, chi)
    env_leaves_init, env_treedef = jax.tree.flatten(env_init)
    config = CTMConfig(chi=chi, max_iter=max_iter)
    config_tuple = _config_to_tuple(config)
    env_leaves = ctm_tensor_converge(
        {(0, 0): A}, None, SINGLE_SITE_NEIGHBORS, config_tuple
    )
    env = jax.tree.unflatten(env_treedef, list(env_leaves))
    return env


class TestRDM1Site:
    """Tests for the 1-site reduced density matrix."""

    def test_product_state_trace(self):
        """For |up,down> product state, rdm has Tr=1 and is a pure state."""
        up = np.array([1.0, 0.0])
        down = np.array([0.0, 1.0])
        A = _make_product_state_cg_tensor(up, down)

        env = _converge_ctm_env(A, chi=4, max_iter=20)
        rdm = _rdm_1site_tensor(A, env)

        # Trace should be 1 (normalised)
        np.testing.assert_allclose(jnp.trace(rdm).real, 1.0, atol=1e-10)

        # Pure state: rho^2 = rho
        rdm2 = rdm @ rdm
        np.testing.assert_allclose(rdm2, rdm, atol=1e-10)

    def test_honeycomb_product_state_energy(self):
        """Tr(rdm . h_intra) for |up,down> = -1/4 (Sz.Sz of antiparallel)."""
        up = np.array([1.0, 0.0])
        down = np.array([0.0, 1.0])
        A = _make_product_state_cg_tensor(up, down)

        env = _converge_ctm_env(A, chi=4, max_iter=20)
        rdm = _rdm_1site_tensor(A, env)

        gates = honeycomb_cg_gates()
        h_intra = gates.h_intra  # (4, 4) Heisenberg S.S

        energy = jnp.trace(rdm @ h_intra).real
        # |up,down> -> Sz.Sz = (1/2)(-1/2) = -1/4, Sp.Sm + Sm.Sp = 0
        np.testing.assert_allclose(energy, -0.25, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_energy_cg tests
# ---------------------------------------------------------------------------


class TestComputeEnergyCG:
    def test_honeycomb_neel_product_state(self):
        """Honeycomb Neel |up,down>: E/site = -3/8.

        3 bonds (x, y, z) per unit cell of 2 sites.
        For Neel |up,down> state, every bond has S.S = -1/4.
        E/site = 3 * (-1/4) / 2 = -3/8 = -0.375.
        """
        up = np.array([1.0, 0.0])
        down = np.array([0.0, 1.0])
        A = _make_product_state_cg_tensor(up, down)
        env = _converge_ctm_env(A, chi=4, max_iter=20)
        gates = honeycomb_cg_gates()
        E = compute_energy_cg(A, env, gates, d_eff=4)
        np.testing.assert_allclose(float(E), -0.375, atol=1e-4)

    def test_honeycomb_ferro_product_state(self):
        """Honeycomb ferro |up,up>: E/site = +3/8.

        For |up,up> state, every bond has Sz.Sz = +1/4, S+S-=S-S+=0.
        E/site = 3 * (1/4) / 2 = 3/8 = 0.375.
        """
        up = np.array([1.0, 0.0])
        A = _make_product_state_cg_tensor(up, up)
        env = _converge_ctm_env(A, chi=4, max_iter=20)
        gates = honeycomb_cg_gates()
        E = compute_energy_cg(A, env, gates, d_eff=4)
        np.testing.assert_allclose(float(E), 0.375, atol=1e-4)
