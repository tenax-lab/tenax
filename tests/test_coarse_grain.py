"""Tests for coarse-grained iPEPS gate construction and 1-site RDM."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor import initialize_ctm_tensor_env
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import _rdm_1site_tensor, _rdm_diagonal_tensor
from tenax.algorithms.ad_utils import _config_to_tuple, ctm_tensor_converge
from tenax.algorithms.coarse_grain import (
    CGGates,
    compute_energy_cg,
    honeycomb_cg_gates,
    kagome_cg_gates,
)
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


class TestComputeEnergyCGSplit:
    """``compute_energy_cg_split`` must match ``compute_energy_cg`` for bosonic A."""

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_kagome_split_matches_std_at_small_D(self, D, chi):
        """Split-aware kagome CG energy == std-path energy on the same env.

        Drives ``ctm_split_tensor`` to convergence on a random U(1)-trivial
        kagome supersite, computes the energy via the split-aware CG path,
        and compares against the std-path energy on the shim-converted env.
        Both paths share the same converged env data — the test isolates
        the energy-RDM dispatch.
        """
        from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
        )
        from tenax.algorithms.coarse_grain import compute_energy_cg_split
        from tenax.algorithms.pess import kagome_xxz_pess_cg_gates

        d_eff = 8  # spin-1/2 kagome 3-PESS supersite
        gates = kagome_xxz_pess_cg_gates(delta=1.0, d=2)

        rng = np.random.default_rng(seed=0)
        data = rng.normal(size=(D, D, D, D, d_eff))
        data /= np.linalg.norm(data)
        sym = U1Symmetry()
        zD = np.zeros(D, dtype=np.int32)
        zd = np.zeros(d_eff, dtype=np.int32)
        A = DenseTensor(
            jnp.asarray(data),
            (
                TensorIndex.from_charges(sym, zD.copy(), FlowDirection.OUT, label="u"),
                TensorIndex.from_charges(sym, zD.copy(), FlowDirection.IN, label="d"),
                TensorIndex.from_charges(sym, zD.copy(), FlowDirection.OUT, label="l"),
                TensorIndex.from_charges(sym, zD.copy(), FlowDirection.IN, label="r"),
                TensorIndex.from_charges(
                    sym, zd.copy(), FlowDirection.IN, label="phys"
                ),
            ),
        )

        split_env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
        std_env = _split_env_to_tensor_standard(split_env)

        E_split = compute_energy_cg_split(A, split_env, gates, d_eff)
        E_std = compute_energy_cg(A, std_env, gates, d_eff)
        np.testing.assert_allclose(float(E_split), float(E_std), atol=1e-10)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_kagome_split_grad_matches_std_at_small_D(self, D, chi):
        """Split-aware kagome CG energy GRADIENT == std-path gradient.

        Mirrors ``test_kagome_split_matches_std_at_small_D`` at the gradient
        level: holds the converged split env fixed (a constant w.r.t. the
        differentiated supersite A) and differentiates the CG energy w.r.t. A
        through both the split-aware RDM path and the shim path built from the
        *same* env. Isolates the energy-RDM-dispatch backward, complementing the
        forward parity test. (Analogue of
        ``test_compute_energy_split_native_grad_matches_shim`` for the
        coarse-grained kagome supersite.)
        """
        from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
        )
        from tenax.algorithms.coarse_grain import compute_energy_cg_split
        from tenax.algorithms.pess import kagome_xxz_pess_cg_gates

        d_eff = 8  # spin-1/2 kagome 3-PESS supersite
        gates = kagome_xxz_pess_cg_gates(delta=1.0, d=2)

        rng = np.random.default_rng(seed=0)
        data = rng.normal(size=(D, D, D, D, d_eff))
        data /= np.linalg.norm(data)
        sym = U1Symmetry()
        zD = np.zeros(D, dtype=np.int32)
        zd = np.zeros(d_eff, dtype=np.int32)
        A = DenseTensor(
            jnp.asarray(data),
            (
                TensorIndex.from_charges(sym, zD.copy(), FlowDirection.OUT, label="u"),
                TensorIndex.from_charges(sym, zD.copy(), FlowDirection.IN, label="d"),
                TensorIndex.from_charges(sym, zD.copy(), FlowDirection.OUT, label="l"),
                TensorIndex.from_charges(sym, zD.copy(), FlowDirection.IN, label="r"),
                TensorIndex.from_charges(
                    sym, zd.copy(), FlowDirection.IN, label="phys"
                ),
            ),
        )

        split_env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
        std_env = _split_env_to_tensor_standard(split_env)

        def loss_split(a):
            return jnp.real(compute_energy_cg_split(a, split_env, gates, d_eff))

        def loss_std(a):
            return jnp.real(compute_energy_cg(a, std_env, gates, d_eff))

        g_split = jax.tree_util.tree_leaves(jax.grad(loss_split)(A))
        g_std = jax.tree_util.tree_leaves(jax.grad(loss_std)(A))
        assert len(g_split) == len(g_std) and g_split, "gradient pytree mismatch"
        for ls, lh in zip(g_split, g_std):
            np.testing.assert_allclose(
                np.asarray(ls),
                np.asarray(lh),
                atol=1e-8,
                rtol=1e-8,
                err_msg="split-aware CG energy gradient diverges from std gradient",
            )


# ---------------------------------------------------------------------------
# AD optimization integration test
# ---------------------------------------------------------------------------


class TestHoneycombAD:
    @pytest.mark.slow
    def test_honeycomb_d2_energy_is_physical(self):
        """Honeycomb D=2 AD optimization must produce finite negative E/site."""
        from tenax.algorithms.coarse_grain import honeycomb_cg_gates
        from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
        from tenax.algorithms.ipeps_optimize import optimize_gs_ad

        gates = honeycomb_cg_gates()
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=8, max_iter=20, min_iter=5),
            gs_num_steps=10,
            gs_learning_rate=1e-2,
            gs_optimizer="lbfgs",
            gs_line_search=True,
            gs_line_search_method="armijo",
            gs_c4v=True,
            gs_explicit_ad=True,
            gs_explicit_ad_steps=10,
            gs_explicit_ad_warmup=3,
            gs_metric_precond=False,
            gs_verbose=False,
            su_init=False,
            cg_gates=gates,
        )
        # Pass a dummy gate with correct d_eff shape for the Hamiltonian placeholder
        dummy_gate = jnp.zeros((4, 4, 4, 4))
        _, _, E_gs = optimize_gs_ad(dummy_gate, None, config)
        assert np.isfinite(E_gs), f"E_gs = {E_gs} is not finite"
        assert E_gs < 0.0, f"E_gs = {E_gs} should be negative"


# ---------------------------------------------------------------------------
# Kagome coarse-grained gate tests
# ---------------------------------------------------------------------------


class TestKagomeCGGates:
    def test_returns_cg_gates(self):
        gates = kagome_cg_gates()
        assert isinstance(gates, CGGates)
        assert gates.n_sites == 3

    def test_h_intra_shape_and_hermiticity(self):
        gates = kagome_cg_gates()
        assert gates.h_intra.shape == (8, 8)
        np.testing.assert_allclose(gates.h_intra, gates.h_intra.conj().T, atol=1e-14)

    def test_h_intra_eigenvalues(self):
        """H_tri = S_u.S_v + S_u.S_w + S_v.S_w.

        S_total(S_total+1)/2 - 3*s(s+1)/2 gives:
          S_total=3/2 (quadruplet): (15/4 - 9/4)/2 = 3/4
          S_total=1/2 (two doublets): (3/4 - 9/4)/2 = -3/4
        """
        gates = kagome_cg_gates()
        eigvals = np.sort(np.linalg.eigvalsh(np.array(gates.h_intra)))
        expected = np.array([-0.75, -0.75, -0.75, -0.75, 0.75, 0.75, 0.75, 0.75])
        np.testing.assert_allclose(eigvals, expected, atol=1e-12)

    def test_h_inter_keys_and_shapes(self):
        gates = kagome_cg_gates()
        assert set(gates.h_inter.keys()) == {"h", "v", "diag"}
        for key in ("h", "v", "diag"):
            assert gates.h_inter[key].shape == (8, 8, 8, 8)

    def test_h_inter_hermiticity(self):
        gates = kagome_cg_gates()
        for key in ("h", "v", "diag"):
            H = np.array(gates.h_inter[key]).reshape(64, 64)
            np.testing.assert_allclose(H, H.conj().T, atol=1e-14)


# ---------------------------------------------------------------------------
# Diagonal (NNN) RDM tests
# ---------------------------------------------------------------------------


def _make_product_state_cg_tensor_3site(s_u, s_v, s_w, D=1):
    """Build a D=1 CG tensor |s_u>x|s_v>x|s_w> as a DenseTensor.

    The effective physical dimension is d_eff = 2^3 = 8 for spin-1/2.
    Virtual indices are trivial (D=1, charge=0).
    """
    phys = np.kron(np.kron(s_u, s_v), s_w)
    d_eff = len(phys)
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


class TestRDMDiagonal:
    """Tests for the diagonal (NNN) reduced density matrix."""

    def test_product_state_trace_and_factorization(self):
        """D=1 product state: diagonal RDM should factorize as rho_A x rho_B."""
        up = np.array([1.0, 0.0])
        A = _make_product_state_cg_tensor_3site(up, up, up)

        env = _converge_ctm_env(A, chi=4, max_iter=20)
        rdm = _rdm_diagonal_tensor(A, env)

        assert rdm.shape == (8, 8, 8, 8)

        rdm_mat = rdm.reshape(64, 64)
        # Trace should be 1
        np.testing.assert_allclose(jnp.trace(rdm_mat), 1.0, atol=1e-10)

        # For a product state, rho_{TL,BR} = rho_TL x rho_BR
        rdm_1 = _rdm_1site_tensor(A, env)
        expected = np.einsum("ij,kl->ikjl", rdm_1, rdm_1)
        np.testing.assert_allclose(rdm, expected, atol=1e-6)

    def test_product_state_pure_state(self):
        """D=1 product state: diagonal RDM reshaped is a projector (rho^2=rho)."""
        up = np.array([1.0, 0.0])
        down = np.array([0.0, 1.0])
        A = _make_product_state_cg_tensor_3site(up, down, up)

        env = _converge_ctm_env(A, chi=4, max_iter=20)
        rdm = _rdm_diagonal_tensor(A, env)
        rdm_mat = rdm.reshape(64, 64)

        # Pure state: rho^2 = rho
        rdm2 = rdm_mat @ rdm_mat
        np.testing.assert_allclose(rdm2, rdm_mat, atol=1e-10)


class TestCGGatesValidation:
    def test_cg_gates_requires_1x1_unit_cell(self):
        from tenax.algorithms.ipeps_config import iPEPSConfig

        gates = honeycomb_cg_gates()
        with pytest.raises(ValueError, match="cg_gates requires unit_cell='1x1'"):
            iPEPSConfig(unit_cell="2site", cg_gates=gates)

    def test_cg_gates_rejects_su_init(self):
        from tenax.algorithms.ipeps_config import iPEPSConfig

        gates = honeycomb_cg_gates()
        with pytest.raises(ValueError, match="incompatible with su_init=True"):
            iPEPSConfig(cg_gates=gates, su_init=True)


# ---------------------------------------------------------------------------
# Regression tests for PR #352 review comments (Codex P1)
# ---------------------------------------------------------------------------


class TestCGOptimizerGuards:
    @pytest.mark.slow
    def test_default_metric_precond_no_crash(self):
        """Default iPEPSConfig with cg_gates must not crash on metric L-BFGS.

        ``gs_metric_precond=True`` is the default and the metric path calls
        ``.todense()`` on grads/params, which fails on tuple params produced
        by ``cg_gates.map_fn``.  The optimizer must auto-disable metric
        preconditioning (with a warning) for that case rather than raise
        ``AttributeError`` on the first step.
        """
        from tenax.algorithms.coarse_grain import honeycomb_cg_gates
        from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
        from tenax.algorithms.ipeps_optimize import optimize_gs_ad

        gates = honeycomb_cg_gates()
        # All preconditioning + L-BFGS defaults retained except: short run +
        # disable c4v + use explicit AD to keep this test fast.
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=8, max_iter=20, min_iter=5),
            gs_num_steps=3,
            gs_optimizer="lbfgs",
            gs_metric_precond=True,  # the bug trigger
            gs_line_search=True,
            gs_line_search_method="armijo",
            gs_implicit_ad=False,
            gs_explicit_ad_steps=10,
            gs_explicit_ad_warmup=3,
            gs_verbose=False,
            su_init=False,
            cg_gates=gates,
        )
        dummy_gate = jnp.zeros((4, 4, 4, 4))
        with pytest.warns(UserWarning, match="gs_metric_precond=True is incompatible"):
            _, _, E_gs = optimize_gs_ad(dummy_gate, None, config)
        assert np.isfinite(E_gs)

    def test_user_supplied_raw_params_are_respected(self):
        """A_init as a raw-params tuple must not be overwritten by init_fn.

        Compares two ``gs_num_steps=0`` runs: one with no A_init (uses
        ``init_fn`` default PRNG seed) and one with raw params from a
        different PRNG seed.  If the user's params were ignored both runs
        would produce identical energies; with the fix they must differ.
        """
        from tenax.algorithms.coarse_grain import honeycomb_cg_gates
        from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
        from tenax.algorithms.ipeps_optimize import optimize_gs_ad

        gates = honeycomb_cg_gates()
        D = 2

        config = iPEPSConfig(
            max_bond_dim=D,
            ctm=CTMConfig(chi=8, max_iter=20, min_iter=5),
            gs_num_steps=0,  # evaluate-only
            gs_optimizer="lbfgs",
            gs_metric_precond=False,
            gs_implicit_ad=False,
            gs_explicit_ad_steps=10,
            gs_explicit_ad_warmup=3,
            gs_verbose=False,
            su_init=False,
            cg_gates=gates,
        )
        dummy_gate = jnp.zeros((4, 4, 4, 4))

        # Run 1: A_init=None → init_fn called with default PRNGKey(0).
        _, _, E_default = optimize_gs_ad(dummy_gate, None, config)

        # Run 2: A_init is a tuple of raw params from a different PRNG key.
        custom_params = gates.init_fn(D, jax.random.PRNGKey(42))
        _, _, E_user = optimize_gs_ad(dummy_gate, custom_params, config)

        # Energies must differ — proves user params weren't silently
        # overwritten with init_fn(PRNGKey(0)).
        assert not np.isclose(E_default, E_user, atol=1e-8), (
            f"User-supplied raw params were ignored: E_default={E_default} "
            f"E_user={E_user} (identical means init_fn override is back)"
        )

    def test_tensor_a_init_with_map_fn_raises(self):
        """Single-tensor A_init is ambiguous when CG+map_fn — must raise."""
        from tenax.algorithms.coarse_grain import honeycomb_cg_gates
        from tenax.algorithms.ipeps_config import iPEPSConfig
        from tenax.algorithms.ipeps_optimize import (
            _wrap_as_dense_tensor,
            optimize_gs_ad,
        )

        gates = honeycomb_cg_gates()
        D = 2
        bogus_A = _wrap_as_dense_tensor(jnp.zeros((D, D, D, D, 4)))
        config = iPEPSConfig(
            max_bond_dim=D,
            cg_gates=gates,
            su_init=False,
        )
        with pytest.raises(ValueError, match="A_init must be either None"):
            optimize_gs_ad(jnp.zeros((4, 4, 4, 4)), bogus_A, config)
